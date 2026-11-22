"""End-to-end pipeline: load -> train -> detect -> evaluate -> write."""
from __future__ import annotations

import json
import pickle
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from .config import Config
from .evaluation import evaluate
from .features import complete_genes, resolve_column
from .intervals import COLDSPOT_COLUMNS, decode_genes
from .model import marginalize_params, train_params_mv

__all__ = ["run", "LoadedData", "load_data"]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@dataclass
class LoadedData:
    residues: pd.DataFrame      # per-residue table derived from the input
    clinvar_eval: pd.DataFrame  # input rows for ClinVar evaluation
    mave_eval: pd.DataFrame     # input rows with a MAVE call
    mave_col: str               # resolved MAVE label column name
    clinvar_genes: set[str]     # genes passing the ClinVar count filter


_PROTEIN_VARIANT = r"^[A-Za-z*]{1,3}(\d+)"   # "C2R" / "Cys2Arg" -> 2


def read_variant_table(cfg: Config) -> tuple[pd.DataFrame, str]:
    """Read only the needed columns of the input table; add ``aa_pos``.

    Returns the table and the resolved MAVE column name.
    """
    cols, feats = cfg.columns, cfg.features
    path = cfg.data.variant_csv
    header = pd.read_csv(path, nrows=0).columns
    mave_col = resolve_column(pd.DataFrame(columns=header), cols.mave,
                              cols.mave_candidates, "MAVE label")
    has_pos = cols.aa_pos in header
    pos_source = cols.aa_pos if has_pos else cols.protein_variant
    needed = list(dict.fromkeys(
        [cols.gene, pos_source, cols.clinvar, mave_col] + feats.primary))
    missing = [c for c in needed if c not in header]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}. "
                       f"See docs/DATA.md for the expected schema.")

    # Precomputed membership columns are simply not read, so a stale
    # assignment can never leak into the evaluation.
    stale = [c for c in cols.drop_precomputed if c in header]
    if stale:
        print(f"  Ignoring precomputed membership columns: {stale}")

    # Read in chunks so the protein_variant strings never all sit in memory at
    # once: each chunk keeps only the parsed integer position. Gene and label
    # columns repeat a handful of values across millions of rows, so they are
    # held as categoricals.
    cat_cols = list(dict.fromkeys([cols.gene, cols.clinvar, mave_col]))
    parts = []
    for chunk in pd.read_csv(path, usecols=needed, chunksize=1_000_000,
                             low_memory=False,
                             dtype={c: "category" for c in cat_cols}):
        # A chunk where a label column is entirely empty infers categories of
        # a different dtype; give every chunk object categories so they unify.
        for c in cat_cols:
            chunk[c] = chunk[c].cat.set_categories(
                chunk[c].cat.categories.astype(object))
        if not has_pos:
            parsed = chunk[cols.protein_variant].str.extract(_PROTEIN_VARIANT)[0]
            bad = parsed.isna()
            if bad.any():
                raise ValueError(
                    f"Could not read an amino-acid position from "
                    f"{cols.protein_variant!r} values such as "
                    f"{chunk.loc[bad, cols.protein_variant].head(3).tolist()}")
            chunk[cols.aa_pos] = parsed.astype(np.int64)
            chunk = chunk.drop(columns=cols.protein_variant)
        parts.append(chunk)
    if not parts:
        raise ValueError(f"{path}: the table has no rows")
    # Chunks infer their own categories; unify them (sorted, as a single
    # read_csv would) rather than letting concat fall back to strings.
    cats = {c: union_categoricals([p[c] for p in parts], sort_categories=True)
            for c in cat_cols}
    df = pd.concat([p.drop(columns=cat_cols) for p in parts],
                   ignore_index=True)
    del parts
    for c in cat_cols:
        df[c] = cats[c]
    return df, mave_col


def residue_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Collapse the variant table to one row per (gene, aa_pos).

    The predictor scores are per-residue summaries, so every substitution at
    a residue must carry the same values; anything else means the table is
    not what the method expects, and is an error.
    """
    cols, feats = cfg.columns, cfg.features
    key = [cols.gene, cols.aa_pos]
    residues = df.drop_duplicates(key)
    n_distinct = len(df.drop_duplicates(key + feats.primary))
    if n_distinct != len(residues):
        raise ValueError(
            f"{n_distinct - len(residues)} residues carry different "
            f"{feats.primary} values across their substitutions; these "
            f"scores must be per-residue (e.g. max/min over substitutions).")
    return (residues[key + feats.primary]
            .sort_values(key).reset_index(drop=True))


def load_data(cfg: Config) -> LoadedData:
    cols, ev = cfg.columns, cfg.evaluation
    print("Loading data ...")
    df, mave_col = read_variant_table(cfg)
    residues = residue_table(df, cfg)

    # Training genes are never evaluated.
    evaluable = df[~df[cols.gene].isin(cfg.exclude_genes)]

    # MAVE evaluation set: rows carrying a functional call. Not subject to the
    # ClinVar count filter - the two evaluations gate genes independently.
    mave_eval = evaluable[evaluable[mave_col].notna()]

    # ClinVar evaluation set: genes with enough PLP and BLB rows. reindex so a
    # table with no PLP (or no BLB) rows at all still yields a column of zeros.
    plp, blb = ev.clinvar_labels["PLP"], ev.clinvar_labels["BLB"]
    counts = (evaluable.groupby(cols.gene)[cols.clinvar]
              .value_counts().unstack(fill_value=0)
              .reindex(columns=[plp, blb], fill_value=0))
    min_n = ev.min_class_count
    clinvar_genes = set(counts[(counts[plp] >= min_n) &
                               (counts[blb] >= min_n)].index)
    clinvar_eval = evaluable[evaluable[cols.gene].isin(clinvar_genes)]

    _report_inputs(cfg, residues, clinvar_eval, mave_eval, mave_col)
    return LoadedData(residues, clinvar_eval, mave_eval, mave_col,
                      clinvar_genes)


def _report_inputs(cfg, residues, clinvar_eval, mave_eval, mave_col) -> None:
    cols, ev = cfg.columns, cfg.evaluation
    n_lab = int(clinvar_eval[cols.clinvar]
                .isin(list(ev.clinvar_labels.values())).sum())
    n_pb = int(mave_eval[mave_col]
               .isin([ev.mave_labels["PLP"], ev.mave_labels["BLB"]]).sum())
    train_desc = ", ".join(
        f"{g} {iv}" for g, ivs in cfg.training_intervals.items() for iv in ivs
    )
    filt = ("no gene filter" if ev.min_class_count == 0
            else f">={ev.min_class_count} PLP & BLB")
    print(f"  Input table  : {cfg.data.variant_csv}")
    print(f"  Residues     : "
          f"{residues[cols.gene].nunique()} genes, "
          f"{len(residues):,} residues (training + detection)")
    print(f"  Trained on   : {train_desc}")
    print(f"  ClinVar eval : {clinvar_eval[cols.gene].nunique()} genes "
          f"({'/'.join(sorted(cfg.exclude_genes))} excluded, "
          f"{filt})")
    print(f"                 {len(clinvar_eval):,} variant rows, {n_lab:,} with "
          f"a {cols.clinvar} call, "
          f"{clinvar_eval[[cols.gene, cols.aa_pos]].drop_duplicates().shape[0]:,}"
          f" distinct residues")
    print(f"  MAVE eval    : {mave_eval[cols.gene].nunique()} genes "
          f"(same table, {mave_col!r} column)")
    print(f"                 {len(mave_eval):,} rows with a MAVE call, "
          f"{n_pb:,} PLP/BLB, "
          f"{mave_eval[[cols.gene, cols.aa_pos]].drop_duplicates().shape[0]:,}"
          f" distinct residues")
    print(f"  Counting unit: {ev.unit}"
          + ("  (primary analysis)" if ev.unit == "variant"
             else "  (pseudo-replication sensitivity analysis)"))
    cov = (f", >={ev.min_mave_coverage:.0%} of the interval assayed"
           if ev.min_mave_coverage > 0 else "")
    print(f"  MAVE filter  : >={ev.min_mave_positions} distinct PLP/BLB aa "
          f"positions per {ev.mave_filter_level}{cov}")


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def _train(cfg: Config, residues: pd.DataFrame):
    """Train the primary model, then marginalise it to the fallback model."""
    cols, feats = cfg.columns, cfg.features
    kw = dict(gene_col=cols.gene, pos_col=cols.aa_pos,
              training_intervals=cfg.training_intervals,
              invert=feats.invert, winsorize=feats.winsorize)

    # Gene sets for training include the training gene(s).
    genes_primary = complete_genes(residues, feats.primary, cols.gene)
    genes_fallback = complete_genes(
        residues, cfg.fallback_features, cols.gene) - genes_primary

    missing_train = [g for g in cfg.training_intervals
                     if g not in genes_primary]
    if missing_train:
        raise ValueError(
            f"Training gene(s) {missing_train} lack complete scores for "
            f"{feats.primary}; cannot estimate parameters."
        )

    print("\nTraining parameters ...")
    train_df = residues[residues[cols.gene].isin(genes_primary)].copy()
    primary = train_params_mv(train_df, feats.primary, **kw)
    print(f"  primary ({len(feats.primary)}D {feats.primary}): "
          f"cold mean={primary.means[primary.cold_state].round(3)}")

    fallback = marginalize_params(primary, cfg.fallback_keep_indices)
    print(f"  fallback ({len(cfg.fallback_features)}D "
          f"{cfg.fallback_features}, marginalised): "
          f"cold mean={fallback.means[fallback.cold_state].round(3)}")

    detect_primary = genes_primary - cfg.exclude_genes
    detect_fallback = genes_fallback - cfg.exclude_genes
    all_genes = set(residues[cols.gene].unique())
    fb = residues[residues[cols.gene].isin(genes_fallback)]
    n_partial = int(fb[feats.fallback_drop].notna().any(axis=1)
                    .groupby(fb[cols.gene]).any().sum())
    n_dropped = len(all_genes - genes_primary - genes_fallback)
    print(f"\nGene sets (train)  : primary={len(genes_primary)}, "
          f"fallback={len(genes_fallback)} "
          f"({n_partial} of them have some, but not all, "
          f"{'/'.join(feats.fallback_drop)} values)")
    print(f"Not decoded        : {n_dropped} genes with a missing value in "
          f"{cfg.fallback_features}")
    print(f"Gene sets (detect) : primary={len(detect_primary)}, "
          f"fallback={len(detect_fallback)} "
          f"({'/'.join(sorted(cfg.exclude_genes))} excluded)")
    return [(detect_primary, primary), (detect_fallback, fallback)]


def detect(cfg: Config, residues: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Detect cold spots across all genes, primary model then fallback."""
    runs = _train(cfg, residues)
    frames = []
    for gene_set, params in runs:
        print(f"  [{len(params.features)}D: {params.features}]")
        cdf = decode_genes(
            gene_set, params, residues,
            gene_col=cfg.columns.gene, pos_col=cfg.columns.aa_pos,
            min_cold_run=cfg.detection.min_cold_run,
            invert=cfg.features.invert, winsorize=cfg.features.winsorize,
        )
        cdf = cdf.assign(model=f"{len(params.features)}D")
        if not cdf.empty:
            frames.append(cdf)

    if not frames:
        combined = pd.DataFrame(columns=COLDSPOT_COLUMNS + ["model"])
    else:
        combined = (pd.concat(frames, ignore_index=True)
                    .sort_values(["gene", "aa_start"])
                    .reset_index(drop=True))
        print(f"  -> Combined: {len(combined)} intervals | "
              f"{combined['gene'].nunique()} genes | "
              f"{int(combined['length'].sum()):,} AAs")
    return combined, runs


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def report(cfg: Config, coldspots: pd.DataFrame, ev: dict,
           clinvar_genes: set[str], mave_col: str) -> None:
    if ev is None:
        print("\nNo cold spots detected; nothing to evaluate.")
        return
    e = cfg.evaluation
    print(f"\n{'=' * 78}")
    print("  RESULTS")
    print(f"  Counting unit : {ev['unit']}")
    print(f"  ClinVar eval  : input table"
          + (f", genes with >={e.min_class_count} PLP & BLB"
             if e.min_class_count else ", all genes"))
    print(f"  MAVE eval     : same table ({mave_col!r}), "
          f">={e.min_mave_positions} distinct PLP/BLB aa positions "
          f"per {e.mave_filter_level}")
    print(f"{'=' * 78}")

    n_cv = len(set(coldspots["gene"].unique()) & clinvar_genes)
    print(f"  Cold spots          : {ev['n_intervals']} intervals, "
          f"{ev['n_genes']} genes, {ev['total_aas']:,} AAs")
    print(f"  Cold spots (ClinVar): {n_cv} eval-set genes")
    if ev["total_mave_aas"]:
        print(f"  Cold spots (MAVE)   : {ev['n_mave_intervals']} intervals, "
              f"{ev['n_mave_genes']} genes, {ev['total_mave_aas']:,} AAs "
              f"({ev['assayed_mave_aas']:,} assayed, "
              f"{ev['assayed_mave_aas'] / ev['total_mave_aas']:.0%})")
    else:
        print("  Cold spots (MAVE)   : none pass the support filter")

    for name, summary, order in (
        ("ClinVar", ev["clinvar_summary"], ("BLB", "PLP", "VUS")),
        ("MAVE", ev["mave_summary"], ("BLB", "PLP", "IR")),
    ):
        print(f"\n  {name} ({ev['unit']}-level counts inside / total):")
        for lbl in order:
            if lbl not in summary:
                continue
            s = summary[lbl]
            rate = ("    -- " if np.isnan(s["rate"])
                    else f"{100 * s['rate']:>6.2f}%")
            print(f"    {lbl:<4} {s['inside']:>9,} / {s['total']:>9,}  ({rate})")

    for name, key, lrkey in (("ClinVar", "or_clinvar", "lr_clinvar"),
                             ("MAVE", "or_mave", "lr_mave")):
        o = ev[key]
        print(f"\n  {name}: OR(benign inside) = {o['OR']} "
              f"[{o['OR_lo']}-{o['OR_hi']}] p={_format_p(o['p'])}"
              if not isinstance(o["p"], float) or not np.isnan(o["p"])
              else f"\n  {name}: OR not estimable")
        print(f"          rate(PLP)/rate(BLB) = {ev[lrkey]:.4f}")


def _format_p(p: float) -> str:
    """Fisher p-values for large tables underflow to exactly 0.0 in floating
    point; print that as a bound rather than as "p=0"."""
    return "<1e-300" if p == 0 else f"{p:.4g}"


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def _git_commit() -> str | None:
    """Commit of the checkout this package was imported from.

    Returns None if the package is not running from its own git checkout
    (e.g. a non-editable install inside some other repository), and appends
    ``-dirty`` when tracked files have uncommitted changes, since the commit
    alone then does not identify the code that produced the outputs.
    """
    pkg_dir = Path(__file__).resolve().parent

    def _git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=pkg_dir, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""

    try:
        top = _git("rev-parse", "--show-toplevel")
        if not top or Path(top).resolve() != pkg_dir.parent:
            return None
        commit = _git("rev-parse", "HEAD")
        if not commit:
            return None
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return None


def _versions() -> dict[str, str]:
    import hmmlearn
    import scipy
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "hmmlearn": hmmlearn.__version__,
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()
                if not isinstance(v, pd.DataFrame)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, set):
        return sorted(obj)
    return obj


def write_outputs(cfg: Config, coldspots: pd.DataFrame, ev: dict | None,
                  runs: list, mave_col: str) -> dict[str, Path]:
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.output.prefix
    written: dict[str, Path] = {}

    # Plain-text tables are the archival record; the pickle is a convenience.
    f = out_dir / f"{p}_intervals.csv"
    coldspots.to_csv(f, index=False)
    written["intervals"] = f

    if ev is not None and not ev["coldspot_df_mave"].empty:
        f = out_dir / f"{p}_intervals_mave_filtered.csv"
        ev["coldspot_df_mave"].to_csv(f, index=False)
        written["intervals_mave"] = f

    if ev is not None:
        f = out_dir / f"{p}_metrics.json"
        f.write_text(json.dumps(_jsonable(ev), indent=2))
        written["metrics"] = f

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config_fingerprint": cfg.fingerprint(),
        "config": _jsonable(cfg.to_dict()),
        "resolved_mave_column": mave_col,
        "package_versions": _versions(),
        "trained_models": [params.summary() for _, params in runs],
    }
    f = out_dir / f"{p}_manifest.json"
    f.write_text(json.dumps(manifest, indent=2, default=str))
    written["manifest"] = f

    if cfg.output.write_pickle:
        f = out_dir / f"{p}.pkl"
        with open(f, "wb") as fh:
            pickle.dump({"coldspot_df": coldspots, "eval": ev,
                         "manifest": manifest}, fh)
        written["pickle"] = f

    print("\nWrote:")
    for k, v in written.items():
        print(f"  {k:<16} {v}")
    return written


# ----------------------------------------------------------------------------
def run(cfg: Config) -> dict:
    """Execute the full pipeline and return the results bundle."""
    np.random.seed(cfg.seed)  # decoding is deterministic; set for good measure
    data = load_data(cfg)
    print(f"\n{'=' * 60}\n  DETECTION\n{'=' * 60}")
    coldspots, runs = detect(cfg, data.residues)
    ev = evaluate(coldspots, data.clinvar_eval, data.mave_eval,
                  cfg=cfg, mave_col=data.mave_col)
    report(cfg, coldspots, ev, data.clinvar_genes, data.mave_col)
    written = write_outputs(cfg, coldspots, ev, runs, data.mave_col)
    return {"coldspots": coldspots, "eval": ev, "written": written}
