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
    residues: pd.DataFrame     # collapsed per-residue table (detection input)
    clinvar_eval: pd.DataFrame  # expanded table, ClinVar gene filter applied
    mave_eval: pd.DataFrame     # expanded table, rows with a MAVE call
    mave_col: str               # resolved MAVE label column name
    clinvar_genes: set[str]     # genes passing the ClinVar count filter


def load_data(cfg: Config) -> LoadedData:
    cols = cfg.columns
    print("Loading data ...")
    residues = (pd.read_csv(cfg.data.vep_residue_csv)
                .sort_values([cols.gene_residue, cols.aa_pos]))
    expanded = pd.read_csv(cfg.data.vep_expanded_csv, low_memory=False)

    # Cold-spot membership is recomputed from this run's intervals, so any
    # membership column shipped with the table is dropped to prevent silent
    # use of a stale assignment.
    stale = [c for c in cols.drop_precomputed if c in expanded.columns]
    if stale:
        expanded = expanded.drop(columns=stale)
        print(f"  Dropped precomputed membership columns: {stale}")

    mave_col = resolve_column(expanded, cols.mave, cols.mave_candidates,
                              "MAVE label")

    required = {cols.gene_expanded, cols.aa_pos, cols.clinvar, mave_col}
    missing = required - set(expanded.columns)
    if missing:
        raise KeyError(f"Expanded table missing columns: {sorted(missing)}")

    # Training genes are never evaluated.
    expanded = expanded[
        ~expanded[cols.gene_expanded].isin(cfg.exclude_genes)
    ].copy()

    # MAVE evaluation set: rows carrying a functional call. Not subject to the
    # ClinVar count filter - the two evaluations gate genes independently.
    mave_eval = expanded[expanded[mave_col].notna()].copy()

    # ClinVar evaluation set: genes with enough PLP and BLB rows.
    counts = (expanded.groupby(cols.gene_expanded)[cols.clinvar]
              .value_counts().unstack(fill_value=0))
    min_n = cfg.evaluation.min_class_count
    clinvar_genes = set(counts[(counts.get("PLP", 0) >= min_n) &
                               (counts.get("BLB", 0) >= min_n)].index)
    clinvar_eval = expanded[
        expanded[cols.gene_expanded].isin(clinvar_genes)
    ].copy()

    _report_inputs(cfg, residues, clinvar_eval, mave_eval, mave_col)
    return LoadedData(residues, clinvar_eval, mave_eval, mave_col,
                      clinvar_genes)


def _report_inputs(cfg, residues, clinvar_eval, mave_eval, mave_col) -> None:
    cols, ev = cfg.columns, cfg.evaluation
    n_lab = int(clinvar_eval[cols.clinvar]
                .isin(list(ev.clinvar_labels.values())).sum())
    n_pb = int(mave_eval[mave_col].isin(["PLP", "BLB"]).sum())
    train_desc = ", ".join(
        f"{g} {iv}" for g, ivs in cfg.training_intervals.items() for iv in ivs
    )
    filt = ("no gene filter" if ev.min_class_count == 0
            else f">={ev.min_class_count} PLP & BLB")
    print(f"  Residue table: "
          f"{residues[cols.gene_residue].nunique()} genes, "
          f"{len(residues):,} residues (detection input)")
    print(f"  Trained on   : {train_desc}")
    print(f"  ClinVar eval : {clinvar_eval[cols.gene_expanded].nunique()} genes "
          f"(expanded table, {'/'.join(sorted(cfg.exclude_genes))} excluded, "
          f"{filt})")
    print(f"                 {len(clinvar_eval):,} variant rows, {n_lab:,} with "
          f"a {cols.clinvar} call, "
          f"{clinvar_eval[[cols.gene_expanded, cols.aa_pos]].drop_duplicates().shape[0]:,}"
          f" distinct residues")
    print(f"  MAVE eval    : {mave_eval[cols.gene_expanded].nunique()} genes "
          f"(same table, {mave_col!r} column)")
    print(f"                 {len(mave_eval):,} rows with a MAVE call, "
          f"{n_pb:,} PLP/BLB, "
          f"{mave_eval[[cols.gene_expanded, cols.aa_pos]].drop_duplicates().shape[0]:,}"
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
    kw = dict(gene_col=cols.gene_residue, pos_col=cols.aa_pos,
              training_intervals=cfg.training_intervals,
              invert=feats.invert, winsorize=feats.winsorize)

    # Gene sets for training include the training gene(s).
    genes_primary = complete_genes(residues, feats.primary, cols.gene_residue)
    genes_fallback = complete_genes(
        residues, cfg.fallback_features, cols.gene_residue) - genes_primary

    missing_train = [g for g in cfg.training_intervals
                     if g not in genes_primary]
    if missing_train:
        raise ValueError(
            f"Training gene(s) {missing_train} lack complete scores for "
            f"{feats.primary}; cannot estimate parameters."
        )

    print("\nTraining parameters ...")
    train_df = residues[residues[cols.gene_residue].isin(genes_primary)].copy()
    primary = train_params_mv(train_df, feats.primary, **kw)
    print(f"  primary ({len(feats.primary)}D {feats.primary}): "
          f"cold mean={primary.means[primary.cold_state].round(3)}")
    if primary.state_swapped:
        print("  WARNING: cold/hot state indices were swapped during training. "
              "This did not occur in the published run - see "
              "docs/CHANGES_FROM_ORIGINAL.md before trusting these numbers.")

    fallback = marginalize_params(primary, cfg.fallback_keep_indices)
    print(f"  fallback ({len(cfg.fallback_features)}D "
          f"{cfg.fallback_features}, marginalised): "
          f"cold mean={fallback.means[fallback.cold_state].round(3)}")

    detect_primary = genes_primary - cfg.exclude_genes
    detect_fallback = genes_fallback - cfg.exclude_genes
    print(f"\nGene sets (train)  : primary={len(genes_primary)}, "
          f"fallback={len(genes_fallback)}")
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
            gene_col=cfg.columns.gene_residue, pos_col=cfg.columns.aa_pos,
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
    print(f"  ClinVar eval  : expanded table"
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
              f"[{o['OR_lo']}-{o['OR_hi']}] p={o['p']:.4g}"
              if not isinstance(o["p"], float) or not np.isnan(o["p"])
              else f"\n  {name}: OR not estimable")
        print(f"          rate(PLP)/rate(BLB) = {ev[lrkey]:.4f}")


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _versions() -> dict[str, str]:
    import hmmlearn
    import scipy
    import sklearn
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
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
