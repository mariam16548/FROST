"""End-to-end pipeline: load, train, detect, (optionally) evaluate, write."""
from __future__ import annotations

import gzip
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from .config import Config
from .evaluation import evaluate, in_intervals
from .features import complete_genes
from .intervals import COLDSPOT_COLUMNS, decode_genes
from .model import marginalize_params, train_params_mv

__all__ = ["run", "load_data", "detect"]

_CHUNK_ROWS = 1_000_000
_PROTEIN_VARIANT = r"^[A-Za-z*]{1,3}(\d+)"   # "C2R" / "Cys2Arg" -> 2


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@dataclass
class LoadedData:
    residues: pd.DataFrame       # one row per (gene, position), with features
    transcripts: dict[str, str]  # gene -> transcript (empty if no column)
    clinvar_eval: pd.DataFrame | None = None  # rows with a ClinVar call
    mave_eval: pd.DataFrame | None = None     # rows with a MAVE call


def _positions(chunk: pd.DataFrame, cfg: Config, has_pos: bool) -> pd.Series:
    """Integer amino-acid positions for a chunk of the input table."""
    cols = cfg.columns
    if has_pos:
        values = chunk[cols.aa_pos]
        pos = pd.to_numeric(values, errors="coerce")
        bad = pos.isna() | (pos % 1 != 0)
        source = cols.aa_pos
    else:
        values = chunk[cols.protein_variant]
        pos = pd.to_numeric(values.str.extract(_PROTEIN_VARIANT)[0])
        bad = pos.isna()
        source = cols.protein_variant
    if bad.any():
        raise ValueError(
            f"Could not read an amino-acid position from {source!r} values "
            f"such as {values[bad].head(3).tolist()}")
    return pos.astype(np.int64)


def read_variant_table(cfg: Config) -> pd.DataFrame:
    """Read the columns FROST needs, with an integer ``aa_pos`` column."""
    cols, feats = cfg.columns, cfg.features
    path = cfg.data.variant_csv
    header = pd.read_csv(path, nrows=0).columns
    has_pos = cols.aa_pos in header
    labels = [cols.clinvar, cols.mave] if cfg.evaluation.enabled else []
    transcript = [cols.transcript] if cols.transcript in header else []
    needed = list(dict.fromkeys(
        [cols.gene, cols.aa_pos if has_pos else cols.protein_variant]
        + transcript + labels + feats.primary))
    missing = [c for c in needed if c not in header]
    if missing:
        raise KeyError(f"{path}: missing columns {missing} "
                       f"(see docs/DATA.md)")

    # Read in chunks, keeping only the parsed integer position, so the
    # protein_variant strings never all sit in memory. Gene and label columns
    # repeat few values over millions of rows, so they are categoricals.
    cat_cols = list(dict.fromkeys([cols.gene] + transcript + labels))
    parts = []
    for chunk in pd.read_csv(path, usecols=needed, chunksize=_CHUNK_ROWS,
                             low_memory=False,
                             dtype={c: "category" for c in cat_cols}):
        if chunk[cols.gene].isna().any():
            raise ValueError(f"{path}: some rows have no {cols.gene!r}")
        # Give every chunk object-dtype categories so they can be unified
        # (an all-empty column would otherwise infer a different dtype).
        for c in cat_cols:
            chunk[c] = chunk[c].cat.set_categories(
                chunk[c].cat.categories.astype(object))
        chunk[cols.aa_pos] = _positions(chunk, cfg, has_pos)
        if not has_pos:
            chunk = chunk.drop(columns=cols.protein_variant)
        parts.append(chunk)
    if not parts:
        raise ValueError(f"{path}: the table has no rows")

    cats = {c: union_categoricals([p[c] for p in parts], sort_categories=True)
            for c in cat_cols}
    df = pd.concat([p.drop(columns=cat_cols) for p in parts],
                   ignore_index=True)
    for c in cat_cols:
        df[c] = cats[c]
    return df


def residue_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per (gene, position). The features must be per-residue."""
    cols, feats = cfg.columns, cfg.features
    key = [cols.gene, cols.aa_pos]
    residues = df.drop_duplicates(key)
    n_distinct = len(df.drop_duplicates(key + feats.primary))
    if n_distinct != len(residues):
        raise ValueError(
            f"{n_distinct - len(residues)} residues have different "
            f"{feats.primary} values across their substitutions; these "
            f"must be per-residue scores.")
    return (residues[key + feats.primary]
            .sort_values(key).reset_index(drop=True))


def _transcripts(df: pd.DataFrame, cfg: Config) -> dict[str, str]:
    cols = cfg.columns
    if cols.transcript not in df.columns:
        return {}
    pairs = df[[cols.gene, cols.transcript]].drop_duplicates()
    multi = pairs[cols.gene].duplicated(keep=False)
    if multi.any():
        raise ValueError(
            f"Expected one {cols.transcript!r} per gene; e.g. "
            f"{sorted(pairs.loc[multi, cols.gene].astype(str).unique())[:3]} "
            f"have several.")
    return dict(zip(pairs[cols.gene].astype(str),
                    pairs[cols.transcript].astype(str)))


def load_data(cfg: Config) -> LoadedData:
    cols = cfg.columns
    print(f"Reading {cfg.data.variant_csv} ...")
    df = read_variant_table(cfg)
    residues = residue_table(df, cfg)
    print(f"  {residues[cols.gene].nunique()} genes, {len(residues):,} "
          f"residues, {len(df):,} substitutions")
    data = LoadedData(residues, _transcripts(df, cfg))
    if cfg.evaluation.enabled:
        evaluable = df[~df[cols.gene].isin(cfg.exclude_genes)]
        data.clinvar_eval = evaluable[evaluable[cols.clinvar].notna()]
        data.mave_eval = evaluable[evaluable[cols.mave].notna()]
    return data


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def _train(cfg: Config, residues: pd.DataFrame):
    """Fit the model on the training gene, and its reduced (no-EVE) form."""
    cols, feats = cfg.columns, cfg.features
    genes_full = complete_genes(residues, feats.primary, cols.gene)
    genes_reduced = complete_genes(
        residues, cfg.fallback_features, cols.gene) - genes_full

    not_ready = [g for g in cfg.training_intervals if g not in genes_full]
    if not_ready:
        raise ValueError(f"Training gene(s) {not_ready} are missing or lack "
                         f"complete scores for {feats.primary}.")

    full = train_params_mv(
        residues, feats.primary, gene_col=cols.gene, pos_col=cols.aa_pos,
        training_intervals=cfg.training_intervals, invert=feats.invert,
        winsorize=feats.winsorize)
    reduced = marginalize_params(full, cfg.fallback_keep_indices)

    n_skipped = residues[cols.gene].nunique() - len(genes_full) \
        - len(genes_reduced)
    print(f"\nTrained on {', '.join(cfg.training_intervals)}. Genes: "
          f"{len(genes_full)} with all features, {len(genes_reduced)} "
          f"without {'/'.join(feats.fallback_drop)}, {n_skipped} with other "
          f"missing scores (not analysed).")
    return [(genes_full - cfg.exclude_genes, full),
            (genes_reduced - cfg.exclude_genes, reduced)]


def detect(cfg: Config, residues: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Find coldspots in every gene. Returns them and the fitted models."""
    runs = _train(cfg, residues)
    frames = []
    for gene_set, params in runs:
        cdf = decode_genes(
            gene_set, params, residues,
            gene_col=cfg.columns.gene, pos_col=cfg.columns.aa_pos,
            min_cold_run=cfg.detection.min_cold_run,
            invert=cfg.features.invert, winsorize=cfg.features.winsorize,
        )
        if not cdf.empty:
            frames.append(cdf)
    if not frames:
        return pd.DataFrame(columns=COLDSPOT_COLUMNS), runs
    coldspots = (pd.concat(frames, ignore_index=True)
                 .sort_values(["gene", "aa_start"]).reset_index(drop=True))
    print(f"Found {len(coldspots)} coldspots in "
          f"{coldspots['gene'].nunique()} genes "
          f"({int(coldspots['length'].sum()):,} residues).")
    return coldspots, runs


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _format_p(p: float) -> str:
    # Fisher p-values for very large tables underflow to exactly 0.0.
    return "p < 1e-300" if p == 0 else f"p = {p:.3g}"


def report(cfg: Config, ev: dict | None) -> None:
    """Print the benchmarking results."""
    if ev is None:
        print("\nNo coldspots found; nothing to evaluate.")
        return
    print(f"\nBenchmarking ({ev['n_genes']} genes with coldspots; MAVE: "
          f"{ev['n_mave_genes']} genes with >= "
          f"{cfg.evaluation.min_mave_positions} assayed positions inside "
          f"their coldspots)")
    for name, summary, order in (
        ("ClinVar", ev["clinvar_summary"], ("BLB", "PLP", "VUS")),
        ("MAVE", ev["mave_summary"], ("BLB", "PLP", "IR")),
    ):
        print(f"  {name}, inside coldspots / total:")
        for lbl in order:
            if lbl in summary:
                s = summary[lbl]
                pct = "" if np.isnan(s["rate"]) else f"  ({s['rate']:.1%})"
                print(f"    {lbl:<4} {s['inside']:>9,} / {s['total']:>9,}{pct}")
    for name, key, lr in (("ClinVar", "or_clinvar", "lr_clinvar"),
                          ("MAVE", "or_mave", "lr_mave")):
        o = ev[key]
        if np.isnan(o["p"]):
            print(f"  {name}: OR not estimable")
        else:
            print(f"  {name}: OR = {o['OR']:.2f} (95% CI {o['OR_lo']:.2f}-"
                  f"{o['OR_hi']:.2f}), {_format_p(o['p'])}, "
                  f"LR- = {ev[lr]:.4f}")


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def _git_commit() -> str | None:
    """Commit of this checkout, with "-dirty" if files are modified."""
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
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    return obj


def regions_table(coldspots: pd.DataFrame, cfg: Config,
                  transcripts: dict[str, str]) -> pd.DataFrame:
    """Coldspots as gene, transcript, aa_start, aa_end."""
    cols = cfg.columns
    out = pd.DataFrame({cols.gene: coldspots["gene"].astype(str)})
    if transcripts:
        out[cols.transcript] = out[cols.gene].map(transcripts)
    out["aa_start"] = coldspots["aa_start"].astype(int).values
    out["aa_end"] = coldspots["aa_end"].astype(int).values
    return out


def write_variant_table(path: Path, cfg: Config,
                        coldspots: pd.DataFrame) -> None:
    """Write every input row with gene_has_coldspot and variant_in_coldspot.

    The input is streamed in chunks. Both columns are blank for training
    genes, which are never analysed.
    """
    cols = cfg.columns
    header = pd.read_csv(cfg.data.variant_csv, nrows=0).columns
    has_pos = cols.aa_pos in header
    pos_col = cols.aa_pos if has_pos else cols.protein_variant
    keep = [c for c in (cols.transcript, cols.gene, pos_col) if c in header]
    cold_genes = set(coldspots["gene"].astype(str))
    with gzip.open(path, "wt", newline="", compresslevel=6) as fh:
        for i, chunk in enumerate(pd.read_csv(
                cfg.data.variant_csv, usecols=keep, chunksize=_CHUNK_ROWS,
                dtype={c: str for c in keep})):
            genes = chunk[cols.gene]
            probe = pd.DataFrame({"gene": genes.values,
                                  "pos": _positions(chunk, cfg, has_pos).values})
            training = genes.isin(cfg.exclude_genes).values
            has = pd.array(genes.isin(cold_genes).values, dtype="boolean")
            ins = pd.array(in_intervals(probe, coldspots, gene_col="gene",
                                        aa_col="pos"), dtype="boolean")
            has[training] = pd.NA
            ins[training] = pd.NA
            chunk.assign(gene_has_coldspot=has, variant_in_coldspot=ins).to_csv(
                fh, index=False, header=(i == 0), na_rep="")


def write_outputs(cfg: Config, coldspots: pd.DataFrame, ev: dict | None,
                  runs: list, data: LoadedData) -> dict[str, Path]:
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.output.prefix
    written = {
        "regions": out_dir / f"{p}_regions.tsv",
        "variants": out_dir / f"{p}_variants.csv.gz",
    }
    regions_table(coldspots, cfg, data.transcripts).to_csv(
        written["regions"], sep="\t", index=False)
    write_variant_table(written["variants"], cfg, coldspots)
    if ev is not None:
        written["metrics"] = out_dir / f"{p}_metrics.json"
        written["metrics"].write_text(json.dumps(_jsonable(ev), indent=2))

    written["manifest"] = out_dir / f"{p}_manifest.json"
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config_fingerprint": cfg.fingerprint(),
        "config": _jsonable(cfg.to_dict()),
        "package_versions": _versions(),
        "fitted_models": [params.summary() for _, params in runs],
        "n_coldspots": int(len(coldspots)),
        "n_genes_with_coldspots": int(coldspots["gene"].nunique()),
    }
    written["manifest"].write_text(json.dumps(manifest, indent=2, default=str))

    print("\nWrote:")
    for path in written.values():
        print(f"  {path}")
    return written


# ----------------------------------------------------------------------------
def run(cfg: Config) -> dict:
    """Run FROST end to end and return the coldspots, metrics and file paths."""
    data = load_data(cfg)
    coldspots, runs = detect(cfg, data.residues)
    ev = None
    if cfg.evaluation.enabled:
        ev = evaluate(coldspots, data.clinvar_eval, data.mave_eval, cfg=cfg)
        report(cfg, ev)
    written = write_outputs(cfg, coldspots, ev, runs, data)
    return {"coldspots": coldspots, "eval": ev, "written": written}
