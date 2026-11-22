"""Benchmarking against ClinVar and MAVE data (``--evaluate``).

Each row of the input table is one amino-acid substitution and counts once.

* ClinVar: benign (BLB) vs pathogenic (PLP) substitutions inside vs outside
  coldspots, in genes with at least one coldspot.
* MAVE: normal vs abnormal substitutions inside vs outside coldspots, in genes
  whose normal/abnormal calls cover at least ``min_mave_positions`` distinct
  positions inside their coldspots, summed across all of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

__all__ = [
    "odds_ratio", "in_intervals", "coldspot_inclusion",
    "mave_genes", "evaluate",
]

NA = float("nan")


def odds_ratio(summary: dict) -> dict:
    """Odds ratio for a benign rather than pathogenic call inside a coldspot.

    ``OR = (blb_in * plp_out) / (blb_out * plp_in)``, so OR > 1 means benign
    substitutions are enriched inside coldspots. The OR and its 95% Woolf
    confidence interval use a Haldane-Anscombe correction (+0.5 to every
    cell); the p-value is a two-sided Fisher's exact test on the uncorrected
    counts.
    """
    blb_in = summary["BLB"]["inside"]
    blb_out = summary["BLB"]["total"] - blb_in
    plp_in = summary["PLP"]["inside"]
    plp_out = summary["PLP"]["total"] - plp_in
    table = [[blb_in, blb_out], [plp_in, plp_out]]
    if (blb_in + blb_out == 0) or (plp_in + plp_out == 0):
        return {"OR": NA, "OR_lo": NA, "OR_hi": NA, "p": NA, "table": table}

    _, p = fisher_exact(np.array(table))
    a, b, c, d = blb_in + 0.5, blb_out + 0.5, plp_in + 0.5, plp_out + 0.5
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {
        "OR": float(np.exp(log_or)),
        "OR_lo": float(np.exp(log_or - 1.96 * se)),
        "OR_hi": float(np.exp(log_or + 1.96 * se)),
        "p": float(p),
        "table": table,
    }


def in_intervals(df: pd.DataFrame, intervals_df: pd.DataFrame, *,
                 gene_col: str, aa_col: str) -> np.ndarray:
    """Boolean mask: is each row's (gene, aa_pos) inside any interval?

    Endpoints are inclusive. Uses one ``searchsorted`` per gene; overlapping
    intervals are handled.
    """
    mask = np.zeros(len(df), dtype=bool)
    if len(df) == 0 or intervals_df is None or intervals_df.empty:
        return mask
    pos_all = df[aa_col].to_numpy(dtype=float)
    row_groups = df.groupby(gene_col, sort=False, observed=True).indices
    for gene, iv in intervals_df.groupby("gene", sort=False, observed=True):
        rows = row_groups.get(gene)
        if rows is None:
            continue
        iv = iv.sort_values("aa_start")
        starts = iv["aa_start"].to_numpy(dtype=float)
        ends = np.maximum.accumulate(iv["aa_end"].to_numpy(dtype=float))
        pos = pos_all[rows]
        k = np.searchsorted(starts, pos, side="right") - 1
        hit = k >= 0
        hit[hit] = pos[hit] <= ends[k[hit]]
        mask[rows] = hit
    return mask


def coldspot_inclusion(*, source_df: pd.DataFrame, coldspot_df: pd.DataFrame,
                       gene_col: str, aa_col: str, label_col: str,
                       label_map: dict[str, str]) -> dict:
    """Count labelled substitutions inside vs outside coldspots, in genes
    with at least one coldspot."""
    coldspot_genes = set(coldspot_df["gene"].unique())
    src = source_df[source_df[gene_col].isin(coldspot_genes)]
    src = src.assign(_in_coldspot=in_intervals(
        src, coldspot_df, gene_col=gene_col, aa_col=aa_col))

    summary = {}
    for canon, raw in label_map.items():
        sub = src[src[label_col] == raw]
        inside, total = int(sub["_in_coldspot"].sum()), int(len(sub))
        summary[canon] = {
            "inside": inside,
            "total": total,
            "rate": inside / total if total > 0 else NA,
        }
    return summary


def mave_genes(coldspot_df: pd.DataFrame, mave_df: pd.DataFrame, *,
               min_count: int, label_col: str, gene_col: str, pos_col: str,
               valid_labels: tuple[str, str]) -> set[str]:
    """Genes whose ``valid_labels`` calls (abnormal or normal) cover at least
    ``min_count`` distinct positions inside their coldspots."""
    calls = (mave_df[mave_df[label_col].isin(valid_labels)]
             [[gene_col, pos_col]].drop_duplicates())
    inside = in_intervals(calls, coldspot_df, gene_col=gene_col, aa_col=pos_col)
    counts = calls[inside].groupby(gene_col, observed=True).size()
    return set(counts[counts >= min_count].index.astype(str))


def _negative_lr(summary: dict) -> float:
    """LR-: fraction of PLP inside coldspots / fraction of BLB inside."""
    blb = summary["BLB"]["rate"]
    return summary["PLP"]["rate"] / blb if blb > 0 else NA


def evaluate(coldspot_df: pd.DataFrame, clinvar_df: pd.DataFrame,
             mave_df: pd.DataFrame, *, cfg) -> dict | None:
    """Run both evaluations; None if there are no coldspots."""
    if coldspot_df is None or coldspot_df.empty:
        return None

    cols, ev = cfg.columns, cfg.evaluation
    gene_col, pos_col, mave_col = cols.gene, cols.aa_pos, cols.mave

    cv = coldspot_inclusion(
        source_df=clinvar_df, coldspot_df=coldspot_df,
        gene_col=gene_col, aa_col=pos_col, label_col=cols.clinvar,
        label_map=ev.clinvar_labels,
    )

    genes = mave_genes(
        coldspot_df, mave_df, min_count=ev.min_mave_positions,
        label_col=mave_col, gene_col=gene_col, pos_col=pos_col,
        valid_labels=(ev.mave_labels["PLP"], ev.mave_labels["BLB"]),
    )
    cs_mave = coldspot_df[coldspot_df["gene"].astype(str).isin(genes)]
    if cs_mave.empty:
        mv = {lbl: {"inside": 0, "total": 0, "rate": NA}
              for lbl in ev.mave_labels}
        or_mv = {"OR": NA, "OR_lo": NA, "OR_hi": NA, "p": NA, "table": None}
    else:
        mv = coldspot_inclusion(
            source_df=mave_df, coldspot_df=cs_mave,
            gene_col=gene_col, aa_col=pos_col, label_col=mave_col,
            label_map=ev.mave_labels,
        )
        or_mv = odds_ratio(mv)

    return {
        "or_clinvar": odds_ratio(cv),
        "or_mave": or_mv,
        "lr_clinvar": _negative_lr(cv),
        "lr_mave": _negative_lr(mv),
        "n_intervals": int(len(coldspot_df)),
        "n_mave_intervals": int(len(cs_mave)),
        "n_genes": int(coldspot_df["gene"].nunique()),
        "n_mave_genes": len(genes),
        "mave_genes": sorted(genes),
        "total_aas": int(coldspot_df["length"].sum()),
        "total_mave_aas": int(cs_mave["length"].sum()),
        "clinvar_summary": cv,
        "mave_summary": mv,
    }
