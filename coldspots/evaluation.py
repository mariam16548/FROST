"""Optional evaluation: ClinVar and MAVE odds ratios for the cold spots.

Implements the manuscript's benchmarking metrics. Each row of the input table
is one amino-acid substitution, and is one observation.

* **ClinVar** - BLB vs PLP inside vs outside cold spots, over genes with at
  least one predicted cold spot.
* **MAVE**    - functionally normal vs abnormal calls inside vs outside cold
  spots, over genes whose MAVE calls cover at least ``min_mave_positions``
  distinct positions with a non-indeterminate call inside their cold spots,
  summed across all cold spots in the gene.

Cold-spot membership is always recomputed from the intervals produced by the
current run.
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
    """Odds ratio for a benign, rather than pathogenic, call inside a cold spot.

    The 2x2 table is::

                        inside    outside
        BLB (benign)    blb_in    blb_out
        PLP (pathogen.) plp_in    plp_out

    so ``OR = (blb_in * plp_out) / (blb_out * plp_in)``. **OR > 1 means benign
    variants are over-represented inside cold spots**, i.e. cold spots are
    depleted of pathogenic variation. Report the direction explicitly; the
    reciprocal convention is equally common in the literature.

    The point estimate and the 95% Woolf (log-scale) confidence interval use
    the Haldane-Anscombe correction (+0.5 to every cell) so they remain finite
    when a cell is empty. The p-value is a two-sided Fisher's exact test on
    the uncorrected counts.
    """
    blb_in = summary["BLB"]["inside"]
    blb_out = summary["BLB"]["total"] - blb_in
    plp_in = summary["PLP"]["inside"]
    plp_out = summary["PLP"]["total"] - plp_in
    if (blb_in + blb_out == 0) or (plp_in + plp_out == 0):
        return {"OR": NA, "OR_lo": NA, "OR_hi": NA, "p": NA,
                "table": [[blb_in, blb_out], [plp_in, plp_out]]}

    _, p = fisher_exact(np.array([[blb_in, blb_out], [plp_in, plp_out]]))
    a, b, c, d = blb_in + 0.5, blb_out + 0.5, plp_in + 0.5, plp_out + 0.5
    or_val = (a * d) / (b * c)
    log_or = np.log(or_val)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {
        "OR": round(or_val, 2),
        "OR_lo": round(float(np.exp(log_or - 1.96 * se)), 2),
        "OR_hi": round(float(np.exp(log_or + 1.96 * se)), 2),
        "p": float(p),
        "table": [[blb_in, blb_out], [plp_in, plp_out]],
    }


def in_intervals(df: pd.DataFrame, intervals_df: pd.DataFrame, *,
                 gene_col: str, aa_col: str) -> np.ndarray:
    """Boolean mask: is each row's (gene, aa_pos) inside any interval?

    Endpoints are inclusive. Vectorised with one ``searchsorted`` per gene, so
    it scales linearly with the number of rows. Overlapping intervals are
    handled (the running maximum of interval ends is used), although the
    intervals produced by decoding never overlap within a gene.
    """
    mask = np.zeros(len(df), dtype=bool)
    if len(df) == 0 or intervals_df is None or intervals_df.empty:
        return mask
    pos_all = df[aa_col].to_numpy(dtype=float)
    row_groups = df.groupby(gene_col, sort=False).indices
    for gene, iv in intervals_df.groupby("gene", sort=False):
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
    """Count labelled substitutions inside vs. outside the predicted cold spots.

    Only genes carrying at least one cold spot contribute; a gene with no
    predicted cold spot has no inside/outside contrast to offer.
    """
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
    """Genes with enough MAVE evidence inside their cold spots.

    A gene qualifies if its MAVE calls in ``valid_labels`` (abnormal or normal,
    i.e. non-zero ExCALIBR points) cover at least ``min_count`` distinct
    amino-acid positions inside its predicted cold spots, summed across all
    of its cold spots.
    """
    calls = (mave_df[mave_df[label_col].isin(valid_labels)]
             [[gene_col, pos_col]].drop_duplicates())
    inside = in_intervals(calls, coldspot_df, gene_col=gene_col, aa_col=pos_col)
    counts = calls[inside].groupby(gene_col, observed=True).size()
    return set(counts[counts >= min_count].index.astype(str))


def _rate_ratio(summary: dict) -> float:
    """LR-: rate(PLP inside) / rate(BLB inside)."""
    blb = summary["BLB"]["rate"]
    if not (isinstance(blb, float) and blb > 0) or np.isnan(blb):
        return NA
    return summary["PLP"]["rate"] / blb


def evaluate(coldspot_df: pd.DataFrame, clinvar_df: pd.DataFrame,
             mave_df: pd.DataFrame, *, cfg, mave_col: str) -> dict | None:
    """Run both evaluations and return a results dict."""
    if coldspot_df is None or coldspot_df.empty:
        return None

    cols, ev = cfg.columns, cfg.evaluation
    gene_col, pos_col = cols.gene, cols.aa_pos

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
        "lr_clinvar": _rate_ratio(cv),
        "lr_mave": _rate_ratio(mv),
        "n_intervals": int(len(coldspot_df)),
        "n_mave_intervals": int(len(cs_mave)),
        "n_genes": int(coldspot_df["gene"].nunique()),
        "n_mave_genes": len(genes),
        "mave_genes": sorted(genes),
        "total_aas": int(coldspot_df["length"].sum()),
        "total_mave_aas": int(cs_mave["length"].sum()),
        "vus_count": cv.get("VUS", {}).get("inside", 0),
        "clinvar_summary": cv,
        "mave_summary": mv,
    }
