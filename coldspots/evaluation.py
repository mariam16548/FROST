"""Cold-spot evaluation: membership, enrichment, odds ratios.

Two independent evaluations run against the same variant-level input
table:

* **ClinVar** - uses the clinical significance column, over the full cold-spot
  set.
* **MAVE**    - uses the functional-assay call column, restricted to cold spots
  with enough assayed positions to support a claim.

Cold-spot membership is always recomputed from the intervals produced by the
current run. Any ``in_coldspot``-style column shipped with the input table is
dropped on load and never consulted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

__all__ = [
    "odds_ratio", "in_intervals", "coldspot_inclusion",
    "filter_by_mave_support", "evaluate",
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

    The point estimate and Wald CI use the Haldane-Anscombe correction (+0.5 to
    every cell) so they remain finite when a cell is empty. Fisher's exact
    p-value is computed on the *uncorrected* counts, which is the conventional
    pairing but means p and the CI are not derived from an identical table.
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


def _collapse_to_positions(df: pd.DataFrame, gene_col: str, aa_col: str,
                           label_col: str) -> pd.DataFrame:
    """One row per (gene, aa_pos, label) - the pseudo-replication control.

    The input table holds one row per substitution, so a residue with several
    classified substitutions contributes several rows and Fisher's exact test
    treats them as independent. Collapsing removes that inflation. A position
    with both a PLP and a BLB call is retained once per label.
    """
    return df.drop_duplicates(subset=[gene_col, aa_col, label_col])


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
                       label_map: dict[str, str],
                       unit: str = "variant",
                       exclude_df: pd.DataFrame | None = None) -> dict:
    """Count labelled observations inside vs. outside the predicted cold spots.

    Only genes carrying at least one cold spot contribute; a gene with no
    predicted cold spot has no inside/outside contrast to offer.

    ``exclude_df`` lists intervals whose positions are removed from **both**
    cells before counting. It is used by the interval-level MAVE evaluation:
    a cold spot that fails the assay-support filter is neither "inside" (it was
    not retained) nor "outside" (it is still a predicted cold spot), so its
    variants must not be counted at all.
    """
    coldspot_genes = set(coldspot_df["gene"].unique())
    src = source_df[source_df[gene_col].isin(coldspot_genes)]
    if exclude_df is not None and not exclude_df.empty:
        drop = in_intervals(src, exclude_df, gene_col=gene_col, aa_col=aa_col)
        src = src[~drop]
    if unit == "position":
        src = _collapse_to_positions(src, gene_col, aa_col, label_col)
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


def filter_by_mave_support(coldspot_df: pd.DataFrame, mave_df: pd.DataFrame, *,
                           min_count: int, min_coverage: float, level: str,
                           label_col: str, gene_col: str, pos_col: str,
                           valid_labels: tuple[str, ...] = ("PLP", "BLB"),
                           ) -> pd.DataFrame:
    """Keep cold spots with enough functional-assay evidence to interpret.

    "Enough" means at least ``min_count`` **distinct aa positions** carrying a
    call in ``valid_labels`` (the raw labels for PLP and BLB, in that order),
    and optionally at least ``min_coverage`` of the region assayed at all.
    Positions are deduplicated before counting because the input table has
    one row per substitution.

    ``level="interval"``
        every cold spot must clear the threshold on its own.
    ``level="gene"``
        the threshold applies to the **sum over a gene's cold spots**, and all
        cold spots in a passing gene are retained - including ones with one,
        two, or zero assayed positions. Cold spots within a gene are disjoint,
        so summing distinct-position counts equals counting their union. This
        is the looser of the two settings; report which was used.

    Columns added per region:
        ``n_PLP`` / ``n_BLB``  distinct aa positions with that call
        ``n_positions``        distinct aa positions with any valid call
        ``n_assayed``          distinct aa positions with any call, incl. IR
        ``frac_assayed``       ``n_assayed / length``
    plus, when ``level="gene"``, the gene-level totals actually thresholded:
        ``gene_n_positions``, ``gene_n_assayed``, ``gene_frac_assayed``

    ``frac_assayed`` matters: a region can clear ``min_count`` on a handful of
    clustered residues at one edge, so quote it alongside any depletion claim.
    """
    if level not in ("interval", "gene"):
        raise ValueError(f"level must be 'interval' or 'gene', got {level!r}")
    if coldspot_df.empty:
        return coldspot_df.copy()

    # One row per (gene, aa_pos, label) with a call we score on.
    valid = (mave_df[mave_df[label_col].isin(valid_labels)]
             [[gene_col, pos_col, label_col]]
             .drop_duplicates())

    c = coldspot_df.reset_index().rename(columns={"index": "_orig_idx"})
    merged = c.merge(valid, left_on="gene", right_on=gene_col, how="left")
    hits = merged[(merged[pos_col] >= merged["aa_start"]) &
                  (merged[pos_col] <= merged["aa_end"])]

    counts = (
        hits.groupby(["_orig_idx", label_col])[pos_col].nunique()
            .unstack(fill_value=0)
            .reindex(columns=list(valid_labels), fill_value=0)
    )
    canon = {raw: name for name, raw in zip(("PLP", "BLB"), valid_labels)}
    counts.columns = [f"n_{canon.get(lbl, lbl)}" for lbl in counts.columns]
    counts["n_positions"] = hits.groupby("_orig_idx")[pos_col].nunique()

    # Assay coverage: any MAVE call at all, whatever the label.
    any_call = mave_df[[gene_col, pos_col]].drop_duplicates()
    m_any = c.merge(any_call, left_on="gene", right_on=gene_col, how="left")
    in_any = ((m_any[pos_col] >= m_any["aa_start"]) &
              (m_any[pos_col] <= m_any["aa_end"]))
    counts["n_assayed"] = m_any[in_any].groupby("_orig_idx")[pos_col].nunique()

    count_cols = counts.columns.tolist()
    out = coldspot_df.join(counts, how="left").fillna(
        {col: 0 for col in count_cols}
    )
    for col in count_cols:
        out[col] = out[col].astype(int)
    out["frac_assayed"] = out["n_assayed"] / out["length"]

    if level == "gene":
        g = out.groupby("gene")
        out["gene_n_positions"] = g["n_positions"].transform("sum")
        out["gene_n_assayed"] = g["n_assayed"].transform("sum")
        out["gene_frac_assayed"] = (out["gene_n_assayed"] /
                                    g["length"].transform("sum"))
        keep = ((out["gene_n_positions"] >= min_count) &
                (out["gene_frac_assayed"] >= min_coverage))
    else:
        keep = ((out["n_positions"] >= min_count) &
                (out["frac_assayed"] >= min_coverage))
    return out[keep].copy()


def _rate_ratio(summary: dict) -> float:
    """rate(PLP inside) / rate(BLB inside).

    Reported as ``LR-`` in the original script. Read it as the likelihood ratio
    for cold-spot membership given a pathogenic vs. a benign variant: values
    below 1 mean membership argues against pathogenicity.
    """
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
        label_map=ev.clinvar_labels, unit=ev.unit,
    )

    cs_mave = filter_by_mave_support(
        coldspot_df, mave_df,
        min_count=ev.min_mave_positions, min_coverage=ev.min_mave_coverage,
        level=ev.mave_filter_level, label_col=mave_col,
        gene_col=gene_col, pos_col=pos_col,
        valid_labels=(ev.mave_labels["PLP"], ev.mave_labels["BLB"]),
    )
    if cs_mave.empty:
        mv = {lbl: {"inside": 0, "total": 0, "rate": NA}
              for lbl in ev.mave_labels}
        or_mv = {"OR": NA, "OR_lo": NA, "OR_hi": NA, "p": NA, "table": None}
    else:
        # Cold spots that failed the support filter in a gene that still has
        # a passing one (only possible with level="interval"): drop their
        # variants from both cells rather than counting them as "outside".
        failed = coldspot_df.loc[~coldspot_df.index.isin(cs_mave.index)]
        failed = failed[failed["gene"].isin(set(cs_mave["gene"]))]
        mv = coldspot_inclusion(
            source_df=mave_df, coldspot_df=cs_mave,
            gene_col=gene_col, aa_col=pos_col, label_col=mave_col,
            label_map=ev.mave_labels, unit=ev.unit, exclude_df=failed,
        )
        or_mv = odds_ratio(mv)

    return {
        "unit": ev.unit,
        "or_clinvar": odds_ratio(cv),
        "or_mave": or_mv,
        "lr_clinvar": _rate_ratio(cv),
        "lr_mave": _rate_ratio(mv),
        "n_intervals": int(len(coldspot_df)),
        "n_mave_intervals": int(len(cs_mave)),
        "n_genes": int(coldspot_df["gene"].nunique()),
        "n_mave_genes": int(cs_mave["gene"].nunique()) if not cs_mave.empty else 0,
        "total_aas": int(coldspot_df["length"].sum()),
        "total_mave_aas": int(cs_mave["length"].sum()) if not cs_mave.empty else 0,
        "assayed_mave_aas": int(cs_mave["n_assayed"].sum()) if not cs_mave.empty else 0,
        "vus_count": cv.get("VUS", {}).get("inside", 0),
        "clinvar_summary": cv,
        "mave_summary": mv,
        "coldspot_df_mave": cs_mave,
    }
