"""Viterbi decoding and cold-spot interval extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import prep_scores
from .model import TrainedParams, build_fixed_hmm

__all__ = [
    "filter_short_regions", "extract_intervals", "decode_genes",
    "COLDSPOT_COLUMNS",
]

COLDSPOT_COLUMNS = ["gene", "aa_start", "aa_end", "length", "n_residues_scored"]


def filter_short_regions(cold_mask: np.ndarray, positions: np.ndarray,
                         min_length: int = 40) -> np.ndarray:
    """Drop cold runs spanning fewer than ``min_length`` amino-acid positions.

    Length is measured in aa coordinates (``end - start + 1``), not in scored
    residues. Residues with missing scores are absent from ``positions``, so a
    run can span 40 aa while containing fewer than 40 observations. The count
    of observations is reported separately as ``n_residues_scored``.
    """
    filtered = cold_mask.copy()
    in_cold, start = False, 0
    for i in range(len(cold_mask)):
        if cold_mask[i] and not in_cold:
            start, in_cold = i, True
        elif not cold_mask[i] and in_cold:
            if positions[i - 1] - positions[start] + 1 < min_length:
                filtered[start:i] = False
            in_cold = False
    if in_cold and positions[-1] - positions[start] + 1 < min_length:
        filtered[start:] = False
    return filtered


def extract_intervals(cold_mask: np.ndarray, pos: np.ndarray,
                      gene: str) -> list[dict]:
    """Convert a boolean per-residue mask into closed [aa_start, aa_end] runs."""
    intervals: list[dict] = []
    in_cold, start_pos, start_i = False, 0, 0

    def _close(end_i: int) -> dict:
        return {
            "gene": gene,
            "aa_start": int(start_pos),
            "aa_end": int(pos[end_i]),
            "length": int(pos[end_i]) - int(start_pos) + 1,
            "n_residues_scored": int(end_i - start_i + 1),
        }

    for i in range(len(cold_mask)):
        if cold_mask[i] and not in_cold:
            start_pos, start_i, in_cold = pos[i], i, True
        elif not cold_mask[i] and in_cold:
            intervals.append(_close(i - 1))
            in_cold = False
    if in_cold:
        intervals.append(_close(len(cold_mask) - 1))
    return intervals


def decode_genes(gene_set, params: TrainedParams, df: pd.DataFrame, *,
                 gene_col: str, pos_col: str, min_cold_run: int,
                 invert: list[str], winsorize: tuple[float, float],
                 verbose: bool = True) -> pd.DataFrame:
    """Run Viterbi decoding for every gene in ``gene_set``.

    Genes with fewer than ``2 * min_cold_run`` scored residues are skipped: a
    protein that short cannot support a cold/hot contrast at this resolution.
    """
    score_cols = params.features
    cold_state = params.cold_state
    model = build_fixed_hmm(params)

    all_intervals: list[dict] = []
    skipped: list[str] = []
    # Row indices per gene, computed once. Selecting each gene with a boolean
    # mask over the whole table is O(genes x rows) and dominated run time at
    # proteome scale.
    rows_by_gene = df.groupby(gene_col, sort=False).indices
    empty = np.array([], dtype=int)
    for gene in sorted(gene_set):
        gdf = df.iloc[rows_by_gene.get(gene, empty)].dropna(subset=score_cols)
        gdf = gdf.sort_values(pos_col).reset_index(drop=True)
        if len(gdf) < 2 * min_cold_run:
            skipped.append(gene)
            continue
        scores_mv = np.column_stack([
            prep_scores(gdf[col].values, col, invert=invert,
                        winsorize=winsorize)
            for col in score_cols
        ])
        pos = gdf[pos_col].values
        cold_mask = filter_short_regions(
            model.predict(scores_mv) == cold_state, pos, min_cold_run
        )
        all_intervals.extend(extract_intervals(cold_mask, pos, gene))

    if verbose:
        print(f"    {len(gene_set)} genes -> "
              f"{len(gene_set) - len(skipped)} processed, "
              f"{len(skipped)} skipped (<{2 * min_cold_run} scored residues) "
              f"| {len(all_intervals)} intervals")
    if not all_intervals:
        return pd.DataFrame(columns=COLDSPOT_COLUMNS)
    return pd.DataFrame(all_intervals)[COLDSPOT_COLUMNS]
