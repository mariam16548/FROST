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


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end indices (inclusive) of each run of True in ``mask``."""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(m[1:] != m[:-1])
    return edges[0::2], edges[1::2] - 1


def filter_short_regions(cold_mask: np.ndarray, positions: np.ndarray,
                         min_length: int = 40) -> np.ndarray:
    """Drop cold runs spanning fewer than ``min_length`` amino-acid positions.

    Length is measured in aa coordinates (``end - start + 1``), not in scored
    residues. Residues missing from the table are absent from ``positions``,
    so a run can span 40 aa while containing fewer than 40 observations. The
    count of observations is reported separately as ``n_residues_scored``.
    """
    positions = np.asarray(positions)
    filtered = np.zeros(len(cold_mask), dtype=bool)
    for a, b in zip(*_runs(cold_mask)):
        if positions[b] - positions[a] + 1 >= min_length:
            filtered[a:b + 1] = True
    return filtered


def extract_intervals(cold_mask: np.ndarray, pos: np.ndarray,
                      gene: str) -> list[dict]:
    """Convert a boolean per-residue mask into closed [aa_start, aa_end] runs."""
    pos = np.asarray(pos)
    return [
        {
            "gene": gene,
            "aa_start": int(pos[a]),
            "aa_end": int(pos[b]),
            "length": int(pos[b]) - int(pos[a]) + 1,
            "n_residues_scored": int(b - a + 1),
        }
        for a, b in zip(*_runs(cold_mask))
    ]


def decode_genes(gene_set, params: TrainedParams, df: pd.DataFrame, *,
                 gene_col: str, pos_col: str, min_cold_run: int,
                 invert: list[str], winsorize: tuple[float, float],
                 verbose: bool = True) -> pd.DataFrame:
    """Run Viterbi decoding for every gene in ``gene_set``.

    Genes with fewer than ``2 * min_cold_run`` scored residues are skipped: a
    protein that short cannot support a Cold/Not-Cold contrast at this resolution.
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
