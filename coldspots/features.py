"""Feature normalisation and gene-completeness helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["prep_scores", "complete_genes"]


def prep_scores(scores: np.ndarray, name: str, *,
                invert: list[str], winsorize: tuple[float, float]) -> np.ndarray:
    """Orient, clip and min-max scale one feature within a single gene.

    ``invert`` features are sign-flipped so that higher = more deleterious;
    values are clipped to the ``winsorize`` percentiles and scaled to [0, 1].
    Scaling uses only the values given, i.e. one gene at a time.
    """
    scores = np.asarray(scores, dtype=float)
    if name in invert:
        scores = -scores
    lo_pct, hi_pct = winsorize
    p_low, p_high = np.percentile(scores, lo_pct), np.percentile(scores, hi_pct)
    clipped = np.clip(scores, p_low, p_high)
    return (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)


def complete_genes(df: pd.DataFrame, cols: list[str], gene_col: str) -> set[str]:
    """Genes with no missing values in every column of ``cols``."""
    complete = (df[cols].notna().all(axis=1)
                .groupby(df[gene_col], observed=True).all())
    return set(complete[complete].index)
