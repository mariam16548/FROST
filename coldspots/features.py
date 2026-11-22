"""Feature normalisation and gene-completeness helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["prep_scores", "complete_genes"]


def prep_scores(scores: np.ndarray, name: str, *,
                invert: list[str], winsorize: tuple[float, float]) -> np.ndarray:
    """Orient, winsorise and min-max scale a single feature vector.

    IMPORTANT (methods): normalisation is computed **within the vector it is
    given**. During training that vector is the training gene's residues;
    during detection it is the target gene's residues. Emission means are
    therefore in each gene's own normalised units, not a global scale. This is
    a deliberate property of the method and must be preserved to reproduce the
    published numbers - do not "fix" it by fitting the scaler globally.
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
    complete = df[cols].notna().all(axis=1).groupby(df[gene_col]).all()
    return set(complete[complete].index)
