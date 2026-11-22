"""Two-state (Cold / Not-Cold) Gaussian HMM.

Parameters are computed once from the labelled training gene and then held
fixed; they are never re-estimated on other genes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm

from .features import prep_scores

__all__ = [
    "train_params_1d", "train_params_mv", "marginalize_params",
    "build_fixed_hmm", "TrainedParams", "COLD_STATE",
]


COLD_STATE = 0  # by construction: state 0 is the labelled training region


class TrainedParams:
    """Fixed parameters of the two-state model. State 0 is Cold."""

    __slots__ = ("startprob", "transmat", "means", "covars", "features")

    def __init__(self, startprob, transmat, means, covars, features):
        self.startprob = startprob
        self.transmat = transmat
        self.means = means
        self.covars = covars
        self.features = list(features)

    @property
    def cold_state(self) -> int:
        """Index of the Cold state (always 0)."""
        return COLD_STATE

    def summary(self) -> dict:
        return {
            "features": self.features,
            "startprob": self.startprob.round(6).tolist(),
            "transmat": self.transmat.round(6).tolist(),
            "means": self.means.round(6).tolist(),
            "variances": np.array(
                [np.diag(c) for c in self.covars]
            ).round(6).tolist(),
            "cold_state": self.cold_state,
        }


def train_params_1d(df: pd.DataFrame, score_col: str, *, gene_col: str,
                    pos_col: str, training_intervals: dict,
                    invert: list[str], winsorize: tuple[float, float]):
    """Estimate start, transition and per-state Gaussian moments for one feature.

    State 0 (Cold) is the labelled ``training_intervals``; state 1
    (Not-Cold) is the rest of the training gene. Transition counts use add-one
    (Laplace) smoothing.
    """
    trans = np.zeros((2, 2))
    state_vals: dict[int, list[np.ndarray]] = {0: [], 1: []}

    for gene, intervals in training_intervals.items():
        gdf = df[df[gene_col] == gene].dropna(subset=[score_col])
        gdf = gdf.sort_values(pos_col).reset_index(drop=True)
        if gdf.empty:
            raise ValueError(f"Training gene {gene} has no {score_col!r} "
                             f"values.")
        pos = gdf[pos_col].values
        scores = prep_scores(gdf[score_col].values, score_col,
                             invert=invert, winsorize=winsorize)
        states = np.ones(len(scores), dtype=int)
        for a, b in intervals:
            if b < pos[0] or a > pos[-1]:
                raise ValueError(
                    f"Training interval ({a}, {b}) for {gene} does not overlap "
                    f"its observed {pos_col} range ({pos[0]}-{pos[-1]}) in "
                    f"column {score_col!r}."
                )
            i = max(0, min(np.searchsorted(pos, a, "left"), len(pos) - 1))
            j = max(0, min(np.searchsorted(pos, b, "right") - 1, len(pos) - 1))
            if j >= i:
                states[i:j + 1] = 0
        for x, y in zip(states[:-1], states[1:]):
            trans[x, y] += 1
        for k in (0, 1):
            state_vals[k].append(scores[states == k])

    cold_all = np.concatenate(state_vals[0])
    hot_all = np.concatenate(state_vals[1])
    if len(cold_all) == 0 or len(hot_all) == 0:
        raise ValueError(
            f"Training produced an empty state for {score_col!r}. Check that "
            f"training.intervals fall inside the observed {pos_col} range."
        )
    total = len(cold_all) + len(hot_all)
    startprob = np.array([len(cold_all) / total, len(hot_all) / total])
    means = np.array([cold_all.mean(), hot_all.mean()])
    scales = np.array([cold_all.std() + 1e-6, hot_all.std() + 1e-6])
    transmat = (trans + 1) / (trans + 1).sum(axis=1, keepdims=True)
    return startprob, transmat, means, scales


def train_params_mv(df: pd.DataFrame, score_cols: list[str], **kw) -> TrainedParams:
    """Combine per-feature estimates into one model with independent
    features (diagonal covariance). Raises if the labelled region does not
    have the lower mean on the first feature, i.e. it is not Cold."""
    all_means, all_scales = [], []
    startprob = transmat = None
    for col in score_cols:
        sp, tm, mu, sd = train_params_1d(df, col, **kw)
        all_means.append(mu)
        all_scales.append(sd)
        if startprob is None:
            # The same for every feature: they depend only on the labels.
            startprob, transmat = sp, tm

    means = np.column_stack(all_means)
    if means[0, 0] > means[1, 0]:
        raise ValueError(
            f"The labelled training region has a higher mean "
            f"{score_cols[0]!r} ({means[0, 0]:.3f}) than the rest of the "
            f"training gene ({means[1, 0]:.3f}) after orientation, so it "
            f"cannot be the cold state. Check training.intervals, the order "
            f"of features.primary, and features.invert."
        )

    covars = np.array([
        np.diag([all_scales[j][k] ** 2 for j in range(len(score_cols))])
        for k in (0, 1)
    ])
    return TrainedParams(startprob, transmat, means, covars, score_cols)


def marginalize_params(params: TrainedParams,
                       keep_indices: list[int]) -> TrainedParams:
    """The same model restricted to a subset of features (used for genes
    without EVE). Exact for a diagonal covariance; the start and transition
    probabilities are unchanged."""
    return TrainedParams(
        params.startprob.copy(),
        params.transmat.copy(),
        params.means[:, keep_indices],
        params.covars[:, keep_indices, :][:, :, keep_indices],
        [params.features[i] for i in keep_indices],
    )


def build_fixed_hmm(params: TrainedParams) -> hmm.GaussianHMM:
    """Wrap ``params`` in a GaussianHMM with all fitting disabled."""
    model = hmm.GaussianHMM(
        n_components=2, covariance_type="full",
        n_iter=0, init_params="", params="",
    )
    model.startprob_ = params.startprob.copy()
    model.transmat_ = params.transmat.copy()
    model.means_ = params.means.copy()
    model.covars_ = params.covars.copy()
    return model
