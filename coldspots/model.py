"""Two-state Gaussian HMM: supervised parameter estimation and marginalisation.

The model is never fitted by EM. Parameters are estimated once, in closed
form, from labelled intervals in the training gene(s), and the resulting model
is applied to every other gene with ``n_iter=0``. Decoding is therefore
deterministic given the input table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm

from .features import prep_scores

__all__ = [
    "train_params_1d", "train_params_mv", "marginalize_params",
    "build_fixed_hmm", "TrainedParams",
]


class TrainedParams:
    """Container for a fixed two-state Gaussian HMM parameterisation."""

    __slots__ = ("startprob", "transmat", "means", "covars", "features",
                 "state_swapped")

    def __init__(self, startprob, transmat, means, covars, features,
                 state_swapped=False):
        self.startprob = startprob
        self.transmat = transmat
        self.means = means
        self.covars = covars
        self.features = list(features)
        # True if state indices had to be swapped so that state 0 is cold.
        # Expected to be False for the published configuration; see
        # docs/CHANGES_FROM_ORIGINAL.md.
        self.state_swapped = bool(state_swapped)

    @property
    def cold_state(self) -> int:
        """Index of the cold state: lower mean on the first feature."""
        return int(np.argmin(self.means[:, 0]))

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
            "state_swapped": self.state_swapped,
        }


def train_params_1d(df: pd.DataFrame, score_col: str, *, gene_col: str,
                    pos_col: str, training_intervals: dict,
                    invert: list[str], winsorize: tuple[float, float]):
    """Estimate start, transition and per-state Gaussian moments for one feature.

    State 0 is the labelled ("cold") state inside ``training_intervals``;
    state 1 is everything else in the training gene(s). Transition counts use
    add-one (Laplace) smoothing.
    """
    trans = np.zeros((2, 2))
    state_vals: dict[int, list[np.ndarray]] = {0: [], 1: []}

    for gene, intervals in training_intervals.items():
        gdf = df[df[gene_col] == gene].dropna(subset=[score_col])
        gdf = gdf.sort_values(pos_col).reset_index(drop=True)
        if gdf.empty:
            continue
        pos = gdf[pos_col].values
        scores = prep_scores(gdf[score_col].values, score_col,
                             invert=invert, winsorize=winsorize)
        states = np.ones(len(scores), dtype=int)
        for a, b in intervals:
            # searchsorted indices are clamped into range below, which would
            # silently relabel the terminal residue if the interval sat wholly
            # outside the observed positions. Refuse instead: a misconfigured
            # training interval must not pass unnoticed.
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

    for k in (0, 1):
        if not state_vals[k]:
            state_vals[k] = [np.zeros(1)]

    cold_all = np.concatenate(state_vals[0])
    hot_all = np.concatenate(state_vals[1])
    if len(cold_all) == 0 or len(hot_all) == 0:
        raise ValueError(
            f"Training produced an empty state for {score_col!r}. Check that "
            f"training.intervals fall inside the observed aa_pos range."
        )
    total = len(cold_all) + len(hot_all)
    startprob = np.array([len(cold_all) / total, len(hot_all) / total])
    means = np.array([cold_all.mean(), hot_all.mean()])
    scales = np.array([cold_all.std() + 1e-6, hot_all.std() + 1e-6])
    transmat = (trans + 1) / (trans + 1).sum(axis=1, keepdims=True)
    return startprob, transmat, means, scales


def train_params_mv(df: pd.DataFrame, score_cols: list[str], **kw) -> TrainedParams:
    """Stack per-feature moments into a multivariate, diagonal-covariance model.

    Features are assumed conditionally independent given the state, so the
    covariance is diagonal even though ``covariance_type="full"`` is used for
    decoding. States are ordered so that state 0 is cold on the first feature.
    """
    all_means, all_scales = [], []
    startprob = transmat = None
    for col in score_cols:
        sp, tm, mu, sd = train_params_1d(df, col, **kw)
        all_means.append(mu)
        all_scales.append(sd)
        if startprob is None:
            # start and transition probabilities depend only on the state
            # labels, which are identical across features.
            startprob, transmat = sp, tm

    means = np.column_stack(all_means)
    swapped = bool(means[0, 0] > means[1, 0])
    if swapped:
        means = means[[1, 0], :]
        all_scales = [s[[1, 0]] for s in all_scales]
        startprob = startprob[[1, 0]]
        transmat = transmat[[1, 0], :][:, [1, 0]]

    covars = np.array([
        np.diag([all_scales[j][k] ** 2 for j in range(len(score_cols))])
        for k in (0, 1)
    ])
    return TrainedParams(startprob, transmat, means, covars, score_cols,
                         state_swapped=swapped)


def marginalize_params(params: TrainedParams,
                       keep_indices: list[int]) -> TrainedParams:
    """Marginalise a fitted Gaussian down to a subset of features.

    For a diagonal covariance this is exact: the marginal of a multivariate
    normal over a coordinate subset is the normal formed by the corresponding
    sub-vector of means and sub-matrix of the covariance. Start and transition
    probabilities are unchanged.
    """
    return TrainedParams(
        params.startprob.copy(),
        params.transmat.copy(),
        params.means[:, keep_indices],
        params.covars[:, keep_indices, :][:, :, keep_indices],
        [params.features[i] for i in keep_indices],
        state_swapped=params.state_swapped,
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
