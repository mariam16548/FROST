"""Parameter estimation, marginalisation, and Viterbi decoding."""
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, norm

from coldspots.features import prep_scores
from coldspots.intervals import decode_genes
from coldspots.model import (COLD_STATE, TrainedParams, build_fixed_hmm,
                             marginalize_params, train_params_mv)

KW = dict(gene_col="gene", pos_col="pos", invert=[], winsorize=(0.0, 100.0))


def _training_gene(cold=(31, 70), n=100, seed=0, cold_high=False):
    rng = np.random.default_rng(seed)
    pos = np.arange(1, n + 1)
    in_cold = (pos >= cold[0]) & (pos <= cold[1])
    lo, hi = (0.8, 0.2) if cold_high else (0.2, 0.8)
    base = np.where(in_cold, lo, hi)
    return pd.DataFrame({
        "gene": "T", "pos": pos,
        "f1": base + rng.normal(0, 0.05, n),
        "f2": base + rng.normal(0, 0.05, n),
    })


def test_supervised_estimates_match_labels():
    df = _training_gene()
    p = train_params_mv(df, ["f1", "f2"],
                        training_intervals={"T": [(31, 70)]}, **KW)
    # 40 cold residues, 60 hot.
    np.testing.assert_allclose(p.startprob, [0.4, 0.6])
    # Adjacent transitions: cold->cold 39, cold->hot 1, hot->cold 1,
    # hot->hot 58; add-one smoothing, row-normalised.
    expected = np.array([[40, 2], [2, 59]], dtype=float)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(p.transmat, expected)
    # Emission means are the per-state means of the within-gene scaled scores.
    scaled = prep_scores(df["f1"].values, "f1", invert=[], winsorize=(0, 100))
    cold = (df["pos"] >= 31) & (df["pos"] <= 70)
    np.testing.assert_allclose(p.means[:, 0],
                               [scaled[cold].mean(), scaled[~cold].mean()])
    assert p.cold_state == COLD_STATE == 0
    assert p.means[0, 0] < p.means[1, 0]


def test_labelled_region_that_is_not_cold_raises():
    df = _training_gene(cold_high=True)
    with pytest.raises(ValueError, match="cannot be the cold state"):
        train_params_mv(df, ["f1", "f2"],
                        training_intervals={"T": [(31, 70)]}, **KW)


def test_training_interval_outside_gene_raises():
    df = _training_gene()
    with pytest.raises(ValueError, match="does not overlap"):
        train_params_mv(df, ["f1", "f2"],
                        training_intervals={"T": [(500, 600)]}, **KW)


def test_absent_training_gene_raises():
    df = _training_gene()
    with pytest.raises(ValueError, match="has no"):
        train_params_mv(df, ["f1", "f2"],
                        training_intervals={"ABSENT": [(1, 10)]}, **KW)


def test_marginalisation_is_exact():
    df = _training_gene()
    p = train_params_mv(df, ["f1", "f2"],
                        training_intervals={"T": [(31, 70)]}, **KW)
    m = marginalize_params(p, [1])
    assert m.features == ["f2"]
    np.testing.assert_array_equal(m.means, p.means[:, [1]])
    np.testing.assert_array_equal(m.covars, p.covars[:, [1]][:, :, [1]])
    # With a diagonal covariance the joint density factorises, so the
    # marginal is the product's remaining factor.
    x = np.array([0.3, 0.6])
    for k in (0, 1):
        joint = multivariate_normal(p.means[k], p.covars[k]).pdf(x)
        f1 = norm(p.means[k, 0], np.sqrt(p.covars[k, 0, 0])).pdf(x[0])
        f2 = norm(m.means[k, 0], np.sqrt(m.covars[k, 0, 0])).pdf(x[1])
        assert joint == pytest.approx(f1 * f2)


def _brute_force_viterbi(params, X):
    """Most probable state path by enumerating every path."""
    log_e = np.column_stack([
        multivariate_normal(params.means[k], params.covars[k]).logpdf(X)
        for k in (0, 1)
    ])
    best, best_lp = None, -np.inf
    for path in itertools.product((0, 1), repeat=len(X)):
        lp = np.log(params.startprob[path[0]]) + log_e[0, path[0]]
        for t in range(1, len(X)):
            lp += np.log(params.transmat[path[t - 1], path[t]])
            lp += log_e[t, path[t]]
        if lp > best_lp:
            best, best_lp = np.array(path), lp
    return best


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(25):
        means = np.sort(rng.uniform(0, 1, (2, 2)), axis=0)
        covars = np.array([np.diag(rng.uniform(0.01, 0.1, 2)) for _ in (0, 1)])
        stay = rng.uniform(0.6, 0.99, 2)
        transmat = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
        params = TrainedParams(np.array([0.5, 0.5]), transmat, means, covars,
                               ["a", "b"])
        X = rng.uniform(0, 1, (9, 2))
        got = build_fixed_hmm(params).predict(X)
        np.testing.assert_array_equal(got, _brute_force_viterbi(params, X))


def test_decoding_is_deterministic():
    """Seeding the global RNG differently changes nothing."""
    train = _training_gene(n=200, cold=(50, 150))
    p = train_params_mv(train, ["f1", "f2"],
                        training_intervals={"T": [(50, 150)]}, **KW)
    genes = [_training_gene(n=300, cold=(80, 200), seed=s).assign(gene=f"G{s}")
             for s in range(5)]
    df = pd.concat(genes, ignore_index=True)
    runs = []
    for seed in (0, 12345):
        np.random.seed(seed)
        runs.append(decode_genes({f"G{s}" for s in range(5)}, p, df,
                                 min_cold_run=40, verbose=False, **KW))
    pd.testing.assert_frame_equal(runs[0], runs[1])
    assert len(runs[0]) == 5
