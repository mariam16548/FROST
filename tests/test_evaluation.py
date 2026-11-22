"""Cold-spot membership, MAVE support filtering, and odds ratios."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from coldspots.config import Config, DataPaths, Evaluation
from coldspots.evaluation import evaluate, in_intervals, mave_genes, odds_ratio
from coldspots.features import complete_genes


def test_in_intervals_matches_naive_check():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({
        "g": rng.choice(["A", "B", "C"], 2000),
        "pos": rng.integers(1, 500, 2000).astype(float),
    })
    rows.loc[::97, "pos"] = np.nan          # missing positions are never inside
    ivs = pd.DataFrame({
        "gene": ["A", "A", "A", "B"],
        "aa_start": [10, 40, 45, 200],      # A's second and third overlap
        "aa_end": [30, 60, 50, 200],
    })
    expected = [
        any(a <= p <= b for a, b in
            zip(ivs.loc[ivs.gene == g, "aa_start"],
                ivs.loc[ivs.gene == g, "aa_end"]))
        for g, p in zip(rows["g"], rows["pos"])
    ]
    got = in_intervals(rows, ivs, gene_col="g", aa_col="pos")
    np.testing.assert_array_equal(got, expected)


def _mave_case():
    """Gene G: two cold spots and MAVE calls (released wording).

    A = [1, 100]:   100 positions with one Normal call each
    B = [201, 204]: 4 positions with 50 Normal calls each
    outside:        50 positions (301-350) with one Abnormal call each
    Gene H: one cold spot [1, 50] with only 5 assayed positions inside.
    """
    rows = ([("G", p, "Normal") for p in range(1, 101)]
            + [("G", p, "Normal") for p in range(201, 205) for _ in range(50)]
            + [("G", p, "Abnormal") for p in range(301, 351)]
            + [("H", p, "Normal") for p in range(1, 6)]
            + [("H", p, "Abnormal") for p in range(100, 140)])
    mave = pd.DataFrame(rows, columns=["gene_symbol", "aa_pos", "call"])
    mave["clinvar"] = None
    cold = pd.DataFrame({"gene": ["G", "G", "H"], "aa_start": [1, 201, 1],
                         "aa_end": [100, 204, 50], "length": [100, 4, 50],
                         "n_residues_scored": [100, 4, 50]})
    return cold, mave


def _cfg():
    cfg = Config(data=DataPaths("unused"),
                 evaluation=Evaluation(enabled=True, min_mave_positions=10))
    cfg.columns.clinvar = "clinvar"
    cfg.columns.mave = "call"
    return cfg


def test_mave_filter_sums_positions_across_a_genes_cold_spots():
    """G passes (104 positions over two cold spots); H has 5 and is left out.

    Every cold spot of a passing gene counts, including B with only 4
    assayed positions of its own.
    """
    cold, mave = _mave_case()
    ev = evaluate(cold, mave.iloc[:0], mave, cfg=_cfg(), mave_col="call")
    assert ev["mave_genes"] == ["G"]
    assert ev["n_mave_intervals"] == 2
    # Each substitution counts once: 100 + 200 Normal inside, 50 Abnormal out.
    assert ev["or_mave"]["table"] == [[300, 0], [0, 50]]


def test_mave_genes_threshold():
    cold, mave = _mave_case()
    kw = dict(label_col="call", gene_col="gene_symbol", pos_col="aa_pos",
              valid_labels=("Abnormal", "Normal"))
    assert mave_genes(cold, mave, min_count=5, **kw) == {"G", "H"}
    assert mave_genes(cold, mave, min_count=6, **kw) == {"G"}


def test_odds_ratio_values_and_direction():
    summary = {"BLB": {"inside": 10, "total": 15},
               "PLP": {"inside": 2, "total": 10}}
    o = odds_ratio(summary)
    assert o["table"] == [[10, 5], [2, 8]]
    # Haldane-Anscombe corrected point estimate: (10.5 * 8.5) / (5.5 * 2.5)
    assert o["OR"] == pytest.approx(round(10.5 * 8.5 / (5.5 * 2.5), 2))
    assert o["OR"] > 1  # benign enriched inside = pathogenic depletion
    assert o["p"] == pytest.approx(fisher_exact([[10, 5], [2, 8]])[1])
    assert o["OR_lo"] < o["OR"] < o["OR_hi"]


def test_odds_ratio_with_empty_class_is_nan():
    o = odds_ratio({"BLB": {"inside": 0, "total": 0},
                    "PLP": {"inside": 2, "total": 10}})
    assert np.isnan(o["OR"]) and np.isnan(o["p"])


def test_complete_genes():
    df = pd.DataFrame({"g": ["A", "A", "B", "B", "C"],
                       "x": [1, 2, 3, np.nan, 5],
                       "y": [1, 2, 3, 4, np.nan]})
    assert complete_genes(df, ["x"], "g") == {"A", "C"}
    assert complete_genes(df, ["x", "y"], "g") == {"A"}
