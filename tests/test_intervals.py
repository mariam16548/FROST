"""Short-run filtering, interval extraction, and per-gene decoding."""
import numpy as np
import pandas as pd

from coldspots.intervals import (decode_genes, extract_intervals,
                                 filter_short_regions)
from coldspots.model import train_params_mv

from test_model import KW, _training_gene


def _runs(mask):
    """(start_index, end_index) of each run of True values."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def test_filter_and_extract_match_reference():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(1, 120))
        mask = rng.random(n) < rng.uniform(0.2, 0.9)
        # Positions with occasional gaps (missing residues).
        pos = np.cumsum(rng.choice([1, 1, 1, 2, 5], size=n))
        min_len = int(rng.integers(1, 30))

        kept = [(a, b) for a, b in _runs(mask)
                if pos[b] - pos[a] + 1 >= min_len]
        expected_mask = np.zeros(n, dtype=bool)
        for a, b in kept:
            expected_mask[a:b + 1] = True

        got_mask = filter_short_regions(mask, pos, min_len)
        np.testing.assert_array_equal(got_mask, expected_mask)

        got = extract_intervals(got_mask, pos, "G")
        assert [(d["aa_start"], d["aa_end"]) for d in got] == \
            [(pos[a], pos[b]) for a, b in kept]
        assert [d["n_residues_scored"] for d in got] == \
            [b - a + 1 for a, b in kept]
        assert all(d["length"] == d["aa_end"] - d["aa_start"] + 1 for d in got)


def test_decode_all_genes_equals_decoding_each_gene_alone():
    train = _training_gene(n=200, cold=(50, 150))
    p = train_params_mv(train, ["f1", "f2"],
                        training_intervals={"T": [(50, 150)]}, **KW)
    genes = [_training_gene(n=250 + 10 * s, cold=(60, 180), seed=s)
             .assign(gene=f"G{s}") for s in range(6)]
    # Shuffle rows so genes are interleaved, as decode_genes must not rely on
    # the table being grouped.
    df = (pd.concat(genes, ignore_index=True)
          .sample(frac=1, random_state=0).reset_index(drop=True))
    names = {f"G{s}" for s in range(6)}

    together = decode_genes(names, p, df, min_cold_run=40, verbose=False, **KW)
    alone = pd.concat(
        [decode_genes({g}, p, df[df["gene"] == g], min_cold_run=40,
                      verbose=False, **KW) for g in sorted(names)],
        ignore_index=True)
    pd.testing.assert_frame_equal(together.reset_index(drop=True), alone)


def test_short_and_absent_genes_are_skipped():
    train = _training_gene(n=200, cold=(50, 150))
    p = train_params_mv(train, ["f1", "f2"],
                        training_intervals={"T": [(50, 150)]}, **KW)
    short = _training_gene(n=60, cold=(10, 50)).assign(gene="SHORT")
    out = decode_genes({"SHORT", "ABSENT"}, p, short, min_cold_run=40,
                       verbose=False, **KW)
    assert out.empty
