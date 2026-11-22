"""Synthetic input table, in the layout of docs/DATA.md, for testing.

Each gene gets zero to two planted coldspots. The four scores are noisy
functions of the planted state, and ClinVar / MAVE labels are less often
pathogenic inside planted coldspots. The planted intervals are written to
``truth.csv``. The data is for checking that FROST runs and recovers what was
planted; it says nothing about real proteins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["make_synthetic"]


def _features(state: np.ndarray, rng: np.random.Generator):
    # A shared latent "damage" signal makes the four features correlated, as
    # they are in real data.
    n = len(state)
    z = np.where(state == 0, 0.25, 0.7) + rng.normal(0, 0.12, n)
    am = np.clip(z + rng.normal(0, 0.10, n), 0, 1)
    esm = -(z * 15 + rng.normal(0, 2.0, n))      # more negative = damaging
    eve = np.clip(z + rng.normal(0, 0.12, n), 0, 1)
    phylop = z * 8 - 1 + rng.normal(0, 1.5, n)
    return am, esm, eve, phylop


_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def make_synthetic(out_dir: str | Path, n_genes: int = 150,
                   seed: int = 0) -> dict[str, Path]:
    """Write ``variants.csv.gz`` and ``truth.csv`` to ``out_dir``.

    Includes BRCA1 with the published training interval (224-1366), genes
    lacking EVE entirely or partly (routed to the 3-feature model), and one
    gene with a missing AlphaMissense value (not decoded).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames, truth = [], []

    def add_gene(gene: str, length: int, cold: list[tuple[int, int]], *,
                 eve_mode: str = "full", mave: bool = False,
                 missing_am: bool = False) -> None:
        state = np.ones(length, dtype=int)
        for a, b in cold:
            state[a - 1:b] = 0
            truth.append((gene, a, b))
        am, esm, eve, phylop = _features(state, rng)
        if eve_mode == "none":
            eve = np.full(length, np.nan)
        elif eve_mode == "partial":
            eve[: length // 10] = np.nan
        if missing_am:
            am[length // 2] = np.nan

        # 1-3 substitutions per residue. The predictor scores are per-residue
        # summaries, so they repeat across a residue's substitutions.
        reps = rng.integers(1, 4, length)
        idx = np.repeat(np.arange(length), reps)
        vstate = state[idx]
        ref = rng.choice(_AA, length)
        protein_variant = np.char.add(
            np.char.add(ref[idx], (idx + 1).astype(str)),
            rng.choice(_AA, len(idx)))
        u = rng.random(len(idx))
        p_plp = np.where(vstate == 0, 0.03, 0.15)
        p_blb = np.where(vstate == 0, 0.15, 0.05)
        clinvar = np.where(u < p_plp, "PLP",
                  np.where(u < p_plp + p_blb, "BLB",
                  np.where(u < p_plp + p_blb + 0.2, "VUS", None)))
        if mave:
            u2 = rng.random(len(idx))
            calls = np.where(u2 < np.where(vstate == 0, 0.1, 0.5), "Abnormal",
                     np.where(u2 < 0.9, "Normal", "Indeterminate")
                     ).astype(object)
            calls[rng.random(len(idx)) < 0.4] = None
        else:
            calls = np.full(len(idx), None, dtype=object)
        frames.append(pd.DataFrame({
            "transcript_id": f"ENST_{gene}", "gene_symbol": gene,
            "protein_variant": protein_variant,
            "max_AlphaMissense": am[idx], "max_EVE": eve[idx],
            "min_ESM1b": esm[idx], "max_phyloP": phylop[idx],
            "pLDDT": rng.uniform(20, 95, length)[idx],
            "CLNSIG": clinvar, "functional_assignment": calls,
            # Stale membership columns, which the pipeline must ignore.
            "gene_has_coldspot": True, "variant_in_coldspot": True,
        }))

    add_gene("BRCA1", 1863, [(224, 1366)], mave=True)
    for k in range(n_genes):
        length = int(rng.integers(150, 1500))
        cold, cur = [], 1
        for _ in range(int(rng.integers(0, 3))):
            a = int(cur + rng.integers(20, max(21, length // 3)))
            b = int(a + rng.integers(60, 300))
            if b >= length - 10:
                break
            cold.append((a, b))
            cur = b + 1
        r = rng.random()
        add_gene(f"G{k:05d}", length, cold,
                 eve_mode="none" if r < 0.2 else "partial" if r < 0.3 else "full",
                 mave=bool(rng.random() < 0.3),
                 missing_am=(k == 0))

    paths = {"variants": out / "variants.csv.gz", "truth": out / "truth.csv"}
    pd.concat(frames).to_csv(paths["variants"], index=False)
    pd.DataFrame(truth, columns=["gene", "aa_start", "aa_end"]).to_csv(
        paths["truth"], index=False)
    return paths
