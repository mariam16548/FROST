# Find Regions of Substition Tolerance (FROST)

Detection and evaluation of **mutational coldspots** — contiguous protein
regions depleted of pathogenic variation — using a two-state Gaussian hidden
Markov model over variant-effect-predictor and conservation scores.

Model parameters are estimated once, in closed form, from a single labelled
region in **BRCA1**, then applied unchanged to every other gene. BRCA1 is
excluded from all evaluation. Predicted coldspots are assessed against
(1) ClinVar clinical significance and (2) multiplexed assay of variant effect
(MAVE) functional calls.

This repository contains the code accompanying:

> *[preprint citation]*

---

## Quick start

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Confirm the pipeline runs, using synthetic data (no real data needed)
python scripts/make_synthetic_data.py
python -m coldspots.cli --config config/synthetic.yaml

# Reproduce the published analysis (requires the real input tables)
#   1. obtain the inputs        -> see docs/DATA.md
#   2. edit the two paths in    -> config/default.yaml
python -m coldspots.cli --config config/default.yaml
```


---

## Repository layout

```
coldspots/
  config.py       All parameters and paths, loaded from YAML. Nothing is
                  hard-coded elsewhere.
  features.py     Feature orientation, winsorisation, min-max scaling.
  model.py        Supervised estimation of the two-state HMM; marginalisation
                  to the reduced feature set.
  intervals.py    Viterbi decoding, short-run filtering, interval extraction.
  evaluation.py   Membership, MAVE support filtering, odds ratios.
  pipeline.py     Orchestration, reporting, output writing.
  cli.py          Command-line interface.
config/
  default.yaml    The published primary analysis.
  sensitivity_position_level.yaml
                  Pseudo-replication control (see Caveats).

docs/
  METHODS.md      Full method description, in prose, with the statistics.
  DATA.md         Input schemas and how to obtain the real tables.

```

---

## Method in brief

1. **Train.** For each feature, residues inside the labelled BRCA1 interval
   (residues 224–1366) are treated as the cold state and the rest of the
   protein as the hot state. Per-state means and variances, initial-state
   probabilities, and Laplace-smoothed transition probabilities are computed
   directly from these labels. There is **no EM step**; `n_iter=0`.
2. **Marginalise.** Genes lacking EVE scores entirely are handled by
   marginalising the trained Gaussian down to the remaining three features.
   Because the covariance is diagonal, this marginal is exact, and the reduced
   model is not retrained — it is the same model, viewed in fewer dimensions.
3. **Detect.** Viterbi decoding per gene on the collapsed per-residue table.
   Cold runs spanning fewer than 40 amino acids are discarded.
4. **Evaluate.** ClinVar and MAVE enrichment are computed on the expanded
   variant-level table, with cold-spot membership recomputed from the intervals
   this run produced.

See [docs/METHODS.md](docs/METHODS.md) for the full description.

---

## Configuration

Every parameter lives in a YAML file. To change the analysis, edit the config —
not the code. The main knobs:

| Key | Default | Meaning |
| --- | --- | --- |
| `features.primary` | AlphaMissense, ESM1b, EVE, phyloP | Feature set; **order matters**, the first feature defines which state is cold |
| `features.fallback_drop` | `[max_EVE]` | Marginalised out for genes missing these scores |
| `training.intervals` | `BRCA1: [[224, 1366]]` | Labelled cold region(s) used for parameter estimation |
| `training.exclude_genes` | `[BRCA1]` | Never decoded; removed from both evaluations |
| `detection.min_cold_run` | `40` | Minimum cold-run span, in amino acids |
| `evaluation.min_mave_positions` | `10` | Distinct assayed PLP/BLB positions required |
| `evaluation.mave_filter_level` | `gene` | Apply that threshold per gene or per interval |
| `evaluation.unit` | `variant` | Counting unit for the enrichment tests |
| `evaluation.min_class_count` | `0` | ClinVar gene filter; `0` disables it |

Any of these can be overridden on the command line without editing the file:

```bash
python -m coldspots.cli --config config/default.yaml \
    --unit position --mave-filter-level interval --out-dir results/sensitivity
```

`python -m coldspots.cli --help` lists all overrides.

---

## Outputs

Written to `output.dir`, prefixed with `output.prefix`:

| File | Contents |
| --- | --- |
| `*_intervals.csv` | Every predicted cold spot: gene, `aa_start`, `aa_end`, `length`, `n_residues_scored`, and which model produced it |
| `*_intervals_mave_filtered.csv` | The subset passing the MAVE support filter, with per-region assay counts and `frac_assayed` |
| `*_metrics.json` | All reported statistics, including the raw 2×2 tables behind each odds ratio |
| `*_manifest.json` | Git commit, config fingerprint, full resolved config, package versions, and the fitted HMM parameters |
| `*.pkl` | Pickled bundle, for continuity with the original workflow |

The CSV and JSON files are the archival record. The pickle is a convenience and
is not guaranteed to load across pandas versions — **do not rely on it for
long-term reproducibility.**

---

## Reproducibility notes

* **The pipeline is deterministic.** Parameters are estimated in closed form and
  Viterbi decoding is exact, so no result depends on the RNG. `seed` is set
  anyway and recorded, but changing it changes nothing (`test_model.py`
  asserts this).
* **Versions are recorded, not just pinned.** Every run writes the exact
  versions of numpy, pandas, scipy, scikit-learn, and hmmlearn to the manifest,
  alongside the git commit and a hash of the resolved config. `requirements.txt`
  pins the versions used for the published run.
* **Membership is always recomputed.** Any `in_coldspot` / `in_coldspot_bw`
  column present in the input is dropped on load, so a stale assignment from a
  previous run cannot leak into the evaluation.
* **The input data is not redistributed here.** See
  [docs/DATA.md](docs/DATA.md) for schemas, provenance, and access.

---

## Caveats a reader should know about

These are properties of the method, not bugs. They are documented so that the
numbers are read correctly.

1. **Variant-level counting inflates the sample size.** By default each row of
   the expanded table is one observation, so a residue with three codon changes
   contributes three observations to Fisher's exact test as though they were
   independent. This makes p-values anti-conservative. The bundled
   `config/sensitivity_position_level.yaml` collapses to one observation per
   (gene, position, label); **report both**, and treat the position-level
   p-value as the defensible one.
2. **`mave_filter_level: gene` is the looser setting.** It thresholds the *sum*
   of assayed positions across a gene's cold spots and then keeps them all —
   including cold spots with two assayed positions, or none. Use
   `interval` to require each region to stand on its own evidence.
3. **Normalisation is per-gene.** Features are winsorised and min-max scaled
   *within each gene* at detection time, and within the training gene at
   training time. Emission means are therefore in each gene's own units, so a
   "cold" call is relative to that protein's own score distribution rather than
   to an absolute scale. This is intentional and load-bearing; do not replace
   it with a global scaler.
4. **`length` is an amino-acid span, not a residue count.** Residues with
   missing scores are dropped before decoding, so a 40-aa cold spot may contain
   fewer than 40 scored residues. `n_residues_scored` is reported alongside.
5. **`frac_assayed` should accompany any depletion claim.** A region can clear
   the MAVE threshold on a handful of clustered residues at one edge while most
   of it is unassayed.
6. **Odds-ratio direction.** `OR` is the odds of a *benign* call inside a cold
   spot relative to outside, so **OR > 1 means depletion of pathogenic
   variation**. The point estimate and Wald CI use the Haldane–Anscombe (+0.5)
   correction; the Fisher p-value uses the uncorrected counts. The raw 2×2
   table is in `*_metrics.json` for anyone wanting a different convention.
7. **Only genes with at least one predicted cold spot enter the tests.** Genes
   without one provide no inside/outside contrast, so they are excluded rather
   than counted as all-outside.

---

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

[MIT](LICENSE) for the code. The input data is governed by the terms of its
original sources; see [docs/DATA.md](docs/DATA.md).
