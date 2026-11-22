# Find Regions Of Substitution Tolerance (FROST)

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

The only input is one table, archived on Zenodo:
[doi:10.5281/zenodo.24398228](https://doi.org/10.5281/zenodo.24398228).

---

## Quick start

Requires Python 3.10 or later (the published run used 3.12).

```bash
git clone https://github.com/mariam16548/FROST.git
cd FROST
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact versions used for the published run
pip install -e .

# 1. Check the install on synthetic data (a few seconds, no download needed)
python scripts/make_synthetic_data.py
python -m coldspots.cli --config config/synthetic.yaml

# 2. Reproduce the published analysis
#    Download variant_level_coldspots.csv.gz from Zenodo (see docs/DATA.md)
#    into data/, then:
python -m coldspots.cli --config config/default.yaml
python -m coldspots.cli --config config/sensitivity_position_level.yaml
```

Run the commands from the repository root: paths in the configs are relative
to the directory you run from. If the table is elsewhere, pass
`--variant-csv /path/to/variant_level_coldspots.csv.gz`. The full run takes
about 30 seconds and 2.5 GB of memory on a laptop, and should report
**2,729 coldspots in 1,336 genes**.

To run the tests: `pytest` (installed by `requirements.txt`).

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
  pipeline.py     Loading, orchestration, reporting, output writing.
  synthetic.py    Synthetic input table for smoke tests.
  cli.py          Command-line interface.
config/
  default.yaml    The published primary analysis.
  sensitivity_position_level.yaml
                  Pseudo-replication control (see Caveats).
  synthetic.yaml  Smoke test on synthetic data.
scripts/
  make_synthetic_data.py
tests/            pytest suite (HMM correctness, intervals, evaluation,
                  end-to-end runs).
docs/
  METHODS.md      Full method description, in prose, with the statistics.
  DATA.md         The input table: schema and where to get it.
```

---

## Method in brief

1. **Load.** The input table has one row per missense substitution. Its
   per-residue scores are collapsed to one row per residue for training and
   detection; the evaluations use the substitution-level rows.
2. **Train.** For each feature, residues inside the labelled BRCA1 interval
   (residues 224–1366) are treated as the cold state and the rest of the
   protein as the hot state. Per-state means and variances, initial-state
   probabilities, and Laplace-smoothed transition probabilities are computed
   directly from these labels. There is **no EM step**; `n_iter=0`.
3. **Marginalise.** Genes with any missing EVE score are decoded with the
   trained Gaussian marginalised onto the remaining three features. Because
   the covariance is diagonal, this marginal is exact, and the reduced model
   is not retrained — it is the same model, viewed in fewer dimensions.
4. **Detect.** Viterbi decoding per gene. Cold runs spanning fewer than 40
   amino acids are discarded.
5. **Evaluate.** ClinVar and MAVE enrichment, with cold-spot membership
   recomputed from the intervals this run produced.

See [docs/METHODS.md](docs/METHODS.md) for the full description.

---

## Configuration

Every parameter lives in a YAML file. To change the analysis, edit the config —
not the code. The main knobs:

| Key | Default | Meaning |
| --- | --- | --- |
| `data.variant_csv` | `data/variant_level_coldspots.csv.gz` | The input table |
| `features.primary` | AlphaMissense, ESM1b, EVE, phyloP | Feature set; **order matters**, the first feature defines which state is cold |
| `features.fallback_drop` | `[max_EVE]` | Marginalised out for genes with any missing value of these scores |
| `training.intervals` | `BRCA1: [[224, 1366]]` | Labelled cold region(s) used for parameter estimation |
| `training.exclude_genes` | `[BRCA1]` | Never decoded; removed from both evaluations |
| `detection.min_cold_run` | `40` | Minimum cold-run span, in amino acids |
| `evaluation.min_mave_positions` | `10` | Distinct assayed PLP/BLB positions required |
| `evaluation.mave_filter_level` | `gene` | Apply that threshold per gene or per interval |
| `evaluation.unit` | `variant` | Counting unit for the enrichment tests |
| `evaluation.min_class_count` | `0` | ClinVar gene filter; `0` disables it |
| `evaluation.mave_labels` | `PLP: Abnormal, BLB: Normal, IR: Indeterminate` | How MAVE calls are worded in the table |

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
| `*_manifest.json` | Git commit (marked `-dirty` if the code had uncommitted changes), config fingerprint, full resolved config, package versions, and the fitted HMM parameters |
| `*.pkl` | Pickled bundle of the above, as a convenience |

The CSV and JSON files are the archival record. The pickle is a convenience and
is not guaranteed to load across pandas versions — **do not rely on it for
long-term reproducibility.**

---

## Reproducibility notes

* **The pipeline is deterministic.** Parameters are estimated in closed form and
  Viterbi decoding is exact, so no result depends on the RNG. `seed` is set
  anyway and recorded, but changing it changes nothing
  (`tests/test_model.py::test_decoding_is_deterministic` and
  `tests/test_pipeline.py::test_two_runs_give_identical_files` check this).
* **Versions are recorded, not just pinned.** Every run writes the exact
  versions of numpy, pandas, scipy and hmmlearn to the manifest, alongside the
  git commit and a hash of the resolved config. `requirements.txt` pins the
  versions used for the published run.
* **Membership is always recomputed.** The `variant_in_coldspot` and
  `gene_has_coldspot` columns shipped in the table are never read, so a stale
  assignment cannot leak into the evaluation. (Running the pipeline on the
  released table reproduces both columns exactly.)

---

## Caveats a reader should know about

These are properties of the method, not bugs. They are documented so that the
numbers are read correctly.

1. **Variant-level counting inflates the sample size.** By default each row of
   the table (one substitution) is one observation, so a residue with three
   classified substitutions contributes three observations to Fisher's exact
   test as though they were independent. This makes p-values
   anti-conservative. `config/sensitivity_position_level.yaml` collapses to
   one observation per (gene, position, label); **report both**. The
   position-level test is less inflated but still not exact: neighbouring
   residues in a contiguous region are not independent, and the single 2×2
   table pools genes with very different baseline rates. A stratified
   (per-gene) or permutation-based test would address both.
2. **`mave_filter_level: gene` is the looser setting.** It thresholds the *sum*
   of assayed positions across a gene's cold spots and then keeps them all —
   including cold spots with two assayed positions, or none. Use
   `interval` to require each region to stand on its own evidence; cold
   spots that fail are then left out of both the inside and outside counts.
3. **Normalisation is per-gene.** Features are winsorised and min-max scaled
   *within each gene* at detection time, and within the training gene at
   training time. Emission means are therefore in each gene's own units, so a
   "cold" call is relative to that protein's own score distribution rather than
   to an absolute scale. This is intentional and load-bearing; do not replace
   it with a global scaler.
4. **Features are treated as independent given the state.** The four scores
   are correlated, but the emission covariance is diagonal, so shared signal is
   counted more than once and per-residue evidence is overconfident. The 40-aa
   minimum run length absorbs much of the resulting fragmentation.
5. **Missing data.** A gene with any missing AlphaMissense, ESM1b or phyloP
   value is not decoded, and a gene with any missing EVE value uses the
   3-feature model. Residues absent from the table are skipped, and the HMM
   treats the residues on either side of a gap as neighbours, so an interval
   can span a gap. `length` is the amino-acid span; `n_residues_scored` is
   the number of scored residues inside it. In the released table 0.5% of
   positions are absent, and 430 of 2,729 cold spots span at least one gap.
6. **`frac_assayed` should accompany any depletion claim.** A region can clear
   the MAVE threshold on a handful of clustered residues at one edge while most
   of it is unassayed.
7. **Odds-ratio direction.** `OR` is the odds of a *benign* call inside a cold
   spot relative to outside, so **OR > 1 means depletion of pathogenic
   variation**. The point estimate and Wald CI use the Haldane–Anscombe (+0.5)
   correction; the Fisher p-value uses the uncorrected counts. The raw 2×2
   table is in `*_metrics.json` for anyone wanting a different convention.
8. **Only genes with at least one predicted cold spot enter the tests.** Genes
   without one provide no inside/outside contrast, so they are excluded rather
   than counted as all-outside.

---

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

[MIT](LICENSE) for the code. The input data is released under CC BY 4.0 and
is governed by the terms of its original sources; see
[docs/DATA.md](docs/DATA.md).
