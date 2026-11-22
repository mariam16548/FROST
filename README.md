# Find Regions Of Substitution Tolerance (FROST)

Detection of **mutational coldspots** — contiguous protein regions depleted
of pathogenic variation — using a two-state Gaussian hidden Markov model over
variant-effect-predictor and conservation scores.

Model parameters are estimated once, in closed form, from a single labelled
region in **BRCA1**, then applied unchanged to every other gene.

Running it on the released table reproduces the published results:

* **the coldspot regions** (`*_regions.tsv`: gene, transcript, `aa_start`,
  `aa_end`), and
* **the per-variant membership columns** `gene_has_coldspot` and
  `variant_in_coldspot` (`*_variants.csv.gz`, one row per input row).

An optional analysis (`--evaluate`) also tests the regions against ClinVar
clinical significance and multiplexed assay of variant effect (MAVE)
functional calls.

This repository contains the code accompanying:

> *[preprint citation]*

The only input is one table, archived on Zenodo:
[doi:10.5281/zenodo.24398228](https://doi.org/10.5281/zenodo.24398228) [variant_level_coldspots.csv.gz].

---

## Quick start

Requires Python 3.11 or later (the published run used 3.12).

```bash
git clone https://github.com/mariam16548/FROST.git
cd FROST
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact versions used for the published run
pip install -e .

# 1. Check the install on synthetic data (a few seconds, no download needed)
python scripts/make_synthetic_data.py
python -m coldspots.cli --config config/synthetic.yaml

# 2. Reproduce the published regions and membership columns
#    Download variant_level_coldspots.csv.gz from Zenodo (see docs/DATA.md)
#    into data/, then:
python -m coldspots.cli --config config/default.yaml

# Optional: the ClinVar / MAVE enrichment analysis
python -m coldspots.cli --config config/default.yaml --evaluate
python -m coldspots.cli --config config/sensitivity_position_level.yaml
```

Run the commands from the repository root: paths in the configs are relative
to the directory you run from. After `pip install -e .`, `frost` works as a
shorter alias for `python -m coldspots.cli`. If the table is elsewhere, pass
`--variant-csv /path/to/variant_level_coldspots.csv.gz`. The run takes about
a minute and 2.5 GB of memory on a laptop, and should report **2,729
coldspots in 1,336 genes**. `results/coldspots_regions.tsv` then matches the
published regions table (its protein-coordinate columns), and
`results/coldspots_variants.csv.gz` matches the table's own
`gene_has_coldspot` / `variant_in_coldspot` columns on every row.

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
                  Optional evaluation, counted per position.
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
   detection.
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
5. **Write.** The regions, and each substitution's membership
   (`variant_in_coldspot`) and its gene's (`gene_has_coldspot`). Training
   genes (BRCA1) are left empty in both columns.
6. **Evaluate (optional).** ClinVar and MAVE enrichment of the regions.

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
| `evaluation.enabled` | `false` | Run the optional ClinVar / MAVE analysis (also `--evaluate`) |
| `evaluation.min_mave_positions` | `10` | Distinct positions with an Abnormal/Normal MAVE call required |
| `evaluation.mave_filter_level` | `gene` | `gene`: the 10 positions may be summed across a gene's cold spots; `interval`: each cold spot needs its own 10 |
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
| `*_regions.tsv` | Every predicted cold spot: `gene_symbol`, `transcript_id`, `aa_start`, `aa_end` (the published regions table's protein-coordinate columns) |
| `*_variants.csv.gz` | One row per input row: `transcript_id`, `gene_symbol`, `protein_variant`, `gene_has_coldspot`, `variant_in_coldspot` (empty for training genes) |
| `*_manifest.json` | Git commit (marked `-dirty` if the code had uncommitted changes), config fingerprint, full resolved config, package versions, the fitted HMM parameters, and region counts |

With `--evaluate`, also:

| File | Contents |
| --- | --- |
| `*_regions_mave_filtered.csv` | Regions passing the MAVE support filter, with per-region assay counts and `frac_assayed` |
| `*_metrics.json` | All reported statistics, including the raw 2×2 tables behind each odds ratio |

`output.variant_table: false` skips the per-variant file; `output.write_pickle:
true` adds a pickled bundle (not guaranteed to load across pandas versions).

---

## Reproducibility notes

* **The pipeline is deterministic.** Parameters are estimated in closed form and
  Viterbi decoding is exact; nothing is random, so there is no seed. The same
  input and config give byte-identical outputs
  (`tests/test_model.py::test_decoding_is_deterministic` and
  `tests/test_pipeline.py::test_two_runs_give_identical_files` check this).
* **Versions are recorded, not just pinned.** Every run writes the exact
  versions of numpy, pandas, scipy and hmmlearn to the manifest, alongside the
  git commit and a hash of the resolved config. `requirements.txt` pins the
  versions used for the published run.
* **Membership is always recomputed.** The `variant_in_coldspot` and
  `gene_has_coldspot` columns shipped in the table are never read; the
  pipeline recomputes them (and reproduces both exactly).

---

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

[MIT](LICENSE) for the code. The input data is released under CC BY 4.0 and
is governed by the terms of its original sources; see
[docs/DATA.md](docs/DATA.md).
