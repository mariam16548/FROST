# FROST: Find Regions Of Substitution Tolerance

FROST finds **coldspots**: stretches of a protein where missense variants are
rarely pathogenic. It combines four per-residue scores (AlphaMissense, ESM1b,
EVE and phyloP) in a two-state hidden Markov model trained once on the known
BRCA1 coldspot, then applies it to any gene.

This is the code for:

> Benazouz M, Fayer S, McEwen AE, Stone J, Gupta P, Rubin AF, Mooney SD,
> Fowler DM, Radivojac P, Starita LM. *Scalable identification of mutationally
> tolerant regions in 1,336 disease-associated genes.* (2026)

## Install

Requires Python 3.11 or later.

```bash
git clone https://github.com/mariam16548/FROST.git
cd FROST
pip install -r requirements.txt
pip install -e .
```

## Run

Try it on a small synthetic dataset first (takes about 10 seconds):

```bash
python scripts/make_synthetic_data.py
frost --config config/default.yaml \
    --variant-csv data/synthetic/variants.csv.gz --out-dir results/synthetic
```

To reproduce the published coldspots, download
`variant_level_coldspots.csv.gz` from Zenodo
([doi:10.5281/zenodo.24398228](https://doi.org/10.5281/zenodo.24398228)) into a
`data/` folder, then run:

```bash
mkdir -p data   # put variant_level_coldspots.csv.gz here
frost --config config/default.yaml
```

This takes about a minute and finds 2,729 coldspots in 1,336 genes. If the
file is somewhere else, add `--variant-csv path/to/file`. Run commands from
the repository root.

## Input

One table with a row per missense substitution. FROST needs the gene, the
substitution (e.g. `C2R`) and the four per-residue scores; see
[docs/DATA.md](docs/DATA.md) for the columns.

## Output

Written to `results/`:

| File | Contents |
| --- | --- |
| `coldspots_regions.tsv` | One row per coldspot: `gene_symbol`, `transcript_id`, `aa_start`, `aa_end` |
| `coldspots_variants.csv.gz` | Every input substitution with `gene_has_coldspot` and `variant_in_coldspot` |
| `coldspots_manifest.json` | Settings, package versions and fitted model parameters for the run |

BRCA1 is the training gene, so it is left blank in the variant table.

## Benchmarking against ClinVar and MAVE data

Add `--evaluate` to also compute how strongly coldspots are enriched for benign
ClinVar variants and functionally normal MAVE variants, as reported in the
paper. The odds ratios and counts are written to `coldspots_metrics.json`.

```bash
frost --config config/default.yaml --evaluate
```

## How it works

1. Scale each score (the most deleterious value at each residue) to 0–1
   within the gene.
2. Estimate the model from BRCA1, with residues 224–1366 as Cold and the rest
   as Not-Cold.
3. Find the most likely Cold/Not-Cold path through each gene (Viterbi).
4. Keep Cold runs of at least 40 residues as coldspots.

Genes missing EVE scores use the other three. The method settings are in
`config/default.yaml`, and [docs/METHODS.md](docs/METHODS.md) describes each
step in detail.

## Tests

```bash
pytest
```

## License

Code: [MIT](LICENSE). Data: CC BY 4.0; see [docs/DATA.md](docs/DATA.md).
