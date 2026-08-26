# Input data

The input tables are **not redistributed in this repository**. They are derived
from third-party resources with their own licence and access terms, and are too
large for version control. This document specifies the schemas precisely enough
to rebuild them, and records the versions used.

> **Fill in the bracketed placeholders before submission.** Reviewers will check
> that every score source is named with a version and access route.

## Obtaining the data

Both input tables are archived on Zenodo:

doi:10.5281/zenodo.22106323

`VEP_scores_with_plddt13_with_08_2026_clinvar.csv`            
`VEP_scores_with_plddt13_expanded_with_08_2026_clinvar_mave.csv`    


## Table 1 — collapsed per-residue table

`data.vep_residue_csv`. One row per (gene, amino-acid position). Used for
parameter estimation and cold-spot detection.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `gene_symbol` | str | yes | HGNC symbol |
| `aa_pos` | int | yes | 1-based amino-acid position |
| `am_pathogenicity` | float | yes | higher = more damaging |
| `ESM1B_score` | float | yes | **more negative = more damaging**; sign-flipped by the pipeline |
| `max_EVE` | float | no | may be missing for an entire gene, which routes it to the marginalised model |
| `phyloP100way` | float | yes | higher = more conserved |
| `plddt` | float | no | carried through; not used by the model |

A gene missing any of `am_pathogenicity`, `ESM1B_score`, `phyloP100way` at any
position is not decoded. `max_EVE` missingness is handled per gene, not per
residue: a gene is routed to the marginalised model only if it lacks EVE
entirely.

## Table 2 — expanded variant-level table

`data.vep_expanded_csv`. One row per nucleotide-level variant. Used for both
evaluations.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `gene_symbol` | str | yes | must match Table 1 |
| `aa_pos` | int | yes | must match Table 1's coordinate system |
| `new_CLNSIG_simple` | str | yes | `PLP`, `BLB`, `VUS`, or missing |
| `ExC_assignment` | str | yes | `PLP`, `BLB`, `IR`, or missing |
| `in_coldspot`, `in_coldspot_bw` | bool | no | **dropped on load** — membership is always recomputed |

Several rows may share an `aa_pos` (one per codon change). This is what makes
the counting-unit distinction in `evaluation.unit` necessary; see
[METHODS.md §6.5](METHODS.md).

The MAVE column name is resolved at load time: if `columns.mave` is absent, the
names in `columns.mave_candidates` are tried in order, and the resolved name is
written to the run manifest.

## Coordinate consistency

The two tables must agree on transcript and amino-acid numbering. The pipeline
joins on `(gene_symbol, aa_pos)` and cannot detect a numbering mismatch — it
would silently produce wrong membership. If the tables were built from
different transcript sets, fix that before running.

Worth stating in the paper: which transcript was used per gene (MANE Select,
canonical Ensembl, or a curated list), and where that mapping lives.

## Synthetic substitute

For testing the code without the real data:

```bash
python scripts/make_synthetic_data.py --out-dir data/synthetic
python -m coldspots.cli --config config/synthetic.yaml
```

The generator writes schema-faithful tables with an artificial cold/hot signal,
a gene lacking EVE entirely, a gene too short to decode, and stale
`in_coldspot` columns — exercising every branch of the pipeline. **Results from
synthetic data carry no biological meaning.**
