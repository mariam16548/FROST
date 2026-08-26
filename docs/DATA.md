# Obtaining the data

Both input tables are archived on Zenodo:

doi:10.5281/zenodo.22106323

`VEP_scores_with_plddt13_with_08_2026_clinvar.csv`            
`VEP_scores_with_plddt13_expanded_with_08_2026_clinvar_mave.csv`    

## Table 1 — collapsed per-residue table

Config key `data.vep_residue_csv`; archived as `VEP_scores_with_plddt13_with_08_2026_clinvar.csv`. 
One row per (gene, amino-acid position). Used for parameter estimation and coldspot detection — not for evaluation.

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

Config key `data.vep_expanded_csv`; archived as `VEP_scores_with_plddt13_expanded_with_08_2026_clinvar_mave.csv`. 
One row per nucleotide-level variant. Used for both evaluations — not for detection.

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

