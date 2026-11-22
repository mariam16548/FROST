# The input table

FROST needs one file, archived on Zenodo under CC BY 4.0:

**[doi:10.5281/zenodo.24398228](https://doi.org/10.5281/zenodo.24398228)** —
`variant_level_coldspots.csv.gz`

Download it into `data/` in the repository (the path `config/default.yaml`
expects), or point the pipeline at it with
`--variant-csv /path/to/variant_level_coldspots.csv.gz`. The file is read
directly; there is no need to decompress it.

The released table has 12,849,029 rows covering 2,749 genes (one transcript
per gene) and 2,255,310 residues.

## Schema

One row per missense substitution. Columns used by the pipeline are marked;
the rest are carried for annotation and ignored.

| Column | Used | Notes |
| --- | --- | --- |
| `gene_symbol` | yes | HGNC symbol |
| `protein_variant` | yes | e.g. `C2R`; the amino-acid position (`2`) is parsed from it. A table may supply an integer `aa_pos` column instead |
| `max_AlphaMissense` | yes | per-residue maximum AlphaMissense; higher = more damaging |
| `min_ESM1b` | yes | per-residue minimum ESM1b; **more negative = more damaging**, sign-flipped by the pipeline |
| `max_EVE` | yes | per-residue maximum EVE; may be missing — a gene with any missing value uses the 3-feature model |
| `max_phyloP` | yes | per-residue maximum phyloP; higher = more conserved |
| `transcript_id` | yes | copied into the outputs; one per gene |
| `CLNSIG` | evaluation only | ClinVar: `PLP`, `BLB`, `VUS`, or empty |
| `functional_assignment` | evaluation only | MAVE call: `Abnormal` (counted as PLP), `Normal` (BLB), `Indeterminate` (IR), or empty |
| `pLDDT`, `AlphaMissense`, `REVEL`, `MutPred2`, `pfam_domain` | no | annotation |
| `gene_has_coldspot`, `variant_in_coldspot` | **never read** | the published calls, for convenience; the pipeline recomputes them and its `*_variants.csv.gz` matches both columns on every row |

The four predictor columns are per-residue summaries, so they repeat across
the substitutions at a residue. The pipeline checks this and stops with an
error if a residue carries different values. A gene with any missing
`max_AlphaMissense`, `min_ESM1b` or `max_phyloP` value is not decoded (none
are in the released table).

`CLNSIG` and `functional_assignment` are read only when the optional
evaluation is on (`--evaluate`).

## Using a different table

Column names and label wording are set in the config (`columns.*`,
`features.*`, `evaluation.clinvar_labels`, `evaluation.mave_labels`), so a
table with other names can be used without changing code.
