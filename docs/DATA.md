# The input table

FROST reads one file, `variant_level_coldspots.csv.gz`, available on Zenodo
under CC BY 4.0:
[doi:10.5281/zenodo.24398228](https://doi.org/10.5281/zenodo.24398228).

Put it in `data/`, or pass its path with `--variant-csv`. There is no need to
decompress it. It covers 2,749 genes (one transcript each), 2,255,310
residues and 12,849,029 substitutions.

## Schema

One row per missense substitution.

| Column | Used | Notes |
| --- | --- | --- |
| `gene_symbol` | yes | HGNC symbol |
| `protein_variant` | yes | e.g. `C2R`; the position (`2`) is read from it. An integer `aa_pos` column can be used instead |
| `max_AlphaMissense` | yes | per-residue maximum AlphaMissense; higher = more damaging |
| `min_ESM1b` | yes | per-residue minimum ESM1b; more negative = more damaging (FROST inverts it) |
| `max_EVE` | yes | per-residue maximum EVE; may be missing |
| `max_phyloP` | yes | per-residue maximum phyloP; higher = more conserved |
| `transcript_id` | yes | copied into the outputs |
| `CLNSIG` | `--evaluate` only | ClinVar: `PLP`, `BLB`, `VUS`, or empty |
| `functional_assignment` | `--evaluate` only | MAVE call: `Abnormal`, `Normal`, `Indeterminate`, or empty |
| `pLDDT`, `AlphaMissense`, `REVEL`, `MutPred2`, `pfam_domain` | no | annotation |
| `gene_has_coldspot`, `variant_in_coldspot` | no | the published coldspot calls; FROST recomputes them |

The four feature columns are per-residue values, so they repeat across the
substitutions at a residue; FROST stops with an error if they don't.

## Using a different table

Column names and label wording are set in the config (`columns`,
`features`, `evaluation`), so a table with other names works without code
changes.
