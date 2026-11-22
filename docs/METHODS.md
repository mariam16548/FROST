# Methods

How the code implements the method described in the paper. All settings are
in `config/default.yaml`.

## Input

One table with a row per missense substitution (see [DATA.md](DATA.md)). The
four feature scores are per-residue values, so FROST keeps one row per
(gene, amino-acid position) for training and detection.

## Features

Four features per residue: AlphaMissense (`max_AlphaMissense`), ESM1b
(`min_ESM1b`), EVE (`max_EVE`) and phyloP (`max_phyloP`). ESM1b is
sign-inverted so that higher values are more deleterious for every feature.
Within each gene, each feature is clipped to its 1st and 99th percentiles and
min–max scaled to [0, 1].

Genes with at least one position lacking an EVE score are analysed with the
other three features. Genes missing any AlphaMissense, ESM1b or phyloP value
are not analysed.

## Model

A two-state hidden Markov model with states Cold and Not-Cold. All parameters
are estimated once from BRCA1, taking positions 224–1366 as Cold and all other
positions as Not-Cold, and are then held fixed:

- **Initial state probabilities:** the proportion of positions in each state.
- **Transition probabilities:** the observed transitions of the labelled
  BRCA1 state sequence, with Laplace (add-one) smoothing.
- **Emissions:** Gaussian, with a mean and standard deviation per feature and
  state, and no covariance between features.

The Cold state is the state with the lower mean AlphaMissense score. For genes
analysed without EVE, the EVE terms are dropped from the same fitted model.

## Coldspot calling

For each gene except BRCA1, positions are ordered by residue number and the
most likely state sequence is found by Viterbi decoding. Runs of Cold
spanning fewer than 40 positions are reassigned to Not-Cold; the remaining
runs are the coldspots.

A coldspot's length is measured in amino-acid positions. Positions absent from
the table (mainly where variants affecting splicing were filtered out) are not
observed, so a coldspot can span a few unscored positions.

## Outputs

- **Regions:** gene, transcript, and the first and last position of each
  coldspot.
- **Variants:** each input substitution with `variant_in_coldspot` (its
  position lies in a coldspot) and `gene_has_coldspot` (its gene has at least
  one). Both are blank for BRCA1.

## Benchmarking (`--evaluate`)

Each substitution counts once, and BRCA1 is excluded.

- **ClinVar:** benign (BLB) and pathogenic (PLP) substitutions inside and
  outside coldspots, in genes with at least one coldspot. VUS inside coldspots
  are counted.
- **MAVE:** functionally normal and abnormal substitutions inside and outside
  coldspots. A gene is included if its normal or abnormal calls cover at least
  10 distinct positions inside its coldspots, summed across all of its
  coldspots.

For each, the odds ratio is

    OR = (benign_in × pathogenic_out) / (benign_out × pathogenic_in)

with a Haldane–Anscombe correction (+0.5 to every cell), a 95% Woolf
confidence interval, and a two-sided Fisher's exact test on the uncorrected
counts. The negative likelihood ratio is LR− = (fraction of pathogenic inside
coldspots) / (fraction of benign inside coldspots).

## Reproducibility

Nothing is random: parameters are computed directly and Viterbi decoding is
exact, so the same input always gives the same output. Each run writes a
manifest with the settings, package versions and fitted parameters.
