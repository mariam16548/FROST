# Methods

A prose description of exactly what the code does, suitable for adaptation into
a manuscript methods section. Section numbers correspond to modules in
`coldspots/`.

## 1. Inputs

Two tables (schemas in [DATA.md](DATA.md)):

* a **collapsed per-residue table**, one row per (gene, amino-acid position),
  carrying the four predictor scores;
* an **expanded variant-level table**, one row per nucleotide-level variant,
  carrying ClinVar clinical significance and MAVE functional calls.

Detection runs on the collapsed table. Both evaluations run on the expanded
table. There is no separate MAVE file — functional calls are already merged
into the expanded table, so multiple pathogenic or benign calls at the same
amino-acid position each contribute to both evaluations.

Training genes (`training.exclude_genes`, default BRCA1) are removed from the
expanded table before either evaluation.

## 2. Feature preparation

Four features are used: AlphaMissense pathogenicity (`am_pathogenicity`),
ESM1b (`ESM1B_score`), EVE (`max_EVE`), and phyloP 100-way conservation
(`phyloP100way`).

ESM1b is sign-flipped so that larger values indicate greater predicted damage
for every feature. Each feature vector is then winsorised at its 1st and 99th
percentiles and min–max scaled to [0, 1].

**Scaling is computed within each gene.** At training time the vector is the
training gene's residues; at detection time it is the target gene's residues.
Consequently the emission means are expressed in each gene's own normalised
units, and a cold-state call is relative to that protein's own score
distribution rather than to an absolute pathogenicity threshold. This is a
deliberate property of the method: it makes the model portable across proteins
with systematically different baseline conservation, at the cost of
interpretability on an absolute scale.

Genes are partitioned by score availability. Genes with complete values for all
four features are decoded with the full model; genes complete for the other
three but entirely missing EVE are decoded with the marginalised model
(§4). Genes missing any of the remaining three are not decoded.

## 3. Model and parameter estimation

A two-state Gaussian HMM. State 0 is "cold" (tolerant), state 1 is "hot".

Parameters are estimated **supervised, in closed form**, from labelled
intervals in the training gene. With BRCA1 residues 224–1366 labelled cold and
the remainder of the protein labelled hot:

* **Emissions.** Per-state, per-feature mean and standard deviation of the
  normalised scores. Features are treated as conditionally independent given
  the state, so the covariance is diagonal (a small constant, 1e-6, is added to
  each standard deviation for numerical stability). `covariance_type="full"` is
  used at decode time, with off-diagonal entries zero.
* **Initial state.** The empirical proportion of labelled cold and hot residues.
* **Transitions.** Adjacent-residue transition counts with add-one (Laplace)
  smoothing, row-normalised.

State indices are ordered so that state 0 has the lower mean on the first
feature. There is **no EM step**: the model is constructed with `n_iter=0` and
`params=""`, so `hmmlearn` performs decoding only. A training interval that
does not overlap the gene's observed position range raises an error rather than
being silently clamped to the terminal residue.

## 4. Marginalisation for genes lacking EVE

Rather than fitting a second model on three features, the trained
four-dimensional Gaussian is **marginalised** onto the three retained
coordinates: the means become the corresponding sub-vector and the covariance
the corresponding sub-matrix. For a diagonal covariance this marginal is exact.
Initial-state and transition probabilities are unchanged, since they do not
depend on the emission dimension.

This matters for interpretation: the reduced model is not an independently
fitted three-feature model but the same model observed in fewer dimensions, so
its cold state means the same thing.

## 5. Detection

For each gene outside the training set:

1. drop residues with any missing feature value and sort by position;
2. skip the gene if fewer than `2 × min_cold_run` (default 80) scored residues
   remain — a protein that short cannot support a cold/hot contrast at this
   resolution;
3. normalise each feature (§2) and Viterbi-decode;
4. discard cold runs spanning fewer than `min_cold_run` (default 40) amino
   acids;
5. emit closed intervals `[aa_start, aa_end]`.

`length` is `aa_end - aa_start + 1`, an amino-acid span. Because unscored
residues were dropped in step 1, an interval may contain fewer scored residues
than its length; the count is reported as `n_residues_scored`.

## 6. Evaluation

### 6.1 Membership

Cold-spot membership is recomputed from the intervals produced by the current
run. Any `in_coldspot`-style column in the input table is dropped on load and
never consulted. Membership is inclusive of both interval endpoints.

Only genes carrying at least one predicted cold spot contribute to the tests. A
gene with no predicted cold spot offers no inside/outside contrast and is
excluded rather than counted as entirely outside.

### 6.2 ClinVar evaluation

Runs over the full cold-spot set. Variants are grouped as PLP
(pathogenic/likely pathogenic), BLB (benign/likely benign), and VUS. VUS counts
inside cold spots are reported but not tested.

`evaluation.min_class_count` optionally restricts to genes with at least that
many PLP and BLB rows; the published configuration sets it to 0, i.e. no gene
filter.

### 6.3 MAVE evaluation

Runs over the same expanded table, keyed on the functional-call column, and is
restricted to cold spots with enough assay support: at least
`min_mave_positions` (default 10) **distinct amino-acid positions** carrying a
PLP or BLB functional call. Positions are deduplicated before counting, since
the expanded table holds one row per nucleotide change.

`mave_filter_level` controls how the threshold is applied:

* `interval` — each cold spot must clear the threshold on its own;
* `gene` (published default) — the threshold applies to the **sum** across a
  gene's cold spots, and every cold spot in a passing gene is retained,
  including those with few or no assayed positions. Cold spots within a gene
  are disjoint, so summing distinct-position counts equals counting their
  union. This is the looser setting and should be stated as such.

Each retained region reports `n_PLP`, `n_BLB`, `n_positions`, `n_assayed`
(distinct positions with any call, including intermediate/indeterminate), and
`frac_assayed = n_assayed / length`. `frac_assayed` should be quoted alongside
any depletion claim: a region can clear the position threshold on a handful of
clustered residues at one edge while most of it is unassayed.

### 6.4 Statistics

For each evaluation, a 2×2 table is formed:

|                     | inside cold spot | outside |
| ------------------- | ---------------- | ------- |
| BLB (benign)        | `blb_in`         | `blb_out` |
| PLP (pathogenic)    | `plp_in`         | `plp_out` |

The odds ratio is `(blb_in · plp_out) / (blb_out · plp_in)`, i.e. **the odds of
a benign rather than pathogenic call inside a cold spot relative to outside**.
OR > 1 indicates depletion of pathogenic variation. The point estimate and the
95% Wald confidence interval on the log scale use the Haldane–Anscombe
correction (+0.5 to every cell) so they remain finite when a cell is empty. The
p-value is Fisher's exact test on the **uncorrected** counts; this is the
conventional pairing, but it does mean the p-value and the interval are not
derived from an identical table. Raw counts are written to `*_metrics.json` so
that any other convention can be applied.

Also reported is the rate ratio `rate(PLP inside) / rate(BLB inside)`, labelled
`LR-` in the original script: the likelihood ratio for cold-spot membership
given a pathogenic versus a benign variant. Values below 1 mean that membership
argues against pathogenicity.

### 6.5 Counting unit and pseudo-replication

`evaluation.unit` sets what counts as one observation.

* `variant` (published primary analysis) — one observation per row of the
  expanded table. A residue with several codon changes contributes several
  observations. Since those observations share a residue and are not
  independent, Fisher's exact test overstates the effective sample size and
  p-values are anti-conservative.
* `position` — the table is collapsed to one observation per (gene, position,
  label) before testing. A position with both a pathogenic and a benign call is
  retained once per label.

`config/sensitivity_position_level.yaml` runs the position-level analysis
together with `mave_filter_level: interval`. Reporting both, and quoting the
position-level p-value as the conservative bound, is strongly recommended.

## 7. Determinism

Nothing in the pipeline is stochastic. Parameters are estimated in closed form,
Viterbi decoding is exact, and no bootstrap or permutation step is used. The
RNG seed is set and recorded for completeness but does not affect any output;
`tests/test_model.py::test_decoding_is_deterministic` asserts this. Given
identical input tables and config, two runs produce byte-identical interval and
metrics files.
