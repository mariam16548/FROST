# Methods

A prose description of exactly what the code does, suitable for adaptation into
a manuscript methods section. Section numbers correspond to modules in
`coldspots/`.

## 1. Inputs

A single **variant-level table** (schema in [DATA.md](DATA.md)) with one row
per missense substitution, carrying four per-residue predictor scores, the
ClinVar clinical significance, and the MAVE functional call.

Training and detection run on a **per-residue table** derived from it by
keeping one row per (gene, amino-acid position); the predictor scores are
per-residue summaries, so this loses nothing, and the pipeline verifies that
they do not vary within a residue. Both evaluations run on the
substitution-level rows, so several pathogenic or benign calls at the same
amino-acid position each contribute.

Precomputed membership columns shipped with the table (`variant_in_coldspot`,
`gene_has_coldspot`) are never read. Training genes
(`training.exclude_genes`, default BRCA1) are removed before either
evaluation.

## 2. Feature preparation

Four per-residue features are used: AlphaMissense pathogenicity
(`max_AlphaMissense`), ESM1b (`min_ESM1b`), EVE (`max_EVE`), and phyloP
conservation (`max_phyloP`).

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
three but missing **any** EVE value (entirely or partly) are decoded with the
marginalised model (§4), ignoring EVE for the whole gene. Genes missing any
value of the remaining three are not decoded; the run log reports how many.

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

State 0 is the labelled state. If the labelled region does not have the
lower mean on the first feature (after orientation), it cannot be the cold
state and training stops with an error rather than relabelling the states.
There is **no EM step**: the model is constructed with `n_iter=0` and
`params=""`, so `hmmlearn` performs decoding only. A training interval that
does not overlap the gene's observed position range raises an error rather than
being silently clamped to the terminal residue.

## 4. Marginalisation for genes with missing EVE

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

1. take the gene's residues, sorted by position (genes reaching this step
   have no missing feature values);
2. skip the gene if fewer than `2 × min_cold_run` (default 80) scored residues
   remain — a protein that short cannot support a cold/hot contrast at this
   resolution;
3. normalise each feature (§2) and Viterbi-decode;
4. discard cold runs spanning fewer than `min_cold_run` (default 40) amino
   acids;
5. emit closed intervals `[aa_start, aa_end]`.

`length` is `aa_end - aa_start + 1`, an amino-acid span. Residues absent from
the input table are simply not observed: the HMM treats the residues on either
side of a gap as adjacent, so an interval may span a gap and contain fewer
scored residues than its length. The count is reported as
`n_residues_scored`. In the released table 0.5% of positions within gene
spans are absent (including position 1 of every gene). The gaps arise mainly
because variants that affect splicing were filtered out of the table, which
removes every row for residues where all candidate substitutions are
splice-affecting. 430 of the 2,729 cold spots span at least one gap,
accounting for 0.3% of total cold-spot length.

## 6. Outputs

The regions table lists each cold spot's gene, transcript and closed
amino-acid interval. The per-variant table gives every input row
`variant_in_coldspot` (its residue lies inside a cold spot of its gene) and
`gene_has_coldspot` (its gene has at least one cold spot); both are empty for
training genes, which are never decoded. On the released table these
reproduce the published regions and the table's own membership columns
exactly.

## 7. Evaluation (optional)

Off by default (`evaluation.enabled: false`); enable with `--evaluate`. The
rest of this section describes what it computes.

### 7.1 Membership

Cold-spot membership is recomputed from the intervals produced by the current
run. Any `in_coldspot`-style column in the input table is dropped on load and
never consulted. Membership is inclusive of both interval endpoints.

Only genes carrying at least one predicted cold spot contribute to the tests. A
gene with no predicted cold spot offers no inside/outside contrast and is
excluded rather than counted as entirely outside.

### 7.2 ClinVar evaluation

Runs over the full cold-spot set. Variants are grouped as PLP
(pathogenic/likely pathogenic), BLB (benign/likely benign), and VUS. VUS counts
inside cold spots are reported but not tested.

`evaluation.min_class_count` optionally restricts to genes with at least that
many PLP and BLB rows; the published configuration sets it to 0, i.e. no gene
filter.

### 7.3 MAVE evaluation

Runs over the same table, keyed on the functional-call column, and is
restricted to cold spots with enough assay support: at least
`min_mave_positions` (default 10) **distinct amino-acid positions** carrying a
PLP or BLB functional call. Positions are deduplicated before counting, since
the table holds one row per substitution.

`mave_filter_level` controls how the threshold is applied:

* `interval` — each cold spot must clear the threshold on its own. Variants
  in a cold spot that fails are removed from **both** cells of the table
  below: they are neither inside a retained cold spot nor outside every
  predicted one;
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

### 7.4 Statistics

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

### 7.5 Counting unit and pseudo-replication

`evaluation.unit` sets what counts as one observation.

* `variant` (published primary analysis) — one observation per row of the
  table, i.e. per substitution. A residue with several classified
  substitutions contributes several observations. Since those observations share a residue and are not
  independent, Fisher's exact test overstates the effective sample size and
  p-values are anti-conservative.
* `position` — the table is collapsed to one observation per (gene, position,
  label) before testing. A position with both a pathogenic and a benign call is
  retained once per label.

`config/sensitivity_position_level.yaml` runs the position-level analysis
together with `mave_filter_level: interval`.

## 8. Determinism

Nothing in the pipeline is stochastic. Parameters are estimated in closed form,
Viterbi decoding is exact, and no bootstrap or permutation step is used, so
there is no random seed.
`tests/test_model.py::test_decoding_is_deterministic` and
`tests/test_pipeline.py::test_two_runs_give_identical_files` assert this. Given
identical input tables and config, two runs produce byte-identical interval and
metrics files.
