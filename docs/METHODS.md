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

A two-state Gaussian HMM with states Cold (state 0) and Not-Cold (state 1).

Parameters are estimated **supervised, in closed form**, from labelled
intervals in the training gene. With BRCA1 residues 224–1366 labelled Cold and
the remainder of the protein labelled Not-Cold:

* **Emissions.** Per-state, per-feature mean and standard deviation of the
  normalised scores. Features are treated as conditionally independent given
  the state, so the covariance is diagonal (a small constant, 1e-6, is added to
  each standard deviation for numerical stability). `covariance_type="full"` is
  used at decode time, with off-diagonal entries zero.
* **Initial state.** The empirical proportion of labelled Cold and Not-Cold residues.
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
   remain — a protein that short cannot support a Cold/Not-Cold contrast at this
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

Off by default (`evaluation.enabled: false`); enable with `--evaluate`. This
computes the manuscript's benchmarking metrics. Each row of the table is one
amino-acid substitution and is one observation. Cold-spot membership is
recomputed from this run's intervals and is inclusive of both endpoints.
Training genes (BRCA1) are excluded.

### 7.1 ClinVar

Restricted to genes containing at least one predicted cold spot. Substitutions
are grouped as PLP, BLB and VUS; VUS inside cold spots are counted but not
tested.

### 7.2 MAVE

Calls are Abnormal (ExCALIBR points > 0, counted as PLP), Normal (< 0,
counted as BLB) and Indeterminate (0). Genes without predicted cold spots are
excluded, as are genes whose Abnormal/Normal calls cover fewer than
`min_mave_positions` (10) distinct amino-acid positions within their
predicted cold spots, summed across all cold spots in the gene.

### 7.3 Statistics

For each evaluation, a 2×2 table is formed:

|                     | inside cold spot | outside |
| ------------------- | ---------------- | ------- |
| BLB / Normal        | `blb_in`         | `blb_out` |
| PLP / Abnormal      | `plp_in`         | `plp_out` |

The odds ratio is `(blb_in · plp_out) / (blb_out · plp_in)`, so OR > 1 means
benign (normal) substitutions are enriched inside cold spots relative to
pathogenic (abnormal) ones. A Haldane–Anscombe correction (+0.5) is applied to
every cell; the 95% confidence interval is the Woolf interval, and the p-value
is a two-sided Fisher's exact test on the uncorrected counts. The negative
likelihood ratio is `LR- = rate(PLP inside) / rate(BLB inside)`. Raw counts
are written to `*_metrics.json`.

## 8. Determinism

Nothing in the pipeline is stochastic. Parameters are estimated in closed form
and Viterbi decoding is exact, so there is no random seed. Given identical
input and config, two runs produce byte-identical outputs
(`tests/test_model.py::test_decoding_is_deterministic` and
`tests/test_pipeline.py::test_two_runs_give_identical_files`).
