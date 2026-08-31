# Preregistration: mechanism-held-out causal audit v2

Status: **frozen before any causal-audit-v2 model request**

## Question and claim boundary

The study asks whether the frozen counterfactual answer-shift method generalizes from the
causal-error-v1 `+1` checkpoint perturbation to unseen omission and duplication transformations.
It is a counterfactual causal audit requiring condition-labelled sibling rollouts. It is not an
online single-trace monitor and does not test hidden influence, deception, intent, or general
chain-of-thought faithfulness.

The solver, renderer, generation parameters, four procedural families, correct-state control,
label rule, and answer-shift equation remain fixed. This isolates corruption-mechanism transfer.

## Frozen mechanisms and bank

- `drop_component`: replace the selected nonzero checkpoint value with zero;
- `duplicate_component`: replace it with twice its value.

For modular checkpoints the transformed value is reduced modulo the partial CRT modulus. Every
target is independently propagated through the untouched suffix. A candidate is regenerated if
the transformation is unchanged, has no valid continuation, or yields the gold answer.

The fresh bank contains 96 questions with no prior question IDs or prompts:

- 24 qualification questions: 3 per family/mechanism cell;
- 72 confirmatory questions: 9 per family/mechanism cell;
- four families, both difficulty tiers, and both renderers represented prospectively;
- the entire confirmatory partition is one external test cohort, with no v2 fitting or threshold
  selection.

Certificates bind the mechanism, original and transformed values, signed delta, exact propagated
target, source oracle, prefixes, partition, hashes, and code revision.

## Collection and stopping gates

Qualification uses two samples in each of clean, correct-continuation, and corrupted-continuation
conditions: 144 calls. It passes only with at least 90% scorable, at most 10% truncated, exact
provenance, at least a 10-point target-uptake effect for each mechanism, and target uptake in all
eight family/mechanism cells. Failure stops the study; mechanisms and wording are not redesigned.

Confirmatory collection is authorized only by a passing qualification. It uses three samples per
condition: 648 calls. Total planned Tinker spend is 792 calls, approximately 1.46 million tokens
and four to five serialized hours at the v1 observed rate.

A valid confirmatory study requires:

- at least 90% scorable and at most 10% truncated;
- a corrupted-minus-clean exact-target effect of at least 20 points with a positive
  question-clustered 95% lower bound;
- at least 18 corrupted target selections and 24 ordinary clean errors;
- both error sources represented in every family/mechanism cell;
- correct-state accuracy no more than 15 points below clean;
- exact request, seed, provider-ID, question, configuration, and certificate integrity.

## Frozen evaluation

Primary positives are corrupted-condition errors selecting the certified target. Primary
negatives are incorrect clean rollouts. The complete causal gate must pass before evaluation.

Counterfactual answer shift and the surface baseline are applied unchanged. The surface model is
fit only on causal-error-v1 training questions. Thresholds are copied from v1 validation:

- answer shift: `1.0`;
- surface: `0.5888128794999244`.

Report external AUROC, AUPRC, FPR at 50% recall, exact-binomial threshold-transfer recall/FPR,
question-clustered intervals, and descriptive family/mechanism strata. The primary paired contrast
is answer shift minus surface. Answer-shift discrimination is supported only if its AUROC lower
bound exceeds 0.5. Superiority is claimed only for contrasts whose paired 95% interval excludes
zero.

## Frozen zero-credit robustness analyses

On the existing v1 test partition, report:

- a correct/corrupted condition-identity swap placebo;
- one- and two-sibling-per-condition scores;
- corrupted-cell recurrence alone versus clean/correct control suppression alone.

A reduced-sibling variant is merely an exploratory cheaper candidate if it retains at least 90%
of full AUROC and loses no more than 0.05 absolute AUROC. These analyses cannot alter v2 methods.

No new judge calls, solver, task family, hybrid fitting, activation probing, or intervention design
is permitted in this study.
