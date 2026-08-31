# Preregistration: qualification-informed causal audit v2.1

Status: **frozen before any v2.1 model request**

## Motivation and claim boundary

Causal-audit-v2 qualification found strong pooled transfer for omission and duplication
checkpoint corruptions but zero exact-target uptake in both subset-counting cells. The original v2
stopping rule was enforced and its confirmatory run remains closed.

This separately preregistered follow-up asks whether the frozen v1 counterfactual answer-shift
method transfers to the two unseen corruption mechanisms **within the three qualification-supported
recurrence families**: affine modular, conditional DAG, and finite state. Family selection is
explicitly informed by v2 qualification outcomes. The result cannot establish generalization to
subset counting or to a representative task distribution.

This remains a counterfactual causal audit requiring condition-labelled siblings, not an online
single-trace monitor or evidence of hidden influence, deception, or chain-of-thought faithfulness.

## Frozen construction and collection

- 72 entirely fresh questions, with no overlap against any previous bank;
- 12 questions in each of six family/mechanism cells;
- balanced difficulty tiers and renderers within every cell;
- `drop_component` and `duplicate_component` transformations unchanged from v2;
- clean, correct-continuation, and corrupted-continuation conditions;
- three samples per condition, for exactly 648 planned Tinker calls;
- the entire cohort is an external test: no v2.1 training, validation, threshold selection, item
  filtering, or replacement.

Every source and intervention is independently verified and content-addressed before collection.
The certificates explicitly record the qualification-informed selection basis.

## Causal and integrity gate

Monitor evaluation is authorized only if all checks pass:

- 648/648 requests stored, unique seeds/rollout IDs/provider IDs, and zero request errors;
- at least 90% scorable, at most 10% truncated, and reasoning present for every scorable response;
- exact source, mechanism, condition, configuration, and certificate verification;
- corrupted-minus-clean target effect at least 20 points with positive question-clustered 95%
  lower bound;
- at least 18 corrupted exact-target selections and 24 ordinary clean errors;
- both causal targets and ordinary errors occur in all six family/mechanism cells;
- correct-state accuracy is no more than 15 points below clean accuracy.

Failure stops monitor evaluation. Gates, families, prompts, and mechanisms cannot be changed after
collection.

## Frozen external evaluation

Primary positives are corrupted-condition errors selecting their certified propagated target.
Primary negatives are completed incorrect clean rollouts. Apply:

- counterfactual answer shift with all available siblings;
- the surface baseline fit only on causal-error-v1 training examples.

Transfer the v1 validation thresholds unchanged: answer shift `1.0`, surface
`0.5888128794999244`. Report AUROC, AUPRC, FPR at 50% recall, question-clustered 95% intervals,
exact-binomial threshold-transfer recall/FPR, and descriptive family/mechanism strata.

The primary contrast is answer shift minus surface. Answer-shift discrimination is supported only
if its clustered AUROC lower bound exceeds 0.5. Metric superiority is claimed only when the paired
95% interval excludes zero. No judge, hybrid, activation probe, or alternative threshold is added.
