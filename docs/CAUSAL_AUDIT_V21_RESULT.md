# Causal-audit-v2.1 external replication result

The qualification-informed three-family replication **passed every causal, integrity, and
external-evaluation criterion**. Counterfactual answer shift transferred to fresh questions and
two unseen checkpoint transformations without v2.1 fitting or threshold selection.

This is evidence for a domain-restricted counterfactual causal audit. It is not evidence for
subset-counting generalization, hidden influence, deception, chain-of-thought faithfulness, or a
deployable single-trace monitor.

## Causal environment

All 648 planned Tinker requests were stored with zero request errors, truncations, parser failures,
or duplicate provider IDs. Exact grading classified 647 responses as correct or incorrect and one
clean response for review.

Corrupted-state target uptake was 118/216 (54.6%), compared with 3/216 (1.4%) under clean prompts:
a 53.2-point effect with a question-clustered 95% interval from 46.3 to 61.1 points. The frozen
gate passed in all six family/mechanism cells. Clean prompts supplied 108 ordinary errors.

| Mechanism | Target effect | Clustered 95% interval |
|---|---:|---:|
| Drop component | +50.0 points | +38.9 to +63.0 |
| Duplicate component | +56.5 points | +47.2 to +66.7 |

Affine drop-component was the weakest cell at 8/36 corrupted targets, but still met the cell
diversity requirement. The other five cells produced 17–24 targets each.

## Untouched external monitor evaluation

The primary external dataset contains 226 known-wrong responses: 118 certified causal errors and
108 ordinary errors. The surface model was fit only on causal-error-v1 training data. Answer shift
used its unchanged equation and all available siblings.

| Monitor | AUROC | AUPRC | FPR at ≥50% recall |
|---|---:|---:|---:|
| Counterfactual answer shift | **0.919** | **0.922** | **0.019** |
| Frozen v1 surface baseline | 0.735 | 0.717 | 0.148 |

Question-clustered 95% intervals were 0.880–0.952 for answer-shift AUROC and 0.877–0.956 for
AUPRC. The answer-shift AUROC lower bound therefore exceeded the preregistered 0.5 criterion.

Every paired contrast supported answer shift over surface:

| Answer shift minus surface | Point difference | Paired clustered 95% interval |
|---|---:|---:|
| AUROC | +0.184 | +0.105 to +0.258 |
| AUPRC | +0.205 | +0.099 to +0.287 |
| FPR at ≥50% recall | −0.130 | −0.269 to −0.057 |

Answer-shift AUROC remained 0.893–0.943 across the three families and was 0.930 for drop-component
and 0.906 for duplicate-component errors.

## Frozen-threshold transfer

At the threshold selected on v1 validation and transferred unchanged:

- answer shift (`1.0`) detected 45/118 causal errors: 38.1% recall, exact 95% interval
  29.4–47.5%; it flagged 0/108 ordinary errors, exact upper bound 3.36%;
- surface (`0.5888128794999244`) detected 74/118 causal errors: 62.7% recall, but flagged 32/108
  ordinary errors: 29.6% FPR, exact interval 21.2–39.2%.

The zero observed answer-shift false positives are a successful independent threshold transfer,
but 108 negatives still cannot establish a sub-1% deployment FPR. The more conservative threshold
also trades away substantial recall.

## Interpretation

The v1 result was not confined to the exact `+1` corruption. Distributional recurrence of a focal
answer under the corrupted condition generalizes to omission and duplication perturbations on
fresh affine, DAG, and finite-state problems. The condition-swap placebo and v2.1 transfer together
support the interpretation that the method measures intervention-conditioned answer stability,
not merely textual difficulty.

The boundary is equally important: subset counting failed qualification and is absent by design.
V2.1 therefore validates the audit in a selected supported domain. It does not erase the original
four-family external-validity failure.

## Artifact binding

- run: `data/generated/tinker_causal_audit_v21_confirmatory_20260831T121459Z`;
- rollout SHA-256: `9104a59e9e6e2fb962fb72b6a4997f849f02386ad18f220cd6b4f340632debe5`;
- run-manifest SHA-256: `5fef44032ddef1a868ce26bca16ba176931453c0a82af74bde7b3b1e2cbcce81`;
- causal report: `results/causal_audit_v21_confirmatory.json`;
- causal-report SHA-256: `1358600408cc4305cf398fc814ebdfec32d8a4c5ecf20eee52b67b68eb93a899`;
- monitor report: `results/causal_audit_v21.monitor_metrics.json`;
- monitor-report SHA-256: `f3b2f1293d24ba35a004555d54cbc15b65d0465cc831fbae6c42b4ab4cf5e9c4`.
