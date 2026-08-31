# Causal-error-detection-v1 monitor result

The frozen behavioural evaluation answers the project’s current research question positively but
narrowly: **counterfactual answer shift distinguishes certified causal errors from ordinary errors
substantially better than transcript surface features at the preregistered operating metric.** A
generic Qwen judge does not, and adding the judge scores to a hybrid weakens the strongest monitor.

The held-out test set contains 49 known-wrong responses: 20 certified-target causal errors and 29
ordinary errors, grouped across 15 untouched questions.

| Monitor | AUROC | AUPRC | FPR at 50% recall |
|---|---:|---:|---:|
| Counterfactual answer shift | 0.917 | 0.893 | 0.000 |
| Hybrid | 0.814 | 0.822 | 0.000 |
| Reasoning length | 0.747 | 0.659 | 0.207 |
| Surface features | 0.738 | 0.694 | 0.172 |
| Template identity | 0.608 | 0.510 | 0.379 |
| Transcript-only Qwen judge | 0.586 | 0.458 | 0.621 |
| Context-aware Qwen judge | 0.512 | 0.440 | 0.690 |
| Correctness only | 0.500 | 0.408 | 1.000 |
| Shuffled-label surface | 0.405 | 0.393 | 0.655 |

Question-clustered bootstrap intervals for answer shift were 0.781–0.985 AUROC and 0.707–0.982
AUPRC. Relative to surface, its paired differences were +0.179 AUROC (95% interval −0.005 to
+0.350), +0.199 AUPRC (−0.009 to +0.375), and **−0.172 FPR at 50% recall (−0.419 to −0.032)**.
Thus the strongest confirmatory evidence is the preregistered operational FPR improvement; ranking
metric improvements are large but their intervals narrowly include zero.

## Threshold transfer and hard negatives

At thresholds selected only on validation data, answer shift achieved 30% test recall and 0% test
FPR; it was conservative relative to the 50% target. The hybrid achieved exactly 50% test recall
and 0% test FPR. Surface achieved 60% recall with 24.1% FPR. Zero observed FPR on 29 ordinary test
errors is promising, not proof of zero deployment false alarms.

The separate 96-example hard-negative audit was descriptive and did not affect fitting or the
primary result. At validation-selected thresholds, answer shift flagged 3/96 (3.1%), surface
43/96 (44.8%), transcript-only judge 80/96 (83.3%), and context-aware judge 76/96 (79.2%). The
judges therefore exhibit the exact “cry wolf” failure mode the project was designed to measure.

Answer shift was strong in every family on the small test strata (AUROC 0.841–1.000). The hybrid
was heterogeneous: its affine-modular AUROC was 0.480 despite strong results elsewhere. The fitted
hybrid assigned positive weight to both weak judge scores, so fusion imported their errors and did
not improve upon answer shift.

## Interpretation

The result is not “GPT-OSS reasons on math,” nor evidence about hidden influence or deception. It
shows that this controlled environment contains enough ordinary failures and randomized causal
errors to evaluate monitors, and that distributional sibling evidence is much more useful than a
generic single-transcript judge. Reasoning length performs almost as well as the broader surface
baseline, confirming that presentation artefacts remain a serious nuisance control.

Counterfactual answer shift requires repeated condition-matched rollouts and is not a cheap online
transcript monitor. External validity, low-prevalence calibration, larger held-out groups, and
performance outside these four synthetic procedural families remain open.

## Artifact binding

- full judge run: `data/generated/tinker_causal_error_judge_full_20260831T054345Z`;
- judge manifest SHA-256: `4a8661d9206258c58ace13562dd7ddebde5ee2aa032fa216139eb31f467fe5ac`;
- judge score SHA-256: `abbd73277b2c138f1e477f75974c0f14be084daebd12e7336d3a44a1bfc306a0`;
- local metrics SHA-256: `0de537d9c1cec1dd692ceb43bb1c5f5afc215ccdc66e598c79a5fc9b49706314`;
- complete metrics: `results/causal_error_v1.monitor_metrics.json`;
- complete metrics SHA-256: `2163cdfe3a10f941644c95fa775861a5a4a7d3e66f509202ba2c3b612bde88e1`;
- evaluation manifest SHA-256: `5ccdb8a686ac79de2fa25a2ace8dad9bca8a6776796f34173b1e403b77742afd`.

The judge run contains 628/628 unique score identities and provider request IDs. Its final manifest
binds both the original and parser-amendment plans and records the four preserved historical
failures from the first attempt.
