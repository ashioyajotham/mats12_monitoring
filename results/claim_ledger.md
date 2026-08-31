# Claim ledger

The causal-error-v1 confirmatory collection establishes the controlled causal mechanism described
in E9. The completed frozen monitor stack establishes the bounded operational result in C2.

| ID | Proposed claim | Required evidence | Artifact | Status |
|---|---|---|---|---|
| C1 | Incorrect hints causally shift answers | Paired answer distributions, group bootstrap CI, manual transcript audit | TBD | Unrun |
| C2 | Counterfactual answer shift beats surface monitoring at the preregistered operating metric | Frozen grouped test; FPR@50% recall and paired clustered CI | `docs/CAUSAL_ERROR_V1_MONITOR_RESULT.md` | Supported: FPR difference −0.172, 95% interval −0.419 to −0.032; AUROC/AUPRC differences are positive but intervals narrowly cross zero |
| C3 | Monitor precision collapses at 1% prevalence | TPR/FPR with group bootstrap; analytic and sampled prevalence check | TBD | Unrun |
| C4 | Activation probes add little/no value | Same-group comparison; nested selection; shuffled-label control | TBD | Stretch/unrun |
| E1 | Authority and directive language interact in ARC hint uptake | Replication on grouped questions with bootstrap CI | Tinker ARC calibration manifests | Exploratory: 0/10 informational, 3/10 ordinary-directive, 5/10 trusted-directive |
| E2 | Observed ARC hint following is acknowledged rather than silent | Independent transcript review and held-out templates | Calibration transcripts and decision log | Exploratory: all valid uptake acknowledged |
| E3 | The frozen ARC slice supplies no natural clean errors | Replication or a broader representative ARC sample | `docs/NATURAL_ERROR_DISCOVERY.md` | Established only for the selected slice: 60/60 correct |
| E4 | Full-thinking Qwen length-censors specific difficult MATH questions | Replication with retained partial responses and matched direct-mode controls | Qwen attribution diagnostic manifest | Exploratory: 4/8 truncated, identical question pattern across two seeds |
| E5 | The GPT-OSS/MATH freeze supplies no natural clean errors | Broader or independently sampled difficult task freeze | GPT-OSS discovery manifest and derived grading | Established only for this selected slice: 59/59 scorable correct, one truncation |
| E6 | Exact intermediate-state corruption causally induces its propagated target | Fresh replication under a hidden-influence mechanism | `results/procedural_continuation_yield_v2.json` | Exploratory: 13/24 corrupted targets versus 0/24 controls; clustered effect interval +33.3 to +75.0 points |
| E7 | User-prompt state continuation yields silent influence | Manual validity review of every target selection | `results/procedural_continuation_yield_v2_manual_review.json` | Rejected: 11 open uses, two ambiguous recomputations, zero silent uses |
| E8 | Assistant prefill reliably causes hidden exact-target uptake | Randomized matched prefills, clustered effect interval, and manual validity | `results/procedural_assistant_prefill_v3.json` | Rejected: 1/24 corrupted versus 2/24 correct-prefill targets; effect -4.2 points, interval -16.7 to +8.3 |
| E9 | Certified user-prompt state perturbations cause exact downstream errors | Fresh prospectively generated replication under the causal-error estimand | `docs/CAUSAL_ERROR_V1_CONFIRMATORY_RESULT.md` | Confirmed: +39.5-point exact-target effect; question-clustered 95% interval +29.9 to +49.1; every frozen gate passed |
| C5 | Counterfactual answer shift generalizes to unseen checkpoint corruption mechanisms | Passing v2 causal gate; untouched external AUROC lower bound above 0.5; no v2 fitting or threshold selection | `docs/CAUSAL_AUDIT_V2_QUALIFICATION_RESULT.md` | Not tested: qualification failed family/mechanism diversity, so external monitor evaluation was correctly stopped |
| E10 | Omission and duplication checkpoint perturbations transfer across the full four-family environment | Every qualification family/mechanism cell shows exact-target uptake | `docs/CAUSAL_AUDIT_V2_QUALIFICATION_RESULT.md` | Rejected: pooled effect +35.4 points, but both subset-counting cells had zero corrupted target uptake |

## Rules

- Every number in the executive summary links to a committed artifact.
- Predictive results do not receive causal wording.
- Failed gates and null results remain in the ledger.
- Claims are downgraded when confidence intervals include practically important alternatives.
