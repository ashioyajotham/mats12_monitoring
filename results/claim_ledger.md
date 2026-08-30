# Claim ledger

No confirmatory empirical claims are established yet. The exploratory observations below define
which effects require replication; they are not monitor-performance claims.

| ID | Proposed claim | Required evidence | Artifact | Status |
|---|---|---|---|---|
| C1 | Incorrect hints causally shift answers | Paired answer distributions, group bootstrap CI, manual transcript audit | TBD | Unrun |
| C2 | Resampling beats surface monitoring | Frozen grouped test; FPR@50% recall and CI | TBD | Unrun |
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
| E9 | Certified user-prompt state perturbations cause exact downstream errors | Fresh prospectively generated replication under the causal-error estimand | `docs/PREREGISTRATION_CAUSAL_ERROR_DETECTION_V1.md` | Proposed; prior 13/24 versus 0/24 result is exploratory only |

## Rules

- Every number in the executive summary links to a committed artifact.
- Predictive results do not receive causal wording.
- Failed gates and null results remain in the ledger.
- Claims are downgraded when confidence intervals include practically important alternatives.
