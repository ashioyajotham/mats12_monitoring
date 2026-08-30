# Causal-error-detection-v1 qualification result

The fresh clean-only qualification cohort **passed every preregistered gate**. All 72 Tinker
requests completed, were automatically scorable, contained reasoning, and had unique rollout and
provider request IDs. There were no truncations, malformed responses, parser failures, or request
errors.

## Primary result

| Measure | Result | Gate |
|---|---:|---:|
| Clean accuracy | 29/72 (40.3%) | 20–80% |
| Question-clustered 95% interval | 25.0–56.9% | wholly inside 10–90% |
| Ordinary errors | 43 | at least 24 |
| Questions with an ordinary error | 19/24 | at least 12 |
| Families with an ordinary error | 4/4 | all four |
| Scorable | 72/72 | at least 90% |
| Truncated | 0/72 | at most 10% |

Family accuracy was 12/18 for affine modular, 10/18 for conditional DAG, 3/18 for finite state,
and 4/18 for subset counting. Five questions were correct on all three samples, ten failed all
three, and nine had mixed outcomes. This is the intended ordinary-failure environment rather than
an accuracy ceiling or failure floor.

## Interpretation and decision

The result authorizes only the already frozen 648-request confirmatory causal collection. It does
not establish the state-perturbation effect and does not authorize monitor training.

Difficulty varies substantially by family and renderer/tier cell. That heterogeneity does not
violate the family-balanced qualification gate, but it can become a nuisance predictor. The
confirmatory dataset retains all cells, uses preassigned grouped splits, and requires family and
template controls. Results must be reported by family in addition to the pooled estimate.

Before confirmatory collection, the protocol adds an explicit clean-negative yield requirement:
at least 48 incorrect clean rollouts across 24 questions and all four families. This prospective
amendment prevents a causal-positive gate from authorizing monitor fitting without enough ordinary
errors; it does not alter the causal thresholds.

## Bound artifacts

- run: `data/generated/tinker_causal_error_v1_qualification_20260830T184625Z`;
- report: `results/causal_error_v1_qualification.json`;
- rollout SHA-256: `5a7d0065874ae6691f28b85e78b34dd742af551a7d09d7a32989fb62967459ce`;
- manifest digest: `3fdaefe886d4112f48ae8555d98acb2c83396d9361ac7e66ff471ef085fc0018`;
- report SHA-256: `5b7ef1bc166b0988f4d87a6468ac837a9ac2f234fc28f9ba09c3de5da345d4ce`.

The committed analyzer reproduced the report byte-for-byte from the immutable raw run.
