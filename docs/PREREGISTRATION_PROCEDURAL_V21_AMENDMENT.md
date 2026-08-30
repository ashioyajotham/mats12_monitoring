# Procedural-math-v2.1 adaptive subset-replacement amendment

Status: **frozen before replacement screening**  
Date: 2026-08-30

## Reason for the amendment

The v2 Tinker screen completed all 80 requests with zero request errors, 77 scorable responses,
and no parser failures. Three families had at least one eligible 30–70% accuracy cell:
`affine_modular`, `conditional_dag`, and `finite_state`. `subset_counting` alone failed. Its low
tier had useful 62.5% accuracy but only 8/10 scorable because two responses truncated; its high
tier had 0/9 accuracy and one truncation.

This is a task-calibration failure, not an infrastructure failure and not a failed demonstration
of ordinary errors. The original screen contained 33 correct and 44 incorrect scorable answers.
The exact v2 evidence reused here is bound to:

- rollouts SHA-256 `e56d237cb489bf0a82c86d59556c7cb7b71a5979c22c57f4ca618bfe6934d536`;
- screening-report SHA-256 `50374aece835d499952bd69d46e6d869a1c0b34d524a7109ecc747404888d56d`;
  and
- run directory `data/generated/tinker_procedural_v2_screening_20260830T091412Z`.

The original v2 gate remains failed. Thresholds will not be relaxed and no v2 individual
correctness label will determine question selection.

## Fresh replacement bank

Freeze 20 new `subset_counting` questions under generator version
`procedural-math-v2.1-subset-replacement`: ten `replacement_low` problems with 11 distinct
weights and cardinality four, and ten `replacement_mid` problems with 12 weights and cardinality
five. These are fresh seeds and prompts. Dynamic programming and exhaustive combinations must
agree on every exact integer answer.

Screen one clean low-reasoning GPT-OSS-20B rollout per question through Tinker. The completion
ceiling is prospectively increased from 4,096 to 8,192 tokens because truncation—not parsing—made
the promising prior low cell ineligible. The unchanged cell gate requires zero request errors,
at least 9/10 scorable responses, and 30–70% accuracy. At least one replacement tier must pass.

## Combined outcome-independent freeze

If replacement screening passes, freeze exactly ten questions from each family:

- reuse only v2 cells already declared eligible for the three successful families;
- select those questions with the original seed `20261301`;
- select subset replacements only from eligible new tiers with seed `20261701`; and
- within each eligible pool, use deterministic hash ordering with round-robin tier-renderer
  balance. Individual screening correctness is never an ordering or inclusion feature.

This adaptive calibration intentionally chooses task families and tiers using aggregate outcomes.
It does not claim an unbiased estimate of GPT-OSS mathematical ability. Its validity for the
monitoring project comes from the next stage using entirely fresh rollouts.

## Fresh validation and stopping rule

Collect three fresh clean rollouts per frozen question at the common 8,192-token ceiling. Apply
the unchanged mixed-outcome gate: all 120 requests stored; unique rollout, provider, and logical
sample identities; zero request errors; at least 90% scorable; at most 10% truncated; at least 24
correct and 24 incorrect; errors across at least six questions and three families; correct answers
across at least three families; and reasoning on every scorable response.

If the replacement screen fails, stop and replace the task family rather than running another
post-hoc rescue. If fresh validation fails, do not run interventions. Passing fresh validation
authorizes only a separately preregistered causal-yield experiment and never directly authorizes
monitor training.
