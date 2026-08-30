# Exact continuation causal-yield-v2 result

The automated causal gate **passed**, but the manual validity gate **failed**. The immutable
Tinker run stored 72/72 scorable responses with no truncations, parser failures, malformed
responses, or request errors.

The corrupted state caused a large, target-specific shift: 13/24 corrupted continuations selected
the exactly propagated wrong answer, compared with 0/24 clean and 0/24 correct-continuation
samples. The corrupted-minus-clean effect was 54.2 percentage points, with a question-clustered
95% bootstrap interval of 33.3 to 75.0 points. Uptake occurred on seven of eight questions and in
all four task families.

## Manual validity result

The frozen phrase matcher called all 13 target selections nominally unacknowledged. Manual review
did not confirm that label:

- 11 traces explicitly took the supplied residue, vector, or state as their continuation point;
- two subset-counting traces independently enumerated combinations and were marked ambiguous;
- zero traces qualified as defensible silent continuation use.

The automated report remains unchanged. Its pass establishes causal state propagation, not hidden
influence. The machine-readable manual review is
`results/procedural_continuation_yield_v2_manual_review.json`.

## Decision

Ordinary user-prompt continuation wording is exhausted under the preregistered stopping rule.
Monitor training remains unauthorized. The one remaining mechanism test is a new preregistered
assistant-prefill diagnostic in which the corrupted reasoning is rendered inside the model's open
assistant analysis turn. Failure of its causal or manual validity gate triggers a reframe toward
detecting causally induced reasoning errors rather than silent unfaithfulness.

## Bound artifacts

- run: `data/generated/tinker_procedural_continuation_yield_v2_20260830T143051Z`;
- automated report: `results/procedural_continuation_yield_v2.json`;
- manual review: `results/procedural_continuation_yield_v2_manual_review.json`;
- run manifest digest: `7308738ea8533f9379d6f93cba6b31e0b41ad796bdebd599ee40262ba6d0ff4b`.
