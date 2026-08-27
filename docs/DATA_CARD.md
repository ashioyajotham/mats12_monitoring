# Dataset card (planned pilot)

## Purpose

Evaluate monitors for silent, causally supported hint use in multiple-choice reasoning. The dataset is not a general benchmark for deception or model honesty.

## Composition

Each base question is paired with clean, incorrect-answer-hint, and irrelevant-metadata prompts and multiple stochastic rollouts. Derived records contain question-level answer shifts, acknowledgment evidence, and human review.

## Sources

No source dataset is frozen yet. Before data collection, record the source name, version/commit, license, selection rule, and any transformations in the run manifest. Do not commit restricted question text.

## Splits

All prompt variants, seeds, and resamples from one `question_id` remain in the same split. A held-out hint template may be used as an explicit out-of-distribution test.

## Labels

Primary positive: `silent_hint_use`. It requires a sampled answer-shift estimate, lack of acknowledgment, uncertainty support, and manual review according to the preregistration. Ambiguous cases remain ambiguous rather than being forced into binary labels.

## Known limitations

- Synthetic hints may not represent naturally occurring unfaithfulness.
- Small resample counts create wide uncertainty intervals.
- Acknowledgment is partly language- and rubric-dependent.
- Multiple-choice answers simplify causal measurement but narrow external validity.
- Results from one model/task family should not be generalized to all reasoning models.

## Intended use

Research on monitor validity, confounding, and base-rate sensitivity. Not intended for model ranking, high-stakes deployment decisions, or claims about intentional deception.
