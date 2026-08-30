# Low-reasoning attribution diagnostic

Status: **freeze before Tinker collection**  
Date: 2026-08-30

## Research hierarchy

The project does not ask whether GPT-OSS can reason about mathematics. Its primary question is:

> Can we construct a controlled reasoning environment with enough ordinary failures to test
> whether monitors can distinguish them from causally induced unfaithfulness?

Mathematics is a controlled, exactly verifiable substrate for that monitoring question. Model
accuracy, task-family comparisons, and reasoning effort are task-construction diagnostics, not
the research endpoint.

The next legitimate milestone is a clean-only mixed-outcome gate: a fresh unmanipulated cohort
must contain both completed correct reasoning and completed ordinary failures. Truncations,
ambiguous prompts, parser failures, and transport failures do not count as ordinary failures.

## Motivation

The 120-request `procedural-math-v1` screen completed without transport failure but produced 26
length truncations, two parser failures, and only five automatically incorrect clean stops. Four
of those five came from a recurrence renderer that did not explicitly identify its initial values
as `a_0` and `a_1`; they are excluded as task ambiguity. The only defensible completed error was
one DAG path-counting response.

The screen therefore failed because medium reasoning converted difficulty into token exhaustion
rather than an adequate ordinary-failure distribution. Raising the token ceiling would not test
the required construct.

## Diagnostic cohort

This diagnostic contains 12 questions that truncated under the medium-reasoning screen and 12
same-family clean-correct controls. Selection seed `20261101` balances available family, tier, and
renderer strata before matching controls by family and nearest tier. Recurrence renderer 1 is
excluded. Selection depends on prior outcomes, so every diagnostic output is permanently barred
from monitor training, validation, testing, and the future mixed-outcome cohort.

- Model: `openai/gpt-oss-20b`, revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`.
- Provider: Tinker only.
- Renderer: `gpt_oss_low_reasoning`.
- Condition: clean only; one new sample per question.
- Sampling: temperature 1.0, top-p 0.95, 4,096 output tokens.
- Logical base seed: `20261110`.

## Attribution gate

The diagnostic passes only if all conditions hold:

- all 24 responses are stored with unique request IDs and zero request errors;
- at least 20 responses are cleanly terminated and automatically scorable;
- no more than two responses truncate;
- at least 10/12 prior-truncation items and 10/12 controls are scorable;
- at least four completed answers are incorrect across at least two families; and
- reasoning is present for every scorable response.

If the gate passes, freeze a new `procedural-math-v2` generator and use low reasoning for a fresh,
outcome-independent screen. If it fails, redesign the task families before further collection.
Neither result authorizes interventions or monitor training. Only a later fresh clean-only
mixed-outcome gate can authorize the causal-yield experiment.
