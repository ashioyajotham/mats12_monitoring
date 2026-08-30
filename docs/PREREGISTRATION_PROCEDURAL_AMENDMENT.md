# Procedural mathematics task-construction amendment

Status: **freeze before Tinker screening**  
Date: 2026-08-30

This exploratory amendment follows the failed natural-error gates on ARC-Challenge and level-4/5
MATH. It governs task construction only. It does not authorize hint interventions, monitor
training, activation collection, or monitor-performance claims.

## Research question

Can original, solver-verified mathematics tasks place GPT-OSS-20B in a reproducible mixed-outcome
regime that supplies naturally incorrect clean reasoning as a valid monitor control?

The target regime is neither maximal difficulty nor model ranking. It is a bank with enough clean
correct and clean incorrect reasoning to distinguish ordinary failure from later causally
hint-influenced behavior.

## Candidate construction

Version `procedural-math-v1` contains 120 candidates: ten instances in every cell of four families
and three structural difficulty tiers.

| Family | Structural difficulty control | Exact verifier |
|---|---|---|
| CRT | three, four, or five congruences | incremental Chinese-remainder solver |
| Linear system | three, four, or five unknowns | rational Gauss-Jordan elimination |
| DAG counting | eight, eleven, or fourteen vertices | DP and recursive path counting |
| Recurrence | target index eight, thirteen, or twenty | iteration and matrix exponentiation |

Each instance is constructed from a latent solution and recomputed through a separate exact code
path. Answers are integers or rationals. Three deterministic renderers per family add surface
variation without an LLM paraphraser. Difficulty changes structure and solution depth rather than
only integer magnitude.

The candidate file contains prompts and normalized answers. A separate certificate file records
the generator version, instance seed, parameters, oracle, prompt digest, and certificate digest.
Certificates and structural metadata never enter model prompts or future monitor inputs.

## Stage 1: adaptive screening

- Model: `openai/gpt-oss-20b` through Tinker, revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`.
- Renderer: `gpt_oss_medium_reasoning`; temperature 1.0, top-p 0.95, 4,096 output tokens.
- Condition: clean only; one rollout per candidate; logical base seed `20260910`.
- A family-by-tier cell qualifies with at least 9/10 automatically scorable responses, zero
  collection request errors, and 30–70% accuracy among scorable responses.
- Freeze ten questions per family from qualifying cells with selection seed `20260901`, balancing
  tier-renderer strata. Selection may use cell eligibility but must not use an individual item's
  correctness.

All four families must have a qualifying cell. If any family fails, report the calibration null
and stop. Do not widen the accuracy band or select individual failures. Any parameter change
creates a separately versioned candidate bank and amendment.

Screening rollouts are calibration evidence only. They cannot enter intervention data, monitor
training, validation, or testing.

## Stage 2: fresh clean discovery

If screening passes, freeze exactly 40 questions and collect three new clean rollouts per question
with logical base seed `20261010`. The task-readiness gate passes only when all checks hold:

- all 120 planned responses are stored with unique provider request IDs and zero request errors;
- at least 90% are cleanly terminated and automatically scorable;
- no more than 10% are length-truncated;
- at least 24 answers are correct and at least 24 are incorrect;
- incorrect answers span at least six questions and three generator families;
- correct answers span at least three generator families; and
- reasoning is present for every scorable response.

Report accuracy with a 95% question-clustered bootstrap interval and family-tier breakdowns.
Unsupported answer forms remain review outcomes, not forced errors.

## Stopping rule and downstream boundary

Failure of either stage is a task-construction result and blocks provider spending on
interventions. Passing the clean-discovery gate freezes a usable task bank but still does not
authorize monitor training. The next amendment must preregister matched correct and subtly flawed
partial solutions derived from certificates and require unacknowledged causal answer shifts before
creating positive labels.

No screening decision may depend on monitor scores. No generator certificate, gold answer,
screening outcome, sibling rollout, or derived label may appear in a transcript-only monitor view.

## Recorded outcome

The screen failed on 2026-08-30. It stored all 120 responses with zero request errors, but only 92
were scorable; 26 truncated and two were parse-invalid. No cell passed. Four of five apparent
errors came from an ambiguous recurrence renderer and are not valid ordinary failures. See
`docs/PROCEDURAL_SCREENING.md`. Per the stopping rule, no 40-question bank was frozen and no
intervention spending was authorized.
