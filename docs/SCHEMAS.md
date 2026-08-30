# Record schemas

Pydantic models in `src/` are authoritative. This document explains their relationships.

| Record | Stable key | Created from | Mutable? |
|---|---|---|---:|
| `Question` | `question_id` | licensed source data | no after run starts |
| `MathProblem` | `question_id` | licensed or procedural free-response task | no after run starts |
| Procedural certificate | `question_id` + certificate digest | generator parameters and exact oracle | no |
| `PromptVariant` | question + condition + template | question and hint constructor | no |
| `Rollout` | `rollout_id` | prompt, model, seed, generation config | append-only |
| `AnswerShift` | question + hinted option | sibling clean/hinted rollouts | reproducible |
| `LabelledRollout` | `rollout_id` + label method | rollout and answer shift | reproducible |
| Human review | rollout + reviewer + form version | blinded and causal review passes | append-only |
| Monitor score | rollout + monitor + version | frozen model and allowed view | reproducible |

## Identity rules

- `question_id` identifies the underlying question, not a rendered prompt.
- `rollout_id` hashes question, condition, seed, and model.
- A generation rerun with different decoding parameters must use a distinct run manifest; decoding parameters are also stored on each rollout.
- Derived artifacts always retain `question_id` so group-bootstrap and leakage checks remain possible.

Hosted reasoning-model rollouts retain the reasoning content, final response, combined transcript,
provider request ID, returned provider model, token usage, finish reason, and request latency. A
logical `seed` remains part of stable rollout identity even when the provider does not support
seeded inference; the run manifest declares that limitation explicitly.

The committed pilot question freeze has a sibling manifest containing the source repository and
revision, source-file SHA-256, license, normalization and selection rule, selected collection
counts, code revision, and output SHA-256. Dataset records retain their upstream identifier and
original choice labels in `metadata`.

Procedural `MathProblem` records retain generator version, family, difficulty tier, renderer,
instance seed, lineage ID, structural parameters, oracle kind, and certificate digest in
`metadata`. The matching certificate is an audit artifact, not a prompt field or permitted monitor
input. Screening selection is performed at the family-by-tier cell level; screening correctness
is not copied into the frozen question record.

Procedural v2 uses the same separation with a new generator namespace. Its candidate screen may
make a cell eligible from aggregate accuracy, but the deterministic within-cell hash order cannot
see individual outcomes. A v2 clean mixed-outcome report may authorize causal-yield design only;
it is not a label file and cannot authorize monitor training.

The v2.1 amendment permits aggregate task-level adaptation after the failed v2 screen. The
combined freeze may reuse previously eligible cells, but question ordering remains blind to item
correctness. A subsequent three-sample-per-question collection supplies fresh validation data;
screening rollouts never become monitor examples.

The causal-yield freeze adds matched intervention metadata to copied `MathProblem` records:
correct and corrupted note text, a strictly incorrect target answer, source-certificate binding,
and an intervention-certificate digest. `Rollout.hinted_option` stores the planted numeric target
for the corrupted condition. This legacy field name denotes a target answer for free-response
math; it is not a multiple-choice option. Automated unacknowledged candidates remain review
candidates rather than causal labels.

## Separation of evidence and monitor inputs

Fields such as `hint_effect`, sibling answer counts, `causal_label`, and manual review are ground-truth evidence. They must not enter transcript-only, context-aware, surface, or activation monitors. Counterfactual-resampling monitors may use declared sibling statistics because those statistics are their method.

## Required run manifest fields

- run purpose (`pipeline_test`, `pilot`, or `confirmatory`);
- config hash and path;
- model identifier and revision;
- backend/library version;
- generation parameters;
- source dataset name, revision, license, and selection rule;
- code revision;
- start/end time and runtime environment;
- counts of requested, completed, invalid, and excluded generations;
- content hashes of produced raw files.
