# Record schemas

Pydantic models in `src/` are authoritative. This document explains their relationships.

| Record | Stable key | Created from | Mutable? |
|---|---|---|---:|
| `Question` | `question_id` | licensed source data | no after run starts |
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
