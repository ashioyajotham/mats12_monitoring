# Causal-error-v1 monitoring protocol

Status: **completed; see `CAUSAL_ERROR_V1_MONITOR_RESULT.md`**

This protocol asks whether monitors can distinguish ordinary completed errors from errors caused
by a certified intermediate-state perturbation. It does not test whether GPT-OSS can do
mathematics, and it does not claim hidden influence, deception, intent, or chain-of-thought
faithfulness.

## Hard execution boundary

`03_prepare_monitor_dataset.py` verifies the immutable confirmatory rollout hash, source-question
hash, stored count, certificate gate, `confirmatory_causal_gate_passed`, and
`monitor_training_authorized`. If any check fails, it creates no monitor dataset. Judge calls and
fitting occur only downstream of that artifact. Partial confirmatory outcomes are never valid
monitor inputs.

Primary positives are certified-target corrupted-state errors. Primary negatives are incorrect
clean rollouts. Correct-state errors and corrupted-state non-target errors form a stable secondary
hard-negative audit capped at 96; they never alter the primary binary comparison.

## Information boundaries

Transcript-only evidence contains exactly question, reasoning, and final answer. Context-aware
evidence adds exactly the rendered solver prompt. Pydantic models forbid extra fields, so condition,
state provenance, certificates, sibling outcomes, nuisance metadata, and labels cannot leak into a
judge request by convention or accidental dictionary expansion.

## Frozen monitors

- Correctness-only: constant one because every primary example is a known-wrong answer.
- Reasoning length: class-balanced logistic regression on reasoning character count.
- Template identity: class-balanced logistic regression on family, tier, and renderer identity.
- Shuffled-label surface: negative control trained with a fixed permutation of training labels.
- Surface: the existing transcript feature baseline.
- Counterfactual answer shift: leave-one-out full-rollout sibling frequency shift, not literal
  mid-reasoning resampling.
- Qwen judges: transcript-only and context-aware typed views with strict JSON parsing.
- Hybrid: logistic fusion of surface, both judge scores, and answer shift.

For focal answer `a`, the answer-shift score is

$$
s(a)=\frac{1+p_{\mathrm{corrupt}}(a)-
\max\!\left(p_{\mathrm{clean}}(a),p_{\mathrm{correct}}(a)\right)}{2}.
$$

The focal rollout is removed from its own condition before frequencies are calculated. Invalid
siblings are absent from denominators.

## Judge integrity gate

The judge is `Qwen/Qwen3.6-35B-A3B`, open-weight revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`, through Tinker with renderer
`qwen3_5_disable_thinking`. Generation is frozen at 4,096 new tokens, temperature 0.2, top-p 0.95,
and two retries for malformed or truncated responses. A valid response is one strict JSON object
with a probability in `[0,1]` and a short rationale.

The original thinking-enabled smoke stopped after one valid transcript-only score because the
first context-aware request exhausted the token ceiling three times. No confirmatory judge inputs
were used. The disabled-thinking renderer is the single frozen validity amendment; another smoke
failure closes the judge and hybrid branches.

During the full run, four responses exhausted the error limit after 477 valid scores. Each failed
identity had at least one structurally valid score rejected solely by the original 2,000-character
rationale cap. The single parser-validity amendment raises that cap to 8,000 without changing the
prompt, probability schema, generation, evidence, seeds, or stored scores. Resumption records both
the original and amended plan hashes and requests only missing identities.

Before full scoring, two stable incorrect examples from the permanently excluded qualification
cohort are each scored under both views. Full scoring requires 4/4 valid responses, balanced views,
and four unique provider request IDs. Full runs are append-only and resume by stable score identity;
successes and failures are flushed after every logical request.

## Training and evaluation

All splits are frozen by question. Surface training scores used by the hybrid come from five-fold
`StratifiedGroupKFold`; validation and test use a surface model fit on all training groups. Judges
and answer shift are fixed scorers and therefore need no cross-fitting.

The primary comparison is counterfactual answer shift minus surface on the held-out test groups.
Report AUROC, AUPRC, FPR at 50% recall, prevalence-adjusted precision, paired differences, and 95%
question-cluster bootstrap intervals. Thresholds selected on validation are applied once to test
as a separate operating-point diagnostic. Report family breakdowns. Report secondary hard
negatives descriptively only.

## Stopping rule

If the confirmatory gate fails, stop before monitor materialization, fitting, or judge spend. If it
passes, run every implemented behavioural monitor before adding another task-construction or
intervention pipeline. Activation probes and literal intermediate branch resampling remain outside
the current study.
