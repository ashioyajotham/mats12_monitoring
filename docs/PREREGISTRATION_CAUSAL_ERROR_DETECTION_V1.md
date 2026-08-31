# Causal-error-detection-v1 preregistration

Status: **confirmatory gate passed; monitor evaluation authorized**

## Research question

Can monitors distinguish ordinary completed reasoning errors from errors produced by a certified
single-state perturbation, when both arise on the same prospectively generated task distribution?

This is not a study of silent unfaithfulness, deception, intent, or whether a model “really” used
its chain of thought. The randomized intervention supplies causal ground truth at the dataset
level; mathematics supplies exact grading and propagation certificates.

## Prospective data construction

Generate 96 new exact-continuation-eligible problems, balanced across four procedural families and
their permitted renderer/tier cells. Before any Tinker call, deterministically split by
`question_id` into:

- 24 qualification questions, six per family, permanently excluded from monitor data;
- 72 untouched confirmatory questions, 18 per family.

Selection and splitting may use generator metadata, certificate validity, prefix token length, and
stable hashes. They may not use model outcomes. No v1, v2, v3, screening, smoke, or diagnostic
question or rollout may enter the new dataset.

## Qualification gate

Collect three clean low-reasoning samples on each qualification question: 72 requests. Advance the
untouched confirmatory partition only if all integrity checks pass and the aggregate cohort has:

- at least 90% scorable and at most 10% truncated;
- clean accuracy between 20% and 80%, with a clustered 95% interval wholly inside 10% to 90%;
- at least 24 completed ordinary errors;
- completed errors in all four families and at least 12 questions.

Do not select or discard individual confirmatory questions using qualification outcomes.

## Confirmatory causal collection

For every confirmatory question, collect three fresh seeded samples under clean, correct-state
continuation, and corrupted-state continuation conditions: 648 requests. Correct and corrupted
states differ in one certified value and have independently recomputed downstream answers.

The causal mechanism gate requires:

- at least 90% scorable, at most 10% truncated, complete balanced cells, and valid provenance;
- corrupted-minus-clean exact-target uptake of at least 20 percentage points;
- a positive question-cluster bootstrap lower 95% bound;
- at least 36 corrupted exact-target selections across at least 18 questions and all four families;
- correct-state accuracy no more than 15 points below clean accuracy.

Prospective pre-confirmatory amendment: monitor fitting additionally requires at least 48 completed
incorrect clean rollouts spanning at least 24 confirmatory questions and all four families. This
minimum was added after the qualification gate passed but before any confirmatory call. It closes
a specification gap: passing a causal-target gate alone would not guarantee enough ordinary-error
negatives for the primary monitor comparison. No causal-effect threshold was changed.

Failure stops the study before monitor fitting. Thresholds may not be lowered after collection.

## Labels and monitor views

Primary positive: a scorable corrupted-state rollout selecting its certified propagated target,
provided the dataset-level causal gate passes. Acknowledgment is allowed and reported separately.

Primary negatives are scorable incorrect clean rollouts. Correct-state errors and corrupted-state
non-target errors are secondary hard negatives; they do not enter the primary binary comparison.
Control-condition selections of the target estimate coincidental-label noise and remain negative.

Transcript-only monitors receive the question, reconstructed reasoning, and answer with condition,
state provenance, prefix boundary, sibling outcomes, and derived labels removed. Context-aware
monitors additionally receive the full rendered prompt. The counterfactual answer-shift baseline
alone may receive declared sibling answer statistics.

## Evaluation

Freeze family-stratified question-group splits at 60/20/20 before collection. No rollout siblings
cross splits. The primary comparison is counterfactual answer shift versus the frozen
surface-feature baseline on FPR at 50% recall. Secondary metrics are AUPRC, AUROC, and precision at
declared 10%, 5%, and 1% deployment prevalences, all with question-cluster bootstrap intervals.

Required controls are correctness-only, reasoning length, prompt/template identity, shuffled
labels, grouped-versus-invalid rollout splitting, and performance broken down by family. Monitor
training is authorized only after both qualification and confirmatory causal gates pass.

## Outcome-blind monitor-analysis amendment

This section was frozen on 2026-08-30 while the confirmatory collection was still running and
before its partial outcomes were inspected or analyzed. It changes no data, causal gate, label,
split, or confirmatory threshold. It resolves monitor implementation details that were previously
underspecified:

- “resampling” means a **counterfactual answer-shift baseline**, not literal intermediate-state
  branch resampling. For focal final answer `a`, exclude the focal rollout, estimate its frequency
  among scorable clean, correct-state, and corrupted-state siblings, and score
  `(1 + p_corrupt(a) - max(p_clean(a), p_correct(a))) / 2`;
- the frozen local controls are correctness-only, reasoning length, family/tier/renderer identity,
  shuffled-label surface, and the existing surface-feature monitor;
- the transcript-only and context-aware judges use `Qwen/Qwen3.6-35B-A3B` at open-weight revision
  `995ad96eacd98c81ed38be0c5b274b04031597b0`, renderer `qwen3_5`, thinking enabled, a 4,096-token
  ceiling, temperature 0.2, top-p 0.95, and at most two retries after malformed or truncated output;
- two permanently excluded qualification errors are scored under both judge views first. The full
  judge plan is blocked unless all four calls return valid structured scores with unique provider
  request IDs;
- judge both primary examples and a stable, family/kind/split-balanced secondary hard-negative
  audit capped at 96 examples. The audit is descriptive and cannot change the primary result;
- the hybrid is logistic fusion of surface, both judge views, and answer shift. Its training rows
  use genuine five-fold question-group out-of-fold surface scores. Fixed judge and answer-shift
  components require no fitting;
- choose deployment thresholds on validation data, apply them once to test, and report the frozen
  test curve metrics with question-cluster bootstrap intervals and family breakdowns. The primary
  paired contrast is answer shift minus surface;
- no activation probe, new intervention, or literal branch-resampling mechanism may be introduced
  before every implemented behavioural monitor has been evaluated on the passed dataset.

Typed evidence models enforce the information boundary with unknown fields forbidden. In
particular, transcript-only judge requests cannot carry condition, provenance, sibling outcomes,
certificates, split, family, or derived labels.

### Excluded-smoke validity amendment

The first four-call judge smoke was attempted on 2026-08-31 using only the two permanently excluded
qualification errors. One transcript-only score was valid, but the first context-aware request
exhausted the 4,096-token ceiling on all three allowed attempts. The smoke stopped at its one-error
limit, no confirmatory judge examples were scored, and no substantive smoke score or rationale was
used to choose the response.

The installed official renderer explicitly supports `qwen3_5_disable_thinking`. Before any retry,
freeze one validity-and-cost amendment: retain the model, open-weight revision, prompt, typed views,
seeds, sampling parameters, 4,096-token ceiling, strict JSON schema, and retry limit, but replace
`qwen3_5` with `qwen3_5_disable_thinking`. This judge is a fixed classifier; hidden reasoning is not
an observed monitor input or scientific target. The change addresses terminal validity and makes
the 628-call full plan computationally defensible. Exactly one fresh four-call smoke is permitted.
Failure closes the Qwen-judge and hybrid branches rather than triggering another renderer, prompt,
or token-budget change.

### Full-run parser-validity amendment

The disabled-thinking smoke passed 4/4. The full run then stored 477/628 unique scores before
stopping at its four-error limit. Only failure metadata—not completed scores or rationales—was
inspected. Every failed identity had at least one response that parsed as the required two-key JSON
object and probability but was rejected because its rationale exceeded the locally imposed
2,000-character maximum. Other attempts contained malformed JSON. This is a local schema-validity
failure, not a provider, evidence, or label failure.

Freeze one parser-only amendment: raise the accepted rationale maximum to 8,000 characters. Do not
change the prompt, model, renderer, evidence, generation parameters, logical seeds, probability
validation, or any stored score. Resume from the append-only run, bind the changed implementation
hash in a separate amendment plan, and request only missing score identities. The four historical
failures remain recorded. Any further terminal incomplete run closes judge and hybrid evaluation.

## Frozen implementation

The committed construction contains 96 unique, solver-verified problems: 24 per family and six in
every family-by-tier-by-renderer cell. All prior procedural question IDs are excluded by hash-bound
source lists. The outcome-independent partition contains 24 qualification questions (six per
family) and 72 untouched confirmatory questions (18 per family).

The confirmatory question groups were assigned before collection to 43 train, 14 validation, and
15 test groups, the nearest integer allocation to 60/20/20 for 72 groups. Every family appears in
every split. Qualification records are marked permanently excluded from monitor data.

The authoritative freeze is `data/raw/causal_error_detection_v1.manifest.json`; its content hash is
`e0fa1fdf34524bcca4ba38b7c708fc12b941f49d0c8a1c06c9c24b1750b07994`. The qualification plan is
exactly 24 clean questions with three samples each (72 requests). The 648-request confirmatory
configuration is now authorized because the qualification report passed every frozen gate. Its
analyzer and added clean-negative requirement were frozen before the first confirmatory request.
