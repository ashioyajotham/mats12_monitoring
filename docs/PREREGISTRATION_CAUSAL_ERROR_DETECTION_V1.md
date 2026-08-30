# Causal-error-detection-v1 preregistration

Status: **qualification passed; frozen confirmatory collection authorized**

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
monitors additionally receive the full rendered prompt. Resampling monitors alone may receive
declared sibling answer statistics.

## Evaluation

Freeze family-stratified question-group splits at 60/20/20 before collection. No rollout siblings
cross splits. The primary comparison is resampling versus the frozen surface-feature baseline on
FPR at 50% recall. Secondary metrics are AUPRC, AUROC, and precision at declared 10%, 5%, and 1%
deployment prevalences, all with question-cluster bootstrap intervals.

Required controls are correctness-only, reasoning length, prompt/template identity, shuffled
labels, grouped-versus-invalid rollout splitting, and performance broken down by family. Monitor
training is authorized only after both qualification and confirmatory causal gates pass.

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
