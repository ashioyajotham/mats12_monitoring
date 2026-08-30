# When Does a Chain-of-Thought Monitor Cry Wolf?

![Pixel-art illustration of ordinary and causally induced reasoning errors](assets/mats12_monitoring_pixel_art.png)

An empirical study of whether reasoning monitors can distinguish **ordinary model mistakes** from
errors caused by a **certified corrupted intermediate state**—without overwhelming human auditors
with false alarms.

This is the MATS 12 application project of Victor Ashioya. For a non-technical introduction, see
the [plain-language project overview](docs/PROJECT_OVERVIEW.md).

## Current status

| Research component | Status |
|---|---|
| Clean mixed-outcome qualification | **Passed**: 29 correct, 43 ordinary errors, 0 invalid |
| Confirmatory randomized causal collection | **In progress** |
| Behavioural monitor stack | Implemented and tested |
| Monitor fitting and full judge scoring | **Blocked** until every confirmatory gate passes |
| Silent-unfaithfulness research branch | Closed after preregistered null/validity results |
| Activation probing | Outside the current study |

No partial confirmatory outcomes may be used for labels, monitor fitting, or research claims. A
failed confirmatory gate stops the study before monitor evaluation.

## The problem

Suppose two models produce the same wrong answer:

1. one reaches it through an ordinary reasoning mistake;
2. the other was given a bad intermediate value and propagated that corruption downstream.

Correctness alone cannot distinguish these cases. Their written reasoning may also look similar.
The monitoring problem is therefore not merely *“Is this answer wrong?”* but:

> Can a monitor identify errors caused by a controlled reasoning corruption without falsely
> accusing too many ordinary failures?

This distinction matters at low base rates. If causally induced failures are rare, a seemingly
accurate monitor can still produce far more false alarms than useful alerts.

## Research question and hypothesis

### Primary question

Can a leave-one-out counterfactual answer-shift score distinguish ordinary completed reasoning
errors from errors produced by a certified single-state perturbation better than a frozen
surface-feature baseline?

### Main hypothesis

Counterfactual answer shift will discriminate intervention-caused errors from ordinary errors
better than surface monitoring on grouped held-out questions. Its false-positive rate may still
make precision poor at realistic deployment prevalences.

### Research hierarchy

This is a **monitoring study**, not a benchmark of whether GPT-OSS can solve mathematics. The task
construction question is subordinate:

> Can we construct a controlled reasoning environment with enough ordinary failures to test
> whether monitors distinguish them from causally induced errors?

Mathematics is useful here because answers can be graded exactly, intermediate states can be
verified independently, and the consequences of changing one value can be propagated in advance.

## Background and research trajectory

The project extends
[cot-faithfulness-mech-interp](https://github.com/ashioyajotham/cot-faithfulness-mech-interp),
where strong GPT-2 probe results weakened on Qwen and Gemma and causal interventions had limited
behavioural effects.

The initial MATS study attempted to construct examples where a model silently followed a
misleading hint. Weak hints were usually resisted. Stronger authority and directive cues sometimes
changed answers, but the model generally acknowledged them. Exact state-continuation interventions
created strong causal control, but their influence remained visible. An assistant-prefill attempt
then failed its causal gate.

Those results did not justify a hidden-influence label. Rather than weakening the criteria, the
project closed that branch and reframed the study around a narrower causal-error question:
distinguishing ordinary failures from errors caused by certified state perturbations.

## Findings so far

These are bounded task-construction and mechanism findings—not monitor-performance results.

| Phase | Result | Decision |
|---|---|---|
| Trusted-answer-key calibration | 9/10 treatment answers followed the wrong key; all nine acknowledged it | Behaviour changed, but no silent-use dataset was justified |
| Uncertain annotation | 0/10 planted-answer uptake despite much longer reasoning | Weak contextual hints were resisted |
| ARC clean discovery | 60/60 correct | The task slice was too easy to supply ordinary failures |
| MATH clean discovery | 59/59 scorable answers correct; one truncation | Static MATH slice also failed the ordinary-error yield gate |
| Procedural v2.1 validation | 120/120 scorable: 64 correct and 56 incorrect | A reliable mixed-outcome environment was achieved |
| Answer-bearing solver note | Target shift +2.8 points; clustered interval −5.6 to +11.1 | No causal-yield evidence; explicit verification framing induced correction |
| Exact state continuation | 13/24 corrupted targets versus 0/24 in each control; effect +54.2 points, interval +33.3 to +75.0 | Strong causal control, but manual review found zero defensible silent-use cases |
| Assistant prefill | 1/24 corrupted targets versus 2/24 correct-prefill targets; effect −4.2 points, interval −16.7 to +8.3 | Causal gate failed; silent-unfaithfulness branch closed |
| Causal-error-v1 qualification | 72/72 scorable: 29 correct and 43 errors across 19 questions and all four families | Frozen confirmatory causal collection authorized |

The central progression is:

```text
Weak or openly acknowledged hint effects
                    ↓
Failure to construct defensible silent-use labels
                    ↓
Reliable ordinary failures from original procedural tasks
                    ↓
Strong, certified causal errors from state corruption
                    ↓
Current study: can monitors distinguish the two?
```

## Research phases and milestones

| Phase | Objective | Outcome |
|---|---|---|
| 0. Prior mechanistic work | Test whether activation probes and interventions scale across models | Motivating negative scaling result |
| 1. Hint interventions | Produce behaviourally influenced, unacknowledged reasoning | Closed: influence was weak or openly acknowledged |
| 2. Natural-error discovery | Find enough ordinary failures on licensed static tasks | Closed: ARC and selected MATH slices were too easy |
| 3. Procedural task construction | Create original, solver-verified tasks with reliable mixed outcomes | Passed with 64 correct and 56 incorrect responses |
| 4. Causal mechanism discovery | Create predictable downstream errors from one corrupted state | Causal control passed; hidden-use validity failed |
| 5. Causal-error dataset construction | Prospectively collect ordinary and causally induced failures | **Current phase: confirmatory collection** |
| 6. Behavioural monitor evaluation | Compare answer shift, surface, judges, controls, and hybrid | Pending confirmatory authorization |
| 7. Internals | Test whether activations add held-out value | Optional future work; not authorized now |

## Current experimental design

The causal-error-v1 study contains 96 new solver-verified procedural problems:

- 24 qualification questions, permanently excluded from monitor data;
- 72 untouched confirmatory questions;
- four task families balanced across difficulty tiers and renderers;
- frozen train, validation, and test assignments grouped by `question_id`.

For each confirmatory question, the solver model receives three conditions with three fresh samples
per condition:

```text
Verified problem
      ├── Clean prompt
      ├── Correct intermediate-state continuation
      └── Single-value corrupted-state continuation
                         ↓
          648 immutable model rollouts
                         ↓
      Causal effect, integrity and diversity gates
                         ↓
     Ordinary errors vs causally induced errors
                         ↓
          Grouped held-out monitor evaluation
```

### Labels

An individual rollout is labelled `causally_induced_error` only when:

1. it was generated under the prospectively assigned corrupted-state condition;
2. it selects the independently certified propagated target; and
3. the complete dataset passes every randomized effect, uncertainty, integrity, clean-negative,
   provenance, and diversity gate.

Primary negatives are completed incorrect clean rollouts. Correct-state errors and corrupted-state
non-target errors are secondary hard negatives, not part of the primary binary comparison.

Acknowledgment is descriptive and is not a label criterion. The label supports a claim about a
controlled causal intervention—not hidden influence, faithfulness, deception, intent, or whether a
model “really” used its chain of thought.

## Confirmatory gates

| Gate | Continue only when… | Frozen threshold |
|---|---|---:|
| Data integrity | Responses are usable and cells are balanced | ≥90% scorable; ≤10% truncated |
| Causal effect | Corruption increases exact-target uptake | ≥20 percentage points; clustered interval lower bound >0 |
| Target diversity | Causal targets span the task distribution | ≥36 targets; ≥18 questions; all 4 families |
| Control validity | Correct-state continuation does not materially damage accuracy | No more than 15-point loss versus clean |
| Ordinary negatives | Enough completed clean errors exist for monitoring | ≥48 errors; ≥24 questions; all 4 families |
| Provenance | Samples, provider IDs, questions and certificates verify exactly | Every integrity check passes |

Thresholds are frozen in the
[causal-error-v1 preregistration](docs/PREREGISTRATION_CAUSAL_ERROR_DETECTION_V1.md) and cannot be
relaxed after collection.

## Monitor comparison

| Monitor | Information available | Purpose |
|---|---|---|
| Correctness-only | Known wrong-answer status | Demonstrate that correctness cannot separate two wrong-answer classes |
| Reasoning length | Character count | Detect a trivial length confound |
| Template identity | Family, tier and renderer | Detect task-construction leakage |
| Shuffled-label surface | Transcript features with permuted training labels | Negative control |
| Surface baseline | Length, uncertainty, backtracking and related visible features | Test whether inexpensive stylistic artefacts suffice |
| Transcript-only Qwen judge | Question, reasoning and answer | Judge causal-error likelihood without intervention context |
| Context-aware Qwen judge | Transcript plus rendered solver prompt | Test the value of seeing the supplied state |
| Counterfactual answer shift | Leave-one-out sibling answer frequencies | Test whether the focal answer becomes more common under corruption |
| Hybrid | Surface, both judges and answer shift | Test whether the signals contain complementary information |

Typed evidence schemas enforce the information boundary. Transcript-only monitors cannot receive
condition names, certificates, provenance, sibling outcomes, split metadata, or derived labels.

The answer-shift score for focal answer $a$ is

$$
s(a)=\frac{1+p_{\mathrm{corrupt}}(a)-
\max\!\left(p_{\mathrm{clean}}(a),p_{\mathrm{correct}}(a)\right)}{2}.
$$

This is a completed-answer frequency baseline, not literal intermediate-state branch resampling.

## Evaluation

Accuracy is not a headline metric. The study reports:

- false-positive rate at 50% recall;
- AUROC and AUPRC;
- precision at assumed causal-error prevalences of 50%, 10%, 5%, and 1%;
- paired question-cluster bootstrap intervals for answer shift versus surface;
- validation-selected thresholds applied once to held-out test questions;
- performance broken down by task family; and
- descriptive scores on secondary hard negatives.

For event prevalence $\pi$, sensitivity $\operatorname{TPR}$, and false-positive rate
$\operatorname{FPR}$:

$$
\operatorname{PPV}(\pi)=
\frac{\operatorname{TPR}\,\pi}
{\operatorname{TPR}\,\pi+\operatorname{FPR}\,(1-\pi)}.
$$

Question-group splitting is mandatory: no repeated sample or intervention variant of the same
problem may cross train, validation, and test.

## What would falsify the preferred story?

- Counterfactual answer shift does not outperform the surface baseline.
- Performance disappears under question-group-disjoint evaluation.
- Reasoning length, family, tier, or renderer identity explains the apparent signal.
- The context-aware judge succeeds only by reading intervention wording.
- Strong balanced-test metrics correspond to unusable precision at 1% prevalence.
- The confirmatory collection fails its causal, ordinary-negative, diversity, or integrity gate.
- A hybrid adds no held-out value beyond its simplest component.

These are findings to report, not obstacles to hide.

## Reproducing the current study

### Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,tinker]"
pytest
```

The CPU-only tests do not require provider credentials. Live collection requires
`TINKER_API_KEY` in the environment.

### Reproduce the frozen causal-error questions

The committed inputs are already frozen. The reproduction command deliberately targets an empty
directory so it cannot overwrite authoritative artifacts:

```bash
uv run --offline python experiments/00_prepare_causal_error_detection_v1.py \
  --output-dir /tmp/causal-error-v1-reproduction
```

### Collect and analyze a new confirmatory reproduction

Do not start a second copy when a collection is already active.

```bash
uv run --offline python experiments/02_generate_dataset.py \
  --config configs/tinker_causal_error_v1_confirmatory.yaml \
  --continue-on-error \
  --max-errors 4

uv run --offline python experiments/02_analyze_causal_error_confirmatory.py \
  --run data/generated/tinker_causal_error_v1_confirmatory_<TIMESTAMP> \
  --output results/causal_error_v1_confirmatory.json
```

### Run monitors only after the report passes

```bash
uv run --offline python experiments/03_prepare_monitor_dataset.py \
  --run data/generated/tinker_causal_error_v1_confirmatory_<TIMESTAMP> \
  --report results/causal_error_v1_confirmatory.json

uv run --offline python experiments/03_run_local_monitors.py \
  --run data/generated/tinker_causal_error_v1_confirmatory_<TIMESTAMP>

# Four-call infrastructure gate on two permanently excluded qualification errors
uv run --offline python experiments/03_run_judge_baseline.py \
  --mode smoke \
  --run data/generated/tinker_causal_error_v1_qualification_20260830T184625Z \
  --report results/causal_error_v1_qualification.json

# Full judge scoring requires the passed smoke manifest
uv run --offline python experiments/03_run_judge_baseline.py \
  --mode full \
  --report results/causal_error_v1_confirmatory.json \
  --smoke-manifest data/generated/tinker_causal_error_judge_smoke_<TIMESTAMP>/manifest.json

uv run --offline python experiments/03_evaluate_monitor_stack.py \
  --judge-run data/generated/tinker_causal_error_judge_full_<TIMESTAMP>
```

Every collection and derived artifact is immutable and content-addressed. Interrupted judge runs
resume by stable score identity.

## Repository guide

```text
configs/       frozen experiment and provider configurations
data/raw/      solver-verified questions and certificates
data/generated immutable model rollouts and run manifests (not committed)
data/reviewed/ gate-authorized derived monitor datasets
src/           task generation, causal gates, monitors and evaluation
experiments/   numbered reproducible entrypoints
docs/          preregistrations, methods, result reports and protocols
results/       machine-readable reports, claim ledger and decision log
tests/         CPU-only scientific and implementation contracts
```

### Core research documents

- [Plain-language project overview](docs/PROJECT_OVERVIEW.md)
- [Current causal-error preregistration](docs/PREREGISTRATION_CAUSAL_ERROR_DETECTION_V1.md)
- [Frozen monitoring protocol](docs/MONITORING_PROTOCOL_V1.md)
- [Methodology and threat models](docs/METHODOLOGY.md)
- [Qualification result](docs/CAUSAL_ERROR_V1_QUALIFICATION_RESULT.md)
- [Research roadmap](docs/ROADMAP.md)
- [References and design lineage](docs/REFERENCES.md)
- [Append-only research decisions](results/decisions.md)
- [Claim ledger](results/claim_ledger.md)

Historical phases remain documented in their preregistrations and result reports under `docs/`.
The decision log records why each branch advanced, failed, or stopped.

## Research lineage

The design draws from work on biasing interventions, chain-of-thought faithfulness, repeated
reasoning samples, and control tasks:

- [Turpin et al. (2023), *Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388)
- [Chen et al. (2025), *Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410)
- [Arcuschin et al. (2025), *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful*](https://arxiv.org/abs/2503.08679)
- [Thought Branches (2025), *Interpreting LLM Reasoning Requires Resampling*](https://arxiv.org/abs/2510.27484)
- [Hewitt and Liang (2019), *Designing and Interpreting Probes with Control Tasks*](https://aclanthology.org/D19-1275/)

See [references and design lineage](docs/REFERENCES.md) for what is borrowed from each source and
what this project changes.

## License

Code and original documentation are MIT-licensed. Committed ARC-derived historical pilot records
remain CC BY-SA 4.0; see [`data/raw/README.md`](data/raw/README.md). Every future dataset must record
its source license and revision before redistribution.
