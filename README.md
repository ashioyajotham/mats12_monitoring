# When Does a Chain-of-Thought Monitor Cry Wolf?

An empirical study of whether behavioural, language-model, and activation-based monitors can distinguish **silent hint use** from ordinary reasoning failures—especially when genuine unfaithfulness is rare.

This repository is the MATS 12 application project of Victor Ashioya. It extends [cot-faithfulness-mech-interp](https://github.com/ashioyajotham/cot-faithfulness-mech-interp), where strong GPT-2 probe results degraded on Qwen and Gemma and causal interventions had weak behavioural effects. The new project asks the operational question that those results leave open: **can a monitor detect causally hint-influenced reasoning without overwhelming an auditor with false positives?**

## Research contract

### Primary question

Can counterfactual resampling distinguish hint-induced rationalization from ordinary reasoning errors better than transcript-only and context-aware baselines?

### Research hierarchy

This is a monitoring project, not a benchmark of whether GPT-OSS can solve mathematics. The
task-construction question is subordinate and operational: **can we construct a controlled
reasoning environment with enough ordinary failures to test whether monitors distinguish them
from causally induced unfaithfulness?**

The next enabling milestone is a fresh clean-only mixed-outcome gate. Truncations, parser failures,
and ambiguous prompts do not count as ordinary errors. Passing that gate permits a causal-yield
experiment; it does not by itself permit monitor training.

### Main hypothesis

Monitors will discriminate silent hint use on balanced data, but difficult clean reasoning will create enough false positives that precision collapses at a 1% deployment prevalence.

### Unit of analysis

A base multiple-choice question, all prompt conditions derived from it, and repeated model rollouts. Splits are grouped by `question_id`; variants or resamples of the same question never cross splits.

### Causal positive label

An individual hinted rollout is labelled `silent_hint_use` only when:

1. it selects the hinted option;
2. the hint produces a sufficiently large answer-probability shift relative to clean resamples;
3. the visible reasoning does not acknowledge the hint; and
4. the effect has enough sampling support to pass the configured uncertainty gate.

Raw generations are immutable. Derived labels can always be rebuilt from them.

## Original pilot scope

- One current open reasoning model, configured in `configs/pilot.yaml`.
- One multiple-choice task family: the pinned ARC-Challenge validation split.
- 20 deterministically selected base questions spanning six source collections.
- Three prompt conditions: clean, incorrect-answer hint, and irrelevant metadata.
- Five samples per condition (approximately 300 rollouts if clean counterfactual resamples are included).
- Full manual review of candidate positives and high-confidence monitor errors.
- Behavioural baselines first; activation probes only after the pilot gates pass.

The default configuration uses a `mock` backend so the full data and evaluation pipeline can be tested on CPU. A real generation adapter must be explicitly configured before collecting research results.

## Pilot gates

| Gate | Continue when… | Default threshold |
|---|---|---:|
| Phenomenon | Incorrect hints measurably alter answers | ≥ 15% candidate rate |
| Validity | Manual review confirms candidate silent use | ≥ 70% agreement |
| Diversity | Positives span more than one task/template group | ≥ 2 groups |
| Feasibility | Generation and resampling fit the time/compute budget | ≤ 4 hours |

Failure is informative. If the phenomenon gate fails, increase task difficulty or change the model. If label validity fails, pivot to naturally occurring rationalization rather than relaxing the definition.

## Repository map

```text
mats12_monitoring/
├── configs/pilot.yaml             # frozen pilot choices and gates
├── data/
│   ├── raw/                       # source questions; never model outputs
│   ├── generated/                 # immutable raw rollouts and manifests
│   └── reviewed/                  # human review and derived labels
├── src/
│   ├── tasks.py                   # question schema, loading and grouped splits
│   ├── datasets/                  # licensed source normalization and freeze selection
│   ├── hints.py                   # controlled prompt interventions
│   ├── generate_rollouts.py       # backend-neutral rollout generation
│   ├── causal_labels.py           # causal label derivation
│   ├── resampling.py              # answer-shift and branch evidence
│   ├── monitors/                  # surface, LLM, activation and hybrid monitors
│   ├── metrics.py                 # AUROC/AUPRC/FPR/low-base-rate PPV
│   └── audit.py                   # manifests, hashes and claim checks
├── experiments/                   # numbered, reproducible experiment entrypoints
├── docs/                          # preregistration, method and reference notes
├── results/
│   ├── claim_ledger.md            # evidence required for every public claim
│   └── decisions.md               # timestamped research decisions and pivots
└── tests/                         # CPU-only contract tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest

# Validate config and generate deterministic mock pilot data
python experiments/01_pilot.py --config configs/pilot.yaml

# Validate the nine-request GLM smoke plan without making API calls
python experiments/02_generate_dataset.py \
  --config configs/glm_smoke.yaml \
  --limit 3 \
  --samples-per-condition 1 \
  --dry-run

# Validate the seeded Tinker/Qwen smoke plan without spending credits
python experiments/02_generate_dataset.py \
  --config configs/tinker_smoke.yaml \
  --limit 1 \
  --samples-per-condition 1 \
  --dry-run

# Validate the frozen 30-request intervention-calibration plan
python experiments/02_generate_dataset.py \
  --config configs/tinker_calibration.yaml \
  --dry-run

# Validate the intermediate-strength annotation calibration
python experiments/02_generate_dataset.py \
  --config configs/tinker_intermediate_calibration.yaml \
  --dry-run

# Validate the clean-only natural-error discovery plan
python experiments/02_generate_dataset.py \
  --config configs/tinker_natural_failures.yaml \
  --dry-run

# Reproduce and verify the original 120-question procedural candidate freeze
python experiments/00_generate_procedural_math.py

# Validate the 120-request Tinker screening plan without spending credits
python experiments/02_generate_dataset.py \
  --config configs/tinker_procedural_screen.yaml \
  --dry-run

# After screening, freeze eligible cells without using individual item outcomes
python experiments/02_select_procedural_pilot.py \
  --run data/generated/tinker_procedural_screening_<TIMESTAMP>

# After a successful freeze, validate the 120-request fresh discovery plan
python experiments/02_generate_dataset.py \
  --config configs/tinker_procedural_discovery.yaml \
  --dry-run

# Validate the diagnostic-only 24-request low-reasoning plan
python experiments/02_generate_dataset.py \
  --config configs/tinker_procedural_low_reasoning_diagnostic.yaml \
  --dry-run

# Reproduce the prospective 80-question procedural-v2 freeze
python experiments/00_generate_procedural_math_v2.py

# Validate the v2 Tinker screen without spending credits
python experiments/02_generate_dataset.py \
  --config configs/tinker_procedural_v2_screen.yaml \
  --dry-run

# After the live screen, apply aggregate cell gates and freeze 40 questions
python experiments/02_select_procedural_v2.py \
  --run data/generated/tinker_procedural_v2_screening_<TIMESTAMP>

# Only after selection passes, validate the fresh 120-request mixed-outcome run
python experiments/02_generate_dataset.py \
  --config configs/tinker_procedural_v2_discovery.yaml \
  --dry-run

# Reproduce and dry-run the prospective 20-question v2.1 subset replacement
python experiments/00_generate_procedural_math_v21.py
python experiments/02_generate_dataset.py \
  --config configs/tinker_procedural_v21_subset_screen.yaml \
  --dry-run

# After its live screen, combine it with the three eligible v2 families
python experiments/02_select_procedural_v21.py \
  --replacement-run data/generated/tinker_procedural_v21_subset_screening_<TIMESTAMP>

# Only after the combined selector passes, validate the fresh 120-request cohort
python experiments/02_generate_dataset.py \
  --config configs/tinker_procedural_v21_discovery.yaml \
  --dry-run

# Reproduce and validate the diagnostic-only 12-question causal-yield freeze
python experiments/00_prepare_causal_yield_pilot.py
python experiments/02_generate_dataset.py \
  --config configs/tinker_procedural_causal_yield.yaml \
  --dry-run

# Evaluate monitors from a labelled JSONL file
python experiments/06_low_base_rate_eval.py \
  --input data/reviewed/monitor_scores.jsonl \
  --output results/low_base_rate_metrics.json
```

To reproduce the committed question freeze, download the exact pinned source file and run the
preparation entrypoint. The command refuses a source whose SHA-256 differs from the configured
hash and never overwrites an existing freeze.

```bash
curl -fL \
  https://huggingface.co/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/validation-00000-of-00001.parquet \
  -o /tmp/arc_challenge_validation.parquet

python experiments/00_prepare_pilot_dataset.py \
  --config configs/pilot.yaml \
  --source /tmp/arc_challenge_validation.parquet
```

## Experiment order

0. `00_prepare_pilot_dataset.py` and `00_generate_procedural_math.py`: reproduce immutable,
   hashed question freezes and solver certificates.
1. `01_pilot.py`: validate config, prompts, schemas and causal-label plumbing with mock data.
2. `02_select_calibration_questions.py`: freeze a difficulty-proxy subset from clean telemetry.
3. `02_generate_dataset.py`: collect immutable Z.AI or Tinker rollouts with manifests, resume,
   provider provenance, and smoke gates.
4. `03_run_baselines.py`: correctness, surface and judge baselines.
5. `04_resampling_monitor.py`: calculate counterfactual answer-shift evidence.
6. `05_activation_probe.py`: optional residual-stream probe after behavioural gates.
7. `06_low_base_rate_eval.py`: report balanced metrics and deployment PPV.

Every script writes to a new run-specific file. Never overwrite raw generations.

## Headline metrics

Accuracy is not a headline metric. Report:

- AUROC and AUPRC with bootstrap confidence intervals;
- false-positive rate at fixed recall;
- precision at assumed prevalences of 50%, 10%, 5%, and 1%;
- performance on clean-incorrect and difficult-clean negatives;
- generalization to held-out questions and hint styles;
- manually inspected false positives and false negatives.

For prevalence `π`, sensitivity `TPR`, and false-positive rate `FPR`:

$$
\operatorname{PPV}(\pi)=\frac{\operatorname{TPR}\,\pi}{\operatorname{TPR}\,\pi + \operatorname{FPR}\,(1-\pi)}.
$$

## What would falsify the project’s preferred story?

- A correctness-only or CoT-length baseline matches the main monitor.
- Performance disappears when split by base question.
- The label is predictable from hint-template identity alone.
- “Silent use” candidates do not survive repeated sampling or manual review.
- A method’s impressive AUROC corresponds to unusable precision at 1% prevalence.
- Internals add no held-out value beyond behavioural evidence.

These are results to report, not obstacles to hide.

## Research lineage

The experimental design draws from:

- [Turpin et al. (2023), *Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388): paired biasing interventions and post-hoc rationalization.
- [Chen et al. (2025), *Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410): controlled hints, influence, and verbalized acknowledgment.
- [Arcuschin et al. (2025), *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful*](https://arxiv.org/abs/2503.08679): naturally occurring shortcuts and realistic negative controls.
- [Thought Branches (2025), *Interpreting LLM Reasoning Requires Resampling*](https://arxiv.org/abs/2510.27484): reasoning as a distribution rather than a single trace.
- [Hewitt & Liang (2019), *Designing and Interpreting Probes with Control Tasks*](https://aclanthology.org/D19-1275/): probe selectivity and control tasks.
- [Victor Ashioya, *Mechanistic Analysis of Chain-of-Thought*](https://github.com/ashioyajotham/cot-faithfulness-mech-interp): activation probing, causal checks, bootstrap analysis, and the motivating negative scaling result.

See [`docs/REFERENCES.md`](docs/REFERENCES.md) for what is borrowed from each source and what this project adds.

## Status

**Discovery gate stopped.** The schemas, controlled prompts, deterministic mock
pipeline, causal-label derivation, baseline interfaces, rare-event metrics, audit manifests, and
CPU-only tests are implemented. The licensed ARC pilot input freeze is committed with provenance.
The Tinker/Qwen integration has passed live infrastructure checks. A trusted-answer-key treatment
caused 90% wrong-answer uptake with explicit acknowledgment, while an uncertain automated
annotation caused longer deliberation but zero uptake. A follow-up authority-by-directive
factorial confirmed their interaction but produced no valid silent-use candidates and eight
16K-token ceiling failures. Human review, grouped-bootstrap reporting, and empirical monitor
evaluation remain unrun. A non-authority peer scratch-note pivot then produced 0/10 uptake with
no invalid outputs.

Clean-only discovery across all 20 frozen ARC questions produced 60/60 correct responses. A
subsequent level-4/5 MATH pivot exposed deterministic full-thinking Qwen truncation, leading to an
auditable partial-response collector and a GPT-OSS-20B medium-reasoning cohort on Tinker. That
cohort stored 60/60 responses with 59 clean stops, one truncation, and 59/59 scorable answers
correct. The natural-error yield gate therefore failed for both selected task slices. Per the
amended stopping rule, intervention spending remains paused.

The first procedural screen also failed: it produced one defensible completed error, 26
truncations, two parser failures, and four apparent recurrence errors attributable to ambiguous
indexing. No family-by-tier cell qualified and no 40-question bank was frozen. The result is
reported in [`docs/PROCEDURAL_SCREENING.md`](docs/PROCEDURAL_SCREENING.md).

The low-reasoning diagnostic formally failed with 19/24 automatically scorable responses, four
parser-invalid outputs, and one truncation; its selected questions remain barred from monitor
data. The result is recorded in
[`docs/LOW_REASONING_DIAGNOSTIC.md`](docs/LOW_REASONING_DIAGNOSTIC.md). The next step is now frozen
prospectively. The 80-question v2 screen then completed reliably and produced a strong overall
mixture—33 correct and 44 incorrect scorable answers—but formally failed because subset counting
had no eligible cell. Three other families qualified. The frozen adaptive response is a fresh
20-question subset replacement under
[`docs/PREREGISTRATION_PROCEDURAL_V21_AMENDMENT.md`](docs/PREREGISTRATION_PROCEDURAL_V21_AMENDMENT.md).
Both replacement tiers passed, and the combined fresh validation then passed all preregistered
checks with 120/120 scorable responses, 64 correct, and 56 incorrect. The result is bound in
[`docs/PROCEDURAL_V21_RESULT.md`](docs/PROCEDURAL_V21_RESULT.md). This authorizes the bounded,
diagnostic-only matched partial-solution pilot frozen in
[`docs/PREREGISTRATION_CAUSAL_YIELD_V1.md`](docs/PREREGISTRATION_CAUSAL_YIELD_V1.md), not monitor
training.

These calibration results are bounded pilot evidence, not a monitor-performance claim. Raw model
outputs remain excluded from version control; their hashes and protocol-level summaries are
recorded in [`docs/CALIBRATION_PROTOCOL.md`](docs/CALIBRATION_PROTOCOL.md) and
[`docs/MATH_DISCOVERY.md`](docs/MATH_DISCOVERY.md).

## License

Code and original documentation are MIT-licensed. The committed ARC-derived question records are
licensed separately under CC BY-SA 4.0; see [`data/raw/README.md`](data/raw/README.md). Future
dataset licenses must be recorded in the run manifest before source data is committed or
redistributed.
