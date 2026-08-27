# When Does a Chain-of-Thought Monitor Cry Wolf?

An empirical study of whether behavioural, language-model, and activation-based monitors can distinguish **silent hint use** from ordinary reasoning failures—especially when genuine unfaithfulness is rare.

This repository is the MATS 12 application project of Victor Ashioya. It extends [cot-faithfulness-mech-interp](https://github.com/ashioyajotham/cot-faithfulness-mech-interp), where strong GPT-2 probe results degraded on Qwen and Gemma and causal interventions had weak behavioural effects. The new project asks the operational question that those results leave open: **can a monitor detect causally hint-influenced reasoning without overwhelming an auditor with false positives?**

## Research contract

### Primary question

Can counterfactual resampling distinguish hint-induced rationalization from ordinary reasoning errors better than transcript-only and context-aware baselines?

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

## Pilot scope

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

0. `00_prepare_pilot_dataset.py`: reproduce the licensed, hashed ARC question freeze.
1. `01_pilot.py`: validate config, prompts, schemas and causal-label plumbing with mock data.
2. `02_generate_dataset.py`: collect immutable Z.AI GLM rollouts with a run manifest and smoke gate.
3. `03_run_baselines.py`: correctness, surface and judge baselines.
4. `04_resampling_monitor.py`: calculate counterfactual answer-shift evidence.
5. `05_activation_probe.py`: optional residual-stream probe after behavioural gates.
6. `06_low_base_rate_eval.py`: report balanced metrics and deployment PPV.

Every script writes to a new run-specific file. Never overwrite raw generations.

## Headline metrics

Accuracy is not a headline metric. Report:

- AUROC and AUPRC with bootstrap confidence intervals;
- false-positive rate at fixed recall;
- precision at assumed prevalences of 50%, 10%, 5%, and 1%;
- performance on clean-incorrect and difficult-clean negatives;
- generalization to held-out questions and hint styles;
- manually inspected false positives and false negatives.

For prevalence \(\pi\), sensitivity \(TPR\), and false-positive rate \(FPR\):

\[
PPV(\pi)=\frac{TPR\pi}{TPR\pi + FPR(1-\pi)}.
\]

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

**Hardened scaffold / preregistration stage.** The schemas, controlled prompts, deterministic mock
pipeline, causal-label derivation, baseline interfaces, rare-event metrics, audit manifests, and
CPU-only tests are implemented. The licensed ARC pilot input freeze is committed with provenance.
The GLM-4.7-Flash adapter and bounded smoke protocol are implemented, but no live API request has
been made. Human review, grouped-bootstrap reporting, and empirical evaluation remain deliberately
unrun.

No empirical claims have been established. Mock outputs are plumbing checks and are excluded from
version control. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the ordered path from this scaffold to
a reviewed pilot; the claim ledger remains intentionally empty of results.

## License

Code and original documentation are MIT-licensed. The committed ARC-derived question records are
licensed separately under CC BY-SA 4.0; see [`data/raw/README.md`](data/raw/README.md). Future
dataset licenses must be recorded in the run manifest before source data is committed or
redistributed.
