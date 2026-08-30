# References and design lineage

This is a living reading map, not a claim that every method has already been implemented.

## Core building blocks

### Clark et al. (2018) — AI2 Reasoning Challenge

**Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.**
[Dataset](https://huggingface.co/datasets/allenai/ai2_arc) ·
[arXiv:1803.05457](https://arxiv.org/abs/1803.05457)

Use: the pinned ARC-Challenge validation split supplies the licensed multiple-choice questions for
the pilot input freeze. The project adds controlled hint conditions and causal monitoring labels;
it does not reinterpret ARC accuracy as a faithfulness measurement.

### Turpin et al. (2023)

**Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.** [arXiv:2305.04388](https://arxiv.org/abs/2305.04388)

Borrow: matched biasing interventions, answer-shift evidence, and post-hoc rationalization framing. Improve: treat influence as a repeated-sampling estimate and evaluate false positives among ordinary difficult reasoning.

### Chen et al. (2025)

**Reasoning Models Don't Always Say What They Think.** [arXiv:2505.05410](https://arxiv.org/abs/2505.05410)

Borrow: controlled hint types and the distinction between using and verbalizing a hint. Improve: make monitoring utility under prevalence shift the target rather than reporting acknowledgment rates alone.

### Arcuschin et al. (2025)

**Chain-of-Thought Reasoning In The Wild Is Not Always Faithful.** [arXiv:2503.08679](https://arxiv.org/abs/2503.08679)

Borrow: naturally occurring shortcuts, restoration errors, and realistic task failures. Use as an exploratory external-validity set only after synthetic causal labels work.

### Thought Branches (2025)

**Thought Branches: Interpreting LLM Reasoning Requires Resampling.** [arXiv:2510.27484](https://arxiv.org/abs/2510.27484)

Borrow: treat a reasoning trace as one sample from a distribution and test claims with repeated
samples. The current counterfactual answer-shift score uses only completed-rollout sibling
frequencies; it is not a literal implementation of intermediate-state branch resampling.

### Hewitt & Liang (2019)

**Designing and Interpreting Probes with Control Tasks.** [ACL Anthology](https://aclanthology.org/D19-1275/)

Borrow: selectivity and control tasks for probes. The project additionally requires grouped splits and incremental value over behavioural monitors.

### Saxton et al. (2019) — Mathematics Dataset

**Analysing Mathematical Reasoning Abilities of Neural Models.**
[Generator](https://github.com/google-deepmind/mathematics_dataset) ·
[arXiv:1904.01557](https://arxiv.org/abs/1904.01557)

Borrow: deterministic parameterized task generation, structural difficulty tiers, and exact
answers. Improve for this project: use harder multi-constraint families, retain per-instance
certificates, and calibrate cells with a target model before freezing a clean-discovery bank.

### Tsoukalas et al. (2024) — PutnamBench

**PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition.**
[Repository](https://github.com/trishullab/PutnamBench) ·
[arXiv:2407.11214](https://arxiv.org/abs/2407.11214)

Use as a possible static hardness comparison because its formalizations are machine-checkable and
its informal statements are available with permission. Do not use it as the primary pilot: proof
grading changes the response modality, and a fixed public benchmark does not remove contamination
risk.

## Prior project

### Ashioya — cot-faithfulness-mech-interp

[Repository](https://github.com/ashioyajotham/cot-faithfulness-mech-interp)

Reuse conceptually:

- activation extraction and logistic probing;
- label-shuffle/selectivity controls;
- bootstrap confidence intervals;
- explicit comparison of predictive and causal evidence;
- model-agnostic loading patterns.

Do not import as assumptions:

- arithmetic clean/corrupted pairs;
- GPT-2-specific circuit locations;
- preselected heads;
- difference-of-means steering as a primary monitor;
- the prior faithful/unfaithful label definition.

Motivating result: circuit-probe discrimination weakened on stronger architectures and interventions moved behaviour only weakly. This project tests whether behavioural counterfactual evidence is a better operational abstraction.

## Literature review questions

For each added paper, record:

1. What causal definition of faithfulness is used?
2. Is the label per transcript, per question, or distributional?
3. What information does the monitor receive?
4. Are clean-incorrect and hinted-but-resisted negatives present?
5. Are splits grouped by underlying question/template?
6. Is utility evaluated under realistic prevalence?
7. Which method or failure mode should change this repo?
