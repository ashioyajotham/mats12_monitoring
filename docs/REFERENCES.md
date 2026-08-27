# References and design lineage

This is a living reading map, not a claim that every method has already been implemented.

## Core building blocks

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

Borrow: treat a reasoning trace as one sample from a distribution; test claims via resampling. Start with full-rollout counterfactuals before introducing expensive intermediate branches.

### Hewitt & Liang (2019)

**Designing and Interpreting Probes with Control Tasks.** [ACL Anthology](https://aclanthology.org/D19-1275/)

Borrow: selectivity and control tasks for probes. The project additionally requires grouped splits and incremental value over behavioural monitors.

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
