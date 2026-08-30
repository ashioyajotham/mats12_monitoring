# Project overview: When does a reasoning monitor cry wolf?

![Pixel-art illustration of the monitoring problem](../assets/mats12_monitoring_pixel_art.png)

## The project in plain English

Imagine that two students submit the same wrong answer:

1. One made an ordinary calculation mistake.
2. The other was given a bad intermediate value, trusted it, and carried that error through the
   rest of the solution.

The final answer alone cannot tell us what happened. Even their written working might look
similar.

This project asks:

> Can an AI monitor reliably distinguish an ordinary reasoning mistake from one caused by a
> deliberately corrupted intermediate step?

## Why this matters

A monitor that flags every wrong answer is not very useful—we already know that the answer is
wrong. A useful monitor must identify the *kind* of failure without constantly raising false
alarms.

This matters especially when intervention-caused errors are rare. Even a monitor that looks good
on a balanced test could overwhelm human reviewers if it incorrectly flags many ordinary mistakes
in real use. This is the project's “cry wolf” problem.

## Background

The project originally investigated whether reasoning models would secretly follow misleading
hints while presenting an apparently independent explanation. The experiments found that:

- weak hints were usually ignored;
- strong hints sometimes changed answers, but the model generally acknowledged them;
- assistant-prefill interventions did not reliably create hidden influence; and
- a corrupted intermediate state could strongly cause a predictable downstream error, although
  that influence was usually visible in the reasoning context.

Those results closed the project's silent-unfaithfulness branch. They motivated a narrower and
more defensible monitoring question.

The project is therefore **not** framed as “Does GPT-OSS reason on mathematics?” Mathematics is
used as a controlled laboratory because answers can be checked automatically and the consequences
of changing one intermediate value can be calculated exactly.

## The controlled environment

The experiment constructs problems where the model can produce two important types of failure:

- **Ordinary error:** the model receives an unmodified problem and reaches a wrong answer through
  its normal reasoning process.
- **Causally induced error:** one certified intermediate value is deliberately corrupted, and the
  model reaches the exact wrong answer that should follow from that corruption.

Repeated clean, correct-state, and corrupted-state attempts provide comparison data. Independently
generated solver certificates establish the correct intermediate state and the propagated wrong
target before model outputs are analyzed.

The label supports a claim about a controlled causal intervention. It does not establish hidden
influence, deception, intent, or whether a model “really” used its chain of thought.

## What the monitors compare

The project evaluates several approaches on the same question-group-disjoint held-out data:

- **Correctness-only control:** demonstrates that correctness cannot distinguish two classes that
  both contain wrong answers.
- **Reasoning-length and template controls:** test whether simple nuisance patterns explain the
  result.
- **Surface monitor:** uses inexpensive features of the visible reasoning, such as length,
  uncertainty language, and backtracking.
- **Transcript-only LLM judge:** sees only the question, reasoning, and final answer.
- **Context-aware LLM judge:** additionally sees the prompt and supplied intermediate state.
- **Counterfactual answer shift:** asks whether the focal answer becomes more common in repeated
  corrupted attempts than in clean or correct-state attempts.
- **Hybrid monitor:** combines the surface, judge, and answer-shift signals.

The transcript-only information boundary is enforced structurally: it cannot receive condition
names, intervention certificates, sibling outcomes, provenance, or derived labels.

## How success is measured

Overall accuracy is not enough. The main comparison asks whether counterfactual answer shift beats
the surface baseline while keeping false alarms low at useful recall.

The evaluation reports:

- false-positive rate at 50% recall;
- AUROC and AUPRC;
- precision at assumed event rates of 10%, 5%, and 1%;
- question-clustered uncertainty intervals;
- performance by task family; and
- descriptive behavior on difficult hard-negative examples.

Train, validation, and test splits are grouped by question so that repeated versions of the same
problem cannot leak across splits.

## What the illustration means

- The **yellow branch** represents an ordinary mistake.
- The **red branch** represents a corrupted intermediate state propagating into a wrong answer.
- The **central scanner** represents a monitor comparing the two error patterns.
- The **uncertainty display** emphasizes that monitoring is probabilistic and can raise false
  alarms.

## Current research stage

The clean qualification cohort passed its mixed-outcome gate, demonstrating that the controlled
task environment produces enough ordinary failures. The confirmatory randomized collection is the
next gate.

Monitor fitting and judge spending remain blocked until the completed confirmatory dataset passes
all causal-effect, clean-negative, diversity, provenance, and integrity requirements. Partial or
failed collections cannot be used for monitor evaluation.

The intended end result is not a claim that one model is good or bad at mathematics. It is an
evidence-backed answer to a monitoring question:

> Can we detect errors caused by a controlled corruption without falsely accusing too many
> ordinary reasoning failures?
