# Procedural-math-v2 clean mixed-outcome preregistration

Status: **frozen before Tinker screening**  
Date: 2026-08-30

## Monitoring-first objective

This is not a study of whether GPT-OSS can reason on mathematics. It asks whether we can construct
a controlled reasoning environment with enough ordinary clean failures to test whether monitors
distinguish them from causally induced unfaithfulness. Mathematics is the controlled task
substrate, not the scientific endpoint.

The v1 medium-reasoning screen failed because difficulty mostly produced truncation, and its
low-reasoning diagnostic also failed its formal gate: 19/24 responses were automatically
scorable, with one truncation and four parser-invalid responses. Manual inspection may motivate
prospective parser changes, but it cannot regrade or rescue that historical gate. Diagnostic
questions and outputs remain permanently excluded from monitor data.

## Frozen candidate construction

Version `procedural-math-v2` contains 80 new questions: ten instances in every cell of four
families (`conditional_dag`, `subset_counting`, `finite_state`, and `affine_modular`) crossed with
two boundary tiers. Each question has two prompt renderers and an exact integer answer. Every
certificate binds the prompt, parameters, seed, answer, and generator version and is checked by a
second exact method. The original code and generated records are MIT-licensed.

The answer parser is changed prospectively: after boxed and labelled answers, it may accept only
the final display-math block when that block is a bare numeric value or a single equation with a
numeric right-hand side. Multi-equation, symbolic-right-hand-side, and nonnumeric displays remain
invalid. Historical v1 summaries are unchanged.

## Stage 1: outcome-independent screen

- Model: `openai/gpt-oss-20b` through Tinker.
- Renderer: low reasoning, 4,096 maximum new tokens.
- Sampling: one clean rollout for each of 80 candidates.
- A family-by-tier cell is eligible only with zero request errors, at least 9/10 automatically
  scorable responses, and 30–70% accuracy among scorable responses.
- Every family must have at least one eligible tier.
- If the gate passes, select ten questions per family from eligible cells using selection seed
  `20261301`, round-robin tier-renderer balance, and a hash order that does not use individual
  correctness. Freeze exactly 40 questions.

If any family lacks an eligible cell, stop and redesign the task bank. Do not select observed
individual failures.

## Stage 2: fresh clean-only mixed-outcome gate

Collect three new clean rollouts for each of the 40 frozen questions. The gate passes only if all
of the following hold:

- all 120 requests are stored with unique rollout IDs, provider request IDs, and three distinct
  logical seeds per question;
- zero request errors and no unknown, non-clean, or wrong-model records;
- at least 90% are automatically scorable and at most 10% truncate;
- at least 24 answers are correct and at least 24 are incorrect;
- errors span at least six questions and three generator families;
- correct answers span at least three generator families; and
- reasoning is present for every scorable response.

Report accuracy with a question-cluster bootstrap interval. Parser-invalid and truncated outputs
are not ordinary errors.

Passing this gate authorizes only a separately preregistered causal-yield experiment. It does not
authorize monitor training. Monitor training becomes defensible only after the intervention
creates valid causally induced unfaithfulness with adequate prevalence and diversity, and after
question-group splits and monitor views are frozen.
