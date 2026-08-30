# Exploratory MATH amendment

Status: **freeze before GPT-OSS discovery generation**  
Date: 2026-08-29

This amendment preserves the original ARC/GLM preregistration as historical. The ARC slice
produced no clean errors and no valid silent-use candidates, so subsequent work is exploratory
until a new behavioural cohort passes the gates below.

## Discovery question

Can a current open reasoning model produce a sufficiently complete and diverse set of natural
errors on licensed level-4/5 MATH problems to support difficult-clean monitor controls?

The frozen input contains 30 records selected with seed `8700` from six MATH training subjects at
dataset revision `21a5633873b6a120296cce3e2df9d5550074f4a3`. The committed manifest records every
source-shard digest. Formatting novelization does not establish decontamination.

## Model and collection

- Provider: Tinker only.
- Attribution diagnostic: Qwen3.6-35B-A3B, full thinking versus a non-thinking capability control.
- Discovery model: `openai/gpt-oss-20b`, revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`, Apache-2.0.
- Renderer: `gpt_oss_medium_reasoning`; temperature 1.0, top-p 0.95, 4,096 output tokens.
- Smoke: six subject-balanced questions, two seeds each.
- Cohort: all 30 questions, two new seeds each; deterministic subject-balanced request order.

Every provider response is retained. Length truncation, malformed format, answer parse failure,
and transport failure are separate outcomes. Only cleanly terminated, parseable responses with
reasoning are scorable. Mathematical grading is exact first, then numeric/restricted symbolic
equivalence; unsupported forms require review and are not counted as incorrect.

## Gates

- Smoke: at least 10/12 scorable, reasoning present for every scorable response, unique provider
  request IDs, and zero transport failures.
- Discovery feasibility: no more than 20% length truncation.
- Natural-error yield: at least six scorable incorrect rollouts spanning two subjects.

If either discovery gate fails, report the null yield result and stop intervention spending. If
both pass, natural errors become difficult-clean negatives for a separately frozen causal-yield
experiment using matched correct and subtly flawed partial solutions. Monitor training remains
blocked until that experiment produces unacknowledged, causally shifted continuations.

## Claims and exclusions

No prior ARC or MATH calibration is pooled with this cohort. Runs with different prompts, models,
token limits, source freezes, or reasoning modes are reported separately. Question-level grouping
is mandatory for intervals and train/test splits. The primary eventual metric remains FPR at 50%
recall, with PPV at 1% prevalence; no monitor-performance claim is currently established.
