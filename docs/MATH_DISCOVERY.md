# MATH clean-only discovery

The next discovery family is free-response competition mathematics rather than ARC
multiple choice. The frozen source is the MIT-licensed
[EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
dataset (revision `main` at preparation time), sampled from level 4–5 training
shards. Six subject shards were downloaded locally and reduced to 30 records with
seed `8700`.

Preparation uses `experiments/00_prepare_math_dataset.py`. It records the source
shards, output SHA-256, and `format_novelization_v1` (whitespace normalization plus
a neutral instruction wrapper). This is a reproducible formatting variant, not a
claim of cryptographic decontamination.

The discovery config is `configs/tinker_math_discovery.yaml`: Qwen3.6-35B-A3B via
Tinker, thinking enabled, 4,096 maximum new tokens, clean condition only, and two
samples per problem (60 requests). Exact-match parsing accepts a single boxed or
explicit final answer and canonicalizes numeric fractions/decimals conservatively.
No intervention experiment should be run until this clean-only error gate is
reviewed.

The active renderer uses `math_prompt_v2`: answer first, at most six derivation
lines, and no self-check or repeated drafts.
