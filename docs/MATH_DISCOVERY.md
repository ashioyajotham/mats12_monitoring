# MATH clean-only discovery

The next discovery family is free-response competition mathematics rather than ARC
multiple choice. The frozen source is the MIT-licensed
[EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
dataset (revision `21a5633873b6a120296cce3e2df9d5550074f4a3`), sampled from level 4–5 training
shards. Six subject shards were downloaded locally and reduced to 30 records with
seed `8700`.

Preparation uses `experiments/00_prepare_math_dataset.py`. It records the source
shards, output SHA-256, and `format_novelization_v1` (whitespace normalization plus
a neutral instruction wrapper). This is a reproducible formatting variant, not a
claim of cryptographic decontamination.

The first discovery attempts used Qwen3.6-35B-A3B through Tinker. Full-thinking
generation repeatedly length-censored difficult questions at 16K, 8K, and 4K.
After the collector began retaining partial responses, an eight-request diagnostic
showed deterministic question-level censoring: 4/8 clean stops and 4/8 truncations,
with both seeds agreeing on which questions truncated. A non-thinking control removed
length truncation but exposed answer-first prompt revisions, so that prompt was retired.

The amended discovery uses `openai/gpt-oss-20b` with medium reasoning and
`concise_final_last_v3`. Its six-subject smoke yielded 10/12 scorable responses,
16.7% truncation, reasoning and unique request IDs on all responses, and no transport
errors. Although the generic manifest recorded the older all-valid smoke gate, this
meets the pre-run amended gate of at least 10/12 scorable and at most 20% truncation.

The full 30-problem, two-seed cohort stored 60/60 responses: 59 clean stops, one
length truncation, no parse-invalid or transport failures, and reasoning plus unique
request IDs for every request. Conservative equivalence grading found 59/59 scorable
answers correct. The natural-error gate therefore failed: zero errors versus the
required six across two subjects. Output SHA-256:
`79c75e0b19678e4d6a159ba8ea30eed1f6ccce5a183f92ddda4d70e62175e450`.
Manifest SHA-256:
`41384ca9f0622493db286d4cdd76ae7035d7a841921da91a610e90e322fed0bf`.

This is a valid null-yield result for this selected slice, not a general MATH accuracy
claim. Per the amendment, intervention spending stops here.
