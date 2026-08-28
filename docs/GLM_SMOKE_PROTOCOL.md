# GLM-4.7-Flash smoke protocol

## Purpose

Validate the hosted generation boundary before spending the pilot's 300-request budget. Smoke
outputs test infrastructure only and do not enter monitor metrics, causal labels, or the claim
ledger.

## Frozen model contract

- Provider: Z.AI general-purpose API (`https://api.z.ai/api/paas/v4`).
- API model alias: `glm-4.7-flash`.
- Corresponding open weights: `zai-org/GLM-4.7-Flash` at revision
  `7dd20894a642a0aa287e9827cb1a1f7f91386b67`, MIT license.
- Thinking: enabled; preserve `reasoning_content` separately from final `content`.
- Sampling: temperature 1.0, top-p 0.95, maximum 4,096 output tokens.
- Seeds: Z.AI does not document seeded inference. Stored seeds are logical sample identifiers and
  are not sent to the provider.

The open-weight revision identifies the public model artifact associated with the API alias; it
does not prove that the hosted service serves that exact immutable checkpoint. Provider response
model names, request IDs, timestamps, usage, and the code/config hashes are therefore retained.

## Procedure

First validate the nine-request plan without credentials or network calls:

```bash
python experiments/02_generate_dataset.py \
  --config configs/glm_smoke.yaml \
  --limit 3 \
  --samples-per-condition 1 \
  --dry-run
```

Then export the API key in the current shell and run the same command without `--dry-run`:

```bash
export ZAI_API_KEY="..."
python experiments/02_generate_dataset.py \
  --config configs/glm_smoke.yaml \
  --limit 3 \
  --samples-per-condition 1 \
  --request-delay-seconds 15
```

Never pass the key as a command-line argument or commit it to a file.

## Automatic integrity gate

The smoke run passes only when all nine requests complete, every final answer parses, every rollout
contains reasoning, every request ID is present and unique, and every finish reason is `stop`.
Transient transport failures and HTTP 408/429/5xx responses receive bounded exponential retries.
Partial rollouts and a failure manifest remain available if collection stops early.

To continue after a transient failure without overwriting evidence, start a new run that imports
the compatible completed rollouts and skips their stable identities:

```bash
python experiments/02_generate_dataset.py \
  --config configs/glm_smoke.yaml \
  --limit 3 \
  --samples-per-condition 1 \
  --request-delay-seconds 15 \
  --resume-from data/generated/glm_smoke_test_only_<timestamp>
```

Resume refuses runs with a different configuration hash. The new manifest records the source run
and imported count; the source artifacts remain untouched.

## Local fallback feasibility

The official Hugging Face checkpoint is a 31B-parameter BF16 model split across 48 weight shards,
roughly 60 GB before inference overhead. Ollama provides a Q4_K_M build at roughly 19 GB, plus
runtime and KV-cache memory, and currently requires Ollama 0.14.3 or newer. Neither is a reliable
fit for a 16 GB unified-memory Mac: the Ollama build may rely heavily on swap or fail to load, while
the BF16 checkpoint is decisively too large. Hosted Z.AI therefore remains the pilot default.

If local hardware with at least 24--32 GB available memory becomes available, prefer the Ollama
quantization for a feasibility smoke test. Treat it as a distinct backend/model condition: do not
combine its generations with hosted API rollouts, because quantization and serving differences can
change the monitored behavior.

Before a full pilot, manually inspect all nine reasoning/final-response pairs and confirm that the
prompt intervention is rendered correctly, reasoning is not truncated, no provider safety filter
changed the task, and the returned model identity is stable. Record the decision in
`results/decisions.md`; do not relax the gate based on desirable hint behavior.
