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
- Sampling: temperature 1.0, top-p 0.95, maximum 2,048 output tokens.
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
  --samples-per-condition 1
```

Never pass the key as a command-line argument or commit it to a file.

## Automatic integrity gate

The smoke run passes only when all nine requests complete, every final answer parses, every rollout
contains reasoning, every request ID is present and unique, and every finish reason is `stop`.
Transient transport failures and HTTP 408/429/5xx responses receive bounded exponential retries.
Partial rollouts and a failure manifest remain available if collection stops early.

Before a full pilot, manually inspect all nine reasoning/final-response pairs and confirm that the
prompt intervention is rendered correctly, reasoning is not truncated, no provider safety filter
changed the task, and the returned model identity is stable. Record the decision in
`results/decisions.md`; do not relax the gate based on desirable hint behavior.
