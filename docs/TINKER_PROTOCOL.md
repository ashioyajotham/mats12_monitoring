# Tinker Qwen sampling protocol

## Frozen model contract

- Provider: Thinking Machines Lab Tinker native Sampling API.
- Model: `Qwen/Qwen3.6-35B-A3B`.
- Public artifact revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Public artifact license: Apache-2.0.
- Renderer: official Tinker Cookbook `qwen3_5` renderer with thinking enabled.
- Sampling: temperature 1.0, top-p 0.95, maximum 8,192 completion tokens.
- Seeds: sent to Tinker as true provider sampling seeds.

Tinker's sequence ID, stop reason, prompt/completion token counts, prompt-cache hits, latency,
reasoning, final response, and exact sampling parameters are retained for every rollout.

## Commands

Install the optional provider dependencies and validate the three-request plan:

```bash
uv sync --extra dev --extra tinker
python experiments/02_generate_dataset.py \
  --config configs/tinker_smoke.yaml \
  --limit 1 \
  --samples-per-condition 1 \
  --dry-run
```

Store `TINKER_API_KEY` only in the ignored `.env`, load it into the shell, and remove `--dry-run`
to collect. Resume from a compatible prior run with `--resume-from <run-directory>`.

## 2026-08-29 yield result

The consistent 8,192-token pilot completed 30/30 rollouts across five questions, three conditions,
and two samples. Every response parsed, contained reasoning, used a unique Tinker sequence ID, and
finished with `stop`. Output SHA-256:
`15d9b4a29faee77a813a15a26ae09ab9c9d7089d285cc4c5052f49b8f78fc87d`.

No question shifted toward its incorrect hint, giving a 0% candidate rate against the
preregistered 15% minimum. These outputs are pilot evidence only. Do not launch the 300-rollout
collection without revising and separately piloting the intervention or task difficulty.

The collector supports `--continue-on-error` for bounded provider/model failures. Such failures are
written to `request_errors.jsonl`, counted as invalid in the manifest, and never silently omitted.

## Assistant-prefill extension

The v3 diagnostic requires Tinker `>=0.26.1` and Tinker Cookbook `>=0.5.5`. It calls the official
renderer's `build_generation_prompt` with an assistant-role prefill that opens GPT-OSS's Harmony
`analysis` channel. Parsing prepends the exact intervention tokens to the sampled suffix so the
stored full reasoning is faithful to the model context. Prompt usage includes the prefix; completion
usage and `generated_reasoning` include only newly sampled tokens.

The Z.AI adapter rejects this field explicitly because equivalent continuation semantics have not
been established there.
