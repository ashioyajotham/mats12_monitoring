# Natural-error discovery

## Protocol

All 20 frozen ARC-Challenge validation questions were sampled under the clean prompt only, with
three fresh provider seeds per question. No hint or metadata intervention was present. The model,
renderer, decoding parameters, and seeds are frozen in `configs/tinker_natural_failures.yaml`.

## Result (2026-08-29)

The cohort completed 60/60 requests with reasoning in every output, 60 unique Tinker sequence IDs,
no invalid generations, and only `stop` finishes. Output SHA-256:
`e9834905dc95574fdbdc6651870ad68cf1e89962e6f4aba2b7579656667a9267`.

Every question was answered correctly in all three clean samples: 60/60 accuracy and zero
naturally occurring clean errors. Mean completion was 1,493.5 tokens and the estimated cost was
approximately $0.123 at the recorded 2026-08-29 Tinker rates.

This is evidence about this model/task slice, not a general capability claim. It means ARC-Challenge
items selected for the current freeze cannot supply the difficult-clean negative class needed for
the monitor study. Do not manufacture errors with stronger hints. Freeze a harder, licensed task
family or benchmark first, then repeat clean-only discovery before any intervention.
