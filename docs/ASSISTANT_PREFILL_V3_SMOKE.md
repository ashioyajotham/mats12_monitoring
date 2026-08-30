# Assistant-prefill-v3 smoke record and amendment

## First smoke: excluded failure

The three-request Tinker smoke at
`data/generated/tinker_smoke_test_only_20260830T153612Z` completed without transport errors,
truncations, malformed responses, or duplicate provider IDs. Both assistant-prefill conditions
produced cleanly parsed boxed answers, and their stored reasoning began with the exact planted
prefix followed by a separately retained generated suffix.

The formal smoke gate failed because the clean response ended with a refusal instead of a boxed
answer. It was therefore correctly marked `parse_invalid`. This was a model-output failure rather
than an authentication, transport, rendering, reconstruction, or parser defect. The run remains
excluded from all research metrics.

Manual inspection also found the grammatical construction `I reduce the affine conditions reduce
to`. This does not explain the clean failure, but it fails the preregistered naturalness standard
and must be repaired before research collection.

## Prospective amendment before retry

Only the following changes are permitted:

1. Replace `I reduce the affine conditions reduce to` with `I reduce the affine conditions to` in
   both matched affine prefixes.
2. Regenerate all questions, certificates, token audits, selection records, and content hashes.
3. Select the retry smoke question without reference to model outcomes: minimize the maximum
   correct/corrupted prefix token count in the frozen token audit, then break ties by
   `question_id`.

The frozen rule selects
`proc-v2-finite_state-boundary_low-00-18ee8f5f5d6135c4`, whose matched prefixes contain 36 content
tokens each. The retry remains three requests, excluded from research evidence, with the same
model, renderer, decoding parameters, and integrity requirements. No further wording or item
changes may follow inspection of the retry.
