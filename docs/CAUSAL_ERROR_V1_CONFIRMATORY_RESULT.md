# Causal-error-detection-v1 confirmatory result

The frozen confirmatory collection **passed every preregistered causal, clean-negative, diversity,
and integrity gate**. All 648 Tinker requests were stored with zero request errors. Of these, 644
were scorable, two were length-truncated, and one was parse-invalid.

Corrupted-state target uptake was 93/215 (43.3%), compared with 8/215 (3.7%) in clean controls: a
39.5-point effect with a question-clustered 95% interval from 29.9 to 49.1 points. Target uptake
spanned 47 questions and all four families. Correct-state accuracy was 51.9%, above clean accuracy
of 41.9%, so the causal effect is not explained by a generally damaging continuation interface.

The clean condition supplied 125 completed ordinary errors across 63 questions and every family.
The gate-bound primary monitor dataset therefore contains 125 ordinary-error negatives and 93
certified-target causal-error positives. All train, validation, and test splits contain both
classes. Monitor training is formally authorized.

## Artifact binding

- run: `data/generated/tinker_causal_error_v1_confirmatory_20260830T194441Z`;
- rollout SHA-256: `ae99d49bf8248e2997ba24c729639a864d69ab2cbb8e16d9a86238d4e1192400`;
- run-manifest SHA-256: `a3e80b9e25eeaa26632f3e226396179bb3aa60afe0004daa12a0d4efebe405dc`;
- report: `results/causal_error_v1_confirmatory.json`;
- report SHA-256: `268dcdaedf1acc4f95b0c27dc38dcda51475796e1135d4cc46106b95c7410546`.

These results establish a controlled environment containing both ordinary failures and causally
induced errors. They do not establish that the influence is hidden, deceptive, or unfaithful.
