# Procedural-v2.1 clean mixed-outcome result

The fresh clean-only gate **passed** on 2026-08-30. The Tinker collection stored and automatically
scored all 120 planned GPT-OSS-20B responses: 64 correct and 56 incorrect, with zero truncations,
parser failures, malformed responses, or request errors. Accuracy was 53.3%; the preregistered
question-cluster bootstrap 95% interval was 42.5–64.2%.

Correct and incorrect answers occurred in all four families. Errors appeared on 31/40 questions.
Across the three samples per prompt, seven questions were always wrong, eleven were wrong twice,
thirteen were correct twice, and nine were always correct. Thus 24 questions produced both clean
successes and clean failures under identical prompts.

The immutable evidence bindings are:

- run: `data/generated/tinker_procedural_v21_clean_mixed_outcome_20260830T105719Z`;
- rollouts SHA-256: `b52413a585a45f5bee5a2a952040152c0f8ce788e0672c302ca98759e9da4cb4`;
- run-manifest digest: `b610777b69c3eddda6ae15e9e92f7f7bed88b2dce8a3558b78356c72e376b0e0`;
- frozen questions SHA-256: `fa52e11b830181ed914d652c0ac0657f9abbba729319f4ab1c66ea329733178a`;
  and
- result SHA-256: `85a1ee81e6937eed26ede6b25751080e863b865e318fb09bb4e608307760ba11`.

Difficulty tier and renderer remain nuisance predictors of correctness. Any later causal dataset
must pair conditions within question, split by question group, report correctness and trace-length
baselines, and prevent family, tier, or renderer imbalance from masquerading as monitoring.

This result answers the enabling task-construction question positively. It does not establish
causal unfaithfulness or monitor performance. Per the preregistration, it authorizes only a
separately frozen causal-yield pilot.
