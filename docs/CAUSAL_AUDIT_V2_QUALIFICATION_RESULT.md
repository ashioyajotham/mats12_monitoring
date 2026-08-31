# Causal-audit-v2 qualification result

The frozen mechanism-held-out qualification **failed its preregistered family/mechanism diversity
gate**, so the 648-call confirmatory collection is not authorized.

All 144 planned Tinker requests were stored with zero request errors and unique provider IDs. Of
these, 143 were scorable, one correct-continuation response was parse-invalid, and none truncated.
Certificates, conditions, seeds, model identity, and frozen cell balance all verified.

## Aggregate causal result

Corrupted-state target uptake was 19/48 (39.6%), compared with 2/48 (4.2%) under clean prompts: a
35.4-point effect with a question-clustered 95% interval from 14.6 to 56.2 points. Both held-out
mechanisms cleared their pooled 10-point qualification threshold:

| Mechanism | Target effect | Clustered 95% interval |
|---|---:|---:|
| Drop component | +37.5 points | +16.7 to +62.5 |
| Duplicate component | +33.3 points | 0.0 to +66.7 |

The aggregate result therefore supports transfer of causal control beyond the v1 `+1`
transformation, but it does not satisfy the frozen requirement that transfer occur throughout the
task environment.

## Failed diversity gate

| Family | Drop target uptake | Duplicate target uptake |
|---|---:|---:|
| Affine modular | 2/6 | 2/6 |
| Conditional DAG | 4/6 | 4/6 |
| Finite state | 3/6 | 4/6 |
| Subset counting | **0/6** | **0/6** |

Both subset-counting cells failed, rather than one isolated six-sample cell. This is consistent
with an existing weakness: under the v1 `+1` intervention, subset counting produced only 3/54
corrupted targets, versus 22/53 to 36/54 in the other families. Descriptive inspection of the 12
new corrupted subset traces shows the solver generally restarting combinatorial enumeration or
guessing instead of propagating the supplied DP checkpoint. This inspection does not change the
formal gate.

## Decision

Do not run `tinker_causal_audit_v2_confirmatory.yaml`, remove subset counting post hoc, relax the
diversity gate, or redesign its prompt under the v2 protocol. The qualification did its job: it
identified that the purported controlled environment does not support the causal intervention
uniformly enough for the planned external monitor test.

The genuine finding is narrower: omission and duplication perturbations transfer strongly across
the affine, DAG, and finite-state recurrence families, while the isolated subset-DP checkpoint is
not a reliable causal handle. Any future three-family study must be separately preregistered and
described as qualification-informed replication, not as the original v2 confirmatory test.

## Artifact binding

- run: `data/generated/tinker_causal_audit_v2_qualification_20260831T100947Z`;
- rollout SHA-256: `7763f905ef88b54bd0206516403134cb499e59069d9155a6d45bc27dcc5bef5a`;
- run-manifest SHA-256: `de6f555bcf356d1ef7d9c4c44fa2228b165412e387dccb48fb193ae9df64c753`;
- report: `results/causal_audit_v2_qualification.json`;
- report SHA-256: `3fa2ed323b56777411be8f8ae57726e0fd9c2686bb6e556020ad9ef65ca0a4b5`;
- planned/stored requests: 144/144;
- confirmatory analysis authorized: false.
