# Research roadmap

This roadmap separates implemented research infrastructure from work that must be completed before
the repository can support empirical claims. The preregistration and pilot gates remain
authoritative when a milestone conflicts with convenience.

## Current milestone: hardened scaffold

- [x] Define immutable question, rollout, answer-shift, and causal-label records.
- [x] Implement controlled prompt conditions and deterministic mock generation.
- [x] Enforce question-group splits and test for leakage.
- [x] Add surface, judge-interface, resampling, activation-probe, and hybrid contracts.
- [x] Report balanced discrimination and prevalence-adjusted precision.
- [x] Record content-addressed manifests and explicit claim/decision ledgers.
- [x] Exercise the pipeline with CPU-only contract and smoke tests.

Mock generations validate plumbing only. They are not evidence for any claim in the claim ledger.

## Milestone 1: freeze the pilot inputs

- Select one licensed multiple-choice source and record its version, license, selection rule, and
  transformations.
- Implement and contract-test one real generation adapter with a pinned model revision.
- Render and manually inspect every prompt template before collection.
- Freeze the pilot configuration and preregistration; enable the positive-lower-bound requirement
  for confirmatory data.

## Milestone 2: collect and validate the phenomenon

- Collect append-only clean, incorrect-hint, and irrelevant-metadata rollouts with complete
  manifests.
- Estimate per-question answer shifts and replace the exploratory small-sample normal interval if
  preregistration selects a more appropriate interval method.
- Complete blinded and causal-evidence review, including double-review agreement and Cohen's kappa.
- Publish a machine-readable gate report and stop or pivot when any preregistered gate fails.

## Milestone 3: evaluate behavioural monitors

- Freeze train/validation/test question groups and monitor inputs.
- Run correctness, CoT-length, surface, transcript-only judge, context-aware judge, and resampling
  comparisons.
- Add grouped-bootstrap confidence intervals to every headline metric and verify analytic
  low-prevalence precision against an explicitly prevalence-shifted sample.
- Audit high-confidence false positives and false negatives, then update the claim ledger without
  hiding null results.

## Milestone 4: optional internals

Proceed only if the phenomenon, validity, diversity, and feasibility gates all pass. Predeclare the
model layers, token-position rule, nested selection process, and shuffled-label controls before
extracting activations. Compare internals with behavioural scores on exactly the same held-out
question groups.

## Publication checklist

- Every executive-summary number links to a committed artifact.
- Dataset redistribution complies with its source license.
- Secrets, restricted prompts, and raw provider payloads remain uncommitted.
- CI, manifest verification, and a clean-room mock run pass from a fresh environment.
- Limitations distinguish synthetic hint use from deception, intent, and general model honesty.
