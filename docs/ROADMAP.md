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
- [x] Freeze a licensed, version-pinned 20-question ARC-Challenge pilot input set.
- [x] Implement and contract-test the Z.AI GLM-4.7-Flash generation adapter.
- [x] Implement deterministic, solver-verified procedural task generation and adaptive cell
  selection without item-level outcome leakage.

Mock generations validate plumbing only. They are not evidence for any claim in the claim ledger.

## Current empirical milestone: procedural task construction

- [x] Freeze and verify the 120-question procedural v1 candidate bank.
- [x] Screen one clean medium-reasoning rollout per candidate; record the failed mixed-outcome
  gate and ambiguous recurrence renderer.
- [x] Run the 24-request low-reasoning attribution diagnostic; record its formal failure and never
  reuse its selected outputs in monitor data.
- [x] Freeze a fresh 80-question procedural v2 bank and its outcome-independent screening rule.
- [ ] Run the 80-request v2 low-reasoning screen and, only if all families qualify, freeze 40
  questions without using individual outcomes.
- [ ] Pass the clean-only mixed-outcome gate on fresh questions.
- [ ] Preregister and run a causal-yield experiment only after the clean task gate passes.
- [ ] Train monitors only after causal positives also pass validity and diversity gates.

## Milestone 1: freeze the pilot inputs

- Pass and manually inspect the nine-request GLM smoke collection.
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
