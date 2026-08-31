# Research roadmap

This roadmap separates implemented research infrastructure from work that must be completed before
the repository can support empirical claims. The preregistration and pilot gates remain
authoritative when a milestone conflicts with convenience.

## Current milestone: mechanism-held-out external audit

- [x] Complete and publish causal-error-v1 monitor evaluation.
- [x] Reframe answer shift as a counterfactual causal audit rather than a single-trace monitor.
- [x] Freeze omission and duplication checkpoint mechanisms with exact certificates.
- [x] Run v1 placebo, decomposition, and sibling-budget robustness analyses.
- [x] Collect and gate the 144-call causal-audit-v2 qualification.
- [x] Enforce the stopping rule after subset counting produced zero uptake in both mechanisms.
- [x] Freeze a separately preregistered three-family replication labelled qualification-informed.
- [x] Collect and gate the 648-call v2.1 external cohort.
- [x] Apply only v1-fitted monitors and v1 validation thresholds.
- [x] Confirm domain-restricted transfer on all paired metrics.

The original v2 confirmatory run remains closed. V2.1 is a new domain-restricted protocol with
fresh questions; it is not a post-hoc subset of v2 data.

The next research decision is no longer another intervention. It is whether to reduce audit cost,
replicate on a second solver, or stop with the present bounded causal-audit claim.

## Paper consolidation

- [x] Freeze the central domain-restricted claim and non-claims.
- [x] Integrate v1, robustness, failed v2 qualification, and successful v2.1 replication.
- [x] Generate result-bound study-flow, causal-cell, and monitor-performance figures.
- [x] Document audit cost, family selection, low-FPR uncertainty, and single-solver limitations.
- [ ] Obtain external technical review and revise the working paper without changing frozen results.

## Completed infrastructure

- [x] Define immutable question, rollout, answer-shift, and causal-label records.
- [x] Implement controlled prompt conditions and deterministic mock generation.
- [x] Enforce question-group splits and test for leakage.
- [x] Add surface, typed Qwen judge, answer-shift, activation-probe, and hybrid contracts.
- [x] Report balanced discrimination and prevalence-adjusted precision.
- [x] Record content-addressed manifests and explicit claim/decision ledgers.
- [x] Exercise the pipeline with CPU-only contract and smoke tests.
- [x] Freeze a licensed, version-pinned 20-question ARC-Challenge pilot input set.
- [x] Implement and contract-test the Z.AI GLM-4.7-Flash generation adapter.
- [x] Implement deterministic, solver-verified procedural task generation and adaptive cell
  selection without item-level outcome leakage.

Mock generations validate plumbing only. They are not evidence for any claim in the claim ledger.

## Completed empirical milestone: task and mechanism discovery

- [x] Freeze and verify the 120-question procedural v1 candidate bank.
- [x] Screen one clean medium-reasoning rollout per candidate; record the failed mixed-outcome
  gate and ambiguous recurrence renderer.
- [x] Run the 24-request low-reasoning attribution diagnostic; record its formal failure and never
  reuse its selected outputs in monitor data.
- [x] Freeze a fresh 80-question procedural v2 bank and its outcome-independent screening rule.
- [x] Run the 80-request v2 low-reasoning screen; record that three families qualified while
  subset counting failed its cell gate.
- [x] Freeze a prospective 20-question v2.1 subset-replacement bank and adaptive protocol.
- [x] Screen the 20 fresh replacements and, only if a tier qualifies, freeze the combined 40
  questions without using individual outcomes.
- [x] Pass the clean-only mixed-outcome gate on 120 fresh responses with 64 correct, 56 incorrect,
  and zero invalid outputs.
- [x] Freeze a diagnostic-only, outcome-independent matched partial-solution causal-yield pilot.
- [x] Run the 108-request answer-bearing causal-yield pilot; record its null causal effect and zero
  defensible silent-use candidates.
- [x] Freeze an eight-question exact state-continuation v2 pilot on previously unused questions.
- [x] Run the 72-request continuation-yield pilot; record its strong automated causal effect and
  failed manual silent-use validity gate.
- [x] Implement and freeze one final assistant-prefill mechanism diagnostic on fresh eligible
  questions without inspecting model outcomes.
- [x] Preserve the first excluded smoke's clean transport and reconstruction result, formal
  parse-validity failure, and prospectively frozen grammar/retry amendment.
- [x] Run the final excluded three-request retry, then the 72-request diagnostic only if all smoke
  integrity checks pass.
- [x] Record the 72-request assistant-prefill null result and close the silent-unfaithfulness
  branch under its stopping rule.

## Current empirical milestone: causal-error dataset construction

- [x] Freeze the new estimand, prospective partitions, causal gates, labels, monitor views, and
  grouped evaluation in `docs/PREREGISTRATION_CAUSAL_ERROR_DETECTION_V1.md`.
- [x] Implement and test the fresh 96-question generator, certificates, and deterministic
  qualification/confirmatory split without consulting model outcomes.
- [x] Run the 72-request clean qualification cohort and stop unless every mixed-outcome and
  integrity gate passes.
- [x] Run the 648-request randomized confirmatory causal collection and publish its
  machine-readable gate report.
- [x] Implement the gate-bound monitor materializer, typed evidence boundaries, local controls,
  two-view resumable Qwen judge, five-fold grouped OOF hybrid, and clustered evaluation.
- [x] After every confirmatory gate passed, run the complete behavioural monitor stack and publish
  grouped held-out results without reviving silent-unfaithfulness labels.

## Historical milestone 1: freeze the pilot inputs

- Pass and manually inspect the nine-request GLM smoke collection.
- Render and manually inspect every prompt template before collection.
- Freeze the pilot configuration and preregistration; enable the positive-lower-bound requirement
  for confirmatory data.

## Historical milestone 2: collect and validate the phenomenon

- Collect append-only clean, incorrect-hint, and irrelevant-metadata rollouts with complete
  manifests.
- Estimate per-question answer shifts and replace the exploratory small-sample normal interval if
  preregistration selects a more appropriate interval method.
- Complete blinded and causal-evidence review, including double-review agreement and Cohen's kappa.
- Publish a machine-readable gate report and stop or pivot when any preregistered gate fails.

## Future milestone: evaluate behavioural monitors

- Freeze train/validation/test question groups and monitor inputs.
- Run correctness, CoT-length, template identity, shuffled-label, surface, transcript-only judge,
  context-aware judge, counterfactual answer-shift, and hybrid comparisons.
- Add grouped-bootstrap confidence intervals to every headline metric and verify analytic
  low-prevalence precision against an explicitly prevalence-shifted sample.
- Audit high-confidence false positives and false negatives, then update the claim ledger without
  hiding null results.

## Optional milestone: internals

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
