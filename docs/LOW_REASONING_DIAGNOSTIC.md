# Low-reasoning attribution diagnostic result

The preregistered diagnostic gate **failed** on 2026-08-30. All 24 Tinker requests were stored
without transport errors and had reasoning, but only 19 were automatically scorable: 11 correct,
8 incorrect, 4 parser-invalid, and 1 truncated. The matched-control stratum supplied only 9
scorable responses, below its required 10, and the overall total was below the required 20.

Manual inspection found final display equations in the four parser-invalid responses. That is
useful parser-development evidence, not permission to change the historical outcome. The parser
hardening introduced for procedural-math-v2 applies prospectively; this report remains failed.
Because the diagnostic was selected using v1 outcomes, its questions and every resulting rollout
remain excluded from monitor training, validation, testing, and the v2 mixed-outcome cohort.

The result supports using low reasoning to reduce length censoring, but it does not establish a
valid monitor-ready ordinary-error distribution. The next legitimate milestone is the frozen v2
screen and fresh clean-only mixed-outcome gate in
[`PREREGISTRATION_PROCEDURAL_V2.md`](PREREGISTRATION_PROCEDURAL_V2.md).
