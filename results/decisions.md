# Research decision log

| UTC time | Observation | Hypothesis / risk | Decision | Cost |
|---|---|---|---|---|
| 2026-08-27 | Prior circuit probes degraded across stronger models and interventions were weak | Activation separability may not be the operationally useful abstraction | Make causal resampling the main method; keep activation probing as stretch | Probe work deferred |
| 2026-08-27 | Low prevalence can make modest FPR operationally disastrous | Balanced AUROC may overstate monitor utility | Pre-register FPR@50% recall and PPV at 1% | None |
| 2026-08-27 | The pilot required a licensed, difficult MCQ source with stable provenance and multiple source groups | A placeholder source blocks real generation and a dominant source group could fail the diversity gate by construction | Freeze 20 ARC-Challenge validation questions from six eligible assessment collections at AllenAI revision `210d026f`; retain CC BY-SA 4.0 on data | Public benchmark contamination and non-representative group balancing are documented limitations |

Append decisions during research. Never rewrite an earlier entry; add a correction row.
