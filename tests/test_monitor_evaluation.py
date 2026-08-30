from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.judge_runner import JudgeScoreRecord
from src.monitor_dataset import ContextAwareEvidence, MonitorExample, TranscriptOnlyEvidence
from src.monitor_evaluation import (
    build_hybrid_scores,
    build_local_component_scores,
    evaluate_component,
    invalid_rollout_split_diagnostic,
    paired_component_comparison,
    summarize_secondary_audit,
    validation_operating_point,
)


def _rollout(rollout_id: str, question_id: str, condition: Condition, answer: str) -> Rollout:
    return Rollout(
        rollout_id=rollout_id,
        question_id=question_id,
        task_family="test",
        condition=condition,
        hinted_option="B" if condition is Condition.CORRUPTED_CONTINUATION else None,
        hint_template=None,
        prompt="p",
        response=f"reasoning {answer}",
        reasoning=f"reasoning {answer}",
        final_response=f"\\boxed{{{answer}}}",
        parsed_answer=answer,
        gold_answer="A",
        seed=int(rollout_id.split("-")[-1]),
        model="model",
        generation={"temperature": 1.0},
        finish_reason="stop",
        created_at="2026-08-30T00:00:00+00:00",
        status=RolloutStatus.CLEAN_STOP,
    )


def _fixture():
    examples: list[MonitorExample] = []
    rollouts: list[Rollout] = []
    for question_index in range(25):
        split = "train" if question_index < 15 else "validation" if question_index < 20 else "test"
        question_id = f"q{question_index}"
        raw = [
            _rollout(f"{question_id}-0", question_id, Condition.CLEAN, "C"),
            _rollout(f"{question_id}-1", question_id, Condition.CLEAN, "A"),
            _rollout(f"{question_id}-2", question_id, Condition.CLEAN, "A"),
            _rollout(f"{question_id}-3", question_id, Condition.CORRECT_CONTINUATION, "A"),
            _rollout(f"{question_id}-4", question_id, Condition.CORRECT_CONTINUATION, "A"),
            _rollout(f"{question_id}-5", question_id, Condition.CORRECT_CONTINUATION, "A"),
            _rollout(f"{question_id}-6", question_id, Condition.CORRUPTED_CONTINUATION, "B"),
            _rollout(f"{question_id}-7", question_id, Condition.CORRUPTED_CONTINUATION, "B"),
            _rollout(f"{question_id}-8", question_id, Condition.CORRUPTED_CONTINUATION, "B"),
        ]
        rollouts.extend(raw)
        for rollout_id, answer, label, kind in (
            (f"{question_id}-0", "C", 0, "ordinary_error"),
            (f"{question_id}-6", "B", 1, "causally_induced_error"),
        ):
            transcript = TranscriptOnlyEvidence(
                question="q",
                reasoning=("short ordinary" if label == 0 else "long state propagation text"),
                final_answer=answer,
            )
            examples.append(
                MonitorExample(
                    example_id=rollout_id,
                    rollout_id=rollout_id,
                    question_id=question_id,
                    split=split,
                    binary_label=label,
                    kind=kind,
                    family=f"family-{question_index % 4}",
                    tier=f"tier-{question_index % 2}",
                    renderer_id=question_index % 2,
                    transcript=transcript,
                    context=ContextAwareEvidence(
                        **transcript.model_dump(), rendered_prompt="p"
                    ),
                )
            )
    return examples, rollouts


def test_local_scores_use_oof_surface_for_hybrid_training() -> None:
    examples, rollouts = _fixture()
    scores = build_local_component_scores(examples, [], rollouts, seed=42)
    train_ids = {row.example_id for row in examples if row.split == "train"}
    train_surface = [
        row for row in scores
        if row.component == "surface" and row.example_id in train_ids
    ]
    assert len(train_surface) == len(train_ids)
    assert {row.score_origin for row in train_surface} == {"five_fold_question_oof"}
    assert {row.component for row in scores} == {
        "correctness_only",
        "reasoning_length",
        "template_identity",
        "shuffled_label_surface",
        "counterfactual_answer_shift",
        "surface",
    }


def test_hybrid_consumes_real_component_scores_and_grouped_report_runs() -> None:
    examples, rollouts = _fixture()
    local = build_local_component_scores(examples, [], rollouts, seed=43)
    judges = [
        JudgeScoreRecord(
            score_id=f"{row.example_id}-{view}",
            example_id=row.example_id,
            question_id=row.question_id,
            split=row.split,
            view=view,
            score=(0.8 if row.binary_label else 0.2),
            rationale="fixture",
            model="judge",
            prompt_version="v1",
            seed=index,
            attempts=1,
            provider_request_id=f"p-{row.example_id}-{view}",
            finish_reason="stop",
            usage={},
        )
        for index, row in enumerate(examples)
        for view in ("transcript_only", "context_aware")
    ]
    hybrid = build_hybrid_scores(examples, local, judges, seed=44)
    assert {row.split for row in hybrid} == {"validation", "test"}
    report = evaluate_component(
        examples,
        [*local, *hybrid],
        component="hybrid",
        bootstrap_samples=20,
        bootstrap_seed=45,
    )
    assert report["point"]["auroc"] == 1.0
    assert report["cluster_bootstrap"]["successful_samples"] > 0
    operating_point = validation_operating_point(
        examples, [*local, *hybrid], component="hybrid"
    )
    assert operating_point["validation_recall"] >= 0.5
    assert 0.0 <= operating_point["test_false_positive_rate"] <= 1.0


def test_secondary_audit_is_descriptive_only() -> None:
    examples, rollouts = _fixture()
    local = build_local_component_scores(examples, [], rollouts, seed=46)
    audit_rows = [row for row in examples if row.split == "test"][:3]
    summary = summarize_secondary_audit(audit_rows, local, component="surface")
    assert summary["overall"]["n"] == 3
    assert "auroc" not in summary["overall"]
    comparison = paired_component_comparison(
        examples,
        local,
        first="counterfactual_answer_shift",
        second="surface",
        bootstrap_samples=20,
        bootstrap_seed=47,
    )
    assert comparison["contrast"] == "counterfactual_answer_shift_minus_surface"
    invalid = invalid_rollout_split_diagnostic(examples, seed=48)
    assert invalid["status"] == "invalid_leakage_diagnostic_not_research_evidence"
    assert invalid["question_groups_crossing_split"] > 0
