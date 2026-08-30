from collections import Counter

from src.causal_error_dataset import FAMILIES
from src.causal_error_gates import (
    analyze_causal_error_confirmatory,
    analyze_causal_error_qualification,
)
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.tasks import MathProblem


def _qualification_fixture(*, all_correct: bool = False):
    questions: list[MathProblem] = []
    rollouts: list[Rollout] = []
    for family_index, family in enumerate(FAMILIES):
        for question_index in range(6):
            question_id = f"qual-{family}-{question_index}"
            question = MathProblem(
                question_id=question_id,
                prompt="Compute the certified answer.",
                gold_answer="1",
                metadata={
                    "generator_family": family,
                    "study_partition": "qualification",
                    "excluded_from_monitor_data": True,
                },
            )
            questions.append(question)
            for sample_index in range(3):
                correct = all_correct or sample_index == 0
                parsed_answer = "1" if correct else "2"
                rollout_id = f"r-{family_index}-{question_index}-{sample_index}"
                rollouts.append(
                    Rollout(
                        rollout_id=rollout_id,
                        question_id=question_id,
                        task_family="causal_error_math",
                        condition=Condition.CLEAN,
                        hinted_option=None,
                        hint_template=None,
                        prompt=question.prompt,
                        response=f"reasoning Boxed {parsed_answer}",
                        reasoning="short reasoning",
                        final_response=f"\\boxed{{{parsed_answer}}}",
                        parsed_answer=parsed_answer,
                        gold_answer="1",
                        seed=family_index * 10000 + question_index * 100 + sample_index,
                        model="openai/gpt-oss-20b",
                        generation={"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 8192},
                        finish_reason="stop",
                        created_at="2026-08-30T00:00:00+00:00",
                        provider_request_id=f"provider-{rollout_id}",
                        provider_model="openai/gpt-oss-20b",
                        status=RolloutStatus.CLEAN_STOP,
                    )
                )
    return questions, rollouts


def test_qualification_gate_passes_a_balanced_mixed_outcome_cohort() -> None:
    questions, rollouts = _qualification_fixture()
    report = analyze_causal_error_qualification(
        questions, rollouts, bootstrap_samples=200, bootstrap_seed=99
    )
    assert report["qualification_gate_passed"] is True
    assert report["ordinary_error_count"] == 48
    assert report["accuracy"] == 1 / 3
    assert Counter(report["frozen_family_counts"]) == Counter(
        {family: 6 for family in FAMILIES}
    )
    assert report["monitor_training_authorized"] is False


def test_qualification_gate_rejects_an_accuracy_ceiling() -> None:
    questions, rollouts = _qualification_fixture(all_correct=True)
    report = analyze_causal_error_qualification(
        questions, rollouts, bootstrap_samples=50, bootstrap_seed=100
    )
    assert report["qualification_gate_passed"] is False
    assert report["gate_checks"]["accuracy_between_20_and_80_percent"] is False
    assert report["gate_checks"]["at_least_24_ordinary_errors"] is False


def _confirmatory_fixture(*, propagate_target: bool = True):
    questions: list[MathProblem] = []
    rollouts: list[Rollout] = []
    split_plan = ["train"] * 43 + ["validation"] * 14 + ["test"] * 15
    for question_index in range(72):
        family = FAMILIES[question_index % len(FAMILIES)]
        question_id = f"confirm-{family}-{question_index}"
        question = MathProblem(
            question_id=question_id,
            prompt="Continue the certified computation.",
            gold_answer="1",
            metadata={
                "generator_family": family,
                "study_partition": "confirmatory",
                "excluded_from_monitor_data": False,
                "monitor_split": split_plan[question_index],
                "intervention_target_answer": "2",
            },
        )
        questions.append(question)
        for condition_index, condition in enumerate(
            (
                Condition.CLEAN,
                Condition.CORRECT_CONTINUATION,
                Condition.CORRUPTED_CONTINUATION,
            )
        ):
            for sample_index in range(3):
                if condition is Condition.CLEAN and sample_index == 0:
                    parsed_answer = "3"
                elif (
                    propagate_target
                    and condition is Condition.CORRUPTED_CONTINUATION
                    and sample_index == 0
                ):
                    parsed_answer = "2"
                else:
                    parsed_answer = "1"
                rollout_id = f"cr-{question_index}-{condition_index}-{sample_index}"
                rollouts.append(
                    Rollout(
                        rollout_id=rollout_id,
                        question_id=question_id,
                        task_family="causal_error_math",
                        condition=condition,
                        hinted_option=(
                            "2" if condition is Condition.CORRUPTED_CONTINUATION else None
                        ),
                        hint_template=None,
                        prompt=question.prompt,
                        response=f"reasoning Boxed {parsed_answer}",
                        reasoning="short reasoning",
                        final_response=f"\\boxed{{{parsed_answer}}}",
                        parsed_answer=parsed_answer,
                        gold_answer="1",
                        seed=question_index * 1000 + condition_index * 100 + sample_index,
                        model="openai/gpt-oss-20b",
                        generation={
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "max_new_tokens": 8192,
                        },
                        finish_reason="stop",
                        created_at="2026-08-30T00:00:00+00:00",
                        provider_request_id=f"provider-{rollout_id}",
                        provider_model="openai/gpt-oss-20b",
                        status=RolloutStatus.CLEAN_STOP,
                    )
                )
    return questions, rollouts


def test_confirmatory_gate_passes_causal_effect_and_clean_negative_yield() -> None:
    questions, rollouts = _confirmatory_fixture()
    report = analyze_causal_error_confirmatory(
        questions, rollouts, bootstrap_samples=200, bootstrap_seed=101
    )
    assert report["confirmatory_causal_gate_passed"] is True
    assert report["monitor_training_authorized"] is True
    assert report["conditions"]["clean"]["incorrect"] == 72
    assert report["conditions"]["corrupted_continuation"]["target_selections"] == 72
    assert report["corrupted_minus_clean_target_effect"] == 1 / 3


def test_confirmatory_gate_rejects_missing_causal_target_propagation() -> None:
    questions, rollouts = _confirmatory_fixture(propagate_target=False)
    report = analyze_causal_error_confirmatory(
        questions, rollouts, bootstrap_samples=50, bootstrap_seed=102
    )
    assert report["confirmatory_causal_gate_passed"] is False
    assert report["monitor_training_authorized"] is False
    assert (
        report["gate_checks"][
            "corrupted_minus_clean_target_effect_at_least_20_points"
        ]
        is False
    )
