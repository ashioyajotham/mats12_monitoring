from collections import Counter

from src.causal_error_dataset import FAMILIES
from src.causal_error_gates import analyze_causal_error_qualification
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
