from collections import deque

import pytest
from pydantic import ValidationError

from src.generate_rollouts import GenerationResult
from src.judge_runner import JudgeScoreRecord, judge_plan, score_judge_plan
from src.monitor_dataset import ContextAwareEvidence, MonitorExample, TranscriptOnlyEvidence
from src.monitors.llm_judge import (
    JudgeBackendError,
    TinkerQwenJudge,
    parse_judge_payload,
)


class FakeBackend:
    def __init__(self, results: list[GenerationResult]):
        self.results = deque(results)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return self.results.popleft()


def _result(text: str, request_id: str, *, clean: bool = True) -> GenerationResult:
    return GenerationResult(
        text=text,
        finish_reason="stop" if clean else "length",
        termination_clean=clean,
        provider_request_id=request_id,
        usage={"total_tokens": 12},
    )


def _example() -> MonitorExample:
    transcript = TranscriptOnlyEvidence(question="q", reasoning="r", final_answer="2")
    return MonitorExample(
        example_id="e1",
        rollout_id="e1",
        question_id="q1",
        split="train",
        binary_label=1,
        kind="causally_induced_error",
        family="f",
        tier="t",
        renderer_id=0,
        transcript=transcript,
        context=ContextAwareEvidence(
            **transcript.model_dump(), rendered_prompt="state and q"
        ),
    )


def test_judge_payload_is_strict_but_accepts_one_json_fence() -> None:
    payload = parse_judge_payload(
        '```json\n{"causal_error_probability":0.75,"rationale":"state propagation"}\n```'
    )
    assert payload.causal_error_probability == 0.75
    with pytest.raises(ValidationError, match="extra"):
        parse_judge_payload(
            '{"causal_error_probability":0.5,"rationale":"x","condition":"corrupt"}'
        )
    verbose = parse_judge_payload(
        '{"causal_error_probability":0.5,"rationale":"' + "x" * 4000 + '"}'
    )
    assert len(verbose.rationale) == 4000
    with pytest.raises(ValidationError, match="at most 8000"):
        parse_judge_payload(
            '{"causal_error_probability":0.5,"rationale":"' + "x" * 8001 + '"}'
        )


def test_tinker_judge_retries_truncation_and_malformed_json() -> None:
    backend = FakeBackend(
        [
            _result("", "p1", clean=False),
            _result("not json", "p2"),
            _result(
                '{"causal_error_probability":0.6,"rationale":"matched state"}', "p3"
            ),
        ]
    )
    result = TinkerQwenJudge(backend).score(
        "e1", TranscriptOnlyEvidence(question="q", reasoning="r", final_answer="2"), seed=10
    )
    assert result.attempts == 3
    assert result.provider_request_id == "p3"
    assert [request.seed for request in backend.requests] == [10, 11, 12]
    assert "previous response was invalid" in backend.requests[-1].prompt


def test_tinker_judge_fails_after_bounded_retries() -> None:
    backend = FakeBackend([_result("bad", f"p{i}") for i in range(3)])
    with pytest.raises(JudgeBackendError, match="attempt 3"):
        TinkerQwenJudge(backend).score(
            "e1", TranscriptOnlyEvidence(question="q", reasoning="r", final_answer="2"), seed=1
        )


def test_judge_plan_scores_both_views_and_resumes_by_score_identity() -> None:
    example = _example()
    plan = judge_plan(
        [example], model="Qwen/Qwen3.6-35B-A3B", prompt_version="causal_error_judge_v1",
        base_seed=100,
    )
    backend = FakeBackend(
        [
            _result('{"causal_error_probability":0.2,"rationale":"ordinary"}', "p1"),
            _result('{"causal_error_probability":0.8,"rationale":"state"}', "p2"),
        ]
    )
    judge = TinkerQwenJudge(backend)
    records, failures = score_judge_plan(plan, judge)
    assert not failures
    assert {row.view for row in records} == {"transcript_only", "context_aware"}
    resumed, failures = score_judge_plan(plan, judge, existing=records)
    assert resumed == records
    assert not failures
    assert len(backend.requests) == 2


def test_judge_resume_rejects_scores_outside_frozen_plan() -> None:
    example = _example()
    plan = judge_plan([example], model="judge", prompt_version="v1", base_seed=100)
    foreign = JudgeScoreRecord(
        score_id="foreign",
        example_id="other",
        question_id="other",
        split="train",
        view="transcript_only",
        score=0.5,
        rationale="fixture",
        model="judge",
        prompt_version="v1",
        seed=1,
        attempts=1,
        provider_request_id="p-foreign",
        finish_reason="stop",
        usage={},
    )
    with pytest.raises(ValueError, match="outside the frozen plan"):
        score_judge_plan(plan, TinkerQwenJudge(FakeBackend([])), existing=[foreign])
