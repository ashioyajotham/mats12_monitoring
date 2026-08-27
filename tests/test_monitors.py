import numpy as np
import pytest

from src.monitors.activation_probe import ActivationProbe, shuffled_control
from src.monitors.hybrid import HybridMonitor
from src.monitors.llm_judge import JudgeRequest, JudgeView, build_judge_prompt
from src.monitors.surface import SurfaceFeatureMonitor, extract_surface_features


def test_judge_prompt_respects_information_boundary():
    transcript = JudgeRequest(
        question="Question", reasoning="Reasoning", final_answer="A", view=JudgeView.TRANSCRIPT_ONLY
    )
    aware = transcript.model_copy(
        update={"view": JudgeView.CONTEXT_AWARE, "hint_text": "External hint says B"}
    )
    assert "External hint says B" not in build_judge_prompt(transcript)
    assert "External hint says B" in build_judge_prompt(aware)


def test_surface_features_and_monitor_produce_scores():
    features = [
        extract_surface_features("Maybe A", "A"),
        extract_surface_features("Definitely B", "B"),
        extract_surface_features("Wait, perhaps A", "A"),
        extract_surface_features("Clearly B", "B"),
    ]
    scores = SurfaceFeatureMonitor().fit(features, [1, 0, 1, 0]).predict_score(features)
    assert scores.shape == (4,)
    assert np.all((0 <= scores) & (scores <= 1))


def test_activation_and_hybrid_monitors_validate_matrix_shape():
    with pytest.raises(ValueError, match="hidden_size"):
        ActivationProbe().fit(np.array([1.0, 2.0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="monitors"):
        HybridMonitor().fit(np.array([0.1, 0.9]), np.array([0, 1]))


def test_activation_probe_and_shuffle_control_are_reproducible():
    activations = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 0.9], [0.8, 1.0]])
    labels = np.array([0, 0, 1, 1])
    scores = ActivationProbe().fit(activations, labels).predict_score(activations)
    assert scores.shape == (4,)
    assert np.array_equal(shuffled_control(labels, 4), shuffled_control(labels, 4))
