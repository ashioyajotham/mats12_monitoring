"""Contract tests for conservative free-response math parsing."""

from src.math_answers import extract_math_answer, normalize_math_answer


def test_compact_and_braced_fractions_share_a_canonical_form() -> None:
    """Normalize common MATH answer-key fraction spellings identically."""
    assert normalize_math_answer(r"\frac53") == "5/3"
    assert normalize_math_answer(r"\frac{5}{3}") == "5/3"


def test_conflicting_boxed_answers_are_rejected() -> None:
    """Avoid silently accepting a transcript with multiple different finals."""
    assert extract_math_answer(r"First \boxed{2}, then \boxed{3}") is None
