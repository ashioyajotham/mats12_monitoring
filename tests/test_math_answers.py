"""Contract tests for conservative free-response math parsing."""

from src.math_answers import (
    MathGrade,
    extract_math_answer,
    grade_math_answer,
    normalize_math_answer,
)


def test_compact_and_braced_fractions_share_a_canonical_form() -> None:
    """Normalize common MATH answer-key fraction spellings identically."""
    assert normalize_math_answer(r"\frac53") == "5/3"
    assert normalize_math_answer(r"\frac{5}{3}") == "5/3"


def test_conflicting_boxed_answers_are_rejected() -> None:
    """Avoid silently accepting a transcript with multiple different finals."""
    assert extract_math_answer(r"First \boxed{2}, then \boxed{3}") is None


def test_nested_boxed_fraction_is_extracted() -> None:
    """Handle the standard ``\\boxed{\\frac{a}{b}}`` rendering."""
    assert extract_math_answer(r"Therefore, \boxed{\frac{5}{3}}") == "5/3"


def test_math_grader_handles_numeric_and_equivalent_equations() -> None:
    assert grade_math_answer("0.5", "1/2") is MathGrade.CORRECT
    assert grade_math_answer(r"\dfrac{5}{3}", "5/3") is MathGrade.CORRECT
    assert grade_math_answer("(-15,0)", "(-15, 0)") is MathGrade.CORRECT
    assert grade_math_answer("2*x + 2*y = 2", "x + y = 1") is MathGrade.CORRECT
    assert grade_math_answer("3", "4") is MathGrade.INCORRECT


def test_math_grader_defers_unsupported_latex() -> None:
    assert grade_math_answer(r"\sin x", "0") is MathGrade.REVIEW
