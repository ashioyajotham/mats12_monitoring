"""Conservative extraction and canonicalization for competition-math answers."""

from __future__ import annotations

import re
from enum import StrEnum
from fractions import Fraction

import sympy

_FINAL = re.compile(r"(?:final answer|answer)\s*:\s*(.+)", re.IGNORECASE)
_DISPLAY_MATH = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_FRAC = re.compile(r"^\\frac\{([^{}]+)\}\{([^{}]+)\}$")
_COMPACT_FRAC = re.compile(r"^\\frac\s*([+-]?\d+)\s*([+-]?\d+)$")
_NUMERIC_FINAL = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+|\d+/\d+|\\(?:d?frac)\{[+-]?\d+\}\{[+-]?\d+\})$"
)
_SAFE_SYMBOLIC = re.compile(r"^[A-Za-z0-9+*/^=()., \-]+$")


class MathGrade(StrEnum):
    """Conservative mathematical equivalence outcome."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    REVIEW = "review"


def _strip_wrappers(value: str) -> str:
    value = value.strip().strip("$ ")
    value = value.replace("\\dfrac", "\\frac")
    value = value.replace("\\,", "").replace("\\!", "")
    value = re.sub(r"^\\(?:text|mathrm)\{(.+)\}$", r"\1", value)
    return value.strip()


def normalize_math_answer(value: str | None) -> str | None:
    """Return a stable exact-match representation, or ``None`` if ambiguous."""
    if value is None:
        return None
    value = _strip_wrappers(value.split("\n", 1)[0].strip().rstrip("."))
    frac = _FRAC.fullmatch(value)
    compact_frac = _COMPACT_FRAC.fullmatch(value)
    if compact_frac:
        frac = compact_frac
    if frac:
        try:
            return str(Fraction(int(frac.group(1)), int(frac.group(2))))
        except (ValueError, ZeroDivisionError):
            return None
    compact = value.replace(" ", "")
    if re.fullmatch(r"[-+]?\d+(?:/\d+)?", compact):
        try:
            return str(Fraction(compact))
        except (ValueError, ZeroDivisionError):
            return None
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", compact):
        try:
            return str(Fraction(compact))
        except (ValueError, ZeroDivisionError):
            return None
    if not value or len(value) > 200:
        return None
    return re.sub(r"\s+", " ", value).strip().casefold()


def extract_math_answer(text: str) -> str | None:
    """Extract a boxed, labelled, or final display-math numeric answer conservatively."""
    boxed: list[str] = []
    marker = "\\boxed{"
    start = 0
    while (position := text.find(marker, start)) >= 0:
        depth = 1
        index = position + len(marker)
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            boxed.append(text[position + len(marker) : index - 1])
        start = position + len(marker)
    labelled = [match.group(1).split("\n", 1)[0] for match in _FINAL.finditer(text)]
    candidates = boxed or labelled
    if not candidates:
        displays = [match.group(1).strip() for match in _DISPLAY_MATH.finditer(text)]
        if displays:
            final_display = displays[-1]
            if final_display.count("=") == 1:
                left, right = final_display.split("=", 1)
                if left.strip() and _NUMERIC_FINAL.fullmatch(right.replace(" ", "")):
                    candidates = [right]
            elif _NUMERIC_FINAL.fullmatch(final_display.replace(" ", "")):
                candidates = [final_display]
    if not candidates:
        return None
    normalized = [normalize_math_answer(item) for item in candidates]
    normalized = [item for item in normalized if item is not None]
    if not normalized or len(set(normalized)) > 1:
        return None
    return normalized[-1]


def _safe_expression(value: str) -> sympy.Expr | None:
    """Parse a deliberately small symbolic grammar without implicit Python names."""
    value = value.replace("^", "**")
    if not _SAFE_SYMBOLIC.fullmatch(value):
        return None
    names = set(re.findall(r"[A-Za-z]+", value))
    local_dict = {name: sympy.Symbol(name) for name in names}
    try:
        return sympy.sympify(value, locals=local_dict, evaluate=True)
    except (TypeError, ValueError, SyntaxError, sympy.SympifyError):
        return None


def _equation_expression(value: str) -> sympy.Expr | None:
    """Convert an expression or single equality to a zero-valued expression."""
    if value.count("=") > 1:
        return None
    if "=" in value:
        left, right = value.split("=", 1)
        left_expr, right_expr = _safe_expression(left), _safe_expression(right)
        return left_expr - right_expr if left_expr is not None and right_expr is not None else None
    return _safe_expression(value)


def grade_math_answer(predicted: str | None, gold: str) -> MathGrade:
    """Grade exact, numeric, and restricted symbolic equivalence; otherwise defer."""
    predicted_normalized = normalize_math_answer(predicted)
    gold_normalized = normalize_math_answer(gold)
    if predicted_normalized is None or gold_normalized is None:
        return MathGrade.REVIEW
    if predicted_normalized == gold_normalized or re.sub(
        r"\s+", "", predicted_normalized
    ) == re.sub(r"\s+", "", gold_normalized):
        return MathGrade.CORRECT
    predicted_expr = _equation_expression(predicted_normalized)
    gold_expr = _equation_expression(gold_normalized)
    if predicted_expr is None or gold_expr is None:
        return MathGrade.REVIEW
    try:
        difference = sympy.simplify(predicted_expr - gold_expr)
        if difference == 0:
            return MathGrade.CORRECT
        if "=" in predicted_normalized and "=" in gold_normalized and gold_expr != 0:
            ratio = sympy.simplify(predicted_expr / gold_expr)
            if not ratio.free_symbols and ratio != 0:
                return MathGrade.CORRECT
        if not (predicted_expr.free_symbols | gold_expr.free_symbols):
            return MathGrade.INCORRECT
    except (TypeError, ValueError, ZeroDivisionError):
        return MathGrade.REVIEW
    return MathGrade.INCORRECT
