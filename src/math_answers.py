"""Conservative extraction and canonicalization for competition-math answers."""

from __future__ import annotations

import re
from fractions import Fraction

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\s*:\s*(.+)", re.IGNORECASE)
_FRAC = re.compile(r"^\\frac\{([^{}]+)\}\{([^{}]+)\}$")
_COMPACT_FRAC = re.compile(r"^\\frac\s*([+-]?\d+)\s*([+-]?\d+)$")


def _strip_wrappers(value: str) -> str:
    value = value.strip().strip("$ ")
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
    """Extract the last boxed/final answer and canonicalize it conservatively."""
    boxed = _BOXED.findall(text)
    candidates = boxed or [match.group(1).split("\n", 1)[0] for match in _FINAL.finditer(text)]
    if not candidates:
        return None
    normalized = [normalize_math_answer(item) for item in candidates]
    normalized = [item for item in normalized if item is not None]
    if not normalized or len(set(normalized)) > 1:
        return None
    return normalized[-1]
