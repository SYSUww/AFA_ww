from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Iterable


ACCURACY_WEIGHT = 0.6
REASONING_WEIGHT = 0.2
TOKEN_EFFICIENCY_WEIGHT = 0.2

_ALLOWED_MODEL_RE = re.compile(
    r"(?:^|[/_-])qwen[-_]?3[._-]?[56](?:$|[-_/.:])",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class BBoardScore:
    accuracy_score: float
    reasoning_score: float
    token_efficiency_score: float
    total_score: float
    token_total: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def token_efficiency_score(token_total: int) -> float:
    """Return the official 0-100 token score from the July 2026 rules."""

    if isinstance(token_total, bool) or not isinstance(token_total, int) or token_total < 0:
        raise ValueError("token_total must be a non-negative integer")
    if token_total == 0:
        return 0.0
    if token_total < 500_000:
        return token_total / 500_000 * 100.0
    if token_total <= 5_000_000:
        return 100.0
    if token_total <= 10_000_000:
        return 100.0 * (1.0 - (token_total - 5_000_000) / 5_000_000)
    return 0.0


def score_submission(
    *,
    accuracy_score: float,
    reasoning_scores: Iterable[float],
    token_total: int,
) -> BBoardScore:
    accuracy = _score_value(accuracy_score, "accuracy_score")
    reasoning_values = [
        _score_value(value, f"reasoning_scores[{index}]")
        for index, value in enumerate(reasoning_scores)
    ]
    reasoning = fmean(reasoning_values) if reasoning_values else 0.0
    token_score = token_efficiency_score(token_total)
    total = (
        accuracy * ACCURACY_WEIGHT
        + reasoning * REASONING_WEIGHT
        + token_score * TOKEN_EFFICIENCY_WEIGHT
    )
    return BBoardScore(
        accuracy_score=accuracy,
        reasoning_score=reasoning,
        token_efficiency_score=token_score,
        total_score=total,
        token_total=token_total,
    )


def is_allowed_submission_model(model_name: str) -> bool:
    """Whether a configured model belongs to the stated Qwen3.5/3.6 families."""

    return bool(_ALLOWED_MODEL_RE.search(str(model_name).strip()))


def require_allowed_submission_model(model_name: str) -> None:
    if not is_allowed_submission_model(model_name):
        raise ValueError(
            "B-board submission generation requires a Qwen3.5/Qwen3.6 model; "
            f"configured model is {model_name!r}"
        )


def _score_value(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number in 0..100")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in 0..100") from exc
    if not 0.0 <= number <= 100.0:
        raise ValueError(f"{name} must be a number in 0..100")
    return number
