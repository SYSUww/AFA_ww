from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from afa_agent.b_board.scoring import BBoardScore, score_submission


TOTAL_SCORE_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class PromotionSnapshot:
    """Offline loop metrics; never interpreted as official leaderboard scores."""

    accuracy_score: float
    reasoning_score: float
    token_total: int
    retry_count: int = 0
    failure_count: int = 0
    audit_issue_count: int = 0
    stability_score: float | None = None
    complete: bool = True
    unobservable_usage_risk: bool = False

    def __post_init__(self) -> None:
        for field_name in ("accuracy_score", "reasoning_score"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric in 0..100")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 100.0:
                raise ValueError(f"{field_name} must be numeric in 0..100")
        for field_name in (
            "token_total",
            "retry_count",
            "failure_count",
            "audit_issue_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{field_name} must be a non-negative integer"
                )
        if self.stability_score is not None:
            value = self.stability_score
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("stability_score must be numeric in 0..100")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 100.0:
                raise ValueError("stability_score must be numeric in 0..100")
        if not isinstance(self.complete, bool):
            raise ValueError("complete must be boolean")
        if not isinstance(self.unobservable_usage_risk, bool):
            raise ValueError("unobservable_usage_risk must be boolean")

    def score(self) -> BBoardScore:
        return score_submission(
            accuracy_score=self.accuracy_score,
            reasoning_scores=[self.reasoning_score],
            token_total=self.token_total,
        )


@dataclass(frozen=True, slots=True)
class TotalScorePromotionDecision:
    promote: bool
    baseline_score: BBoardScore
    candidate_score: BBoardScore
    total_score_delta: float
    component_deltas: dict[str, float | int]
    tie_breaker_improvements: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "baseline_score": self.baseline_score.to_dict(),
            "candidate_score": self.candidate_score.to_dict(),
            "total_score_delta": self.total_score_delta,
            "component_deltas": self.component_deltas,
            "tie_breaker_improvements": list(self.tie_breaker_improvements),
            "reasons": list(self.reasons),
        }


def decide_total_score_promotion(
    *,
    baseline: PromotionSnapshot,
    candidate: PromotionSnapshot,
    epsilon: float = TOTAL_SCORE_EPSILON,
) -> TotalScorePromotionDecision:
    """Apply the agreed proxy promotion gate: weighted total must not decline."""

    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or epsilon < 0
    ):
        raise ValueError("epsilon must be finite and non-negative")
    baseline_score = baseline.score()
    candidate_score = candidate.score()
    delta = candidate_score.total_score - baseline_score.total_score
    reasons: list[str] = []
    if not candidate.complete:
        reasons.append("candidate_incomplete")
    if candidate.unobservable_usage_risk:
        reasons.append("unobservable_usage_risk")
    if delta < -epsilon:
        reasons.append("weighted_total_regressed")

    tie_breakers = _tie_breaker_improvements(baseline, candidate)
    if abs(delta) <= epsilon and not tie_breakers:
        reasons.append("no_total_or_operational_gain")

    return TotalScorePromotionDecision(
        promote=not reasons,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        total_score_delta=delta,
        component_deltas={
            "accuracy_score": (
                candidate_score.accuracy_score - baseline_score.accuracy_score
            ),
            "reasoning_score": (
                candidate_score.reasoning_score - baseline_score.reasoning_score
            ),
            "token_efficiency_score": (
                candidate_score.token_efficiency_score
                - baseline_score.token_efficiency_score
            ),
            "token_total": candidate.token_total - baseline.token_total,
        },
        tie_breaker_improvements=tie_breakers,
        reasons=tuple(sorted(set(reasons))),
    )


def _tie_breaker_improvements(
    baseline: PromotionSnapshot,
    candidate: PromotionSnapshot,
) -> tuple[str, ...]:
    improvements: list[str] = []
    for field in ("retry_count", "failure_count", "audit_issue_count"):
        if getattr(candidate, field) < getattr(baseline, field):
            improvements.append(field)
    if (
        baseline.stability_score is not None
        and candidate.stability_score is not None
        and candidate.stability_score > baseline.stability_score
    ):
        improvements.append("stability_score")
    return tuple(improvements)
