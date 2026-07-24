#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.total_score_promotion import (  # noqa: E402
    PromotionSnapshot,
    decide_total_score_promotion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the offline B-board weighted-total non-regression gate"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _load_snapshot(args.baseline)
    candidate = _load_snapshot(args.candidate)
    decision = decide_total_score_promotion(
        baseline=baseline,
        candidate=candidate,
    )
    payload = {
        "score_type": "offline_proxy_not_official_leaderboard_score",
        "formula": "accuracy*0.5 + reasoning*0.3 + token_efficiency*0.2",
        "gate": "candidate_total_score >= baseline_total_score",
        "decision": decision.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_snapshot(path: Path) -> PromotionSnapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: snapshot must be a JSON object")
    allowed = {
        "accuracy_score",
        "reasoning_score",
        "token_total",
        "retry_count",
        "failure_count",
        "audit_issue_count",
        "stability_score",
        "complete",
        "unobservable_usage_risk",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{path}: unsupported fields: {unknown}")
    return PromotionSnapshot(**_coerce_snapshot(payload))


def _coerce_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("accuracy_score", "reasoning_score", "token_total")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"snapshot is missing required fields: {missing}")
    return dict(payload)


if __name__ == "__main__":
    main()
