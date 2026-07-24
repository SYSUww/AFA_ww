#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.reasoning_evaluation import (  # noqa: E402
    MIN_REASONING_NON_WHITESPACE,
    REASONING_JUDGE_MODEL,
    FixedReasoningEvaluator,
    ReasoningEvaluation,
    build_reasoning_sentinels,
    validate_reasoning_sentinels,
)
from afa_agent.client import OpenAICompatibleClient  # noqa: E402
from afa_agent.config import build_model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the fixed GPT-5.6 reasoning shadow judge"
    )
    parser.add_argument(
        "--env-root",
        type=Path,
        default=Path("/Users/abandon/Documents/AFA_ww"),
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    base = build_model_config(args.env_root, env_prefix="LLM")
    if base is None:
        raise RuntimeError("LLM judge model configuration is missing")
    config = replace(
        base,
        model_name=REASONING_JUDGE_MODEL,
        temperature=0.0,
    )
    sentinels = build_reasoning_sentinels()

    def evaluate(item: dict[str, object]) -> dict[str, object]:
        sentinel_id = str(item["sentinel_id"])
        reasoning = str(item["reasoning"])
        if len("".join(reasoning.split())) < MIN_REASONING_NON_WHITESPACE:
            evaluation = ReasoningEvaluation(
                qid=sentinel_id,
                logical=0.0,
                completeness=0.0,
                clarity=0.0,
                status="below_minimum_length",
            )
            return {
                "evaluation": evaluation,
                "usage": _zero_usage(),
                "calls": [],
                "failure": None,
            }
        try:
            outcome = FixedReasoningEvaluator(
                OpenAICompatibleClient(config)
            ).evaluate(reasoning)
            evaluation = ReasoningEvaluation(
                qid=sentinel_id,
                logical=outcome.final_dimensions["logical"],
                completeness=outcome.final_dimensions["completeness"],
                clarity=outcome.final_dimensions["clarity"],
                rubric_scores=outcome.rubric_scores,
                auditor_scores=outcome.auditor_scores,
                hard_cap_violations=outcome.violations,
                hard_caps=outcome.hard_caps,
            )
            return {
                "evaluation": evaluation,
                "usage": outcome.total_usage,
                "calls": list(outcome.calls),
                "failure": None,
            }
        except Exception as exc:
            evaluation = ReasoningEvaluation(
                qid=sentinel_id,
                logical=0.0,
                completeness=0.0,
                clarity=0.0,
                status="judge_error",
            )
            return {
                "evaluation": evaluation,
                "usage": dict(getattr(exc, "token_usage", _zero_usage())),
                "calls": list(getattr(exc, "calls", ())),
                "failure": {
                    "error_type": type(exc).__name__,
                    "error": _sanitize_error(str(exc), config),
                },
            }

    results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate, dict(item)): str(item["sentinel_id"])
            for item in sentinels
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    evaluations = {
        sentinel_id: result["evaluation"]
        for sentinel_id, result in results.items()
    }
    calibration = validate_reasoning_sentinels(evaluations)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": config.model_name,
        "temperature": config.temperature,
        "workers": args.workers,
        "calibration": calibration,
        "sentinels": [
            {
                "sentinel_id": str(item["sentinel_id"]),
                "target_band": int(item["target"]),
                "reasoning": str(item["reasoning"]),
                "evaluation": evaluations[str(item["sentinel_id"])].to_dict(),
                "usage": results[str(item["sentinel_id"])]["usage"],
                "calls": results[str(item["sentinel_id"])]["calls"],
                "failure": results[str(item["sentinel_id"])]["failure"],
            }
            for item in sentinels
        ],
        "judge_usage": _sum_usage(
            [
                dict(result["usage"])
                for result in results.values()
            ]
        ),
        "included_in_submission_token_usage": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not calibration["passed"]:
        raise SystemExit(2)


def _zero_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _sanitize_error(message: str, config: object) -> str:
    sanitized = message
    for secret in (
        str(getattr(config, "api_key", "") or ""),
        str(getattr(config, "api_base", "") or ""),
    ):
        if secret:
            sanitized = sanitized.replace(secret, "[redacted]")
    return sanitized


def _sum_usage(items: list[dict[str, int]]) -> dict[str, int]:
    return {
        key: sum(int(item.get(key, 0)) for item in items)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


if __name__ == "__main__":
    main()
