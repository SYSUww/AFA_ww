#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import load_b_questions
from afa_agent.b_board.runner import (
    CALCULATION_PROVIDER_FORMATS,
    DEFAULT_INDEX_ROOT,
    DEFAULT_PARSED_ROOT,
    DEFAULT_STRATEGY_PATH,
    RUN_MODES,
    RUN_MODE_SUBMISSION,
    RUN_STAGES,
    RUN_STAGE_FULL,
    BBoardActualRunner,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real 100-question B-board answering chain")
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT.relative_to(ROOT)))
    parser.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT.relative_to(ROOT)))
    parser.add_argument("--strategy-config", default=str(DEFAULT_STRATEGY_PATH.relative_to(ROOT)))
    parser.add_argument("--locator-attempt", default="attempt_43")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qid-file", default="")
    parser.add_argument(
        "--answer-format",
        action="append",
        default=[],
        help=(
            "Run every question whose normalized answer_format matches this "
            "value. May be repeated and composes with --qid/--qid-file."
        ),
    )
    parser.add_argument("--run-dir", default="")
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default=RUN_MODE_SUBMISSION,
        help="submission enforces the official model allowlist; research writes only research_submit.csv",
    )
    parser.add_argument(
        "--stage",
        choices=RUN_STAGES,
        default=RUN_STAGE_FULL,
        help="answer stops after frozen answer checkpoints; full also generates reasoning and CSV",
    )
    parser.add_argument(
        "--calculation-profile",
        action="store_true",
        help=(
            "Research-only: ask the allowed model for a generic calculation "
            "profile before retrieval. Disabled by default because the current "
            "profile direction did not pass the full-26 promotion gate."
        ),
    )
    parser.add_argument(
        "--calculation-modular-prompt",
        action="store_true",
        help=(
            "Research-only: use the shorter domain-specific calculation "
            "prompt. Disabled by default because its Top-12 A1 did not pass."
        ),
    )
    parser.add_argument(
        "--calculation-thinking-budget",
        type=_positive_int,
        default=None,
        help=(
            "Research-only Qwen thinking budget for each calculation-plan "
            "call. Omit to preserve the provider default."
        ),
    )
    parser.add_argument(
        "--calculation-thinking-mode",
        choices=("default", "on", "off"),
        default="default",
        help=(
            "Research-only Qwen thinking mode for calculation-plan calls. "
            "default preserves provider behavior."
        ),
    )
    parser.add_argument(
        "--calculation-provider-format",
        choices=CALCULATION_PROVIDER_FORMATS,
        default=CALCULATION_PROVIDER_FORMATS[0],
        help=(
            "Research control for provider-side calculation output: native "
            "strict JSON Schema or JSON object with full local validation."
        ),
    )
    parser.add_argument(
        "--calculation-repair-retry",
        action="store_true",
        help=(
            "Research-only: repair same-evidence structural failures from "
            "the previous JSON with thinking disabled."
        ),
    )
    parser.add_argument(
        "--calculation-structured-retrieval",
        action="store_true",
        help=(
            "Research-only: enable document/period/lexical first-pass "
            "retrieval and typed retry retrieval. Disabled by default "
            "because the Top-12 direction did not pass."
        ),
    )
    parser.add_argument(
        "--calculation-unit-repair",
        action="store_true",
        help=(
            "Research-only: insert deterministic currency conversions for "
            "same-dimension add/sub inputs before local replay."
        ),
    )
    parser.add_argument(
        "--calculation-guarded-adaptive-thinking",
        action="store_true",
        help=(
            "Research-only bundle: use an answer-blind structural policy "
            "for the first calculation attempt, inject runtime evidence "
            "semantic constraints, and return retries to provider default "
            "thinking."
        ),
    )
    parser.add_argument(
        "--calculation-joint-reasoning",
        action="store_true",
        help=(
            "Research-only: require the calculation-plan decision_summary "
            "to be submission-ready reasoning and reuse it with zero "
            "incremental reasoning model calls."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_b_questions(ROOT / args.question_root, ROOT / args.submission_template)
    qids = list(args.qid)
    if args.qid_file:
        qids.extend(
            line.strip()
            for line in (ROOT / args.qid_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    requested_formats = {
        str(item).strip() for item in args.answer_format if str(item).strip()
    }
    if requested_formats:
        known_formats = {item.answer_format for item in questions}
        unknown_formats = sorted(requested_formats - known_formats)
        if unknown_formats:
            raise ValueError(
                f"Unknown B answer formats: {unknown_formats}"
            )
        qids.extend(
            item.qid
            for item in questions
            if item.answer_format in requested_formats
        )
    qids = list(dict.fromkeys(qids)) or None
    if qids is not None:
        known = {item.qid for item in questions}
        unknown = sorted(set(qids) - known)
        if unknown:
            raise ValueError(f"Unknown B qids: {unknown}")
    run_dir = (
        (ROOT / args.run_dir)
        if args.run_dir
        else ROOT
        / "artifacts"
        / "b_board_actual"
        / f"b0_attempt43_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    runner = BBoardActualRunner(
        questions=questions,
        parsed_root=ROOT / args.parsed_root,
        index_root=ROOT / args.index_root,
        strategy_path=ROOT / args.strategy_config,
        locator_attempt_id=args.locator_attempt,
        run_mode=args.run_mode,
        calculation_profile_enabled=args.calculation_profile,
        calculation_modular_prompt_enabled=(
            args.calculation_modular_prompt
        ),
        calculation_structured_retrieval_enabled=(
            args.calculation_structured_retrieval
        ),
        calculation_unit_repair_enabled=args.calculation_unit_repair,
        calculation_enable_thinking=(
            None
            if args.calculation_thinking_mode == "default"
            else args.calculation_thinking_mode == "on"
        ),
        calculation_provider_format=args.calculation_provider_format,
        calculation_repair_retry_enabled=args.calculation_repair_retry,
        calculation_guarded_adaptive_thinking_enabled=(
            args.calculation_guarded_adaptive_thinking
        ),
        calculation_joint_reasoning_enabled=(
            args.calculation_joint_reasoning
        ),
        calculation_thinking_budget=(
            args.calculation_thinking_budget
        ),
    )
    manifest = runner.run(
        run_dir=run_dir,
        qids=qids,
        workers=max(1, args.workers),
        stage=args.stage,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
