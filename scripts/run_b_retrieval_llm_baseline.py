#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping
from uuid import uuid4

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import BAnswer, load_b_questions, write_b_submission
from afa_agent.b_board.evidence_compaction import compact_retrieval_payload
from afa_agent.b_board.retrieval_llm_baseline import (
    DOCUMENT_CANDIDATE_STRATEGIES,
    EVIDENCE_QUOTA_STRATEGIES,
    PIPELINE_VERSION,
    PROMPT_VERSION,
    build_answer_messages,
    build_answer_schema,
    build_frozen_answer_reasoning_messages,
    build_frozen_answer_reasoning_schema,
    build_reasoning_canonical_schema,
    load_domain_indexes,
    make_retriever,
    is_verified_calculation_path,
    prepare_evidence_payload,
    public_run_config_fingerprint,
    retrieval_policy_version,
    retrieve_question_evidence,
    sha256_file,
    validate_answer_payload,
    validate_answer_parts_shape,
    validate_answer_shape_payload,
    validate_frozen_answer_reasoning_payload,
    validate_reasoning_canonical_payload,
)
from afa_agent.b_board.submission_policy import require_allowed_submission_model
from afa_agent.client import (
    TRANSPORT_RETRY_POLICY_VERSION,
    OpenAICompatibleClient,
)
from afa_agent.config import build_model_config
from afa_agent.io_utils import write_json as write_json_atomic
from afa_agent.models import TokenUsage


DEFAULT_SOURCE_ROOT = Path("/Users/abandon/Documents/AFA_ww")
DEFAULT_INDEX_ROOT = (
    DEFAULT_SOURCE_ROOT / "artifacts" / "preprocessed_loop_candidates" / "index"
)
DEFAULT_MODEL = "qwen3.7-plus-2026-05-26"
MAX_VERIFIED_REASONING_CALLS = 2
RESEARCH_ONLY_EVIDENCE_QUOTA_STRATEGIES = {
    "document_balanced",
    "adaptive_multi_report_calculation",
    "metric_slot_coverage",
}
PROHIBITED_GENERATION_MODULE_FRAGMENTS = (
    ".solver",
    "candidate_scorecard",
    "official_answer",
    "pseudo_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an answer-blind retrieval + single-Qwen B-board baseline"
    )
    parser.add_argument(
        "--question-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "upload_b/question_b",
    )
    parser.add_argument(
        "--submission-template",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "upload_b/submit.csv",
    )
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-query-top-k", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--max-doc-candidates", type=int, default=6)
    parser.add_argument(
        "--document-candidate-strategy",
        choices=DOCUMENT_CANDIDATE_STRATEGIES,
        default="anchor_union",
        help=(
            "anchor_union keeps anchor hits plus global document discovery; "
            "anchor_first keeps only anchor-hit documents when anchors exist "
            "and otherwise falls back to global discovery"
        ),
    )
    parser.add_argument(
        "--evidence-quota-strategy",
        choices=EVIDENCE_QUOTA_STRATEGIES,
        default="primary_guard",
        help=(
            "primary_guard reserves the global primary Top4 plus one item per "
            "document; document_balanced reserves two items per document "
            "before the remaining global ranking; metric_slot_coverage "
            "reserves one hit for each generic disclosed metric in every "
            "anchor-selected report"
        ),
    )
    parser.add_argument(
        "--allow-research-only-strategy",
        action="store_true",
        help=(
            "Explicitly permit a non-production evidence quota experiment. "
            "Such runs are permanently ineligible for submission assembly."
        ),
    )
    parser.add_argument("--supplemental-weight", type=float, default=0.11)
    parser.add_argument("--max-queries-per-option", type=int, default=12)
    parser.add_argument("--max-hit-chars", type=int, default=1800)
    parser.add_argument("--max-evidence-chars", type=int, default=12000)
    parser.add_argument(
        "--evidence-compaction",
        choices=("off", "adjacent"),
        default="off",
        help=(
            "off preserves the established ranking; adjacent retrieves a deeper "
            "pool and merges only structurally adjacent chunks into TopK blocks"
        ),
    )
    parser.add_argument("--compaction-pool-multiplier", type=int, default=3)
    parser.add_argument("--compaction-max-block-chars", type=int, default=3600)
    parser.add_argument("--compaction-max-components", type=int, default=2)
    parser.add_argument("--thinking-budget", type=int, default=2048)
    parser.add_argument("--max-format-retries", type=int, default=1)
    parser.add_argument(
        "--output-contract",
        choices=("joint", "reasoning-canonical"),
        default="joint",
        help=(
            "joint returns answer_parts plus reasoning; reasoning-canonical makes "
            "the unchanged final reasoning conclusion the sole answer source"
        ),
    )
    parser.add_argument(
        "--calculation-mode",
        choices=("direct", "verified"),
        default="direct",
        help=(
            "direct keeps calculation questions on the established single-call path; "
            "verified enables the checkpointed Qwen-plan + Decimal audited path"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    with _exclusive_run_lock(args.run_dir):
        _main_locked(args)


def _main_locked(args: argparse.Namespace) -> None:
    all_questions = load_b_questions(
        args.question_root,
        args.submission_template,
    )
    question_by_qid = {question.qid: question for question in all_questions}
    unknown = sorted(set(args.qid) - set(question_by_qid))
    if unknown:
        raise ValueError(f"unknown qids: {unknown}")
    questions = (
        [question_by_qid[qid] for qid in args.qid]
        if args.qid
        else all_questions
    )

    model_config = build_model_config(args.env_root, env_prefix="OPENAI")
    if model_config is None:
        raise RuntimeError("OPENAI model configuration is missing")
    model_config = replace(
        model_config,
        model_name=args.model,
        temperature=0.0,
    )
    require_allowed_submission_model(model_config.model_name)
    client = OpenAICompatibleClient(model_config)
    if args.calculation_mode == "verified":
        __import__("afa_agent.b_board.retrieval_llm_calculation")
    _assert_no_prohibited_generation_modules_loaded()

    index_payloads = load_domain_indexes(args.index_root)
    retrievers = {
        domain: make_retriever(domain, payload["units"])
        for domain, payload in index_payloads.items()
    }
    doc_ids = {
        domain: sorted(
            {
                str(unit["doc_id"])
                for unit in payload["units"]
                if unit.get("doc_id")
            }
        )
        for domain, payload in index_payloads.items()
    }
    public_config = _public_config(args, questions, model_config, index_payloads)
    fingerprint = public_run_config_fingerprint(public_config)
    public_config = _initialize_run_dir(
        args.run_dir,
        public_config,
        fingerprint,
    )

    existing_answers = _read_rows(args.run_dir / "answers.json")
    existing_failures = _read_rows(args.run_dir / "failures.json")
    finished_qids = _finished_qids(
        answers=existing_answers,
        failures=existing_failures,
        run_dir=args.run_dir,
        calculation_mode=args.calculation_mode,
    )
    pending = [question for question in questions if question.qid not in finished_qids]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                question,
                args=args,
                client=client,
                retriever=retrievers[question.domain],
                all_doc_ids=doc_ids[question.domain],
                domain_units=index_payloads[question.domain]["units"],
            ): question
            for question in pending
        }
        for future in as_completed(futures):
            question = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "qid": question.qid,
                    "domain": question.domain,
                    "answer_format": question.answer_format,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "calls": [],
                    "token_usage": TokenUsage().to_dict(),
                    "unobservable_usage_risk": True,
                    "preserve_existing_raw_calls": True,
                }
            results.append(result)
            _persist_result(
                args.run_dir,
                result,
                question_order=[item.qid for item in questions],
                fingerprint=fingerprint,
                public_config=public_config,
            )
            print(
                json.dumps(
                    {
                        "qid": question.qid,
                        "status": result["status"],
                        "answer_parts": result.get("answer_parts"),
                        "calls": len(result.get("calls", [])),
                        "tokens": result.get("token_usage", {}).get("total_tokens", 0),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    manifest = _write_manifest(
        args.run_dir,
        question_order=[item.qid for item in questions],
        fingerprint=fingerprint,
        public_config=public_config,
    )
    if len(questions) == len(all_questions) and manifest["failed_question_count"] == 0:
        answers_by_qid = {
            str(row["qid"]): row
            for row in _read_rows(args.run_dir / "answers.json")
        }
        submission_answers = [
            BAnswer(
                qid=question.qid,
                answer_parts=tuple(answers_by_qid[question.qid]["answer_parts"]),
                prompt_tokens=int(
                    answers_by_qid[question.qid]["token_usage"]["prompt_tokens"]
                ),
                completion_tokens=int(
                    answers_by_qid[question.qid]["token_usage"]["completion_tokens"]
                ),
                total_tokens=int(
                    answers_by_qid[question.qid]["token_usage"]["total_tokens"]
                ),
                reasoning=str(answers_by_qid[question.qid]["reasoning"]),
            )
            for question in all_questions
        ]
        write_b_submission(
            args.run_dir / "research_submit.csv",
            all_questions,
            submission_answers,
            audit_ready=True,
        )
        manifest["research_submission_path"] = str(
            args.run_dir / "research_submit.csv"
        )
        manifest["research_submission_sha256"] = sha256_file(
            args.run_dir / "research_submit.csv"
        )
        _write_json(args.run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _run_one(
    question: Any,
    *,
    args: argparse.Namespace,
    client: OpenAICompatibleClient,
    retriever: Any,
    all_doc_ids: list[str],
    domain_units: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    compaction_mode = getattr(args, "evidence_compaction", "off")
    retrieval_top_k = (
        args.final_top_k * int(getattr(args, "compaction_pool_multiplier", 3))
        if compaction_mode == "adjacent"
        else args.final_top_k
    )
    retrieval = retrieve_question_evidence(
        question,
        retriever=retriever,
        all_doc_ids=all_doc_ids,
        per_query_top_k=args.per_query_top_k,
        final_top_k=retrieval_top_k,
        supplemental_weight=args.supplemental_weight,
        max_queries_per_option=args.max_queries_per_option,
        max_doc_candidates=args.max_doc_candidates,
        document_candidate_strategy=getattr(
            args, "document_candidate_strategy", "anchor_union"
        ),
        evidence_quota_strategy=getattr(
            args, "evidence_quota_strategy", "primary_guard"
        ),
    )
    active_retrieval_policy_version = str(
        retrieval.get(
            "policy_version",
            retrieval_policy_version(
                getattr(args, "document_candidate_strategy", "anchor_union"),
                getattr(args, "evidence_quota_strategy", "primary_guard"),
            ),
        )
    )
    if compaction_mode == "adjacent":
        if domain_units is None:
            raise ValueError("adjacent evidence compaction requires domain units")
        retrieval = compact_retrieval_payload(
            retrieval,
            units=domain_units,
            top_k=args.final_top_k,
            max_block_chars=int(
                getattr(args, "compaction_max_block_chars", 3600)
            ),
            max_components_per_block=int(
                getattr(args, "compaction_max_components", 2)
            ),
        )
    evidence = prepare_evidence_payload(
        retrieval,
        max_hit_chars=args.max_hit_chars,
        max_total_chars=args.max_evidence_chars,
    )
    evidence_alias_map = _evidence_alias_map(evidence)
    if is_verified_calculation_path(
        getattr(args, "calculation_mode", "direct"),
        question.answer_format,
    ):
        return _run_one_verified_calculation(
            question,
            args=args,
            client=client,
            retrieval=retrieval,
            evidence=evidence,
            evidence_alias_map=evidence_alias_map,
        )
    output_contract = getattr(args, "output_contract", "joint")
    canonical_output = output_contract == "reasoning-canonical"
    initial_schema = (
        build_reasoning_canonical_schema()
        if canonical_output
        else build_answer_schema(question)
    )
    validation_error = ""
    previous_response = ""
    frozen_answer_parts: list[str] | None = None
    try:
        calls = _load_raw_call_checkpoint(
            args.run_dir,
            question.qid,
            evidence_alias_map,
        )
    except Exception as exc:
        return {
            "qid": question.qid,
            "domain": question.domain,
            "answer_format": question.answer_format,
            "question_type": question.type,
            "pipeline_version": PIPELINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "retrieval_policy_version": active_retrieval_policy_version,
            "retrieval": retrieval,
            "evidence_items": evidence,
            "evidence_alias_map": evidence_alias_map,
            "calls": [],
            "token_usage": TokenUsage().to_dict(),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "unobservable_usage_risk": True,
            "preserve_existing_raw_calls": True,
        }
    if not canonical_output:
        try:
            frozen_answer_parts = _load_direct_frozen_answer(
                args.run_dir,
                question=question,
                evidence_alias_map=evidence_alias_map,
                calls=calls,
            )
        except Exception as exc:
            return {
                "qid": question.qid,
                "domain": question.domain,
                "answer_format": question.answer_format,
                "question_type": question.type,
                "pipeline_version": PIPELINE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "retrieval_policy_version": active_retrieval_policy_version,
                "retrieval": retrieval,
                "evidence_items": evidence,
                "evidence_alias_map": evidence_alias_map,
                "calls": calls,
                "token_usage": _sum_call_usage(calls),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "unobservable_usage_risk": False,
                "preserve_existing_raw_calls": True,
            }
    parsed: dict[str, Any] | None = None
    if calls:
        previous_response = str(calls[-1]["content"])
        candidate: dict[str, Any] | None = None
        try:
            candidate = json.loads(previous_response)
            if not isinstance(candidate, dict):
                raise ValueError("model response must be a JSON object")
            if calls[-1].get("purpose") == "reasoning_only_retry_from_frozen_answer":
                raw_frozen_parts = calls[-1].get("frozen_answer_parts")
                if not isinstance(raw_frozen_parts, list):
                    raise ValueError("reasoning-only checkpoint lost frozen answer")
                frozen_answer_parts = [str(item) for item in raw_frozen_parts]
                parsed = validate_frozen_answer_reasoning_payload(
                    frozen_answer_parts,
                    candidate,
                )
            else:
                if not canonical_output and frozen_answer_parts is None:
                    frozen_answer_parts = validate_answer_parts_shape(
                        question,
                        candidate.get("answer_parts"),
                    )
                    _freeze_direct_answer(
                        args.run_dir,
                        question=question,
                        evidence_alias_map=evidence_alias_map,
                        answer_parts=frozen_answer_parts,
                        calls=calls,
                    )
                parsed = (
                    validate_reasoning_canonical_payload(question, candidate)
                    if canonical_output
                    else validate_answer_payload(question, candidate)
                )
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            validation_error = _validation_error_code(exc)
            if (
                not canonical_output
                and validation_error
                in {
                    "reasoning_missing_explicit_conclusion",
                    "answer_reasoning_conclusion_mismatch",
                }
                and candidate is not None
            ):
                try:
                    frozen_answer_parts = validate_answer_parts_shape(
                        question,
                        candidate.get("answer_parts"),
                    )
                    _freeze_direct_answer(
                        args.run_dir,
                        question=question,
                        evidence_alias_map=evidence_alias_map,
                        answer_parts=frozen_answer_parts,
                        calls=calls,
                    )
                except (ValueError, TypeError, KeyError):
                    frozen_answer_parts = None
    for attempt in range(len(calls), args.max_format_retries + 1):
        if parsed is not None:
            break
        reasoning_only_retry = frozen_answer_parts is not None
        active_schema = (
            build_frozen_answer_reasoning_schema(frozen_answer_parts)
            if reasoning_only_retry
            else initial_schema
        )
        try:
            messages = (
                build_frozen_answer_reasoning_messages(
                    question,
                    evidence,
                    frozen_answer_parts=frozen_answer_parts,
                )
                if reasoning_only_retry
                else build_answer_messages(
                    question,
                    evidence,
                    validation_error=validation_error,
                    previous_response=previous_response,
                    output_contract=(
                        "reasoning_canonical" if canonical_output else "joint"
                    ),
                )
            )
        except Exception as exc:
            return {
                "qid": question.qid,
                "domain": question.domain,
                "answer_format": question.answer_format,
                "question_type": question.type,
                "pipeline_version": PIPELINE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "retrieval_policy_version": active_retrieval_policy_version,
                "retrieval": retrieval,
                "evidence_items": evidence,
                "evidence_alias_map": evidence_alias_map,
                "calls": calls,
                "token_usage": _sum_call_usage(calls),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "unobservable_usage_risk": False,
            }
        try:
            call_purpose = (
                "reasoning_only_retry_from_frozen_answer"
                if reasoning_only_retry
                else "initial_answer"
                if attempt == 0
                else "format_consistency_retry"
            )
            _record_generation_call_intent(
                args.run_dir,
                qid=question.qid,
                call_index=attempt + 1,
                purpose=call_purpose,
                evidence_alias_map=evidence_alias_map,
            )
            response = client.chat_json(
                messages,
                response_schema=active_schema,
                schema_name=(
                    "b_retrieval_llm_frozen_answer_reasoning_v3"
                    if reasoning_only_retry
                    else "b_retrieval_llm_reasoning_canonical_v1"
                    if canonical_output
                    else "b_retrieval_llm_final_submission_v2"
                ),
                extra_body=(
                    {"enable_thinking": False}
                    if reasoning_only_retry
                    else {
                        "enable_thinking": args.thinking_budget > 0,
                        **(
                            {"thinking_budget": args.thinking_budget}
                            if args.thinking_budget > 0
                            else {}
                        ),
                    }
                ),
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            provider_error = ""
            if response is not None:
                provider_error = str(getattr(response, "text", ""))[:2000]
            return {
                "qid": question.qid,
                "domain": question.domain,
                "answer_format": question.answer_format,
                "question_type": question.type,
                "pipeline_version": PIPELINE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "retrieval_policy_version": active_retrieval_policy_version,
                "retrieval": retrieval,
                "evidence_items": evidence,
                "evidence_alias_map": evidence_alias_map,
                "calls": calls,
                "token_usage": _sum_call_usage(calls),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "provider_error": provider_error,
                "failed_transport_stage": call_purpose,
                "failed_transport_attempt_count": int(
                    getattr(exc, "transport_attempt_count", 1)
                ),
                "failed_transport_rejections": list(
                    getattr(exc, "transport_rejections", ())
                ),
                "transport_retry_policy_version": (
                    TRANSPORT_RETRY_POLICY_VERSION
                ),
                "unobservable_usage_risk": _unobservable_usage_risk(exc),
            }
        call = {
            "call_index": attempt + 1,
            "purpose": call_purpose,
            "model_name": client.config.model_name,
            "response_format_mode": response.response_format_mode,
            "messages": messages,
            "response_schema": active_schema,
            "raw_response": response.raw_payload,
            "content": response.content,
            "token_usage": response.token_usage.to_dict(),
            "transport_attempt_count": response.transport_attempt_count,
            "transport_rejections": list(response.transport_rejections),
        }
        if reasoning_only_retry:
            call["frozen_answer_parts"] = list(frozen_answer_parts)
        calls.append(call)
        _checkpoint_raw_calls(
            args.run_dir,
            question.qid,
            calls,
            evidence_alias_map,
        )
        previous_response = response.content
        candidate = None
        try:
            candidate = json.loads(response.content)
            if not isinstance(candidate, dict):
                raise ValueError("model response must be a JSON object")
            if not canonical_output and not reasoning_only_retry:
                frozen_answer_parts = validate_answer_parts_shape(
                    question,
                    candidate.get("answer_parts"),
                )
                _freeze_direct_answer(
                    args.run_dir,
                    question=question,
                    evidence_alias_map=evidence_alias_map,
                    answer_parts=frozen_answer_parts,
                    calls=calls,
                )
            parsed = (
                validate_frozen_answer_reasoning_payload(
                    frozen_answer_parts,
                    candidate,
                )
                if reasoning_only_retry
                else validate_reasoning_canonical_payload(question, candidate)
                if canonical_output
                else validate_answer_payload(question, candidate)
            )
            break
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            validation_error = _validation_error_code(exc)
            if (
                not canonical_output
                and not reasoning_only_retry
                and validation_error
                in {
                    "reasoning_missing_explicit_conclusion",
                    "answer_reasoning_conclusion_mismatch",
                }
                and candidate is not None
            ):
                try:
                    frozen_answer_parts = validate_answer_parts_shape(
                        question,
                        candidate.get("answer_parts"),
                    )
                    _freeze_direct_answer(
                        args.run_dir,
                        question=question,
                        evidence_alias_map=evidence_alias_map,
                        answer_parts=frozen_answer_parts,
                        calls=calls,
                    )
                except (ValueError, TypeError, KeyError):
                    frozen_answer_parts = None
    usage = _sum_call_usage(calls)
    base = {
        "qid": question.qid,
        "domain": question.domain,
        "answer_format": question.answer_format,
        "question_type": question.type,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_policy_version": active_retrieval_policy_version,
        "retrieval": retrieval,
        "evidence_items": evidence,
        "evidence_alias_map": evidence_alias_map,
        "calls": calls,
        "token_usage": usage,
    }
    if parsed is None:
        return {
            **base,
            "status": "failed",
            "error_type": "response_validation_failed",
            "error": validation_error,
            "unobservable_usage_risk": False,
        }
    return {
        **base,
        "status": "answered",
        **parsed,
        "format_retry_count": sum(
            call.get("purpose") == "format_consistency_retry"
            for call in calls
        ),
        "reasoning_only_call_count": sum(
            call.get("purpose")
            == "reasoning_only_retry_from_frozen_answer"
            for call in calls
        ),
        "unobservable_usage_risk": False,
    }


def _run_one_verified_calculation(
    question: Any,
    *,
    args: argparse.Namespace,
    client: OpenAICompatibleClient,
    retrieval: dict[str, Any],
    evidence: list[dict[str, Any]],
    evidence_alias_map: list[dict[str, Any]],
) -> dict[str, Any]:
    from afa_agent.b_board.retrieval_llm_calculation import (
        VerifiedCalculationStageError,
        run_verified_calculation,
    )

    base = {
        "qid": question.qid,
        "domain": question.domain,
        "answer_format": question.answer_format,
        "question_type": question.type,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_policy_version": str(
            retrieval.get(
                "policy_version",
                retrieval_policy_version(
                    getattr(
                        args,
                        "document_candidate_strategy",
                        "anchor_union",
                    ),
                    getattr(args, "evidence_quota_strategy", "primary_guard"),
                ),
            )
        ),
        "calculation_mode": "verified",
        "retrieval": retrieval,
        "evidence_items": evidence,
        "evidence_alias_map": evidence_alias_map,
        "format_retry_count": 0,
    }
    try:
        outcome = run_verified_calculation(
            question,
            evidence=evidence,
            client=client,
            run_dir=args.run_dir,
            thinking_budget=args.thinking_budget,
        )
    except VerifiedCalculationStageError as exc:
        raw_checkpoint_exists = (
            args.run_dir / "raw_calls" / f"{question.qid}.json"
        ).exists()
        return {
            **base,
            "calls": exc.calls,
            "token_usage": exc.token_usage,
            "decision_trace": exc.decision_trace,
            "frozen_answer_checkpoint": (
                {
                    "path": exc.frozen_checkpoint_path,
                    "preserved": True,
                }
                if exc.frozen_checkpoint_path
                else None
            ),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "error_stage": exc.stage,
            "error_code": exc.error_code,
            "retry_route": exc.retry_route,
            "failed_transport_stage": exc.stage,
            "failed_transport_attempt_count": (
                exc.failed_transport_attempt_count
            ),
            "failed_transport_rejections": (
                exc.failed_transport_rejections
            ),
            "transport_retry_policy_version": (
                TRANSPORT_RETRY_POLICY_VERSION
            ),
            "unobservable_usage_risk": exc.unobservable_usage_risk,
            "preserve_existing_raw_calls": (
                raw_checkpoint_exists and not exc.calls
            ),
        }
    return {
        **base,
        **outcome,
        "status": "answered",
        "unobservable_usage_risk": False,
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "workers",
        "per_query_top_k",
        "final_top_k",
        "max_doc_candidates",
        "max_queries_per_option",
        "max_hit_chars",
        "max_evidence_chars",
        "compaction_pool_multiplier",
        "compaction_max_block_chars",
        "compaction_max_components",
    )
    for field in positive_fields:
        if int(getattr(args, field)) < 1:
            raise ValueError(f"{field} must be positive")
    qids = [str(qid) for qid in getattr(args, "qid", [])]
    if len(qids) != len(set(qids)):
        raise ValueError("qid arguments must not contain duplicates")
    if args.thinking_budget < 0:
        raise ValueError("thinking_budget must not be negative")
    if args.max_format_retries not in {0, 1}:
        raise ValueError("max_format_retries must be zero or one")
    if (
        getattr(args, "evidence_quota_strategy", "primary_guard")
        in {
            "adaptive_multi_report_calculation",
            "metric_slot_coverage",
        }
        and getattr(args, "document_candidate_strategy", "anchor_union")
        != "anchor_first"
    ):
        raise ValueError(
            "adaptive multi-document evidence strategies require "
            "anchor_first documents"
        )
    research_only_strategy = (
        getattr(args, "evidence_quota_strategy", "primary_guard")
        in RESEARCH_ONLY_EVIDENCE_QUOTA_STRATEGIES
    )
    if research_only_strategy and not getattr(
        args,
        "allow_research_only_strategy",
        False,
    ):
        raise ValueError(
            "research-only evidence quota strategy requires "
            "--allow-research-only-strategy"
        )
    if (
        not research_only_strategy
        and getattr(args, "allow_research_only_strategy", False)
    ):
        raise ValueError(
            "--allow-research-only-strategy is only valid with a "
            "research-only evidence quota strategy"
        )
    if (
        args.calculation_mode == "verified"
        and args.output_contract != "joint"
    ):
        raise ValueError(
            "verified calculation requires the joint output contract; "
            "reasoning-canonical is a direct-mode experiment"
        )
    if not 0 <= args.supplemental_weight <= 1:
        raise ValueError("supplemental_weight must be between zero and one")


@contextmanager
def _exclusive_run_lock(run_dir: Path) -> Iterator[None]:
    """Prevent two processes from charging and overwriting the same run."""

    resolved = run_dir.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved.parent / f".{resolved.name}.run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"run directory is already active: {resolved}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "run_dir": str(resolved),
                    "pid": int(__import__("os").getpid()),
                    "acquired_at": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _public_config(
    args: argparse.Namespace,
    questions: list[Any],
    model_config: Any,
    index_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    question_files = sorted(
        path
        for path in args.question_root.iterdir()
        if path.suffix.lower() in {".json", ".jsonl"}
    )
    source_files = _generation_source_files(
        getattr(args, "calculation_mode", "direct"),
        getattr(args, "evidence_compaction", "off"),
    )
    document_candidate_strategy = getattr(
        args, "document_candidate_strategy", "anchor_union"
    )
    evidence_quota_strategy = getattr(
        args, "evidence_quota_strategy", "primary_guard"
    )
    research_only_strategy = (
        evidence_quota_strategy in RESEARCH_ONLY_EVIDENCE_QUOTA_STRATEGIES
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_policy_version": retrieval_policy_version(
            document_candidate_strategy,
            evidence_quota_strategy,
        ),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "model_name": model_config.model_name,
            "temperature": model_config.temperature,
            "structured_output_mode": "native_json_schema_strict",
            "thinking_budget": args.thinking_budget,
            "api_base_sha256": _sha256_text(
                str(model_config.api_base).rstrip("/")
            ),
            "connect_timeout_seconds": model_config.connect_timeout_seconds,
            "read_timeout_seconds": model_config.read_timeout_seconds,
            "max_retries": model_config.max_retries,
            "retry_backoff_seconds": model_config.retry_backoff_seconds,
        },
        "scope": {
            "question_count": len(questions),
            "qids": [question.qid for question in questions],
        },
        "retrieval": {
            "index_root": str(args.index_root.resolve()),
            "per_query_top_k": args.per_query_top_k,
            "final_top_k": args.final_top_k,
            "max_doc_candidates": args.max_doc_candidates,
            "document_candidate_strategy": document_candidate_strategy,
            "evidence_quota_strategy": evidence_quota_strategy,
            "research_only_strategy": research_only_strategy,
            "supplemental_weight": args.supplemental_weight,
            "max_queries_per_option": args.max_queries_per_option,
            "max_hit_chars": args.max_hit_chars,
            "max_evidence_chars": args.max_evidence_chars,
            "evidence_compaction": getattr(args, "evidence_compaction", "off"),
            "compaction_pool_multiplier": getattr(
                args, "compaction_pool_multiplier", 3
            ),
            "compaction_max_block_chars": getattr(
                args, "compaction_max_block_chars", 3600
            ),
            "compaction_max_components": getattr(
                args, "compaction_max_components", 2
            ),
            "domain_unit_counts": {
                domain: len(payload["units"])
                for domain, payload in index_payloads.items()
            },
            "index_sha256": {
                domain: sha256_file(args.index_root / domain / "index.json")
                for domain in index_payloads
            },
        },
        "generation": {
            "workers": args.workers,
            "max_format_retries": args.max_format_retries,
            "calculation_mode": getattr(args, "calculation_mode", "direct"),
            "output_contract": getattr(args, "output_contract", "joint"),
            "final_payload_policy": "model_fields_only_no_code_rewrite",
            "transport_retry": "explicit HTTP 429 pre-generation rejection only",
            "read_timeout_retry": False,
        },
        "runtime": {
            "python": sys.version,
            "requests": requests.__version__,
            "jieba": _package_version("jieba"),
            "charset_normalizer": _package_version("charset-normalizer"),
        },
        "input_sha256": {
            "submission_template": sha256_file(args.submission_template),
            "question_files": {
                str(path.resolve()): sha256_file(path) for path in question_files
            },
            "source_files": {
                str(path.relative_to(ROOT)): sha256_file(path) for path in source_files
            },
        },
        "answer_blind_contract": {
            "qid_in_model_messages": False,
            "reference_loaded_by_generation": False,
            "official_locks_loaded_by_generation": False,
            "solver_or_rule_layer_used": False,
            "fixed_locator_used": False,
            "deterministic_calculation_executor_used": (
                getattr(args, "calculation_mode", "direct") == "verified"
            ),
            "document_scope": "all documents in the question domain index",
            "research_only_strategy": research_only_strategy,
        },
    }


def _generation_source_files(
    calculation_mode: str,
    evidence_compaction: str = "off",
) -> list[Path]:
    if calculation_mode == "verified":
        __import__("afa_agent.b_board.retrieval_llm_calculation")
    source_files = {Path(__file__).resolve()}
    source_root = (ROOT / "src").resolve()
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "afa_agent" and not module_name.startswith("afa_agent."):
            continue
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        if path.suffix == ".pyc":
            path = Path(str(path)[:-1])
        if path.is_file() and (path == source_root or source_root in path.parents):
            source_files.add(path)
    return sorted(source_files)


def _assert_no_prohibited_generation_modules_loaded() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if name == "afa_agent" or name.startswith("afa_agent.")
        if any(
            fragment in name
            for fragment in PROHIBITED_GENERATION_MODULE_FRAGMENTS
        )
    )
    if loaded:
        raise RuntimeError(
            "direct generation process loaded prohibited modules: "
            + ", ".join(loaded)
        )


def _package_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "not-installed"


def _finished_qids(
    *,
    answers: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    run_dir: Path,
    calculation_mode: str,
) -> set[str]:
    finished = {str(row["qid"]) for row in answers}
    for failure in failures:
        qid = str(failure["qid"])
        checkpoint_exists = (
            run_dir / "frozen_answers" / f"{qid}.json"
        ).is_file()
        safe_reasoning_resume = (
            calculation_mode == "verified"
            and failure.get("error_stage") == "reasoning"
            and failure.get("error_code") == "reasoning_contract_error"
            and failure.get("retry_route")
            == "resume_reasoning_from_frozen_checkpoint"
            and failure.get("unobservable_usage_risk") is False
            and checkpoint_exists
            and _raw_call_purpose_count(
                run_dir / "raw_calls" / f"{qid}.json",
                "verified_calculation_reasoning",
            )
            < MAX_VERIFIED_REASONING_CALLS
        )
        if not safe_reasoning_resume:
            finished.add(qid)
    return finished


def _raw_call_purpose_count(path: Path, purpose: str) -> int:
    if not path.exists():
        return 0
    payload = _read_json_object(path, label="raw call ledger")
    calls = payload.get("calls")
    if not isinstance(calls, list):
        raise ValueError("raw call ledger calls must be an array")
    return sum(
        isinstance(call, Mapping) and call.get("purpose") == purpose
        for call in calls
    )


def _initialize_run_dir(
    run_dir: Path,
    public_config: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("run directory fingerprint mismatch")
        existing_config = existing.get("config")
        if not isinstance(existing_config, dict):
            raise ValueError("stored run config is invalid")
        if not isinstance(existing.get("run_instance_id"), str) or not existing[
            "run_instance_id"
        ]:
            raise ValueError(
                "stored run config has no immutable run instance id"
            )
        return existing_config
    if any(run_dir.iterdir()):
        raise ValueError("new run directory must be empty")
    (run_dir / "raw_calls").mkdir()
    (run_dir / "call_intents").mkdir()
    (run_dir / "frozen_answers").mkdir()
    (run_dir / "retrieval").mkdir()
    _write_json(
        config_path,
        {
            "fingerprint": fingerprint,
            "run_instance_id": uuid4().hex,
            "config": public_config,
        },
    )
    _write_json(run_dir / "answers.json", [])
    _write_json(run_dir / "failures.json", [])
    return public_config


def _persist_result(
    run_dir: Path,
    result: dict[str, Any],
    *,
    question_order: list[str],
    fingerprint: str,
    public_config: dict[str, Any],
) -> None:
    qid = str(result["qid"])
    _write_immutable_or_equal_json(
        run_dir / "retrieval" / f"{qid}.json",
        result.get("retrieval", {}),
        label=f"{qid} retrieval",
    )
    raw_path = run_dir / "raw_calls" / f"{qid}.json"
    raw_payload = {
        "qid": qid,
        **_run_instance_binding(run_dir),
        "evidence_alias_map": result.get("evidence_alias_map", []),
        "calls": result.get("calls", []),
        "final_call_index": (
            len(result.get("calls", []))
            if result.get("status") == "answered"
            else None
        ),
        "final_content_sha256": _sha256_text(
            str(result.get("calls", [{}])[-1].get("content", ""))
            if result.get("calls")
            else ""
        ),
        "submitted_answer_parts_sha256": _sha256_text(
            json.dumps(
                result.get("answer_parts", []),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
        "submitted_reasoning_sha256": _sha256_text(
            str(result.get("reasoning", ""))
        ),
        "postprocessing": _postprocessing_record(result),
        "ledger_state": (
            "finalized_answer"
            if result.get("status") == "answered"
            else "checkpoint_resume_allowed"
            if (
                result.get("retry_route")
                == "resume_reasoning_from_frozen_checkpoint"
            )
            else "finalized_failure"
        ),
    }
    if result.get("preserve_existing_raw_calls") and raw_path.exists():
        _read_json_object(raw_path, label=f"{qid} raw call ledger")
    else:
        _write_raw_ledger_append_only(raw_path, raw_payload)
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"retrieval", "calls"}
    }
    target = (
        run_dir / "answers.json"
        if result["status"] == "answered"
        else run_dir / "failures.json"
    )
    for result_path in (
        run_dir / "answers.json",
        run_dir / "failures.json",
    ):
        rows = [
            row
            for row in _read_rows(result_path)
            if str(row["qid"]) != qid
        ]
        _write_json(result_path, rows)
    rows = _read_rows(target)
    rows = [row for row in rows if str(row["qid"]) != qid]
    rows.append(compact)
    order = {item: index for index, item in enumerate(question_order)}
    rows.sort(key=lambda row: order[str(row["qid"])])
    _write_json(target, rows)
    _write_manifest(
        run_dir,
        question_order=question_order,
        fingerprint=fingerprint,
        public_config=public_config,
    )


def _postprocessing_record(result: dict[str, Any]) -> dict[str, Any]:
    trace = result.get("decision_trace") or {}
    mode = str(trace.get("postprocessing_mode", "none"))
    if mode in {
        "none",
        "multi_choice_conclusion_separator_equivalence",
    }:
        return {
            "answer_modified": False,
            "reasoning_modified": False,
            "csv_escaping_only": True,
        }
    if mode not in {
        "model_field_assembly",
        "deterministic_format_normalization",
        "same_response_contract_recovery",
        "same_response_conclusion_assembly",
    }:
        raise ValueError(f"unsupported postprocessing mode: {mode}")
    normalized = bool(trace.get("deterministic_format_normalization", False))
    return {
        "mode": mode,
        "answer_modified": mode
        in {
            "deterministic_format_normalization",
            "same_response_contract_recovery",
        },
        "reasoning_modified": (
            mode == "deterministic_format_normalization"
            or (mode == "same_response_contract_recovery" and normalized)
            or mode == "same_response_conclusion_assembly"
        ),
        "model_fields_assembled": mode == "model_field_assembly",
        "semantic_correction": False,
        "csv_escaping_only": False,
    }


def _write_manifest(
    run_dir: Path,
    *,
    question_order: list[str],
    fingerprint: str,
    public_config: dict[str, Any],
) -> dict[str, Any]:
    answers = _read_rows(run_dir / "answers.json")
    failures = _read_rows(run_dir / "failures.json")
    answered = {str(row["qid"]) for row in answers}
    failed = {str(row["qid"]) for row in failures}
    if answered & failed:
        raise ValueError("a qid cannot be both answered and failed")
    usage = TokenUsage()
    raw_call_count = 0
    format_retry_count = 0
    transport_attempt_count = 0
    transport_rejection_count = 0
    call_purpose_counts: Counter[str] = Counter()
    usage_reconciliation_problems: list[str] = []
    for row in [*answers, *failures]:
        qid = str(row["qid"])
        if qid in failed:
            failed_attempts, failed_rejections = (
                _validate_failed_transport_audit(
                    row,
                    label=f"{qid} failed provider stage",
                )
            )
            transport_attempt_count += failed_attempts
            transport_rejection_count += failed_rejections
        row_usage = _strict_usage(
            row.get("token_usage") or {},
            label=f"{qid} row token usage",
        )
        call_path = run_dir / "raw_calls" / f"{row['qid']}.json"
        if not call_path.exists():
            usage_reconciliation_problems.append(f"{qid}:missing_raw_ledger")
            continue
        call_usage = TokenUsage()
        usage_added = False
        try:
            raw_payload = _read_json_object(
                call_path, label=f"{qid} raw call ledger"
            )
            if raw_payload.get("qid") != qid:
                raise ValueError("qid mismatch")
            calls = raw_payload.get("calls")
            if not isinstance(calls, list):
                raise ValueError("calls must be an array")
            raw_call_count += len(calls)
            for expected_index, call in enumerate(calls, start=1):
                if not isinstance(call, dict):
                    raise ValueError("call must be an object")
                if int(call.get("call_index", -1)) != expected_index:
                    raise ValueError("call indexes are not contiguous")
                purpose = str(call.get("purpose", "unknown"))
                call_purpose_counts[purpose] += 1
                if purpose == "format_consistency_retry":
                    format_retry_count += 1
                attempt_count, rejection_count = _validate_transport_audit(
                    call,
                    label=f"{qid} call {expected_index}",
                )
                transport_attempt_count += attempt_count
                transport_rejection_count += rejection_count
                current_usage = _strict_usage(
                    call.get("token_usage") or {},
                    label=f"{qid} call {expected_index} token usage",
                )
                raw_provider_usage = _strict_usage(
                    (call.get("raw_response") or {}).get("usage") or {},
                    label=f"{qid} call {expected_index} provider usage",
                )
                if current_usage.to_dict() != raw_provider_usage.to_dict():
                    raise ValueError("call usage differs from provider raw usage")
                call_usage.add(current_usage)
            _validate_generation_call_intents(
                run_dir,
                qid=qid,
                evidence_alias_map=list(
                    raw_payload.get("evidence_alias_map") or []
                ),
                calls=[dict(call) for call in calls],
                terminal_failure=(row if qid in failed else None),
            )
            usage.add(call_usage)
            usage_added = True
            if call_usage.to_dict() != row_usage.to_dict():
                usage_reconciliation_problems.append(
                    f"{qid}:row usage differs from raw call ledger"
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if not usage_added:
                usage.add(call_usage)
            usage_reconciliation_problems.append(
                f"{qid}:{type(exc).__name__}:{exc}"
            )
    finished = answered | failed
    expected = set(question_order)
    unobservable_usage_risk = bool(usage_reconciliation_problems) or any(
        bool(row.get("unobservable_usage_risk")) for row in failures
    )
    status = "complete" if finished == expected and not failed else (
        "incomplete" if finished == expected else "running"
    )
    manifest = {
        "run_id": run_dir.name,
        "runner": PIPELINE_VERSION,
        "status": status,
        "created_at": public_config["created_at"],
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        "model": public_config["model"],
        "scope": public_config["scope"],
        "expected_question_count": len(question_order),
        "answered_question_count": len(answered),
        "failed_question_count": len(failed),
        "answered_qids": [qid for qid in question_order if qid in answered],
        "failed_qids": [qid for qid in question_order if qid in failed],
        "raw_call_count": raw_call_count,
        "format_retry_count": format_retry_count,
        "transport_attempt_count": transport_attempt_count,
        "transport_rejection_count": transport_rejection_count,
        "call_purpose_counts": dict(sorted(call_purpose_counts.items())),
        "token_usage": usage.to_dict(),
        "all_observed_usage_from_provider_raw_fields": (
            not usage_reconciliation_problems
        ),
        "usage_reconciliation_problems": usage_reconciliation_problems,
        "unobservable_usage_risk": unobservable_usage_risk,
        "answer_blind_contract": public_config["answer_blind_contract"],
        "submission_eligible": False,
        "submission_ineligibility_reasons": [
            "research_only_answer_blind_baseline",
            "accuracy_reference_is_evaluation_only",
            *(
                ["research_only_evidence_quota_strategy"]
                if bool(
                    (public_config.get("retrieval") or {}).get(
                        "research_only_strategy"
                    )
                )
                else []
            ),
        ],
        "official_submission_count": 0,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _sum_call_usage(calls: list[dict[str, Any]]) -> dict[str, int]:
    usage = TokenUsage()
    for call in calls:
        raw = call["token_usage"]
        usage.add(
            TokenUsage(
                prompt_tokens=int(raw["prompt_tokens"]),
                completion_tokens=int(raw["completion_tokens"]),
                total_tokens=int(raw["total_tokens"]),
            )
        )
    return usage.to_dict()


def _evidence_alias_map(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "alias": str(item["evidence_key"]),
            "source_alias": str(item["source_key"]),
            "source_evidence_id": str(item["evidence_id"]),
            "doc_id": str(item["doc_id"]),
            "retrieval_rank": int(item["rank"]),
            "unit_type": str(item["unit_type"]),
            "title_path": list(item["title_path"]),
            "prompt_text_sha256": str(item["prompt_text_sha256"]),
            "source_text_sha256": str(item["source_text_sha256"]),
            "prompt_text_chars": len(str(item["text"])),
            "truncated": bool(item["truncated"]),
            "merged_from": list(item.get("merged_from", [])),
            "source_order": list(item.get("source_order", [])),
            "overlap_chars": int(item.get("overlap_chars", 0)),
            "component_hashes": list(item.get("component_hashes", [])),
            "compaction_truncation_provenance": dict(
                item.get("compaction_truncation_provenance", {})
            ),
        }
        for item in evidence
    ]


def _checkpoint_raw_calls(
    run_dir: Path,
    qid: str,
    calls: list[dict[str, Any]],
    evidence_alias_map: list[dict[str, Any]],
) -> None:
    _write_raw_ledger_append_only(
        run_dir / "raw_calls" / f"{qid}.json",
        {
            "qid": qid,
            **_run_instance_binding(run_dir),
            "evidence_alias_map": evidence_alias_map,
            "calls": calls,
            "checkpoint_only": True,
            "ledger_state": "checkpoint",
        },
    )


def _record_generation_call_intent(
    run_dir: Path,
    *,
    qid: str,
    call_index: int,
    purpose: str,
    evidence_alias_map: list[dict[str, Any]],
) -> None:
    path = run_dir / "call_intents" / f"{qid}.{call_index}.json"
    payload = {
        "qid": qid,
        "call_index": call_index,
        "purpose": purpose,
        **_run_instance_binding(run_dir),
        "evidence_alias_map_sha256": _sha256_text(
            json.dumps(
                evidence_alias_map,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "state": "provider_call_started",
    }
    if path.exists():
        if _read_json_object(path, label=f"{qid} call intent") != payload:
            raise ValueError("generation call intent changed")
        raise ValueError("generation call intent already exists")
    write_json_atomic(path, payload)


def _validate_generation_call_intents(
    run_dir: Path,
    *,
    qid: str,
    evidence_alias_map: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    terminal_failure: Mapping[str, Any] | None = None,
) -> None:
    intent_dir = run_dir / "call_intents"
    if not intent_dir.exists():
        return
    intent_paths = sorted(intent_dir.glob(f"{qid}.*.json"))
    expected_alias_sha256 = _sha256_text(
        json.dumps(
            evidence_alias_map,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    intents: dict[int, dict[str, Any]] = {}
    for path in intent_paths:
        payload = _read_json_object(path, label=f"{qid} call intent")
        _validate_run_instance_binding(run_dir, payload)
        call_index = payload.get("call_index")
        if (
            payload.get("qid") != qid
            or isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index < 1
            or path.name != f"{qid}.{call_index}.json"
            or call_index in intents
            or payload.get("evidence_alias_map_sha256")
            != expected_alias_sha256
            or payload.get("state") != "provider_call_started"
        ):
            raise ValueError("generation call intent binding is invalid")
        intents[call_index] = payload
    for call in calls:
        call_index = int(call.get("call_index", -1))
        intent = intents.get(call_index)
        if intent is None or intent.get("purpose") != call.get("purpose"):
            raise ValueError("observed generation call has no matching intent")
    unmatched = sorted(set(intents) - {
        int(call.get("call_index", -1)) for call in calls
    })
    if unmatched:
        expected_purpose_by_stage = {
            "answer": "calculation_plan",
            "reasoning": "verified_calculation_reasoning",
        }
        failed_stage = (
            str(terminal_failure.get("failed_transport_stage", ""))
            if terminal_failure is not None
            else ""
        )
        expected_purpose = expected_purpose_by_stage.get(
            failed_stage,
            failed_stage,
        )
        if (
            terminal_failure is not None
            and len(unmatched) == 1
            and unmatched[0] == len(calls) + 1
            and intents[unmatched[0]].get("purpose") == expected_purpose
            and int(
                terminal_failure.get(
                    "failed_transport_attempt_count", 0
                )
            )
            >= 1
        ):
            return
        raise RuntimeError(
            "unobservable provider attempt exists for call indexes "
            + ",".join(str(item) for item in unmatched)
        )


def _freeze_direct_answer(
    run_dir: Path,
    *,
    question: Any,
    evidence_alias_map: list[dict[str, Any]],
    answer_parts: list[str],
    calls: list[dict[str, Any]],
) -> None:
    if not calls:
        raise ValueError("direct answer cannot freeze without an observed call")
    answer_call = calls[-1]
    if answer_call.get("purpose") == "reasoning_only_retry_from_frozen_answer":
        raise ValueError("reasoning-only call cannot replace the frozen answer")
    payload = {
        "checkpoint_version": "direct_model_answer_v1",
        "checkpoint_kind": "direct_model_answer",
        "qid": question.qid,
        **_run_instance_binding(run_dir),
        "evidence_alias_map": evidence_alias_map,
        "answer_parts": list(answer_parts),
        "answer_call_index": int(answer_call["call_index"]),
        "answer_content_sha256": _sha256_text(str(answer_call["content"])),
    }
    _write_immutable_or_equal_json(
        run_dir / "frozen_answers" / f"{question.qid}.json",
        payload,
        label=f"{question.qid} direct frozen answer",
    )


def _load_direct_frozen_answer(
    run_dir: Path,
    *,
    question: Any,
    evidence_alias_map: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[str] | None:
    path = run_dir / "frozen_answers" / f"{question.qid}.json"
    if not path.exists():
        return None
    payload = _read_json_object(path, label=f"{question.qid} frozen answer")
    _validate_run_instance_binding(run_dir, payload)
    if (
        payload.get("checkpoint_version") != "direct_model_answer_v1"
        or payload.get("checkpoint_kind") != "direct_model_answer"
        or payload.get("qid") != question.qid
        or payload.get("evidence_alias_map") != evidence_alias_map
    ):
        raise ValueError("direct frozen answer binding mismatch")
    call_index = payload.get("answer_call_index")
    if (
        isinstance(call_index, bool)
        or not isinstance(call_index, int)
        or call_index < 1
        or call_index > len(calls)
    ):
        raise ValueError("direct frozen answer call index is invalid")
    answer_call = calls[call_index - 1]
    if (
        int(answer_call.get("call_index", -1)) != call_index
        or answer_call.get("purpose")
        == "reasoning_only_retry_from_frozen_answer"
        or payload.get("answer_content_sha256")
        != _sha256_text(str(answer_call.get("content", "")))
    ):
        raise ValueError("direct frozen answer call binding mismatch")
    answer_parts = validate_answer_parts_shape(
        question,
        payload.get("answer_parts"),
    )
    raw_payload = json.loads(str(answer_call["content"]))
    if raw_payload.get("answer_parts") != answer_parts:
        raise ValueError("direct frozen answer differs from model response")
    return answer_parts


def _load_raw_call_checkpoint(
    run_dir: Path,
    qid: str,
    evidence_alias_map: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = run_dir / "raw_calls" / f"{qid}.json"
    if not path.exists():
        _validate_generation_call_intents(
            run_dir,
            qid=qid,
            evidence_alias_map=evidence_alias_map,
            calls=[],
        )
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_run_instance_binding(run_dir, payload)
    if payload.get("qid") != qid:
        raise ValueError("raw call checkpoint qid mismatch")
    if payload.get("evidence_alias_map") != evidence_alias_map:
        raise ValueError("raw call checkpoint evidence fingerprint mismatch")
    raw_calls = payload.get("calls")
    if not isinstance(raw_calls, list):
        raise ValueError("raw call checkpoint calls must be an array")
    calls: list[dict[str, Any]] = []
    for expected_index, raw_call in enumerate(raw_calls, start=1):
        if not isinstance(raw_call, dict):
            raise ValueError("raw call checkpoint contains a non-object call")
        if int(raw_call.get("call_index", -1)) != expected_index:
            raise ValueError("raw call checkpoint call indexes are not contiguous")
        if not isinstance(raw_call.get("content"), str):
            raise ValueError("raw call checkpoint content must be a string")
        recorded_usage = _strict_usage(
            raw_call.get("token_usage") or {},
            label=f"{qid} checkpoint call {expected_index}",
        )
        provider_usage = _strict_usage(
            (raw_call.get("raw_response") or {}).get("usage") or {},
            label=f"{qid} checkpoint provider call {expected_index}",
        )
        if recorded_usage.to_dict() != provider_usage.to_dict():
            raise ValueError(
                "raw call checkpoint usage differs from provider raw usage"
            )
        _validate_transport_audit(
            raw_call,
            label=f"{qid} checkpoint call {expected_index}",
        )
        calls.append(dict(raw_call))
    _validate_generation_call_intents(
        run_dir,
        qid=qid,
        evidence_alias_map=evidence_alias_map,
        calls=calls,
    )
    return calls


def _validation_error_code(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "json_parse_error"
    message = str(exc)
    if "explicit 结论" in message or "conclusion must not be empty" in message:
        return "reasoning_missing_explicit_conclusion"
    if "conclusion" in message:
        return "answer_reasoning_conclusion_mismatch"
    if (
        ".answer_" in message
        or "numeric/date slot" in message
        or "percentage slot" in message
    ):
        return "answer_slot_format_error"
    if "response fields" in message or "must be an array" in message:
        return "schema_fields_error"
    return "answer_shape_error"


def _unobservable_usage_risk(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    explicit_429 = (
        isinstance(exc, requests.HTTPError)
        and response is not None
        and int(getattr(response, "status_code", 0)) == 429
    )
    return not explicit_429


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected an array")
    return [dict(row) for row in payload]


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _write_immutable_or_equal_json(
    path: Path,
    payload: Any,
    *,
    label: str,
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"{label} changed inside an immutable run")
        return
    _write_json(path, payload)


def _write_raw_ledger_append_only(
    path: Path,
    payload: dict[str, Any],
) -> None:
    new_calls = payload.get("calls")
    if not isinstance(new_calls, list):
        raise ValueError("raw call ledger calls must be an array")
    if path.exists():
        existing = _read_json_object(path, label="raw call ledger")
        if existing.get("qid") != payload.get("qid"):
            raise ValueError("raw call ledger qid changed")
        if existing.get("evidence_alias_map") != payload.get(
            "evidence_alias_map"
        ):
            raise ValueError("raw call ledger evidence fingerprint changed")
        if (
            existing.get("run_fingerprint") != payload.get("run_fingerprint")
            or existing.get("run_instance_id")
            != payload.get("run_instance_id")
        ):
            raise ValueError("raw call ledger run binding changed")
        if existing.get("ledger_state") in {
            "finalized_answer",
            "finalized_failure",
        }:
            if existing != payload:
                raise ValueError("finalized raw call ledger is immutable")
            return
        existing_calls = existing.get("calls")
        if not isinstance(existing_calls, list):
            raise ValueError("existing raw call ledger calls must be an array")
        if len(new_calls) < len(existing_calls):
            raise ValueError("raw call ledger cannot lose observed calls")
        if new_calls[: len(existing_calls)] != existing_calls:
            raise ValueError("raw call ledger can only append observed calls")
    _write_json(path, payload)


def _run_instance_binding(run_dir: Path) -> dict[str, str]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    payload = _read_json_object(config_path, label="run config")
    fingerprint = payload.get("fingerprint")
    run_instance_id = payload.get("run_instance_id")
    if (
        not isinstance(fingerprint, str)
        or not fingerprint
        or not isinstance(run_instance_id, str)
        or not run_instance_id
    ):
        raise ValueError("run config binding is invalid")
    return {
        "run_fingerprint": fingerprint,
        "run_instance_id": run_instance_id,
    }


def _validate_run_instance_binding(
    run_dir: Path,
    payload: Mapping[str, Any],
) -> None:
    expected = _run_instance_binding(run_dir)
    if expected and any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("raw call checkpoint run instance mismatch")


def _strict_usage(payload: Any, *, label: str) -> TokenUsage:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    values: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{label} {key} must be a non-negative integer"
            )
        values[key] = value
    if values["total_tokens"] != (
        values["prompt_tokens"] + values["completion_tokens"]
    ):
        raise ValueError(f"{label} total_tokens is inconsistent")
    return TokenUsage(**values)


def _validate_transport_audit(
    call: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, int]:
    if (
        "transport_attempt_count" not in call
        or "transport_rejections" not in call
    ):
        raise ValueError(f"{label} transport audit is missing")
    attempt_count = call["transport_attempt_count"]
    rejections = call["transport_rejections"]
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
        or not isinstance(rejections, list)
    ):
        raise ValueError(f"{label} transport audit is invalid")
    for index, rejection in enumerate(rejections, start=1):
        if (
            not isinstance(rejection, Mapping)
            or rejection.get("attempt_index") != index
            or rejection.get("status_code") != 429
            or rejection.get("pre_generation_rejection") is not True
            or rejection.get("token_usage_observed") is not False
        ):
            raise ValueError(f"{label} transport rejection is invalid")
    if attempt_count != len(rejections) + 1:
        raise ValueError(f"{label} transport attempt count is inconsistent")
    return attempt_count, len(rejections)


def _validate_failed_transport_audit(
    row: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, int]:
    attempt_count = row.get("failed_transport_attempt_count", 0)
    rejections = row.get("failed_transport_rejections", [])
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 0
        or not isinstance(rejections, list)
    ):
        raise ValueError(f"{label} audit is invalid")
    if attempt_count == 0:
        if rejections:
            raise ValueError(f"{label} has rejections without attempts")
        return 0, 0
    if (
        row.get("transport_retry_policy_version")
        != TRANSPORT_RETRY_POLICY_VERSION
        or not row.get("failed_transport_stage")
    ):
        raise ValueError(f"{label} binding is invalid")
    for index, rejection in enumerate(rejections, start=1):
        if (
            not isinstance(rejection, Mapping)
            or rejection.get("attempt_index") != index
            or rejection.get("status_code") != 429
            or rejection.get("pre_generation_rejection") is not True
            or rejection.get("token_usage_observed") is not False
        ):
            raise ValueError(f"{label} rejection is invalid")
    expected_attempts = len(rejections) + (
        1 if bool(row.get("unobservable_usage_risk")) else 0
    )
    if attempt_count != expected_attempts:
        raise ValueError(f"{label} attempt count is inconsistent")
    return attempt_count, len(rejections)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
