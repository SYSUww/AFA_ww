#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import BAnswer, load_b_questions, write_b_submission
from afa_agent.b_board.retrieval_llm_baseline import (
    PIPELINE_VERSION,
    PROMPT_VERSION,
    RETRIEVAL_POLICY_VERSION,
    build_answer_messages,
    build_answer_schema,
    load_domain_indexes,
    make_retriever,
    prepare_evidence_payload,
    public_run_fingerprint,
    retrieve_question_evidence,
    sha256_file,
    validate_answer_payload,
)
from afa_agent.b_board.submission_policy import require_allowed_submission_model
from afa_agent.client import OpenAICompatibleClient
from afa_agent.config import build_model_config
from afa_agent.models import TokenUsage


DEFAULT_SOURCE_ROOT = Path("/Users/abandon/Documents/AFA_ww")
DEFAULT_INDEX_ROOT = (
    DEFAULT_SOURCE_ROOT / "artifacts" / "preprocessed_loop_candidates" / "index"
)
DEFAULT_MODEL = "qwen3.7-plus-2026-05-26"


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
    parser.add_argument("--supplemental-weight", type=float, default=0.11)
    parser.add_argument("--max-queries-per-option", type=int, default=12)
    parser.add_argument("--max-hit-chars", type=int, default=1800)
    parser.add_argument("--max-evidence-chars", type=int, default=12000)
    parser.add_argument("--thinking-budget", type=int, default=2048)
    parser.add_argument("--max-format-retries", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
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
    fingerprint = public_run_fingerprint(
        {
            key: value
            for key, value in public_config.items()
            if key != "created_at"
        }
    )
    public_config = _initialize_run_dir(
        args.run_dir,
        public_config,
        fingerprint,
    )

    existing_answers = _read_rows(args.run_dir / "answers.json")
    existing_failures = _read_rows(args.run_dir / "failures.json")
    finished_qids = {
        str(row["qid"]) for row in [*existing_answers, *existing_failures]
    }
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
                    "unobservable_usage_risk": False,
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
) -> dict[str, Any]:
    retrieval = retrieve_question_evidence(
        question,
        retriever=retriever,
        all_doc_ids=all_doc_ids,
        per_query_top_k=args.per_query_top_k,
        final_top_k=args.final_top_k,
        supplemental_weight=args.supplemental_weight,
        max_queries_per_option=args.max_queries_per_option,
        max_doc_candidates=args.max_doc_candidates,
    )
    evidence = prepare_evidence_payload(
        retrieval,
        max_hit_chars=args.max_hit_chars,
        max_total_chars=args.max_evidence_chars,
    )
    evidence_alias_map = _evidence_alias_map(evidence)
    schema = build_answer_schema(question)
    validation_error = ""
    previous_response = ""
    calls: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    for attempt in range(args.max_format_retries + 1):
        try:
            messages = build_answer_messages(
                question,
                evidence,
                validation_error=validation_error,
                previous_response=previous_response,
            )
        except Exception as exc:
            return {
                "qid": question.qid,
                "domain": question.domain,
                "answer_format": question.answer_format,
                "question_type": question.type,
                "pipeline_version": PIPELINE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
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
            response = client.chat_json(
                messages,
                response_schema=schema,
                schema_name="b_retrieval_llm_final_submission_v2",
                extra_body={
                    "enable_thinking": args.thinking_budget > 0,
                    **(
                        {"thinking_budget": args.thinking_budget}
                        if args.thinking_budget > 0
                        else {}
                    ),
                },
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
                "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
                "retrieval": retrieval,
                "evidence_items": evidence,
                "evidence_alias_map": evidence_alias_map,
                "calls": calls,
                "token_usage": _sum_call_usage(calls),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "provider_error": provider_error,
                "unobservable_usage_risk": _unobservable_usage_risk(exc),
            }
        call = {
            "call_index": attempt + 1,
            "purpose": "initial_answer" if attempt == 0 else "format_consistency_retry",
            "model_name": client.config.model_name,
            "response_format_mode": response.response_format_mode,
            "messages": messages,
            "response_schema": schema,
            "raw_response": response.raw_payload,
            "content": response.content,
            "token_usage": response.token_usage.to_dict(),
        }
        calls.append(call)
        _checkpoint_raw_calls(
            args.run_dir,
            question.qid,
            calls,
            evidence_alias_map,
        )
        previous_response = response.content
        try:
            candidate = json.loads(response.content)
            if not isinstance(candidate, dict):
                raise ValueError("model response must be a JSON object")
            parsed = validate_answer_payload(question, candidate)
            break
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            validation_error = _validation_error_code(exc)
    usage = _sum_call_usage(calls)
    base = {
        "qid": question.qid,
        "domain": question.domain,
        "answer_format": question.answer_format,
        "question_type": question.type,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
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
        "format_retry_count": max(0, len(calls) - 1),
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
    )
    for field in positive_fields:
        if int(getattr(args, field)) < 1:
            raise ValueError(f"{field} must be positive")
    if args.thinking_budget < 0:
        raise ValueError("thinking_budget must not be negative")
    if args.max_format_retries not in {0, 1}:
        raise ValueError("max_format_retries must be zero or one")
    if not 0 <= args.supplemental_weight <= 1:
        raise ValueError("supplemental_weight must be between zero and one")


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
    source_files = [
        ROOT / "src/afa_agent/b_board/retrieval_llm_baseline.py",
        ROOT / "src/afa_agent/retrieval_query.py",
        ROOT / "scripts/run_b_retrieval_llm_baseline.py",
    ]
    return {
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "model_name": model_config.model_name,
            "temperature": model_config.temperature,
            "structured_output_mode": "native_json_schema_strict",
            "thinking_budget": args.thinking_budget,
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
            "supplemental_weight": args.supplemental_weight,
            "max_queries_per_option": args.max_queries_per_option,
            "max_hit_chars": args.max_hit_chars,
            "max_evidence_chars": args.max_evidence_chars,
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
            "transport_retry": "explicit HTTP 429 pre-generation rejection only",
            "read_timeout_retry": False,
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
            "document_scope": "all documents in the question domain index",
        },
    }


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
        return existing_config
    if any(run_dir.iterdir()):
        raise ValueError("new run directory must be empty")
    (run_dir / "raw_calls").mkdir()
    (run_dir / "retrieval").mkdir()
    _write_json(
        config_path,
        {
            "fingerprint": fingerprint,
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
    _write_json(
        run_dir / "retrieval" / f"{qid}.json",
        result.get("retrieval", {}),
    )
    _write_json(
        run_dir / "raw_calls" / f"{qid}.json",
        {
            "qid": qid,
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
            "postprocessing": {
                "answer_modified": False,
                "reasoning_modified": False,
                "csv_escaping_only": True,
            },
        },
    )
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
    for row in [*answers, *failures]:
        row_usage = row.get("token_usage") or {}
        usage.add(
            TokenUsage(
                prompt_tokens=int(row_usage.get("prompt_tokens", 0)),
                completion_tokens=int(row_usage.get("completion_tokens", 0)),
                total_tokens=int(row_usage.get("total_tokens", 0)),
            )
        )
        call_path = run_dir / "raw_calls" / f"{row['qid']}.json"
        if call_path.exists():
            call_count = len(
                json.loads(call_path.read_text(encoding="utf-8")).get("calls", [])
            )
            raw_call_count += call_count
            format_retry_count += max(0, call_count - 1)
    finished = answered | failed
    expected = set(question_order)
    unobservable_usage_risk = any(
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
        "token_usage": usage.to_dict(),
        "all_observed_usage_from_provider_raw_fields": True,
        "unobservable_usage_risk": unobservable_usage_risk,
        "answer_blind_contract": public_config["answer_blind_contract"],
        "submission_eligible": False,
        "submission_ineligibility_reasons": [
            "research_only_answer_blind_baseline",
            "accuracy_reference_is_evaluation_only",
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
        }
        for item in evidence
    ]


def _checkpoint_raw_calls(
    run_dir: Path,
    qid: str,
    calls: list[dict[str, Any]],
    evidence_alias_map: list[dict[str, Any]],
) -> None:
    _write_json(
        run_dir / "raw_calls" / f"{qid}.json",
        {
            "qid": qid,
            "evidence_alias_map": evidence_alias_map,
            "calls": calls,
            "checkpoint_only": True,
        },
    )


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
    return isinstance(
        exc,
        (
            requests.ReadTimeout,
            ValueError,
            KeyError,
            TypeError,
        ),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected an array")
    return [dict(row) for row in payload]


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
