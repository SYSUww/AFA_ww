#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import (
    BAnswer,
    load_b_questions,
    validate_b_answer,
    write_b_submission,
)
from afa_agent.b_board.submission_policy import is_allowed_submission_model
from afa_agent.client import TRANSPORT_RETRY_POLICY_VERSION
from afa_agent.b_board.retrieval_llm_baseline import (
    is_verified_calculation_path,
    public_run_config_fingerprint,
    suspicious_generation_prompt_literals,
    validate_answer_payload,
    validate_frozen_answer_reasoning_payload,
)
from afa_agent.b_board.retrieval_llm_calculation import (
    validate_verified_calculation_checkpoint,
    validate_verified_frozen_answer_reasoning_payload,
)


DEFAULT_SOURCE_ROOT = Path("/Users/abandon/Documents/AFA_ww")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble disjoint immutable retrieval+Qwen runs into one audited "
            "100-question submission without changing answers or reasoning"
        )
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = assemble_submission(
        source_run_dirs=args.run_dir,
        output_dir=args.output_dir,
        question_root=args.question_root,
        submission_template=args.submission_template,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def assemble_submission(
    *,
    source_run_dirs: Iterable[Path],
    output_dir: Path,
    question_root: Path,
    submission_template: Path,
) -> dict[str, Any]:
    run_dirs = [Path(path).resolve() for path in source_run_dirs]
    if len(run_dirs) < 2:
        raise ValueError("at least two source runs are required")
    if len(run_dirs) != len(set(run_dirs)):
        raise ValueError("source run directories must be unique")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be new or empty")

    questions = load_b_questions(question_root, submission_template)
    question_by_qid = {question.qid: question for question in questions}
    expected_qids = [question.qid for question in questions]
    final_answers: dict[str, dict[str, Any]] = {}
    call_history: dict[str, list[dict[str, Any]]] = {}
    evidence_aliases: dict[str, list[dict[str, Any]]] = {}
    retrieval_payloads: dict[str, Any] = {}
    retrieval_history: dict[str, list[dict[str, Any]]] = {}
    source_records: list[dict[str, Any]] = []
    terminal_transport_attempt_count = 0
    terminal_transport_rejection_count = 0
    common_model_contract: dict[str, Any] | None = None
    pipeline_versions: set[str] = set()
    prompt_versions: set[str] = set()
    retrieval_policy_versions: set[str] = set()

    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "run_manifest.json")
        source_config = _audit_source_run_configuration(run_dir)
        answers = _read_rows(run_dir / "answers.json")
        failures = _read_rows(run_dir / "failures.json")
        result_rows = [*answers, *failures]
        if manifest.get("status") not in {"complete", "incomplete"}:
            raise ValueError(f"{run_dir}: source run has not finished its scope")
        model_name = str((manifest.get("model") or {}).get("model_name", ""))
        if not is_allowed_submission_model(model_name):
            raise ValueError(f"{run_dir}: model is not submission-allowed")
        answer_blind = manifest.get("answer_blind_contract") or {}
        required_blind_flags = {
            "qid_in_model_messages": False,
            "reference_loaded_by_generation": False,
            "official_locks_loaded_by_generation": False,
            "solver_or_rule_layer_used": False,
            "fixed_locator_used": False,
        }
        for key, expected in required_blind_flags.items():
            if answer_blind.get(key) is not expected:
                raise ValueError(f"{run_dir}: answer-blind contract failed at {key}")
        if answer_blind.get("research_only_strategy") is not False:
            raise ValueError(f"{run_dir}: research-only strategy is not assemblable")

        contract = _source_contract(manifest, result_rows)
        model_contract = {
            key: contract[key]
            for key in (
                "model_name",
                "temperature",
                "structured_output_mode",
                "thinking_budget",
            )
        }
        if common_model_contract is None:
            common_model_contract = model_contract
        elif model_contract != common_model_contract:
            raise ValueError(
                f"{run_dir}: model or structured-generation contract drift"
            )
        pipeline_versions.add(contract["pipeline_version"])
        prompt_versions.add(contract["prompt_version"])
        retrieval_policy_versions.add(contract["retrieval_policy_version"])

        run_qids = [str(qid) for qid in ((manifest.get("scope") or {}).get("qids") or [])]
        answer_qids = [str(row.get("qid", "")) for row in answers]
        failure_qids = [str(row.get("qid", "")) for row in failures]
        if set(answer_qids) & set(failure_qids):
            raise ValueError(f"{run_dir}: qid appears in both answers and failures")
        if set(run_qids) != set(answer_qids) | set(failure_qids):
            raise ValueError(f"{run_dir}: manifest scope and result rows disagree")

        run_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        run_call_count = 0
        run_terminal_transport_attempt_count = 0
        run_terminal_transport_rejection_count = 0
        for row in result_rows:
            qid = str(row["qid"])
            if qid not in question_by_qid:
                raise ValueError(f"{run_dir}: unknown qid {qid}")
            is_answered = qid in set(answer_qids)
            raw_artifact = _audit_result_row(
                question=question_by_qid[qid],
                row=row,
                raw_call_path=run_dir / "raw_calls" / f"{qid}.json",
                run_dir=run_dir,
                expected_model=model_name,
                is_answered=is_answered,
            )
            for field in run_usage:
                run_usage[field] += int((row.get("token_usage") or {}).get(field, 0))
            calls = raw_artifact.get("calls", [])
            run_call_count += len(calls)
            if not is_answered:
                failed_attempts, failed_rejections = (
                    _audit_failed_transport(row, qid=qid)
                )
                run_terminal_transport_attempt_count += failed_attempts
                run_terminal_transport_rejection_count += (
                    failed_rejections
                )

            aliases = raw_artifact.get("evidence_alias_map", [])
            retrieval = _read_json(run_dir / "retrieval" / f"{qid}.json")
            retrieval_digest = _sha256_json(retrieval)
            retrieval_history.setdefault(qid, []).append(
                {
                    "source_run": run_dir.name,
                    "retrieval_sha256": retrieval_digest,
                }
            )
            if qid not in evidence_aliases or is_answered:
                evidence_aliases[qid] = aliases
                retrieval_payloads[qid] = retrieval

            history = call_history.setdefault(qid, [])
            for source_call in calls:
                combined_call = dict(source_call)
                combined_call["source_run"] = run_dir.name
                combined_call["source_call_index"] = int(
                    source_call.get("call_index", len(history) + 1)
                )
                combined_call["call_index"] = len(history) + 1
                if history and source_call.get("purpose") == "initial_answer":
                    combined_call["purpose"] = "recovery_initial_answer"
                combined_call["evidence_alias_map"] = aliases
                combined_call["retrieval_sha256"] = retrieval_digest
                history.append(combined_call)

            if is_answered:
                # Source runs are ordered. A later answer may intentionally
                # supersede an earlier structurally valid answer after a
                # generic retrieval/prompt defect was discovered. Every call
                # from both runs remains in the cumulative usage ledger.
                final_answers[qid] = dict(row)

        if run_usage != manifest.get("token_usage"):
            raise ValueError(f"{run_dir}: result usage does not match manifest")
        if run_call_count != int(manifest.get("raw_call_count", -1)):
            raise ValueError(f"{run_dir}: raw call count does not match manifest")
        call_transport_attempts = sum(
            int(call["transport_attempt_count"])
            for row in result_rows
            for call in _read_json(
                run_dir / "raw_calls" / f"{row['qid']}.json"
            ).get("calls", [])
        )
        call_transport_rejections = sum(
            len(call["transport_rejections"])
            for row in result_rows
            for call in _read_json(
                run_dir / "raw_calls" / f"{row['qid']}.json"
            ).get("calls", [])
        )
        if (
            call_transport_attempts
            + run_terminal_transport_attempt_count
            != int(manifest.get("transport_attempt_count", -1))
            or call_transport_rejections
            + run_terminal_transport_rejection_count
            != int(manifest.get("transport_rejection_count", -1))
        ):
            raise ValueError(
                f"{run_dir}: transport audit does not match manifest"
            )
        terminal_transport_attempt_count += (
            run_terminal_transport_attempt_count
        )
        terminal_transport_rejection_count += (
            run_terminal_transport_rejection_count
        )
        source_records.append(
            {
                "run_dir": str(run_dir),
                "run_manifest_sha256": _sha256_file(run_dir / "run_manifest.json"),
                "answers_sha256": _sha256_file(run_dir / "answers.json"),
                "failures_sha256": _sha256_file(run_dir / "failures.json"),
                "question_count": len(result_rows),
                "answered_question_count": len(answers),
                "failed_question_count": len(failures),
                "raw_call_count": run_call_count,
                "terminal_transport_attempt_count": (
                    run_terminal_transport_attempt_count
                ),
                "terminal_transport_rejection_count": (
                    run_terminal_transport_rejection_count
                ),
                "token_usage": run_usage,
                "source_config_sha256": _sha256_json(source_config),
            }
        )

    if set(final_answers) != set(expected_qids):
        missing = sorted(set(expected_qids) - set(final_answers))
        extra = sorted(set(final_answers) - set(expected_qids))
        raise ValueError(f"assembled qid coverage mismatch: missing={missing}, extra={extra}")
    assert common_model_contract is not None

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_calls").mkdir()
    (output_dir / "retrieval").mkdir()
    ordered_rows: list[dict[str, Any]] = []
    for qid in expected_qids:
        calls = call_history[qid]
        cumulative_usage = {
            field: sum(int(call["token_usage"][field]) for call in calls)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        row = dict(final_answers[qid])
        row["token_usage"] = cumulative_usage
        row["format_retry_count"] = sum(
            call.get("purpose") == "format_consistency_retry"
            for call in calls
        )
        ordered_rows.append(row)
        final_content = str(calls[-1]["content"])
        raw_artifact = {
            "qid": qid,
            "evidence_alias_map": evidence_aliases[qid],
            "retrieval_history": retrieval_history[qid],
            "calls": calls,
            "final_call_index": len(calls),
            "final_content_sha256": _sha256_text(final_content),
            "submitted_answer_parts_sha256": _sha256_text(
                json.dumps(
                    row["answer_parts"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
            "submitted_reasoning_sha256": _sha256_text(str(row["reasoning"])),
            "postprocessing": {
                "answer_modified": False,
                "reasoning_modified": False,
                "csv_escaping_only": True,
            },
        }
        _write_json(
            output_dir / "raw_calls" / f"{qid}.json",
            raw_artifact,
        )
        _write_json(
            output_dir / "retrieval" / f"{qid}.json",
            retrieval_payloads[qid],
        )
    _write_json(output_dir / "answers.json", ordered_rows)
    _write_json(output_dir / "failures.json", [])

    submission_answers = [
        BAnswer(
            qid=qid,
            answer_parts=tuple(str(part) for part in row["answer_parts"]),
            reasoning=str(row["reasoning"]),
            prompt_tokens=int(row["token_usage"]["prompt_tokens"]),
            completion_tokens=int(row["token_usage"]["completion_tokens"]),
            total_tokens=int(row["token_usage"]["total_tokens"]),
        )
        for qid, row in zip(expected_qids, ordered_rows)
    ]
    submission_path = output_dir / "submit.csv"
    write_b_submission(
        submission_path,
        questions,
        submission_answers,
        audit_ready=True,
    )
    token_usage = {
        field: sum(int(row["token_usage"][field]) for row in ordered_rows)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    raw_call_count = sum(len(call_history[qid]) for qid in expected_qids)
    call_purpose_counts: dict[str, int] = {}
    for calls in call_history.values():
        for call in calls:
            purpose = str(call.get("purpose", "unknown"))
            call_purpose_counts[purpose] = (
                call_purpose_counts.get(purpose, 0) + 1
            )
    format_retry_count = call_purpose_counts.get(
        "format_consistency_retry",
        0,
    )
    transport_attempt_count = terminal_transport_attempt_count + sum(
        int(call.get("transport_attempt_count", 1))
        for calls in call_history.values()
        for call in calls
    )
    transport_rejection_count = terminal_transport_rejection_count + sum(
        len(call.get("transport_rejections", []))
        for calls in call_history.values()
        for call in calls
    )
    manifest = {
        "run_id": output_dir.name,
        "runner": "b_retrieval_llm_immutable_run_assembler_v1",
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "model_name": common_model_contract["model_name"],
            "temperature": common_model_contract["temperature"],
            "structured_output_mode": common_model_contract[
                "structured_output_mode"
            ],
            "thinking_budget": common_model_contract["thinking_budget"],
        },
        "pipeline_versions": sorted(pipeline_versions),
        "prompt_versions": sorted(prompt_versions),
        "retrieval_policy_versions": sorted(retrieval_policy_versions),
        "expected_question_count": len(expected_qids),
        "answered_question_count": len(expected_qids),
        "failed_question_count": 0,
        "scope": {"question_count": len(expected_qids), "qids": expected_qids},
        "raw_call_count": raw_call_count,
        "format_retry_count": format_retry_count,
        "call_purpose_counts": dict(sorted(call_purpose_counts.items())),
        "transport_attempt_count": transport_attempt_count,
        "transport_rejection_count": transport_rejection_count,
        "token_usage": token_usage,
        "all_observed_usage_from_provider_raw_fields": True,
        "unobservable_usage_risk": False,
        "answer_blind_contract": {
            "qid_in_model_messages": False,
            "reference_loaded_by_generation": False,
            "official_locks_loaded_by_generation": False,
            "solver_or_rule_layer_used": False,
            "fixed_locator_used": False,
            "document_scope": "all documents in each question domain index",
        },
        "assembly": {
            "answers_modified": False,
            "reasoning_modified": False,
            "all_source_call_usage_aggregated": True,
            "retrieval_drift_qids": [
                qid
                for qid in expected_qids
                if len(
                    {
                        item["retrieval_sha256"]
                        for item in retrieval_history[qid]
                    }
                )
                > 1
            ],
            "csv_escaping_only": True,
            "source_runs": source_records,
        },
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": _sha256_file(submission_path),
        "submission_eligible": True,
        "submission_ineligibility_reasons": [],
        "official_submission_count": 0,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _source_contract(
    manifest: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not result_rows:
        raise ValueError("source run has no result rows")
    versions = {
        (
            str(row.get("pipeline_version", "")),
            str(row.get("prompt_version", "")),
            str(row.get("retrieval_policy_version", "")),
        )
        for row in result_rows
    }
    if len(versions) != 1:
        raise ValueError("source run contains version drift")
    pipeline, prompt, retrieval = versions.pop()
    model = manifest.get("model") or {}
    return {
        "model_name": str(model.get("model_name", "")),
        "temperature": float(model.get("temperature", -1)),
        "structured_output_mode": str(model.get("structured_output_mode", "")),
        "thinking_budget": int(model.get("thinking_budget", -1)),
        "pipeline_version": pipeline,
        "prompt_version": prompt,
        "retrieval_policy_version": retrieval,
    }


def _audit_result_row(
    *,
    question: Any,
    row: dict[str, Any],
    raw_call_path: Path,
    run_dir: Path,
    expected_model: str,
    is_answered: bool,
) -> dict[str, Any]:
    usage = row.get("token_usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", -1))
    completion_tokens = int(usage.get("completion_tokens", -1))
    total_tokens = int(usage.get("total_tokens", -1))
    if min(prompt_tokens, completion_tokens, total_tokens) < 0:
        raise ValueError(f"{question.qid}: missing token usage")
    if prompt_tokens + completion_tokens != total_tokens:
        raise ValueError(f"{question.qid}: answer usage arithmetic mismatch")
    if is_answered:
        answer = BAnswer(
            qid=question.qid,
            answer_parts=tuple(str(part) for part in row.get("answer_parts", [])),
            reasoning=str(row.get("reasoning", "")),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        validate_b_answer(question, answer)
        if not answer.reasoning.strip():
            raise ValueError(f"{question.qid}: reasoning is empty")

    raw = _read_json(raw_call_path)
    run_config_payload = _read_json(run_dir / "run_config.json")
    run_binding = {
        "run_fingerprint": str(run_config_payload.get("fingerprint", "")),
        "run_instance_id": str(run_config_payload.get("run_instance_id", "")),
    }
    if any(raw.get(key) != value for key, value in run_binding.items()):
        raise ValueError(f"{question.qid}: raw run binding mismatch")
    calls = raw.get("calls", [])
    if is_answered and not calls:
        raise ValueError(f"{question.qid}: raw calls are missing")
    if not is_answered and row.get("unobservable_usage_risk") is not False:
        raise ValueError(
            f"{question.qid}: failed source has unobservable usage"
        )
    if not is_answered:
        _audit_failed_transport(row, qid=question.qid)
    summed = {
        field: sum(int(call["token_usage"][field]) for call in calls)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if summed != usage:
        raise ValueError(f"{question.qid}: raw call usage mismatch")
    for call in calls:
        if call.get("model_name") != expected_model:
            raise ValueError(f"{question.qid}: model drift")
        if call.get("response_format_mode") != "native_json_schema_strict":
            raise ValueError(f"{question.qid}: structured output mode drift")
        serialized_messages = json.dumps(call.get("messages"), ensure_ascii=False)
        if question.qid in serialized_messages:
            raise ValueError(f"{question.qid}: qid leaked into model messages")
        if any(
            fragment in serialized_messages
            for fragment in ("pseudo99", "官网答案", "official_answer_locks")
        ):
            raise ValueError(f"{question.qid}: prohibited answer reference in messages")
        provider_usage = (call.get("raw_response") or {}).get("usage") or {}
        if {
            field: int(provider_usage.get(field, -1))
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        } != {
            field: int((call.get("token_usage") or {}).get(field, -1))
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        }:
            raise ValueError(
                f"{question.qid}: call usage differs from provider raw usage"
            )
        if (
            "transport_attempt_count" not in call
            or "transport_rejections" not in call
        ):
            raise ValueError(
                f"{question.qid}: transport audit is missing"
            )
        attempt_count = call["transport_attempt_count"]
        rejections = call["transport_rejections"]
        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or attempt_count < 1
            or not isinstance(rejections, list)
            or attempt_count != len(rejections) + 1
            or any(
                not isinstance(item, dict)
                or item.get("attempt_index") != rejection_index
                or item.get("status_code") != 429
                or item.get("pre_generation_rejection") is not True
                or item.get("token_usage_observed") is not False
                for rejection_index, item in enumerate(
                    rejections, start=1
                )
            )
        ):
            raise ValueError(f"{question.qid}: transport audit is invalid")
    _verify_call_intents(
        run_dir=run_dir,
        qid=question.qid,
        aliases=list(raw.get("evidence_alias_map") or []),
        calls=calls,
        run_binding=run_binding,
        terminal_failure=(row if not is_answered else None),
    )
    if is_answered:
        final_payload = _reconstruct_final_payload(
            question=question,
            calls=calls,
            run_dir=run_dir,
        )
        if final_payload["answer_parts"] != row.get("answer_parts"):
            raise ValueError(f"{question.qid}: final answer differs from raw response")
        if final_payload["reasoning"] != row.get("reasoning"):
            raise ValueError(f"{question.qid}: final reasoning differs from raw response")
        if raw.get("postprocessing") != {
            "answer_modified": False,
            "reasoning_modified": False,
            "csv_escaping_only": True,
        }:
            raise ValueError(f"{question.qid}: postprocessing contract mismatch")
        generation = dict(
            ((run_config_payload.get("config") or {}).get("generation") or {})
        )
        effective_verified_calculation = is_verified_calculation_path(
            str(generation.get("calculation_mode", "")),
            question.answer_format,
        )
        if (
            not effective_verified_calculation
            and generation.get("output_contract") == "joint"
        ):
            _verify_direct_frozen_answer(
                run_dir=run_dir,
                question=question,
                aliases=list(raw.get("evidence_alias_map") or []),
                calls=calls,
                answer_parts=list(row.get("answer_parts") or []),
                run_binding=run_binding,
            )
        elif effective_verified_calculation:
            checkpoint_path = (
                run_dir / "frozen_answers" / f"{question.qid}.json"
            )
            checkpoint_record = row.get("frozen_answer_checkpoint")
            evidence = row.get("evidence_items")
            if (
                not isinstance(checkpoint_record, dict)
                or not isinstance(evidence, list)
                or checkpoint_record.get("sha256")
                != _sha256_file(checkpoint_path)
            ):
                raise ValueError(
                    f"{question.qid}: verified checkpoint record is invalid"
                )
            checkpoint = validate_verified_calculation_checkpoint(
                checkpoint_path,
                question=question,
                evidence=evidence,
                expected_model_name=expected_model,
                expected_run_binding=run_binding,
            )
            checkpoint_calls = checkpoint.get("calls")
            if (
                not isinstance(checkpoint_calls, list)
                or calls[: len(checkpoint_calls)] != checkpoint_calls
            ):
                raise ValueError(
                    f"{question.qid}: verified checkpoint call prefix drift"
                )
            if checkpoint.get("answer_parts") != row.get("answer_parts"):
                raise ValueError(
                    f"{question.qid}: verified checkpoint answer drift"
                )
    return raw


def _reconstruct_final_payload(
    *,
    question: Any,
    calls: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    final_call = calls[-1]
    payload = json.loads(str(final_call["content"]))
    purpose = str(final_call.get("purpose", ""))
    if purpose == "reasoning_only_retry_from_frozen_answer":
        return validate_frozen_answer_reasoning_payload(
            list(final_call.get("frozen_answer_parts") or []),
            payload,
        )
    if purpose == "verified_calculation_reasoning":
        checkpoint = _read_json(
            run_dir / "frozen_answers" / f"{question.qid}.json"
        )
        return validate_verified_frozen_answer_reasoning_payload(
            list(checkpoint.get("answer_parts") or []),
            payload,
        )
    return validate_answer_payload(question, payload)


def _verify_call_intents(
    *,
    run_dir: Path,
    qid: str,
    aliases: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    run_binding: dict[str, str],
    terminal_failure: Mapping[str, Any] | None = None,
) -> None:
    alias_sha256 = _sha256_text(
        json.dumps(
            aliases,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    intents: dict[int, dict[str, Any]] = {}
    for path in sorted((run_dir / "call_intents").glob(f"{qid}.*.json")):
        payload = _read_json(path)
        call_index = payload.get("call_index")
        if (
            payload.get("qid") != qid
            or isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or path.name != f"{qid}.{call_index}.json"
            or call_index in intents
            or payload.get("evidence_alias_map_sha256") != alias_sha256
            or payload.get("state") != "provider_call_started"
            or any(
                payload.get(key) != value
                for key, value in run_binding.items()
            )
        ):
            raise ValueError(f"{qid}: call intent binding mismatch")
        intents[call_index] = payload
    for call in calls:
        call_index = int(call.get("call_index", -1))
        if (
            call_index not in intents
            or intents[call_index].get("purpose") != call.get("purpose")
        ):
            raise ValueError(f"{qid}: call has no matching intent")
    unmatched = sorted(
        set(intents)
        - {int(call.get("call_index", -1)) for call in calls}
    )
    if not unmatched:
        if (
            terminal_failure is not None
            and int(
                terminal_failure.get(
                    "failed_transport_attempt_count", 0
                )
            )
            > 0
        ):
            raise ValueError(
                f"{qid}: failed transport has no unmatched call intent"
            )
        return
    failed_stage = (
        str(terminal_failure.get("failed_transport_stage", ""))
        if terminal_failure is not None
        else ""
    )
    expected_purpose = {
        "answer": "calculation_plan",
        "reasoning": "verified_calculation_reasoning",
    }.get(failed_stage, failed_stage)
    if (
        terminal_failure is None
        or len(unmatched) != 1
        or unmatched[0] != len(calls) + 1
        or intents[unmatched[0]].get("purpose") != expected_purpose
        or int(
            terminal_failure.get("failed_transport_attempt_count", 0)
        )
        < 1
    ):
        raise ValueError(f"{qid}: call intent coverage mismatch")


def _audit_failed_transport(
    row: Mapping[str, Any],
    *,
    qid: str,
) -> tuple[int, int]:
    attempt_count = row.get("failed_transport_attempt_count", 0)
    rejections = row.get("failed_transport_rejections", [])
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 0
        or not isinstance(rejections, list)
    ):
        raise ValueError(f"{qid}: failed transport audit is invalid")
    if attempt_count == 0:
        if rejections:
            raise ValueError(
                f"{qid}: failed transport has rejections without attempts"
            )
        return 0, 0
    if (
        row.get("unobservable_usage_risk") is not False
        or row.get("transport_retry_policy_version")
        != TRANSPORT_RETRY_POLICY_VERSION
        or not row.get("failed_transport_stage")
        or attempt_count != len(rejections)
    ):
        raise ValueError(f"{qid}: failed transport binding is invalid")
    for index, rejection in enumerate(rejections, start=1):
        if (
            not isinstance(rejection, Mapping)
            or rejection.get("attempt_index") != index
            or rejection.get("status_code") != 429
            or rejection.get("pre_generation_rejection") is not True
            or rejection.get("token_usage_observed") is not False
        ):
            raise ValueError(
                f"{qid}: failed transport rejection is invalid"
            )
    return attempt_count, len(rejections)


def _verify_direct_frozen_answer(
    *,
    run_dir: Path,
    question: Any,
    aliases: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    answer_parts: list[str],
    run_binding: dict[str, str],
) -> None:
    payload = _read_json(
        run_dir / "frozen_answers" / f"{question.qid}.json"
    )
    call_index = payload.get("answer_call_index")
    if (
        payload.get("checkpoint_version") != "direct_model_answer_v1"
        or payload.get("checkpoint_kind") != "direct_model_answer"
        or payload.get("qid") != question.qid
        or payload.get("evidence_alias_map") != aliases
        or payload.get("answer_parts") != answer_parts
        or any(
            payload.get(key) != value for key, value in run_binding.items()
        )
        or isinstance(call_index, bool)
        or not isinstance(call_index, int)
        or call_index < 1
        or call_index > len(calls)
    ):
        raise ValueError(f"{question.qid}: direct frozen answer is invalid")
    answer_call = calls[call_index - 1]
    if (
        payload.get("answer_content_sha256")
        != _sha256_text(str(answer_call.get("content", "")))
        or answer_call.get("purpose")
        == "reasoning_only_retry_from_frozen_answer"
        or json.loads(str(answer_call.get("content", ""))).get(
            "answer_parts"
        )
        != answer_parts
    ):
        raise ValueError(f"{question.qid}: direct frozen answer drift")


def _audit_source_run_configuration(run_dir: Path) -> dict[str, Any]:
    payload = _read_json(run_dir / "run_config.json")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{run_dir}: run config is missing")
    expected_fingerprint = public_run_config_fingerprint(config)
    if payload.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"{run_dir}: run config fingerprint is invalid")
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"{run_dir}: manifest fingerprint is invalid")
    retrieval = config.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError(f"{run_dir}: retrieval config is missing")
    if retrieval.get("evidence_quota_strategy") != "primary_guard":
        raise ValueError(f"{run_dir}: evidence quota strategy is research-only")
    if retrieval.get("research_only_strategy") is not False:
        raise ValueError(f"{run_dir}: research-only retrieval flag is not false")
    generation = config.get("generation")
    blind_contract = config.get("answer_blind_contract")
    if not isinstance(generation, dict) or not isinstance(
        blind_contract, dict
    ):
        raise ValueError(f"{run_dir}: generation contract is missing")
    calculation_mode = generation.get("calculation_mode")
    if calculation_mode not in {"direct", "verified"}:
        raise ValueError(f"{run_dir}: unsupported calculation mode")
    if bool(
        blind_contract.get("deterministic_calculation_executor_used")
    ) != (calculation_mode == "verified"):
        raise ValueError(
            f"{run_dir}: calculation executor contract is inconsistent"
        )

    source_files = (
        (config.get("input_sha256") or {}).get("source_files") or {}
    )
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError(f"{run_dir}: generation source fingerprint is missing")
    prohibited_import_fragments = (
        ".solver",
        "candidate_scorecard",
        "official_answer",
        "pseudo_accuracy",
    )
    qid_pattern = re.compile(r"\b(?:fc|fin|ins|res|reg)_b_\d{3}\b")
    for relative, expected_sha256 in sorted(source_files.items()):
        path = (ROOT / str(relative)).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"{run_dir}: invalid generation source {relative}")
        if _sha256_file(path) != str(expected_sha256):
            raise ValueError(f"{run_dir}: generation source drifted at {relative}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        literals: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(str(node.module or ""))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.extend(qid_pattern.findall(node.value))
        if any(
            fragment in item
            for item in imports
            for fragment in prohibited_import_fragments
        ):
            raise ValueError(f"{run_dir}: prohibited generation import in {relative}")
        if literals:
            raise ValueError(f"{run_dir}: hardcoded B QID in {relative}")
        suspicious_prompt_literals = _suspicious_prompt_literals(tree)
        if suspicious_prompt_literals:
            raise ValueError(
                f"{run_dir}: fixed prompt literal risk in {relative}: "
                + ",".join(suspicious_prompt_literals)
            )
    return config


def _suspicious_prompt_literals(tree: ast.AST) -> list[str]:
    return suspicious_generation_prompt_literals(tree)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON array")
    return [dict(row) for row in payload]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
