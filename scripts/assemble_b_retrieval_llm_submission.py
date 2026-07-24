#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import (
    BAnswer,
    load_b_questions,
    validate_b_answer,
    write_b_submission,
)
from afa_agent.b_board.submission_policy import is_allowed_submission_model


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
    common_model_contract: dict[str, Any] | None = None
    pipeline_versions: set[str] = set()
    prompt_versions: set[str] = set()
    retrieval_policy_versions: set[str] = set()

    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "run_manifest.json")
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
        for row in result_rows:
            qid = str(row["qid"])
            if qid not in question_by_qid:
                raise ValueError(f"{run_dir}: unknown qid {qid}")
            is_answered = qid in set(answer_qids)
            raw_artifact = _audit_result_row(
                question=question_by_qid[qid],
                row=row,
                raw_call_path=run_dir / "raw_calls" / f"{qid}.json",
                expected_model=model_name,
                is_answered=is_answered,
            )
            for field in run_usage:
                run_usage[field] += int((row.get("token_usage") or {}).get(field, 0))
            calls = raw_artifact.get("calls", [])
            run_call_count += len(calls)

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
                "token_usage": run_usage,
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
        row["format_retry_count"] = max(0, len(calls) - 1)
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
    format_retry_count = raw_call_count - len(expected_qids)
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
    calls = raw.get("calls", [])
    if not calls:
        raise ValueError(f"{question.qid}: raw calls are missing")
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
    if is_answered:
        final_payload = json.loads(str(calls[-1]["content"]))
        if final_payload.get("answer_parts") != row.get("answer_parts"):
            raise ValueError(f"{question.qid}: final answer differs from raw response")
        if final_payload.get("reasoning") != row.get("reasoning"):
            raise ValueError(f"{question.qid}: final reasoning differs from raw response")
        if raw.get("postprocessing") != {
            "answer_modified": False,
            "reasoning_modified": False,
            "csv_escaping_only": True,
        }:
            raise ValueError(f"{question.qid}: postprocessing contract mismatch")
    return raw


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
