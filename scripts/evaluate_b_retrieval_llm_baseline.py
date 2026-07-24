#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.candidate_scorecard import answer_parts_equivalent
from afa_agent.b_board.io import load_b_questions
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
from afa_agent.b_board.scoring import token_efficiency_score
from afa_agent.b_board.submission_policy import is_allowed_submission_model


DEFAULT_REFERENCE = (
    Path("/Users/abandon/Documents/AFA_ww")
    / "artifacts/b_board_actual/references"
    / "pseudo99_from_official98_ins016_bd_v1/reference_answers.json"
)
DEFAULT_REFERENCE_MANIFEST = DEFAULT_REFERENCE.parent / "reference_manifest.json"
DEFAULT_SOURCE_ROOT = Path("/Users/abandon/Documents/AFA_ww")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an answer-blind retrieval+Qwen run against pseudo99"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
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
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=DEFAULT_REFERENCE_MANIFEST,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_b_questions(args.question_root, args.submission_template)
    question_by_qid = {question.qid: question for question in questions}
    run_manifest = _read_json(args.run_dir / "run_manifest.json")
    expected_qids = [
        str(qid) for qid in ((run_manifest.get("scope") or {}).get("qids") or [])
    ]
    if not expected_qids:
        expected_qids = [question.qid for question in questions]
    if len(expected_qids) != len(set(expected_qids)):
        raise ValueError("run scope contains duplicate qids")
    unknown_qids = sorted(set(expected_qids) - set(question_by_qid))
    if unknown_qids:
        raise ValueError(f"run scope contains unknown qids: {unknown_qids}")
    answers = _index_rows(_read_json(args.run_dir / "answers.json"))
    failures = _index_rows(_read_json(args.run_dir / "failures.json"))
    reference = _index_rows(_read_json(args.reference))
    reference_manifest = _read_json(args.reference_manifest)
    missing_reference = sorted(set(expected_qids) - set(reference))
    if missing_reference:
        raise ValueError(f"reference is missing run-scope qids: {missing_reference}")

    exact: list[str] = []
    equivalent: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for qid in expected_qids:
        candidate = answers.get(qid)
        reference_parts = list(reference[qid]["answer_parts"])
        if candidate is None:
            failure = failures.get(qid)
            mismatches.append(
                {
                    "qid": qid,
                    "domain": question_by_qid[qid].domain,
                    "answer_format": question_by_qid[qid].answer_format,
                    "candidate": None,
                    "reference": reference_parts,
                    "failure": failure,
                }
            )
            continue
        candidate_parts = list(candidate["answer_parts"])
        if candidate_parts == reference_parts:
            exact.append(qid)
        if answer_parts_equivalent(candidate_parts, reference_parts):
            equivalent.append(qid)
        else:
            mismatches.append(
                {
                    "qid": qid,
                    "domain": question_by_qid[qid].domain,
                    "answer_format": question_by_qid[qid].answer_format,
                    "candidate": candidate_parts,
                    "reference": reference_parts,
                    "retrieved_doc_ids": sorted(
                        {
                            str(item.get("doc_id", ""))
                            for item in candidate.get("evidence_items", [])
                            if item.get("doc_id")
                        }
                    ),
                }
            )

    compliance = _audit_generation(
        run_dir=args.run_dir,
        expected_qids=expected_qids,
        question_by_qid=question_by_qid,
        answers=answers,
        failures=failures,
        run_manifest=run_manifest,
    )
    by_domain = _segment_metrics(
        expected_qids,
        equivalent,
        lambda qid: question_by_qid[qid].domain,
    )
    by_format = _segment_metrics(
        expected_qids,
        equivalent,
        lambda qid: question_by_qid[qid].answer_format,
    )
    token_total = int((run_manifest.get("token_usage") or {}).get("total_tokens", 0))
    token_score_field = (
        "token_efficiency_score"
        if len(expected_qids) == len(questions)
        else "sample_token_efficiency_score_not_full_run_comparable"
    )
    result = {
        "score_type": "offline_pseudo99_reference_match_not_official_accuracy",
        "official_accuracy": None,
        "reference": {
            "label": reference_manifest.get("label"),
            "officially_submitted": bool(
                reference_manifest.get("officially_submitted", False)
            ),
            "predicted_accuracy": reference_manifest.get("predicted_accuracy"),
            "question_count": len(expected_qids),
            "reference_sha256": reference_manifest.get("reference_sha256"),
        },
        "run": {
            "run_dir": str(args.run_dir.resolve()),
            "status": run_manifest.get("status"),
            "model": (run_manifest.get("model") or {}).get("model_name"),
            "answered_question_count": len(answers),
            "failed_question_count": len(failures),
            "raw_call_count": run_manifest.get("raw_call_count"),
            "format_retry_count": run_manifest.get("format_retry_count"),
            "token_usage": run_manifest.get("token_usage"),
            token_score_field: token_efficiency_score(token_total),
        },
        "reference_exact_match_count": len(exact),
        "reference_equivalent_match_count": len(equivalent),
        "reference_equivalent_accuracy_percent": round(
            len(equivalent) / len(expected_qids) * 100,
            6,
        ),
        "reference_mismatch_count": len(mismatches),
        "reference_mismatches": mismatches,
        "by_domain": by_domain,
        "by_answer_format": by_format,
        "compliance": compliance,
        "notes": [
            "该准确率只表示与当前pseudo99候选的等价匹配率，不是官网准确率。",
            "生成进程不加载pseudo99、官网答案锁或历史提交；比较在冻结生成结果后单独执行。",
            f"缺失或失败题按不匹配计入本次{len(expected_qids)}题样本分母。",
            (
                "Token效率分按完整100题总Token计算。"
                if len(expected_qids) == len(questions)
                else "样本Token效率分仅按样本总Token代入，不能与100题正式提交直接比较。"
            ),
        ],
    }
    output = args.output or args.run_dir / "accuracy_evaluation.json"
    _write_json(output, result)
    _write_badcases(
        args.run_dir / "accuracy_badcases.md",
        mismatches,
        question_by_qid,
        result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _audit_generation(
    *,
    run_dir: Path,
    expected_qids: list[str],
    question_by_qid: dict[str, Any],
    answers: dict[str, dict[str, Any]],
    failures: dict[str, dict[str, Any]],
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    problems: list[str] = []
    try:
        run_config_payload = _read_json(run_dir / "run_config.json")
        generation_config = dict(run_config_payload.get("config") or {})
        if run_config_payload.get("fingerprint") != public_run_config_fingerprint(
            generation_config
        ):
            raise ValueError("run config fingerprint is invalid")
        if run_manifest.get("fingerprint") != run_config_payload.get(
            "fingerprint"
        ):
            raise ValueError("manifest fingerprint is invalid")
        run_binding = {
            "run_fingerprint": str(run_config_payload["fingerprint"]),
            "run_instance_id": str(run_config_payload["run_instance_id"]),
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        generation_config = {}
        run_binding = {}
        problems.append("run_config_binding_invalid")
    retrieval_config = dict(generation_config.get("retrieval") or {})
    if retrieval_config.get("research_only_strategy") is not False:
        problems.append("research_only_strategy_not_isolated")
    generation_contract = dict(generation_config.get("generation") or {})
    blind_config = dict(
        generation_config.get("answer_blind_contract") or {}
    )
    calculation_mode = generation_contract.get("calculation_mode")
    if calculation_mode not in {"direct", "verified"}:
        problems.append("unsupported_calculation_mode")
    if bool(
        blind_config.get("deterministic_calculation_executor_used")
    ) != (calculation_mode == "verified"):
        problems.append("calculation_executor_contract_inconsistent")
    model_name = str((run_manifest.get("model") or {}).get("model_name", ""))
    if not is_allowed_submission_model(model_name):
        problems.append("model_not_allowed")
    if set(answers) | set(failures) != set(expected_qids):
        problems.append("qid_coverage_incomplete")
    call_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    for qid in expected_qids:
        call_path = run_dir / "raw_calls" / f"{qid}.json"
        if not call_path.exists():
            problems.append(f"{qid}:missing_raw_calls")
            continue
        raw_artifact = _read_json(call_path)
        if raw_artifact.get("qid") != qid:
            problems.append(f"{qid}:raw_ledger_qid_mismatch")
        if run_binding and any(
            raw_artifact.get(key) != value
            for key, value in run_binding.items()
        ):
            problems.append(f"{qid}:raw_ledger_run_binding_mismatch")
        calls = raw_artifact.get("calls", [])
        if not isinstance(calls, list):
            problems.append(f"{qid}:raw_calls_not_array")
            calls = []
        aliases = raw_artifact.get("evidence_alias_map", [])
        alias_names = [str(item.get("alias", "")) for item in aliases]
        source_ids = [str(item.get("source_evidence_id", "")) for item in aliases]
        if (
            not aliases
            or not all(alias_names)
            or len(alias_names) != len(set(alias_names))
            or not all(source_ids)
            or len(source_ids) != len(set(source_ids))
        ):
            problems.append(f"{qid}:invalid_evidence_alias_map")
        row = answers.get(qid) or failures.get(qid) or {}
        if aliases != row.get("evidence_alias_map"):
            problems.append(f"{qid}:evidence_alias_map_drift")
        call_count += len(calls)
        for expected_index, call in enumerate(calls, start=1):
            if (
                not isinstance(call, dict)
                or int(call.get("call_index", -1)) != expected_index
            ):
                problems.append(f"{qid}:invalid_call_sequence")
                continue
            messages = call.get("messages")
            if qid in json.dumps(messages, ensure_ascii=False):
                problems.append(f"{qid}:qid_leaked_to_model")
            if call.get("model_name") != model_name:
                problems.append(f"{qid}:model_drift")
            if call.get("response_format_mode") != "native_json_schema_strict":
                problems.append(f"{qid}:structured_output_mode_drift")
            attempt_count = call.get("transport_attempt_count", 1)
            rejections = call.get("transport_rejections", [])
            if (
                isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count < 1
                or not isinstance(rejections, list)
                or attempt_count != len(rejections) + 1
                or any(
                    not isinstance(item, dict)
                    or item.get("status_code") != 429
                    or item.get("pre_generation_rejection") is not True
                    or item.get("token_usage_observed") is not False
                    for item in rejections
                )
            ):
                problems.append(f"{qid}:transport_audit_invalid")
            usage = call.get("token_usage") or {}
            prompt = int(usage.get("prompt_tokens", -1))
            completion = int(usage.get("completion_tokens", -1))
            total = int(usage.get("total_tokens", -1))
            if min(prompt, completion, total) < 0 or total != prompt + completion:
                problems.append(f"{qid}:invalid_raw_usage")
            provider_usage = (call.get("raw_response") or {}).get("usage") or {}
            try:
                provider_prompt = int(provider_usage["prompt_tokens"])
                provider_completion = int(provider_usage["completion_tokens"])
                provider_total = int(provider_usage["total_tokens"])
            except (KeyError, TypeError, ValueError):
                problems.append(f"{qid}:missing_provider_raw_usage")
            else:
                if (
                    min(
                        provider_prompt,
                        provider_completion,
                        provider_total,
                    )
                    < 0
                    or provider_total
                    != provider_prompt + provider_completion
                    or (prompt, completion, total)
                    != (
                        provider_prompt,
                        provider_completion,
                        provider_total,
                    )
                ):
                    problems.append(f"{qid}:provider_raw_usage_drift")
            prompt_tokens += max(0, prompt)
            completion_tokens += max(0, completion)
        intent_problems = _audit_call_intents(
            run_dir=run_dir,
            qid=qid,
            aliases=aliases,
            calls=calls,
            run_binding=run_binding,
        )
        problems.extend(intent_problems)
        if qid in answers:
            final_content = str(calls[-1].get("content", "")) if calls else ""
            try:
                final_payload = _effective_submitted_payload(
                    run_dir=run_dir,
                    qid=qid,
                    raw_artifact=raw_artifact,
                    calls=calls,
                    question=question_by_qid[qid],
                )
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                final_payload = {}
                problems.append(f"{qid}:final_content_not_json")
            if (
                final_payload.get("answer_parts") != answers[qid].get("answer_parts")
                or final_payload.get("reasoning") != answers[qid].get("reasoning")
            ):
                problems.append(f"{qid}:submitted_payload_drift")
            expected_parts_hash = _sha256_text(
                json.dumps(
                    answers[qid].get("answer_parts", []),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            if raw_artifact.get("final_call_index") != len(calls):
                problems.append(f"{qid}:final_call_index_mismatch")
            if raw_artifact.get("final_content_sha256") != _sha256_text(final_content):
                problems.append(f"{qid}:final_content_hash_mismatch")
            if (
                raw_artifact.get("submitted_answer_parts_sha256")
                != expected_parts_hash
            ):
                problems.append(f"{qid}:submitted_answer_hash_mismatch")
            if raw_artifact.get("submitted_reasoning_sha256") != _sha256_text(
                str(answers[qid].get("reasoning", ""))
            ):
                problems.append(f"{qid}:submitted_reasoning_hash_mismatch")
            if not _valid_postprocessing_record(
                raw_artifact.get("postprocessing")
            ):
                problems.append(f"{qid}:postprocessing_contract_mismatch")
            generation = dict(generation_config.get("generation") or {})
            effective_verified_calculation = is_verified_calculation_path(
                str(generation.get("calculation_mode", "")),
                question_by_qid[qid].answer_format,
            )
            if (
                not effective_verified_calculation
                and generation.get("output_contract") == "joint"
            ):
                problems.extend(
                    _audit_direct_frozen_answer(
                        run_dir=run_dir,
                        qid=qid,
                        answer=answers[qid],
                        aliases=aliases,
                        calls=calls,
                        run_binding=run_binding,
                    )
                )
            elif effective_verified_calculation:
                checkpoint_path = (
                    run_dir / "frozen_answers" / f"{qid}.json"
                )
                checkpoint_record = answers[qid].get(
                    "frozen_answer_checkpoint"
                )
                evidence = answers[qid].get("evidence_items")
                try:
                    if (
                        not isinstance(checkpoint_record, dict)
                        or not isinstance(evidence, list)
                        or checkpoint_record.get("sha256")
                        != _sha256_file(checkpoint_path)
                    ):
                        raise ValueError(
                            "verified checkpoint record is invalid"
                        )
                    checkpoint = validate_verified_calculation_checkpoint(
                        checkpoint_path,
                        question=question_by_qid[qid],
                        evidence=evidence,
                        expected_model_name=model_name,
                        expected_run_binding=run_binding,
                    )
                    checkpoint_calls = checkpoint.get("calls")
                    if (
                        not isinstance(checkpoint_calls, list)
                        or calls[: len(checkpoint_calls)]
                        != checkpoint_calls
                    ):
                        raise ValueError(
                            "verified checkpoint call prefix drift"
                        )
                    if checkpoint.get("answer_parts") != answers[qid].get(
                        "answer_parts"
                    ):
                        raise ValueError(
                            "verified checkpoint answer drift"
                        )
                except (
                    OSError,
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    problems.append(
                        f"{qid}:verified_checkpoint_invalid:"
                        f"{type(exc).__name__}:{exc}"
                    )
    manifest_usage = run_manifest.get("token_usage") or {}
    if (
        prompt_tokens != int(manifest_usage.get("prompt_tokens", -1))
        or completion_tokens != int(manifest_usage.get("completion_tokens", -1))
        or prompt_tokens + completion_tokens
        != int(manifest_usage.get("total_tokens", -1))
    ):
        problems.append("manifest_usage_mismatch")
    if call_count != int(run_manifest.get("raw_call_count", -1)):
        problems.append("manifest_call_count_mismatch")
    if not run_manifest.get("all_observed_usage_from_provider_raw_fields"):
        problems.append("manifest_provider_usage_not_fully_observed")
    if run_manifest.get("unobservable_usage_risk"):
        problems.append("manifest_unobservable_usage_risk")
    try:
        source_paths = _generation_source_paths(run_dir)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        source_paths = []
        problems.append(
            f"generation_source_fingerprint_invalid:{type(exc).__name__}:{exc}"
        )
    prohibited_imports = _prohibited_generation_imports(source_paths)
    problems.extend(f"prohibited_import:{item}" for item in prohibited_imports)
    hardcoded_qid_literals = _hardcoded_qid_literals(source_paths)
    problems.extend(
        f"hardcoded_qid_literal:{item}" for item in hardcoded_qid_literals
    )
    prompt_literal_risks = _fixed_prompt_literal_risks(source_paths)
    problems.extend(
        f"fixed_prompt_literal_risk:{item}"
        for item in prompt_literal_risks
    )
    contract = dict(run_manifest.get("answer_blind_contract") or {})
    if contract.get("research_only_strategy") is not False:
        problems.append("manifest_research_only_strategy_not_false")
    generation_reference_loaded = bool(
        contract.get("reference_loaded_by_generation")
        or contract.get("official_locks_loaded_by_generation")
        or any(
            fragment in item
            for item in prohibited_imports
            for fragment in (
                "candidate_scorecard",
                "official_answer",
                "pseudo_accuracy",
            )
        )
    )
    solver_used = bool(contract.get("solver_or_rule_layer_used"))
    fixed_locator_used = bool(
        contract.get("fixed_locator_used") or hardcoded_qid_literals
    )
    return {
        "passed": not problems,
        "problems": problems,
        "model_allowed": is_allowed_submission_model(model_name),
        "qid_absent_from_model_messages": not any(
            problem.endswith("qid_leaked_to_model") for problem in problems
        ),
        "raw_usage_reconciled": not any(
            "usage" in problem for problem in problems
        ),
        "generation_prohibited_imports": prohibited_imports,
        "generation_hardcoded_qid_literals": hardcoded_qid_literals,
        "generation_source_files_scanned": [
            str(path.relative_to(ROOT)) for path in source_paths
        ],
        "generation_reference_loaded": generation_reference_loaded,
        "solver_used": solver_used,
        "fixed_locator_used": fixed_locator_used,
    }


def _effective_submitted_payload(
    *,
    run_dir: Path | None = None,
    qid: str = "",
    raw_artifact: dict[str, Any],
    calls: list[dict[str, Any]],
    question: Any,
) -> dict[str, Any]:
    if not calls:
        return {}
    final_payload = json.loads(str(calls[-1].get("content", "")))
    if calls[-1].get("purpose") == "reasoning_only_retry_from_frozen_answer":
        return validate_frozen_answer_reasoning_payload(
            list(calls[-1].get("frozen_answer_parts") or []),
            final_payload,
        )
    if calls[-1].get("purpose") == "verified_calculation_reasoning":
        if run_dir is None or not qid:
            raise ValueError(
                "verified calculation reconstruction requires run_dir and qid"
            )
        checkpoint = _read_json(
            run_dir / "frozen_answers" / f"{qid}.json"
        )
        answer_parts = checkpoint.get("answer_parts")
        if not isinstance(answer_parts, list):
            raise ValueError("verified calculation artifact is incomplete")
        return validate_verified_frozen_answer_reasoning_payload(
            [str(item) for item in answer_parts],
            final_payload,
        )
    if "answer_parts" in final_payload:
        return validate_answer_payload(
            question,
            final_payload,
        )
    return final_payload


def _audit_call_intents(
    *,
    run_dir: Path,
    qid: str,
    aliases: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    run_binding: dict[str, str],
) -> list[str]:
    problems: list[str] = []
    intent_dir = run_dir / "call_intents"
    intent_paths = sorted(intent_dir.glob(f"{qid}.*.json"))
    alias_sha256 = _sha256_text(
        json.dumps(
            aliases,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    intents: dict[int, dict[str, Any]] = {}
    for path in intent_paths:
        try:
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
                raise ValueError("binding mismatch")
            intents[call_index] = payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            problems.append(f"{qid}:invalid_call_intent:{path.name}")
    for call in calls:
        call_index = int(call.get("call_index", -1))
        intent = intents.get(call_index)
        if intent is None or intent.get("purpose") != call.get("purpose"):
            problems.append(f"{qid}:call_without_matching_intent:{call_index}")
    for call_index in sorted(
        set(intents)
        - {int(call.get("call_index", -1)) for call in calls}
    ):
        problems.append(f"{qid}:unobservable_call_intent:{call_index}")
    return problems


def _audit_direct_frozen_answer(
    *,
    run_dir: Path,
    qid: str,
    answer: dict[str, Any],
    aliases: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    run_binding: dict[str, str],
) -> list[str]:
    path = run_dir / "frozen_answers" / f"{qid}.json"
    if not path.exists():
        return [f"{qid}:missing_direct_frozen_answer"]
    try:
        payload = _read_json(path)
        call_index = payload.get("answer_call_index")
        if (
            payload.get("checkpoint_version") != "direct_model_answer_v1"
            or payload.get("checkpoint_kind") != "direct_model_answer"
            or payload.get("qid") != qid
            or payload.get("evidence_alias_map") != aliases
            or any(
                payload.get(key) != value for key, value in run_binding.items()
            )
            or isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index < 1
            or call_index > len(calls)
            or payload.get("answer_parts") != answer.get("answer_parts")
        ):
            raise ValueError("binding mismatch")
        answer_call = calls[call_index - 1]
        if (
            payload.get("answer_content_sha256")
            != _sha256_text(str(answer_call.get("content", "")))
            or answer_call.get("purpose")
            == "reasoning_only_retry_from_frozen_answer"
        ):
            raise ValueError("answer call mismatch")
        model_payload = json.loads(str(answer_call.get("content", "")))
        if model_payload.get("answer_parts") != payload.get("answer_parts"):
            raise ValueError("answer field mismatch")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return [f"{qid}:invalid_direct_frozen_answer"]
    return []


def _valid_postprocessing_record(payload: Any) -> bool:
    return payload == {
        "answer_modified": False,
        "reasoning_modified": False,
        "csv_escaping_only": True,
    }


def _generation_source_paths(run_dir: Path) -> list[Path]:
    run_config = _read_json(run_dir / "run_config.json")
    config = run_config.get("config")
    if not isinstance(config, dict):
        raise ValueError("run config is missing")
    source_files = (
        (config.get("input_sha256") or {}).get("source_files") or {}
    )
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("run config has no generation source fingerprint")
    paths: list[Path] = []
    for relative, expected_sha256 in sorted(source_files.items()):
        path = (ROOT / str(relative)).resolve()
        if ROOT.resolve() not in path.parents:
            raise ValueError("generation source path escapes the repository")
        if not path.is_file():
            raise ValueError(f"generation source file is missing: {relative}")
        if _sha256_file(path) != str(expected_sha256):
            raise ValueError(f"generation source file drifted: {relative}")
        paths.append(path)
    return paths


def _prohibited_generation_imports(paths: list[Path]) -> list[str]:
    prohibited_fragments = (
        ".solver",
        "candidate_scorecard",
        "official_answer",
        "pseudo_accuracy",
    )
    imports: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(str(node.module or ""))
    return sorted(
        {
            name
            for name in imports
            if any(fragment in name for fragment in prohibited_fragments)
        }
    )


def _hardcoded_qid_literals(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    qid_pattern = re.compile(r"\b(?:fc|fin|ins|res|reg)_b_\d{3}\b")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(
                node.value, str
            ):
                continue
            for qid in qid_pattern.findall(node.value):
                findings.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}:{qid}"
                )
    return sorted(set(findings))


def _fixed_prompt_literal_risks(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for literal in suspicious_generation_prompt_literals(tree):
            findings.append(f"{path.relative_to(ROOT)}:{literal}")
    return sorted(set(findings))


def _segment_metrics(
    expected_qids: list[str],
    equivalent_qids: list[str],
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    totals = Counter(key_fn(qid) for qid in expected_qids)
    matches = Counter(key_fn(qid) for qid in equivalent_qids)
    return {
        key: {
            "question_count": totals[key],
            "match_count": matches[key],
            "match_rate_percent": round(matches[key] / totals[key] * 100, 6),
        }
        for key in sorted(totals)
    }


def _write_badcases(
    path: Path,
    mismatches: list[dict[str, Any]],
    question_by_qid: dict[str, Any],
    result: dict[str, Any],
) -> None:
    lines = [
        "# Retrieval + Qwen 样本代理准确率 badcases",
        "",
        (
            f"- 与pseudo99等价匹配：{result['reference_equivalent_match_count']}/"
            f"{result['reference']['question_count']}"
        ),
        f"- 代理匹配率：{result['reference_equivalent_accuracy_percent']:.2f}%",
        "- 注意：这不是官网准确率。",
        "",
    ]
    for item in mismatches:
        question = question_by_qid[item["qid"]]
        lines.extend(
            [
                f"## {item['qid']}",
                "",
                f"- 领域/题型：`{item['domain']}` / `{item['answer_format']}`",
                f"- 题目：{question.question}",
                f"- 模型答案：`{item.get('candidate')}`",
                f"- pseudo99：`{item.get('reference')}`",
            ]
        )
        if item.get("failure"):
            failure = item["failure"]
            lines.append(
                f"- 失败：`{failure.get('error_type')}` {failure.get('error', '')}"
            )
        else:
            lines.append(f"- 检索文档：`{item.get('retrieved_doc_ids', [])}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _index_rows(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("expected a JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("answer/reference row must be an object")
        qid = str(row.get("qid", "")).strip()
        if not qid or qid in indexed:
            raise ValueError("answer/reference contains missing or duplicate qid")
        indexed[qid] = row
    return indexed


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
