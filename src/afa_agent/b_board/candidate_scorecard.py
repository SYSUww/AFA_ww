from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from afa_agent.b_board.io import SUBMISSION_COLUMNS
from afa_agent.b_board.reasoning_evaluation import (
    PROMPT_VERSION as REASONING_PROMPT_VERSION,
    REASONING_JUDGE_MODEL,
    SCHEMA_VERSION as REASONING_SCHEMA_VERSION,
)
from afa_agent.b_board.runner import SUBMISSION_REASONING_PROMPT_VERSION
from afa_agent.b_board.scoring import score_submission, token_efficiency_score
from afa_agent.b_board.submission_policy import is_allowed_submission_model


_NUMERIC_ANSWER_RE = re.compile(
    r"^(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?P<suffix>%?)$"
)


def evaluate_candidate_run(
    *,
    run_dir: Path,
    reference_manifest_path: Path,
    official_locks_path: Path,
    reasoning_evaluation_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a proxy scorecard without treating the pseudo reference as truth.

    The accuracy proxy is available only when a complete, compliant Qwen run
    matches every frozen pseudo-reference answer.  It is never labelled as an
    official score.
    """

    run_dir = Path(run_dir).resolve()
    manifest = _read_object(run_dir / "run_manifest.json")
    answers = _read_answer_rows(run_dir / "answers.json")
    ledger = _read_jsonl(run_dir / "usage_ledger.jsonl")
    submission_path = (run_dir / "submit.csv").resolve()
    reference_manifest_path = Path(reference_manifest_path).resolve()
    reference_manifest = _read_object(reference_manifest_path)
    reference_path = Path(str(reference_manifest["reference_answers"])).resolve()
    if _sha256(reference_path) != str(reference_manifest.get("reference_sha256", "")):
        raise ValueError("pseudo reference SHA-256 does not match its sealed manifest")
    reference = _read_answer_rows(reference_path)
    locks_payload = _read_object(Path(official_locks_path).resolve())
    locks = _normalize_answer_map(locks_payload.get("answers"), "official locks")

    expected_qids = list(reference)
    failures: list[str] = []
    declared_submission = str(manifest.get("submission_path") or "")
    if not declared_submission or Path(declared_submission).resolve() != submission_path:
        failures.append("manifest_submission_path_mismatch")
    submission_audit = _audit_submission_csv(
        submission_path=submission_path,
        answers=answers,
        expected_qids=expected_qids,
    )
    failures.extend(submission_audit["failures"])
    if list(answers) != expected_qids:
        failures.append("candidate_qids_or_order_do_not_match_reference")
    expected_count = len(expected_qids)
    if int(manifest.get("expected_question_count", -1)) != expected_count:
        failures.append("manifest_expected_question_count_mismatch")
    if int(manifest.get("answered_question_count", -1)) != expected_count:
        failures.append("manifest_answered_question_count_mismatch")
    if manifest.get("status") != "complete":
        failures.append("run_is_not_complete")
    if manifest.get("failed_qids"):
        failures.append("run_has_failed_qids")
    if manifest.get("run_mode") != "submission":
        failures.append("run_mode_is_not_submission")
    if manifest.get("submission_eligible") is not True:
        failures.append("runner_did_not_mark_submission_eligible")

    model_name = str(dict(manifest.get("model") or {}).get("model_name", ""))
    if not is_allowed_submission_model(model_name):
        failures.append("manifest_model_is_not_allowed_qwen")

    ledger_by_qid: dict[str, dict[str, Any]] = {}
    for row in ledger:
        qid = str(row.get("qid", "")).strip()
        if not qid or qid in ledger_by_qid:
            failures.append("usage_ledger_has_missing_or_duplicate_qid")
            continue
        ledger_by_qid[qid] = row
    if list(ledger_by_qid) != expected_qids:
        failures.append("usage_ledger_qids_or_order_do_not_match_reference")

    answer_token_total = 0
    raw_call_count = 0
    for qid, answer in answers.items():
        usage = _normalize_usage(answer.get("token_usage"), f"{qid}.token_usage")
        answer_token_total += usage["total_tokens"]
        reasoning = str(answer.get("decision_summary", "")).strip()
        if len(re.sub(r"\s+", "", reasoning)) < 20:
            failures.append(f"{qid}:reasoning_is_too_short")
        reasoning_trace = dict(
            dict(answer.get("decision_trace") or {}).get("submission_reasoning") or {}
        )
        if (
            reasoning_trace.get("prompt_version")
            != SUBMISSION_REASONING_PROMPT_VERSION
        ):
            failures.append(f"{qid}:reasoning_prompt_version_mismatch")
        if reasoning_trace.get("grounding_status") != "supported":
            failures.append(f"{qid}:reasoning_is_not_grounded")
        if reasoning_trace.get("answer_parts_preserved") is not True:
            failures.append(f"{qid}:reasoning_did_not_freeze_answer")
        if str(reasoning_trace.get("model_name", "")) != model_name:
            failures.append(f"{qid}:reasoning_model_mismatch")
        for part in answer.get("answer_parts") or []:
            if not _reasoning_mentions_answer_part(reasoning, part):
                failures.append(f"{qid}:reasoning_omits_frozen_answer_part")
                break
        evidence_ids = {
            str(item.get("unit_id", "")).strip()
            for item in answer.get("evidence_items") or []
            if isinstance(item, Mapping) and str(item.get("unit_id", "")).strip()
        }
        if not evidence_ids:
            failures.append(f"{qid}:answer_has_no_evidence")
        used_evidence_ids = {
            str(item).strip()
            for item in answer.get("used_evidence_ids") or []
            if str(item).strip()
        }
        unknown_used_ids = sorted(
            item
            for item in used_evidence_ids
            if item not in evidence_ids and not item.startswith("question:")
        )
        if unknown_used_ids:
            failures.append(f"{qid}:used_evidence_is_not_in_final_evidence")
        rescued_ids = {
            str(item).strip()
            for item in reasoning_trace.get("rescued_evidence_ids") or []
            if str(item).strip()
        }
        if rescued_ids - evidence_ids:
            failures.append(f"{qid}:rescued_evidence_is_not_in_final_evidence")
        if answer.get("answer_format") == "calculation":
            calculation_trace = dict(answer.get("calculation_trace") or {})
            if calculation_trace.get("replay_verified") is not True:
                failures.append(f"{qid}:calculation_replay_is_not_verified")
            if calculation_trace.get("grounding_verified") is not True:
                failures.append(f"{qid}:calculation_grounding_is_not_verified")

        ledger_row = ledger_by_qid.get(qid)
        if ledger_row is None:
            continue
        if ledger_row.get("status") != "success":
            failures.append(f"{qid}:usage_ledger_status_is_not_success")
        calls = ledger_row.get("calls")
        if not isinstance(calls, list) or not calls:
            failures.append(f"{qid}:usage_ledger_has_no_raw_calls")
            calls = []
        raw_call_count += len(calls)
        call_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                failures.append(f"{qid}:usage_call_{index + 1}_is_invalid")
                continue
            call_model = str(call.get("model_name", ""))
            if call_model != model_name or not is_allowed_submission_model(call_model):
                failures.append(f"{qid}:usage_call_{index + 1}_model_is_not_allowed")
            call_usage = _normalize_usage(
                call.get("token_usage"), f"{qid}.usage_call_{index + 1}"
            )
            for key in call_total:
                call_total[key] += call_usage[key]
        if call_total != usage:
            failures.append(f"{qid}:raw_call_usage_does_not_match_answer_usage")
        ledger_usage = _normalize_usage(
            ledger_row.get("token_usage"), f"{qid}.usage_ledger_total"
        )
        if ledger_usage != usage:
            failures.append(f"{qid}:usage_ledger_total_does_not_match_answer_usage")

    generation_usage = _normalize_usage(
        manifest.get("generation_token_usage"), "manifest.generation_token_usage"
    )
    answer_usage_total = _sum_answer_usage(answers)
    if generation_usage != answer_usage_total:
        failures.append("manifest_generation_usage_does_not_match_answers")
    manifest_usage = _normalize_usage(
        manifest.get("token_usage"), "manifest.token_usage"
    )
    if manifest_usage != answer_usage_total:
        failures.append("manifest_submission_usage_does_not_match_answers")
    if answer_usage_total["total_tokens"] != answer_token_total:
        failures.append("internal_answer_usage_sum_mismatch")

    exact_matches: list[str] = []
    equivalent_matches: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for qid in expected_qids:
        candidate_parts = list(answers.get(qid, {}).get("answer_parts") or [])
        reference_parts = list(reference[qid].get("answer_parts") or [])
        if candidate_parts == reference_parts:
            exact_matches.append(qid)
        if answer_parts_equivalent(candidate_parts, reference_parts):
            equivalent_matches.append(qid)
        else:
            mismatches.append(
                {
                    "qid": qid,
                    "candidate": candidate_parts,
                    "reference": reference_parts,
                }
            )

    lock_regressions: list[dict[str, Any]] = []
    for qid, locked_parts in locks.items():
        candidate = list(answers.get(qid, {}).get("answer_parts") or [])
        if not answer_parts_equivalent(candidate, locked_parts):
            lock_regressions.append(
                {"qid": qid, "candidate": candidate, "locked": locked_parts}
            )
    if lock_regressions:
        failures.append("official_answer_lock_regression")

    compliance_failures = list(dict.fromkeys(failures))
    complete_reference_match = len(equivalent_matches) == expected_count
    accuracy_proxy = (
        float(reference_manifest["predicted_accuracy"])
        if complete_reference_match and not compliance_failures
        else None
    )
    token_score = token_efficiency_score(answer_token_total)
    reasoning_shadow = None
    reasoning_score = None
    if reasoning_evaluation_dir is not None:
        reasoning_shadow = _load_reasoning_shadow(
            evaluation_dir=Path(reasoning_evaluation_dir).resolve(),
            submission_path=submission_path,
            answers=answers,
        )
        reasoning_score = float(reasoning_shadow["reasoning_score"])
    proxy_total = None
    if accuracy_proxy is not None and reasoning_score is not None:
        proxy_total = score_submission(
            accuracy_score=accuracy_proxy,
            reasoning_scores=[reasoning_score],
            token_total=answer_token_total,
        ).total_score
    predicted_accuracy = float(reference_manifest["predicted_accuracy"])
    changed_answer_count = len(mismatches)
    accuracy_scenario = {
        "assumption": (
            "The frozen reference is exactly the predicted-99 answer set with "
            "one remaining wrong answer."
        ),
        "changed_answer_count": changed_answer_count,
        "lower_bound": max(0.0, predicted_accuracy - changed_answer_count),
        "upper_bound": (
            predicted_accuracy if changed_answer_count == 0 else 100.0
        ),
        "is_official": False,
    }
    proxy_total_range = None
    if not compliance_failures and reasoning_score is not None:
        proxy_total_range = {
            "lower_bound": score_submission(
                accuracy_score=accuracy_scenario["lower_bound"],
                reasoning_scores=[reasoning_score],
                token_total=answer_token_total,
            ).total_score,
            "upper_bound": score_submission(
                accuracy_score=accuracy_scenario["upper_bound"],
                reasoning_scores=[reasoning_score],
                token_total=answer_token_total,
            ).total_score,
        }

    return {
        "score_type": "offline_proxy",
        "official_accuracy_score": None,
        "official_total_score": None,
        "reference": {
            "label": reference_manifest.get("label"),
            "officially_submitted": bool(
                reference_manifest.get("officially_submitted", False)
            ),
            "predicted_accuracy": float(reference_manifest["predicted_accuracy"]),
            "question_count": expected_count,
            "reference_sha256": reference_manifest.get("reference_sha256"),
        },
        "run": {
            "run_dir": str(run_dir),
            "model_name": model_name,
            "question_count": len(answers),
            "raw_call_count": raw_call_count,
            "token_total": answer_token_total,
            "submission_path": str(submission_path),
            "submission_sha256": submission_audit["sha256"],
        },
        "compliance_passed": not compliance_failures,
        "compliance_failures": compliance_failures,
        "reference_exact_match_count": len(exact_matches),
        "reference_equivalent_match_count": len(equivalent_matches),
        "reference_match_rate": (
            len(equivalent_matches) / expected_count if expected_count else 0.0
        ),
        "reference_mismatches": mismatches,
        "official_lock_count": len(locks),
        "official_lock_regressions": lock_regressions,
        "accuracy_proxy_score": accuracy_proxy,
        "accuracy_scenario": accuracy_scenario,
        "reasoning_shadow": reasoning_shadow,
        "reasoning_shadow_score": reasoning_score,
        "token_efficiency_score": token_score,
        "proxy_total_score": proxy_total,
        "proxy_total_score_range": proxy_total_range,
        "notes": (
            "Reference match is a pseudo-label preservation check, not official "
            "accuracy. A point proxy total is emitted only for a compliant full "
            "match with a sealed GPT-5.6 shadow evaluation."
        ),
    }


def answer_parts_equivalent(
    left: Sequence[object], right: Sequence[object]
) -> bool:
    if len(left) != len(right):
        return False
    return all(_answer_part_key(a) == _answer_part_key(b) for a, b in zip(left, right))


def _answer_part_key(value: object) -> tuple[str, str, str]:
    text = str(value).strip()
    match = _NUMERIC_ANSWER_RE.fullmatch(text)
    if match is None:
        return ("text", text, "")
    try:
        number = Decimal(match.group("number").replace(",", "")).normalize()
    except InvalidOperation:
        return ("text", text, "")
    return ("number", str(number), match.group("suffix"))


def _reasoning_mentions_answer_part(reasoning: str, answer_part: object) -> bool:
    compact_reasoning = re.sub(r"\s+", "", reasoning).replace(",", "")
    compact_part = re.sub(r"\s+", "", str(answer_part)).replace(",", "")
    if compact_part and compact_part in compact_reasoning:
        return True
    expected_key = _answer_part_key(answer_part)
    if expected_key[0] != "number":
        return False
    numeric_tokens = re.findall(
        r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?",
        reasoning,
    )
    return any(_answer_part_key(token) == expected_key for token in numeric_tokens)


def _audit_submission_csv(
    *,
    submission_path: Path,
    answers: Mapping[str, Mapping[str, Any]],
    expected_qids: Sequence[str],
) -> dict[str, Any]:
    failures: list[str] = []
    if not submission_path.is_file():
        return {"failures": ["submission_csv_is_missing"], "sha256": None}
    with submission_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        rows = list(reader)
    if columns != SUBMISSION_COLUMNS:
        failures.append("submission_csv_columns_do_not_match_new_rule")
    if not rows or str(rows[0].get("qid", "")).strip() != "summary":
        failures.append("submission_csv_summary_row_is_missing_or_not_first")
        summary = None
        question_rows = rows
    else:
        summary = rows[0]
        question_rows = rows[1:]
    csv_qids = [str(row.get("qid", "")).strip() for row in question_rows]
    if csv_qids != list(expected_qids):
        failures.append("submission_csv_qids_or_order_do_not_match_reference")

    csv_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in question_rows:
        qid = str(row.get("qid", "")).strip()
        answer = answers.get(qid)
        if answer is None:
            continue
        csv_parts = [
            str(row.get(f"answer{index}", "")).strip()
            for index in range(1, 5)
            if str(row.get(f"answer{index}", "")).strip()
        ]
        if csv_parts != list(answer.get("answer_parts") or []):
            failures.append(f"{qid}:submission_csv_answer_mismatch")
        try:
            row_usage = {
                key: _strict_csv_int(row.get(key), f"{qid}.{key}")
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            row_usage = _normalize_usage(row_usage, f"{qid}.submission_csv_usage")
        except ValueError:
            failures.append(f"{qid}:submission_csv_usage_is_invalid")
            continue
        expected_usage = _normalize_usage(
            answer.get("token_usage"), f"{qid}.token_usage"
        )
        if row_usage != expected_usage:
            failures.append(f"{qid}:submission_csv_usage_mismatch")
        for key in csv_usage:
            csv_usage[key] += row_usage[key]
        if str(row.get("reasoning", "")) != str(
            answer.get("decision_summary", "")
        ):
            failures.append(f"{qid}:submission_csv_reasoning_mismatch")

    if summary is not None:
        if any(str(summary.get(f"answer{index}", "")).strip() for index in range(1, 5)):
            failures.append("submission_csv_summary_contains_answers")
        if str(summary.get("reasoning", "")).strip():
            failures.append("submission_csv_summary_contains_reasoning")
        try:
            summary_usage = {
                key: _strict_csv_int(
                    summary.get(key), f"summary.{key}"
                )
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            summary_usage = _normalize_usage(
                summary_usage, "submission_csv_summary_usage"
            )
            if summary_usage != csv_usage:
                failures.append("submission_csv_summary_usage_mismatch")
        except ValueError:
            failures.append("submission_csv_summary_usage_is_invalid")
    return {
        "failures": list(dict.fromkeys(failures)),
        "sha256": _sha256(submission_path),
    }


def _strict_csv_int(value: object, name: str) -> int:
    text = str(value if value is not None else "").strip()
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"{name} is not a non-negative integer")
    return int(text)


def _load_reasoning_shadow(
    *,
    evaluation_dir: Path,
    submission_path: Path,
    answers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = _read_object(
        evaluation_dir / "reasoning_evaluator_manifest.json"
    )
    if manifest.get("status") != "complete":
        raise ValueError("reasoning shadow evaluation is not complete")
    identity = manifest.get("evaluator_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("reasoning shadow evaluator identity is missing")
    if str(identity.get("model_name", "")).lower() != REASONING_JUDGE_MODEL:
        raise ValueError("reasoning shadow model is not the fixed GPT-5.6 judge")
    if identity.get("prompt_version") != REASONING_PROMPT_VERSION:
        raise ValueError("reasoning shadow prompt version changed")
    if int(identity.get("schema_version", -1)) != REASONING_SCHEMA_VERSION:
        raise ValueError("reasoning shadow schema version changed")
    if float(identity.get("temperature", -1)) != 0.0:
        raise ValueError("reasoning shadow temperature is not zero")

    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ValueError("reasoning shadow fingerprint is missing")
    components = fingerprint.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("reasoning shadow fingerprint components are missing")
    if str(fingerprint.get("sha256", "")) != _payload_sha256(components):
        raise ValueError("reasoning shadow fingerprint was modified")
    submission_component = components.get("submission")
    if not isinstance(submission_component, Mapping):
        raise ValueError("reasoning shadow submission fingerprint is missing")
    if str(submission_component.get("sha256", "")) != _sha256(submission_path):
        raise ValueError("reasoning shadow was evaluated against another submission")

    sealed_path = (evaluation_dir / "sealed_reasoning.json").resolve()
    declared_sealed = str(manifest.get("sealed_reasoning_path") or "")
    if not declared_sealed or Path(declared_sealed).resolve() != sealed_path:
        raise ValueError("reasoning shadow sealed path mismatch")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    current = [
        {"qid": qid, "reasoning": str(answer.get("decision_summary", ""))}
        for qid, answer in answers.items()
    ]
    sealed_component = components.get("sealed_reasoning")
    if not isinstance(sealed_component, Mapping):
        raise ValueError("reasoning shadow sealed fingerprint is missing")
    if (
        int(sealed_component.get("count", -1)) != len(current)
        or _payload_sha256(sealed) != str(sealed_component.get("sha256", ""))
        or _payload_sha256(current) != _payload_sha256(sealed)
    ):
        raise ValueError("reasoning shadow sealed text differs from this candidate")

    score_rows = json.loads(
        (evaluation_dir / "reasoning_scores.json").read_text(encoding="utf-8")
    )
    if not isinstance(score_rows, list):
        raise ValueError("reasoning shadow scores must be an array")
    expected_qids = set(answers)
    seen_qids: set[str] = set()
    per_qid_scores: list[float] = []
    dimension_values = {"logical": [], "completeness": [], "clarity": []}
    for row in score_rows:
        if not isinstance(row, Mapping):
            raise ValueError("reasoning shadow score row must be an object")
        qid = str(row.get("qid", "")).strip()
        if not qid or qid in seen_qids or qid not in expected_qids:
            raise ValueError("reasoning shadow score qid coverage is invalid")
        seen_qids.add(qid)
        values = [
            _finite_score(row.get(key), f"{qid}.{key}")
            for key in ("logical", "completeness", "clarity")
        ]
        per_qid_scores.append(fmean(values))
        for key, value in zip(dimension_values, values):
            dimension_values[key].append(value)
    if seen_qids != expected_qids:
        raise ValueError("reasoning shadow score qid coverage is incomplete")

    reasoning_score = fmean(per_qid_scores) if per_qid_scores else 0.0
    aggregate = _read_object(evaluation_dir / "reasoning_aggregate.json")
    if int(aggregate.get("question_count", -1)) != len(answers):
        raise ValueError("reasoning shadow aggregate question count mismatch")
    if not math.isclose(
        float(aggregate.get("reasoning_score", -1)),
        reasoning_score,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("reasoning shadow aggregate score mismatch")
    manifest_aggregate = manifest.get("reasoning_aggregate")
    if (
        not isinstance(manifest_aggregate, Mapping)
        or not math.isclose(
            float(manifest_aggregate.get("reasoning_score", -1)),
            reasoning_score,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("reasoning shadow manifest aggregate mismatch")
    if (
        int(manifest.get("expected_reasoning_count", -1)) != len(answers)
        or int(manifest.get("evaluated_reasoning_count", -1)) != len(answers)
    ):
        raise ValueError("reasoning shadow manifest coverage mismatch")

    return {
        "evaluation_dir": str(evaluation_dir),
        "model_name": REASONING_JUDGE_MODEL,
        "prompt_version": REASONING_PROMPT_VERSION,
        "fingerprint_sha256": fingerprint.get("sha256"),
        "reasoning_score": reasoning_score,
        "dimension_means": {
            key: fmean(values) if values else 0.0
            for key, values in dimension_values.items()
        },
        "failure_count": int(manifest.get("failure_count", 0)),
        "judge_tokens_included_in_submission": False,
    }


def _finite_score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric in 0..100")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise ValueError(f"{name} must be numeric in 0..100")
    return number


def _sum_answer_usage(
    answers: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for qid, answer in answers.items():
        usage = _normalize_usage(answer.get("token_usage"), f"{qid}.token_usage")
        for key in total:
            total[key] += usage[key]
    return total


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_answer_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON array")
    rows: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: answer row must be an object")
        qid = str(item.get("qid", "")).strip()
        parts = item.get("answer_parts")
        if not qid or qid in rows:
            raise ValueError(f"{path}: missing or duplicate qid")
        if (
            not isinstance(parts, list)
            or not parts
            or any(not str(part).strip() for part in parts)
        ):
            raise ValueError(f"{path}: {qid} has invalid answer_parts")
        rows[qid] = item
    return rows


def _normalize_answer_map(payload: object, name: str) -> dict[str, list[str]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    normalized: dict[str, list[str]] = {}
    for raw_qid, raw_parts in payload.items():
        qid = str(raw_qid).strip()
        if not qid or not isinstance(raw_parts, list) or not raw_parts:
            raise ValueError(f"{name} contains an invalid answer")
        normalized[qid] = [str(part).strip() for part in raw_parts]
    return normalized


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}: JSONL row must be an object")
        rows.append(item)
    return rows


def _normalize_usage(payload: object, name: str) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer")
        usage[key] = value
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError(f"{name}.total_tokens is inconsistent")
    return usage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
