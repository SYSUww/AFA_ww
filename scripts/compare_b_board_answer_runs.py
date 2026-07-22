#!/usr/bin/env python3
"""Compare two no-docids answer runs against the Group A v20 reference.

The comparison is deliberately QID-based.  It never relies on CSV row order or
on where an ``answer.csv`` summary row happens to be placed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


EXPECTED_DOMAIN_COUNTS = {
    "financial_reports": 20,
    "insurance": 20,
    "regulatory": 20,
    "research": 18,
}
DOMAIN_ORDER = {domain: index for index, domain in enumerate(EXPECTED_DOMAIN_COUNTS)}
SUPPORT_STATUSES = {
    "supported",
    "weak_supported",
    "unsupported",
    "contradicted",
    "format_conflict",
}
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
ANSWER_RE = re.compile(r"^[ABCD]+$")


class IntegrityError(ValueError):
    """Raised when a comparison input violates the expected artifact contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def read_json(path: Path) -> Any:
    require(path.is_file(), f"missing required JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"invalid JSON in {path}: {exc}") from exc


def read_csv(path: Path, required_fields: Iterable[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"missing required CSV file: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = set(required_fields) - fields
            require(not missing, f"{path} is missing columns: {sorted(missing)}")
            return list(reader)
    except UnicodeError as exc:
        raise IntegrityError(f"cannot decode CSV {path}: {exc}") from exc


def index_unique(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        qid = str(row.get("qid", "")).strip()
        require(qid, f"{path}:{line_number} has an empty qid")
        require(qid not in indexed, f"{path} contains duplicate qid {qid!r}")
        indexed[qid] = row
    return indexed


def parse_int(value: Any, context: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{context} must be an integer, got {value!r}") from exc
    return parsed


def parse_float(value: Any, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{context} must be numeric, got {value!r}") from exc


def validate_tokens(row: dict[str, Any], context: str) -> tuple[int, int, int]:
    prompt = parse_int(row.get("prompt_tokens"), f"{context}.prompt_tokens")
    completion = parse_int(row.get("completion_tokens"), f"{context}.completion_tokens")
    total = parse_int(row.get("total_tokens"), f"{context}.total_tokens")
    require(prompt >= 0 and completion >= 0 and total >= 0, f"{context} has negative token counts")
    require(prompt + completion == total, f"{context} token mismatch: {prompt} + {completion} != {total}")
    return prompt, completion, total


def validate_answer(answer: str, answer_format: str, context: str) -> None:
    require(bool(ANSWER_RE.fullmatch(answer)), f"{context} has illegal answer {answer!r}")
    require(len(answer) == len(set(answer)), f"{context} has duplicate answer labels: {answer!r}")
    require(answer == "".join(sorted(answer)), f"{context} answer labels are not in canonical order: {answer!r}")
    if answer_format == "tf":
        require(len(answer) == 1 and answer in {"A", "B"}, f"{context} tf answer must be one of A/B")
    elif answer_format == "mcq":
        require(len(answer) == 1, f"{context} mcq answer must contain exactly one label")
    elif answer_format == "multi":
        require(2 <= len(answer) <= 4, f"{context} multi answer must contain two to four labels")
    else:
        raise IntegrityError(f"{context} has unknown answer_format {answer_format!r}")


def assert_same_qids(label: str, expected: set[str], actual: set[str]) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    require(not missing and not extra, f"{label} qid mismatch; missing={missing}, extra={extra}")


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def close_enough(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(left - right) <= tolerance


def validate_run(
    run_dir: Path,
    label: str,
    expected_question_count: int,
    expected_attempt: str | None,
) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    answer_domain_path = run_dir / "answer_with_domain.csv"
    token_path = run_dir / "token_usage_breakdown.csv"
    audit_path = run_dir / "evidence_answer_audit.csv"
    audit_summary_path = run_dir / "evidence_answer_audit_summary.json"
    answer_path = run_dir / "answer.csv"

    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), f"{manifest_path} must contain a JSON object")
    attempt = manifest.get("attempt") or {}
    require(isinstance(attempt, dict), f"{manifest_path}.attempt must be an object")
    attempt_id = str(attempt.get("attempt_id", "")).strip()
    require(attempt_id, f"{manifest_path} is missing attempt.attempt_id")
    if expected_attempt:
        require(
            attempt_id == expected_attempt,
            f"{label} attempt mismatch: expected {expected_attempt!r}, found {attempt_id!r}",
        )

    manifest_count = parse_int(manifest.get("question_count"), f"{manifest_path}.question_count")
    failed_count = parse_int(manifest.get("failed_count"), f"{manifest_path}.failed_count")
    failed_qids = manifest.get("failed_qids")
    require(isinstance(failed_qids, list), f"{manifest_path}.failed_qids must be a list")
    require(manifest_count == expected_question_count, f"{label} expected {expected_question_count} questions, got {manifest_count}")
    require(failed_count == 0, f"{label} has failed_count={failed_count}; failed_qids={failed_qids}")
    require(not failed_qids, f"{label} has non-empty failed_qids: {failed_qids}")

    answer_rows = read_csv(
        answer_domain_path,
        ["qid", "answer", "domain", "answer_format", *TOKEN_FIELDS],
    )
    answer_by_qid = index_unique(answer_rows, answer_domain_path)
    require(len(answer_by_qid) == expected_question_count, f"{answer_domain_path} expected {expected_question_count} rows, got {len(answer_by_qid)}")
    require("summary" not in answer_by_qid, f"{answer_domain_path} must not contain a summary row")

    domain_counts = Counter()
    answer_tokens: dict[str, tuple[int, int, int]] = {}
    for qid, row in answer_by_qid.items():
        answer = str(row["answer"]).strip().upper()
        answer_format = str(row["answer_format"]).strip()
        domain = str(row["domain"]).strip()
        validate_answer(answer, answer_format, f"{answer_domain_path}:{qid}")
        require(domain in EXPECTED_DOMAIN_COUNTS, f"{answer_domain_path}:{qid} has unexpected domain {domain!r}")
        domain_counts[domain] += 1
        answer_tokens[qid] = validate_tokens(row, f"{answer_domain_path}:{qid}")
        row["answer"] = answer
        row["answer_format"] = answer_format
        row["domain"] = domain
    require(dict(domain_counts) == EXPECTED_DOMAIN_COUNTS, f"{label} domain counts mismatch: expected {EXPECTED_DOMAIN_COUNTS}, got {dict(domain_counts)}")

    token_rows = read_csv(token_path, ["qid", "domain", *TOKEN_FIELDS])
    token_by_qid = index_unique(token_rows, token_path)
    assert_same_qids(str(token_path), set(answer_by_qid), set(token_by_qid))
    for qid, row in token_by_qid.items():
        require(row["domain"].strip() == answer_by_qid[qid]["domain"], f"{token_path}:{qid} domain differs from answer_with_domain.csv")
        token_tuple = validate_tokens(row, f"{token_path}:{qid}")
        require(token_tuple == answer_tokens[qid], f"{token_path}:{qid} token values differ from answer_with_domain.csv")

    answer_csv_rows = read_csv(answer_path, ["qid", "answer", *TOKEN_FIELDS])
    summary_rows = [row for row in answer_csv_rows if row["qid"].strip() == "summary"]
    data_rows = [row for row in answer_csv_rows if row["qid"].strip() != "summary"]
    require(len(summary_rows) == 1, f"{answer_path} must contain exactly one summary row, got {len(summary_rows)}")
    answer_csv_by_qid = index_unique(data_rows, answer_path)
    assert_same_qids(str(answer_path), set(answer_by_qid), set(answer_csv_by_qid))
    for qid, row in answer_csv_by_qid.items():
        require(row["answer"].strip().upper() == answer_by_qid[qid]["answer"], f"{answer_path}:{qid} answer differs from answer_with_domain.csv")
        token_tuple = validate_tokens(row, f"{answer_path}:{qid}")
        require(token_tuple == answer_tokens[qid], f"{answer_path}:{qid} token values differ from answer_with_domain.csv")

    prompt_total = sum(tokens[0] for tokens in answer_tokens.values())
    completion_total = sum(tokens[1] for tokens in answer_tokens.values())
    token_total = sum(tokens[2] for tokens in answer_tokens.values())
    summary_tokens = validate_tokens(summary_rows[0], f"{answer_path}:summary")
    require(
        summary_tokens == (prompt_total, completion_total, token_total),
        f"{answer_path} summary tokens {summary_tokens} differ from row sums {(prompt_total, completion_total, token_total)}",
    )
    manifest_tokens = parse_int(manifest.get("total_tokens"), f"{manifest_path}.total_tokens")
    require(manifest_tokens == token_total, f"{manifest_path}.total_tokens={manifest_tokens} but CSV rows sum to {token_total}")

    audit_rows = read_csv(
        audit_path,
        ["qid", "domain", "answer_format", "pred_answer", "support_status", *TOKEN_FIELDS],
    )
    audit_by_qid = index_unique(audit_rows, audit_path)
    assert_same_qids(str(audit_path), set(answer_by_qid), set(audit_by_qid))
    status_counts: Counter[str] = Counter()
    domain_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    domain_token_totals: Counter[str] = Counter()
    for qid, row in audit_by_qid.items():
        answer_row = answer_by_qid[qid]
        domain = row["domain"].strip()
        answer_format = row["answer_format"].strip()
        pred_answer = row["pred_answer"].strip().upper()
        status = row["support_status"].strip()
        require(domain == answer_row["domain"], f"{audit_path}:{qid} domain differs from answer_with_domain.csv")
        require(answer_format == answer_row["answer_format"], f"{audit_path}:{qid} answer_format differs from answer_with_domain.csv")
        require(pred_answer == answer_row["answer"], f"{audit_path}:{qid} pred_answer differs from answer_with_domain.csv")
        require(status in SUPPORT_STATUSES, f"{audit_path}:{qid} has unknown support_status {status!r}")
        audit_tokens = validate_tokens(row, f"{audit_path}:{qid}")
        require(audit_tokens == answer_tokens[qid], f"{audit_path}:{qid} token values differ from answer_with_domain.csv")
        row["pred_answer"] = pred_answer
        row["support_status"] = status
        status_counts[status] += 1
        domain_status_counts[domain][status] += 1
        domain_token_totals[domain] += audit_tokens[2]

    audit_summary = read_json(audit_summary_path)
    require(isinstance(audit_summary, dict), f"{audit_summary_path} must contain a JSON object")
    require(parse_int(audit_summary.get("question_count"), f"{audit_summary_path}.question_count") == expected_question_count, f"{audit_summary_path} question_count mismatch")
    for status in SUPPORT_STATUSES:
        summary_count = parse_int(audit_summary.get(status), f"{audit_summary_path}.{status}")
        require(summary_count == status_counts[status], f"{audit_summary_path}.{status}={summary_count}, raw audit has {status_counts[status]}")
    require(sum(status_counts.values()) == expected_question_count, f"{label} evidence status counts do not sum to {expected_question_count}: {dict(status_counts)}")
    supported_rate = ratio(status_counts["supported"], expected_question_count)
    supported_or_weak_rate = ratio(status_counts["supported"] + status_counts["weak_supported"], expected_question_count)
    require(close_enough(parse_float(audit_summary.get("supported_rate"), f"{audit_summary_path}.supported_rate"), supported_rate), f"{audit_summary_path}.supported_rate differs from raw audit")
    require(close_enough(parse_float(audit_summary.get("supported_or_weak_rate"), f"{audit_summary_path}.supported_or_weak_rate"), supported_or_weak_rate), f"{audit_summary_path}.supported_or_weak_rate differs from raw audit")

    summary_by_domain = audit_summary.get("by_domain")
    require(isinstance(summary_by_domain, dict), f"{audit_summary_path}.by_domain must be an object")
    require(set(summary_by_domain) == set(EXPECTED_DOMAIN_COUNTS), f"{audit_summary_path}.by_domain keys mismatch: {sorted(summary_by_domain)}")
    for domain, expected_count in EXPECTED_DOMAIN_COUNTS.items():
        payload = summary_by_domain[domain]
        require(isinstance(payload, dict), f"{audit_summary_path}.by_domain.{domain} must be an object")
        require(parse_int(payload.get("question_count"), f"{audit_summary_path}.{domain}.question_count") == expected_count, f"{audit_summary_path}.{domain} question_count mismatch")
        for status in SUPPORT_STATUSES:
            count = parse_int(payload.get(status), f"{audit_summary_path}.{domain}.{status}")
            require(count == domain_status_counts[domain][status], f"{audit_summary_path}.{domain}.{status}={count}, raw audit has {domain_status_counts[domain][status]}")
        require(parse_int(payload.get("total_tokens"), f"{audit_summary_path}.{domain}.total_tokens") == domain_token_totals[domain], f"{audit_summary_path}.{domain}.total_tokens differs from raw audit")
        domain_supported_rate = ratio(domain_status_counts[domain]["supported"], expected_count)
        domain_supported_or_weak_rate = ratio(
            domain_status_counts[domain]["supported"] + domain_status_counts[domain]["weak_supported"],
            expected_count,
        )
        require(close_enough(parse_float(payload.get("supported_rate"), f"{audit_summary_path}.{domain}.supported_rate"), domain_supported_rate), f"{audit_summary_path}.{domain}.supported_rate differs from raw audit")
        require(close_enough(parse_float(payload.get("supported_or_weak_rate"), f"{audit_summary_path}.{domain}.supported_or_weak_rate"), domain_supported_or_weak_rate), f"{audit_summary_path}.{domain}.supported_or_weak_rate differs from raw audit")

    return {
        "run_dir": run_dir.resolve(),
        "manifest": manifest,
        "attempt": attempt,
        "attempt_id": attempt_id,
        "answer_by_qid": answer_by_qid,
        "audit_by_qid": audit_by_qid,
        "token_by_qid": answer_tokens,
        "status_counts": status_counts,
        "domain_status_counts": domain_status_counts,
        "domain_token_totals": domain_token_totals,
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "total_tokens": token_total,
        "zero_token_questions": sum(tokens[2] == 0 for tokens in answer_tokens.values()),
        "supported_rate": supported_rate,
        "supported_or_weak_rate": supported_or_weak_rate,
    }


def validate_reference(path: Path, expected_count: int) -> dict[str, Any]:
    rows = read_csv(path, ["qid", "answer", *TOKEN_FIELDS])
    summary_rows = [row for row in rows if row["qid"].strip() == "summary"]
    data_rows = [row for row in rows if row["qid"].strip() != "summary"]
    require(len(summary_rows) == 1, f"{path} must contain exactly one summary row, got {len(summary_rows)}")
    by_qid = index_unique(data_rows, path)
    require(len(by_qid) == expected_count, f"{path} expected {expected_count} reference rows, got {len(by_qid)}")
    tokens: dict[str, tuple[int, int, int]] = {}
    for qid, row in by_qid.items():
        answer = row["answer"].strip().upper()
        require(bool(ANSWER_RE.fullmatch(answer)), f"{path}:{qid} has illegal reference answer {answer!r}")
        require(len(answer) == len(set(answer)), f"{path}:{qid} has duplicate reference labels: {answer!r}")
        require(answer == "".join(sorted(answer)), f"{path}:{qid} reference labels are not in canonical order: {answer!r}")
        row["answer"] = answer
        tokens[qid] = validate_tokens(row, f"{path}:{qid}")
    totals = tuple(sum(values[index] for values in tokens.values()) for index in range(3))
    summary_tokens = validate_tokens(summary_rows[0], f"{path}:summary")
    require(summary_tokens == totals, f"{path} summary tokens {summary_tokens} differ from row sums {totals}")
    return {
        "path": path.resolve(),
        "by_qid": by_qid,
        "token_by_qid": tokens,
        "prompt_tokens": totals[0],
        "completion_tokens": totals[1],
        "total_tokens": totals[2],
        "zero_token_questions": sum(values[2] == 0 for values in tokens.values()),
    }


def proxy_transition(baseline_matches: bool, current_matches: bool) -> str:
    if baseline_matches and current_matches:
        return "stable_match"
    if baseline_matches and not current_matches:
        return "regressed"
    if not baseline_matches and current_matches:
        return "improved"
    return "stable_mismatch"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def serializable_status_counts(counts: Counter[str]) -> dict[str, int]:
    return {status: counts[status] for status in sorted(SUPPORT_STATUSES)}


def run_summary(run: dict[str, Any], matches: int, question_count: int) -> dict[str, Any]:
    return {
        "attempt_id": run["attempt_id"],
        "variant_name": run["attempt"].get("variant_name", ""),
        "question_count": question_count,
        "failed_count": 0,
        "matches_v20": matches,
        "proxy_accuracy_vs_v20": ratio(matches, question_count),
        "manifest_proxy_accuracy_vs_reference_88": parse_float(
            run["manifest"].get("proxy_accuracy_vs_reference_88"),
            f"{run['run_dir']}/run_manifest.json.proxy_accuracy_vs_reference_88",
        ),
        "evidence": {
            **serializable_status_counts(run["status_counts"]),
            "supported_rate": run["supported_rate"],
            "supported_or_weak_rate": run["supported_or_weak_rate"],
        },
        "tokens": {
            "prompt_tokens": run["prompt_tokens"],
            "completion_tokens": run["completion_tokens"],
            "total_tokens": run["total_tokens"],
            "zero_token_questions": run["zero_token_questions"],
        },
    }


def compare(
    current: dict[str, Any],
    baseline: dict[str, Any],
    reference: dict[str, Any],
    expected_question_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    current_qids = set(current["answer_by_qid"])
    baseline_qids = set(baseline["answer_by_qid"])
    reference_qids = set(reference["by_qid"])
    assert_same_qids("current versus baseline", baseline_qids, current_qids)
    missing_reference = sorted(current_qids - reference_qids)
    require(not missing_reference, f"reference answer CSV does not cover comparison qids: {missing_reference}")

    for qid in sorted(current_qids):
        current_row = current["answer_by_qid"][qid]
        baseline_row = baseline["answer_by_qid"][qid]
        require(current_row["domain"] == baseline_row["domain"], f"{qid} domain differs between current and baseline")
        require(current_row["answer_format"] == baseline_row["answer_format"], f"{qid} answer_format differs between current and baseline")

    qids = sorted(current_qids, key=lambda qid: (DOMAIN_ORDER[current["answer_by_qid"][qid]["domain"]], qid))
    per_question: list[dict[str, Any]] = []
    for qid in qids:
        current_answer_row = current["answer_by_qid"][qid]
        baseline_answer_row = baseline["answer_by_qid"][qid]
        reference_row = reference["by_qid"][qid]
        current_answer = current_answer_row["answer"]
        baseline_answer = baseline_answer_row["answer"]
        reference_answer = reference_row["answer"]
        current_matches = current_answer == reference_answer
        baseline_matches = baseline_answer == reference_answer
        current_status = current["audit_by_qid"][qid]["support_status"]
        baseline_status = baseline["audit_by_qid"][qid]["support_status"]
        current_tokens = current["token_by_qid"][qid][2]
        baseline_tokens = baseline["token_by_qid"][qid][2]
        reference_tokens = reference["token_by_qid"][qid][2]
        per_question.append(
            {
                "qid": qid,
                "domain": current_answer_row["domain"],
                "answer_format": current_answer_row["answer_format"],
                "baseline_attempt_id": baseline["attempt_id"],
                "current_attempt_id": current["attempt_id"],
                "baseline_answer": baseline_answer,
                "current_answer": current_answer,
                "v20_answer": reference_answer,
                "answer_changed": baseline_answer != current_answer,
                "baseline_matches_v20": baseline_matches,
                "current_matches_v20": current_matches,
                "proxy_transition": proxy_transition(baseline_matches, current_matches),
                "baseline_support_status": baseline_status,
                "current_support_status": current_status,
                "evidence_transition": f"{baseline_status}->{current_status}",
                "baseline_total_tokens": baseline_tokens,
                "current_total_tokens": current_tokens,
                "delta_tokens_vs_baseline": current_tokens - baseline_tokens,
                "v20_total_tokens": reference_tokens,
                "delta_tokens_vs_v20": current_tokens - reference_tokens,
            }
        )

    by_domain: list[dict[str, Any]] = []
    for domain in EXPECTED_DOMAIN_COUNTS:
        rows = [row for row in per_question if row["domain"] == domain]
        count = len(rows)
        baseline_matches = sum(bool(row["baseline_matches_v20"]) for row in rows)
        current_matches = sum(bool(row["current_matches_v20"]) for row in rows)
        baseline_supported = sum(row["baseline_support_status"] == "supported" for row in rows)
        current_supported = sum(row["current_support_status"] == "supported" for row in rows)
        baseline_supported_or_weak = sum(row["baseline_support_status"] in {"supported", "weak_supported"} for row in rows)
        current_supported_or_weak = sum(row["current_support_status"] in {"supported", "weak_supported"} for row in rows)
        baseline_tokens = sum(int(row["baseline_total_tokens"]) for row in rows)
        current_tokens = sum(int(row["current_total_tokens"]) for row in rows)
        v20_tokens = sum(int(row["v20_total_tokens"]) for row in rows)
        by_domain.append(
            {
                "domain": domain,
                "question_count": count,
                "answer_changed": sum(bool(row["answer_changed"]) for row in rows),
                "baseline_matches_v20": baseline_matches,
                "current_matches_v20": current_matches,
                "baseline_proxy_accuracy": ratio(baseline_matches, count),
                "current_proxy_accuracy": ratio(current_matches, count),
                "delta_proxy_accuracy": ratio(current_matches - baseline_matches, count),
                "baseline_supported": baseline_supported,
                "current_supported": current_supported,
                "delta_supported": current_supported - baseline_supported,
                "baseline_supported_or_weak": baseline_supported_or_weak,
                "current_supported_or_weak": current_supported_or_weak,
                "delta_supported_or_weak": current_supported_or_weak - baseline_supported_or_weak,
                "baseline_total_tokens": baseline_tokens,
                "current_total_tokens": current_tokens,
                "delta_tokens_vs_baseline": current_tokens - baseline_tokens,
                "v20_subset_total_tokens": v20_tokens,
                "delta_tokens_vs_v20": current_tokens - v20_tokens,
                "baseline_zero_token_questions": sum(int(row["baseline_total_tokens"]) == 0 for row in rows),
                "current_zero_token_questions": sum(int(row["current_total_tokens"]) == 0 for row in rows),
            }
        )

    transition_counts = Counter(str(row["proxy_transition"]) for row in per_question)
    evidence_transitions = Counter(str(row["evidence_transition"]) for row in per_question)
    baseline_matches = sum(bool(row["baseline_matches_v20"]) for row in per_question)
    current_matches = sum(bool(row["current_matches_v20"]) for row in per_question)
    baseline_summary = run_summary(baseline, baseline_matches, expected_question_count)
    current_summary = run_summary(current, current_matches, expected_question_count)
    require(
        close_enough(
            baseline_summary["proxy_accuracy_vs_v20"],
            baseline_summary["manifest_proxy_accuracy_vs_reference_88"],
        ),
        f"baseline manifest proxy accuracy {baseline_summary['manifest_proxy_accuracy_vs_reference_88']} differs from recomputed {baseline_summary['proxy_accuracy_vs_v20']}",
    )
    require(
        close_enough(
            current_summary["proxy_accuracy_vs_v20"],
            current_summary["manifest_proxy_accuracy_vs_reference_88"],
        ),
        f"current manifest proxy accuracy {current_summary['manifest_proxy_accuracy_vs_reference_88']} differs from recomputed {current_summary['proxy_accuracy_vs_v20']}",
    )

    reference_subset_tokens = sum(reference["token_by_qid"][qid][2] for qid in current_qids)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sources": {
            "current_run": str(current["run_dir"]),
            "baseline_run": str(baseline["run_dir"]),
            "v20_reference": str(reference["path"]),
        },
        "integrity": {
            "ok": True,
            "question_count": expected_question_count,
            "qid_sets_equal": True,
            "reference_covers_all_qids": True,
            "domain_counts": EXPECTED_DOMAIN_COUNTS,
            "failed_questions": 0,
        },
        "baseline": baseline_summary,
        "current": current_summary,
        "delta_current_minus_baseline": {
            "answer_changed": sum(bool(row["answer_changed"]) for row in per_question),
            "matches_v20": current_matches - baseline_matches,
            "proxy_accuracy": ratio(current_matches - baseline_matches, expected_question_count),
            "proxy_transitions": dict(sorted(transition_counts.items())),
            "supported": current["status_counts"]["supported"] - baseline["status_counts"]["supported"],
            "supported_or_weak": (
                current["status_counts"]["supported"]
                + current["status_counts"]["weak_supported"]
                - baseline["status_counts"]["supported"]
                - baseline["status_counts"]["weak_supported"]
            ),
            "evidence_transitions": dict(sorted(evidence_transitions.items())),
            "total_tokens": current["total_tokens"] - baseline["total_tokens"],
            "zero_token_questions": current["zero_token_questions"] - baseline["zero_token_questions"],
        },
        "v20_reference": {
            "question_count": len(reference["by_qid"]),
            "full_total_tokens": reference["total_tokens"],
            "comparison_subset_count": expected_question_count,
            "comparison_subset_total_tokens": reference_subset_tokens,
            "current_delta_tokens_vs_subset": current["total_tokens"] - reference_subset_tokens,
            "note": "The 78-question subset token total is the comparable value; the 100-question full total is context only.",
        },
        "by_domain": by_domain,
        "metric_note": "Matches against v20 are a regression proxy, not official B-board accuracy. v20 has no evidence audit, so evidence is compared only between the two no-docids runs.",
    }
    return per_question, by_domain, summary


def format_number(value: float) -> str:
    return f"{value:.6f}"


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    current = summary["current"]
    delta = summary["delta_current_minus_baseline"]
    reference = summary["v20_reference"]
    lines = [
        "# B-board No-Docids Answer Comparison",
        "",
        f"- baseline: `{baseline['attempt_id']}` `{baseline['variant_name']}`",
        f"- current: `{current['attempt_id']}` `{current['variant_name']}`",
        f"- comparison questions: `{summary['integrity']['question_count']}`",
        f"- answer changes: `{delta['answer_changed']}`",
        f"- proxy matches vs v20: `{baseline['matches_v20']} -> {current['matches_v20']}` (`{delta['matches_v20']:+d}`)",
        f"- proxy accuracy vs v20: `{format_number(baseline['proxy_accuracy_vs_v20'])} -> {format_number(current['proxy_accuracy_vs_v20'])}` (`{format_number(delta['proxy_accuracy'])}`)",
        f"- evidence supported: `{baseline['evidence']['supported']} -> {current['evidence']['supported']}` (`{delta['supported']:+d}`)",
        f"- evidence supported or weak: `{baseline['evidence']['supported'] + baseline['evidence']['weak_supported']} -> {current['evidence']['supported'] + current['evidence']['weak_supported']}` (`{delta['supported_or_weak']:+d}`)",
        f"- no-docids total tokens: `{baseline['tokens']['total_tokens']} -> {current['tokens']['total_tokens']}` (`{delta['total_tokens']:+d}`)",
        f"- v20 comparable 78-question tokens: `{reference['comparison_subset_total_tokens']}`",
        f"- v20 full 100-question tokens, context only: `{reference['full_total_tokens']}`",
        "",
        "The v20 match rate is a regression proxy, not official B-board accuracy. Evidence is compared only between the two no-docids runs because v20 has no evidence audit.",
        "",
        "## By Domain",
        "",
        "| domain | count | changed | proxy matches | proxy rate | supported | supported or weak | tokens | v20 subset tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_domain"]:
        lines.append(
            f"| `{row['domain']}` | {row['question_count']} | {row['answer_changed']} | "
            f"{row['baseline_matches_v20']} -> {row['current_matches_v20']} | "
            f"{format_number(row['baseline_proxy_accuracy'])} -> {format_number(row['current_proxy_accuracy'])} | "
            f"{row['baseline_supported']} -> {row['current_supported']} | "
            f"{row['baseline_supported_or_weak']} -> {row['current_supported_or_weak']} | "
            f"{row['baseline_total_tokens']} -> {row['current_total_tokens']} | "
            f"{row['v20_subset_total_tokens']} |"
        )
    lines.extend(["", "## Proxy Transitions", ""])
    for key, value in delta["proxy_transitions"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Evidence Transitions", ""])
    for key, value in delta["evidence_transitions"].items():
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two B-board no-docids answer runs against the v20 reference.")
    parser.add_argument("--current", required=True, help="Current no_docids_clean_subset_run directory")
    parser.add_argument("--baseline", required=True, help="Baseline no-docids answer run directory")
    parser.add_argument("--reference", required=True, help="Group A v20 answer.csv")
    parser.add_argument("--output", required=True, help="Output directory for comparison artifacts")
    parser.add_argument("--expect-current-attempt", help="Fail unless current manifest has this attempt_id")
    parser.add_argument("--expect-baseline-attempt", help="Fail unless baseline manifest has this attempt_id")
    parser.add_argument("--expected-question-count", type=int, default=78)
    parser.add_argument("--expected-reference-count", type=int, default=100)
    args = parser.parse_args()

    try:
        require(args.expected_question_count > 0, "--expected-question-count must be positive")
        require(args.expected_reference_count > 0, "--expected-reference-count must be positive")
        current = validate_run(
            Path(args.current),
            "current",
            args.expected_question_count,
            args.expect_current_attempt,
        )
        baseline = validate_run(
            Path(args.baseline),
            "baseline",
            args.expected_question_count,
            args.expect_baseline_attempt,
        )
        reference = validate_reference(Path(args.reference), args.expected_reference_count)
        per_question, by_domain, summary = compare(
            current,
            baseline,
            reference,
            args.expected_question_count,
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            output_dir / "per_question.csv",
            per_question,
            [
                "qid",
                "domain",
                "answer_format",
                "baseline_attempt_id",
                "current_attempt_id",
                "baseline_answer",
                "current_answer",
                "v20_answer",
                "answer_changed",
                "baseline_matches_v20",
                "current_matches_v20",
                "proxy_transition",
                "baseline_support_status",
                "current_support_status",
                "evidence_transition",
                "baseline_total_tokens",
                "current_total_tokens",
                "delta_tokens_vs_baseline",
                "v20_total_tokens",
                "delta_tokens_vs_v20",
            ],
        )
        write_csv(
            output_dir / "by_domain.csv",
            by_domain,
            [
                "domain",
                "question_count",
                "answer_changed",
                "baseline_matches_v20",
                "current_matches_v20",
                "baseline_proxy_accuracy",
                "current_proxy_accuracy",
                "delta_proxy_accuracy",
                "baseline_supported",
                "current_supported",
                "delta_supported",
                "baseline_supported_or_weak",
                "current_supported_or_weak",
                "delta_supported_or_weak",
                "baseline_total_tokens",
                "current_total_tokens",
                "delta_tokens_vs_baseline",
                "v20_subset_total_tokens",
                "delta_tokens_vs_v20",
                "baseline_zero_token_questions",
                "current_zero_token_questions",
            ],
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_summary_markdown(output_dir / "summary.md", summary)
        print(output_dir.resolve())
        return 0
    except IntegrityError as exc:
        print(f"integrity check failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"file operation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
