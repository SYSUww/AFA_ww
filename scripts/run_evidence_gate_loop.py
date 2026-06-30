#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.client import OpenAICompatibleClient, extract_json_object
from afa_agent.config import build_run_config
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl


DOMAINS = ["regulatory", "financial_reports", "insurance", "research", "financial_contracts"]
DEFAULT_RISK_CSV = ROOT / "artifacts" / "answer_audit" / "group_a_20260628_preprocessed_loop_full" / "risk_cause_breakdown_p0_p1_p2.csv"
DEFAULT_BASELINE = ROOT / "artifacts" / "submissions" / "group_a_20260628_preprocessed_loop_full" / "answer_with_domain.csv"
DEFAULT_STRATEGY = ROOT / "configs" / "autoresearch" / "evidence_gate_rescue.json"
DEFAULT_PARSED_ROOT = ROOT / "artifacts" / "preprocessed_loop_candidates" / "parsed"
DEFAULT_INDEX_ROOT = ROOT / "artifacts" / "preprocessed_loop_candidates" / "index"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "evidence_gate_eval" / "risk53"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_questions() -> dict[str, dict[str, Any]]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    questions: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        rows = read_json(Path(manifest["domains"][domain]["question_path"]))
        for row in rows:
            if row.get("split") == "A":
                questions[row["qid"]] = row
    return questions


def load_baseline_answers(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    return {row["qid"]: row for row in rows if row.get("qid") and row["qid"] != "summary"}


def group_by_domain(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    return dict(grouped)


def write_qid_files(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    qid_dir = ensure_dir(output_dir / "qid_files")
    paths = {}
    for domain, items in sorted(group_by_domain(rows).items()):
        path = qid_dir / f"{domain}.txt"
        path.write_text("\n".join(row["qid"] for row in items) + "\n", encoding="utf-8")
        paths[domain] = path
    return paths


def build_round_strategy(base_strategy_path: Path, round_dir: Path, round_index: int) -> Path:
    if round_index == 0:
        return base_strategy_path
    strategy = read_json(base_strategy_path)
    shared = strategy.setdefault("domains", {}).setdefault("__all__", {})
    retrieval = shared.setdefault("retrieval", {})
    gate = shared.setdefault("evidence_gate", {})
    retrieval["top_k"] = int(retrieval.get("top_k", 8)) + round_index * 2
    retrieval["max_hits_for_prompt"] = int(retrieval.get("max_hits_for_prompt", 8)) + round_index * 2
    gate["rescue_top_k"] = int(gate.get("rescue_top_k", 12)) + round_index * 4
    gate["max_hits_after_rescue"] = int(gate.get("max_hits_after_rescue", 12)) + round_index * 4
    gate["per_doc_quota"] = int(gate.get("per_doc_quota", 2)) + (1 if round_index >= 2 else 0)
    gate["max_rescue_rounds"] = max(int(gate.get("max_rescue_rounds", 7)), 7)
    path = round_dir / "round_strategy.json"
    write_json(path, strategy)
    return path


def run_domain_round(
    *,
    domain: str,
    qid_file: Path,
    round_dir: Path,
    strategy_path: Path,
    parsed_root: Path,
    index_root: Path,
    dry_run: bool,
) -> Path:
    run_id = f"{domain}_round_{round_dir.name.split('_')[-1]}"
    run_root = round_dir / "runs"
    run_dir = run_root / run_id
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_answering.py"),
        "--domain",
        domain,
        "--split",
        "A",
        "--qid-file",
        str(qid_file),
        "--strategy-config",
        str(strategy_path),
        "--parsed-path",
        str(parsed_root / domain / "parsed.json"),
        "--index-path",
        str(index_root / domain / "index.json"),
        "--run-root-dir",
        str(run_root),
        "--run-id",
        run_id,
    ]
    if dry_run:
        write_json(run_dir / "dry_run_command.json", {"cmd": cmd})
        return run_dir
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return run_dir


def load_round_answers(run_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    answers = []
    for run_dir in run_dirs.values():
        answers_path = run_dir / "outputs" / "debug" / "answers.json"
        if answers_path.exists():
            answers.extend(read_json(answers_path))
    return answers


def format_error(answer: str, answer_format: str, options: dict[str, str]) -> str:
    if answer_format == "tf":
        tf_answer = "".join(ch for ch in answer.upper() if ch in {"A", "B"})
        if len(tf_answer) != 1:
            return "tf_requires_a_or_b"
        return ""
    cleaned = "".join(ch for ch in answer.upper() if ch in options)
    if answer_format == "mcq" and len(cleaned) != 1:
        return "mcq_requires_one"
    if answer_format == "multi" and len(set(cleaned)) < 2:
        return "multi_requires_two_or_more"
    return ""


def compact_text(text: str, limit: int = 600) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def parse_probability(value: Any) -> int:
    try:
        return max(0, min(100, int(float(str(value).replace("%", "").strip()))))
    except (TypeError, ValueError):
        return 0


def audit_with_llm(
    *,
    client: OpenAICompatibleClient,
    question: dict[str, Any],
    answer_row: dict[str, Any],
    rule_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    evidence_items = [
        {
            "idx": idx,
            "doc_id": item.get("doc_id", ""),
            "title_path": item.get("title_path", []),
            "text": compact_text(item.get("text", "")),
        }
        for idx, item in enumerate(answer_row.get("evidence_items", [])[:8], start=1)
    ]
    payload = {
        "qid": answer_row["qid"],
        "domain": answer_row["domain"],
        "question": question["question"],
        "answer_format": question["answer_format"],
        "options": question.get("options", {}),
        "pred_answer": answer_row.get("pred_answer", ""),
        "rule_audit": rule_audit,
        "evidence_items": evidence_items,
    }
    response = client.chat_json(
        [
            {
                "role": "system",
                "content": (
                    "你是金融文档问答证据链审计员。只根据给定题目、选项、证据和rule_audit判断答案确定性。"
                    "不要使用外部知识。若证据缺关键文档、关键指标、公式或条款，必须降低分数。只输出JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请输出字段："
                    '{"correct_probability":0-100整数,"evidence_grade":"strong|medium|weak|contradictory",'
                    '"issue_type":"strong_support|partial_support|retrieval_miss|wrong_chunk|partial_doc_coverage|'
                    'metric_parse_error|calculation_error|option_semantics_error|multi_choice_forced|prompt_reasoning_error|question_ambiguous",'
                    '"why":"一句话","suggested_fix":"一句话"}。\n\n'
                    f"样本：{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ]
    )
    parsed = extract_json_object(response.content)
    usage = response.token_usage.to_dict()
    return parsed, usage


def rule_audit_answer(
    *,
    answer_row: dict[str, Any],
    question: dict[str, Any],
    baseline_answer: str,
    high_threshold: float,
) -> dict[str, Any]:
    pred_answer = str(answer_row.get("pred_answer", "")).strip().upper()
    option_debug = answer_row.get("debug_meta", {}).get("option_debug", [])
    final_issues = answer_row.get("debug_meta", {}).get("final_consistency_check", {}).get("issues", [])
    answer_finalization = answer_row.get("debug_meta", {}).get("answer_finalization", {}) or {}
    gates = {}
    rescue_rounds = 0
    for item in option_debug:
        option = str(item.get("option", "")).upper()
        gate = item.get("evidence_gate", {}) or {}
        final_gate = gate.get("final_gate", {}) or {}
        if option:
            gates[option] = final_gate
        rescue_rounds += len(gate.get("rounds", []) or [])
    selected = {ch for ch in pred_answer if ch in question.get("options", {})}
    selected_scores = [
        float(gates[option].get("certainty_score", 0.0))
        for option in selected
        if option in gates
    ]
    all_scores = [float(gate.get("certainty_score", 0.0)) for gate in gates.values() if gate]
    certainty_score = min(selected_scores) if selected_scores else (max(all_scores) if all_scores else 0.0)
    rule_outputs = answer_row.get("debug_meta", {}).get("rule_outputs", []) or []
    rule_confidences = [
        float(item.get("confidence", 0.0))
        for item in rule_outputs
        if item.get("confidence") is not None
    ]
    if certainty_score <= 0 and rule_confidences and answer_row.get("evidence_items"):
        certainty_score = max(rule_confidences)
    expected_docs, expected_doc_scope = expected_docs_for_audit(
        answer_row=answer_row,
        question=question,
        selected=selected,
    )
    evidence_docs = {item.get("doc_id") for item in answer_row.get("evidence_items", []) if item.get("doc_id")}
    missing_docs = sorted(expected_docs - evidence_docs)
    error = format_error(pred_answer, question.get("answer_format", ""), question.get("options", {}))
    low_reasons = []
    if error:
        low_reasons.append(error)
    if not answer_row.get("evidence_items"):
        low_reasons.append("empty_evidence")
    if missing_docs:
        low_reasons.append("missing_doc")
    if final_issues:
        low_reasons.extend(final_issues)
    if answer_finalization.get("format_forced"):
        low_reasons.append("multi_choice_forced" if question.get("answer_format") == "multi" else "answer_format_forced")
    if answer_finalization.get("no_supported_fallback"):
        low_reasons.append("no_supported_option")
    if answer_finalization.get("invalid_model_answer"):
        low_reasons.append("invalid_model_answer")
    if certainty_score < high_threshold:
        low_reasons.append("low_certainty_score")
    changed = bool(baseline_answer and pred_answer != baseline_answer)
    high_certainty = not low_reasons
    blocker_type, retrieval_rescue_needed = classify_blocker(
        low_reasons=low_reasons,
        answer_finalization=answer_finalization,
    )
    return {
        "qid": answer_row["qid"],
        "domain": answer_row["domain"],
        "pred_answer": pred_answer,
        "baseline_answer": baseline_answer,
        "answer_changed": changed,
        "certainty_score": round(certainty_score, 4),
        "high_certainty": high_certainty,
        "low_reasons": sorted(set(low_reasons)),
        "blocker_type": blocker_type,
        "retrieval_rescue_needed": retrieval_rescue_needed,
        "format_error": error,
        "empty_evidence": not bool(answer_row.get("evidence_items")),
        "missing_doc_ids": missing_docs,
        "expected_doc_scope": expected_doc_scope,
        "final_consistency_issues": final_issues,
        "answer_finalization": answer_finalization,
        "rescue_rounds": rescue_rounds,
        "gate_statuses": {option: gate.get("status", "") for option, gate in gates.items()},
        "gate_reasons": {option: gate.get("reasons", []) for option, gate in gates.items()},
        "token_usage": answer_row.get("token_usage", {}),
    }


def classify_blocker(*, low_reasons: list[str], answer_finalization: dict[str, Any]) -> tuple[str, bool]:
    issue_set = set(low_reasons)
    if not issue_set:
        return "none", False

    retrieval_markers = {"empty_evidence", "missing_doc", "low_certainty_score"}
    has_retrieval_gap = bool(issue_set & retrieval_markers) or any(
        reason.startswith("selected_gate_fail") for reason in issue_set
    )
    has_selected_false = any(reason.startswith("selected_false_option") for reason in issue_set)

    if "mcq_ambiguous_supported" in issue_set and not has_retrieval_gap:
        return "question_ambiguity", False

    if "single_supported_multi" in issue_set and not has_retrieval_gap:
        if has_selected_false:
            return "format_forced_false_option", False
        return "question_format_conflict", False

    if has_selected_false and not has_retrieval_gap:
        return "answer_evidence_contradiction", False

    if has_retrieval_gap:
        return "retrieval_or_evidence_gap", True

    if answer_finalization.get("invalid_model_answer") or answer_finalization.get("format_forced"):
        return "answer_format_or_generation", False

    return "answer_or_format_issue", False


def expected_docs_for_audit(
    *,
    answer_row: dict[str, Any],
    question: dict[str, Any],
    selected: set[str],
) -> tuple[set[str], str]:
    question_docs = {str(doc_id) for doc_id in question.get("doc_ids", []) if str(doc_id)}
    option_debug = answer_row.get("debug_meta", {}).get("option_debug", []) or []
    option_docs: set[str] = set()
    for item in option_debug:
        option = str(item.get("option", "")).upper()
        if option not in selected:
            continue
        for doc_id in item.get("search_doc_ids", []) or []:
            if str(doc_id):
                option_docs.add(str(doc_id))
    if selected and option_docs:
        return option_docs, "selected_option_docs"
    return question_docs, "question_docs"


def flatten_rescue_logs(answer_rows: list[dict[str, Any]], audit_by_qid: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    logs = []
    for answer in answer_rows:
        qid = answer["qid"]
        llm_audit = audit_by_qid.get(qid, {}).get("llm_audit_result", {})
        for item in answer.get("debug_meta", {}).get("option_debug", []):
            option = item.get("option", "")
            gate = item.get("evidence_gate", {}) or {}
            initial_gate = gate.get("initial_gate", {})
            final_gate = gate.get("final_gate", {})
            rounds = gate.get("rounds", []) or []
            if not rounds:
                logs.append(
                    {
                        "qid": qid,
                        "domain": answer["domain"],
                        "option": option,
                        "round": 0,
                        "trigger_reason": ",".join(initial_gate.get("rescue_trigger", [])),
                        "retrieval_channel": "initial_gate",
                        "query": "",
                        "top_k": "",
                        "unit_type_boosts": "",
                        "hits_before": "",
                        "hits_after": "",
                        "gate_before": json.dumps(initial_gate, ensure_ascii=False),
                        "gate_after": json.dumps(final_gate, ensure_ascii=False),
                        "certainty_score": final_gate.get("certainty_score", ""),
                        "llm_audit_result": json.dumps(llm_audit, ensure_ascii=False),
                        "token_usage": json.dumps(answer.get("token_usage", {}), ensure_ascii=False),
                    }
                )
            for event in rounds:
                logs.append(
                    {
                        "qid": qid,
                        "domain": answer["domain"],
                        "option": option,
                        "round": event.get("round", ""),
                        "trigger_reason": ",".join(initial_gate.get("rescue_trigger", [])),
                        "retrieval_channel": event.get("retrieval_channel", ""),
                        "query": event.get("query", ""),
                        "top_k": event.get("top_k", ""),
                        "unit_type_boosts": json.dumps(event.get("unit_type_boosts", {}), ensure_ascii=False),
                        "hits_before": event.get("hits_before", ""),
                        "hits_after": event.get("hits_after", ""),
                        "gate_before": json.dumps(event.get("gate_before", {}), ensure_ascii=False),
                        "gate_after": json.dumps(event.get("gate_after", {}), ensure_ascii=False),
                        "certainty_score": (event.get("gate_after", {}) or {}).get("certainty_score", ""),
                        "llm_audit_result": json.dumps(llm_audit, ensure_ascii=False),
                        "token_usage": json.dumps(answer.get("token_usage", {}), ensure_ascii=False),
                    }
                )
    return logs


def audit_round(
    *,
    answer_rows: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    baseline_answers: dict[str, dict[str, str]],
    high_threshold: float,
    llm_audit_mode: str,
    client: OpenAICompatibleClient | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    audits = []
    audit_by_qid = {}
    for answer in answer_rows:
        qid = answer["qid"]
        question = questions[qid]
        baseline = baseline_answers.get(qid, {}).get("answer", "")
        audit = rule_audit_answer(answer_row=answer, question=question, baseline_answer=baseline, high_threshold=high_threshold)
        should_llm_audit = llm_audit_mode == "all" or (
            llm_audit_mode == "low_or_changed" and (not audit["high_certainty"] or audit["answer_changed"])
        )
        if should_llm_audit and client:
            llm_result, llm_usage = audit_with_llm(client=client, question=question, answer_row=answer, rule_audit=audit)
            audit["llm_audit_result"] = llm_result
            audit["llm_token_usage"] = llm_usage
            probability = parse_probability(llm_result.get("correct_probability", 0))
            audit["llm_correct_probability"] = probability
            audit["certainty_score"] = round(min(float(audit["certainty_score"]), probability / 100), 4)
            if probability < high_threshold * 100:
                audit["high_certainty"] = False
                audit["low_reasons"] = sorted(set([*audit["low_reasons"], "llm_low_probability"]))
        else:
            audit["llm_audit_result"] = {}
            audit["llm_token_usage"] = {}
            audit["llm_correct_probability"] = ""
        audits.append(audit)
        audit_by_qid[qid] = audit
    return audits, audit_by_qid


def write_round_reports(round_dir: Path, answer_rows: list[dict[str, Any]], audits: list[dict[str, Any]], audit_by_qid: dict[str, dict[str, Any]]) -> None:
    write_json(round_dir / "answers.json", answer_rows)
    write_json(round_dir / "certainty_audit.json", audits)
    low_rows = [row for row in audits if not row["high_certainty"]]
    write_csv(round_dir / "low_certainty_cases.csv", normalize_csv_rows(low_rows), comparison_fields())
    write_jsonl(round_dir / "rescue_logs.jsonl", flatten_rescue_logs(answer_rows, audit_by_qid))


def comparison_fields() -> list[str]:
    return [
        "qid",
        "domain",
        "baseline_answer",
        "pred_answer",
        "answer_changed",
        "certainty_score",
        "high_certainty",
        "low_reasons",
        "blocker_type",
        "retrieval_rescue_needed",
        "format_error",
        "empty_evidence",
        "missing_doc_ids",
        "final_consistency_issues",
        "answer_finalization",
        "rescue_rounds",
        "gate_statuses",
        "gate_reasons",
        "llm_correct_probability",
        "token_usage",
    ]


def normalize_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = dict(row)
        for key in ["low_reasons", "missing_doc_ids", "final_consistency_issues"]:
            item[key] = ",".join(map(str, item.get(key, [])))
        for key in ["answer_finalization", "gate_statuses", "gate_reasons", "token_usage"]:
            item[key] = json.dumps(item.get(key, {}), ensure_ascii=False)
        normalized.append(item)
    return normalized


def write_final_reports(output_dir: Path, latest_audits: dict[str, dict[str, Any]], risk_rows: list[dict[str, str]], round_summaries: list[dict[str, Any]]) -> None:
    risk_by_qid = {row["qid"]: row for row in risk_rows}
    rows = []
    for qid in [row["qid"] for row in risk_rows]:
        audit = dict(latest_audits[qid])
        risk = risk_by_qid[qid]
        audit["review_priority"] = risk.get("review_priority", "")
        audit["issue_type"] = risk.get("issue_type", "")
        audit["evidence_grade"] = risk.get("evidence_grade", "")
        rows.append(audit)
    write_csv(output_dir / "risk53_comparison.csv", normalize_csv_rows(rows), [
        "qid",
        "domain",
        "review_priority",
        "issue_type",
        "evidence_grade",
        *comparison_fields()[2:],
    ])
    changed = [row for row in rows if row["answer_changed"]]
    low = [row for row in rows if not row["high_certainty"]]
    write_csv(output_dir / "changed_answer_cases.csv", normalize_csv_rows(changed), comparison_fields())
    write_csv(output_dir / "still_low_certainty_cases.csv", normalize_csv_rows(low), comparison_fields())
    summary = {
        "question_count": len(rows),
        "high_certainty_count": sum(1 for row in rows if row["high_certainty"]),
        "high_certainty_ratio": round(sum(1 for row in rows if row["high_certainty"]) / max(1, len(rows)), 4),
        "changed_answer_count": len(changed),
        "low_certainty_count": len(low),
        "by_domain": dict(Counter(row["domain"] for row in rows)),
        "low_by_domain": dict(Counter(row["domain"] for row in low)),
        "rounds": round_summaries,
    }
    write_json(output_dir / "risk53_summary.json", summary)
    lines = [
        "# Evidence Gate Risk53 Loop Summary",
        "",
        f"- questions: {summary['question_count']}",
        f"- high certainty: {summary['high_certainty_count']} ({summary['high_certainty_ratio']:.2%})",
        f"- changed answers: {summary['changed_answer_count']}",
        f"- still low certainty: {summary['low_certainty_count']}",
        "",
        "## Rounds",
    ]
    for item in round_summaries:
        lines.append(
            f"- round {item['round']}: ran {item['ran_count']} qids, high={item['high_certainty_count']}, "
            f"ratio={item['high_certainty_ratio']:.2%}, low={item['low_certainty_count']}"
        )
    if low:
        lines.extend(["", "## Still Low Certainty", ""])
        for row in low:
            lines.append(f"- {row['qid']} ({row['domain']}): {','.join(row['low_reasons'])}")
    (output_dir / "risk53_loop_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-gate loop-engine rescue/eval over risk cases.")
    parser.add_argument("--risk-csv", default=str(DEFAULT_RISK_CSV))
    parser.add_argument("--baseline-answer-csv", default=str(DEFAULT_BASELINE))
    parser.add_argument("--strategy-config", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--target-high-certainty", type=float, default=0.7)
    parser.add_argument("--min-improvement", type=float, default=0.05)
    parser.add_argument("--llm-audit", choices=["off", "low_or_changed", "all"], default="low_or_changed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    risk_rows = read_csv_rows(Path(args.risk_csv))
    if args.limit:
        risk_rows = risk_rows[: args.limit]
    questions = load_questions()
    baseline_answers = load_baseline_answers(Path(args.baseline_answer_csv))
    client = None
    if args.llm_audit != "off" and not args.dry_run:
        config = build_run_config(ROOT)
        if not config.model:
            raise RuntimeError("Missing LLM model config for --llm-audit")
        client = OpenAICompatibleClient(config.model)

    latest_answers: dict[str, dict[str, Any]] = {}
    latest_audits: dict[str, dict[str, Any]] = {}
    remaining_rows = risk_rows[:]
    previous_high_ratio = -1.0
    round_summaries = []

    for round_index in range(args.max_rounds + 1):
        round_dir = ensure_dir(output_dir / f"round_{round_index:02d}")
        strategy_path = build_round_strategy(Path(args.strategy_config), round_dir, round_index)
        qid_files = write_qid_files(remaining_rows, round_dir)
        run_dirs: dict[str, Path] = {}
        jobs = sorted(qid_files.items())
        if args.workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        run_domain_round,
                        domain=domain,
                        qid_file=qid_file,
                        round_dir=round_dir,
                        strategy_path=strategy_path,
                        parsed_root=Path(args.parsed_root),
                        index_root=Path(args.index_root),
                        dry_run=args.dry_run,
                    ): domain
                    for domain, qid_file in jobs
                }
                for future in as_completed(futures):
                    domain = futures[future]
                    run_dirs[domain] = future.result()
        else:
            for domain, qid_file in jobs:
                run_dirs[domain] = run_domain_round(
                    domain=domain,
                    qid_file=qid_file,
                    round_dir=round_dir,
                    strategy_path=strategy_path,
                    parsed_root=Path(args.parsed_root),
                    index_root=Path(args.index_root),
                    dry_run=args.dry_run,
                )
        write_json(
            round_dir / "run_manifest.json",
            {
                "round": round_index,
                "risk_csv": str(Path(args.risk_csv).resolve()),
                "strategy_config": str(strategy_path.resolve()),
                "run_dirs": {domain: str(path.resolve()) for domain, path in run_dirs.items()},
                "qid_count": len(remaining_rows),
                "dry_run": args.dry_run,
            },
        )
        if args.dry_run:
            break

        round_answers = load_round_answers(run_dirs)
        for answer in round_answers:
            latest_answers[answer["qid"]] = answer
        round_audits, round_audit_by_qid = audit_round(
            answer_rows=round_answers,
            questions=questions,
            baseline_answers=baseline_answers,
            high_threshold=args.target_high_certainty,
            llm_audit_mode=args.llm_audit,
            client=client,
        )
        for audit in round_audits:
            latest_audits[audit["qid"]] = audit
        write_round_reports(round_dir, round_answers, round_audits, round_audit_by_qid)

        all_latest = [latest_audits[row["qid"]] for row in risk_rows if row["qid"] in latest_audits]
        high_count = sum(1 for row in all_latest if row["high_certainty"])
        high_ratio = high_count / max(1, len(risk_rows))
        low_qids = {row["qid"] for row in all_latest if not row["high_certainty"]}
        summary = {
            "round": round_index,
            "ran_count": len(round_answers),
            "latest_count": len(all_latest),
            "high_certainty_count": high_count,
            "high_certainty_ratio": round(high_ratio, 4),
            "low_certainty_count": len(low_qids),
            "low_qids": sorted(low_qids),
        }
        round_summaries.append(summary)
        write_json(round_dir / "round_summary.json", summary)

        if high_ratio >= args.target_high_certainty:
            break
        if round_index > 0 and high_ratio - previous_high_ratio < args.min_improvement:
            break
        previous_high_ratio = high_ratio
        remaining_rows = [row for row in risk_rows if row["qid"] in low_qids]
        if not remaining_rows:
            break

    if not args.dry_run and latest_audits:
        write_final_reports(output_dir, latest_audits, risk_rows, round_summaries)
    print(output_dir)


if __name__ == "__main__":
    main()
