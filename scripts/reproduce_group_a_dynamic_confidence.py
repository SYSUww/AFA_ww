#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.client import OpenAICompatibleClient, extract_json_object
from afa_agent.config import build_run_config
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl


DOMAINS = ["regulatory", "financial_reports", "insurance", "research", "financial_contracts"]
DEFAULT_ROUND0_STRATEGY = ROOT / "configs" / "autoresearch" / "default_strategy.json"
DEFAULT_RESCUE_STRATEGY = ROOT / "configs" / "autoresearch" / "evidence_gate_rescue_accuracy_first.json"
DEFAULT_PARSED_ROOT = ROOT / "artifacts" / "preprocessed_loop_candidates" / "parsed"
DEFAULT_INDEX_ROOT = ROOT / "artifacts" / "preprocessed_loop_candidates" / "index"
DEFAULT_ORDER_CSV = ROOT / "artifacts" / "submissions" / "group_a_20260628_preprocessed_loop_full" / "answer.csv"
DEFAULT_REFERENCE_88_CSV = ROOT / "artifacts" / "submissions" / "group_a_candidate_accuracy_first_v20_20260630" / "answer.csv"


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


def token_payload(row: dict[str, Any] | None) -> dict[str, int]:
    payload = (row or {}).get("token_usage", {}) or {}
    return {
        "prompt_tokens": int(payload.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(payload.get("completion_tokens", 0) or 0),
        "total_tokens": int(payload.get("total_tokens", 0) or 0),
    }


def add_tokens(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "prompt_tokens": left.get("prompt_tokens", 0) + right.get("prompt_tokens", 0),
        "completion_tokens": left.get("completion_tokens", 0) + right.get("completion_tokens", 0),
        "total_tokens": left.get("total_tokens", 0) + right.get("total_tokens", 0),
    }


def load_questions() -> tuple[dict[str, dict[str, Any]], list[str]]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    questions: dict[str, dict[str, Any]] = {}
    fallback_order: list[str] = []
    for domain in DOMAINS:
        path = Path(manifest["domains"][domain]["question_path"])
        for row in read_json(path):
            if row.get("split") != "A":
                continue
            questions[row["qid"]] = row
            fallback_order.append(row["qid"])
    return questions, fallback_order


def load_order(order_csv: Path, fallback_order: list[str], questions: dict[str, dict[str, Any]], limit: int) -> list[str]:
    order: list[str] = []
    if order_csv.exists():
        for row in read_csv_rows(order_csv):
            qid = row.get("qid", "")
            if qid and qid != "summary" and qid in questions:
                order.append(qid)
    if not order:
        order = [qid for qid in fallback_order if qid in questions]
    if limit > 0:
        order = order[:limit]
    return order


def write_qid_files(qids: list[str], questions: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    qid_dir = ensure_dir(output_dir / "qid_files")
    grouped: dict[str, list[str]] = defaultdict(list)
    for qid in qids:
        grouped[questions[qid]["domain"]].append(qid)
    paths: dict[str, Path] = {}
    for domain in DOMAINS:
        domain_qids = grouped.get(domain, [])
        if not domain_qids:
            continue
        path = qid_dir / f"{domain}.txt"
        path.write_text("\n".join(domain_qids) + "\n", encoding="utf-8")
        paths[domain] = path
    return paths


def run_domain_batch(
    *,
    stage_name: str,
    qids: list[str],
    questions: dict[str, dict[str, Any]],
    output_dir: Path,
    strategy_path: Path,
    parsed_root: Path,
    index_root: Path,
    workers: int,
    dry_run: bool,
) -> dict[str, Path]:
    stage_dir = ensure_dir(output_dir / stage_name)
    qid_files = write_qid_files(qids, questions, stage_dir)
    run_root = ensure_dir(stage_dir / "runs")
    commands: dict[str, list[str]] = {}

    def run_one(domain: str, qid_file: Path) -> tuple[str, Path]:
        run_id = f"{domain}_{stage_name}"
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
        commands[domain] = cmd
        ensure_dir(run_dir)
        write_json(run_dir / "planned_command.json", {"domain": domain, "stage": stage_name, "cmd": cmd})
        if dry_run:
            return domain, run_dir
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        try:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            write_json(
                stage_dir / "failed_command.json",
                {
                    "stage": stage_name,
                    "domain": domain,
                    "returncode": exc.returncode,
                    "cmd": cmd,
                    "run_dir": str(run_dir),
                },
            )
            raise
        return domain, run_dir

    run_dirs: dict[str, Path] = {}
    jobs = list(qid_files.items())
    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, domain, path): domain for domain, path in jobs}
            for future in as_completed(futures):
                domain, run_dir = future.result()
                run_dirs[domain] = run_dir
    else:
        for domain, path in jobs:
            domain, run_dir = run_one(domain, path)
            run_dirs[domain] = run_dir

    write_json(
        stage_dir / "run_manifest.json",
        {
            "stage": stage_name,
            "strategy_path": str(strategy_path.resolve()),
            "qid_count": len(qids),
            "workers": workers,
            "dry_run": dry_run,
            "commands": commands,
            "run_dirs": {domain: str(path.resolve()) for domain, path in run_dirs.items()},
        },
    )
    return run_dirs


def load_answers(run_dirs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs.values():
        path = run_dir / "outputs" / "debug" / "answers.json"
        if not path.exists():
            continue
        for row in read_json(path):
            answers[row["qid"]] = row
    return answers


def answer_format_error(answer: str, answer_format: str, options: dict[str, str]) -> str:
    cleaned = "".join(ch for ch in str(answer).upper() if ch in options or ch in {"A", "B"})
    if answer_format == "tf":
        return "" if len(cleaned) == 1 and cleaned in {"A", "B"} else "tf_requires_one_a_or_b"
    option_cleaned = "".join(ch for ch in str(answer).upper() if ch in options)
    if answer_format == "mcq":
        return "" if len(option_cleaned) == 1 else "mcq_requires_one"
    if answer_format == "multi":
        return "" if len(set(option_cleaned)) >= 2 else "multi_requires_two_or_more"
    return ""


def option_scores(answer_row: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], bool]:
    model_scores: dict[str, float] = {}
    gate_scores: dict[str, float] = {}
    missing_model_confidence = False
    debug = answer_row.get("debug_meta", {}) or {}
    top_confidence = debug.get("model_confidence")
    if top_confidence is not None:
        try:
            model_scores["__answer__"] = float(top_confidence)
        except (TypeError, ValueError):
            missing_model_confidence = True
    for item in debug.get("rule_outputs", []) or []:
        option = str(item.get("option") or item.get("answer") or "").upper()
        if not option:
            option = "__rule__"
        try:
            model_scores[option] = float(item.get("confidence", 0.95))
        except (TypeError, ValueError):
            model_scores[option] = 0.95
    for item in debug.get("option_debug", []) or []:
        option = str(item.get("option", "")).upper()
        if not option:
            continue
        if item.get("model_confidence") is not None:
            try:
                model_scores[option] = float(item.get("model_confidence"))
            except (TypeError, ValueError):
                missing_model_confidence = True
        else:
            missing_model_confidence = True
        gate = item.get("evidence_gate", {}) or {}
        final_gate = gate.get("final_gate", {}) or {}
        if final_gate.get("certainty_score") is not None:
            try:
                gate_scores[option] = float(final_gate.get("certainty_score"))
            except (TypeError, ValueError):
                pass
    return model_scores, gate_scores, missing_model_confidence


def selected_score(scores: dict[str, float], selected: set[str]) -> float | None:
    if "__answer__" in scores:
        return scores["__answer__"]
    option_scores_only = {key: value for key, value in scores.items() if key and not key.startswith("__")}
    if selected and option_scores_only:
        selected_values = [option_scores_only[option] for option in selected if option in option_scores_only]
        if selected_values:
            return min(selected_values)
    if option_scores_only:
        return max(option_scores_only.values())
    if scores:
        return max(scores.values())
    return None


def audit_answer(answer_row: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    pred_answer = str(answer_row.get("pred_answer", "")).strip().upper()
    options = question.get("options", {}) or {"A": "", "B": ""}
    selected = {ch for ch in pred_answer if ch in options or question.get("answer_format") == "tf" and ch in {"A", "B"}}
    model_scores, gate_scores, missing_model_confidence = option_scores(answer_row)
    model_score = selected_score(model_scores, selected)
    gate_score = selected_score(gate_scores, selected)
    score_parts = [score for score in [model_score, gate_score] if score is not None]
    score = min(score_parts) if score_parts else 0.5
    low_reasons: list[str] = []
    if not score_parts:
        low_reasons.append("missing_confidence")
    elif missing_model_confidence and model_score is None:
        low_reasons.append("missing_model_confidence")

    evidence_items = answer_row.get("evidence_items", []) or []
    evidence_docs = {str(item.get("doc_id", "")) for item in evidence_items if item.get("doc_id")}
    expected_docs = {str(doc_id) for doc_id in question.get("doc_ids", []) if str(doc_id)}
    missing_docs = sorted(expected_docs - evidence_docs)
    if not evidence_items:
        low_reasons.append("empty_evidence")
        score = min(score, 0.2)
    elif missing_docs and len(expected_docs) > 1:
        low_reasons.append("partial_doc_coverage")
        score = min(score, 0.65)

    fmt_error = answer_format_error(pred_answer, question.get("answer_format", ""), options)
    if fmt_error:
        low_reasons.append(fmt_error)
        score = min(score, 0.35)

    debug = answer_row.get("debug_meta", {}) or {}
    final_issues = debug.get("final_consistency_check", {}).get("issues", []) or []
    if final_issues:
        low_reasons.extend(str(item) for item in final_issues)
        score = min(score, 0.55)
    finalization = debug.get("answer_finalization", {}) or {}
    if finalization.get("format_forced"):
        low_reasons.append("format_forced")
        score = min(score, 0.45)
    if finalization.get("no_supported_fallback"):
        low_reasons.append("no_supported_fallback")
        score = min(score, 0.35)
    if finalization.get("invalid_model_answer"):
        low_reasons.append("invalid_model_answer")
        score = min(score, 0.35)

    score = max(0.0, min(1.0, score))
    return {
        "qid": answer_row["qid"],
        "domain": answer_row["domain"],
        "answer_format": question.get("answer_format", ""),
        "pred_answer": pred_answer,
        "confidence_score": round(score, 4),
        "model_confidence": "" if model_score is None else round(model_score, 4),
        "gate_certainty": "" if gate_score is None else round(gate_score, 4),
        "low_reasons": sorted(set(low_reasons)),
        "format_error": fmt_error,
        "missing_doc_ids": missing_docs,
        "evidence_count": len(evidence_items),
        "token_usage": token_payload(answer_row),
    }


def choose_uncertain_qids(audits: list[dict[str, Any]], target_count: int) -> tuple[list[str], float]:
    ranked = sorted(audits, key=lambda row: (float(row["confidence_score"]), row["qid"]))
    selected = ranked[: min(target_count, len(ranked))]
    threshold = float(selected[-1]["confidence_score"]) if selected else 0.0
    return [row["qid"] for row in selected], threshold


def read_answer_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        row["qid"]: row.get("answer", "")
        for row in read_csv_rows(path)
        if row.get("qid") and row["qid"] != "summary"
    }


def api_preflight(output_dir: Path) -> None:
    config = build_run_config(ROOT)
    if not config.model:
        write_json(output_dir / "api_preflight_failed.json", {"error": "missing_model_config"})
        raise RuntimeError("Missing model config for API preflight")
    client = OpenAICompatibleClient(config.model)
    try:
        response = client.chat_json(
            [
                {"role": "system", "content": "只输出JSON。"},
                {"role": "user", "content": '输出 {"ok": true}。'},
            ]
        )
        parsed = extract_json_object(response.content)
    except Exception as exc:
        write_json(
            output_dir / "api_preflight_failed.json",
            {
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "model": config.model.model_name,
                "api_base": config.model.api_base,
            },
        )
        raise
    write_json(
        output_dir / "api_preflight.json",
        {
            "ok": bool(parsed.get("ok", False)),
            "model": config.model.model_name,
            "token_usage": response.token_usage.to_dict(),
        },
    )


def write_answer_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    prompt_total = sum(int(row["prompt_tokens"]) for row in rows)
    completion_total = sum(int(row["completion_tokens"]) for row in rows)
    total = sum(int(row["total_tokens"]) for row in rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer": "",
                "prompt_tokens": prompt_total,
                "completion_tokens": completion_total,
                "total_tokens": total,
            }
        )
        for row in rows:
            writer.writerow({key: row[key] for key in ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]})


def merge_results(
    *,
    qid_order: list[str],
    questions: dict[str, dict[str, Any]],
    round0_answers: dict[str, dict[str, Any]],
    rescue_answers: dict[str, dict[str, Any]],
    round0_audits: dict[str, dict[str, Any]],
    uncertain_qids: set[str],
    reference_answers: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    answer_rows: list[dict[str, Any]] = []
    answer_with_domain: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for qid in qid_order:
        question = questions[qid]
        round0 = round0_answers[qid]
        rescued = qid in uncertain_qids
        rescue = rescue_answers.get(qid) if rescued else None
        final = rescue or round0
        round0_tokens = token_payload(round0)
        rescue_tokens = token_payload(rescue)
        final_tokens = add_tokens(round0_tokens, rescue_tokens if rescued else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        final_answer = str(final.get("pred_answer", "")).strip().upper()
        audit = round0_audits[qid]
        answer_rows.append(
            {
                "qid": qid,
                "answer": final_answer,
                "prompt_tokens": final_tokens["prompt_tokens"],
                "completion_tokens": final_tokens["completion_tokens"],
                "total_tokens": final_tokens["total_tokens"],
            }
        )
        answer_with_domain.append(
            {
                "qid": qid,
                "answer": final_answer,
                "domain": question["domain"],
                "answer_format": question.get("answer_format", ""),
                "source_round": "rescue" if rescued and rescue else "round0",
                "rescued": rescued,
                "round0_answer": str(round0.get("pred_answer", "")).strip().upper(),
                "rescue_answer": str((rescue or {}).get("pred_answer", "")).strip().upper(),
                "confidence_score": audit["confidence_score"],
                "low_reasons": ",".join(audit.get("low_reasons", [])),
                **final_tokens,
            }
        )
        token_rows.append(
            {
                "qid": qid,
                "domain": question["domain"],
                "rescued": rescued,
                "round0_prompt_tokens": round0_tokens["prompt_tokens"],
                "round0_completion_tokens": round0_tokens["completion_tokens"],
                "round0_total_tokens": round0_tokens["total_tokens"],
                "rescue_prompt_tokens": rescue_tokens["prompt_tokens"],
                "rescue_completion_tokens": rescue_tokens["completion_tokens"],
                "rescue_total_tokens": rescue_tokens["total_tokens"],
                "final_prompt_tokens": final_tokens["prompt_tokens"],
                "final_completion_tokens": final_tokens["completion_tokens"],
                "final_total_tokens": final_tokens["total_tokens"],
            }
        )
        reference = reference_answers.get(qid, "")
        comparison_rows.append(
            {
                "qid": qid,
                "domain": question["domain"],
                "reference_answer": reference,
                "reproduced_answer": final_answer,
                "matches_reference": bool(reference and reference == final_answer),
                "round0_answer": str(round0.get("pred_answer", "")).strip().upper(),
                "rescue_answer": str((rescue or {}).get("pred_answer", "")).strip().upper(),
                "rescued": rescued,
                "confidence_score": audit["confidence_score"],
                "low_reasons": ",".join(audit.get("low_reasons", [])),
            }
        )
    return answer_rows, answer_with_domain, token_rows, comparison_rows


def write_report(
    *,
    output_dir: Path,
    qid_order: list[str],
    uncertain_qids: list[str],
    threshold: float,
    answer_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    round0_run_dirs: dict[str, Path],
    rescue_run_dirs: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    total_prompt = sum(int(row["prompt_tokens"]) for row in answer_rows)
    total_completion = sum(int(row["completion_tokens"]) for row in answer_rows)
    total_tokens = sum(int(row["total_tokens"]) for row in answer_rows)
    diff_rows = [row for row in comparison_rows if row["reference_answer"] and not row["matches_reference"]]
    config = build_run_config(ROOT)
    model_name = config.model.model_name if config.model else "unknown"
    lines = [
        "# Group A Dynamic Confidence Reproduction",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- model: `{model_name}`",
        f"- question_count: `{len(qid_order)}`",
        f"- uncertain_target_count: `{args.low_confidence_count}`",
        f"- uncertain_selected_count: `{len(uncertain_qids)}`",
        f"- effective_confidence_threshold: `{threshold:.4f}`",
        f"- prompt_tokens: `{total_prompt}`",
        f"- completion_tokens: `{total_completion}`",
        f"- total_tokens: `{total_tokens}`",
        f"- diff_vs_current_88: `{len(diff_rows)}`",
        "",
        "## Flow",
        "",
        "1. Round0 runs all selected Group A questions with the baseline strategy and records evidence-based confidence.",
        "2. The script ranks questions by confidence and selects the lowest-confidence questions, defaulting to 53 to mirror the historical risk set size.",
        "3. Round1 reruns only those low-confidence questions with the accuracy-first evidence gate rescue strategy.",
        "4. Final answer uses Round1 for rescued questions and Round0 for the rest; token usage is Round0 plus Round1 for rescued questions.",
        "",
        "## Output Files",
        "",
        "- `answer.csv`: submission-style final answers with recomputed token totals.",
        "- `answer_with_domain.csv`: final answers plus source round and confidence fields.",
        "- `confidence_audit.csv`: Round0 confidence audit for every question.",
        "- `uncertain_qids.txt`: dynamically selected rescue set.",
        "- `token_usage_breakdown.csv`: per-question token accounting.",
        "- `comparison_vs_current_88.csv`: diff against the current v20 88%-measured answer vector.",
        "- `round0/` and `rescue/`: raw run outputs from `run_answering.py`.",
        "",
        "## Run Dirs",
        "",
        f"- round0: `{json.dumps({k: str(v) for k, v in round0_run_dirs.items()}, ensure_ascii=False)}`",
        f"- rescue: `{json.dumps({k: str(v) for k, v in rescue_run_dirs.items()}, ensure_ascii=False)}`",
    ]
    if diff_rows:
        lines.extend(["", "## Diff Vs Current 88", ""])
        for row in diff_rows[:30]:
            lines.append(
                f"- `{row['qid']}`: reference `{row['reference_answer']}` vs reproduced `{row['reproduced_answer']}` "
                f"(round0 `{row['round0_answer']}`, rescue `{row['rescue_answer']}`)"
            )
        if len(diff_rows) > 30:
            lines.append(f"- ... {len(diff_rows) - 30} more rows in `comparison_vs_current_88.csv`")
    (output_dir / "reproduce_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Group A with dynamic confidence and evidence-gate rescue.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--round0-strategy", default=str(DEFAULT_ROUND0_STRATEGY))
    parser.add_argument("--rescue-strategy", default=str(DEFAULT_RESCUE_STRATEGY))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT))
    parser.add_argument("--order-csv", default=str(DEFAULT_ORDER_CSV))
    parser.add_argument("--reference-answer-csv", default=str(DEFAULT_REFERENCE_88_CSV))
    parser.add_argument("--low-confidence-count", type=int, default=53)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-api-preflight", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("group_a_dynamic_confidence_%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "reproducible_runs" / run_id)
    questions, fallback_order = load_questions()
    qid_order = load_order(Path(args.order_csv), fallback_order, questions, args.limit)
    if not qid_order:
        raise RuntimeError("No Group A qids found")
    if not args.dry_run and not args.skip_api_preflight:
        api_preflight(output_dir)

    round0_run_dirs = run_domain_batch(
        stage_name="round0",
        qids=qid_order,
        questions=questions,
        output_dir=output_dir,
        strategy_path=Path(args.round0_strategy),
        parsed_root=Path(args.parsed_root),
        index_root=Path(args.index_root),
        workers=args.workers,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(output_dir)
        return
    round0_answers = load_answers(round0_run_dirs)
    missing_round0 = [qid for qid in qid_order if qid not in round0_answers]
    if missing_round0:
        raise RuntimeError(f"Round0 missing answers: {missing_round0}")

    audits = [audit_answer(round0_answers[qid], questions[qid]) for qid in qid_order]
    uncertain_qids, threshold = choose_uncertain_qids(audits, args.low_confidence_count)
    (output_dir / "uncertain_qids.txt").write_text("\n".join(uncertain_qids) + "\n", encoding="utf-8")
    audit_by_qid = {row["qid"]: row for row in audits}
    write_csv(
        output_dir / "confidence_audit.csv",
        [
            {
                **row,
                "low_reasons": ",".join(row.get("low_reasons", [])),
                "missing_doc_ids": ",".join(row.get("missing_doc_ids", [])),
                "token_usage": json.dumps(row.get("token_usage", {}), ensure_ascii=False),
            }
            for row in audits
        ],
        [
            "qid",
            "domain",
            "answer_format",
            "pred_answer",
            "confidence_score",
            "model_confidence",
            "gate_certainty",
            "low_reasons",
            "format_error",
            "missing_doc_ids",
            "evidence_count",
            "token_usage",
        ],
    )

    rescue_run_dirs = run_domain_batch(
        stage_name="rescue",
        qids=uncertain_qids,
        questions=questions,
        output_dir=output_dir,
        strategy_path=Path(args.rescue_strategy),
        parsed_root=Path(args.parsed_root),
        index_root=Path(args.index_root),
        workers=args.workers,
        dry_run=False,
    )
    rescue_answers = load_answers(rescue_run_dirs)
    missing_rescue = [qid for qid in uncertain_qids if qid not in rescue_answers]
    if missing_rescue:
        raise RuntimeError(f"Rescue missing answers: {missing_rescue}")

    reference_answers = read_answer_csv(Path(args.reference_answer_csv))
    answer_rows, answer_with_domain, token_rows, comparison_rows = merge_results(
        qid_order=qid_order,
        questions=questions,
        round0_answers=round0_answers,
        rescue_answers=rescue_answers,
        round0_audits=audit_by_qid,
        uncertain_qids=set(uncertain_qids),
        reference_answers=reference_answers,
    )
    write_answer_csv(output_dir / "answer.csv", answer_rows)
    write_csv(
        output_dir / "answer_with_domain.csv",
        answer_with_domain,
        [
            "qid",
            "answer",
            "domain",
            "answer_format",
            "source_round",
            "rescued",
            "round0_answer",
            "rescue_answer",
            "confidence_score",
            "low_reasons",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ],
    )
    write_csv(
        output_dir / "token_usage_breakdown.csv",
        token_rows,
        [
            "qid",
            "domain",
            "rescued",
            "round0_prompt_tokens",
            "round0_completion_tokens",
            "round0_total_tokens",
            "rescue_prompt_tokens",
            "rescue_completion_tokens",
            "rescue_total_tokens",
            "final_prompt_tokens",
            "final_completion_tokens",
            "final_total_tokens",
        ],
    )
    write_csv(
        output_dir / "comparison_vs_current_88.csv",
        comparison_rows,
        [
            "qid",
            "domain",
            "reference_answer",
            "reproduced_answer",
            "matches_reference",
            "round0_answer",
            "rescue_answer",
            "rescued",
            "confidence_score",
            "low_reasons",
        ],
    )
    write_json(output_dir / "final_answers.json", [round0_answers[qid] if qid not in rescue_answers else rescue_answers[qid] for qid in qid_order])
    write_jsonl(output_dir / "audit_logs.jsonl", audits)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "qid_count": len(qid_order),
            "round0_strategy": str(Path(args.round0_strategy).resolve()),
            "rescue_strategy": str(Path(args.rescue_strategy).resolve()),
            "parsed_root": str(Path(args.parsed_root).resolve()),
            "index_root": str(Path(args.index_root).resolve()),
            "order_csv": str(Path(args.order_csv).resolve()),
            "reference_answer_csv": str(Path(args.reference_answer_csv).resolve()),
            "low_confidence_count": args.low_confidence_count,
            "effective_confidence_threshold": threshold,
            "round0_run_dirs": {domain: str(path.resolve()) for domain, path in round0_run_dirs.items()},
            "rescue_run_dirs": {domain: str(path.resolve()) for domain, path in rescue_run_dirs.items()},
        },
    )
    write_report(
        output_dir=output_dir,
        qid_order=qid_order,
        uncertain_qids=uncertain_qids,
        threshold=threshold,
        answer_rows=answer_rows,
        comparison_rows=comparison_rows,
        round0_run_dirs=round0_run_dirs,
        rescue_run_dirs=rescue_run_dirs,
        args=args,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
