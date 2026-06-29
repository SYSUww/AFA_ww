#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.client import OpenAICompatibleClient, extract_json_object
from afa_agent.config import build_run_config


DOMAINS = ["regulatory", "financial_reports", "insurance", "research", "financial_contracts"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions() -> dict[str, dict[str, Any]]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    by_qid: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        rows = read_json(Path(manifest["domains"][domain]["question_path"]))
        for row in rows:
            if row.get("split") == "A":
                by_qid[row["qid"]] = row
    return by_qid


def truncate(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    tail = text[-min(160, limit // 4) :]
    return text[: limit - len(tail) - 20] + " ...[truncated]... " + tail


def compact_evidence(items: list[dict[str, Any]], max_items: int, text_chars: int) -> list[dict[str, Any]]:
    compact = []
    for idx, item in enumerate(items[:max_items], start=1):
        compact.append(
            {
                "idx": idx,
                "unit_id": item.get("unit_id", ""),
                "doc_id": item.get("doc_id", ""),
                "score": item.get("score", ""),
                "title_path": item.get("title_path", []),
                "text": truncate(item.get("text", ""), text_chars),
                "metadata": item.get("metadata", {}),
            }
        )
    return compact


def load_debug_answers(mapping_path: Path) -> list[dict[str, Any]]:
    mapping = read_json(mapping_path)
    rows = []
    for domain in DOMAINS:
        answers_path = Path(mapping["domains"][domain]["answers_json"])
        rows.extend(read_json(answers_path))
    rows.sort(key=lambda row: row["qid"])
    return rows


def build_payload(question: dict[str, Any], answer_row: dict[str, Any], args: argparse.Namespace) -> str:
    payload = {
        "qid": answer_row["qid"],
        "domain": answer_row["domain"],
        "question_type": answer_row["question_type"],
        "question": question["question"],
        "options": question.get("options", {}),
        "answer_format": question.get("answer_format", ""),
        "pred_answer": answer_row["pred_answer"],
        "previous_option_labels": answer_row.get("option_labels", {}),
        "previous_reasoning_summary": truncate(answer_row.get("reasoning_summary", ""), args.reasoning_chars),
        "evidence_items": compact_evidence(
            answer_row.get("evidence_items", []),
            max_items=args.max_evidence_items,
            text_chars=args.evidence_text_chars,
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def audit_one(
    question: dict[str, Any],
    answer_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = build_run_config(ROOT)
    if not config.model:
        raise RuntimeError("Missing LLM model config")
    client = OpenAICompatibleClient(config.model)
    q_payload = build_payload(question, answer_row, args)
    response = client.chat_json(
        [
            {
                "role": "system",
                "content": (
                    "你是金融文档问答结果审计员。你只根据给定 evidence_items、题目、选项和预测答案，"
                    "评估该预测答案正确的概率。不要读取外部知识，不要假设 evidence 之外的信息。"
                    "previous_reasoning_summary 和 previous_option_labels 只是上一轮模型的解释，不能当作事实。"
                    "判断重点：被选中的选项是否均被 evidence 直接支持；未选选项是否被 evidence 反驳；"
                    "涉及数值比较、日期、公式、期限时必须能从 evidence 中算出或直接读出。"
                    "如果 evidence 缺关键文档、缺关键指标、只命中泛化段落或与答案矛盾，要显著降分。"
                    "多选题还要检查是否只是被格式规则强制补成两个选项。只输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请审计下面这道题的证据链。\n\n"
                    "概率校准：\n"
                    "- 90-100：证据直接、完整支持答案，且非选项也有明确反驳或排除理由。\n"
                    "- 70-89：主要结论被支持，但有轻微信息缺口或表述需要推断。\n"
                    "- 45-69：部分选项有证据，部分选项缺证据；答案可能对也可能错。\n"
                    "- 20-44：关键证据缺失、错 chunk、跨文档覆盖不足、计算不可验证。\n"
                    "- 0-19：证据直接反驳预测答案，或预测答案明显来自格式兜底/误读。\n\n"
                    "issue_type 只能从这些值中选一个："
                    "strong_support, partial_support, retrieval_miss, wrong_chunk, partial_doc_coverage, "
                    "metric_parse_error, calculation_error, option_semantics_error, multi_choice_forced, "
                    "prompt_reasoning_error, question_ambiguous。\n\n"
                    "输出 JSON 字段：\n"
                    "{\n"
                    '  "correct_probability": 0-100整数,\n'
                    '  "evidence_grade": "strong|medium|weak|contradictory",\n'
                    '  "issue_type": "...",\n'
                    '  "suspected_correct_answer": "若仅凭证据能判断则给选项字母，否则空字符串",\n'
                    '  "risky_options": ["A"],\n'
                    '  "why": "一句话说明为什么这个概率",\n'
                    '  "wrong_reason": "若疑似会错，说明最可能错因；若很稳，写证据充分",\n'
                    '  "suggested_fix": "对检索/chunk/prompt的改进建议"\n'
                    "}\n\n"
                    f"待审计样本：\n{q_payload}"
                ),
            },
        ]
    )
    parsed = extract_json_object(response.content)
    prob = int(parsed.get("correct_probability", 0))
    prob = max(0, min(100, prob))
    return {
        "qid": answer_row["qid"],
        "domain": answer_row["domain"],
        "question_type": answer_row["question_type"],
        "pred_answer": answer_row["pred_answer"],
        "correct_probability": prob,
        "evidence_grade": str(parsed.get("evidence_grade", "")).strip(),
        "issue_type": str(parsed.get("issue_type", "")).strip(),
        "suspected_correct_answer": str(parsed.get("suspected_correct_answer", "")).strip(),
        "risky_options": parsed.get("risky_options", []),
        "why": str(parsed.get("why", "")).strip(),
        "wrong_reason": str(parsed.get("wrong_reason", "")).strip(),
        "suggested_fix": str(parsed.get("suggested_fix", "")).strip(),
        "evidence_count": len(answer_row.get("evidence_items", [])),
        "doc_ids": question.get("doc_ids", []),
        "token_usage": {
            "prompt_tokens": response.token_usage.prompt_tokens,
            "completion_tokens": response.token_usage.completion_tokens,
            "total_tokens": response.token_usage.total_tokens,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "qid",
        "domain",
        "question_type",
        "pred_answer",
        "correct_probability",
        "evidence_grade",
        "issue_type",
        "suspected_correct_answer",
        "risky_options",
        "why",
        "wrong_reason",
        "suggested_fix",
        "evidence_count",
        "doc_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["risky_options"] = ",".join(map(str, row.get("risky_options", [])))
            out["doc_ids"] = "|".join(map(str, row.get("doc_ids", [])))
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def bucket(prob: int) -> str:
    if prob >= 90:
        return "90-100 strong"
    if prob >= 70:
        return "70-89 likely"
    if prob >= 45:
        return "45-69 uncertain"
    if prob >= 20:
        return "20-44 high_risk"
    return "0-19 very_high_risk"


def write_markdown(path: Path, rows: list[dict[str, Any]], answer_rows: list[dict[str, Any]]) -> None:
    from collections import Counter, defaultdict

    by_bucket = Counter(bucket(row["correct_probability"]) for row in rows)
    by_issue = Counter(row["issue_type"] for row in rows)
    by_domain_prob: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_domain_prob[row["domain"]].append(row["correct_probability"])
    avg_by_domain = {
        domain: round(sum(values) / len(values), 1)
        for domain, values in sorted(by_domain_prob.items())
        if values
    }
    total_tokens = sum(row.get("token_usage", {}).get("total_tokens", 0) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Chunk Answer Confidence Audit\n\n")
        f.write("Scope: `group_a_20260628_preprocessed_loop_full`, 100 questions. Sort order is low confidence first.\n\n")
        f.write(f"- audited_questions: {len(rows)}\n")
        f.write(f"- audit_llm_total_tokens: {total_tokens}\n")
        f.write(f"- source_answers: {len(answer_rows)}\n\n")
        f.write("## Probability Buckets\n\n")
        for key, value in by_bucket.most_common():
            f.write(f"- {key}: {value}\n")
        f.write("\n## Average Probability By Domain\n\n")
        for domain, value in avg_by_domain.items():
            f.write(f"- {domain}: {value}\n")
        f.write("\n## Issue Types\n\n")
        for key, value in by_issue.most_common():
            f.write(f"- {key or 'unknown'}: {value}\n")
        f.write("\n## Lowest Confidence Cases\n\n")
        f.write("| qid | domain | pred | prob | issue | suspected | why |\n")
        f.write("|---|---:|---:|---:|---|---:|---|\n")
        for row in rows[:30]:
            why = str(row["why"]).replace("|", "/")
            f.write(
                f"| {row['qid']} | {row['domain']} | {row['pred_answer']} | "
                f"{row['correct_probability']} | {row['issue_type']} | "
                f"{row.get('suspected_correct_answer','')} | {why} |\n"
            )
        f.write("\n## Suggested Fix Themes\n\n")
        for row in rows[:30]:
            f.write(
                f"- `{row['qid']}` ({row['correct_probability']}%, {row['issue_type']}): "
                f"{row['wrong_reason']} 修复：{row['suggested_fix']}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        default=str(ROOT / "artifacts" / "submissions" / "group_a_20260628_preprocessed_loop_full" / "run_mapping.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "answer_audit" / "group_a_20260628_preprocessed_loop_full"),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-evidence-items", type=int, default=12)
    parser.add_argument("--evidence-text-chars", type=int, default=900)
    parser.add_argument("--reasoning-chars", type=int, default=1400)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "confidence_audit.json"

    questions = load_questions()
    answer_rows = load_debug_answers(Path(args.mapping))
    if args.limit:
        answer_rows = answer_rows[: args.limit]

    results = read_json(partial_path) if args.resume and partial_path.exists() else []
    done = {row["qid"] for row in results}
    pending = [row for row in answer_rows if row["qid"] not in done]

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(audit_one, questions[row["qid"]], row, args): row["qid"]
                for row in pending
            }
            for future in as_completed(futures):
                qid = futures[future]
                result = future.result()
                results.append(result)
                results.sort(key=lambda row: (row["correct_probability"], row["qid"]))
                write_json(partial_path, results)
                print(f"audited {qid}: {result['correct_probability']} {result['issue_type']}", flush=True)

    results.sort(key=lambda row: (row["correct_probability"], row["qid"]))
    write_json(partial_path, results)
    write_csv(output_dir / "confidence_audit_low_first.csv", results)
    write_csv(output_dir / "confidence_audit_high_first.csv", list(reversed(results)))
    write_markdown(output_dir / "confidence_audit.md", results, answer_rows)
    summary = {
        "question_count": len(results),
        "source_answer_count": len(answer_rows),
        "low_first_csv": str((output_dir / "confidence_audit_low_first.csv").resolve()),
        "high_first_csv": str((output_dir / "confidence_audit_high_first.csv").resolve()),
        "markdown": str((output_dir / "confidence_audit.md").resolve()),
        "total_audit_tokens": sum(row.get("token_usage", {}).get("total_tokens", 0) for row in results),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
