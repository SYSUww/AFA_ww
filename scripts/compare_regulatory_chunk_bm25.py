#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.regulatory.retriever import RegulatoryRetriever  # noqa: E402
from afa_agent.io_utils import read_json, write_json  # noqa: E402
from afa_agent.text_utils import tokenize_zh  # noqa: E402


TARGETS = {
    "reg_a_006": [
        "strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）::第三十九条"
    ],
    "reg_a_008": [
        "strict_v3_009_中国人民银行_国家金融监督管理总局_中国证券监督管理委员会令〔2025〕第11号（金融机构客户尽职调查和客户身份资料及交易记录保存管理办法）::第十三条"
    ],
    "reg_a_009": [
        "strict_v3_015_中国人民银行令〔2025〕第3号（中国人民银行业务领域数据安全管理办法）::第十六条",
        "strict_v3_015_中国人民银行令〔2025〕第3号（中国人民银行业务领域数据安全管理办法）::第四十二条",
    ],
    "reg_a_004": [
        "strict_v3_016_中国人民银行_国家金融监督管理总局令〔2025〕第2号（银行卡清算机构管理办法）::第二十六条",
        "strict_v3_016_中国人民银行_国家金融监督管理总局令〔2025〕第2号（银行卡清算机构管理办法）::第二十七条",
    ],
    "reg_a_005": [
        "strict_v3_018_中国人民银行令〔2024〕第4号（非银行支付机构监督管理条例实施细则）::第六十二条"
    ],
    "reg_a_010": [
        "strict_v3_018_中国人民银行令〔2024〕第4号（非银行支付机构监督管理条例实施细则）::第六十二条"
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regulatory BM25 retrieval between old and chunk indexes.")
    parser.add_argument("--questions-path", default="test/regulatory_questions.json")
    parser.add_argument("--old-index", default="artifacts/index/regulatory/index.json")
    parser.add_argument("--new-index", default="artifacts/preprocessed/regulatory/chunks_no_manifest/best/index.json")
    parser.add_argument("--output-dir", default="artifacts/preprocessed/regulatory/chunks_no_manifest/bm25_compare")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--include-missing-doc-questions", action="store_true")
    args = parser.parse_args()

    questions = read_json(ROOT / args.questions_path)
    old_index = read_json(ROOT / args.old_index)
    new_index = read_json(ROOT / args.new_index)

    old_retriever = RegulatoryRetriever(old_index["units"])
    new_retriever = RegulatoryRetriever(new_index["units"])
    old_docs = {unit["doc_id"] for unit in old_index["units"]}
    new_docs = {unit["doc_id"] for unit in new_index["units"]}

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for question in questions:
        doc_ids = question.get("doc_ids", [])
        missing_in_new = sorted(set(doc_ids) - new_docs)
        missing_in_old = sorted(set(doc_ids) - old_docs)
        if missing_in_new and not args.include_missing_doc_questions:
            skipped.append(
                {
                    "qid": question["qid"],
                    "missing_in_new": missing_in_new,
                    "missing_in_old": missing_in_old,
                }
            )
            continue
        for option_key, option_text in question["options"].items():
            query = f"{question['question']}\n{option_text}"
            old_hits = old_retriever.search(doc_ids, query, top_k=args.top_k, expand_neighbors=False)
            new_hits = new_retriever.search(doc_ids, query, top_k=args.top_k, expand_neighbors=False)
            old_ids = [hit.unit_id for hit in old_hits]
            new_ids = [hit.unit_id for hit in new_hits]
            targets = TARGETS.get(question["qid"], [])
            rows.append(
                {
                    "qid": question["qid"],
                    "option": option_key,
                    "query_token_count": len(tokenize_zh(query)),
                    "old_top1": old_ids[0] if old_ids else "",
                    "new_top1": new_ids[0] if new_ids else "",
                    "top1_changed": bool(old_ids and new_ids and old_ids[0] != new_ids[0]),
                    "topk_overlap": len(set(old_ids) & set(new_ids)),
                    "old_target_hits": ";".join(target for target in targets if _has_target(old_ids, target)),
                    "new_target_hits": ";".join(target for target in targets if _has_target(new_ids, target)),
                    "old_hits": [hit.to_dict() for hit in old_hits],
                    "new_hits": [hit.to_dict() for hit in new_hits],
                }
            )

    summary = {
        "mode": "official_bm25_jieba",
        "questions_total": len(questions),
        "questions_compared": len({row["qid"] for row in rows}),
        "questions_skipped_missing_new_docs": len(skipped),
        "skipped": skipped,
        "options_compared": len(rows),
        "old_units": len(old_index["units"]),
        "new_units": len(new_index["units"]),
        "old_doc_count": len(old_docs),
        "new_doc_count": len(new_docs),
        "top1_changed_options": sum(1 for row in rows if row["top1_changed"]),
        "top1_changed_rate": round(sum(1 for row in rows if row["top1_changed"]) / max(1, len(rows)), 4),
        "avg_topk_overlap": round(sum(row["topk_overlap"] for row in rows) / max(1, len(rows)), 3),
        "target_rows": sum(1 for row in rows if TARGETS.get(row["qid"])),
        "old_target_hit_rows": sum(1 for row in rows if row["old_target_hits"]),
        "new_target_hit_rows": sum(1 for row in rows if row["new_target_hits"]),
    }

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "details.json", rows)
    _write_csv(output_dir / "summary.csv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_dir)


def _has_target(hit_ids: list[str], target: str) -> bool:
    return any(unit_id == target or unit_id.startswith(f"{target}::") for unit_id in hit_ids)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "qid",
        "option",
        "query_token_count",
        "top1_changed",
        "topk_overlap",
        "old_target_hits",
        "new_target_hits",
        "old_top1",
        "new_top1",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


if __name__ == "__main__":
    main()
