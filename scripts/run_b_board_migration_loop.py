#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.bm25 import BM25Index
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.exporters import export_answer_csv, export_answers_json, export_evidence_json
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl
from afa_agent.models import AnswerResult, Question
from afa_agent.text_utils import tokenize_zh


DOMAINS = ["regulatory", "financial_reports", "insurance", "research", "financial_contracts"]
DEFAULT_PARSED_ROOT = ROOT / "artifacts" / "preprocessed_loop_candidates" / "parsed"
DEFAULT_INDEX_ROOT = ROOT / "artifacts" / "preprocessed_loop_candidates" / "index"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "b_board_migration"
DEFAULT_REFERENCE_ANSWER_CSV = ROOT / "artifacts" / "submissions" / "group_a_candidate_accuracy_first_v20_20260630" / "answer.csv"
DEFAULT_ANSWER_STRATEGY_CONFIG = ROOT / "configs" / "autoresearch" / "evidence_gate_rescue_accuracy_first.json"
PROFILE_INDEX_CACHE: dict[str, dict[str, tuple[list[dict[str, Any]], BM25Index]]] = {}
RETRIEVER_CACHE: dict[str, GenericBM25Retriever] = {}

DOC_ORDER_RE = re.compile(
    r"第一(?:份|个|篇|本|则)?(?:文档|报告|合同|募集说明书|文件)|"
    r"第二(?:份|个|篇|本|则)?(?:文档|报告|合同|募集说明书|文件)|"
    r"两(?:份|个|篇|本)?(?:文档|报告|合同|文件)|"
    r"上述(?:两份|两个|文档|报告|合同|文件)|"
    r"前者|后者|其中一(?:份|个)|另一(?:份|个)"
)
TEXT_DOCID_RE = re.compile(
    r"\b(?:fc_)?text_?0*\d+\b|pack\d+_text\d+|annual_[a-z0-9_]+|csrc_\d+|strict_v\d+_\d+",
    re.IGNORECASE,
)

SYNONYMS: dict[str, list[str]] = {
    "营收": ["营业收入", "营业总收入"],
    "归母净利润": ["归属于上市公司股东的净利润"],
    "经营现金流": ["经营活动产生的现金流量净额"],
    "基本保额": ["基本保险金额"],
    "赔付": ["给付", "赔偿", "报销"],
    "发行规模": ["发行金额", "募集资金总额"],
    "评级": ["主体信用评级", "债项信用评级"],
    "报告": ["报送", "备案", "披露"],
    "市场空间": ["市场规模", "行业规模"],
}

INSURANCE_DOC_ALIASES: dict[str, list[str]] = {
    "1": ["平安智盈金生", "智盈金生"],
    "2": ["国寿增益宝", "中国人寿增益宝", "增益宝"],
    "3": ["众安白血病医疗险", "白血病医疗险", "急性白血病复发医疗保险"],
    "4": ["平安安佑福", "安佑福", "重大疾病保险"],
    "5": ["平安e生保", "平安 e 生保", "e生保", "住院7.0医疗保险"],
    "6": ["太保团体百万医疗", "太平洋团体百万医疗", "团体百万医疗"],
    "7": ["平安预防接种意外险", "预防接种意外伤害保险"],
    "8": ["众安营运交通意外险", "营运交通工具团体意外伤害保险", "营运交通意外"],
    "10": ["众安特种车", "特种车商业保险"],
    "11": ["平安家财险", "平安产险家庭财产保险", "家庭财产保险"],
    "12": ["众安家财险", "家庭财产综合保险"],
    "13": ["众安食责险", "众安食品安全责任险", "食品安全责任保险"],
    "14": ["平安食品安全责任险", "平安产险食品安全责任保险"],
    "15": ["国寿鑫享添盈", "中国人寿鑫享添盈", "鑫享添盈"],
    "16": ["平安富鸿金生", "富鸿金生"],
}

REGULATORY_DOC_ALIASES: dict[str, list[str]] = {
    "csrc_0009_att1": ["上市公司信息披露管理办法", "信息披露管理办法", "定期报告", "年度报告和中报"],
    "csrc_0023_att1": ["上市公司治理准则", "公司治理准则", "治理准则", "董事候选人", "现金分红"],
    "csrc_0027_att1": ["证券公司分类监管规定", "证券公司分类评价规定", "分类监管规定", "分类评价规定"],
    "csrc_0035_att1": ["上市公司章程指引", "章程指引", "股东大会职权", "股东会职权"],
    "csrc_0037_att1": ["半年度报告的内容与格式", "半年度报告", "中期报告"],
    "csrc_0038_att1": ["年度报告的内容与格式", "年度报告", "公开发行证券的公司信息披露内容与格式准则第 2 号"],
    "csrc_0262": ["行政处罚时效", "处罚时效", "世纪华通", "浙江世纪华通"],
    "csrc_0271": ["苏亚金诚", "宏图高科", "审计报告", "签字注册会计师", "审计责任"],
    "strict_v3_009_中国人民银行_国家金融监督管理总局_中国证券监督管理委员会令〔2025〕第11号（金融机构客户尽职调查和客户身份资料及交易记录保存管理办法）": [
        "金融机构客户尽职调查",
        "客户尽职调查",
        "客户身份资料",
        "交易记录保存",
        "空壳银行",
    ],
    "strict_v3_015_中国人民银行令〔2025〕第3号（中国人民银行业务领域数据安全管理办法）": [
        "中国人民银行业务领域数据安全管理办法",
        "业务领域数据安全",
        "重要数据",
        "核心数据",
        "风险评估报告",
    ],
}


@dataclass(frozen=True)
class AttemptConfig:
    attempt_id: str
    round_id: str
    priority: str
    direction: str
    variant_name: str
    hypothesis: str
    top_k: int = 10
    answer_top_k: int = 5
    include_options: bool = True
    include_type: bool = True
    include_synonyms: bool = False
    include_entities: bool = False
    include_numbers: bool = False
    profile_mode: str = "balanced"
    multi_doc_bonus: bool = False
    evidence_top_k: int = 8
    retrieval_unit_type_boosts: dict[str, float] | None = None
    token_strategy: str = "baseline"
    answer_strategy: str = "baseline_prompt"
    answer_doc_policy: str = "alias_pruned"
    query_suffix: str = ""
    domain_query_suffixes: dict[str, str] | None = None
    domain_locator_boosts: dict[str, dict[str, float]] | None = None
    locator_rerank_policy: str = "none"
    first_stage_top_k: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "round_id": self.round_id,
            "priority": self.priority,
            "direction": self.direction,
            "variant_name": self.variant_name,
            "hypothesis": self.hypothesis,
            "top_k": self.top_k,
            "answer_top_k": self.answer_top_k,
            "include_options": self.include_options,
            "include_type": self.include_type,
            "include_synonyms": self.include_synonyms,
            "include_entities": self.include_entities,
            "include_numbers": self.include_numbers,
            "profile_mode": self.profile_mode,
            "multi_doc_bonus": self.multi_doc_bonus,
            "evidence_top_k": self.evidence_top_k,
            "retrieval_unit_type_boosts": self.retrieval_unit_type_boosts or {},
            "token_strategy": self.token_strategy,
            "answer_strategy": self.answer_strategy,
            "answer_doc_policy": self.answer_doc_policy,
            "query_suffix": self.query_suffix,
            "domain_query_suffixes": self.domain_query_suffixes or {},
            "domain_locator_boosts": self.domain_locator_boosts or {},
            "locator_rerank_policy": self.locator_rerank_policy,
            "first_stage_top_k": self.first_stage_top_k,
        }


class CsvWriter:
    @staticmethod
    def write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_cell(row.get(key, "")) for key in fieldnames})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def load_questions() -> dict[str, dict[str, Any]]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    questions: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        path = Path(manifest["domains"][domain]["question_path"])
        for row in read_json(path):
            if row.get("split") == "A":
                questions[row["qid"]] = row
    return questions


def question_text(row: dict[str, Any]) -> str:
    return "\n".join([row.get("question", ""), row.get("type", ""), *list((row.get("options") or {}).values())])


def classify_question(row: dict[str, Any]) -> dict[str, Any]:
    text = question_text(row)
    matched_patterns: list[str] = []
    if DOC_ORDER_RE.search(text):
        matched_patterns.append("doc_order_ref")
    if TEXT_DOCID_RE.search(text):
        matched_patterns.append("text_docid_ref")
    has_order = "doc_order_ref" in matched_patterns
    has_text_docid = "text_docid_ref" in matched_patterns
    if has_order and has_text_docid:
        category = "doc_order_plus_text_leak"
        reason = "题干或选项同时依赖文档顺序/集合关系，并出现显式文档编号。"
    elif has_order:
        category = "doc_order_dependent"
        reason = "题干或选项依赖第一份、第二份、两份文档等 A榜给定文档关系。"
    elif has_text_docid:
        category = "text_docid_leak"
        reason = "题干或选项直接出现文档编号，不能作为严格无 doc_ids blind 样本。"
    else:
        category = "clean_blind_candidate"
        reason = "题目没有明显依赖 A榜文档顺序或显式文档编号，适合模拟 B榜无 doc_ids 场景。"
    return {
        "qid": row["qid"],
        "domain": row["domain"],
        "category": category,
        "doc_ids": list(row.get("doc_ids") or []),
        "doc_count": len(row.get("doc_ids") or []),
        "matched_patterns": matched_patterns,
        "reason": reason,
        "question": row.get("question", ""),
        "options": row.get("options", {}),
    }


def write_mask_outputs(output_dir: Path, questions: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = [classify_question(questions[qid]) for qid in sorted(questions)]
    CsvWriter.write(
        output_dir / "docid_mask_eligibility.csv",
        rows,
        ["qid", "domain", "category", "doc_ids", "doc_count", "matched_patterns", "reason", "question", "options"],
    )
    category_counts = Counter(row["category"] for row in rows)
    domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        domain_counts[row["domain"]][row["category"]] += 1
    clean_qids = [row["qid"] for row in rows if row["category"] == "clean_blind_candidate"]
    special_rows = [row for row in rows if row["category"] != "clean_blind_candidate"]
    CsvWriter.write(
        output_dir / "special_subset_diagnosis.csv",
        [
            {
                **row,
                "suitable_next_experiment": "pair_locator/doc_order_locator" if "doc_order" in row["category"] else "leak_audit_only",
                "main_metric_policy": "exclude_from_strict_b_board_metric",
            }
            for row in special_rows
        ],
        [
            "qid",
            "domain",
            "category",
            "doc_ids",
            "doc_count",
            "matched_patterns",
            "reason",
            "suitable_next_experiment",
            "main_metric_policy",
            "question",
            "options",
        ],
    )
    lines = [
        "# A榜 Doc IDs Mask Eligibility",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- total_questions: `{len(rows)}`",
        f"- clean_blind_candidate: `{category_counts.get('clean_blind_candidate', 0)}`",
        f"- special_subset: `{len(special_rows)}`",
        "",
        "## Category Counts",
        "",
        "| category | count |",
        "|---|---:|",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| `{category}` | {count} |")
    lines.extend(["", "## Domain Counts", "", "| domain | clean | order | leak | order+leak |", "|---|---:|---:|---:|---:|"])
    for domain in DOMAINS:
        counts = domain_counts[domain]
        lines.append(
            f"| `{domain}` | {counts.get('clean_blind_candidate', 0)} | "
            f"{counts.get('doc_order_dependent', 0)} | {counts.get('text_docid_leak', 0)} | "
            f"{counts.get('doc_order_plus_text_leak', 0)} |"
        )
    lines.extend(["", "## Clean Subset QIDs", "", ", ".join(f"`{qid}`" for qid in clean_qids)])
    lines.extend(["", "## Special Subset Policy", ""])
    lines.append(
        "Special subset 不计入严格 B榜迁移主指标，因为这些题的题面语义依赖 A榜提供的文档集合、"
        "文档顺序，或直接泄漏文档编号。后续可以单独设计 pair locator / doc-order locator 诊断。"
    )
    if special_rows:
        lines.extend(["", "## Special Subset QIDs", ""])
        for row in special_rows:
            lines.append(f"- `{row['qid']}` `{row['domain']}` `{row['category']}`: {row['reason']}")
    (output_dir / "docid_mask_eligibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, clean_qids


def load_domain_payloads(parsed_root: Path, index_root: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        parsed = read_json(parsed_root / domain / "parsed.json")
        index = read_json(index_root / domain / "index.json")
        payloads[domain] = {"parsed": parsed, "index": index}
    return payloads


def build_doc_profiles(payloads: dict[str, dict[str, Any]], profile_mode: str) -> dict[str, list[dict[str, Any]]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain, payload in payloads.items():
        docs = {doc["doc_id"]: doc for doc in payload["parsed"].get("documents", [])}
        units_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for unit in payload["index"].get("units", []):
            units_by_doc[unit["doc_id"]].append(unit)
        profiles: list[dict[str, Any]] = []
        for doc_id, doc in docs.items():
            units = units_by_doc.get(doc_id, [])
            title_parts = [doc_id, doc.get("title", ""), doc.get("metadata", {}).get("source_relpath", "")]
            heading_parts: list[str] = []
            first_parts: list[str] = []
            metric_parts: list[str] = []
            for unit in units[:80]:
                heading_parts.extend(str(item) for item in unit.get("title_path", [])[:3])
                text = unit.get("text", "")
                heading_parts.extend(_leading_headings(text)[:3])
                if len(" ".join(first_parts)) < 5000:
                    first_parts.append(text[:800])
                unit_type = str(unit.get("unit_type", ""))
                if unit_type in {"metric_row", "formula_block", "element_block", "article", "conclusion_block", "clause_block"}:
                    metric_parts.append(text[:800])
            repetitions = {
                "lean_title": (5, 1, 1),
                "balanced": (3, 2, 2),
                "content_heavy": (2, 2, 4),
                "structured_heavy": (3, 2, 5),
            }.get(profile_mode, (3, 2, 2))
            title_weight, heading_weight, content_weight = repetitions
            profile_text = "\n".join(
                title_parts * title_weight
                + heading_parts[:80] * heading_weight
                + metric_parts[:40] * content_weight
                + first_parts[:12] * content_weight
            )
            profiles.append(
                {
                    "doc_id": doc_id,
                    "domain": domain,
                    "title": doc.get("title", ""),
                    "source_path": doc.get("source_path", ""),
                    "metadata": doc.get("metadata", {}),
                    "profile_text": profile_text,
                    "unit_count": len(units),
                }
            )
        by_domain[domain] = profiles
    return by_domain


def _leading_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in str(text).splitlines()[:8]:
        cleaned = line.strip().strip("# ").strip()
        if 4 <= len(cleaned) <= 120 and (line.lstrip().startswith("#") or any(term in cleaned for term in ["准则", "办法", "规定", "指引", "报告", "条款"])):
            headings.append(cleaned)
    return headings


def extract_entities(text: str) -> list[str]:
    patterns = [
        r"[\u4e00-\u9fffA-Za-z0-9]{2,}(?:公司|银行|保险|集团|证券|基金|能源|科技|股份|建筑|移动|控股|投资)",
        r"《[^》]{2,80}》",
        r"第[一二三四五六七八九十百零〇两\d]+[章节条]",
        r"〔\d{4}〕第?\d+号",
        r"\d{4}\s*年",
        r"\d+(?:\.\d+)?\s*(?:%|％|亿元|万元|元|美元|倍|日|个月|年)",
    ]
    seen: set[str] = set()
    entities: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            item = match.group(0).strip()
            if item not in seen:
                seen.add(item)
                entities.append(item)
    return entities


def expand_synonyms(text: str) -> list[str]:
    expanded: list[str] = []
    for term, aliases in SYNONYMS.items():
        if term in text or any(alias in text for alias in aliases):
            expanded.extend([term, *aliases])
    return expanded


def _dedupe_strings(items: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = str(item).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def build_query(row: dict[str, Any], attempt: AttemptConfig) -> str:
    parts = [row.get("question", "")]
    if attempt.include_type:
        parts.append(row.get("type", ""))
    if attempt.include_options:
        parts.extend((row.get("options") or {}).values())
    if attempt.query_suffix:
        parts.append(attempt.query_suffix)
    domain_suffix = (attempt.domain_query_suffixes or {}).get(row.get("domain", ""))
    if domain_suffix:
        parts.append(domain_suffix)
    base = "\n".join(part for part in parts if part)
    extras: list[str] = []
    if attempt.include_synonyms:
        extras.extend(expand_synonyms(base))
    if attempt.include_entities:
        extras.extend(extract_entities(base))
    if attempt.include_numbers:
        extras.extend(re.findall(r"\d+(?:\.\d+)?\s*(?:%|％|亿元|万元|元|美元|倍|日|个月|年)?", base))
    return "\n".join([base, " ".join(extras)]).strip()


def locate_docs(
    questions: dict[str, dict[str, Any]],
    qids: list[str],
    payloads: dict[str, dict[str, Any]],
    attempt: AttemptConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in qids:
        row = questions[qid]
        domain = row["domain"]
        effective_attempt = _effective_attempt_for_domain(attempt, domain)
        indexes = get_profile_indexes(payloads, effective_attempt.profile_mode)
        profiles, index = indexes[domain]
        query = build_query(row, effective_attempt)
        query_tokens = tokenize_zh(query)
        first_stage_k = max(effective_attempt.top_k, 20, effective_attempt.first_stage_top_k or 0)
        scored = index.top_k(query_tokens, first_stage_k)
        candidates: list[dict[str, Any]] = []
        for idx, score in scored:
            profile = profiles[idx]
            bonus = 0.0
            if effective_attempt.include_entities or effective_attempt.include_synonyms:
                bonus += _alias_bonus(query, profile)
            if effective_attempt.multi_doc_bonus and len(row.get("doc_ids") or []) > 1:
                bonus += _multi_doc_bonus(query, profile)
            bonus += _domain_locator_bonus(query, profile, domain, effective_attempt)
            candidates.append(
                {
                    "doc_id": profile["doc_id"],
                    "score": round(score + bonus, 6),
                    "base_score": round(score, 6),
                    "bonus": round(bonus, 6),
                    "title": profile.get("title", ""),
                    "locator_reason": _locator_reason(query, profile),
                    "_profile_text": profile.get("profile_text", ""),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        candidates = _apply_locator_rerank(candidates, query, row, effective_attempt)
        selected = [_public_candidate(item) for item in candidates[: effective_attempt.top_k]]
        rows.append(
            {
                "qid": qid,
                "domain": domain,
                "query_terms": query[:2000],
                "candidate_doc_ids": [item["doc_id"] for item in selected],
                "scores": [item["score"] for item in selected],
                "locator_reason": [item["locator_reason"] for item in selected],
                "true_doc_ids_for_eval_only": row.get("doc_ids", []),
                "candidates": selected,
            }
        )
    return rows


def _effective_attempt_for_domain(attempt: AttemptConfig, domain: str) -> AttemptConfig:
    if not attempt.locator_rerank_policy.startswith("domain_mix"):
        return attempt
    overrides: dict[str, Any] = {
        "top_k": attempt.top_k,
        "first_stage_top_k": max(attempt.first_stage_top_k, 60),
        "include_options": True,
    }
    if domain in {"financial_reports", "research"}:
        overrides.update(
            {
                "profile_mode": "balanced",
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": domain == "financial_reports",
                "multi_doc_bonus": True,
            }
        )
    elif domain == "insurance":
        overrides.update(
            {
                "profile_mode": "lean_title",
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "multi_doc_bonus": True,
                "domain_query_suffixes": {
                    **(attempt.domain_query_suffixes or {}),
                    "insurance": "保险产品 条款 产品名称 身故保险金 满期保险金 年金 现金价值 账户价值 基本保险金额 等待期 宽限期 免赔额",
                },
            }
        )
    elif domain == "regulatory":
        overrides.update(
            {
                "profile_mode": "content_heavy",
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "multi_doc_bonus": True,
                "domain_query_suffixes": {
                    **(attempt.domain_query_suffixes or {}),
                    "regulatory": "法条 处罚 决定书 市场禁入 披露 报告 期限 金额 监管措施 违法行为 上市公司治理 信息披露 年度报告 半年度报告 章程指引",
                },
            }
        )
    return replace(attempt, **overrides)


def _apply_locator_rerank(
    candidates: list[dict[str, Any]],
    query: str,
    question: dict[str, Any],
    attempt: AttemptConfig,
) -> list[dict[str, Any]]:
    policy = attempt.locator_rerank_policy or "none"
    if policy == "none":
        return candidates
    reranked = [dict(item) for item in candidates]
    for item in reranked:
        bonus, reasons = _second_stage_locator_bonus(item, query, question, policy)
        if bonus:
            item["rerank_bonus"] = round(bonus, 6)
            item["score"] = round(float(item.get("score", 0.0)) + bonus, 6)
            item["locator_reason"] = f"{item.get('locator_reason', '')}; rerank=" + ",".join(reasons)
    reranked.sort(key=lambda item: item["score"], reverse=True)
    if "anchor_quota" in policy or policy in {"insurance_anchor_quota", "regulatory_anchor_quota", "all_anchor_quota"}:
        reranked = _apply_anchor_quota(reranked, query, question, policy)
    if "dedupe" in policy or "domain_mix" in policy or "canonical" in policy:
        reranked = _dedupe_locator_candidates(reranked, inherit_score="inherit" in policy)
    return reranked


def _second_stage_locator_bonus(item: dict[str, Any], query: str, question: dict[str, Any], policy: str) -> tuple[float, list[str]]:
    domain = question.get("domain", "")
    text = f"{item.get('doc_id', '')} {item.get('title', '')} {item.get('_profile_text', '')[:8000]}"
    bonus = 0.0
    reasons: list[str] = []
    target_ids = set(_anchor_target_doc_ids(query, domain))
    if item.get("doc_id") in target_ids or _locator_canonical_doc_id(str(item.get("doc_id", ""))) in {
        _locator_canonical_doc_id(doc_id) for doc_id in target_ids
    }:
        bonus += 24.0
        reasons.append("anchor_doc")
    matched_aliases = [alias for alias in _doc_aliases({"doc_id": item.get("doc_id", ""), "title": item.get("title", ""), "profile_text": item.get("_profile_text", "")}) if alias in query]
    if matched_aliases and any(key in policy for key in ["alias", "second_stage", "domain_mix", "anchor_quota"]):
        bonus += min(8.0, 2.0 + 1.0 * len(matched_aliases))
        reasons.append("alias")
    if any(key in policy for key in ["numeric", "second_stage", "domain_mix"]):
        overlap = set(_numbers_for_rerank(query)) & set(_numbers_for_rerank(text))
        if overlap:
            bonus += min(3.0, 0.35 * len(overlap))
            reasons.append("number")
    if domain == "regulatory" and any(term in query for term in ["上市公司", "年度报告", "半年度报告", "治理", "章程", "处罚", "市场禁入"]):
        regulatory_overlap = sum(1 for term in ["上市公司", "年度报告", "半年度报告", "治理", "章程", "处罚", "市场禁入", "信息披露"] if term in query and term in text)
        if regulatory_overlap:
            bonus += min(4.0, 0.7 * regulatory_overlap)
            reasons.append("reg_terms")
    if domain == "insurance":
        insurance_overlap = sum(1 for term in ["等待期", "宽限期", "免赔额", "身故", "现金价值", "账户价值", "保险金", "责任免除"] if term in query and term in text)
        if insurance_overlap:
            bonus += min(4.0, 0.6 * insurance_overlap)
            reasons.append("insurance_terms")
    doc_id = str(item.get("doc_id", ""))
    if "__dup" in doc_id or doc_id.endswith("_extracted"):
        bonus -= 0.25
        reasons.append("dup_penalty")
    return bonus, reasons


def _apply_anchor_quota(
    candidates: list[dict[str, Any]],
    query: str,
    question: dict[str, Any],
    policy: str,
) -> list[dict[str, Any]]:
    domain = question.get("domain", "")
    if policy == "insurance_anchor_quota" and domain != "insurance":
        return candidates
    if policy == "regulatory_anchor_quota" and domain != "regulatory":
        return candidates
    target_ids = _anchor_target_doc_ids(query, domain)
    if not target_ids:
        return candidates
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    by_doc = {item["doc_id"]: item for item in candidates}
    by_canonical: dict[str, dict[str, Any]] = {}
    for item in candidates:
        by_canonical.setdefault(_locator_canonical_doc_id(item["doc_id"]), item)
    for doc_id in target_ids:
        item = by_doc.get(doc_id) or by_canonical.get(_locator_canonical_doc_id(doc_id))
        if item and item["doc_id"] not in used:
            promoted = dict(item)
            promoted["score"] = round(float(promoted.get("score", 0.0)) + 30.0, 6)
            promoted["locator_reason"] = f"{promoted.get('locator_reason', '')}; quota_anchor={doc_id}"
            selected.append(promoted)
            used.add(promoted["doc_id"])
    for item in candidates:
        if item["doc_id"] not in used:
            selected.append(item)
    selected.sort(key=lambda item: item["score"], reverse=True)
    return selected


def _anchor_target_doc_ids(query: str, domain: str) -> list[str]:
    alias_map: dict[str, list[str]]
    if domain == "insurance":
        alias_map = INSURANCE_DOC_ALIASES
    elif domain == "regulatory":
        alias_map = REGULATORY_DOC_ALIASES
    else:
        alias_map = {}
    targets: list[str] = []
    for doc_id, aliases in alias_map.items():
        if any(alias and alias in query for alias in aliases):
            targets.append(doc_id)
    return _dedupe_strings(targets)


def _dedupe_locator_candidates(candidates: list[dict[str, Any]], *, inherit_score: bool = False) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[_locator_canonical_doc_id(item["doc_id"])].append(item)
    selected: list[dict[str, Any]] = []
    for canonical, items in grouped.items():
        best_score = max(float(item.get("score", 0.0)) for item in items)
        preferred = None
        for item in items:
            if item["doc_id"] == canonical:
                preferred = item
                break
        if preferred is None:
            preferred = max(items, key=lambda item: float(item.get("score", 0.0)))
        chosen = dict(preferred)
        if inherit_score:
            chosen["score"] = round(best_score, 6)
            chosen["locator_reason"] = f"{chosen.get('locator_reason', '')}; canonical_score_inherit"
        selected.append(chosen)
    selected.sort(key=lambda item: item["score"], reverse=True)
    return selected


def _locator_canonical_doc_id(doc_id: str) -> str:
    cleaned = re.sub(r"__dup\d*$", "", str(doc_id))
    if re.fullmatch(r"csrc_\d{4}_extracted", cleaned):
        cleaned = cleaned.replace("_extracted", "")
    return cleaned


def _numbers_for_rerank(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?\s*(?:%|％|亿元|万元|元|美元|倍|日|个月|年)?", text)


def _public_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def get_profile_indexes(payloads: dict[str, dict[str, Any]], profile_mode: str) -> dict[str, tuple[list[dict[str, Any]], BM25Index]]:
    cached = PROFILE_INDEX_CACHE.get(profile_mode)
    if cached is not None:
        return cached
    profiles_by_domain = build_doc_profiles(payloads, profile_mode)
    indexes: dict[str, tuple[list[dict[str, Any]], BM25Index]] = {}
    for domain, profiles in profiles_by_domain.items():
        indexes[domain] = (profiles, BM25Index([tokenize_zh(profile["profile_text"]) for profile in profiles]))
    PROFILE_INDEX_CACHE[profile_mode] = indexes
    return indexes


def _multi_doc_bonus(query: str, profile: dict[str, Any]) -> float:
    title = f"{profile.get('doc_id', '')} {profile.get('title', '')} {profile.get('profile_text', '')[:2000]}"
    query_entities = set(extract_entities(query))
    if not query_entities:
        return 0.0
    overlap = sum(1 for item in query_entities if item and item in title)
    return min(1.5, overlap * 0.25)


def _domain_locator_bonus(query: str, profile: dict[str, Any], domain: str, attempt: AttemptConfig) -> float:
    boosts = (attempt.domain_locator_boosts or {}).get(domain, {})
    if not boosts:
        return 0.0
    text = f"{profile.get('doc_id', '')} {profile.get('title', '')} {profile.get('profile_text', '')[:6000]}"
    bonus = 0.0
    for term, value in boosts.items():
        if term.startswith("doc_id:"):
            needle = term.split(":", 1)[1]
            if needle and needle in str(profile.get("doc_id", "")):
                bonus += float(value)
            continue
        if term.startswith("always:"):
            needle = term.split(":", 1)[1]
            if needle and needle in text:
                bonus += float(value)
            continue
        if term in text and term in query:
            bonus += float(value)
    return min(4.0, bonus)


def _alias_bonus(query: str, profile: dict[str, Any]) -> float:
    aliases = _doc_aliases(profile)
    if not aliases:
        return 0.0
    matched = [alias for alias in aliases if alias and alias in query]
    if not matched:
        return 0.0
    if any(alias in _hard_doc_aliases(str(profile.get("doc_id", ""))) for alias in matched):
        return min(10.0, 7.0 + 0.5 * len(matched))
    return min(3.0, 1.2 + 0.3 * len(matched))


def _doc_aliases(profile: dict[str, Any]) -> list[str]:
    doc_id = str(profile.get("doc_id", ""))
    title = str(profile.get("title", ""))
    text = str(profile.get("profile_text", ""))[:2000]
    aliases: list[str] = []
    aliases.extend(_hard_doc_aliases(doc_id))
    for pattern in [
        r"[\u4e00-\u9fffA-Za-z0-9]{2,}(?:股份有限公司|有限公司|集团有限公司|银行|保险|证券|集团|控股)",
        r"《([^》]{2,80})》",
        r"[\u4e00-\u9fffA-Za-z0-9]{2,}(?:保险|条款|办法|报告|募集说明书|准则|规定|指引|规则|决定|责任险|医疗险)",
    ]:
        for match in re.finditer(pattern, f"{title}\n{text}"):
            item = match.group(1) if match.groups() else match.group(0)
            cleaned = item.strip("# ：:，,。 ")
            if 2 <= len(cleaned) <= 80 and not _is_generic_alias(cleaned):
                aliases.append(cleaned)
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            deduped.append(alias)
    return deduped[:30]


def _hard_doc_aliases(doc_id: str) -> list[str]:
    hardcoded = {
        "byd": ["比亚迪", "比亚迪股份有限公司"],
        "catl": ["宁德时代", "宁德时代新能源科技"],
        "midea": ["美的集团", "美的"],
        "cscec": ["中国建筑", "中国建筑股份有限公司"],
        "cmb": ["招商银行", "招行"],
        "chinamobile": ["中国移动", "中国移动有限公司"],
    }
    aliases: list[str] = []
    lowered = doc_id.lower()
    for key, values in hardcoded.items():
        if key in lowered:
            aliases.extend(values)
    aliases.extend(INSURANCE_DOC_ALIASES.get(doc_id, []))
    aliases.extend(REGULATORY_DOC_ALIASES.get(doc_id, []))
    return aliases


def _is_generic_alias(alias: str) -> bool:
    generic_exact = {
        "年度报告",
        "年年度报告",
        "报告",
        "行业深度报告",
        "募集说明书",
        "债券募集说明书",
        "保险条款",
    }
    if alias in generic_exact:
        return True
    if re.fullmatch(r"\d{4}\s*年?年度报告", alias):
        return True
    return False


def _locator_reason(query: str, profile: dict[str, Any]) -> str:
    text = f"{profile.get('doc_id', '')} {profile.get('title', '')} {profile.get('profile_text', '')[:3000]}"
    matched_entities = [item for item in extract_entities(query) if item in text]
    matched_synonyms = [item for item in expand_synonyms(query) if item in text]
    reasons = []
    if matched_entities:
        reasons.append("entity=" + ",".join(matched_entities[:5]))
    if matched_synonyms:
        reasons.append("synonym=" + ",".join(matched_synonyms[:5]))
    matched_aliases = [alias for alias in _doc_aliases(profile) if alias in query]
    if matched_aliases:
        reasons.append("alias=" + ",".join(matched_aliases[:5]))
    if not reasons:
        reasons.append("bm25_profile_match")
    return "; ".join(reasons)


def evaluate_locator(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        true_docs = list(dict.fromkeys(str(item) for item in row["true_doc_ids_for_eval_only"]))
        candidates = row["candidate_doc_ids"]
        eval_row: dict[str, Any] = {
            "qid": row["qid"],
            "domain": row["domain"],
            "true_doc_ids": true_docs,
            "candidate_doc_ids": candidates,
        }
        for k in [1, 3, 5, 10]:
            top = set(candidates[:k])
            covered = [doc_id for doc_id in true_docs if doc_id in top]
            eval_row[f"doc_recall@{k}"] = 1.0 if true_docs and len(covered) == len(true_docs) else 0.0
            eval_row[f"doc_any_recall@{k}"] = 1.0 if true_docs and covered else 0.0
            eval_row[f"doc_coverage@{k}"] = round(len(covered) / len(true_docs), 4) if true_docs else 0.0
        eval_row["failure_reason"] = _doc_failure_reason(eval_row, row)
        eval_rows.append(eval_row)
    metrics = aggregate_locator_metrics(eval_rows)
    return eval_rows, metrics


def _doc_failure_reason(eval_row: dict[str, Any], candidate_row: dict[str, Any]) -> str:
    if eval_row["doc_recall@5"] >= 1.0:
        return "ok"
    if eval_row["doc_any_recall@10"] <= 0:
        query = candidate_row.get("query_terms", "")
        if len(extract_entities(query)) <= 1:
            return "entity_missing"
        if len(tokenize_zh(query)) <= 8:
            return "generic_question"
        return "profile_too_weak"
    if eval_row["doc_coverage@10"] < 1.0:
        return "multi_doc_pair_missing"
    return "rank_too_low"


def aggregate_locator_metrics(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"question_count": len(eval_rows)}
    for k in [1, 3, 5, 10]:
        metrics[f"doc_recall@{k}"] = _avg(row[f"doc_recall@{k}"] for row in eval_rows)
        metrics[f"doc_any_recall@{k}"] = _avg(row[f"doc_any_recall@{k}"] for row in eval_rows)
        metrics[f"doc_coverage@{k}"] = _avg(row[f"doc_coverage@{k}"] for row in eval_rows)
    metrics["failure_counts"] = dict(Counter(row["failure_reason"] for row in eval_rows))
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        domain_rows = [row for row in eval_rows if row["domain"] == domain]
        if domain_rows:
            by_domain[domain] = {
                "question_count": len(domain_rows),
                "doc_recall@5": _avg(row["doc_recall@5"] for row in domain_rows),
                "doc_coverage@5": _avg(row["doc_coverage@5"] for row in domain_rows),
                "failure_counts": dict(Counter(row["failure_reason"] for row in domain_rows)),
            }
    metrics["by_domain"] = by_domain
    return metrics


def _avg(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(float(item) for item in items) / len(items), 6)


def evaluate_evidence_retrieval(
    questions: dict[str, dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    attempt: AttemptConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrievers = get_retrievers(payloads)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        qid = candidate["qid"]
        question = questions[qid]
        domain = question["domain"]
        effective_attempt = _effective_attempt_for_domain(attempt, domain)
        doc_ids = select_answer_doc_ids(candidate, question, effective_attempt)
        query = build_query(question, effective_attempt)
        hits = retrievers[domain].search(
            doc_ids,
            query,
            top_k=effective_attempt.evidence_top_k,
            unit_type_boosts=effective_attempt.retrieval_unit_type_boosts or None,
            ensure_per_doc=True,
            expand_neighbors=True,
        )
        hit_docs = list(dict.fromkeys(hit.doc_id for hit in hits))
        true_docs = list(question.get("doc_ids") or [])
        covered_docs = [doc_id for doc_id in true_docs if doc_id in hit_docs]
        rows.append(
            {
                "qid": qid,
                "domain": domain,
                "candidate_doc_ids": doc_ids,
                "true_doc_ids": true_docs,
                "hit_doc_ids": hit_docs,
                "evidence_count": len(hits),
                "evidence_doc_coverage": round(len(covered_docs) / len(true_docs), 4) if true_docs else 0.0,
                "evidence_doc_full_recall": 1.0 if true_docs and len(covered_docs) == len(true_docs) else 0.0,
                "top_hit_unit_ids": [hit.unit_id for hit in hits[:5]],
                "estimated_prompt_tokens": estimate_prompt_tokens(question, hits, effective_attempt),
                "failure_reason": "ok" if true_docs and len(covered_docs) == len(true_docs) else "evidence_retrieval_fail",
            }
        )
    metrics = {
        "evidence_doc_full_recall": _avg(row["evidence_doc_full_recall"] for row in rows),
        "evidence_doc_coverage": _avg(row["evidence_doc_coverage"] for row in rows),
        "estimated_prompt_tokens": sum(int(row["estimated_prompt_tokens"]) for row in rows),
        "failure_counts": dict(Counter(row["failure_reason"] for row in rows)),
    }
    return rows, metrics


def get_retrievers(payloads: dict[str, dict[str, Any]]) -> dict[str, GenericBM25Retriever]:
    if RETRIEVER_CACHE:
        return RETRIEVER_CACHE
    for domain, payload in payloads.items():
        RETRIEVER_CACHE[domain] = GenericBM25Retriever(payload["index"].get("units", []))
    return RETRIEVER_CACHE


def estimate_prompt_tokens(question: dict[str, Any], hits: list[Any], attempt: AttemptConfig) -> int:
    evidence_chars = sum(len(hit.text) for hit in hits[: attempt.evidence_top_k])
    question_chars = len(question.get("question", "")) + sum(len(item) for item in (question.get("options") or {}).values())
    compression = {
        "baseline": 1.0,
        "dynamic_topk": 0.8,
        "evidence_pool_dedupe": 0.7,
        "structured_evidence_card": 0.55,
        "low_confidence_only_rescue": 0.75,
    }.get(attempt.token_strategy, 1.0)
    return int((evidence_chars + question_chars + 800) / 1.8 * compression)


def select_answer_doc_ids(candidate_row: dict[str, Any], question: dict[str, Any], attempt: AttemptConfig) -> list[str]:
    candidates = candidate_row.get("candidates", [])
    alias_matched = [
        item["doc_id"]
        for item in candidates
        if "alias=" in str(item.get("locator_reason", ""))
    ]
    candidate_doc_ids = candidate_row.get("candidate_doc_ids", [])
    policy = attempt.answer_doc_policy
    domain = question.get("domain", "")

    if policy == "expanded_topk":
        return _unique_doc_ids(candidate_doc_ids, attempt.answer_top_k)
    if policy == "multi_doc_fill":
        return _unique_doc_ids([*alias_matched, *candidate_doc_ids], attempt.answer_top_k)
    if policy == "canonical_dedupe_fill":
        return _unique_doc_ids([*alias_matched, *candidate_doc_ids], attempt.answer_top_k, dedupe_canonical=True)
    if policy == "regulatory_per_doc_fill" and domain == "regulatory":
        return _unique_doc_ids([*alias_matched, *candidate_doc_ids], attempt.answer_top_k, dedupe_canonical=True)
    if policy == "alias_pruned":
        # Product/report aliases are strong document identifiers for these two
        # domains. In regulatory and research they are often generic phrases
        # such as "定期报告" or "深度报告"; pruning to those aliases can discard
        # the locator's higher-ranked documents even when the locator is right.
        if domain in {"financial_reports", "insurance"} and alias_matched:
            return _unique_doc_ids(alias_matched, attempt.answer_top_k)
        return _unique_doc_ids(candidate_doc_ids, attempt.answer_top_k)

    if alias_matched:
        if domain in {"financial_reports", "insurance"}:
            return _unique_doc_ids(alias_matched, attempt.answer_top_k)
        if len(alias_matched) >= 2:
            return _unique_doc_ids(alias_matched, attempt.answer_top_k)
    return _unique_doc_ids([*alias_matched, *candidate_doc_ids], attempt.answer_top_k)


def _unique_doc_ids(doc_ids: Iterable[str], limit: int, *, dedupe_canonical: bool = False) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        if not doc_id:
            continue
        key = _canonical_doc_id(str(doc_id)) if dedupe_canonical else str(doc_id)
        if key in seen:
            continue
        seen.add(key)
        selected.append(str(doc_id))
        if len(selected) >= limit:
            break
    return selected


def _canonical_doc_id(doc_id: str) -> str:
    return re.sub(r"__dup\d*$", "", doc_id)


def default_attempts() -> list[AttemptConfig]:
    specs = [
        ("round_00", "P0", "doc_locator", "profile_bm25_balanced", "文档 profile BM25 可建立 no-docids 初始召回。", {"profile_mode": "balanced", "include_options": False}),
        ("round_00", "P0", "doc_locator", "title_filename_weighted", "提高标题、文件名、doc_id 权重可改善实体明确题。", {"profile_mode": "lean_title"}),
        ("round_00", "P0", "doc_locator", "question_options_joint", "题目和选项联合 query 可补充候选实体。", {"include_options": True, "include_type": True}),
        ("round_00", "P0", "doc_locator", "entity_year_metric", "抽取实体、年份、指标、金额可改善跨 domain 定位。", {"include_entities": True, "include_numbers": True}),
        ("round_00", "P0", "doc_locator", "synonym_expansion", "领域同义词扩展可降低题目词与文档词不一致。", {"include_synonyms": True, "include_entities": True}),
        ("round_01", "P0", "dynamic_topk_rescue", "topk5_low_cost", "较小候选集可降低 token，观察 doc recall 损失。", {"top_k": 5, "answer_top_k": 3, "token_strategy": "dynamic_topk"}),
        ("round_01", "P0", "dynamic_topk_rescue", "topk10_default", "默认 top10 作为召回和成本折中点。", {"top_k": 10, "answer_top_k": 5, "token_strategy": "baseline"}),
        ("round_01", "P0", "dynamic_topk_rescue", "topk15_rescue", "扩大候选集检验 locator 排名不足是否是主因。", {"top_k": 15, "answer_top_k": 8, "token_strategy": "low_confidence_only_rescue"}),
        ("round_01", "P0", "dynamic_topk_rescue", "structured_card_cost", "结构化 evidence card 预计能压缩 prompt token。", {"top_k": 10, "answer_top_k": 5, "token_strategy": "structured_evidence_card"}),
        ("round_01", "P0", "dynamic_topk_rescue", "evidence_pool_cost", "整题 evidence pool 去重预计降低重复上下文。", {"top_k": 10, "answer_top_k": 5, "token_strategy": "evidence_pool_dedupe"}),
        ("round_02", "P0", "regulatory_clause_gate", "article_boost", "法规条款 unit_type boost 可改善条款证据覆盖。", {"include_entities": True, "retrieval_unit_type_boosts": {"article": 1.8, "preamble": 1.2}}),
        ("round_02", "P0", "regulatory_clause_gate", "penalty_entity_boost", "处罚决定题按当事人和处罚类型检索可提升证据命中。", {"include_entities": True, "include_synonyms": True}),
        ("round_02", "P0", "regulatory_clause_gate", "neighbor_clause_expansion", "条款邻居扩展可补齐上下文。", {"evidence_top_k": 12, "include_entities": True}),
        ("round_02", "P0", "financial_contracts_structured", "contract_element_boost", "合同 element block boost 可改善发行要素召回。", {"retrieval_unit_type_boosts": {"element_block": 1.8, "paragraph": 1.0}, "include_entities": True}),
        ("round_02", "P0", "financial_contracts_structured", "contract_multi_doc", "多文档覆盖 locator 观察 pair 缺失问题。", {"multi_doc_bonus": True, "top_k": 15, "answer_top_k": 8, "include_entities": True}),
        ("round_02", "P0", "financial_contracts_structured", "contract_numbers_fields", "发行规模、评级、价格等数字字段进入 query。", {"include_numbers": True, "include_entities": True, "profile_mode": "structured_heavy"}),
        ("round_03", "P1", "insurance_calculator", "formula_clause_boost", "保险公式和条款 boost 可提升计算题证据。", {"retrieval_unit_type_boosts": {"formula_block": 2.0, "clause_block": 1.6}, "include_entities": True}),
        ("round_03", "P1", "insurance_calculator", "product_option_locator", "产品名和选项合并检索可改善产品级覆盖。", {"include_options": True, "include_entities": True, "profile_mode": "lean_title"}),
        ("round_03", "P1", "insurance_calculator", "structured_card_formula", "公式证据卡可压缩 token。", {"token_strategy": "structured_evidence_card", "retrieval_unit_type_boosts": {"formula_block": 2.0, "clause_block": 1.5}}),
        ("round_03", "P1", "financial_reports_metric_cell", "metric_row_boost", "财报 metric row boost 可改善表格题证据。", {"retrieval_unit_type_boosts": {"metric_row": 2.0, "paragraph": 1.0}, "include_numbers": True}),
        ("round_03", "P1", "financial_reports_metric_cell", "company_year_metric_query", "公司、年份、指标抽取可改善财报定位。", {"include_entities": True, "include_numbers": True, "include_synonyms": True}),
        ("round_03", "P1", "financial_reports_metric_cell", "metric_structured_card", "表格证据卡可降低 token。", {"token_strategy": "structured_evidence_card", "retrieval_unit_type_boosts": {"metric_row": 2.0}}),
        ("round_04", "P1", "evidence_pool_single_call", "evidence_pool_default", "整题 evidence pool 可减少选项重复证据。", {"token_strategy": "evidence_pool_dedupe"}),
        ("round_04", "P1", "evidence_pool_single_call", "supported_only_guard", "多选题 supported-only guard 可降低答案层矛盾。", {"answer_strategy": "supported_only_multiselect"}),
        ("round_04", "P1", "evidence_pool_single_call", "final_consistency_retry", "最终一致性复核可拦截 evidence/answer 矛盾。", {"answer_strategy": "final_consistency_guard"}),
        ("round_04", "P2", "research_data_point", "research_conclusion_boost", "研报 conclusion block boost 可改善趋势/结论题。", {"retrieval_unit_type_boosts": {"conclusion_block": 1.8}, "include_numbers": True}),
        ("round_04", "P2", "research_data_point", "research_entity_time_metric", "时间、地区、指标抽取可改善研报定位。", {"include_entities": True, "include_numbers": True}),
        ("round_04", "P2", "research_data_point", "research_structured_card", "data point 证据卡可压缩研报上下文。", {"token_strategy": "structured_evidence_card"}),
        (
            "round_05",
            "P0",
            "regulatory_round2_targeted",
            "reg_docno_penalty_alias",
            "监管处罚题按文号、当事人、处罚决定、市场禁入扩展 query，并保留更多候选文档做 evidence 判断。",
            {
                "top_k": 15,
                "answer_top_k": 8,
                "evidence_top_k": 12,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "regulatory_per_doc_fill",
                "domain_query_suffixes": {
                    "regulatory": "证监会 行政处罚 处罚决定书 市场禁入 当事人 文号 罚款 没收 违法所得 信息披露 报告 备案"
                },
                "domain_locator_boosts": {
                    "regulatory": {
                        "doc_id:csrc_": 0.45,
                        "行政处罚": 0.35,
                        "处罚决定": 0.35,
                        "市场禁入": 0.5,
                        "当事人": 0.2,
                    }
                },
            },
        ),
        (
            "round_05",
            "P0",
            "regulatory_round2_targeted",
            "reg_article_strict_law",
            "法规条款题按法条号、义务动作、期限、报告对象扩展 query，优先补齐 strict_v3 类法规文档。",
            {
                "top_k": 15,
                "answer_top_k": 8,
                "evidence_top_k": 12,
                "include_entities": True,
                "include_numbers": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "regulatory_per_doc_fill",
                "retrieval_unit_type_boosts": {"article": 2.0, "paragraph": 1.0, "preamble": 1.2},
                "domain_query_suffixes": {
                    "regulatory": "第 条 款 办法 规定 义务 应当 不得 期限 日内 报送 报告 披露 备案 金额 对象"
                },
                "domain_locator_boosts": {
                    "regulatory": {
                        "doc_id:strict_v3_": 0.45,
                        "办法": 0.2,
                        "规定": 0.2,
                        "报告": 0.15,
                        "披露": 0.15,
                    }
                },
            },
        ),
        (
            "round_05",
            "P0",
            "regulatory_round2_targeted",
            "reg_high_recall_neighbor",
            "监管低确定性 case 暂时用更高 topK 和邻居扩展换取证据覆盖，观察 token 增量是否值得。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "evidence_top_k": 14,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "content_heavy",
                "answer_doc_policy": "canonical_dedupe_fill",
                "retrieval_unit_type_boosts": {"article": 1.7, "paragraph": 1.0, "penalty_decision": 1.5},
                "domain_query_suffixes": {
                    "regulatory": "法条 处罚 决定书 市场禁入 披露 报告 期限 金额 监管措施 违法行为"
                },
            },
        ),
        (
            "round_06",
            "P0",
            "financial_reports_answer_layer",
            "fin_metric_supported_card",
            "财报题优先 metric_row，压缩成结构化证据卡，减少答案层在多选格式上的漂移。",
            {
                "top_k": 10,
                "answer_top_k": 5,
                "evidence_top_k": 10,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "multi_doc_fill",
                "retrieval_unit_type_boosts": {"metric_row": 2.4, "paragraph": 1.0},
                "token_strategy": "structured_evidence_card",
                "answer_strategy": "supported_only_multiselect",
                "domain_query_suffixes": {
                    "financial_reports": "营业收入 净利润 归母净利润 经营现金流 同比 增长 下降 2024 2025 合并报表 单位"
                },
            },
        ),
        (
            "round_06",
            "P0",
            "financial_reports_answer_layer",
            "fin_metric_high_recall",
            "财报数值题扩大候选和 evidence topK，要求同一指标两年数值尽量同时进入证据。",
            {
                "top_k": 15,
                "answer_top_k": 8,
                "evidence_top_k": 12,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "multi_doc_fill",
                "retrieval_unit_type_boosts": {"metric_row": 2.5, "table_row": 1.8, "paragraph": 1.0},
                "domain_query_suffixes": {
                    "financial_reports": "指标 年份 金额 百分比 单位 同比 增减 营业收入 现金流 资产负债率"
                },
            },
        ),
        (
            "round_06",
            "P0",
            "financial_reports_answer_layer",
            "fin_compact_format_guard",
            "财报高 token case 用较小答案文档集合和 supported-only guard 优先降低不必要证据重复。",
            {
                "top_k": 8,
                "answer_top_k": 4,
                "evidence_top_k": 8,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "canonical_dedupe_fill",
                "retrieval_unit_type_boosts": {"metric_row": 2.4, "paragraph": 1.0},
                "token_strategy": "evidence_pool_dedupe",
                "answer_strategy": "final_consistency_guard",
            },
        ),
        (
            "round_07",
            "P1",
            "insurance_multi_product_calc",
            "ins_product_alias_fill",
            "保险多产品题按产品标题和选项实体补文档，避免只命中一个产品就推理。",
            {
                "top_k": 15,
                "answer_top_k": 8,
                "evidence_top_k": 10,
                "include_options": True,
                "include_entities": True,
                "include_synonyms": True,
                "profile_mode": "lean_title",
                "answer_doc_policy": "multi_doc_fill",
                "domain_query_suffixes": {
                    "insurance": "保险产品 条款 产品名称 身故保险金 满期保险金 年金 现金价值 账户价值 基本保险金额"
                },
                "domain_locator_boosts": {
                    "insurance": {
                        "保险条款": 0.35,
                        "保险金": 0.25,
                        "现金价值": 0.2,
                        "账户价值": 0.2,
                    }
                },
            },
        ),
        (
            "round_07",
            "P1",
            "insurance_multi_product_calc",
            "ins_formula_clause_expanded",
            "保险计算题按公式块、条款标题和给付关键词定向召回，减少模型自由计算。",
            {
                "top_k": 15,
                "answer_top_k": 8,
                "evidence_top_k": 12,
                "include_options": True,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "multi_doc_fill",
                "retrieval_unit_type_boosts": {"formula_block": 2.6, "clause_block": 1.8, "paragraph": 1.0},
                "domain_query_suffixes": {
                    "insurance": "计算公式 给付比例 基本保险金额 现金价值 账户价值 已交保险费 身故 满期 重大疾病"
                },
            },
        ),
        (
            "round_07",
            "P1",
            "insurance_multi_product_calc",
            "ins_formula_compact_guard",
            "保险题保留公式/条款 boost，但压缩 evidence card 并使用 canonical dedupe 控制 token。",
            {
                "top_k": 10,
                "answer_top_k": 5,
                "evidence_top_k": 8,
                "include_options": True,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "canonical_dedupe_fill",
                "retrieval_unit_type_boosts": {"formula_block": 2.5, "clause_block": 1.7, "paragraph": 1.0},
                "token_strategy": "structured_evidence_card",
                "answer_strategy": "supported_only_multiselect",
            },
        ),
        (
            "round_08",
            "P1",
            "token_control_high_cost",
            "token_topk8_answer3_card",
            "高 token case 先验证小文档集合和结构化 evidence card 的保守压缩效果。",
            {
                "top_k": 8,
                "answer_top_k": 3,
                "evidence_top_k": 7,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "canonical_dedupe_fill",
                "token_strategy": "structured_evidence_card",
                "answer_strategy": "supported_only_multiselect",
            },
        ),
        (
            "round_08",
            "P1",
            "token_control_high_cost",
            "token_topk10_answer4_pool",
            "整题 evidence pool 去重配合 answer_top_k=4，目标是少丢证据同时降低重复上下文。",
            {
                "top_k": 10,
                "answer_top_k": 4,
                "evidence_top_k": 8,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "multi_doc_fill",
                "token_strategy": "evidence_pool_dedupe",
            },
        ),
        (
            "round_08",
            "P1",
            "token_control_high_cost",
            "token_low_confidence_rescue",
            "默认保留较高召回，但通过 low-confidence-only rescue 估计将二次检索限制在低确定性 case。",
            {
                "top_k": 12,
                "answer_top_k": 5,
                "evidence_top_k": 8,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "structured_heavy",
                "answer_doc_policy": "multi_doc_fill",
                "token_strategy": "low_confidence_only_rescue",
                "answer_strategy": "final_consistency_guard",
            },
        ),
        (
            "round_09",
            "P0",
            "locator_canonical_dedupe",
            "canonical_dedupe_rerank",
            "按 canonical doc 去重，避免 __dup/extracted 候选占据 top5 名额。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 60,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "content_heavy",
                "locator_rerank_policy": "canonical_dedupe",
            },
        ),
        (
            "round_09",
            "P0",
            "locator_canonical_dedupe",
            "csrc_base_dedupe",
            "将 csrc_xxxx、csrc_xxxx_extracted、__dup 视作同一 base，保留可评估的 base 文档。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 60,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "content_heavy",
                "locator_rerank_policy": "csrc_base_dedupe",
            },
        ),
        (
            "round_09",
            "P0",
            "locator_canonical_dedupe",
            "canonical_score_inherit",
            "canonical 组内保留优先 doc_id，同时继承组内最高分，减少去重造成的排序损失。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 60,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "content_heavy",
                "locator_rerank_policy": "canonical_score_inherit",
            },
        ),
        (
            "round_10",
            "P0",
            "locator_domain_strategy_mix",
            "domain_best_static",
            "按 domain 复用当前离线最优定位形态：财报/研报偏多文档，保险偏标题产品，监管偏内容。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 60,
                "locator_rerank_policy": "domain_mix",
            },
        ),
        (
            "round_10",
            "P0",
            "locator_domain_strategy_mix",
            "domain_best_dedupe",
            "domain-best 基础上叠加 canonical 去重，减少法规 duplicate 对 top5 的挤占。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 60,
                "locator_rerank_policy": "domain_mix_canonical_dedupe",
            },
        ),
        (
            "round_10",
            "P0",
            "locator_domain_strategy_mix",
            "domain_best_top60_rerank",
            "domain-best 使用 top60 候选池并启用二阶段 alias/数字/领域词重排。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 60,
                "locator_rerank_policy": "domain_mix_second_stage_rerank",
            },
        ),
        (
            "round_11",
            "P0",
            "locator_anchor_quota",
            "insurance_product_quota",
            "保险题按产品名 anchor 保底候选，解决多产品题漏一份文档。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 60,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "lean_title",
                "locator_rerank_policy": "insurance_anchor_quota",
            },
        ),
        (
            "round_11",
            "P0",
            "locator_anchor_quota",
            "regulatory_doc_anchor_quota",
            "监管题按法规名、处罚主体、报告类型 anchor 保底候选，解决法规/处罚组合题缺文档。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 80,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "content_heavy",
                "locator_rerank_policy": "regulatory_anchor_quota",
            },
        ),
        (
            "round_11",
            "P0",
            "locator_anchor_quota",
            "all_domain_anchor_quota",
            "全域启用 anchor quota，观察是否提升多文档题完整覆盖且不伤财报。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 80,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "content_heavy",
                "locator_rerank_policy": "all_anchor_quota",
            },
        ),
        (
            "round_12",
            "P0",
            "locator_second_stage_rerank",
            "exact_alias_rerank",
            "top60 后按标题/别名精确命中重排，优先解决 rank_too_low。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 60,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "content_heavy",
                "locator_rerank_policy": "second_stage_alias_rerank",
            },
        ),
        (
            "round_12",
            "P0",
            "locator_second_stage_rerank",
            "numeric_year_rerank",
            "top60 后按年份、金额、百分比等数值锚点重排，优先解决研报和财报 rank_too_low。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 60,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "content_heavy",
                "locator_rerank_policy": "second_stage_numeric_rerank",
            },
        ),
        (
            "round_12",
            "P0",
            "locator_second_stage_rerank",
            "multi_doc_balanced_rerank",
            "top80 后综合 alias、数字、anchor quota 与 canonical 去重，优先提升多文档完整召回。",
            {
                "top_k": 20,
                "answer_top_k": 10,
                "first_stage_top_k": 80,
                "include_entities": True,
                "include_numbers": True,
                "include_synonyms": True,
                "profile_mode": "content_heavy",
                "multi_doc_bonus": True,
                "locator_rerank_policy": "all_anchor_quota_canonical_dedupe_second_stage",
            },
        ),
    ]
    attempts: list[AttemptConfig] = []
    for index, (round_id, priority, direction, variant, hypothesis, overrides) in enumerate(specs, start=1):
        base = {
            "attempt_id": f"attempt_{index:02d}",
            "round_id": round_id,
            "priority": priority,
            "direction": direction,
            "variant_name": variant,
            "hypothesis": hypothesis,
        }
        base.update(overrides)
        attempts.append(AttemptConfig(**base))
    return attempts


def write_attempt_outputs(
    attempt_dir: Path,
    attempt: AttemptConfig,
    candidate_rows: list[dict[str, Any]],
    locator_eval_rows: list[dict[str, Any]],
    locator_metrics: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    evidence_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    ensure_dir(attempt_dir)
    write_json(attempt_dir / "attempt_config.json", attempt.to_dict())
    write_json(attempt_dir / "run_manifest.json", {"created_at": datetime.now().isoformat(timespec="seconds"), "attempt": attempt.to_dict()})
    write_jsonl(attempt_dir / "doc_locator_candidates.jsonl", candidate_rows)
    CsvWriter.write(
        attempt_dir / "doc_locator_eval.csv",
        locator_eval_rows,
        [
            "qid",
            "domain",
            "true_doc_ids",
            "candidate_doc_ids",
            "doc_recall@1",
            "doc_any_recall@1",
            "doc_coverage@1",
            "doc_recall@3",
            "doc_any_recall@3",
            "doc_coverage@3",
            "doc_recall@5",
            "doc_any_recall@5",
            "doc_coverage@5",
            "doc_recall@10",
            "doc_any_recall@10",
            "doc_coverage@10",
            "failure_reason",
        ],
    )
    CsvWriter.write(
        attempt_dir / "failure_analysis.csv",
        [
            {
                "qid": row["qid"],
                "domain": row["domain"],
                "locator_failure_reason": row["failure_reason"],
                "evidence_failure_reason": evidence_rows[index]["failure_reason"],
                "doc_coverage@5": row["doc_coverage@5"],
                "evidence_doc_coverage": evidence_rows[index]["evidence_doc_coverage"],
                "recommended_action": recommend_action(row["failure_reason"], evidence_rows[index]["failure_reason"]),
            }
            for index, row in enumerate(locator_eval_rows)
        ],
        [
            "qid",
            "domain",
            "locator_failure_reason",
            "evidence_failure_reason",
            "doc_coverage@5",
            "evidence_doc_coverage",
            "recommended_action",
        ],
    )
    CsvWriter.write(
        attempt_dir / "token_usage_breakdown.csv",
        [
            {
                "qid": row["qid"],
                "domain": row["domain"],
                "estimated_prompt_tokens": row["estimated_prompt_tokens"],
                "estimated_completion_tokens": 800,
                "estimated_total_tokens": row["estimated_prompt_tokens"] + 800,
            }
            for row in evidence_rows
        ],
        ["qid", "domain", "estimated_prompt_tokens", "estimated_completion_tokens", "estimated_total_tokens"],
    )
    metrics = {
        **attempt.to_dict(),
        "locator": locator_metrics,
        "evidence": evidence_metrics,
        "token_usage": {
            "estimated_prompt_tokens": evidence_metrics["estimated_prompt_tokens"],
            "estimated_completion_tokens": len(evidence_rows) * 800,
            "estimated_total_tokens": evidence_metrics["estimated_prompt_tokens"] + len(evidence_rows) * 800,
        },
    }
    metrics["selection_score"] = attempt_selection_score(metrics)
    metrics["locator_selection_score"] = locator_selection_score(metrics)
    promote = decide_promote(metrics, baseline_metrics)
    metrics["promoted"] = promote["promoted"]
    metrics["promote_reason"] = promote["reason"]
    write_json(attempt_dir / "metrics.json", metrics)
    summary = [
        f"# {attempt.attempt_id} {attempt.variant_name}",
        "",
        f"- priority: `{attempt.priority}`",
        f"- direction: `{attempt.direction}`",
        f"- hypothesis: {attempt.hypothesis}",
        f"- doc_recall@5: `{locator_metrics['doc_recall@5']}`",
        f"- doc_coverage@5: `{locator_metrics['doc_coverage@5']}`",
        f"- evidence_doc_full_recall: `{evidence_metrics['evidence_doc_full_recall']}`",
        f"- locator_first_selection_score: `{metrics['locator_selection_score']}`",
        f"- estimated_total_tokens: `{metrics['token_usage']['estimated_total_tokens']}`",
        f"- evidence_first_selection_score: `{metrics['selection_score']}`",
        f"- promoted: `{promote['promoted']}`",
        f"- reason: {promote['reason']}",
    ]
    (attempt_dir / "attempt_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return metrics


def recommend_action(locator_reason: str, evidence_reason: str) -> str:
    if locator_reason in {"entity_missing", "generic_question", "profile_too_weak", "rank_too_low"}:
        return "improve_doc_locator"
    if locator_reason == "multi_doc_pair_missing":
        return "add_multi_doc_locator_or_per_doc_quota"
    if evidence_reason == "evidence_retrieval_fail":
        return "improve_chunk_retrieval_or_neighbor_expansion"
    return "answer_layer_or_token_optimization"


def decide_promote(metrics: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if baseline is None:
        return {"promoted": True, "reason": "baseline attempt"}
    current_score = metrics.get("selection_score", attempt_selection_score(metrics))
    base_score = baseline.get("selection_score", attempt_selection_score(baseline))
    current_locator_score = metrics.get("locator_selection_score", locator_selection_score(metrics))
    base_locator_score = baseline.get("locator_selection_score", locator_selection_score(baseline))
    current_recall = metrics["locator"]["doc_recall@5"]
    base_recall = baseline["locator"]["doc_recall@5"]
    current_tokens = metrics["token_usage"]["estimated_total_tokens"]
    base_tokens = baseline["token_usage"]["estimated_total_tokens"]
    if current_locator_score - base_locator_score >= 0.02:
        return {"promoted": True, "reason": "locator-first selection score improved by at least 2pp"}
    if current_score - base_score >= 0.02:
        return {"promoted": True, "reason": "evidence-first selection score improved by at least 2pp"}
    if current_recall - base_recall >= 0.03 and current_tokens <= base_tokens * 1.2:
        return {"promoted": True, "reason": "doc_recall@5 improved by at least 3pp within token budget"}
    if current_tokens <= base_tokens * 0.85 and current_recall >= base_recall:
        return {"promoted": True, "reason": "estimated tokens decreased by at least 15% without recall drop"}
    if metrics["evidence"]["evidence_doc_full_recall"] > baseline["evidence"]["evidence_doc_full_recall"] and current_tokens <= base_tokens * 1.2:
        return {"promoted": True, "reason": "evidence document recall improved within token budget"}
    if current_tokens > base_tokens * 1.3 and current_recall <= base_recall:
        return {"promoted": False, "reason": "token estimate increased over 30% without recall improvement"}
    return {"promoted": False, "reason": "no material improvement over baseline"}


def attempt_selection_score(metrics: dict[str, Any]) -> float:
    locator = metrics.get("locator", {})
    evidence = metrics.get("evidence", {})
    tokens = float((metrics.get("token_usage") or {}).get("estimated_total_tokens", 0) or 0)
    score = (
        0.4 * float(evidence.get("evidence_doc_full_recall", 0.0) or 0.0)
        + 0.25 * float(evidence.get("evidence_doc_coverage", 0.0) or 0.0)
        + 0.2 * float(locator.get("doc_recall@5", 0.0) or 0.0)
        + 0.1 * float(locator.get("doc_recall@10", 0.0) or 0.0)
        + 0.05 * float(locator.get("doc_coverage@5", 0.0) or 0.0)
        - min(tokens / 2_000_000.0, 1.0) * 0.03
    )
    return round(score, 6)


def locator_selection_score(metrics: dict[str, Any]) -> float:
    locator = metrics.get("locator", {})
    tokens = float((metrics.get("token_usage") or {}).get("estimated_total_tokens", 0) or 0)
    score = (
        0.55 * float(locator.get("doc_recall@5", 0.0) or 0.0)
        + 0.2 * float(locator.get("doc_coverage@5", 0.0) or 0.0)
        + 0.15 * float(locator.get("doc_recall@10", 0.0) or 0.0)
        + 0.07 * float(locator.get("doc_any_recall@5", 0.0) or 0.0)
        + 0.03 * float(locator.get("doc_coverage@10", 0.0) or 0.0)
        - min(tokens / 2_000_000.0, 1.0) * 0.015
    )
    return round(score, 6)


def write_round_summaries(output_dir: Path, attempt_metrics: list[dict[str, Any]]) -> None:
    rounds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metrics in attempt_metrics:
        rounds[metrics["round_id"]].append(metrics)
    for round_id, items in rounds.items():
        round_dir = ensure_dir(output_dir / "loop_runs" / round_id)
        promoted = [item for item in items if item.get("promoted")]
        rejected = [item for item in items if not item.get("promoted")]
        lines = [
            f"# {round_id} Summary",
            "",
            f"- attempts: `{len(items)}`",
            f"- promoted: `{len(promoted)}`",
            f"- rejected: `{len(rejected)}`",
            "",
            "| attempt | direction | variant | doc_recall@5 | doc_coverage@5 | locator_score | evidence_full_recall | est_tokens | promoted |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for item in items:
            lines.append(
                f"| `{item['attempt_id']}` | `{item['direction']}` | `{item['variant_name']}` | "
                f"{item['locator']['doc_recall@5']} | {item['locator']['doc_coverage@5']} | "
                f"{item.get('locator_selection_score', locator_selection_score(item))} | "
                f"{item['evidence']['evidence_doc_full_recall']} | "
                f"{item['token_usage']['estimated_total_tokens']} | `{item['promoted']}` |"
            )
        (round_dir / "round_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (round_dir / "promoted_changes.md").write_text(
            "\n".join(f"- `{item['attempt_id']}` `{item['variant_name']}`: {item['promote_reason']}" for item in promoted) + "\n",
            encoding="utf-8",
        )
        (round_dir / "rejected_changes.md").write_text(
            "\n".join(f"- `{item['attempt_id']}` `{item['variant_name']}`: {item['promote_reason']}" for item in rejected) + "\n",
            encoding="utf-8",
        )
        next_lines = ["# Next Round Plan", ""]
        if rejected:
            common_failures = Counter()
            for item in rejected:
                common_failures.update(item["locator"].get("failure_counts", {}))
                common_failures.update(item["evidence"].get("failure_counts", {}))
            next_lines.append("Prioritize the largest remaining failure buckets:")
            for key, count in common_failures.most_common(5):
                next_lines.append(f"- `{key}`: {count}")
        else:
            next_lines.append("No rejected attempts in this round.")
        (round_dir / "next_round_plan.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def choose_best_attempt(attempt_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        attempt_metrics,
        key=lambda item: (
            item["locator"]["doc_recall@5"],
            item["locator"]["doc_coverage@5"],
            item["locator"]["doc_recall@10"],
            item.get("locator_selection_score", locator_selection_score(item)),
            item["evidence"]["evidence_doc_full_recall"],
            -item["token_usage"]["estimated_total_tokens"],
        ),
        reverse=True,
    )[0]


def run_answering_for_best(
    output_dir: Path,
    best: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    clean_qids: list[str],
    payloads: dict[str, dict[str, Any]],
    parsed_root: Path,
    index_root: Path,
    reference_answers: dict[str, str],
    answer_limit: int,
    answer_workers: int,
    answer_strategy_config: Path,
    force_answer: bool,
) -> dict[str, Any]:
    from afa_agent.domains.registry import get_plugin

    os.environ["AFA_STRATEGY_CONFIG"] = str(answer_strategy_config.resolve())
    attempt = AttemptConfig(**{key: best[key] for key in AttemptConfig.__dataclass_fields__ if key in best})
    candidate_rows = locate_docs(questions, clean_qids, payloads, attempt)
    candidates_by_qid = {row["qid"]: row for row in candidate_rows}
    run_dir = ensure_dir(output_dir / "no_docids_clean_subset_run")
    qids = clean_qids[:answer_limit] if answer_limit > 0 else clean_qids
    existing_results = [] if force_answer else load_existing_answer_results(run_dir / "final_answers.json", qids)
    existing_qids = {result.qid for result in existing_results}
    results: list[AnswerResult] = existing_results[:]
    remaining_qids = [qid for qid in qids if qid not in existing_qids]

    def answer_one_qid(qid: str) -> AnswerResult:
        row = questions[qid]
        domain = row["domain"]
        effective_attempt = _effective_attempt_for_domain(attempt, domain)
        candidate_doc_ids = select_answer_doc_ids(candidates_by_qid[qid], row, effective_attempt)
        question = Question(
            qid=row["qid"],
            domain=row["domain"],
            split=row["split"],
            question=row["question"],
            options=row["options"],
            answer_format=row["answer_format"],
            type=row["type"],
            doc_ids=candidate_doc_ids,
            metadata={
                "true_doc_ids_for_eval_only": row.get("doc_ids", []),
                "no_docids_locator_attempt": attempt.attempt_id,
                "doc_ids_are_locator_candidates": True,
            },
        )
        last_exc: Exception | None = None
        for _ in range(2):
            try:
                plugin = get_plugin(domain)
                parsed_path = parsed_root / domain / "parsed.json"
                index_path = index_root / domain / "index.json"
                result = plugin.answer_one(question, parsed_path, index_path)
                result.debug_meta.setdefault("no_docids_locator", candidates_by_qid[qid])
                result.debug_meta.setdefault("true_doc_ids_for_eval_only", row.get("doc_ids", []))
                return result
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    failures: list[dict[str, Any]] = []
    if remaining_qids:
        if answer_workers > 1:
            with ThreadPoolExecutor(max_workers=answer_workers) as executor:
                futures = {executor.submit(answer_one_qid, qid): qid for qid in remaining_qids}
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failures.append({"qid": qid, "error_type": exc.__class__.__name__, "error": str(exc)})
                        write_jsonl(run_dir / "failed_answers.jsonl", failures)
                        continue
                    results.sort(key=lambda item: qids.index(item.qid))
                    export_answers_json(run_dir / "final_answers.json", results)
        else:
            for qid in remaining_qids:
                try:
                    results.append(answer_one_qid(qid))
                except Exception as exc:
                    failures.append({"qid": qid, "error_type": exc.__class__.__name__, "error": str(exc)})
                    write_jsonl(run_dir / "failed_answers.jsonl", failures)
                    continue
                results.sort(key=lambda item: qids.index(item.qid))
                export_answers_json(run_dir / "final_answers.json", results)
    results.sort(key=lambda item: qids.index(item.qid))
    export_answers_json(run_dir / "final_answers.json", results)
    export_evidence_json(run_dir / "evidence.json", results)
    export_answer_csv(run_dir / "answer.csv", results)
    answer_with_domain = []
    comparison = []
    for result in results:
        row = questions[result.qid]
        reference = reference_answers.get(result.qid, "")
        answer_with_domain.append(
            {
                "qid": result.qid,
                "answer": result.pred_answer,
                "domain": result.domain,
                "answer_format": row.get("answer_format", ""),
                "prompt_tokens": result.token_usage.prompt_tokens,
                "completion_tokens": result.token_usage.completion_tokens,
                "total_tokens": result.token_usage.total_tokens,
            }
        )
        comparison.append(
            {
                "qid": result.qid,
                "domain": result.domain,
                "reference_answer": reference,
                "no_docids_answer": result.pred_answer,
                "matches_reference": bool(reference and reference == result.pred_answer),
                "prompt_tokens": result.token_usage.prompt_tokens,
                "completion_tokens": result.token_usage.completion_tokens,
                "total_tokens": result.token_usage.total_tokens,
                "candidate_doc_ids": select_answer_doc_ids(
                    candidates_by_qid[result.qid],
                    row,
                    _effective_attempt_for_domain(attempt, result.domain),
                ),
                "true_doc_ids_for_eval_only": row.get("doc_ids", []),
            }
        )
    CsvWriter.write(
        run_dir / "answer_with_domain.csv",
        answer_with_domain,
        ["qid", "answer", "domain", "answer_format", "prompt_tokens", "completion_tokens", "total_tokens"],
    )
    CsvWriter.write(
        run_dir / "token_usage_breakdown.csv",
        answer_with_domain,
        ["qid", "domain", "prompt_tokens", "completion_tokens", "total_tokens"],
    )
    CsvWriter.write(
        run_dir / "comparison_vs_oracle_docids.csv",
        comparison,
        [
            "qid",
            "domain",
            "reference_answer",
            "no_docids_answer",
            "matches_reference",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "candidate_doc_ids",
            "true_doc_ids_for_eval_only",
        ],
    )
    accuracy = _avg(1.0 if row["matches_reference"] else 0.0 for row in comparison)
    token_total = sum(int(row["total_tokens"]) for row in comparison)
    lines = [
        "# No-Docids Clean Subset Run",
        "",
        f"- attempt_id: `{attempt.attempt_id}`",
        f"- variant: `{attempt.variant_name}`",
        f"- question_count: `{len(results)}`",
        f"- proxy_accuracy_vs_reference_88: `{accuracy}`",
        f"- total_tokens: `{token_total}`",
        "",
        "Accuracy is a proxy against the current 88% answer vector, not official labels.",
    ]
    (run_dir / "comparison_vs_oracle_docids.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    evidence_audit = write_evidence_answer_audit(run_dir, questions, qids, results)
    write_json(
        run_dir / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "attempt": attempt.to_dict(),
            "answer_strategy_config": str(answer_strategy_config.resolve()),
            "question_count": len(results),
            "failed_count": len(failures),
            "failed_qids": [row["qid"] for row in failures],
            "proxy_accuracy_vs_reference_88": accuracy,
            "evidence_answer_audit": evidence_audit,
            "total_tokens": token_total,
        },
    )
    return {
        "question_count": len(results),
        "proxy_accuracy_vs_reference_88": accuracy,
        "evidence_answer_audit": evidence_audit,
        "total_tokens": token_total,
    }


def write_evidence_answer_audit(
    run_dir: Path,
    questions: dict[str, dict[str, Any]],
    qids: list[str],
    results: list[AnswerResult],
) -> dict[str, Any]:
    by_qid = {result.qid: result for result in results}
    rows: list[dict[str, Any]] = []
    for qid in qids:
        result = by_qid.get(qid)
        if result is None:
            continue
        question = questions[qid]
        rows.append(audit_answer_from_evidence(result.to_dict(), question))
    CsvWriter.write(
        run_dir / "evidence_answer_audit.csv",
        rows,
        [
            "qid",
            "domain",
            "answer_format",
            "pred_answer",
            "support_status",
            "support_score",
            "selected_options",
            "selected_gate_statuses",
            "selected_gate_reasons",
            "all_option_gate_statuses",
            "evidence_count",
            "format_error",
            "final_consistency_issues",
            "option_label_mismatch",
            "audit_reasons",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ],
    )
    counts = Counter(row["support_status"] for row in rows)
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in sorted(set(row["domain"] for row in rows)):
        domain_rows = [row for row in rows if row["domain"] == domain]
        by_domain[domain] = {
            "question_count": len(domain_rows),
            "supported": sum(row["support_status"] == "supported" for row in domain_rows),
            "weak_supported": sum(row["support_status"] == "weak_supported" for row in domain_rows),
            "unsupported": sum(row["support_status"] == "unsupported" for row in domain_rows),
            "contradicted": sum(row["support_status"] == "contradicted" for row in domain_rows),
            "format_conflict": sum(row["support_status"] == "format_conflict" for row in domain_rows),
            "supported_rate": _avg(1.0 if row["support_status"] == "supported" else 0.0 for row in domain_rows),
            "supported_or_weak_rate": _avg(
                1.0 if row["support_status"] in {"supported", "weak_supported"} else 0.0
                for row in domain_rows
            ),
            "total_tokens": sum(int(row["total_tokens"]) for row in domain_rows),
        }
    metrics = {
        "question_count": len(rows),
        "supported": counts.get("supported", 0),
        "weak_supported": counts.get("weak_supported", 0),
        "unsupported": counts.get("unsupported", 0),
        "contradicted": counts.get("contradicted", 0),
        "format_conflict": counts.get("format_conflict", 0),
        "supported_rate": _avg(1.0 if row["support_status"] == "supported" else 0.0 for row in rows),
        "supported_or_weak_rate": _avg(
            1.0 if row["support_status"] in {"supported", "weak_supported"} else 0.0
            for row in rows
        ),
        "by_domain": by_domain,
    }
    lines = [
        "# Evidence Answer Audit",
        "",
        "This audit does not use reference answers. It checks whether the final answer is supported by retrieved evidence, evidence gate status, format constraints and final consistency metadata.",
        "",
        f"- question_count: `{metrics['question_count']}`",
        f"- supported: `{metrics['supported']}`",
        f"- weak_supported: `{metrics['weak_supported']}`",
        f"- unsupported: `{metrics['unsupported']}`",
        f"- contradicted: `{metrics['contradicted']}`",
        f"- format_conflict: `{metrics['format_conflict']}`",
        f"- supported_rate: `{metrics['supported_rate']}`",
        f"- supported_or_weak_rate: `{metrics['supported_or_weak_rate']}`",
        "",
        "## By Domain",
        "",
        "| domain | count | supported | weak | unsupported | contradicted | format | supported_rate | supported_or_weak | tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, payload in by_domain.items():
        lines.append(
            f"| `{domain}` | {payload['question_count']} | {payload['supported']} | {payload['weak_supported']} | "
            f"{payload['unsupported']} | {payload['contradicted']} | {payload['format_conflict']} | "
            f"{payload['supported_rate']} | {payload['supported_or_weak_rate']} | {payload['total_tokens']} |"
        )
    low_rows = [row for row in rows if row["support_status"] != "supported"]
    if low_rows:
        lines.extend(["", "## Non-Supported Cases", ""])
        for row in low_rows[:80]:
            lines.append(
                f"- `{row['qid']}` `{row['domain']}` answer `{row['pred_answer']}` -> "
                f"`{row['support_status']}`: {row['audit_reasons']}"
            )
    (run_dir / "evidence_answer_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(run_dir / "evidence_answer_audit_summary.json", metrics)
    return metrics


def audit_answer_from_evidence(answer_row: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    pred_answer = "".join(ch for ch in str(answer_row.get("pred_answer", "")).upper() if ch in {"A", "B", "C", "D"})
    options = question.get("options", {}) or {}
    answer_format = question.get("answer_format", "")
    selected = [ch for ch in pred_answer if ch in options or answer_format == "tf" and ch in {"A", "B"}]
    token_usage = answer_row.get("token_usage", {}) or {}
    debug = answer_row.get("debug_meta", {}) or {}
    final_issues = [str(item) for item in (debug.get("final_consistency_check", {}) or {}).get("issues", []) or []]
    option_labels = answer_row.get("option_labels", {}) or {}
    option_debug = {str(item.get("option", "")).upper(): item for item in (debug.get("option_debug") or [])}
    selected_gate_statuses: dict[str, str] = {}
    selected_gate_reasons: dict[str, list[str]] = {}
    all_gate_statuses: dict[str, str] = {}
    selected_scores: list[float] = []
    reasons: list[str] = []

    for option in sorted(set(options) | set(option_labels) | set(option_debug)):
        gate = ((option_debug.get(option) or {}).get("evidence_gate") or {}).get("final_gate") or {}
        status = str(gate.get("status") or "missing")
        all_gate_statuses[option] = status
        if option in selected:
            selected_gate_statuses[option] = status
            selected_gate_reasons[option] = list(gate.get("reasons") or [])
            try:
                selected_scores.append(float(gate.get("certainty_score", 0.0) or 0.0))
            except (TypeError, ValueError):
                selected_scores.append(0.0)

    fmt_error = answer_format_error(pred_answer, answer_format, options)
    if fmt_error:
        reasons.append(fmt_error)
    if not answer_row.get("evidence_items"):
        reasons.append("empty_evidence")
    if final_issues:
        reasons.extend(f"final_consistency:{item}" for item in final_issues)

    label_selected = sorted(option for option, label in option_labels.items() if label)
    option_label_mismatch = bool(label_selected and sorted(selected) != label_selected)
    if option_label_mismatch:
        reasons.append(f"option_label_mismatch:{''.join(label_selected)}")

    missing_selected_gate = [option for option, status in selected_gate_statuses.items() if status == "missing"]
    failed_selected_gate = [option for option, status in selected_gate_statuses.items() if status == "fail"]
    partial_selected_gate = [option for option, status in selected_gate_statuses.items() if status == "partial"]
    if missing_selected_gate:
        reasons.append("missing_selected_gate:" + ",".join(missing_selected_gate))
    if failed_selected_gate:
        reasons.append("failed_selected_gate:" + ",".join(failed_selected_gate))
    if partial_selected_gate:
        reasons.append("partial_selected_gate:" + ",".join(partial_selected_gate))

    if fmt_error:
        support_status = "format_conflict"
    elif option_label_mismatch or any("contradict" in reason.lower() for reason in reasons + final_issues):
        support_status = "contradicted"
    elif "empty_evidence" in reasons or failed_selected_gate or missing_selected_gate:
        support_status = "unsupported"
    elif final_issues or partial_selected_gate or (selected_scores and min(selected_scores) < 0.7):
        support_status = "weak_supported"
    else:
        support_status = "supported"

    return {
        "qid": answer_row["qid"],
        "domain": answer_row["domain"],
        "answer_format": answer_format,
        "pred_answer": pred_answer,
        "support_status": support_status,
        "support_score": round(min(selected_scores), 4) if selected_scores else "",
        "selected_options": selected,
        "selected_gate_statuses": selected_gate_statuses,
        "selected_gate_reasons": selected_gate_reasons,
        "all_option_gate_statuses": all_gate_statuses,
        "evidence_count": len(answer_row.get("evidence_items") or []),
        "format_error": fmt_error,
        "final_consistency_issues": final_issues,
        "option_label_mismatch": option_label_mismatch,
        "audit_reasons": sorted(set(reasons)),
        "prompt_tokens": int(token_usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(token_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(token_usage.get("total_tokens", 0) or 0),
    }


def answer_format_error(answer: str, answer_format: str, options: dict[str, str]) -> str:
    cleaned = "".join(ch for ch in str(answer).upper() if ch in set(options) | {"A", "B"})
    if answer_format == "tf":
        return "" if len(cleaned) == 1 and cleaned in {"A", "B"} else "tf_requires_one_a_or_b"
    option_cleaned = "".join(ch for ch in str(answer).upper() if ch in options)
    if answer_format == "mcq":
        return "" if len(option_cleaned) == 1 else "mcq_requires_one"
    if answer_format == "multi":
        return "" if len(set(option_cleaned)) >= 2 else "multi_requires_two_or_more"
    return ""


def load_existing_answer_results(path: Path, allowed_qids: list[str]) -> list[AnswerResult]:
    if not path.exists():
        return []
    allowed = set(allowed_qids)
    rows = read_json(path)
    results: list[AnswerResult] = []
    for row in rows:
        if row.get("qid") not in allowed:
            continue
        token = row.get("token_usage", {}) or {}
        from afa_agent.models import TokenUsage

        results.append(
            AnswerResult(
                qid=row["qid"],
                domain=row["domain"],
                question_type=row["question_type"],
                pred_answer=row["pred_answer"],
                option_labels=row.get("option_labels", {}),
                evidence_items=row.get("evidence_items", []),
                reasoning_summary=row.get("reasoning_summary", ""),
                token_usage=TokenUsage(
                    prompt_tokens=int(token.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(token.get("completion_tokens", 0) or 0),
                    total_tokens=int(token.get("total_tokens", 0) or 0),
                ),
                debug_meta=row.get("debug_meta", {}),
            )
        )
    return results


def read_reference_answers(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["qid"]: row.get("answer", "") for row in csv.DictReader(handle) if row.get("qid") and row["qid"] != "summary"}


def write_global_summary(
    output_dir: Path,
    clean_qids: list[str],
    eligibility_rows: list[dict[str, Any]],
    attempt_metrics: list[dict[str, Any]],
    best: dict[str, Any],
    answer_metrics: dict[str, Any] | None,
) -> None:
    write_best_locator_exports(output_dir, best)
    write_locator_optimization_reports(output_dir, attempt_metrics, best)
    special_count = sum(1 for row in eligibility_rows if row["category"] != "clean_blind_candidate")
    lines = [
        "# B Board Migration Loop Summary",
        "",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- clean_subset_count: `{len(clean_qids)}`",
        f"- special_subset_count: `{special_count}`",
        f"- attempts: `{len(attempt_metrics)}`",
        f"- best_attempt: `{best['attempt_id']}` `{best['variant_name']}`",
        f"- best_doc_recall@5: `{best['locator']['doc_recall@5']}`",
        f"- best_doc_recall@10: `{best['locator']['doc_recall@10']}`",
        f"- best_locator_first_selection_score: `{best.get('locator_selection_score', locator_selection_score(best))}`",
        f"- best_evidence_doc_full_recall: `{best['evidence']['evidence_doc_full_recall']}`",
        f"- best_evidence_first_selection_score: `{best.get('selection_score', attempt_selection_score(best))}`",
        f"- best_estimated_total_tokens: `{best['token_usage']['estimated_total_tokens']}`",
        "",
        "## Attempt Leaderboard",
        "",
        "| attempt | priority | direction | variant | doc_recall@5 | doc_coverage@5 | doc_recall@10 | locator_score | evidence_full_recall | est_tokens | promoted |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in sorted(
        attempt_metrics,
        key=lambda row: (
            row["locator"]["doc_recall@5"],
            row["locator"]["doc_coverage@5"],
            row["locator"]["doc_recall@10"],
            row.get("locator_selection_score", locator_selection_score(row)),
            row["evidence"]["evidence_doc_full_recall"],
            -row["token_usage"]["estimated_total_tokens"],
        ),
        reverse=True,
    ):
        lines.append(
            f"| `{item['attempt_id']}` | `{item['priority']}` | `{item['direction']}` | `{item['variant_name']}` | "
            f"{item['locator']['doc_recall@5']} | {item['locator']['doc_coverage@5']} | "
            f"{item['locator']['doc_recall@10']} | {item.get('locator_selection_score', locator_selection_score(item))} | "
            f"{item['evidence']['evidence_doc_full_recall']} | "
            f"{item['token_usage']['estimated_total_tokens']} | `{item['promoted']}` |"
        )
    lines.extend(["", "## Answer Baseline"])
    if answer_metrics:
        evidence_audit = answer_metrics.get("evidence_answer_audit", {}) or {}
        lines.extend(
            [
                "",
                f"- answered_questions: `{answer_metrics['question_count']}`",
                f"- evidence_supported_rate: `{evidence_audit.get('supported_rate', '')}`",
                f"- evidence_supported_or_weak_rate: `{evidence_audit.get('supported_or_weak_rate', '')}`",
                f"- evidence_supported: `{evidence_audit.get('supported', '')}`",
                f"- evidence_weak_supported: `{evidence_audit.get('weak_supported', '')}`",
                f"- evidence_unsupported: `{evidence_audit.get('unsupported', '')}`",
                f"- evidence_contradicted: `{evidence_audit.get('contradicted', '')}`",
                f"- evidence_format_conflict: `{evidence_audit.get('format_conflict', '')}`",
                f"- proxy_accuracy_vs_reference_88_secondary: `{answer_metrics['proxy_accuracy_vs_reference_88']}`",
                f"- total_tokens: `{answer_metrics['total_tokens']}`",
            ]
        )
    else:
        lines.extend(["", "- Answering was skipped in this run; locator/evidence loop artifacts are complete."])
    lines.extend(
        [
            "",
            "## Next Optimization Recommendation",
            "",
            "- If doc_recall@5 is low, prioritize doc profile and entity/synonym locator variants.",
            "- If doc_recall is acceptable but evidence recall is low, prioritize chunk retrieval, unit_type search and neighbor expansion.",
            "- If evidence recall is acceptable but evidence-supported answer rate is low, prioritize prompt, final consistency guard and supported-only multiselect.",
            "- If token grows faster than accuracy, prioritize evidence pool, structured evidence cards and dynamic topK.",
        ]
    )
    (output_dir / "b_board_migration_loop_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "best_strategy_config.json",
        {
            "selected_attempt": best["attempt_id"],
            "strategy": {key: best[key] for key in AttemptConfig.__dataclass_fields__ if key in best},
            "metrics": {
                "locator": best["locator"],
                "evidence": best["evidence"],
                "token_usage": best["token_usage"],
                "locator_first_selection_score": best.get("locator_selection_score", locator_selection_score(best)),
                "evidence_first_selection_score": best.get("selection_score", attempt_selection_score(best)),
                "answer": answer_metrics or {},
            },
            "policy": {
                "use_true_doc_ids_for_retrieval": False,
                "primary_answer_metric": "evidence_supported_without_reference_answers",
                "main_metric_subset": "clean_blind_candidate",
                "special_subset_policy": "diagnosis_only",
            },
        },
    )


def write_locator_optimization_reports(output_dir: Path, attempt_metrics: list[dict[str, Any]], best: dict[str, Any]) -> None:
    if not attempt_metrics:
        return
    baseline = next((item for item in attempt_metrics if item.get("attempt_id") == "attempt_31"), attempt_metrics[0])
    baseline_locator = baseline.get("locator", {})
    best_locator = best.get("locator", {})
    lines = [
        "# Locator Optimization Summary",
        "",
        f"- baseline_attempt: `{baseline.get('attempt_id')}` `{baseline.get('variant_name')}`",
        f"- best_attempt: `{best.get('attempt_id')}` `{best.get('variant_name')}`",
        f"- target_doc_recall@5: `>= 0.8`",
        f"- baseline_doc_recall@5: `{baseline_locator.get('doc_recall@5')}`",
        f"- best_doc_recall@5: `{best_locator.get('doc_recall@5')}`",
        f"- baseline_doc_coverage@5: `{baseline_locator.get('doc_coverage@5')}`",
        f"- best_doc_coverage@5: `{best_locator.get('doc_coverage@5')}`",
        f"- baseline_doc_recall@10: `{baseline_locator.get('doc_recall@10')}`",
        f"- best_doc_recall@10: `{best_locator.get('doc_recall@10')}`",
        f"- achieved_target: `{float(best_locator.get('doc_recall@5', 0.0) or 0.0) >= 0.8}`",
        "",
        "## Failure Buckets",
        "",
        "| bucket | baseline | best |",
        "|---|---:|---:|",
    ]
    buckets = sorted(set((baseline_locator.get("failure_counts") or {})) | set((best_locator.get("failure_counts") or {})))
    for bucket in buckets:
        lines.append(
            f"| `{bucket}` | {(baseline_locator.get('failure_counts') or {}).get(bucket, 0)} | "
            f"{(best_locator.get('failure_counts') or {}).get(bucket, 0)} |"
        )
    lines.extend(["", "## By Domain", "", "| domain | baseline_recall@5 | best_recall@5 | baseline_cov@5 | best_cov@5 |", "|---|---:|---:|---:|---:|"])
    for domain in DOMAINS:
        base_domain = (baseline_locator.get("by_domain") or {}).get(domain, {})
        best_domain = (best_locator.get("by_domain") or {}).get(domain, {})
        if not base_domain and not best_domain:
            continue
        lines.append(
            f"| `{domain}` | {base_domain.get('doc_recall@5', '')} | {best_domain.get('doc_recall@5', '')} | "
            f"{base_domain.get('doc_coverage@5', '')} | {best_domain.get('doc_coverage@5', '')} |"
        )
    (output_dir / "locator_round_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_locator_badcase_diff(output_dir, baseline, best)


def write_locator_badcase_diff(output_dir: Path, baseline: dict[str, Any], best: dict[str, Any]) -> None:
    baseline_eval = _read_locator_eval(output_dir, baseline)
    best_eval = _read_locator_eval(output_dir, best)
    baseline_candidates = _read_locator_candidates(output_dir, baseline)
    best_candidates = _read_locator_candidates(output_dir, best)
    rows: list[dict[str, Any]] = []
    for qid in sorted(set(baseline_eval) | set(best_eval)):
        base_row = baseline_eval.get(qid, {})
        best_row = best_eval.get(qid, {})
        if base_row.get("failure_reason") == "ok" and best_row.get("failure_reason") == "ok":
            continue
        true_doc_ids = baseline_candidates.get(qid, best_candidates.get(qid, {})).get("true_doc_ids_for_eval_only", [])
        base_docs = baseline_candidates.get(qid, {}).get("candidate_doc_ids", [])
        best_docs = best_candidates.get(qid, {}).get("candidate_doc_ids", [])
        rows.append(
            {
                "qid": qid,
                "domain": base_row.get("domain") or best_row.get("domain", ""),
                "baseline_failure": base_row.get("failure_reason", ""),
                "best_failure": best_row.get("failure_reason", ""),
                "baseline_doc_recall@5": base_row.get("doc_recall@5", ""),
                "best_doc_recall@5": best_row.get("doc_recall@5", ""),
                "baseline_doc_coverage@5": base_row.get("doc_coverage@5", ""),
                "best_doc_coverage@5": best_row.get("doc_coverage@5", ""),
                "true_doc_ids": true_doc_ids,
                "baseline_true_ranks": _candidate_ranks(true_doc_ids, base_docs),
                "best_true_ranks": _candidate_ranks(true_doc_ids, best_docs),
                "baseline_top10": base_docs[:10],
                "best_top10": best_docs[:10],
                "status_change": f"{base_row.get('failure_reason', '')}->{best_row.get('failure_reason', '')}",
            }
        )
    CsvWriter.write(
        output_dir / "locator_badcase_diff.csv",
        rows,
        [
            "qid",
            "domain",
            "baseline_failure",
            "best_failure",
            "baseline_doc_recall@5",
            "best_doc_recall@5",
            "baseline_doc_coverage@5",
            "best_doc_coverage@5",
            "true_doc_ids",
            "baseline_true_ranks",
            "best_true_ranks",
            "baseline_top10",
            "best_top10",
            "status_change",
        ],
    )


def _attempt_output_dir(output_dir: Path, metrics: dict[str, Any]) -> Path:
    return output_dir / "loop_runs" / str(metrics.get("round_id", "")) / str(metrics.get("attempt_id", ""))


def _read_locator_eval(output_dir: Path, metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = _attempt_output_dir(output_dir, metrics) / "doc_locator_eval.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["qid"]: row for row in csv.DictReader(handle)}


def _read_locator_candidates(output_dir: Path, metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = _attempt_output_dir(output_dir, metrics) / "doc_locator_candidates.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["qid"]] = row
    return rows


def _candidate_ranks(true_doc_ids: list[str], candidate_doc_ids: list[str]) -> list[int | None]:
    ranks: list[int | None] = []
    for doc_id in true_doc_ids:
        try:
            ranks.append(candidate_doc_ids.index(doc_id) + 1)
        except ValueError:
            ranks.append(None)
    return ranks


def write_best_locator_exports(output_dir: Path, best: dict[str, Any]) -> None:
    attempt_dir = output_dir / "loop_runs" / best["round_id"] / best["attempt_id"]
    for filename in ["doc_locator_candidates.jsonl", "doc_locator_eval.csv"]:
        source = attempt_dir / filename
        if source.exists():
            shutil.copyfile(source, output_dir / filename)
    locator = best.get("locator", {})
    evidence = best.get("evidence", {})
    lines = [
        "# Doc Locator Eval Summary",
        "",
        f"- selected_attempt: `{best.get('attempt_id')}`",
        f"- variant: `{best.get('variant_name')}`",
        f"- clean_subset_count: `{locator.get('question_count', 0)}`",
        f"- doc_recall@1: `{locator.get('doc_recall@1')}`",
        f"- doc_recall@3: `{locator.get('doc_recall@3')}`",
        f"- doc_recall@5: `{locator.get('doc_recall@5')}`",
        f"- doc_recall@10: `{locator.get('doc_recall@10')}`",
        f"- doc_coverage@5: `{locator.get('doc_coverage@5')}`",
        f"- evidence_doc_full_recall: `{evidence.get('evidence_doc_full_recall')}`",
        f"- evidence_doc_coverage: `{evidence.get('evidence_doc_coverage')}`",
        "",
        "## Failure Counts",
        "",
    ]
    for key, value in (locator.get("failure_counts") or {}).items():
        lines.append(f"- locator `{key}`: {value}")
    for key, value in (evidence.get("failure_counts") or {}).items():
        lines.append(f"- evidence `{key}`: {value}")
    (output_dir / "doc_locator_eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B board migration doc locator self-evolving loop.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT))
    parser.add_argument("--reference-answer-csv", default=str(DEFAULT_REFERENCE_ANSWER_CSV))
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--run-answering", action="store_true")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--answer-limit", type=int, default=0)
    parser.add_argument("--answer-workers", type=int, default=1)
    parser.add_argument("--answer-strategy-config", default=str(DEFAULT_ANSWER_STRATEGY_CONFIG))
    parser.add_argument("--force-answer", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    questions = load_questions()
    eligibility_rows, clean_qids = write_mask_outputs(output_dir, questions)
    payloads = load_domain_payloads(Path(args.parsed_root), Path(args.index_root))
    if args.answer_only:
        best_payload = read_json(output_dir / "best_strategy_config.json")
        best = {
            **best_payload["strategy"],
            "locator": best_payload["metrics"].get("locator", {}),
            "evidence": best_payload["metrics"].get("evidence", {}),
            "token_usage": best_payload["metrics"].get("token_usage", {}),
        }
        answer_metrics = run_answering_for_best(
            output_dir=output_dir,
            best=best,
            questions=questions,
            clean_qids=clean_qids,
            payloads=payloads,
            parsed_root=Path(args.parsed_root),
            index_root=Path(args.index_root),
            reference_answers=read_reference_answers(Path(args.reference_answer_csv)),
            answer_limit=args.answer_limit,
            answer_workers=max(1, args.answer_workers),
            answer_strategy_config=Path(args.answer_strategy_config),
            force_answer=args.force_answer,
        )
        attempt_metrics = load_existing_attempt_metrics(output_dir)
        write_global_summary(output_dir, clean_qids, eligibility_rows, attempt_metrics, best, answer_metrics)
        print(output_dir)
        return
    attempts = default_attempts()
    if args.max_attempts > 0:
        attempts = attempts[: args.max_attempts]

    attempt_metrics: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None
    for attempt in attempts:
        attempt_dir = output_dir / "loop_runs" / attempt.round_id / attempt.attempt_id
        candidate_rows = locate_docs(questions, clean_qids, payloads, attempt)
        locator_eval_rows, locator_metrics = evaluate_locator(candidate_rows)
        evidence_rows, evidence_metrics = evaluate_evidence_retrieval(questions, candidate_rows, payloads, attempt)
        metrics = write_attempt_outputs(
            attempt_dir,
            attempt,
            candidate_rows,
            locator_eval_rows,
            locator_metrics,
            evidence_rows,
            evidence_metrics,
            baseline_metrics,
        )
        if baseline_metrics is None:
            baseline_metrics = metrics
        attempt_metrics.append(metrics)

    write_round_summaries(output_dir, attempt_metrics)
    best = choose_best_attempt(attempt_metrics)
    answer_metrics = None
    if args.run_answering:
        answer_metrics = run_answering_for_best(
            output_dir=output_dir,
            best=best,
            questions=questions,
            clean_qids=clean_qids,
            payloads=payloads,
            parsed_root=Path(args.parsed_root),
            index_root=Path(args.index_root),
            reference_answers=read_reference_answers(Path(args.reference_answer_csv)),
            answer_limit=args.answer_limit,
            answer_workers=max(1, args.answer_workers),
            answer_strategy_config=Path(args.answer_strategy_config),
            force_answer=args.force_answer,
        )
    write_global_summary(output_dir, clean_qids, eligibility_rows, attempt_metrics, best, answer_metrics)
    print(output_dir)


def load_existing_attempt_metrics(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "loop_runs").glob("*/*/metrics.json")):
        rows.append(read_json(path))
    return rows


if __name__ == "__main__":
    main()
