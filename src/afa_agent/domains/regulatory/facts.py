from __future__ import annotations

import re
from typing import Any

from afa_agent.models import Question, RetrievalHit
from afa_agent.text_utils import tokenize_zh


DATE_RE = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日")
DEADLINE_RE = re.compile(
    r"(?:\d+|[一二三四五六七八九十百零〇两]+)\s*(?:个工作日|工作日|日|个月|月|年|小时)"
)
AMOUNT_RE = re.compile(
    r"(?:人民币|美元)?\s*(?:\d+(?:\.\d+)?|[一二三四五六七八九十百零〇两]+)\s*(?:亿元|万元|元|万美元|美元)"
)
ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零〇两\d]+条")
MODAL_RE = re.compile(r"应当|不得|可以|无需|不需|不需要|必须|至少|提前|不得早于|原则上")
REPORT_RE = re.compile(r"差异报告|可疑交易报告|大额交易报告|风险评估报告|报告")
ACTION_TERMS = [
    "提交",
    "报送",
    "报告",
    "保存",
    "核实",
    "识别",
    "查询",
    "核对",
    "披露",
    "审议",
    "批准",
    "终止",
    "建立",
    "提供",
    "公示",
    "申请",
    "变更",
    "撤并",
    "施行",
    "废止",
    "停止施行",
    "扣减",
    "处罚",
]
NOISY_TOKENS = {
    "关于",
    "结合",
    "依据",
    "根据",
    "相关",
    "规定",
    "下列",
    "说法",
    "正确",
    "准确",
    "符合",
    "要求",
    "描述",
    "哪些",
    "哪项",
    "以下",
    "有",
    "中",
    "的",
}
TF_OPTION_TEXT = {"正确", "错误", "对", "错"}


def extract_regulatory_facts(text: str) -> dict[str, list[str]]:
    return {
        "articles": _unique(ARTICLE_RE.findall(text)),
        "dates": _unique(DATE_RE.findall(text)),
        "deadlines": _unique(_compact_spaces(item) for item in DEADLINE_RE.findall(text)),
        "amounts": _unique(_compact_spaces(item) for item in AMOUNT_RE.findall(text)),
        "modals": _unique(MODAL_RE.findall(text)),
        "reports": _unique(REPORT_RE.findall(text)),
        "actions": [term for term in ACTION_TERMS if term in text],
    }


def extract_hit_facts(hit: RetrievalHit) -> dict[str, Any]:
    facts = extract_regulatory_facts(hit.text)
    article_no = hit.metadata.get("article_no")
    if article_no and article_no not in facts["articles"]:
        facts["articles"].insert(0, str(article_no))
    return facts


def build_regulatory_query_variants(question: Question, option_key: str, option_text: str) -> list[str]:
    question_text = question.question.strip()
    option_text = option_text.strip()
    if question.answer_format == "tf":
        seeds = [question_text, *split_statement_parts(question_text)]
    else:
        option_keywords = keyword_query(option_text)
        question_keywords = keyword_query(question_text, max_tokens=10)
        seeds = [
            option_text,
            option_keywords,
            numeric_action_query(option_text),
            f"{option_keywords} {question_keywords}".strip(),
            question_keywords,
        ]
    return _dedupe_query(seed for seed in seeds if seed and seed.strip())


def split_statement_parts(text: str) -> list[str]:
    pieces = re.split(r"(?:，且|且|同时|并且|；|;)", text)
    return [piece.strip(" ：:，。 ") for piece in pieces if len(piece.strip()) >= 8]


def keyword_query(text: str, max_tokens: int = 18) -> str:
    facts = extract_regulatory_facts(text)
    selected: list[str] = []
    for group in ("articles", "dates", "deadlines", "amounts", "reports", "modals", "actions"):
        selected.extend(facts[group])
    for token in tokenize_zh(text):
        if token in NOISY_TOKENS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        selected.append(token)
    return " ".join(_unique(selected)[:max_tokens])


def numeric_action_query(text: str) -> str:
    facts = extract_regulatory_facts(text)
    anchors = facts["dates"] + facts["deadlines"] + facts["amounts"] + facts["reports"] + facts["actions"]
    if not anchors:
        return ""
    return " ".join(_unique(anchors))


def summarize_rule_alignment(option_text: str, hits: list[RetrievalHit]) -> dict[str, Any]:
    option_facts = extract_regulatory_facts(option_text)
    hit_facts = [extract_hit_facts(hit) for hit in hits]
    merged = merge_fact_rows(hit_facts)
    checks = {
        "deadline": _coverage(option_facts["deadlines"], merged["deadlines"]),
        "date": _coverage(option_facts["dates"], merged["dates"]),
        "amount": _coverage(option_facts["amounts"], merged["amounts"]),
        "report": _coverage(option_facts["reports"], merged["reports"]),
        "modal": _coverage(option_facts["modals"], merged["modals"]),
    }
    return {
        "option_facts": option_facts,
        "evidence_facts": merged,
        "checks": checks,
    }


def format_rule_summary(rule_summary: dict[str, Any]) -> str:
    checks = rule_summary.get("checks", {})
    rows = []
    for label in ("deadline", "date", "amount", "report", "modal"):
        item = checks.get(label, {})
        expected = item.get("expected") or []
        if not expected:
            continue
        matched = item.get("matched") or []
        missing = item.get("missing") or []
        rows.append(f"{label}: expected={expected}; matched={matched}; missing={missing}")
    if not rows:
        return "未抽取到需要精确核验的期限、日期、金额、报告类型或模态词。"
    return "\n".join(rows)


def merge_fact_rows(rows: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    merged = {
        "articles": [],
        "dates": [],
        "deadlines": [],
        "amounts": [],
        "modals": [],
        "reports": [],
        "actions": [],
    }
    for row in rows:
        for key in merged:
            merged[key].extend(row.get(key, []))
    return {key: _unique(value) for key, value in merged.items()}


def _coverage(expected: list[str], actual: list[str]) -> dict[str, list[str]]:
    matched = [item for item in expected if item in actual]
    return {
        "expected": expected,
        "matched": matched,
        "missing": [item for item in expected if item not in matched],
    }


def _compact_spaces(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _dedupe_query(items) -> list[str]:
    cleaned = []
    seen = set()
    for item in items:
        value = re.sub(r"\s+", " ", item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _unique(items) -> list[str]:
    output = []
    seen = set()
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
