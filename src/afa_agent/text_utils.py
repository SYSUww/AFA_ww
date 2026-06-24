from __future__ import annotations

import re
from typing import Iterable


ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百零〇两\d]+条)\s*(.*)$")
CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百零〇两\d]+章)\s*(.*)$")
SECTION_RE = re.compile(r"^(第[一二三四五六七八九十百零〇两\d]+节)\s*(.*)$")
DOMAIN_TERMS = [
    "A股",
    "营业收入",
    "营业总收入",
    "归母净利润",
    "归属于上市公司股东的净利润",
    "经营活动产生的现金流量净额",
    "经营现金流",
    "研发投入",
    "研发投入占营业收入比例",
    "研发投入占营业收入的比例",
    "现金分红",
    "每10股派",
    "保险责任",
    "身故保险金",
    "保单账户价值",
    "投资组合账户价值",
    "现金价值",
    "已交保费",
    "基本保额",
    "责任免除",
    "养老保险金",
    "退保金额",
    "市场规模",
    "渗透率",
    "同比增长",
    "同比下降",
    "投资建议",
    "风险提示",
    "行业集中度",
    "发行人",
    "发行规模",
    "债券期限",
    "票面利率",
    "利率条款",
    "主体信用评级",
    "债项信用评级",
    "受托管理人",
    "募集资金用途",
    "回售条款",
    "赎回条款",
    "客户尽职调查",
    "受益所有人",
    "反洗钱",
    "数据安全",
    "信息披露",
    "行政许可",
    "监督管理",
]
STOPWORD_SINGLE_CHARS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "为",
    "在",
    "对",
    "中",
    "于",
    "由",
    "等",
    "并",
    "按",
    "将",
    "其",
    "本",
    "该",
    "若",
    "如",
}
KEEP_SINGLE_CHARS = {"年", "条", "章", "节", "款", "股", "元", "亿", "万", "期", "日", "月", "a"}
_JIEBA_READY = False


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_lines(lines: Iterable[str]) -> list[str]:
    cleaned = []
    for line in lines:
        normalized = normalize_whitespace(line)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _require_jieba():
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("jieba is required for BM25 tokenization") from exc
    return jieba


def _ensure_jieba_terms() -> object:
    global _JIEBA_READY
    jieba = _require_jieba()
    if not _JIEBA_READY:
        for term in DOMAIN_TERMS:
            jieba.add_word(term, freq=200000)
        _JIEBA_READY = True
    return jieba


def _append_unique(tokens: list[str], seen: set[str], token: str) -> None:
    cleaned = token.strip().lower()
    if not cleaned or cleaned in seen:
        return
    seen.add(cleaned)
    tokens.append(cleaned)


def _keep_jieba_token(token: str) -> bool:
    if not token.strip():
        return False
    if re.fullmatch(r"\W+", token, flags=re.UNICODE):
        return False
    if len(token) == 1 and token not in KEEP_SINGLE_CHARS and token in STOPWORD_SINGLE_CHARS:
        return False
    return True


def tokenize_zh(text: str) -> list[str]:
    normalized = normalize_whitespace(text).lower()
    jieba = _ensure_jieba_terms()
    tokens: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(r"[a-z0-9_.%]+", normalized):
        _append_unique(tokens, seen, match.group(0))
    for match in re.finditer(r"\d[\d,]*(?:\.\d+)?\s*(?:%|％|亿元|万元|元|年|月|日|股|倍)?", normalized):
        raw = match.group(0).strip()
        _append_unique(tokens, seen, raw)
        if "," in raw:
            _append_unique(tokens, seen, raw.replace(",", ""))
    for match in re.finditer(r"第[一二三四五六七八九十百零〇两\d]+[章节条]", normalized):
        _append_unique(tokens, seen, match.group(0))
    for term in DOMAIN_TERMS:
        if term.lower() in normalized:
            _append_unique(tokens, seen, term)

    for token in jieba.lcut(normalized, cut_all=False):
        token = token.strip()
        if _keep_jieba_token(token):
            _append_unique(tokens, seen, token)

    for segment in re.findall(r"[\u4e00-\u9fff]{1,}", normalized):
        if len(segment) == 1:
            if _keep_jieba_token(segment):
                _append_unique(tokens, seen, segment)
            continue
        for item in (segment[i : i + 2] for i in range(len(segment) - 1)):
            _append_unique(tokens, seen, item)
    return tokens
