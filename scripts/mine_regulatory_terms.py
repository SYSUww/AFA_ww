#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
PURE_NUMBER_RE = re.compile(r"^[0-9０-９一二三四五六七八九十百千万亿零〇两]+$")
ARTICLE_MARK_RE = re.compile(r"^第[一二三四五六七八九十百千万零〇两0-9]+[章节条款项号]$")

STOPWORDS = {
    "一个",
    "一种",
    "上述",
    "下列",
    "不得",
    "以及",
    "但是",
    "或者",
    "可以",
    "应当",
    "进行",
    "有关",
    "相关",
    "规定",
    "办法",
    "通知",
    "公告",
    "决定",
    "情况",
    "如下",
    "以上",
    "以下",
    "其中",
    "其他",
    "本条",
    "本款",
    "本项",
    "本法",
    "本办法",
    "部门",
    "单位",
    "个人",
    "文件",
    "内容",
    "时间",
    "日期",
    "附件",
    "及其",
    "系统",
    "类型",
    "要求",
    "决定书",
    "有限公司",
    "中华人民共和国",
    "中文名称",
    "制定依据",
    "公司",
    "证券",
    "交易",
    "信息",
    "业务",
    "资产",
    "上市",
    "股东",
    "管理",
    "董事",
    "机构",
    "报告",
    "审计",
    "投资",
    "代码",
    "编码",
    "监管",
    "风险",
    "客户",
    "账户",
    "重大",
    "服务",
    "说明",
    "包括",
    "控制",
    "股票",
    "中国",
    "资金",
    "主体",
    "规则",
    "是否",
    "存在",
    "行为",
    "企业",
    "经营",
    "规范",
    "名称",
    "人员",
    "制定",
    "事项",
}

BAD_PREFIXES = ("应当", "不得", "可以", "按照", "根据", "有关", "相关", "以下", "以上")
BAD_SUFFIXES = ("规定", "办法", "通知", "公告", "决定", "情况", "如下", "之一", "以上", "以下", "应当")
BAD_FRAGMENTS = ("本处罚", "收到本", "之日起", "的其他", "的公司", "规定的")
GOOD_HINTS = (
    "监管",
    "证券",
    "基金",
    "期货",
    "上市",
    "发行",
    "披露",
    "账户",
    "客户",
    "交易",
    "处罚",
    "违法",
    "违规",
    "内幕",
    "信息",
    "反洗钱",
    "受益",
    "所有人",
    "尽职",
    "调查",
    "适当性",
    "募集",
    "托管",
    "管理人",
    "股东",
    "董事",
    "审计",
    "风险",
    "资本",
    "资产",
    "净值",
    "信用",
)


def _require_jieba():
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("jieba is required for regulatory term mining") from exc
    return jieba


def _load_existing_domain_terms() -> set[str]:
    try:
        from afa_agent.text_utils import DOMAIN_TERMS
    except Exception:
        return set()
    return {term.lower() for term in DOMAIN_TERMS}


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_candidate(term: str) -> bool:
    term = term.strip().lower()
    if len(term) < 2 or len(term) > 16:
        return False
    if not TOKEN_RE.fullmatch(term):
        return False
    if term in STOPWORDS:
        return False
    if PURE_NUMBER_RE.fullmatch(term) or ARTICLE_MARK_RE.fullmatch(term):
        return False
    if not CHINESE_RE.search(term):
        return False
    if "的" in term and term not in {"公开发行的证券"}:
        return False
    if "应当" in term:
        return False
    if any(fragment in term for fragment in BAD_FRAGMENTS):
        return False
    if re.search(r"\d{4}年", term):
        return False
    if re.search(r"[第年月日号条款项章]{2,}$", term) and len(term) <= 4:
        return False
    if any(term.startswith(prefix) for prefix in BAD_PREFIXES) and len(term) <= 6:
        return False
    if any(term.endswith(suffix) for suffix in BAD_SUFFIXES) and len(term) <= 6:
        return False
    return True


def jieba_terms(text: str, jieba) -> list[str]:
    tokens = [
        token.strip().lower()
        for token in jieba.lcut(text, cut_all=False)
        if TOKEN_RE.fullmatch(token.strip().lower())
    ]
    terms: list[str] = []
    for token in tokens:
        if is_candidate(token):
            terms.append(token)
    for size in (2, 3):
        for idx in range(0, max(0, len(tokens) - size + 1)):
            term = "".join(tokens[idx : idx + size]).strip().lower()
            if is_candidate(term):
                terms.append(term)
    return terms


def char_ngram_terms(text: str, min_n: int, max_n: int) -> list[str]:
    terms = []
    for segment in re.findall(r"[\u4e00-\u9fff]{%d,}" % min_n, text):
        upper = min(max_n, len(segment))
        for n in range(min_n, upper + 1):
            for idx in range(0, len(segment) - n + 1):
                term = segment[idx : idx + n].lower()
                if is_candidate(term):
                    terms.append(term)
    return terms


def compact_example(text: str, term: str, radius: int = 34) -> str:
    idx = text.lower().find(term.lower())
    if idx < 0:
        return text[: radius * 2]
    start = max(0, idx - radius)
    end = min(len(text), idx + len(term) + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def build_candidates(
    units: list[dict],
    documents: list[dict],
    *,
    min_freq: int,
    top_k: int,
    min_ngram: int,
    max_ngram: int,
    include_char_ngrams: bool,
) -> list[dict]:
    jieba = _require_jieba()
    existing_terms = _load_existing_domain_terms()
    for term in existing_terms:
        jieba.add_word(term, freq=200000)

    doc_type_by_id = {doc["doc_id"]: doc.get("doc_type") or "unknown" for doc in documents}
    freq: Counter[str] = Counter()
    doc_freq: dict[str, set[str]] = defaultdict(set)
    unit_type_freq: dict[str, Counter[str]] = defaultdict(Counter)
    doc_type_freq: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)

    for unit in units:
        text = normalize_text(unit.get("text") or "")
        if not text:
            continue
        unit_terms = jieba_terms(text, jieba)
        if include_char_ngrams:
            unit_terms.extend(char_ngram_terms(text, min_ngram, max_ngram))
        seen_in_unit = set()
        for term in unit_terms:
            if not is_candidate(term):
                continue
            freq[term] += 1
            doc_freq[term].add(unit["doc_id"])
            unit_type_freq[term][unit.get("unit_type") or "unknown"] += 1
            doc_type_freq[term][doc_type_by_id.get(unit["doc_id"], "unknown")] += 1
            if term not in seen_in_unit and len(examples[term]) < 3:
                examples[term].append(
                    {
                        "unit_id": unit.get("unit_id"),
                        "unit_type": unit.get("unit_type"),
                        "text": compact_example(text, term),
                    }
                )
            seen_in_unit.add(term)

    total_docs = max(1, len({unit["doc_id"] for unit in units}))
    candidates = []
    for term, count in freq.items():
        if count < min_freq:
            continue
        df = len(doc_freq[term])
        if df < 2 and count < min_freq * 2:
            continue
        length_bonus = min(3.2, 0.8 + len(term) / 4)
        coverage = math.log1p(df)
        rarity = math.log((total_docs + 1) / (df + 1)) + 1.0
        hint_bonus = 1.25 if any(hint in term for hint in GOOD_HINTS) else 1.0
        existing_penalty = 0.72 if term in existing_terms else 1.0
        broad_penalty = 0.55 if len(term) <= 2 else 1.0
        score = count * coverage * rarity * length_bonus * hint_bonus * existing_penalty * broad_penalty
        candidates.append(
            {
                "term": term,
                "score": round(score, 4),
                "freq": count,
                "doc_freq": df,
                "unit_types": dict(unit_type_freq[term].most_common(5)),
                "doc_types": dict(doc_type_freq[term].most_common(5)),
                "already_in_domain_terms": term in existing_terms,
                "examples": examples[term],
            }
        )

    candidates.sort(key=lambda item: (item["score"], item["freq"], len(item["term"])), reverse=True)
    return candidates[:top_k]


def write_csv(path: Path, candidates: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "term",
                "score",
                "freq",
                "doc_freq",
                "unit_types",
                "doc_types",
                "already_in_domain_terms",
                "example",
            ],
        )
        writer.writeheader()
        for idx, item in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "term": item["term"],
                    "score": item["score"],
                    "freq": item["freq"],
                    "doc_freq": item["doc_freq"],
                    "unit_types": json.dumps(item["unit_types"], ensure_ascii=False),
                    "doc_types": json.dumps(item["doc_types"], ensure_ascii=False),
                    "already_in_domain_terms": item["already_in_domain_terms"],
                    "example": item["examples"][0]["text"] if item["examples"] else "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine candidate regulatory domain terms from preprocessed units.")
    parser.add_argument("--preprocessed-root", type=Path, default=Path("artifacts/preprocessed/regulatory"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/preprocessed/regulatory/terms"))
    parser.add_argument("--min-freq", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=600)
    parser.add_argument("--min-ngram", type=int, default=3)
    parser.add_argument("--max-ngram", type=int, default=8)
    parser.add_argument("--include-char-ngrams", action="store_true")
    args = parser.parse_args()

    units = json.loads((args.preprocessed_root / "units.json").read_text(encoding="utf-8"))
    documents = json.loads((args.preprocessed_root / "documents.json").read_text(encoding="utf-8"))
    candidates = build_candidates(
        units,
        documents,
        min_freq=args.min_freq,
        top_k=args.top_k,
        min_ngram=args.min_ngram,
        max_ngram=args.max_ngram,
        include_char_ngrams=args.include_char_ngrams,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "regulatory_terms_candidates.json"
    csv_path = args.output_dir / "regulatory_terms_candidates.csv"
    summary_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, candidates)
    summary_path.write_text(
        json.dumps(
            {
                "source_units": len(units),
                "source_documents": len(documents),
                "candidate_count": len(candidates),
                "min_freq": args.min_freq,
                "top_k": args.top_k,
                "min_ngram": args.min_ngram,
                "max_ngram": args.max_ngram,
                "include_char_ngrams": args.include_char_ngrams,
                "outputs": {
                    "json": str(json_path.resolve()),
                    "csv": str(csv_path.resolve()),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
