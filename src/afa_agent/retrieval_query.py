from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

from afa_agent.text_utils import normalize_whitespace, tokenize_zh


QUERY_PLAN_VERSION = "semantic_slots_v1"

_DOCUMENT_TITLE_RE = re.compile(r"《([^》]{2,100})》")
_COORDINATED_REPORT_SUBJECT_RE = re.compile(
    r"(?:根据|查阅|结合|依据)?\s*"
    r"([^？。\n]{2,100}?)\s*"
    r"(20\d{2})\s*年?\s*(?:年度报告|年报)"
)
_YEAR_RANGE_RE = re.compile(r"20\d{2}\s*(?:—|–|-|至)\s*20\d{2}\s*年?")
_PERIOD_RE = re.compile(
    r"(?:20\d{2}(?:年度|年(?:\d{1,2}月\d{1,2}日|"
    r"\d{1,2}\s*(?:—|–|-|至)\s*\d{1,2}月|"
    r"\d{1,2}月|上半年|下半年|末)?)|"
    r"第[一二三四五六七八九十百零〇两\d]+\s*(?:个)?(?:保单)?年度|"
    r"(?:\d+|[一二三四五六七八九十百零〇两]+)\s*"
    r"(?:个工作日|工作日|个自然日|自然日|日|个月|月|年))"
)
_QUANTITY_RE = re.compile(
    r"(?:人民币|美元)?\s*-?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|％|个百分点|亿元|万元|元|万股|股|万美元|美元|倍|kWh|GWh|人|万人)?",
    flags=re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零〇两\d]+条")
_ORG_SUFFIX_RE = re.compile(
    r"([A-Za-z0-9\u4e00-\u9fff·（）()]{2,32}?"
    r"(?:股份有限公司|有限责任公司|集团有限公司|有限公司|集团|银行|证券|"
    r"保险|交易所|委员会|管理局|研究院))"
)
_INSURANCE_SUBJECT_RE = re.compile(
    r"(?:^|[：；;。])\s*"
    r"([A-Za-z0-9\u4e00-\u9fff·]{2,20}?)"
    r"(?=(?:中|在第|被保险人|已开始|累计|所交|退还|现金价值|"
    r"第[一二三四五六七八九十百零〇两\d]+))"
)

_LEADING_NOISE = (
    "计算题",
    "判断题",
    "多选题",
    "单选题",
    "根据",
    "查阅",
    "关于",
    "结合",
    "依据",
    "请问",
    "以下",
)

_GENERAL_TOPIC_TERMS = (
    "营业收入",
    "营业总收入",
    "归母净利润",
    "净利润",
    "经营活动产生的现金流量净额",
    "经营现金流",
    "资产负债率",
    "净资产收益率",
    "加权平均净资产收益率",
    "基本每股收益",
    "EBITDA",
    "现金分红",
    "分地区收入",
    "毛利率",
    "评估增值率",
    "募集资金用途",
    "债务结构调整",
    "连带责任保证",
    "担保",
    "身故保险金",
    "基本保险金额",
    "基本保额",
    "账户价值",
    "现金价值",
    "已交保费",
    "退保费用",
    "保单年度",
    "责任免除",
    "等待期",
    "免赔额",
    "减额交清",
    "客户尽职调查",
    "受益所有人",
    "信息披露",
    "分类评价",
    "反洗钱",
    "收费项目",
    "公示",
    "市场规模",
    "渗透率",
    "订单",
    "销量",
    "单车带电量",
    "资产配置",
    "金融机构",
    "低利率",
    "资产荒",
    "杠杆",
)

_DOMAIN_TOPIC_TERMS = {
    "financial_reports": (
        "报告期",
        "年度报告",
        "合并报表",
        "母公司",
        "原始金额",
        "每股",
        "每10股",
        "中期分红",
        "全年分红",
        "权益乘数",
    ),
    "financial_contracts": (
        "募集说明书",
        "发行人",
        "评估基准日",
        "交易作价",
        "违约",
        "回售",
        "赎回",
        "转股价格",
    ),
    "insurance": (
        "保险责任",
        "给付比例",
        "领取养老金",
        "解除合同",
        "实付保险费",
        "应付保险费",
        "较大值",
        "差额",
    ),
    "regulatory": (
        "适用主体",
        "法律责任",
        "行政监管措施",
        "报告",
        "报送",
        "核实",
        "披露",
        "审议",
        "批准",
        "支付机构",
        "收费标准",
        "调整",
    ),
    "research": (
        "同比",
        "环比",
        "占比",
        "风险",
        "成本",
        "需求",
        "供给",
        "因果",
        "高股息",
        "FVOCI",
        "方向性自营",
    ),
}

_RELATION_TERMS = (
    "取较大值",
    "取较小值",
    "较大者",
    "较小者",
    "同比增长",
    "同比下降",
    "同比",
    "环比",
    "占比",
    "百分点",
    "平均值",
    "合计",
    "累计",
    "差额",
    "排序",
    "高于",
    "低于",
    "不低于",
    "不高于",
    "增加",
    "减少",
    "提高",
    "下降",
    "适用",
    "承担",
    "负责",
    "给付",
    "赔付",
    "披露",
    "报送",
    "核实",
    "审议",
    "批准",
    "公示",
)

_SCOPE_TERMS = (
    "所有",
    "全部",
    "均",
    "任何",
    "仅",
    "只",
    "至少",
    "以上",
    "以下",
    "不超过",
    "不得",
    "必须",
    "无需",
    "不需要",
    "明确",
    "未明确",
    "未提及",
    "不存在",
    "无论",
    "分别",
    "彼此独立",
)

_EXCEPTION_TERMS = (
    "除外",
    "但",
    "否则",
    "例外",
    "不适用",
    "前提",
    "条件",
    "无民事行为能力",
    "可疑",
)

_ABSENCE_TERMS = ("未提及", "未明确", "不存在", "不包括", "没有", "均未", "从未")

_CONTRAST_REPLACEMENTS = (
    ("至少提前", "未提前满"),
    ("不超过", "超过"),
    ("不得低于", "低于"),
    ("不低于", "低于"),
    ("至少", "不足"),
    ("以上", "以下"),
    ("不得", "可以"),
    ("无需", "需要"),
    ("不需要", "需要"),
    ("增加", "减少"),
    ("提高", "降低"),
    ("上升", "下降"),
    ("减少", "增加"),
    ("降低", "提高"),
    ("下降", "上升"),
    ("包括", "不包括"),
    ("仅", "还包括"),
    ("全部", "部分"),
    ("所有", "部分"),
    ("均", "存在例外"),
    ("取较大值", "取较小值"),
    ("较大值", "较小值"),
    ("较大者", "较小者"),
)

_FALLBACK_STOPWORDS = {
    "根据",
    "查阅",
    "关于",
    "结合",
    "以下",
    "哪些",
    "下列",
    "说法",
    "正确",
    "错误",
    "答案",
    "格式",
    "计算题",
    "判断题",
    "多选题",
    "单选题",
    "分别",
    "多少",
    "进行",
    "使用",
    "其中",
    "以及",
    "可以",
    "是否",
}

_ANCHOR_NOISE_FRAGMENTS = (
    "计算",
    "身故",
    "被保险",
    "已开始",
    "累计",
    "所交",
    "已给付",
    "领取养老",
    "基本保险",
    "责任免除",
    "营业收入",
    "净利润",
    "现金价值",
    "账户价值",
    "条款规定",
)


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Answer-blind input for retrieval-query generation.

    The interface intentionally has no qid, label, expected answer, document id,
    evidence id, or frozen-answer field.
    """

    domain: str
    question: str
    option_text: str = ""
    question_type: str = ""
    answer_format: str = ""
    document_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")


@dataclass(frozen=True, slots=True)
class SemanticSlots:
    anchors: tuple[str, ...]
    topics: tuple[str, ...]
    periods: tuple[str, ...]
    quantities: tuple[str, ...]
    relations: tuple[str, ...]
    scopes: tuple[str, ...]
    exceptions: tuple[str, ...]
    articles: tuple[str, ...]
    atoms: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {key: list(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class QueryVariant:
    channel: str
    query: str
    rationale: str
    atom_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    version: str
    domain: str
    slots: SemanticSlots
    variants: tuple[QueryVariant, ...]
    requires_scope_check: bool

    def query_strings(self) -> list[str]:
        return [variant.query for variant in self.variants]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "domain": self.domain,
            "slots": self.slots.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
            "requires_scope_check": self.requires_scope_check,
        }


def generate_retrieval_plan(
    request: RetrievalRequest,
    *,
    max_queries: int = 12,
) -> RetrievalPlan:
    """Build support, broad, contrast, and coverage queries from text alone."""

    if max_queries < 1:
        raise ValueError("max_queries must be positive")
    question = normalize_whitespace(request.question)
    option_text = normalize_whitespace(request.option_text)
    combined = "\n".join(part for part in (question, option_text) if part)
    atoms = _statement_atoms(option_text or question)
    anchors = _extract_anchors(combined, request.document_hints, request.domain)
    periods = _extract_periods(combined)
    quantities = tuple(
        item for item in _unique(_compact(item) for item in _QUANTITY_RE.findall(combined))
        if item and item not in periods and not re.fullmatch(r"20\d{2}", item)
    )
    topics = _extract_topics(combined, request.domain, anchors, periods)
    relations = tuple(term for term in _RELATION_TERMS if term.lower() in combined.lower())
    scopes = tuple(term for term in _SCOPE_TERMS if _contains_scope_term(combined, term))
    exceptions = tuple(term for term in _EXCEPTION_TERMS if term in combined)
    articles = tuple(_unique(_ARTICLE_RE.findall(combined)))
    slots = SemanticSlots(
        anchors=anchors,
        topics=topics,
        periods=periods,
        quantities=quantities,
        relations=relations,
        scopes=scopes,
        exceptions=exceptions,
        articles=articles,
        atoms=atoms,
    )
    requires_scope_check = any(term in combined for term in _ABSENCE_TERMS)
    variants: list[QueryVariant] = []

    primary_parts = [request.question_type, question, option_text]
    _append_variant(
        variants,
        channel="primary",
        query=_compose(primary_parts),
        rationale="题面与当前待核验命题的无损首轮查询",
    )

    for atom_index, atom in enumerate(atoms):
        _append_variant(
            variants,
            channel="support",
            query=_compose([*anchors, *articles, atom]),
            rationale="保留命题结论值，召回直接支持或直接否定该命题的原文",
            atom_index=atom_index,
        )

    broad_query = _compose(
        [
            *anchors,
            *articles,
            *periods,
            *topics,
            *relations,
            *scopes,
            *exceptions,
        ]
    )
    _append_variant(
        variants,
        channel="broad",
        query=broad_query or question,
        rationale="去除候选结论数值，检索材料实际披露的指标、条款和适用范围",
    )

    if requires_scope_check:
        _append_variant(
            variants,
            channel="scope_check",
            query=_compose([*anchors, *topics, "完整条款 适用范围 责任免除 例外"]),
            rationale="否定存在性命题必须在限定文档或章节内完成覆盖检查",
        )

    if request.answer_format == "calculation" or "计算题" in request.question_type:
        for anchor in _coverage_anchors(anchors)[:4] or ("",):
            _append_variant(
                variants,
                channel="coverage",
                query=_compose(
                    [
                        anchor,
                        *periods,
                        *topics,
                        *relations,
                        *articles,
                        "原始数值 公式 单位 表格行 条款",
                    ]
                ),
                rationale="按实体覆盖计算所需期间、指标、公式和单位，不等待模型报缺值",
            )

    for atom_index, atom in enumerate(atoms):
        contrast = _contrast_statement(atom)
        if contrast and contrast != atom:
            _append_variant(
                variants,
                channel="contrast",
                query=_compose([*anchors, *periods, contrast, *exceptions]),
                rationale="用通用范围、极性或方向反转寻找反证和例外",
                atom_index=atom_index,
            )

    return RetrievalPlan(
        version=QUERY_PLAN_VERSION,
        domain=request.domain,
        slots=slots,
        variants=tuple(variants[:max_queries]),
        requires_scope_check=requires_scope_check,
    )


def _extract_anchors(
    text: str,
    document_hints: Iterable[str],
    domain: str,
) -> tuple[str, ...]:
    anchors: list[str] = []
    anchors.extend(_DOCUMENT_TITLE_RE.findall(text))
    if domain == "financial_reports":
        for match in _COORDINATED_REPORT_SUBJECT_RE.finditer(text):
            subject_text = _strip_leading_noise(match.group(1))
            for subject in re.split(r"[、,，和及与]+", subject_text):
                cleaned = _strip_leading_noise(subject)
                if _valid_anchor(cleaned):
                    anchors.append(cleaned)
    for segment in re.split(r"[\n，。；;、：:\s/]+", text):
        for match in _ORG_SUFFIX_RE.findall(segment):
            cleaned = _strip_leading_noise(match)
            if _valid_anchor(cleaned):
                anchors.append(cleaned)
    if domain == "insurance":
        for match in _INSURANCE_SUBJECT_RE.findall(text):
            cleaned = _strip_leading_noise(match)
            if _valid_anchor(cleaned):
                anchors.append(cleaned)
    anchors.extend(hint.strip() for hint in document_hints if hint and hint.strip())
    return tuple(_unique(anchors))


def _coverage_anchors(anchors: tuple[str, ...]) -> tuple[str, ...]:
    document_markers = ("报告书", "年度报告", "募集说明书", "保险条款", "管理办法", "管理规定")
    entities = [
        anchor
        for anchor in anchors
        if not (
            any(marker in anchor for marker in document_markers)
            and any(other != anchor and other in anchor for other in anchors)
        )
    ]
    return tuple(entities or anchors)


def _extract_periods(text: str) -> tuple[str, ...]:
    matches: list[tuple[int, str]] = []
    for pattern in (_YEAR_RANGE_RE, _PERIOD_RE):
        matches.extend((match.start(), _compact(match.group(0))) for match in pattern.finditer(text))
    matches.sort(key=lambda item: item[0])
    return tuple(_unique(value for _, value in matches if value))


def _extract_topics(
    text: str,
    domain: str,
    anchors: tuple[str, ...],
    periods: tuple[str, ...],
) -> tuple[str, ...]:
    lower = text.lower()
    terms = [*_GENERAL_TOPIC_TERMS, *_DOMAIN_TOPIC_TERMS.get(domain, ())]
    matched = [term for term in terms if term.lower() in lower]
    blocked = " ".join([*anchors, *periods, *matched]).lower()
    fallback: list[str] = []
    for token in tokenize_zh(text):
        cleaned = token.strip()
        subject_form = cleaned[:-1] if cleaned.endswith(("中", "在")) else cleaned
        if (
            len(cleaned) < 2
            or cleaned in _FALLBACK_STOPWORDS
            or cleaned.lower() in blocked
            or (subject_form and subject_form.lower() in blocked)
            or re.fullmatch(r"[\d.,%％_-]+", cleaned)
            or _QUANTITY_RE.fullmatch(cleaned)
        ):
            continue
        fallback.append(cleaned)
        if len(fallback) >= 3:
            break
    return tuple(_unique([*matched, *fallback]))


def _statement_atoms(text: str) -> tuple[str, ...]:
    normalized = normalize_whitespace(text)
    pieces = re.split(r"(?:[；;。]|，(?:且|并且|同时|但|而)|(?:并且|同时))", normalized)
    atoms = [_compact(piece.strip(" ，。；;：:")) for piece in pieces]
    return tuple(_unique(atom for atom in atoms if len(atom) >= 4)[:6]) or (normalized,)


def _contrast_statement(text: str) -> str:
    for source, target in _CONTRAST_REPLACEMENTS:
        if _contains_scope_term(text, source):
            return text.replace(source, target, 1)
    return ""


def _compose(parts: Iterable[str]) -> str:
    return "\n".join(_unique(_compact(part) for part in parts if part and _compact(part)))


def _append_variant(
    variants: list[QueryVariant],
    *,
    channel: str,
    query: str,
    rationale: str,
    atom_index: int | None = None,
) -> None:
    cleaned = normalize_whitespace(query)
    if not cleaned or any(item.query == cleaned for item in variants):
        return
    variants.append(
        QueryVariant(
            channel=channel,
            query=cleaned,
            rationale=rationale,
            atom_index=atom_index,
        )
    )


def _strip_leading_noise(text: str) -> str:
    cleaned = text.strip("《》“”\"'（）() ")
    changed = True
    while changed:
        changed = False
        for prefix in _LEADING_NOISE:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                changed = True
    return cleaned


def _valid_anchor(text: str) -> bool:
    return (
        len(text) >= 2
        and text not in _FALLBACK_STOPWORDS
        and text not in {"保险", "养老保险", "身故保险", "证券"}
        and not any(fragment in text for fragment in _ANCHOR_NOISE_FRAGMENTS)
    )


def _contains_scope_term(text: str, term: str) -> bool:
    if term == "均":
        return "均" in text.replace("平均", "")
    return term in text


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _unique(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = item.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result
