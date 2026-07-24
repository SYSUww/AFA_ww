from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from afa_agent.b_board.io import BAnswer, BQuestion, validate_b_answer
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.regulatory.retriever import RegulatoryRetriever
from afa_agent.models import RetrievalHit
from afa_agent.retrieval_query import (
    RetrievalPlan,
    RetrievalRequest,
    generate_retrieval_plan,
)


PIPELINE_VERSION = "b_retrieval_llm_answer_blind_v2"
PROMPT_VERSION = "b_retrieval_llm_modular_final_submission_v5"
RETRIEVAL_POLICY_VERSION = "semantic_slots_entity_coverage_option_quota_v3"

UNIT_TYPE_BOOSTS: dict[str, dict[str, float]] = {
    "financial_contracts": {"element_block": 1.8, "paragraph": 1.0},
    "financial_reports": {"metric_row": 1.8, "paragraph": 1.0},
    "insurance": {"formula_block": 1.8, "clause_block": 1.1},
    "research": {"conclusion_block": 1.6, "paragraph": 1.0},
    "regulatory": {},
}


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    anchor_queries: tuple[str, ...]
    document_queries: tuple[str, ...]
    primary_queries: tuple[str, ...]
    supplemental_queries: tuple[str, ...]
    option_queries: tuple[tuple[str, ...], ...]
    plans: tuple[RetrievalPlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_queries": list(self.anchor_queries),
            "document_queries": list(self.document_queries),
            "primary_queries": list(self.primary_queries),
            "supplemental_queries": list(self.supplemental_queries),
            "option_queries": [list(items) for items in self.option_queries],
            "plans": [plan.to_dict() for plan in self.plans],
        }


def load_domain_indexes(index_root: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for domain in UNIT_TYPE_BOOSTS:
        path = index_root / domain / "index.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        units = payload.get("units")
        if not isinstance(units, list) or not units:
            raise ValueError(f"{path}: index must contain a non-empty units array")
        payloads[domain] = payload
    return payloads


def make_retriever(domain: str, units: list[dict[str, Any]]) -> Any:
    if domain == "regulatory":
        return RegulatoryRetriever(units)
    return GenericBM25Retriever(units)


def build_retrieval_bundle(
    question: BQuestion,
    *,
    max_queries_per_option: int = 12,
) -> RetrievalBundle:
    requests = _retrieval_requests(question)
    plans = tuple(
        generate_retrieval_plan(request, max_queries=max_queries_per_option)
        for request in requests
    )
    primary = _dedupe(
        variant.query
        for plan in plans
        for variant in plan.variants
        if variant.channel == "primary"
    )
    allowed_supplemental = (
        {"coverage"}
        if question.answer_format == "calculation"
        else {"support", "contrast", "scope_check"}
    )
    supplemental = _dedupe(
        variant.query
        for plan in plans
        for variant in plan.variants
        if variant.channel in allowed_supplemental
    )
    document_queries = _dedupe(
        variant.query
        for plan in plans
        for variant in plan.variants
        if variant.channel in {"primary", "broad"}
    )
    anchor_queries = _document_anchor_queries(question, plans)
    option_queries = tuple(_focused_plan_queries(plan) for plan in plans)
    return RetrievalBundle(
        anchor_queries=tuple(anchor_queries),
        document_queries=tuple(document_queries),
        primary_queries=tuple(primary),
        supplemental_queries=tuple(supplemental),
        option_queries=option_queries,
        plans=plans,
    )


def retrieve_question_evidence(
    question: BQuestion,
    *,
    retriever: Any,
    all_doc_ids: Sequence[str],
    per_query_top_k: int = 10,
    final_top_k: int = 10,
    supplemental_weight: float = 0.11,
    max_queries_per_option: int = 12,
    max_doc_candidates: int = 6,
) -> dict[str, Any]:
    if not all_doc_ids:
        raise ValueError(f"{question.domain}: corpus has no document ids")
    bundle = build_retrieval_bundle(
        question,
        max_queries_per_option=max_queries_per_option,
    )
    boosts = UNIT_TYPE_BOOSTS.get(question.domain, {})
    discovery = _rank_queries(
        retriever,
        list(all_doc_ids),
        list(bundle.document_queries),
        per_query_top_k=max(per_query_top_k, 20),
        final_top_k=max(final_top_k * 5, 50),
        unit_type_boosts=boosts,
        expand_neighbors=False,
    )
    anchored_doc_ids = _anchor_doc_candidates(
        retriever,
        list(all_doc_ids),
        list(bundle.anchor_queries),
        per_query_top_k=max(per_query_top_k, 10),
        unit_type_boosts=boosts,
    )
    selected_doc_ids = (
        anchored_doc_ids[:max_doc_candidates]
        if anchored_doc_ids
        else _ranked_doc_ids(discovery)[:max_doc_candidates]
    )
    if not selected_doc_ids:
        raise ValueError("document discovery produced no candidates")
    primary = _rank_queries(
        retriever,
        selected_doc_ids,
        list(bundle.primary_queries),
        per_query_top_k=per_query_top_k,
        final_top_k=final_top_k,
        unit_type_boosts=boosts,
        expand_neighbors=True,
    )
    supplemental = _rank_queries(
        retriever,
        selected_doc_ids,
        list(bundle.supplemental_queries),
        per_query_top_k=per_query_top_k,
        final_top_k=max(final_top_k * 3, final_top_k),
        unit_type_boosts=boosts,
        expand_neighbors=True,
    )
    document_evidence_query = _document_evidence_query(bundle.plans)
    document_rankings = [
        _rank_queries(
            retriever,
            [doc_id],
            [document_evidence_query],
            per_query_top_k=per_query_top_k,
            final_top_k=max(2, min(final_top_k, 4)),
            unit_type_boosts=boosts,
            expand_neighbors=True,
        )
        for doc_id in selected_doc_ids
        if document_evidence_query
    ]
    option_rankings = [
        _rank_queries(
            retriever,
            selected_doc_ids,
            list(queries),
            per_query_top_k=per_query_top_k,
            final_top_k=max(3, min(final_top_k, 5)),
            unit_type_boosts=boosts,
            expand_neighbors=True,
        )
        for queries in bundle.option_queries
        if queries
    ]
    blended = _blend_rankings(
        primary,
        supplemental,
        document_rankings=document_rankings,
        option_rankings=option_rankings,
        supplemental_weight=supplemental_weight,
        final_top_k=final_top_k,
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "policy_version": RETRIEVAL_POLICY_VERSION,
        "corpus_doc_count": len(set(all_doc_ids)),
        "selected_doc_ids": selected_doc_ids,
        "anchored_doc_ids": anchored_doc_ids,
        "bundle": bundle.to_dict(),
        "document_discovery": _ranking_to_dict(discovery),
        "primary": _ranking_to_dict(primary),
        "supplemental": _ranking_to_dict(supplemental),
        "document_rankings": [
            _ranking_to_dict(ranking) for ranking in document_rankings
        ],
        "option_rankings": [
            _ranking_to_dict(ranking) for ranking in option_rankings
        ],
        "final": _ranking_to_dict(blended),
    }


def _document_anchor_queries(
    question: BQuestion,
    plans: Sequence[RetrievalPlan],
) -> list[str]:
    if not plans:
        return []
    anchors = list(plans[0].slots.anchors)
    periods = list(plans[0].slots.periods)
    document_marker = next(
        (
            marker
            for marker in (
                "年度报告",
                "年报",
                "募集说明书",
                "报告书",
                "保险条款",
                "管理办法",
                "管理规定",
            )
            if marker in question.question
        ),
        "",
    )
    period = periods[0] if periods else ""
    return _dedupe(
        " ".join(part for part in (anchor, period, document_marker) if part)
        for anchor in anchors
    )


def _focused_plan_queries(plan: RetrievalPlan) -> tuple[str, ...]:
    slots = plan.slots
    if slots.atoms:
        return tuple(_dedupe(slots.atoms))
    slot_query = " ".join(
        [
            *slots.articles,
            *slots.periods,
            *slots.topics,
            *slots.quantities,
            *slots.relations,
            *slots.scopes,
            *slots.exceptions,
        ]
    )
    return tuple(_dedupe([slot_query]))


def _document_evidence_query(plans: Sequence[RetrievalPlan]) -> str:
    if not plans:
        return ""
    slots = plans[0].slots
    return " ".join(
        _dedupe(
            [
                *slots.articles,
                *slots.periods,
                *slots.topics,
                *slots.quantities,
                *slots.relations,
                *slots.scopes,
                *slots.exceptions,
            ]
        )
    )


def _retrieval_requests(question: BQuestion) -> list[RetrievalRequest]:
    if question.answer_format in {"calculation", "extraction"}:
        return [
            RetrievalRequest(
                domain=question.domain,
                question=question.question,
                question_type=question.type,
                answer_format=question.answer_format,
            )
        ]
    return [
        RetrievalRequest(
            domain=question.domain,
            question=question.question,
            option_text=option_text,
            question_type=question.type,
            answer_format=question.answer_format,
        )
        for option_text in question.options.values()
    ]


def _rank_queries(
    retriever: Any,
    doc_ids: list[str],
    queries: list[str],
    *,
    per_query_top_k: int,
    final_top_k: int,
    unit_type_boosts: Mapping[str, float],
    expand_neighbors: bool,
) -> dict[str, Any]:
    scores: defaultdict[str, float] = defaultdict(float)
    hit_by_id: dict[str, RetrievalHit] = {}
    for query in queries:
        kwargs = {
            "top_k": per_query_top_k,
            "ensure_per_doc": False,
            "expand_neighbors": expand_neighbors,
        }
        try:
            hits = retriever.search(
                doc_ids,
                query,
                unit_type_boosts=dict(unit_type_boosts),
                **kwargs,
            )
        except TypeError:
            hits = retriever.search(doc_ids, query, **kwargs)
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            unit_id = _normalize_unit_id(hit.unit_id)
            if unit_id in seen:
                continue
            seen.add(unit_id)
            scores[unit_id] += 1.0 / (60 + rank)
            current = hit_by_id.get(unit_id)
            if current is None or hit.score > current.score:
                hit_by_id[unit_id] = hit
    ranked_ids = sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id))
    ranked_ids = ranked_ids[:final_top_k]
    return {
        "queries": list(queries),
        "ranked_ids": ranked_ids,
        "ranked_hits": [hit_by_id[unit_id] for unit_id in ranked_ids],
    }


def _ranked_doc_ids(ranking: Mapping[str, Any]) -> list[str]:
    return _dedupe(hit.doc_id for hit in ranking["ranked_hits"])


def _anchor_doc_candidates(
    retriever: Any,
    doc_ids: list[str],
    anchor_queries: list[str],
    *,
    per_query_top_k: int,
    unit_type_boosts: Mapping[str, float],
) -> list[str]:
    selected: list[str] = []
    for query in anchor_queries:
        kwargs = {
            "top_k": per_query_top_k,
            "ensure_per_doc": False,
            "expand_neighbors": False,
        }
        try:
            hits = retriever.search(
                doc_ids,
                query,
                unit_type_boosts=dict(unit_type_boosts),
                **kwargs,
            )
        except TypeError:
            hits = retriever.search(doc_ids, query, **kwargs)
        if hits:
            selected.append(str(hits[0].doc_id))
    return _dedupe(selected)


def _blend_rankings(
    primary: Mapping[str, Any],
    supplemental: Mapping[str, Any],
    *,
    document_rankings: Sequence[Mapping[str, Any]],
    option_rankings: Sequence[Mapping[str, Any]],
    supplemental_weight: float,
    final_top_k: int,
) -> dict[str, Any]:
    if supplemental_weight < 0:
        raise ValueError("supplemental_weight must not be negative")
    scores: defaultdict[str, float] = defaultdict(float)
    for rank, unit_id in enumerate(primary["ranked_ids"], start=1):
        scores[str(unit_id)] += 1.0 / rank
    for rank, unit_id in enumerate(supplemental["ranked_ids"], start=1):
        scores[str(unit_id)] += supplemental_weight / rank
    hit_by_id = {
        _normalize_unit_id(hit.unit_id): hit
        for hit in [
            *primary["ranked_hits"],
            *supplemental["ranked_hits"],
            *(
                hit
                for ranking in document_rankings
                for hit in ranking["ranked_hits"]
            ),
            *(
                hit
                for ranking in option_rankings
                for hit in ranking["ranked_hits"]
            ),
        ]
    }
    reserved_ids = _dedupe(
        unit_id
        for ranking in document_rankings
        for unit_id in ranking["ranked_ids"][:2]
    )
    reserved_ids = _dedupe(
        [
            *reserved_ids,
            *(
                unit_id
                for ranking in option_rankings
                for unit_id in ranking["ranked_ids"][:1]
            ),
        ]
    )
    for ranking in document_rankings:
        for rank, unit_id in enumerate(ranking["ranked_ids"], start=1):
            scores[str(unit_id)] += 0.5 / rank
    for ranking in option_rankings:
        for rank, unit_id in enumerate(ranking["ranked_ids"], start=1):
            scores[str(unit_id)] += 0.5 / rank
    ranked_ids = [
        *reserved_ids[:final_top_k],
        *(
            unit_id
            for unit_id in sorted(
                scores,
                key=lambda unit_id: (-scores[unit_id], unit_id),
            )
            if unit_id not in reserved_ids
        ),
    ][:final_top_k]
    return {
        "queries": [
            *primary["queries"],
            *supplemental["queries"],
            *(
                query
                for ranking in document_rankings
                for query in ranking["queries"]
            ),
            *(
                query
                for ranking in option_rankings
                for query in ranking["queries"]
            ),
        ],
        "ranked_ids": ranked_ids,
        "ranked_hits": [hit_by_id[unit_id] for unit_id in ranked_ids],
    }


def _ranking_to_dict(ranking: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queries": list(ranking["queries"]),
        "ranked_ids": list(ranking["ranked_ids"]),
        "hits": [
            {
                **hit.to_dict(),
                "unit_id": _normalize_unit_id(hit.unit_id),
            }
            for hit in ranking["ranked_hits"]
        ],
    }


def prepare_evidence_payload(
    retrieval: Mapping[str, Any],
    *,
    max_hit_chars: int = 1800,
    max_total_chars: int = 12000,
) -> list[dict[str, Any]]:
    if max_hit_chars < 1 or max_total_chars < 1:
        raise ValueError("evidence character limits must be positive")
    evidence: list[dict[str, Any]] = []
    used_chars = 0
    source_keys: dict[str, str] = {}
    for rank, hit in enumerate(retrieval["final"]["hits"], start=1):
        remaining = max_total_chars - used_chars
        if remaining <= 0:
            break
        raw_text = str(hit.get("text", "")).strip()
        text = raw_text[: min(max_hit_chars, remaining)]
        if not text:
            continue
        doc_id = str(hit["doc_id"])
        source_key = source_keys.setdefault(doc_id, f"S{len(source_keys) + 1:02d}")
        evidence.append(
            {
                "evidence_key": f"E{len(evidence) + 1:02d}",
                "source_key": source_key,
                "evidence_id": str(hit["unit_id"]),
                "doc_id": doc_id,
                "rank": rank,
                "unit_type": str((hit.get("metadata") or {}).get("unit_type", "")),
                "title_path": [str(item) for item in hit.get("title_path", [])],
                "text": text,
                "prompt_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "source_text_sha256": hashlib.sha256(
                    raw_text.encode("utf-8")
                ).hexdigest(),
                "truncated": len(text) < len(raw_text),
            }
        )
        used_chars += len(text)
    if not evidence:
        raise ValueError("retrieval produced no usable evidence")
    return evidence


def build_answer_schema(
    question: BQuestion,
) -> dict[str, Any]:
    option_labels = list(question.options)
    if question.answer_format in {"tf", "mcq"}:
        answer_item_schema: dict[str, Any] = {
            "type": "string",
            "enum": option_labels,
        }
    elif question.answer_format == "multi":
        legal_answers = [
            "".join(items)
            for size in range(2, len(option_labels) + 1)
            for items in combinations(option_labels, size)
        ]
        answer_item_schema = {
            "type": "string",
            "enum": legal_answers,
        }
    else:
        answer_item_schema = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Let the model complete its evidence-backed audit summary before
            # committing the final structured answer.
            "reasoning": {"type": "string", "minLength": 20},
            "answer_parts": {
                "type": "array",
                "items": answer_item_schema,
                "minItems": question.answer_slots,
                "maxItems": question.answer_slots,
            },
        },
        "required": ["reasoning", "answer_parts"],
    }


def build_answer_messages(
    question: BQuestion,
    evidence: Sequence[Mapping[str, Any]],
    *,
    validation_error: str = "",
    previous_response: str = "",
) -> list[dict[str, str]]:
    options_text = (
        "\n".join(f"{key}. {value}" for key, value in question.options.items())
        if question.options
        else "（无选项）"
    )
    source_lines: list[str] = []
    seen_sources: set[str] = set()
    for item in evidence:
        source_key = str(item["source_key"])
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        source_lines.append(f"[{source_key}] {item['doc_id']}")
    evidence_text = "\n\n".join(
        (
            f"[{item['evidence_key']}|{item['source_key']}] "
            f"title={_prompt_title(item.get('title_path', []))}\n{item['text']}"
        )
        for item in evidence
    )
    format_instruction = _format_instruction(question)
    slot_text = (
        f"提交槽数量：{question.answer_slots}\n"
        f"槽位格式模板：{list(question.answer_slot_templates)}\n"
        if question.answer_format not in {"tf", "mcq", "multi"}
        else ""
    )
    messages = [
        {
            "role": "system",
            "content": _system_prompt(question),
        },
        {
            "role": "user",
            "content": (
                f"领域：{question.domain}\n"
                f"题型：{question.type}（{question.answer_format}）\n"
                f"题目：{question.question}\n"
                f"选项：\n{options_text}\n\n"
                f"{slot_text}"
                f"格式要求：{format_instruction}\n\n"
                f"来源：\n{chr(10).join(source_lines)}\n\n"
                f"检索证据：\n{evidence_text}\n\n"
                "请一次性输出符合JSON Schema、可直接提交的最终answer_parts和最终reasoning。"
                "reasoning不得只引用E01等编号，必须复述支撑答案的关键事实。"
                "输出后不会由代码或其他模型补写、改写或总结。"
            ),
        },
    ]
    if validation_error:
        messages.extend(
            [
                {"role": "assistant", "content": previous_response},
                {
                    "role": "user",
                    "content": _retry_instruction(validation_error),
                },
            ]
        )
    serialized = json.dumps(messages, ensure_ascii=False)
    if question.qid in serialized:
        raise ValueError("qid leaked into model messages")
    return messages


def validate_answer_payload(
    question: BQuestion,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    required = {"answer_parts", "reasoning"}
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    if missing or extra:
        raise ValueError(f"response fields mismatch: missing={missing}, extra={extra}")

    raw_parts = payload["answer_parts"]
    if not isinstance(raw_parts, list):
        raise ValueError("answer_parts must be an array")
    parts = tuple(str(item).strip() for item in raw_parts)
    if any(not part for part in parts):
        raise ValueError("answer_parts must contain non-empty strings")

    if not isinstance(payload["reasoning"], str):
        raise ValueError("reasoning must be a string")
    reasoning = payload["reasoning"]
    if len(reasoning.strip()) < 20:
        raise ValueError("reasoning must contain at least 20 characters")
    answer = BAnswer(
        qid=question.qid,
        answer_parts=parts,
        reasoning=reasoning,
    )
    validate_b_answer(question, answer)
    _validate_reasoning_conclusion(parts, reasoning)
    return {
        "answer_parts": list(parts),
        "reasoning": reasoning,
    }


def public_run_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _system_prompt(question: BQuestion) -> str:
    common = (
        "你是金融材料问答模型。只能依据给定题目与检索证据作答，不得使用题号、"
        "历史答案、隐藏标签或外部知识。输出将直接提交：只返回JSON Schema规定的"
        "answer_parts与reasoning，禁止附加字段。reasoning也是最终提交文本，会在"
        "看不到题目、证据和答案的情况下被单独评审，因此必须是简洁、自包含、可审计"
        "的结论摘要：明确主体、年份或适用范围，复述关键事实，说明必要的判断或计算，"
        "并以“结论：...”结束；不得只引用E01等证据编号，也不要输出冗长的隐藏思维过程。"
        "reasoning中的最终结论必须机械复制answer_parts：单槽写“结论：<槽1>”，多槽按顺序"
        "写“结论：<槽1>；<槽2>”；不得用“正确/错误”、单位或解释替代、增补槽内文本。"
    )
    if question.answer_format in {"tf", "mcq", "multi"}:
        return (
            common
            + "这是选择类问题。应在内部逐项核验主体、时间、指标、范围、否定词和例外条件，"
            "但不要输出逐项布尔标签或中间结构。判断题仅选A或B；单选题仅选一个字母；"
            "多选题必须选择至少两个不同字母并按升序连接。"
        )
    if question.answer_format == "calculation":
        return (
            common
            + "这是计算题。先核对主体、年份、指标口径、单位和取值范围；再写出必要公式、"
            "关键代入值与舍入结果。特别检查汇总值是否已包含分项，避免重复计算，并正确处理"
            "百分比、最大最小值、差额、累计值及排序。严格按槽位格式模板保留小数位、百分号"
            "及分隔符，末尾结论复制格式化后的槽位。"
        )
    return (
        common
        + "这是信息抽取题。核对主体、年份、指标、单位和输出顺序，直接复述证据支持的结果，"
        "严格按题目要求返回全部槽位。"
    )


def _prompt_title(title_path: Sequence[Any]) -> str:
    parts = [str(item).strip() for item in title_path if str(item).strip()]
    if not parts:
        return "无"
    return " / ".join(part[:120] for part in parts[-2:])


def _validate_reasoning_conclusion(
    answer_parts: Sequence[str],
    reasoning: str,
) -> None:
    marker_index = max(reasoning.rfind("结论："), reasoning.rfind("结论:"))
    if marker_index < 0:
        raise ValueError("reasoning must end with an explicit 结论")
    conclusion = reasoning[marker_index + 3 :].strip()
    if not conclusion:
        raise ValueError("reasoning conclusion must not be empty")

    def normalize(value: str) -> str:
        cleaned = re.sub(r"[。.!！]+$", "", str(value).strip())
        return re.sub(r"[\s、,，;；|]+", "", cleaned).upper()

    normalized_conclusion = normalize(conclusion)
    normalized_parts = [normalize(part) for part in answer_parts]
    if normalized_conclusion != "".join(normalized_parts):
        raise ValueError("reasoning conclusion does not exactly match answer_parts")


def _format_instruction(question: BQuestion) -> str:
    if question.answer_format == "tf":
        return "判断题只填一个字母：A=正确，B=错误。"
    if question.answer_format == "mcq":
        return "单选题只填一个选项字母。"
    if question.answer_format == "multi":
        return "多选题在一个槽内填写至少两个不同字母，按字母升序连接。"
    return (
        f"严格返回{question.answer_slots}个槽；数值精度、百分号、日期和排序分隔符"
        "完全遵循题面及槽位模板。"
    )


def _retry_instruction(validation_error: str) -> str:
    repairs = {
        "reasoning_missing_explicit_conclusion": (
            "reasoning末尾缺少机械结论。请在完整reasoning末尾增加“结论：”，"
            "随后逐字复制answer_parts各槽；多槽用中文分号连接。"
        ),
        "answer_reasoning_conclusion_mismatch": (
            "answer_parts与reasoning末尾结论不一致。重新核验内容后让两者完全相同；"
            "判断题结论必须写A或B，不得写“正确/错误”；结论不得增加单位或解释。"
        ),
        "answer_slot_format_error": (
            "answer_parts未遵循题面和槽位模板。保持原计算语义，修正小数位、百分号、"
            "日期、排序或分隔符；reasoning末尾结论必须逐字复制修正后的槽位。"
        ),
        "schema_fields_error": "只保留Schema规定的两个字段，并满足数组和字符串类型。",
        "json_parse_error": "输出必须是一个可解析且符合Schema的JSON对象。",
        "answer_shape_error": "重新核对题型、选项数量和槽位数量，并让结论逐字复制槽位。",
    }
    repair = repairs.get(validation_error, repairs["answer_shape_error"])
    return (
        "上一响应未通过确定性格式/一致性校验。只根据同一题目和证据，重新输出一整份"
        "可直接提交的answer_parts与reasoning；不要增加证据中不存在的事实。"
        f"\n修复要求：{repair}\n错误码：{validation_error}"
    )


def _normalize_unit_id(unit_id: str) -> str:
    return str(unit_id).replace("__dup2", "").replace("__dup", "")


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = str(item).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
