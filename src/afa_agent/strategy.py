from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from afa_agent.io_utils import read_json


DEFAULT_STAGE_SETTINGS: dict[str, dict[str, Any]] = {
    "pdf_parse": {
        "pdf_backend": "pypdf",
        "keep_page_markers": True,
        "normalize_whitespace": True,
        "drop_short_lines": False,
    },
    "segmentation": {
        "paragraph_max_chars": 1000,
        "article_max_chars": 1200,
        "metric_max_chars": 600,
        "element_max_chars": 700,
        "clause_max_chars": 900,
        "duplicate_highlight_units": True,
    },
    "retrieval": {
        "query_mode": "question_option",
        "include_question_type": True,
        "include_doc_id_hint": False,
        "top_k": 6,
        "max_hits_for_prompt": 6,
        "ensure_per_doc": True,
        "expand_neighbors": True,
        "unit_type_boosts": {},
        "query_suffix": "",
    },
    "rule_layer": {
        "enabled": True,
        "high_confidence_threshold": 0.9,
        "prefer_rule_result": True,
    },
    "answering": {
        "prompt_template_id": "default",
        "fallback_template_id": "default",
        "max_hits": 6,
        "extra_context": "",
        "consistency_repeats": 1,
    },
    "evidence_gate": {
        "enabled": False,
        "max_rescue_rounds": 7,
        "rescue_top_k": 12,
        "per_doc_quota": 2,
        "min_hit_chars": 24,
        "max_hits_after_rescue": 12,
        "use_llm_coverage_check": False,
        "final_consistency_retry": True,
        "high_certainty_threshold": 0.7,
        "low_certainty_threshold": 0.45,
        "rescue_channels": [
            "query_rewrite_search",
            "title_search",
            "unit_type_search",
            "table_metric_search",
            "clause_formula_search",
            "per_doc_search",
            "neighbor_expansion",
        ],
    },
}


DEFAULT_STRATEGY_CONFIG: dict[str, Any] = {
    "version": "autoresearch_v1",
    "strategy_id": "baseline",
    "proxy_objective_version": "proxy_v1",
    "domains": {
        "__all__": deepcopy(DEFAULT_STAGE_SETTINGS),
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_strategy_path(explicit_path: str | Path | None = None) -> Path | None:
    if explicit_path:
        return Path(explicit_path)
    env_path = os.environ.get("AFA_STRATEGY_CONFIG")
    return Path(env_path) if env_path else None


def load_strategy_config(path: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_strategy_path(path)
    if not resolved:
        return deepcopy(DEFAULT_STRATEGY_CONFIG)
    payload = read_json(resolved)
    return deep_merge(DEFAULT_STRATEGY_CONFIG, payload)


def get_domain_strategy(domain: str, path: str | Path | None = None) -> dict[str, Any]:
    payload = load_strategy_config(path)
    shared = payload.get("domains", {}).get("__all__", {})
    domain_payload = payload.get("domains", {}).get(domain, {})
    return deep_merge(shared, domain_payload)


def get_stage_settings(domain: str, stage: str, path: str | Path | None = None) -> dict[str, Any]:
    strategy = get_domain_strategy(domain, path)
    stage_settings = strategy.get(stage, {})
    return deep_merge(DEFAULT_STAGE_SETTINGS.get(stage, {}), stage_settings)


def build_query_variants(question, option_key: str, option_text: str, retrieval_settings: dict[str, Any]) -> list[str]:
    mode = retrieval_settings.get("query_mode", "question_option")
    question_type = getattr(question, "type", "")
    question_text = getattr(question, "question", "")
    doc_hint = " ".join(getattr(question, "doc_ids", [])[:2]) if retrieval_settings.get("include_doc_id_hint") else ""
    suffix = retrieval_settings.get("query_suffix", "").strip()
    variants: list[str] = []

    if mode == "question_only":
        variants.append(question_text)
    elif mode == "type_question_option":
        variants.append(f"{question_type}\n{question_text}\n{option_text}".strip())
    elif mode == "question_option_doc":
        variants.append(f"{question_text}\n{option_text}\n{doc_hint}".strip())
    else:
        variants.append(f"{question_text}\n{option_text}".strip())

    if retrieval_settings.get("include_question_type") and mode != "type_question_option":
        variants.append(f"{question_type}\n{question_text}\n{option_text}".strip())
    variants.append(question_text.strip())
    if doc_hint:
        variants.append(f"{question_text}\n{doc_hint}".strip())
    if suffix:
        variants = [f"{variant}\n{suffix}".strip() for variant in variants]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped or [question_text.strip()]


def serialize_hits(hits: list[Any], limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in hits[:limit] if limit else hits:
        rows.append(
            {
                "unit_id": hit.unit_id,
                "doc_id": hit.doc_id,
                "score": hit.score,
                "title_path": hit.title_path,
                "unit_type": hit.metadata.get("unit_type") or hit.metadata.get("source_unit_type", ""),
                "text_preview": hit.text[:240],
                "metadata": hit.metadata,
            }
        )
    return rows
