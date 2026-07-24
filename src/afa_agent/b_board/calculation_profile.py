from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


CALCULATION_PROFILE_SCHEMA_VERSION = "calculation_profile_v1"
CALCULATION_THINKING_POLICY_VERSION = (
    "calculation_guarded_adaptive_thinking_v1"
)
CALCULATION_FIRST_ATTEMPT_EVIDENCE_POLICY_VERSION = (
    "calculation_self_contained_question_first_v1"
)

CALCULATION_TASK_TYPES = (
    "direct_extraction",
    "arithmetic",
    "ratio",
    "percentage_change",
    "percentage_point_change",
    "aggregation",
    "ranking",
    "date_calculation",
    "conditional_rule",
    "scenario_projection",
)

CALCULATION_OPERATORS = (
    "add",
    "sub",
    "mul",
    "div",
    "mean",
    "max",
    "min",
    "abs",
    "pct_change",
    "pct_point_delta",
    "count_gte",
    "count_gt",
    "sort_desc",
    "date_add_days",
    "next_workday",
    "days_between",
)

CALCULATION_RISK_CHECKS = (
    "period_binding",
    "entity_binding",
    "aggregate_scope",
    "unit_conversion",
    "table_row_binding",
    "ranking_coverage",
    "conditional_branch",
    "full_period_components",
    "raw_amount_ratio",
    "date_boundary",
)

CALCULATION_PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "task_types",
        "required_facts",
        "operators",
        "outputs",
        "complexity",
        "risk_checks",
    ],
    "properties": {
        "task_types": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "string",
                "enum": list(CALCULATION_TASK_TYPES),
            },
        },
        "required_facts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "subject",
                    "metric",
                    "period",
                    "unit",
                    "role",
                    "source_kind",
                ],
                "properties": {
                    "subject": {"type": "string"},
                    "metric": {"type": "string", "minLength": 1},
                    "period": {"type": "string"},
                    "unit": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": [
                            "input",
                            "condition",
                            "comparison",
                        ],
                    },
                    "source_kind": {
                        "type": "string",
                        "enum": [
                            "question_input",
                            "disclosed_metric",
                            "table_row",
                            "contract_clause",
                            "period_components",
                        ],
                    },
                },
            },
        },
        "operators": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "string",
                "enum": list(CALCULATION_OPERATORS),
            },
        },
        "outputs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "format"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "number",
                            "percentage",
                            "percentage_points",
                            "date",
                            "text",
                        ],
                    },
                    "format": {
                        "type": "string",
                        "enum": [
                            "decimal0",
                            "decimal1",
                            "decimal2",
                            "percent2",
                            "date",
                            "text",
                        ],
                    },
                },
            },
        },
        "complexity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "risk_checks": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "string",
                "enum": list(CALCULATION_RISK_CHECKS),
            },
        },
    },
}

_PROFILE_VALIDATOR = Draft202012Validator(CALCULATION_PROFILE_SCHEMA)

CALCULATION_PROFILE_SYSTEM_PROMPT = """你是计算任务的结构化预检器。
只分析题目要求，不求答案，不补充题目未提供的事实，不输出检索结果。
根据题目语义列出求解所需的事实、算子、输出格式、复杂度和通用风险检查。
required_facts 中的 subject、metric、period、unit 必须直接概括题目要求，
不得包含答案、推导结果、文档ID、题号或历史运行信息。
每个 required_fact 必须标记 source_kind：
question_input 表示题面直接假设；disclosed_metric 表示正文直接披露指标；
table_row 表示必须从指定表格同行取得；contract_clause 表示合同条件或规则；
period_components 表示目标期间需要合并多个组成部分。
risk_checks 必须覆盖 source_kind 所隐含的通用风险，不能只列计算算子。
只输出符合 JSON Schema 的对象。"""

_SOURCE_KIND_RISK_CHECKS = {
    "table_row": ("table_row_binding",),
    "contract_clause": ("conditional_branch",),
    "period_components": ("full_period_components",),
}

_RISK_GUIDANCE = {
    "period_binding": "每个原始变量必须同时绑定正确期间，禁止用相邻年度或时点代替。",
    "entity_binding": "每个原始变量必须绑定正确主体，不得跨公司、产品或合同借值。",
    "aggregate_scope": "直接披露的汇总值与单期值必须区分，禁止重复聚合。",
    "unit_conversion": "证据未逐字写出单位时unit留空；不同尺度运算必须显式换算。",
    "table_row_binding": "表格行标签、期间和数值必须来自同一证据行，不得把其他行数值改名。",
    "ranking_coverage": "排序前必须计算并覆盖题目要求的全部比较对象。",
    "conditional_branch": "条件规则必须逐分支执行，取大、取小、保底和差额不得改成相加。",
    "full_period_components": "全年或完整期间指标必须覆盖证据明确要求的全部期间组成部分。",
    "raw_amount_ratio": "要求使用原始金额时，必须用同期间原始分子分母重新计算比率。",
    "date_boundary": "日期和区间计算必须明确起止边界及自然日或工作日口径。",
}


@dataclass(frozen=True, slots=True)
class RequiredFact:
    subject: str
    metric: str
    period: str
    unit: str
    role: str
    source_kind: str

    def retrieval_query(self) -> str:
        return " ".join(
            value
            for value in (
                self.subject,
                self.metric,
                self.period,
                self.unit,
            )
            if value
        )


@dataclass(frozen=True, slots=True)
class ProfileOutput:
    kind: str
    format: str


@dataclass(frozen=True, slots=True)
class CalculationProfile:
    domain: str
    task_types: tuple[str, ...]
    required_facts: tuple[RequiredFact, ...]
    operators: tuple[str, ...]
    outputs: tuple[ProfileOutput, ...]
    complexity: str
    risk_checks: tuple[str, ...]
    version: str = CALCULATION_PROFILE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "task_types": list(self.task_types),
            "required_facts": [
                asdict(item) for item in self.required_facts
            ],
            "operators": list(self.operators),
            "outputs": [asdict(item) for item in self.outputs],
            "complexity": self.complexity,
            "risk_checks": list(self.risk_checks),
            "version": self.version,
        }

    def retrieval_queries(self) -> tuple[str, ...]:
        queries = dict.fromkeys(
            item.retrieval_query().strip()
            for item in self.required_facts
            if item.retrieval_query().strip()
        )
        return tuple(queries)


@dataclass(frozen=True, slots=True)
class CalculationThinkingPolicy:
    """Answer-blind provider thinking policy derived from task structure."""

    mode: str
    thinking_budget: int | None
    reasons: tuple[str, ...]
    version: str = CALCULATION_THINKING_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalculationFirstAttemptEvidencePolicy:
    """Answer-blind evidence scope for the first calculation request."""

    mode: str
    max_non_question_hits: int | None
    reasons: tuple[str, ...]
    version: str = CALCULATION_FIRST_ATTEMPT_EVIDENCE_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_calculation_first_attempt_evidence_policy(
    *,
    domain: str,
    question: str,
    answer_slots: int,
) -> CalculationFirstAttemptEvidencePolicy:
    """Avoid irrelevant documents only for dense, self-contained questions.

    Any later attempt restores the runner's normal progressive evidence
    expansion. The decision uses no qid, answer, document id, or prior run.
    """

    compact = re.sub(r"\s+", "", str(question))
    non_year_question = re.sub(r"20\d{2}年?", "", compact)
    numeric_literals = re.findall(
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?",
        non_year_question,
    )
    has_external_source_reference = any(
        marker in compact
        for marker in ("查阅", "根据", "报告", "合同", "材料", "文档")
    )
    if (
        domain == "research"
        and answer_slots == 1
        and len(numeric_literals) >= 4
        and not has_external_source_reference
    ):
        return CalculationFirstAttemptEvidencePolicy(
            mode="question_only_first_attempt",
            max_non_question_hits=0,
            reasons=(
                "single_slot_question_supplies_dense_numeric_inputs",
                "no_external_source_reference",
                "normal_retrieval_restored_on_retry",
            ),
        )
    return CalculationFirstAttemptEvidencePolicy(
        mode="progressive_retrieval",
        max_non_question_hits=None,
        reasons=("evidence_sensitive_or_not_self_contained",),
    )


def infer_calculation_thinking_policy(
    *,
    domain: str,
    question: str,
    answer_slots: int,
    semantic_constraint_types: Sequence[str] = (),
) -> CalculationThinkingPolicy:
    """Choose a conservative first-attempt thinking mode without answer data.

    The policy uses only domain and structural properties visible in the
    question. Any retry returns to the provider default in the runner.
    """

    compact = re.sub(r"\s+", "", str(question))
    years = tuple(dict.fromkeys(re.findall(r"20\d{2}", compact)))
    non_year_question = re.sub(r"20\d{2}年?", "", compact)
    numeric_literals = re.findall(
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?",
        non_year_question,
    )
    quoted_formulae = re.findall(
        r"[“\"]([^”\"]*(?:=|＝)[^”\"]*)[”\"]",
        str(question),
    )
    has_formula = bool(quoted_formulae)
    has_ranking = any(
        marker in compact
        for marker in ("排序", "从高到低", "从低到高")
    )
    has_reconciliation = any(
        marker in compact
        for marker in ("偏差", "差异", "核对", "吻合程度")
    )
    has_external_source_reference = any(
        marker in compact
        for marker in ("查阅", "根据", "报告", "合同", "材料", "文档")
    )
    has_semantic_constraint = bool(
        tuple(
            item
            for item in semantic_constraint_types
            if str(item).strip()
        )
    )

    if (
        domain == "research"
        and answer_slots == 1
        and len(numeric_literals) >= 4
        and not has_external_source_reference
        and not has_semantic_constraint
    ):
        return CalculationThinkingPolicy(
            mode="off",
            thinking_budget=None,
            reasons=(
                "single_slot_question_supplies_dense_numeric_inputs",
                "no_external_source_reference",
                "no_runtime_semantic_constraint",
            ),
        )

    if (
        domain == "financial_reports"
        and len(years) == 1
        and answer_slots <= 2
        and has_formula
        and not has_ranking
        and has_reconciliation
    ):
        return CalculationThinkingPolicy(
            mode="off",
            thinking_budget=None,
            reasons=(
                "single_period_explicit_formula_reconciliation",
                "no_ranking_or_cross_period_comparison",
            ),
        )

    if (
        domain == "financial_reports"
        and len(years) == 1
        and answer_slots == 2
        and has_formula
        and has_ranking
    ):
        return CalculationThinkingPolicy(
            mode="budget",
            thinking_budget=4096,
            reasons=(
                "single_period_explicit_formula_ranking",
                "bounded_first_attempt_then_default_retry",
            ),
        )

    return CalculationThinkingPolicy(
        mode="default",
        thinking_budget=None,
        reasons=("complex_or_evidence_sensitive_task",),
    )


def build_calculation_profile_messages(
    *,
    domain: str,
    question: str,
    answer_format: str,
    answer_slots: int,
    answer_slot_templates: Sequence[str],
) -> list[dict[str, str]]:
    """Build the answer-blind profile request.

    The interface deliberately has no qid, document identifiers, retrieved
    evidence, previous answer, or retry state.
    """

    payload = {
        "domain": str(domain),
        "question": str(question),
        "answer_format": str(answer_format),
        "answer_slots": int(answer_slots),
        "answer_slot_templates": [
            str(item) for item in answer_slot_templates
        ],
    }
    return [
        {
            "role": "system",
            "content": CALCULATION_PROFILE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def parse_calculation_profile(
    payload: Mapping[str, Any],
    *,
    domain: str,
    expected_output_count: int,
) -> CalculationProfile:
    normalized = _normalize_profile_payload(payload)
    errors = sorted(
        _PROFILE_VALIDATOR.iter_errors(normalized),
        key=lambda item: tuple(str(part) for part in item.path),
    )
    if errors:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in item.path) or '<root>'}: "
            f"{item.message}"
            for item in errors
        )
        raise ValueError(f"CalculationProfile schema violation: {detail}")
    if len(normalized["outputs"]) != int(expected_output_count):
        raise ValueError(
            "CalculationProfile output count does not match answer slots"
        )
    risk_checks = list(normalized["risk_checks"])
    for fact in normalized["required_facts"]:
        risk_checks.extend(
            _SOURCE_KIND_RISK_CHECKS.get(fact["source_kind"], ())
        )
    return CalculationProfile(
        domain=str(domain),
        task_types=tuple(normalized["task_types"]),
        required_facts=tuple(
            RequiredFact(
                subject=item["subject"],
                metric=item["metric"],
                period=item["period"],
                unit=item["unit"],
                role=item["role"],
                source_kind=item["source_kind"],
            )
            for item in normalized["required_facts"]
        ),
        operators=tuple(normalized["operators"]),
        outputs=tuple(
            ProfileOutput(kind=item["kind"], format=item["format"])
            for item in normalized["outputs"]
        ),
        complexity=normalized["complexity"],
        risk_checks=tuple(dict.fromkeys(risk_checks)),
    )


def calculation_profile_solver_guidance(
    profile: CalculationProfile | None,
) -> tuple[str, ...]:
    if profile is None:
        return ()
    return tuple(
        _RISK_GUIDANCE[item]
        for item in profile.risk_checks
        if item in _RISK_GUIDANCE
    )


def _normalize_profile_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(payload)
    for key in ("task_types", "operators", "risk_checks"):
        raw_items = normalized.get(key)
        if isinstance(raw_items, list):
            normalized[key] = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_items
                    if str(item).strip()
                )
            )
    raw_facts = normalized.get("required_facts")
    if isinstance(raw_facts, list):
        normalized["required_facts"] = [
            {
                "subject": str(item.get("subject", "")).strip(),
                "metric": str(item.get("metric", "")).strip(),
                "period": str(item.get("period", "")).strip(),
                "unit": str(item.get("unit", "")).strip(),
                "role": str(item.get("role", "")).strip(),
                "source_kind": str(
                    item.get("source_kind", "")
                ).strip(),
            }
            if isinstance(item, Mapping)
            else item
            for item in raw_facts
        ]
    raw_outputs = normalized.get("outputs")
    if isinstance(raw_outputs, list):
        normalized["outputs"] = [
            {
                "kind": str(item.get("kind", "")).strip(),
                "format": str(item.get("format", "")).strip(),
            }
            if isinstance(item, Mapping)
            else item
            for item in raw_outputs
        ]
    if "complexity" in normalized:
        normalized["complexity"] = str(
            normalized["complexity"]
        ).strip()
    return normalized
