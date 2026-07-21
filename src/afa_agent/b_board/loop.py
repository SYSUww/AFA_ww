from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from afa_agent.b_board.evaluator import ConfidenceEvaluation, decide_candidate_promotion
from afa_agent.experiment_registry import (
    DECISION_EXECUTE,
    DECISION_REFINE_EXISTING,
    DECISION_RETRY_AFTER_CONTEXT_CHANGE,
    DECISION_REUSE_PROMOTED,
    DECISION_SKIP_DUPLICATE,
    ExperimentRegistry,
    HistoryDecision,
    sanitize_registry_payload,
)


MAX_COMPARABLE_ATTEMPTS = 3
EXECUTABLE_HISTORY_DECISIONS = {
    DECISION_EXECUTE,
    DECISION_REFINE_EXISTING,
    DECISION_RETRY_AFTER_CONTEXT_CHANGE,
}

_TIER_ORDER = {"blocked": 0, "low": 1, "medium": 2, "high": 3}
_LOW_TIERS = {"blocked", "low"}


@dataclass(frozen=True, slots=True)
class Direction:
    direction_id: str
    pipeline_stage: str
    root_cause_cluster: str
    hypothesis: str
    change_vector: Mapping[str, Any]
    priority: int = 100
    source: str = "initial"
    target_qids: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    question_types: tuple[str, ...] = ()
    material_delta: Mapping[str, Any] | None = None

    def candidate(self, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "direction_id": self.direction_id,
            "pipeline_stage": self.pipeline_stage,
            "root_cause_cluster": self.root_cause_cluster,
            "hypothesis": self.hypothesis,
            "change_vector": dict(self.change_vector),
            "domains": list(self.domains),
            "question_types": list(self.question_types),
            "target_qids": list(self.target_qids),
            "direction_source": self.source,
        }
        if self.material_delta:
            payload["material_delta"] = dict(self.material_delta)
        payload.update(dict(context or {}))
        return payload


@dataclass(frozen=True, slots=True)
class Attempt:
    experiment_id: str
    direction: Direction
    attempt_index: int
    action: str
    candidate: Mapping[str, Any]
    history: HistoryDecision
    reason: str

    @property
    def executable(self) -> bool:
        return self.action == "execute"


@dataclass(frozen=True, slots=True)
class SchedulerSignal:
    kind: str
    reason: str
    unresolved_clusters: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromotionMergeResult:
    answers: dict[str, Any]
    evaluations: dict[str, ConfidenceEvaluation]
    decisions: dict[str, dict[str, Any]]
    promoted_qids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoundGateResult:
    valid: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]


@dataclass(order=True, slots=True)
class _QueuedDirection:
    priority: int
    sequence: int
    direction: Direction = field(compare=False)


def initial_directions() -> tuple[Direction, ...]:
    """Return the initial open search space; it is intentionally not a whitelist."""

    specs = (
        (0, "run_integrity", "runtime", "integrity", "补齐运行、检查点、格式与Token完整性"),
        (0, "evidence_contract", "evidence", "citation", "强制答案引用真实使用的证据单元"),
        (0, "calculation_executor", "calculation", "calculation", "使用可重放工具执行计算、单位和日期逻辑"),
        (10, "document_cleanup", "preprocessing", "parsing", "清理OCR、乱码、断行、页眉页脚与重复文本"),
        (10, "table_formula_recovery", "preprocessing", "parsing", "恢复跨页表格、脚注、公式及单位绑定"),
        (10, "document_structure", "preprocessing", "parsing", "提取标题、章节、条款、主体、产品和年份元数据"),
        (10, "multi_granularity_chunking", "chunking", "chunking", "比较语义块、条款块、表格块、父子块和多粒度块"),
        (10, "document_routing", "routing", "routing", "用实体、年份、文档类型、法规和产品别名路由文档"),
        (10, "sparse_retrieval", "retrieval", "retrieval", "优化BM25、查询拆解、多查询和迭代检索"),
        (10, "hybrid_retrieval", "retrieval", "retrieval", "验证稀疏、向量和混合召回及重排"),
        (10, "evidence_reranking", "retrieval", "evidence", "按主体、年份、指标、单位、反证和覆盖重排证据"),
        (10, "context_assembly", "context", "evidence", "验证父子块、证据卡、表格重建和跨文档对齐"),
        (20, "domain_prompt", "prompt", "decision", "按领域与题型优化结构化答案和引用Prompt"),
        (20, "question_decomposition", "reasoning", "decision", "将跨主体、跨年份和多步骤问题拆为可验证子问题"),
        (20, "option_verification", "reasoning", "decision", "逐项验证选项并检查最强竞争项和多选完整性"),
        (20, "counterevidence", "reasoning", "decision", "主动搜索错主体、错年份、错单位和相反条款"),
        (20, "iterative_reasoning", "reasoning", "stability", "执行初答、证据复核、反证、修正和封存"),
        (20, "self_consistency", "verification", "stability", "比较多个检索视图、拆解方法和推理结果"),
        (20, "domain_rules", "rules", "decision", "引入可追溯的财报、保险、监管和合同领域规则"),
        (20, "local_tools", "tools", "calculation", "使用计算器、日历、单位换算和表格工具"),
        (30, "multi_agent_adjudication", "verification", "stability", "比较多候选、critic、debate与裁决流程"),
        (30, "multimodal_document", "preprocessing", "parsing", "验证扫描PDF、复杂表格与公式页面视觉读取"),
        (30, "graph_iterative_rag", "retrieval", "retrieval", "验证证据图、跨文档事实链与查询反馈"),
        (30, "output_normalization", "postprocess", "format", "修复多选顺序、精度、日期、百分数和多槽输出"),
    )
    return tuple(
        Direction(
            direction_id=direction_id,
            pipeline_stage=stage,
            root_cause_cluster=cluster,
            hypothesis=hypothesis,
            change_vector={"strategy": direction_id},
            priority=priority,
        )
        for priority, direction_id, stage, cluster, hypothesis in specs
    )


_CLUSTER_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("calculation", ("计算", "公式", "算术", "舍入", "round", "unit", "单位", "日期", "calendar")),
    ("citation", ("citation", "引用", "证据id", "evidence id", "provenance", "溯源")),
    ("chunking", ("chunk", "分块", "截断", "overlap", "父子块")),
    ("parsing", ("ocr", "解析", "乱码", "表格", "table", "公式恢复", "页眉", "断行")),
    ("routing", ("wrong document", "错文档", "文档路由", "主体错误", "错主体", "年份错误", "错年份")),
    ("retrieval", ("retrieval", "检索", "召回", "找不到", "rerank", "topk")),
    ("evidence", ("evidence", "证据不足", "证据覆盖", "support", "支撑不足")),
    ("format", ("format", "格式", "答案槽", "slot", "多选顺序")),
    ("stability", ("stability", "不一致", "冲突", "漂移", "复核", "consistency")),
    ("decision", ("entail", "判别", "选项", "答案错误", "反证", "竞争项", "推理")),
)

_CLUSTER_STAGE = {
    "calculation": "calculation",
    "citation": "evidence",
    "chunking": "chunking",
    "parsing": "preprocessing",
    "routing": "routing",
    "retrieval": "retrieval",
    "evidence": "evidence",
    "format": "postprocess",
    "stability": "verification",
    "decision": "reasoning",
    "unknown": "analysis",
}


def cluster_low_confidence_reasons(
    evaluations: Mapping[str, ConfidenceEvaluation | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for qid, evaluation in evaluations.items():
        tier = str(_value(evaluation, "tier", ""))
        if tier not in _LOW_TIERS:
            continue
        reasons = _string_items(_value(evaluation, "low_confidence_reasons", ()))
        reasons += _string_items(_value(evaluation, "blocking_reasons", ()))
        reasons += _string_items(_value(evaluation, "hard_failures", ()))
        if not reasons:
            reasons = ["未分类低置信原因"]
        for reason in reasons:
            cluster = _classify_reason(reason)
            bucket = clusters.setdefault(cluster, {"qids": set(), "reasons": set()})
            bucket["qids"].add(str(qid))
            bucket["reasons"].add(reason)
    return {
        cluster: {
            "qids": tuple(sorted(bucket["qids"])),
            "reasons": tuple(sorted(bucket["reasons"])),
        }
        for cluster, bucket in sorted(clusters.items())
    }


def directions_from_low_confidence(
    evaluations: Mapping[str, ConfidenceEvaluation | Mapping[str, Any]],
    *,
    qid_domains: Mapping[str, str] | None = None,
) -> tuple[Direction, ...]:
    directions: list[Direction] = []
    for cluster, details in cluster_low_confidence_reasons(evaluations).items():
        reasons = tuple(details["reasons"])
        qids = tuple(details["qids"])
        signature = hashlib.sha256(
            json.dumps(reasons, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        domains = tuple(
            sorted({str(qid_domains[qid]) for qid in qids if qid_domains and qid in qid_domains})
        )
        directions.append(
            Direction(
                direction_id=f"dynamic_{cluster}_{signature}",
                pipeline_stage=_CLUSTER_STAGE[cluster],
                root_cause_cluster=cluster,
                hypothesis=f"针对{cluster}低置信根因进行定向优化",
                change_vector={"root_cause_reasons": list(reasons), "strategy": "evaluator_driven"},
                priority=5,
                source="evaluator",
                target_qids=qids,
                domains=domains,
                material_delta={"new_evaluator_root_cause": cluster},
            )
        )
    return tuple(directions)


class OpenEndedLoopScheduler:
    def __init__(
        self,
        registry: ExperimentRegistry,
        *,
        directions: Sequence[Direction] | None = None,
        max_comparable_attempts: int = MAX_COMPARABLE_ATTEMPTS,
    ) -> None:
        if max_comparable_attempts < 1:
            raise ValueError("max_comparable_attempts must be positive")
        self.registry = registry
        self.max_comparable_attempts = max_comparable_attempts
        self._queue: list[_QueuedDirection] = []
        self._queued_keys: set[str] = set()
        self._seen_keys: set[str] = set()
        self._sequence = 0
        self._running: dict[str, Attempt] = {}
        self._direction_status: dict[str, str] = {}
        self._directions: dict[str, Direction] = {}
        self._unresolved_clusters: set[str] = set()
        self._external_research_exhausted = False
        self._research_waves = 0
        for direction in directions if directions is not None else initial_directions():
            self.enqueue(direction)

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def direction_status(self) -> dict[str, str]:
        return dict(self._direction_status)

    def enqueue(self, direction: Direction) -> bool:
        key = _direction_queue_key(direction)
        if key in self._queued_keys or key in self._seen_keys:
            return False
        self._sequence += 1
        self._queue.append(_QueuedDirection(direction.priority, self._sequence, direction))
        self._queue.sort()
        self._queued_keys.add(key)
        self._directions[direction.direction_id] = direction
        self._direction_status[direction.direction_id] = "queued"
        return True

    def add_evaluator_directions(
        self,
        evaluations: Mapping[str, ConfidenceEvaluation | Mapping[str, Any]],
        *,
        qid_domains: Mapping[str, str] | None = None,
    ) -> tuple[Direction, ...]:
        clusters = cluster_low_confidence_reasons(evaluations)
        self._unresolved_clusters.update(clusters)
        added = tuple(
            direction
            for direction in directions_from_low_confidence(evaluations, qid_domains=qid_domains)
            if self.enqueue(direction)
        )
        if added:
            self._external_research_exhausted = False
        return added

    def next_attempt(self, *, context: Mapping[str, Any] | None = None) -> Attempt | None:
        if not self._queue:
            return None
        queued = self._queue.pop(0)
        direction = queued.direction
        queue_key = _direction_queue_key(direction)
        self._queued_keys.discard(queue_key)
        self._seen_keys.add(queue_key)

        candidate = direction.candidate(context)
        # This is deliberately the first history operation for every candidate.
        history = self.registry.decide(candidate)
        index = history.comparable_attempt_count + 1
        experiment_id = _experiment_id(direction.direction_id, index, history.candidate_fingerprint)

        if history.comparable_attempt_count >= self.max_comparable_attempts:
            action = "attempt_limit"
            reason = f"方向已有{history.comparable_attempt_count}个可比完整尝试，达到上限"
            self._direction_status[direction.direction_id] = "exhausted"
        elif history.decision in EXECUTABLE_HISTORY_DECISIONS:
            action = "execute"
            reason = history.reason
            self._direction_status[direction.direction_id] = "running"
        elif history.decision == DECISION_REUSE_PROMOTED:
            action = "reuse_promoted"
            reason = history.reason
            self._direction_status[direction.direction_id] = "resolved"
        elif history.decision == DECISION_SKIP_DUPLICATE:
            action = "skip_duplicate"
            reason = history.reason
            self._direction_status[direction.direction_id] = "exhausted"
        else:
            action = "skip"
            reason = history.reason
            self._direction_status[direction.direction_id] = "exhausted"

        attempt = Attempt(
            experiment_id=experiment_id,
            direction=direction,
            attempt_index=index,
            action=action,
            candidate=candidate,
            history=history,
            reason=reason,
        )
        if attempt.executable:
            self._running[experiment_id] = attempt
        return attempt

    def complete_attempt(
        self,
        attempt: Attempt,
        *,
        status: str,
        record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not attempt.executable:
            raise ValueError("Only executable attempts can be completed")
        if attempt.experiment_id not in self._running:
            raise ValueError(f"Attempt is not running: {attempt.experiment_id}")
        payload = {
            **dict(attempt.candidate),
            **dict(record or {}),
            "experiment_id": attempt.experiment_id,
            "direction_id": attempt.direction.direction_id,
            "attempt_index": attempt.attempt_index,
            "status": status,
            "history_decision": attempt.history.to_dict(),
        }
        stored = self.registry.append(payload)
        self._running.pop(attempt.experiment_id, None)
        normalized = status.strip().lower()
        if normalized in {"accepted", "effective", "promoted", "resolved"}:
            self._direction_status[attempt.direction.direction_id] = "resolved"
            self._unresolved_clusters.discard(attempt.direction.root_cause_cluster)
        elif normalized in {"failed", "error", "blocked_technical", "incomplete", "interrupted"}:
            self._direction_status[attempt.direction.direction_id] = "retryable"
        elif self.registry.count_comparable_attempts(attempt.candidate) >= self.max_comparable_attempts:
            self._direction_status[attempt.direction.direction_id] = "exhausted"
        else:
            self._direction_status[attempt.direction.direction_id] = "awaiting_refinement"
        return stored

    def record_external_research_wave(self, directions: Sequence[Direction]) -> int:
        self._research_waves += 1
        added = 0
        for direction in directions:
            external = direction if direction.source == "online" else replace(direction, source="online")
            added += int(self.enqueue(external))
        self._external_research_exhausted = added == 0
        return added

    def signal(self) -> SchedulerSignal | None:
        if self._queue or self._running:
            return None
        unresolved = tuple(sorted(self._unresolved_clusters))
        refinable = tuple(
            sorted(
                direction_id
                for direction_id, status in self._direction_status.items()
                if status in {"awaiting_refinement", "retryable"}
                and self._directions[direction_id].root_cause_cluster in self._unresolved_clusters
            )
        )
        if refinable:
            return SchedulerSignal(
                "local_refinement_required",
                "仍有未达到三次上限的本地方向需要生成实质增量候选: " + ",".join(refinable),
                unresolved,
            )
        if unresolved and not self._external_research_exhausted:
            return SchedulerSignal(
                "online_research_required",
                "本地非重复可执行方向已枯竭，仍有低置信根因",
                unresolved,
            )
        if unresolved and self._external_research_exhausted:
            return SchedulerSignal(
                "stop",
                "最近一次外部研究波次没有产生新的非重复可执行方向",
                unresolved,
            )
        return SchedulerSignal("complete", "没有未解决的低置信根因")


def merge_promoted_answers(
    *,
    incumbent_answers: Mapping[str, Any],
    candidate_answers: Mapping[str, Any],
    incumbent_evaluations: Mapping[str, ConfidenceEvaluation],
    candidate_evaluations: Mapping[str, ConfidenceEvaluation],
    blind_winners: Mapping[str, str] | None = None,
    candidate_blind_labels: Mapping[str, str] | None = None,
) -> PromotionMergeResult:
    answers = dict(incumbent_answers)
    evaluations = dict(incumbent_evaluations)
    decisions: dict[str, dict[str, Any]] = {}
    promoted: list[str] = []
    for qid in sorted(candidate_answers):
        if qid not in incumbent_answers:
            decisions[qid] = {"promote": False, "reasons": ["missing_incumbent_answer"]}
            continue
        if qid not in incumbent_evaluations or qid not in candidate_evaluations:
            decisions[qid] = {"promote": False, "reasons": ["missing_confidence_evaluation"]}
            continue
        answer_changed = _answer_identity(incumbent_answers[qid]) != _answer_identity(
            candidate_answers[qid]
        )
        decision = decide_candidate_promotion(
            incumbent=incumbent_evaluations[qid],
            candidate=candidate_evaluations[qid],
            answer_changed=answer_changed,
            blind_winner=(blind_winners or {}).get(qid),
            candidate_blind_label=(candidate_blind_labels or {}).get(qid),
        )
        decisions[qid] = decision
        if decision["promote"]:
            answers[qid] = candidate_answers[qid]
            evaluations[qid] = candidate_evaluations[qid]
            promoted.append(qid)
    return PromotionMergeResult(answers, evaluations, decisions, tuple(promoted))


def evaluate_round_gate(
    *,
    incumbent_evaluations: Mapping[str, ConfidenceEvaluation],
    candidate_evaluations: Mapping[str, ConfidenceEvaluation],
    qid_domains: Mapping[str, str],
    expected_qids: Iterable[str] | None = None,
    sentinels_passed: bool,
    tests_passed: bool,
    integrity_passed: bool,
    unexplained_regressions: Sequence[str] = (),
    changed_qids: Iterable[str] | None = None,
) -> RoundGateResult:
    reasons: list[str] = []
    expected = set(expected_qids if expected_qids is not None else incumbent_evaluations)
    candidate_qids = set(candidate_evaluations)
    if candidate_qids != expected:
        reasons.append("incomplete_evaluation_coverage")
    if not sentinels_passed:
        reasons.append("calibration_sentinels_failed")
    if not tests_passed:
        reasons.append("tests_failed")
    if not integrity_passed:
        reasons.append("run_integrity_failed")
    if unexplained_regressions:
        reasons.append("unexplained_answer_regressions")
    changed = set(expected if changed_qids is None else changed_qids)
    if changed - expected:
        reasons.append("changed_qids_outside_expected")
    changed &= expected
    comparison_evaluations = {
        qid: (
            candidate_evaluations[qid]
            if qid in changed and qid in candidate_evaluations
            else incumbent_evaluations[qid]
        )
        for qid in expected
        if qid in incumbent_evaluations
        and (qid in candidate_evaluations or qid not in changed)
    }
    unchanged_raw_score_drift = {
        qid: candidate_evaluations[qid].confidence_score
        - incumbent_evaluations[qid].confidence_score
        for qid in sorted((expected - changed) & candidate_qids & set(incumbent_evaluations))
        if candidate_evaluations[qid].confidence_score
        != incumbent_evaluations[qid].confidence_score
    }
    unchanged_raw_tier_drift = [
        qid
        for qid in sorted((expected - changed) & candidate_qids & set(incumbent_evaluations))
        if candidate_evaluations[qid].tier != incumbent_evaluations[qid].tier
    ]
    new_hard_failures: dict[str, list[str]] = {}
    for qid in sorted(candidate_qids & expected):
        incumbent_hard_failures = set(
            incumbent_evaluations[qid].hard_failures
            if qid in incumbent_evaluations
            else ()
        )
        introduced = sorted(
            set(candidate_evaluations[qid].hard_failures) - incumbent_hard_failures
        )
        if introduced:
            new_hard_failures[qid] = introduced
    if new_hard_failures:
        reasons.append("candidate_introduces_new_hard_failures")

    domains = sorted({qid_domains[qid] for qid in expected if qid in qid_domains})
    domain_metrics: dict[str, dict[str, Any]] = {}
    any_low_tier_improvement = False
    any_p10_gain = False
    for domain in domains:
        qids = sorted(qid for qid in expected if qid_domains.get(qid) == domain)
        before = [incumbent_evaluations[qid] for qid in qids if qid in incumbent_evaluations]
        after = [comparison_evaluations[qid] for qid in qids if qid in comparison_evaluations]
        before_low = sum(item.tier in _LOW_TIERS for item in before)
        after_low = sum(item.tier in _LOW_TIERS for item in after)
        before_p10 = _p10([item.confidence_score for item in before])
        after_p10 = _p10([item.confidence_score for item in after])
        p10_delta = after_p10 - before_p10
        improved_qids = [
            qid
            for qid in qids
            if qid in incumbent_evaluations
            and qid in comparison_evaluations
            and incumbent_evaluations[qid].tier in _LOW_TIERS
            and _TIER_ORDER[comparison_evaluations[qid].tier]
            > _TIER_ORDER[incumbent_evaluations[qid].tier]
        ]
        any_low_tier_improvement |= bool(improved_qids)
        any_p10_gain |= p10_delta >= 3
        if after_low > before_low:
            reasons.append(f"blocked_low_increased:{domain}")
        domain_metrics[domain] = {
            "blocked_low_before": before_low,
            "blocked_low_after": after_low,
            "p10_before": before_p10,
            "p10_after": after_p10,
            "p10_delta": p10_delta,
            "low_tier_improved_qids": improved_qids,
        }
    if not (any_low_tier_improvement or any_p10_gain):
        reasons.append("no_low_tail_improvement")
    metrics = {
        "expected_qid_count": len(expected),
        "evaluated_qid_count": len(candidate_qids & expected),
        "domain_metrics": domain_metrics,
        "has_low_tier_improvement": any_low_tier_improvement,
        "has_domain_p10_gain": any_p10_gain,
        "new_hard_failures": new_hard_failures,
        "causal_changed_qids": sorted(changed),
        "unchanged_raw_score_drift": unchanged_raw_score_drift,
        "unchanged_raw_tier_drift": unchanged_raw_tier_drift,
    }
    return RoundGateResult(not reasons, tuple(sorted(set(reasons))), metrics)


def append_markdown_log(path: Path, entry: Mapping[str, Any]) -> None:
    """Append a sanitized structured event without ever rewriting prior log bytes."""

    cleaned = sanitize_registry_payload(entry)
    cleaned = _redact_sensitive_text(cleaned)
    if not isinstance(cleaned, Mapping):
        raise TypeError("Markdown log entry must be a mapping")
    recorded_at = str(cleaned.get("recorded_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    experiment_id = str(cleaned.get("experiment_id") or cleaned.get("event") or "loop-event")
    payload = dict(cleaned)
    payload["recorded_at"] = recorded_at
    markdown = (
        f"## {experiment_id}\n\n"
        f"- recorded_at: `{recorded_at}`\n\n"
        "```json\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n```\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        separator = "\n" if os.fstat(descriptor).st_size else ""
        encoded = (separator + markdown).encode("utf-8")
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _classify_reason(reason: str) -> str:
    normalized = " ".join(reason.lower().split())
    for cluster, patterns in _CLUSTER_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return cluster
    return "unknown"


def _direction_queue_key(direction: Direction) -> str:
    identity = {
        "direction_id": direction.direction_id,
        "change_vector": direction.change_vector,
        "target_qids": direction.target_qids,
        "material_delta": direction.material_delta,
    }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _experiment_id(direction_id: str, index: int, fingerprint: Mapping[str, Any]) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", direction_id).strip("-") or "direction"
    return f"b-loop-{safe}-a{index}-{str(fingerprint.get('semantic_sha256', ''))[:8]}"


def _value(item: ConfidenceEvaluation | Mapping[str, Any], name: str, default: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _answer_identity(answer: Any) -> str:
    if isinstance(answer, Mapping):
        selected = {
            key: value
            for key, value in answer.items()
            if key in {"answer", "answer_parts", "answer_1", "answer_2", "answer_3", "answer_4"}
        }
        answer = selected or answer
    return json.dumps(answer, ensure_ascii=False, sort_keys=True, default=str)


def _p10(scores: Sequence[int]) -> int:
    if not scores:
        return 0
    ordered = sorted(scores)
    return int(ordered[max(0, math.ceil(len(ordered) * 0.10) - 1)])


_SENSITIVE_TEXT_PATTERNS = (
    re.compile(
        r"(?i)\b(?:llm_|openai_)?api[_ -]?(?:key|base|url)\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/-]+"),
    re.compile(r"\bsk-[a-zA-Z0-9_-]{8,}\b"),
)


def _redact_sensitive_text(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_sensitive_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_sensitive_text(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive_text(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SENSITIVE_TEXT_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    return value
