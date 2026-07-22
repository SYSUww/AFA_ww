from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from afa_agent.b_board.calculation import CalculationExecutor, CalculationPlanError
from afa_agent.b_board.io import (
    BAnswer,
    BQuestion,
    infer_requested_decimal_places,
    infer_percent_suffix_requirement,
    validate_b_answer,
    write_b_submission,
)
from afa_agent.b_board.scoring import require_allowed_submission_model
from afa_agent.client import OpenAICompatibleClient, capture_llm_usage, extract_json_object
from afa_agent.config import build_run_config
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.registry import get_plugin
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl
from afa_agent.models import Question, TokenUsage
from afa_agent.run_metadata import (
    RunFingerprintError,
    build_run_fingerprint,
    validate_resume_fingerprint,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARSED_ROOT = ROOT / "artifacts" / "preprocessed_loop_candidates" / "parsed"
DEFAULT_INDEX_ROOT = ROOT / "artifacts" / "preprocessed_loop_candidates" / "index"
DEFAULT_STRATEGY_PATH = ROOT / "configs" / "autoresearch" / "evidence_gate_rescue_accuracy_first.json"


CALCULATION_SYSTEM_PROMPT = """你是金融长文计算题的结构化求解器。只使用题目和给定证据，不补充未给出的事实。
输出一个 JSON 对象，字段为 variables、steps、outputs、decision_summary。
variables: [{name,value,value_type,unit,evidence_ids}]，value_type 仅 decimal/date/text，所有变量必须给 evidence_ids。
每个变量的 value 必须以同一数值或日期直接出现在所引证据中，unit 也必须与证据一致；不得把 5.55% 擅自写成 0.0555。
只有证据同一片段明确写出单位时才填 unit；表格只有裸金额但未标单位时必须填空字符串，不得推断或补写“元”。比率或百分比计算可直接使用同口径原始金额。
证据缺变量时不要用“无法计算”等文本冒充数值输出；该题应让计划校验失败并等待重新检索。
steps: [{id,op,args,...}]，引用写成 {"ref":"变量或步骤id"}。
方向性运算禁止使用位置参数：pct_change 必须写 new 和 old 字段，严格按
(new / old - 1) * 100 计算；pct_point_delta 也必须写 new 和 old，严格按 new - old 计算。
日期运算使用具名参数：date_add_days 的 args 写 {"date":{"ref":"日期变量"},"days":{"ref":"天数变量"}}；
next_workday 的 args 写 {"date":{"ref":"日期变量"}}；days_between 的 args 写
{"end":{"ref":"结束日期"},"start":{"ref":"开始日期"}}，严格按 end - start 计算。
允许 op: add,sub,mul,div,mean,abs,max,min,pct_change,pct_point_delta,count_gte,count_gt,sort_desc,date_add_days,next_workday,days_between。
sort_desc 使用 items:[{label,source}]。outputs 数量必须等于答案槽数；每项为 {source,format}。
format 仅 raw,decimal0,decimal1,decimal2,percent2,date_cn,text。中间过程不得舍入，最终才按格式四舍五入。
格式优先级为：题干具体要求 > README通用规则 > 提交模板占位。题干未规定时，README要求百分数答案带%并保留两位小数，其他数值不带单位并保留两位小数。
证据 ID 必须原样使用给定 evidence_id。题目本身给出的数值可引用 question:<qid>。只输出 JSON。"""

SUBMISSION_REASONING_PROMPT_VERSION = "b_submission_reasoning_v1"
SUBMISSION_REASONING_SYSTEM_PROMPT = f"""你是金融长文问答的提交摘要生成器。只使用给定题目、答案、已有求解摘要和证据，不补充外部事实，也不得改变答案。
输出一个 JSON 对象，字段仅为 answer_parts 和 reasoning。answer_parts 必须逐字复制给定答案。
reasoning 是可审计但不暴露冗长思维链的推理摘要，使用中文，建议 60-220 字；必须具体说明定位到的主体/条款/指标，给出关键证据事实或计算关系，再说明这些依据如何支持答案。不得只复述题目或答案，不得写空泛模板，不得声称未提供的页码、条款号或事实。
即使已有求解摘要很短，也要依据给定证据形成自洽摘要。只输出 JSON。prompt_version={SUBMISSION_REASONING_PROMPT_VERSION}。"""

RUNNER_VERSION = "b_actual_v10_reasoning_audit"
CALCULATION_RETRIEVAL_VERSION = "phrase_constrained_v2"


@dataclass(slots=True)
class BAnswerArtifact:
    qid: str
    domain: str
    answer_format: str
    answer_slot_count: int
    answer_parts: list[str]
    used_evidence_ids: list[str]
    evidence_items: list[dict[str, Any]]
    decision_summary: str
    decision_trace: dict[str, Any]
    calculation_trace: dict[str, Any]
    token_usage: dict[str, int]
    locator: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "domain": self.domain,
            "answer_format": self.answer_format,
            "answer_slot_count": self.answer_slot_count,
            "answer_parts": self.answer_parts,
            "used_evidence_ids": self.used_evidence_ids,
            "evidence_items": self.evidence_items,
            "decision_summary": self.decision_summary,
            "decision_trace": self.decision_trace,
            "calculation_trace": self.calculation_trace,
            "token_usage": self.token_usage,
            "locator": self.locator,
        }

    def to_submission_answer(self) -> BAnswer:
        return BAnswer(
            qid=self.qid,
            answer_parts=tuple(self.answer_parts),
            prompt_tokens=int(self.token_usage.get("prompt_tokens", 0)),
            completion_tokens=int(self.token_usage.get("completion_tokens", 0)),
            total_tokens=int(self.token_usage.get("total_tokens", 0)),
            reasoning=self.decision_summary,
        )


class BAnswerGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        token_usage: Mapping[str, int],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> None:
        super().__init__(message)
        self.token_usage = {key: int(value) for key, value in token_usage.items()}
        self.diagnostics = [dict(item) for item in diagnostics]


class BBoardActualRunner:
    def __init__(
        self,
        *,
        questions: Sequence[BQuestion],
        parsed_root: Path = DEFAULT_PARSED_ROOT,
        index_root: Path = DEFAULT_INDEX_ROOT,
        strategy_path: Path = DEFAULT_STRATEGY_PATH,
        locator_attempt_id: str = "attempt_43",
        calculation_top_k: int = 18,
    ) -> None:
        self.questions = list(questions)
        self.question_by_qid = {item.qid: item for item in questions}
        self.parsed_root = Path(parsed_root).resolve()
        self.index_root = Path(index_root).resolve()
        self.strategy_path = Path(strategy_path).resolve()
        self.locator_attempt_id = locator_attempt_id
        self.calculation_top_k = calculation_top_k
        self.config = build_run_config(ROOT)
        if self.config.model is None:
            raise RuntimeError("Missing model config in .env")
        require_allowed_submission_model(self.config.model.model_name)
        self.client = OpenAICompatibleClient(self.config.model)
        self.calculator = CalculationExecutor()
        self._migration = _migration_module()
        self.attempt = _find_locator_attempt(self._migration, locator_attempt_id)
        self.payloads = self._migration.load_domain_payloads(self.parsed_root, self.index_root)
        self.retrievers = {
            domain: GenericBM25Retriever(payload["index"].get("units", []))
            for domain, payload in self.payloads.items()
        }
        os.environ["AFA_STRATEGY_CONFIG"] = str(self.strategy_path)

    def locate(self, questions: Sequence[BQuestion] | None = None) -> dict[str, dict[str, Any]]:
        selected = list(questions or self.questions)
        rows = {item.qid: _question_row(item) for item in selected}
        located = self._migration.locate_docs(rows, [item.qid for item in selected], self.payloads, self.attempt)
        return {row["qid"]: row for row in located}

    def answer_one(self, question: BQuestion, locator: Mapping[str, Any]) -> BAnswerArtifact:
        with capture_llm_usage() as usage_ledger:
            try:
                effective_attempt = self._migration._effective_attempt_for_domain(
                    self.attempt, question.domain
                )
                candidate_doc_ids = self._migration.select_answer_doc_ids(
                    dict(locator), _question_row(question), effective_attempt
                )
                if question.answer_format == "calculation":
                    artifact = self._answer_calculation(question, candidate_doc_ids, locator)
                else:
                    artifact = self._answer_choice(question, candidate_doc_ids, locator)
            except Exception as exc:
                diagnostics = list(getattr(exc, "diagnostics", []))
                diagnostics.append({"stage": "api_usage_ledger", "calls": usage_ledger.calls})
                raise BAnswerGenerationError(
                    str(exc),
                    token_usage=usage_ledger.total(),
                    diagnostics=diagnostics,
                ) from exc

        ledger_total = usage_ledger.total()
        if artifact.token_usage != ledger_total:
            artifact.decision_trace = {
                **artifact.decision_trace,
                "solver_reported_token_usage": dict(artifact.token_usage),
            }
        artifact.token_usage = ledger_total
        artifact.decision_trace = {
            **artifact.decision_trace,
            "api_usage_ledger": {
                "call_count": len(usage_ledger.calls),
                "calls": usage_ledger.calls,
            },
        }
        return artifact

    def run(
        self,
        *,
        run_dir: Path,
        qids: Sequence[str] | None = None,
        workers: int = 4,
        force: bool = False,
    ) -> dict[str, Any]:
        selected = self.questions if qids is None else [self.question_by_qid[qid] for qid in qids]
        run_dir = Path(run_dir).resolve()
        fingerprint = self._build_fingerprint(selected, workers)
        existing = self._prepare_run(run_dir, fingerprint, force=force)
        artifacts_by_qid = {row["qid"]: _artifact_from_dict(row) for row in existing}
        remaining = [item for item in selected if item.qid not in artifacts_by_qid]
        locator_by_qid = self.locate(selected)
        write_jsonl(run_dir / "locator.jsonl", [locator_by_qid[item.qid] for item in selected])
        failures: list[dict[str, Any]] = []

        def persist() -> None:
            ordered = [artifacts_by_qid[item.qid].to_dict() for item in selected if item.qid in artifacts_by_qid]
            write_json(run_dir / "answers.json", ordered)

        if workers > 1 and remaining:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self.answer_one, item, locator_by_qid[item.qid]): item for item in remaining}
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        artifact = future.result()
                        validate_b_answer(item, artifact.to_submission_answer())
                        artifacts_by_qid[item.qid] = artifact
                        persist()
                    except Exception as exc:
                        failures.append(_failure_record(item.qid, exc))
                        write_jsonl(run_dir / "failures.jsonl", failures)
        else:
            for item in remaining:
                try:
                    artifact = self.answer_one(item, locator_by_qid[item.qid])
                    validate_b_answer(item, artifact.to_submission_answer())
                    artifacts_by_qid[item.qid] = artifact
                    persist()
                except Exception as exc:
                    failures.append(_failure_record(item.qid, exc))
                    write_jsonl(run_dir / "failures.jsonl", failures)

        ordered_artifacts = [artifacts_by_qid[item.qid] for item in selected if item.qid in artifacts_by_qid]
        missing = [item.qid for item in selected if item.qid not in artifacts_by_qid]
        if not missing:
            write_b_submission(
                run_dir / "submit.csv",
                selected,
                [item.to_submission_answer() for item in ordered_artifacts],
                audit_ready=True,
            )
        write_json(run_dir / "answers.json", [item.to_dict() for item in ordered_artifacts])
        write_jsonl(run_dir / "failures.jsonl", failures)
        totals = _sum_tokens(ordered_artifacts)
        failed_totals = _sum_failure_tokens(failures)
        manifest = read_json(run_dir / "run_manifest.json")
        manifest.update(
            {
                "status": "complete" if not missing else "incomplete",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "expected_question_count": len(selected),
                "answered_question_count": len(ordered_artifacts),
                "failed_qids": missing,
                "token_usage": totals,
                "failed_token_usage": failed_totals,
                "generation_token_usage": _add_token_usage(totals, failed_totals),
                "submission_path": str(run_dir / "submit.csv") if not missing else None,
            }
        )
        write_json(run_dir / "run_manifest.json", manifest)
        return manifest

    def _answer_choice(
        self,
        question: BQuestion,
        candidate_doc_ids: list[str],
        locator: Mapping[str, Any],
    ) -> BAnswerArtifact:
        plugin_question = Question(
            qid=question.qid,
            domain=question.domain,
            split="B",
            question=question.question,
            options=dict(question.options),
            answer_format=question.answer_format,
            type=question.type,
            doc_ids=candidate_doc_ids,
            metadata={"doc_ids_are_locator_candidates": True, "locator_attempt": self.locator_attempt_id},
        )
        plugin = get_plugin(question.domain)
        result = plugin.answer_one(
            plugin_question,
            self.parsed_root / question.domain / "parsed.json",
            self.index_root / question.domain / "index.json",
        )
        evidence = [_normalize_evidence(item) for item in result.evidence_items]
        used_ids = list(dict.fromkeys(str(item["unit_id"]) for item in evidence if item.get("unit_id")))
        finalization = result.debug_meta.get("answer_finalization", {}) or {}
        consistency = result.debug_meta.get("final_consistency_check", {}) or {}
        decision_trace = {**finalization, "final_consistency_check": consistency}
        artifact = BAnswerArtifact(
            qid=question.qid,
            domain=question.domain,
            answer_format=question.answer_format,
            answer_slot_count=question.answer_slots,
            answer_parts=[result.pred_answer],
            used_evidence_ids=used_ids,
            evidence_items=evidence,
            decision_summary=result.reasoning_summary,
            decision_trace=decision_trace,
            calculation_trace={},
            token_usage=result.token_usage.to_dict(),
            locator={**dict(locator), "selected_doc_ids": candidate_doc_ids},
        )
        return self._attach_submission_reasoning(question, artifact)

    def _answer_calculation(
        self,
        question: BQuestion,
        candidate_doc_ids: list[str],
        locator: Mapping[str, Any],
    ) -> BAnswerArtifact:
        query = "\n".join([question.question, question.type, "数值 公式 单位 日期 条款 计算"])
        hits = self.retrievers[question.domain].search(
            candidate_doc_ids,
            query,
            top_k=self.calculation_top_k,
            unit_type_boosts={"metric_row": 1.8, "formula_block": 2.0, "clause_block": 1.5, "article": 1.3},
            ensure_per_doc=True,
            expand_neighbors=True,
        )
        question_evidence_id = f"question:{question.qid}"
        evidence_items = [
            {
                "unit_id": question_evidence_id,
                "doc_id": "__question__",
                "title_path": ["题目"],
                "text": question.question,
                "score": 1.0,
            },
            *[_normalize_evidence(hit.to_dict()) for hit in hits],
        ]
        usage = TokenUsage()
        last_error: Exception | None = None
        feedback = ""
        diagnostics: list[dict[str, Any]] = []
        retrieval_rounds: list[dict[str, Any]] = []
        for attempt_number in range(1, 4):
            evidence_payload = _calculation_evidence_payload(evidence_items)
            messages = [
                {"role": "system", "content": CALCULATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"qid：{question.qid}\n题目：{question.question}\n答案槽数：{question.answer_slots}\n"
                        f"提交模板占位：{json.dumps(question.answer_slot_templates, ensure_ascii=False)}\n"
                        f"证据：{json.dumps(evidence_payload, ensure_ascii=False)}\n{feedback}"
                    ),
                },
            ]
            response = self.client.chat_json(messages)
            usage.add(response.token_usage)
            plan: dict[str, Any] | None = None
            try:
                plan = extract_json_object(response.content)
                result = self.calculator.execute(
                    plan,
                    expected_slots=question.answer_slots,
                    evidence_text_by_id={
                        str(item["unit_id"]): str(item.get("text", ""))
                        for item in evidence_items
                    },
                    expected_slot_templates=question.answer_slot_templates,
                    expected_numeric_decimal_places=infer_requested_decimal_places(
                        question.question
                    ),
                    expected_percent_suffixes=tuple(
                        infer_percent_suffix_requirement(
                            question.question,
                            slot_index=index,
                            slot_count=question.answer_slots,
                        )
                        for index in range(1, question.answer_slots + 1)
                    ),
                )
                available = {str(item["unit_id"]) for item in evidence_items}
                missing = sorted(set(result.used_evidence_ids) - available)
                if missing:
                    raise CalculationPlanError("Plan cited unknown evidence IDs: " + ",".join(missing))
                selected_evidence = [item for item in evidence_items if str(item["unit_id"]) in result.used_evidence_ids]
                artifact = BAnswerArtifact(
                    qid=question.qid,
                    domain=question.domain,
                    answer_format=question.answer_format,
                    answer_slot_count=question.answer_slots,
                    answer_parts=list(result.answer_parts),
                    used_evidence_ids=list(result.used_evidence_ids),
                    evidence_items=selected_evidence,
                    decision_summary=str(plan.get("decision_summary", "")).strip(),
                    decision_trace={"source": "structured_calculation_plan", "format_forced": False},
                    calculation_trace=result.trace,
                    token_usage=usage.to_dict(),
                    locator={
                        **dict(locator),
                        "selected_doc_ids": candidate_doc_ids,
                        "calculation_retrieval_rounds": retrieval_rounds,
                    },
                )
                artifact = self._attach_submission_reasoning(question, artifact)
                validate_b_answer(question, artifact.to_submission_answer())
                return artifact
            except Exception as exc:
                if isinstance(exc, BAnswerGenerationError):
                    raise BAnswerGenerationError(
                        str(exc),
                        token_usage=exc.token_usage,
                        diagnostics=[*diagnostics, *exc.diagnostics],
                    ) from exc
                last_error = exc
                diagnostics.append(
                    {
                        "attempt": attempt_number,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc)[:1000],
                        "plan": plan,
                    }
                )
                added_ids: list[str] = []
                retry_query = ""
                if attempt_number < 3:
                    retry_query = _calculation_retry_query(question, plan, exc)
                    retry_hits = self.retrievers[question.domain].search(
                        candidate_doc_ids,
                        retry_query,
                        top_k=max(12, self.calculation_top_k),
                        unit_type_boosts={
                            "metric_row": 2.4,
                            "formula_block": 2.0,
                            "clause_block": 1.5,
                            "article": 1.3,
                        },
                        ensure_per_doc=True,
                        expand_neighbors=True,
                    )
                    phrase_hits = _diagnostic_phrase_evidence(
                        self.retrievers[question.domain],
                        candidate_doc_ids,
                        f"{question.question}\n{retry_query}",
                        top_k=max(12, self.calculation_top_k),
                    )
                    evidence_items, added_ids = _merge_calculation_evidence(
                        evidence_items,
                        [
                            *phrase_hits,
                            *[_normalize_evidence(hit.to_dict()) for hit in retry_hits],
                        ],
                        max_items=1 + self.calculation_top_k * 3,
                    )
                    retrieval_round = {
                        "after_attempt": attempt_number,
                        "query": retry_query,
                        "phrase_overlay_count": len(phrase_hits),
                        "added_evidence_ids": added_ids,
                        "evidence_count": len(evidence_items),
                    }
                    retrieval_rounds.append(retrieval_round)
                    diagnostics[-1]["retrieval"] = retrieval_round
                feedback = (
                    f"上一次计划无法本地重放：{exc}。"
                    f"已按缺失变量定向补充 {len(added_ids)} 条新证据。"
                    "请重新检查全部证据、补齐变量并只输出完整 JSON。"
                )
        assert last_error is not None
        raise BAnswerGenerationError(
            f"Calculation failed after {len(diagnostics)} grounded replay attempts: {last_error}",
            token_usage=usage.to_dict(),
            diagnostics=diagnostics,
        ) from last_error

    def _attach_submission_reasoning(
        self,
        question: BQuestion,
        artifact: BAnswerArtifact,
    ) -> BAnswerArtifact:
        evidence_payload = [
            {
                "unit_id": str(item.get("unit_id", "")),
                "doc_id": str(item.get("doc_id", "")),
                "title": " > ".join(str(value) for value in item.get("title_path", [])),
                "text": str(item.get("text", ""))[:1800],
            }
            for item in artifact.evidence_items[:12]
        ]
        messages = [
            {"role": "system", "content": SUBMISSION_REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "qid": question.qid,
                        "question": question.question,
                        "options": question.options,
                        "answer_parts": artifact.answer_parts,
                        "existing_solution_summary": artifact.decision_summary,
                        "evidence": evidence_payload,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        response = self.client.chat_json(messages)
        combined_usage = _add_token_usage(artifact.token_usage, response.token_usage.to_dict())
        diagnostics = [
            {
                "stage": "submission_reasoning",
                "prompt_version": SUBMISSION_REASONING_PROMPT_VERSION,
                "response_preview": response.content[:1000],
            }
        ]
        try:
            payload = extract_json_object(response.content)
            answer_parts = payload.get("answer_parts")
            if not isinstance(answer_parts, list) or [str(item) for item in answer_parts] != artifact.answer_parts:
                raise ValueError("submission reasoning response changed answer_parts")
            reasoning = str(payload.get("reasoning", "")).strip()
            if len(re.sub(r"\s+", "", reasoning)) < 20:
                raise ValueError("submission reasoning is shorter than 20 non-whitespace characters")
            if response.token_usage.prompt_tokens <= 0 or response.token_usage.completion_tokens <= 0:
                raise ValueError("submission reasoning API response is missing positive raw usage")
        except Exception as exc:
            diagnostics[0]["error_type"] = exc.__class__.__name__
            diagnostics[0]["error"] = str(exc)[:1000]
            raise BAnswerGenerationError(
                f"Submission reasoning finalization failed: {exc}",
                token_usage=combined_usage,
                diagnostics=diagnostics,
            ) from exc

        artifact.decision_summary = reasoning
        artifact.token_usage = combined_usage
        artifact.decision_trace = {
            **artifact.decision_trace,
            "submission_reasoning": {
                "prompt_version": SUBMISSION_REASONING_PROMPT_VERSION,
                "model_name": self.config.model.model_name,
                "token_usage": response.token_usage.to_dict(),
                "answer_parts_preserved": True,
            },
        }
        return artifact

    def _build_fingerprint(self, questions: Sequence[BQuestion], workers: int) -> dict[str, Any]:
        public_model = {
            "model_name": self.config.model.model_name,
            "temperature": self.config.model.temperature,
            "api_base_sha256": hashlib.sha256(self.config.model.api_base.encode("utf-8")).hexdigest(),
        }
        strategy_payload = {
            "locator_attempt": self.attempt.to_dict(),
            "answer_strategy": read_json(self.strategy_path),
            "calculation_contract": {
                "prompt_sha256": hashlib.sha256(
                    CALCULATION_SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                "trace_schema_version": 2,
                "grounding_required": True,
                "iterative_retrieval_version": CALCULATION_RETRIEVAL_VERSION,
            },
        }
        return build_run_fingerprint(
            project_root=ROOT,
            arguments={
                "runner": RUNNER_VERSION,
                "locator_attempt_id": self.locator_attempt_id,
                "workers": workers,
                "calculation_top_k": self.calculation_top_k,
            },
            questions=[item.to_dict() for item in questions],
            parsed_path=self.parsed_root,
            index_path=self.index_root,
            strategy_payload=strategy_payload,
            strategy_path=self.strategy_path,
            model_settings=public_model,
        )

    def _prepare_run(self, run_dir: Path, fingerprint: Mapping[str, Any], *, force: bool) -> list[dict[str, Any]]:
        manifest_path = run_dir / "run_manifest.json"
        answers_path = run_dir / "answers.json"
        if run_dir.exists() and any(run_dir.iterdir()):
            if force:
                raise RunFingerprintError(
                    "force does not delete an existing B run; choose a new immutable run directory"
                )
            if not manifest_path.exists():
                raise RunFingerprintError("Existing B run has no run_manifest.json")
            manifest = read_json(manifest_path)
            validate_resume_fingerprint(manifest, dict(fingerprint))
            return read_json(answers_path) if answers_path.exists() else []
        ensure_dir(run_dir)
        write_json(
            manifest_path,
            {
                "run_id": run_dir.name,
                "runner": RUNNER_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "status": "running",
                "fingerprint": fingerprint,
                "model": {
                    "model_name": self.config.model.model_name,
                    "temperature": self.config.model.temperature,
                },
                "locator_attempt_id": self.locator_attempt_id,
            },
        )
        return []


def _migration_module():
    import scripts.run_b_board_migration_loop as migration

    return migration


def _find_locator_attempt(migration: Any, attempt_id: str):
    for item in migration.default_attempts():
        if item.attempt_id == attempt_id:
            return item
    raise ValueError(f"Unknown locator attempt: {attempt_id}")


def _question_row(question: BQuestion) -> dict[str, Any]:
    return {
        "qid": question.qid,
        "domain": question.domain,
        "split": "B",
        "question": question.question,
        "options": dict(question.options),
        "answer_format": question.answer_format,
        "type": question.type,
    }


def _normalize_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    if "unit_id" not in payload and "evidence_id" in payload:
        payload["unit_id"] = str(payload["evidence_id"])
    payload.setdefault("title_path", [])
    payload.setdefault("text", "")
    return payload


def _calculation_evidence_payload(
    evidence_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": item["unit_id"],
            "doc_id": item.get("doc_id", ""),
            "title": " > ".join(item.get("title_path", [])),
            "text": str(item.get("text", ""))[:5000],
        }
        for item in evidence_items
    ]


def _calculation_retry_query(
    question: BQuestion,
    plan: Mapping[str, Any] | None,
    error: Exception,
) -> str:
    summary = str((plan or {}).get("decision_summary", "")).strip()
    return "\n".join(
        item
        for item in (
            (summary or question.question)[:1500],
            str(error)[:800],
        )
        if item
    )


def _diagnostic_phrase_evidence(
    retriever: GenericBM25Retriever,
    doc_ids: Sequence[str],
    query: str,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    normalized_query = _compact_text(query)
    concept_groups: list[tuple[tuple[str, ...], int]] = []
    if "境外" in normalized_query or "分地区" in normalized_query:
        concept_groups.extend([(("境外",), 12), (("分地区",), 10), (("营业收入",), 4)])
    elif "资产负债率" in normalized_query:
        concept_groups.extend(
            [
                (("资产负债率",), 12),
                (("总负债", "负债合计"), 8),
                (("总资产", "资产总计"), 8),
            ]
        )
    elif "评估增值率" in normalized_query or "增值率" in normalized_query:
        concept_groups.extend([(("评估增值率", "增值率"), 12), (("评估基准日",), 5)])
    elif "营业收入" in normalized_query:
        concept_groups.extend([(("营业收入", "营业收入合计", "营业总收入"), 10)])
    if not concept_groups:
        return []

    years = set(re.findall(r"20\d{2}", normalized_query))
    dates = set(re.findall(r"20\d{2}年\d{1,2}月\d{1,2}日", normalized_query))
    entities = [
        item
        for item in ("冠鸿智能", "比亚迪", "宁德时代", "美的集团")
        if item in normalized_query
    ]
    rows_by_doc: dict[str, list[tuple[int, int, Mapping[str, Any]]]] = {}
    allowed_docs = set(doc_ids)
    for position, unit in enumerate(retriever.units):
        doc_id = str(unit.get("doc_id", ""))
        if doc_id not in allowed_docs:
            continue
        text = _compact_text(
            " ".join(str(item) for item in unit.get("title_path", []))
            + "\n"
            + str(unit.get("text", ""))
        )
        matched_groups = 0
        score = 0
        for alternatives, weight in concept_groups:
            if any(term in text for term in alternatives):
                matched_groups += 1
                score += weight
        if not matched_groups:
            continue
        score += 3 * sum(year in text for year in years)
        score += 8 * sum(date in text for date in dates)
        if any(year in doc_id for year in years):
            score += 5
        score += 6 * sum(entity in text for entity in entities)
        score += 2 if str(unit.get("unit_type", "")) == "metric_row" else 0
        rows_by_doc.setdefault(doc_id, []).append((score, -position, unit))

    selected: list[tuple[int, int, Mapping[str, Any]]] = []
    per_doc = max(1, min(3, top_k // max(1, len(rows_by_doc))))
    for doc_id in doc_ids:
        ranked = sorted(rows_by_doc.get(doc_id, []), reverse=True)
        selected.extend(ranked[:per_doc])
    selected.sort(reverse=True)
    evidence: list[dict[str, Any]] = []
    for score, _, unit in selected[:top_k]:
        evidence.append(
            {
                "unit_id": str(unit["unit_id"]),
                "doc_id": str(unit["doc_id"]),
                "title_path": list(unit.get("title_path", [])),
                "text": str(unit.get("text", "")),
                "score": float(1000 + score),
                "metadata": {
                    **dict(unit.get("metadata", {})),
                    "unit_type": unit.get("unit_type", ""),
                    "retrieval_source": CALCULATION_RETRIEVAL_VERSION,
                },
            }
        )
    return evidence


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).replace(",", "")


def _merge_calculation_evidence(
    existing: Sequence[Mapping[str, Any]],
    additions: Sequence[Mapping[str, Any]],
    *,
    max_items: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if max_items < 1:
        raise ValueError("max_items must be positive")
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    added_ids: list[str] = []
    for source, is_addition in ((existing, False), (additions, True)):
        for raw in source:
            item = dict(raw)
            unit_id = str(item.get("unit_id", "")).strip()
            if not unit_id or unit_id in seen or len(merged) >= max_items:
                continue
            seen.add(unit_id)
            merged.append(item)
            if is_addition:
                added_ids.append(unit_id)
    return merged, added_ids


def _artifact_from_dict(row: Mapping[str, Any]) -> BAnswerArtifact:
    return BAnswerArtifact(
        qid=str(row["qid"]),
        domain=str(row["domain"]),
        answer_format=str(row["answer_format"]),
        answer_slot_count=int(row["answer_slot_count"]),
        answer_parts=[str(item) for item in row.get("answer_parts", [])],
        used_evidence_ids=[str(item) for item in row.get("used_evidence_ids", [])],
        evidence_items=[dict(item) for item in row.get("evidence_items", [])],
        decision_summary=str(row.get("decision_summary", "")),
        decision_trace=dict(row.get("decision_trace", {})),
        calculation_trace=dict(row.get("calculation_trace", {})),
        token_usage={key: int(value) for key, value in dict(row.get("token_usage", {})).items()},
        locator=dict(row.get("locator", {})),
    )


def _sum_tokens(artifacts: Sequence[BAnswerArtifact]) -> dict[str, int]:
    prompt = sum(int(item.token_usage.get("prompt_tokens", 0)) for item in artifacts)
    completion = sum(int(item.token_usage.get("completion_tokens", 0)) for item in artifacts)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _failure_record(qid: str, exc: Exception) -> dict[str, Any]:
    record: dict[str, Any] = {
        "qid": qid,
        "error_type": exc.__class__.__name__,
        "error": str(exc)[:2000],
    }
    if isinstance(exc, BAnswerGenerationError):
        record["token_usage"] = dict(exc.token_usage)
        record["diagnostics"] = list(exc.diagnostics)
    return record


def _sum_failure_tokens(failures: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    prompt = sum(int(dict(item.get("token_usage") or {}).get("prompt_tokens", 0)) for item in failures)
    completion = sum(
        int(dict(item.get("token_usage") or {}).get("completion_tokens", 0))
        for item in failures
    )
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _add_token_usage(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    prompt = int(left.get("prompt_tokens", 0)) + int(right.get("prompt_tokens", 0))
    completion = int(left.get("completion_tokens", 0)) + int(right.get("completion_tokens", 0))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
