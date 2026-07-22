#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.experiment_journal import ExperimentJournal
from afa_agent.b_board.io import load_b_questions, write_b_submission
from afa_agent.b_board.reasoning_evaluation import (
    PROMPT_VERSION as REASONING_EVALUATOR_VERSION,
    REASONING_JUDGE_MODEL,
    run_reasoning_evaluation,
)
from afa_agent.b_board.runner import (
    RUN_MODE_RESEARCH,
    SUBMISSION_REASONING_FEEDBACK_PROMPT_VERSION,
    SUBMISSION_REASONING_REFINE_PROMPT_VERSION,
    BAnswerArtifact,
    BBoardActualRunner,
    _artifact_from_dict,
)
from afa_agent.b_board.scoring import score_submission
from afa_agent.experiment_registry import ExperimentRegistry
from afa_agent.io_utils import ensure_dir, read_json, write_json


DEFAULT_TARGET_QIDS = (
    "fc_b_007",
    "fin_b_014",
    "reg_b_003",
    "reg_b_024",
    "res_b_008",
    "res_b_014",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one answer-frozen feedback/refine reasoning experiment"
    )
    parser.add_argument(
        "--base-run",
        default="artifacts/b_board_score_loop/full_chain_82d4492_research_v1",
    )
    parser.add_argument(
        "--amount-scale-run",
        default="artifacts/b_board_score_loop/amount_scale_a1_res_b012",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/b_board_score_loop/reasoning_self_refine_a1_lowtail6",
    )
    parser.add_argument("--target-qids", nargs="+", default=list(DEFAULT_TARGET_QIDS))
    parser.add_argument("--judge-workers", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_run = (ROOT / args.base_run).resolve()
    amount_scale_run = (ROOT / args.amount_scale_run).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    target_qids = tuple(dict.fromkeys(str(qid).strip() for qid in args.target_qids))
    candidate = _candidate(target_qids)
    journal = ExperimentJournal(
        registry=ExperimentRegistry(ROOT / "experiments/b_board_actual/experiment_registry.jsonl"),
        markdown_log_path=ROOT / "wiki/b_board_actual_loop_log.md",
    )
    review = journal.review(candidate)
    print(json.dumps({"history_review": review.to_dict()}, ensure_ascii=False), flush=True)
    if not review.executable:
        raise RuntimeError(
            "history review rejected this attempt: " + review.history_decision.reason
        )

    try:
        result = _run(
            base_run=base_run,
            amount_scale_run=amount_scale_run,
            output_dir=output_dir,
            target_qids=target_qids,
            judge_workers=max(1, args.judge_workers),
        )
    except Exception as exc:
        journal.append_result(
            review=review,
            candidate=candidate,
            result={
                "experiment_id": "b-loop-reasoning-self-refine-verification-a1-lowtail6",
                "status": "blocked_technical",
                "approach": "冻结6题答案，使用同一gpt-5.5先按new.md三维质检，再基于证据重写reasoning。",
                "effect": "实验未形成可比较的完整评分结果。",
                "failure_analysis": f"{exc.__class__.__name__}: {str(exc)[:1000]}",
                "next_step": "排除技术故障后重新审查日志；技术失败不计入3轮材料尝试上限。",
                "metrics": {"error_type": exc.__class__.__name__},
                "artifact_path": str(output_dir.relative_to(ROOT)),
                "submission_effect": "not_submitted",
            },
        )
        raise

    stored = journal.append_result(review=review, candidate=candidate, result=result)
    print(json.dumps({"experiment_result": stored}, ensure_ascii=False, indent=2), flush=True)


def _run(
    *,
    base_run: Path,
    amount_scale_run: Path,
    output_dir: Path,
    target_qids: tuple[str, ...],
    judge_workers: int,
) -> dict[str, Any]:
    questions = load_b_questions(ROOT / "upload_b/question_b", ROOT / "upload_b/submit.csv")
    question_by_qid = {item.qid: item for item in questions}
    missing = sorted(set(target_qids) - set(question_by_qid))
    if missing:
        raise ValueError(f"unknown target qids: {missing}")

    base_artifacts = {
        str(row["qid"]): _artifact_from_dict(row)
        for row in read_json(base_run / "answers.json")
    }
    amount_artifacts = {
        str(row["qid"]): _artifact_from_dict(row)
        for row in read_json(amount_scale_run / "answers.json")
    }
    if set(base_artifacts) != set(question_by_qid):
        raise ValueError("base artifact coverage does not match the 100-question dataset")
    base_artifacts["res_b_012"] = amount_artifacts["res_b_012"]

    runner = BBoardActualRunner(questions=questions, run_mode=RUN_MODE_RESEARCH)
    if runner.config.model.model_name != "gpt-5.5":
        raise ValueError(
            "self-refine generator is frozen to gpt-5.5; configured model is "
            f"{runner.config.model.model_name!r}"
        )

    refined: dict[str, BAnswerArtifact] = {}
    for index, qid in enumerate(target_qids, start=1):
        artifact = _artifact_from_dict(base_artifacts[qid].to_dict())
        refined[qid] = runner.refine_submission_reasoning(question_by_qid[qid], artifact)
        print(f"refined {index}/{len(target_qids)} {qid}", flush=True)

    answer_changes = {
        qid: {
            "before": base_artifacts[qid].answer_parts,
            "after": refined[qid].answer_parts,
        }
        for qid in target_qids
        if refined[qid].answer_parts != base_artifacts[qid].answer_parts
    }
    if answer_changes:
        raise ValueError(f"answer freeze gate failed: {answer_changes}")

    ensure_dir(output_dir)
    write_json(output_dir / "answers.json", [refined[qid].to_dict() for qid in target_qids])
    target_questions = [question_by_qid[qid] for qid in target_qids]
    write_b_submission(
        output_dir / "research_submit.csv",
        target_questions,
        [refined[qid].to_submission_answer() for qid in target_qids],
        audit_ready=False,
    )

    judge_model = replace(
        runner.config.model,
        model_name=REASONING_JUDGE_MODEL,
        temperature=0.0,
    )
    evaluation = run_reasoning_evaluation(
        submission_path=output_dir / "research_submit.csv",
        questions=target_questions,
        model_config=judge_model,
        output_dir=output_dir / "reasoning_eval",
        workers=judge_workers,
    )
    print(
        json.dumps({"target_reasoning_aggregate": evaluation.aggregate}, ensure_ascii=False),
        flush=True,
    )

    base_scores = {
        str(row["qid"]): float(row["reasoning_score"])
        for row in read_json(base_run / "reasoning_eval/reasoning_scores.json")
    }
    amount_score_rows = read_json(amount_scale_run / "reasoning_eval/reasoning_scores.json")
    base_scores["res_b_012"] = float(amount_score_rows[0]["reasoning_score"])
    candidate_scores = dict(base_scores)
    for qid, row in evaluation.evaluations.items():
        candidate_scores[qid] = row.reasoning_score

    base_token_total = sum(item.token_usage["total_tokens"] for item in base_artifacts.values())
    candidate_token_total = base_token_total + sum(
        refined[qid].token_usage["total_tokens"]
        - base_artifacts[qid].token_usage["total_tokens"]
        for qid in target_qids
    )
    baseline_scorecard = score_submission(
        accuracy_score=100.0,
        reasoning_scores=base_scores.values(),
        token_total=base_token_total,
    ).to_dict()
    candidate_scorecard = score_submission(
        accuracy_score=100.0,
        reasoning_scores=candidate_scores.values(),
        token_total=candidate_token_total,
    ).to_dict()
    target_before = [base_scores[qid] for qid in target_qids]
    target_after = [candidate_scores[qid] for qid in target_qids]
    total_delta = candidate_scorecard["total_score"] - baseline_scorecard["total_score"]
    target_delta = fmean(target_after) - fmean(target_before)
    status = "effective" if not answer_changes and total_delta > 0 else "rejected"

    causal_composite = {
        "scope": "full100_causal_composite_res_b_012_fixed_plus_6_refined",
        "base_run": str(base_run.relative_to(ROOT)),
        "amount_scale_replacement_run": str(amount_scale_run.relative_to(ROOT)),
        "target_qids": list(target_qids),
        "answer_changes": answer_changes,
        "target_reasoning_before": fmean(target_before),
        "target_reasoning_after": fmean(target_after),
        "target_reasoning_delta": target_delta,
        "baseline_scorecard": baseline_scorecard,
        "candidate_scorecard": candidate_scorecard,
        "total_score_delta": total_delta,
        "incumbent_official_aggregate_accuracy": 97.0,
        "incumbent_per_qid_truth_available": False,
    }
    write_json(output_dir / "causal_composite_scorecard.json", causal_composite)
    write_json(
        output_dir / "experiment_manifest.json",
        {
            "status": status,
            "generator_model": runner.config.model.model_name,
            "feedback_prompt_version": SUBMISSION_REASONING_FEEDBACK_PROMPT_VERSION,
            "refine_prompt_version": SUBMISSION_REASONING_REFINE_PROMPT_VERSION,
            "reasoning_evaluator_model": REASONING_JUDGE_MODEL,
            "reasoning_evaluator_version": REASONING_EVALUATOR_VERSION,
            "judge_receives_reasoning_only": True,
            "target_qids": list(target_qids),
            "causal_composite": causal_composite,
        },
    )

    return {
        "experiment_id": "b-loop-reasoning-self-refine-verification-a1-lowtail6",
        "status": status,
        "approach": "冻结6个reasoning低尾题的答案；gpt-5.5先按new.md的logical/completeness/clarity及题型检查项生成反馈，再使用同一模型依据原证据重写一次，完整累加两次API原始usage。",
        "effect": (
            f"目标6题reasoning均值 {fmean(target_before):.3f}→{fmean(target_after):.3f}"
            f"（{target_delta:+.3f}）；全100题因果代理总分 "
            f"{baseline_scorecard['total_score']:.6f}→{candidate_scorecard['total_score']:.6f}"
            f"（{total_delta:+.6f}），答案变化{len(answer_changes)}。"
        ),
        "failure_analysis": (
            "通过答案硬冻结、仅reasoning judge和完整usage记账约束；仍是基于97%提交构造的伪标签代理，不能证明逐题真实正确。"
            if status == "effective"
            else "目标集平均或全量加权总分未提升；下一轮需针对具体退化维度做材料性修改，不能原样重试。"
        ),
        "next_step": (
            "运行全量单测并提交推送有效分支；随后先复查日志，再决定是否以更严格选择门扩到更多低尾题。"
            if status == "effective"
            else "复查逐题三维分和反馈内容；若有明确可修正退化，再执行A2，最多共3轮。"
        ),
        "metrics": {
            **causal_composite,
            "target_reasoning_aggregate": evaluation.aggregate,
            "judge_token_usage": evaluation.manifest["judge_token_usage"],
            "judge_tokens_included_in_submission": False,
        },
        "artifact_path": str(output_dir.relative_to(ROOT)),
        "generator_model": runner.config.model.model_name,
        "evaluator_model": REASONING_JUDGE_MODEL,
        "evaluator_version": REASONING_EVALUATOR_VERSION,
        "submission_effect": "not_submitted",
        "sources": [
            "https://arxiv.org/abs/2303.17651",
            "https://arxiv.org/abs/2309.11495",
        ],
    }


def _candidate(target_qids: tuple[str, ...]) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    code_hash = hashlib.sha256(
        (ROOT / "src/afa_agent/b_board/runner.py").read_bytes()
    ).hexdigest()
    return {
        "direction_id": "reasoning_self_refine_verification",
        "pipeline_stage": "reasoning",
        "root_cause_cluster": "reasoning_verification_lowtail",
        "hypothesis": "对低尾reasoning先生成三维具体反馈再做一次证据约束重写，可在冻结答案且Token仍满分时提升最终加权总分",
        "change_vector": {
            "variant": "a1",
            "strategy": "two_call_feedback_then_refine",
            "answer_freeze_gate": True,
            "feedback_prompt_version": SUBMISSION_REASONING_FEEDBACK_PROMPT_VERSION,
            "refine_prompt_version": SUBMISSION_REASONING_REFINE_PROMPT_VERSION,
        },
        "material_delta": {"feedback_then_refine": True},
        "target_qids": list(target_qids),
        "question_types": [],
        "domains": [],
        "base_commit": head,
        "code_hash": code_hash,
        "generator_model": "gpt-5.5",
        "evaluator_model": REASONING_JUDGE_MODEL,
        "evaluator_version": REASONING_EVALUATOR_VERSION,
    }


if __name__ == "__main__":
    main()
