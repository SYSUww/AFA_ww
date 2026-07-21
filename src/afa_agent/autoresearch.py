from __future__ import annotations

import math
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from afa_agent.io_utils import ensure_dir, read_json, timestamp_id, write_json, write_jsonl
from afa_agent.strategy import deep_merge, load_strategy_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = ROOT / "artifacts" / "manifest" / "dataset_manifest.json"
DEFAULT_AUTORESEARCH_DIR = ROOT / "artifacts" / "autoresearch"
QUESTION_DOMAINS = [
    "regulatory",
    "financial_reports",
    "insurance",
    "research",
    "financial_contracts",
]
PROXY_WEIGHTS = {
    "format_valid": 0.20,
    "evidence_coverage": 0.20,
    "multi_doc_coverage": 0.15,
    "retrieval_focus": 0.15,
    "consistency": 0.15,
    "rule_alignment": 0.10,
    "token_efficiency": 0.05,
}


def load_plan_config(plan_config_path: Path) -> dict[str, Any]:
    return read_json(plan_config_path)


def load_dataset_slices(path: Path) -> dict[str, Any]:
    return read_json(path)


def load_candidate_sets(path: Path) -> dict[str, Any]:
    return read_json(path)


def normalize_domains(domains: list[str] | None) -> list[str]:
    if not domains:
        return QUESTION_DOMAINS[:]
    if len(domains) == 1 and domains[0] == "all":
        return QUESTION_DOMAINS[:]
    return domains


def load_questions_by_domain(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    payload: dict[str, list[dict[str, Any]]] = {}
    for domain, domain_manifest in manifest["domains"].items():
        rows = read_json(Path(domain_manifest["question_path"]))
        payload[domain] = rows
    return payload


def build_slice_qids(
    slice_name: str,
    domains: list[str],
    dataset_slices_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, list[str]]:
    slices = load_dataset_slices(dataset_slices_path)
    questions_by_domain = load_questions_by_domain(manifest_path)
    if slice_name not in slices:
        raise ValueError(f"Unknown dataset slice: {slice_name}")
    selected: dict[str, list[str]] = {}
    for domain in domains:
        domain_value = slices[slice_name].get(domain)
        if domain_value == "__all__":
            selected[domain] = [
                row["qid"]
                for row in questions_by_domain[domain]
                if row.get("split") == "A"
            ]
        else:
            selected[domain] = domain_value or []
    return selected


def write_slice_files(experiment_dir: Path, qids_by_domain: dict[str, list[str]]) -> dict[str, Path]:
    slice_dir = ensure_dir(experiment_dir / "dataset_slice")
    paths: dict[str, Path] = {}
    for domain, qids in qids_by_domain.items():
        path = slice_dir / f"{domain}_qids.txt"
        path.write_text("\n".join(qids) + ("\n" if qids else ""), encoding="utf-8")
        paths[domain] = path
    return paths


def run_command(cmd: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def run_pipeline_for_candidate(
    *,
    domain: str,
    qid_file: Path,
    strategy_config_path: Path,
    candidate_dir: Path,
) -> Path:
    parsed_path = candidate_dir / "parsed" / f"{domain}.json"
    index_path = candidate_dir / "index" / f"{domain}.json"
    runs_dir = candidate_dir / "runs"
    run_id = f"{domain}_a_{candidate_dir.name}"

    run_command(
        [
            "python3",
            "scripts/parse_domain.py",
            "--domain",
            domain,
            "--strategy-config",
            str(strategy_config_path),
            "--output-path",
            str(parsed_path),
        ]
    )
    run_command(
        [
            "python3",
            "scripts/build_index.py",
            "--domain",
            domain,
            "--strategy-config",
            str(strategy_config_path),
            "--parsed-path",
            str(parsed_path),
            "--output-path",
            str(index_path),
        ]
    )
    run_command(
        [
            "python3",
            "scripts/run_answering.py",
            "--domain",
            domain,
            "--split",
            "A",
            "--strategy-config",
            str(strategy_config_path),
            "--parsed-path",
            str(parsed_path),
            "--index-path",
            str(index_path),
            "--run-root-dir",
            str(runs_dir),
            "--run-id",
            run_id,
            "--qid-file",
            str(qid_file),
        ]
    )
    return runs_dir / run_id


def build_failed_candidate_result(
    *,
    candidate_id: str,
    stage: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "avg_proxy_score": -1.0,
        "invalid_answer_rate": 1.0,
        "avg_total_tokens": 0.0,
        "domain_breakdown": [],
        "question_logs": [],
        "disqualified": True,
        "error_message": error_message,
    }


def _is_valid_answer(answer: str, question: dict[str, Any]) -> bool:
    allowed = set(question["options"].keys())
    if question["answer_format"] == "tf":
        return answer in {"A", "B"}
    if question["answer_format"] == "mcq":
        return len(answer) == 1 and answer in allowed
    if question["answer_format"] == "multi":
        return bool(answer) and all(ch in allowed for ch in answer) and "".join(sorted(set(answer))) == answer
    return bool(answer)


def _evidence_doc_coverage(evidence_items: list[dict[str, Any]], question: dict[str, Any]) -> float:
    if not evidence_items:
        return 0.0
    allowed_docs = set(question.get("doc_ids", []))
    if not allowed_docs:
        return 1.0
    matched = [item for item in evidence_items if item.get("doc_id") in allowed_docs]
    return len(matched) / max(1, len(evidence_items))


def _multi_doc_coverage(evidence_items: list[dict[str, Any]], question: dict[str, Any]) -> float:
    doc_ids = list(dict.fromkeys(question.get("doc_ids", [])))
    if len(doc_ids) <= 1:
        return 1.0
    if not evidence_items:
        return 0.0
    evidence_docs = {item.get("doc_id") for item in evidence_items if item.get("doc_id")}
    return len(evidence_docs & set(doc_ids)) / len(doc_ids)


def _retrieval_focus(question_log: dict[str, Any]) -> float:
    retrieval_topk = question_log.get("retrieval_topk", [])
    if not retrieval_topk:
        return 0.0
    preferred = {"metric_row", "formula_block", "element_block", "conclusion_block", "article", "article_chunk", "clause_block"}
    boosted = 0
    for item in retrieval_topk:
        unit_type = item.get("unit_type") or item.get("metadata", {}).get("unit_type") or ""
        if unit_type in preferred:
            boosted += 1
    return boosted / len(retrieval_topk)


def _answer_consistency(question_log: dict[str, Any]) -> float:
    values = question_log.get("consistency_answers", [])
    if not values:
        return 1.0
    most_common = max(values.count(item) for item in set(values))
    return most_common / len(values)


def _rule_alignment(question_log: dict[str, Any], pred_answer: str) -> float:
    rule_outputs = question_log.get("rule_outputs", [])
    if not rule_outputs:
        return 0.8
    aligned = 0
    total = 0
    for item in rule_outputs:
        label = item.get("label")
        confidence = float(item.get("confidence", 0.0))
        if label is None or confidence < 0.5:
            continue
        total += 1
        if item.get("answer") == pred_answer:
            aligned += 1
    if total == 0:
        return 0.8
    return aligned / total


def _token_efficiency(total_tokens: int) -> float:
    if total_tokens <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (math.log10(total_tokens + 1) / 6.0)))


def _empty_penalty(pred_answer: str, evidence_items: list[dict[str, Any]], reasoning_summary: str) -> float:
    if not pred_answer or not evidence_items:
        return 0.25
    if "无法判断" in reasoning_summary or "证据不足" in reasoning_summary:
        return 0.15
    return 0.0


def score_run_proxy(
    *,
    run_dir: Path,
    domain: str,
    qids: list[str],
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    questions = {
        row["qid"]: row
        for row in load_questions_by_domain(manifest_path)[domain]
        if row["qid"] in qids
    }
    answers = read_json(run_dir / "outputs" / "debug" / "answers.json")
    question_logs: list[dict[str, Any]] = []
    invalid_count = 0
    total_tokens = 0
    by_type_scores: dict[str, list[float]] = defaultdict(list)

    for row in answers:
        question = questions[row["qid"]]
        pred_answer = row["pred_answer"]
        evidence_items = row.get("evidence_items", [])
        debug_meta = row.get("debug_meta", {})
        reasoning_summary = row.get("reasoning_summary", "")
        token_usage = row.get("token_usage", {})
        total_tokens += int(token_usage.get("total_tokens", 0))
        format_valid = 1.0 if _is_valid_answer(pred_answer, question) else 0.0
        if not format_valid:
            invalid_count += 1
        evidence_coverage = _evidence_doc_coverage(evidence_items, question)
        multi_doc_coverage = _multi_doc_coverage(evidence_items, question)
        retrieval_focus = _retrieval_focus(debug_meta)
        consistency = _answer_consistency(debug_meta)
        rule_alignment = _rule_alignment(debug_meta, pred_answer)
        token_efficiency = _token_efficiency(int(token_usage.get("total_tokens", 0)))
        penalty = _empty_penalty(pred_answer, evidence_items, reasoning_summary)
        proxy_score = (
            PROXY_WEIGHTS["format_valid"] * format_valid
            + PROXY_WEIGHTS["evidence_coverage"] * evidence_coverage
            + PROXY_WEIGHTS["multi_doc_coverage"] * multi_doc_coverage
            + PROXY_WEIGHTS["retrieval_focus"] * retrieval_focus
            + PROXY_WEIGHTS["consistency"] * consistency
            + PROXY_WEIGHTS["rule_alignment"] * rule_alignment
            + PROXY_WEIGHTS["token_efficiency"] * token_efficiency
            - penalty
        )
        type_name = question.get("type", question.get("answer_format", "unknown"))
        by_type_scores[type_name].append(proxy_score)
        question_logs.append(
            {
                "qid": row["qid"],
                "domain": domain,
                "question_type": question["answer_format"],
                "question_label": type_name,
                "pred_answer": pred_answer,
                "query_variants": debug_meta.get("query_variants", []),
                "retrieval_topk": debug_meta.get("retrieval_topk", []),
                "selected_evidence_ids": debug_meta.get("selected_evidence_ids", []),
                "rule_outputs": debug_meta.get("rule_outputs", []),
                "prompt_template_id": debug_meta.get("prompt_template_id", "default"),
                "token_usage": token_usage,
                "proxy_subscores": {
                    "format_valid": format_valid,
                    "evidence_coverage": evidence_coverage,
                    "multi_doc_coverage": multi_doc_coverage,
                    "retrieval_focus": retrieval_focus,
                    "consistency": consistency,
                    "rule_alignment": rule_alignment,
                    "token_efficiency": token_efficiency,
                    "empty_or_fallback_penalty": penalty,
                    "proxy_score": proxy_score,
                },
            }
        )

    proxy_scores = [row["proxy_subscores"]["proxy_score"] for row in question_logs]
    avg_proxy_score = sum(proxy_scores) / max(1, len(proxy_scores))
    invalid_answer_rate = invalid_count / max(1, len(question_logs))
    aggregate = {
        "domain": domain,
        "question_count": len(question_logs),
        "avg_proxy_score": avg_proxy_score,
        "invalid_answer_rate": invalid_answer_rate,
        "avg_total_tokens": total_tokens / max(1, len(question_logs)),
        "total_tokens": total_tokens,
        "by_type": {
            type_name: sum(values) / len(values)
            for type_name, values in by_type_scores.items()
        },
        "questions": question_logs,
        "disqualified": invalid_answer_rate > 0.25,
    }
    return aggregate


def score_candidate(
    *,
    candidate_id: str,
    stage: str,
    candidate_dir: Path,
    qids_by_domain: dict[str, list[str]],
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    domains_payload = []
    all_question_logs: list[dict[str, Any]] = []
    for domain, qids in qids_by_domain.items():
        run_dir = candidate_dir / "runs" / f"{domain}_a_{candidate_dir.name}"
        domain_payload = score_run_proxy(run_dir=run_dir, domain=domain, qids=qids, manifest_path=manifest_path)
        domains_payload.append(domain_payload)
        for row in domain_payload["questions"]:
            enriched = dict(row)
            enriched["candidate_id"] = candidate_id
            enriched["stage"] = stage
            all_question_logs.append(enriched)

    avg_proxy = sum(item["avg_proxy_score"] for item in domains_payload) / max(1, len(domains_payload))
    invalid_rate = sum(item["invalid_answer_rate"] for item in domains_payload) / max(1, len(domains_payload))
    avg_tokens = sum(item["avg_total_tokens"] for item in domains_payload) / max(1, len(domains_payload))
    disqualified = any(item["disqualified"] for item in domains_payload)
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "avg_proxy_score": -1.0 if disqualified else avg_proxy,
        "invalid_answer_rate": invalid_rate,
        "avg_total_tokens": avg_tokens,
        "domain_breakdown": domains_payload,
        "question_logs": all_question_logs,
        "disqualified": disqualified,
    }


def rank_candidates(scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        scored_candidates,
        key=lambda item: (
            item["avg_proxy_score"],
            -item["invalid_answer_rate"],
            -item["avg_total_tokens"],
        ),
        reverse=True,
    )


def update_leaderboard(base_dir: Path, stage: str, ranking: list[dict[str, Any]], experiment_id: str) -> None:
    leaderboard_path = base_dir / "leaderboard.json"
    leaderboard = read_json(leaderboard_path) if leaderboard_path.exists() else {"stages": {}}
    best = ranking[0] if ranking else {}
    leaderboard["stages"][stage] = {
        "experiment_id": experiment_id,
        "candidate_id": best.get("candidate_id"),
        "avg_proxy_score": best.get("avg_proxy_score"),
        "invalid_answer_rate": best.get("invalid_answer_rate"),
        "avg_total_tokens": best.get("avg_total_tokens"),
    }
    write_json(leaderboard_path, leaderboard)


def build_experiment_manifest(
    *,
    experiment_id: str,
    branch_name: str,
    commit_hash: str,
    stage: str,
    domains: list[str],
    dataset_slice: str,
    proxy_objective_version: str,
    candidate_ids: list[str],
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "branch_name": branch_name,
        "commit_hash": commit_hash,
        "stage": stage,
        "domains": domains,
        "dataset_slice": dataset_slice,
        "proxy_objective_version": proxy_objective_version,
        "candidate_ids": candidate_ids,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_git_branch_and_commit() -> tuple[str, str]:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return branch, commit


def run_experiment(
    *,
    stage: str,
    domains: list[str],
    dataset_slice: str,
    baseline_config_path: Path,
    candidate_set_path: Path,
    dataset_slices_path: Path,
    output_root: Path = DEFAULT_AUTORESEARCH_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    base_strategy = load_strategy_config(baseline_config_path)
    candidate_sets = load_candidate_sets(candidate_set_path)
    if stage not in candidate_sets["stage_candidates"]:
        raise ValueError(f"Candidate set does not contain stage: {stage}")
    domains = normalize_domains(domains)
    experiment_id = timestamp_id(f"{stage}_{'all_domains' if len(domains) == len(QUESTION_DOMAINS) else '_'.join(domains)}")
    experiment_dir = ensure_dir(output_root / "experiments" / experiment_id)
    qids_by_domain = build_slice_qids(dataset_slice, domains, dataset_slices_path, manifest_path)
    slice_files = write_slice_files(experiment_dir, qids_by_domain)
    branch_name, commit_hash = get_git_branch_and_commit()

    candidates = candidate_sets["stage_candidates"][stage]
    manifest = build_experiment_manifest(
        experiment_id=experiment_id,
        branch_name=branch_name,
        commit_hash=commit_hash,
        stage=stage,
        domains=domains,
        dataset_slice=dataset_slice,
        proxy_objective_version=base_strategy.get("proxy_objective_version", "proxy_v1"),
        candidate_ids=[item["candidate_id"] for item in candidates],
    )
    write_json(experiment_dir / "experiment_manifest.json", manifest)
    write_json(
        experiment_dir / "candidate_config.json",
        {
            "baseline_config_path": str(baseline_config_path),
            "candidates": [
                {
                    "candidate_id": item["candidate_id"],
                    "strategy_override": item["strategy_override"],
                }
                for item in candidates
            ],
        },
    )

    scored_candidates: list[dict[str, Any]] = []
    for item in candidates:
        candidate_id = item["candidate_id"]
        candidate_dir = ensure_dir(experiment_dir / "candidates" / candidate_id)
        candidate_strategy = deep_merge(base_strategy, item["strategy_override"])
        candidate_strategy["strategy_id"] = candidate_id
        strategy_path = candidate_dir / "candidate_config.json"
        write_json(strategy_path, candidate_strategy)
        for domain in domains:
            run_pipeline_for_candidate(
                domain=domain,
                qid_file=slice_files[domain],
                strategy_config_path=strategy_path,
                candidate_dir=candidate_dir,
            )
        scored = score_candidate(
            candidate_id=candidate_id,
            stage=stage,
            candidate_dir=candidate_dir,
            qids_by_domain=qids_by_domain,
            manifest_path=manifest_path,
        )
        scored_candidates.append(scored)
        write_json(candidate_dir / "aggregate_metrics.json", scored)
        write_jsonl(candidate_dir / "question_logs.jsonl", scored["question_logs"])
        notes = [
            f"# Candidate {candidate_id}",
            "",
            f"- stage: `{stage}`",
            f"- avg_proxy_score: `{scored['avg_proxy_score']}`",
            f"- invalid_answer_rate: `{scored['invalid_answer_rate']}`",
            f"- avg_total_tokens: `{scored['avg_total_tokens']}`",
            f"- disqualified: `{scored['disqualified']}`",
        ]
        (candidate_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    ranking = rank_candidates(scored_candidates)
    gap = 0.0
    if len(ranking) >= 2:
        gap = ranking[0]["avg_proxy_score"] - ranking[1]["avg_proxy_score"]
    ranking_payload = {
        "experiment_id": experiment_id,
        "stage": stage,
        "dataset_slice": dataset_slice,
        "ranking": ranking,
        "winner_gap": gap,
    }
    write_json(experiment_dir / "ranking.json", ranking_payload)
    write_json(
        experiment_dir / "aggregate_metrics.json",
        {
            "candidate_count": len(scored_candidates),
            "best_candidate_id": ranking[0]["candidate_id"] if ranking else None,
            "winner_gap": gap,
            "avg_proxy_scores": {item["candidate_id"]: item["avg_proxy_score"] for item in ranking},
        },
    )
    notes = [
        f"# Experiment {experiment_id}",
        "",
        f"- stage: `{stage}`",
        f"- domains: `{', '.join(domains)}`",
        f"- dataset_slice: `{dataset_slice}`",
        f"- candidate_count: `{len(scored_candidates)}`",
        f"- best_candidate: `{ranking[0]['candidate_id'] if ranking else ''}`",
        f"- winner_gap: `{gap}`",
    ]
    (experiment_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    best_candidate = ranking[0] if ranking else {}
    if best_candidate:
        best_path = experiment_dir / "best_candidate.json"
        write_json(
            best_path,
            {
                "experiment_id": experiment_id,
                "stage": stage,
                "candidate_id": best_candidate["candidate_id"],
                "candidate_config_path": str(experiment_dir / "candidates" / best_candidate["candidate_id"] / "candidate_config.json"),
                "avg_proxy_score": best_candidate["avg_proxy_score"],
                "invalid_answer_rate": best_candidate["invalid_answer_rate"],
                "avg_total_tokens": best_candidate["avg_total_tokens"],
            },
        )
        update_leaderboard(output_root, stage, ranking, experiment_id)
    return {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "ranking": ranking,
        "best_candidate_config_path": str(experiment_dir / "candidates" / ranking[0]["candidate_id"] / "candidate_config.json") if ranking else "",
    }


def run_loop_plan(plan_config_path: Path) -> dict[str, Any]:
    plan = load_plan_config(plan_config_path)
    execution = plan.get("execution", {}) or {}
    if execution.get("runner") == "b_actual_open_loop":
        from afa_agent.b_board.orchestrator import run_b_actual_loop_plan

        return run_b_actual_loop_plan(plan, plan_config_path)
    if execution.get("compatible_with_run_loop_engine") is False:
        runner = execution.get("runner", "a dedicated runner")
        raise ValueError(
            f"Plan {plan_config_path} declares runner {runner!r} and cannot be executed "
            "by the generic loop engine"
        )
    output_root = ROOT / plan.get("output_root", "artifacts/autoresearch")
    stages = plan["stages"]
    baseline_config_path = ROOT / plan["baseline_config"]
    candidate_set_path = ROOT / plan["candidate_set"]
    dataset_slices_path = ROOT / plan["dataset_slices"]
    domains = normalize_domains(plan.get("domains"))
    validation_slice = plan.get("validation_slice", "full_group_a")
    results: list[dict[str, Any]] = []
    current_baseline = baseline_config_path

    for stage in stages:
        stage_slice = plan.get("stage_dataset_overrides", {}).get(stage, plan.get("default_stage_slice", "dev_mini"))
        stage_result = run_experiment(
            stage=stage,
            domains=domains,
            dataset_slice=stage_slice,
            baseline_config_path=current_baseline,
            candidate_set_path=candidate_set_path,
            dataset_slices_path=dataset_slices_path,
            output_root=output_root,
        )
        results.append(
            {
                "stage": stage,
                "search_experiment_id": stage_result["experiment_id"],
                "search_experiment_dir": stage_result["experiment_dir"],
                "best_candidate_config_path": stage_result["best_candidate_config_path"],
            }
        )
        current_baseline = Path(stage_result["best_candidate_config_path"])

        validation_result = run_experiment(
            stage=stage,
            domains=domains,
            dataset_slice=validation_slice,
            baseline_config_path=current_baseline,
            candidate_set_path=_write_single_candidate_set(output_root, current_baseline, stage),
            dataset_slices_path=dataset_slices_path,
            output_root=output_root,
        )
        results[-1]["validation_experiment_id"] = validation_result["experiment_id"]
        results[-1]["validation_experiment_dir"] = validation_result["experiment_dir"]

    loop_id = timestamp_id("loop")
    loop_dir = ensure_dir(output_root / "loops" / loop_id)
    payload = {
        "loop_id": loop_id,
        "plan_config_path": str(plan_config_path),
        "final_baseline_config_path": str(current_baseline),
        "stages": results,
    }
    write_json(loop_dir / "loop_summary.json", payload)
    return payload


def _write_single_candidate_set(output_root: Path, config_path: Path, stage: str) -> Path:
    single_path = ensure_dir(output_root / "generated") / f"{stage}_single_candidate_set.json"
    payload = {
        "stage_candidates": {
            stage: [
                {
                    "candidate_id": f"validate_{config_path.stem}",
                    "strategy_override": {},
                }
            ]
        }
    }
    write_json(single_path, payload)
    return single_path
