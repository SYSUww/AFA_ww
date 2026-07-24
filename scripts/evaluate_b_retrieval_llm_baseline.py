#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.candidate_scorecard import answer_parts_equivalent
from afa_agent.b_board.io import load_b_questions
from afa_agent.b_board.scoring import token_efficiency_score
from afa_agent.b_board.submission_policy import is_allowed_submission_model


DEFAULT_REFERENCE = (
    Path("/Users/abandon/Documents/AFA_ww")
    / "artifacts/b_board_actual/references"
    / "pseudo99_from_official98_ins016_bd_v1/reference_answers.json"
)
DEFAULT_REFERENCE_MANIFEST = DEFAULT_REFERENCE.parent / "reference_manifest.json"
DEFAULT_SOURCE_ROOT = Path("/Users/abandon/Documents/AFA_ww")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an answer-blind retrieval+Qwen run against pseudo99"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--question-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "upload_b/question_b",
    )
    parser.add_argument(
        "--submission-template",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "upload_b/submit.csv",
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=DEFAULT_REFERENCE_MANIFEST,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_b_questions(args.question_root, args.submission_template)
    question_by_qid = {question.qid: question for question in questions}
    run_manifest = _read_json(args.run_dir / "run_manifest.json")
    expected_qids = [
        str(qid) for qid in ((run_manifest.get("scope") or {}).get("qids") or [])
    ]
    if not expected_qids:
        expected_qids = [question.qid for question in questions]
    if len(expected_qids) != len(set(expected_qids)):
        raise ValueError("run scope contains duplicate qids")
    unknown_qids = sorted(set(expected_qids) - set(question_by_qid))
    if unknown_qids:
        raise ValueError(f"run scope contains unknown qids: {unknown_qids}")
    answers = _index_rows(_read_json(args.run_dir / "answers.json"))
    failures = _index_rows(_read_json(args.run_dir / "failures.json"))
    reference = _index_rows(_read_json(args.reference))
    reference_manifest = _read_json(args.reference_manifest)
    missing_reference = sorted(set(expected_qids) - set(reference))
    if missing_reference:
        raise ValueError(f"reference is missing run-scope qids: {missing_reference}")

    exact: list[str] = []
    equivalent: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for qid in expected_qids:
        candidate = answers.get(qid)
        reference_parts = list(reference[qid]["answer_parts"])
        if candidate is None:
            failure = failures.get(qid)
            mismatches.append(
                {
                    "qid": qid,
                    "domain": question_by_qid[qid].domain,
                    "answer_format": question_by_qid[qid].answer_format,
                    "candidate": None,
                    "reference": reference_parts,
                    "failure": failure,
                }
            )
            continue
        candidate_parts = list(candidate["answer_parts"])
        if candidate_parts == reference_parts:
            exact.append(qid)
        if answer_parts_equivalent(candidate_parts, reference_parts):
            equivalent.append(qid)
        else:
            mismatches.append(
                {
                    "qid": qid,
                    "domain": question_by_qid[qid].domain,
                    "answer_format": question_by_qid[qid].answer_format,
                    "candidate": candidate_parts,
                    "reference": reference_parts,
                    "retrieved_doc_ids": sorted(
                        {
                            str(item.get("doc_id", ""))
                            for item in candidate.get("evidence_items", [])
                            if item.get("doc_id")
                        }
                    ),
                }
            )

    compliance = _audit_generation(
        run_dir=args.run_dir,
        expected_qids=expected_qids,
        answers=answers,
        failures=failures,
        run_manifest=run_manifest,
    )
    by_domain = _segment_metrics(
        expected_qids,
        equivalent,
        lambda qid: question_by_qid[qid].domain,
    )
    by_format = _segment_metrics(
        expected_qids,
        equivalent,
        lambda qid: question_by_qid[qid].answer_format,
    )
    token_total = int((run_manifest.get("token_usage") or {}).get("total_tokens", 0))
    token_score_field = (
        "token_efficiency_score"
        if len(expected_qids) == len(questions)
        else "sample_token_efficiency_score_not_full_run_comparable"
    )
    result = {
        "score_type": "offline_pseudo99_reference_match_not_official_accuracy",
        "official_accuracy": None,
        "reference": {
            "label": reference_manifest.get("label"),
            "officially_submitted": bool(
                reference_manifest.get("officially_submitted", False)
            ),
            "predicted_accuracy": reference_manifest.get("predicted_accuracy"),
            "question_count": len(expected_qids),
            "reference_sha256": reference_manifest.get("reference_sha256"),
        },
        "run": {
            "run_dir": str(args.run_dir.resolve()),
            "status": run_manifest.get("status"),
            "model": (run_manifest.get("model") or {}).get("model_name"),
            "answered_question_count": len(answers),
            "failed_question_count": len(failures),
            "raw_call_count": run_manifest.get("raw_call_count"),
            "format_retry_count": run_manifest.get("format_retry_count"),
            "token_usage": run_manifest.get("token_usage"),
            token_score_field: token_efficiency_score(token_total),
        },
        "reference_exact_match_count": len(exact),
        "reference_equivalent_match_count": len(equivalent),
        "reference_equivalent_accuracy_percent": round(
            len(equivalent) / len(expected_qids) * 100,
            6,
        ),
        "reference_mismatch_count": len(mismatches),
        "reference_mismatches": mismatches,
        "by_domain": by_domain,
        "by_answer_format": by_format,
        "compliance": compliance,
        "notes": [
            "该准确率只表示与当前pseudo99候选的等价匹配率，不是官网准确率。",
            "生成进程不加载pseudo99、官网答案锁或历史提交；比较在冻结生成结果后单独执行。",
            f"缺失或失败题按不匹配计入本次{len(expected_qids)}题样本分母。",
            (
                "Token效率分按完整100题总Token计算。"
                if len(expected_qids) == len(questions)
                else "样本Token效率分仅按样本总Token代入，不能与100题正式提交直接比较。"
            ),
        ],
    }
    output = args.output or args.run_dir / "accuracy_evaluation.json"
    _write_json(output, result)
    _write_badcases(
        args.run_dir / "accuracy_badcases.md",
        mismatches,
        question_by_qid,
        result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _audit_generation(
    *,
    run_dir: Path,
    expected_qids: list[str],
    answers: dict[str, dict[str, Any]],
    failures: dict[str, dict[str, Any]],
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    problems: list[str] = []
    model_name = str((run_manifest.get("model") or {}).get("model_name", ""))
    if not is_allowed_submission_model(model_name):
        problems.append("model_not_allowed")
    if set(answers) | set(failures) != set(expected_qids):
        problems.append("qid_coverage_incomplete")
    call_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    for qid in expected_qids:
        call_path = run_dir / "raw_calls" / f"{qid}.json"
        if not call_path.exists():
            problems.append(f"{qid}:missing_raw_calls")
            continue
        raw_artifact = _read_json(call_path)
        calls = raw_artifact.get("calls", [])
        aliases = raw_artifact.get("evidence_alias_map", [])
        alias_names = [str(item.get("alias", "")) for item in aliases]
        source_ids = [str(item.get("source_evidence_id", "")) for item in aliases]
        if (
            not aliases
            or not all(alias_names)
            or len(alias_names) != len(set(alias_names))
            or not all(source_ids)
            or len(source_ids) != len(set(source_ids))
        ):
            problems.append(f"{qid}:invalid_evidence_alias_map")
        row = answers.get(qid) or failures.get(qid) or {}
        if aliases != row.get("evidence_alias_map"):
            problems.append(f"{qid}:evidence_alias_map_drift")
        call_count += len(calls)
        for call in calls:
            messages = call.get("messages")
            if qid in json.dumps(messages, ensure_ascii=False):
                problems.append(f"{qid}:qid_leaked_to_model")
            if call.get("model_name") != model_name:
                problems.append(f"{qid}:model_drift")
            if call.get("response_format_mode") != "native_json_schema_strict":
                problems.append(f"{qid}:structured_output_mode_drift")
            usage = call.get("token_usage") or {}
            prompt = int(usage.get("prompt_tokens", -1))
            completion = int(usage.get("completion_tokens", -1))
            total = int(usage.get("total_tokens", -1))
            if min(prompt, completion, total) < 0 or total != prompt + completion:
                problems.append(f"{qid}:invalid_raw_usage")
            prompt_tokens += max(0, prompt)
            completion_tokens += max(0, completion)
        if qid in answers:
            final_content = str(calls[-1].get("content", "")) if calls else ""
            try:
                final_payload = json.loads(final_content)
            except (json.JSONDecodeError, TypeError):
                final_payload = {}
                problems.append(f"{qid}:final_content_not_json")
            if (
                final_payload.get("answer_parts") != answers[qid].get("answer_parts")
                or final_payload.get("reasoning") != answers[qid].get("reasoning")
            ):
                problems.append(f"{qid}:submitted_payload_drift")
            expected_parts_hash = _sha256_text(
                json.dumps(
                    answers[qid].get("answer_parts", []),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            if raw_artifact.get("final_call_index") != len(calls):
                problems.append(f"{qid}:final_call_index_mismatch")
            if raw_artifact.get("final_content_sha256") != _sha256_text(final_content):
                problems.append(f"{qid}:final_content_hash_mismatch")
            if (
                raw_artifact.get("submitted_answer_parts_sha256")
                != expected_parts_hash
            ):
                problems.append(f"{qid}:submitted_answer_hash_mismatch")
            if raw_artifact.get("submitted_reasoning_sha256") != _sha256_text(
                str(answers[qid].get("reasoning", ""))
            ):
                problems.append(f"{qid}:submitted_reasoning_hash_mismatch")
            if raw_artifact.get("postprocessing") != {
                "answer_modified": False,
                "reasoning_modified": False,
                "csv_escaping_only": True,
            }:
                problems.append(f"{qid}:postprocessing_contract_mismatch")
    manifest_usage = run_manifest.get("token_usage") or {}
    if (
        prompt_tokens != int(manifest_usage.get("prompt_tokens", -1))
        or completion_tokens != int(manifest_usage.get("completion_tokens", -1))
        or prompt_tokens + completion_tokens
        != int(manifest_usage.get("total_tokens", -1))
    ):
        problems.append("manifest_usage_mismatch")
    if call_count != int(run_manifest.get("raw_call_count", -1)):
        problems.append("manifest_call_count_mismatch")
    prohibited_imports = _prohibited_generation_imports()
    problems.extend(f"prohibited_import:{item}" for item in prohibited_imports)
    return {
        "passed": not problems,
        "problems": problems,
        "model_allowed": is_allowed_submission_model(model_name),
        "qid_absent_from_model_messages": not any(
            problem.endswith("qid_leaked_to_model") for problem in problems
        ),
        "raw_usage_reconciled": "manifest_usage_mismatch" not in problems,
        "generation_prohibited_imports": prohibited_imports,
        "generation_reference_loaded": False,
        "solver_used": False,
        "fixed_locator_used": False,
    }


def _prohibited_generation_imports() -> list[str]:
    prohibited_fragments = (
        ".solver",
        "candidate_scorecard",
        "official_answer",
        "pseudo_accuracy",
    )
    imports: list[str] = []
    for path in (
        ROOT / "src/afa_agent/b_board/retrieval_llm_baseline.py",
        ROOT / "scripts/run_b_retrieval_llm_baseline.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(str(node.module or ""))
    return sorted(
        {
            name
            for name in imports
            if any(fragment in name for fragment in prohibited_fragments)
        }
    )


def _segment_metrics(
    expected_qids: list[str],
    equivalent_qids: list[str],
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    totals = Counter(key_fn(qid) for qid in expected_qids)
    matches = Counter(key_fn(qid) for qid in equivalent_qids)
    return {
        key: {
            "question_count": totals[key],
            "match_count": matches[key],
            "match_rate_percent": round(matches[key] / totals[key] * 100, 6),
        }
        for key in sorted(totals)
    }


def _write_badcases(
    path: Path,
    mismatches: list[dict[str, Any]],
    question_by_qid: dict[str, Any],
    result: dict[str, Any],
) -> None:
    lines = [
        "# Retrieval + Qwen 样本代理准确率 badcases",
        "",
        (
            f"- 与pseudo99等价匹配：{result['reference_equivalent_match_count']}/"
            f"{result['reference']['question_count']}"
        ),
        f"- 代理匹配率：{result['reference_equivalent_accuracy_percent']:.2f}%",
        "- 注意：这不是官网准确率。",
        "",
    ]
    for item in mismatches:
        question = question_by_qid[item["qid"]]
        lines.extend(
            [
                f"## {item['qid']}",
                "",
                f"- 领域/题型：`{item['domain']}` / `{item['answer_format']}`",
                f"- 题目：{question.question}",
                f"- 模型答案：`{item.get('candidate')}`",
                f"- pseudo99：`{item.get('reference')}`",
            ]
        )
        if item.get("failure"):
            failure = item["failure"]
            lines.append(
                f"- 失败：`{failure.get('error_type')}` {failure.get('error', '')}"
            )
        else:
            lines.append(f"- 检索文档：`{item.get('retrieved_doc_ids', [])}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _index_rows(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("expected a JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("answer/reference row must be an object")
        qid = str(row.get("qid", "")).strip()
        if not qid or qid in indexed:
            raise ValueError("answer/reference contains missing or duplicate qid")
        indexed[qid] = row
    return indexed


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
