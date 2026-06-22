from __future__ import annotations

import csv
from pathlib import Path

from .io_utils import ensure_dir, write_json
from .models import AnswerResult


def export_answers_json(path: Path, results: list[AnswerResult]) -> None:
    write_json(path, [result.to_dict() for result in results])


def export_evidence_json(path: Path, results: list[AnswerResult]) -> None:
    payload = {
        result.qid: {
            "qid": result.qid,
            "pred_answer": result.pred_answer,
            "question_type": result.question_type,
            "option_labels": result.option_labels,
            "reasoning_summary": result.reasoning_summary,
            "evidence_items": result.evidence_items,
            "token_usage": result.token_usage.to_dict(),
            "debug_meta": result.debug_meta,
        }
        for result in results
    }
    write_json(path, payload)


def export_answer_csv(path: Path, results: list[AnswerResult]) -> None:
    ensure_dir(path.parent)
    prompt_tokens = sum(item.token_usage.prompt_tokens for item in results)
    completion_tokens = sum(item.token_usage.completion_tokens for item in results)
    total_tokens = sum(item.token_usage.total_tokens for item in results)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer": "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
        for result in results:
            writer.writerow(
                {
                    "qid": result.qid,
                    "answer": result.pred_answer,
                    "prompt_tokens": result.token_usage.prompt_tokens,
                    "completion_tokens": result.token_usage.completion_tokens,
                    "total_tokens": result.token_usage.total_tokens,
                }
            )


def export_grouped_results(base_dir: Path, results: list[AnswerResult]) -> None:
    grouped: dict[str, list[AnswerResult]] = {}
    for result in results:
        grouped.setdefault(result.question_type, []).append(result)
    for question_type, items in grouped.items():
        type_dir = base_dir / question_type
        ensure_dir(type_dir)
        export_answers_json(type_dir / "answers.json", items)
        export_evidence_json(type_dir / "evidence.json", items)
        (type_dir / "qids.txt").write_text(
            "\n".join(result.qid for result in items) + "\n",
            encoding="utf-8",
        )
