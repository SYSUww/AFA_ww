#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.config import build_run_config
from afa_agent.domains.regulatory import RegulatoryPlugin
from afa_agent.exporters import export_answer_csv, export_answers_json, export_evidence_json
from afa_agent.io_utils import read_json, timestamp_id, write_json, write_jsonl
from afa_agent.models import Question


def load_questions(domain: str, split: str) -> list[Question]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    question_path = Path(manifest["domains"][domain]["question_path"])
    rows = read_json(question_path)
    questions = []
    for row in rows:
        if row.get("split") != split:
            continue
        questions.append(
            Question(
                qid=row["qid"],
                domain=row["domain"],
                split=row["split"],
                question=row["question"],
                options=row["options"],
                answer_format=row["answer_format"],
                type=row["type"],
                doc_ids=row.get("doc_ids", []),
            )
        )
    return questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split", default="A")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume-run-dir", default="")
    parser.add_argument("--qid", default="")
    args = parser.parse_args()

    questions = load_questions(args.domain, args.split)
    if args.qid:
        questions = [question for question in questions if question.qid == args.qid]
    if args.limit > 0:
        questions = questions[: args.limit]
    question_order = {question.qid: index for index, question in enumerate(load_questions(args.domain, args.split))}

    parsed_path = ROOT / "artifacts" / "parsed" / args.domain / "parsed.json"
    index_path = ROOT / "artifacts" / "index" / args.domain / "index.json"

    if args.domain != "regulatory":
        raise ValueError(f"Unsupported domain for answering: {args.domain}")

    run_id = timestamp_id(f"{args.domain.lower()}_{args.split.lower()}")
    run_dir = Path(args.resume_run_dir) if args.resume_run_dir else (ROOT / "artifacts" / "runs" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_config.json", build_run_config().to_public_dict())

    plugin = RegulatoryPlugin()
    existing_results = []
    if (run_dir / "answers.json").exists():
        existing_results = read_json(run_dir / "answers.json")
    completed_qids = {row["qid"] for row in existing_results}
    if args.qid:
        existing_results = [row for row in existing_results if row["qid"] != args.qid]
        completed_qids = {row["qid"] for row in existing_results}
    results = existing_results[:]
    for question in questions:
        if question.qid in completed_qids:
            continue
        result = plugin.answer_one(question, parsed_path, index_path)
        results.append(result.to_dict())
        export_answers_json(run_dir / "answers.json", [
            type("AnswerProxy", (), {"to_dict": lambda self, row=row: row})() for row in results
        ])

    results.sort(key=lambda row: question_order.get(row["qid"], 10**9))

    class AnswerProxy:
        def __init__(self, row):
            self.row = row

        def to_dict(self):
            return self.row

        @property
        def qid(self):
            return self.row["qid"]

        @property
        def question_type(self):
            return self.row["question_type"]

        @property
        def pred_answer(self):
            return self.row["pred_answer"]

        @property
        def option_labels(self):
            return self.row["option_labels"]

        @property
        def evidence_items(self):
            return self.row["evidence_items"]

        @property
        def reasoning_summary(self):
            return self.row["reasoning_summary"]

        @property
        def token_usage(self):
            class TokenProxy:
                def __init__(self, payload):
                    self.prompt_tokens = payload["prompt_tokens"]
                    self.completion_tokens = payload["completion_tokens"]
                    self.total_tokens = payload["total_tokens"]

                def to_dict(self):
                    return {
                        "prompt_tokens": self.prompt_tokens,
                        "completion_tokens": self.completion_tokens,
                        "total_tokens": self.total_tokens,
                    }

            return TokenProxy(self.row["token_usage"])

        @property
        def debug_meta(self):
            return self.row["debug_meta"]

    proxy_results = [AnswerProxy(row) for row in results]
    export_answers_json(run_dir / "answers.json", proxy_results)
    export_evidence_json(run_dir / "evidence.json", proxy_results)
    export_answer_csv(run_dir / "answer.csv", proxy_results)
    write_json(
        run_dir / "token_usage.json",
        {
            "prompt_tokens": sum(item.token_usage.prompt_tokens for item in proxy_results),
            "completion_tokens": sum(item.token_usage.completion_tokens for item in proxy_results),
            "total_tokens": sum(item.token_usage.total_tokens for item in proxy_results),
            "question_count": len(proxy_results),
        },
    )
    write_jsonl(run_dir / "logs.jsonl", [item.to_dict() for item in proxy_results])
    print(run_dir)


if __name__ == "__main__":
    main()
