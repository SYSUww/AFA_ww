#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.config import build_run_config
from afa_agent.domains.registry import get_plugin
from afa_agent.exporters import export_answer_csv, export_answers_json, export_evidence_json
from afa_agent.exporters import export_grouped_results
from afa_agent.io_utils import ensure_run_subdirs, read_json, timestamp_id, write_json, write_jsonl
from afa_agent.models import Question
from afa_agent.run_metadata import build_run_manifest, initialize_run_layout


def load_questions(domain: str, split: str, qid_filter: set[str] | None = None) -> list[Question]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    question_path = Path(manifest["domains"][domain]["question_path"])
    rows = read_json(question_path)
    questions = []
    for row in rows:
        if row.get("split") != split:
            continue
        if qid_filter and row["qid"] not in qid_filter:
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
    parser.add_argument("--qid-file", default="")
    parser.add_argument("--strategy-config", default="")
    parser.add_argument("--parsed-path", default="")
    parser.add_argument("--index-path", default="")
    parser.add_argument("--run-root-dir", default="")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    if args.strategy_config:
        os.environ["AFA_STRATEGY_CONFIG"] = str(Path(args.strategy_config).resolve())

    qid_filter = None
    if args.qid_file:
        qid_filter = {
            line.strip()
            for line in Path(args.qid_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    questions = load_questions(args.domain, args.split, qid_filter=qid_filter)
    if args.qid:
        questions = [question for question in questions if question.qid == args.qid]
    if args.limit > 0:
        questions = questions[: args.limit]
    question_order = {
        question.qid: index
        for index, question in enumerate(load_questions(args.domain, args.split, qid_filter=qid_filter))
    }

    parsed_path = Path(args.parsed_path) if args.parsed_path else (ROOT / "artifacts" / "parsed" / args.domain / "parsed.json")
    index_path = Path(args.index_path) if args.index_path else (ROOT / "artifacts" / "index" / args.domain / "index.json")

    run_id = args.run_id or timestamp_id(f"{args.domain.lower()}_{args.split.lower()}")
    run_root_dir = Path(args.run_root_dir) if args.run_root_dir else (ROOT / "artifacts" / "runs")
    run_dir = Path(args.resume_run_dir) if args.resume_run_dir else (run_root_dir / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    plugin = get_plugin(args.domain)
    config = build_run_config()
    run_manifest = build_run_manifest(
        run_id=run_dir.name,
        run_dir=run_dir,
        domain=args.domain,
        split=args.split,
        question_count=len(questions),
        qid=args.qid,
        limit=args.limit,
        plugin_name=plugin.__class__.__name__,
        strategy_label=plugin.strategy_label,
        strategy_details=plugin.strategy_details,
        model_name=config.model.model_name if config.model else "unknown",
        resumed=bool(args.resume_run_dir),
    )
    if args.strategy_config:
        run_manifest["generation_method"]["strategy_config_path"] = str(Path(args.strategy_config).resolve())
    if args.qid_file:
        run_manifest["question_scope"]["qid_file"] = str(Path(args.qid_file).resolve())
    layout = initialize_run_layout(run_dir, run_manifest, config.to_public_dict())
    existing_results = []
    answers_path = layout["debug"] / "answers.json"
    if answers_path.exists():
        existing_results = read_json(answers_path)
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
        export_answers_json(layout["debug"] / "answers.json", [
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
    export_answers_json(layout["debug"] / "answers.json", proxy_results)
    export_evidence_json(layout["debug"] / "evidence.json", proxy_results)
    export_answer_csv(layout["submission"] / "answer.csv", proxy_results)
    export_grouped_results(layout["by_type"], proxy_results)
    write_json(
        layout["debug"] / "token_usage.json",
        {
            "prompt_tokens": sum(item.token_usage.prompt_tokens for item in proxy_results),
            "completion_tokens": sum(item.token_usage.completion_tokens for item in proxy_results),
            "total_tokens": sum(item.token_usage.total_tokens for item in proxy_results),
            "question_count": len(proxy_results),
        },
    )
    write_jsonl(layout["debug"] / "logs.jsonl", [item.to_dict() for item in proxy_results])
    print(run_dir)


if __name__ == "__main__":
    main()
