#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.exporters import export_grouped_results
from afa_agent.io_utils import ensure_run_subdirs, read_json
from afa_agent.domains.registry import PLUGIN_REGISTRY
from afa_agent.run_metadata import initialize_run_layout, build_run_manifest


def move_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src != dst:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> None:
    runs_dir = ROOT / "artifacts" / "runs"
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        parts = run_dir.name.split("_")
        domain = "_".join(parts[:-3]) if len(parts) >= 4 else run_dir.name
        split = parts[-3].upper() if len(parts) >= 4 else "A"
        config_path = first_existing(run_dir / "run_config.json", run_dir / "meta" / "run_config.json")
        config = read_json(config_path) if config_path else {}
        model_name = config.get("model", {}).get("model_name", "unknown")
        answers_path = first_existing(run_dir / "answers.json", run_dir / "outputs" / "debug" / "answers.json")
        answers_rows = read_json(answers_path) if answers_path.exists() else []
        question_count = len(answers_rows)
        plugin_cls = PLUGIN_REGISTRY.get(domain)
        strategy_label = plugin_cls.strategy_label if plugin_cls else "migrated-legacy-run"
        strategy_details = list(plugin_cls.strategy_details) if plugin_cls else ["由旧版平铺 run 目录迁移而来"]
        manifest = build_run_manifest(
            run_id=run_dir.name,
            run_dir=run_dir,
            domain=domain,
            split=split,
            question_count=question_count,
            qid="",
            limit=0,
            plugin_name=plugin_cls.__name__ if plugin_cls else "unknown",
            strategy_label=strategy_label,
            strategy_details=strategy_details,
            model_name=model_name,
            resumed=False,
        )
        layout = initialize_run_layout(run_dir, manifest, config)
        move_if_exists(run_dir / "run_config.json", layout["meta"] / "run_config.json")
        move_if_exists(run_dir / "answer.csv", layout["submission"] / "answer.csv")
        move_if_exists(run_dir / "answers.json", layout["debug"] / "answers.json")
        move_if_exists(run_dir / "evidence.json", layout["debug"] / "evidence.json")
        move_if_exists(run_dir / "logs.jsonl", layout["debug"] / "logs.jsonl")
        move_if_exists(run_dir / "token_usage.json", layout["debug"] / "token_usage.json")
        move_if_exists(run_dir / "metrics.json", layout["analysis"] / "metrics.json")
        move_if_exists(run_dir / "wrong_cases.json", layout["analysis"] / "wrong_cases.json")
        migrated_answers_path = layout["debug"] / "answers.json"
        if migrated_answers_path.exists():
            rows = read_json(migrated_answers_path)

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
                            self.payload = payload

                        def to_dict(self):
                            return self.payload

                    return TokenProxy(self.row["token_usage"])

                @property
                def debug_meta(self):
                    return self.row["debug_meta"]

            export_grouped_results(layout["by_type"], [AnswerProxy(row) for row in rows])
        print(run_dir)


if __name__ == "__main__":
    main()
