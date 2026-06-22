from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from afa_agent.io_utils import ensure_run_subdirs, write_json


def build_run_manifest(
    *,
    run_id: str,
    run_dir: Path,
    domain: str,
    split: str,
    question_count: int,
    qid: str,
    limit: int,
    plugin_name: str,
    strategy_label: str,
    strategy_details: list[str],
    model_name: str,
    resumed: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "domain": domain,
        "split": split,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "plugin_name": plugin_name,
        "generation_method": {
            "label": strategy_label,
            "details": strategy_details,
            "model_name": model_name,
        },
        "question_scope": {
            "question_count": question_count,
            "single_qid": qid or None,
            "limit": limit or None,
        },
        "resumed": resumed,
        "layout": {
            "meta": "meta/",
            "submission": "outputs/submission/",
            "debug": "outputs/debug/",
            "by_type": "outputs/by_type/",
            "analysis": "analysis/",
        },
    }


def write_run_readme(run_dir: Path, manifest: dict[str, Any]) -> None:
    text = "\n".join(
        [
            f"# Run {manifest['run_id']}",
            "",
            f"- domain: `{manifest['domain']}`",
            f"- split: `{manifest['split']}`",
            f"- method: `{manifest['generation_method']['label']}`",
            f"- model: `{manifest['generation_method']['model_name']}`",
            f"- question_count: `{manifest['question_scope']['question_count']}`",
            f"- resumed: `{manifest['resumed']}`",
            "",
            "## Layout",
            "",
            "- `meta/run_manifest.json`: 运行方式、领域、模型、题目范围",
            "- `meta/run_config.json`: 脱敏后的运行配置",
            "- `outputs/submission/answer.csv`: 提交文件",
            "- `outputs/debug/answers.json`: 全题答案",
            "- `outputs/debug/evidence.json`: 全题证据链",
            "- `outputs/debug/token_usage.json`: token 汇总",
            "- `outputs/debug/logs.jsonl`: 逐题原始结果",
            "- `outputs/by_type/<type>/`: 按题型拆分的结果",
            "- `analysis/metrics.json`: 汇总指标",
            "- `analysis/wrong_cases.json`: 错题位",
        ]
    )
    (run_dir / "README.md").write_text(text + "\n", encoding="utf-8")


def initialize_run_layout(run_dir: Path, manifest: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Path]:
    layout = ensure_run_subdirs(run_dir)
    write_json(layout["meta"] / "run_manifest.json", manifest)
    write_json(layout["meta"] / "run_config.json", run_config)
    write_run_readme(run_dir, manifest)
    return layout
