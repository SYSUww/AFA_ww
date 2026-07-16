from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from afa_agent.io_utils import ensure_run_subdirs, write_json


RUN_FINGERPRINT_SCHEMA_VERSION = 1


class RunFingerprintError(RuntimeError):
    """Raised when a run cannot be safely resumed."""


def _canonical_json_bytes(payload: Any) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return text.encode("utf-8", errors="replace")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_input_path(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Fingerprint input does not exist: {resolved}")
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "size_bytes": resolved.stat().st_size,
            "sha256": _file_sha256(resolved),
        }
    if not resolved.is_dir():
        raise ValueError(f"Fingerprint input is neither a file nor a directory: {resolved}")

    entries: list[dict[str, Any]] = []
    total_size = 0
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        size = child.stat().st_size
        total_size += size
        entries.append(
            {
                "path": child.relative_to(resolved).as_posix(),
                "size_bytes": size,
                "sha256": _file_sha256(child),
            }
        )
    return {
        "path": str(resolved),
        "kind": "directory",
        "size_bytes": total_size,
        "file_count": len(entries),
        "sha256": _payload_sha256(entries),
    }


def collect_git_state(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()

    def git(*args: str) -> bytes:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    try:
        commit = git("rev-parse", "HEAD").decode("utf-8").strip()
        branch = git("rev-parse", "--abbrev-ref", "HEAD").decode("utf-8").strip()
        tracked_diff = git("diff", "--binary", "HEAD", "--")
        untracked_output = git("ls-files", "--others", "--exclude-standard", "-z")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunFingerprintError(f"Unable to capture git state under {root}: {exc}") from exc

    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    untracked_paths: list[str] = []
    for raw_path in sorted(item for item in untracked_output.split(b"\0") if item):
        relative_path = raw_path.decode("utf-8", errors="surrogateescape")
        untracked_paths.append(relative_path)
        digest.update(b"\0untracked-path\0")
        digest.update(raw_path)
        file_path = root / relative_path
        if file_path.is_file():
            digest.update(b"\0untracked-content\0")
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)

    return {
        "branch": branch,
        "commit": commit,
        "dirty": bool(tracked_diff or untracked_paths),
        "dirty_diff_sha256": digest.hexdigest(),
        "untracked_paths": untracked_paths,
    }


def build_run_fingerprint(
    *,
    project_root: Path,
    arguments: dict[str, Any],
    questions: list[Any],
    parsed_path: Path,
    index_path: Path,
    strategy_payload: dict[str, Any],
    strategy_path: Path | None,
    model_settings: dict[str, Any],
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question_rows = [question.to_dict() if hasattr(question, "to_dict") else question for question in questions]
    qids = [str(row["qid"]) for row in question_rows]
    components = {
        "git": git_state or collect_git_state(project_root),
        "arguments": arguments,
        "questions": {
            "count": len(question_rows),
            "qids_sha256": _payload_sha256(qids),
            "content_sha256": _payload_sha256(question_rows),
        },
        "inputs": {
            "parsed": describe_input_path(parsed_path),
            "index": describe_input_path(index_path),
        },
        "strategy": {
            "path": str(strategy_path.resolve()) if strategy_path else None,
            "content_sha256": _payload_sha256(strategy_payload),
        },
        "model": model_settings,
    }
    return {
        "schema_version": RUN_FINGERPRINT_SCHEMA_VERSION,
        "sha256": _payload_sha256(components),
        "components": components,
    }


def validate_resume_fingerprint(existing_manifest: dict[str, Any], current_fingerprint: dict[str, Any]) -> None:
    existing = existing_manifest.get("fingerprint")
    if not isinstance(existing, dict):
        raise RunFingerprintError("Existing run manifest has no fingerprint; refusing unsafe resume")

    for label, fingerprint in (("existing", existing), ("current", current_fingerprint)):
        if fingerprint.get("schema_version") != RUN_FINGERPRINT_SCHEMA_VERSION:
            raise RunFingerprintError(f"Unsupported {label} run fingerprint schema")
        components = fingerprint.get("components")
        if not isinstance(components, dict) or fingerprint.get("sha256") != _payload_sha256(components):
            raise RunFingerprintError(f"The {label} run fingerprint is incomplete or has been modified")

    if existing["sha256"] == current_fingerprint["sha256"]:
        return

    existing_components = existing["components"]
    current_components = current_fingerprint["components"]
    changed = [
        key
        for key in sorted(set(existing_components) | set(current_components))
        if existing_components.get(key) != current_components.get(key)
    ]
    details = ", ".join(changed) if changed else "unknown components"
    raise RunFingerprintError(f"Run fingerprint mismatch ({details}); refusing unsafe resume")


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
    model_temperature: float | None = None,
    fingerprint: dict[str, Any] | None = None,
    invocation_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
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
            "temperature": model_temperature,
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
    if fingerprint is not None:
        manifest["fingerprint"] = fingerprint
    if invocation_args is not None:
        manifest["invocation_args"] = invocation_args
    return manifest


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
