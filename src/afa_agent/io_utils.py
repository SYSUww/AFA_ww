from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    safe_text = text.encode("utf-8", errors="replace").decode("utf-8")
    path.write_text(safe_text, encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def timestamp_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def ensure_run_subdirs(run_dir: Path) -> dict[str, Path]:
    layout = {
        "meta": ensure_dir(run_dir / "meta"),
        "outputs": ensure_dir(run_dir / "outputs"),
        "submission": ensure_dir(run_dir / "outputs" / "submission"),
        "debug": ensure_dir(run_dir / "outputs" / "debug"),
        "by_type": ensure_dir(run_dir / "outputs" / "by_type"),
        "analysis": ensure_dir(run_dir / "analysis"),
    }
    return layout
