from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    safe_text = text.encode("utf-8", errors="replace").decode("utf-8")
    _atomic_write_text(path, lambda handle: handle.write(safe_text))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    def write_rows(handle: TextIO) -> None:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    _atomic_write_text(path, write_rows)


def _atomic_write_text(path: Path, writer: Callable[[TextIO], object]) -> None:
    """Write a UTF-8 text file atomically without leaving partial checkpoints."""

    ensure_dir(path.parent)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


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
