from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def load_env(env_path: Path | None = None) -> dict[str, str]:
    env_file = env_path or (DEFAULT_ROOT / ".env")
    data: dict[str, str] = {}
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    for key, value in os.environ.items():
        data[key] = value
    return data


@dataclass(slots=True)
class ModelConfig:
    api_key: str
    api_base: str
    model_name: str
    temperature: float = 0.0
    timeout_seconds: int = 120
    connect_timeout_seconds: int = 20
    read_timeout_seconds: int = 120
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0


@dataclass(slots=True)
class RunConfig:
    project_root: Path = DEFAULT_ROOT
    artifacts_dir: Path = field(default_factory=lambda: DEFAULT_ROOT / "artifacts")
    model: ModelConfig | None = None

    def ensure_directories(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "manifest").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "parsed").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "index").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "runs").mkdir(parents=True, exist_ok=True)

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["model"]:
            payload["model"]["api_key"] = "<redacted>"
        payload["project_root"] = str(self.project_root)
        payload["artifacts_dir"] = str(self.artifacts_dir)
        return payload

    def to_public_json(self) -> str:
        return json.dumps(self.to_public_dict(), ensure_ascii=False, indent=2)


def build_run_config(project_root: Path | None = None) -> RunConfig:
    root = project_root or DEFAULT_ROOT
    env = load_env(root / ".env")
    model = None
    if env.get("LLM_API_KEY") and env.get("LLM_API_BASE") and env.get("LLM_MODEL"):
        timeout_seconds = _env_int(env, "LLM_TIMEOUT_SECONDS", 120)
        model = ModelConfig(
            api_key=env["LLM_API_KEY"],
            api_base=env["LLM_API_BASE"],
            model_name=env["LLM_MODEL"],
            temperature=_env_float(env, "LLM_TEMPERATURE", 0.0),
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=_env_int(env, "LLM_CONNECT_TIMEOUT_SECONDS", 20),
            read_timeout_seconds=_env_int(env, "LLM_READ_TIMEOUT_SECONDS", timeout_seconds),
            max_retries=_env_int(env, "LLM_MAX_RETRIES", 2),
            retry_backoff_seconds=_env_float(env, "LLM_RETRY_BACKOFF_SECONDS", 2.0),
        )
    config = RunConfig(project_root=root, artifacts_dir=root / "artifacts", model=model)
    config.ensure_directories()
    return config


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return default
