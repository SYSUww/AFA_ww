from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
STRUCTURED_OUTPUT_LOCAL = "json_object_local_schema"
STRUCTURED_OUTPUT_NATIVE = "native_json_schema_strict"
STRUCTURED_OUTPUT_MODES = (
    STRUCTURED_OUTPUT_LOCAL,
    STRUCTURED_OUTPUT_NATIVE,
)
_MODEL_ENV_FIELDS = {
    "LLM": {
        "api_key": "LLM_API_KEY",
        "api_base": "LLM_API_BASE",
        "model_name": "LLM_MODEL",
    },
    "OPENAI": {
        "api_key": "OPENAI_API_KEY",
        "api_base": "OPENAI_BASE_URL",
        "model_name": "OPENAI_MODEL",
    },
}


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
    structured_output_mode: str = STRUCTURED_OUTPUT_LOCAL


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
    model = build_model_config(root)
    config = RunConfig(project_root=root, artifacts_dir=root / "artifacts", model=model)
    config.ensure_directories()
    return config


def build_model_config(
    project_root: Path | None = None,
    *,
    env_prefix: str | None = None,
) -> ModelConfig | None:
    """Build one complete connection without mixing environment namespaces."""

    root = project_root or DEFAULT_ROOT
    env = load_env(root / ".env")
    selected = _select_model_env(env, requested_prefix=env_prefix)
    if selected is None:
        return None
    prefix, values = selected
    timeout_seconds = _env_int(env, f"{prefix}_TIMEOUT_SECONDS", 120)
    structured_output_mode = str(
        env.get(f"{prefix}_STRUCTURED_OUTPUT_MODE", STRUCTURED_OUTPUT_LOCAL)
    ).strip()
    if structured_output_mode not in STRUCTURED_OUTPUT_MODES:
        raise ValueError(
            f"{prefix}_STRUCTURED_OUTPUT_MODE must be one of "
            + ", ".join(STRUCTURED_OUTPUT_MODES)
        )
    return ModelConfig(
        api_key=values["api_key"],
        api_base=values["api_base"],
        model_name=values["model_name"],
        temperature=_env_float(env, f"{prefix}_TEMPERATURE", 0.0),
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=_env_int(
            env, f"{prefix}_CONNECT_TIMEOUT_SECONDS", 20
        ),
        read_timeout_seconds=_env_int(
            env, f"{prefix}_READ_TIMEOUT_SECONDS", timeout_seconds
        ),
        max_retries=_env_int(env, f"{prefix}_MAX_RETRIES", 2),
        retry_backoff_seconds=_env_float(
            env, f"{prefix}_RETRY_BACKOFF_SECONDS", 2.0
        ),
        structured_output_mode=structured_output_mode,
    )


def _select_model_env(
    env: dict[str, str],
    *,
    requested_prefix: str | None = None,
) -> tuple[str, dict[str, str]] | None:
    selected_prefix = str(
        requested_prefix
        if requested_prefix is not None
        else env.get("AFA_MODEL_ENV_PREFIX", "")
    ).strip().upper()
    if selected_prefix and selected_prefix not in _MODEL_ENV_FIELDS:
        raise ValueError("model env prefix must be one of LLM or OPENAI")
    prefixes = (selected_prefix,) if selected_prefix else ("LLM", "OPENAI")
    for prefix in prefixes:
        fields = _MODEL_ENV_FIELDS[prefix]
        present = {
            name: str(env.get(key, "")).strip()
            for name, key in fields.items()
        }
        if prefix == "OPENAI" and not present["model_name"]:
            present["model_name"] = str(env.get("MODEL_NAME", "")).strip()
        supplied = [name for name, value in present.items() if value]
        if not supplied:
            if selected_prefix:
                raise ValueError(
                    f"{prefix} model configuration was explicitly selected but is missing"
                )
            continue
        if len(supplied) != len(fields):
            missing = sorted(set(fields) - set(supplied))
            raise ValueError(
                f"incomplete {prefix} model configuration; missing {', '.join(missing)}"
            )
        return prefix, present
    return None


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
