#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.client import OpenAICompatibleClient, extract_json_object  # noqa: E402
from afa_agent.config import build_model_config  # noqa: E402


PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "echo": {"type": "string", "enum": ["schema_probe"]},
    },
    "required": ["ok", "echo"],
    "additionalProperties": False,
}
PROBE_MESSAGES = [
    {
        "role": "system",
        "content": "Return only the JSON object required by the supplied schema.",
    },
    {
        "role": "user",
        "content": "This is a de-identified capability probe. Set ok=true and echo=schema_probe.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe native strict JSON Schema support with a sanitized request"
    )
    parser.add_argument(
        "--env-root",
        type=Path,
        default=Path("/Users/abandon/Documents/AFA_ww"),
    )
    parser.add_argument("--model", default="qwen3.7-plus-2026-05-26")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_model_config(args.env_root, env_prefix="OPENAI")
    if config is None:
        raise RuntimeError("OPENAI model configuration is missing")
    config = replace(config, model_name=args.model, temperature=0.0, max_retries=0)
    response = OpenAICompatibleClient(config).chat_json(
        PROBE_MESSAGES,
        response_schema=PROBE_SCHEMA,
        schema_name="qwen37_capability_probe_v1",
    )
    parsed = extract_json_object(response.content)
    schema_adherent = (
        set(parsed) == {"ok", "echo"}
        and parsed.get("ok") is True
        and parsed.get("echo") == "schema_probe"
    )
    payload = {
        "probe_version": "qwen37_native_strict_schema_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": config.model_name,
        "temperature": config.temperature,
        "response_format_mode": response.response_format_mode,
        "schema_adherent": schema_adherent,
        "response": parsed,
        "token_usage": response.token_usage.to_dict(),
        "request_hashes": {
            "messages_sha256": _sha256_json(PROBE_MESSAGES),
            "schema_sha256": _sha256_json(PROBE_SCHEMA),
        },
        "contains_competition_question_or_answer": False,
        "included_in_submission_token_usage": False,
    }
    if response.response_format_mode != "native_json_schema_strict":
        raise RuntimeError("provider response was not marked as native strict schema")
    if not schema_adherent:
        raise RuntimeError("provider accepted the request but violated the strict schema")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()
