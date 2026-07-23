from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


SUBMISSION_REASONING_SCHEMA_VERSION = "submission_reasoning_v1"
SUBMISSION_REASONING_NORMALIZATION_VERSION = (
    "deterministic_payload_normalization_v3_explicit_frozen_conclusion"
)

SUBMISSION_REASONING_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "answer_parts",
        "grounding_status",
        "missing_support",
        "reasoning",
    ],
    "properties": {
        "answer_parts": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
        "grounding_status": {
            "type": "string",
            "enum": ["supported", "insufficient"],
        },
        "missing_support": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
    },
}

_VALIDATOR = Draft202012Validator(SUBMISSION_REASONING_SCHEMA)
_ALLOWED_KEYS = frozenset(SUBMISSION_REASONING_SCHEMA["properties"])


def normalize_submission_reasoning_payload(
    payload: Mapping[str, Any],
    *,
    frozen_answer_parts: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Repair only representation drift without inventing reasoning content."""

    normalized = copy.deepcopy(dict(payload))
    normalizations: list[dict[str, Any]] = []

    extra_keys = sorted(str(key) for key in normalized if key not in _ALLOWED_KEYS)
    if extra_keys:
        normalized = {
            key: value for key, value in normalized.items() if key in _ALLOWED_KEYS
        }
        normalizations.append(
            {
                "reason": "drop_noncontract_fields",
                "removed_keys": extra_keys,
            }
        )

    answer_parts = normalized.get("answer_parts")
    expected = [str(item) for item in frozen_answer_parts]
    if (
        isinstance(answer_parts, str)
        and len(expected) == 1
        and answer_parts == expected[0]
    ):
        normalized["answer_parts"] = [answer_parts]
        normalizations.append(
            {"reason": "single_frozen_answer_string_to_array"}
        )
    elif (
        isinstance(answer_parts, list)
        and len(expected) == 1
        and len(answer_parts) > 1
        and all(isinstance(item, str) for item in answer_parts)
        and "".join(answer_parts) == expected[0]
    ):
        normalized["answer_parts"] = [expected[0]]
        normalizations.append(
            {
                "reason": "join_exact_split_single_slot_answer_parts",
                "part_count": len(answer_parts),
            }
        )

    grounding_status = normalized.get("grounding_status")
    if isinstance(grounding_status, str):
        compact_status = grounding_status.strip().lower()
        if compact_status != grounding_status:
            normalized["grounding_status"] = compact_status
            normalizations.append(
                {"reason": "normalize_grounding_status_whitespace_case"}
            )
        grounding_status = compact_status

    missing_support = normalized.get("missing_support")
    if missing_support is None and grounding_status == "supported":
        normalized["missing_support"] = []
        normalizations.append(
            {"reason": "supported_null_missing_support_to_empty_array"}
        )
    elif isinstance(missing_support, str) and missing_support.strip():
        normalized["missing_support"] = [missing_support.strip()]
        normalizations.append(
            {"reason": "missing_support_string_to_array"}
        )

    reasoning = normalized.get("reasoning")
    if (
        normalized.get("answer_parts") == expected
        and grounding_status == "supported"
        and isinstance(reasoning, str)
        and reasoning.strip()
    ):
        compact_reasoning = "".join(reasoning.split()).replace(",", "")
        missing_parts = [
            part
            for part in expected
            if "".join(part.split()).replace(",", "") not in compact_reasoning
        ]
        if missing_parts:
            separator = "" if reasoning.rstrip().endswith(("。", "！", "？", ";", "；")) else "。"
            if len(expected) == 1:
                conclusion = f"最终答案为{expected[0]}。"
            else:
                conclusion = f"最终答案依次为{'；'.join(expected)}。"
            normalized["reasoning"] = f"{reasoning.rstrip()}{separator}{conclusion}"
            normalizations.append(
                {
                    "reason": "append_exact_frozen_answer_conclusion",
                    "missing_parts": missing_parts,
                }
            )

    return normalized, normalizations


def validate_submission_reasoning_schema(payload: Mapping[str, Any]) -> None:
    errors = sorted(
        _VALIDATOR.iter_errors(dict(payload)),
        key=lambda item: [str(part) for part in item.absolute_path],
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise ValueError(
        f"SubmissionReasoning schema violation at {path}: {error.message}"
    )
