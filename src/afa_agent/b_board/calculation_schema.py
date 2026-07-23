from __future__ import annotations

from typing import Any, Mapping

from jsonschema import Draft202012Validator

from afa_agent.b_board.calculation import CalculationPlanError


CALCULATION_PLAN_SCHEMA_VERSION = "calculation_plan_v1"

_VALUE_NODE = {
    "oneOf": [
        {"$ref": "#/$defs/reference"},
        {"$ref": "#/$defs/literal"},
    ]
}

CALCULATION_PLAN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "variables",
        "steps",
        "outputs",
        "supporting_evidence_ids",
        "decision_summary",
    ],
    "properties": {
        "variables": {
            "type": "array",
            "items": {"$ref": "#/$defs/variable"},
        },
        "steps": {
            "type": "array",
            "items": {"$ref": "#/$defs/step"},
        },
        "outputs": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/output"},
        },
        "supporting_evidence_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "decision_summary": {"type": "string"},
    },
    "$defs": {
        "reference": {
            "type": "object",
            "additionalProperties": False,
            "required": ["ref"],
            "properties": {"ref": {"type": "string", "minLength": 1}},
        },
        "literal": {
            "type": "object",
            "additionalProperties": False,
            "required": ["literal", "value_type", "unit"],
            "properties": {
                "literal": {"type": ["string", "number"]},
                "value_type": {
                    "type": "string",
                    "enum": ["decimal", "date", "text"],
                },
                "unit": {"type": "string"},
            },
        },
        "variable": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "value", "value_type", "unit", "evidence_ids"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "value": {"type": ["string", "number"]},
                "value_type": {
                    "type": "string",
                    "enum": ["decimal", "date", "text"],
                },
                "unit": {"type": "string"},
                "evidence_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
        "step": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "args"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "op": {
                            "type": "string",
                            "enum": ["add", "mul", "mean", "max", "min"],
                        },
                        "args": {
                            "type": "array",
                            "minItems": 1,
                            "items": _VALUE_NODE,
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "args"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "op": {"type": "string", "enum": ["sub", "div"]},
                        "args": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": _VALUE_NODE,
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "args"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "op": {"const": "abs"},
                        "args": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 1,
                            "items": _VALUE_NODE,
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "new", "old"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "op": {
                            "type": "string",
                            "enum": ["pct_change", "pct_point_delta"],
                        },
                        "new": _VALUE_NODE,
                        "old": _VALUE_NODE,
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "args", "threshold"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "op": {
                            "type": "string",
                            "enum": ["count_gte", "count_gt"],
                        },
                        "args": {
                            "type": "array",
                            "minItems": 1,
                            "items": _VALUE_NODE,
                        },
                        "threshold": _VALUE_NODE,
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "items"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "op": {"const": "sort_desc"},
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["label", "source"],
                                "properties": {
                                    "label": {"type": "string", "minLength": 1},
                                    "source": _VALUE_NODE,
                                },
                            },
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "args"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "op": {"const": "date_add_days"},
                        "args": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["date", "days"],
                            "properties": {
                                "date": _VALUE_NODE,
                                "days": _VALUE_NODE,
                            },
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "args"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "op": {"const": "next_workday"},
                        "args": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["date"],
                            "properties": {"date": _VALUE_NODE},
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "args"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "op": {"const": "days_between"},
                        "args": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["end", "start"],
                            "properties": {
                                "end": _VALUE_NODE,
                                "start": _VALUE_NODE,
                            },
                        },
                    },
                },
            ]
        },
        "output": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source", "format"],
            "properties": {
                "source": {
                    "oneOf": [
                        {"$ref": "#/$defs/reference"},
                        {"type": "string", "minLength": 1},
                    ]
                },
                "format": {
                    "type": "string",
                    "enum": [
                        "raw",
                        "decimal0",
                        "decimal1",
                        "decimal2",
                        "percent2",
                        "date_cn",
                        "text",
                    ],
                },
            },
        },
    },
}


_VALIDATOR = Draft202012Validator(CALCULATION_PLAN_SCHEMA)


def validate_calculation_plan_schema(plan: Mapping[str, Any]) -> None:
    errors = sorted(
        _VALIDATOR.iter_errors(dict(plan)),
        key=lambda item: [str(part) for part in item.absolute_path],
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise CalculationPlanError(
        f"CalculationPlan schema violation at {path}: {error.message}"
    )
