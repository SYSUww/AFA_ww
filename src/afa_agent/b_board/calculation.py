from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any, Mapping, Sequence

from afa_agent.b_board.io import validate_freeform_slot


@dataclass(frozen=True, slots=True)
class BusinessCalendar:
    holidays: frozenset[date] = frozenset()
    working_weekends: frozenset[date] = frozenset()

    def is_workday(self, value: date) -> bool:
        if value in self.working_weekends:
            return True
        return value.weekday() < 5 and value not in self.holidays

    def next_workday(self, value: date) -> date:
        current = value + timedelta(days=1)
        while not self.is_workday(current):
            current += timedelta(days=1)
        return current


@dataclass(frozen=True, slots=True)
class CalculationResult:
    answer_parts: tuple[str, ...]
    used_evidence_ids: tuple[str, ...]
    trace: dict[str, Any]


class CalculationPlanError(ValueError):
    pass


class CalculationExecutor:
    """Replay a small declarative calculation plan without eval or binary floats."""

    def __init__(self, calendar: BusinessCalendar | None = None):
        self.calendar = calendar or BusinessCalendar()

    def execute(
        self,
        plan: Mapping[str, Any],
        *,
        expected_slots: int,
        evidence_text_by_id: Mapping[str, str] | None = None,
        expected_slot_templates: Sequence[str] | None = None,
    ) -> CalculationResult:
        variables: dict[str, Any] = {}
        value_kinds: dict[str, str] = {}
        evidence_ids: list[str] = []
        normalized_variables: list[dict[str, Any]] = []
        grounding_checks: list[dict[str, Any]] = []
        for item in _require_list(plan, "variables"):
            if not isinstance(item, Mapping):
                raise CalculationPlanError("Each variable must be an object")
            name = str(item.get("name", "")).strip()
            if not name or name in variables:
                raise CalculationPlanError(f"Invalid or duplicate variable name: {name!r}")
            value_type = str(item.get("value_type", "decimal"))
            value = _parse_value(item.get("value"), value_type)
            variable_evidence = [str(value) for value in item.get("evidence_ids", []) if str(value)]
            if not variable_evidence:
                raise CalculationPlanError(f"Variable {name} has no evidence_ids")
            evidence_ids.extend(variable_evidence)
            variables[name] = value
            value_kinds[name] = _variable_kind(value_type, str(item.get("unit", "")))
            normalized_variables.append(
                {
                    "name": name,
                    "value": _serialize_value(value),
                    "value_type": value_type,
                    "unit": str(item.get("unit", "")),
                    "evidence_ids": variable_evidence,
                }
            )
            grounding_checks.append(
                _grounding_check(
                    name=name,
                    value=value,
                    value_type=value_type,
                    unit=str(item.get("unit", "")),
                    evidence_ids=variable_evidence,
                    evidence_text_by_id=evidence_text_by_id,
                )
            )

        grounding_verified = bool(evidence_text_by_id) and all(
            item["verified"] for item in grounding_checks
        )
        if evidence_text_by_id is not None and not grounding_verified:
            failed = [item["name"] for item in grounding_checks if not item["verified"]]
            raise CalculationPlanError(
                "Variables are not grounded in cited evidence: " + ",".join(failed)
            )

        values = dict(variables)
        normalized_steps: list[dict[str, Any]] = []
        for step in _require_list(plan, "steps", allow_missing=True):
            if not isinstance(step, Mapping):
                raise CalculationPlanError("Each step must be an object")
            step_id = str(step.get("id", "")).strip()
            op = str(step.get("op", "")).strip()
            if not step_id or step_id in values:
                raise CalculationPlanError(f"Invalid or duplicate step id: {step_id!r}")
            result, result_kind, conversions = self._run_operation(
                op, step, values, value_kinds
            )
            values[step_id] = result
            value_kinds[step_id] = result_kind
            normalized_steps.append(
                {
                    "id": step_id,
                    "op": op,
                    "args": step.get("args", []),
                    "operand_roles": {
                        key: step[key] for key in ("new", "old") if key in step
                    },
                    "result": _serialize_value(result),
                    "value_kind": result_kind,
                    "unit_conversions": conversions,
                }
            )

        raw_outputs = _require_list(plan, "outputs")
        if len(raw_outputs) != expected_slots:
            raise CalculationPlanError(
                f"Expected {expected_slots} output slots, got {len(raw_outputs)}"
            )
        answer_parts: list[str] = []
        normalized_outputs: list[dict[str, Any]] = []
        for position, output in enumerate(raw_outputs, start=1):
            if not isinstance(output, Mapping):
                raise CalculationPlanError("Each output must be an object")
            source = output.get("source")
            value = _resolve(source, values)
            value_kind = _resolve_kind(source, value_kinds)
            requested_format = str(output.get("format", "raw"))
            format_name = requested_format
            if expected_slot_templates is not None:
                if len(expected_slot_templates) != expected_slots:
                    raise CalculationPlanError("expected_slot_templates count mismatch")
                format_name = _format_for_slot_template(
                    str(expected_slot_templates[position - 1]),
                    value,
                    requested_format=requested_format,
                )
            rendered = _format_value(value, format_name, value_kind=value_kind)
            if not rendered:
                raise CalculationPlanError(f"Output slot {position} rendered empty")
            if expected_slot_templates is not None:
                try:
                    validate_freeform_slot(
                        rendered,
                        str(expected_slot_templates[position - 1]),
                        f"output slot {position}",
                    )
                except ValueError as exc:
                    raise CalculationPlanError(str(exc)) from exc
            answer_parts.append(rendered)
            normalized_outputs.append(
                {
                    "slot": position,
                    "source": source,
                    "requested_format": requested_format,
                    "format": format_name,
                    "value_kind": value_kind,
                    "value": rendered,
                }
            )

        deduped_evidence = tuple(dict.fromkeys(evidence_ids))
        trace = {
            "schema_version": 2,
            "variables": normalized_variables,
            "grounding_checks": grounding_checks,
            "steps": normalized_steps,
            "outputs": normalized_outputs,
            "replay_verified": True,
            "grounding_verified": grounding_verified,
        }
        return CalculationResult(tuple(answer_parts), deduped_evidence, trace)

    def _run_operation(
        self,
        op: str,
        step: Mapping[str, Any],
        values: Mapping[str, Any],
        value_kinds: Mapping[str, str],
    ) -> tuple[Any, str, list[dict[str, str]]]:
        args = [_resolve(item, values) for item in step.get("args", [])]
        arg_specs = list(step.get("args", []))
        arg_kinds = [_resolve_kind(item, value_kinds) for item in arg_specs]
        conversions: list[dict[str, str]] = []
        if op == "add":
            return sum((_decimal(item) for item in args), Decimal("0")), _first_kind(arg_kinds), conversions
        if op == "sub":
            _require_arg_count(op, args, 2)
            return _decimal(args[0]) - _decimal(args[1]), _first_kind(arg_kinds), conversions
        if op == "mul":
            result = Decimal("1")
            for item in args:
                result *= _decimal(item)
            return result, "decimal", conversions
        if op == "div":
            _require_arg_count(op, args, 2)
            denominator = _decimal(args[1])
            if arg_kinds[1] == "percent_points":
                denominator /= Decimal("100")
                conversions.append(
                    {
                        "argument": "denominator",
                        "from": "percent_points",
                        "to": "ratio",
                    }
                )
            if denominator == 0:
                raise CalculationPlanError("Division by zero")
            result_kind = arg_kinds[0] if arg_kinds[1] == "percent_points" else "ratio"
            return _decimal(args[0]) / denominator, result_kind, conversions
        if op == "mean":
            if not args:
                raise CalculationPlanError("mean requires at least one argument")
            return (
                sum((_decimal(item) for item in args), Decimal("0")) / Decimal(len(args)),
                _first_kind(arg_kinds),
                conversions,
            )
        if op == "abs":
            _require_arg_count(op, args, 1)
            return abs(_decimal(args[0])), _first_kind(arg_kinds), conversions
        if op == "max":
            return max(_decimal(item) for item in args), _first_kind(arg_kinds), conversions
        if op == "min":
            return min(_decimal(item) for item in args), _first_kind(arg_kinds), conversions
        if op == "pct_change":
            new_spec, old_spec = _require_named_directional_operands(op, step)
            new = _decimal(_resolve(new_spec, values))
            old = _decimal(_resolve(old_spec, values))
            if old == 0:
                raise CalculationPlanError("pct_change old value is zero")
            return (
                (new / old - Decimal("1")) * Decimal("100"),
                "percent_points",
                conversions,
            )
        if op == "pct_point_delta":
            new_spec, old_spec = _require_named_directional_operands(op, step)
            return (
                _decimal(_resolve(new_spec, values))
                - _decimal(_resolve(old_spec, values)),
                "percent_points",
                conversions,
            )
        if op == "count_gte":
            threshold = _decimal(_resolve(step.get("threshold"), values))
            return Decimal(sum(1 for item in args if _decimal(item) >= threshold)), "decimal", conversions
        if op == "count_gt":
            threshold = _decimal(_resolve(step.get("threshold"), values))
            return Decimal(sum(1 for item in args if _decimal(item) > threshold)), "decimal", conversions
        if op == "sort_desc":
            items = step.get("items")
            if not isinstance(items, list) or not items:
                raise CalculationPlanError("sort_desc requires items")
            pairs: list[tuple[str, Decimal]] = []
            for item in items:
                if not isinstance(item, Mapping):
                    raise CalculationPlanError("sort_desc item must be an object")
                label = str(item.get("label", "")).strip()
                if not label:
                    raise CalculationPlanError("sort_desc item has empty label")
                pairs.append((label, _decimal(_resolve(item.get("source"), values))))
            pairs.sort(key=lambda pair: (-pair[1], pair[0]))
            return ">".join(label for label, _ in pairs), "text", conversions
        if op == "date_add_days":
            _require_arg_count(op, args, 2)
            return _date(args[0]) + timedelta(days=int(_decimal(args[1]))), "date", conversions
        if op == "next_workday":
            _require_arg_count(op, args, 1)
            return self.calendar.next_workday(_date(args[0])), "date", conversions
        if op == "days_between":
            _require_arg_count(op, args, 2)
            return Decimal((_date(args[0]) - _date(args[1])).days), "decimal", conversions
        raise CalculationPlanError(f"Unsupported operation: {op!r}")


def _require_list(plan: Mapping[str, Any], key: str, *, allow_missing: bool = False) -> list[Any]:
    value = plan.get(key, [] if allow_missing else None)
    if not isinstance(value, list):
        raise CalculationPlanError(f"{key} must be a list")
    return value


def _require_named_directional_operands(
    op: str, step: Mapping[str, Any]
) -> tuple[Any, Any]:
    if "new" not in step or "old" not in step:
        raise CalculationPlanError(
            f"{op} requires named 'new' and 'old' operands"
        )
    return step["new"], step["old"]


def _parse_value(value: Any, value_type: str) -> Any:
    if value_type == "decimal":
        return _decimal(value)
    if value_type == "date":
        return _date(value)
    if value_type == "text":
        text = str(value).strip()
        if not text:
            raise CalculationPlanError("Text variable cannot be empty")
        return text
    raise CalculationPlanError(f"Unsupported value_type: {value_type!r}")


def _resolve(value: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping) and "ref" in value:
        ref = str(value["ref"])
        if ref not in values:
            raise CalculationPlanError(f"Unknown reference: {ref}")
        return values[ref]
    if isinstance(value, Mapping) and "literal" in value:
        value_type = str(value.get("value_type", "text"))
        return _parse_value(value["literal"], value_type)
    return value


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise CalculationPlanError("Boolean is not a decimal")
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, AttributeError) as exc:
        raise CalculationPlanError(f"Invalid decimal: {value!r}") from exc


def _date(value: Any) -> date:
    if isinstance(value, date):
        return value
    text = str(value).strip().replace("年", "-").replace("月", "-").replace("日", "")
    try:
        year, month, day = (int(part) for part in text.split("-"))
        return date(year, month, day)
    except (TypeError, ValueError) as exc:
        raise CalculationPlanError(f"Invalid date: {value!r}") from exc


def _format_value(value: Any, format_name: str, *, value_kind: str = "decimal") -> str:
    if format_name == "raw":
        return _serialize_value(value)
    if format_name == "decimal0":
        return str(_decimal(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if format_name == "decimal1":
        return format(_decimal(value).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP), "f")
    if format_name == "decimal2":
        return format(_decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")
    if format_name in {"percent2", "percent2_bare"}:
        numeric = _decimal(value)
        if value_kind == "ratio":
            numeric *= Decimal("100")
        rendered = format(numeric.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")
        return rendered + ("%" if format_name == "percent2" else "")
    if format_name == "date_cn":
        parsed = _date(value)
        return f"{parsed.year}年{parsed.month}月{parsed.day}日"
    if format_name == "text":
        return str(value).strip()
    raise CalculationPlanError(f"Unsupported output format: {format_name!r}")


def _serialize_value(value: Any) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _require_arg_count(op: str, args: Sequence[Any], count: int) -> None:
    if len(args) != count:
        raise CalculationPlanError(f"{op} requires {count} arguments, got {len(args)}")


def _variable_kind(value_type: str, unit: str) -> str:
    if value_type == "date":
        return "date"
    if value_type == "text":
        return "text"
    normalized_unit = "".join(unit.split()).lower()
    if normalized_unit in {"%", "百分点", "percent", "percentage"}:
        return "percent_points"
    return "amount" if normalized_unit else "decimal"


def _resolve_kind(value: Any, kinds: Mapping[str, str]) -> str:
    if isinstance(value, Mapping) and "ref" in value:
        return kinds.get(str(value["ref"]), "decimal")
    if isinstance(value, Mapping) and "literal" in value:
        return _variable_kind(str(value.get("value_type", "text")), str(value.get("unit", "")))
    return "decimal"


def _first_kind(kinds: Sequence[str]) -> str:
    return kinds[0] if kinds else "decimal"


def _format_for_slot_template(
    template: str,
    value: Any,
    *,
    requested_format: str,
) -> str:
    if template.endswith("%"):
        return "percent2"
    if ">" in template:
        return "text"
    if re.fullmatch(r"9+\.99", template):
        if isinstance(value, date):
            return "date_cn"
        if requested_format == "percent2":
            return "percent2_bare"
        return "decimal2"
    raise CalculationPlanError(f"Unsupported output slot template: {template!r}")


_NUMBER_RE = re.compile(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?(?![\d.])")


def _grounding_check(
    *,
    name: str,
    value: Any,
    value_type: str,
    unit: str,
    evidence_ids: Sequence[str],
    evidence_text_by_id: Mapping[str, str] | None,
) -> dict[str, Any]:
    if evidence_text_by_id is None:
        return {
            "name": name,
            "verified": False,
            "reason": "evidence_text_not_supplied",
            "matched_evidence_ids": [],
        }
    available = {
        evidence_id: str(evidence_text_by_id[evidence_id])
        for evidence_id in evidence_ids
        if evidence_id in evidence_text_by_id
    }
    matched: list[str] = []
    for evidence_id, text in available.items():
        if _value_appears(value, value_type, text) and _unit_appears(unit, text):
            matched.append(evidence_id)
    reason = "matched_literal_value_and_unit" if matched else "value_or_unit_not_found"
    return {
        "name": name,
        "verified": bool(matched),
        "reason": reason,
        "matched_evidence_ids": matched,
    }


def _value_appears(value: Any, value_type: str, text: str) -> bool:
    if value_type == "decimal":
        target = _decimal(value)
        for raw in _NUMBER_RE.findall(text):
            try:
                if _decimal(raw) == target:
                    return True
            except CalculationPlanError:
                continue
        return False
    if value_type == "date":
        target = _date(value)
        variants = {
            target.isoformat(),
            f"{target.year}年{target.month}月{target.day}日",
            f"{target.year}年{target.month:02d}月{target.day:02d}日",
        }
        return any(item in text for item in variants)
    normalized_value = "".join(str(value).split())
    normalized_text = "".join(text.split())
    return bool(normalized_value) and normalized_value in normalized_text


def _unit_appears(unit: str, text: str) -> bool:
    normalized = "".join(unit.split())
    if not normalized or normalized in {"无", "个", "日", "天"}:
        return True
    aliases = {
        "%": ("%", "百分比"),
        "百分点": ("百分点", "%"),
        "元/股": ("元/股", "元／股", "每股"),
    }
    return any(token in text for token in aliases.get(normalized, (normalized,)))
