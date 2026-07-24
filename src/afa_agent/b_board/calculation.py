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
        semantic_constraints: Sequence[Mapping[str, Any]] | None = None,
        expected_slot_templates: Sequence[str] | None = None,
        expected_numeric_decimal_places: int | None = None,
        expected_percent_suffixes: Sequence[bool | None] | None = None,
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
            unit = str(item.get("unit", ""))
            value = _parse_variable_value(item.get("value"), value_type, unit)
            variable_evidence = [str(value) for value in item.get("evidence_ids", []) if str(value)]
            if not variable_evidence:
                raise CalculationPlanError(f"Variable {name} has no evidence_ids")
            evidence_ids.extend(variable_evidence)
            variables[name] = value
            value_kinds[name] = _variable_kind(value_type, unit)
            normalized_variables.append(
                {
                    "name": name,
                    "value": _serialize_value(value),
                    "value_type": value_type,
                    "unit": unit,
                    "evidence_ids": variable_evidence,
                }
            )
            grounding_checks.append(
                _grounding_check(
                    name=name,
                    value=value,
                    value_type=value_type,
                    unit=unit,
                    evidence_ids=variable_evidence,
                    evidence_text_by_id=evidence_text_by_id,
                )
            )

        grounding_verified = bool(evidence_text_by_id) and all(
            item["verified"] for item in grounding_checks
        )
        if evidence_text_by_id is not None and not grounding_verified:
            failed = [
                f'{item["name"]}[{item["reason"]}]'
                for item in grounding_checks
                if not item["verified"]
            ]
            raise CalculationPlanError(
                "Variables are not grounded in cited evidence: " + ",".join(failed)
            )

        aggregation_scope_checks = _validate_disclosed_aggregate_scope(
            plan,
            semantic_constraints or (),
        )
        _validate_amount_unit_scales(plan)

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
            question_rounded_value: str | None = None
            if expected_slot_templates is not None:
                if len(expected_slot_templates) != expected_slots:
                    raise CalculationPlanError("expected_slot_templates count mismatch")
                if (
                    expected_percent_suffixes is not None
                    and len(expected_percent_suffixes) != expected_slots
                ):
                    raise CalculationPlanError("expected_percent_suffixes count mismatch")
                slot_template = str(expected_slot_templates[position - 1])
                percent_suffix = (
                    expected_percent_suffixes[position - 1]
                    if expected_percent_suffixes is not None
                    else None
                )
                format_name = _format_for_slot_contract(
                    slot_template,
                    value,
                    requested_format=requested_format,
                    numeric_decimal_places=expected_numeric_decimal_places,
                    percent_suffix=percent_suffix,
                )
            rendered = _format_value(value, format_name, value_kind=value_kind)
            if (
                expected_slot_templates is not None
                and expected_numeric_decimal_places is not None
                and re.fullmatch(r"-?\d+(?:\.\d+)?%?", rendered)
            ):
                question_rounded_value = rendered.removesuffix("%")
            if not rendered:
                raise CalculationPlanError(f"Output slot {position} rendered empty")
            if expected_slot_templates is not None:
                try:
                    validate_freeform_slot(
                        rendered,
                        str(expected_slot_templates[position - 1]),
                        f"output slot {position}",
                        numeric_decimal_places=expected_numeric_decimal_places,
                        percent_suffix=(
                            expected_percent_suffixes[position - 1]
                            if expected_percent_suffixes is not None
                            else None
                        ),
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
                    "question_decimal_places": expected_numeric_decimal_places,
                    "question_rounded_value": question_rounded_value,
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
            "aggregation_scope_checks": aggregation_scope_checks,
            "aggregation_scope_verified": all(
                item["verified"] for item in aggregation_scope_checks
            ),
        }
        return CalculationResult(tuple(answer_parts), deduped_evidence, trace)

    def replay_legacy_trace(
        self,
        trace: Mapping[str, Any],
        *,
        expected_slots: int,
        evidence_text_by_id: Mapping[str, str],
        expected_slot_templates: Sequence[str] | None = None,
        expected_numeric_decimal_places: int | None = None,
        expected_percent_suffixes: Sequence[bool | None] | None = None,
        preserve_incumbent_answer: bool = True,
    ) -> CalculationResult:
        """Revalidate an old normalized trace without asking a model to replan it.

        Legacy traces stored ratios such as ``0.75`` while citing clauses that
        literally state ``75%``.  This method keeps the original dependency
        graph and output fixed, but makes that conversion explicit as a replayed
        division by 100.  Variables and steps that do not contribute to an
        output are discarded before grounding so narrative helper fields cannot
        masquerade as calculation inputs.
        """

        plan, replay_meta = _legacy_trace_replay_plan(trace, evidence_text_by_id)
        result = self.execute(
            plan,
            expected_slots=expected_slots,
            evidence_text_by_id=evidence_text_by_id,
            expected_slot_templates=expected_slot_templates,
            expected_numeric_decimal_places=expected_numeric_decimal_places,
            expected_percent_suffixes=expected_percent_suffixes,
        )
        expected_parts = tuple(replay_meta["expected_answer_parts"])
        answer_preserved = not expected_parts or result.answer_parts == expected_parts
        if preserve_incumbent_answer and not answer_preserved:
            raise CalculationPlanError(
                "Legacy replay changed the incumbent answer: "
                f"expected {expected_parts}, got {result.answer_parts}"
            )
        if not preserve_incumbent_answer and not _answers_differ_only_in_format(
            expected_parts, result.answer_parts
        ):
            raise CalculationPlanError(
                "Legacy format migration changed the incumbent value: "
                f"expected {expected_parts}, got {result.answer_parts}"
            )
        replayed_trace = {
            **result.trace,
            "revalidated_from_schema_version": replay_meta["source_schema_version"],
            "revalidation": {
                "answer_preserved": answer_preserved,
                "format_change_allowed": not preserve_incumbent_answer,
                "converted_percent_ratio_variables": replay_meta[
                    "converted_percent_ratio_variables"
                ],
                "pruned_variable_names": replay_meta["pruned_variable_names"],
                "pruned_step_ids": replay_meta["pruned_step_ids"],
            },
        }
        return CalculationResult(
            result.answer_parts,
            result.used_evidence_ids,
            replayed_trace,
        )

    def _run_operation(
        self,
        op: str,
        step: Mapping[str, Any],
        values: Mapping[str, Any],
        value_kinds: Mapping[str, str],
    ) -> tuple[Any, str, list[dict[str, str]]]:
        arg_specs = _operation_arg_specs(op, step)
        args = [_resolve(item, values) for item in arg_specs]
        arg_kinds = [_resolve_kind(item, value_kinds) for item in arg_specs]
        conversions: list[dict[str, str]] = []
        if op == "add":
            return sum((_decimal(item) for item in args), Decimal("0")), _first_kind(arg_kinds), conversions
        if op == "sub":
            _require_arg_count(op, args, 2)
            left = _decimal(args[0])
            right = _decimal(args[1])
            result_kind = _first_kind(arg_kinds)
            if arg_kinds[0] == "percent_points" and _is_ratio_operand(
                right, arg_kinds[1]
            ):
                left /= Decimal("100")
                conversions.append(
                    {
                        "argument": "left",
                        "from": "percent_points",
                        "to": "ratio",
                    }
                )
                result_kind = "ratio"
            elif arg_kinds[1] == "percent_points" and _is_ratio_operand(
                left, arg_kinds[0]
            ):
                right /= Decimal("100")
                conversions.append(
                    {
                        "argument": "right",
                        "from": "percent_points",
                        "to": "ratio",
                    }
                )
                result_kind = "ratio"
            return left - right, result_kind, conversions
        if op == "mul":
            result = Decimal("1")
            for index, (item, kind) in enumerate(zip(args, arg_kinds), start=1):
                operand = _decimal(item)
                if kind == "percent_points":
                    operand /= Decimal("100")
                    conversions.append(
                        {
                            "argument": f"argument_{index}",
                            "from": "percent_points",
                            "to": "ratio",
                        }
                    )
                result *= operand
            return result, _multiplication_kind(arg_kinds), conversions
        if op == "div":
            _require_arg_count(op, args, 2)
            numerator = _decimal(args[0])
            denominator = _decimal(args[1])
            if arg_kinds[0] == "percent_points" and arg_kinds[1] == "percent_points":
                numerator /= Decimal("100")
                denominator /= Decimal("100")
                conversions.extend(
                    [
                        {
                            "argument": "numerator",
                            "from": "percent_points",
                            "to": "ratio",
                        },
                        {
                            "argument": "denominator",
                            "from": "percent_points",
                            "to": "ratio",
                        },
                    ]
                )
                result_kind = "ratio"
            elif arg_kinds[1] == "percent_points":
                denominator /= Decimal("100")
                conversions.append(
                    {
                        "argument": "denominator",
                        "from": "percent_points",
                        "to": "ratio",
                    }
                )
                result_kind = arg_kinds[0]
            else:
                result_kind = "ratio"
            if denominator == 0:
                raise CalculationPlanError("Division by zero")
            return numerator / denominator, result_kind, conversions
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
            new = _decimal(_resolve(new_spec, values))
            old = _decimal(_resolve(old_spec, values))
            new_kind = _resolve_kind(new_spec, value_kinds)
            old_kind = _resolve_kind(old_spec, value_kinds)
            if new_kind == "ratio":
                new *= Decimal("100")
                conversions.append(
                    {"argument": "new", "from": "ratio", "to": "percent_points"}
                )
            if old_kind == "ratio":
                old *= Decimal("100")
                conversions.append(
                    {"argument": "old", "from": "ratio", "to": "percent_points"}
                )
            return (
                new - old,
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


_CURRENCY_UNIT_FACTORS = {
    "元": Decimal("1"),
    "万元": Decimal("10000"),
    "百万元": Decimal("1000000"),
    "亿元": Decimal("100000000"),
}
_COUNT_UNIT_FACTORS = {
    "人": Decimal("1"),
    "万人": Decimal("10000"),
}
_DIMENSIONLESS_UNITS = {"", "%", "百分比", "比例", "倍", "ratio", "百分点"}


def _validate_amount_unit_scales(plan: Mapping[str, Any]) -> None:
    """Reject direct addition/subtraction across known currency scales.

    This is intentionally a validation-only unit pass. It recognizes explicit
    powers-of-ten conversion steps and the common ``万人 * 元 = 万元`` relation,
    then lets the normal Decimal executor replay the model's arithmetic.
    """

    units: dict[str, str] = {}
    scalar_values: dict[str, Decimal] = {}
    for item in _require_list(plan, "variables"):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        unit = _normalize_amount_unit(item.get("unit", ""))
        if name:
            units[name] = unit
            if not unit:
                try:
                    scalar_values[name] = _decimal(item.get("value"))
                except CalculationPlanError:
                    pass

    for step in _require_list(plan, "steps", allow_missing=True):
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("id", "")).strip()
        op = str(step.get("op", "")).strip()
        if not step_id:
            continue
        result_unit = ""
        if op in {"add", "sub", "mean", "abs", "max", "min", "mul", "div"}:
            arg_specs = _operation_arg_specs(op, step)
            arg_units = [_unit_for_spec(spec, units) for spec in arg_specs]
            if op in {"add", "sub"}:
                currency_units = [unit for unit in arg_units if unit in _CURRENCY_UNIT_FACTORS]
                if len(set(currency_units)) > 1:
                    raise CalculationPlanError(
                        f"{op} amount unit mismatch: {' vs '.join(currency_units)}; "
                        "convert explicitly before add/sub (1亿元=10000万元)"
                    )
                result_unit = currency_units[0] if currency_units else _first_known_unit(arg_units)
            elif op == "mul":
                result_unit = _infer_multiplication_unit(
                    arg_specs,
                    arg_units,
                    scalar_values,
                )
            elif op == "div":
                result_unit = _infer_division_unit(
                    arg_specs,
                    arg_units,
                    scalar_values,
                )
            else:
                result_unit = _first_known_unit(arg_units)
        elif op in {"pct_change", "pct_point_delta"}:
            result_unit = "%"
        units[step_id] = result_unit


def _normalize_amount_unit(value: Any) -> str:
    unit = re.sub(r"\s+", "", str(value or "")).replace("％", "%").lower()
    return "" if unit in _DIMENSIONLESS_UNITS else unit


def _unit_for_spec(spec: Any, units: Mapping[str, str]) -> str:
    if isinstance(spec, Mapping) and "ref" in spec:
        return units.get(str(spec["ref"]), "")
    return ""


def _first_known_unit(units: Sequence[str]) -> str:
    return next((unit for unit in units if unit), "")


def _infer_multiplication_unit(
    arg_specs: Sequence[Any],
    arg_units: Sequence[str],
    scalar_values: Mapping[str, Decimal],
) -> str:
    currency_units = [unit for unit in arg_units if unit in _CURRENCY_UNIT_FACTORS]
    count_units = [unit for unit in arg_units if unit in _COUNT_UNIT_FACTORS]
    if len(currency_units) == 1 and len(count_units) == 1:
        factor = (
            _CURRENCY_UNIT_FACTORS[currency_units[0]]
            * _COUNT_UNIT_FACTORS[count_units[0]]
        )
        return _currency_unit_for_factor(factor)
    if len(currency_units) == 1:
        scalar = _explicit_scalar_product(arg_specs, arg_units, scalar_values)
        if scalar is not None and scalar != 0:
            converted = _currency_unit_for_factor(
                _CURRENCY_UNIT_FACTORS[currency_units[0]] / scalar
            )
            if converted:
                return converted
        return currency_units[0]
    if len(count_units) == 1:
        scalar = _explicit_scalar_product(
            arg_specs,
            arg_units,
            scalar_values,
        )
        if scalar is not None and scalar != 0:
            converted = _count_unit_for_factor(
                _COUNT_UNIT_FACTORS[count_units[0]] / scalar
            )
            if converted:
                return converted
        return count_units[0]
    return _first_known_unit(arg_units) if len([unit for unit in arg_units if unit]) == 1 else ""


def _infer_division_unit(
    arg_specs: Sequence[Any],
    arg_units: Sequence[str],
    scalar_values: Mapping[str, Decimal],
) -> str:
    numerator_unit = arg_units[0] if arg_units else ""
    denominator_unit = arg_units[1] if len(arg_units) > 1 else ""
    if numerator_unit in _CURRENCY_UNIT_FACTORS and not denominator_unit:
        scalar = _scalar_decimal(arg_specs[1], scalar_values) if len(arg_specs) > 1 else None
        if scalar is not None and scalar != 0:
            converted = _currency_unit_for_factor(
                _CURRENCY_UNIT_FACTORS[numerator_unit] * scalar
            )
            if converted:
                return converted
        return numerator_unit
    if numerator_unit in _CURRENCY_UNIT_FACTORS and denominator_unit in _CURRENCY_UNIT_FACTORS:
        return ""
    if numerator_unit in _COUNT_UNIT_FACTORS and not denominator_unit:
        scalar = (
            _scalar_decimal(arg_specs[1], scalar_values)
            if len(arg_specs) > 1
            else None
        )
        if scalar is not None and scalar != 0:
            converted = _count_unit_for_factor(
                _COUNT_UNIT_FACTORS[numerator_unit] * scalar
            )
            if converted:
                return converted
        return numerator_unit
    return numerator_unit


def _explicit_scalar_product(
    arg_specs: Sequence[Any],
    arg_units: Sequence[str],
    scalar_values: Mapping[str, Decimal],
) -> Decimal | None:
    scalar = Decimal("1")
    found = False
    for spec, unit in zip(arg_specs, arg_units):
        if unit:
            continue
        value = _scalar_decimal(spec, scalar_values)
        if value is None:
            continue
        scalar *= value
        found = True
    return scalar if found else None


def _scalar_decimal(spec: Any, scalar_values: Mapping[str, Decimal]) -> Decimal | None:
    if isinstance(spec, Mapping):
        if "ref" in spec:
            return scalar_values.get(str(spec["ref"]))
        if "literal" in spec:
            spec = spec["literal"]
        else:
            return None
    if isinstance(spec, bool):
        return None
    try:
        return _decimal(spec)
    except CalculationPlanError:
        return None


def _currency_unit_for_factor(factor: Decimal) -> str:
    return next(
        (unit for unit, candidate in _CURRENCY_UNIT_FACTORS.items() if candidate == factor),
        "",
    )


def _count_unit_for_factor(factor: Decimal) -> str:
    return next(
        (
            unit
            for unit, candidate in _COUNT_UNIT_FACTORS.items()
            if candidate == factor
        ),
        "",
    )


def _legacy_trace_replay_plan(
    trace: Mapping[str, Any],
    evidence_text_by_id: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_variables = _require_list(trace, "variables")
    raw_steps = _require_list(trace, "steps", allow_missing=True)
    raw_outputs = _require_list(trace, "outputs")
    variables_by_name: dict[str, dict[str, Any]] = {}
    for raw in raw_variables:
        if not isinstance(raw, Mapping):
            raise CalculationPlanError("Legacy trace variable must be an object")
        name = str(raw.get("name", "")).strip()
        if not name or name in variables_by_name:
            raise CalculationPlanError(f"Invalid legacy variable name: {name!r}")
        variables_by_name[name] = dict(raw)

    steps_by_id: dict[str, dict[str, Any]] = {}
    ordered_step_ids: list[str] = []
    for raw in raw_steps:
        if not isinstance(raw, Mapping):
            raise CalculationPlanError("Legacy trace step must be an object")
        step_id = str(raw.get("id", "")).strip()
        if not step_id or step_id in steps_by_id or step_id in variables_by_name:
            raise CalculationPlanError(f"Invalid legacy step id: {step_id!r}")
        step = dict(raw)
        operand_roles = step.get("operand_roles")
        if isinstance(operand_roles, Mapping):
            for role in ("new", "old"):
                if role in operand_roles:
                    step.setdefault(role, operand_roles[role])
        steps_by_id[step_id] = step
        ordered_step_ids.append(step_id)

    needed_variables: set[str] = set()
    needed_steps: set[str] = set()
    visiting: set[str] = set()

    def visit_reference(ref: str) -> None:
        if ref in variables_by_name:
            needed_variables.add(ref)
            return
        if ref not in steps_by_id:
            raise CalculationPlanError(f"Legacy trace has unknown reference: {ref}")
        if ref in needed_steps:
            return
        if ref in visiting:
            raise CalculationPlanError(f"Legacy trace has a dependency cycle at: {ref}")
        visiting.add(ref)
        for dependency in _reference_names(steps_by_id[ref]):
            visit_reference(dependency)
        visiting.remove(ref)
        needed_steps.add(ref)

    for output in raw_outputs:
        if not isinstance(output, Mapping):
            raise CalculationPlanError("Legacy trace output must be an object")
        for ref in _reference_names(output.get("source")):
            visit_reference(ref)

    evidence_ids = [str(value) for value in evidence_text_by_id]
    normalized_variables: list[dict[str, Any]] = []
    conversion_steps: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []
    for name in (str(item.get("name", "")) for item in raw_variables if isinstance(item, Mapping)):
        if name not in needed_variables:
            continue
        variable = dict(variables_by_name[name])
        value_type = str(variable.get("value_type", "decimal"))
        value = _parse_value(variable.get("value"), value_type)
        unit = str(variable.get("unit", ""))
        declared_ids = [
            str(value)
            for value in variable.get("evidence_ids", [])
            if str(value) in evidence_text_by_id
        ]
        direct = _grounding_check(
            name=name,
            value=value,
            value_type=value_type,
            unit=unit,
            evidence_ids=declared_ids,
            evidence_text_by_id=evidence_text_by_id,
        )
        if not direct["verified"]:
            direct = _grounding_check(
                name=name,
                value=value,
                value_type=value_type,
                unit=unit,
                evidence_ids=evidence_ids,
                evidence_text_by_id=evidence_text_by_id,
            )
        if direct["verified"]:
            variable["evidence_ids"] = list(direct["matched_evidence_ids"])
            normalized_variables.append(variable)
            continue

        percent = _legacy_ratio_as_percent(
            name=name,
            value=value,
            value_type=value_type,
            unit=unit,
            preferred_evidence_ids=declared_ids,
            evidence_ids=evidence_ids,
            evidence_text_by_id=evidence_text_by_id,
        )
        if percent is None:
            raise CalculationPlanError(
                f"Legacy variable {name} cannot be grounded literally or as an explicit percentage"
            )
        percent_name = f"{name}__percent_points"
        if percent_name in variables_by_name or percent_name in steps_by_id:
            raise CalculationPlanError(f"Legacy percentage conversion id collision: {percent_name}")
        normalized_variables.append(
            {
                "name": percent_name,
                "value": percent["value"],
                "value_type": "decimal",
                "unit": "%",
                "evidence_ids": percent["evidence_ids"],
            }
        )
        conversion_steps.append(
            {
                "id": name,
                "op": "div",
                "args": [
                    {"ref": percent_name},
                    {"literal": "100", "value_type": "decimal"},
                ],
            }
        )
        converted.append(
            {
                "name": name,
                "source_value": _serialize_value(value),
                "literal_percent_value": percent["value"],
                "evidence_ids": percent["evidence_ids"],
            }
        )

    normalized_steps = [
        steps_by_id[step_id] for step_id in ordered_step_ids if step_id in needed_steps
    ]
    expected_parts = [
        str(output.get("value", "")).strip()
        for output in raw_outputs
        if isinstance(output, Mapping) and str(output.get("value", "")).strip()
    ]
    return (
        {
            "variables": normalized_variables,
            "steps": [*conversion_steps, *normalized_steps],
            "outputs": [dict(output) for output in raw_outputs],
        },
        {
            "source_schema_version": int(trace.get("schema_version", 1)),
            "expected_answer_parts": expected_parts,
            "converted_percent_ratio_variables": converted,
            "pruned_variable_names": sorted(set(variables_by_name) - needed_variables),
            "pruned_step_ids": sorted(set(steps_by_id) - needed_steps),
        },
    )


def _reference_names(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        names = {str(value["ref"])} if "ref" in value else set()
        for nested in value.values():
            names.update(_reference_names(nested))
        return names
    if isinstance(value, list):
        names: set[str] = set()
        for nested in value:
            names.update(_reference_names(nested))
        return names
    return set()


def _legacy_ratio_as_percent(
    *,
    name: str,
    value: Any,
    value_type: str,
    unit: str,
    preferred_evidence_ids: Sequence[str],
    evidence_ids: Sequence[str],
    evidence_text_by_id: Mapping[str, str],
) -> dict[str, Any] | None:
    if value_type != "decimal":
        return None
    normalized_unit = "".join(unit.split()).lower()
    if normalized_unit not in {"比例", "倍", "ratio"}:
        return None
    percent_value = _decimal(value) * Decimal("100")
    check = _grounding_check(
        name=f"{name}__percent_points",
        value=percent_value,
        value_type="decimal",
        unit="%",
        evidence_ids=preferred_evidence_ids,
        evidence_text_by_id=evidence_text_by_id,
    )
    if not check["verified"]:
        check = _grounding_check(
            name=f"{name}__percent_points",
            value=percent_value,
            value_type="decimal",
            unit="%",
            evidence_ids=evidence_ids,
            evidence_text_by_id=evidence_text_by_id,
        )
    if not check["verified"]:
        return None
    return {
        "value": _serialize_value(percent_value),
        "evidence_ids": list(check["matched_evidence_ids"]),
    }


def _require_named_directional_operands(
    op: str, step: Mapping[str, Any]
) -> tuple[Any, Any]:
    if "new" in step and "old" in step:
        return step["new"], step["old"]
    raw_args = step.get("args")
    if isinstance(raw_args, Mapping) and "new" in raw_args and "old" in raw_args:
        extra = sorted(str(key) for key in raw_args if key not in {"new", "old"})
        if extra:
            raise CalculationPlanError(
                f"{op} named args mismatch: extra=" + ",".join(extra)
            )
        return raw_args["new"], raw_args["old"]
    else:
        raise CalculationPlanError(
            f"{op} requires named 'new' and 'old' operands"
        )


def _operation_arg_specs(op: str, step: Mapping[str, Any]) -> list[Any]:
    raw_args = step.get("args", [])
    if isinstance(raw_args, list):
        return list(raw_args)
    if not isinstance(raw_args, Mapping):
        raise CalculationPlanError(f"{op} args must be a list or supported named object")

    named_roles = {
        "abs": ("value",),
        "div": ("numerator", "denominator"),
        "pct_change": ("new", "old"),
        "pct_point_delta": ("new", "old"),
        "date_add_days": ("date", "days"),
        "next_workday": ("date",),
        "days_between": ("end", "start"),
    }.get(op)
    if named_roles is None:
        raise CalculationPlanError(f"{op} args must be a list")
    missing = [role for role in named_roles if role not in raw_args]
    extra = sorted(str(key) for key in raw_args if key not in named_roles)
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise CalculationPlanError(f"{op} named args mismatch: {'; '.join(details)}")
    return [raw_args[role] for role in named_roles]


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


def _parse_variable_value(value: Any, value_type: str, unit: str) -> Any:
    if value_type != "decimal":
        return _parse_value(value, value_type)
    text = str(value).strip()
    if text.endswith("%"):
        if _variable_kind(value_type, unit) != "percent_points":
            raise CalculationPlanError(
                "A decimal value with a % suffix must also declare a percentage unit"
            )
        text = text[:-1].strip()
    return _decimal(text)


def _resolve(value: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping) and "ref" in value:
        ref = str(value["ref"])
        if ref not in values:
            raise CalculationPlanError(f"Unknown reference: {ref}")
        return values[ref]
    if isinstance(value, Mapping) and "literal" in value:
        value_type = str(value.get("value_type", "text"))
        return _parse_value(value["literal"], value_type)
    if isinstance(value, str) and value in values:
        return values[value]
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
    percent_match = re.fullmatch(r"percent([012])(_bare)?", format_name)
    if percent_match:
        numeric = _decimal(value)
        if value_kind == "ratio":
            numeric *= Decimal("100")
        places = int(percent_match.group(1))
        quantum = Decimal("1").scaleb(-places)
        rendered = format(numeric.quantize(quantum, rounding=ROUND_HALF_UP), "f")
        return rendered + ("" if percent_match.group(2) else "%")
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


def _answers_differ_only_in_format(
    before: Sequence[str], after: Sequence[str]
) -> bool:
    if len(before) != len(after):
        return False
    for old, new in zip(before, after):
        if old == new:
            continue
        try:
            old_number = Decimal(str(old).strip().removesuffix("%"))
            new_number = Decimal(str(new).strip().removesuffix("%"))
        except InvalidOperation:
            return False
        if old_number != new_number:
            return False
    return True


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
    if isinstance(value, str) and value in kinds:
        return kinds[value]
    return "decimal"


def _first_kind(kinds: Sequence[str]) -> str:
    return kinds[0] if kinds else "decimal"


def _multiplication_kind(kinds: Sequence[str]) -> str:
    if "amount" in kinds:
        return "amount"
    if kinds and all(kind in {"ratio", "percent_points"} for kind in kinds):
        return "ratio"
    return "decimal"


def _is_ratio_operand(value: Decimal, kind: str) -> bool:
    return kind == "ratio" or (kind == "decimal" and abs(value) == Decimal("1"))


def _format_for_slot_contract(
    template: str,
    value: Any,
    *,
    requested_format: str,
    numeric_decimal_places: int | None,
    percent_suffix: bool | None,
) -> str:
    if ">" in template:
        return "text"
    if template.endswith("%") or re.fullmatch(r"9+\.99", template):
        if isinstance(value, date):
            return "date_cn"
        places = 2 if numeric_decimal_places is None else numeric_decimal_places
        if places not in {0, 1, 2}:
            raise CalculationPlanError(
                f"Unsupported question numeric precision: {numeric_decimal_places}"
            )
        template_requires_percent = template.endswith("%")
        requires_percent = (
            template_requires_percent if percent_suffix is None else percent_suffix
        )
        percent_semantics = requested_format.startswith("percent")
        if requires_percent or percent_semantics:
            return f"percent{places}" + ("" if requires_percent else "_bare")
        return f"decimal{places}"
    raise CalculationPlanError(f"Unsupported output slot template: {template!r}")


_NUMBER_RE = re.compile(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?(?![\d.])")


def _validate_disclosed_aggregate_scope(
    plan: Mapping[str, Any],
    semantic_constraints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep an explicitly disclosed aggregate from masquerading as a period input."""

    checks: list[dict[str, Any]] = []
    variables = [
        item
        for item in plan.get("variables", [])
        if isinstance(item, Mapping)
    ]
    direct_output_refs = {
        str(source.get("ref", "")).strip()
        if isinstance(source, Mapping)
        else str(source).strip()
        for output in plan.get("outputs", [])
        if isinstance(output, Mapping)
        for source in [output.get("source")]
    }
    for constraint in semantic_constraints:
        if constraint.get("type") != "direct_disclosed_aggregate":
            continue
        evidence_id = str(constraint.get("evidence_id", "")).strip()
        disclosed_value = str(constraint.get("disclosed_value", "")).strip()
        unit = str(constraint.get("unit", "")).strip()
        period = str(constraint.get("period", "")).strip()
        metric = str(constraint.get("metric", "")).strip()
        aggregation_scope = str(
            constraint.get("aggregation_scope", "")
        ).strip()
        if (
            not evidence_id
            or not disclosed_value
            or not period
            or not metric
            or aggregation_scope != "multi_period_mean"
        ):
            raise CalculationPlanError(
                "disclosed aggregate scope mismatch: incomplete semantic constraint"
            )
        try:
            normalized_value = _decimal(disclosed_value.rstrip("%"))
        except CalculationPlanError as exc:
            raise CalculationPlanError(
                "disclosed aggregate scope mismatch: invalid disclosed value"
            ) from exc

        period_years = set(re.findall(r"20\d{2}", period))
        matched_variables: list[str] = []
        mislabeled_variables: list[str] = []
        for variable in variables:
            raw_value = str(variable.get("value", "")).strip().rstrip("%")
            try:
                value_matches = _decimal(raw_value) == normalized_value
            except CalculationPlanError:
                continue
            evidence_ids = variable.get("evidence_ids", [])
            if (
                not value_matches
                or str(variable.get("unit", "")).strip().casefold()
                != unit.casefold()
                or not isinstance(evidence_ids, list)
                or evidence_id not in {str(item) for item in evidence_ids}
            ):
                continue
            name = str(variable.get("name", "")).strip()
            if metric not in name:
                continue
            variable_years = set(re.findall(r"20\d{2}", name))
            if (
                len(variable_years) == 1
                and len(period_years) > 1
                and variable_years < period_years
                and not any(marker in name for marker in ("平均", "年均", "均值"))
            ):
                mislabeled_variables.append(name)
                continue
            matched_variables.append(name)

        direct_matches = sorted(set(matched_variables) & direct_output_refs)
        verified = bool(direct_matches)
        check = {
            "type": "direct_disclosed_aggregate",
            "aggregation_scope": aggregation_scope,
            "period": period,
            "metric": metric,
            "evidence_id": evidence_id,
            "verified": verified,
            "direct_output_variable_names": direct_matches,
            "mislabeled_variable_names": mislabeled_variables,
        }
        checks.append(check)
        if not verified:
            detail = (
                f" aggregate value was mislabeled as {','.join(mislabeled_variables)}"
                if mislabeled_variables
                else " disclosed aggregate must be used as a direct output"
            )
            raise CalculationPlanError(
                "disclosed aggregate scope mismatch:" + detail
            )
    return checks


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
    value_matched: list[str] = []
    unit_matched: list[str] = []
    for evidence_id, text in available.items():
        value_appears = _value_appears(value, value_type, text)
        unit_appears = _unit_appears(unit, text)
        if value_appears:
            value_matched.append(evidence_id)
        if unit_appears:
            unit_matched.append(evidence_id)
        if value_appears and unit_appears:
            matched.append(evidence_id)
    if matched:
        reason = "matched_literal_value_and_unit"
    elif value_matched and not unit_matched:
        reason = "unit_not_found"
    elif unit_matched and not value_matched:
        reason = "value_not_found"
    else:
        reason = "value_and_unit_not_found"
    return {
        "name": name,
        "verified": bool(matched),
        "reason": reason,
        "matched_evidence_ids": matched,
        "value_matched_evidence_ids": value_matched,
        "unit_matched_evidence_ids": unit_matched,
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
        if target == target.to_integral_value():
            chinese = _integer_to_chinese(int(target))
            if chinese and re.search(
                rf"(?<![负零〇一二两三四五六七八九十百千万亿])"
                rf"{re.escape(chinese)}"
                rf"(?![零〇一二两三四五六七八九十百千万亿])",
                text,
            ):
                return True
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
    normalized = "".join(unit.split()).casefold()
    if not normalized or normalized in {"无", "个", "日", "天"}:
        return True
    aliases = {
        "%": ("%", "百分比"),
        "百分点": ("百分点", "%"),
        "元/股": ("元/股", "元／股", "每股"),
    }
    compact_text = "".join(text.split()).casefold()
    return any(
        token.casefold() in compact_text
        for token in aliases.get(normalized, (normalized,))
    )


def check_variable_grounding(
    *,
    name: str,
    value: Any,
    value_type: str,
    unit: str,
    evidence_ids: Sequence[str],
    evidence_text_by_id: Mapping[str, str],
) -> dict[str, Any]:
    """Expose the executor's literal grounding result to plan normalizers."""

    return _grounding_check(
        name=name,
        value=value,
        value_type=value_type,
        unit=unit,
        evidence_ids=evidence_ids,
        evidence_text_by_id=evidence_text_by_id,
    )


def _integer_to_chinese(value: int) -> str:
    if value < 0:
        positive = _integer_to_chinese(-value)
        return f"负{positive}" if positive else ""
    if value > 9999:
        return ""
    if value == 0:
        return "零"
    digits = "零一二三四五六七八九"
    units = ("", "十", "百", "千")
    chars: list[str] = []
    pending_zero = False
    for position in range(3, -1, -1):
        factor = 10**position
        digit = value // factor
        value %= factor
        if digit == 0:
            if chars and value:
                pending_zero = True
            continue
        if pending_zero:
            chars.append("零")
            pending_zero = False
        if not (digit == 1 and position == 1 and not chars):
            chars.append(digits[digit])
        chars.append(units[position])
    return "".join(chars)
