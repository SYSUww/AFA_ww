from __future__ import annotations

import ast
import re
import unittest
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CALCULATION_PATHS = (
    ROOT / "src" / "afa_agent" / "b_board" / "runner.py",
    ROOT / "src" / "afa_agent" / "b_board" / "calculation.py",
    ROOT / "src" / "afa_agent" / "b_board" / "calculation_profile.py",
)

_QID_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:fc|fin|ins|reg|res)_[ab]_\d{3}"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_DOCUMENT_ID_LITERAL_RE = re.compile(
    r"^(?:"
    r"(?:pack\d+_)?text\d{1,3}"
    r"|annual_[a-z0-9_]+_report"
    r"|csrc_\d{4}_att\d+"
    r")$",
    re.IGNORECASE,
)
_ROUTING_FUNCTION_MARKERS = (
    "answer_calculation",
    "doc_id",
    "evidence",
    "overlay",
    "query_terms",
    "retriev",
    "route",
    "select",
)
_HISTORICAL_MAPPING_NAME_RE = re.compile(
    r"(?:"
    r"benchmark|gold|historical|known_answers?|official_answers?"
    r"|answer_(?:map|overrides?|corrections?|by_qid)"
    r"|qid_(?:to|by)_answers?"
    r"|doc(?:ument)?_(?:map|overrides?|by_qid)"
    r"|qid_(?:to|by)_doc(?:ument)?_ids?"
    r")",
    re.IGNORECASE,
)
_REGEX_CALL_NAMES = {
    "findall",
    "finditer",
    "fullmatch",
    "match",
    "search",
}
_GENERIC_SYNTAX_FUNCTION_PREFIXES = (
    "_extract_",
    "_infer_",
    "_normalize_",
    "_parse_",
    "_requested_",
    "_validate_",
    "parse_",
)


@dataclass(frozen=True, slots=True)
class AuditViolation:
    path: str
    line: int
    category: str
    detail: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.category}: {self.detail}"


def audit_python_source(source: str, *, path: str) -> list[AuditViolation]:
    tree = ast.parse(source, filename=path)
    violations: list[AuditViolation] = []
    routing_roots = _routing_roots_for_path(path)
    reachable_functions = (
        _reachable_function_names(tree, routing_roots)
        if routing_roots
        else None
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if match := _QID_LITERAL_RE.search(node.value):
                violations.append(
                    AuditViolation(
                        path,
                        node.lineno,
                        "qid_literal",
                        repr(match.group(0)),
                    )
                )
            if _DOCUMENT_ID_LITERAL_RE.fullmatch(node.value.strip()):
                violations.append(
                    AuditViolation(
                        path,
                        node.lineno,
                        "document_id_literal",
                        repr(node.value),
                    )
                )

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_names = _assignment_target_names(node)
            value = node.value
            if (
                value is not None
                and _is_literal_mapping(value)
                and any(
                    _HISTORICAL_MAPPING_NAME_RE.search(name)
                    for name in target_names
                )
            ):
                violations.append(
                    AuditViolation(
                        path,
                        node.lineno,
                        "historical_answer_or_document_mapping",
                        ", ".join(sorted(target_names)),
                    )
                )

    branch_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (
                ast.If,
                ast.IfExp,
                ast.Match,
                ast.While,
                ast.comprehension,
            ),
        )
    ]
    for node in branch_nodes:
        predicates = _branch_predicates(node)
        if any(_contains_question_qid_reference(item) for item in predicates):
            violations.append(
                AuditViolation(
                    path,
                    node.lineno,
                    "question_qid_branch",
                    "branching on question.qid is forbidden",
                )
            )

    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        if (
            reachable_functions is not None
            and function.name not in reachable_functions
        ):
            continue
        violations.extend(_audit_routing_function(function, path=path))

    return _deduplicate_violations(violations)


def audit_production_calculation_chain() -> list[AuditViolation]:
    violations: list[AuditViolation] = []
    for source_path in PRODUCTION_CALCULATION_PATHS:
        if not source_path.is_file():
            continue
        violations.extend(
            audit_python_source(
                source_path.read_text(encoding="utf-8"),
                path=str(source_path.relative_to(ROOT)),
            )
        )
    return violations


def _audit_routing_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    path: str,
) -> list[AuditViolation]:
    if function.name.startswith(_GENERIC_SYNTAX_FUNCTION_PREFIXES):
        return []
    routing_name = any(
        marker in function.name.casefold()
        for marker in _ROUTING_FUNCTION_MARKERS
    )
    tainted_names = _question_derived_names(function)
    directly_reads_question_text = any(
        isinstance(node, ast.Attribute)
        and node.attr == "question"
        and isinstance(node.value, ast.Name)
        and node.value.id == "question"
        for node in ast.walk(function)
    )
    if not routing_name and not tainted_names and not directly_reads_question_text:
        return []

    violations: list[AuditViolation] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Compare):
            if (
                _expression_references_question_text(node, tainted_names)
                and _contains_literal_string(node)
            ):
                violations.append(
                    AuditViolation(
                        path,
                        node.lineno,
                        "question_text_routing",
                        f"{function.name} branches on a literal question fragment",
                    )
                )
        elif isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            if (
                _expression_references_question_text(node, tainted_names)
                and _contains_literal_string(node)
                and any(
                    isinstance(item, ast.Compare)
                    and any(
                        isinstance(operator, (ast.In, ast.NotIn))
                        for operator in item.ops
                    )
                    for item in ast.walk(node)
                )
            ):
                violations.append(
                    AuditViolation(
                        path,
                        node.lineno,
                        "question_text_routing",
                        f"{function.name} routes on fixed question fragments",
                    )
                )
        elif isinstance(node, ast.Call) and _is_regex_call(node):
            if any(
                _expression_references_question_text(argument, tainted_names)
                for argument in node.args
            ):
                violations.append(
                    AuditViolation(
                        path,
                        node.lineno,
                        "question_regex_routing",
                        f"{function.name} routes with a regex over question text",
                    )
                )
    return violations


def _question_derived_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    names = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        if argument.arg != "question"
        and _looks_question_derived_name(argument.arg)
    }
    changed = True
    while changed:
        changed = False
        for node in ast.walk(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                continue
            value = node.value
            if value is None:
                continue
            if not _expression_references_question_text(value, names):
                continue
            for target_name in _assignment_target_names(node):
                if (
                    _looks_question_derived_name(target_name)
                    and target_name not in names
                ):
                    names.add(target_name)
                    changed = True
    return names


def _looks_question_derived_name(name: str) -> bool:
    normalized = name.casefold()
    return normalized in {
        "compact_question",
        "compact_query",
        "normalized_query",
        "prompt",
        "query",
        "question_query",
        "question_text",
    } or normalized.endswith(("_question", "_query", "_prompt", "_text"))


def _routing_roots_for_path(path: str) -> set[str]:
    filename = Path(path).name
    if filename == "runner.py":
        return {"_answer_calculation"}
    if filename == "calculation.py":
        return {"execute"}
    if filename == "calculation_profile.py":
        return {
            "build_calculation_profile_messages",
            "infer_calculation_thinking_policy",
            "parse_calculation_profile",
            "retrieval_queries",
        }
    return set()


def _reachable_function_names(
    tree: ast.Module,
    roots: set[str],
) -> set[str]:
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reachable = set(roots) & set(functions)
    pending = list(reachable)
    while pending:
        function_name = pending.pop()
        function = functions[function_name]
        called_names = {
            called
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            for called in [_called_function_name(node)]
            if called
        }
        for called in called_names & set(functions):
            if called not in reachable:
                reachable.add(called)
                pending.append(called)
    return reachable


def _called_function_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _expression_references_question_text(
    expression: ast.AST,
    tainted_names: set[str],
) -> bool:
    for node in ast.walk(expression):
        if isinstance(node, ast.Name) and node.id in tainted_names:
            return True
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "question"
            and isinstance(node.value, ast.Name)
            and node.value.id == "question"
        ):
            return True
    return False


def _contains_question_qid_reference(expression: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "qid"
        and isinstance(node.value, ast.Name)
        and node.value.id == "question"
        for node in ast.walk(expression)
    )


def _contains_literal_string(expression: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and bool(node.value.strip())
        for node in ast.walk(expression)
    )


def _is_regex_call(node: ast.Call) -> bool:
    function = node.func
    if not isinstance(function, ast.Attribute):
        return False
    if function.attr not in _REGEX_CALL_NAMES:
        return False
    if isinstance(function.value, ast.Name):
        return (
            function.value.id == "re"
            or function.value.id.endswith("_RE")
        )
    return False


def _branch_predicates(node: ast.AST) -> tuple[ast.AST, ...]:
    if isinstance(node, (ast.If, ast.IfExp, ast.While)):
        return (node.test,)
    if isinstance(node, ast.Match):
        return (node.subject,)
    if isinstance(node, ast.comprehension):
        return tuple(node.ifs)
    return ()


def _assignment_target_names(
    node: ast.Assign | ast.AnnAssign | ast.NamedExpr,
) -> set[str]:
    if isinstance(node, ast.Assign):
        targets = node.targets
    else:
        targets = [node.target]
    return {
        nested.id
        for target in targets
        for nested in ast.walk(target)
        if isinstance(nested, ast.Name)
    }


def _is_literal_mapping(node: ast.AST) -> bool:
    if isinstance(node, ast.Dict):
        return bool(node.keys) and all(
            key is None or isinstance(key, ast.Constant)
            for key in node.keys
        )
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and bool(node.keywords)
        and not node.args
    )


def _deduplicate_violations(
    violations: list[AuditViolation],
) -> list[AuditViolation]:
    return list(
        dict.fromkeys(
            sorted(
                violations,
                key=lambda item: (
                    item.path,
                    item.line,
                    item.category,
                    item.detail,
                ),
            )
        )
    )


class BBoardCalculationNoHardcodingTests(unittest.TestCase):
    def test_production_calculation_chain_has_no_hardcoded_routes(self) -> None:
        violations = audit_production_calculation_chain()
        self.assertEqual(
            violations,
            [],
            "\n" + "\n".join(item.render() for item in violations),
        )

    def test_auditor_rejects_qid_literals_and_qid_branches(self) -> None:
        violations = audit_python_source(
            dedent(
                """
                def choose(question):
                    if question.qid == "fin_b_018":
                        return "special"
                    return "generic"
                """
            ),
            path="synthetic.py",
        )

        self.assertEqual(
            {item.category for item in violations},
            {"qid_literal", "question_qid_branch"},
        )

    def test_auditor_rejects_question_routing_and_sealed_maps(self) -> None:
        violations = audit_python_source(
            dedent(
                """
                import re

                KNOWN_ANSWERS = {"case": "14.41"}
                QID_TO_DOC_IDS = {"case": ["annual_midea_2025_report"]}

                def select_calculation_evidence(question, retriever):
                    compact_question = question.question.replace(" ", "")
                    if "西部证券2023至2025年" in compact_question:
                        return retriever.search(["text14"], compact_question)
                    if re.search(r"美的集团.*资产负债率", compact_question):
                        return retriever.search([], compact_question)
                    return []
                """
            ),
            path="synthetic.py",
        )
        categories = {item.category for item in violations}

        self.assertIn("historical_answer_or_document_mapping", categories)
        self.assertIn("document_id_literal", categories)
        self.assertIn("question_text_routing", categories)
        self.assertIn("question_regex_routing", categories)

    def test_auditor_allows_generic_syntax_parsing(self) -> None:
        violations = audit_python_source(
            dedent(
                """
                import re

                OPERATOR_SYNTAX = {"percentage": "pct_change"}

                def _requested_period(question_text):
                    match = re.search(
                        r"(20\\d{2})年至(20\\d{2})年",
                        question_text,
                    )
                    return match.groups() if match else ()
                """
            ),
            path="synthetic.py",
        )

        self.assertEqual(violations, [])

    def test_audit_scope_excludes_logs_and_test_fixtures(self) -> None:
        self.assertTrue(PRODUCTION_CALCULATION_PATHS)
        self.assertTrue(
            all(
                path.is_relative_to(ROOT / "src")
                for path in PRODUCTION_CALCULATION_PATHS
            )
        )


if __name__ == "__main__":
    unittest.main()
