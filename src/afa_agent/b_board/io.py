from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUBMISSION_COLUMNS = (
    "qid",
    "answer1",
    "answer2",
    "answer3",
    "answer4",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning",
)

# The question package predates the July 2026 scoring change.  Keep accepting
# its eight-column CSV as a slot template, but never emit that legacy schema as
# a submission.
LEGACY_TEMPLATE_COLUMNS = (
    "qid",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)

_QUESTION_TYPE_ALIASES = {
    "判断题": "tf",
    "判断": "tf",
    "tf": "tf",
    "true_false": "tf",
    "单选题": "mcq",
    "单选": "mcq",
    "mcq": "mcq",
    "single_choice": "mcq",
    "多选题": "multi",
    "多选": "multi",
    "multi": "multi",
    "multiple_choice": "multi",
    "计算题": "calculation",
    "计算": "calculation",
    "calculation": "calculation",
    "抽取题": "extraction",
    "抽取": "extraction",
    "extraction": "extraction",
}

_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_EXPLICIT_DECIMAL_PLACES_RE = re.compile(
    r"(?:保留|精确到|四舍五入到|四舍五入至)(?:小数点后)?\s*([零一二两012])\s*位小数"
)
_DECIMAL_PLACE_VALUES = {"零": 0, "0": 0, "一": 1, "1": 1, "二": 2, "两": 2, "2": 2}


@dataclass(slots=True)
class BQuestion:
    """A real B-board question with its submission-slot contract attached."""

    qid: str
    domain: str
    split: str
    question: str
    options: dict[str, str]
    answer_format: str
    type: str
    answer_slots: int
    answer_slot_templates: tuple[str, ...]
    doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def slot_count(self) -> int:
        return self.answer_slots

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BAnswer:
    """The single answer contract consumed by the official B-board writer."""

    qid: str
    answer_parts: tuple[str, ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int | None = None
    reasoning: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer_parts", tuple(self.answer_parts))
        _require_nonempty_text(self.qid, "answer qid")
        _require_token(self.prompt_tokens, f"{self.qid}.prompt_tokens")
        _require_token(self.completion_tokens, f"{self.qid}.completion_tokens")
        expected_total = self.prompt_tokens + self.completion_tokens
        if self.total_tokens is None:
            object.__setattr__(self, "total_tokens", expected_total)
        else:
            _require_token(self.total_tokens, f"{self.qid}.total_tokens")
            if self.total_tokens != expected_total:
                raise ValueError(
                    f"{self.qid}: total_tokens={self.total_tokens} does not equal "
                    f"prompt_tokens + completion_tokens ({expected_total})"
                )
        if not isinstance(self.reasoning, str):
            raise ValueError(f"{self.qid}.reasoning must be a string")
        if "\x00" in self.reasoning:
            raise ValueError(f"{self.qid}.reasoning contains a NUL character")

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "answer_parts": list(self.answer_parts),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning": self.reasoning,
        }


# A descriptive alias for callers that prefer parity with the existing AnswerResult name.
BAnswerResult = BAnswer


@dataclass(slots=True)
class SubmissionTemplate:
    qid_order: tuple[str, ...]
    answer_slot_templates: dict[str, tuple[str, ...]]

    def slot_count(self, qid: str) -> int:
        try:
            return len(self.answer_slot_templates[qid])
        except KeyError as exc:
            raise KeyError(f"qid {qid!r} is absent from the submission template") from exc


def normalize_question_type(raw_type: str) -> str:
    value = str(raw_type).strip()
    try:
        return _QUESTION_TYPE_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f"unsupported B-board question type: {raw_type!r}") from exc


def infer_requested_decimal_places(question_text: str) -> int | None:
    """Return an explicit 0-2 decimal-place instruction from a question."""

    match = _EXPLICIT_DECIMAL_PLACES_RE.search(str(question_text))
    return _DECIMAL_PLACE_VALUES[match.group(1)] if match else None


def infer_percent_suffix_requirement(
    question_text: str,
    *,
    slot_index: int,
    slot_count: int,
) -> bool | None:
    """Return the percent-sign requirement for one answer slot.

    Explicit question instructions take precedence.  Otherwise, percentage
    semantics in the requested answer are interpreted according to README:
    percentage answers carry ``%`` while percentage-point answers do not.
    ``None`` means no percentage semantics were identified, so the submission
    template remains the fallback.
    """

    text = re.sub(r"\s+", "", str(question_text)).replace("％", "%")
    if "均不带%" in text or "均不带百分号" in text:
        return False
    scoped_negative = False
    if slot_index == 1 and re.search(r"前者不带(?:%|百分号)", text):
        return False
    if re.search(r"前者不带(?:%|百分号)", text):
        scoped_negative = True
    if slot_index == slot_count and re.search(r"后者不带(?:%|百分号)", text):
        return False
    if re.search(r"后者不带(?:%|百分号)", text):
        scoped_negative = True
    if not scoped_negative and re.search(r"(?:答案|结果)?不带(?:%|百分号)", text):
        return False
    scoped_positive = False
    if re.search(
        r"前者(?:(?:须|需|必须)?(?:填写|带|添加)(?:百分号|%)|以百分数计)",
        text,
    ):
        scoped_positive = True
        if slot_index == 1:
            return True
    if re.search(
        r"后者(?:(?:须|需|必须)?(?:填写|带|添加)(?:百分号|%)|以百分数计)",
        text,
    ):
        scoped_positive = True
        if slot_index == slot_count:
            return True
    if scoped_positive:
        return None
    if re.search(r"(?:须|需|必须)(?:填写|带|添加)(?:百分号|%)", text):
        return True

    format_match = re.search(r"答案格式为[‘“\"]([^’”\"]+)[’”\"]", text)
    if format_match:
        descriptors = re.split(r"[；;]", format_match.group(1))
        if len(descriptors) == slot_count:
            descriptor = descriptors[slot_index - 1]
            if "百分点" in descriptor:
                return False
            if "降幅" in descriptor and "百分点" in text and "同比" not in descriptor:
                return False
            if re.search(
                r"(?:百分数|同比增幅|同比增速|增长率|相对偏差|收益率|毛利率|净利率|现金流率|负债率|比例|占比)",
                descriptor,
            ):
                return True

    if slot_count == 1:
        request_clause = re.split(r"[。；;]|则", text.rstrip("。？！?!"))[-1]
        if "百分点" in request_clause:
            return False
        if re.search(
            r"(?:百分数|同比增幅|同比增速|增长率|相对偏差|收益率|毛利率|净利率|现金流率|负债率|比例|占比)",
            request_clause,
        ):
            return True
    return None


def read_b_question_file(path: Path | str) -> list[dict[str, Any]]:
    """Read either a JSON array/object or JSONL file, including an UTF-8 BOM."""

    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"question file is empty: {source}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {source}:{line_number}")
            rows.append(row)
        if not rows:
            raise ValueError(f"question file contains no rows: {source}")
        return rows

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and payload and all(isinstance(row, dict) for row in payload):
        return payload
    raise ValueError(f"expected a non-empty JSON object or array of objects: {source}")


def load_submission_template(path: Path | str) -> SubmissionTemplate:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns not in {SUBMISSION_COLUMNS, LEGACY_TEMPLATE_COLUMNS}:
            raise ValueError(
                f"{source}: expected current columns {SUBMISSION_COLUMNS} or legacy template "
                f"columns {LEGACY_TEMPLATE_COLUMNS}, got {columns}"
            )
        rows = list(reader)

    answer_prefix = "answer" if columns == SUBMISSION_COLUMNS else "answer_"

    if not rows or rows[0]["qid"].strip() != "summary":
        raise ValueError(f"{source}: summary must be the first data row")
    _validate_summary_row_shape(rows[0], source)

    order: list[str] = []
    slot_templates: dict[str, tuple[str, ...]] = {}
    prompt_total = 0
    completion_total = 0
    for row_number, row in enumerate(rows[1:], start=3):
        qid = row["qid"].strip()
        _require_nonempty_text(qid, f"{source}:{row_number} qid")
        if qid == "summary" or qid in slot_templates:
            raise ValueError(f"{source}:{row_number}: duplicate or reserved qid {qid!r}")
        slots = tuple(row[f"{answer_prefix}{index}"].strip() for index in range(1, 5))
        used = sum(bool(value) for value in slots)
        if used == 0 or slots[:used] != tuple(value for value in slots if value):
            raise ValueError(
                f"{source}:{row_number}: answer slots must contain 1-4 contiguous placeholders"
            )
        prompt, completion, _ = _parse_row_tokens(row, f"{source}:{row_number}")
        prompt_total += prompt
        completion_total += completion
        order.append(qid)
        slot_templates[qid] = slots[:used]

    if not order:
        raise ValueError(f"{source}: submission template contains no question rows")
    summary = _parse_row_tokens(rows[0], f"{source}:2")
    expected_summary = (prompt_total, completion_total, prompt_total + completion_total)
    if summary != expected_summary:
        raise ValueError(
            f"{source}: summary token totals do not equal the sum of question rows: "
            f"expected {expected_summary}, got {summary}"
        )
    return SubmissionTemplate(tuple(order), slot_templates)


def load_b_questions(
    question_root: Path | str,
    submission_template: Path | str | None = None,
) -> list[BQuestion]:
    """Load all real B questions and return them in official submission order."""

    root = Path(question_root)
    if root.name != "question_b" and (root / "question_b").is_dir():
        question_dir = root / "question_b"
        default_template = root / "submit.csv"
    else:
        question_dir = root
        default_template = root.parent / "submit.csv"
    template = load_submission_template(submission_template or default_template)

    files = sorted(
        path for path in question_dir.iterdir() if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    if not files:
        raise ValueError(f"no JSON or JSONL question files found under {question_dir}")

    raw_by_qid: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in files:
        for row in read_b_question_file(path):
            qid = str(row.get("qid", "")).strip()
            _require_nonempty_text(qid, f"{path} qid")
            if qid in raw_by_qid:
                raise ValueError(f"duplicate B-board qid {qid!r} in {path} and {raw_by_qid[qid][1]}")
            raw_by_qid[qid] = (row, path)

    expected = set(template.qid_order)
    actual = set(raw_by_qid)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"question/template qid mismatch: missing={missing}, extra={extra}")

    questions: list[BQuestion] = []
    for qid in template.qid_order:
        row, source = raw_by_qid[qid]
        questions.append(_build_question(row, source, template.answer_slot_templates[qid]))
    return questions


load_b_dataset = load_b_questions


def validate_b_answer(question: BQuestion, answer: BAnswer) -> None:
    if answer.qid != question.qid:
        raise ValueError(f"answer qid {answer.qid!r} does not match question {question.qid!r}")
    if len(answer.answer_parts) != question.answer_slots:
        raise ValueError(
            f"{question.qid}: expected {question.answer_slots} answer parts, "
            f"got {len(answer.answer_parts)}"
        )

    for index, part in enumerate(answer.answer_parts, start=1):
        if not isinstance(part, str) or not part:
            raise ValueError(f"{question.qid}.answer_{index} must be a non-empty string")
        if part != part.strip() or "\n" in part or "\r" in part:
            raise ValueError(f"{question.qid}.answer_{index} contains surrounding whitespace or newlines")

    if question.answer_format in {"tf", "mcq", "multi"}:
        if question.answer_slots != 1:
            raise ValueError(f"{question.qid}: choice questions must use exactly one answer slot")
        value = answer.answer_parts[0]
        if not re.fullmatch(r"[A-Z]+", value):
            raise ValueError(f"{question.qid}: choice answer must contain uppercase letters only")
        letters = list(value)
        if any(letter not in question.options for letter in letters):
            raise ValueError(f"{question.qid}: answer contains a letter absent from its options")
        if len(set(letters)) != len(letters) or letters != sorted(letters):
            raise ValueError(f"{question.qid}: choice letters must be unique and sorted")
        expected_length = 1 if question.answer_format in {"tf", "mcq"} else None
        if expected_length is not None and len(letters) != expected_length:
            raise ValueError(f"{question.qid}: {question.answer_format} requires one option")
        if question.answer_format == "multi" and len(letters) < 2:
            raise ValueError(f"{question.qid}: multi-choice answer requires at least two options")
        return

    decimal_places = infer_requested_decimal_places(question.question)
    for index, (part, template) in enumerate(
        zip(answer.answer_parts, question.answer_slot_templates), start=1
    ):
        validate_freeform_slot(
            part,
            template,
            f"{question.qid}.answer_{index}",
            numeric_decimal_places=decimal_places,
            percent_suffix=infer_percent_suffix_requirement(
                question.question,
                slot_index=index,
                slot_count=question.answer_slots,
            ),
        )


def write_b_submission(
    path: Path | str,
    questions: Sequence[BQuestion],
    answers: Iterable[BAnswer] | Mapping[str, BAnswer],
    *,
    audit_ready: bool = False,
) -> None:
    """Write and re-validate the current nine-column B-board CSV.

    ``audit_ready`` additionally rejects rows that would receive an automatic
    zero reasoning score or cannot plausibly originate from a model API call.
    """

    destination = Path(path)
    answer_by_qid = _index_answers(answers)
    _validate_question_answer_sets(questions, answer_by_qid)

    prompt_total = sum(answer_by_qid[question.qid].prompt_tokens for question in questions)
    completion_total = sum(answer_by_qid[question.qid].completion_tokens for question in questions)
    total = prompt_total + completion_total
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUBMISSION_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer1": "",
                "answer2": "",
                "answer3": "",
                "answer4": "",
                "prompt_tokens": prompt_total,
                "completion_tokens": completion_total,
                "total_tokens": total,
                "reasoning": "",
            }
        )
        for question in questions:
            answer = answer_by_qid[question.qid]
            row: dict[str, str | int | None] = {
                "qid": question.qid,
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "total_tokens": answer.total_tokens,
                "reasoning": answer.reasoning,
            }
            for index in range(1, 5):
                row[f"answer{index}"] = (
                    answer.answer_parts[index - 1] if index <= len(answer.answer_parts) else ""
                )
            writer.writerow(row)

    validate_b_submission(destination, questions, audit_ready=audit_ready)


write_submission_csv = write_b_submission


def validate_b_submission(
    path: Path | str,
    questions: Sequence[BQuestion],
    *,
    audit_ready: bool = False,
) -> list[BAnswer]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing_columns = [column for column in SUBMISSION_COLUMNS if column not in columns]
        if missing_columns:
            raise ValueError(
                f"{source}: submission is missing required columns {missing_columns}; "
                f"required columns are {SUBMISSION_COLUMNS}"
            )
        rows = list(reader)

    summary_rows = [row for row in rows if row["qid"] == "summary"]
    if len(summary_rows) > 1:
        raise ValueError(f"{source}: submission contains multiple summary rows")
    question_rows = [row for row in rows if row["qid"] != "summary"]
    if len(question_rows) != len(questions):
        raise ValueError(f"{source}: expected exactly {len(questions)} answer rows")
    if summary_rows:
        _validate_summary_row_shape(summary_rows[0], source)

    parsed: list[BAnswer] = []
    for row_number, (row, question) in enumerate(zip(question_rows, questions), start=2):
        if row["qid"] != question.qid:
            raise ValueError(
                f"{source}:{row_number}: expected qid {question.qid!r}, got {row['qid']!r}"
            )
        prompt, completion, total = _parse_row_tokens(row, f"{source}:{row_number}")
        parts = tuple(row[f"answer{index}"] for index in range(1, question.answer_slots + 1))
        if any(row[f"answer{index}"] for index in range(question.answer_slots + 1, 5)):
            raise ValueError(f"{source}:{row_number}: unused answer slots must be empty")
        answer = BAnswer(
            question.qid,
            parts,
            prompt,
            completion,
            total,
            row["reasoning"],
        )
        validate_b_answer(question, answer)
        if audit_ready:
            _validate_audit_ready_answer(answer, f"{source}:{row_number}")
        parsed.append(answer)

    expected_prompt = sum(answer.prompt_tokens for answer in parsed)
    expected_completion = sum(answer.completion_tokens for answer in parsed)
    expected_total = expected_prompt + expected_completion
    if summary_rows:
        summary_prompt, summary_completion, summary_total = _parse_row_tokens(
            summary_rows[0], f"{source}:summary"
        )
        if (summary_prompt, summary_completion, summary_total) != (
            expected_prompt,
            expected_completion,
            expected_total,
        ):
            raise ValueError(
                f"{source}: summary token totals do not equal the sum of question rows: "
                f"expected {(expected_prompt, expected_completion, expected_total)}, "
                f"got {(summary_prompt, summary_completion, summary_total)}"
            )
    return parsed


def _build_question(
    row: Mapping[str, Any], source: Path, slot_templates: tuple[str, ...]
) -> BQuestion:
    required = ("qid", "domain", "split", "question", "type", "options")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{source}: question row is missing required fields {missing}")

    qid = str(row["qid"]).strip()
    domain = str(row["domain"]).strip()
    split = str(row["split"]).strip()
    question = str(row["question"]).strip()
    raw_type = str(row["type"]).strip()
    for value, label in ((qid, "qid"), (domain, "domain"), (question, "question"), (raw_type, "type")):
        _require_nonempty_text(value, f"{source} {label}")
    if split != "B":
        raise ValueError(f"{source}:{qid}: split must be 'B', got {split!r}")

    answer_format = normalize_question_type(raw_type)
    raw_options = row["options"]
    if not isinstance(raw_options, dict):
        raise ValueError(f"{source}:{qid}: options must be an object")
    options = {str(key): str(value) for key, value in raw_options.items()}
    if answer_format in {"tf", "mcq", "multi"}:
        if not options:
            raise ValueError(f"{source}:{qid}: choice question has no options")
        if any(not re.fullmatch(r"[A-Z]", key) or not value.strip() for key, value in options.items()):
            raise ValueError(f"{source}:{qid}: choice options require uppercase labels and non-empty text")
        if len(slot_templates) != 1:
            raise ValueError(f"{source}:{qid}: choice question must have one submission slot")
    elif options:
        raise ValueError(f"{source}:{qid}: calculation/extraction options must be empty")

    metadata = {
        "source_path": str(source),
        "answer_slot_templates": list(slot_templates),
    }
    return BQuestion(
        qid=qid,
        domain=domain,
        split=split,
        question=question,
        options=options,
        answer_format=answer_format,
        type=raw_type,
        answer_slots=len(slot_templates),
        answer_slot_templates=slot_templates,
        metadata=metadata,
    )


def _index_answers(answers: Iterable[BAnswer] | Mapping[str, BAnswer]) -> dict[str, BAnswer]:
    values = answers.values() if isinstance(answers, Mapping) else answers
    indexed: dict[str, BAnswer] = {}
    for answer in values:
        if not isinstance(answer, BAnswer):
            raise TypeError(f"expected BAnswer, got {type(answer).__name__}")
        if answer.qid in indexed:
            raise ValueError(f"duplicate answer qid {answer.qid!r}")
        indexed[answer.qid] = answer
    return indexed


def _validate_question_answer_sets(
    questions: Sequence[BQuestion], answer_by_qid: Mapping[str, BAnswer]
) -> None:
    qids = [question.qid for question in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("question list contains duplicate qids")
    expected = set(qids)
    actual = set(answer_by_qid)
    if expected != actual:
        raise ValueError(
            f"question/answer qid mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for question in questions:
        validate_b_answer(question, answer_by_qid[question.qid])


def validate_freeform_slot(
    value: str,
    template: str,
    context: str = "answer",
    *,
    numeric_decimal_places: int | None = None,
    percent_suffix: bool | None = None,
) -> None:
    """Validate one freeform value against the official slot shape.

    The numeric placeholder is also used by the official template for Chinese
    dates, so it accepts either the question-specific numeric precision or a
    valid date. It never accepts an explanatory sentence.
    """

    decimal_places = 2 if numeric_decimal_places is None else numeric_decimal_places
    if decimal_places not in {0, 1, 2}:
        raise ValueError(f"{context}: unsupported decimal-place requirement {decimal_places}")
    numeric_pattern = rf"-?(?:0|[1-9]\d*)" + (
        "" if decimal_places == 0 else rf"\.\d{{{decimal_places}}}"
    )
    decimal_label = "decimal" if decimal_places == 1 else "decimals"
    requires_percent = template.endswith("%") if percent_suffix is None else percent_suffix

    if ">" in template:
        if not re.fullmatch(r"[^>\s]+(?:>[^>\s]+)+", value):
            raise ValueError(
                f"{context}: ordering slot requires non-empty labels joined by half-width '>'"
            )
        return
    date_match = _DATE_RE.fullmatch(value)
    if date_match and re.fullmatch(r"9+\.99", template):
        year, month, day = (int(part) for part in date_match.groups())
        try:
            date(year, month, day)
        except ValueError as exc:
            raise ValueError(f"{context}: invalid Chinese date {value!r}") from exc
        return
    if template.endswith("%") or (
        re.fullmatch(r"9+\.99", template) and percent_suffix is not None
    ):
        suffix = "%" if requires_percent else ""
        if not re.fullmatch(numeric_pattern + re.escape(suffix), value):
            suffix_description = " and '%' suffix" if requires_percent else " without '%' suffix"
            raise ValueError(
                f"{context}: numeric slot requires exactly {decimal_places} {decimal_label}"
                f"{suffix_description}"
            )
        return
    if re.fullmatch(r"9+\.99", template):
        if not re.fullmatch(numeric_pattern, value):
            raise ValueError(
                f"{context}: numeric/date slot requires exactly {decimal_places} {decimal_label}"
                " or valid Chinese date"
            )
        return
    raise ValueError(f"{context}: unsupported freeform slot template {template!r}")


def _parse_row_tokens(row: Mapping[str, str], context: str) -> tuple[int, int, int]:
    values: list[int] = []
    for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        raw = row[field_name]
        if not re.fullmatch(r"\d+", raw):
            raise ValueError(f"{context}: {field_name} must be a non-negative integer")
        values.append(int(raw))
    prompt, completion, total = values
    if total != prompt + completion:
        raise ValueError(f"{context}: total_tokens must equal prompt_tokens + completion_tokens")
    return prompt, completion, total


def _validate_summary_row_shape(row: Mapping[str, str], source: Path) -> None:
    answer_prefix = "answer" if "answer1" in row else "answer_"
    if any(row[f"{answer_prefix}{index}"] for index in range(1, 5)):
        raise ValueError(f"{source}: summary answer slots must be empty")
    if "reasoning" in row and row["reasoning"]:
        raise ValueError(f"{source}: summary reasoning must be empty")
    _parse_row_tokens(row, f"{source}:2")


def _validate_audit_ready_answer(answer: BAnswer, context: str) -> None:
    reasoning = re.sub(r"\s+", "", answer.reasoning)
    if len(reasoning) < 20:
        raise ValueError(f"{context}: reasoning must contain at least 20 non-whitespace characters")
    if answer.prompt_tokens <= 0 or answer.completion_tokens <= 0:
        raise ValueError(
            f"{context}: audit-ready rows require positive prompt_tokens and completion_tokens"
        )


def _require_nonempty_text(value: object, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")


def _require_token(value: object, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
