from __future__ import annotations

import re


_ALLOWED_MODEL_RE = re.compile(
    r"(?:^|[/_:\-])qwen[-_]?3[._-]?[567](?:$|[-_/.:])",
    flags=re.IGNORECASE,
)


def is_allowed_submission_model(model_name: str) -> bool:
    """Whether a configured model belongs to an allowed Qwen3.5/3.6/3.7 family."""

    return bool(_ALLOWED_MODEL_RE.search(str(model_name).strip()))


def require_allowed_submission_model(model_name: str) -> None:
    if not is_allowed_submission_model(model_name):
        raise ValueError(
            "B-board submission generation requires a Qwen3.5/Qwen3.6/Qwen3.7 model; "
            f"configured model is {model_name!r}"
        )
