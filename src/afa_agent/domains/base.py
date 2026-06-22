from __future__ import annotations

from pathlib import Path

from afa_agent.models import AnswerResult, Document, EvidenceUnit, Question


class DomainPlugin:
    name: str
    strategy_label: str = "unspecified"
    strategy_details: list[str] = []

    def parse(self, manifest_path: Path, output_path: Path) -> None:
        raise NotImplementedError

    def build_index(self, parsed_path: Path, output_path: Path) -> None:
        raise NotImplementedError

    def answer_questions(
        self,
        questions: list[Question],
        parsed_path: Path,
        index_path: Path,
    ) -> list[AnswerResult]:
        raise NotImplementedError
