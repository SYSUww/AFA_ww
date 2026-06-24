from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from afa_agent.client import OpenAICompatibleClient
from afa_agent.config import build_run_config
from afa_agent.domains.base import DomainPlugin
from afa_agent.domains.common import detect_title, load_text_by_source, make_units_from_sections, split_text_into_sections
from afa_agent.domains.financial_contracts.solver import FinancialContractsSolver
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.io_utils import read_json, write_json
from afa_agent.models import Document, Question
from afa_agent.strategy import get_stage_settings


CONTRACT_KEYWORDS = [
    "发行人",
    "发行金额",
    "主体信用评级",
    "债项信用评级",
    "受托管理人",
    "主承销商",
    "期限",
    "利率",
    "回售",
    "赎回",
    "偿付",
    "违约",
]


class FinancialContractsPlugin(DomainPlugin):
    name = "financial_contracts"
    strategy_label = "element-block+bm25+clause-verification"
    strategy_details = [
        "抽取发行要素和关键条款块",
        "发行主体、评级、金额、机构词驱动检索",
        "逐选项核对发行要素与权利义务条款",
    ]

    def parse(self, manifest_path: Path, output_path: Path) -> None:
        parse_settings = get_stage_settings(self.name, "pdf_parse")
        segmentation_settings = get_stage_settings(self.name, "segmentation")
        manifest = read_json(manifest_path)
        domain_manifest = manifest["domains"][self.name]
        referenced_doc_ids = set(domain_manifest.get("referenced_doc_ids", []))
        docs_payload = []
        units_payload = []
        for doc_id, record in domain_manifest["documents"].items():
            if referenced_doc_ids and doc_id not in referenced_doc_ids:
                continue
            source_path = Path(record["source_path"])
            text, metadata = load_text_by_source(source_path, record["source_type"], options=parse_settings)
            title = detect_title(text, doc_id)
            document = Document(
                doc_id=doc_id,
                domain=self.name,
                title=title,
                source_type=record["source_type"],
                source_path=str(source_path),
                metadata=metadata,
            )
            docs_payload.append(document.to_dict())
            sections = split_text_into_sections(
                text,
                title,
                "sec",
                unit_type="paragraph",
                max_chars=segmentation_settings.get("paragraph_max_chars", 900),
            )
            element_sections = self._extract_element_sections(
                text,
                title,
                max_chars=segmentation_settings.get("element_max_chars", 700),
            )
            units = make_units_from_sections(document, sections) + make_units_from_sections(document, element_sections)
            units_payload.extend([unit.to_dict() for unit in units])
        write_json(output_path, {"documents": docs_payload, "units": units_payload})

    def build_index(self, parsed_path: Path, output_path: Path) -> None:
        parsed = read_json(parsed_path)
        write_json(output_path, {"domain": self.name, "documents": parsed["documents"], "units": parsed["units"]})

    def answer_questions(self, questions: list[Question], parsed_path: Path, index_path: Path):
        return [self.answer_one(question, parsed_path, index_path) for question in questions]

    def answer_one(self, question: Question, parsed_path: Path, index_path: Path):
        index_payload = read_json(index_path)
        retriever = GenericBM25Retriever(index_payload["units"])
        config = build_run_config()
        if not config.model:
            raise RuntimeError("Missing model config in .env")
        solver = FinancialContractsSolver(OpenAICompatibleClient(config.model), retriever, strategy=self.name)
        return solver.solve(question)

    def _extract_element_sections(self, text: str, title: str, max_chars: int = 700) -> list[dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections: list[dict[str, Any]] = []
        for idx, line in enumerate(lines):
            if not any(keyword in line for keyword in CONTRACT_KEYWORDS):
                continue
            window = "\n".join(lines[max(0, idx - 1): min(len(lines), idx + 3)])
            matched = next(keyword for keyword in CONTRACT_KEYWORDS if keyword in line)
            sections.append(
                {
                    "section_id": f"element_{idx+1}",
                    "title_path": [title, matched],
                    "text": window,
                    "unit_type": "element_block",
                    "metadata": {
                        "element_name": matched,
                        "numbers": re.findall(r"\d[\d,]*(?:\.\d+)?", window),
                    },
                    "max_chars": max_chars,
                }
            )
        return sections
