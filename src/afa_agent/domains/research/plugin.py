from __future__ import annotations

from pathlib import Path

from afa_agent.client import OpenAICompatibleClient
from afa_agent.config import build_run_config
from afa_agent.domains.base import DomainPlugin
from afa_agent.domains.common import detect_title, load_text_by_source, make_units_from_sections, split_text_into_sections
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.research.solver import ResearchSolver
from afa_agent.io_utils import read_json, write_json
from afa_agent.models import Document, Question


RESEARCH_KEYWORDS = ["预计", "同比", "市场规模", "渗透率", "增速", "结论", "判断", "投资建议"]


class ResearchPlugin(DomainPlugin):
    name = "research"
    strategy_label = "conclusion-block+bm25+cross-doc-check"
    strategy_details = [
        "按段落切分并额外标记结论块",
        "跨研报题优先保留多文档证据",
        "逐选项核对数据和结论",
    ]

    def parse(self, manifest_path: Path, output_path: Path) -> None:
        manifest = read_json(manifest_path)
        domain_manifest = manifest["domains"][self.name]
        referenced_doc_ids = set(domain_manifest.get("referenced_doc_ids", []))
        docs_payload = []
        units_payload = []
        for doc_id, record in domain_manifest["documents"].items():
            if referenced_doc_ids and doc_id not in referenced_doc_ids:
                continue
            source_path = Path(record["source_path"])
            text, metadata = load_text_by_source(source_path, record["source_type"])
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
            sections = split_text_into_sections(text, title, "sec", unit_type="paragraph", max_chars=1000)
            highlight_sections = []
            for section in sections:
                if any(keyword in section["text"] for keyword in RESEARCH_KEYWORDS):
                    section_copy = dict(section)
                    section_copy["unit_type"] = "conclusion_block"
                    highlight_sections.append(section_copy)
            units = make_units_from_sections(document, sections) + make_units_from_sections(document, highlight_sections)
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
        solver = ResearchSolver(OpenAICompatibleClient(config.model), retriever)
        return solver.solve(question)
