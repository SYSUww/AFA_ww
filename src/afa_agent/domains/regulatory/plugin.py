from __future__ import annotations

from pathlib import Path

from afa_agent.client import OpenAICompatibleClient
from afa_agent.config import build_run_config
from afa_agent.domains.base import DomainPlugin
from afa_agent.domains.regulatory.parser import (
    detect_title,
    extract_text_from_html,
    extract_text_from_markdown,
    extract_text_from_pdf,
    extract_text_from_txt,
    split_regulatory_units,
)
from afa_agent.domains.regulatory.retriever import RegulatoryRetriever
from afa_agent.domains.regulatory.solver import RegulatorySolver
from afa_agent.io_utils import read_json, write_json
from afa_agent.models import Document, Question
from afa_agent.strategy import get_stage_settings


class RegulatoryPlugin(DomainPlugin):
    name = "regulatory"
    strategy_label = "rule+bm25+article-judgment"
    strategy_details = [
        "txt/html优先，附件PDF兜底",
        "按法条切分并做相邻条款回溯",
        "逐选项判别，必要时单选/多选复核",
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
            source_type = record["source_type"]
            if source_type == "txt":
                text, metadata = extract_text_from_txt(source_path)
            elif source_type in {"md", "markdown"}:
                text, metadata = extract_text_from_markdown(source_path)
            elif source_type == "html":
                text, metadata = extract_text_from_html(source_path)
            elif source_type == "pdf":
                text, metadata = extract_text_from_pdf(source_path, options=parse_settings)
            else:
                raise ValueError(f"Unsupported source_type: {source_type}")
            title = metadata.get("title") or detect_title(text, doc_id)
            document = Document(
                doc_id=doc_id,
                domain=self.name,
                title=title,
                source_type=source_type,
                source_path=str(source_path),
                metadata=metadata,
            )
            docs_payload.append(document.to_dict())
            units = split_regulatory_units(
                document,
                text,
                max_chars=segmentation_settings.get("article_max_chars", 1200),
            )
            units_payload.extend([unit.to_dict() for unit in units])
        write_json(output_path, {"documents": docs_payload, "units": units_payload})

    def build_index(self, parsed_path: Path, output_path: Path) -> None:
        parsed = read_json(parsed_path)
        output = {
            "domain": self.name,
            "unit_count": len(parsed["units"]),
            "doc_count": len(parsed["documents"]),
            "units": parsed["units"],
        }
        write_json(output_path, output)

    def answer_questions(
        self,
        questions: list[Question],
        parsed_path: Path,
        index_path: Path,
    ):
        return [self.answer_one(question, parsed_path, index_path) for question in questions]

    def answer_one(
        self,
        question: Question,
        parsed_path: Path,
        index_path: Path,
    ):
        parsed = read_json(parsed_path)
        index_payload = read_json(index_path)
        retriever = RegulatoryRetriever(index_payload["units"])
        config = build_run_config()
        if not config.model:
            raise RuntimeError("Missing model config in .env")
        client = OpenAICompatibleClient(config.model)
        solver = RegulatorySolver(client, retriever, strategy=self.name)
        return solver.solve(question)
