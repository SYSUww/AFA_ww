from __future__ import annotations

from pathlib import Path

from afa_agent.client import OpenAICompatibleClient
from afa_agent.config import build_run_config
from afa_agent.domains.base import DomainPlugin
from afa_agent.domains.common import detect_title, load_text_by_source, make_units_from_sections, split_text_into_sections
from afa_agent.domains.generic_retriever import GenericBM25Retriever, ensure_unique_unit_ids
from afa_agent.domains.insurance.solver import InsuranceSolver
from afa_agent.io_utils import read_json, write_json
from afa_agent.models import Document, Question
from afa_agent.strategy import get_stage_settings


INSURANCE_KEYWORDS = [
    "保险责任",
    "身故保险金",
    "现金价值",
    "账户价值",
    "已交保费",
    "基本保额",
    "退保",
    "领取",
    "年金",
]


class InsurancePlugin(DomainPlugin):
    name = "insurance"
    strategy_label = "clause-formula+bm25+single-shot-judge"
    strategy_details = [
        "抽取责任条款和公式块",
        "产品名、金额、场景条件驱动检索",
        "整题一次裁决以降低长计算题开销",
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
            title = detect_title(text, f"insurance_{doc_id}")
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
                unit_type="clause_block",
                max_chars=segmentation_settings.get("clause_max_chars", 900),
            )
            for section in sections:
                if any(keyword in section["text"] for keyword in INSURANCE_KEYWORDS):
                    section["unit_type"] = "formula_block" if any(keyword in section["text"] for keyword in ["账户价值", "已交保费", "基本保额"]) else "clause_block"
            units = make_units_from_sections(document, sections)
            units_payload.extend([unit.to_dict() for unit in units])
        ensure_unique_unit_ids(units_payload, context="insurance parsed units")
        write_json(output_path, {"documents": docs_payload, "units": units_payload})

    def build_index(self, parsed_path: Path, output_path: Path) -> None:
        parsed = read_json(parsed_path)
        ensure_unique_unit_ids(parsed["units"], context="insurance index units")
        write_json(output_path, {"domain": self.name, "documents": parsed["documents"], "units": parsed["units"]})

    def answer_questions(self, questions: list[Question], parsed_path: Path, index_path: Path):
        return [self.answer_one(question, parsed_path, index_path) for question in questions]

    def answer_one(self, question: Question, parsed_path: Path, index_path: Path):
        index_payload = read_json(index_path)
        retriever = GenericBM25Retriever(index_payload["units"])
        config = build_run_config()
        if not config.model:
            raise RuntimeError("Missing model config in .env")
        solver = InsuranceSolver(OpenAICompatibleClient(config.model), retriever, strategy=self.name)
        return solver.solve(question)
