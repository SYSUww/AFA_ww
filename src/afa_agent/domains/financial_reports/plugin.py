from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from afa_agent.client import OpenAICompatibleClient
from afa_agent.config import build_run_config
from afa_agent.domains.base import DomainPlugin
from afa_agent.domains.common import detect_title, load_text_by_source, make_units_from_sections, split_text_into_sections
from afa_agent.domains.financial_reports.solver import FinancialReportsSolver
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.io_utils import read_json, write_json
from afa_agent.models import Document, Question


FINANCIAL_METRICS = [
    "营业收入",
    "营业总收入",
    "归属于上市公司股东的净利润",
    "归母净利润",
    "经营活动产生的现金流量净额",
    "研发投入",
    "研发投入占营业收入的比例",
    "每10股派",
    "现金分红",
    "分红比例",
]


class FinancialReportsPlugin(DomainPlugin):
    name = "financial_reports"
    strategy_label = "metric-extraction+bm25+rule-compare"
    strategy_details = [
        "PDF抽文本后切段并抽取metric_row",
        "指标名、年份、公司名驱动检索",
        "同指标双文档命中时优先做本地数值比较",
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
            metric_sections = self._extract_metric_sections(text, title, doc_id)
            units = make_units_from_sections(document, sections) + make_units_from_sections(document, metric_sections)
            units_payload.extend([unit.to_dict() for unit in units])
        write_json(output_path, {"documents": docs_payload, "units": units_payload})

    def build_index(self, parsed_path: Path, output_path: Path) -> None:
        parsed = read_json(parsed_path)
        write_json(
            output_path,
            {
                "domain": self.name,
                "documents": parsed["documents"],
                "units": parsed["units"],
                "unit_count": len(parsed["units"]),
            },
        )

    def answer_questions(self, questions: list[Question], parsed_path: Path, index_path: Path):
        return [self.answer_one(question, parsed_path, index_path) for question in questions]

    def answer_one(self, question: Question, parsed_path: Path, index_path: Path):
        index_payload = read_json(index_path)
        retriever = GenericBM25Retriever(index_payload["units"])
        config = build_run_config()
        if not config.model:
            raise RuntimeError("Missing model config in .env")
        client = OpenAICompatibleClient(config.model)
        solver = FinancialReportsSolver(client, retriever, index_payload["units"])
        return solver.solve(question)

    def _extract_metric_sections(self, text: str, title: str, doc_id: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections: list[dict[str, Any]] = []
        section_index = 1
        for idx, line in enumerate(lines):
            if not any(metric in line for metric in FINANCIAL_METRICS):
                continue
            if not re.search(r"\d", line):
                continue
            if "注" in line[:3] and "同比" not in line:
                continue
            window = "\n".join(lines[max(0, idx - 1): min(len(lines), idx + 2)])
            matched_metric = next(metric for metric in FINANCIAL_METRICS if metric in line)
            year_match = re.search(r"(20\d{2})", doc_id)
            sections.append(
                {
                    "section_id": f"metric_{section_index}",
                    "title_path": [title, matched_metric],
                    "text": window,
                    "unit_type": "metric_row",
                    "metadata": {
                        "metric_name": matched_metric,
                        "year": year_match.group(1) if year_match else "",
                        "numbers": [str(num) for num in re.findall(r"\d[\d,]*(?:\.\d+)?", window)],
                    },
                    "max_chars": 600,
                }
            )
            section_index += 1
        return sections
