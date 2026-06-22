#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.io_utils import write_json


def build_regulatory_manifest(raw_dir: Path) -> dict:
    txt_dir = raw_dir / "regulatory" / "txt"
    html_dir = raw_dir / "regulatory" / "html"
    attachments_dir = raw_dir / "regulatory" / "attachments"
    documents = {}
    for path in sorted(txt_dir.glob("*.txt")):
        documents[path.stem] = {"source_path": str(path), "source_type": "txt"}
    for path in sorted(html_dir.glob("*.html")):
        documents.setdefault(path.stem, {"source_path": str(path), "source_type": "html"})
    for path in sorted(attachments_dir.glob("*.pdf")):
        documents[path.stem] = {"source_path": str(path), "source_type": "pdf"}
    return {"documents": documents}


def build_generic_manifest(raw_dir: Path, domain: str) -> dict:
    domain_dir = raw_dir / domain
    documents = {}
    for path in sorted(domain_dir.glob("*")):
        if path.is_file():
            documents[path.stem] = {"source_path": str(path), "source_type": path.suffix.lstrip(".").lower()}
    return {"documents": documents}


def main() -> None:
    raw_dir = ROOT / "public_dataset_upload" / "raw"
    questions_dir = ROOT / "public_dataset_upload" / "questions" / "group_a"
    domains = {}
    for question_file in sorted(questions_dir.glob("*_questions.json")):
        domain = question_file.name.replace("_questions.json", "")
        question_rows = json.loads(question_file.read_text(encoding="utf-8"))
        referenced_doc_ids = sorted({doc_id for row in question_rows for doc_id in row.get("doc_ids", [])})
        domains[domain] = {
            "question_path": str(question_file),
            "documents": {},
            "referenced_doc_ids": referenced_doc_ids,
        }
    if "regulatory" in domains:
        domains["regulatory"] = {
            **domains["regulatory"],
            **build_regulatory_manifest(raw_dir),
        }
    for domain in list(domains.keys()):
        if domain == "regulatory":
            continue
        domains[domain] = {**domains[domain], **build_generic_manifest(raw_dir, domain)}

    payload = {"project_root": str(ROOT), "domains": domains}
    out_path = ROOT / "artifacts" / "manifest" / "dataset_manifest.json"
    write_json(out_path, payload)
    print(out_path)


if __name__ == "__main__":
    main()
