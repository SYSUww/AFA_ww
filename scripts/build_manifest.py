#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.io_utils import write_json


MINERU_MANIFEST_PATH = ROOT / "artifacts" / "mineru" / "manifest.json"
EXTRACTED_ROOT = ROOT / "artifacts" / "extracted"
EXTRACTED_CLEANED_ROOT = ROOT / "artifacts" / "extracted_cleaned"
EXTRACTED_SOURCE_ROOTS = [
    ("artifacts/extracted_cleaned", EXTRACTED_CLEANED_ROOT),
    ("artifacts/extracted", EXTRACTED_ROOT),
]


def load_mineru_manifest() -> dict[str, dict]:
    if not MINERU_MANIFEST_PATH.exists():
        return {}
    payload = json.loads(MINERU_MANIFEST_PATH.read_text(encoding="utf-8"))
    return payload.get("documents", {})


def source_type_for(path: Path) -> str:
    suffix = path.suffix.lstrip(".").lower()
    return "md" if suffix == "markdown" else suffix


def extracted_candidates(root: Path, domain: str, doc_id: str, default_type: str) -> list[Path]:
    domain_dir = root / domain
    if domain == "regulatory":
        if default_type == "pdf":
            return [domain_dir / "attachments" / f"{doc_id}.md"]
        if default_type == "txt":
            return [
                domain_dir / "txt" / f"{doc_id}.md",
                domain_dir / "txt" / f"{doc_id}.txt",
            ]
        if default_type == "html":
            return [
                domain_dir / "html" / f"{doc_id}.md",
                domain_dir / "html" / f"{doc_id}_extracted.html",
                domain_dir / "html" / f"{doc_id}.html",
            ]
    return [
        domain_dir / f"{doc_id}.md",
        domain_dir / f"{doc_id}.markdown",
        domain_dir / f"{doc_id}.txt",
        domain_dir / f"{doc_id}.html",
    ]


def maybe_use_extracted_source(
    *,
    domain: str,
    doc_id: str,
    default_path: Path,
    default_type: str,
) -> dict[str, str] | None:
    for source_label, root in EXTRACTED_SOURCE_ROOTS:
        for path in extracted_candidates(root, domain, doc_id, default_type):
            if path.exists():
                return {
                    "source_path": str(path),
                    "source_type": source_type_for(path),
                    "original_source_path": str(default_path),
                    "original_source_type": default_type,
                    "extracted_source": source_label,
                }
    return None


def maybe_use_mineru_source(
    *,
    domain: str,
    doc_id: str,
    default_path: Path,
    default_type: str,
    mineru_docs: dict[str, dict],
) -> dict[str, str]:
    extracted = maybe_use_extracted_source(
        domain=domain,
        doc_id=doc_id,
        default_path=default_path,
        default_type=default_type,
    )
    if extracted is not None:
        return extracted

    key = f"{domain}::{doc_id}"
    mineru_meta = mineru_docs.get(key, {})
    content_md = mineru_meta.get("content_md")
    if default_type == "pdf" and mineru_meta.get("status") == "success" and content_md and Path(content_md).exists():
        return {
            "source_path": content_md,
            "source_type": "md",
            "original_source_path": str(default_path),
            "original_source_type": default_type,
            "mineru_meta_path": str(Path(mineru_meta.get("normalized_dir", "")).parent / "meta.json") if mineru_meta.get("normalized_dir") else "",
        }
    return {"source_path": str(default_path), "source_type": default_type}


def build_regulatory_manifest(raw_dir: Path, mineru_docs: dict[str, dict]) -> dict:
    txt_dir = raw_dir / "regulatory" / "txt"
    html_dir = raw_dir / "regulatory" / "html"
    attachments_dir = raw_dir / "regulatory" / "attachments"
    documents = {}
    for path in sorted(txt_dir.glob("*.txt")):
        documents[path.stem] = maybe_use_mineru_source(
            domain="regulatory",
            doc_id=path.stem,
            default_path=path,
            default_type="txt",
            mineru_docs=mineru_docs,
        )
    for path in sorted(html_dir.glob("*.html")):
        documents.setdefault(
            path.stem,
            maybe_use_mineru_source(
                domain="regulatory",
                doc_id=path.stem,
                default_path=path,
                default_type="html",
                mineru_docs=mineru_docs,
            ),
        )
    for path in sorted(attachments_dir.glob("*.pdf")):
        documents[path.stem] = maybe_use_mineru_source(
            domain="regulatory",
            doc_id=path.stem,
            default_path=path,
            default_type="pdf",
            mineru_docs=mineru_docs,
        )
    return {"documents": documents}


def build_generic_manifest(raw_dir: Path, domain: str, mineru_docs: dict[str, dict]) -> dict:
    domain_dir = raw_dir / domain
    documents = {}
    for path in sorted(domain_dir.glob("*")):
        if path.is_file():
            default_type = path.suffix.lstrip(".").lower()
            documents[path.stem] = maybe_use_mineru_source(
                domain=domain,
                doc_id=path.stem,
                default_path=path,
                default_type=default_type,
                mineru_docs=mineru_docs,
            )
    return {"documents": documents}


def main() -> None:
    raw_dir = ROOT / "public_dataset_upload" / "raw"
    questions_dir = ROOT / "public_dataset_upload" / "questions" / "group_a"
    mineru_docs = load_mineru_manifest()
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
            **build_regulatory_manifest(raw_dir, mineru_docs),
        }
    for domain in list(domains.keys()):
        if domain == "regulatory":
            continue
        domains[domain] = {**domains[domain], **build_generic_manifest(raw_dir, domain, mineru_docs)}

    payload = {"project_root": str(ROOT), "domains": domains}
    out_path = ROOT / "artifacts" / "manifest" / "dataset_manifest.json"
    write_json(out_path, payload)
    print(out_path)


if __name__ == "__main__":
    main()
