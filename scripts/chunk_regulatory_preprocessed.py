#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.regulatory_preprocess.preprocessor import (  # noqa: E402
    RegulatoryDocument,
    clean_regulatory_text,
    classify_doc_type,
    extract_document_metadata,
    split_units,
)


@dataclass(frozen=True)
class ChunkProfile:
    profile_id: str
    max_chars: int
    soft_min_chars: int
    target_chars: int


PROFILES = [
    ChunkProfile("compact", max_chars=900, soft_min_chars=180, target_chars=720),
    ChunkProfile("balanced", max_chars=1200, soft_min_chars=220, target_chars=980),
    ChunkProfile("long", max_chars=1500, soft_min_chars=260, target_chars=1250),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build retrieval chunks from regulatory preprocessing outputs and validate coverage."
    )
    parser.add_argument("--preprocessed-root", default="artifacts/preprocessed/regulatory")
    parser.add_argument("--output-root", default="artifacts/preprocessed/regulatory/chunks")
    parser.add_argument("--manifest-path", default="artifacts/manifest/dataset_manifest.json")
    parser.add_argument("--skip-manifest-fill", action="store_true")
    parser.add_argument("--profile", choices=[profile.profile_id for profile in PROFILES] + ["all"], default="all")
    parser.add_argument("--min-unit-coverage", type=float, default=1.0)
    parser.add_argument("--min-doc-coverage", type=float, default=1.0)
    args = parser.parse_args()

    preprocessed_root = (ROOT / args.preprocessed_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    documents = _read_json(preprocessed_root / "documents.json")
    units = _read_json(preprocessed_root / "units.json")
    manifest_fill = {"added_documents": [], "missing_documents": []}
    if not args.skip_manifest_fill:
        manifest_fill = fill_manifest_referenced_docs(
            documents,
            units,
            (ROOT / args.manifest_path).resolve(),
        )

    selected_profiles = PROFILES if args.profile == "all" else [p for p in PROFILES if p.profile_id == args.profile]
    output_root.mkdir(parents=True, exist_ok=True)

    reports = []
    for profile in selected_profiles:
        profile_dir = output_root / profile.profile_id
        report = build_chunks_for_profile(documents, units, profile, profile_dir)
        reports.append(report)

    leaderboard = sorted(reports, key=_score_report, reverse=True)
    best = leaderboard[0] if leaderboard else {}
    best_profile = best.get("profile_id")
    if best_profile:
        best_dir = output_root / "best"
        if best_dir.exists() or best_dir.is_symlink():
            if best_dir.is_symlink() or best_dir.is_file():
                best_dir.unlink()
            else:
                shutil.rmtree(best_dir)
        shutil.copytree(output_root / best_profile, best_dir)

    summary = {
        "preprocessed_root": str(preprocessed_root),
        "output_root": str(output_root),
        "manifest_fill": manifest_fill,
        "profiles": leaderboard,
        "best_profile": best_profile,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failures = []
    for report in reports:
        if report["unit_coverage"] < args.min_unit_coverage:
            failures.append(f"{report['profile_id']} unit coverage {report['unit_coverage']:.6f}")
        if report["document_coverage"] < args.min_doc_coverage:
            failures.append(f"{report['profile_id']} document coverage {report['document_coverage']:.6f}")
        if report["bad_case_counts"].get("text_mismatch", 0):
            failures.append(f"{report['profile_id']} has text mismatches")
        if report["bad_case_counts"].get("overlong_chunk", 0):
            failures.append(f"{report['profile_id']} has overlong chunks")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        sys.exit(1)


def build_chunks_for_profile(
    documents: list[dict[str, Any]],
    units: list[dict[str, Any]],
    profile: ChunkProfile,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_by_id = {doc["doc_id"]: doc for doc in documents}
    chunks: list[dict[str, Any]] = []
    bad_cases: list[dict[str, Any]] = []
    source_unit_ids = set()
    covered_unit_ids = set()
    chunk_counts_by_doc: Counter[str] = Counter()

    for unit in units:
        unit_id = unit.get("unit_id", "")
        doc_id = unit.get("doc_id", "")
        text = unit.get("text", "")
        if not unit_id or not doc_id:
            bad_cases.append({"reason": "missing_unit_identity", "unit_id": unit_id, "doc_id": doc_id})
            continue
        source_unit_ids.add(unit_id)
        if not text.strip():
            bad_cases.append({"reason": "empty_unit_text", "unit_id": unit_id, "doc_id": doc_id})
            continue

        spans = split_text_spans(text, profile)
        rebuilt = "".join(text[start:end] for start, end in spans)
        if rebuilt != text:
            bad_cases.append(
                {
                    "reason": "text_mismatch",
                    "unit_id": unit_id,
                    "doc_id": doc_id,
                    "source_chars": len(text),
                    "rebuilt_chars": len(rebuilt),
                    "source_sha1": _sha1(text),
                    "rebuilt_sha1": _sha1(rebuilt),
                }
            )
            continue

        covered_unit_ids.add(unit_id)
        doc = docs_by_id.get(doc_id, {})
        for index, (start, end) in enumerate(spans, start=1):
            chunk_text = text[start:end]
            chunk_id = unit_id if len(spans) == 1 else f"{unit_id}::chunk_{index}"
            unit_type = unit.get("unit_type", "")
            if len(spans) > 1 and not unit_type.endswith("_chunk"):
                unit_type = f"{unit_type}_chunk"
            if len(chunk_text) > profile.max_chars:
                bad_cases.append(
                    {
                        "reason": "overlong_chunk",
                        "chunk_id": chunk_id,
                        "unit_id": unit_id,
                        "doc_id": doc_id,
                        "chars": len(chunk_text),
                        "max_chars": profile.max_chars,
                    }
                )
            if len(chunk_text.strip()) < profile.soft_min_chars and len(text.strip()) >= profile.soft_min_chars:
                bad_cases.append(
                    {
                        "reason": "short_fragment",
                        "chunk_id": chunk_id,
                        "unit_id": unit_id,
                        "doc_id": doc_id,
                        "chars": len(chunk_text.strip()),
                    }
                )
            chunks.append(
                {
                    "unit_id": chunk_id,
                    "doc_id": doc_id,
                    "domain": "regulatory",
                    "unit_type": unit_type,
                    "title_path": unit.get("title_path", []),
                    "text": chunk_text,
                    "page_refs": [],
                    "parent_unit_id": None if len(spans) == 1 else unit_id,
                    "metadata": {
                        "source_unit_id": unit_id,
                        "chunk_index": index,
                        "chunk_count": len(spans),
                        "char_start": start,
                        "char_end": end,
                        "source_path": unit.get("source_path") or doc.get("source_path", ""),
                        "source_doc_type": doc.get("doc_type", ""),
                        "source_bucket": doc.get("source_bucket", ""),
                    },
                }
            )
            chunk_counts_by_doc[doc_id] += 1

    document_ids = {doc["doc_id"] for doc in documents}
    unit_doc_ids = {unit.get("doc_id", "") for unit in units if unit.get("doc_id")}
    covered_doc_ids = set(chunk_counts_by_doc)
    for doc_id in sorted(document_ids - unit_doc_ids):
        bad_cases.append({"reason": "document_without_preprocessed_units", "doc_id": doc_id})
    for doc_id in sorted(unit_doc_ids - covered_doc_ids):
        bad_cases.append({"reason": "document_without_chunks", "doc_id": doc_id})

    bad_case_counts = Counter(item["reason"] for item in bad_cases)
    lengths = [len(chunk["text"]) for chunk in chunks]
    chunk_docs_payload = [_to_document_payload(doc) for doc in documents]
    report = {
        "profile_id": profile.profile_id,
        "max_chars": profile.max_chars,
        "soft_min_chars": profile.soft_min_chars,
        "target_chars": profile.target_chars,
        "source_document_count": len(documents),
        "source_unit_count": len(units),
        "chunk_count": len(chunks),
        "covered_unit_count": len(covered_unit_ids),
        "unit_coverage": len(covered_unit_ids) / max(1, len(source_unit_ids)),
        "covered_document_count": len(covered_doc_ids),
        "document_coverage": len(covered_doc_ids) / max(1, len(unit_doc_ids)),
        "source_document_coverage": len(covered_doc_ids) / max(1, len(document_ids)),
        "unit_doc_count": len(unit_doc_ids),
        "bad_case_counts": dict(sorted(bad_case_counts.items())),
        "lengths": _length_summary(lengths),
        "bad_cases_sample": bad_cases[:200],
    }

    _write_json(output_dir / "documents.json", chunk_docs_payload)
    _write_json(output_dir / "chunks.json", chunks)
    _write_json(output_dir / "parsed.json", {"documents": chunk_docs_payload, "units": chunks})
    _write_json(
        output_dir / "index.json",
        {
            "domain": "regulatory",
            "unit_count": len(chunks),
            "doc_count": len(chunk_docs_payload),
            "units": chunks,
        },
    )
    _write_json(output_dir / "coverage_report.json", report)
    return report


def fill_manifest_referenced_docs(
    documents: list[dict[str, Any]],
    units: list[dict[str, Any]],
    manifest_path: Path,
) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"added_documents": [], "missing_documents": [f"manifest not found: {manifest_path}"]}
    manifest = _read_json(manifest_path)
    domain_manifest = manifest.get("domains", {}).get("regulatory", {})
    referenced = set(domain_manifest.get("referenced_doc_ids", []))
    records = domain_manifest.get("documents", {})
    existing_doc_ids = {doc.get("doc_id") for doc in documents}
    missing_doc_ids = sorted(doc_id for doc_id in referenced if doc_id not in existing_doc_ids)
    added_documents = []
    still_missing = []

    for doc_id in missing_doc_ids:
        record = records.get(doc_id)
        if not record:
            still_missing.append({"doc_id": doc_id, "reason": "missing_manifest_record"})
            continue
        source_path = Path(record.get("source_path", ""))
        if not source_path.exists():
            still_missing.append({"doc_id": doc_id, "reason": "missing_source_path", "source_path": str(source_path)})
            continue
        try:
            document, doc_units = preprocess_manifest_document(doc_id, source_path, record.get("source_type", ""))
        except Exception as exc:  # pragma: no cover - reported in artifact for manual inspection.
            still_missing.append({"doc_id": doc_id, "reason": "preprocess_failed", "error": str(exc)})
            continue
        documents.append(document)
        units.extend(doc_units)
        added_documents.append({"doc_id": doc_id, "source_path": str(source_path), "unit_count": len(doc_units)})

    return {"added_documents": added_documents, "missing_documents": still_missing}


def preprocess_manifest_document(
    doc_id: str,
    source_path: Path,
    source_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_bucket = source_type or source_path.suffix.lstrip(".") or "unknown"
    raw_text, metadata_hint = extract_manifest_text(source_path, source_type)
    cleaned, _stats = clean_regulatory_text(raw_text, source_bucket)
    metadata = extract_document_metadata(cleaned, source_bucket, doc_id)
    for key, value in metadata_hint.items():
        if value:
            metadata[key] = value
    doc_type = classify_doc_type(cleaned, source_bucket, metadata["title"], doc_id)
    document = RegulatoryDocument(
        doc_id=doc_id,
        source_path=str(source_path),
        source_bucket=source_bucket,
        doc_type=doc_type,
        title=metadata["title"],
        doc_no=metadata["doc_no"],
        agency=metadata["agency"],
        publish_date=metadata["publish_date"],
        effective_date=metadata["effective_date"],
    )
    doc_units = split_units(document, cleaned)
    return document.to_dict(), [unit.to_dict() for unit in doc_units]


def extract_manifest_text(source_path: Path, source_type: str) -> tuple[str, dict[str, str]]:
    if source_type == "html" or source_path.suffix.lower() in {".html", ".htm"}:
        html = source_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "form", "header", "footer"]):
            tag.decompose()
        title = _meta_content(soup, "ArticleTitle")
        pub_date = _meta_content(soup, "PubDate")[:10]
        content = (
            soup.select_one(".detail-news")
            or soup.select_one(".TRS_Editor")
            or soup.select_one("#zoom")
            or soup.select_one(".content")
            or soup.body
            or soup
        )
        return content.get_text("\n", strip=True), {"title": title, "publish_date": pub_date}
    return source_path.read_text(encoding="utf-8", errors="ignore"), {}


def _meta_content(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name})
    if not tag:
        return ""
    value = tag.get("content")
    return value.strip() if isinstance(value, str) else ""


def split_text_spans(text: str, profile: ChunkProfile) -> list[tuple[int, int]]:
    if len(text) <= profile.max_chars:
        return [(0, len(text))]

    boundaries = collect_split_boundaries(text)
    spans: list[tuple[int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        hard_end = min(text_len, start + profile.max_chars)
        if hard_end == text_len:
            spans.append((start, text_len))
            break

        target_end = min(text_len, start + profile.target_chars)
        candidate = choose_boundary(boundaries, start, target_end, hard_end)
        if candidate <= start:
            candidate = hard_end
        spans.append((start, candidate))
        start = candidate

    return merge_short_tail(spans, profile.max_chars, profile.soft_min_chars)


def collect_split_boundaries(text: str) -> list[int]:
    boundaries = {match.end() for match in re.finditer(r"\n{1,2}|[。！？；;]\s*|[，,]\s*", text)}
    boundaries.add(len(text))
    return sorted(boundaries)


def choose_boundary(boundaries: list[int], start: int, target_end: int, hard_end: int) -> int:
    usable = [pos for pos in boundaries if start < pos <= hard_end]
    if not usable:
        return hard_end
    before_target = [pos for pos in usable if pos <= target_end]
    if before_target:
        return before_target[-1]
    return usable[0]


def merge_short_tail(spans: list[tuple[int, int]], max_chars: int, soft_min_chars: int) -> list[tuple[int, int]]:
    if len(spans) < 2:
        return spans
    prev_start, prev_end = spans[-2]
    tail_start, tail_end = spans[-1]
    if tail_end - tail_start < soft_min_chars and tail_end - prev_start <= max_chars:
        return [*spans[:-2], (prev_start, tail_end)]
    return spans


def _to_document_payload(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": doc.get("doc_id", ""),
        "domain": "regulatory",
        "title": doc.get("title", ""),
        "source_type": doc.get("source_bucket", ""),
        "source_path": doc.get("source_path", ""),
        "metadata": {
            "doc_type": doc.get("doc_type", ""),
            "doc_no": doc.get("doc_no", ""),
            "agency": doc.get("agency", ""),
            "publish_date": doc.get("publish_date", ""),
            "effective_date": doc.get("effective_date", ""),
        },
    }


def _score_report(report: dict[str, Any]) -> tuple[float, float, float, float]:
    bad_counts = report.get("bad_case_counts", {})
    hard_bad = bad_counts.get("text_mismatch", 0) + bad_counts.get("overlong_chunk", 0)
    soft_bad = bad_counts.get("short_fragment", 0)
    return (
        report.get("unit_coverage", 0.0) + report.get("document_coverage", 0.0),
        -float(hard_bad),
        -float(soft_bad),
        -abs(report.get("lengths", {}).get("median", 0) - report.get("target_chars", 0)),
    )


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "max": 0, "avg": 0}
    ordered = sorted(lengths)
    return {
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "median": _percentile(ordered, 0.5),
        "p75": _percentile(ordered, 0.75),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "avg": sum(ordered) / len(ordered),
    }


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * ratio)))
    return values[index]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
