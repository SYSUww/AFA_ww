#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.io_utils import ensure_dir, read_json, write_json


RAW_ROOT = ROOT / "public_dataset_upload" / "raw"
OUTPUT_ROOT = ROOT / "artifacts" / "mineru"
DEFAULT_MINERU_BIN = Path("/opt/miniconda3/envs/afa-autoresearch/bin/mineru")


def discover_pdf_documents() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path in sorted(RAW_ROOT.rglob("*.pdf")):
        relative = path.relative_to(RAW_ROOT)
        domain = relative.parts[0]
        doc_id = path.stem
        items.append(
            {
                "domain": domain,
                "doc_id": doc_id,
                "source_path": str(path),
                "relative_path": str(relative),
            }
        )
    return items


def resolve_mineru_bin(explicit: str) -> str:
    if explicit:
        return explicit
    if DEFAULT_MINERU_BIN.exists():
        return str(DEFAULT_MINERU_BIN)
    candidate = shutil.which("mineru")
    if candidate:
        return candidate
    raise FileNotFoundError("Could not locate MinerU binary. Use --mineru-bin to specify it.")


def build_doc_dirs(domain: str, doc_id: str) -> dict[str, Path]:
    base_dir = OUTPUT_ROOT / "docs" / domain / doc_id
    return {
        "base": ensure_dir(base_dir),
        "raw": ensure_dir(base_dir / "raw_output"),
        "normalized": ensure_dir(base_dir / "normalized"),
        "logs": ensure_dir(base_dir / "logs"),
    }


def collect_output_files(raw_dir: Path) -> dict[str, list[str]]:
    files = [path for path in raw_dir.rglob("*") if path.is_file()]
    markdown_files = sorted([str(path) for path in files if path.suffix.lower() in {".md", ".markdown"}])
    text_files = sorted([str(path) for path in files if path.suffix.lower() == ".txt"])
    json_files = sorted([str(path) for path in files if path.suffix.lower() == ".json"])
    image_files = sorted([str(path) for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    return {
        "markdown_files": markdown_files,
        "text_files": text_files,
        "json_files": json_files,
        "image_files": image_files,
    }


def choose_primary_text_file(files: dict[str, list[str]]) -> Path | None:
    for key in ["markdown_files", "text_files"]:
        if files[key]:
            return Path(files[key][0])
    return None


def materialize_normalized_outputs(primary_file: Path, normalized_dir: Path) -> dict[str, str]:
    content_md = normalized_dir / "content.md"
    content_txt = normalized_dir / "content.txt"
    text = primary_file.read_text(encoding="utf-8", errors="ignore")
    content_md.write_text(text, encoding="utf-8")
    content_txt.write_text(text, encoding="utf-8")
    return {
        "content_md": str(content_md),
        "content_txt": str(content_txt),
        "primary_source_file": str(primary_file),
    }


def load_previous_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload.get("documents", {})


def run_one_document(
    *,
    item: dict[str, str],
    mineru_bin: str,
    backend: str,
    method: str,
    lang: str,
    force: bool,
    timeout_seconds: int,
    model_source: str,
    enable_formula: bool,
    enable_table: bool,
    enable_image_analysis: bool,
) -> dict:
    doc_dirs = build_doc_dirs(item["domain"], item["doc_id"])
    metadata_path = doc_dirs["base"] / "meta.json"

    if metadata_path.exists() and not force:
        existing = read_json(metadata_path)
        if existing.get("status") == "success":
            return existing

    if doc_dirs["raw"].exists():
        shutil.rmtree(doc_dirs["raw"])
    ensure_dir(doc_dirs["raw"])

    command = [
        mineru_bin,
        "-p",
        item["source_path"],
        "-o",
        str(doc_dirs["raw"]),
        "--backend",
        backend,
        "--method",
        method,
        "--lang",
        lang,
        "--formula",
        str(enable_formula).lower(),
        "--table",
        str(enable_table).lower(),
        "--image-analysis",
        str(enable_image_analysis).lower(),
    ]
    started_at = time.time()
    timeout_message = ""
    env = os.environ.copy()
    if model_source:
        env["MINERU_MODEL_SOURCE"] = model_source
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, env=env)
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(command, returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "")
        timeout_message = f"Timed out after {timeout_seconds} seconds"
    elapsed = round(time.time() - started_at, 3)

    stdout_path = doc_dirs["logs"] / "stdout.log"
    stderr_path = doc_dirs["logs"] / "stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="ignore")

    files = collect_output_files(doc_dirs["raw"])
    primary_file = choose_primary_text_file(files)
    status = "success" if proc.returncode == 0 and primary_file is not None else "failed"
    normalized = {}
    if primary_file is not None:
        normalized = materialize_normalized_outputs(primary_file, doc_dirs["normalized"])

    metadata = {
        "domain": item["domain"],
        "doc_id": item["doc_id"],
        "source_path": item["source_path"],
        "relative_path": item["relative_path"],
        "status": status,
        "command": command,
        "return_code": proc.returncode,
        "backend": backend,
        "method": method,
        "lang": lang,
        "model_source": model_source,
        "enable_formula": enable_formula,
        "enable_table": enable_table,
        "enable_image_analysis": enable_image_analysis,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": elapsed,
        "raw_output_dir": str(doc_dirs["raw"]),
        "normalized_dir": str(doc_dirs["normalized"]),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "collected_files": files,
        "error_message": timeout_message,
        **normalized,
    }
    write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="")
    parser.add_argument("--doc-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--backend", default="pipeline")
    parser.add_argument("--method", default="auto")
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--model-source", default="modelscope")
    parser.add_argument("--enable-formula", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-image-analysis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--mineru-bin", default="")
    args = parser.parse_args()

    items = discover_pdf_documents()
    if args.domain:
        items = [item for item in items if item["domain"] == args.domain]
    if args.doc_id:
        items = [item for item in items if item["doc_id"] == args.doc_id]
    if args.limit > 0:
        items = items[: args.limit]

    ensure_dir(OUTPUT_ROOT)
    mineru_bin = resolve_mineru_bin(args.mineru_bin)
    manifest_path = OUTPUT_ROOT / "manifest.json"
    existing_documents = load_previous_manifest(manifest_path)
    updated_documents = dict(existing_documents)

    results = []
    for item in items:
        metadata = run_one_document(
            item=item,
            mineru_bin=mineru_bin,
            backend=args.backend,
            method=args.method,
            lang=args.lang,
            model_source=args.model_source,
            enable_formula=args.enable_formula,
            enable_table=args.enable_table,
            enable_image_analysis=args.enable_image_analysis,
            force=args.force,
            timeout_seconds=args.timeout_seconds,
        )
        updated_documents[f"{item['domain']}::{item['doc_id']}"] = metadata
        results.append(metadata)
        print(f"{item['domain']}::{item['doc_id']} -> {metadata['status']}")

    success_count = sum(1 for item in updated_documents.values() if item.get("status") == "success")
    failed_count = sum(1 for item in updated_documents.values() if item.get("status") == "failed")
    payload = {
        "mineru_bin": mineru_bin,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "document_count": len(updated_documents),
        "success_count": success_count,
        "failed_count": failed_count,
        "documents": updated_documents,
    }
    write_json(manifest_path, payload)
    print(manifest_path)


if __name__ == "__main__":
    main()
