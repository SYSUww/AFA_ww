#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "manifest" / "dataset_manifest.json"
DEFAULT_OUTPUT = ROOT / "configs" / "autoresearch" / "dataset_slices_v2.json"
DEFAULT_METADATA_OUTPUT = ROOT / "configs" / "autoresearch" / "dataset_slices_v2_manifest.json"
BUCKET_CAPACITIES = {"dev_v2": 10, "gate_v2": 5, "holdout_v2": 5}
BUCKET_RATIOS = {"dev_v2": 0.50, "gate_v2": 0.25, "holdout_v2": 0.25}
SPLIT_SEED = "afa-no-embedding-accuracy-token-v2"
KNOWN_DEV_QIDS = {
    "financial_reports": {
        "fin_a_003",
        "fin_a_008",
        "fin_a_011",
        "fin_a_013",
        "fin_a_015",
        "fin_a_018",
    },
    "insurance": {
        "ins_a_001",
        "ins_a_003",
        "ins_a_006",
        "ins_a_009",
        "ins_a_012",
        "ins_a_014",
        "ins_a_016",
        "ins_a_020",
    },
    "regulatory": {"reg_a_010"},
    "research": {"res_a_006", "res_a_013"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_rank(value: str) -> str:
    return sha256_bytes(f"{SPLIT_SEED}:{value}".encode("utf-8"))


def broad_category(row: dict[str, Any]) -> str:
    text = f"{row.get('type', '')} {row.get('question', '')}".lower()
    category_patterns = [
        ("calculation", ["计算", "比例", "比率", "金额", "percentage", "ratio"]),
        ("comparison", ["比较", "对比", "高于", "低于", "排序", "趋势", "comparison"]),
        ("clause", ["条款", "合同", "规定", "办法", "责任", "期限", "clause"]),
        ("fact", ["事实", "查询", "识别", "核对", "data", "fact"]),
    ]
    for category, keywords in category_patterns:
        if any(keyword in text for keyword in keywords):
            return category
    return "judgment"


def row_features(row: dict[str, Any]) -> tuple[str, ...]:
    doc_count = len(row.get("doc_ids") or [])
    return (
        f"answer_format:{row.get('answer_format', '')}",
        f"question_type:{row.get('type', '')}",
        f"doc_count:{doc_count}",
        f"category:{broad_category(row)}",
    )


def feature_weight(feature: str) -> float:
    if feature.startswith("answer_format:"):
        return 50.0
    if feature.startswith("question_type:"):
        return 3.0
    if feature.startswith("category:"):
        return 2.0
    return 1.0


def split_domain(rows: list[dict[str, Any]], required_dev_qids: set[str] | None = None) -> dict[str, list[str]]:
    total_capacity = sum(BUCKET_CAPACITIES.values())
    if len(rows) != total_capacity:
        raise ValueError(f"Expected {total_capacity} Group A rows per domain, found {len(rows)}")

    required_dev_qids = required_dev_qids or set()
    rows_by_qid = {str(row["qid"]): row for row in rows}
    missing_required = required_dev_qids - rows_by_qid.keys()
    if missing_required:
        raise ValueError(f"Required dev qids are missing: {sorted(missing_required)}")
    if len(required_dev_qids) > BUCKET_CAPACITIES["dev_v2"]:
        raise ValueError("Required dev qids exceed dev_v2 capacity")

    feature_totals = Counter(feature for row in rows for feature in row_features(row))
    ordered = sorted(
        [row for row in rows if str(row["qid"]) not in required_dev_qids],
        key=lambda row: (
            sum(feature_totals[feature] for feature in row_features(row)),
            stable_rank(str(row["qid"])),
        ),
    )
    assigned: dict[str, list[str]] = {
        bucket: sorted(required_dev_qids) if bucket == "dev_v2" else []
        for bucket in BUCKET_CAPACITIES
    }
    feature_counts: dict[str, Counter[str]] = {bucket: Counter() for bucket in BUCKET_CAPACITIES}
    for qid in required_dev_qids:
        feature_counts["dev_v2"].update(row_features(rows_by_qid[qid]))

    for row in ordered:
        features = row_features(row)
        candidates: list[tuple[float, str, str]] = []
        for bucket, capacity in BUCKET_CAPACITIES.items():
            if len(assigned[bucket]) >= capacity:
                continue
            incremental_error = 0.0
            for feature in features:
                target = feature_totals[feature] * BUCKET_RATIOS[bucket]
                before = feature_counts[bucket][feature] - target
                after = feature_counts[bucket][feature] + 1 - target
                incremental_error += feature_weight(feature) * (after * after - before * before)
            capacity_pressure = len(assigned[bucket]) / capacity
            tie_break = stable_rank(f"{row['qid']}:{bucket}")
            candidates.append((incremental_error + capacity_pressure * 0.01, tie_break, bucket))
        if not candidates:
            raise RuntimeError(f"No split capacity remains for {row['qid']}")
        bucket = min(candidates)[2]
        assigned[bucket].append(str(row["qid"]))
        feature_counts[bucket].update(features)

    for bucket, qids in assigned.items():
        if len(qids) != BUCKET_CAPACITIES[bucket]:
            raise AssertionError(f"{bucket} expected {BUCKET_CAPACITIES[bucket]} qids, found {len(qids)}")
        qids.sort()
    if len({qid for qids in assigned.values() for qid in qids}) != total_capacity:
        raise AssertionError("Generated slices overlap")
    return assigned


def choose_smoke(dev_qids: list[str], rows_by_qid: dict[str, dict[str, Any]]) -> list[str]:
    ordered = sorted(dev_qids, key=stable_rank)
    selected: list[str] = []
    seen_formats: set[str] = set()
    for qid in ordered:
        answer_format = str(rows_by_qid[qid].get("answer_format", ""))
        if answer_format in seen_formats:
            continue
        selected.append(qid)
        seen_formats.add(answer_format)
        if len(selected) == 2:
            return sorted(selected)
    return sorted(ordered[:2])


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build_payloads(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(manifest_path)
    domains = sorted(manifest.get("domains", {}))
    slices: dict[str, dict[str, list[str] | str]] = {
        "smoke_v2": {},
        "dev_v2": {},
        "gate_v2": {},
        "holdout_v2": {},
        "full_group_a": {},
    }
    sources: dict[str, Any] = {}
    distribution: dict[str, Any] = defaultdict(dict)

    for domain in domains:
        question_path = Path(manifest["domains"][domain]["question_path"])
        if not question_path.is_absolute():
            question_path = ROOT / question_path
        rows = [row for row in read_json(question_path) if row.get("split") == "A"]
        rows_by_qid = {str(row["qid"]): row for row in rows}
        if len(rows_by_qid) != len(rows):
            raise ValueError(f"Duplicate qid values found in {question_path}")
        assigned = split_domain(rows, KNOWN_DEV_QIDS.get(domain, set()))
        for bucket, qids in assigned.items():
            slices[bucket][domain] = qids
            distribution[domain][bucket] = {
                "answer_format": dict(Counter(rows_by_qid[qid].get("answer_format", "") for qid in qids)),
                "question_type": dict(Counter(rows_by_qid[qid].get("type", "") for qid in qids)),
                "category": dict(Counter(broad_category(rows_by_qid[qid]) for qid in qids)),
                "doc_count": dict(Counter(str(len(rows_by_qid[qid].get("doc_ids") or [])) for qid in qids)),
            }
        slices["smoke_v2"][domain] = choose_smoke(assigned["dev_v2"], rows_by_qid)
        slices["full_group_a"][domain] = "__all__"
        sources[domain] = {
            "path": relative_to_root(question_path),
            "sha256": sha256_file(question_path),
            "question_count": len(rows),
        }

    slice_hashes = {
        split_name: sha256_bytes(canonical_json_bytes(payload))
        for split_name, payload in slices.items()
    }
    metadata = {
        "version": "no_embedding_slices_v2",
        "algorithm": "deterministic_multilabel_greedy_v1",
        "seed": SPLIT_SEED,
        "created_from_manifest": relative_to_root(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sources": sources,
        "known_dev_qids": {domain: sorted(qids) for domain, qids in KNOWN_DEV_QIDS.items()},
        "slice_hashes": slice_hashes,
        "distribution": distribution,
        "rules": {
            "dev_v2": "10 questions per domain; tuning and per-question analysis allowed",
            "gate_v2": "5 questions per domain; aggregate metrics only",
            "holdout_v2": "5 questions per domain; run at most once per research batch",
            "smoke_v2": "2 questions selected from dev_v2; pipeline check only",
        },
    }
    return slices, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA_OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    slices, metadata = build_payloads(Path(args.manifest))
    slices_bytes = canonical_json_bytes(slices)
    metadata["dataset_slices_sha256"] = sha256_bytes(slices_bytes)
    metadata_bytes = canonical_json_bytes(metadata)

    if args.check:
        if not output_path.exists() or output_path.read_bytes() != slices_bytes:
            raise SystemExit(f"slice file is stale: {output_path}")
        if not metadata_path.exists() or metadata_path.read_bytes() != metadata_bytes:
            raise SystemExit(f"slice metadata is stale: {metadata_path}")
        print(output_path)
        print(metadata_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(slices_bytes)
    metadata_path.write_bytes(metadata_bytes)
    print(output_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
