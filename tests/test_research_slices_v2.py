from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.build_research_slices_v2 import (
    BUCKET_CAPACITIES,
    DEFAULT_MANIFEST,
    KNOWN_DEV_QIDS,
    build_payloads,
    canonical_json_bytes,
)
from afa_agent.autoresearch import run_loop_plan


ROOT = Path(__file__).resolve().parents[1]
SLICES_PATH = ROOT / "configs" / "autoresearch" / "dataset_slices_v2.json"
SLICES_MANIFEST_PATH = ROOT / "configs" / "autoresearch" / "dataset_slices_v2_manifest.json"
PLAN_PATH = ROOT / "configs" / "autoresearch" / "plan_config_v2.json"


class ResearchSliceV2Tests(unittest.TestCase):
    def test_committed_slices_are_deterministic_and_current(self) -> None:
        slices, metadata = build_payloads(DEFAULT_MANIFEST)
        slices_bytes = canonical_json_bytes(slices)
        metadata["dataset_slices_sha256"] = hashlib.sha256(slices_bytes).hexdigest()

        self.assertEqual(SLICES_PATH.read_bytes(), slices_bytes)
        self.assertEqual(SLICES_MANIFEST_PATH.read_bytes(), canonical_json_bytes(metadata))

    def test_domain_splits_are_disjoint_and_known_cases_stay_in_dev(self) -> None:
        slices = json.loads(SLICES_PATH.read_text(encoding="utf-8"))
        domains = set(slices["dev_v2"])
        self.assertTrue(set(KNOWN_DEV_QIDS) <= domains)
        for domain in domains:
            required_dev = KNOWN_DEV_QIDS.get(domain, set())
            with self.subTest(domain=domain):
                buckets = {
                    name: set(slices[name][domain])
                    for name in ("dev_v2", "gate_v2", "holdout_v2")
                }
                for name, expected_size in BUCKET_CAPACITIES.items():
                    self.assertEqual(len(buckets[name]), expected_size)
                self.assertFalse(buckets["dev_v2"] & buckets["gate_v2"])
                self.assertFalse(buckets["dev_v2"] & buckets["holdout_v2"])
                self.assertFalse(buckets["gate_v2"] & buckets["holdout_v2"])
                self.assertTrue(required_dev <= buckets["dev_v2"])
                self.assertTrue(set(slices["smoke_v2"][domain]) <= buckets["dev_v2"])

    def test_plan_explicitly_forbids_embedding(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.assertIs(plan["promotion_rules"]["embedding_forbidden"], True)
        self.assertEqual(plan["dataset_slices"], "configs/autoresearch/dataset_slices_v2.json")
        self.assertIs(plan["execution"]["requires_explicit_paid_run"], True)

    def test_manual_plan_cannot_accidentally_start_generic_paid_loop(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be executed by the generic loop engine"):
            run_loop_plan(PLAN_PATH)


if __name__ == "__main__":
    unittest.main()
