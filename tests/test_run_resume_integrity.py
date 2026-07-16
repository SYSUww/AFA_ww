from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from afa_agent.io_utils import read_json, write_json, write_jsonl
from afa_agent.models import Question
from afa_agent.run_metadata import (
    RunFingerprintError,
    build_run_fingerprint,
    build_run_manifest,
    validate_resume_fingerprint,
)
from scripts import run_answering


class AtomicJsonWriteTests(unittest.TestCase):
    def test_json_and_jsonl_replace_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "state.json"
            jsonl_path = root / "state.jsonl"
            json_path.write_text('{"old": true}', encoding="utf-8")
            jsonl_path.write_text('{"old": true}\n', encoding="utf-8")

            write_json(json_path, {"new": "值"})
            write_jsonl(jsonl_path, [{"row": 1}, {"row": 2}])

            self.assertEqual(read_json(json_path), {"new": "值"})
            self.assertEqual(
                [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()],
                [{"row": 1}, {"row": 2}],
            )

    def test_jsonl_failure_keeps_old_file_and_cleans_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "checkpoint.jsonl"
            original = '{"old": true}\n'
            path.write_text(original, encoding="utf-8")

            with self.assertRaises(TypeError):
                write_jsonl(path, [{"row": 1}, {"not_json": {1, 2}}])

            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(root.glob(".checkpoint.jsonl.*.tmp")), [])


class RunFingerprintTests(unittest.TestCase):
    def _build_fingerprint(self, root: Path) -> dict[str, object]:
        parsed_path = root / "parsed.json"
        index_path = root / "index.json"
        parsed_path.write_text('{"documents": []}', encoding="utf-8")
        index_path.write_text('{"units": []}', encoding="utf-8")
        question = Question(
            qid="q1",
            domain="regulatory",
            split="A",
            question="测试问题",
            options={"A": "是", "B": "否"},
            answer_format="single",
            type="判断题",
            doc_ids=["doc1"],
        )
        return build_run_fingerprint(
            project_root=root,
            arguments={
                "domain": "regulatory",
                "split": "A",
                "limit": 1,
                "resume_run_dir": str(root / "run"),
                "qid": "q1",
                "qid_file": None,
                "strategy_config": None,
                "parsed_path": str(parsed_path),
                "index_path": str(index_path),
                "run_root_dir": str(root),
                "run_id": "run",
            },
            questions=[question],
            parsed_path=parsed_path,
            index_path=index_path,
            strategy_payload={"strategy_id": "baseline"},
            strategy_path=None,
            model_settings={"model_name": "test-model", "temperature": 0.0},
            git_state={
                "branch": "codex/test",
                "commit": "a" * 40,
                "dirty": False,
                "dirty_diff_sha256": "b" * 64,
                "untracked_paths": [],
            },
        )

    def test_manifest_fingerprint_is_self_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = self._build_fingerprint(root)
            manifest = build_run_manifest(
                run_id="run",
                run_dir=root / "run",
                domain="regulatory",
                split="A",
                question_count=1,
                qid="q1",
                limit=1,
                plugin_name="FakePlugin",
                strategy_label="test",
                strategy_details=[],
                model_name="test-model",
                model_temperature=0.0,
                resumed=False,
                fingerprint=fingerprint,
                invocation_args={"domain": "regulatory"},
            )

            validate_resume_fingerprint(manifest, fingerprint)
            components = fingerprint["components"]
            self.assertIn("dirty_diff_sha256", components["git"])
            self.assertEqual(components["git"]["branch"], "codex/test")
            self.assertIn("content_sha256", components["questions"])
            self.assertIn("qids_sha256", components["questions"])
            self.assertIn("parsed", components["inputs"])
            self.assertIn("index", components["inputs"])
            self.assertIn("content_sha256", components["strategy"])
            self.assertEqual(components["model"]["model_name"], "test-model")
            self.assertEqual(components["model"]["temperature"], 0.0)

    def test_modified_or_mismatched_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fingerprint = self._build_fingerprint(Path(directory))
            manifest = {"fingerprint": fingerprint}
            changed = copy.deepcopy(fingerprint)
            changed["components"]["model"]["temperature"] = 0.5

            with self.assertRaisesRegex(RunFingerprintError, "current.*modified"):
                validate_resume_fingerprint(manifest, changed)


class AnsweringResumeTests(unittest.TestCase):
    class FakePlugin:
        strategy_label = "test-no-model"
        strategy_details = ["unit test"]

        def answer_one(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("answer_one must not be called for an empty question selection")

    def _invoke(self, arguments: list[str]) -> None:
        with mock.patch.object(sys, "argv", ["run_answering.py", *arguments]), mock.patch.object(
            run_answering, "get_plugin", return_value=self.FakePlugin()
        ), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AFA_STRATEGY_CONFIG", None)
            run_answering.main()

    def test_new_run_matching_resume_and_mismatch_before_manifest_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed_path = root / "parsed.json"
            index_path = root / "index.json"
            parsed_path.write_text('{"documents": [], "units": []}', encoding="utf-8")
            index_path.write_text('{"documents": [], "units": []}', encoding="utf-8")
            run_dir = root / "runs" / "resume_test"
            shared_args = [
                "--domain",
                "regulatory",
                "--split",
                "A",
                "--qid",
                "__missing_qid__",
                "--parsed-path",
                str(parsed_path),
                "--index-path",
                str(index_path),
            ]

            self._invoke(
                [
                    *shared_args,
                    "--run-root-dir",
                    str(run_dir.parent),
                    "--run-id",
                    run_dir.name,
                ]
            )
            manifest_path = run_dir / "meta" / "run_manifest.json"
            first_manifest = read_json(manifest_path)
            self.assertFalse(first_manifest["resumed"])
            self.assertIn("fingerprint", first_manifest)
            fingerprint_model = first_manifest["fingerprint"]["components"]["model"]
            self.assertNotIn("api_key", fingerprint_model)
            self.assertNotIn("api_base", fingerprint_model)

            self._invoke([*shared_args, "--resume-run-dir", str(run_dir)])
            resumed_manifest = read_json(manifest_path)
            self.assertTrue(resumed_manifest["resumed"])
            self.assertEqual(resumed_manifest["resume_count"], 1)
            self.assertEqual(resumed_manifest["fingerprint"], first_manifest["fingerprint"])

            manifest_before_mismatch = manifest_path.read_bytes()
            index_path.write_text('{"documents": [], "units": [{"changed": true}]}', encoding="utf-8")
            with self.assertRaisesRegex(RunFingerprintError, "inputs"):
                self._invoke([*shared_args, "--resume-run-dir", str(run_dir)])
            self.assertEqual(manifest_path.read_bytes(), manifest_before_mismatch)


if __name__ == "__main__":
    unittest.main()
