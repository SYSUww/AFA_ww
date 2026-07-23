from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from afa_agent.config import build_model_config, build_run_config


class RunConfigEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_env(self, text: str) -> None:
        (self.root / ".env").write_text(text, encoding="utf-8")

    def build(self):
        with patch.dict(os.environ, {}, clear=True):
            return build_run_config(self.root)

    def test_complete_llm_configuration_has_priority(self) -> None:
        self.write_env(
            "\n".join(
                (
                    "LLM_API_KEY=llm-key",
                    "LLM_API_BASE=https://llm.example/v1",
                    "LLM_MODEL=qwen3.6-plus",
                    "OPENAI_API_KEY=openai-key",
                    "OPENAI_BASE_URL=https://openai.example/v1",
                    "OPENAI_MODEL=qwen3.7-plus",
                )
            )
        )

        config = self.build()

        assert config.model is not None
        self.assertEqual(config.model.api_key, "llm-key")
        self.assertEqual(config.model.api_base, "https://llm.example/v1")
        self.assertEqual(config.model.model_name, "qwen3.6-plus")

    def test_complete_openai_configuration_is_the_fallback(self) -> None:
        self.write_env(
            "\n".join(
                (
                    "OPENAI_API_KEY=openai-key",
                    "OPENAI_BASE_URL=https://openai.example/v1",
                    "OPENAI_MODEL=qwen3.7-plus-2026-05-26",
                    "OPENAI_TIMEOUT_SECONDS=45",
                    "OPENAI_MAX_RETRIES=4",
                )
            )
        )

        config = self.build()

        assert config.model is not None
        self.assertEqual(config.model.model_name, "qwen3.7-plus-2026-05-26")
        self.assertEqual(config.model.timeout_seconds, 45)
        self.assertEqual(config.model.max_retries, 4)

    def test_openai_connection_accepts_generic_model_name_alias(self) -> None:
        self.write_env(
            "\n".join(
                (
                    "OPENAI_API_KEY=openai-key",
                    "OPENAI_BASE_URL=https://openai.example/v1",
                    "MODEL_NAME=qwen3.7-plus-2026-05-26",
                )
            )
        )

        config = self.build()

        assert config.model is not None
        self.assertEqual(config.model.model_name, "qwen3.7-plus-2026-05-26")

    def test_structured_output_mode_requires_explicit_native_opt_in(self) -> None:
        self.write_env(
            "\n".join(
                (
                    "OPENAI_API_KEY=openai-key",
                    "OPENAI_BASE_URL=https://openai.example/v1",
                    "OPENAI_MODEL=qwen3.7-plus-2026-05-26",
                    "OPENAI_STRUCTURED_OUTPUT_MODE=native_json_schema_strict",
                )
            )
        )

        config = self.build()

        assert config.model is not None
        self.assertEqual(
            config.model.structured_output_mode,
            "native_json_schema_strict",
        )

    def test_unknown_structured_output_mode_is_rejected(self) -> None:
        self.write_env(
            "\n".join(
                (
                    "OPENAI_API_KEY=openai-key",
                    "OPENAI_BASE_URL=https://openai.example/v1",
                    "OPENAI_MODEL=qwen3.7-plus-2026-05-26",
                    "OPENAI_STRUCTURED_OUTPUT_MODE=auto",
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "must be one of"):
            self.build()

    def test_explicit_openai_selection_overrides_complete_llm_configuration(self) -> None:
        self.write_env(
            "\n".join(
                (
                    "AFA_MODEL_ENV_PREFIX=OPENAI",
                    "LLM_API_KEY=llm-key",
                    "LLM_API_BASE=https://llm.example/v1",
                    "LLM_MODEL=gpt-5.5",
                    "OPENAI_API_KEY=openai-key",
                    "OPENAI_BASE_URL=https://openai.example/v1",
                    "OPENAI_MODEL=qwen3.7-plus-2026-05-26",
                )
            )
        )

        config = self.build()

        assert config.model is not None
        self.assertEqual(config.model.api_key, "openai-key")
        self.assertEqual(config.model.model_name, "qwen3.7-plus-2026-05-26")

    def test_partial_llm_configuration_does_not_mix_with_openai(self) -> None:
        self.write_env(
            "\n".join(
                (
                    "LLM_API_KEY=llm-key",
                    "OPENAI_API_KEY=openai-key",
                    "OPENAI_BASE_URL=https://openai.example/v1",
                    "OPENAI_MODEL=qwen3.7-plus",
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "incomplete LLM"):
            self.build()

    def test_partial_openai_configuration_is_rejected(self) -> None:
        self.write_env(
            "\n".join(
                (
                    "OPENAI_API_KEY=openai-key",
                    "OPENAI_BASE_URL=https://openai.example/v1",
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "incomplete OPENAI"):
            self.build()

    def test_missing_model_configuration_remains_optional(self) -> None:
        self.write_env("UNRELATED=value\n")

        config = self.build()

        self.assertIsNone(config.model)

    def test_unknown_model_environment_selector_is_rejected(self) -> None:
        self.write_env("AFA_MODEL_ENV_PREFIX=OTHER\n")

        with self.assertRaisesRegex(ValueError, "must be one of"):
            self.build()

    def test_evaluator_prefix_is_independent_of_production_selector(self) -> None:
        self.write_env(
            "\n".join(
                (
                    "AFA_MODEL_ENV_PREFIX=OPENAI",
                    "OPENAI_API_KEY=qwen-key",
                    "OPENAI_BASE_URL=https://qwen.example/v1",
                    "OPENAI_MODEL=qwen3.7-plus",
                    "LLM_API_KEY=judge-key",
                    "LLM_API_BASE=https://judge.example/v1",
                    "LLM_MODEL=gpt-5.5",
                )
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            production = build_run_config(self.root)
            judge_base = build_model_config(self.root, env_prefix="LLM")

        assert production.model is not None
        assert judge_base is not None
        self.assertEqual(production.model.model_name, "qwen3.7-plus")
        self.assertEqual(production.model.api_base, "https://qwen.example/v1")
        self.assertEqual(judge_base.model_name, "gpt-5.5")
        self.assertEqual(judge_base.api_base, "https://judge.example/v1")


if __name__ == "__main__":
    unittest.main()
