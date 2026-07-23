from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from afa_agent.client import OpenAICompatibleClient, capture_llm_usage
from afa_agent.config import ModelConfig


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class ClientUsageLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OpenAICompatibleClient(
            ModelConfig(
                api_key="test",
                api_base="https://example.invalid/v1",
                model_name="qwen3.5-plus",
                max_retries=0,
            )
        )

    def test_captures_each_successful_raw_usage_response(self) -> None:
        responses = [
            FakeHTTPResponse(
                {
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }
            ),
            FakeHTTPResponse(
                {
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
                }
            ),
        ]
        with patch("afa_agent.client.requests.post", side_effect=responses):
            with capture_llm_usage() as ledger:
                self.client.chat_json([{"role": "user", "content": "first"}])
                self.client.chat_json([{"role": "user", "content": "second"}])

        self.assertEqual(ledger.total(), {
            "prompt_tokens": 30,
            "completion_tokens": 5,
            "total_tokens": 35,
        })
        self.assertEqual([call["call_index"] for call in ledger.calls], [1, 2])
        self.assertTrue(all(call["model_name"] == "qwen3.5-plus" for call in ledger.calls))

    def test_rejects_inconsistent_provider_usage(self) -> None:
        response = FakeHTTPResponse(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 99},
            }
        )
        with patch("afa_agent.client.requests.post", return_value=response):
            with self.assertRaisesRegex(ValueError, "must equal"):
                self.client.chat_json([{"role": "user", "content": "bad usage"}])

    def test_native_strict_schema_is_explicit_and_recorded(self) -> None:
        response = FakeHTTPResponse(
            {
                "choices": [{"message": {"content": '{"value":"ok"}'}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
        }
        with patch(
            "afa_agent.client.requests.post",
            return_value=response,
        ) as post:
            with capture_llm_usage() as ledger:
                result = self.client.chat_json(
                    [{"role": "user", "content": "strict"}],
                    response_schema=schema,
                    schema_name="probe-v1",
                )

        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            payload["response_format"]["json_schema"]["name"],
            "probe-v1",
        )
        self.assertEqual(
            result.response_format_mode,
            "native_json_schema_strict",
        )
        self.assertEqual(
            ledger.calls[0]["response_format_mode"],
            "native_json_schema_strict",
        )


if __name__ == "__main__":
    unittest.main()
