from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import requests

from afa_agent.client import OpenAICompatibleClient, capture_llm_usage
from afa_agent.config import ModelConfig


class FakeHTTPResponse:
    def __init__(self, payload: dict, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"status={self.status_code}",
                response=self,
            )

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

    def test_extra_body_adds_provider_specific_options(self) -> None:
        response = FakeHTTPResponse(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )
        with patch(
            "afa_agent.client.requests.post",
            return_value=response,
        ) as post:
            self.client.chat_json(
                [{"role": "user", "content": "direct"}],
                extra_body={"enable_thinking": False},
            )

        payload = json.loads(post.call_args.kwargs["data"])
        self.assertIs(payload["enable_thinking"], False)

    def test_extra_body_cannot_override_core_request_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected request fields"):
            self.client.chat_json(
                [{"role": "user", "content": "direct"}],
                extra_body={"model": "different-model"},
            )

    def test_read_timeout_is_not_retried_without_a_usage_response(self) -> None:
        client = OpenAICompatibleClient(
            ModelConfig(
                api_key="test",
                api_base="https://example.invalid/v1",
                model_name="qwen3.7-plus",
                max_retries=2,
            )
        )
        with patch(
            "afa_agent.client.requests.post",
            side_effect=requests.ReadTimeout("slow generation"),
        ) as post:
            with capture_llm_usage() as ledger:
                with self.assertRaises(requests.ReadTimeout):
                    client.chat_json([{"role": "user", "content": "slow"}])

        self.assertEqual(post.call_count, 1)
        self.assertEqual(ledger.calls, [])

    def test_connection_error_is_not_retried_after_post_dispatch(self) -> None:
        client = OpenAICompatibleClient(
            ModelConfig(
                api_key="test",
                api_base="https://example.invalid/v1",
                model_name="qwen3.7-plus",
                max_retries=2,
            )
        )
        with patch(
            "afa_agent.client.requests.post",
            side_effect=requests.ConnectionError("ambiguous disconnect"),
        ) as post:
            with self.assertRaises(requests.ConnectionError):
                client.chat_json([{"role": "user", "content": "ambiguous"}])

        self.assertEqual(post.call_count, 1)

    def test_http_500_is_not_retried_without_provider_usage(self) -> None:
        client = OpenAICompatibleClient(
            ModelConfig(
                api_key="test",
                api_base="https://example.invalid/v1",
                model_name="qwen3.7-plus",
                max_retries=2,
            )
        )
        with patch(
            "afa_agent.client.requests.post",
            return_value=FakeHTTPResponse({}, status_code=500),
        ) as post:
            with self.assertRaises(requests.HTTPError):
                client.chat_json([{"role": "user", "content": "server error"}])

        self.assertEqual(post.call_count, 1)

    def test_invalid_json_response_is_not_retried(self) -> None:
        class InvalidJSONResponse(FakeHTTPResponse):
            def json(self) -> dict:
                raise ValueError("invalid response body")

        client = OpenAICompatibleClient(
            ModelConfig(
                api_key="test",
                api_base="https://example.invalid/v1",
                model_name="qwen3.7-plus",
                max_retries=2,
            )
        )
        with patch(
            "afa_agent.client.requests.post",
            return_value=InvalidJSONResponse({}),
        ) as post:
            with self.assertRaisesRegex(ValueError, "invalid response body"):
                client.chat_json([{"role": "user", "content": "bad json"}])

        self.assertEqual(post.call_count, 1)

    def test_explicit_http_429_rejection_can_retry(self) -> None:
        client = OpenAICompatibleClient(
            ModelConfig(
                api_key="test",
                api_base="https://example.invalid/v1",
                model_name="qwen3.7-plus",
                max_retries=2,
                retry_backoff_seconds=0,
            )
        )
        success = FakeHTTPResponse(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )
        with patch(
            "afa_agent.client.requests.post",
            side_effect=[
                FakeHTTPResponse({}, status_code=429),
                success,
            ],
        ) as post:
            with capture_llm_usage() as ledger:
                response = client.chat_json(
                    [{"role": "user", "content": "rate limit"}]
                )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(ledger.total()["total_tokens"], 12)
        self.assertEqual(response.transport_attempt_count, 2)
        self.assertEqual(
            response.transport_rejections,
            (
                {
                    "attempt_index": 1,
                    "status_code": 429,
                    "pre_generation_rejection": True,
                    "token_usage_observed": False,
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
