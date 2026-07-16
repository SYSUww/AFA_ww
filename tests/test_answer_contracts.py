from __future__ import annotations

import unittest

from afa_agent.client import LLMResponse
from afa_agent.domains.llm_utils import ask_answer_fallback, ask_option_judgment, finalize_answer
from afa_agent.models import TokenUsage


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self._contents = iter(contents)
        self.calls: list[list[dict[str, str]]] = []

    def chat_json(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(
            content=next(self._contents),
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            raw_payload={},
        )


def ask_judgment(client: FakeClient) -> tuple[dict[str, object], TokenUsage]:
    return ask_option_judgment(
        client,  # type: ignore[arg-type]
        "system",
        "question",
        "mcq",
        "A",
        "option",
        "evidence",
    )


class OptionJudgmentContractTests(unittest.TestCase):
    def test_string_false_is_retried_instead_of_becoming_true(self) -> None:
        client = FakeClient([
            '{"label": "false", "reasoning_summary": "first"}',
            '{"label": false, "reasoning_summary": "second"}',
        ])

        parsed, usage = ask_judgment(client)

        self.assertIs(parsed["label"], False)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("label 必须是 JSON 布尔值", client.calls[1][1]["content"])
        self.assertEqual(usage.total_tokens, 24)

    def test_numeric_zero_and_one_are_normalized_to_bool(self) -> None:
        for raw_label, expected in ((0, False), (1, True)):
            with self.subTest(raw_label=raw_label):
                client = FakeClient([f'{{"label": {raw_label}, "reasoning_summary": "ok"}}'])

                parsed, _ = ask_judgment(client)

                self.assertIs(parsed["label"], expected)
                self.assertEqual(len(client.calls), 1)

    def test_consecutive_invalid_label_types_fail_explicitly(self) -> None:
        client = FakeClient([
            '{"label": "false"}',
            '{"label": null}',
        ])

        with self.assertRaisesRegex(ValueError, "Option judgment label must be boolean"):
            ask_judgment(client)

        self.assertEqual(len(client.calls), 2)


class AnswerFallbackContractTests(unittest.TestCase):
    def test_consecutive_invalid_responses_do_not_default_to_first_option(self) -> None:
        client = FakeClient(["not json", '{"answer": 1}'])

        with self.assertRaisesRegex(ValueError, "no valid answer after 2 attempts"):
            ask_answer_fallback(
                client,  # type: ignore[arg-type]
                "system",
                "question",
                [],
                "mcq",
                ["A", "B", "C", "D"],
            )

        self.assertEqual(len(client.calls), 2)


class SupportedOnlyContractTests(unittest.TestCase):
    POLICY = {
        "enabled": True,
        "mode": "supported_only",
        "allow_supported_only_output": True,
        "supported_only_formats": ["mcq", "multi"],
    }

    def test_supported_only_does_not_shrink_valid_multi_to_one_option(self) -> None:
        answer, metadata = finalize_answer(
            "AB",
            answer_format="multi",
            allowed_options=["A", "B", "C", "D"],
            option_labels={"A": True, "B": False, "C": False, "D": False},
            answer_policy_settings=self.POLICY,
        )

        self.assertEqual(answer, "AB")
        self.assertFalse(metadata["answer_policy"]["is_supported_only_format_compliant"])
        self.assertNotIn("applied_mode", metadata["answer_policy"])

    def test_supported_only_applies_when_multi_minimum_is_met(self) -> None:
        answer, metadata = finalize_answer(
            "ABC",
            answer_format="multi",
            allowed_options=["A", "B", "C", "D"],
            option_labels={"A": True, "B": True, "C": False, "D": False},
            answer_policy_settings=self.POLICY,
        )

        self.assertEqual(answer, "AB")
        self.assertEqual(metadata["answer_policy"]["applied_mode"], "supported_only")

    def test_known_format_conflicts_replay_to_valid_multi_without_model_calls(self) -> None:
        cases = {
            "fin_a_011": ("D", {"A": False, "B": False, "C": False, "D": True}),
            "ins_a_009": ("C", {"A": False, "B": False, "C": True, "D": False}),
            "ins_a_012": ("A", {"A": True, "B": False, "C": False, "D": False}),
            "ins_a_014": ("A", {"A": True, "B": False, "C": False, "D": False}),
            "ins_a_016": ("D", {"A": False, "B": False, "C": False, "D": True}),
        }
        for qid, (raw_answer, labels) in cases.items():
            with self.subTest(qid=qid):
                answer, metadata = finalize_answer(
                    raw_answer,
                    answer_format="multi",
                    allowed_options=["A", "B", "C", "D"],
                    option_labels=labels,
                    answer_policy_settings=self.POLICY,
                )

                self.assertGreaterEqual(len(set(answer)), 2)
                self.assertNotIn("applied_mode", metadata["answer_policy"])


if __name__ == "__main__":
    unittest.main()
