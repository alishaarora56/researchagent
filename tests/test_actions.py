"""Tests for actions proposed by a model."""

import unittest
from dataclasses import FrozenInstanceError

from research_agent.actions import FinishAction, ToolCall


class ToolCallTests(unittest.TestCase):
    def test_stores_a_named_tool_and_structured_arguments(self) -> None:
        action = ToolCall(
            name="  save_note  ",
            arguments={"note": "Speculative decoding uses a draft model."},
        )

        self.assertEqual(action.name, "save_note")
        self.assertEqual(
            action.arguments,
            {"note": "Speculative decoding uses a draft model."},
        )

    def test_allows_a_tool_with_no_arguments(self) -> None:
        action = ToolCall(name="read_notes")

        self.assertEqual(action.arguments, {})

    def test_rejects_an_invalid_tool_name(self) -> None:
        invalid_names = (
            "",
            "   ",
            42,
            "x" * 101,
            "invalid \ud800",
        )

        for name in invalid_names:
            with self.subTest(name=name):
                expected_error = TypeError if not isinstance(name, str) else ValueError
                with self.assertRaises(expected_error):
                    ToolCall(name=name)  # type: ignore[arg-type]

    def test_rejects_non_mapping_arguments(self) -> None:
        with self.assertRaises(TypeError):
            ToolCall(name="save_note", arguments=["a note"])  # type: ignore[arg-type]

    def test_copies_and_protects_arguments_from_mutation(self) -> None:
        original_arguments = {"note": "Original finding"}
        action = ToolCall(name="save_note", arguments=original_arguments)

        original_arguments["note"] = "Changed outside the action"

        self.assertEqual(action.arguments["note"], "Original finding")
        with self.assertRaises(TypeError):
            action.arguments["note"] = "Changed on the action"  # type: ignore[index]

    def test_deeply_freezes_json_arguments(self) -> None:
        original = {"filters": {"domains": ["example.com"]}}
        action = ToolCall("search", original)

        original["filters"]["domains"].append("changed.example")
        original["filters"]["extra"] = True

        filters = action.arguments["filters"]
        self.assertEqual(
            filters["domains"],  # type: ignore[index]
            ("example.com",),
        )
        self.assertNotIn("extra", filters)  # type: ignore[operator]
        with self.assertRaises(TypeError):
            filters["extra"] = True  # type: ignore[index]

    def test_rejects_unbounded_or_non_json_arguments(self) -> None:
        invalid_arguments = (
            {"text": "x" * 50_001},
            {"value": object()},
            {"value": float("inf")},
            {"items": list(range(1_001))},
        )

        for arguments in invalid_arguments:
            argument_type = (
                type(arguments["value"])
                if "value" in arguments
                else "collection"
            )
            with self.subTest(arguments_type=argument_type):
                with self.assertRaises((TypeError, ValueError)):
                    ToolCall("tool", arguments)


class FinishActionTests(unittest.TestCase):
    def test_stores_a_trimmed_final_answer(self) -> None:
        action = FinishAction(answer="  Research complete.  ")

        self.assertEqual(action.answer, "Research complete.")

    def test_rejects_an_invalid_final_answer(self) -> None:
        invalid_answers = (
            "",
            "   ",
            None,
            "x" * 50_001,
            "invalid \ud800",
        )

        for answer in invalid_answers:
            with self.subTest(answer=answer):
                expected_error = (
                    TypeError if not isinstance(answer, str) else ValueError
                )
                with self.assertRaises(expected_error):
                    FinishAction(answer=answer)  # type: ignore[arg-type]

    def test_cannot_be_changed_after_validation(self) -> None:
        action = FinishAction(answer="Research complete.")

        with self.assertRaises(FrozenInstanceError):
            action.answer = ""  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
