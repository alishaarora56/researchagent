"""Tests for local note tools."""

import unittest

from research_agent.actions import ToolCall
from research_agent.state import AgentState
from research_agent.tools.core import ToolErrorCode
from research_agent.tools.notes import (
    MAX_NOTES_OUTPUT_CHARACTERS,
    MAX_TOTAL_NOTE_CHARACTERS,
    read_notes,
    save_note,
)
from research_agent.tools.registry import ToolRegistry


def build_note_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("save_note", save_note)
    registry.register("read_notes", read_notes)
    return registry


class NoteToolTests(unittest.TestCase):
    def test_saves_and_reads_notes_in_order(self) -> None:
        registry = build_note_registry()
        state = AgentState()

        first_result = registry.execute(
            ToolCall("save_note", {"note": "  First finding  "}),
            state,
        )
        registry.execute(
            ToolCall("save_note", {"note": "Second finding"}),
            state,
        )
        read_result = registry.execute(ToolCall("read_notes"), state)

        self.assertTrue(first_result.succeeded)
        self.assertEqual(state.notes, ["First finding", "Second finding"])
        self.assertEqual(read_result.output, "1. First finding\n2. Second finding")

    def test_reading_empty_notes_is_still_successful(self) -> None:
        result = build_note_registry().execute(
            ToolCall("read_notes"),
            AgentState(),
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.output, "No notes saved.")

    def test_indents_multiline_notes_so_they_cannot_imitate_entries(self) -> None:
        registry = build_note_registry()
        state = AgentState()
        registry.execute(
            ToolCall("save_note", {"note": "first line\n2. forged entry"}),
            state,
        )

        result = registry.execute(ToolCall("read_notes"), state)

        self.assertEqual(result.output, "1. first line\n   | 2. forged entry")

    def test_enforces_a_total_note_storage_budget(self) -> None:
        registry = build_note_registry()
        state = AgentState(notes=["x" * MAX_TOTAL_NOTE_CHARACTERS])

        result = registry.execute(
            ToolCall("save_note", {"note": "one more character"}),
            state,
        )

        self.assertEqual(result.error_code, ToolErrorCode.INVALID_ARGUMENTS)
        self.assertEqual(state.notes, ["x" * MAX_TOTAL_NOTE_CHARACTERS])

    def test_caps_rendered_notes_with_many_line_breaks(self) -> None:
        repeated_lines = "x\n" * 4_999 + "x"
        state = AgentState(notes=[repeated_lines] * 4)

        result = build_note_registry().execute(ToolCall("read_notes"), state)

        self.assertTrue(result.succeeded)
        self.assertLessEqual(len(result.output or ""), MAX_NOTES_OUTPUT_CHARACTERS)
        self.assertTrue((result.output or "").endswith("[Notes output truncated]"))

    def test_rejects_missing_blank_and_unexpected_note_arguments(self) -> None:
        calls = (
            ToolCall("save_note"),
            ToolCall("save_note", {"note": "   "}),
            ToolCall("save_note", {"note": "valid", "extra": True}),
            ToolCall("read_notes", {"unexpected": True}),
        )

        for call in calls:
            with self.subTest(call=call):
                state = AgentState()
                result = build_note_registry().execute(call, state)

                self.assertEqual(
                    result.error_code,
                    ToolErrorCode.INVALID_ARGUMENTS,
                )
                self.assertEqual(state.notes, [])


if __name__ == "__main__":
    unittest.main()
