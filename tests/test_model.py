"""Tests for model contexts, structured history, and scripted models."""

import json
import unittest
from dataclasses import FrozenInstanceError

from research_agent.actions import FinishAction, ToolCall
from research_agent.model import (
    MAX_ARGUMENTS_DISPLAY_CHARS,
    HistoryEntry,
    ModelContext,
    ModelError,
    ScriptedModel,
    build_model_context,
    create_history_entry,
)
from research_agent.state import AgentState
from research_agent.tools import ToolErrorCode, ToolResult


def make_context(**changes: object) -> ModelContext:
    values: dict[str, object] = {
        "instructions": "Use tools and report evidence.",
        "objective": "Research speculative decoding",
        "step_number": 1,
        "steps_remaining": 4,
        "available_tools": ("read_notes", "save_note"),
        "history": (),
        "omitted_history_count": 0,
        "notes": (),
        "omitted_note_count": 0,
        "visited_urls": (),
        "omitted_visited_url_count": 0,
    }
    values.update(changes)
    return ModelContext(**values)  # type: ignore[arg-type]


def make_entry(
    step_number: int,
    *,
    output: str = "ok",
) -> HistoryEntry:
    call = ToolCall("read_notes")
    return create_history_entry(
        step_number,
        call,
        ToolResult.success(call.name, output),
    )


class HistoryEntryTests(unittest.TestCase):
    def test_renders_arguments_as_deterministic_compact_json(self) -> None:
        call = ToolCall(
            "save_note",
            {"z": "café", "a": 1},
        )

        entry = create_history_entry(
            2,
            call,
            ToolResult.success("save_note", "Saved note 1."),
        )

        self.assertEqual(entry.step_number, 2)
        self.assertEqual(entry.arguments_display, '{"a":1,"z":"café"}')

    def test_caps_large_argument_displays(self) -> None:
        entry = create_history_entry(
            1,
            ToolCall("save_note", {"note": "x" * 20_000}),
            ToolResult.success("save_note", "saved"),
        )

        self.assertEqual(
            len(entry.arguments_display),
            MAX_ARGUMENTS_DISPLAY_CHARS,
        )
        self.assertTrue(entry.arguments_display.endswith("[truncated]"))

    def test_renders_deeply_frozen_json_arguments(self) -> None:
        entry = create_history_entry(
            1,
            ToolCall(
                "unknown",
                {"filters": {"domains": ["example.com"]}},
            ),
            ToolResult.failure(
                "unknown",
                ToolErrorCode.UNKNOWN_TOOL,
                "unknown tool",
            ),
        )

        rendered = json.loads(entry.arguments_display)
        self.assertEqual(
            rendered,
            {"filters": {"domains": ["example.com"]}},
        )

    def test_rejects_a_result_for_another_tool(self) -> None:
        with self.assertRaises(ValueError):
            create_history_entry(
                1,
                ToolCall("read_notes"),
                ToolResult.success("save_note", "saved"),
            )

    def test_validates_step_number_and_result_type(self) -> None:
        result = ToolResult.success("read_notes", "none")

        with self.assertRaises(TypeError):
            HistoryEntry(True, "read_notes", "{}", result)
        with self.assertRaises(ValueError):
            HistoryEntry(0, "read_notes", "{}", result)
        with self.assertRaises(TypeError):
            HistoryEntry(1, "read_notes", "{}", "bad")  # type: ignore[arg-type]


class ModelContextTests(unittest.TestCase):
    def test_builds_the_next_step_from_current_state(self) -> None:
        history = [make_entry(1), make_entry(2)]
        state = AgentState(
            step_count=2,
            notes=["first", "second"],
            visited_urls={"https://z.example", "https://a.example"},
        )

        context = build_model_context(
            instructions="  Use tools.  ",
            objective="  Research agents  ",
            state=state,
            max_steps=5,
            available_tools=["save_note", "read_notes"],
            history=history,
        )

        self.assertEqual(context.instructions, "Use tools.")
        self.assertEqual(context.objective, "Research agents")
        self.assertEqual(context.step_number, 3)
        self.assertEqual(context.steps_remaining, 3)
        self.assertEqual(context.available_tools, ("read_notes", "save_note"))
        self.assertEqual(context.history, tuple(history))
        self.assertEqual(context.notes, ("first", "second"))
        self.assertEqual(
            context.visited_urls,
            ("https://a.example", "https://z.example"),
        )

    def test_context_is_detached_from_mutable_inputs(self) -> None:
        notes = ["saved"]
        urls = {"https://example.com"}
        history = [make_entry(1)]
        tools = ["read_notes"]
        state = AgentState(
            step_count=1,
            notes=notes,
            visited_urls=urls,
        )

        context = build_model_context(
            instructions="Use tools.",
            objective="Research agents",
            state=state,
            max_steps=4,
            available_tools=tools,
            history=history,
        )
        notes.append("later")
        urls.add("https://later.example")
        history.clear()
        tools.append("save_note")

        self.assertEqual(context.notes, ("saved",))
        self.assertEqual(context.visited_urls, ("https://example.com",))
        self.assertEqual(len(context.history), 1)
        self.assertEqual(context.available_tools, ("read_notes",))
        with self.assertRaises(FrozenInstanceError):
            context.objective = "changed"  # type: ignore[misc]

    def test_keeps_recent_history_within_the_budget(self) -> None:
        history = [
            make_entry(1, output="a" * 39_000),
            make_entry(2, output="b" * 39_000),
            make_entry(3, output="c" * 39_000),
        ]

        context = build_model_context(
            instructions="Use tools.",
            objective="Research agents",
            state=AgentState(step_count=3),
            max_steps=5,
            available_tools=("read_notes",),
            history=history,
        )

        self.assertEqual(
            tuple(entry.step_number for entry in context.history),
            (3,),
        )
        self.assertEqual(context.omitted_history_count, 2)

    def test_keeps_newest_notes_and_reports_omissions(self) -> None:
        state = AgentState(
            notes=[
                "old" * 3_000,
                "middle" * 1_500,
                "new" * 3_000,
            ]
        )

        context = build_model_context(
            instructions="Use notes.",
            objective="Research agents",
            state=state,
            max_steps=5,
            available_tools=("read_notes",),
        )

        self.assertEqual(context.notes, tuple(state.notes[-2:]))
        self.assertEqual(context.omitted_note_count, 1)

    def test_sorts_and_bounds_visited_urls(self) -> None:
        urls = {
            "https://c.example/" + "c" * 3_500,
            "https://a.example/" + "a" * 3_500,
            "https://b.example/" + "b" * 3_500,
        }

        context = build_model_context(
            instructions="Use sources.",
            objective="Research agents",
            state=AgentState(visited_urls=urls),
            max_steps=5,
            available_tools=("read_page",),
        )

        self.assertEqual(context.visited_urls, tuple(sorted(urls))[:2])
        self.assertEqual(context.omitted_visited_url_count, 1)

    def test_rejects_invalid_limits_and_duplicate_tools(self) -> None:
        invalid_limits = (
            (AgentState(step_count=3), 2),
            (AgentState(step_count=2), 2),
        )
        for state, max_steps in invalid_limits:
            with self.subTest(
                step_count=state.step_count,
                max_steps=max_steps,
            ):
                with self.assertRaises(ValueError):
                    build_model_context(
                        instructions="Use tools.",
                        objective="Research agents",
                        state=state,
                        max_steps=max_steps,
                        available_tools=(),
                    )
        with self.assertRaises(ValueError):
            build_model_context(
                instructions="Use tools.",
                objective="Research agents",
                state=AgentState(),
                max_steps=2,
                available_tools=("read_notes", " read_notes "),
            )

    def test_rejects_too_many_available_tools(self) -> None:
        with self.assertRaises(ValueError):
            build_model_context(
                instructions="Use tools.",
                objective="Research agents",
                state=AgentState(),
                max_steps=2,
                available_tools=tuple(
                    f"tool_{index}" for index in range(101)
                ),
            )

    def test_rejects_invalid_unicode_in_seeded_state(self) -> None:
        states = (
            AgentState(notes=["invalid \ud800"]),
            AgentState(visited_urls={"https://example.com/\ud800"}),
        )

        for state in states:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    build_model_context(
                        instructions="Use tools.",
                        objective="Research agents",
                        state=state,
                        max_steps=2,
                        available_tools=(),
                    )

    def test_rejects_duplicate_out_of_order_or_future_history(self) -> None:
        cases = (
            (AgentState(step_count=2), (make_entry(1), make_entry(1))),
            (AgentState(step_count=2), (make_entry(2), make_entry(1))),
            (AgentState(step_count=1), (make_entry(2),)),
        )

        for state, history in cases:
            with self.subTest(steps=tuple(e.step_number for e in history)):
                with self.assertRaises(ValueError):
                    build_model_context(
                        instructions="Use tools.",
                        objective="Research agents",
                        state=state,
                        max_steps=3,
                        available_tools=(),
                        history=history,
                    )

    def test_direct_context_validates_tuple_and_count_fields(self) -> None:
        with self.assertRaises(TypeError):
            make_context(notes=[])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            make_context(omitted_note_count=-1)
        with self.assertRaises(ValueError):
            make_context(steps_remaining=0)

    def test_direct_context_rejects_contradictory_history(self) -> None:
        invalid_values = (
            {
                "step_number": 2,
                "history": (make_entry(1), make_entry(1)),
            },
            {
                "step_number": 2,
                "history": (make_entry(2), make_entry(1)),
            },
            {"step_number": 2, "history": (make_entry(2),)},
            {
                "step_number": 4,
                "history": (make_entry(1), make_entry(3)),
            },
            {
                "step_number": 4,
                "history": (make_entry(3),),
                "omitted_history_count": 1,
            },
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    make_context(**values)

    def test_direct_context_accepts_a_declared_history_suffix(self) -> None:
        context = make_context(
            step_number=4,
            history=(make_entry(3),),
            omitted_history_count=2,
        )

        self.assertEqual(context.omitted_history_count, 2)
        self.assertEqual(context.history[0].step_number, 3)


class ScriptedModelTests(unittest.TestCase):
    def test_copies_actions_and_returns_them_in_order(self) -> None:
        actions = [
            ToolCall("read_notes"),
            FinishAction("Research complete."),
        ]
        model = ScriptedModel(actions)
        actions.clear()
        first_context = make_context()
        second_context = make_context(
            step_number=2,
            steps_remaining=3,
            history=(make_entry(1),),
        )

        self.assertIsInstance(model.choose_action(first_context), ToolCall)
        self.assertIsInstance(model.choose_action(second_context), FinishAction)
        self.assertEqual(model.contexts, (first_context, second_context))

    def test_exhaustion_is_a_safe_model_error_and_records_context(self) -> None:
        model = ScriptedModel(())
        context = make_context()

        with self.assertRaisesRegex(ModelError, "ran out of actions"):
            model.choose_action(context)

        self.assertEqual(model.contexts, (context,))

    def test_rejects_invalid_script_members_and_contexts(self) -> None:
        with self.assertRaises(TypeError):
            ScriptedModel(["not an action"])  # type: ignore[list-item]

        model = ScriptedModel([FinishAction("done")])
        with self.assertRaises(TypeError):
            model.choose_action("not a context")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
