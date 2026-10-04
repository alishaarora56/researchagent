"""Tests for the deterministic model-tool orchestration loop."""

import unittest
from dataclasses import FrozenInstanceError

from research_agent.actions import FinishAction, ToolCall
from research_agent.config import AgentConfig, ResearchRequest
from research_agent.model import (
    HistoryEntry,
    ModelContext,
    ModelError,
    ScriptedModel,
)
from research_agent.orchestrator import (
    MAX_MODEL_ERROR_CHARACTERS,
    RunResult,
    StopReason,
    run_agent,
)
from research_agent.state import AgentState, AgentStateSnapshot, RunStatus
from research_agent.tools.core import ToolErrorCode, ToolResult
from research_agent.tools.notes import read_notes, save_note
from research_agent.tools.registry import ToolRegistry


def request_with(
    *,
    max_steps: int = 8,
    timeout_seconds: int = 120,
) -> ResearchRequest:
    return ResearchRequest(
        objective="Research speculative decoding",
        config=AgentConfig(
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        ),
    )


def note_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("save_note", save_note)
    registry.register("read_notes", read_notes)
    return registry


class ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class AdvancingModel:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.call_count = 0

    def choose_action(self, context: ModelContext) -> ToolCall:
        self.call_count += 1
        self.clock.value = 1.0
        return ToolCall("save_note", {"note": "too late"})


class InvalidModel:
    def choose_action(self, context: ModelContext) -> object:
        return "not an action"


class FailingModel:
    def choose_action(self, context: ModelContext) -> object:
        raise ModelError("x" * 3_000)


class LateFailingModel:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock

    def choose_action(self, context: ModelContext) -> object:
        self.clock.value = 1.0
        raise ModelError("provider failed after the deadline")


class OrchestratorTests(unittest.TestCase):
    def test_runs_tools_then_finishes_with_immutable_context_snapshots(self) -> None:
        model = ScriptedModel(
            (
                ToolCall("save_note", {"note": "Draft models propose tokens."}),
                ToolCall("read_notes"),
                FinishAction("Research complete."),
            )
        )

        result = run_agent(
            request_with(),
            model=model,
            tools=note_registry(),
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.stop_reason, StopReason.FINISHED)
        self.assertEqual(result.answer, "Research complete.")
        self.assertEqual(result.state.status, RunStatus.FINISHED)
        self.assertEqual(result.state.step_count, 3)
        self.assertEqual(result.state.tool_history, ("save_note", "read_notes"))
        self.assertEqual(result.state.notes, ("Draft models propose tokens.",))
        self.assertEqual(len(result.history), 2)

        first_context, second_context, third_context = model.contexts
        self.assertEqual(first_context.history, ())
        self.assertEqual(first_context.notes, ())
        self.assertEqual(len(second_context.history), 1)
        self.assertEqual(second_context.notes, ("Draft models propose tokens.",))
        self.assertEqual(len(third_context.history), 2)
        self.assertEqual(first_context.notes, ())

    def test_can_finish_immediately(self) -> None:
        result = run_agent(
            request_with(max_steps=1),
            model=ScriptedModel((FinishAction("Done."),)),
            tools=ToolRegistry(),
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.state.step_count, 1)
        self.assertEqual(result.history, ())

    def test_tool_failure_is_an_observation_and_model_can_recover(self) -> None:
        model = ScriptedModel(
            (
                ToolCall("missing_tool"),
                FinishAction("Recovered from the tool error."),
            )
        )

        result = run_agent(
            request_with(),
            model=model,
            tools=ToolRegistry(),
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(
            result.history[0].result.error_code,
            ToolErrorCode.UNKNOWN_TOOL,
        )
        self.assertEqual(len(model.contexts[1].history), 1)

    def test_finish_on_the_last_allowed_step_succeeds(self) -> None:
        result = run_agent(
            request_with(max_steps=2),
            model=ScriptedModel(
                (ToolCall("missing"), FinishAction("Done on step two."))
            ),
            tools=ToolRegistry(),
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.state.step_count, 2)

    def test_tool_on_last_step_stops_without_an_extra_model_call(self) -> None:
        model = ScriptedModel(
            (
                ToolCall("missing"),
                ToolCall("still_missing"),
                FinishAction("This action must never be requested."),
            )
        )

        result = run_agent(
            request_with(max_steps=2),
            model=model,
            tools=ToolRegistry(),
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.stop_reason, StopReason.STEP_LIMIT)
        self.assertEqual(result.state.status, RunStatus.FAILED)
        self.assertEqual(result.state.step_count, 2)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(len(model.contexts), 2)

    def test_timeout_before_first_model_call_uses_no_step(self) -> None:
        calls = iter((0.0, 1.0))

        result = run_agent(
            request_with(timeout_seconds=1),
            model=ScriptedModel((FinishAction("too late"),)),
            tools=ToolRegistry(),
            clock=lambda: next(calls),
        )

        self.assertEqual(result.stop_reason, StopReason.TIMEOUT)
        self.assertEqual(result.state.step_count, 0)

    def test_action_returned_at_deadline_is_not_executed(self) -> None:
        clock = ManualClock()
        model = AdvancingModel(clock)

        result = run_agent(
            request_with(timeout_seconds=1),
            model=model,
            tools=note_registry(),
            clock=clock,
        )

        self.assertEqual(result.stop_reason, StopReason.TIMEOUT)
        self.assertEqual(result.state.step_count, 1)
        self.assertEqual(result.state.notes, ())
        self.assertEqual(result.history, ())
        self.assertEqual(model.call_count, 1)

    def test_tool_result_is_recorded_when_tool_reaches_deadline(self) -> None:
        clock = ManualClock()
        registry = ToolRegistry()

        def slow_tool(arguments: object, state: AgentState) -> str:
            clock.value = 1.0
            return "completed side effect"

        registry.register("slow", slow_tool)
        result = run_agent(
            request_with(timeout_seconds=1),
            model=ScriptedModel((ToolCall("slow"), FinishAction("unused"))),
            tools=registry,
            clock=clock,
        )

        self.assertEqual(result.stop_reason, StopReason.TIMEOUT)
        self.assertEqual(result.state.step_count, 1)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0].result.output, "completed side effect")

    def test_script_exhaustion_is_a_safe_model_failure(self) -> None:
        result = run_agent(
            request_with(),
            model=ScriptedModel(()),
            tools=ToolRegistry(),
        )

        self.assertEqual(result.stop_reason, StopReason.MODEL_ERROR)
        self.assertEqual(result.state.status, RunStatus.FAILED)
        self.assertEqual(result.state.step_count, 1)
        self.assertIn("ran out", result.error_message or "")

    def test_model_error_message_is_bounded(self) -> None:
        result = run_agent(
            request_with(),
            model=FailingModel(),
            tools=ToolRegistry(),
        )

        self.assertEqual(result.stop_reason, StopReason.MODEL_ERROR)
        self.assertLessEqual(
            len(result.error_message or ""),
            MAX_MODEL_ERROR_CHARACTERS,
        )
        self.assertTrue(
            (result.error_message or "").endswith("[Model error truncated]")
        )

    def test_model_error_after_deadline_is_classified_as_timeout(self) -> None:
        clock = ManualClock()

        result = run_agent(
            request_with(timeout_seconds=1),
            model=LateFailingModel(clock),
            tools=ToolRegistry(),
            clock=clock,
        )

        self.assertEqual(result.stop_reason, StopReason.TIMEOUT)

    def test_unexpected_tool_bug_surfaces_but_state_is_not_left_running(self) -> None:
        state = AgentState()
        registry = ToolRegistry()

        def broken(arguments: object, received_state: AgentState) -> str:
            received_state.step_count = 0
            received_state.status = RunStatus.FINISHED
            raise RuntimeError("programming bug")

        registry.register("broken", broken)

        with self.assertRaisesRegex(RuntimeError, "programming bug"):
            run_agent(
                request_with(),
                model=ScriptedModel((ToolCall("broken"),)),
                tools=registry,
                state=state,
            )

        self.assertEqual(state.status, RunStatus.FAILED)
        self.assertEqual(state.step_count, 1)

    def test_tool_cannot_rewind_control_state_to_bypass_limits(self) -> None:
        state = AgentState()
        registry = ToolRegistry()

        def tamper(arguments: object, received_state: AgentState) -> str:
            received_state.step_count = 0
            received_state.status = RunStatus.FINISHED
            received_state.tool_history.clear()
            return "attempted tampering"

        registry.register("tamper", tamper)
        model = ScriptedModel(
            (
                ToolCall("tamper"),
                ToolCall("tamper"),
                FinishAction("must not run"),
            )
        )

        result = run_agent(
            request_with(max_steps=2),
            model=model,
            tools=registry,
            state=state,
        )

        self.assertEqual(result.stop_reason, StopReason.STEP_LIMIT)
        self.assertEqual(result.state.step_count, 2)
        self.assertEqual(result.state.tool_history, ("tamper", "tamper"))
        self.assertEqual(len(model.contexts), 2)

    def test_tool_cannot_rewrite_nested_arguments_before_audit(self) -> None:
        registry = ToolRegistry()

        def attempt_mutation(arguments: object, state: AgentState) -> str:
            nested = arguments["nested"]  # type: ignore[index]
            with self.assertRaises(TypeError):
                nested["value"] = "changed"  # type: ignore[index]
            return "arguments stayed immutable"

        registry.register("inspect", attempt_mutation)
        result = run_agent(
            request_with(),
            model=ScriptedModel(
                (
                    ToolCall(
                        "inspect",
                        {"nested": {"value": "original"}},
                    ),
                    FinishAction("Done."),
                )
            ),
            tools=registry,
        )

        self.assertTrue(result.succeeded)
        self.assertIn('"value":"original"', result.history[0].arguments_display)

    def test_result_contains_an_immutable_state_snapshot(self) -> None:
        state = AgentState()
        result = run_agent(
            request_with(),
            model=ScriptedModel((FinishAction("Done."),)),
            tools=ToolRegistry(),
            state=state,
        )

        state.status = RunStatus.FAILED
        state.notes.append("changed later")

        self.assertTrue(result.succeeded)
        self.assertEqual(result.state.status, RunStatus.FINISHED)
        self.assertEqual(result.state.notes, ())
        with self.assertRaises(FrozenInstanceError):
            result.state.status = RunStatus.FAILED  # type: ignore[misc]

    def test_run_result_rejects_history_that_disagrees_with_state(self) -> None:
        state = AgentStateSnapshot(
            step_count=1,
            visited_urls=(),
            notes=(),
            tool_history=("save_note",),
            status=RunStatus.FAILED,
        )

        with self.assertRaisesRegex(ValueError, "tool_history"):
            RunResult(
                stop_reason=StopReason.STEP_LIMIT,
                state=state,
                history=(),
                error_message="Stopped.",
            )

    def test_run_result_rejects_nonconsecutive_history(self) -> None:
        call = ToolCall("read_notes")
        entry = HistoryEntry(
            step_number=2,
            tool_name=call.name,
            arguments_display="{}",
            result=ToolResult.success(call.name, "No notes."),
        )
        state = AgentStateSnapshot(
            step_count=2,
            visited_urls=(),
            notes=(),
            tool_history=("read_notes",),
            status=RunStatus.FAILED,
        )

        with self.assertRaisesRegex(ValueError, "consecutive"):
            RunResult(
                stop_reason=StopReason.STEP_LIMIT,
                state=state,
                history=(entry,),
                error_message="Stopped.",
            )

    def test_run_result_enforces_reason_specific_step_counts(self) -> None:
        impossible_results = (
            {
                "stop_reason": StopReason.MODEL_ERROR,
                "state": AgentStateSnapshot(
                    step_count=0,
                    visited_urls=(),
                    notes=(),
                    tool_history=(),
                    status=RunStatus.FAILED,
                ),
            },
            {
                "stop_reason": StopReason.STEP_LIMIT,
                "state": AgentStateSnapshot(
                    step_count=1,
                    visited_urls=(),
                    notes=(),
                    tool_history=(),
                    status=RunStatus.FAILED,
                ),
            },
        )

        for values in impossible_results:
            with self.subTest(reason=values["stop_reason"]):
                with self.assertRaises(ValueError):
                    RunResult(
                        **values,  # type: ignore[arg-type]
                        history=(),
                        error_message="Stopped.",
                    )

    def test_invalid_model_return_surfaces_and_marks_state_failed(self) -> None:
        state = AgentState()

        with self.assertRaisesRegex(TypeError, "ToolCall or FinishAction"):
            run_agent(
                request_with(),
                model=InvalidModel(),  # type: ignore[arg-type]
                tools=ToolRegistry(),
                state=state,
            )

        self.assertEqual(state.status, RunStatus.FAILED)
        self.assertEqual(state.step_count, 1)

    def test_rejects_reused_or_inconsistent_state_without_mutating_it(self) -> None:
        states = (
            AgentState(status=RunStatus.RUNNING),
            AgentState(status=RunStatus.FINISHED),
            AgentState(status=RunStatus.FAILED),
            AgentState(step_count=1),
            AgentState(tool_history=["old_call"]),
        )

        for state in states:
            original_status = state.status
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    run_agent(
                        request_with(),
                        model=ScriptedModel((FinishAction("Done."),)),
                        tools=ToolRegistry(),
                        state=state,
                    )

                self.assertEqual(state.status, original_status)

    def test_context_sorts_visited_urls_and_copies_seed_notes(self) -> None:
        state = AgentState(
            visited_urls={"https://z.example", "https://a.example"},
            notes=["seed note"],
        )
        model = ScriptedModel((FinishAction("Done."),))

        run_agent(
            request_with(),
            model=model,
            tools=ToolRegistry(),
            state=state,
        )

        context = model.contexts[0]
        self.assertEqual(
            context.visited_urls,
            ("https://a.example", "https://z.example"),
        )
        self.assertEqual(context.notes, ("seed note",))


if __name__ == "__main__":
    unittest.main()
