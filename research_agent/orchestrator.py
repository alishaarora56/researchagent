"""Deterministic agent loop connecting models, tools, state, and limits."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from research_agent.actions import FinishAction, ToolCall
from research_agent.config import ResearchRequest
from research_agent.model import (
    AgentModel,
    HistoryEntry,
    ModelError,
    build_model_context,
    create_history_entry,
)
from research_agent.state import AgentState, AgentStateSnapshot, RunStatus
from research_agent.tools.registry import ToolRegistry


MAX_MODEL_ERROR_CHARACTERS = 2_000


class StopReason(str, Enum):
    """Why an agent run returned control to its caller."""

    FINISHED = "finished"
    STEP_LIMIT = "step_limit"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"


@dataclass(frozen=True)
class RunResult:
    """Final outcome and structured audit trail for one agent run."""

    stop_reason: StopReason
    state: AgentStateSnapshot
    history: tuple[HistoryEntry, ...]
    answer: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stop_reason, StopReason):
            raise TypeError("stop_reason must be a StopReason")
        if not isinstance(self.state, AgentStateSnapshot):
            raise TypeError("state must be an AgentStateSnapshot")
        if not isinstance(self.history, tuple) or not all(
            isinstance(entry, HistoryEntry) for entry in self.history
        ):
            raise TypeError("history must be a tuple of HistoryEntry objects")

        expected_steps = tuple(range(1, len(self.history) + 1))
        history_steps = tuple(
            entry.step_number for entry in self.history
        )
        if history_steps != expected_steps:
            raise ValueError("history must contain consecutive tool steps")
        history_tools = tuple(entry.tool_name for entry in self.history)
        if history_tools != self.state.tool_history:
            raise ValueError("history must match state tool_history")
        if not len(self.history) <= self.state.step_count <= len(self.history) + 1:
            raise ValueError("history does not match state step_count")

        if self.stop_reason is StopReason.FINISHED:
            if self.state.status is not RunStatus.FINISHED:
                raise ValueError("a finished result requires FINISHED state")
            if self.state.step_count != len(self.history) + 1:
                raise ValueError("a finished result requires a final model step")
            if not isinstance(self.answer, str) or not self.answer.strip():
                raise ValueError("a finished result requires an answer")
            if self.error_message is not None:
                raise ValueError("a finished result cannot have an error message")
            return

        if self.state.status is not RunStatus.FAILED:
            raise ValueError("a stopped result requires FAILED state")
        if self.answer is not None:
            raise ValueError("a stopped result cannot have an answer")
        if (
            not isinstance(self.error_message, str)
            or not self.error_message.strip()
        ):
            raise ValueError("a stopped result requires an error message")
        if (
            self.stop_reason is StopReason.MODEL_ERROR
            and self.state.step_count != len(self.history) + 1
        ):
            raise ValueError(
                "a model error requires one attempted model step"
            )
        if self.stop_reason is StopReason.STEP_LIMIT and (
            self.state.step_count == 0
            or self.state.step_count != len(self.history)
        ):
            raise ValueError(
                "a step-limit result requires all attempted steps in history"
            )

    @property
    def succeeded(self) -> bool:
        """Return whether the model explicitly finished the run."""

        return self.stop_reason is StopReason.FINISHED


def _failed_result(
    *,
    stop_reason: StopReason,
    state: AgentState,
    history: list[HistoryEntry],
    message: str,
) -> RunResult:
    state.status = RunStatus.FAILED
    return RunResult(
        stop_reason=stop_reason,
        state=AgentStateSnapshot.from_state(state),
        history=tuple(history),
        error_message=message,
    )


def run_agent(
    request: ResearchRequest,
    *,
    model: AgentModel,
    tools: ToolRegistry,
    state: AgentState | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Run model actions until finish, a safety stop, or an expected failure.

    Timeouts are cooperative boundaries around synchronous model and tool calls.
    A blocking dependency still needs its own timeout because this function
    cannot interrupt Python code that has not returned.
    """

    if not isinstance(request, ResearchRequest):
        raise TypeError("request must be a ResearchRequest")
    if not isinstance(tools, ToolRegistry):
        raise TypeError("tools must be a ToolRegistry")
    if not callable(getattr(model, "choose_action", None)):
        raise TypeError("model must provide choose_action(context)")
    if not callable(clock):
        raise TypeError("clock must be callable")

    run_state = AgentState() if state is None else state
    if not isinstance(run_state, AgentState):
        raise TypeError("state must be an AgentState")
    if run_state.status is not RunStatus.READY:
        raise ValueError("agent state must be READY before a run")
    if run_state.step_count != 0 or run_state.tool_history:
        raise ValueError("agent state must not contain an earlier run")

    objective = request.objective
    max_steps = request.config.max_steps
    timeout_seconds = request.config.timeout_seconds
    available_tools = tools.names
    history: list[HistoryEntry] = []
    deadline = clock() + timeout_seconds
    run_state.status = RunStatus.RUNNING

    try:
        while True:
            if clock() >= deadline:
                return _failed_result(
                    stop_reason=StopReason.TIMEOUT,
                    state=run_state,
                    history=history,
                    message=(
                        "Agent stopped after reaching "
                        f"timeout_seconds={timeout_seconds}."
                    ),
                )
            if run_state.step_count >= max_steps:
                return _failed_result(
                    stop_reason=StopReason.STEP_LIMIT,
                    state=run_state,
                    history=history,
                    message=(
                        "Agent stopped after reaching "
                        f"max_steps={max_steps} before the model finished."
                    ),
                )

            context = build_model_context(
                objective=objective,
                max_steps=max_steps,
                state=run_state,
                available_tools=available_tools,
                history=tuple(history),
            )
            run_state.step_count += 1

            try:
                action = model.choose_action(context)
            except ModelError as error:
                if clock() >= deadline:
                    return _failed_result(
                        stop_reason=StopReason.TIMEOUT,
                        state=run_state,
                        history=history,
                        message=(
                            "Agent stopped after reaching "
                            f"timeout_seconds={timeout_seconds}."
                        ),
                    )
                message = str(error).strip() or "Model execution failed."
                message = message.encode(
                    "utf-8",
                    errors="replace",
                ).decode("utf-8")
                if len(message) > MAX_MODEL_ERROR_CHARACTERS:
                    marker = "\n[Model error truncated]"
                    available = MAX_MODEL_ERROR_CHARACTERS - len(marker)
                    message = message[:available].rstrip() + marker
                return _failed_result(
                    stop_reason=StopReason.MODEL_ERROR,
                    state=run_state,
                    history=history,
                    message=message,
                )

            if not isinstance(action, (ToolCall, FinishAction)):
                raise TypeError(
                    "model must return a ToolCall or FinishAction"
                )
            if clock() >= deadline:
                return _failed_result(
                    stop_reason=StopReason.TIMEOUT,
                    state=run_state,
                    history=history,
                    message=(
                        "Agent stopped after reaching "
                        f"timeout_seconds={timeout_seconds}."
                    ),
                )

            if isinstance(action, FinishAction):
                run_state.status = RunStatus.FINISHED
                return RunResult(
                    stop_reason=StopReason.FINISHED,
                    state=AgentStateSnapshot.from_state(run_state),
                    history=tuple(history),
                    answer=action.answer,
                )

            run_state.tool_history.append(action.name)
            expected_step_count = run_state.step_count
            expected_tool_history = list(run_state.tool_history)
            try:
                tool_result = tools.execute(action, run_state)
            finally:
                run_state.step_count = expected_step_count
                run_state.tool_history = expected_tool_history
                run_state.status = RunStatus.RUNNING
            history.append(
                create_history_entry(
                    run_state.step_count,
                    action,
                    tool_result,
                )
            )

            if clock() >= deadline:
                return _failed_result(
                    stop_reason=StopReason.TIMEOUT,
                    state=run_state,
                    history=history,
                    message=(
                        "Agent stopped after reaching "
                        f"timeout_seconds={timeout_seconds}."
                    ),
                )
    except BaseException:
        run_state.status = RunStatus.FAILED
        raise
    finally:
        if run_state.status is RunStatus.RUNNING:
            run_state.status = RunStatus.FAILED
