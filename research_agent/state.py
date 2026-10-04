"""Mutable state accumulated during one research-agent run."""

from dataclasses import dataclass, field
from enum import Enum


class RunStatus(Enum):
    """Lifecycle stages for a research-agent run."""

    READY = "ready"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class AgentState:
    """Information that changes while the agent works on an objective."""

    step_count: int = 0
    visited_urls: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    tool_history: list[str] = field(default_factory=list)
    status: RunStatus = RunStatus.READY


@dataclass(frozen=True)
class AgentStateSnapshot:
    """Immutable final view of state returned from an agent run."""

    step_count: int
    visited_urls: tuple[str, ...]
    notes: tuple[str, ...]
    tool_history: tuple[str, ...]
    status: RunStatus

    def __post_init__(self) -> None:
        if type(self.step_count) is not int:
            raise TypeError("step_count must be an integer")
        if self.step_count < 0:
            raise ValueError("step_count must not be negative")

        self._validate_text_tuple(
            self.visited_urls,
            "visited_urls",
            allow_empty=False,
        )
        self._validate_text_tuple(self.notes, "notes", allow_empty=True)
        self._validate_text_tuple(
            self.tool_history,
            "tool_history",
            allow_empty=False,
        )
        if len(self.tool_history) > self.step_count:
            raise ValueError("tool_history cannot exceed step_count")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")

    @staticmethod
    def _validate_text_tuple(
        values: object,
        name: str,
        *,
        allow_empty: bool,
    ) -> None:
        if not isinstance(values, tuple):
            raise TypeError(f"{name} must be a tuple")
        for value in values:
            if not isinstance(value, str):
                raise TypeError(f"{name} values must be strings")
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(
                    f"{name} values contain invalid Unicode"
                ) from error
            if not allow_empty and not value.strip():
                raise ValueError(f"{name} values must not be empty")

    @classmethod
    def from_state(cls, state: AgentState) -> "AgentStateSnapshot":
        """Copy mutable run state into deterministic immutable collections."""

        if not isinstance(state, AgentState):
            raise TypeError("state must be an AgentState")
        return cls(
            step_count=state.step_count,
            visited_urls=tuple(sorted(state.visited_urls)),
            notes=tuple(state.notes),
            tool_history=tuple(state.tool_history),
            status=state.status,
        )
