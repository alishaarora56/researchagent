"""Shared contracts and expected errors for executable tools."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from research_agent.state import AgentState


class ToolErrorCode(str, Enum):
    """Stable error categories that the agent loop can understand."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True)
class ToolResult:
    """A successful observation or an expected, model-correctable failure."""

    tool_name: str
    output: str | None = None
    error_code: ToolErrorCode | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str):
            raise TypeError("tool_name must be a string")
        if not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        object.__setattr__(self, "tool_name", self.tool_name.strip())

        if self.error_code is None:
            if not isinstance(self.output, str):
                raise ValueError("a successful result requires string output")
            if self.error_message is not None:
                raise ValueError("a successful result cannot have an error message")
            return

        if not isinstance(self.error_code, ToolErrorCode):
            raise TypeError("error_code must be a ToolErrorCode")
        if self.output is not None:
            raise ValueError("a failed result cannot have output")
        if not isinstance(self.error_message, str) or not self.error_message.strip():
            raise ValueError("a failed result requires an error message")
        object.__setattr__(self, "error_message", self.error_message.strip())

    @property
    def succeeded(self) -> bool:
        """Return whether the tool completed successfully."""

        return self.error_code is None

    @classmethod
    def success(cls, tool_name: str, output: str) -> "ToolResult":
        """Construct a successful tool result."""

        return cls(tool_name=tool_name, output=output)

    @classmethod
    def failure(
        cls,
        tool_name: str,
        error_code: ToolErrorCode,
        error_message: str,
    ) -> "ToolResult":
        """Construct an expected tool failure safe to expose to a model."""

        return cls(
            tool_name=tool_name,
            error_code=error_code,
            error_message=error_message,
        )


class ToolInputError(ValueError):
    """The model supplied missing, unexpected, or invalid arguments."""


class ToolExecutionError(RuntimeError):
    """An expected external failure prevented a valid tool from completing."""


ToolHandler: TypeAlias = Callable[[Mapping[str, object], AgentState], str]
