"""Explicit allow-list and dispatcher for executable tools."""

from research_agent.actions import MAX_TOOL_NAME_CHARACTERS, ToolCall
from research_agent.state import AgentState
from research_agent.tools.core import (
    ToolErrorCode,
    ToolExecutionError,
    ToolHandler,
    ToolInputError,
    ToolResult,
)


MAX_TOOL_OUTPUT_CHARACTERS = 50_000
MAX_REGISTERED_TOOLS = 100


def _bounded_tool_text(value: str, marker: str) -> str:
    safe_value = value.encode("utf-8", errors="replace").decode("utf-8")
    if len(safe_value) <= MAX_TOOL_OUTPUT_CHARACTERS:
        return safe_value
    available_characters = MAX_TOOL_OUTPUT_CHARACTERS - len(marker)
    return safe_value[:available_characters].rstrip() + marker


class ToolRegistry:
    """Register known tools and safely dispatch validated tool calls."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered tool names in deterministic order."""

        return tuple(sorted(self._handlers))

    def register(self, name: str, handler: ToolHandler) -> None:
        """Add one handler to the allow-list."""

        if not isinstance(name, str):
            raise TypeError("tool name must be a string")
        if not name.strip():
            raise ValueError("tool name must be a non-empty string")
        cleaned_name = name.strip()
        try:
            cleaned_name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("tool name contains invalid Unicode") from error
        if len(cleaned_name) > MAX_TOOL_NAME_CHARACTERS:
            raise ValueError(
                "tool name must be at most "
                f"{MAX_TOOL_NAME_CHARACTERS} characters"
            )
        if not callable(handler):
            raise TypeError("tool handler must be callable")

        if cleaned_name in self._handlers:
            raise ValueError(f"tool already registered: {cleaned_name}")
        if len(self._handlers) >= MAX_REGISTERED_TOOLS:
            raise ValueError(
                f"at most {MAX_REGISTERED_TOOLS} tools may be registered"
            )
        self._handlers[cleaned_name] = handler

    def execute(self, call: ToolCall, state: AgentState) -> ToolResult:
        """Dispatch a call and translate expected failures into observations."""

        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolResult.failure(
                call.name,
                ToolErrorCode.UNKNOWN_TOOL,
                f"unknown tool: {call.name}",
            )

        try:
            output = handler(call.arguments, state)
        except ToolInputError as error:
            message = str(error).strip() or "Tool arguments were invalid."
            return ToolResult.failure(
                call.name,
                ToolErrorCode.INVALID_ARGUMENTS,
                _bounded_tool_text(message, "\n[Tool error truncated]"),
            )
        except ToolExecutionError as error:
            message = str(error).strip() or "Tool execution failed."
            return ToolResult.failure(
                call.name,
                ToolErrorCode.EXECUTION_FAILED,
                _bounded_tool_text(message, "\n[Tool error truncated]"),
            )

        if not isinstance(output, str):
            raise TypeError("tool handlers must return strings")
        output = _bounded_tool_text(
            output,
            "\n[Tool output truncated by registry]",
        )
        return ToolResult.success(call.name, output)
