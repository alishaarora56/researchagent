"""Actions a model may propose to the agent harness."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias


MAX_TOOL_NAME_CHARACTERS = 100
MAX_FINAL_ANSWER_CHARACTERS = 50_000
MAX_TOOL_ARGUMENT_DEPTH = 20
MAX_TOOL_ARGUMENT_ITEMS = 1_000
MAX_TOOL_ARGUMENT_TEXT_CHARACTERS = 50_000


def _freeze_arguments(arguments: Mapping[object, object]) -> Mapping[str, object]:
    item_count = 0
    text_characters = 0

    def freeze(value: object, depth: int) -> object:
        nonlocal item_count, text_characters
        item_count += 1
        if item_count > MAX_TOOL_ARGUMENT_ITEMS:
            raise ValueError(
                "tool arguments contain too many nested values"
            )
        if depth > MAX_TOOL_ARGUMENT_DEPTH:
            raise ValueError("tool arguments are nested too deeply")

        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("tool argument numbers must be finite")
            return value
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(
                    "tool argument text contains invalid Unicode"
                ) from error
            text_characters += len(value)
            if text_characters > MAX_TOOL_ARGUMENT_TEXT_CHARACTERS:
                raise ValueError("tool argument text is too large")
            return value
        if isinstance(value, Mapping):
            frozen_mapping: dict[str, object] = {}
            for key, nested_value in value.items():
                if not isinstance(key, str):
                    raise TypeError("tool argument names must be strings")
                frozen_key = freeze(key, depth + 1)
                frozen_mapping[str(frozen_key)] = freeze(
                    nested_value,
                    depth + 1,
                )
            return MappingProxyType(frozen_mapping)
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item, depth + 1) for item in value)
        raise TypeError("tool argument values must be JSON-compatible")

    frozen = freeze(arguments, 0)
    if not isinstance(frozen, Mapping):
        raise TypeError("tool arguments must be a mapping")
    return frozen


@dataclass(frozen=True)
class ToolCall:
    """An immutable request to run one tool with structured arguments.

    This boundary validates the action's general shape. The tool registry then
    validates which argument names and values each individual tool accepts.
    """

    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("tool name must be a string")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")

        cleaned_name = self.name.strip()
        if not cleaned_name:
            raise ValueError("tool name must not be empty")
        try:
            cleaned_name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("tool name contains invalid Unicode") from error
        if len(cleaned_name) > MAX_TOOL_NAME_CHARACTERS:
            raise ValueError(
                "tool name must be at most "
                f"{MAX_TOOL_NAME_CHARACTERS} characters"
            )

        object.__setattr__(self, "name", cleaned_name)
        object.__setattr__(
            self,
            "arguments",
            _freeze_arguments(self.arguments),
        )


@dataclass(frozen=True)
class FinishAction:
    """A model decision to stop the loop and return a final answer."""

    answer: str

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise TypeError("final answer must be a string")

        cleaned_answer = self.answer.strip()
        if not cleaned_answer:
            raise ValueError("final answer must not be empty")
        try:
            cleaned_answer.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("final answer contains invalid Unicode") from error
        if len(cleaned_answer) > MAX_FINAL_ANSWER_CHARACTERS:
            raise ValueError(
                "final answer must be at most "
                f"{MAX_FINAL_ANSWER_CHARACTERS} characters"
            )

        object.__setattr__(self, "answer", cleaned_answer)


AgentAction: TypeAlias = ToolCall | FinishAction
