"""Small argument-validation helpers shared by tool handlers."""

from collections.abc import Mapping

from research_agent.tools.core import ToolInputError


def require_exact_arguments(
    arguments: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    """Require exactly the declared argument names."""

    optional_names = optional or set()
    non_string_names = [name for name in arguments if not isinstance(name, str)]
    if non_string_names:
        raise ToolInputError("argument names must be strings")

    supplied = set(arguments)
    missing = required - supplied
    unexpected = supplied - required - optional_names

    if missing:
        raise ToolInputError(f"missing required argument: {sorted(missing)[0]}")
    if unexpected:
        raise ToolInputError(f"unexpected argument: {sorted(unexpected)[0]}")


def require_text(
    arguments: Mapping[str, object],
    name: str,
    *,
    max_length: int,
) -> str:
    """Return one trimmed, non-empty string argument within a size limit."""

    value = arguments[name]
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be a string")

    cleaned = value.strip()
    if not cleaned:
        raise ToolInputError(f"{name} must not be empty")
    try:
        cleaned.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ToolInputError(f"{name} contains invalid Unicode") from error
    if len(cleaned) > max_length:
        raise ToolInputError(f"{name} must be at most {max_length} characters")
    return cleaned


def optional_integer(
    arguments: Mapping[str, object],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Return an optional integer constrained to a closed range."""

    value = arguments.get(name, default)
    if type(value) is not int:
        raise ToolInputError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ToolInputError(f"{name} must be between {minimum} and {maximum}")
    return value
