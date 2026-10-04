"""Command-line interface for the deterministic research-agent demo."""

import argparse
import sys
import time
from collections.abc import Callable, Sequence

from research_agent.actions import FinishAction, ToolCall
from research_agent.config import (
    MAX_OBJECTIVE_CHARACTERS,
    AgentConfig,
    ResearchRequest,
)
from research_agent.model import AgentModel, ScriptedModel
from research_agent.orchestrator import RunResult, run_agent
from research_agent.tools import ToolRegistry, build_default_registry


ModelFactory = Callable[[ResearchRequest], AgentModel]
RegistryFactory = Callable[[], ToolRegistry]

_BIDI_CONTROL_CODE_POINTS = frozenset(
    {
        0x061C,
        0x200E,
        0x200F,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    }
)


def _terminal_safe_text(value: str) -> str:
    safe_characters: list[str] = []
    for character in value:
        code_point = ord(character)
        if character == "\n":
            safe_characters.append(character)
        elif (
            code_point < 32
            or 127 <= code_point <= 159
            or code_point in _BIDI_CONTROL_CODE_POINTS
        ):
            if code_point <= 255:
                safe_characters.append(f"\\x{code_point:02x}")
            elif code_point <= 65_535:
                safe_characters.append(f"\\u{code_point:04x}")
            else:
                safe_characters.append(f"\\U{code_point:08x}")
        else:
            safe_characters.append(character)
    return "".join(safe_characters)


def non_empty_text(value: str) -> str:
    """Trim a command-line value and reject blank text."""

    cleaned_value = value.strip()
    if not cleaned_value:
        raise argparse.ArgumentTypeError("must not be empty")
    return cleaned_value


def objective_text(value: str) -> str:
    """Validate an objective before it enters model context or notes."""

    cleaned_value = non_empty_text(value)
    try:
        cleaned_value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise argparse.ArgumentTypeError(
            "contains invalid Unicode"
        ) from error
    if len(cleaned_value) > MAX_OBJECTIVE_CHARACTERS:
        raise argparse.ArgumentTypeError(
            f"must be at most {MAX_OBJECTIVE_CHARACTERS} characters"
        )
    return cleaned_value


def step_limit(value: str) -> int:
    """Convert a command-line value to a safe agent step limit."""

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= parsed_value <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return parsed_value


def timeout_limit(value: str) -> int:
    """Convert a command-line value to a safe timeout in seconds."""

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= parsed_value <= 3600:
        raise argparse.ArgumentTypeError("must be between 1 and 3600")
    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    """Create the parser that defines the public CLI contract."""

    parser = argparse.ArgumentParser(
        prog="python3 -m research_agent",
        description="Run the offline scripted research-agent demo.",
    )
    parser.add_argument(
        "objective",
        type=objective_text,
        help=(
            "the objective used by the offline demo, for example "
            '"speculative decoding"'
        ),
    )
    parser.add_argument(
        "--model",
        type=non_empty_text,
        choices=("scripted",),
        default="scripted",
        help="offline deterministic stand-in for a future LLM (default: scripted)",
    )
    parser.add_argument(
        "--max-steps",
        type=step_limit,
        default=8,
        help="maximum number of model actions (default: 8)",
    )
    parser.add_argument(
        "--timeout-seconds",
        dest="timeout_seconds",
        type=timeout_limit,
        default=120,
        help="run timeout checked between model and tool calls (default: 120)",
    )
    return parser


def parse_request(arguments: Sequence[str] | None = None) -> ResearchRequest:
    """Parse command-line arguments into validated domain objects."""

    parsed = build_parser().parse_args(arguments)
    config = AgentConfig(
        model=parsed.model,
        max_steps=parsed.max_steps,
        timeout_seconds=parsed.timeout_seconds,
    )
    return ResearchRequest(objective=parsed.objective, config=config)


def build_scripted_model(request: ResearchRequest) -> ScriptedModel:
    """Build the honest, network-free action sequence used by the CLI."""

    return ScriptedModel(
        (
            ToolCall(
                "save_note",
                {"note": f"Demo objective: {request.objective}"},
            ),
            ToolCall("read_notes"),
            FinishAction(
                answer=(
                    f'Scripted demo completed for: "{request.objective}".\n'
                    "No external research was performed."
                )
            ),
        )
    )


def render_request(request: ResearchRequest) -> str:
    """Render the stable header shown before a run trace."""

    return "\n".join(
        (
            "Research Agent — scripted offline demo",
            f"Objective: {request.objective}",
            f"Model: {request.config.model}",
            (
                "Limits: "
                f"{request.config.max_steps} steps, "
                f"{request.config.timeout_seconds} seconds"
            ),
            "Network/API: not used by this script",
        )
    )


def _append_indented(lines: list[str], value: str) -> None:
    for line in value.splitlines() or [""]:
        lines.append(f"    {line}")


def render_run_result(
    request: ResearchRequest,
    result: RunResult,
) -> str:
    """Render an educational trace without mixing printing into the loop."""

    lines = [render_request(request), ""]
    for entry in result.history:
        lines.extend(
            (
                f"Step {entry.step_number}/{request.config.max_steps}",
                f"  model -> tool: {entry.tool_name}",
                f"  arguments: {entry.arguments_display}",
            )
        )
        if entry.result.succeeded:
            lines.append("  tool -> success:")
            _append_indented(lines, entry.result.output or "")
        else:
            error_code = entry.result.error_code
            code_text = error_code.value if error_code is not None else "unknown"
            lines.append(f"  tool -> failure ({code_text}):")
            _append_indented(lines, entry.result.error_message or "Unknown error.")
        lines.append("")

    if result.succeeded:
        lines.extend(
            (
                f"Step {result.state.step_count}/{request.config.max_steps}",
                "  model -> finish",
                "",
                "Final answer",
            )
        )
        lines.extend((result.answer or "").splitlines())
    else:
        lines.extend(("Run stopped", result.error_message or "Unknown error."))

    lines.extend(
        (
            "",
            (
                f"Run summary: {result.state.status.value} | "
                f"{result.state.step_count} steps | "
                f"{len(result.history)} tool calls"
            ),
        )
    )
    return _terminal_safe_text("\n".join(lines))


def main(
    arguments: Sequence[str] | None = None,
    *,
    model_factory: ModelFactory | None = None,
    registry_factory: RegistryFactory | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Parse a request, run the scripted agent, and print its trace."""

    request = parse_request(arguments)
    selected_model_factory = model_factory or build_scripted_model
    selected_registry_factory = registry_factory or build_default_registry

    try:
        result = run_agent(
            request,
            model=selected_model_factory(request),
            tools=selected_registry_factory(),
            clock=clock,
        )
    except KeyboardInterrupt:
        print("Research agent interrupted.", file=sys.stderr)
        return 130

    output_stream = sys.stdout if result.succeeded else sys.stderr
    print(render_run_result(request, result), file=output_stream)
    return 0 if result.succeeded else 1
