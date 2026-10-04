"""Local note tools backed by the current AgentState."""

from collections.abc import Mapping

from research_agent.state import AgentState
from research_agent.tools.core import ToolInputError
from research_agent.tools.validation import require_exact_arguments, require_text


MAX_TOTAL_NOTE_CHARACTERS = 40_000
MAX_NOTES_OUTPUT_CHARACTERS = 50_000


def save_note(arguments: Mapping[str, object], state: AgentState) -> str:
    """Validate and append one useful research note."""

    require_exact_arguments(arguments, required={"note"})
    note = require_text(arguments, "note", max_length=10_000)
    total_characters = sum(len(existing_note) for existing_note in state.notes)
    if total_characters + len(note) > MAX_TOTAL_NOTE_CHARACTERS:
        raise ToolInputError(
            "saved notes may contain at most "
            f"{MAX_TOTAL_NOTE_CHARACTERS} characters in total"
        )
    state.notes.append(note)
    return f"Saved note {len(state.notes)}."


def read_notes(arguments: Mapping[str, object], state: AgentState) -> str:
    """Return the notes accumulated so far in a model-readable format."""

    require_exact_arguments(arguments, required=set())
    if not state.notes:
        return "No notes saved."

    rendered_notes: list[str] = []
    for index, note in enumerate(state.notes, start=1):
        first_line, *continuation_lines = note.splitlines()
        rendered_notes.append(f"{index}. {first_line}")
        rendered_notes.extend(
            f"   | {line}"
            for line in continuation_lines
        )
    rendered = "\n".join(rendered_notes)
    if len(rendered) <= MAX_NOTES_OUTPUT_CHARACTERS:
        return rendered

    marker = "\n[Notes output truncated]"
    available_characters = MAX_NOTES_OUTPUT_CHARACTERS - len(marker)
    return rendered[:available_characters].rstrip() + marker
