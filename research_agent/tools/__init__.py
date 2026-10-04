"""Composition helpers for the research agent's built-in tools."""

from research_agent.tools.core import (
    ToolErrorCode,
    ToolExecutionError,
    ToolHandler,
    ToolInputError,
    ToolResult,
)
from research_agent.tools.notes import read_notes, save_note
from research_agent.tools.registry import ToolRegistry
from research_agent.tools.web import (
    BraveSearchBackend,
    HttpPageBackend,
    PageBackend,
    PageDocument,
    SearchBackend,
    SearchResult,
    make_read_page_handler,
    make_search_web_handler,
)


def build_default_registry(
    *,
    search_backend: SearchBackend | None = None,
    page_backend: PageBackend | None = None,
) -> ToolRegistry:
    """Build the four-tool registry with injectable web dependencies."""

    selected_search_backend = (
        BraveSearchBackend() if search_backend is None else search_backend
    )
    selected_page_backend = (
        HttpPageBackend() if page_backend is None else page_backend
    )

    registry = ToolRegistry()
    registry.register("save_note", save_note)
    registry.register("read_notes", read_notes)
    registry.register(
        "search_web",
        make_search_web_handler(selected_search_backend),
    )
    registry.register(
        "read_page",
        make_read_page_handler(selected_page_backend),
    )
    return registry


__all__ = [
    "BraveSearchBackend",
    "HttpPageBackend",
    "PageBackend",
    "PageDocument",
    "SearchBackend",
    "SearchResult",
    "ToolErrorCode",
    "ToolExecutionError",
    "ToolHandler",
    "ToolInputError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
]
