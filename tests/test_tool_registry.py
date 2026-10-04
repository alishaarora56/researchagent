"""Tests for structured tool results and registry dispatch."""

import unittest

import research_agent.tools as public_tools
from research_agent.actions import ToolCall
from research_agent.state import AgentState
from research_agent.tools import build_default_registry
from research_agent.tools.core import (
    ToolErrorCode,
    ToolExecutionError,
    ToolInputError,
    ToolResult,
)
from research_agent.tools.registry import ToolRegistry
from research_agent.tools.web import PageDocument, SearchResult


class ToolResultTests(unittest.TestCase):
    def test_success_and_failure_have_distinct_shapes(self) -> None:
        success = ToolResult.success("read_notes", "No notes saved.")
        failure = ToolResult.failure(
            "missing_tool",
            ToolErrorCode.UNKNOWN_TOOL,
            "unknown tool: missing_tool",
        )

        self.assertTrue(success.succeeded)
        self.assertEqual(success.output, "No notes saved.")
        self.assertFalse(failure.succeeded)
        self.assertEqual(failure.error_code, ToolErrorCode.UNKNOWN_TOOL)

    def test_rejects_an_inconsistent_result(self) -> None:
        with self.assertRaises(ValueError):
            ToolResult(tool_name="read_notes")

        with self.assertRaises(ValueError):
            ToolResult(
                tool_name="read_notes",
                output="unexpected output",
                error_code=ToolErrorCode.EXECUTION_FAILED,
                error_message="failed",
            )

    def test_uses_type_error_for_a_non_string_tool_name(self) -> None:
        with self.assertRaises(TypeError):
            ToolResult.success(123, "output")  # type: ignore[arg-type]


class ToolRegistryTests(unittest.TestCase):
    def test_default_registry_contains_all_four_tools(self) -> None:
        registry = build_default_registry(
            search_backend=lambda query, limit: [
                SearchResult("Title", "https://example.com")
            ],
            page_backend=lambda url, limit: PageDocument(
                url,
                "Title",
                "Content",
            ),
        )

        self.assertEqual(
            registry.names,
            ("read_notes", "read_page", "save_note", "search_web"),
        )

    def test_dispatches_a_registered_tool(self) -> None:
        registry = ToolRegistry()
        state = AgentState()

        def echo(arguments: object, received_state: AgentState) -> str:
            self.assertIs(received_state, state)
            return str(arguments)

        registry.register("echo", echo)
        result = registry.execute(
            ToolCall("echo", {"message": "hello"}),
            state,
        )

        self.assertTrue(result.succeeded)
        self.assertIn("hello", result.output or "")

    def test_caps_output_from_every_registered_tool(self) -> None:
        registry = ToolRegistry()
        registry.register("large", lambda arguments, state: "x" * 60_000)

        result = registry.execute(ToolCall("large"), AgentState())

        self.assertTrue(result.succeeded)
        self.assertLessEqual(len(result.output or ""), 50_000)
        self.assertTrue(
            (result.output or "").endswith(
                "[Tool output truncated by registry]"
            )
        )

    def test_caps_expected_error_messages(self) -> None:
        registry = ToolRegistry()

        def fail(arguments: object, state: AgentState) -> str:
            raise ToolExecutionError("x" * 60_000)

        registry.register("fail", fail)
        result = registry.execute(ToolCall("fail"), AgentState())

        self.assertEqual(result.error_code, ToolErrorCode.EXECUTION_FAILED)
        self.assertLessEqual(len(result.error_message or ""), 50_000)
        self.assertTrue(
            (result.error_message or "").endswith("[Tool error truncated]")
        )

    def test_sanitizes_invalid_unicode_from_custom_tool_output(self) -> None:
        registry = ToolRegistry()
        registry.register("unsafe", lambda arguments, state: "bad \ud800")

        result = registry.execute(ToolCall("unsafe"), AgentState())

        (result.output or "").encode("utf-8")

    def test_unknown_tool_returns_a_failure(self) -> None:
        result = ToolRegistry().execute(ToolCall("unknown"), AgentState())

        self.assertEqual(result.error_code, ToolErrorCode.UNKNOWN_TOOL)

    def test_expected_handler_errors_become_failures(self) -> None:
        cases = (
            (
                ToolInputError("bad input"),
                ToolErrorCode.INVALID_ARGUMENTS,
            ),
            (
                ToolExecutionError("service unavailable"),
                ToolErrorCode.EXECUTION_FAILED,
            ),
            (
                ToolInputError(),
                ToolErrorCode.INVALID_ARGUMENTS,
            ),
            (
                ToolExecutionError(),
                ToolErrorCode.EXECUTION_FAILED,
            ),
        )

        for raised_error, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                registry = ToolRegistry()

                def fail(arguments: object, state: AgentState) -> str:
                    raise raised_error

                registry.register("fail", fail)
                result = registry.execute(ToolCall("fail"), AgentState())

                self.assertEqual(result.error_code, expected_code)
                self.assertTrue(result.error_message)

    def test_unexpected_handler_errors_propagate(self) -> None:
        registry = ToolRegistry()

        def broken(arguments: object, state: AgentState) -> str:
            raise RuntimeError("programming bug")

        registry.register("broken", broken)

        with self.assertRaisesRegex(RuntimeError, "programming bug"):
            registry.execute(ToolCall("broken"), AgentState())

    def test_rejects_duplicate_registration(self) -> None:
        registry = ToolRegistry()
        registry.register("echo", lambda arguments, state: "ok")

        with self.assertRaises(ValueError):
            registry.register("echo", lambda arguments, state: "also ok")

    def test_caps_the_number_of_registered_tools(self) -> None:
        registry = ToolRegistry()
        for index in range(100):
            registry.register(
                f"tool_{index}",
                lambda arguments, state: "ok",
            )

        with self.assertRaises(ValueError):
            registry.register("one_too_many", lambda arguments, state: "ok")

    def test_uses_type_error_for_a_non_string_registration_name(self) -> None:
        with self.assertRaises(TypeError):
            ToolRegistry().register(  # type: ignore[arg-type]
                123,
                lambda arguments, state: "ok",
            )

    def test_public_package_exports_extension_contracts(self) -> None:
        expected_names = (
            "PageBackend",
            "PageDocument",
            "SearchBackend",
            "SearchResult",
            "ToolExecutionError",
            "ToolHandler",
            "ToolInputError",
        )

        for name in expected_names:
            with self.subTest(name=name):
                self.assertIn(name, public_tools.__all__)
                self.assertTrue(hasattr(public_tools, name))


if __name__ == "__main__":
    unittest.main()
