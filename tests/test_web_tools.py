"""Tests for search and page-reading tool handlers."""

import http.client
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

from research_agent.actions import ToolCall
from research_agent.state import AgentState
from research_agent.tools.core import ToolErrorCode, ToolExecutionError
from research_agent.tools.registry import ToolRegistry
from research_agent.tools.web import (
    BraveSearchBackend,
    BRAVE_SEARCH_ENDPOINT,
    MAX_SEARCH_OUTPUT_CHARS,
    PageDocument,
    SearchResult,
    _NoRedirectHandler,
    make_read_page_handler,
    make_search_web_handler,
)


class FakeApiResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeApiResponse":
        return self

    def __exit__(self, *arguments: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self._body[:size]


class FakeApiOpener:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.request = None
        self.timeout = None

    def open(self, request: object, *, timeout: float) -> FakeApiResponse:
        self.request = request
        self.timeout = timeout
        return FakeApiResponse(self.payload)


class FakeRawApiResponse:
    def __init__(self, body: bytes, *, read_error: Exception | None = None) -> None:
        self._body = body
        self._read_error = read_error

    def __enter__(self) -> "FakeRawApiResponse":
        return self

    def __exit__(self, *arguments: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._body[:size]


class FakeRawApiOpener:
    def __init__(self, response: FakeRawApiResponse) -> None:
        self.response = response
        self.request = None

    def open(self, request: object, *, timeout: float) -> FakeRawApiResponse:
        self.request = request
        return self.response


class RaisingApiOpener:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def open(self, request: object, *, timeout: float) -> FakeApiResponse:
        raise self.error


class SearchWebToolTests(unittest.TestCase):
    def test_searches_with_trimmed_query_and_formats_results(self) -> None:
        calls: list[tuple[str, int]] = []

        def fake_search(query: str, limit: int) -> list[SearchResult]:
            calls.append((query, limit))
            return [
                SearchResult(
                    "Speculative decoding",
                    "https://example.com/paper",
                    "A decoding optimization.",
                )
            ]

        registry = ToolRegistry()
        registry.register("search_web", make_search_web_handler(fake_search))
        result = registry.execute(
            ToolCall(
                "search_web",
                {"query": "  speculative decoding  ", "max_results": 3},
            ),
            AgentState(),
        )

        self.assertEqual(calls, [("speculative decoding", 3)])
        self.assertTrue(result.succeeded)
        self.assertIn("https://example.com/paper", result.output or "")
        self.assertIn("untrusted external content", result.output or "")

    def test_caps_oversized_search_observations(self) -> None:
        def fake_search(query: str, limit: int) -> list[SearchResult]:
            return [
                SearchResult(
                    "T" * 20_000,
                    "https://example.com/" + ("u" * 20_000),
                    "S" * 20_000,
                )
                for _ in range(10)
            ]

        registry = ToolRegistry()
        registry.register("search_web", make_search_web_handler(fake_search))

        result = registry.execute(
            ToolCall("search_web", {"query": "agent harness", "max_results": 10}),
            AgentState(),
        )

        self.assertTrue(result.succeeded)
        self.assertLessEqual(len(result.output or ""), MAX_SEARCH_OUTPUT_CHARS)
        self.assertTrue((result.output or "").endswith("[Tool output truncated]"))

    def test_rejects_invalid_search_arguments_before_calling_backend(self) -> None:
        call_count = 0

        def fake_search(query: str, limit: int) -> list[SearchResult]:
            nonlocal call_count
            call_count += 1
            return []

        registry = ToolRegistry()
        registry.register("search_web", make_search_web_handler(fake_search))
        invalid_calls = (
            ToolCall("search_web"),
            ToolCall("search_web", {"query": "   "}),
            ToolCall("search_web", {"query": "valid", "max_results": 0}),
            ToolCall("search_web", {"query": "valid", "extra": True}),
            ToolCall("search_web", {"query": " ".join(["word"] * 76)}),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                result = registry.execute(call, AgentState())
                self.assertEqual(
                    result.error_code,
                    ToolErrorCode.INVALID_ARGUMENTS,
                )

        self.assertEqual(call_count, 0)

    def test_brave_backend_maps_the_documented_response_shape(self) -> None:
        opener = FakeApiOpener(
            {
                "web": {
                    "results": [
                        {
                            "title": "Result title",
                            "url": "https://example.com/result",
                            "description": "Result description",
                        }
                    ]
                }
            }
        )
        backend = BraveSearchBackend(
            "test-key",
            opener=opener,
            timeout_seconds=7,
        )

        results = backend("agent harness", 4)

        self.assertEqual(results[0].title, "Result title")
        self.assertEqual(opener.timeout, 7)
        request = opener.request
        self.assertIsNotNone(request)
        query = parse_qs(urlsplit(request.full_url).query)  # type: ignore[union-attr]
        headers = {
            name.lower(): value
            for name, value in request.header_items()  # type: ignore[union-attr]
        }
        self.assertEqual(query["q"], ["agent harness"])
        self.assertEqual(query["count"], ["4"])
        self.assertEqual(headers["x-subscription-token"], "test-key")

    def test_brave_backend_requires_configuration(self) -> None:
        backend = BraveSearchBackend(api_key="")

        with self.assertRaisesRegex(ToolExecutionError, "BRAVE_SEARCH_API_KEY"):
            backend("agent harness", 5)

    def test_brave_backend_rejects_blank_or_unsafe_keys(self) -> None:
        for api_key in ("   ", "bad\nkey", "emoji-🔑"):
            with self.subTest(api_key=api_key):
                opener = FakeApiOpener({})
                backend = BraveSearchBackend(api_key=api_key, opener=opener)

                with self.assertRaisesRegex(
                    ToolExecutionError,
                    "BRAVE_SEARCH_API_KEY",
                ):
                    backend("agent harness", 5)

                self.assertIsNone(opener.request)

    def test_brave_backend_reads_key_from_environment(self) -> None:
        opener = FakeApiOpener({"web": {"results": []}})
        with patch.dict(
            os.environ,
            {"BRAVE_SEARCH_API_KEY": "environment-key"},
            clear=True,
        ):
            backend = BraveSearchBackend(opener=opener)

        backend("agent harness", 5)

        headers = {
            name.lower(): value
            for name, value in opener.request.header_items()  # type: ignore[union-attr]
        }
        self.assertEqual(headers["x-subscription-token"], "environment-key")

    def test_brave_redirect_handler_never_forwards_the_key(self) -> None:
        request = Request(
            BRAVE_SEARCH_ENDPOINT,
            headers={"X-Subscription-Token": "secret"},
        )

        redirected_request = _NoRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://attacker.example/",
        )

        self.assertIsNone(redirected_request)

    def test_brave_backend_converts_truncated_response_to_safe_error(self) -> None:
        opener = FakeRawApiOpener(
            FakeRawApiResponse(
                b"",
                read_error=http.client.IncompleteRead(b"{}", 10),
            )
        )
        backend = BraveSearchBackend("test-key", opener=opener)

        with self.assertRaisesRegex(ToolExecutionError, "request failed"):
            backend("agent harness", 5)

    def test_brave_backend_rejects_oversized_or_invalid_data(self) -> None:
        responses = (
            FakeRawApiResponse(b"x" * 1_000_001),
            FakeRawApiResponse(b"not json"),
            FakeRawApiResponse(
                (b"[" * 2_000) + b"0" + (b"]" * 2_000)
            ),
        )

        for response in responses:
            with self.subTest(response=response):
                backend = BraveSearchBackend(
                    "test-key",
                    opener=FakeRawApiOpener(response),
                )

                with self.assertRaises(ToolExecutionError):
                    backend("agent harness", 5)

    def test_brave_backend_converts_http_and_network_failures(self) -> None:
        cases = (
            (
                HTTPError(
                    "https://api.search.brave.com",
                    429,
                    "rate limited",
                    {},
                    None,
                ),
                "HTTP 429",
            ),
            (URLError("dns"), "request failed"),
            (http.client.RemoteDisconnected("closed"), "request failed"),
            (TimeoutError("slow"), "request failed"),
        )

        for error, message in cases:
            with self.subTest(error=error):
                backend = BraveSearchBackend(
                    "test-key",
                    opener=RaisingApiOpener(error),
                )

                with self.assertRaisesRegex(ToolExecutionError, message):
                    backend("agent harness", 5)


class ReadPageToolTests(unittest.TestCase):
    def test_reads_page_and_records_only_the_final_url(self) -> None:
        calls: list[tuple[str, int]] = []

        def fake_reader(url: str, max_chars: int) -> PageDocument:
            calls.append((url, max_chars))
            return PageDocument(
                "https://example.com/final",
                "Example",
                "Useful page content.",
            )

        registry = ToolRegistry()
        registry.register("read_page", make_read_page_handler(fake_reader))
        state = AgentState()
        result = registry.execute(
            ToolCall(
                "read_page",
                {"url": "https://example.com/start", "max_chars": 2_000},
            ),
            state,
        )

        self.assertEqual(calls, [("https://example.com/start", 2_000)])
        self.assertEqual(state.visited_urls, {"https://example.com/final"})
        self.assertTrue(result.succeeded)
        self.assertIn("not instructions", result.output or "")

    def test_failed_page_read_does_not_mark_url_as_visited(self) -> None:
        def failing_reader(url: str, max_chars: int) -> PageDocument:
            raise ToolExecutionError("Page request failed.")

        registry = ToolRegistry()
        registry.register("read_page", make_read_page_handler(failing_reader))
        state = AgentState()

        result = registry.execute(
            ToolCall("read_page", {"url": "https://example.com"}),
            state,
        )

        self.assertEqual(result.error_code, ToolErrorCode.EXECUTION_FAILED)
        self.assertEqual(state.visited_urls, set())

    def test_caps_the_entire_page_observation_to_max_chars(self) -> None:
        def fake_reader(url: str, max_chars: int) -> PageDocument:
            final_url = "https://example.com/" + ("u" * 2_000)
            return PageDocument(final_url, "T" * 1_000, "body " * 1_000)

        registry = ToolRegistry()
        registry.register("read_page", make_read_page_handler(fake_reader))
        result = registry.execute(
            ToolCall(
                "read_page",
                {"url": "https://example.com", "max_chars": 500},
            ),
            AgentState(),
        )

        self.assertTrue(result.succeeded)
        self.assertLessEqual(len(result.output or ""), 500)
        self.assertTrue(
            (result.output or "").startswith("Untrusted webpage content")
        )
        self.assertTrue((result.output or "").endswith("[Tool output truncated]"))


if __name__ == "__main__":
    unittest.main()
