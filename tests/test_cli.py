"""Tests for the first command-line milestone."""

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

from research_agent.cli import main, parse_request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ParseRequestTests(unittest.TestCase):
    def test_uses_beginner_friendly_defaults(self) -> None:
        request = parse_request(["Research speculative decoding"])

        self.assertEqual(request.objective, "Research speculative decoding")
        self.assertEqual(request.config.model, "scripted")
        self.assertEqual(request.config.max_steps, 8)
        self.assertEqual(request.config.timeout_seconds, 120)

    def test_accepts_explicit_run_configuration(self) -> None:
        request = parse_request(
            [
                "Research tool-calling agents",
                "--model",
                "scripted",
                "--max-steps",
                "4",
                "--timeout-seconds",
                "30",
            ]
        )

        self.assertEqual(request.config.model, "scripted")
        self.assertEqual(request.config.max_steps, 4)
        self.assertEqual(request.config.timeout_seconds, 30)

    def test_trims_the_objective_and_preserves_its_wording(self) -> None:
        request = parse_request(["  How do agents use tools? 🤖  "])

        self.assertEqual(request.objective, "How do agents use tools? 🤖")

    def test_rejects_blank_objective_and_model(self) -> None:
        invalid_arguments = (["   "], ["Research agents", "--model", "   "])

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parse_request(arguments)

                self.assertEqual(error.exception.code, 2)

    def test_rejects_a_model_that_is_not_implemented_yet(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_request(["Research agents", "--model", "live-model"])

        self.assertEqual(error.exception.code, 2)

    def test_rejects_non_positive_limits(self) -> None:
        invalid_arguments = (
            ["Research agents", "--max-steps", "0"],
            ["Research agents", "--timeout-seconds", "0"],
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parse_request(arguments)

                self.assertEqual(error.exception.code, 2)

    def test_rejects_limits_above_the_safe_range(self) -> None:
        invalid_arguments = (
            ["Research agents", "--max-steps", "101"],
            ["Research agents", "--timeout-seconds", "3601"],
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parse_request(arguments)

                self.assertEqual(error.exception.code, 2)

    def test_rejects_non_integer_limits(self) -> None:
        invalid_arguments = (
            ["Research agents", "--max-steps", "1.5"],
            ["Research agents", "--timeout-seconds", "ten"],
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parse_request(arguments)

                self.assertEqual(error.exception.code, 2)


class MainTests(unittest.TestCase):
    def test_runs_the_scripted_agent_and_prints_the_trace(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main(["  Research speculative decoding  "])

        self.assertEqual(exit_code, 0)
        rendered = output.getvalue()
        self.assertIn("Objective: Research speculative decoding", rendered)
        self.assertIn("model -> tool: save_note", rendered)
        self.assertIn("model -> tool: read_notes", rendered)
        self.assertIn("model -> finish", rendered)
        self.assertIn("No external research was performed.", rendered)
        self.assertIn("Run summary: finished | 3 steps | 2 tool calls", rendered)

    def test_returns_one_when_the_script_hits_the_step_limit(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stderr(output):
            exit_code = main(
                ["Research speculative decoding", "--max-steps", "2"]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("max_steps=2", output.getvalue())
        self.assertNotIn("Final answer", output.getvalue())

    def test_escapes_terminal_control_characters(self) -> None:
        output = io.StringIO()
        objective = "safe\x1b[2Jspoof"

        with contextlib.redirect_stdout(output):
            exit_code = main([objective])

        self.assertEqual(exit_code, 0)
        self.assertNotIn("\x1b", output.getvalue())
        self.assertIn("\\x1b[2J", output.getvalue())

    def test_preserves_joiners_but_escapes_bidi_controls(self) -> None:
        output = io.StringIO()
        objective = "Emoji 👩‍💻 and Persian می‌روم \u202espoof"

        with contextlib.redirect_stdout(output):
            exit_code = main([objective])

        self.assertEqual(exit_code, 0)
        rendered = output.getvalue()
        self.assertIn("👩‍💻", rendered)
        self.assertIn("می‌روم", rendered)
        self.assertNotIn("\u202e", rendered)
        self.assertIn("\\u202e", rendered)

    def test_package_module_exposes_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "research_agent", "--help"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("python3 -m research_agent", completed.stdout)
        self.assertIn("--max-steps", completed.stdout)
        self.assertIn("offline deterministic", completed.stdout)


if __name__ == "__main__":
    unittest.main()
