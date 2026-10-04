"""Tests for validated research request types."""

import unittest
from dataclasses import FrozenInstanceError

from research_agent.config import (
    MAX_OBJECTIVE_CHARACTERS,
    AgentConfig,
    ResearchRequest,
)


class AgentConfigTests(unittest.TestCase):
    def test_uses_the_scripted_model_by_default(self) -> None:
        self.assertEqual(AgentConfig().model, "scripted")

    def test_trims_the_model_name(self) -> None:
        config = AgentConfig(model="  mock  ")

        self.assertEqual(config.model, "mock")

    def test_rejects_non_integer_limits(self) -> None:
        invalid_values = (
            {"max_steps": 1.5},
            {"max_steps": True},
            {"timeout_seconds": 2.5},
            {"timeout_seconds": False},
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    AgentConfig(**values)

    def test_rejects_limits_outside_the_safe_range(self) -> None:
        invalid_values = (
            {"max_steps": 0},
            {"max_steps": 101},
            {"timeout_seconds": 0},
            {"timeout_seconds": 3601},
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    AgentConfig(**values)

    def test_validated_configuration_is_immutable(self) -> None:
        config = AgentConfig()

        with self.assertRaises(FrozenInstanceError):
            config.max_steps = 999  # type: ignore[misc]


class ResearchRequestTests(unittest.TestCase):
    def test_rejects_a_blank_objective(self) -> None:
        with self.assertRaises(ValueError):
            ResearchRequest(objective="   ", config=AgentConfig())

    def test_rejects_oversized_or_invalid_unicode_objective(self) -> None:
        invalid_objectives = (
            "x" * (MAX_OBJECTIVE_CHARACTERS + 1),
            "invalid \ud800",
        )

        for objective in invalid_objectives:
            with self.subTest(objective=repr(objective)):
                with self.assertRaises(ValueError):
                    ResearchRequest(objective=objective, config=AgentConfig())

    def test_rejects_an_invalid_config_type(self) -> None:
        with self.assertRaises(TypeError):
            ResearchRequest(  # type: ignore[arg-type]
                objective="Research agents",
                config="scripted",
            )

    def test_validated_request_is_immutable(self) -> None:
        request = ResearchRequest(
            objective="Research agents",
            config=AgentConfig(),
        )

        with self.assertRaises(FrozenInstanceError):
            request.objective = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
