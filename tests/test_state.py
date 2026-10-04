"""Tests for state accumulated during a research run."""

import unittest
from dataclasses import FrozenInstanceError

from research_agent.state import AgentState, AgentStateSnapshot, RunStatus


class AgentStateTests(unittest.TestCase):
    def test_new_state_starts_ready_and_empty(self) -> None:
        state = AgentState()

        self.assertEqual(state.step_count, 0)
        self.assertEqual(state.visited_urls, set())
        self.assertEqual(state.notes, [])
        self.assertEqual(state.tool_history, [])
        self.assertEqual(state.status, RunStatus.READY)

    def test_states_do_not_share_their_collections(self) -> None:
        first_state = AgentState()
        second_state = AgentState()

        first_state.visited_urls.add("https://example.com")
        first_state.notes.append("A useful finding")
        first_state.tool_history.append("search_web")

        self.assertEqual(second_state.visited_urls, set())
        self.assertEqual(second_state.notes, [])
        self.assertEqual(second_state.tool_history, [])


class AgentStateSnapshotTests(unittest.TestCase):
    def test_copies_state_into_immutable_collections(self) -> None:
        state = AgentState(
            step_count=1,
            visited_urls={"https://example.com"},
            notes=["finding"],
            tool_history=["save_note"],
            status=RunStatus.FINISHED,
        )

        snapshot = AgentStateSnapshot.from_state(state)
        state.notes.append("later")

        self.assertEqual(snapshot.notes, ("finding",))
        self.assertEqual(snapshot.visited_urls, ("https://example.com",))
        with self.assertRaises(FrozenInstanceError):
            snapshot.status = RunStatus.FAILED  # type: ignore[misc]

    def test_direct_constructor_rejects_mutable_collections(self) -> None:
        with self.assertRaises(TypeError):
            AgentStateSnapshot(
                step_count=0,
                visited_urls=[],  # type: ignore[arg-type]
                notes=(),
                tool_history=(),
                status=RunStatus.READY,
            )

    def test_direct_constructor_validates_step_status_and_text(self) -> None:
        invalid_values = (
            {"step_count": True},
            {"step_count": -1},
            {"status": "finished"},
            {"visited_urls": ("",)},
            {"notes": ("invalid \ud800",)},
            {"tool_history": (1,)},
            {"tool_history": ("save_note",)},
        )

        defaults: dict[str, object] = {
            "step_count": 0,
            "visited_urls": (),
            "notes": (),
            "tool_history": (),
            "status": RunStatus.READY,
        }
        for changes in invalid_values:
            with self.subTest(changes=changes):
                values = defaults | changes
                with self.assertRaises((TypeError, ValueError)):
                    AgentStateSnapshot(**values)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
