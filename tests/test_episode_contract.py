"""Dependency-free checks of the shared episode contract and phase log."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scenarios.ball_bowl.episode import (  # noqa: E402
    TASK_PHASES,
    EpisodePolicy,
    PlanningError,
    PolicyOptions,
    load_policy_class,
)
from superdex_scenarios.recording.phases import PHASE_SEQUENCE, PhaseLog  # noqa: E402


class _FakePolicy:
    def __init__(self, scenario: object, options: PolicyOptions) -> None:
        self.scenario = scenario
        self.options = options

    def plan(self) -> None:
        raise PlanningError("infeasible sample")

    def trajectory_points(self) -> np.ndarray:
        return np.zeros((2, 3))

    def home_pose(self) -> np.ndarray:
        return np.zeros(3)

    def preshape_pose(self) -> np.ndarray:
        return np.zeros(3)

    def run(self, runner: object) -> bool:
        return True


class EpisodeContractTest(unittest.TestCase):
    def test_runtime_protocol_recognises_a_conforming_policy(self) -> None:
        self.assertIsInstance(_FakePolicy(object(), PolicyOptions()), EpisodePolicy)

    def test_runtime_protocol_rejects_a_partial_policy(self) -> None:
        class Partial:
            def plan(self) -> None: ...

        self.assertNotIsInstance(Partial(), EpisodePolicy)

    def test_planning_error_is_a_runtime_error(self) -> None:
        self.assertTrue(issubclass(PlanningError, RuntimeError))

    def test_phase_names_match_the_phase_log(self) -> None:
        self.assertEqual(tuple(TASK_PHASES), tuple(PHASE_SEQUENCE))

    def test_load_policy_class_resolves_module_and_class(self) -> None:
        cls = load_policy_class("scenarios.ball_bowl.human_project:HumanPolicy")
        self.assertEqual(cls.__name__, "HumanPolicy")
        with self.assertRaises(ValueError):
            load_policy_class("no-colon")

    def test_default_options(self) -> None:
        options = PolicyOptions()
        self.assertTrue(options.optimize_trajectory)
        self.assertFalse(options.allow_failed_grasp)


class PhaseLogTest(unittest.TestCase):
    def test_begin_closes_previous_phase_and_serialises(self) -> None:
        clock = {"step": 0}

        def sampler():
            clock["step"] += 100
            return clock["step"], clock["step"] / 400.0, [0.1, 0.2, 0.3], [0, 0, 0, 1], [0, 0, 0.4]

        log = PhaseLog("test_embodiment", sampler)
        log.begin("home")
        log.begin("preshape")
        log.event("grasp_verified", note="ok")
        log.finish_episode(completed=False)
        payload = log.to_dict()
        self.assertEqual(payload["format"], PhaseLog.FORMAT)
        self.assertEqual([p["name"] for p in payload["phases"]], ["home", "preshape"])
        self.assertEqual(payload["phases"][0]["end"]["step"], payload["phases"][1]["start"]["step"])
        self.assertIsNotNone(payload["phases"][1]["end"])
        self.assertEqual(payload["events"][0]["name"], "grasp_verified")
        self.assertEqual(payload["events"][0]["note"], "ok")
        self.assertEqual(payload["phase_sequence"], list(PHASE_SEQUENCE))

    def test_rejects_out_of_order_or_incomplete_successful_episodes(self) -> None:
        def sampler():
            return 0, 0.0, [0, 0, 0], [0, 0, 0, 1], [0, 0, 0]

        log = PhaseLog("test_embodiment", sampler)
        with self.assertRaisesRegex(ValueError, "Expected phase 'home'"):
            log.begin("approach")

        log.begin("home")
        with self.assertRaisesRegex(RuntimeError, "missing"):
            log.finish_episode(completed=True)

    def test_accepts_the_complete_shared_phase_sequence(self) -> None:
        def sampler():
            return 0, 0.0, [0, 0, 0], [0, 0, 0, 1], [0, 0, 0]

        log = PhaseLog("test_embodiment", sampler)
        for phase in PHASE_SEQUENCE:
            log.begin(phase)
        log.finish_episode(completed=True)
        self.assertEqual([record.name for record in log.records], list(PHASE_SEQUENCE))


if __name__ == "__main__":
    unittest.main()
