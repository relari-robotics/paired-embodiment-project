"""The blank human policy must satisfy the episode contract and stay blank."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scenarios.ball_bowl.episode import EpisodePolicy, PolicyOptions  # noqa: E402
from scenarios.ball_bowl.human_project import HumanPolicy  # noqa: E402


class HumanProjectScaffoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = HumanPolicy(object(), PolicyOptions())

    def test_policy_satisfies_episode_contract(self) -> None:
        self.assertIsInstance(self.policy, EpisodePolicy)

    def test_every_method_is_intentionally_unimplemented(self) -> None:
        for method in ("plan", "trajectory_points", "home_pose", "preshape_pose"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(NotImplementedError, "intentionally unimplemented"):
                    getattr(self.policy, method)()
        with self.assertRaisesRegex(NotImplementedError, "intentionally unimplemented"):
            self.policy.run(object())


if __name__ == "__main__":
    unittest.main()
