# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenArm v2 reference policy for the sponge-and-plate wiping task.

The gripper pinches the upper part of the sponge with its two jaws, carries it
over the plate, presses it into the dish floor, and drags it through three
serpentine passes.  The sponge is held and moved by contact friction alone; the
wipe is executed as direct inverse-kinematics references so the pressing depth
is honoured, while free-space segments are collision-optimized.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from superdex_scenarios.planning import sample_natural_cubic_spline

from . import scenario as task
from .episode import PlanningError, PolicyOptions
from .runner import EpisodeRunner


class OpenArmPolicy:
    """Reference implementation for the OpenArm v2 right arm and gripper."""

    def __init__(self, scenario: task.SpongePlateScenario, options: PolicyOptions) -> None:
        self.scenario = scenario
        self.options = options
        self.info = scenario.bot_info
        self.motion: task.PlannedMotion | None = None

    # -- planning ---------------------------------------------------------

    def plan(self) -> None:
        try:
            self.motion = task.plan_motion(
                self.scenario.kinematics,
                self.info,
                self.scenario.specification,
                optimize_trajectory=self.options.optimize_trajectory,
            )
        except RuntimeError as error:
            raise PlanningError(str(error)) from error

    def _motion(self) -> task.PlannedMotion:
        if self.motion is None:
            raise RuntimeError("OpenArmPolicy.plan() must run before execution.")
        return self.motion

    def trajectory_points(self) -> npt.NDArray[np.float64]:
        """Sample the same cubic splines that the simulator executes."""
        kinematics = self.scenario.kinematics
        motion = self._motion()
        points = []
        for segment in motion.segments():
            samples = sample_natural_cubic_spline(segment)
            points.extend(kinematics.grasp_point_world(q) for q in samples[:-1])
        points.append(kinematics.grasp_point_world(motion.retreat_to_home[-1]))
        return np.asarray(points, dtype=float)

    def home_pose(self) -> npt.NDArray[np.float64]:
        return self.info.target_pose(self._motion().home_to_pre_pick[0], "home")

    def preshape_pose(self) -> npt.NDArray[np.float64]:
        return self.info.target_pose(self._motion().pre_pick_to_pick[-1], "preshape")

    # -- execution --------------------------------------------------------

    def run(self, runner: EpisodeRunner) -> bool:
        motion = self._motion()
        home = motion.home_to_pre_pick[0]
        pregrasp = motion.pre_pick_to_pick[-1]
        grip = np.full(2, task.FINGERS_WIPE_CLOSED)
        place = motion.place_safe_to_place[-1]

        print(f"Executing {self.info.display_name} sponge-wipes-plate episode...")
        runner.phase("home")
        if not runner.hold(home, "home", 0.20):
            return False
        runner.phase("preshape")
        if not runner.follow(np.vstack([home, home]), 0.55, "home", "preshape"):
            return False
        runner.phase("approach")
        if not runner.follow(motion.home_to_pre_pick, 1.5, "preshape"):
            return False
        runner.phase("pre_grasp")
        if not runner.follow(motion.pre_pick_to_pick, 1.2, "preshape"):
            return False
        if not runner.hold(pregrasp, "preshape", 0.5):
            return False
        runner.print_tracking_error(pregrasp)
        runner.check_grasp_alignment()

        # Parallel jaws: a position ramp pinches the upper part of the sponge;
        # it is held by the resulting contact friction alone.
        runner.phase("grasp")
        if not runner.follow(np.vstack([pregrasp, pregrasp]), 0.9, "open", grip):
            return False
        if not runner.hold(pregrasp, grip, 0.35):
            return False
        runner.phase("lift")
        if not runner.follow(motion.pick_to_lift, 1.2, grip):
            return False
        runner.verify_physical_grasp()
        runner.phase("carry")
        if not runner.follow(motion.lift_to_hover, 2.0, grip):
            return False
        runner.phase("lower")
        if not runner.follow(motion.hover_to_press, 1.0, grip):
            return False
        if not runner.hold(motion.hover_to_press[-1], grip, 0.4):
            return False
        runner.verify_press()

        runner.phase("wipe")
        for index, segment in enumerate(motion.wipe_segments):
            is_stroke = index % 2 == 0
            if not runner.follow(segment, 1.4 if is_stroke else 0.7, grip):
                return False
            if is_stroke:
                runner.print_wipe_progress(index // 2 + 1)
        runner.phase("lift_off")
        if not runner.follow(motion.press_to_lift_off, 0.8, grip):
            return False
        runner.phase("carry_back")
        if not runner.follow(motion.lift_off_to_place_safe, 2.0, grip):
            return False
        runner.phase("lower_back")
        if not runner.follow(motion.place_safe_to_place, 1.0, grip):
            return False

        runner.phase("release")
        runner.print_release_position()
        if not runner.follow(np.vstack([place, place]), 0.55, grip, "open"):
            return False
        if not runner.hold(place, "open", 0.6):
            return False
        runner.phase("retreat")
        if not runner.follow(motion.place_to_retreat, 0.9, "open"):
            return False
        runner.phase("return_home")
        if not runner.follow(motion.retreat_to_home, 1.5, "open", "home"):
            return False
        return runner.hold(home, "home", 0.35)


__all__ = ["OpenArmPolicy"]
