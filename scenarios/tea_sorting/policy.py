"""Workspace-wide, pose-relative tea sorting with bimanual relay support."""

from __future__ import annotations

import numpy as np

from scenarios.organizer.episode import PlanningError
from scenarios.organizer.openarm_policy import OpenArmPolicy
from superdex_scenarios.planning import TrajOptTrajectoryOptimizer

from .geometry import BOX_SIZE, GRASP, HANDOFF_XY, PACKET_SIZE
from .scenario import DESK_TOP_Z


class TeaPolicy(OpenArmPolicy):
    """Select arms by reachability and use TrajOpt for free-space motion.

    A direct transfer is preferred. If no one arm can both pick a packet and
    reach its color bin, the reachable arm deposits it in the shared transfer
    cradle and the other arm finishes the job. Objects are never attached or
    teleported; the relay is two ordinary physical releases and grasps.
    """

    def __init__(self, scenario, options=None):
        super().__init__(scenario, options)
        jobs = [
            ("part", i, actor, GRASP) for i, actor in enumerate(scenario.parts)
        ]
        # Relay routes pass behind the box and approach through its center.
        # Fill that compartment last so chamomile cannot obstruct the corridor.
        self.jobs = sorted(
            jobs,
            key=lambda job: scenario.specification.tea_order[job[1]] == 1,
        )
        self.routes: list[tuple[str, ...]] = []

    def select_arm(self, kind, index, side=None):
        if side is None:
            side = (
                "left"
                if self.position(self.scenario.parts[index])[1] > 0
                else "right"
            )
        # The organizer parent configures the correct mirrored grasp rotation
        # when part index 2 denotes left and index 0 denotes right.
        super().select_arm("part", 2 if side == "left" else 0)
        mask = np.isin(
            self.info.hand_dofs, self.info.dof_groups[f"{self.side}_gripper"]
        )
        self.grip[mask] = 0.0

    def _active_arm_mask(self):
        return np.isin(
            self.info.arm_dofs, self.info.dof_groups[f"{self.side}_arm"]
        )

    def _optimize(self, reference):
        reference = np.asarray(reference, dtype=float)
        optimizer = TrajOptTrajectoryOptimizer(
            self.kin,
            # Collision-free Cartesian references are already optimal for the
            # grasp frame. TrajOpt validates them and runs SQP only when the
            # obstacle constraints require a joint-space detour.
            accept_valid_reference=True,
        )
        path = optimizer.optimize(reference, max_iterations=28)
        path[:, ~self._active_arm_mask()] = self.home[~self._active_arm_mask()]
        clearance = optimizer.minimum_clearance(path)
        if clearance < -1.0e-4:
            raise RuntimeError(
                "TrajOpt route is invalid after parking the unused arm: "
                f"clearance={clearance:+.4f} m"
            )
        return path

    @staticmethod
    def _resting_pick(xy):
        return np.array(
            [
                xy[0],
                xy[1],
                DESK_TOP_Z + PACKET_SIZE[2] / 2 + 0.0003 + GRASP[2],
            ]
        )

    @staticmethod
    def _release_position(target, *, staging=False):
        # Release with the packet bottom one packet-height above the receiving
        # floor. This is the same clearance used by the proven fixed policy.
        result = np.asarray(target, dtype=float).copy()
        drop = 0.018 if staging else PACKET_SIZE[2]
        result[2] += drop + PACKET_SIZE[2] / 2 + GRASP[2] + 0.0003
        return result

    def _waypoints(self, pick, target, *, staging=False):
        pick = np.asarray(pick, dtype=float)
        target = np.asarray(target, dtype=float)
        # Pickup clearance and receiver clearance are independent. This keeps
        # the shared handoff reachable by both arms while still clearing the
        # taller tea-box walls on the final carry.
        lift_z = max(DESK_TOP_Z + 0.115, pick[2] + 0.063)
        placement = self._release_position(target, staging=staging)
        hover_z = max(lift_z, placement[2] + 0.008)
        from_handoff = np.linalg.norm(pick[:2] - HANDOFF_XY) < 0.035
        approach_x = 0.025 if from_handoff and self.side == "left" else -0.025
        pre = pick + [approach_x, 0.0, 0.045]
        lift = np.array([pick[0], pick[1], lift_z])
        hover = np.array([target[0], target[1], hover_z])
        return pre, lift, hover, placement

    def _free_reference(self, seed, destination):
        destination = np.asarray(destination, dtype=float)
        try:
            return self.line(seed, destination)
        except RuntimeError:
            # A Cartesian continuation can cross an IK branch boundary even
            # when both endpoints are reachable. Use the destination's home
            # branch as a seed and let TrajOpt repair the joint-space route.
            goal = self.solve(destination, self.home)
            knots = max(8, int(np.ceil(np.linalg.norm(goal - seed) / 0.20)) + 1)
            return np.linspace(seed, goal, knots)

    def _carry_waypoints(self, lift, hover, *, staging):
        if not staging:
            return [hover]
        box = np.asarray(self.scenario.organizer.get_root_transform().translation)
        side = 1.0 if lift[1] >= box[1] else -1.0
        outside_y = box[1] + side * (BOX_SIZE[1] / 2 + 0.050)
        z = max(lift[2], hover[2])
        # Stay behind the box while crossing its Y span. Splitting this into
        # separate TrajOpt problems fixes the route's homotopy explicitly.
        return [
            np.array([HANDOFF_XY[0], outside_y, z]),
            np.array([HANDOFF_XY[0], HANDOFF_XY[1], z]),
            hover,
        ]

    def _path(self, seed, destination, *, free, optimize):
        path = (
            self._free_reference(seed, destination)
            if free
            else self.line(seed, np.asarray(destination, dtype=float))
        )
        return self._optimize(path) if optimize and self.options.check_collisions else path

    def _preview_leg(self, side, pick, target, *, optimize, staging=False):
        """Plan one home-to-home transfer without advancing physics."""
        self.select_arm("part", 0, side)
        pre, lift, hover, placement = self._waypoints(
            pick, target, staging=staging
        )
        q = self.home.copy()
        points = []
        travel = 0.0

        def append(destination, free=False):
            nonlocal q, travel
            path = self._path(
                q, destination, free=free, optimize=free and optimize
            )
            points.extend(self.kin.grasp_point_world(v) for v in path[:-1])
            travel += float(
                np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
            )
            q = path[-1]

        safe = self.kin.grasp_point_world(q).copy()
        safe[2] = lift[2]
        # The authored parked pose is physically valid but overlaps the coarse
        # parked-arm proxy by a few millimetres, so it is a boundary condition,
        # not a TrajOpt clearance constraint. Optimize once above that pose.
        append(safe)
        append([pre[0], pre[1], lift[2]], True)
        append(pre)
        append(pick)
        append(lift)
        for waypoint in self._carry_waypoints(lift, hover, staging=staging):
            append(waypoint, True)
        append(placement)
        append(hover)
        exit_point = np.array(
            [0.010, -0.230 if side == "right" else 0.230, hover[2]]
        )
        append(exit_point, True)
        home_path = np.linspace(q, self.home, 12)
        points.extend(self.kin.grasp_point_world(v) for v in home_path)
        travel += float(
            np.sum(np.linalg.norm(np.diff(home_path, axis=0), axis=1))
        )
        return np.asarray(points), travel

    def _route_candidates(self, actor, target):
        pick = self.position(actor, GRASP)
        handoff_pick = self._resting_pick(HANDOFF_XY)
        handoff_target = np.array([*HANDOFF_XY, DESK_TOP_Z])
        candidates = []
        for side in ("right", "left"):
            try:
                _, score = self._preview_leg(side, pick, target, optimize=False)
                candidates.append((0, score, (side,)))
            except RuntimeError:
                pass
        for source in ("right", "left"):
            for destination in ("right", "left"):
                if source == destination:
                    continue
                try:
                    _, first = self._preview_leg(
                        source,
                        pick,
                        handoff_target,
                        optimize=False,
                        staging=True,
                    )
                    _, second = self._preview_leg(
                        destination, handoff_pick, target, optimize=False
                    )
                    candidates.append((1, first + second, (source, destination)))
                except RuntimeError:
                    pass
        return sorted(candidates)

    def plan(self):
        self.points = []
        self.routes = []
        handoff_pick = self._resting_pick(HANDOFF_XY)
        handoff_target = np.array([*HANDOFF_XY, DESK_TOP_Z])
        try:
            for _, index, actor, _ in self.jobs:
                self.kin.collision_model.active = actor
                target = self.scenario.target("part", index)
                errors = []
                for _, _, route in self._route_candidates(actor, target):
                    try:
                        if len(route) == 1:
                            legs = [
                                (
                                    route[0],
                                    self.position(actor, GRASP),
                                    target,
                                    False,
                                )
                            ]
                        else:
                            legs = [
                                (
                                    route[0],
                                    self.position(actor, GRASP),
                                    handoff_target,
                                    True,
                                ),
                                (route[1], handoff_pick, target, False),
                            ]
                        route_points = []
                        for side, pick, destination, staging in legs:
                            points, _ = self._preview_leg(
                                side,
                                pick,
                                destination,
                                optimize=True,
                                staging=staging,
                            )
                            if route_points:
                                route_points.append(np.full(3, np.nan))
                            route_points.extend(points)
                        self.routes.append(route)
                        if self.points:
                            self.points.append(np.full(3, np.nan))
                        self.points.extend(route_points)
                        break
                    except RuntimeError as exc:
                        errors.append(f"{route}: {exc}")
                else:
                    detail = "; ".join(errors) if errors else "no IK-reachable arm pair"
                    raise PlanningError(
                        f"tea_{index} is outside the usable workspace: {detail}"
                    )
        except RuntimeError as exc:
            if isinstance(exc, PlanningError):
                raise
            raise PlanningError(str(exc)) from exc
        finally:
            self.kin.collision_model.active = None

    def trajectory_points(self):
        return np.asarray(self.points)

    def move(
        self, runner, destination, hand, duration=1.0, *, free=False, precise=False
    ):
        seed = runner.measured_pose()[self.info.arm_dofs]
        path = (
            self._free_reference(seed, destination)
            if free
            else self.line(seed, np.asarray(destination))
        )
        if free and self.options.check_collisions:
            path = self._optimize(path)
        if not runner.follow(path, duration, hand):
            return False
        if precise:
            command = np.asarray(destination).copy()
            for _ in range(5):
                error = destination - runner.grasp_position()
                if np.linalg.norm(error) < 0.0008:
                    break
                if np.linalg.norm(error) > 0.020:
                    raise RuntimeError(
                        f"Blocked motion: tracking error {np.linalg.norm(error):.3f} m"
                    )
                command += np.clip(error, -0.006, 0.006)
                q = self.solve(command, runner.measured_pose()[self.info.arm_dofs])
                if not runner.follow(
                    np.vstack([runner.measured_pose()[self.info.arm_dofs], q]),
                    0.25,
                    hand,
                ):
                    return False
        return True

    def _return_home(self, runner, hand):
        q = runner.measured_pose()[self.info.arm_dofs]
        path = np.linspace(q, self.home, 12)
        return runner.follow(path, 1.5, hand, "home")

    def _staging_status(self, actor):
        bounds = actor.get_aabb_world()
        lo, hi = np.asarray(bounds.min), np.asarray(bounds.max)
        position = self.position(actor)
        return {
            "position_m": position.tolist(),
            "near_cradle": bool(np.linalg.norm(position[:2] - HANDOFF_XY) < 0.025),
            "upright": bool(hi[2] - lo[2] > 0.060),
            "resting": bool(abs(lo[2] - DESK_TOP_Z) < 0.006),
            "speed_m_s": float(np.linalg.norm(actor.get_linear_velocity())),
            "angular_speed_rad_s": float(np.linalg.norm(actor.get_angular_velocity())),
        }

    def _transfer(self, runner, index, actor, side, target, *, final):
        self.select_arm("part", index, side)
        runner.select_arm(self.side)
        stage = "sort" if final else "relay"
        name = f"tea_{index}/{stage}_{side}"
        self.kin.collision_model.active = actor
        pick = self.position(actor, GRASP)
        from_handoff = np.linalg.norm(pick[:2] - HANDOFF_XY) < 0.035
        pre, lift, hover, _ = self._waypoints(
            pick, target, staging=not final
        )
        runner.phase(f"{name}/approach")
        safe = runner.grasp_position().copy()
        safe[2] = lift[2]
        for point, duration, free in (
            (safe, 0.8, False),
            ([pre[0], pre[1], lift[2]], 1.4, True),
            (pre, 0.8, False),
            (pick, 1.0, False),
        ):
            if not self.move(
                runner,
                np.asarray(point),
                self.open_hand,
                duration,
                free=free,
                precise=not free,
            ):
                return False
        runner.phase(f"{name}/grasp")
        q = runner.measured_pose()[self.info.arm_dofs]
        if not runner.follow(np.vstack([q, q]), 0.8, self.open_hand, self.grip):
            return False
        if not runner.hold(q, self.grip, 0.3):
            return False
        before = self.position(actor)
        runner.phase(f"{name}/lift")
        # The left arm reaches the relay rack close to a joint limit. Its
        # physical lift check below is a better completion condition than
        # chasing the final millimetres of compliant tracking sag there.
        if not self.move(
            runner,
            lift,
            self.grip,
            1.3,
            precise=not from_handoff,
        ):
            return False
        rise = self.position(actor)[2] - before[2]
        if rise < 0.055:
            raise RuntimeError(f"{name}: physical grasp failed (rise {rise:.4f} m)")
        held = runner.grasp_position() - self.position(actor)
        hover[:2] = (np.asarray(target) + held)[:2]
        runner.phase(f"{name}/carry")
        waypoints = self._carry_waypoints(lift, hover, staging=not final)
        for waypoint in waypoints:
            if not self.move(
                runner,
                waypoint,
                self.grip,
                1.8 / len(waypoints),
                free=True,
                precise=final and waypoint is waypoints[-1],
            ):
                return False
        bounds = actor.get_aabb_world()
        centre = (np.asarray(bounds.min) + np.asarray(bounds.max)) / 2
        placement = runner.grasp_position().copy()
        placement[:2] += np.asarray(target)[:2] - centre[:2]
        drop = PACKET_SIZE[2] if final else 0.018
        placement[2] += target[2] + drop - np.asarray(bounds.min)[2]
        runner.phase(f"{name}/place")
        if not self.move(
            runner, placement, self.grip, 1.5, precise=final
        ):
            return False
        runner.phase(f"{name}/release")
        q = runner.measured_pose()[self.info.arm_dofs]
        release = self.open_hand.copy()
        mask = np.isin(
            self.info.hand_dofs, self.info.dof_groups[f"{self.side}_gripper"]
        )
        release[mask] = 0.13 if self.side == "left" else -0.13
        if not runner.follow(np.vstack([q, q]), 0.6, self.grip, release):
            return False
        if not runner.hold(q, release, 0.8):
            return False
        runner.phase(f"{name}/retreat")
        if not self.move(runner, hover, release, 0.9):
            return False
        q = runner.measured_pose()[self.info.arm_dofs]
        if not runner.follow(np.vstack([q, q]), 0.6, release, self.open_hand):
            return False
        if not runner.hold(q, self.open_hand, 0.8):
            return False

        status = None
        passed = False
        for _ in range(30 if final else 16):
            status = (
                self.scenario.outcome()["parts"][index]
                if final
                else self._staging_status(actor)
            )
            passed = (
                status["success"]
                if final
                else status["near_cradle"]
                and status["upright"]
                and status["resting"]
                and status["speed_m_s"] < 0.015
                and status["angular_speed_rad_s"] < 0.15
            )
            if passed:
                break
            if not runner.hold(q, self.open_hand, 0.5):
                return False
        runner.event("placement_checked" if final else "relay_checked", **status)
        if not passed:
            raise RuntimeError(f"{name}: placement failed: {status}")
        exit_point = np.array(
            [0.010, -0.230 if self.side == "right" else 0.230, hover[2]]
        )
        if not self.move(runner, exit_point, self.open_hand, 1.3, free=True):
            return False
        return self._return_home(runner, self.open_hand)

    def run(self, runner):
        if len(self.routes) != len(self.jobs):
            raise RuntimeError("TeaPolicy.plan() must run before execution")
        runner.phase("home")
        if not runner.hold(self.home, "open", 0.5):
            return False
        handoff_target = np.array([*HANDOFF_XY, DESK_TOP_Z])
        for (_, index, actor, _), route in zip(self.jobs, self.routes):
            target = self.scenario.target("part", index)
            if len(route) == 2:
                if not self._transfer(
                    runner,
                    index,
                    actor,
                    route[0],
                    handoff_target,
                    final=False,
                ):
                    return False
            if not self._transfer(
                runner, index, actor, route[-1], target, final=True
            ):
                return False
        runner.phase("return_home")
        self.kin.collision_model.active = None
        return runner.hold(self.home, "home", 1.0)
