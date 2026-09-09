"""Embodiment-neutral motor-effort and contact-force recording."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Self

import h5py
import numpy as np
import numpy.typing as npt
from superdex import physics

from superdex_scenarios.embodiments import ContactGroup, EmbodimentModel


def _vector(value: object) -> list[float]:
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def _joint_dof_components(joint: object) -> tuple[str, ...]:
    joint_type = joint.type
    if joint_type == physics.ArticulatedJointType.HARD:
        return ()
    if joint_type == physics.ArticulatedJointType.SPHERICAL:
        return ("rx", "ry", "rz")
    if joint_type == physics.ArticulatedJointType.FREE:
        return ("tx", "ty", "tz", "rx", "ry", "rz")
    return ("axis",)


def _task_object(scenario: Any) -> tuple[physics.Actor, str]:
    """Return the manipulated actor and its label.

    A workcell may expose ``task_object``/``task_object_label`` (for example
    the soft sponge of the sponge-and-plate task); the ball-and-bowl workcell
    exposes ``ball``.
    """
    workcell = scenario.workcell
    actor = getattr(workcell, "task_object", None)
    if actor is None:
        actor = workcell.ball
    return actor, str(getattr(workcell, "task_object_label", "ball"))


class TelemetryRecorder:
    """Stream full-rate physical telemetry and sparse contacts to disk.

    Motor torque is Mochi's generalized force from the articulated pose
    controller. Contact forces are reported in the simulation world frame.
    Logical hand forces are derived only from link-to-object contact queries,
    where the object is the workcell's manipulated actor (ball, sponge, ...).
    """

    FORMAT = "superdex-embodiment-telemetry-v2"

    def __init__(
        self,
        scenario: Any,
        output_dir: Path,
        *,
        time_step: float,
        cameras: list[dict[str, object]] | None = None,
    ) -> None:
        self.scenario = scenario
        self.scene = scenario.scene
        self.info: EmbodimentModel = scenario.bot_info
        self.task_object, self.object_label = _task_object(scenario)
        self.output_dir = output_dir
        self.time_step = float(time_step)
        self.cameras = list(cameras or ())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self.sample_count = 0
        self.contact_sample_count = 0
        self._query_registrations: list[tuple[physics.Actor, object]] = []

        self.joints = self._joint_metadata()
        if len(self.joints) != self.info.actor.get_num_dofs():
            raise RuntimeError(
                f"Joint metadata has {len(self.joints)} entries for "
                f"{self.info.actor.get_num_dofs()} actor DoFs."
            )

        self.actors = self._collect_contact_actors()
        self.actor_names = [actor.get_name() for actor in self.actors]
        self.torque_actor_names = {
            actor.get_name()
            for actor in self.actors
            if actor.get_type() in (physics.ActorType.RIGID, physics.ActorType.ARTICULATED)
        }
        self.contact_group_actors = {
            group.name: self._actors_for_group(group)
            for group in self.info.contact_groups
        }

        self._register_query(
            self.info.actor, physics.QueryType.ARTICULATED_CONTROLLER_FORCE
        )
        for actor in self.actors:
            self._register_query(actor, physics.QueryType.TOTAL_CONTACT_FORCE)

        # Embodiment links are queried for embodiment-to-world contacts. The
        # object query adds its desk/bowl/plate contacts; embodiment contacts
        # from that query are skipped to prevent duplicates.
        self.embodiment_contact_actors: list[physics.Actor] = []
        for handle in self.info.actor.get_nested_link_actors():
            actor = self.scene.get_actor(handle)
            if actor.is_query_supported(physics.QueryType.CONTACT_POINTS):
                self._register_query(actor, physics.QueryType.CONTACT_POINTS)
                self.embodiment_contact_actors.append(actor)
        self._register_query(self.task_object, physics.QueryType.CONTACT_POINTS)
        self.embodiment_actor_names = {
            actor.get_name() for actor in self.embodiment_contact_actors
        }

        dof_count = self.info.actor.get_num_dofs()
        self.pose = physics.DynamicArrayReal(dof_count)
        self.velocity = physics.DynamicArrayReal(dof_count)

        self._csv_file = (self.output_dir / "telemetry.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._csv = csv.writer(self._csv_file)
        self.columns = self._columns()
        self._csv.writerow(self.columns)

        self.all_actor_names = self._all_actor_names()
        self.actor_ids = {
            name: index for index, name in enumerate(self.all_actor_names)
        }
        self._contact_dtype = np.dtype(
            [
                ("step", "<i4"),
                ("query_actor_id", "<u2"),
                ("actor_a_id", "<u2"),
                ("actor_b_id", "<u2"),
                ("position_a_world_m", "<f4", (3,)),
                ("position_b_world_m", "<f4", (3,)),
                ("normal_world", "<f4", (3,)),
                ("force_on_actor_a_world_n", "<f4", (3,)),
                ("distance_m", "<f4"),
                ("point_velocity_a_world_m_s", "<f4", (3,)),
                ("point_velocity_b_world_m_s", "<f4", (3,)),
                ("sample_index", "<i4"),
                ("integration_weight", "<f4"),
            ]
        )
        self._contacts_h5 = h5py.File(self.output_dir / "contacts.h5", "w")
        string_type = h5py.string_dtype(encoding="utf-8")
        self._contacts_h5.create_dataset(
            "actor_names",
            data=np.asarray(self.all_actor_names, dtype=object),
            dtype=string_type,
        )
        self._contact_dataset = self._contacts_h5.create_dataset(
            "contacts",
            shape=(0,),
            maxshape=(None,),
            chunks=(16384,),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            dtype=self._contact_dtype,
        )
        self._contacts_h5.attrs["format"] = "superdex-contact-forces-v1"
        self._contacts_h5.attrs["physics_rate_hz"] = 1.0 / self.time_step
        self._contacts_h5.attrs["embodiment"] = self.info.embodiment_id
        self._contacts_h5.attrs["force_definition"] = (
            "World-frame force applied to actor_a; zero-force near-contact "
            "candidates are omitted."
        )
        self._contact_buffer: list[npt.NDArray[Any]] = []
        self._contact_buffer_size = 0

        self.peak_abs_motor_torque = np.zeros(len(self.joints), dtype=float)
        self.peak_contact_force = {name: 0.0 for name in self.actor_names}
        self.peak_group_ball_force = {
            group.name: 0.0 for group in self.info.contact_groups
        }
        self.peak_equivalent_grip_force = 0.0
        self.peak_hand_motor_torque_abs_sum = 0.0

    def _register_query(self, actor: physics.Actor, query_type: object) -> None:
        if not actor.is_query_supported(query_type):
            raise RuntimeError(
                f"Actor {actor.get_name()!r} does not support required query "
                f"{query_type}."
            )
        handle = actor.register_query(query_type)
        self._query_registrations.append((actor, handle))

    def _joint_metadata(self) -> list[dict[str, object]]:
        controlled = set(self.info.controlled_dofs.tolist())
        result: list[dict[str, object]] = []
        dof = 0
        for joint in self.info.prefab.joints:
            components = _joint_dof_components(joint)
            for component in components:
                name = (
                    joint.name if len(components) == 1 else f"{joint.name}/{component}"
                )
                result.append(
                    {
                        "dof_index": dof,
                        "name": name,
                        "joint_name": joint.name,
                        "component": component,
                        "controlled": dof in controlled,
                        "effort_limit_nm": float(joint.effort_limit),
                        "axis_local": _vector(joint.axis),
                    }
                )
                dof += 1
        return result

    def _collect_contact_actors(self) -> list[physics.Actor]:
        actors: dict[object, physics.Actor] = {}

        def collect(actor: physics.Actor) -> None:
            if actor.is_query_supported(physics.QueryType.TOTAL_CONTACT_FORCE):
                actors[actor.get_handle()] = actor

        self.scene.for_each_actor(collect)
        return sorted(actors.values(), key=lambda actor: actor.get_name())

    def _actors_for_group(self, group: ContactGroup) -> tuple[physics.Actor, ...]:
        matched: list[physics.Actor] = []
        for suffix in group.actor_suffixes:
            candidates = [
                actor for actor in self.actors if actor.get_name().endswith(suffix)
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one contact actor ending in {suffix!r} for "
                    f"{group.name!r}, found {len(candidates)}."
                )
            matched.append(candidates[0])
        return tuple(matched)

    def _all_actor_names(self) -> list[str]:
        names: set[str] = set()
        self.scene.for_each_actor(lambda actor: names.add(actor.get_name()))
        return sorted(names)

    def _columns(self) -> list[str]:
        columns = ["step", "time_s"]
        for joint in self.joints:
            name = str(joint["name"])
            columns.extend(
                [
                    f"joint/{name}/position_rad",
                    f"joint/{name}/velocity_rad_s",
                    f"joint/{name}/target_rad",
                    f"joint/{name}/motor_torque_nm",
                ]
            )
        for name in self.actor_names:
            columns.extend(
                [
                    f"contact/{name}/force_world_x_n",
                    f"contact/{name}/force_world_y_n",
                    f"contact/{name}/force_world_z_n",
                    f"contact/{name}/force_norm_n",
                    f"contact/{name}/torque_com_world_x_nm",
                    f"contact/{name}/torque_com_world_y_nm",
                    f"contact/{name}/torque_com_world_z_nm",
                    f"contact/{name}/torque_com_norm_nm",
                ]
            )
        prefix = "gripper" if self.info.embodiment_id == "openarm_v2" else "hand"
        obj = self.object_label
        for group in self.info.contact_groups:
            columns.extend(
                [
                    f"{prefix}/{group.name}_{obj}_force_world_x_n",
                    f"{prefix}/{group.name}_{obj}_force_world_y_n",
                    f"{prefix}/{group.name}_{obj}_force_world_z_n",
                    f"{prefix}/{group.name}_{obj}_force_norm_n",
                ]
            )
        if self.info.embodiment_id == "openarm_v2":
            columns.extend(
                [
                    f"gripper/two_jaw_{obj}_force_sum_n",
                    "gripper/pinch_force_single_jaw_equivalent_n",
                    "gripper/finger1_motor_torque_nm",
                    "gripper/finger2_motor_torque_nm",
                    "gripper/coupled_1to1_motor_torque_nm",
                    "gripper/motor_torque_abs_sum_nm",
                ]
            )
        else:
            columns.extend(
                [
                    f"hand/aggregate_{obj}_force_sum_n",
                    "hand/opposition_force_equivalent_n",
                    "hand/joint_motor_torque_abs_sum_nm",
                ]
            )
        return columns

    def _group_ball_forces(self) -> tuple[list[np.ndarray], np.ndarray]:
        vectors = []
        for group in self.info.contact_groups:
            vector = np.zeros(3, dtype=float)
            for actor in self.contact_group_actors[group.name]:
                vector += np.asarray(
                    actor.get_contact_force_from_actor_world(self.task_object),
                    dtype=float,
                )
            vectors.append(vector)
        return vectors, np.asarray([np.linalg.norm(v) for v in vectors], dtype=float)

    def record_sample(self, step: int, target_pose: npt.ArrayLike) -> None:
        """Record the state produced by the just-completed simulation step."""
        self.info.actor.get_articulated_pose(self.pose)
        self.info.actor.get_articulated_joint_velocities(self.velocity)
        position = np.asarray(self.pose, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        target = np.asarray(target_pose, dtype=float)
        motor_torque = np.asarray(
            self.info.actor.get_articulated_controller_force(), dtype=float
        )
        if motor_torque.shape != position.shape:
            raise RuntimeError(
                f"Controller force has shape {motor_torque.shape}, expected "
                f"{position.shape}."
            )

        row: list[float | int] = [step, step * self.time_step]
        for dof in range(len(self.joints)):
            row.extend([position[dof], velocity[dof], target[dof], motor_torque[dof]])
        self.peak_abs_motor_torque = np.maximum(
            self.peak_abs_motor_torque, np.abs(motor_torque)
        )

        for actor in self.actors:
            force = np.asarray(actor.get_contact_force_world(), dtype=float)
            if actor.get_name() in self.torque_actor_names:
                torque = np.asarray(actor.get_contact_torque_world(), dtype=float)
            else:
                # Soft actors report a total contact force but no wrench torque.
                torque = np.zeros(3, dtype=float)
            force_norm = float(np.linalg.norm(force))
            torque_norm = float(np.linalg.norm(torque))
            row.extend([*force, force_norm, *torque, torque_norm])
            name = actor.get_name()
            self.peak_contact_force[name] = max(
                self.peak_contact_force[name], force_norm
            )

        group_forces, group_norms = self._group_ball_forces()
        for group, force, norm in zip(
            self.info.contact_groups, group_forces, group_norms
        ):
            row.extend([*force, norm])
            self.peak_group_ball_force[group.name] = max(
                self.peak_group_ball_force[group.name], float(norm)
            )

        hand_torques = motor_torque[self.info.hand_dofs]
        hand_abs_sum = float(np.sum(np.abs(hand_torques)))
        if self.info.embodiment_id == "openarm_v2":
            force_sum = float(np.sum(group_norms))
            equivalent_force = 0.5 * force_sum
            coupled_torque = float(np.sum(hand_torques))
            row.extend(
                [
                    force_sum,
                    equivalent_force,
                    hand_torques[0],
                    hand_torques[1],
                    coupled_torque,
                    hand_abs_sum,
                ]
            )
        else:
            force_sum = float(np.sum(group_norms))
            thumb_force = float(group_norms[0]) if len(group_norms) else 0.0
            opposing_force = float(np.sum(group_norms[1:]))
            equivalent_force = min(thumb_force, opposing_force)
            row.extend([force_sum, equivalent_force, hand_abs_sum])
        self._csv.writerow(row)

        self.peak_equivalent_grip_force = max(
            self.peak_equivalent_grip_force, equivalent_force
        )
        self.peak_hand_motor_torque_abs_sum = max(
            self.peak_hand_motor_torque_abs_sum, hand_abs_sum
        )
        self._record_contacts(step)
        self.sample_count += 1

    def _record_contacts(self, step: int) -> None:
        contacts: list[tuple[object, ...]] = []
        for actor in self.embodiment_contact_actors:
            contacts.extend(self._contacts_from(actor, skip_embodiment=False))
        contacts.extend(self._contacts_from(self.task_object, skip_embodiment=True))
        if not contacts:
            return
        block = np.empty(len(contacts), dtype=self._contact_dtype)
        block["step"] = step
        for index, contact in enumerate(contacts):
            (
                query_actor,
                actor_a,
                actor_b,
                position_a,
                position_b,
                normal,
                force,
                distance,
                velocity_a,
                velocity_b,
                sample_index,
                integration_weight,
            ) = contact
            block[index] = (
                step,
                self.actor_ids[str(query_actor)],
                self.actor_ids[str(actor_a)],
                self.actor_ids[str(actor_b)],
                position_a,
                position_b,
                normal,
                force,
                distance,
                velocity_a,
                velocity_b,
                sample_index,
                integration_weight,
            )
        self._contact_buffer.append(block)
        self._contact_buffer_size += len(block)
        if self._contact_buffer_size >= 16384:
            self._flush_contacts()
        self.contact_sample_count += len(contacts)

    def _flush_contacts(self) -> None:
        if not self._contact_buffer:
            return
        block = np.concatenate(self._contact_buffer)
        start = len(self._contact_dataset)
        self._contact_dataset.resize((start + len(block),))
        self._contact_dataset[start:] = block
        self._contact_buffer.clear()
        self._contact_buffer_size = 0

    def _contacts_from(
        self,
        query_actor: physics.Actor,
        *,
        skip_embodiment: bool,
    ) -> list[tuple[object, ...]]:
        records = []
        for point in query_actor.get_contact_points_world():
            force = np.asarray(point.force, dtype=float)
            if float(np.linalg.norm(force)) <= 1.0e-10:
                continue
            actor_a = self.scene.get_actor(point.actor_a).get_name()
            actor_b = self.scene.get_actor(point.actor_b).get_name()
            if skip_embodiment and (
                actor_a in self.embodiment_actor_names
                or actor_b in self.embodiment_actor_names
            ):
                continue
            records.append(
                (
                    query_actor.get_name(),
                    actor_a,
                    actor_b,
                    np.asarray(point.pos_a, dtype=np.float32),
                    np.asarray(point.pos_b, dtype=np.float32),
                    np.asarray(point.normal, dtype=np.float32),
                    force.astype(np.float32),
                    float(point.distance),
                    np.asarray(point.point_velocity_a, dtype=np.float32),
                    np.asarray(point.point_velocity_b, dtype=np.float32),
                    int(point.sample_index),
                    float(point.int_weight),
                )
            )
        return records

    def close(self) -> None:
        if self._closed:
            return
        self._csv_file.close()
        self._flush_contacts()
        self._contacts_h5.close()
        for actor, handle in reversed(self._query_registrations):
            actor.cancel_query(handle)
        self._query_registrations.clear()

        metadata = {
            "format": self.FORMAT,
            "embodiment": self.info.embodiment_id,
            "scenario": self.scenario.specification.to_dict(),
            "physics_rate_hz": 1.0 / self.time_step,
            "coordinate_system": "world +X forward, +Y left, +Z up",
            "samples": self.sample_count,
            "files": {
                "dense_telemetry": "telemetry.csv",
                "individual_contacts": "contacts.h5",
                "summary": "telemetry_summary.json",
            },
            "motor_torque_definition": (
                "Joint-side generalized force applied by Mochi's articulated pose "
                "controller; not electrical current or pre-gearbox shaft torque."
            ),
            "contact_definition": (
                "World-frame contact force; aggregate contact torque is about the "
                "actor center of mass. Individual contact force is applied to actor_a."
            ),
            "logical_contact_groups": [
                {
                    "name": group.name,
                    "actor_suffixes": list(group.actor_suffixes),
                    "dofs": list(group.dofs),
                }
                for group in self.info.contact_groups
            ],
            "joints": self.joints,
            "contact_wrench_actors": self.actor_names,
            "contact_hdf5": {
                "dataset": "contacts",
                "actor_name_table": "actor_names",
                "time_definition": "time_s = step / physics_rate_hz",
                "fields": list(self._contact_dtype.names or ()),
            },
            "cameras": self.cameras,
        }
        summary = {
            "format": self.FORMAT,
            "embodiment": self.info.embodiment_id,
            "samples": self.sample_count,
            "duration_s": self.sample_count * self.time_step,
            "individual_contact_samples": self.contact_sample_count,
            "peak_abs_motor_torque_nm": {
                str(joint["name"]): float(self.peak_abs_motor_torque[index])
                for index, joint in enumerate(self.joints)
            },
            "peak_contact_force_norm_n": self.peak_contact_force,
            "hand": {
                f"peak_group_{self.object_label}_force_norm_n": self.peak_group_ball_force,
                "peak_equivalent_grip_force_n": self.peak_equivalent_grip_force,
                "peak_joint_motor_torque_abs_sum_nm": (
                    self.peak_hand_motor_torque_abs_sum
                ),
            },
        }
        if self.info.embodiment_id == "openarm_v2":
            summary["gripper"] = {
                f"peak_finger1_{self.object_label}_force_norm_n": (
                    self.peak_group_ball_force.get("finger1", 0.0)
                ),
                f"peak_finger2_{self.object_label}_force_norm_n": (
                    self.peak_group_ball_force.get("finger2", 0.0)
                ),
                "peak_pinch_force_single_jaw_equivalent_n": (
                    self.peak_equivalent_grip_force
                ),
            }
        (self.output_dir / "telemetry_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        (self.output_dir / "telemetry_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Telemetry exported: {self.output_dir / 'telemetry.csv'} "
            f"({self.sample_count} samples at {1.0 / self.time_step:.0f} Hz)"
        )
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
