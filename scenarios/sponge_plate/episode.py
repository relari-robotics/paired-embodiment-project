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

"""Episode contract for the sponge-and-plate task.

The policy protocol, options, and planning error are the same objects the
ball-and-bowl task uses, so a policy author moves between tasks without
relearning the runner.  Only the semantic phase sequence differs: wiping adds
the press, wipe, lift-off, and return-trip phases.

This module deliberately imports no simulator code so the contract can be unit
tested without SuperDex.
"""

from __future__ import annotations

from scenarios.ball_bowl.episode import (
    EpisodePolicy,
    PlanningError,
    PolicyOptions,
    load_policy_class,
)

TASK_PHASES: tuple[str, ...] = (
    "home",
    "preshape",
    "approach",
    "pre_grasp",
    "grasp",
    "lift",
    "carry",
    "lower",
    "wipe",
    "lift_off",
    "carry_back",
    "lower_back",
    "release",
    "retreat",
    "return_home",
)
"""Semantic task phases, in order; every embodiment marks all of them."""

__all__ = [
    "TASK_PHASES",
    "EpisodePolicy",
    "PlanningError",
    "PolicyOptions",
    "load_policy_class",
]
