"""Shared episode policy contract; organizer phases repeat once per object."""

from dataclasses import dataclass
from scenarios.ball_bowl.episode import EpisodePolicy, PlanningError


@dataclass(frozen=True)
class PolicyOptions:
    check_collisions: bool = True


__all__ = ["EpisodePolicy", "PlanningError", "PolicyOptions"]
