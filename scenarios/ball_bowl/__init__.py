"""Embodiment-neutral ball-and-bowl manipulation scenario.

Simulator-dependent names are imported lazily so the episode contract
(``episode.py``) and the blank project policy can be unit tested without
SuperDex.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scenario import (
        EMBODIMENTS,
        BallBowlScenario,
        HumanBallBowlScenario,
        OpenArmBallBowlScenario,
    )

__all__ = [
    "EMBODIMENTS",
    "BallBowlScenario",
    "HumanBallBowlScenario",
    "OpenArmBallBowlScenario",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from . import scenario

        return getattr(scenario, name)
    raise AttributeError(name)
