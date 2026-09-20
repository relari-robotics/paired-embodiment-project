"""Embodiment-neutral ball-and-bowl manipulation scenario.

Simulator-dependent names are imported lazily so the episode contract
(``episode.py``) and the blank human policy can be unit tested without
SuperDex.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .embodiments import EMBODIMENTS
    from .scenario import BallBowlScenario

__all__ = ["EMBODIMENTS", "BallBowlScenario"]


def __getattr__(name: str) -> object:
    if name == "EMBODIMENTS":
        from . import embodiments

        return embodiments.EMBODIMENTS
    if name == "BallBowlScenario":
        from . import scenario

        return scenario.BallBowlScenario
    raise AttributeError(name)
