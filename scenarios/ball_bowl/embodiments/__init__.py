"""Embodiments that can perform the ball-and-bowl task, one subpackage each.

A subpackage holds everything specific to one body: how it is built
(``embodiment.py``), its policies, its teleoperation mapping, and its Studio
render scene.  The task itself (:mod:`scenarios.ball_bowl.scenario`) and the
runner are shared.

:data:`EMBODIMENTS` is assembled lazily so a policy module can be imported, and
unit tested, without SuperDex.
"""

from __future__ import annotations

DEFAULT_EMBODIMENT = "openarm_v2_bimanual"

__all__ = ["DEFAULT_EMBODIMENT", "EMBODIMENTS"]


def __getattr__(name: str) -> object:
    if name == "EMBODIMENTS":
        from .human_right_hand.embodiment import EMBODIMENTS as human
        from .openarm_v2.embodiment import EMBODIMENTS as openarm

        embodiments = {**openarm, **human}
        globals()[name] = embodiments
        return embodiments
    raise AttributeError(name)
