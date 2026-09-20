"""OpenArm v2: the right arm alone, or both arms.

``policy`` picks the ball up with the right arm.  ``bimanual_policy`` first
drags the bowl to its target with the left arm; the runner selects it when the
specification has a ``bowl_target_xy``.  ``teleop_policy`` drives both arms
from live Kyber packets through ``teleop_mapping.json``.
"""
