"""Open-loop joint-trajectory replay shared by every scenario.

``trajectory`` defines the file format and is simulator-free; ``policy`` is the
episode policy that executes a file; ``cli`` wires both into a scenario runner.
"""

from .trajectory import (
    FORMAT,
    JointTrajectory,
    PhaseMark,
    aperture_from_finger_joints,
    finger_joints_from_aperture,
    phase_sequence_for,
)

__all__ = [
    "FORMAT",
    "JointTrajectory",
    "PhaseMark",
    "aperture_from_finger_joints",
    "finger_joints_from_aperture",
    "phase_sequence_for",
]
