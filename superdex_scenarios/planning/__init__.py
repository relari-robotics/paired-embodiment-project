"""Scenario-independent motion-planning utilities."""

from .trajopt import TrajOptTrajectoryOptimizer, sample_natural_cubic_spline

__all__ = [
    "TrajOptTrajectoryOptimizer",
    "sample_natural_cubic_spline",
]
