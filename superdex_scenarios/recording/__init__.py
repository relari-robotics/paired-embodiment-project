"""Reusable recording utilities with lazy simulator-dependent imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .phases import PHASE_SEQUENCE, PhaseLog

if TYPE_CHECKING:
    from .telemetry import TelemetryRecorder
    from .transforms import TransformRecorder

__all__ = ["PHASE_SEQUENCE", "PhaseLog", "TelemetryRecorder", "TransformRecorder"]


def __getattr__(name: str) -> object:
    if name == "TelemetryRecorder":
        from .telemetry import TelemetryRecorder

        return TelemetryRecorder
    if name == "TransformRecorder":
        from .transforms import TransformRecorder

        return TransformRecorder
    raise AttributeError(name)
