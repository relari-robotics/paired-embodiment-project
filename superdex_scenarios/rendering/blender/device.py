"""Cross-platform Cycles device selection.

The functions here are called inside Blender, but do not import ``bpy`` at
module import time. This keeps the backend policy reusable and makes the
selection rules easy to test with small fakes.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

DEVICE_AUTO = "AUTO"
DEVICE_CPU = "CPU"
DEVICE_OPTIX = "OPTIX"
DEVICE_CUDA = "CUDA"
DEVICE_METAL = "METAL"

SUPPORTED_DEVICES = (DEVICE_AUTO, DEVICE_OPTIX, DEVICE_CUDA, DEVICE_METAL, DEVICE_CPU)


@dataclass(frozen=True)
class DeviceSelection:
    """The backend and device names selected for a render."""

    backend: str
    devices: tuple[str, ...]


def _candidate_backends(requested: str) -> tuple[str, ...]:
    requested = requested.upper()
    if requested != DEVICE_AUTO:
        return (requested,)

    # OptiX is the preferred NVIDIA backend, while Metal is the native Apple
    # backend. They are mutually exclusive technologies; AUTO is the small
    # compatibility layer that hides that platform difference from callers.
    if platform.system() == "Darwin":
        return (DEVICE_METAL, DEVICE_OPTIX, DEVICE_CUDA)
    return (DEVICE_OPTIX, DEVICE_CUDA, DEVICE_METAL)


def _set_backend(preferences: Any, backend: str) -> bool:
    try:
        preferences.compute_device_type = backend
        preferences.refresh_devices()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False
    return True


def configure_cycles_device(
    preferences: Any,
    scene: Any,
    requested: str = DEVICE_AUTO,
    device_index: int | None = None,
) -> DeviceSelection:
    """Select a Cycles backend and devices, raising a useful error if needed.

    ``device_index`` addresses the GPU list exposed by the chosen backend. In
    a multi-process render, each worker selects one index so workers do not
    compete for the same GPU. With no index, all visible GPUs are enabled.
    """

    requested = requested.upper()
    if requested not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported Cycles device {requested!r}")
    if device_index is not None and device_index < 0:
        raise ValueError("device_index must be non-negative")

    if requested == DEVICE_CPU:
        for device in preferences.devices:
            device.use = False
        scene.cycles.device = DEVICE_CPU
        return DeviceSelection(DEVICE_CPU, ())

    for backend in _candidate_backends(requested):
        if not _set_backend(preferences, backend):
            continue
        devices = [device for device in preferences.devices if getattr(device, "type", "") != DEVICE_CPU]
        if not devices:
            continue
        if device_index is not None and device_index >= len(devices):
            raise RuntimeError(
                f"Cycles backend {backend} exposes {len(devices)} GPU(s), "
                f"but device index {device_index} was requested"
            )

        selected = devices if device_index is None else [devices[device_index]]
        selected_ids = {id(device) for device in selected}
        for device in preferences.devices:
            device.use = id(device) in selected_ids
        scene.cycles.device = "GPU"
        return DeviceSelection(backend, tuple(str(device.name) for device in selected))

    if requested == DEVICE_AUTO:
        # A CPU fallback makes headless exports portable to machines without a
        # compatible GPU. It is deliberately only an AUTO fallback; explicit
        # GPU requests should fail loudly instead of silently becoming slow.
        for device in preferences.devices:
            device.use = False
        scene.cycles.device = DEVICE_CPU
        return DeviceSelection(DEVICE_CPU, ())

    raise RuntimeError(
        f"Requested Cycles backend {requested} is unavailable. "
        "Install a compatible Blender build and GPU driver, or use --device AUTO."
    )
