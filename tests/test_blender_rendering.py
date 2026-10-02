import unittest
from pathlib import Path

from superdex_scenarios.rendering.blender.device import configure_cycles_device
from superdex_scenarios.rendering.blender.make_video import render_worker_command
from superdex_scenarios.rendering.blender.render_plan import partition_frames


class FakeDevice:
    def __init__(self, name: str, device_type: str) -> None:
        self.name = name
        self.type = device_type
        self.use = False


class FakePreferences:
    def __init__(self) -> None:
        self.devices = [FakeDevice("CPU", "CPU"), FakeDevice("GPU 0", "OPTIX"), FakeDevice("GPU 1", "OPTIX")]
        self.compute_device_type = None

    def refresh_devices(self) -> None:
        pass


class FakeCycles:
    device = None


class FakeScene:
    cycles = FakeCycles()


class BlenderRenderingTests(unittest.TestCase):
    def test_partitions_are_contiguous_and_cover_all_frames(self) -> None:
        partitions = [partition_frames(10, index, 3) for index in range(3)]
        self.assertEqual([(part.start, part.stop) for part in partitions], [(0, 3), (3, 6), (6, 10)])
        self.assertEqual(sum(part.count for part in partitions), 10)

    def test_device_index_selects_only_one_gpu(self) -> None:
        preferences = FakePreferences()
        scene = FakeScene()
        selection = configure_cycles_device(preferences, scene, "OPTIX", device_index=1)
        self.assertEqual(selection.backend, "OPTIX")
        self.assertEqual(selection.devices, ("GPU 1",))
        self.assertEqual([device.use for device in preferences.devices], [False, False, True])
        self.assertEqual(scene.cycles.device, "GPU")

    def test_worker_command_has_disjoint_frame_worker_arguments(self) -> None:
        command = render_worker_command(
            blender="blender",
            recording=Path("/recording"),
            output=Path("/frames"),
            fps=24.0,
            passthrough=["--camera", "front"],
            worker_index=1,
            worker_count=3,
            device="OPTIX",
            device_index=0,
            resume=True,
            overwrite=False,
        )
        self.assertIn("--worker-index", command)
        self.assertIn("1", command)
        self.assertIn("--worker-count", command)
        self.assertIn("3", command)
        self.assertIn("--device-index", command)
        self.assertIn("--resume", command)


if __name__ == "__main__":
    unittest.main()
