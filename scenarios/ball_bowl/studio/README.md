# Ball-and-bowl Studio assets

This folder contains self-contained public SuperDex Studio manifests for both
physical embodiments:

- `scene/openarm_ball_bowl_studio.mochi_scene`
- `human_scene/human_ball_bowl_studio.mochi_scene`

Each bundle includes the embodiment, wooden desk, tennis-size ball, bowl, ground
collision actor, generated Mochi collision assets, and copied GLB render assets.
The OpenArm manifest contains the complete robot. The human manifest contains
the fixed-shoulder seven-DOF right arm and articulated Meta XR right hand.

Rebuild them against the installed SuperDex asset set:

```bash
cd /path/to/project_superdex
uv run --no-project \
  /path/to/superdex-scenarios/scenarios/ball_bowl/studio/build_studio_scene.py
uv run --no-project \
  /path/to/superdex-scenarios/scenarios/ball_bowl/studio/build_studio_scene.py --human
```

The builder stages, reloads, and instantiates each exported scene before
replacing the checked-in bundle. It therefore validates that all manifest and
render-asset paths resolve through the released prefab loader.

Studio assets are useful for visual inspection. The executable task—including
fresh randomization, TrajOpt, compliant physical control, telemetry, and camera
exports—lives in `../runner.py`. The generic PBR viewer is under
`superdex_scenarios/rendering/pbr` at the repository root.
