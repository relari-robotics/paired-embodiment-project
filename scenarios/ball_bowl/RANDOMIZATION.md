# Scenario randomization

The programmatic runner samples a new task on every invocation and prints the
seed plus the complete specification before executing. A seed is sufficient to
reproduce the mass, colors, bowl geometry, object poses, task targets, and fresh
TrajOpt result:

```bash
uv run --no-project scenarios/ball_bowl/runner.py --headless --seed 1234
uv run --no-project scenarios/ball_bowl/runner.py --human --plan-only --seed 1234
```

Use `--fixed` for the original deterministic scene. `--headless` automatically
writes a uniquely named export bundle, including `scenario.json`, and embeds the
sample in all higher-level recordings. Use `--dry-run` for a full simulation
without persisted output.

## Distributions

| Parameter | Distribution |
| --- | --- |
| Ball mass | Uniform over 50, 100, ..., 500 g |
| Ball color | Uniform over blue, red, yellow, green, orange, and purple |
| Bowl color | Uniform over gray, ivory, teal, terracotta, navy, and mustard |
| Bowl shape | Uniform over shallow round, deep round, X-oval, and Y-oval |
| Ball X/Y | Continuous uniform in `[-0.015, 0.030] × [-0.310, -0.190] m` |
| Bowl X/Y | Continuous uniform in `[-0.140, 0.015] × [-0.035, 0.065] m` |

The bowl variants scale the same open collision/render mesh. Their scales are
`[2.40, 2.40, 0.35]`, `[2.55, 2.55, 0.55]`, `[2.90, 2.40, 0.34]`, and
`[2.40, 2.90, 0.34]`. Consequently, both the physical geometry and TrajOpt rim
proxy change, including elliptical X/Y radii and the rim height.

## Validity and planning

Sampling is limited to a conservatively tested subset of the shared right-arm tabletop
workspace. The ball and bowl must also be separated by the sampled bowl's
largest outer radius, the ball radius, and a 45 mm buffer. The ball band excludes
the gripper's parked home footprint, so the robot cannot disturb it before the
planned approach.

For OpenArm, each candidate constructs continuous-IK references and runs all
seven TrajOpt segments from scratch. A candidate is accepted only when IK and
collision validation succeed; invalid samples advance the seeded generator for
up to 24 attempts. The human scaffold builds the first deterministic sample and
does not judge solvability, leaving that policy to the project.

The ball remains a 67 mm rigid sphere. Its shell inertia is recomputed from the
sampled mass. Pickup is always contact-only: no weld, teleport, disabled contact,
or forced ball velocity is used.
