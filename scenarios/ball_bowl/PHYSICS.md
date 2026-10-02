# Physics model and limitations

The scene runs at 400 Hz under gravity with BDF2 integration and finite,
effort-limited compliant pose tracking. Nothing welds, parents, teleports, or
otherwise attaches the ball to an end effector.

## Workcell

The ball is a rigid 67 mm shell with tennis-ball shell inertia. Mass is sampled
from 50–500 g. Contact uses finite friction, a 0.5 mm smoothing distance, and a
0.2 mm contact threshold. The bowl is a revolved profile measured from the
recorded bowl (175 mm across, 70 mm tall, flat floor, no lip) at a randomized
anisotropic scale. The desk is a static rigid body; the bowl is a dynamic one
(150-400 g, SDF collider), so it can be pushed, dragged by its rim, or tipped.
`--fixed` alone restores the original regression scene, whose bowl is static. Soft tissue,
shell indentation, ceramic compliance, air drag, and rolling resistance are not
modeled.

## OpenArm v2 reference

The complete robot asset is imported. By default both seven-DOF arms and
both two-jaw grippers receive controller targets and retain gravity and finite
effort limits; the single-arm reference policies hold the left arm at its
parked pose. With `--embodiment openarm_v2` only the right side is controlled
and the left side remains collision geometry. The included
reference uses continuation IK followed by global spline trajectory optimization
and executes the result with a minimum-jerk clock.

## Human hand scaffold

The human embodiment is the articulated 27-DOF Meta XR right hand on an
invisible, controlled six-DOF Cartesian carrier. The low-poly collision
decomposition shipped with SuperDex is used for real-time contact while the
high-detail render meshes are retained.

Adjacent anatomical collision shells overlap by construction, so internal
hand-link contacts are disabled. Every hand link still collides with the ball
and workcell. Finger joints use finite, effort-limited compliant tracking.

The carrier is a simulation mechanism, not an anatomical arm model. The
scaffold does not select a shoulder model, reach constraint, wrist orientation,
approach direction, grasp pose, trajectory generator, contact controller, or
failure-recovery policy. Those choices belong to the project.

The hand remains a mechanics-oriented asset rather than a subject-specific
biomechanical model. It does not model deformable skin, tendons, tactile pads,
nail compliance, muscle fatigue, or involuntary motion.
