# Physics model and limitations

The scene runs at 400 Hz under gravity with BDF2 integration and finite,
effort-limited compliant pose tracking. Nothing welds, parents, teleports, or
otherwise attaches the ball to an end effector.

## Workcell

The ball is a rigid 67 mm shell with tennis-ball shell inertia. Mass is sampled
from 50–500 g. Contact uses finite friction, a 0.5 mm smoothing distance, and a
0.2 mm contact threshold. The bowl uses open paper-cup collision geometry at a
randomized anisotropic scale. The desk and bowl are rigid bodies. Soft tissue,
shell indentation, ceramic compliance, air drag, and rolling resistance are not
modeled.

## OpenArm v2 reference

The complete robot asset is imported. Only the seven right-arm and two
right-gripper DOFs receive controller targets. Right-side links retain gravity
and finite effort limits. The left side remains collision geometry. The included
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
