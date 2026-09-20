"""Deterministic visual-only standing figure placement and two-segment arm IK.

All coordinates are metres, Z-up. The figure faces -X at the work surface.
Only the inferred elbow moves; body placement is computed from the first frame.
"""
import json
from pathlib import Path
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / 'embodiments/assets/artec_figure'
ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
UPPER_LENGTH = .32
FORE_LENGTH = .29


def placement(wrist, shoulder=None):
    anchor = np.array(json.loads((ASSETS / 'placement.json').read_text())['standing']['right_shoulder'])
    if shoulder is None:
        shoulder = np.array([wrist[0] + .22, wrist[1], anchor[2]])
    else:
        shoulder = np.asarray(shoulder, dtype=float)
    origin = shoulder - ROTATION @ anchor
    return origin, shoulder


def arm_joints(shoulder, wrist):
    shoulder, wrist = np.asarray(shoulder), np.asarray(wrist)
    delta = wrist - shoulder
    distance = max(float(np.linalg.norm(delta)), 1e-8)
    direction = delta / distance
    pole = np.array([0., 1., -.35])
    pole -= direction * np.dot(pole, direction)
    if np.linalg.norm(pole) < 1e-6:
        pole = np.cross(direction, [1., 0., 0.])
    pole /= np.linalg.norm(pole)
    # Unreachable targets remain attached, with extension confined to the arm.
    extension = max(1., distance / (UPPER_LENGTH + FORE_LENGTH - .001))
    upper, fore = UPPER_LENGTH * extension, FORE_LENGTH * extension
    along = (upper*upper - fore*fore + distance*distance) / (2*distance)
    along = np.clip(along, -upper, upper)
    elbow = shoulder + direction * along + pole * np.sqrt(max(0., upper*upper-along*along))
    return elbow


def segment_matrix(head, tail, lateral):
    x = np.asarray(tail) - head
    x /= np.linalg.norm(x)
    y = np.asarray(lateral) - x * np.dot(lateral, x)
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(x, np.eye(3)[np.argmin(np.abs(x))])
    y /= np.linalg.norm(y)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
    matrix[:3, 3] = head
    return matrix
