"""Instrument tips for the rig, from the da Vinci Research Kit (dVRK) model.

- Grasper: Large Needle Driver (EndoWrist 400006), two jaws.
- Knife: a #11 scalpel blade in a cap mounted on the EndoWrist wrist (scripts/make_scalpel.py).

The meshes come from https://github.com/jhu-dvrk/dvrk_model (MIT licence) and are
converted by scripts/prepare_dvrk_tools.py into assets/dvrk_tools.npz. Each part
has a render mesh (~6000 triangles) and a simplified collision mesh (~400).

Tool frame (every part's body frame at zero jaw angle): origin on the jaw pivot,
jaws pointing to -z, pivot axis along y, jaws closing along x. In the scene each
part is a rigid body tied to its commanded pose by a 6-DOF spring; the spring
wrench is the part's force/torque sensor (see rig_tools.py).
"""

import os

import numpy as np

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
ASSETS = np.load(os.path.join(_ASSET_DIR, "dvrk_tools.npz"))
# Knife: a #11 scalpel blade in a cap on the EndoWrist yaw pivot (scripts/make_scalpel.py).
SCALPEL = np.load(os.path.join(_ASSET_DIR, "scalpel.npz"))

JAW_LENGTH = 0.00965             # LND pivot to jaw tip [m]
JAW_TIP_OFFSET = 0.00064         # tip centre off the closing plane at zero angle
JAW_WIDTH = 0.003                # jaw extent along the pivot axis
JAW_OPEN = np.radians(35.0)      # per jaw; tip separation ~12 mm
JAW_CLOSED = np.radians(3.0)     # per jaw; a little foam between the jaws
# Knife tip in its body frame (scalpel point; pivot origin, tool along -z). The blade is a plate in
# the x-z plane: flat faces +-y, cutting edge facing +x.
KNIFE_TIP = SCALPEL["scalpel_tip"]
KNIFE_LENGTH = float(-KNIFE_TIP[2])  # pivot to the point along the tool axis
WRIST_LIMIT = 1.39626            # wrist pitch and yaw joint limits [rad] (dVRK URDF)


def mesh(part, kind):
    """(points, triangles) of a part in its body frame; kind is "col" or "vis"."""
    src = SCALPEL if part.startswith("scalpel") else ASSETS
    return src[f"{part}_{kind}_pts"], src[f"{part}_{kind}_tris"]


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def jaw_rotation(side, angle):
    """Body rotation of a jaw: side -1 (left, dVRK jaw 1) swings to -x, +1 to +x."""
    return rot_y(-side * angle)


def tip_separation(angle):
    return 2 * (JAW_TIP_OFFSET * np.cos(angle) + JAW_LENGTH * np.sin(angle))


def quat_from_matrix(r):
    """Unit quaternion (x, y, z, w), SOFA's order, of a rotation matrix."""
    w = np.sqrt(max(0.0, 1.0 + np.trace(r))) / 2
    if w > 1e-6:
        q = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], 4 * w * w]) / (4 * w)
    else:                                    # rotation by ~pi
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = np.zeros(4)
        q[i], q[j], q[k], q[3] = s / 4, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s, (r[k, j] - r[j, k]) / s
    return q / np.linalg.norm(q)


def matrix_from_quat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec(r):
    """Rotation vector (axis * angle) of a rotation matrix."""
    q = quat_from_matrix(r)
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    return np.zeros(3) if s < 1e-15 else q[:3] / s * 2 * np.arctan2(s, q[3])


def rigid(pos, rot):
    """SOFA Rigid3d coordinate [x, y, z, qx, qy, qz, qw]."""
    return np.r_[pos, quat_from_matrix(rot)]


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _wrist_frames(base, q):
    """World rotations of the roll, pitch and jaw (tool) frames for joints q = (roll, pitch, yaw).

    `base` is the roll link frame at zero roll. Chain from the dVRK URDF:
    roll link -Rz(roll)- pitch joint origin -Rz(pitch)- yaw joint origin -Rz(yaw)- jaw link.
    """
    roll = base @ rot_z(q[0])
    pitch = roll @ ASSETS["pitch_joint"][:3, :3] @ rot_z(q[1])
    jaw = pitch @ ASSETS["yaw_joint"][:3, :3] @ rot_z(q[2]) @ ASSETS["r_tool"].T
    return roll, pitch, jaw


def _turn(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else rot_y(np.pi)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def wrist(shaft_dir, body_rot=np.eye(3)):
    """Wrist joint angles that keep a tool's body orientation with the shaft along `shaft_dir`.

    shaft_dir points from the wrist up the shaft towards the arm. Returns the joints
    (roll, pitch, yaw) [rad], the pitch link (clevis) rotation, and the pitch axis
    point relative to the jaw pivot. Solved numerically (coarse grid, then Gauss-Newton),
    preferring the least wrist bend.
    """
    d = np.asarray(shaft_dir, float) / np.linalg.norm(shaft_dir)
    # Roll link at zero roll: the straight-wrist roll frame, turned so its z (towards the tip) is -d.
    zero = ASSETS["r_tool"] @ np.linalg.inv(ASSETS["yaw_joint"])[:3, :3] \
        @ np.linalg.inv(ASSETS["pitch_joint"])[:3, :3]
    base = _turn(zero[:, 2], -d) @ zero

    def residual(q):
        return (_wrist_frames(base, q)[2] - body_rot).ravel()

    grid = np.linspace(-np.pi, np.pi, 37)
    bend = np.linspace(-WRIST_LIMIT, WRIST_LIMIT, 17)
    starts = sorted(((np.linalg.norm(residual((r, p, y))), (r, p, y)) for r in grid for p in bend for y in bend),
                    key=lambda e: e[0])[:20]
    best = None
    for _, q in starts:
        q = np.array(q, float)
        for _ in range(50):
            r = residual(q)
            jac = np.column_stack([(residual(q + h) - r) / 1e-7 for h in np.eye(3) * 1e-7])
            q -= np.linalg.lstsq(jac, r, rcond=None)[0]
        q[0] = (q[0] + np.pi) % (2 * np.pi) - np.pi
        if np.linalg.norm(residual(q)) < 1e-9 and np.all(np.abs(q[1:]) <= WRIST_LIMIT):
            if best is None or np.abs(q[1:]).sum() < np.abs(best[1:]).sum():
                best = q
    if best is None:
        raise ValueError(f"shaft direction {d} needs a wrist bend beyond the joint limits")
    _, pitch, _ = _wrist_frames(base, best)
    pitch_point = -pitch @ ASSETS["yaw_joint"][:3, 3]          # yaw axis is 9.1 mm along the pitch link
    return best, pitch, pitch_point
