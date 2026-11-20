"""6-D wrench (force + torque) per instrument, as a wrist force/torque sensor would report it.

Convention (same as the per-part columns): the wrench the instrument applies to the sponge,
the force in N and the torque in N m about the instrument's pivot. A physical F/T sensor on the
tool measures the reaction, i.e. the negative.

- grasper: left jaw + right jaw + grasp attachment springs, about the shared jaw pivot.
  The attachment moment is logged about the world origin (grasp_moment) and is moved to the
  pivot: T_P = T_O - P x F.
- scalpel: the blade body (blade + cap), about the cap pin (the blade's yaw pivot).

Two frames:
- world: SOFA axes (x, y, z; z up).
- tool: the instrument's body frame (R^T applied). Grasper: the jaw body frame (the halfway
  rotation between the two jaws, so the opening angle cancels). Scalpel: the blade frame
  (z from the tip towards the cap, x = cutting edge direction, y = blade flat normal).
"""

import numpy as np

import tools


def _exp(v):
    """Rotation matrix of a rotation vector (Rodrigues)."""
    a = np.linalg.norm(v)
    if a < 1e-15:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _rots(pose):
    return np.array([tools.matrix_from_quat(q) for q in pose[:, 3:]])


def wrenches(d):
    """{"grasper"|"scalpel": dict(F, T, F_tool, T_tool, pivot, quat)}, each (T, 3) or (T, 4)."""
    out = {}
    if "jaw_left_torque" in d and "jaw_right_torque" in d:
        pl, pr = d["jaw_left_pose"][:, :3], d["jaw_right_pose"][:, :3]
        P = 0.5 * (pl + pr)
        Fa = d["grasp_force"]
        F = d["jaw_left_force"] + d["jaw_right_force"] + Fa
        T = (d["jaw_left_torque"] + np.cross(pl - P, d["jaw_left_force"])
             + d["jaw_right_torque"] + np.cross(pr - P, d["jaw_right_force"])
             + d["grasp_moment"] - np.cross(P, Fa))
        RL, RR = _rots(d["jaw_left_pose"]), _rots(d["jaw_right_pose"])
        R = np.array([rl @ _exp(0.5 * tools.rotvec(rl.T @ rr)) for rl, rr in zip(RL, RR)])
        out["grasper"] = dict(F=F, T=T, R=R, pivot=P)
    if "blade_torque" in d:
        out["scalpel"] = dict(F=d["blade_force"], T=d["blade_torque"], R=_rots(d["blade_pose"]),
                              pivot=d["blade_pose"][:, :3])
    for w in out.values():
        w["F_tool"] = np.einsum("tji,tj->ti", w["R"], w["F"])      # R^T F
        w["T_tool"] = np.einsum("tji,tj->ti", w["R"], w["T"])
        w["quat"] = np.array([tools.quat_from_matrix(r) for r in w["R"]])
    return out


def columns(d, rows=slice(None)):
    """(list of (n, k) arrays, list of column names) of every instrument's 6-D wrench.

    Names: <tool>_{Fx,Fy,Fz}_N, <tool>_{Tx,Ty,Tz}_Nm (world frame), the same with a _tool
    suffix (tool frame), then the pivot <tool>_pivot_{x,y,z}_m and <tool>_q{x,y,z,w}.
    """
    cols, names = [], []
    for tool, w in wrenches(d).items():
        cols += [w["F"][rows], w["T"][rows], w["F_tool"][rows], w["T_tool"][rows], w["pivot"][rows],
                 w["quat"][rows]]
        names += ([f"{tool}_F{a}_N" for a in "xyz"] + [f"{tool}_T{a}_Nm" for a in "xyz"]
                  + [f"{tool}_F{a}_tool_N" for a in "xyz"] + [f"{tool}_T{a}_tool_Nm" for a in "xyz"]
                  + [f"{tool}_pivot_{a}_m" for a in "xyz"] + [f"{tool}_q{a}" for a in "xyzw"])
    return cols, names
