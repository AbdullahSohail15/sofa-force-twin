"""Static balance checks on a recorded tool run (scenes 04-05).

Force: the tools' vertical force on the sponge equals the sum of the cells minus
its weight. Moment (horizontal axes, about the origin at the sponge bottom): the
cell springs, the grasp springs, the tool wrenches (pivot x force + torque about
the pivot) and the weight at the current centre of mass sum to zero. The moment
check exercises the tools' torque sensors.

    python scripts/check_balance.py --run outputs/rig_tools
"""

import argparse
import os

import numpy as np


def balance(d):
    t, pos, tets, rest = d["t"], d["pos"].astype(float), d["tets"], d["rest"]
    cut_step = d["cut_step"] if "cut_step" in d else np.full(len(tets), -1)
    mass = d["mass_now"] if "mass_now" in d else np.full(len(t), float(d["mass"]))
    g = d["gravity"]
    parts = [str(p) for p in d["part_names"]]
    a, b, c, e = (rest[tets[:, k]] for k in range(4))
    vol = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), e - a)) / 6     # rest volume: mass per element

    tool_z = sum(d[p + "_force"][:, 2] for p in parts) + d["grasp_force"][:, 2]
    f_err = -tool_z - (d["cells"].sum(1) - mass * -g[2])
    m_err = np.zeros((len(t), 3))
    for i in range(len(t)):
        alive = (cut_step < 0) | (cut_step > i)
        com = (vol[alive, None] * pos[i][tets[alive]].mean(1)).sum(0) / vol[alive].sum()
        m = d["cells_moment"][i] + d["grasp_moment"][i] + np.cross(com, mass[i] * g)
        for p in parts:
            m += np.cross(d[p + "_pose"][i, :3], d[p + "_force"][i]) + d[p + "_torque"][i]
        m_err[i] = m
    return t, f_err, m_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/rig_tools")
    args = parser.parse_args()
    d = np.load(os.path.join(args.run, "sim.npz"))
    t, f_err, m_err = balance(d)
    parts = [str(p) for p in d["part_names"]]
    tool_m = np.array([np.linalg.norm(sum(np.cross(d[p + "_pose"][i, :3], d[p + "_force"][i]) + d[p + "_torque"][i]
                                          for p in parts)[:2]) for i in range(len(t))])
    mxy = np.linalg.norm(m_err[:, :2], axis=1)
    print(f"force  error: median {np.median(np.abs(f_err)) * 1e3:.4f} mN, max {np.abs(f_err).max() * 1e3:.3f} mN "
          f"at t={t[np.abs(f_err).argmax()]:.2f}s")
    print(f"moment error (x,y): median {np.median(mxy) * 1e3:.4f} N mm, max {mxy.max() * 1e3:.3f} N mm "
          f"at t={t[mxy.argmax()]:.2f}s; tool moments reach {tool_m.max() * 1e3:.1f} N mm")
    torques = {p: np.abs(d[p + "_torque"]).max() * 1e3 for p in parts}
    print("largest sensor torque per part [N mm]:", {p: round(v, 2) for p, v in torques.items()})


if __name__ == "__main__":
    main()
