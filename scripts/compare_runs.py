"""Compare two runs of the same scene (e.g. two machines, two SOFA versions, or a solver change).

    python scripts/compare_runs.py --a outputs/rig_cut --b outputs/rig_cut_other

Reports the cut, the load cells, the tool sensors and the node positions, step by step.
"""
import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="reference run directory (contains sim.npz)")
    parser.add_argument("--b", required=True, help="run directory to compare")
    args = parser.parse_args()
    a = np.load(os.path.join(args.a, "sim.npz"))
    b = np.load(os.path.join(args.b, "sim.npz"))
    n = min(len(a["t"]), len(b["t"]))
    print(f"steps: a {len(a['t'])}, b {len(b['t'])}; comparing {n}")

    if "cut_step" in a.files and "cut_step" in b.files:
        ca, cb = a["cut_step"] >= 0, b["cut_step"] >= 0
        both = ca & cb
        dstep = np.abs(a["cut_step"][both] - b["cut_step"][both])
        print(f"cut tets: a {ca.sum()}, b {cb.sum()}, both {both.sum()}, only a {(ca & ~cb).sum()}, only b {(cb & ~ca).sum()}; "
              f"cut step difference median {np.median(dstep) if both.any() else 0:.0f}, max {dstep.max() if both.any() else 0}")
        print(f"mass at end: a {a['mass_now'][n - 1] * 1e3:.4f} g, b {b['mass_now'][n - 1] * 1e3:.4f} g")

    if "tie_step" in a.files and "tie_step" in b.files:
        ca, cb = a["tie_step"] >= 0, b["tie_step"] >= 0
        both = ca & cb
        dstep = np.abs(a["tie_step"][both] - b["tie_step"][both])
        print(f"cut ties: a {ca.sum()}, b {cb.sum()}, both {both.sum()}, only a {(ca & ~cb).sum()}, only b {(cb & ~ca).sum()}; "
              f"cut step difference median {np.median(dstep) if both.any() else 0:.0f}, max {dstep.max() if both.any() else 0}")

    def row(name, xa, xb, unit, scale):
        d = np.abs(xa[:n] - xb[:n])
        d = d.reshape(n, -1).max(axis=1)
        peak = np.abs(xa[:n]).max()
        print(f"  {name:<18} max |a-b| {d.max() * scale:9.4f} {unit} (median {np.median(d) * scale:8.4f})  "
              f"peak |a| {peak * scale:8.4f} {unit}")

    print("signals:")
    row("cells (8 ch)", a["cells"], b["cells"], "N", 1)
    row("sum of cells", a["cells"].sum(1), b["cells"].sum(1), "N", 1)
    for p in ("jaw_left", "jaw_right", "blade"):
        if f"{p}_force" in a.files:
            row(f"{p} force", a[f"{p}_force"], b[f"{p}_force"], "N", 1)
            row(f"{p} torque", a[f"{p}_torque"], b[f"{p}_torque"], "N mm", 1e3)
    row("grasp force", a["grasp_force"], b["grasp_force"], "N", 1)
    row("node positions", a["pos"], b["pos"], "mm", 1e3)
    if "blade_push" in a.files:
        row("blade push", a["blade_push"], b["blade_push"], "N", 1)


if __name__ == "__main__":
    main()
