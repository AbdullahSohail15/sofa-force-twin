"""Basic hard-coded script: press the scalpel into the sponge, drag it, lift it out.

This is the smallest complete use of the rig (scenes/rig.py) and the tool machinery
(scenes/rig_tools.py). Copy it and edit the block marked EDIT HERE to make your own motion.

- The grasper is parked 20 mm above a corner and never grasps.
- The scalpel follows SCALPEL_PATH: a list of (time [s], x, y, z of the scalpel point [m]),
  linearly interpolated between waypoints. Heights are relative to the undeformed top surface,
  so z < 0 means "inside the sponge".
- Every step SOFA records the scalpel's 6-axis wrench and the eight load cells.

Outputs (in --out):
    sim.npz      every recorded signal (same format as scene 04, so render_tools.py and
                 check_balance.py work on it)
    press_forces.csv   time, scalpel force and torque, eight cells, and the force balance
    press_forces.png   the same as a plot
(render_tools.py adds the full forces.csv, forces.png and an mp4 next to them)

    python examples/basic_press.py --out outputs/basic_press
    python scripts/check_balance.py --run outputs/basic_press
    python scripts/render_tools.py --run outputs/basic_press      # optional mp4

Frame: origin at the centre of the sponge bottom, x along the 86 mm side, y along the
71 mm side, z up. Units are SI (m, s, N). The sponge top is at z = rig_tools.TOP (21 mm).
"""

import argparse
import os
import sys
import time

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scenes"))
import rig  # noqa: E402
import rig_tools  # noqa: E402
from block_on_supports import GRAVITY  # noqa: E402

# ---------------------------------------------------------------- EDIT HERE ----------------
X0, Y0 = 0.010, -0.010        # where the scalpel goes in [m] (sponge is +-43 x +-35.5 mm)
DEPTH = 0.003                 # how far the point goes below the top surface [m]
DRAG = 0.012                  # drag length along +y while inside [m]
SCALPEL_PATH = [              # (time [s], x [m], y [m], height of the point above the top [m])
    (0.0, X0, Y0, 0.005),         # start 5 mm above the surface
    (0.5, X0, Y0, 0.005),         # let the sponge settle under gravity
    (1.5, X0, Y0, -DEPTH),        # press in
    (2.0, X0, Y0, -DEPTH),        # hold
    (3.5, X0, Y0 + DRAG, -DEPTH), # drag along +y
    (4.0, X0, Y0 + DRAG, -DEPTH), # hold
    (4.8, X0, Y0 + DRAG, 0.005),  # lift out
    (5.5, X0, Y0 + DRAG, 0.005),  # rest
]
FRICTION = 1.0                # Coulomb friction between tools and foam
rig.YOUNG_BASE = 30e3         # cream foam Young's modulus [Pa] (guess, to be calibrated)
rig.YOUNG_SCRUB = 60e3        # orange scrub layer Young's modulus [Pa]
# ---------------------------------------------------------------------------------------------

T_END = SCALPEL_PATH[-1][0]


def scalpel_point(t):
    """Commanded position of the scalpel point at time t (world frame)."""
    x, y, h = rig_tools.interp(SCALPEL_PATH, t)
    return np.array([x, y, rig_tools.TOP + h])


def park_grasper():
    """Keep the grasper open, 20 mm above the top-left corner, for the whole run."""
    rig_tools.GRASP_XY = np.array([-0.035, 0.028])
    rig_tools.GRASP_PATH = [(0.0, 0.020, rig_tools.G_OPEN), (1e6, 0.020, rig_tools.G_OPEN)]
    rig_tools.T_GRASP = rig_tools.T_RELEASE = 1e6      # never grasp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/basic_press")
    parser.add_argument("--t-end", type=float, default=T_END)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. Build the scene: sponge + eight load cells + grasper (parked) + scalpel.
    park_grasper()
    rig_tools.FRICTION = FRICTION
    parts = rig_tools.make_parts(blade=scalpel_point)
    root = Sofa.Core.Node("root")
    rig_tools.createScene(root, parts)
    Sofa.Simulation.init(root)
    rig_tools.check_parts(root, parts)        # SOFA ignores misspelt fields silently: assert they took
    mass = float(root.Sponge.mass.totalMass.value)
    print(f"sponge: {len(root.Sponge.dofs.position.value)} nodes, {mass * 1e3:.2f} g")

    # 2. Step the simulation. The controller (rig_tools.Recorder) moves the tools to their commanded
    #    poses before each step and records forces after it.
    n = int(round(args.t_end / rig_tools.DT))
    t0 = time.time()
    for k in range(n):
        Sofa.Simulation.animate(root, rig_tools.DT)
        if k % 50 == 0:
            log = root.controller.log
            print(f"  t={log['t'][-1]:.2f} s  scalpel Fz {log['blade_force'][-1][2]:+.3f} N  "
                  f"cells sum {np.sum(log['cells'][-1]):+.3f} N  ({time.time() - t0:.0f} s wall)", flush=True)

    # 3. Save. sim.npz uses the same keys as scene 04 so the render and check scripts read it.
    log = {k: np.array(v) for k, v in root.controller.log.items()}
    log["pos"] = log["pos"].astype(np.float32)
    points = np.array(root.Sponge.dofs.rest_position.value)
    tets = np.array(root.Sponge.topo.tetrahedra.value)
    young = np.array(root.Sponge.fem.youngModulus.value)
    np.savez_compressed(
        os.path.join(args.out, "sim.npz"), **log,
        rest=points, tets=tets, scrub=young > rig.YOUNG_BASE, mass=mass, dt=rig_tools.DT,
        gravity=np.array(GRAVITY), cell_names=np.array([c[0] for c in rig.CELLS]),
        part_names=np.array([p.name for p in parts]), **rig_tools.tool_meshes(parts),
        grasp_cmd=np.array([np.r_[rig_tools.grasper_pose(t)[0], rig_tools.grasper_pose(t)[1]] for t in log["t"]]),
        blade_cmd=np.array([scalpel_point(t) for t in log["t"]]),
    )

    # press_forces.csv: force the scalpel applies to the sponge, its torque about the scalpel's wrist pivot,
    # the eight cell readings, and the vertical balance error (should be ~0 when nothing moves fast).
    t, F, T, C = log["t"], log["blade_force"], log["blade_torque"], log["cells"]
    weight = mass * -GRAVITY[2]
    tools_z = F[:, 2] + log["jaw_left_force"][:, 2] + log["jaw_right_force"][:, 2] + log["grasp_force"][:, 2]
    balance = -tools_z - (C.sum(1) - weight)
    names = [c[0] for c in rig.CELLS]
    header = ("t_s,scalpel_fx_N,scalpel_fy_N,scalpel_fz_N,scalpel_tx_Nm,scalpel_ty_Nm,scalpel_tz_Nm,"
              + ",".join(f"{c}_N" for c in names) + ",cells_sum_N,balance_err_N")
    np.savetxt(os.path.join(args.out, "press_forces.csv"),
               np.column_stack([t, F, T, C, C.sum(1), balance]), delimiter=",", header=header, comments="",
               fmt="%.6g")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, a in enumerate("xyz"):
        ax[0].plot(t, F[:, i], label=f"scalpel F{a}")
    ax[0].plot(t, C.sum(1) - weight, "k--", label="cells sum - weight")
    ax[0].set_ylabel("force on sponge [N]")
    ax[0].legend(fontsize=8)
    for i, c in enumerate(names):
        ax[1].plot(t, C[:, i], label=c)
    ax[1].set_ylabel("cell reading [N]")
    ax[1].set_xlabel("time [s]")
    ax[1].legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "press_forces.png"), dpi=120)

    i = int(np.argmax(-F[:, 2]))
    print(f"peak downward scalpel force {-F[i, 2]:.3f} N at t={t[i]:.2f} s; "
          f"worst balance error {np.abs(balance).max() * 1e3:.1f} mN")
    print(f"saved {args.out}/sim.npz, press_forces.csv, press_forces.png  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
