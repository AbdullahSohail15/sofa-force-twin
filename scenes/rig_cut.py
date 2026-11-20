"""Scene 05: the blade cuts the sponge while the grasper holds it (scene 04 rig and tools).

The knife is the #11 scalpel from scene 04, its thin direction across the cut. Cutting has zero width (scenes/cutting.py): the sponge is
split along the knife plane x = CUT_X and the two sides are tied together; the blade releases
the ties it pushes through, the faces part around it, and the foam closes again behind it. No
material is removed, so there is no gap after the blade has left.

Limits: the knife can only cut along the pre-built plane; the two faces of the cut do not
collide with each other (they only meet again, they are never pressed into one another here);
F_CUT is a guess to be calibrated against real cuts.

    python scenes/rig_cut.py --out outputs/rig_cut
"""

import argparse
import os
import sys

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig  # noqa: E402
import rig_tools  # noqa: E402
from block_on_supports import GRAVITY  # noqa: E402
from cutting import TieCutRecorder  # noqa: E402
from rig_tools import DT, TOP, grasper_pose, interp  # noqa: E402

CUT_X = 0.022                    # knife plane (split plane of the sponge)
# Blade: (time, x, y, edge height above the undeformed top [m]). The grasper is scene 04's.
BLADE_PATH = [
    (0.0, CUT_X, -0.010, 0.005),
    (3.5, CUT_X, -0.010, 0.005),
    (4.5, CUT_X, -0.010, -0.008),  # plunge 8 mm, through the 6 mm scrub layer
    (6.0, CUT_X, 0.010, -0.008),   # slice 20 mm along the edge (+y)
    (6.8, CUT_X, 0.010, 0.005),    # pull out through the cut
    (9.0, CUT_X, 0.010, 0.005),
]
T_END = rig_tools.T_END


def blade_pose(t):
    x, y, h = interp(BLADE_PATH, t)
    return np.array([x, y, TOP + h])


def make_parts():
    return rig_tools.make_parts(blade=blade_pose)


def createScene(root, parts=None):
    parts = parts if parts is not None else make_parts()
    rig_tools.createScene(root, parts, cut_x=CUT_X, split=(0, CUT_X), recorder=TieCutRecorder,
                          blade_path=blade_pose)
    root.controller.ties = root.Sponge.ties
    return root


def run(root, parts, out, t_end, extra=None):
    """Simulate, then save sim.npz (shared with scene 06)."""
    Sofa.Simulation.init(root)
    rec = root.controller
    n_ties = len(rec.pairs)
    assert len(root.Sponge.ties.stiffness.value) == n_ties, "tie springs not loaded"
    rest = np.array(root.Sponge.dofs.position.value)
    tets = np.array(root.Sponge.topo.tetrahedra.value)
    assert len(root.Sponge.fem.youngModulus.value) == len(tets), "per-element Young's modulus ignored"
    mass = float(root.Sponge.mass.totalMass.value)
    print(f"sponge nodes {len(rest)}  tets {len(tets)}  ties {n_ties}  mass {mass * 1e3:.2f} g")
    for k in range(int(round(t_end / DT))):
        Sofa.Simulation.animate(root, DT)
        if k % 50 == 0:
            print(f"  t={k * DT:.2f}s  contact rows {len(root.constraints.constraintForces.value)}  "
                  f"ties cut {rec.log['ties_cut'][-1]}  blade push {rec.log['blade_push'][-1]:+.2f} N",
                  flush=True)
    out_d = {k: np.array(v) for k, v in rec.log.items()}
    out_d["pos"] = out_d["pos"].astype(np.float32)
    young = np.array(root.Sponge.fem.youngModulus.value)
    blade_rot = parts[2].pose(0.0)[1]
    np.savez_compressed(
        os.path.join(out, "sim.npz"), **out_d,
        rest=rest, tets=tets, scrub=young > rig.YOUNG_BASE, mass=mass, dt=DT, gravity=np.array(GRAVITY),
        tie_pairs=rec.pairs, tie_step=rec.tie_step, f_cut=rec.f_cut, blade_rot=blade_rot,
        cell_names=np.array([c[0] for c in rig.CELLS]), part_names=np.array([p.name for p in parts]),
        **rig_tools.tool_meshes(parts),
        grasp_cmd=np.array([np.r_[grasper_pose(t)[0], grasper_pose(t)[1]] for t in out_d["t"]]),
        blade_cmd=np.array([rec.blade_path(t) for t in out_d["t"]]),
        **(extra or {}),
    )
    print(f"saved {os.path.join(out, 'sim.npz')}: {int((rec.tie_step >= 0).sum())} of {n_ties} ties cut")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/rig_cut")
    parser.add_argument("--t-end", type=float, default=T_END)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    parts = make_parts()
    root = Sofa.Core.Node("root")
    createScene(root, parts)
    run(root, parts, args.out, args.t_end, extra={"cut_x": CUT_X})


if __name__ == "__main__":
    main()
