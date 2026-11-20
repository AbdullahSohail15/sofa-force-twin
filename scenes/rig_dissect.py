"""Scene 06: dissect the orange scrub layer off the cream foam, starting from a corner.

The grasper pinches the scrub layer near the upper-left corner (workspace view) and lifts it.
The knife (a #11 scalpel in a cap on the wrist, held flat in the layer interface) goes in point
first along the orange/cream interface from the sponge's top edge, then sweeps sideways with its
cutting edge leading, under the lifted flap and out through the left edge. The grasper lifts further as the flap frees, then lays it back down and lets
go; the flap settles back in place on the cream foam.

Cutting is the zero-width tie release of scenes/cutting.py on the split plane z = BASE_THICKNESS
(the layer interface): the two layers are tied there until the knife cuts the ties. Only the
scrub layer's top nodes can be grasped.

Limits: as in scene 05, the cut faces do not collide with each other, so where the flap is laid
back down it can sink into the foam by up to its own (tiny) weight's worth; F_CUT is a guess.

    python scenes/rig_dissect.py --out outputs/rig_dissect
"""

import argparse
import os
import sys

import numpy as np

import Sofa.Core

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig  # noqa: E402
import rig_tools  # noqa: E402
import tools  # noqa: E402
from cutting import TieCutRecorder  # noqa: E402
from rig_cut import run  # noqa: E402
from rig_tools import interp  # noqa: E402

Z_CUT = rig.BASE_THICKNESS        # layer interface, a node plane of the mesh
TOP = rig_tools.TOP
G_OPEN, G_CLOSED = tools.JAW_OPEN, tools.JAW_CLOSED

# Grasper, between four top-surface node columns 7 mm in from the corner (-x, +y).
GRASP_XY = np.array([-0.0358, 0.0284])
T_GRASP, T_RELEASE = 2.3, 7.8
GRASP_PATH = [                     # (time, jaw tip height above the undeformed top [m], jaw angle)
    (0.0, 0.004, G_OPEN),
    (0.5, 0.004, G_OPEN),
    (1.5, -0.002, G_OPEN),         # open tips 2 mm into the 6 mm scrub layer
    (2.3, -0.002, G_CLOSED),       # pinch, then attach (top two node layers only)
    (3.0, 0.003, G_CLOSED),        # lift: tension on the corner before the knife goes in
    (5.7, 0.004, G_CLOSED),        # gentle while the knife sweeps (keeps the interface at the blade)
    (6.6, 0.014, G_CLOSED),        # lift the freed flap
    (7.0, 0.014, G_CLOSED),
    (7.6, -0.001, G_CLOSED),       # lay the flap back down
    (7.8, -0.001, G_CLOSED),
    (8.3, -0.001, G_OPEN),         # release
    (9.0, 0.008, G_OPEN),          # retract
]

# Knife: point direction mostly -y, a little towards -x, level, so the whole 20 mm edge stays in
# the interface (wrist pitch 45 deg, yaw 34 deg, within the 80 deg limits).
TIP_DIR = np.array([-0.3, -1.0, 0.0]) / np.linalg.norm([-0.3, -1.0, 0.0])
# Peeling the bonded layer interface takes far less than cutting through foam (scene 05's 0.8 N):
# the thin scalpel sliding in the interface only meets ~0.2-1 N, and the interface nodes ride up
# over its bevel by up to ~2-3 mm.
F_CUT = 0.15
CUT_RADIUS = 0.003
Z_KNIFE = Z_CUT                   # the 0.8 mm blade straddles the interface, its edge meets the ties
BLADE_PATH = [                     # (time, x, y, z of the scalpel point [m])
    (0.0, -0.018, 0.064, TOP + 0.010),
    (2.4, -0.018, 0.064, TOP + 0.010),
    (3.0, -0.018, 0.046, Z_KNIFE),   # at the interface, outside the top edge
    (3.9, -0.018, 0.016, Z_KNIFE),   # point first, in under the flap
    (4.6, -0.030, 0.016, Z_KNIFE),   # sweep -x with the edge leading...
    (5.7, -0.052, 0.016, Z_KNIFE + 0.002),  # ...rising with the lifted interface near the corner
    (6.4, -0.052, 0.060, Z_KNIFE + 0.002),  # withdraw along +y, clear of the sponge
    (7.0, -0.052, 0.064, TOP + 0.012),
    (9.0, -0.052, 0.064, TOP + 0.012),
]
T_END = 9.0


def blade_pose(t):
    return interp(BLADE_PATH, t)


def blade_rotation():
    """Body rotation of the knife: point (-z) along TIP_DIR, blade lying flat in the layer interface.

    The flat faces (body y) point up as far as they can; the cutting edge (body +x) then faces
    world -x, the direction of the sweep.
    """
    bz = -TIP_DIR
    by = np.array([0.0, 0.0, 1.0]) - TIP_DIR[2] * TIP_DIR
    by /= np.linalg.norm(by)
    return np.column_stack([np.cross(by, bz), by, bz])


def use_grasp_plan():
    """rig_tools reads its grasp plan from module globals at call time; point them at this scene's."""
    rig_tools.GRASP_XY, rig_tools.GRASP_PATH = GRASP_XY, GRASP_PATH
    rig_tools.T_GRASP, rig_tools.T_RELEASE = T_GRASP, T_RELEASE


def make_parts():
    use_grasp_plan()
    return rig_tools.make_parts(blade=blade_pose, blade_rot=blade_rotation())


def createScene(root, parts=None):
    parts = parts if parts is not None else make_parts()
    use_grasp_plan()
    rig_tools.createScene(root, parts, split=(2, Z_CUT), grasp_min_z=Z_CUT + 0.0015,
                          recorder=TieCutRecorder, blade_path=blade_pose, f_cut=F_CUT, radius=CUT_RADIUS)
    root.controller.ties = root.Sponge.ties
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/rig_dissect")
    parser.add_argument("--t-end", type=float, default=T_END)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    parts = make_parts()
    root = Sofa.Core.Node("root")
    createScene(root, parts)
    run(root, parts, args.out, args.t_end, extra={"split_z": Z_CUT})


if __name__ == "__main__":
    main()
