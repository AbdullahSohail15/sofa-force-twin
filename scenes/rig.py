"""Geometry of the real force rig: two-layer sponge on eight bar load cells.

Measured from photos of the lab rig (top view, scaled by the 12.7 mm width of the
aluminium bar cells; the photos are not part of this repo), so the dimensions are approximate:

- Sponge 86 x 71 x 21 mm: a 15 mm cream foam base under a 6 mm orange scrub layer.
- Eight straight-bar load cells, two per side, cantilevered inward from black
  mounts on a black base plate with an open window under the sponge. The sponge
  rests on the bar tips; each cell measures one vertical force (one channel).

Frame: origin at the centre of the sponge bottom, x along the long side (right in
the workspace camera view), y along the short side (up in that view), z up.
Units are SI.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_on_supports import make_mesh  # noqa: E402

SPONGE_SIZE = (0.086, 0.071, 0.021)    # was 0.027 with the 21 mm base
BASE_THICKNESS = 0.015           # cream foam layer (was 0.021); the 6 mm orange scrub layer is above it
RES = (19, 16, 8)                # ~4.8 x 4.7 x 3 mm nodes (z was 10); the layer interface falls on a node plane
YOUNG_BASE = 30e3                # [Pa] cream foam (to be calibrated)
YOUNG_SCRUB = 60e3               # [Pa] orange scrub layer (to be calibrated)
POISSON = 0.3
DENSITY = 30.0                   # [kg/m^3]

BAR_WIDTH = 0.0127               # bar cross-section is 12.7 x 12.7 mm
BAR_LENGTH = 0.080
BAR_OVERLAP = 0.012              # how far each bar tip reaches under the sponge
CELL_K = 3e4                     # vertical stiffness of one bar cell at its tip [N/m]
# Load-cell model. "plate": each bar tip is a rigid plate on a 6-DOF spring (CELL_K, CELL_KA) and the
# sponge nodes over the tip are welded to it (bilateral constraints), so the contact area moves as one
# piece like the real metal tip; the reading is the plate spring's force. "springs": the older model,
# one independent spring of CELL_K / n per contact node.
CELL_MODEL = "plate"
CELL_KA = 50.0                   # tip tilt stiffness [N m/rad]; double-bending-beam cells keep the tip level
PLATE_MASS = 1e-3                # [kg] per plate; its weight is subtracted from the reading
PAIR_OFFSET = 0.014              # the two cells of a side sit at +-14 mm from its midline

# Channels numbered clockwise from the top-left in the workspace camera view.
# The real CH01-CH08 order is not known yet; remap here once it is.
LX, LY = SPONGE_SIZE[0] / 2, SPONGE_SIZE[1] / 2
CELLS = [
    # name, side, centre of the contact patch along the side, inward direction
    ("CH1", "top", (-PAIR_OFFSET, LY), (0, -1)),
    ("CH2", "top", (PAIR_OFFSET, LY), (0, -1)),
    ("CH3", "right", (LX, PAIR_OFFSET), (-1, 0)),
    ("CH4", "right", (LX, -PAIR_OFFSET), (-1, 0)),
    ("CH5", "bottom", (PAIR_OFFSET, -LY), (0, 1)),
    ("CH6", "bottom", (-PAIR_OFFSET, -LY), (0, 1)),
    ("CH7", "left", (-LX, -PAIR_OFFSET), (1, 0)),
    ("CH8", "left", (-LX, PAIR_OFFSET), (1, 0)),
]

# Instrument shafts: unit direction from the wrist up the shaft to each arm, estimated
# from the workspace and side photos (~45 deg from vertical, arms at the upper left
# and upper right of the workspace view). Replace with the arms' real kinematics.
SHAFT_DIRS = {"grasper": np.array([-0.55, 0.45, 0.70]) / np.linalg.norm([-0.55, 0.45, 0.70]),
              "knife": np.array([0.55, 0.45, 0.70]) / np.linalg.norm([0.55, 0.45, 0.70])}

# Colours sampled from the photos.
COLORS = dict(scrub="#e68c3a", foam="#dcc07a", drape="#7fc3f4", fixture="#1e2226",
              aluminium="#a9acae", tape="#e7c76b", wall="#dfe1de", instrument="#1b1b1b")


def make_sponge(cut_x=None, kerf=0.003):
    """Tet mesh of the sponge and the Young's modulus of each tetrahedron.

    With `cut_x`, a node plane at x = cut_x (kerf 0, for split_mesh) or two node planes at
    cut_x +- kerf/2 (a one-cell-wide column that element deletion can remove). Grid planes
    closer than 1 mm to them are dropped so no sliver cells form beside them.
    """
    xs = np.linspace(-SPONGE_SIZE[0] / 2, SPONGE_SIZE[0] / 2, RES[0])
    if cut_x is not None:
        xs = xs[np.abs(xs - cut_x) > kerf / 2 + 0.001]
        xs = np.sort(np.r_[xs, cut_x] if kerf == 0 else np.r_[xs, cut_x - kerf / 2, cut_x + kerf / 2])
    points, tets = make_mesh(RES, SPONGE_SIZE, xs)
    centroid_z = points[tets][:, :, 2].mean(axis=1)
    scrub = centroid_z > BASE_THICKNESS
    young = np.where(scrub, YOUNG_SCRUB, YOUNG_BASE)
    return points, tets, young, scrub


def bar_box(cell):
    """Axis-aligned bounds of a bar: from its tip under the sponge out to the mount."""
    _, _, (ex, ey), (dx, dy) = cell
    tip = np.array([ex, ey]) + BAR_OVERLAP * np.array([dx, dy])
    far = tip - BAR_LENGTH * np.array([dx, dy])
    half = BAR_WIDTH / 2
    lo, hi = np.minimum(tip, far), np.maximum(tip, far)
    if dx == 0:
        lo[0], hi[0] = ex - half, ex + half
    else:
        lo[1], hi[1] = ey - half, ey + half
    return lo, hi


def cell_patches(points):
    """Bottom nodes resting on each bar tip."""
    bottom = np.isclose(points[:, 2], 0.0)
    patches = []
    for cell in CELLS:
        lo, hi = bar_box(cell)
        eps = 1e-9
        inside = ((points[:, 0] >= lo[0] - eps) & (points[:, 0] <= hi[0] + eps)
                  & (points[:, 1] >= lo[1] - eps) & (points[:, 1] <= hi[1] + eps))
        patch = np.flatnonzero(bottom & inside)
        assert len(patch) > 0, f"{cell[0]} touches no nodes"
        patches.append(patch)
    return patches


def split_mesh(points, tets, axis, value):
    """Split the mesh along the node plane points[:, axis] == value, for zero-width cutting.

    Every node on the plane gets a twin; the tetrahedra on the + side use the twins. The
    pairs (node, twin) are returned so the scene can tie them together (stiff zero-length
    springs) and cut by releasing ties: nothing is removed, the two faces just separate where
    the blade passes and close again behind it.
    """
    on = np.flatnonzero(np.isclose(points[:, axis], value, atol=1e-9))
    assert len(on), f"no node plane at {'xyz'[axis]} = {value}"
    twin = np.full(len(points), -1)
    twin[on] = len(points) + np.arange(len(on))
    plus = points[tets][:, :, axis].mean(axis=1) > value
    tets = tets.copy()
    sub = tets[plus]
    sub = np.where(twin[sub] >= 0, twin[sub], sub)
    tets[plus] = sub
    return np.r_[points, points[on]], tets, np.stack([on, twin[on]], 1)
