"""Scene 03: spherical indenter on the real rig (two-layer sponge, eight bar cells).

Same contact and tool-sensor setup as scene 02 (indenter_contact.py), with the
sponge, layers and load-cell layout of the lab rig from rig.py. Each bar cell is
a stiff vertical spring under the sponge nodes resting on its tip, and its
reading is the vertical spring force, one channel per cell.

    python \
        scenes/rig_indenter.py --out outputs/rig_indenter
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
from block_on_supports import GRAVITY  # noqa: E402

DT = 0.01
TOOL_RADIUS = 0.008
TOOL_MASS = 0.002
SENSOR_K = 1e4
FRICTION = 0.3
ALARM_DIST, CONTACT_DIST = 0.002, 0.0002

START_XY = np.array([0.015, 0.008])
SLIDE = np.array([-0.030, 0.0])
DEPTH = 0.006
CLEARANCE = 0.004
Z_TOP = rig.SPONGE_SIZE[2] + TOOL_RADIUS
WAYPOINTS = [
    (0.0, *START_XY, Z_TOP + CLEARANCE),
    (0.5, *START_XY, Z_TOP + CLEARANCE),                 # settle under gravity
    (2.5, *START_XY, Z_TOP - DEPTH),                     # press
    (3.5, *START_XY, Z_TOP - DEPTH),                     # hold
    (5.5, *(START_XY + SLIDE), Z_TOP - DEPTH),           # slide
    (6.5, *(START_XY + SLIDE), Z_TOP - DEPTH),           # hold
    (8.0, *(START_XY + SLIDE), Z_TOP + CLEARANCE),       # retract
    (8.5, *(START_XY + SLIDE), Z_TOP + CLEARANCE),       # rest
]
T_END = WAYPOINTS[-1][0]


def target_at(t):
    w = np.array(WAYPOINTS)
    return np.array([np.interp(t, w[:, 0], w[:, k]) for k in (1, 2, 3)])


class Recorder(Sofa.Core.Controller):
    """Moves the tool target along the path and records the state."""

    def __init__(self, *args, **kwargs):
        sponge, tool, target = kwargs.pop("sponge"), kwargs.pop("tool"), kwargs.pop("target")
        patches, rest = kwargs.pop("patches"), kwargs.pop("rest")
        super().__init__(*args, **kwargs)  # must run before any attribute is set
        self.sponge, self.tool, self.target = sponge, tool, target
        self.patches, self.rest = patches, rest
        self.log = {k: [] for k in ("t", "pos", "tool", "target", "sensor", "cells")}

    def onAnimateBeginEvent(self, event):
        p = target_at(self.getContext().getTime() + DT).tolist()
        self.target.position.value = [p]
        self.target.free_position.value = [p]

    def onAnimateEndEvent(self, event):
        pos = np.array(self.sponge.position.value)
        tool = np.array(self.tool.position.value)[0]
        target = np.array(self.target.position.value)[0]
        self.log["t"].append(self.getContext().getTime())
        self.log["pos"].append(pos)
        self.log["tool"].append(tool)
        self.log["target"].append(target)
        self.log["sensor"].append(SENSOR_K * (target - tool))
        # Vertical spring force of each bar cell; compression reads positive.
        self.log["cells"].append(np.array([
            (rig.CELL_K / len(p)) * (self.rest[p, 2] - pos[p, 2]).sum() for p in self.patches
        ]))


def createScene(root):
    root.gravity = list(GRAVITY)
    root.dt = DT
    for plugin in [
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Mapping",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact",
    ]:
        root.addObject("RequiredPlugin", name=plugin)

    root.addObject("FreeMotionAnimationLoop")
    root.addObject("BlockGaussSeidelConstraintSolver", name="constraints",
                   tolerance=1e-8, maxIterations=500, computeConstraintForces=True)
    root.addObject("CollisionPipeline")
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject("LocalMinDistance", alarmDistance=ALARM_DIST, contactDistance=CONTACT_DIST)
    root.addObject("CollisionResponse", response="FrictionContactConstraint",
                   responseParams=f"mu={FRICTION}")

    points, tets, young, _ = rig.make_sponge()
    patches = rig.cell_patches(points)

    sponge = root.addChild("Sponge")
    sponge.addObject("EulerImplicitSolver", rayleighMass=0.1, rayleighStiffness=0.1)
    # Assembled CG: see indenter_contact.py for why not a direct solver. The tolerance is
    # an absolute squared residual; this 5 g sponge has ~1e-5 N per node, so 1e-12 stopped
    # CG early and the sponge drifted upward.
    sponge.addObject("CGLinearSolver", template="CompressedRowSparseMatrixMat3x3d",
                     iterations=1000, tolerance=1e-18, threshold=1e-25)
    sponge.addObject("TetrahedronSetTopologyContainer", name="topo",
                     position=points.tolist(), tetrahedra=tets.tolist())
    sponge.addObject("TetrahedronSetTopologyModifier")
    mo = sponge.addObject("MechanicalObject", name="dofs", template="Vec3d", position=points.tolist())
    sponge.addObject("MeshMatrixMass", name="mass", massDensity=rig.DENSITY)
    # One Young's modulus per tetrahedron: cream foam base, stiffer orange scrub layer.
    sponge.addObject("TetrahedronFEMForceField", name="fem", method="large",
                     youngModulus=young.tolist(), poissonRatio=rig.POISSON)
    for (name, *_), p in zip(rig.CELLS, patches):
        sponge.addObject("RestShapeSpringsForceField", name=name, points=p.tolist(),
                         stiffness=rig.CELL_K / len(p))
    sponge.addObject("GenericConstraintCorrection")

    surf = sponge.addChild("Surface")
    surf.addObject("TriangleSetTopologyContainer", name="topo")
    surf.addObject("TriangleSetTopologyModifier")
    surf.addObject("Tetra2TriangleTopologicalMapping", input="@../topo", output="@topo")
    surf.addObject("TriangleCollisionModel")
    surf.addObject("LineCollisionModel")
    surf.addObject("PointCollisionModel")

    start = target_at(0.0).tolist()
    target = root.addChild("Target").addObject("MechanicalObject", name="dofs", template="Vec3d",
                                               position=[start])
    tool_node = root.addChild("Indenter")
    # No Rayleigh damping: stiffness damping on the sensor spring would read as drag.
    tool_node.addObject("EulerImplicitSolver", rayleighMass=0.0, rayleighStiffness=0.0)
    tool_node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
    tool = tool_node.addObject("MechanicalObject", name="dofs", template="Vec3d", position=[start])
    tool_node.addObject("UniformMass", totalMass=TOOL_MASS)
    tool_node.addObject("RestShapeSpringsForceField", name="sensor", points=[0],
                        stiffness=SENSOR_K, external_rest_shape="@../Target/dofs", external_points=[0])
    tool_node.addObject("SphereCollisionModel", radius=TOOL_RADIUS)
    tool_node.addObject("LinearSolverConstraintCorrection")

    root.addObject(Recorder(name="controller", sponge=mo, tool=tool, target=target,
                            patches=patches, rest=points))
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/rig_indenter")
    parser.add_argument("--t-end", type=float, default=T_END)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    points, tets, young, scrub = rig.make_sponge()
    patches = rig.cell_patches(points)

    # Fail loudly if SOFA silently ignored a field.
    assert len(root.Sponge.topo.tetrahedra.value) == len(tets), "tetrahedra not loaded"
    assert len(root.Sponge.fem.youngModulus.value) == len(tets), "per-element Young's modulus ignored"
    for (name, *_), p in zip(rig.CELLS, patches):
        got = np.sort(np.array(root.Sponge.getObject(name).points.value).ravel())
        assert np.array_equal(got, np.sort(p)), f"{name} indices ignored"
    mass = float(root.Sponge.mass.totalMass.value)
    print(f"nodes {len(points)}  tets {len(tets)} ({scrub.sum()} scrub)  mass {mass * 1e3:.2f} g  "
          f"cell nodes {[len(p) for p in patches]}")

    for k in range(int(round(args.t_end / DT))):
        Sofa.Simulation.animate(root, DT)
        if k % 100 == 0:
            print(f"  t={k * DT:.1f}s", flush=True)

    log = root.controller.log
    np.savez_compressed(
        os.path.join(args.out, "sim.npz"),
        t=np.array(log["t"]), pos=np.array(log["pos"], dtype=np.float32),
        tool=np.array(log["tool"]), target=np.array(log["target"]),
        sensor=np.array(log["sensor"]), cells=np.array(log["cells"]),
        rest=points, tets=tets, scrub=scrub, mass=mass, tool_mass=TOOL_MASS,
        tool_radius=TOOL_RADIUS, dt=DT, gravity=np.array(GRAVITY),
        cell_names=np.array([c[0] for c in rig.CELLS]), waypoints=np.array(WAYPOINTS),
    )
    print(f"saved {os.path.join(args.out, 'sim.npz')}")


if __name__ == "__main__":
    main()
