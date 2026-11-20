"""Scene 02: rigid spherical indenter pressing and sliding on the foam block.

Same block and load-cell supports as scene 01 (finer mesh), but the load now
comes from frictional contact with a rigid sphere instead of a prescribed force.

The indenter is a simulated sphere tied by a stiff spring to a kinematic target
that follows a scripted path. The spring is the tool's force sensor, so the tool
force is measured independently of the four support load cells. Contact is
solved with Lagrangian constraints (FreeMotionAnimationLoop + Gauss-Seidel) with
Coulomb friction against the block's surface triangles.

Path: settle, descend into the block, hold, slide sideways, hold, retract.

    python \
        scenes/indenter_contact.py --out outputs/indenter_contact
"""

import argparse
import os
import sys

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_on_supports import (DENSITY, GRAVITY, POISSON, SIZE, SUPPORT_HALF,  # noqa: E402
                               SUPPORT_K, SUPPORT_XY, YOUNG, make_mesh, support_patches)

RES = (21, 21, 9)                # 5 mm node spacing, so the sphere sees a smooth surface
DT = 0.01

TOOL_RADIUS = 0.008              # sphere radius [m]
TOOL_MASS = 0.002                # [kg]
SENSOR_K = 1e4                   # tool force-sensor spring stiffness [N/m]
FRICTION = 0.3                   # Coulomb friction coefficient tool-foam
ALARM_DIST, CONTACT_DIST = 0.002, 0.0002

START_XY = np.array([0.02, 0.01])
SLIDE = np.array([-0.02, 0.0])   # sideways travel during the slide [m]
DEPTH = 0.006                    # target indentation below the undeformed top [m]
CLEARANCE = 0.004                # start gap between sphere and top surface [m]
# Waypoints of the target sphere centre: (time [s], x, y, z). Linear in between.
Z_TOP = SIZE[2] + TOOL_RADIUS
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


class IndenterController(Sofa.Core.Controller):
    """Moves the target along the path and records the state."""

    def __init__(self, *args, **kwargs):
        block, tool, target = kwargs.pop("block"), kwargs.pop("tool"), kwargs.pop("target")
        patches, rest = kwargs.pop("patches"), kwargs.pop("rest")
        super().__init__(*args, **kwargs)  # must run before any attribute is set
        self.block, self.tool, self.target = block, tool, target
        self.patches, self.rest = patches, rest
        self.log = {k: [] for k in ("t", "pos", "tool", "target", "sensor", "support")}

    def onAnimateBeginEvent(self, event):
        # Target position for the coming step (time t + dt).
        p = target_at(self.getContext().getTime() + DT).tolist()
        self.target.position.value = [p]
        self.target.free_position.value = [p]

    def onAnimateEndEvent(self, event):
        t = self.getContext().getTime()
        pos = np.array(self.block.position.value)
        tool = np.array(self.tool.position.value)[0]
        target = np.array(self.target.position.value)[0]
        self.log["t"].append(t)
        self.log["pos"].append(pos)
        self.log["tool"].append(tool)
        self.log["target"].append(target)
        # Force the sensor spring applies to the tool.
        self.log["sensor"].append(SENSOR_K * (target - tool))
        self.log["support"].append(np.array([
            (SUPPORT_K / len(p)) * (self.rest[p] - pos[p]).sum(axis=0) for p in self.patches
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

    points, tets = make_mesh(RES)
    patches = support_patches(points)

    block = root.addChild("Block")
    block.addObject("EulerImplicitSolver", rayleighMass=0.1, rayleighStiffness=0.1)
    # Iterative CG: a direct LDL factorisation of this 12k-DOF mesh costs ~0.9 s per step,
    # CG ~0.04 s. GenericConstraintCorrection builds the contact compliance through it.
    block.addObject("CGLinearSolver", template="CompressedRowSparseMatrixMat3x3d",
                    iterations=300, tolerance=1e-12, threshold=1e-14)
    block.addObject("TetrahedronSetTopologyContainer", name="topo",
                    position=points.tolist(), tetrahedra=tets.tolist())
    block.addObject("TetrahedronSetTopologyModifier")
    block_mo = block.addObject("MechanicalObject", name="dofs", template="Vec3d", position=points.tolist())
    block.addObject("MeshMatrixMass", name="mass", massDensity=DENSITY)
    block.addObject("TetrahedronFEMForceField", name="fem", method="large",
                    youngModulus=YOUNG, poissonRatio=POISSON)
    for n, p in enumerate(patches):
        block.addObject("RestShapeSpringsForceField", name=f"support{n}",
                        points=p.tolist(), stiffness=SUPPORT_K / len(p))
    block.addObject("GenericConstraintCorrection")

    # Collision surface: boundary triangles of the tet mesh, on the same DOFs.
    surf = block.addChild("Surface")
    surf.addObject("TriangleSetTopologyContainer", name="topo")
    surf.addObject("TriangleSetTopologyModifier")
    surf.addObject("Tetra2TriangleTopologicalMapping", input="@../topo", output="@topo")
    surf.addObject("TriangleCollisionModel")
    surf.addObject("LineCollisionModel")
    surf.addObject("PointCollisionModel")

    start = target_at(0.0).tolist()
    target_node = root.addChild("Target")
    target = target_node.addObject("MechanicalObject", name="dofs", template="Vec3d", position=[start])

    tool_node = root.addChild("Indenter")
    # No Rayleigh damping here: stiffness damping on the sensor spring would read as drag.
    tool_node.addObject("EulerImplicitSolver", rayleighMass=0.0, rayleighStiffness=0.0)
    tool_node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
    tool = tool_node.addObject("MechanicalObject", name="dofs", template="Vec3d", position=[start])
    tool_node.addObject("UniformMass", totalMass=TOOL_MASS)
    tool_node.addObject("RestShapeSpringsForceField", name="sensor", points=[0],
                        stiffness=SENSOR_K, external_rest_shape="@../Target/dofs", external_points=[0])
    tool_node.addObject("SphereCollisionModel", radius=TOOL_RADIUS)
    tool_node.addObject("LinearSolverConstraintCorrection")

    root.addObject(IndenterController(name="controller", block=block_mo, tool=tool, target=target,
                                      patches=patches, rest=points))
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/indenter_contact")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    points, tets = make_mesh(RES)
    patches = support_patches(points)

    # Fail loudly if SOFA silently ignored a field.
    assert len(root.Block.topo.tetrahedra.value) == len(tets), "tetrahedra not loaded"
    assert len(root.Block.Surface.topo.triangles.value) > 0, "no collision surface"
    for n, p in enumerate(patches):
        got = np.sort(np.array(root.Block.getObject(f"support{n}").points.value).ravel())
        assert np.array_equal(got, np.sort(p)), f"support{n} indices ignored"
    mass = float(root.Block.mass.totalMass.value)
    print(f"nodes {len(points)}  tets {len(tets)}  surface tris "
          f"{len(root.Block.Surface.topo.triangles.value)}  mass {mass * 1e3:.2f} g  "
          f"support nodes {[len(p) for p in patches]}")

    steps = int(round(T_END / DT))
    for k in range(steps):
        Sofa.Simulation.animate(root, DT)
        if k % 100 == 0:
            print(f"  t={k * DT:.1f}s", flush=True)

    log = root.controller.log
    np.savez_compressed(
        os.path.join(args.out, "sim.npz"),
        t=np.array(log["t"]), pos=np.array(log["pos"], dtype=np.float32),
        tool=np.array(log["tool"]), target=np.array(log["target"]),
        sensor=np.array(log["sensor"]), support=np.array(log["support"]),
        rest=points, tets=tets, mass=mass, tool_mass=TOOL_MASS, tool_radius=TOOL_RADIUS,
        dt=DT, gravity=np.array(GRAVITY), support_xy=np.array(SUPPORT_XY),
        support_half=SUPPORT_HALF, size=np.array(SIZE), waypoints=np.array(WAYPOINTS),
    )
    print(f"saved {os.path.join(args.out, 'sim.npz')}")


if __name__ == "__main__":
    main()
