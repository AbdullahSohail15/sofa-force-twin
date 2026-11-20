"""Scene 04: grasper and blade on the real rig (two-layer sponge, eight bar cells).

The grasper (left) descends with its jaws open, pinches the scrub surface, lifts,
holds while the blade (right) presses in and makes a slicing stroke, then lowers,
opens and retracts. Cutting is not modelled yet: the blade indents and drags
along the surface but does not separate the foam.

The tools are da Vinci EndoWrist tips from the dVRK model (see tools.py): a Large
Needle Driver grasper and a #11 scalpel blade in a cap on the EndoWrist wrist as the
knife ("blade" below). Each part (left jaw, right jaw, knife) is a rigid body tied to its
commanded pose by a 6-DOF spring, and it collides through a simplified mesh of
its CAD shape. The spring wrench is the part's force/torque sensor: the force and
the torque about the jaw pivot that the part applies to the sponge. The sponge,
layers and cells are those of rig.py.

Grasping: the jaws squeeze the foam through contact, but friction on this coarse
mesh cannot hold a pinch (the foam slips out as the jaws rise). As in most SOFA
surgical simulators, a closed grasp is therefore an attachment: when the jaws
finish closing, the sponge nodes between the tips are tied by stiff springs to
anchors that move with the grasper, until the jaws open. Those springs are part
of the grasper's force reading.

    python \
        scenes/rig_tools.py --out outputs/rig_tools
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
import tools  # noqa: E402
from block_on_supports import GRAVITY  # noqa: E402

DT = 0.01
FRICTION = 1.0                   # foam on serrated grasper jaws / steel
SENSOR_K = 1e6                   # sensor spring, linear [N/m]: 1 um per N
SENSOR_KA = 50.0                 # sensor spring, angular [N m/rad]: 0.02 mrad per N mm
PART_MASS = 0.003                # [kg] per jaw / knife
PART_INERTIA = 8e-6              # inertia per unit mass [m^2] (~L^2/12 for a 10 mm part)
ALARM_DIST, CONTACT_DIST = 0.002, 0.0002
TOP = rig.SPONGE_SIZE[2]
GRASP_K = 2e3                    # attachment spring per grasped node [N/m]
T_GRASP, T_RELEASE = 2.5, 7.2    # jaws closed / start opening
GRASP_RAMP = 0.2                 # attachment stiffness ramps up over this time [s]
TIE_K = 2e4                      # tie spring across a split plane, per node pair [N/m]; ~70x an element's
#                                  stiffness, so a tied plane deforms like uncut foam (5 um per 0.1 N)

GRASP_XY = np.array([-0.018, 0.0])
# Grasper: (time, jaw tip height above the undeformed top [m], jaw angle [rad], pivot dx [m]).
G_OPEN, G_CLOSED = tools.JAW_OPEN, tools.JAW_CLOSED
GRASP_PATH = [
    (0.0, 0.003, G_OPEN),
    (0.5, 0.003, G_OPEN),        # settle
    (1.5, -0.005, G_OPEN),       # press the open jaw tips 5 mm in (deeper buries the jaw hub)
    (2.5, -0.005, G_CLOSED),     # close: pinch, then attach
    (3.5, 0.004, G_CLOSED),      # lift 8 mm
    (6.5, 0.004, G_CLOSED),      # hold while the blade works
    (7.2, -0.006, G_CLOSED),     # lower, then release
    (7.7, -0.006, G_OPEN),       # open
    (8.5, 0.006, G_OPEN),        # retract
    (9.0, 0.006, G_OPEN),
]
# Knife ("blade"): (time, x, y, height of the scalpel point above the undeformed top [m]).
BLADE_PATH = [
    (0.0, 0.022, -0.008, 0.005),
    (3.5, 0.022, -0.008, 0.005),
    (4.3, 0.022, -0.008, -0.003),  # press the tip 3 mm in
    (6.0, 0.022, 0.008, -0.003),   # drag stroke along +y (no cutting in this scene)
    (6.6, 0.022, 0.008, 0.005),    # retract
    (9.0, 0.022, 0.008, 0.005),
]
T_END = 9.0


def interp(path, t):
    p = np.array(path)
    return np.array([np.interp(t, p[:, 0], p[:, k]) for k in range(1, p.shape[1])])


def grasper_pose(t):
    """Pivot position and jaw angle; the pivot height keeps the tips on the path."""
    tip_h, angle = interp(GRASP_PATH, t)
    pivot = np.array([GRASP_XY[0], GRASP_XY[1], TOP + tip_h + tools.JAW_LENGTH * np.cos(angle)])
    return pivot, angle


def blade_pose(t):
    """Commanded position of the knife tip (the scalpel point)."""
    x, y, h = interp(BLADE_PATH, t)
    return np.array([x, y, TOP + h])


class Part:
    """One rigid tool part: its dVRK mesh name and its body pose (pivot, rotation) at time t."""

    def __init__(self, name, mesh, pose):
        self.name, self.mesh, self.pose = name, mesh, pose
        self.local, self.tris = tools.mesh(mesh, "col")

    def place(self, t):
        return tools.rigid(*self.pose(t))


def jaw_pose(side):
    def pose(t):
        pivot, angle = grasper_pose(t)
        return pivot, tools.jaw_rotation(side, angle)
    return pose


# Knife held upright with its cutting edge (body +x) facing world +y, the slicing direction of
# the blade paths, and its flat faces (body y) normal to world x.
BLADE_ROT = tools.rot_z(np.pi / 2)


def knife_pose(tip_pose, rot=BLADE_ROT):
    """Knife body pose (wrist pivot, body rotation) from a function giving its tip position.

    The scalpel point sits at tools.KNIFE_TIP in the body frame; rot = I is upright with the
    cutting edge facing +x.
    """
    return lambda t: (tip_pose(t) - rot @ tools.KNIFE_TIP, rot)


def make_parts(blade=blade_pose, blade_rot=BLADE_ROT):
    return [
        Part("jaw_left", "lnd_jaw_left", jaw_pose(-1)),
        Part("jaw_right", "lnd_jaw_right", jaw_pose(+1)),
        Part("blade", "scalpel", knife_pose(blade, blade_rot)),
    ]


class Recorder(Sofa.Core.Controller):
    """Moves every tool target to its commanded pose and records the state."""

    def __init__(self, *args, **kwargs):
        sponge, parts = kwargs.pop("sponge"), kwargs.pop("parts")
        patches, rest = kwargs.pop("patches"), kwargs.pop("rest")
        plates = kwargs.pop("plates", None)          # rigid-plate cells (rig.CELL_MODEL == "plate")
        grasp, anchors, candidates = kwargs.pop("grasp"), kwargs.pop("anchors"), kwargs.pop("candidates")
        super().__init__(*args, **kwargs)  # must run before any attribute is set
        self.plates = plates
        self.sponge, self.parts, self.patches, self.rest = sponge, parts, patches, rest
        self.grasp, self.anchors, self.candidates = grasp, anchors, candidates
        self.offsets = None      # grasped node positions relative to the pivot, set on grasp
        self.log = {"t": [], "pos": [], "cells": [], "grasp_force": [], "grasped": []}
        for p in parts:
            for key in ("_pos", "_pose", "_force", "_torque"):
                self.log[p.name + key] = []
        self.log.update({"cells_moment": [], "cells_force": [], "grasp_moment": []})

    def onAnimateBeginEvent(self, event):
        t = self.getContext().getTime() + DT
        for p in self.parts:
            x = [p.place(t).tolist()]
            p.target.position.value = x
            p.target.free_position.value = x
        self.update_grasp(t)

    def update_grasp(self, t):
        pivot, angle = grasper_pose(t)
        if self.offsets is None and T_GRASP <= t < T_RELEASE:
            # Attach the candidate nodes lying between the closed jaw tips.
            x = np.array(self.sponge.position.value)[self.candidates]
            tip_z = pivot[2] - tools.JAW_LENGTH * np.cos(angle)
            half_gap = tools.tip_separation(angle) / 2 + 0.0025
            between = ((np.abs(x[:, 0] - pivot[0]) <= half_gap)
                       & (np.abs(x[:, 1] - pivot[1]) <= tools.JAW_WIDTH / 2 + 0.0025)
                       & (x[:, 2] >= tip_z - 0.002))
            self.offsets, self.t_grasp = x - pivot, t
            # One scalar stiffness on just the grasped nodes: with per-point stiffness, SOFA 25.12's
            # RestShapeSpringsForceField::buildStiffnessMatrix indexes k by node id, not by spring,
            # so the implicit solver misses the springs and the grasp explodes.
            sel = np.flatnonzero(between)
            self.grasp.points.value = self.candidates[sel].tolist()
            self.grasp.external_points.value = sel.tolist()
            print(f"  grasp at t={t:.2f}s: {between.sum()} nodes attached", flush=True)
        elif self.offsets is not None and t >= T_RELEASE:
            self.grasp.stiffness.value = [0.0]
            self.offsets = None
            print(f"  release at t={t:.2f}s", flush=True)
        if self.offsets is not None:
            ramp = min(1.0, (t - self.t_grasp) / GRASP_RAMP)
            self.grasp.stiffness.value = [ramp * GRASP_K]
            self.anchors.position.value = (pivot + self.offsets).tolist()
            self.anchors.free_position.value = (pivot + self.offsets).tolist()

    def onAnimateEndEvent(self, event):
        pos = np.array(self.sponge.position.value)
        self.log["t"].append(self.getContext().getTime())
        self.log["pos"].append(pos)
        if self.plates is not None:
            # Wrench each plate applies to the sponge (quasi-static): plate spring + plate weight; the
            # torque is about the plate centre.
            x = np.array(self.plates.position.value)
            x0 = np.array(self.plates.rest_position.value)
            force = rig.CELL_K * (x0[:, :3] - x[:, :3]) + rig.PLATE_MASS * np.array(GRAVITY)
            torque = np.array([rig.CELL_KA * tools.rotvec(tools.matrix_from_quat(a[3:]) @ tools.matrix_from_quat(b[3:]).T)
                               for a, b in zip(x0, x)])
            self.log["cells"].append(force[:, 2].copy())
            self.log["cells_force"].append(force)
            self.log["cells_moment"].append((np.cross(x[:, :3], force) + torque).sum(axis=0))
        else:
            self.log["cells"].append(np.array([
                (rig.CELL_K / len(p)) * (self.rest[p, 2] - pos[p, 2]).sum() for p in self.patches
            ]))
            self.log["cells_force"].append(np.array([
                (rig.CELL_K / len(p)) * (self.rest[p] - pos[p]).sum(axis=0) for p in self.patches]))
            # Moment of all cell spring forces (3D) on the sponge about the origin, for the balance check.
            self.log["cells_moment"].append(sum(
                np.cross(pos[p], (rig.CELL_K / len(p)) * (self.rest[p] - pos[p])).sum(axis=0) for p in self.patches))
        # Attachment springs: force the grasper applies to the sponge through the grasp.
        k = float(np.array(self.grasp.stiffness.value).ravel()[0])
        pts = np.array(self.grasp.points.value, dtype=int).ravel()
        ext = np.array(self.grasp.external_points.value, dtype=int).ravel()
        anchor = np.array(self.anchors.position.value)
        self.log["grasp_force"].append(k * (anchor[ext] - pos[pts]).sum(axis=0))
        self.log["grasp_moment"].append(np.cross(pos[pts], k * (anchor[ext] - pos[pts])).sum(axis=0))
        self.log["grasped"].append(len(pts) if k > 0 else 0)
        g = np.array(GRAVITY)
        for p in self.parts:
            x = np.array(p.body.position.value)[0]
            target = np.array(p.target.position.value)[0]
            rel = tools.matrix_from_quat(target[3:]) @ tools.matrix_from_quat(x[3:]).T
            # Wrench the part applies to the sponge (quasi-static): sensor spring + weight, and the
            # torque about the part's pivot (its weight acts at the pivot).
            self.log[p.name + "_pos"].append(np.array(p.dofs.position.value, dtype=np.float32))
            self.log[p.name + "_pose"].append(x)
            self.log[p.name + "_force"].append(SENSOR_K * (target[:3] - x[:3]) + PART_MASS * g)
            self.log[p.name + "_torque"].append(SENSOR_KA * tools.rotvec(rel))


def add_part(root, part, group):
    """Rigid body on a 6-DOF sensor spring to a kinematic target, colliding through its mesh."""
    start = [part.place(0.0).tolist()]
    tnode = root.addChild(part.name + "_target")
    part.target = tnode.addObject("MechanicalObject", name="dofs", template="Rigid3d", position=start)
    node = root.addChild(part.name)
    # No Rayleigh damping: stiffness damping on the sensor springs would read as drag.
    node.addObject("EulerImplicitSolver", rayleighMass=0.0, rayleighStiffness=0.0)
    node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
    part.body = node.addObject("MechanicalObject", name="dofs", template="Rigid3d", position=start)
    i = PART_INERTIA
    node.addObject("UniformMass", name="mass", vertexMass=f"{PART_MASS} 1 {i} 0 0 0 {i} 0 0 0 {i}")
    # Scalar stiffnesses: per-point vectors hit the SOFA 25.12 stiffness-matrix bug (see grasp).
    node.addObject("RestShapeSpringsForceField", name="sensor", template="Rigid3d", points=[0],
                   stiffness=[SENSOR_K], angularStiffness=[SENSOR_KA],
                   external_rest_shape=f"@../{part.name}_target/dofs", external_points=[0])
    node.addObject("LinearSolverConstraintCorrection")
    col = node.addChild("collision")
    col.addObject("MeshTopology", position=part.local.tolist(), triangles=part.tris.tolist())
    part.dofs = col.addObject("MechanicalObject", name="dofs", template="Vec3d", position=part.local.tolist())
    for model in ("TriangleCollisionModel", "LineCollisionModel", "PointCollisionModel"):
        col.addObject(model, group=group)
    col.addObject("RigidMapping", input="@..", output="@.", globalToLocalCoords=False)


def add_cell_plates(root, patches, points):
    """Rigid-plate load cells: one rigid plate per bar tip on a 6-DOF spring to its rest pose; the sponge
    nodes over the tip are welded to points on the plate by bilateral constraints."""
    centres = np.array([[*points[p].mean(axis=0), 0.0, 0.0, 0.0, 1.0] for p in patches])
    node = root.addChild("CellPlates")
    node.addObject("EulerImplicitSolver", rayleighMass=0.0, rayleighStiffness=0.0)
    node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
    plates = node.addObject("MechanicalObject", name="dofs", template="Rigid3d", position=centres.tolist())
    i = (0.0127 ** 2) / 6                   # inertia per unit mass of a 12.7 mm square plate [m^2]
    node.addObject("UniformMass", name="mass", vertexMass=f"{rig.PLATE_MASS} 1 {i} 0 0 0 {i} 0 0 0 {i}")
    node.addObject("RestShapeSpringsForceField", name="cells", template="Rigid3d",
                   points=list(range(len(patches))), stiffness=[rig.CELL_K], angularStiffness=[rig.CELL_KA])
    node.addObject("LinearSolverConstraintCorrection")
    local = np.concatenate([points[p] - centres[k, :3] for k, p in enumerate(patches)])
    owner = np.concatenate([[k] * len(p) for k, p in enumerate(patches)])
    pts = node.addChild("points")
    pts.addObject("MechanicalObject", name="dofs", template="Vec3d", position=local.tolist())
    pts.addObject("RigidMapping", input="@..", output="@.", rigidIndexPerPoint=owner.tolist(),
                  globalToLocalCoords=False)
    sponge_idx = np.concatenate(patches)
    root.addObject("BilateralLagrangianConstraint", name="cell_welds", template="Vec3d",
                   object1="@Sponge/dofs", object2="@CellPlates/points/dofs",
                   first_point=sponge_idx.tolist(), second_point=list(range(len(sponge_idx))))
    return plates


def createScene(root, parts=None, cut_x=None, split=None, grasp_min_z=None, recorder=Recorder,
                **recorder_kwargs):
    """The rig with the tools.

    split=(axis, value): the sponge is split along that node plane and the two sides are tied by
    stiff zero-length springs ("ties", see rig.split_mesh); a cutting recorder releases them
    (scenes 05, 06). cut_x without split: the older element-deletion cut (a 3 mm column).
    grasp_min_z: only nodes at or above this height can be grasped.
    """
    root.gravity = list(GRAVITY)
    root.dt = DT
    for plugin in [
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Mapping",
        "Sofa.Component.Topology.Utility",
        "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Model",
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

    points, tets, young, _ = rig.make_sponge(cut_x, kerf=0 if split is not None else 0.003)
    pairs = None
    if split is not None:
        points, tets, pairs = rig.split_mesh(points, tets, *split)
    patches = rig.cell_patches(points)

    sponge = root.addChild("Sponge")
    # No stiffness damping: it acts against absolute velocity, so while the stiff grasp springs move
    # it absorbs ~10 N that never reaches the cells and the tool reading stops balancing them.
    sponge.addObject("EulerImplicitSolver", rayleighMass=0.1, rayleighStiffness=0.0)
    # Synchronous direct solver: ~0.6 s per step whatever the number of contact rows (CG +
    # GenericConstraintCorrection costs one CG solve per row, ~1.7 s here). AsyncSparseLDLSolver
    # is ~0.04 s but reuses the previous factorisation, which diverged when the grasp springs
    # switched on.
    sponge.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
    sponge.addObject("TetrahedronSetTopologyContainer", name="topo",
                     position=points.tolist(), tetrahedra=tets.tolist())
    sponge.addObject("TetrahedronSetTopologyModifier")
    mo = sponge.addObject("MechanicalObject", name="dofs", template="Vec3d", position=points.tolist())
    sponge.addObject("MeshMatrixMass", name="mass", massDensity=rig.DENSITY)
    if cut_x is None or split is not None:
        sponge.addObject("TetrahedronFEMForceField", name="fem", method="large",
                         youngModulus=young.tolist(), poissonRatio=rig.POISSON)
    else:
        # TetrahedronFEMForceField misassigns its per-element modulus after tets are removed;
        # this one keeps per-element data through removals (checked against a pre-cut mesh).
        sponge.addObject("TetrahedralCorotationalFEMForceField", name="fem", method="large",
                         youngModulus=float(young.max()), poissonRatio=rig.POISSON,
                         localStiffnessFactor=(young / young.max()).tolist())
        sponge.addObject("TopologicalChangeProcessor", name="cutter", useDataInputs=True, timeToRemove=1e9)
    if pairs is not None:
        n = len(pairs)
        sponge.addObject("SpringForceField", name="ties", springsIndices1=pairs[:, 0].tolist(),
                         springsIndices2=pairs[:, 1].tolist(), stiffness=[TIE_K] * n, damping=[0.0] * n,
                         lengths=[0.0] * n)
        recorder_kwargs.update(pairs=pairs, split_axis=split[0])
    plates = None
    if rig.CELL_MODEL == "plate":
        plates = add_cell_plates(root, patches, points)
    else:
        for (name, *_), p in zip(rig.CELLS, patches):
            sponge.addObject("RestShapeSpringsForceField", name=name, points=p.tolist(),
                             stiffness=rig.CELL_K / len(p))
    recorder_kwargs["plates"] = plates
    # Grasp attachment: candidate nodes near the grasp site, springs off until the grasp.
    near = ((np.abs(points[:, 0] - GRASP_XY[0]) <= 0.008) & (np.abs(points[:, 1] - GRASP_XY[1]) <= 0.006)
            & (points[:, 2] >= (TOP - 0.010 if grasp_min_z is None else grasp_min_z)))
    candidates = np.flatnonzero(near)
    anchors = root.addChild("GraspAnchors").addObject(
        "MechanicalObject", name="dofs", template="Vec3d", position=points[candidates].tolist())
    grasp = sponge.addObject("RestShapeSpringsForceField", name="grasp", points=candidates.tolist(),
                             stiffness=[0.0],
                             external_rest_shape="@../GraspAnchors/dofs",
                             external_points=list(range(len(candidates))))
    sponge.addObject("LinearSolverConstraintCorrection")

    surf = sponge.addChild("Surface")
    surf.addObject("TriangleSetTopologyContainer", name="topo")
    surf.addObject("TriangleSetTopologyModifier")
    surf.addObject("Tetra2TriangleTopologicalMapping", input="@../topo", output="@topo")
    for model in ("TriangleCollisionModel", "LineCollisionModel", "PointCollisionModel"):
        surf.addObject(model, group=0)

    parts = parts if parts is not None else make_parts()
    # The two jaws share a group so they never collide with each other.
    for part, group in zip(parts, (1, 1, 2)):
        add_part(root, part, group)

    root.addObject(recorder(name="controller", sponge=mo, parts=parts, patches=patches, rest=points,
                            grasp=grasp, anchors=anchors, candidates=candidates, **recorder_kwargs))
    return root


def check_parts(root, parts):
    """SOFA ignores unknown fields silently: confirm the sensors and the inertia took."""
    for p in parts:
        node = root.getChild(p.name)
        assert np.isclose(node.sensor.angularStiffness.value[0], SENSOR_KA), f"{p.name} angular sensor ignored"
        assert np.isclose(float(str(node.mass.vertexMass.value).split()[2].strip("[")), PART_INERTIA), \
            f"{p.name} inertia ignored"
        world = np.array(p.dofs.position.value)
        pivot, rot = p.pose(0.0)
        assert np.allclose(world, p.local @ rot.T + pivot, atol=1e-9), f"{p.name} collision mesh misplaced"


def tool_meshes(parts):
    """Collision triangles plus body-frame render meshes of the tools and their clevises."""
    out = {}
    for p in parts:
        out[p.name + "_tris"] = p.tris
        out[p.name + "_vis_pts"], out[p.name + "_vis_tris"] = tools.mesh(p.mesh, "vis")
    out["clevis_vis_pts"], out["clevis_vis_tris"] = tools.mesh("clevis", "vis")
    out["blade_cap_vis_pts"], out["blade_cap_vis_tris"] = tools.mesh("scalpel_cap", "vis")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/rig_tools")
    parser.add_argument("--t-end", type=float, default=T_END)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    parts = make_parts()
    root = Sofa.Core.Node("root")
    createScene(root, parts)
    Sofa.Simulation.init(root)
    points, tets, young, scrub = rig.make_sponge()

    assert len(root.Sponge.topo.tetrahedra.value) == len(tets), "tetrahedra not loaded"
    assert len(root.Sponge.fem.youngModulus.value) == len(tets), "per-element Young's modulus ignored"
    check_parts(root, parts)
    mass = float(root.Sponge.mass.totalMass.value)
    print(f"sponge nodes {len(points)}  tets {len(tets)}  mass {mass * 1e3:.2f} g  "
          f"tool collision vertices {[len(p.local) for p in parts]}")

    for k in range(int(round(args.t_end / DT))):
        Sofa.Simulation.animate(root, DT)
        if k % 100 == 0:
            print(f"  t={k * DT:.1f}s  contact rows {len(root.constraints.constraintForces.value)}", flush=True)

    log = root.controller.log
    out = {k: np.array(v) for k, v in log.items()}
    out["pos"] = out["pos"].astype(np.float32)
    np.savez_compressed(
        os.path.join(args.out, "sim.npz"), **out,
        rest=points, tets=tets, scrub=scrub, mass=mass, dt=DT, gravity=np.array(GRAVITY),
        cell_names=np.array([c[0] for c in rig.CELLS]), part_names=np.array([p.name for p in parts]),
        **tool_meshes(parts),
        grasp_cmd=np.array([np.r_[grasper_pose(t)[0], grasper_pose(t)[1]] for t in out["t"]]),
        blade_cmd=np.array([blade_pose(t) for t in out["t"]]),
    )
    print(f"saved {os.path.join(args.out, 'sim.npz')}")


if __name__ == "__main__":
    main()
