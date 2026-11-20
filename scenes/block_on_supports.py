"""Scene 01: FEM foam block resting on four load-cell supports.

The block is a tetrahedral FEM volume (corotational linear elasticity). Each
support is a patch of bottom nodes tied to its rest position by a stiff spring,
so the spring force k * (x0 - x) summed over the patch is the load-cell reading.
After the block settles under gravity, an off-centre press is ramped on, held
and released, so the four readings differ.

Units are SI throughout (m, kg, s, N, Pa). z is up.

Run headless (records positions and support forces to an .npz):
    python \
        scenes/block_on_supports.py --out outputs/block_on_supports
Or open in the SOFA GUI where one exists:  runSofa scenes/block_on_supports.py
"""

import argparse
import itertools
import os

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation

# ---- parameters ------------------------------------------------------------
SIZE = (0.10, 0.10, 0.04)        # block extent x, y, z [m]
RES = (11, 11, 5)                # grid nodes along x, y, z
YOUNG = 30e3                     # Young's modulus [Pa], soft foam
POISSON = 0.3
DENSITY = 30.0                   # [kg/m^3], foam
GRAVITY = (0.0, 0.0, -9.81)

SUPPORT_XY = [(-0.035, -0.035), (0.035, -0.035), (0.035, 0.035), (-0.035, 0.035)]
SUPPORT_HALF = 0.0125            # half-width of each square support patch [m]
SUPPORT_K = 5e4                  # stiffness of one load cell [N/m]

PRESS_XY = (0.02, 0.01)          # off-centre press location [m]
PRESS_RADIUS = 0.011             # top nodes within this radius are loaded
PRESS_FORCE = 1.5                # peak downward force [N]
# (start, end) of each phase [s]: settle, ramp up, hold, ramp down, rest
T_RAMP_UP, T_HOLD, T_RAMP_DOWN, T_REST, T_END = 1.0, 2.5, 4.0, 5.0, 6.0
DT = 0.01


def press_force(t):
    """Total downward press force [N] at time t."""
    if t < T_RAMP_UP:
        return 0.0
    if t < T_HOLD:
        return PRESS_FORCE * (t - T_RAMP_UP) / (T_HOLD - T_RAMP_UP)
    if t < T_RAMP_DOWN:
        return PRESS_FORCE
    if t < T_REST:
        return PRESS_FORCE * (T_REST - t) / (T_REST - T_RAMP_DOWN)
    return 0.0


def make_mesh(res=RES, size=SIZE, xs=None):
    """Regular grid of nodes split into tetrahedra (6 per hex, Kuhn split).

    Tetrahedra rather than hexahedra so later scenes can cut the mesh. `xs`
    optionally replaces the uniform x node planes (res[0] is then ignored).
    """
    nx, ny, nz = res
    xs = np.linspace(-size[0] / 2, size[0] / 2, nx) if xs is None else np.asarray(xs)
    nx = len(xs)
    ys = np.linspace(-size[1] / 2, size[1] / 2, ny)
    zs = np.linspace(0.0, size[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def nid(i, j, k):
        return (i * ny + j) * nz + k

    tets = []
    for i, j, k in itertools.product(range(nx - 1), range(ny - 1), range(nz - 1)):
        for perm in itertools.permutations(range(3)):
            corner = np.zeros(3, dtype=int)
            path = [nid(i, j, k)]
            for axis in perm:
                corner[axis] += 1
                path.append(nid(i + corner[0], j + corner[1], k + corner[2]))
            tets.append(path)
    tets = np.array(tets)

    # Orient every tetrahedron to positive volume.
    a, b, c, d = (points[tets[:, n]] for n in range(4))
    vol = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    return points, tets


def support_patches(points):
    bottom = np.isclose(points[:, 2], 0.0)
    patches = []
    for cx, cy in SUPPORT_XY:
        inside = (np.abs(points[:, 0] - cx) <= SUPPORT_HALF) & (np.abs(points[:, 1] - cy) <= SUPPORT_HALF)
        patches.append(np.flatnonzero(bottom & inside))
    return patches


def press_patch(points):
    top = np.isclose(points[:, 2], SIZE[2])
    r = np.hypot(points[:, 0] - PRESS_XY[0], points[:, 1] - PRESS_XY[1])
    return np.flatnonzero(top & (r <= PRESS_RADIUS))


class PressController(Sofa.Core.Controller):
    """Updates the press load each step and records the state."""

    def __init__(self, *args, **kwargs):
        mo, press = kwargs.pop("mo"), kwargs.pop("press")
        patches, rest = kwargs.pop("patches"), kwargs.pop("rest")
        super().__init__(*args, **kwargs)  # must run before any attribute is set
        self.mo, self.press, self.patches, self.rest = mo, press, patches, rest
        self.log = {"t": [], "pos": [], "press": [], "support": []}

    def onAnimateBeginEvent(self, event):
        t = self.getContext().getTime()
        n = len(self.press.indices.value)
        f = np.zeros((n, 3))
        f[:, 2] = -press_force(t) / n
        self.press.forces.value = f.tolist()

    def onAnimateEndEvent(self, event):
        t = self.getContext().getTime()
        pos = np.array(self.mo.position.value)
        # Load-cell reading = spring force k * (x0 - x) summed over the patch,
        # i.e. the support's reaction on the block; compression reads positive z.
        reading = np.array([
            (SUPPORT_K / len(p)) * (self.rest[p] - pos[p]).sum(axis=0)
            for p in self.patches
        ])
        self.log["t"].append(t)
        self.log["pos"].append(pos)
        self.log["press"].append(press_force(t))
        self.log["support"].append(reading)


def createScene(root):
    root.gravity = list(GRAVITY)
    root.dt = DT
    for plugin in [
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.Mass",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.AnimationLoop",
    ]:
        root.addObject("RequiredPlugin", name=plugin)
    root.addObject("DefaultAnimationLoop")

    points, tets = make_mesh()
    patches = support_patches(points)
    loaded = press_patch(points)

    block = root.addChild("Block")
    block.addObject("EulerImplicitSolver", rayleighMass=0.1, rayleighStiffness=0.1)
    block.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
    block.addObject("TetrahedronSetTopologyContainer", name="topo",
                    position=points.tolist(), tetrahedra=tets.tolist())
    block.addObject("TetrahedronSetTopologyModifier")
    mo = block.addObject("MechanicalObject", name="dofs", template="Vec3d", position=points.tolist())
    block.addObject("MeshMatrixMass", name="mass", massDensity=DENSITY)
    block.addObject("TetrahedronFEMForceField", name="fem", method="large",
                    youngModulus=YOUNG, poissonRatio=POISSON)
    for n, p in enumerate(patches):
        block.addObject("RestShapeSpringsForceField", name=f"support{n}",
                        points=p.tolist(), stiffness=SUPPORT_K / len(p))
    press = block.addObject("ConstantForceField", name="press", indices=loaded.tolist(),
                            forces=np.zeros((len(loaded), 3)).tolist())
    block.addObject(PressController(name="controller", mo=mo, press=press,
                                    patches=patches, rest=points))
    return root


def check_scene(root, points, tets, patches):
    """Fail loudly if SOFA silently ignored a field."""
    block = root.Block
    assert len(block.topo.tetrahedra.value) == len(tets), "tetrahedra not loaded"
    for n, p in enumerate(patches):
        got = np.array(block.getObject(f"support{n}").points.value).ravel()
        assert np.array_equal(np.sort(got), np.sort(p)), f"support{n} indices ignored"
    assert len(block.press.indices.value) > 0, "press patch empty"
    mass = float(block.mass.totalMass.value)
    expected = DENSITY * np.prod(SIZE)
    assert abs(mass - expected) / expected < 1e-3, f"mass {mass} vs {expected}"
    return mass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/block_on_supports")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    points, tets = make_mesh()
    patches = support_patches(points)
    mass = check_scene(root, points, tets, patches)
    print(f"nodes {len(points)}  tets {len(tets)}  mass {mass * 1e3:.2f} g  "
          f"support nodes {[len(p) for p in patches]}  press nodes {len(press_patch(points))}")

    for _ in range(int(round(T_END / DT))):
        Sofa.Simulation.animate(root, DT)

    log = root.Block.controller.log
    np.savez_compressed(
        os.path.join(args.out, "sim.npz"),
        t=np.array(log["t"]), pos=np.array(log["pos"], dtype=np.float32),
        press=np.array(log["press"]), support=np.array(log["support"]),
        rest=points, tets=tets, mass=mass, dt=DT,
        support_xy=np.array(SUPPORT_XY), support_half=SUPPORT_HALF,
        press_xy=np.array(PRESS_XY), press_nodes=press_patch(points),
        size=np.array(SIZE),
    )
    print(f"saved {os.path.join(args.out, 'sim.npz')}")


if __name__ == "__main__":
    main()
