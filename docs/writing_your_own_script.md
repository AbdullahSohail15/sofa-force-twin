# Writing your own script

The quickest way to a new experiment is to copy `examples/basic_press.py` and edit its
**EDIT HERE** block. It builds the full rig (two-layer sponge, eight load cells, grasper, scalpel),
moves the scalpel along a hard-coded path, and saves the same data as the main scenes.

```bash
conda activate sofa-twin
export SOFA_ROOT=$CONDA_PREFIX
cp examples/basic_press.py examples/my_press.py
# edit examples/my_press.py
python examples/my_press.py --out outputs/my_press           # simulate (~7 min for 5.5 s)
python scripts/check_balance.py --run outputs/my_press       # force/moment balance
python scripts/render_tools.py --run outputs/my_press        # optional mp4 (+ the full forces.csv)
```

Add `--t-end 2` to simulate only the first 2 s while you try things out.

## 1. The tool path

```python
X0, Y0 = 0.010, -0.010        # where the scalpel goes in [m]
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
```

- Waypoints are **linearly interpolated** (`rig_tools.interp`); the run ends at the last time.
- x ∈ [−43, 43] mm, y ∈ [−35.5, 35.5] mm; the sponge centre is (0, 0). Keep ~5 mm away from the
  edges and ~10 mm away from the load-cell tips if you do not want edge effects.
- The height is relative to the **undeformed** top (z = 21 mm); negative = inside the sponge.
- Always start with a short **settle** (0.5 s) above the surface: the sponge sags under gravity
  at the start.
- **Speed:** move at a few mm/s to ~20 mm/s. Very fast motions give dynamic force spikes (the foam
  has inertia and damping) and need more constraint iterations.
- **Depth:** the scalpel is thin. Presses deeper than ~3–4 mm without a cut can make the blade slip
  through the coarse surface mesh. For deeper work, cut (scene 05) or refine the mesh (below).
- The scalpel stays upright with its cutting edge facing +y (`rig_tools.BLADE_ROT`). Pass
  `blade_rot=` to `rig_tools.make_parts` for another orientation (a 3×3 rotation; see
  `rig_dissect.blade_rotation` for a tilted blade).

### Moving the grasper too

The example parks the grasper. To use it, set the grasper plan instead of `park_grasper()`
(same format as scene 04, `rig_tools.GRASP_PATH`):

```python
rig_tools.GRASP_XY = np.array([-0.018, 0.0])        # x, y of the jaw pivot [m]
rig_tools.GRASP_PATH = [                            # (time, jaw-tip height above the top [m], jaw angle [rad])
    (0.0, 0.003, rig_tools.G_OPEN),
    (1.0, -0.004, rig_tools.G_OPEN),                # open jaws 4 mm into the foam
    (1.8, -0.004, rig_tools.G_CLOSED),              # close
    (2.8, 0.004, rig_tools.G_CLOSED),               # lift
    ...
]
rig_tools.T_GRASP, rig_tools.T_RELEASE = 1.8, 4.5   # attachment on / off [s]
```

At `T_GRASP` the nodes between the closed jaws are attached (see
[physics_model.md](physics_model.md#grasping-rig_toolsrecorderupdate_grasp)); at `T_RELEASE` they are let go.
Set these globals **before** `make_parts()` and `createScene()`: `rig_tools` reads them at run time.

## 2. Materials and contact

```python
rig.YOUNG_BASE = 30e3     # cream foam [Pa]
rig.YOUNG_SCRUB = 60e3    # orange scrub layer [Pa]
rig.POISSON = 0.3
rig.DENSITY = 30.0        # [kg/m^3]
rig_tools.FRICTION = 1.0  # tool-foam Coulomb friction
rig.CELL_K = 3e4          # load-cell stiffness [N/m]
rig.CELL_MODEL = "plate"  # or "springs" (~2x faster, within ~6 mN)
```

Set them before `rig_tools.createScene(...)`. In a linear-elastic model the forces for a given
indentation scale with E, so doubling both moduli doubles every force.

## 3. Mesh resolution

`rig.RES = (19, 16, 8)` nodes along x, y, z. Finer meshes give smoother contact and cuts but cost
much more time (the direct solver grows faster than linearly). Keep the z count such that the layer
interface (15 mm of 21 mm) lands on a node plane: with 8 nodes the spacing is 3 mm (interface on
plane 5). For example `(25, 21, 8)` or `(19, 16, 15)` (1.5 mm in z). These finer meshes have not
been run here; check the balance (`scripts/check_balance.py`) after changing the mesh.

## 4. Reading the results

```python
import numpy as np
d = np.load("outputs/my_press/sim.npz")
t = d["t"]
F = d["blade_force"]                  # (T, 3) force the scalpel applies to the sponge [N]
cells = d["cells"]                    # (T, 8) load cells CH1..CH8 [N]
weight = float(d["mass"]) * 9.81
print("peak push", -F[:, 2].min(), "N; cells at that moment", cells[F[:, 2].argmin()])
```

All keys are listed in [outputs.md](outputs.md).

## 5. How the script works

1. `rig_tools.make_parts(blade=scalpel_point)` creates the three rigid parts (two jaws and the
   scalpel) with their collision meshes and the function giving each part's pose at time t.
2. `rig_tools.createScene(root, parts)` builds the SOFA graph: solvers, collision pipeline, the
   sponge FEM, the load-cell plates, the parts on their sensor springs, and a `Recorder` controller.
3. Every `Sofa.Simulation.animate(root, DT)`:
   - before the step, the recorder moves each part's kinematic target to its commanded pose (and
     handles the grasp);
   - SOFA solves the free motion, detects contacts, solves the contact forces with friction, and
     corrects the motion;
   - after the step, the recorder logs node positions, the cell forces and every part's sensor
     wrench into `root.controller.log`.
4. The log is saved to `sim.npz` with the mesh and the commands.

To see the scene graph or change a solver, read `rig_tools.createScene`: each `addObject` call is
one SOFA component, with comments on why its settings were chosen.

## 6. Writing a scene from scratch

If you need something the rig does not have (another object, another tool shape), start from
`scenes/block_on_supports.py` (≈200 lines, no contact) or `scenes/indenter_contact.py` (one rigid
tool with a force sensor and friction contact). Then read
[troubleshooting.md](troubleshooting.md): SOFA ignores misspelt component fields silently, so
assert after `Sofa.Simulation.init` that every important value took effect (as
`rig_tools.check_parts` does).
