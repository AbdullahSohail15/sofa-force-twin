# Outputs

`./run.sh <scene>` writes into `outputs/<scene name>/` (or the folder you pass):

| File | Written by | Contents |
|---|---|---|
| `sim.npz` | the scene (`scenes/*.py`) | every recorded signal + the mesh, one row per 10 ms step |
| `forces.csv` | the render script | the force signals as a table (one row per step) |
| `forces.png` | the render script | load cells and tool force over time |
| `<scene>.mp4` | the render script | 3D view + section + force traces, 25 fps |

All values are SI (m, s, N, N·m), in the rig frame (origin at the centre of the sponge bottom,
z up; see [physics_model.md](physics_model.md)). Forces are those **applied to the sponge** by the
tool (a physical F/T sensor on the tool reads the negative), and cell readings are positive when the
sponge pushes down on the cell.

Load a run in Python:

```python
import numpy as np
d = np.load("outputs/rig_cut/sim.npz")
print(sorted(d.keys()))
t, cells = d["t"], d["cells"]            # (T,), (T, 8)
blade_F = d["blade_force"]                # (T, 3)
```

## `sim.npz`, tool scenes 04–06 (and `examples/basic_press.py`)

T = number of steps, N = sponge nodes, P = collision-mesh vertices of a part.

| Key | Shape | Meaning |
|---|---|---|
| `t` | (T,) | time at the end of each step [s] |
| `pos` | (T, N, 3) float32 | sponge node positions [m] |
| `rest`, `tets` | (N, 3), (E, 4) | rest node positions and tetrahedra (node indices) |
| `scrub` | (E,) bool | tetrahedron belongs to the orange scrub layer |
| `mass`, `dt`, `gravity` | scalar, scalar, (3,) | sponge mass [kg], time step [s], gravity [m/s²] |
| `cells` | (T, 8) | **load-cell readings** CH1–CH8, vertical [N] |
| `cells_force` | (T, 8, 3) | full 3D force each cell applies to the sponge [N] |
| `cells_moment` | (T, 3) | total moment of the cells on the sponge about the origin [N·m] |
| `cell_names` | (8,) | `CH1` … `CH8` |
| `part_names` | (3,) | `jaw_left`, `jaw_right`, `blade` |
| `<part>_force` | (T, 3) | force the part applies to the sponge [N] |
| `<part>_torque` | (T, 3) | torque the part applies, about its pivot [N·m] |
| `<part>_pose` | (T, 7) | part pivot position [m] + orientation quaternion (x, y, z, w) |
| `<part>_pos` | (T, P, 3) float32 | the part's collision vertices in the world |
| `grasp_force`, `grasp_moment` | (T, 3) | force (and moment about the origin) of the grasp attachment springs on the sponge |
| `grasped` | (T,) | number of attached sponge nodes (0 when not grasping) |
| `grasp_cmd` | (T, 4) | commanded grasper pivot (x, y, z) and jaw angle [rad] |
| `blade_cmd` | (T, 3) | commanded scalpel point (x, y, z) |
| `<part>_tris`, `*_vis_pts`, `*_vis_tris` | | collision and render meshes in the body frame (for rendering) |

Extra keys in the cutting scenes 05–06:

| Key | Shape | Meaning |
|---|---|---|
| `tie_pairs` | (K, 2) | node pairs across the split plane (node, twin) |
| `tie_step` | (K,) | step at which each tie was cut, −1 if never |
| `ties_cut` | (T,) | number of ties cut so far |
| `blade_push` | (T,) | blade force along its direction of motion [N] (compared with `f_cut`) |
| `f_cut` | scalar | cutting resistance used [N] |
| `blade_rot` | (3, 3) | scalpel body rotation |
| `cut_x` (05) / `split_z` (06) | scalar | position of the split plane [m] |

**Grasper total** = `jaw_left_force + jaw_right_force + grasp_force`. `scripts/wrench.py` builds the
6-axis wrench per instrument from these.

## `forces.csv`, tool scenes 04–06 (`scripts/render_tools.py`)

| Columns | Meaning |
|---|---|
| `t_s` | time [s] |
| `grasper_{x,y,z}_N` | total grasper force on the sponge (both jaws + attachment) |
| `jawL_*`, `jawR_*`, `grasp_attach_*` | the three contributions separately |
| `blade_{x,y,z}_N` | scalpel force on the sponge |
| `CH1_N` … `CH8_N` | load cells |
| `jawL_torque_*_Nm`, `jawL_pivot_*_m`, `jawL_q*` (and `jawR_`, `blade_`) | per-part torque about its pivot, pivot position, orientation |
| `grasper_F*_N`, `grasper_T*_Nm` | grasper 6-axis wrench, world frame, about the jaw pivot |
| `grasper_F*_tool_N`, `grasper_T*_tool_Nm` | the same in the grasper's body frame |
| `scalpel_F*_N`, `scalpel_T*_Nm`, `scalpel_*_tool_*` | scalpel 6-axis wrench, world and blade frame (z tip→cap, x cutting edge, y blade normal) |
| `<tool>_pivot_*_m`, `<tool>_q*` | the pivot and frame the wrench is expressed about |

## Scenes 01–03

| Scene | `sim.npz` keys | `forces.csv` columns |
|---|---|---|
| 01 block | `t, pos, press` (applied force), `support` (T, 4, 3), `rest, tets, mass, dt, support_xy, press_xy, ...` | `t_s, press_N, cell{0-3}_{x,y,z}_N` |
| 02 indenter | `t, pos, tool` (sphere centre), `target`, `sensor` (spring force), `support` (T, 4, 3), `rest, tets, ...` | `t_s, tool_{x,y,z}_m, toolforce_{x,y,z}_N, cell{0-3}_{x,y,z}_N` |
| 03 rig indenter | `t, pos, tool, target, sensor, cells` (T, 8), `cell_names, scrub, rest, tets, ...` | `t_s, tool_{x,y,z}_m, toolforce_{x,y,z}_N, CH1_N … CH8_N` |

The tool force on the foam in 02–03 is `sensor + tool_mass * gravity`.

## `examples/basic_press.py`

Writes the same `sim.npz` as the tool scenes, plus its own short `press_forces.csv` (and
`press_forces.png`):
`t_s, scalpel_f{x,y,z}_N, scalpel_t{x,y,z}_Nm, CH1_N … CH8_N, cells_sum_N, balance_err_N`, where
`balance_err_N` = −(vertical force of all tools) − (cells sum − weight), ~0 at equilibrium.
Running `scripts/render_tools.py` on it adds the full `forces.csv`, `forces.png` and the mp4.

## Size

A 9 s tool-scene run is ~15–30 MB (`pos` dominates). Delete `outputs/` freely; it is git-ignored.
