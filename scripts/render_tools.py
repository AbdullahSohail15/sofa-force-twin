"""Render a recorded grasper + blade run (scene 04, or 05 with cutting) to an mp4, plus plots and CSV.

In the 3D view the jaws and the knife ("blade") are the dVRK render meshes placed
at their simulated rigid poses, with the wrist clevis; the section view shows the
collision meshes SOFA used. The wrist is bent through its dVRK joints (roll,
pitch, yaw; tools.wrist) so the shaft runs at the arm's angle (rig.SHAFT_DIRS)
while the jaws keep their simulated pose; the clevis and shaft are drawn only, as
is the rig fixture. Below the 3D view is a section at y = 0 coloured by displacement.
For a cutting run (sim.npz has cut_step), the sponge mesh drops the cut elements
frame by frame, so the kerf and its walls show.

    python scripts/render_tools.py --run outputs/rig_tools
"""

import argparse
import os
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from render_rig import CH_COLORS, add_fixture, sponge_colors

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenes"))
import rig  # noqa: E402
import tools  # noqa: E402
import wrench  # noqa: E402

SHAFT_RADIUS, SHAFT_LENGTH = 0.0042, 0.16
# Wrist joints per instrument: the jaws/knife keep their upright body orientation and the
# shaft runs along the arm's direction. Body part whose pivot carries each wrist.
def wrists(d):
    """Wrist solution per instrument; the knife may be held tilted (sim.npz blade_rot)."""
    return {tool: tools.wrist(rig.SHAFT_DIRS[tool], d["blade_rot"] if tool == "knife" and "blade_rot" in d
                              else np.eye(3)) for tool in ("grasper", "knife")}
WRIST_PART = {"grasper": "jaw_left", "knife": "blade"}


def forces(d):
    """Forces the tools apply to the sponge [N], each (T, 3)."""
    grasper = d["jaw_left_force"] + d["jaw_right_force"] + d["grasp_force"]
    return grasper, d["blade_force"]


def save_forces(d, out):
    t, C = d["t"], d["cells"]
    G, B = forces(d)
    names = [str(n) for n in d["cell_names"]]
    cols = [t, G, d["jaw_left_force"], d["jaw_right_force"], d["grasp_force"], B, C]
    header = ("t_s," + ",".join(f"grasper_{a}_N" for a in "xyz") + ","
              + ",".join(f"jawL_{a}_N" for a in "xyz") + "," + ",".join(f"jawR_{a}_N" for a in "xyz") + ","
              + ",".join(f"grasp_attach_{a}_N" for a in "xyz") + "," + ",".join(f"blade_{a}_N" for a in "xyz")
              + "," + ",".join(f"{n}_N" for n in names))
    # Torques about each part's jaw pivot [N m] and the pivot poses (x y z [m], quaternion x y z w).
    for part, short in (("jaw_left", "jawL"), ("jaw_right", "jawR"), ("blade", "blade")):
        if part + "_torque" in d:
            cols += [d[part + "_torque"], d[part + "_pose"]]
            header += ("," + ",".join(f"{short}_torque_{a}_Nm" for a in "xyz") + ","
                       + ",".join(f"{short}_pivot_{a}_m" for a in "xyz") + ","
                       + ",".join(f"{short}_q{a}" for a in "xyzw"))
    # Combined 6-D wrench per instrument (world and tool frame), see wrench.py.
    wcols, wnames = wrench.columns(d)
    cols += wcols
    header += "".join("," + n for n in wnames)
    np.savetxt(os.path.join(out, "forces.csv"), np.column_stack(cols), delimiter=",", header=header,
               comments="", fmt="%.6g")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    plot_panels(axes, d)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "forces.png"), dpi=150)
    plt.close(fig)


def weight(d):
    """Sponge weight per step [N]; it drops as a cut removes elements."""
    return (d["mass_now"] if "mass_now" in d else np.full(len(d["t"]), float(d["mass"]))) * 9.81


def plot_panels(axes, d):
    t, C = d["t"], d["cells"]
    G, B = forces(d)
    a, b = axes
    for n, name in enumerate(str(x) for x in d["cell_names"]):
        a.plot(t, C[:, n], color=CH_COLORS[n], label=name)
    a.set(ylabel="cell force [N]", title="Simulated bar load cells (vertical)")
    a.legend(fontsize=7, ncol=4, loc="upper right")
    b.plot(t, -G[:, 2], color="tab:blue", label="grasper, down (negative = lifting)")
    b.plot(t, d["jaw_right_force"][:, 0], color="tab:cyan", lw=1, label="grasper squeeze (right jaw, -x)", alpha=0.8)
    b.plot(t, -B[:, 2], color="tab:red", label="blade, down")
    b.plot(t, B[:, 1], color="tab:orange", lw=1, label="blade drag along stroke (+y)")
    b.plot(t, C.sum(1) - weight(d), "k--", lw=1, label="sum of cells - weight")
    b.plot(t, -(G[:, 2] + B[:, 2]), color="0.6", lw=3, alpha=0.4, label="grasper + blade, down")
    b.set(xlabel="time [s]", ylabel="force on sponge [N]", title="Tool force sensors vs load cells")
    b.legend(fontsize=6.5, loc="upper right", ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3)


def placed(points, pose):
    """Body-frame points moved to a Rigid3d pose [x, y, z, qx, qy, qz, qw]."""
    return points @ tools.matrix_from_quat(pose[3:]).T + pose[:3]


def poly(points, tris):
    return pv.PolyData(points, np.column_stack([np.full(len(tris), 3), tris]).ravel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/rig_tools")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--name", help="video file name (default: <run dir name>.mp4)")
    args = parser.parse_args()

    d = np.load(os.path.join(args.run, "sim.npz"))
    t, pos, rest, tets = d["t"], d["pos"], d["rest"], d["tets"]
    G, B = forces(d)
    save_forces(d, args.run)
    WRISTS = wrists(d)

    stride = max(1, int(round(1.0 / (args.fps * float(d["dt"])))))
    frames = range(0, len(t), stride)
    disp_mm = 1e3 * np.linalg.norm(pos - rest, axis=2)
    clim = (0.0, min(4.0, float(disp_mm.max())))
    cut_step = d["cut_step"] if "cut_step" in d else np.full(len(tets), -1)
    sponge = {}

    def sponge_at(i):
        """Grid of the elements still present at frame i, and its surface colours (cached)."""
        alive = (cut_step < 0) | (cut_step > i)
        if sponge.get("alive") is None or not np.array_equal(alive, sponge["alive"]):
            grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets[alive]}, rest.copy())
            rgb = sponge_colors(grid)
            # Faces exposed by the cut (not on the sponge's outer boundary) drawn darker, as in a slit.
            c = grid.extract_surface(algorithm="dataset_surface").cell_centers().points
            half = np.array(rig.SPONGE_SIZE) / 2
            outer = ((np.abs(np.abs(c[:, 0]) - half[0]) < 1e-6) | (np.abs(np.abs(c[:, 1]) - half[1]) < 1e-6)
                     | (np.abs(c[:, 2]) < 1e-6) | (np.abs(c[:, 2] - rig.SPONGE_SIZE[2]) < 1e-6))
            rgb[~outer] = (rgb[~outer] * 0.55).astype(np.uint8)
            sponge.update(alive=alive, grid=grid, rgb=rgb)
        return sponge["grid"], sponge["rgb"]
    style = dict(scalars="disp [mm]", clim=clim, cmap="viridis", above_color="yellow",
                 show_edges=True, edge_color="#333333", line_width=0.3)
    shaft = pv.Cylinder(center=(0, 0, SHAFT_LENGTH / 2), direction=(0, 0, 1), radius=SHAFT_RADIUS,
                        height=SHAFT_LENGTH)

    def shaft_at(start, direction):
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, direction)
        angle = np.degrees(np.arccos(np.dot(z, direction) / np.linalg.norm(direction)))
        return shaft.rotate_vector(axis, angle, point=(0, 0, 0), inplace=False).translate(start, inplace=False)

    pl = pv.Plotter(off_screen=True, window_size=(960, 960), shape=(2, 1), border=False, row_weights=[1.6, 1])
    pl.subplot(0, 0)
    pl.set_background(rig.COLORS["wall"], top="#f4f5f4")
    add_fixture(pl)
    # Closer than scene 03 so the 12 mm jaws and the blade are visible.
    pl.camera_position = [(0.035, -0.15, 0.11), (0.0, 0.0, 0.012), (0, 0, 1)]
    pl.camera.view_angle = 35
    pl.subplot(1, 0)
    pl.set_background("white")
    pl.add_text("section at y = 0 (grasper plane), |displacement| 0-4 mm", font_size=9, color="black")
    pl.camera_position = [(0.0, -0.25, 0.015), (0.0, 0.0, 0.015), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 0.032

    def draw(i):
        """Replace the moving actors (by name) with frame i's geometry."""
        grid, sponge_rgb = sponge_at(i)
        grid.points = pos[i]
        grid["disp [mm]"] = disp_mm[i]
        names = ("jaw_left", "jaw_right", "blade")
        collision = {n: poly(d[f"{n}_pos"][i], d[f"{n}_tris"]) for n in names}
        render = {n: poly(placed(d[f"{n}_vis_pts"], d[f"{n}_pose"][i]), d[f"{n}_vis_tris"]) for n in names}
        if "blade_cap_vis_pts" in d:      # scalpel cap, on the knife's pose
            render["blade_cap"] = poly(placed(d["blade_cap_vis_pts"], d["blade_pose"][i]), d["blade_cap_vis_tris"])
        clevis_pts, clevis_tris = tools.mesh("clevis", "vis")      # pitch link frame
        for sub in ((0, 0), (1, 0)):
            pl.subplot(*sub)
            if sub == (0, 0):
                surf = grid.extract_surface(algorithm="dataset_surface")
                surf.cell_data["rgb"] = sponge_rgb
                pl.add_mesh(surf, name="sponge", scalars="rgb", rgb=True, ambient=0.4, diffuse=0.7)
                for tool, (_, pitch_rot, pitch_point) in WRISTS.items():
                    pitch_axis = d[WRIST_PART[tool] + "_pose"][i, :3] + pitch_point
                    pl.add_mesh(poly(clevis_pts @ pitch_rot.T + pitch_axis, clevis_tris), name=f"{tool}_clevis",
                                color="#4a4f55", specular=0.5, smooth_shading=True)
                    pl.add_mesh(shaft_at(pitch_axis, rig.SHAFT_DIRS[tool]), name=f"{tool}_shaft",
                                color=rig.COLORS["instrument"], smooth_shading=True)
                for n, m in render.items():
                    pl.add_mesh(m, name=n, color={"blade": "#cfd3d7", "blade_cap": "#bdb39f"}.get(n, "#a7abb0"),
                                specular=0.8, specular_power=30, smooth_shading=True)
            else:
                pl.add_mesh(grid.slice(normal="y", origin=(0, 0, 0)), name="section",
                            show_scalar_bar=False, **style)
                for n, m in collision.items():
                    pl.add_mesh(m, name=n, color="#cfd3d7" if n == "blade" else "#9a9ea3")

    video_path = os.path.join(args.run, args.name or os.path.basename(os.path.normpath(args.run)) + ".mp4")
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    fig = plt.figure(figsize=(15.2, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.axis("off")
    axes = [fig.add_subplot(gs[0, 1])]
    axes.append(fig.add_subplot(gs[1, 1], sharex=axes[0]))
    plot_panels(axes, d)
    cursors = [ax.axvline(0, color="k", lw=0.8) for ax in axes]
    title = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    img_artist = None
    for i in frames:
        draw(i)
        img = pl.screenshot(return_img=True)
        if img_artist is None:
            img_artist = ax_img.imshow(img)
        else:
            img_artist.set_data(img)
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        state = "grasped" if d["grasped"][i] else "free"
        if "tie_step" in d:
            n_cut = int(((d["tie_step"] >= 0) & (d["tie_step"] <= i)).sum())
            head = f"Force rig twin: blade cutting ({n_cut} of {len(d['tie_step'])} ties cut)"
        elif "cut_step" in d:
            n_cut = int(((cut_step >= 0) & (cut_step <= i)).sum())
            head = f"Force rig twin: blade cutting ({n_cut} elements cut)"
        else:
            head = "Force rig twin: grasper + blade (cutting not modelled yet)"
        title.set_text(f"{head}   t = {t[i]:.2f} s   "
                       f"grasper {-G[i, 2]:+.2f} N ({state})   blade {-B[i, 2]:.2f} N down")
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    writer.close()
    plt.close(fig)
    pl.close()
    print(f"wrote {video_path} ({len(frames)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
