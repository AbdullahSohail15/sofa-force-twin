"""Render a recorded rig run (scene 03) to an mp4, plus a force plot and CSV.

The 3D view uses the real rig's colours (sampled from photos of the lab rig): orange
scrub layer on cream foam, aluminium bar cells on black mounts and base plate,
blue drape. Below it is a section through the tool path coloured by displacement.
On the right are the eight cell channels and the tool sensor.

    python scripts/render_rig.py --run outputs/rig_indenter
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenes"))
import rig  # noqa: E402

CH_COLORS = plt.get_cmap("tab10").colors[:8]
BAR_Z = (-rig.BAR_WIDTH, 0.0)    # bar top is flush with the sponge bottom
BASE_TOP, BASE_T = -0.030, 0.008


def tool_force_on_sponge(d):
    """Force the tool applies to the sponge, from the sensor spring (quasi-static)."""
    return d["sensor"] + float(d["tool_mass"]) * d["gravity"]


def save_forces(d, out):
    t, C, F = d["t"], d["cells"], tool_force_on_sponge(d)
    names = [str(n) for n in d["cell_names"]]
    header = ("t_s,tool_x_m,tool_y_m,tool_z_m,toolforce_x_N,toolforce_y_N,toolforce_z_N,"
              + ",".join(f"{n}_N" for n in names))
    np.savetxt(os.path.join(out, "forces.csv"), np.column_stack([t, d["tool"], F, C]),
               delimiter=",", header=header, comments="", fmt="%.6g")
    weight = float(d["mass"]) * 9.81
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for n, name in enumerate(names):
        a.plot(t, C[:, n], color=CH_COLORS[n], label=name)
    a.set(ylabel="cell force [N]", title="Eight bar load cells")
    a.legend(ncol=4, fontsize=8)
    b.plot(t, C.sum(1) - weight, "k--", lw=1, label="sum of cells - weight")
    b.plot(t, -F[:, 2], color="0.6", lw=3, alpha=0.5, label="tool sensor (down)")
    b.plot(t, -F[:, 0], color="tab:purple", lw=1, label="tool sensor x (lateral)")
    b.set(xlabel="time [s]", ylabel="force [N]")
    b.legend(fontsize=8)
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "forces.png"), dpi=150)
    plt.close(fig)


def sponge_colors(grid):
    """Per-face colours of the sponge surface, from its rest shape.

    Orange scrub on top and cream foam below, with a little per-face brightness
    noise to suggest the porous texture.
    """
    tri_z = grid.extract_surface(algorithm="dataset_surface").cell_centers().points[:, 2]
    rgb = np.where((tri_z > rig.BASE_THICKNESS)[:, None],
                   np.array(matplotlib.colors.to_rgb(rig.COLORS["scrub"])),
                   np.array(matplotlib.colors.to_rgb(rig.COLORS["foam"])))
    noise = np.random.default_rng(0).normal(1.0, 0.06, size=(len(rgb), 1))
    noise[tri_z <= rig.BASE_THICKNESS] = 1.0 + 0.4 * (noise[tri_z <= rig.BASE_THICKNESS] - 1.0)
    return (np.clip(rgb * noise, 0, 1) * 255).astype(np.uint8)


def add_fixture(pl):
    """Static rig parts: drape, base plate with window, mounts, bars, gauge tape."""
    c = rig.COLORS
    pl.add_mesh(pv.Plane(center=(0, 0, BASE_TOP - BASE_T), i_size=3.0, j_size=3.0), color=c["drape"],
                ambient=0.25)
    ox, oy, wx, wy = 0.105, 0.095, 0.060, 0.052          # half sizes: outer plate and window
    z0, z1 = BASE_TOP - BASE_T, BASE_TOP
    for b in [(-ox, ox, wy, oy), (-ox, ox, -oy, -wy), (-ox, -wx, -wy, wy), (wx, ox, -wy, wy)]:
        pl.add_mesh(pv.Box((*b, z0, z1)), color=c["fixture"], specular=0.2)
    for n, cell in enumerate(rig.CELLS):
        name, _, _, (dx, dy) = cell
        lo, hi = rig.bar_box(cell)
        pl.add_mesh(pv.Box((lo[0], hi[0], lo[1], hi[1], *BAR_Z)), color=c["aluminium"],
                    specular=0.7, specular_power=30)
        # The bar's outer end (opposite the tip) is clamped in a black mount.
        d = np.array([dx, dy])
        tip = np.array(cell[2]) + rig.BAR_OVERLAP * d
        far = tip - rig.BAR_LENGTH * d
        mc = far + 0.010 * d
        along, across = 0.024, 0.026
        hx, hy = (across / 2, along / 2) if dx == 0 else (along / 2, across / 2)
        pl.add_mesh(pv.Box((mc[0] - hx, mc[0] + hx, mc[1] - hy, mc[1] + hy, BASE_TOP, 0.004)),
                    color=c["fixture"], specular=0.2)
        # Gauge tape between the sponge edge and the mount.
        tc = tip - 0.040 * d
        hx, hy = (rig.BAR_WIDTH / 2 + 0.0005, 0.008) if dx == 0 else (0.008, rig.BAR_WIDTH / 2 + 0.0005)
        pl.add_mesh(pv.Box((tc[0] - hx, tc[0] + hx, tc[1] - hy, tc[1] + hy, -0.0002, 0.0006)),
                    color=c["tape"])
        label = far - 0.012 * d
        pl.add_point_labels(np.array([[label[0], label[1], 0.008]]), [name], font_size=11,
                            text_color="black", shape_color="white", shape_opacity=0.7,
                            point_size=0, show_points=False, always_visible=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/rig_indenter")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    d = np.load(os.path.join(args.run, "sim.npz"))
    t, pos, rest, tets, C = d["t"], d["pos"], d["rest"], d["tets"], d["cells"]
    tool, radius = d["tool"], float(d["tool_radius"])
    F = tool_force_on_sponge(d)
    weight = float(d["mass"]) * 9.81
    names = [str(n) for n in d["cell_names"]]
    save_forces(d, args.run)

    stride = max(1, int(round(1.0 / (args.fps * float(d["dt"])))))
    frames = range(0, len(t), stride)
    disp_mm = 1e3 * np.linalg.norm(pos - rest, axis=2)
    clim = (0.0, min(3.0, float(disp_mm.max())))

    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, rest.copy())
    sponge_rgb = sponge_colors(grid)

    y_cut = float(tool[0, 1])
    sphere = pv.Sphere(radius=radius, theta_resolution=40, phi_resolution=40)
    shaft = pv.Cylinder(center=(0, 0, 0.06 + radius), direction=(0, 0, 1), radius=0.0042, height=0.12)
    circle = pv.Circle(radius=radius, resolution=80).rotate_x(90, inplace=False)
    style = dict(scalars="disp [mm]", clim=clim, cmap="viridis", above_color="yellow",
                 show_edges=True, edge_color="#333333", line_width=0.3)

    pl = pv.Plotter(off_screen=True, window_size=(960, 960), shape=(2, 1), border=False,
                    row_weights=[1.6, 1], groups=None)
    pl.subplot(0, 0)
    pl.set_background(rig.COLORS["wall"], top="#f4f5f4")
    add_fixture(pl)
    # Oblique front view, like the lab photo of the rig.
    pl.camera_position = [(0.07, -0.25, 0.17), (0.0, 0.0, 0.0), (0, 0, 1)]
    pl.camera.view_angle = 35
    pl.subplot(1, 0)
    pl.set_background("white")
    pl.add_text(f"section at y = {1e3 * y_cut:.0f} mm (tool path), |displacement| 0-3 mm",
                font_size=9, color="black")
    pl.camera_position = [(0.0, y_cut - 0.25, 0.013), (0.0, y_cut, 0.013), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 0.03

    def draw(i):
        """Replace the moving actors (by name) with frame i's geometry."""
        grid.points = pos[i]
        grid["disp [mm]"] = disp_mm[i]
        pl.subplot(0, 0)
        surf = grid.extract_surface(algorithm="dataset_surface")
        surf.cell_data["rgb"] = sponge_rgb
        pl.add_mesh(surf, name="sponge", scalars="rgb", rgb=True, smooth_shading=False,
                    ambient=0.4, diffuse=0.7)
        pl.add_mesh(sphere.translate(tool[i], inplace=False), name="tip", color="#a7abb0",
                    specular=0.8, specular_power=40, smooth_shading=True)
        pl.add_mesh(shaft.translate(tool[i], inplace=False), name="shaft",
                    color=rig.COLORS["instrument"], specular=0.3, smooth_shading=True)
        pl.subplot(1, 0)
        pl.add_mesh(grid.slice(normal="y", origin=(0, y_cut, 0)), name="section",
                    show_scalar_bar=False, **style)
        pl.add_mesh(circle.translate(tool[i], inplace=False), name="outline", color="#444444",
                    line_width=2, style="wireframe")

    video_path = os.path.join(args.run, "rig_indenter.mp4")
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)

    fig = plt.figure(figsize=(15.2, 8))  # pixel width divisible by 8 for the encoder
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.axis("off")
    ax_c = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1], sharex=ax_c)
    for n, name in enumerate(names):
        ax_c.plot(t, C[:, n], color=CH_COLORS[n], label=name)
    ax_c.set(ylabel="cell force [N]", title="Simulated bar load cells (vertical)")
    ax_c.legend(fontsize=7, ncol=4, loc="upper right")
    ax_t.plot(t, C.sum(1) - weight, "k--", lw=1, label="sum of 8 cells - sponge weight")
    ax_t.plot(t, -F[:, 2], color="0.6", lw=3, alpha=0.5, label="tool sensor, vertical")
    ax_t.plot(t, -F[:, 0], color="tab:purple", lw=1, label="tool sensor, lateral x")
    ax_t.set(xlabel="time [s]", ylabel="force [N]", title="Tool force sensor vs load cells")
    ax_t.legend(fontsize=7, loc="upper right")
    cursors = [ax.axvline(0, color="k", lw=0.8) for ax in (ax_c, ax_t)]
    for ax in (ax_c, ax_t):
        ax.grid(alpha=0.3)
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
        title.set_text(f"Force rig twin: indenter on two-layer sponge, 8 bar load cells   t = {t[i]:.2f} s   "
                       f"tool force {-F[i, 2]:.2f} N down, {-F[i, 0]:+.2f} N lateral")
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    writer.close()
    plt.close(fig)
    pl.close()
    print(f"wrote {video_path} ({len(frames)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
