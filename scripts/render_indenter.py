"""Render a recorded indenter-contact run to an mp4, plus a force plot and CSV.

Each video frame is the deformed block and the spherical indenter (PyVista,
offscreen, true scale) next to the load-cell and tool-sensor traces.

    python scripts/render_indenter.py \
        --run outputs/indenter_contact --fps 25
"""

import argparse
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from render_video import CELL_COLORS


def tool_force_on_block(d):
    """Force the tool applies to the block, from the sensor spring (quasi-static).

    On the tool: sensor + weight + contact = 0, so the push on the block is
    -contact = sensor + weight.
    """
    return d["sensor"] + float(d["tool_mass"]) * d["gravity"]


def save_forces(d, out):
    t, S = d["t"], d["support"]
    F = tool_force_on_block(d)
    header = ("t_s,tool_x_m,tool_y_m,tool_z_m,toolforce_x_N,toolforce_y_N,toolforce_z_N,"
              + ",".join(f"cell{n}_{a}_N" for n in range(4) for a in "xyz"))
    rows = np.column_stack([t, d["tool"], F, S.reshape(len(t), -1)])
    np.savetxt(os.path.join(out, "forces.csv"), rows, delimiter=",", header=header, comments="", fmt="%.6g")

    weight = float(d["mass"]) * 9.81
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for n in range(4):
        a.plot(t, S[:, n, 2], color=CELL_COLORS[n], label=f"cell {n}")
    a.plot(t, S[:, :, 2].sum(1) - weight, "k--", lw=1, label="sum of cells - weight")
    a.plot(t, -F[:, 2], color="0.6", lw=3, alpha=0.5, label="tool sensor (down)")
    a.set(ylabel="vertical force [N]", title="Load cells vs tool force sensor")
    a.legend(ncol=3, fontsize=8)
    b.plot(t, S[:, :, 0].sum(1), "k--", lw=1, label="sum of cells x")
    b.plot(t, -F[:, 0], color="0.6", lw=3, alpha=0.5, label="tool sensor x (reversed)")
    b.set(xlabel="time [s]", ylabel="lateral force [N]")
    b.legend(fontsize=8)
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "forces.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/indenter_contact")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    d = np.load(os.path.join(args.run, "sim.npz"))
    t, pos, rest, tets, S = d["t"], d["pos"], d["rest"], d["tets"], d["support"]
    tool, radius = d["tool"], float(d["tool_radius"])
    F = tool_force_on_block(d)
    weight = float(d["mass"]) * 9.81
    save_forces(d, args.run)

    stride = max(1, int(round(1.0 / (args.fps * float(d["dt"])))))
    frames = range(0, len(t), stride)
    disp_mm = 1e3 * np.linalg.norm(pos - rest, axis=2)
    # Cap the colour scale so the spread of deformation is visible, not just the dent.
    clim = (0.0, min(3.0, float(disp_mm.max())))
    bar = dict(title="|displacement| [mm]", color="black", vertical=True,
               position_x=0.84, position_y=0.2, height=0.6, title_font_size=12)

    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, rest.copy())
    y_cut = float(tool[0, 1])
    sphere = pv.Sphere(radius=radius, theta_resolution=40, phi_resolution=40)
    shaft = pv.Cylinder(center=(0, 0, 0.04 + radius), direction=(0, 0, 1), radius=0.4 * radius, height=0.08)
    circle = pv.Circle(radius=radius, resolution=80).rotate_x(90, inplace=False)
    style = dict(scalars="disp [mm]", clim=clim, cmap="viridis", above_color="yellow",
                 show_edges=True, edge_color="#333333", line_width=0.3)

    # Top: 3D overview. Bottom: section through the tool path (plane y = tool y).
    pl = pv.Plotter(off_screen=True, window_size=(900, 900), shape=(2, 1), border=False)
    pl.set_background("white")
    pl.subplot(0, 0)
    half, h = float(d["support_half"]), 0.03
    for n, (cx, cy) in enumerate(d["support_xy"]):
        pl.add_mesh(pv.Box((cx - half, cx + half, cy - half, cy + half, -h, 0.0)), color=CELL_COLORS[n])
    pl.add_mesh(pv.Plane(center=(0, 0, -h), i_size=0.16, j_size=0.16), color="#dddddd")
    pl.camera_position = [(0.13, -0.17, 0.11), (0.0, 0.0, 0.015), (0, 0, 1)]
    pl.subplot(1, 0)
    pl.add_text(f"section at y = {1e3 * y_cut:.0f} mm (tool path)", font_size=9, color="black")
    pl.camera_position = [(0.0, y_cut - 0.25, 0.02), (0.0, y_cut, 0.02), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 0.042

    def draw(i):
        """Replace the moving actors (by name) with frame i's geometry."""
        grid.points = pos[i]
        grid["disp [mm]"] = disp_mm[i]
        pl.subplot(0, 0)
        pl.add_mesh(grid.extract_surface(algorithm="dataset_surface"), name="block",
                    scalar_bar_args=bar, **style)
        pl.add_mesh(sphere.translate(tool[i], inplace=False), name="tip", color="#b0b6bf",
                    opacity=0.45, smooth_shading=True)
        pl.add_mesh(shaft.translate(tool[i], inplace=False), name="shaft", color="#8a9099",
                    opacity=0.45, smooth_shading=True)
        pl.subplot(1, 0)
        pl.add_mesh(grid.slice(normal="y", origin=(0, y_cut, 0)), name="section",
                    show_scalar_bar=False, **style)
        pl.add_mesh(circle.translate(tool[i], inplace=False), name="outline", color="#444444",
                    line_width=2, style="wireframe")

    video_path = os.path.join(args.run, "indenter_contact.mp4")
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)

    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.axis("off")
    ax_z = fig.add_subplot(gs[0, 1])
    ax_x = fig.add_subplot(gs[1, 1], sharex=ax_z)
    for n in range(4):
        ax_z.plot(t, S[:, n, 2], color=CELL_COLORS[n], label=f"cell {n}")
    ax_z.plot(t, S[:, :, 2].sum(1) - weight, "k--", lw=1, label="cells - weight")
    ax_z.plot(t, -F[:, 2], color="0.6", lw=3, alpha=0.5, label="tool sensor")
    ax_z.set(ylabel="vertical [N]", title="Simulated load cells and tool force sensor")
    ax_z.legend(fontsize=7, ncol=3, loc="upper right")
    ax_x.plot(t, S[:, :, 0].sum(1), "k--", lw=1, label="cells x")
    ax_x.plot(t, -F[:, 0], color="0.6", lw=3, alpha=0.5, label="tool sensor x (reversed)")
    ax_x.set(xlabel="time [s]", ylabel="lateral x [N]")
    ax_x.legend(fontsize=7, loc="upper right")
    cursors = [ax.axvline(0, color="k", lw=0.8) for ax in (ax_z, ax_x)]
    for ax in (ax_z, ax_x):
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
        title.set_text(f"Rigid spherical indenter on FEM foam   t = {t[i]:.2f} s   "
                       f"tool force z {-F[i, 2]:.2f} N  x {-F[i, 0]:+.2f} N   (true scale)")
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    writer.close()
    plt.close(fig)
    pl.close()
    print(f"wrote {video_path} ({len(frames)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
