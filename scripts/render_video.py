"""Render a recorded block-on-supports run to an mp4, plus a force plot and CSV.

Each video frame is the deformed block (PyVista, offscreen) next to the four
load-cell traces with a moving time cursor.

    python scripts/render_video.py \
        --run outputs/block_on_supports --fps 25 --exaggerate 5
"""

import argparse
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

CELL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def save_forces(d, out):
    t, S, P = d["t"], d["support"], d["press"]
    header = "t_s,press_N," + ",".join(f"cell{n}_{a}_N" for n in range(4) for a in "xyz")
    rows = np.column_stack([t, P, S.reshape(len(t), -1)])
    np.savetxt(os.path.join(out, "forces.csv"), rows, delimiter=",", header=header, comments="", fmt="%.6g")

    fig, ax = plt.subplots(figsize=(8, 4))
    for n in range(4):
        ax.plot(t, S[:, n, 2], color=CELL_COLORS[n], label=f"cell {n}")
    ax.plot(t, S[:, :, 2].sum(1), "k--", lw=1, label="sum")
    ax.plot(t, P + float(d["mass"]) * 9.81, color="0.6", lw=3, alpha=0.5, label="applied + weight")
    ax.set(xlabel="time [s]", ylabel="vertical reaction [N]", title="Support load cells")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "forces.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/block_on_supports")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--exaggerate", type=float, default=5.0,
                        help="display-only displacement scale; forces are unaffected")
    args = parser.parse_args()

    d = np.load(os.path.join(args.run, "sim.npz"))
    t, pos, rest, tets = d["t"], d["pos"], d["rest"], d["tets"]
    S, P = d["support"], d["press"]
    save_forces(d, args.run)

    stride = max(1, int(round(1.0 / (args.fps * float(d["dt"])))))
    frames = range(0, len(t), stride)
    disp_mm = 1e3 * np.linalg.norm(pos - rest, axis=2)
    clim = (0.0, float(disp_mm.max()))

    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, rest.copy())
    surf = grid.extract_surface(algorithm="dataset_surface")
    ids = surf["vtkOriginalPointIds"]

    pl = pv.Plotter(off_screen=True, window_size=(900, 720))
    pl.set_background("white")
    surf["disp [mm]"] = disp_mm[0, ids]
    pl.add_mesh(surf, scalars="disp [mm]", clim=clim, cmap="viridis", show_edges=True,
                edge_color="#333333", line_width=0.5,
                scalar_bar_args=dict(title="|displacement| [mm]", color="black", vertical=True,
                                     position_x=0.86, position_y=0.25, height=0.5))
    half, h = float(d["support_half"]), 0.03
    for n, (cx, cy) in enumerate(d["support_xy"]):
        pl.add_mesh(pv.Box((cx - half, cx + half, cy - half, cy + half, -h, 0.0)), color=CELL_COLORS[n])
    pl.add_mesh(pv.Plane(center=(0, 0, -h), i_size=0.16, j_size=0.16), color="#dddddd")
    pl.camera_position = [(0.17, -0.21, 0.16), (0.0, 0.0, 0.0), (0, 0, 1)]
    arrow_actor = None

    os.makedirs(args.run, exist_ok=True)
    video_path = os.path.join(args.run, "block_on_supports.mp4")
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=8)

    fig, (ax_img, ax_f) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax_img.axis("off")
    img_artist = None
    for n in range(4):
        ax_f.plot(t, S[:, n, 2], color=CELL_COLORS[n], label=f"cell {n}")
    ax_f.plot(t, S[:, :, 2].sum(1), "k--", lw=1, label="sum of cells")
    ax_f.plot(t, P, color="0.5", lw=1, label="press force")
    ax_f.set(xlabel="time [s]", ylabel="vertical force [N]", title="Simulated load cells")
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=8, loc="upper right")
    cursor = ax_f.axvline(0, color="k", lw=0.8)
    title = fig.suptitle("")

    for i in frames:
        surf.points = rest[ids] + args.exaggerate * (pos[i, ids] - rest[ids])
        surf["disp [mm]"] = disp_mm[i, ids]
        if arrow_actor is not None:
            pl.remove_actor(arrow_actor)
            arrow_actor = None
        if P[i] > 1e-6:
            length = 0.04 * P[i] / P.max()
            pn = d["press_nodes"]
            tip = rest[pn].mean(0) + args.exaggerate * (pos[i, pn] - rest[pn]).mean(0)
            arrow_actor = pl.add_mesh(pv.Arrow(start=tip + (0, 0, length), direction=(0, 0, -1), scale=length,
                                               tip_length=0.3, shaft_radius=0.04, tip_radius=0.1), color="crimson")
        img = pl.screenshot(return_img=True)
        if img_artist is None:
            img_artist = ax_img.imshow(img)
        else:
            img_artist.set_data(img)
        cursor.set_xdata([t[i], t[i]])
        cells = "  ".join(f"c{n} {S[i, n, 2]:.3f}" for n in range(4))
        title.set_text(f"FEM foam block on 4 load cells   t = {t[i]:.2f} s   press = {P[i]:.2f} N   "
                       f"[{cells}] N   (displacement shown x{args.exaggerate:g})")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(frame)
    writer.close()
    plt.close(fig)
    pl.close()
    print(f"wrote {video_path} ({len(frames)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
