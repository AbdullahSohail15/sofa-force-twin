"""Build the scalpel tool (assets/scalpel.npz): a #11-style surgical blade held in a cap that
mounts on the EndoWrist yaw pivot in place of the scissor jaws.

Same tool frame as the dVRK parts (tools.py): origin on the jaw pivot, the tool points to -z,
the pivot pin runs along y. The blade is a flat plate in the x-z plane (thin along y) with its
cutting edge facing +x and its point at tools.KNIFE_TIP.

Parts:
- cap: a hub that sits between the clevis tines on the pivot pin (like the scissor hub), a neck,
  and a round ferrule with a groove and a chamfered nose with the blade slot. Render only, PEEK.
- blade: stainless #11 shape (straight spine, straight edge to the point), 0.36 mm thick with a
  ground bevel along the edge. 20 mm stands out of the cap.
- col: collision mesh of blade + cap. The blade is thickened to 0.8 mm (thin plates tunnel
  through the foam contact) and resampled (vertices every ~1.6 mm, inside the 2.5 mm cut radius) because the tie cut
  in scenes/cutting.py looks for ties near the blade's collision vertices.

    python scripts/make_scalpel.py
"""

import os

import numpy as np
import trimesh
from manifold3d import Manifold

MM = 1e-3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "scalpel.npz")

# Blade outline in the x-z plane [mm]: tang (inside the cap) down to the point.
BLADE = np.array([(-1.6, -6.0), (1.6, -6.0), (1.6, -11.5), (2.0, -13.0), (-1.2, -31.0), (-1.6, -27.5)])
# The same outline with the edge (heel -> point) moved in by the bevel width.
BLADE_INNER = np.array([(-1.6, -6.0), (1.6, -6.0), (1.6, -11.5), (1.25, -13.0), (-1.45, -28.4), (-1.6, -27.0)])
TIP = np.array([-1.2, 0.0, -31.0]) * MM
NOSE_Z = -11.0                  # bottom of the cap [mm]


def plate(outline, half_t):
    return np.array([(x, y, z) for x, z in outline for y in (-half_t, half_t)])


def blade(half_t=0.18, edge_half_t=0.015):
    """Bevelled blade: hull of the full outline (knife-thin) and the inner outline (full thickness)."""
    return Manifold.hull_points(np.r_[plate(BLADE, edge_half_t), plate(BLADE_INNER, half_t)])


def zcyl(r, z0, z1, n=48):
    return Manifold.cylinder(z1 - z0, r, r, n).translate((0, 0, z0))


def cap():
    hub = Manifold.cylinder(3.0, 2.3, 2.3, 48).rotate((90, 0, 0)).translate((0, 1.5, 0))     # along y
    pin = Manifold.cylinder(3.4, 0.55, 0.55, 24).rotate((90, 0, 0)).translate((0, 1.7, 0))
    neck = Manifold.hull_points(np.array([(x, y, z) for x in (-1.9, 1.9) for y in (-1.45, 1.45)
                                          for z in (-3.6, -0.5)]))
    ferrule = Manifold.batch_hull([zcyl(2.15, -3.25, -2.95), zcyl(2.6, -9.9, -3.4), zcyl(1.9, NOSE_Z, -10.6)])
    groove = zcyl(2.8, -5.4, -5.0) - zcyl(2.38, -5.5, -4.9)
    slot = Manifold.hull_points(np.array([(x, y, z) for x in (-2.1, 2.1) for y in (-0.3, 0.3)
                                          for z in (NOSE_Z - 0.1, -10.3)]))
    screw = Manifold.cylinder(0.5, 0.7, 0.7, 24).rotate((-90, 0, 0)).translate((0, 2.35, -7.6))
    return (hub - pin) + neck + ferrule - groove - slot + screw


def arrays(m, scale=MM):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3] * scale, np.asarray(mesh.tri_verts, np.int32)


def collision():
    b_pts, b_tris = arrays(blade(half_t=0.4, edge_half_t=0.25))
    v, f = trimesh.remesh.subdivide_to_size(b_pts, b_tris, max_edge=0.0016)
    b = trimesh.Trimesh(v, f, process=True)
    c = trimesh.Trimesh(*arrays(Manifold.batch_hull([zcyl(2.6, NOSE_Z, -3.2, n=12), zcyl(2.0, -3.2, 0.0, n=12)])))
    m = trimesh.util.concatenate([b, c])
    return np.asarray(m.vertices), np.asarray(m.faces, np.int32)


def main():
    b_pts, b_tris = arrays(blade())
    c_pts, c_tris = arrays(cap())
    col_pts, col_tris = collision()
    np.savez_compressed(OUT, scalpel_vis_pts=b_pts, scalpel_vis_tris=b_tris,
                        scalpel_cap_vis_pts=c_pts, scalpel_cap_vis_tris=c_tris,
                        scalpel_col_pts=col_pts, scalpel_col_tris=col_tris, scalpel_tip=TIP)
    print(f"blade {len(b_tris)} tris, cap {len(c_tris)} tris, collision {len(col_pts)} vertices "
          f"{len(col_tris)} tris -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
