"""Zero-width cutting by releasing ties across a split plane (scenes 05 and 06).

The sponge is split along a node plane (rig.split_mesh) and every node pair across it is held
together by a stiff zero-length spring (rig_tools.TIE_K). Uncut, the plane behaves like the
rest of the foam. Cutting releases ties: nothing is removed, so there is no kerf. The two faces
part only where the blade (or a pull, as in a dissection) holds them apart, and the foam closes
again behind the blade.

Cutting rule: each step, if the blade pushes on the sponge along its direction of motion harder
than the cutting resistance F_CUT, every tie whose two nodes lie within RADIUS of the blade's
collision surface swept up to REACH ahead along its motion is released (the crack runs a few mm
ahead of the edge). F_CUT is a guess to be calibrated against real cuts.

On release the two nodes of a tie are moved NUDGE apart, each to its own side of the plane. They
coincide exactly on the blade's midplane otherwise, and contact cannot tell which side of the
blade a node belongs to: it pushes it ahead and drags it along. Nudged, the blade's wedge pushes
each side its own way.
"""
import numpy as np
from scipy.spatial import cKDTree

from rig_tools import DT, Recorder


class TieCutRecorder(Recorder):
    """Scene 04's recorder plus the tie-release cutting rule and its log."""

    f_cut = 0.8              # cutting resistance: blade push that cuts [N]
    reach = 0.004            # ties up to this far ahead of the blade are cut [m]
    radius = 0.0025          # ...and up to this far from its surface [m]
    nudge = 0.0008           # each released node moves this far to its own side [m]

    def __init__(self, *args, **kwargs):
        blade_path, pairs, axis = kwargs.pop("blade_path"), kwargs.pop("pairs"), kwargs.pop("split_axis")
        f_cut, radius = kwargs.pop("f_cut", None), kwargs.pop("radius", None)
        super().__init__(*args, **kwargs)  # must run before any attribute is set
        self.blade_path, self.pairs = blade_path, pairs
        if f_cut is not None:
            self.f_cut = f_cut
        if radius is not None:
            self.radius = radius
        self.normal = np.eye(3)[axis]                     # from the - side (pairs[:, 0]) to the + side
        self.ties = None                                  # SpringForceField, set once the scene exists
        self.blade = next(p for p in self.parts if p.name == "blade")
        self.tie_step = np.full(len(pairs), -1)           # step each tie was cut at, -1 if intact
        self.log.update({"blade_push": [], "ties_cut": []})

    def direction(self, t):
        d = self.blade_path(t + DT) - self.blade_path(t - DT)
        n = np.linalg.norm(d)
        return d / n if n > 0 else None

    def onAnimateBeginEvent(self, event):
        super().onAnimateBeginEvent(event)
        self.cut(self.getContext().getTime())

    def cut(self, t):
        u = self.direction(t)
        if u is None or not self.log["t"] or float(self.log["blade_force"][-1] @ u) < self.f_cut:
            return
        live = np.flatnonzero(self.tie_step < 0)
        if not len(live):
            return
        x = np.array(self.sponge.position.value)
        blade = np.array(self.blade.dofs.position.value)
        swept = np.concatenate([blade + s * u for s in np.linspace(0.0, self.reach, 5)])
        tree = cKDTree(swept)
        a, b = self.pairs[live, 0], self.pairs[live, 1]
        near = (tree.query(x[a])[0] < self.radius) & (tree.query(x[b])[0] < self.radius)
        ids = live[near]
        if len(ids):
            with self.sponge.position.writeableArray() as w:
                w[self.pairs[ids, 0]] -= self.nudge / 2 * self.normal
                w[self.pairs[ids, 1]] += self.nudge / 2 * self.normal
            self.tie_step[ids] = len(self.log["t"])
            k = np.array(self.ties.stiffness.value, dtype=float)
            k[ids] = 0.0
            self.ties.stiffness.value = k.tolist()

    def onAnimateEndEvent(self, event):
        super().onAnimateEndEvent(event)
        u = self.direction(self.getContext().getTime())
        self.log["blade_push"].append(float(self.log["blade_force"][-1] @ u) if u is not None else 0.0)
        self.log["ties_cut"].append(int((self.tie_step >= 0).sum()))
