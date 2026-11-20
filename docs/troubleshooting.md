# Troubleshooting and SOFA pitfalls

Problems we ran into while building the scenes, and what fixed them. Read this before changing
solvers or force fields.

## Installation and running

See the table at the end of [installation.md](installation.md#common-errors) for
`import Sofa` errors, `SOFA_ROOT`, VTK/X-server messages and headless rendering.

## SOFA behaviour

**Harmless messages.** `[WARNING] [SpringForceField(ties)] ... list has a different size than
indices1` (scenes 05, 06) and `RequiredPlugin ... pluginName` (SOFA 26.06) can be ignored.

**Misspelt or unknown component fields are ignored silently.** `addObject("X", stifness=...)`
creates the component with its default stiffness and only logs a warning (if that). After
`Sofa.Simulation.init(root)`, assert that the values you rely on took effect, e.g.
`assert len(root.Sponge.fem.youngModulus.value) == n_tets` (see `rig_tools.check_parts`).

**`RestShapeSpringsForceField` with a per-point stiffness vector (SOFA 25.12 bug).** The force uses
the right value, but `buildStiffnessMatrix` looks the stiffness up by node id instead of by spring,
so the implicit solver does not see the springs and stiff ones explode. Use **one scalar stiffness**
and list the active nodes in `points` (as the grasp springs do).

**`RestShapeSpringsForceField` uses `points`, not `indices`,** in SOFA 25.12.

**Rigid bodies: set the inertia.** `UniformMass` on a `Rigid3d` object defaults to an inertia of
1 kg·m², absurd for a 3 g part. Give it explicitly: `vertexMass="m 1 I 0 0 0 I 0 0 0 I"` (mass,
volume, inertia matrix per unit mass).

**Rigid positions** are `[[x, y, z, qx, qy, qz, qw]]` (quaternion last, scalar last).

**`Sofa.Core.Controller` subclasses:** call `super().__init__(*args, **kwargs)` **before** setting
any attribute, or SOFA crashes (segfault).

**Contact with a tool needs `LinearSolverConstraintCorrection` on the tool.**
`UncoupledConstraintCorrection` gave the tool no contact reaction at all.

**No Rayleigh stiffness damping on the sponge or the tools.** It damps against absolute velocity:
on the sensor springs it reads as drag, and on the sponge it absorbed up to ~10 N that the cells
never saw while the stiff grasp springs moved. Use `rayleighStiffness=0`.

**Iterative solver tolerance.** `CGLinearSolver`'s tolerance is an absolute squared residual. For
the 4 g sponge 1e-12 stopped too early (the sponge drifted up and lost ~10 % of its weight reading);
1e-18 worked. The rig scenes now use the direct `SparseLDLSolver`, which has no such issue.

**`CGLinearSolver` + `GenericConstraintCorrection`** costs one CG solve per contact row (~1–2 s per
step with contact). The direct `SparseLDLSolver` + `LinearSolverConstraintCorrection` costs ~0.6 s
per step regardless of the number of contacts.

**`AsyncSparseLDLSolver`** is ~15× faster per step (it reuses the previous factorisation) but
diverged when the grasp springs switched on. Worth retrying with a slower grasp ramp if you need
speed.

## Cutting

**Element deletion and `TetrahedronFEMForceField`.** After tetrahedra are removed,
`TetrahedronFEMForceField` applies its per-element Young's moduli to the wrong elements. Use
`TetrahedralCorotationalFEMForceField` with `localStiffnessFactor` for meshes that lose elements.
(The scenes cut by releasing tie springs instead, so the mesh never changes.)

**`TopologicalChangeProcessor` timing.** With `useDataInputs`, it removes elements on the same step
only if `timeToRemove` is already in the past: set it to `t - dt/2`.

**Python cannot add tetrahedra** in the SOFA 25.12 bindings, so true node splitting along an
arbitrary path is not available from Python; the tie-release cut needs the split plane to be built
into the mesh.

## Physics sanity checks

- At rest the eight cells must sum to the sponge weight (0.038 N for the default sponge).
- `python scripts/check_balance.py --run <run>`: the vertical tool force must equal cells − weight
  (median ~µN, worst ~10 mN at motion starts/stops). Larger errors mean something is wrong: a
  missing force in the log, too few constraint iterations, or a solver that did not converge.
- `python scripts/compare_runs.py --a <run1> --b <run2>`: compares two runs of the same scene
  (cells, tool forces, node positions, cut ties), e.g. after changing a solver or across machines.
