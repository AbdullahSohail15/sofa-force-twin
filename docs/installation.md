# Installation (Linux)

You need SOFA with its Python bindings (SofaPython3) and a few Python packages. The easiest way is
conda: the conda-forge package `sofa-python3` brings SOFA itself, so there is nothing to compile.

Tested on Ubuntu 22.04 x86-64 with conda-forge `sofa-python3` 25.12 and Python 3.12 (a clean
`conda env create` followed by every scene). Other recent Linux distributions should work the same
way. No GPU or display is needed.

## 1. Install conda (skip if you have it)

Any conda works (Miniforge, Miniconda, Anaconda). Miniforge uses conda-forge by default:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda init bash      # then open a new terminal
```

## 2. Create the environment

```bash
git clone https://github.com/AbdullahSohail15/sofa-force-twin.git
cd sofa-force-twin
conda env create -f environment.yml       # creates the env "sofa-twin" (~4 GB, ~5-10 min)
conda activate sofa-twin
```

`environment.yml` installs:

| Package | Why |
|---|---|
| `python=3.12` | SofaPython3 is built per Python version |
| `sofa-python3=25.12` | SOFA v25.12 + its Python bindings |
| `numpy`, `scipy` | arrays; `scipy.spatial.cKDTree` for the cutting rule |
| `matplotlib` | force plots |
| `pyvista` (VTK) | offscreen 3D rendering of the videos |
| `imageio`, `imageio-ffmpeg` | mp4 writing |

To install into an existing environment instead:
`conda install -c conda-forge python=3.12 sofa-python3=25.12 numpy scipy matplotlib pyvista imageio imageio-ffmpeg`.

## 3. Check the install

```bash
conda activate sofa-twin
export SOFA_ROOT=$CONDA_PREFIX            # run.sh does this for you; needed when calling python directly
python -c "import Sofa, Sofa.Simulation; print('SOFA OK')"
./run.sh 01                               # ~30 s
```

Expected end of the output:

```
== scene 01: scenes/block_on_supports.py -> outputs/block_on_supports
nodes 605  tets 2400  mass 12.00 g  support nodes [4, 4, 4, 4]  press nodes 5
saved outputs/block_on_supports/sim.npz
   simulated in 15 s
wrote outputs/block_on_supports/block_on_supports.mp4 (150 frames at 25 fps)
   outputs in outputs/block_on_supports
```

Then try the quick versions of the tool scenes, which simulate only the first seconds:

```bash
T_END=1 NO_RENDER=1 ./run.sh 05           # ~2 min: builds the full cutting scene and runs 1 s
```

## About `SOFA_ROOT`

conda's SOFA finds its plugins through the `SOFA_ROOT` environment variable. `run.sh` sets it to the
active conda env (`$CONDA_PREFIX`) when it is unset. If you run a scene with `python scenes/...`
directly, either `export SOFA_ROOT=$CONDA_PREFIX` once per terminal or make it permanent for the env:

```bash
conda activate sofa-twin
conda env config vars set SOFA_ROOT=$CONDA_PREFIX   # with the env active, so $CONDA_PREFIX is its path
conda deactivate && conda activate sofa-twin         # re-activate to apply
```

Without it, SOFA prints "Checking SOFA_ROOT ..." and may fail to load plugins.

## Running

```bash
./run.sh 01                 # one scene: simulate, check, render  -> outputs/block_on_supports/
./run.sh 05 runs/my_cut     # choose the output folder
T_END=2 ./run.sh 04         # only the first 2 s (scenes 03-06)
NO_RENDER=1 ./run.sh 06     # skip the video (rendering takes a few minutes for the tool scenes)
./run.sh all                # all six scenes one after another (~1.5 h)
PY=/path/to/python ./run.sh 01   # use another Python
```

Or call the pieces yourself (after `export SOFA_ROOT=$CONDA_PREFIX`):

```bash
python scenes/rig_cut.py --out outputs/rig_cut            # simulate -> sim.npz
python scripts/check_balance.py --run outputs/rig_cut     # force/moment balance
python scripts/render_tools.py --run outputs/rig_cut      # mp4 + forces.png + forces.csv
```

The scenes are single-threaded. To use a many-core machine, run several at once in separate
terminals (or with `&`), each with its own output folder.

## Alternative: the SOFA binary release (v26.06)

The scenes also run unchanged on the official SOFA v26.06 binaries (checked on macOS: same cut
elements, forces within a few mN of the conda 25.12 run). The conda route above is simpler; use this
only if you need SOFA 26.06 or its GUI (`runSofa`).

1. Download `SOFA_v26.06.00_Linux.zip` from <https://github.com/sofa-framework/sofa/releases> and
   unzip it, e.g. to `~/sofa/SOFA_v26.06.00_Linux`.
2. Create a Python **3.12** venv (SofaPython3 in that release is built for 3.12) and install the
   packages: `python3.12 -m venv ~/sofa/venv && ~/sofa/venv/bin/pip install numpy scipy matplotlib pyvista imageio imageio-ffmpeg`.
3. Point Python at SOFA (put this in a small wrapper script or your shell profile):
   ```bash
   export SOFA_ROOT=~/sofa/SOFA_v26.06.00_Linux
   export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
   export LD_LIBRARY_PATH=$SOFA_ROOT/lib:$LD_LIBRARY_PATH
   ```
4. Run with `PY=~/sofa/venv/bin/python ./run.sh 01`.

SOFA 26.06 warns that `RequiredPlugin` should use `pluginName` instead of `name`; this is harmless.
These binary steps have not been re-tested on Linux for this repo; the conda route has.

## Viewing a scene in the SOFA GUI

The conda package has no GUI. With the binary release you can open the simpler scenes in `runSofa`
(`runSofa -l SofaPython3 scenes/block_on_supports.py`) to look around. The tool scenes are driven
from Python `main()` functions and record in a controller, so run them headless with `run.sh`.

## Common errors

| Message | Cause / fix |
|---|---|
| `error: 'python' cannot import Sofa` | The env is not active: `conda activate sofa-twin` |
| `ModuleNotFoundError: No module named 'Sofa'` | Same, or a Python version different from SOFA's (must be 3.12 here) |
| `Checking SOFA_ROOT ...` then plugin load errors | Set `SOFA_ROOT` (see above) |
| `bad X server connection` (VTK) | Harmless: rendering is offscreen. `run.sh` hides it |
| Blank or black video frames on a headless server | Install OSMesa/EGL VTK or run under `xvfb-run -a ./run.sh 01` |
| `libGL.so.1: cannot open shared object file` | `sudo apt install libgl1 libglu1-mesa libxrender1` |
| A run seems stuck | The tool scenes take ~1-3 s per 10 ms step; watch the `t=...` progress lines (every 0.5-1 s of simulated time) |

More SOFA-specific problems are in [troubleshooting.md](troubleshooting.md).
