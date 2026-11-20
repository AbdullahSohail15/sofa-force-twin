#!/usr/bin/env bash
# Run one scene of the SOFA force twin: simulate, then render the video + force plot.
#
#   ./run.sh 01            # or: block | indenter | rig | tools | cut | dissect | all
#   ./run.sh 05 my_runs/cut_a      # optional output folder (default outputs/<scene name>)
#   T_END=2 ./run.sh 04            # simulate only the first 2 s (quick test, scenes 03-06)
#   NO_RENDER=1 ./run.sh 05        # simulation only, no video
#
# Uses the `python` on your PATH (activate the conda env first: `conda activate sofa-twin`),
# or set PY=/path/to/python.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
# conda's SOFA finds its plugins through SOFA_ROOT; default it to the active env.
if [[ -z "${SOFA_ROOT:-}" && -n "${CONDA_PREFIX:-}" ]]; then export SOFA_ROOT=$CONDA_PREFIX; fi
"$PY" -c "import Sofa" >/dev/null 2>&1 || {
    echo "error: '$PY' cannot import Sofa. Run 'conda activate sofa-twin' first (see docs/installation.md)." >&2; exit 1; }

# SOFA prints a lot of [INFO] lines and VTK warns about the missing X server; hide them.
quiet() { grep --line-buffered -v -E "^\[INFO\]|^-+$|SOFA_ROOT|SOFAPYTHON3|DEPRECATED|bad X server|^Warning" || true; }

scene() {   # scene() <number>  -> sets SCRIPT NAME RENDER CHECK
    case "$1" in
        01|block)    SCRIPT=block_on_supports; RENDER=render_video;    CHECK=0 ;;
        02|indenter) SCRIPT=indenter_contact;  RENDER=render_indenter; CHECK=0 ;;
        03|rig)      SCRIPT=rig_indenter;      RENDER=render_rig;      CHECK=0 ;;
        04|tools)    SCRIPT=rig_tools;         RENDER=render_tools;    CHECK=1 ;;
        05|cut)      SCRIPT=rig_cut;           RENDER=render_tools;    CHECK=1 ;;
        06|dissect)  SCRIPT=rig_dissect;       RENDER=render_tools;    CHECK=1 ;;
        *) echo "unknown scene '$1' (use 01-06, block, indenter, rig, tools, cut, dissect or all)" >&2; exit 2 ;;
    esac
}

run_one() {
    scene "$1"
    local out=${2:-outputs/$SCRIPT}
    mkdir -p "$out"
    echo "== scene $1: scenes/$SCRIPT.py -> $out"
    local extra=()
    if [[ -n "${T_END:-}" ]]; then
        case $SCRIPT in block_on_supports|indenter_contact) echo "   (T_END ignored: scenes 01-02 have a fixed length)" ;;
                        *) extra=(--t-end "$T_END") ;; esac
    fi
    local t0=$SECONDS
    "$PY" "scenes/$SCRIPT.py" --out "$out" "${extra[@]}" 2>&1 | quiet
    echo "   simulated in $((SECONDS - t0)) s"
    if [[ $CHECK == 1 ]]; then "$PY" scripts/check_balance.py --run "$out" 2>&1 | quiet; fi
    if [[ -z "${NO_RENDER:-}" ]]; then
        "$PY" "scripts/$RENDER.py" --run "$out" 2>&1 | quiet
        echo "   outputs in $out"
    fi
}

if [[ "${1:-}" == all ]]; then
    for s in 01 02 03 04 05 06; do run_one $s; done
else
    run_one "${1:?usage: ./run.sh <01-06|all> [output folder]}" "${2:-}"
fi
