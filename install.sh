#!/usr/bin/env bash
# LISA Analysis Tools — central development installer.
#
# This script lives inside the LAT repo and orchestrates a full dev
# install of the LISA Analysis Tools stack: GBT, Eryn, LAT (this repo),
# BBHx, GBGPU, LATW (tutorials), plus the external phentax + FEW.
#
# Assumes you have already cloned this repo:
#   git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
#   cd lisa-analysis-tools
#   ./install.sh
#
# Sibling repos are cloned into the same parent directory as this clone,
# producing a layout like:
#
#   <dev_root>/
#     ├── lisa-analysis-tools/    (this repo, LAT)
#     ├── GPUBackendTools/        (GBT)
#     ├── Eryn/
#     ├── BBHx/
#     ├── GBGPU/
#     ├── LATW/
#     └── FastEMRIWaveforms/      (optional)
#
# Re-runnable: existing sibling clones are reused, not re-cloned.
#
# Optional env vars:
#   ORG=lisa-analysis-tools       # GitHub org for org-owned repos
#   SKIP_FEW=1                    # skip FastEMRIWaveforms (EMRI users only)
#   SKIP_PHENTAX=1                # skip phentax (MBH PhenomTHM)
#   SKIP_LISA_ON_GPU=1            # skip retiring lisa-on-gpu (default)
#   GIT_PULL=1                    # (default) after checkout, fast-forward-pull
#                                 # origin/<branch> for each package. The install
#                                 # ABORTS if any package has diverged from its
#                                 # remote (non-fast-forward) so you never build
#                                 # on a silently stale/conflicting tree. Set
#                                 # GIT_PULL=0 to skip pulling (offline / pinned).
#   GBT_LAPACKE_DETECT_WITH=PKGCONFIG
#                                 # (default) how the compiled packages locate
#                                 # LAPACKE: AUTO | CMAKE | PKGCONFIG | DISABLE.
#                                 # Passed to every package (GBT, LAT, BBHx,
#                                 # GBGPU, FEW) as
#                                 # cmake.define.GBT_LAPACKE_DETECT_WITH.
#   GBT_LAPACKE_FETCH=            # AUTO | ON | OFF — download + build Reference
#                                 # LAPACK when detection fails. Unset (default)
#                                 # keeps each package's own default (AUTO).
#   GBT_LAPACKE_EXTRA_LIBS=       # extra libs linked alongside LAPACKE, e.g.
#                                 # "gfortran". Unset keeps package defaults.

set -euo pipefail

ORG="${ORG:-lisa-analysis-tools}"
SKIP_FEW="${SKIP_FEW:-0}"
SKIP_PHENTAX="${SKIP_PHENTAX:-0}"
SKIP_LISA_ON_GPU="${SKIP_LISA_ON_GPU:-1}"
GIT_PULL="${GIT_PULL:-1}"
GBT_LAPACKE_DETECT_WITH="${GBT_LAPACKE_DETECT_WITH:-PKGCONFIG}"
GBT_LAPACKE_FETCH="${GBT_LAPACKE_FETCH:-}"
GBT_LAPACKE_EXTRA_LIBS="${GBT_LAPACKE_EXTRA_LIBS:-}"

LAT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEV_ROOT="$(dirname "$LAT_DIR")"
CONSTRAINTS="${LAT_DIR}/constraints/sprint.txt"

if [ ! -f "$CONSTRAINTS" ]; then
    echo "ERROR: constraint file not found at $CONSTRAINTS" >&2
    exit 1
fi
export PIP_CONSTRAINT="$CONSTRAINTS"

echo "LAT repo:    $LAT_DIR"
echo "Dev root:    $DEV_ROOT"
echo "Constraints: $PIP_CONSTRAINT"
echo "LAPACKE:     detect=${GBT_LAPACKE_DETECT_WITH}" \
     "fetch=${GBT_LAPACKE_FETCH:-<package default>}" \
     "extra_libs=${GBT_LAPACKE_EXTRA_LIBS:-<package default>}"
echo ""

# ----------------------------------------------------------------------
# Base build deps
# ----------------------------------------------------------------------
pip install --upgrade pip
pip install \
    scikit_build_core uv uv_build setuptools_scm pybind11 \
    numpy scipy ipython jupyter astropy lisaconstants Cython

# Optional: macOS + brew lapack
#export CC=/usr/bin/clang
#export CXX=/usr/bin/clang++
#export PKG_CONFIG_PATH="/opt/homebrew/opt/lapack/lib/pkgconfig:$PKG_CONFIG_PATH"

# The GBT_LAPACKE_* option family is shared by every compiled package in the
# chain (GBT's get_lapacke() is the single detector; LAT/BBHx/GBGPU include
# it, and FEW's gpu_backend branch reads the same names), so one flag set is
# passed to all of them. FETCH / EXTRA_LIBS are only forwarded when the user
# set them, keeping each package's own defaults otherwise.
LAPACKE_FLAGS=("--config-settings=cmake.define.GBT_LAPACKE_DETECT_WITH=${GBT_LAPACKE_DETECT_WITH}")
if [ -n "$GBT_LAPACKE_FETCH" ]; then
    LAPACKE_FLAGS+=("--config-settings=cmake.define.GBT_LAPACKE_FETCH=${GBT_LAPACKE_FETCH}")
fi
if [ -n "$GBT_LAPACKE_EXTRA_LIBS" ]; then
    LAPACKE_FLAGS+=("--config-settings=cmake.define.GBT_LAPACKE_EXTRA_LIBS=${GBT_LAPACKE_EXTRA_LIBS}")
fi

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
clone_or_reuse_sibling() {
    local org="$1" repo="$2"
    cd "$DEV_ROOT"
    if [ ! -d "$repo" ]; then
        echo ""
        echo "===> cloning $org/$repo into $DEV_ROOT/$repo"
        git clone "https://github.com/${org}/${repo}.git"
    else
        echo ""
        echo "===> reusing $DEV_ROOT/$repo (already cloned)"
    fi
}

# Fast-forward-pull origin/<branch>. Aborts the whole install if the local
# branch has diverged from its remote (non-fast-forward) or the working tree
# conflicts, so a stale/conflicting checkout is never silently built. Skip with
# GIT_PULL=0.
pull_or_stop() {
    local repo_path="$1" branch="$2"
    local name; name="$(basename "$repo_path")"
    [ "$GIT_PULL" = "1" ] || { echo "===> GIT_PULL=0: skipping pull for $name"; return 0; }
    echo "===> pulling origin/$branch in $name (fast-forward only)"
    if ! git -C "$repo_path" pull --ff-only origin "$branch"; then
        echo "" >&2
        echo "ERROR: could not fast-forward '$name' to origin/$branch." >&2
        echo "       The local '$branch' has diverged from the remote (local commits" >&2
        echo "       not on origin, or conflicting local changes), so a plain pull is" >&2
        echo "       not allowed. Resolve it manually, e.g.:" >&2
        echo "         cd $repo_path" >&2
        echo "         git status" >&2
        echo "         git log --oneline --graph --left-right HEAD...origin/$branch" >&2
        echo "       then re-run ./install.sh  (or set GIT_PULL=0 to skip pulls)." >&2
        exit 1
    fi
}

editable_install() {
    local repo_path="$1" branch="$2"
    shift 2
    cd "$repo_path"
    git checkout "$branch"
    pull_or_stop "$repo_path" "$branch"
    pip install --no-build-isolation -e . "$@"
}

# ----------------------------------------------------------------------
# Dependency-ordered installs
#
#   Foundation:     GPUBackendTools  (splines, cuda_complex)
#                   Eryn             (standalone sampler)
#   Mid:            LAT  (this repo; depends on GBT)
#   Source classes: BBHx, GBGPU      (depend on LAT + GBT)
#   Tutorials:      LATW             (pure-Python, no compile)
#   External:       phentax, FEW
# ----------------------------------------------------------------------

clone_or_reuse_sibling "$ORG" GPUBackendTools
editable_install "$DEV_ROOT/GPUBackendTools" spline "${LAPACKE_FLAGS[@]}"

clone_or_reuse_sibling "$ORG" Eryn
editable_install "$DEV_ROOT/Eryn" dev "${LAPACKE_FLAGS[@]}"

echo ""
echo "===> installing LAT (this repo: $LAT_DIR)"
editable_install "$LAT_DIR" dev "${LAPACKE_FLAGS[@]}"

clone_or_reuse_sibling "$ORG" BBHx
editable_install "$DEV_ROOT/BBHx" dev "${LAPACKE_FLAGS[@]}"

clone_or_reuse_sibling "$ORG" GBGPU
editable_install "$DEV_ROOT/GBGPU" dev "${LAPACKE_FLAGS[@]}"

if clone_or_reuse_sibling "$ORG" LATW; then
    # Tutorials repo: pure-Python, never pip-installed. The dev branch is the
    # dev-stack tutorial set (branch policy: LATW main <-> pip releases,
    # LATW dev <-> this install.sh stack).
    git -C "$DEV_ROOT/LATW" checkout dev
    pull_or_stop "$DEV_ROOT/LATW" dev
else
    echo "WARN: LATW clone failed (skipping — tutorials repo is optional)"
fi

# ----------------------------------------------------------------------
# Optional / external
# ----------------------------------------------------------------------
if [ "$SKIP_PHENTAX" != "1" ]; then
    echo ""
    echo "===> installing phentax (MBH IMRPhenomTHM)"
    # Equivalent to BBHx's `phentax` extra (pip install 'bbhx[phentax]');
    # installed directly here since BBHx is already editable-installed above.
    pip install git+https://github.com/asantini29/phentax.git
fi

if [ "$SKIP_FEW" != "1" ]; then
    clone_or_reuse_sibling BlackHolePerturbationToolkit FastEMRIWaveforms
    # FEW's gpu_backend branch reads the shared GBT_LAPACKE_* option names
    # (commit f5f51416), so it takes the same flag set as everything else.
    editable_install "$DEV_ROOT/FastEMRIWaveforms" gpu_backend "${LAPACKE_FLAGS[@]}"
fi

# lisa-on-gpu is being retired — opt-in only.
if [ "$SKIP_LISA_ON_GPU" != "1" ]; then
    clone_or_reuse_sibling mikekatz04 lisa-on-gpu
    editable_install "$DEV_ROOT/lisa-on-gpu" tdi_on_fly "${LAPACKE_FLAGS[@]}"
fi

echo ""
echo "==> done. Verify with:"
echo "    python -c 'import lisatools, eryn, bbhx, gbgpu, gpubackendtools; print(\"all import OK\")'"
