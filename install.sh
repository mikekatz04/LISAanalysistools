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
#   ./install.sh                # full dev install (clone + pull + pip)
#   ./install.sh --pull-only    # update-only: checkout + ff-pull every repo,
#                               # no pip/compile (see PULL_ONLY below)
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
#   SKIP_GBT=1                    # pull GBT checkout but skip its recompile
#   SKIP_BBHX=1                   # pull BBHx checkout but skip its recompile
#   SKIP_GBGPU=1                  # pull GBGPU checkout but skip its (slow CUDA)
#                                 # recompile; rebuild by hand when .cu changed
#   GIT_PULL=1                    # (default) after checkout, fast-forward-pull
#                                 # origin/<branch> for each package. The install
#                                 # ABORTS if any package has diverged from its
#                                 # remote (non-fast-forward) so you never build
#                                 # on a silently stale/conflicting tree. Set
#                                 # GIT_PULL=0 to skip pulling (offline / pinned).
#   PULL_ONLY=0                   # set to 1 (or pass --pull-only) to ONLY
#                                 # checkout + fast-forward-pull every repo and
#                                 # skip ALL pip installs. Every package is
#                                 # editable-installed (Python imports straight
#                                 # from the source trees), so this is a full
#                                 # Python-level update with no C/CUDA rebuild.
#                                 # Re-run the full ./install.sh when native
#                                 # code, dependencies, or packaging change.
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
SKIP_GBT="${SKIP_GBT:-0}"
SKIP_BBHX="${SKIP_BBHX:-0}"
SKIP_GBGPU="${SKIP_GBGPU:-0}"
GIT_PULL="${GIT_PULL:-1}"
PULL_ONLY="${PULL_ONLY:-0}"
if [ "${1:-}" = "--pull-only" ] || [ "${1:-}" = "pull" ]; then
    PULL_ONLY=1
fi
if [ "$PULL_ONLY" = "1" ]; then
    # pulling IS the point of this mode
    GIT_PULL=1
fi
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
if [ "$PULL_ONLY" = "1" ]; then
    echo "Mode:        pull-only (checkout + ff-pull; NO pip installs)"
else
    echo "Mode:        full install"
fi
echo "LAPACKE:     detect=${GBT_LAPACKE_DETECT_WITH}" \
     "fetch=${GBT_LAPACKE_FETCH:-<package default>}" \
     "extra_libs=${GBT_LAPACKE_EXTRA_LIBS:-<package default>}"
echo ""

# ----------------------------------------------------------------------
# Base build deps
# ----------------------------------------------------------------------
if [ "$PULL_ONLY" != "1" ]; then
    pip install --upgrade pip
    pip install \
        scikit_build_core uv uv_build setuptools_scm pybind11 nanobind \
        numpy scipy ipython jupyter astropy lisaconstants Cython
fi

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
    if [ "$PULL_ONLY" = "1" ]; then
        # Editable installs import Python from the source tree, so the pull
        # above already IS the update — skip the (slow, C-compiling) pip step.
        echo "===> PULL_ONLY: skipping pip install for $(basename "$repo_path")"
        return 0
    fi
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

# SKIP_GBT / SKIP_BBHX / SKIP_GBGPU: pull the checkout (source stays current)
# but skip the compiled recompile, same as SKIP_FEW/SKIP_PHENTAX. Rebuild by
# hand when that package's native .cu/.cxx actually changed. NB: GBT is the
# base dependency -- only skip its recompile if it is already installed.
clone_or_reuse_sibling "$ORG" GPUBackendTools
if [ "$SKIP_GBT" != "1" ]; then
    editable_install "$DEV_ROOT/GPUBackendTools" spline "${LAPACKE_FLAGS[@]}"
else
    echo "===> SKIP_GBT=1: checkout pulled, skipping GPUBackendTools recompile"
fi

clone_or_reuse_sibling "$ORG" Eryn
editable_install "$DEV_ROOT/Eryn" dev "${LAPACKE_FLAGS[@]}"

echo ""
echo "===> installing LAT (this repo: $LAT_DIR)"
editable_install "$LAT_DIR" dev "${LAPACKE_FLAGS[@]}"

clone_or_reuse_sibling "$ORG" BBHx
if [ "$SKIP_BBHX" != "1" ]; then
    editable_install "$DEV_ROOT/BBHx" dev "${LAPACKE_FLAGS[@]}"
else
    echo "===> SKIP_BBHX=1: checkout pulled, skipping BBHx recompile"
fi

# GBGPU carries the CUDA sig-het/chunked-het kernels, so its build is the
# slowest in the stack. SKIP_GBGPU=1 leaves the existing editable install in
# place (checkout is still ff-pulled so the source is current) and skips only
# the recompile -- rebuild it by hand when its .cu/.cxx actually changed:
#   cd GBGPU && pip install --no-build-isolation -e . \
#       --config-settings=cmake.define.GBT_LAPACKE_DETECT_WITH=PKGCONFIG
clone_or_reuse_sibling "$ORG" GBGPU
if [ "$SKIP_GBGPU" != "1" ]; then
    editable_install "$DEV_ROOT/GBGPU" dev "${LAPACKE_FLAGS[@]}"
else
    echo "===> SKIP_GBGPU=1: checkout pulled, skipping GBGPU recompile"
fi

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
if [ "$SKIP_PHENTAX" != "1" ] && [ "$PULL_ONLY" != "1" ]; then
    echo ""
    echo "===> installing phentax (MBH IMRPhenomTHM)"
    # Equivalent to BBHx's `phentax` extra (pip install 'bbhx[phentax]');
    # installed directly here since BBHx is already editable-installed above.
    # Not a local editable clone, so there is nothing to pull in PULL_ONLY mode.
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
if [ "$PULL_ONLY" = "1" ]; then
    echo "==> pull-only update done (no pip installs run)."
    echo "    Python changes are live immediately: the editable installs import"
    echo "    from the source trees. Re-run the full ./install.sh after C/CUDA,"
    echo "    dependency, or packaging (pyproject) changes."
else
    echo "==> done. Verify with:"
    echo "    python -c 'import lisatools, eryn, bbhx, gbgpu, gpubackendtools; print(\"all import OK\")'"
fi
