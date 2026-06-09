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

set -euo pipefail

ORG="${ORG:-lisa-analysis-tools}"
SKIP_FEW="${SKIP_FEW:-0}"
SKIP_PHENTAX="${SKIP_PHENTAX:-0}"
SKIP_LISA_ON_GPU="${SKIP_LISA_ON_GPU:-1}"

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

LAPACKE_FLAG="--config-settings=cmake.define.GBT_LAPACKE_FETCH=ON"

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

editable_install() {
    local repo_path="$1" branch="$2"
    shift 2
    cd "$repo_path"
    git checkout "$branch"
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
editable_install "$DEV_ROOT/GPUBackendTools" spline "$LAPACKE_FLAG"

clone_or_reuse_sibling "$ORG" Eryn
editable_install "$DEV_ROOT/Eryn" dev "$LAPACKE_FLAG"

echo ""
echo "===> installing LAT (this repo: $LAT_DIR)"
editable_install "$LAT_DIR" dev "$LAPACKE_FLAG"

clone_or_reuse_sibling "$ORG" BBHx
editable_install "$DEV_ROOT/BBHx" dev "$LAPACKE_FLAG"

clone_or_reuse_sibling "$ORG" GBGPU
editable_install "$DEV_ROOT/GBGPU" dev "$LAPACKE_FLAG"

clone_or_reuse_sibling "$ORG" LATW || \
    echo "WARN: LATW clone failed (skipping — tutorials repo is optional)"

# ----------------------------------------------------------------------
# Optional / external
# ----------------------------------------------------------------------
if [ "$SKIP_PHENTAX" != "1" ]; then
    echo ""
    echo "===> installing phentax (MBH IMRPhenomTHM)"
    pip install git+https://github.com/asantini29/phentax.git
fi

if [ "$SKIP_FEW" != "1" ]; then
    clone_or_reuse_sibling BlackHolePerturbationToolkit FastEMRIWaveforms
    editable_install "$DEV_ROOT/FastEMRIWaveforms" gpu_backend \
        --config-settings=cmake.define.FEW_LAPACKE_DETECT_WITH=PKGCONFIG
fi

# lisa-on-gpu is being retired — opt-in only.
if [ "$SKIP_LISA_ON_GPU" != "1" ]; then
    clone_or_reuse_sibling mikekatz04 lisa-on-gpu
    editable_install "$DEV_ROOT/lisa-on-gpu" tdi_on_fly "$LAPACKE_FLAG"
fi

echo ""
echo "==> done. Verify with:"
echo "    python -c 'import lisatools, eryn, bbhx, gbgpu, gpubackendtools; print(\"all import OK\")'"
