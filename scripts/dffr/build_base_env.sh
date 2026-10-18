#!/usr/bin/env bash
# * Frozen environment of the code before a step, for baselines that must come from that code.
# Why it exists, and how it is checked, is in docs/response/dffr/tests.md.
#
# Usage: build_base_env.sh BASE_DIR
#   Clones LAT and GBGPU at LAT_COMMIT and GBGPU_COMMIT, and GPUBackendTools, Eryn and BBHx at
#   the commits checked out in EREBOR, with their uncommitted changes; copies the third-party
#   versions of MAIN_VENV; then builds the stack into BASE_DIR/.venv with install_erebor_spider.sh.
#   BBHx is needed because lisatools.globalfit.recipe, which holds the Mojito constants, imports it.
# Environment: EREBOR (folder of the clones and the install script, default: two levels above this
#   LAT clone), MAIN_VENV (default EREBOR/.venv), LAT_COMMIT, GBGPU_COMMIT (required).
set -euo pipefail

usage="usage: LAT_COMMIT=<sha> GBGPU_COMMIT=<sha> build_base_env.sh BASE_DIR"
base_dir=${1:?$usage}
lat_commit=${LAT_COMMIT:?$usage}
gbgpu_commit=${GBGPU_COMMIT:?$usage}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
erebor=${EREBOR:-$(cd "$script_dir/../../.." && pwd)}
main_venv=${MAIN_VENV:-$erebor/.venv}

mkdir -p "$base_dir/env"
base_dir=$(cd "$base_dir" && pwd)
cd "$base_dir"

# * Clones, not worktrees: install_erebor_spider.sh requires <repo>/.git to be a directory.
clone_at() {
    local repo=$1 commit=$2
    if [ ! -d "$repo/.git" ]; then
        git clone --quiet "$erebor/$repo" "$repo"
        git -C "$repo" checkout --quiet --detach "$commit"
    fi
    [ "$(git -C "$repo" rev-parse HEAD)" = "$(git -C "$erebor/$repo" rev-parse "$commit")" ] ||
        { echo "$repo is not at $commit" >&2; exit 1; }
}
clone_at lisa-analysis-tools "$lat_commit"
clone_at GBGPU "$gbgpu_commit"
for repo in GPUBackendTools Eryn BBHx; do
    if [ ! -d "$repo/.git" ]; then
        clone_at "$repo" "$(git -C "$erebor/$repo" rev-parse HEAD)"
        # ! The main trees build with their uncommitted changes, so the frozen copies get them too.
        git -C "$erebor/$repo" diff HEAD > "$base_dir/env/$repo.uncommitted.diff"
        [ ! -s "$base_dir/env/$repo.uncommitted.diff" ] ||
            git -C "$repo" apply "$base_dir/env/$repo.uncommitted.diff"
    fi
done

# * The main venv's exact third-party versions, minus the components switched off below.
uv pip freeze --python "$main_venv/bin/python" --exclude-editable |
    grep -vE '^(fastemriwaveforms|phentax|mpi4py)\b' > env/third_party_pins.txt
python_version=$("$main_venv/bin/python" -c 'import platform; print(platform.python_version())')
if [ ! -x .venv/bin/python ]; then
    uv venv --quiet --python "$python_version" --managed-python .venv
fi
# ? Installed first and without dependencies; the install script then finds every requirement met
# ? and keeps these versions.
uv pip install --quiet --python .venv/bin/python --no-deps -r env/third_party_pins.txt

GBT_BRANCH="" ERYN_BRANCH="" LAT_BRANCH="" GBGPU_BRANCH="" BBHX_BRANCH="" GIT_PULL=0 EDITABLE=0 \
    WITH_BBHX=1 WITH_FEW=0 WITH_PHENTAX=0 WITH_MPI=0 WITH_JUPYTER=0 \
    "$erebor/install_erebor_spider.sh" "$base_dir"
