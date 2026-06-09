#!/usr/bin/env bash
# check_single_registrant.sh
#
# Phase 3K (2026-06-02). The third leg of the L2 single-registrant defense:
# a grep gate that refuses any commit / PR / CI run that introduces a
# `py::class_<X>(...)` registration for a LAT-owned shared wrapper class
# outside the LISAanalysistools repo. Pairs with:
#   - The `LISATOOLS_IS_WRAPPER_OWNER` toggle + static_assert in each
#     downstream binding TU (catches accidental re-registration at compile
#     time IF the consumer toggles the macro).
#   - The single-registrant convention itself, documented in
#     LISAanalysistools/src/lisatools/cutils/lisatools_header_abi.hpp.
#
# This script catches the case where a contributor adds `py::class_<...>` to
# the consumer file WITHOUT first flipping the toggle. The static_assert
# wouldn't catch that (the macro stays 0), but this script would.
#
# Usage:
#   tools/check_single_registrant.sh
#
# Exit code:
#   0 -- no forbidden registrations found
#   1 -- one or more forbidden registrations found (printed to stderr)
#
# Intended invocation from CI (GitHub Actions step):
#   - name: Single-registrant rule check
#     run: tools/check_single_registrant.sh
#
# To extend: add wrapper class names to FORBIDDEN_CLASSES below as Phase 3J/4
# carve-out moves more shared classes into LAT.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Repos that consume LAT's shared wrappers but must NOT register them.
CONSUMER_REPOS=(
    "lisa-on-gpu"
    "GBGPU"
    "BBHx"
    "FastEMRIWaveforms"
)

# Classes owned by LISAanalysistools (registered in pycppdetector). A
# `py::class_<X>(...)` for any of these in a CONSUMER_REPOS path is a bug.
FORBIDDEN_CLASSES=(
    "OrbitsWrap"
    "LISAResponseWrap"
    "TDIConfigWrap"
    "CubicSplineWrap_responselisa"
    "OrbitsWrap_responselisa"
    # Phase 3L (2026-06-02): TDIonTheFly carve-out begins. Classes moved
    # so far live in LAT under lisatools/cutils/.
    "FDDomain"
    "FDDomainWrap"
    "WDMSettings"
    "WDMSettingsWrap"
    "WDMDomain"
    "WDMDomainWrap"
    "FDSplineTDIWaveform"
    "FDSplineTDIWaveformWrap"
    "TDSplineTDIWaveform"
    "TDSplineTDIWaveformWrap"
)

violations=0
violation_log=$(mktemp)
trap 'rm -f "$violation_log"' EXIT

for repo in "${CONSUMER_REPOS[@]}"; do
    repo_path="$ROOT/$repo"
    [ -d "$repo_path" ] || continue

    for cls in "${FORBIDDEN_CLASSES[@]}"; do
        # Look for `py::class_<<cls>>(`. We use \\b to anchor on a word boundary
        # so `OrbitsWrap_responselisa` doesn't match `OrbitsWrap`.
        # --include= patterns restrict to pybind11 binding source.
        matches=$(grep -rn \
            --include='*.cxx' --include='*.cpp' --include='*.cu' --include='*.cc' \
            --include='*.hpp' --include='*.hh' --include='*.h' \
            -E "py::class_<\s*${cls}\b" \
            "$repo_path" 2>/dev/null || true)
        if [ -n "$matches" ]; then
            echo "" >> "$violation_log"
            echo "VIOLATION: \`${cls}\` registered outside LISAanalysistools" >> "$violation_log"
            echo "$matches" >> "$violation_log"
            violations=$((violations + 1))
        fi
    done
done

if [ "$violations" -gt 0 ]; then
    {
        echo "==============================================================="
        echo "Single-registrant rule violation(s) found."
        cat "$violation_log"
        echo ""
        echo "==============================================================="
        echo "These classes are owned by LISAanalysistools and registered in"
        echo "its pycppdetector pybind11 module (via response_part(m) or"
        echo "detector_part(m) in binding.cxx). They must NOT be registered"
        echo "by any other repo. See:"
        echo "  LISAanalysistools/src/lisatools/cutils/lisatools_header_abi.hpp"
        echo "  plan section 'OrbitsWrap (and friends) symbol unification'"
        echo "  memory project_phase3j_enforcement"
        echo "==============================================================="
    } >&2
    exit 1
fi

echo "Single-registrant rule check passed."
exit 0
