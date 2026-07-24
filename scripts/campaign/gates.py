"""The testing-campaign gate DAG — the single source of truth.

Structure (user directive 2026-07-23): **one gate per source class at every
tier where sources are tested separately**, so the dashboard shows a
per-source chain of custody (t1-gt-emri -> t2-lite-emri -> t3-gpu-emri ->
t4-heavy-emri), plus cross-cutting gates (foundation, composition,
multi-GPU, full run).  The minimum granularity of a check is "the fit runs
with this component exercised in-sampler": every command imports a stock fit
(``erebor.<name>()``), builds it, and runs it or interrogates its objects.

Source selection uses the mojito id envs (``EMRI_IDS`` / ``MBHB_IDS`` /
``SOBHB_IDS`` — note the SOBHB astro-class spelling; the branch/debug prefix
is SOBBH): an empty value drops that branch, so per-source gates are real
single-branch ``full_year_combined`` runs.

``ledger.json`` holds only mutable state; this module is diffable code.
Command templates may use ``{py}`` which ``campaign.py`` fills with the
running interpreter, so the same definitions serve laptop and cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

STATES = ("pending", "running", "red", "yellow", "green")

# Baselines from the 2026-07-11 mojito fidelity run (scripts/validation/
# run_mojito_null_checks.sh) with the campaign's 2x regression margin applied.
# Keys follow what the scripts emit: mojito_null_check.py prints the stock
# BRANCH name (mbh/emri/sobbh -> MBH/EMRI/SOBBH); gb/vgb_mojito_match.py
# print class=GB/VGB mismatch lines.
NULL_BASELINE_2X = {
    "MBH": 3.0e-3,    # worst source id19, SNR 1009
    "SOBBH": 3.0e-6,
    "EMRI": 1.2e-3,
    "GB": 1.0e-9,
    "VGB": 1.0e-11,
}


@dataclass(frozen=True)
class Check:
    """One sub-check of a gate: a command plus the criteria evaluated on its
    parsed output.  ``command == ""`` marks a manual/aggregation check that is
    confirmed via ``campaign.py ingest --confirm`` or ``campaign.py set``."""

    id: str
    command: str = ""
    criteria: tuple = ()
    notes: str = ""


@dataclass(frozen=True)
class Gate:
    id: str
    tier: int
    branch: str
    title: str
    objective: str
    where: str  # "laptop" | "cluster"
    depends_on: tuple = ()
    checks: tuple = ()
    proof_plots: tuple = ()  # glob patterns under gf_output/campaign/<id>/
    notes: str = ""


def _ll_finite():
    return {"metric": "ll_finite", "op": "==", "value": 1}


# Shell prefixes selecting exactly one source class in full_year_combined
# (empty id list -> branch dropped).
_ONLY_MBH = "EMRI_IDS= SOBHB_IDS= "
_ONLY_EMRI = "MBHB_IDS= SOBHB_IDS= "
_ONLY_SOBBH = "MBHB_IDS= EMRI_IDS= "

GATES: tuple = (
    # ------------------------------------------------------------------ T0
    Gate(
        id="t0-foundation",
        tier=0,
        branch="all",
        title="Foundation",
        objective="The infrastructure is trustworthy.",
        where="laptop",
        checks=(
            Check(
                id="unit-fast",
                command=(
                    "{py} -m unittest tests.test_stock_globalfit "
                    "tests.test_globalfit_functionmove tests.test_noise_globalfit"
                ),
                criteria=({"metric": "tests_failed", "op": "==", "value": 0},),
            ),
            Check(
                id="build-all-lite",
                command="{py} scripts/campaign/runners/build_all_lite.py",
                criteria=(
                    {"metric": "variants_built", "op": ">=", "value": 7},
                    {"metric": "build_failures", "op": "==", "value": 0},
                ),
                notes="deepcopy+pickle round-trip pre-build on every lite variant",
            ),
            Check(
                id="move-timing",
                command="{py} -m unittest tests.test_move_timing",
                criteria=({"metric": "tests_failed", "op": "==", "value": 0},),
            ),
            Check(
                id="campaign-selftest",
                command="{py} scripts/campaign/runners/selftest.py",
                criteria=({"metric": "selftest_failed", "op": "==", "value": 0},),
            ),
        ),
        proof_plots=("selftest_*.png",),
    ),
    # ------------------------------------------------------------ T1 per class
    Gate(
        id="t1-gt-gb",
        tier=1,
        branch="gb",
        title="GB ground truth",
        objective="Top galactic binaries reproduce the mojito stream (band-passed match).",
        where="laptop",
        depends_on=("t0-foundation",),
        checks=(
            Check(
                id="gb-match",
                command="{py} scripts/gb/gb_mojito_match.py",
                criteria=(
                    {"metric": "gb_mismatch_max", "op": "<=", "value": NULL_BASELINE_2X["GB"]},
                ),
            ),
        ),
        proof_plots=("gb_mojito_match.png",),
    ),
    Gate(
        id="t1-gt-vgb",
        tier=1,
        branch="vgb",
        title="VGB ground truth",
        objective="Verification binaries reproduce the clean mojito VGB stream.",
        where="laptop",
        depends_on=("t0-foundation",),
        checks=(
            Check(
                id="vgb-match",
                command="{py} scripts/gb/vgb_mojito_match.py",
                criteria=(
                    {"metric": "vgb_mismatch_max", "op": "<=", "value": NULL_BASELINE_2X["VGB"]},
                ),
            ),
        ),
        proof_plots=("vgb_mojito_match.png",),
    ),
    Gate(
        id="t1-fastlike-gb",
        tier=1,
        branch="gb",
        title="GB fast likelihood ≡ AnalysisContainer",
        objective=(
            "The fast GB WDM chunked-heterodyne likelihood matches the "
            "AnalysisContainer likelihood — the chain-of-custody link tying the "
            "in-sampler fast path back to the data-level checks."
        ),
        where="laptop",
        depends_on=("t1-gt-gb",),
        checks=(
            Check(
                id="gb-fastlike-vs-ac",
                command=(
                    "{py} scripts/campaign/runners/fastlike_vs_ac.py --branch gb --topn 3"
                ),
                criteria=(
                    {"metric": "fastlike_max_reldiff", "op": "<=", "value": 1e-4},
                    {"metric": "fastlike_sources", "op": ">=", "value": 3},
                ),
                notes="3 highest-frequency mojito GB sources; d_h & h_h fast vs AC",
            ),
        ),
        proof_plots=("fastlike_gb*.png",),
    ),
    Gate(
        id="t1-fastlike-vgb",
        tier=1,
        branch="vgb",
        title="VGB fast likelihood ≡ AnalysisContainer",
        objective=(
            "The fast VGB WDM likelihood matches the AnalysisContainer "
            "likelihood for the verification binaries."
        ),
        where="laptop",
        depends_on=("t1-gt-vgb",),
        checks=(
            Check(
                id="vgb-fastlike-vs-ac",
                command=(
                    "{py} scripts/campaign/runners/fastlike_vs_ac.py --branch vgb --topn 3"
                ),
                criteria=(
                    {"metric": "fastlike_max_reldiff", "op": "<=", "value": 1e-4},
                    {"metric": "fastlike_sources", "op": ">=", "value": 3},
                ),
            ),
        ),
        proof_plots=("fastlike_vgb*.png",),
    ),
    Gate(
        id="t1-gt-mbh",
        tier=1,
        branch="mbh",
        title="MBH ground truth",
        objective="MBHB null template inside the stock fit nulls the mojito data.",
        where="laptop",
        depends_on=("t0-foundation",),
        checks=(
            Check(
                id="null-check",
                command="bash scripts/validation/run_mojito_null_checks.sh MBHB",
                criteria=(
                    {"metric": "null_rr_dd_MBH_max", "op": "<=", "value": NULL_BASELINE_2X["MBH"]},
                ),
                notes="merger-centered chop windows, all catalogue ids",
            ),
            Check(
                id="null-proof",
                command=(
                    "{py} scripts/campaign/runners/null_proof.py "
                    "--branch mbh --gate t1-gt-mbh"
                ),
                criteria=({"metric": "null_proof_ok", "op": "==", "value": 1},),
                notes="rr/dd-per-source bar plot vs the null threshold",
            ),
        ),
        proof_plots=("mbh_null_*.png",),
    ),
    Gate(
        id="t1-gt-emri",
        tier=1,
        branch="emri",
        title="EMRI ground truth",
        objective="EMRI null template inside the stock fit nulls the mojito data.",
        where="laptop",
        depends_on=("t0-foundation",),
        checks=(
            Check(
                id="null-check",
                command="bash scripts/validation/run_mojito_null_checks.sh EMRI",
                criteria=(
                    {"metric": "null_rr_dd_EMRI_max", "op": "<=", "value": NULL_BASELINE_2X["EMRI"]},
                ),
                notes="3-month window (driver default)",
            ),
            Check(
                id="null-proof",
                command=(
                    "{py} scripts/campaign/runners/null_proof.py "
                    "--branch emri --gate t1-gt-emri"
                ),
                criteria=({"metric": "null_proof_ok", "op": "==", "value": 1},),
            ),
        ),
        proof_plots=("emri_null_*.png",),
    ),
    Gate(
        id="t1-gt-sobbh",
        tier=1,
        branch="sobbh",
        title="SOBBH ground truth",
        objective="SOBHB null template inside the stock fit nulls the mojito data.",
        where="laptop",
        depends_on=("t0-foundation",),
        checks=(
            Check(
                id="null-check",
                command="bash scripts/validation/run_mojito_null_checks.sh SOBHB",
                criteria=(
                    {"metric": "null_rr_dd_SOBBH_max", "op": "<=", "value": NULL_BASELINE_2X["SOBBH"]},
                ),
            ),
            Check(
                id="null-proof",
                command=(
                    "{py} scripts/campaign/runners/null_proof.py "
                    "--branch sobbh --gate t1-gt-sobbh"
                ),
                criteria=({"metric": "null_proof_ok", "op": "==", "value": 1},),
            ),
        ),
        proof_plots=("sobbh_null_*.png",),
    ),
    Gate(
        id="t1-gt-noise",
        tier=1,
        branch="noise",
        title="Noise ground truth",
        objective="The 731-day NOISE brick reads correctly through MojitoNoiseEstimates.",
        where="laptop",
        depends_on=("t0-foundation",),
        checks=(
            Check(
                id="mojito-noise",
                command="{py} -m unittest tests.test_mojito_noise",
                criteria=({"metric": "tests_failed", "op": "==", "value": 0},),
            ),
            Check(
                id="noise-proof",
                command="{py} scripts/campaign/runners/noise_proof.py",
                criteria=({"metric": "noise_proof_ok", "op": "==", "value": 1},),
                notes="per-channel brick PSD through the stock MojitoNoiseSensitivityMatrix",
            ),
        ),
        proof_plots=("mojito_noise_psd*.png",),
    ),
    Gate(
        id="t1-alignment",
        tier=1,
        branch="all",
        title="Stock waveform alignment",
        objective="Per-source stock waveform defaults equal the erebor builder defaults.",
        where="laptop",
        depends_on=("t0-foundation",),
        checks=(
            Check(
                id="waveform-align",
                command="LAT_SLOW_TESTS=1 {py} -m unittest tests.test_stock_waveform_alignment",
                criteria=({"metric": "tests_failed", "op": "==", "value": 0},),
            ),
            Check(
                id="siggen-parity",
                command="{py} scripts/validation/gf_signal_gen_vs_mojito.py",
                criteria=({"manual": "all [RESULT] mismatches within 2x baseline"},),
            ),
        ),
    ),
    # ------------------------------------------------------------ T2 per class
    Gate(
        id="t2-infra-blank",
        tier=2,
        branch="all",
        title="blank end-to-end",
        objective="erebor.blank samples end-to-end with HDF persist/resume.",
        where="laptop",
        depends_on=("t0-foundation",),
        checks=(
            Check(
                id="blank-e2e",
                command="RUN_GF_SMOKE=1 {py} -m unittest tests.test_globalfit_sample",
                criteria=({"metric": "tests_failed", "op": "==", "value": 0},),
            ),
        ),
    ),
    Gate(
        id="t2-lite-gb",
        tier=2,
        branch="gb",
        title="GB lite sampling",
        objective="gb_no_fg samples end-to-end on a laptop in PE and search modes.",
        where="laptop",
        depends_on=("t1-gt-gb",),
        checks=(
            Check(
                id="gb-pe",
                command=(
                    "GF_MOVE_TIMING=1 {py} scripts/campaign/runners/branch_lite.py "
                    "--variant gb_no_fg --iterations 10"
                ),
                criteria=(
                    {"metric": "s_per_it", "op": "<=", "value": 5.0},
                    _ll_finite(),
                ),
                notes="baseline 2.7-3.5 s/it",
            ),
            Check(
                id="gb-search",
                command=(
                    "GB_MODE=search GB_DEBUG=1 "
                    "GB_DEBUG_DIR=gf_output/campaign/t2-lite-gb "
                    "{py} scripts/campaign/runners/branch_lite.py "
                    "--variant gb_no_fg --iterations 5"
                ),
                criteria=({"metric": "debug_pngs", "op": ">=", "value": 1},),
                notes="THE search-mode check; zero-leaf start, RJ births",
            ),
        ),
        proof_plots=("gb_debug_*.png",),
    ),
    Gate(
        id="t2-lite-vgb",
        tier=2,
        branch="vgb",
        title="VGB lite sampling",
        objective="vgb_lite samples with fixed catalogue leaves.",
        where="laptop",
        depends_on=("t1-gt-vgb",),
        checks=(
            Check(
                id="vgb-run",
                command=(
                    "GF_MOVE_TIMING=1 {py} scripts/campaign/runners/branch_lite.py "
                    "--variant vgb_lite --iterations 5"
                ),
                criteria=(_ll_finite(),),
            ),
        ),
    ),
    Gate(
        id="t2-lite-noise",
        tier=2,
        branch="noise",
        title="Noise lite sampling",
        objective="noise_sgwb_lite samples PSD+galfor+SGWB.",
        where="laptop",
        depends_on=("t1-gt-noise",),
        checks=(
            Check(
                id="noise-run",
                command=(
                    "GF_MOVE_TIMING=1 {py} scripts/campaign/runners/branch_lite.py "
                    "--variant noise_sgwb_lite --iterations 3"
                ),
                criteria=(_ll_finite(),),
            ),
        ),
    ),
    Gate(
        id="t2-lite-mbh",
        tier=2,
        branch="mbh",
        title="MBH lite sampling",
        objective="MBH-only full_year_combined samples with flip-book proof.",
        where="laptop",
        depends_on=("t1-gt-mbh",),
        checks=(
            Check(
                id="mbh-run",
                command=(
                    _ONLY_MBH + "MBH_DEBUG=1 "
                    "MBH_DEBUG_DIR=gf_output/campaign/t2-lite-mbh "
                    "GF_MOVE_TIMING=1 {py} scripts/campaign/runners/branch_lite.py "
                    "--variant full_year_combined --lite --iterations 3"
                ),
                criteria=(
                    _ll_finite(),
                    {"metric": "debug_pngs", "op": ">=", "value": 1},
                ),
            ),
        ),
        proof_plots=("mbh_debug_*.png",),
    ),
    Gate(
        id="t2-lite-emri",
        tier=2,
        branch="emri",
        title="EMRI lite sampling",
        objective="EMRI-only full_year_combined samples, incl. the domain guard in-sampler.",
        where="laptop",
        depends_on=("t1-gt-emri",),
        checks=(
            Check(
                id="emri-run",
                command=(
                    _ONLY_EMRI + "EMRI_DEBUG=1 "
                    "EMRI_DEBUG_DIR=gf_output/campaign/t2-lite-emri "
                    "GF_MOVE_TIMING=1 {py} scripts/campaign/runners/branch_lite.py "
                    "--variant full_year_combined --lite --iterations 3"
                ),
                criteria=(
                    _ll_finite(),
                    {"metric": "debug_pngs", "op": ">=", "value": 1},
                ),
            ),
            Check(
                id="domain-guard",
                command="{py} scripts/campaign/runners/emri_sparse_guard.py",
                criteria=(
                    {"metric": "guard_ll_floor", "op": "==", "value": 1},
                    {"metric": "process_survived", "op": "==", "value": 1},
                ),
                notes=(
                    "FEW <3-point trajectory guard exercised through the stock "
                    "emri signal_gen inside a built fit -> ll=-1e300, process alive"
                ),
            ),
        ),
        proof_plots=("emri_debug_*.png",),
    ),
    Gate(
        id="t2-lite-sobbh",
        tier=2,
        branch="sobbh",
        title="SOBBH lite sampling",
        objective="SOBBH-only full_year_combined samples with flip-book proof.",
        where="laptop",
        depends_on=("t1-gt-sobbh",),
        checks=(
            Check(
                id="sobbh-run",
                command=(
                    _ONLY_SOBBH + "SOBBH_DEBUG=1 "
                    "SOBBH_DEBUG_DIR=gf_output/campaign/t2-lite-sobbh "
                    "GF_MOVE_TIMING=1 {py} scripts/campaign/runners/branch_lite.py "
                    "--variant full_year_combined --lite --iterations 3"
                ),
                criteria=(
                    _ll_finite(),
                    {"metric": "debug_pngs", "op": ">=", "value": 1},
                ),
            ),
        ),
        proof_plots=("sobbh_debug_*.png",),
    ),
    Gate(
        id="t2-composition",
        tier=2,
        branch="all",
        title="all_sources composes",
        objective="all_sources_lite runs with every branch move timed and diagnostics produced.",
        where="laptop",
        depends_on=(
            "t2-infra-blank",
            "t2-lite-gb",
            "t2-lite-vgb",
            "t2-lite-noise",
            "t2-lite-mbh",
            "t2-lite-emri",
            "t2-lite-sobbh",
        ),
        checks=(
            Check(
                id="all-sources",
                command=(
                    "GF_MOVE_TIMING=1 MAKE_DIAGNOSTIC_PLOTS=1 PLOT_ITERATIONS=2 "
                    "{py} scripts/campaign/runners/branch_lite.py "
                    "--variant all_sources --lite --iterations 3"
                ),
                criteria=(
                    _ll_finite(),
                    {"metric": "timed_moves", "op": ">=", "value": 4},
                ),
                notes="first cross-move efficiency table from [GF_TIMING]",
            ),
        ),
        proof_plots=("timing_*.png",),
    ),
    # ------------------------------------------------------------ T3 per class
    Gate(
        id="t3-gpu-gb",
        tier=3,
        branch="gb",
        title="GB on one GPU",
        objective="GB machinery correct and fast on a single GPU (parity, FD twin, memory model).",
        where="cluster",
        depends_on=("t2-lite-gb",),
        checks=(
            Check(
                id="gpu-parity",
                command=(
                    "USE_GPU=1 GPUS=0 GPU_BACKEND=cuda12x GF_MOVE_TIMING=1 "
                    "python scripts/campaign/runners/branch_lite.py "
                    "--variant gb_no_fg --iterations 10"
                ),
                criteria=(
                    {"manual": "|ll - t2 CPU ll| / |ll| <= 1e-8 at the seeded start"},
                    {"manual": "s/it >= 10x faster than the t2 CPU value"},
                ),
            ),
            Check(
                id="fd-domain-twin",
                command=(
                    "GB_DOMAIN=fd USE_GPU=1 GPUS=0 GF_MOVE_TIMING=1 "
                    "python scripts/campaign/runners/branch_lite.py "
                    "--variant gb_no_fg --iterations 10"
                ),
                criteria=({"manual": "cold-chain ll trajectory consistent with same-seed WDM twin"},),
            ),
            Check(
                id="memory-model",
                command="python scripts/diagnostics/gpu_memory_estimate.py",
                criteria=(
                    {"manual": "predicted GPU pool peak within 25% of measured SubBandBuffer/pool lines"},
                ),
                notes="refresh the script's stale prose while touching it",
            ),
        ),
        proof_plots=("parity_*.png", "timing_*.png"),
    ),
    Gate(
        id="t3-gpu-mbh",
        tier=3,
        branch="mbh",
        title="MBH on one GPU",
        objective="MBH-only full_year_combined runs on one GPU.",
        where="cluster",
        depends_on=("t2-lite-mbh",),
        checks=(
            Check(
                id="mbh-gpu",
                command=(
                    _ONLY_MBH + "USE_GPU=1 GPUS=0 NUM_ITERATIONS=25 GF_MOVE_TIMING=1 "
                    "MAKE_DIAGNOSTIC_PLOTS=0 "
                    "python scripts/run_global.py --stock full_year_combined"
                ),
                criteria=(
                    _ll_finite(),
                    {"manual": "GPU pool below GB_GPU_MEM_WARN_GB; mbh move in [GF_TIMING]"},
                ),
            ),
        ),
        proof_plots=("timing_*.png",),
    ),
    Gate(
        id="t3-gpu-emri",
        tier=3,
        branch="emri",
        title="EMRI on one GPU",
        objective="EMRI-only full_year_combined survives boundary proposals on GPU (cusparse work item).",
        where="cluster",
        depends_on=("t2-lite-emri",),
        checks=(
            Check(
                id="emri-gpu",
                command=(
                    _ONLY_EMRI + "USE_GPU=1 GPUS=0 NUM_ITERATIONS=25 GF_MOVE_TIMING=1 "
                    "MAKE_DIAGNOSTIC_PLOTS=0 "
                    "python scripts/run_global.py --stock full_year_combined"
                ),
                criteria=(
                    _ll_finite(),
                    {"manual": "no interpolate.cu exit(-1) death across the run"},
                ),
                notes=(
                    "work items: FEW rebuilt with the <3-point guard (df30fa9f) + "
                    "ERR_NE prints the cusparse status instead of silent exit(-1)"
                ),
            ),
        ),
        proof_plots=("timing_*.png",),
    ),
    Gate(
        id="t3-gpu-sobbh",
        tier=3,
        branch="sobbh",
        title="SOBBH on one GPU",
        objective="SOBBH-only full_year_combined with TDI-on-the-fly on GPU (cbbhx work item).",
        where="cluster",
        depends_on=("t2-lite-sobbh",),
        checks=(
            Check(
                id="sobbh-gpu",
                command=(
                    _ONLY_SOBBH + "USE_GPU=1 GPUS=0 NUM_ITERATIONS=25 GF_MOVE_TIMING=1 "
                    "MAKE_DIAGNOSTIC_PLOTS=0 "
                    "python scripts/run_global.py --stock full_year_combined"
                ),
                criteria=(
                    _ll_finite(),
                    {"manual": "tdi-on-the-fly path active (not USE_TDIONFLY=0 fallback)"},
                ),
                notes=(
                    "work item: rebuild cbbhx (BBHx cuda12x) — undefined "
                    "__device_builtin_variable_blockDim symbol; yellow allowed via "
                    "USE_TDIONFLY=0 until rebuilt"
                ),
            ),
        ),
        proof_plots=("timing_*.png",),
    ),
    # ------------------------------------------------------------ T4 per class
    Gate(
        id="t4-heavy-gb",
        tier=4,
        branch="gb",
        title="GB heavy",
        objective="gb_no_fg at production settings on one GPU, fully profiled.",
        where="cluster",
        depends_on=("t3-gpu-gb",),
        checks=(
            Check(
                id="gb-200it",
                command=(
                    "USE_GPU=1 GPUS=0 NUM_ITERATIONS=200 GF_MOVE_TIMING=1 "
                    "GB_PROP_TIMING_SYNC=1 "
                    "python scripts/run_global.py --stock gb_no_fg"
                ),
                criteria=(
                    _ll_finite(),
                    {"manual": "GB_TIMING stage split + GPU/host memory recorded; drift-repair rate flat"},
                ),
                notes="narrow-slab measurement deferred per user",
            ),
        ),
        proof_plots=("timing_*.png", "gb_debug_*.png"),
    ),
    Gate(
        id="t4-heavy-mbh",
        tier=4,
        branch="mbh",
        title="MBH heavy",
        objective="MBH-only production run; START-AWARE residual criterion.",
        where="cluster",
        depends_on=("t3-gpu-mbh",),
        checks=(
            Check(
                id="mbh-100it",
                command=(
                    _ONLY_MBH + "USE_GPU=1 GPUS=0 NUM_ITERATIONS=100 GF_MOVE_TIMING=1 "
                    "MBH_DEBUG=1 MBH_DEBUG_EVERY=20 "
                    "python scripts/run_global.py --stock full_year_combined"
                ),
                criteria=(
                    _ll_finite(),
                    {
                        "manual": (
                            "START-AWARE: truth-near starts -> residuals stationary at the "
                            "truth-null floor (rr/dd ~ T1, no growth); scattered starts -> "
                            "residuals approach the floor"
                        )
                    },
                ),
            ),
        ),
        proof_plots=("mbh_debug_*.png", "timing_*.png"),
    ),
    Gate(
        id="t4-heavy-emri",
        tier=4,
        branch="emri",
        title="EMRI heavy",
        objective="EMRI-only production run; START-AWARE residual criterion.",
        where="cluster",
        depends_on=("t3-gpu-emri",),
        checks=(
            Check(
                id="emri-100it",
                command=(
                    _ONLY_EMRI + "USE_GPU=1 GPUS=0 NUM_ITERATIONS=100 GF_MOVE_TIMING=1 "
                    "EMRI_DEBUG=1 EMRI_DEBUG_EVERY=20 "
                    "python scripts/run_global.py --stock full_year_combined"
                ),
                criteria=(
                    _ll_finite(),
                    {"manual": "START-AWARE residual criterion (as t4-heavy-mbh)"},
                ),
            ),
        ),
        proof_plots=("emri_debug_*.png", "timing_*.png"),
    ),
    Gate(
        id="t4-heavy-sobbh",
        tier=4,
        branch="sobbh",
        title="SOBBH heavy",
        objective="SOBBH-only production run; START-AWARE residual criterion.",
        where="cluster",
        depends_on=("t3-gpu-sobbh",),
        checks=(
            Check(
                id="sobbh-100it",
                command=(
                    _ONLY_SOBBH + "USE_GPU=1 GPUS=0 NUM_ITERATIONS=100 GF_MOVE_TIMING=1 "
                    "SOBBH_DEBUG=1 SOBBH_DEBUG_EVERY=20 "
                    "python scripts/run_global.py --stock full_year_combined"
                ),
                criteria=(
                    _ll_finite(),
                    {"manual": "START-AWARE residual criterion (as t4-heavy-mbh)"},
                ),
            ),
        ),
        proof_plots=("sobbh_debug_*.png", "timing_*.png"),
    ),
    Gate(
        id="t4-heavy-noise",
        tier=4,
        branch="noise",
        title="Noise heavy + efficiency report",
        objective="noise_sgwb at production settings; cross-branch efficiency table rendered.",
        where="cluster",
        depends_on=("t2-lite-noise",),
        checks=(
            Check(
                id="noise-full",
                command="USE_GPU=1 GPUS=0 python scripts/run_global.py --stock noise_sgwb",
                criteria=(_ll_finite(),),
            ),
            Check(
                id="efficiency-report",
                command="",
                criteria=(
                    {"manual": "cross-branch s/move + memory table rendered into dashboard from T4 [GF_TIMING]"},
                ),
            ),
        ),
        proof_plots=("timing_*.png",),
    ),
    # ------------------------------------------------------------------ T5
    Gate(
        id="t5-mg-correctness",
        tier=5,
        branch="gb",
        title="Multi-GPU correctness",
        objective="Two GPUs produce the same physics as one (P1 gates 1-3).",
        where="cluster",
        depends_on=("t4-heavy-gb",),
        checks=(
            Check(
                id="mg1-baseline",
                command=(
                    "GPUS=0 NUM_ITERATIONS=50 NWALKERS=8 NTEMPS=4 DATA_MODE=synthetic "
                    "MAKE_DIAGNOSTIC_PLOTS=0 python scripts/run_global.py --stock gb_no_fg"
                ),
                criteria=({"manual": "same seed -> chain statistically identical to pre-P1 dev"},),
            ),
            Check(
                id="mg2-identity",
                command=(
                    "GPUS=0,1 NUM_ITERATIONS=50 NWALKERS=8 NTEMPS=4 DATA_MODE=synthetic "
                    "MAKE_DIAGNOSTIC_PLOTS=0 python scripts/run_global.py --stock gb_no_fg"
                ),
                criteria=(
                    {"manual": "acs.likelihood() matches 1-GPU to ~1e-12; ~100% same-shard swaps"},
                ),
            ),
            Check(
                id="mg3-sampling",
                command="",
                criteria=(
                    {"manual": "~500 it 1-vs-2 GPU: tempering acceptance, corners, drift-repair rate indistinguishable"},
                ),
                notes="commands per docs/multigpu-cluster-validation.md gate 3",
            ),
        ),
        proof_plots=("corner_overlay_*.png", "acceptance_*.png"),
        notes="reconcile with the 2026-07-23 multi-GPU hardening (05645d2) at batch time",
    ),
    Gate(
        id="t5-mg-scaling",
        tier=5,
        branch="gb",
        title="Multi-GPU scaling",
        objective="Two GPUs are worth it (P1 gate 4).",
        where="cluster",
        depends_on=("t5-mg-correctness",),
        checks=(
            Check(
                id="mg4-scaling",
                command="",
                criteria=(
                    {"manual": "wall-clock split (proposal/tempering/fill) 1 vs 2 GPUs recorded"},
                    {"manual": "GB_MEMPOOL_FREE_EACH_ROUND=0 vs 1 compared, no OOM at 0"},
                ),
                notes="commands per docs/multigpu-cluster-validation.md gate 4 (wide band)",
            ),
        ),
        proof_plots=("scaling_*.png",),
    ),
    # ------------------------------------------------------------------ T6
    Gate(
        id="t6-allsources-2yr",
        tier=6,
        branch="all",
        title="all_sources, 2 years, multi-GPU",
        objective="The end target runs sustained and survives resume.",
        where="cluster",
        depends_on=(
            "t2-composition",
            "t4-heavy-mbh",
            "t4-heavy-emri",
            "t4-heavy-sobbh",
            "t4-heavy-noise",
            "t5-mg-scaling",
        ),
        checks=(
            Check(
                id="full-run",
                command=(
                    "TOBS_TARGET=63115200 USE_GPU=1 GPUS=0,1 GF_MOVE_TIMING=1 "
                    "mpiexec -n 3 python scripts/run_global.py --stock all_sources"
                ),
                criteria=(
                    _ll_finite(),
                    {"manual": "sustained multi-hundred iterations; async saver keeps up"},
                ),
            ),
            Check(
                id="resume",
                command="",
                criteria=({"manual": "kill after checkpoint, relaunch: iterations continue, ll continuous"},),
            ),
        ),
        proof_plots=("timing_*.png", "s_per_it_timeline_*.png"),
    ),
    Gate(
        id="t6-science-close",
        tier=6,
        branch="all",
        title="Science sanity + close",
        objective="The full run's cold chain is scientifically sane; campaign closes.",
        where="cluster",
        depends_on=("t6-allsources-2yr",),
        checks=(
            Check(
                id="science-sanity",
                command="python scripts/campaign/runners/science_sanity.py",
                criteria=(
                    {"manual": "cold-chain residual rr/dd within few x T1 baselines via fit.acs"},
                    {"manual": "VGB recovered params consistent with catalogue; leaf counts plausible"},
                ),
            ),
            Check(
                id="campaign-close",
                command="",
                criteria=({"manual": "final cleanup sweep done; dashboard republished; all gates green"},),
            ),
        ),
        proof_plots=("residual_panel_*.png", "vgb_recovery_*.png"),
    ),
)

GATES_BY_ID = {g.id: g for g in GATES}

# Cluster batches: batch N -> tier whose cluster gates it carries.
BATCH_TIERS = {1: 3, 2: 4, 3: 5, 4: 6}


def gates_for_batch(n: int):
    tier = BATCH_TIERS[n]
    return [g for g in GATES if g.tier == tier and g.where == "cluster"]
