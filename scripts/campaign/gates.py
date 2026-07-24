"""The testing-campaign gate DAG — the single source of truth.

Gates are LARGE objectives (one to three per tier).  The fine-grained items live
inside each gate as ``Check`` entries, never as separate gates.  The minimum
granularity of a check is "the fit runs with this component exercised
in-sampler": every command below imports a stock fit (``erebor.<name>()``),
builds it, and runs it or interrogates its objects.  No standalone harnesses.

``ledger.json`` holds only mutable state (state/metrics/evidence/history); this
module is diffable code and is what a review of the campaign reviews.

Command templates may use ``{py}`` which ``campaign.py`` fills with the running
interpreter, so the same definitions serve the laptop and the cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STATES = ("pending", "red", "yellow", "green")

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


GATES: tuple[Gate, ...] = (
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
                notes="GF_MOVE_TIMING instrumentation landed; unit-fast re-run after",
            ),
            Check(
                id="campaign-selftest",
                command="{py} scripts/campaign/runners/selftest.py",
                criteria=({"metric": "selftest_failed", "op": "==", "value": 0},),
                notes="fixture log -> parse -> ledger -> dashboard incl. embedded PNG",
            ),
        ),
        proof_plots=("selftest_*.png",),
    ),
    # ------------------------------------------------------------------ T1
    Gate(
        id="t1-mojito-ground-truth",
        tier=1,
        branch="all",
        title="Mojito ground truth",
        objective="Every branch's physics matches the mojito data through the stock fits.",
        where="laptop",
        depends_on=("t0-foundation",),
        checks=(
            Check(
                id="null-checks",
                command="bash scripts/validation/run_mojito_null_checks.sh",
                criteria=(
                    {"metric": "null_rr_dd_MBH_max", "op": "<=", "value": NULL_BASELINE_2X["MBH"]},
                    {"metric": "null_rr_dd_SOBBH_max", "op": "<=", "value": NULL_BASELINE_2X["SOBBH"]},
                    {"metric": "null_rr_dd_EMRI_max", "op": "<=", "value": NULL_BASELINE_2X["EMRI"]},
                ),
                notes="null template inside full_year_combined via fit.acs",
            ),
            Check(
                id="gb-vgb-mismatch",
                command=(
                    "{py} scripts/gb/gb_mojito_match.py && "
                    "{py} scripts/gb/vgb_mojito_match.py"
                ),
                criteria=(
                    {"metric": "gb_mismatch_max", "op": "<=", "value": NULL_BASELINE_2X["GB"]},
                    {"metric": "vgb_mismatch_max", "op": "<=", "value": NULL_BASELINE_2X["VGB"]},
                ),
            ),
            Check(
                id="siggen-parity",
                command="{py} scripts/validation/gf_signal_gen_vs_mojito.py",
                criteria=({"manual": "all [RESULT] mismatches within 2x baseline"},),
            ),
            Check(
                id="waveform-align",
                command="LAT_SLOW_TESTS=1 {py} -m unittest tests.test_stock_waveform_alignment",
                criteria=({"metric": "tests_failed", "op": "==", "value": 0},),
            ),
            Check(
                id="mojito-noise",
                command="{py} -m unittest tests.test_mojito_noise",
                criteria=({"metric": "tests_failed", "op": "==", "value": 0},),
                notes="731-day NOISE brick confirmed usable (2-yr capability)",
            ),
        ),
        proof_plots=("null_*.png", "mismatch_*.png"),
    ),
    # ------------------------------------------------------------------ T2
    Gate(
        id="t2-gbfamily-lite",
        tier=2,
        branch="gb",
        title="GB family lite",
        objective="GB family samples end-to-end on a laptop (PE, search, VGB, blank, noise).",
        where="laptop",
        depends_on=("t1-mojito-ground-truth",),
        checks=(
            Check(
                id="gb-pe-lite",
                command=(
                    "GF_MOVE_TIMING=1 {py} scripts/campaign/runners/branch_lite.py "
                    "--variant gb_no_fg --iterations 10"
                ),
                criteria=(
                    {"metric": "s_per_it", "op": "<=", "value": 5.0},
                    {"metric": "ll_finite", "op": "==", "value": 1},
                ),
                notes="baseline 2.7-3.5 s/it",
            ),
            Check(
                id="gb-search-lite",
                command=(
                    "GB_MODE=search GB_DEBUG=1 "
                    "GB_DEBUG_DIR=gf_output/campaign/t2-gbfamily-lite "
                    "{py} scripts/campaign/runners/branch_lite.py "
                    "--variant gb_no_fg --iterations 5"
                ),
                criteria=({"metric": "debug_pngs", "op": ">=", "value": 1},),
                notes="THE search-mode check; RJ adds leaves from zero-leaf start",
            ),
            Check(
                id="vgb-lite",
                command=(
                    "{py} scripts/campaign/runners/branch_lite.py "
                    "--variant vgb --lite --iterations 5"
                ),
                criteria=({"metric": "ll_finite", "op": "==", "value": 1},),
            ),
            Check(
                id="blank-e2e",
                command="RUN_GF_SMOKE=1 {py} -m unittest tests.test_globalfit_sample",
                criteria=({"metric": "tests_failed", "op": "==", "value": 0},),
                notes="includes HDF persist/resume",
            ),
            Check(
                id="noise-lite",
                command=(
                    "{py} scripts/campaign/runners/branch_lite.py "
                    "--variant noise_sgwb_lite --iterations 3"
                ),
                criteria=({"metric": "ll_finite", "op": "==", "value": 1},),
            ),
        ),
        proof_plots=("gb_debug_*.png", "timing_*.png"),
    ),
    Gate(
        id="t2-sources-lite",
        tier=2,
        branch="mbh/emri/sobbh",
        title="Source branches lite",
        objective="MBH/EMRI/SOBBH sample end-to-end with visual proof, incl. the EMRI domain guard in-sampler.",
        where="laptop",
        depends_on=("t1-mojito-ground-truth",),
        checks=(
            Check(
                id="fyc-lite-debug",
                command=(
                    "MBH_DEBUG=1 EMRI_DEBUG=1 SOBBH_DEBUG=1 "
                    "MBH_DEBUG_DIR=gf_output/campaign/t2-sources-lite "
                    "EMRI_DEBUG_DIR=gf_output/campaign/t2-sources-lite "
                    "SOBBH_DEBUG_DIR=gf_output/campaign/t2-sources-lite "
                    "{py} scripts/campaign/runners/branch_lite.py "
                    "--variant full_year_combined --lite --iterations 3"
                ),
                criteria=(
                    {"metric": "ll_finite", "op": "==", "value": 1},
                    {"metric": "debug_pngs", "op": ">=", "value": 3},
                ),
                notes="template|data|residual flip-books for all three branches",
            ),
            Check(
                id="emri-domain-guard",
                command="{py} scripts/campaign/runners/emri_sparse_guard.py",
                criteria=(
                    {"metric": "guard_ll_floor", "op": "==", "value": 1},
                    {"metric": "process_survived", "op": "==", "value": 1},
                ),
                notes=(
                    "FEW <3-point sparse-trajectory guard, exercised through the "
                    "stock emri signal_gen inside a built fit: boundary proposal "
                    "-> ll=-1e300, process alive"
                ),
            ),
        ),
        proof_plots=("*_debug_leaf*.png",),
    ),
    Gate(
        id="t2-composition-lite",
        tier=2,
        branch="all",
        title="all_sources composes",
        objective="all_sources_lite runs with every branch move timed and diagnostics produced.",
        where="laptop",
        depends_on=("t2-gbfamily-lite", "t2-sources-lite"),
        checks=(
            Check(
                id="all-sources-lite",
                command=(
                    "GF_MOVE_TIMING=1 MAKE_DIAGNOSTIC_PLOTS=1 PLOT_ITERATIONS=2 "
                    "{py} scripts/campaign/runners/branch_lite.py "
                    "--variant all_sources --lite --iterations 3"
                ),
                criteria=(
                    {"metric": "ll_finite", "op": "==", "value": 1},
                    {"metric": "timed_moves", "op": ">=", "value": 4},
                ),
                notes="first cross-move efficiency table from [GF_TIMING]",
            ),
        ),
        proof_plots=("timing_*.png", "diagnostic*/*.png"),
    ),
    # ------------------------------------------------------------------ T3
    Gate(
        id="t3-gb-gpu",
        tier=3,
        branch="gb",
        title="GB on one GPU",
        objective="GB machinery correct and fast on a single GPU (parity, FD twin, memory model).",
        where="cluster",
        depends_on=("t2-gbfamily-lite",),
        checks=(
            Check(
                id="gpu-parity",
                command=(
                    "USE_GPU=1 GPUS=0 GPU_BACKEND=cuda12x GF_MOVE_TIMING=1 "
                    "python scripts/campaign/runners/branch_lite.py "
                    "--variant gb_no_fg --iterations 10"
                ),
                criteria=(
                    {"manual": "|ll_start_gpu - ll_start_cpu|/|ll| <= 1e-8 vs t2 stored value"},
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
                notes="THE FD-domain check",
            ),
            Check(
                id="memory-model",
                command="python scripts/diagnostics/gpu_memory_estimate.py",
                criteria=(
                    {"manual": "predicted GPU pool peak within 25% of measured SubBandBuffer/pool lines"},
                ),
                notes="refresh the script's stale prose (n_subbands default, task-b status) while touching it",
            ),
        ),
        proof_plots=("parity_*.png", "memmodel_*.png"),
    ),
    Gate(
        id="t3-sources-gpu",
        tier=3,
        branch="mbh/emri/sobbh",
        title="Sources on one GPU",
        objective="full_year_combined runs on one GPU with all branches exercised in-sampler.",
        where="cluster",
        depends_on=("t2-sources-lite",),
        checks=(
            Check(
                id="fyc-gpu-smoke",
                command=(
                    "USE_GPU=1 GPUS=0 NUM_ITERATIONS=25 GF_MOVE_TIMING=1 "
                    "MAKE_DIAGNOSTIC_PLOTS=0 "
                    "python scripts/run_global.py --stock full_year_combined"
                ),
                criteria=(
                    {"metric": "ll_finite", "op": "==", "value": 1},
                    {"manual": "GPU pool stays below GB_GPU_MEM_WARN_GB; all branches in [GF_TIMING]"},
                ),
                notes=(
                    "work items INSIDE this gate, proven by the fit run itself: "
                    "(a) FEW interpolate.cu ERR_NE prints cusparse status and the "
                    "EMRI update loop survives boundary proposals (no exit(-1) "
                    "death); (b) cbbhx cuda12x rebuilt so SOBBH tdi-on-the-fly "
                    "generates in-run (yellow allowed via USE_TDIONFLY=0 until then)"
                ),
            ),
        ),
        proof_plots=("*_debug_leaf*.png", "timing_*.png"),
    ),
    # ------------------------------------------------------------------ T4
    Gate(
        id="t4-gb-heavy",
        tier=4,
        branch="gb",
        title="GB heavy",
        objective="gb_no_fg at production settings on one GPU, fully profiled.",
        where="cluster",
        depends_on=("t3-gb-gpu",),
        checks=(
            Check(
                id="gb-200it",
                command=(
                    "USE_GPU=1 GPUS=0 NUM_ITERATIONS=200 GF_MOVE_TIMING=1 "
                    "GB_PROP_TIMING_SYNC=1 "
                    "python scripts/run_global.py --stock gb_no_fg"
                ),
                criteria=(
                    {"metric": "ll_finite", "op": "==", "value": 1},
                    {"manual": "GB_TIMING stage split + GPU/host memory recorded; drift-repair rate flat"},
                ),
                notes="narrow-slab measurement DEFERRED per user (not part of the campaign)",
            ),
        ),
        proof_plots=("timing_*.png", "gb_debug_*.png"),
    ),
    Gate(
        id="t4-sources-noise-heavy",
        tier=4,
        branch="mbh/emri/sobbh/noise",
        title="Sources + noise heavy",
        objective="Source branches and noise at production settings; cross-branch efficiency report.",
        where="cluster",
        depends_on=("t3-sources-gpu",),
        checks=(
            Check(
                id="fyc-100it",
                command=(
                    "USE_GPU=1 GPUS=0 NUM_ITERATIONS=100 GF_MOVE_TIMING=1 "
                    "MBH_DEBUG=1 MBH_DEBUG_EVERY=20 EMRI_DEBUG=1 EMRI_DEBUG_EVERY=20 "
                    "SOBBH_DEBUG=1 SOBBH_DEBUG_EVERY=20 "
                    "python scripts/run_global.py --stock full_year_combined"
                ),
                criteria=(
                    {"metric": "ll_finite", "op": "==", "value": 1},
                    {
                        "manual": (
                            "START-AWARE residual criterion: truth-near starts -> "
                            "cold-chain residuals stationary at the truth-null floor "
                            "(rr/dd consistent with T1, no systematic growth); "
                            "START_FACTOR-scattered starts -> residuals approach the floor"
                        )
                    },
                ),
            ),
            Check(
                id="noise-heavy",
                command="USE_GPU=1 GPUS=0 python scripts/run_global.py --stock noise_sgwb",
                criteria=({"metric": "ll_finite", "op": "==", "value": 1},),
            ),
            Check(
                id="efficiency-report",
                command="",
                criteria=({"manual": "cross-branch s/move + memory table rendered into dashboard"},),
            ),
        ),
        proof_plots=("*_debug_leaf*.png", "timing_*.png"),
    ),
    # ------------------------------------------------------------------ T5
    Gate(
        id="t5-mg-correctness",
        tier=5,
        branch="gb",
        title="Multi-GPU correctness",
        objective="Two GPUs produce the same physics as one (P1 gates 1-3).",
        where="cluster",
        depends_on=("t4-gb-heavy",),
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
        depends_on=("t4-sources-noise-heavy", "t5-mg-scaling"),
        checks=(
            Check(
                id="full-run",
                command=(
                    "TOBS_TARGET=63115200 USE_GPU=1 GPUS=0,1 GF_MOVE_TIMING=1 "
                    "mpiexec -n 3 python scripts/run_global.py --stock all_sources"
                ),
                criteria=(
                    {"metric": "ll_finite", "op": "==", "value": 1},
                    {"manual": "sustained multi-hundred iterations; async saver keeps up"},
                ),
            ),
            Check(
                id="resume",
                command="",
                criteria=({"manual": "kill after checkpoint, relaunch: iteration count continues, ll continuous"},),
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
