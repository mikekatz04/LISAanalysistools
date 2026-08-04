"""THE GB likelihood-engine benchmark: accuracy and speed, one run, one output.

Single entry point.  Measures, in the same sweep and against the same
reference, for chunked-het / sig-het v2 / v3 / v4-PCR / v4-banded:

  * per-candidate cost across an observation-time ladder and a batch sweep
  * tiered likelihood error at every observation time
  * the settings frontier (fit nodes x knots x band) at every observation time

and emits ONE figure, ONE npz, and printed tables including the operational
answer: the cheapest configuration that meets the accuracy bar at each Tobs.

ACCURACY REFERENCE = chunked-het, itself validated against full time-domain
truth (plain h+/hx at every dt through the direct response) at 3e-9.  Errors
are DELTA-vs-DELTA from the reference point -- a constant offset at the
reference cancels in every acceptance ratio -- and are judged against the
tiered spec

    allowed(T) = max(0.1, T/100)      for a candidate whose true |dlnL| is T

with T > ~1000 gated by the trust region rather than required to be accurate.

Run (GPU):
    USE_GPU=1 GPU_BACKEND=cuda12x python gb_engine_benchmark.py
Run (cluster CPU):
    BENCH_NT_LIST="540,1080,2160" BENCH_BATCHES="1,8,64" BENCH_NF=1440 \\
        python gb_engine_benchmark.py
Fast smoke (laptop):
    BENCH_NT_LIST=256 BENCH_BATCHES=4 BENCH_NREF=1 BENCH_TIERS=1,100 \\
        BENCH_SETTINGS=0 python gb_engine_benchmark.py

Env: BENCH_NT_LIST ("540,1080,2160,4320,8640" GPU = 3 mo / 6 mo / 1 / 2 /
     4 yr at Nf=1440, dt=10; "1024" CPU) -- wavelet time-pixel counts; BENCH_BATCHES
     ("16,64,256,1024,4096,16384" GPU / "8" CPU); BENCH_NF (1440 GPU /
     256 CPU); BENCH_NREF (4) reference sources; BENCH_TIERS
     ("1,10,50,100,1000"); BENCH_NR (64) fit nodes at 1 yr; BENCH_K (128)
     knots; BENCH_BAND (16) half-band; BENCH_REPS (5); BENCH_SETTINGS (1)
     run the settings frontier; BENCH_T_REP (100) tier the frontier reports;
     ENV_OUT (./ratio_proto_out)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import WDMBandLikelihoodEngine

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
BACKEND = os.environ.get("GPU_BACKEND", "cuda12x") if USE_GPU else "cpu"

ALL_ENGINES = ["chunked", "v2", "v3", "v4-pcr", "v4-band", "v5", "v5-ctl"]
# "v5"     = v4-banded with the per-candidate fold scratch ELIMINATED
#            (r_sparse/dr_sparse are r_pix times a candidate-independent mask
#            bit, so v5 keeps the bit and rebuilds r/dr in registers) AND the
#            shared arena overlaid by lifetime phase. 27.6 KB, constant in
#            N_sparse_t up to N ~ 450 -> ~5 blocks/SM on an A100 against
#            v4-banded's 1. "v5-ctl" = the same kernel with a FLAT carve:
#            identical arithmetic and traffic at ~3 blocks/SM, so v5-vs-v5-ctl
#            isolates OCCUPANCY and v5-ctl-vs-v4-band isolates everything
#            else. Neither is in the default selection below by design: they
#            are opt-in via BENCH_ENGINES so an unrelated run's grid is never
#            resized by them.
# BENCH_ENGINES selects the engines to build and score. "chunked" is the
# accuracy reference and is always kept. This is not only a time-saver:
# every sig-het engine is built with ONE nt_layer, and v2 costs 960 B per
# sparse-time point against v4-banded's 528, so leaving v2 in PINS the
# sparse grid for everyone at what v2 can afford (this is the whole reason
# N_sparse_t sat at 127 in the 5-baseline sweep). Dropping v2 lets
# fit_nt_layer size the grid to the engines actually under test.
#
# The default is now the chunked/v4/v5 comparison -- the live question. v2,
# v3 and v4-pcr are opt-in (BENCH_ENGINES="chunked,v2,v3,v4-pcr,v4-band")
# for the historical 5-engine sweep.
DEFAULT_ENGINES = ["chunked", "v4-band", "v5", "v5-ctl"]
ENGINES = [n for n in ALL_ENGINES
           if n == "chunked"
           or n in {s.strip() for s in
                    os.environ.get("BENCH_ENGINES",
                                   ",".join(DEFAULT_ENGINES)).split(",")}]
# Categorical hues in FIXED order, CVD-validated (adjacent-pair dE >= 8 under
# deuteranopia and protanopia).  Colour follows the ENGINE, never its rank;
CLR = {"chunked": "#666666", "v2": "#E69F00", "v3": "#0072B2",
       "v4-pcr": "#009E73", "v4-band": "#D55E00", "v5": "#CC79A7",
       "v5-ctl": "#8C6BB1"}
MRK = {"chunked": "o", "v2": "s", "v3": "^", "v4-pcr": "D",
       "v4-band": "v", "v5": "X", "v5-ctl": "P"}
LS = {n: "-" for n in ENGINES}
RAMP = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c",
        "#08306b"]
K_MRK = {64: "o", 128: "s", 256: "^", 512: "P"}
INK, INK2, GRID, OKC, BADC = "#1a1a1a", "#555555", "#d8d8d8", "#2f7d4f", "#b3261e"


# ===================== shared-memory accounting ==========================
# Exact mirrors of the C++ budgets; the harness sizes the sparse-time grid to
# the device instead of guessing (N_PARAMS_MAX = 20 in gb_tdi_on_the_fly.cu).

def v2_shared_bytes(nch, m_half, Nt_layer, N_sparse_t, n_sparse_fd):
    M = 2 * m_half + 1
    return (20 * 8 + n_sparse_fd * 8 + nch * n_sparse_fd * 16
            + (nch * M * Nt_layer + 3 * nch * M * N_sparse_t) * 16)


def sighet_shared_bytes(n_nodes, n_knots, nch, m_half, N_sparse_t, band_len,
                        v4=True):
    M = 2 * m_half + 1
    b = 2 * 20 * 8 + n_nodes * 8 + 2 * nch * n_nodes * 16
    b += 6 * n_nodes * 8 + n_nodes * 4 + n_nodes * 1
    b += 8 * nch * n_nodes * 8
    n_fit = max(n_knots, n_nodes) if v4 else n_nodes
    b += 9 * n_fit * 8
    if v4:
        b += n_knots * 8 + 2 * nch * n_knots * 8
        if band_len <= 0:
            b += 6 * nch * n_knots * 8
        b += nch * N_sparse_t * 16
    return b + 2 * nch * M * N_sparse_t * 16 + 64


def sighet_v5_shared_bytes(n_nodes, n_knots, nch, m_half, N_sparse_t,
                           band_len, alias=True):
    """Per-block dynamic shared request of the V5 scorer.

    Exact mirror of ``gb_sighet_v5_region_sizes`` + ``gb_sighet_v5_shared_bytes``
    in gb_tdi_on_the_fly.cu. Region letters match the table in that file's V5
    header; ``_al`` is its GB_SIGHET_V5_AL 16-byte round-up.

    With ``alias=True`` the regions whose lifetimes are disjoint are overlaid
    (F over E, G over B+C+Bk+D), so the request is the widest PHASE rather
    than the sum of the arrays -- and it is CONSTANT in ``N_sparse_t`` until
    the fold phase overtakes the node phase around N ~ 450. ``alias=False`` is
    the flat carve, i.e. the occupancy control.
    """
    def _al(x):
        return (int(x) + 15) & ~15

    M = 2 * m_half + 1
    nwords = (N_sparse_t + 63) // 64
    n_fit = n_nodes if band_len > 0 else max(n_knots, n_nodes)

    szA = _al(2 * 20 * 8)                            # params_c/r (N_PARAMS_MAX)
    szB = _al(n_nodes * 8)                           # t_nodes
    szC = _al(8 * nch * n_nodes * 8)                 # dlnA/dphi + cA1-3/cP1-3
    szBk = _al(n_knots * 8)                          # t_knots
    szD = _al(9 * n_fit * 8)                         # B_b + pcr
    szE = _al(2 * nch * n_nodes * 16 + 6 * n_nodes * 8
              + n_nodes * 4 + n_nodes * 1)           # extraction scratch
    szF = _al(2 * nch * n_knots * 8)                 # rk_re/rk_im
    szCI = 0 if band_len > 0 else _al(6 * nch * n_knots * 8)
    szG = _al(2 * nch * N_sparse_t * 8               # rpix_re/rpix_im
              + nch * M * nwords * 8)                # staged mask words

    if band_len > 0 and alias:
        # max(szE, szF): F overlays E, and n_knots > 2*n_nodes would make F
        # the larger of the two -- see the note in gb_sighet_v5_shared_bytes.
        return max(szA + max(szE, szF) + szB + szC + szBk + szD,  # node stage
                   szA + szF + szG)                               # r_pix/fold
    return szA + szB + szC + szBk + szD + szE + szF + szCI + szG


def blocks_per_sm(dyn_shared_bytes, nthreads=128, static_shared=2176,
                  per_block_reserved=1024, per_sm_shared=167936,
                  max_blocks_per_sm=32, max_threads_per_sm=2048):
    """Shared-memory-limited resident blocks per SM (register limits aside).

    ``static_shared`` covers the kernel's ``d_h_tmp``/``h_h_tmp`` staging
    (2 x nthreads x 8 B) plus cub's BlockReduce temp storage;
    ``per_block_reserved`` is the 1 KB the driver reserves per block on
    sm_80+. Both are omitted from the design note's arithmetic and together
    they are worth a whole block at the v5 footprint. Defaults are A100
    (per_sm_shared=167936); pass 233472 for H100.
    """
    per_block = dyn_shared_bytes + static_shared + per_block_reserved
    return max(1, min(int(per_sm_shared // per_block), max_blocks_per_sm,
                      int(max_threads_per_sm // nthreads)))


def device_shared_limit(default=163840):
    try:
        import cupy as cp
        return int(cp.cuda.Device(0).attributes["MaxSharedMemoryPerBlockOptin"])
    except Exception:                                        # noqa: BLE001
        return default


def fit_nt_layer(nt, edge, nr, knots, nch=3, m_half=2, n_sparse_fd=512,
                 margin=0.92):
    """Finest sparse resolution fitting every ENABLED engine on this device.

    Sized to ``ENGINES`` rather than to all four: the engines share one
    nt_layer, so an engine that is not under test must not set the grid for
    the ones that are. v2 is the expensive one (960 B per sparse point vs
    528 for v4-banded), so excluding it via BENCH_ENGINES is what lets the
    v3/v4 family run the finer grid it can actually afford.
    """
    lim = device_shared_limit() * margin
    cost = {
        "v2": lambda ntl, nsp: v2_shared_bytes(nch, m_half, ntl, nsp,
                                               n_sparse_fd),
        "v3": lambda ntl, nsp: sighet_shared_bytes(nr, knots, nch, m_half,
                                                   nsp, 0, False),
        "v4-pcr": lambda ntl, nsp: sighet_shared_bytes(nr, knots, nch, m_half,
                                                       nsp, 0, True),
        "v4-band": lambda ntl, nsp: sighet_shared_bytes(nr, knots, nch, m_half,
                                                        nsp, 16, True),
        "v5": lambda ntl, nsp: sighet_v5_shared_bytes(nr, knots, nch, m_half,
                                                      nsp, 16, True),
        "v5-ctl": lambda ntl, nsp: sighet_v5_shared_bytes(nr, knots, nch,
                                                          m_half, nsp, 16,
                                                          False),
    }
    active = [f for n, f in cost.items() if n in ENGINES] or list(cost.values())
    for stride in range(1, nt):
        if nt % stride:
            continue
        ntl, nsp = nt // stride, (nt - 2 * edge) // stride
        if nsp < 8:
            break
        if max(f(ntl, nsp) for f in active) <= lim:
            return ntl, nsp
    return 1, 8


# ===================== scaffold ==========================================

class Holder:
    """xp-aware data holder: device-resident slabs under CUDA."""

    def __init__(self, xp, data, invC):
        self.linear_data_arr = [xp.ascontiguousarray(data).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(invC).ravel()]

    def __len__(self):
        return 1


def asnp(a):
    g = getattr(a, "get", None)
    return g() if callable(g) else np.asarray(a)


# Shortest baseline we tune for.  Below this the node count is simply
# floored -- sub-3-month observations are not an operating regime, and the
# very small node counts the pure law would give there are both unvalidated
# and prone to leaving tiers unreachable.
MIN_TUNE_YR = 0.25


def nr_law(nt, Nf, dt, base=64):
    """Fit nodes predicted at this baseline.

    The ratio h_cand/h_ref is structured by the ANNUAL modulation
    (sky-Doppler + antenna pattern), so the oscillation count a spline must
    represent scales as Tobs / 1 yr.  Polynomial phase (df0, dfdot) costs no
    nodes at any baseline -- cubic splines represent it exactly -- so this
    law keys on the modulation alone.  ``base`` is calibrated at 1 yr and the
    result is floored at the MIN_TUNE_YR (3-month) value: n_r = 16 at
    base = 64.
    """
    yr = max(nt * Nf * dt / 86400.0 / 365.25, MIN_TUNE_YR)
    n = base * yr
    # snap to a power of two: keeps the node loop and its thread mapping
    # clean, and avoids meaningless counts like 63 or 252 that come out of
    # the 0.986-yr sidereal rounding
    n = 2 ** int(round(np.log2(max(n, 1.0))))
    return int(np.clip(n, 2 ** int(round(np.log2(max(base * MIN_TUNE_YR, 1.0)))),
                       256))


def nr_ladder(nt, Nf, dt, base=64):
    """Factor-of-two ladder bracketing the law, floored at MIN_TUNE_YR, so
    the sweep MEASURES the minimum passing node count instead of assuming
    the law is right."""
    n = nr_law(nt, Nf, dt, base)
    lo = max(4, nr_law(0, Nf, dt, base) // 2)     # half the 3-month floor
    return sorted({max(lo, n // 4), max(lo, n // 2), n, min(256, 2 * n)})


def build(nt, Nf, nr, knots, band, quiet=False):
    dt = 10.0
    t0 = int(0.5 * YRSID_SI / dt) * dt
    edge = max(2, int(round(0.027 * nt)))
    tk = max(2, int(round(0.025 * nt)))
    # nt_layer sets the sig-het sparse-time grid: stride = Nt // nt_layer,
    # N_sparse_t = Nt_active // stride. On GPU it is fitted to the shared
    # memory the enabled engines can afford. On CPU there is no such limit,
    # so the old hard-coded 512 gave N_sparse_t ~ 510 at EVERY baseline --
    # 2.5-4x the pixel work of the 127-204 the GPU actually runs, which
    # silently handicapped sig-het in every CPU-vs-GPU comparison. BENCH_NTL
    # was also trapped inside the GPU branch, so there was no way to correct
    # it. It now applies to both paths.
    nt_layer = 512
    if USE_GPU:
        nt_layer, _ = fit_nt_layer(nt, edge, nr, knots)
    nt_layer = int(os.environ.get("BENCH_NTL", str(nt_layer)))
    orbits = ESAOrbits(force_backend=BACKEND)
    ws = WDMSettings(Nf, nt, dt, t0=t0, min_freq=1e-4, max_freq=2e-2,
                     min_time=edge * Nf * dt, max_time=(nt - edge) * Nf * dt,
                     force_backend=BACKEND)
    # chunked-het splits the baseline into Nt_sub blocks because a GPU block
    # must hold one chunk in shared memory. On CPU there is no such limit and
    # chunking is pure overhead: the old hard-coded Nt_sub=128 cost 6 chunks
    # at 3 months rising to 90 at 4 years, each paying per-chunk fixed cost
    # plus the n_pad halo -- an overhead that GROWS WITH Tobs and so inflated
    # both the level and the shape of the CPU curve.
    #
    # Nt_sub must be a power of two (radix-2 FFT, asserted in chunked_het),
    # and Nt never is, so "one chunk" would mean Nt_sub = ceil-power-of-two:
    # a chunk overhanging the data by up to 1.9x (7744 pixels past the end at
    # Nt=8640). That segfaults. Take the FLOOR power of two instead: two
    # chunks, ZERO pixels past the data, and the redundant span is interior
    # OVERLAP that compute_chunk_geometry's partial-slide case already trims
    # via keep_lo/keep_hi -- the same machinery the GPU path uses. Kept
    # pixels then sum to exactly Nt: no double-count, no gap.
    # CPU_NT_SUB_MAX = FAST_WDM_NT_SUB_MAX, the CPU compile-time cap in
    # lisatools/cutils/lat_chunked_het_kernels.hh. On the CPU branch the
    # kernel's "shared" buffers are STACK arrays sized at that maximum
    # (layer_buf_cpu[NCHANNELS_MAX * NT_SUB_MAX]), and nothing checks Nt_sub
    # against it -- so Nt_sub=8192 overruns the stack and segfaults. That is
    # the whole story behind the cluster crashes: absolute size, not overhang.
    #
    # Take the CEILING power of two, capped there. Ceiling gives ONE chunk
    # wherever it fits, and one chunk is measurably faster than two of half
    # the size even though the computed span is identical -- per-chunk fixed
    # cost is real on CPU, where chunks are a sequential loop. Measured at
    # Nt=540, batch 32: Nt_sub=1024 (1 chunk) 4848 us vs Nt_sub=512
    # (2 chunks) 8235 us, i.e. the floor variant was 1.70x SLOWER. The cap
    # only binds for Nt > 4096, which then splits into 2 or 3 chunks.
    CPU_NT_SUB_MAX = 4096
    nt_sub = int(os.environ.get(
        "BENCH_NT_SUB",
        128 if USE_GPU
        else min(1 << int(np.ceil(np.log2(max(nt, 64)))), CPU_NT_SUB_MAX)))
    chunked = GBWDMComputations(
        ws, t_ref=t0, Nt_sub=nt_sub, n_pad=16, N_sparse=256, N_cp_sig=48,
        N_cp_orbit=32, orbits=orbits, tdi_config="2nd generation",
        force_backend=BACKEND, d_d=0.0, tdi_type="XYZ",
        tukey_alpha=2.0 * tk / nt)
    chunked.convert_to_ra_dec = False
    mk = lambda **kw: GBSignalHetComputations.for_band_engine(     # noqa: E731
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=nt_layer,
        m_active_half_width=2, **kw)
    builders = {"v2": lambda: mk(),
                "v3": lambda: mk(v3_n_nodes=nr),
                "v4-pcr": lambda: mk(v3_n_nodes=nr, v4_knots=knots, v4_band=0),
                "v4-band": lambda: mk(v3_n_nodes=nr, v4_knots=knots,
                                      v4_band=band),
                # v5 == v4-banded with the per-pixel fold scratch relocated;
                # v5=1 -> global slab, v5=2 -> shared control.
                "v5": lambda: mk(v3_n_nodes=nr, v4_knots=knots,
                                 v4_band=band, v5=1),
                "v5-ctl": lambda: mk(v3_n_nodes=nr, v4_knots=knots,
                                     v4_band=band, v5=2)}
    engines = {n: f() for n, f in builders.items() if n in ENGINES}
    # ``engines`` holds the SIG-HET engines only -- ``chunked`` is tracked
    # separately and always kept -- so a chunked-only run (BENCH_ENGINES=chunked,
    # the natural first step when bisecting a failure) leaves this dict EMPTY.
    # Build one throwaway engine purely to read the sparse-grid geometry rather
    # than raising StopIteration on ``next(iter(...))``.
    if engines:
        nsp = next(iter(engines.values()))._g["N_sparse_t"]
    else:
        nsp = builders["v4-band"]()._g["N_sparse_t"]

    # Drop engines whose scorer cannot fit this device's shared memory.
    # Without this the kernel launches anyway and dies on
    # "GPUassert: invalid argument" inside the .cu, which reads like a bug in
    # the engine rather than a grid that was simply set too fine -- and it
    # takes the whole sweep down with it. An explicit skip keeps the engines
    # that DO fit measurable, which is the entire point of a grid override.
    if USE_GPU:
        lim = device_shared_limit()
        need = {"v2": v2_shared_bytes(3, 2, nt_layer, nsp, 512),
                "v3": sighet_shared_bytes(nr, knots, 3, 2, nsp, 0, False),
                "v4-pcr": sighet_shared_bytes(nr, knots, 3, 2, nsp, 0, True),
                "v4-band": sighet_shared_bytes(nr, knots, 3, 2, nsp, 16, True),
                "v5": sighet_v5_shared_bytes(nr, knots, 3, 2, nsp, 16, True),
                "v5-ctl": sighet_v5_shared_bytes(nr, knots, 3, 2, nsp, 16,
                                                 False)}
        for n in [k for k in engines if need.get(k, 0) > lim]:
            print(f"  [skip] {n}: scorer needs {need[n]/1024:.0f} KB > "
                  f"{lim/1024:.0f} KB device limit at N_sparse_t={nsp} "
                  f"(nt_layer={nt_layer}). Lower BENCH_NTL to include it.")
            engines.pop(n)
        if not engines:
            raise RuntimeError(
                f"no sig-het engine fits {lim/1024:.0f} KB at "
                f"N_sparse_t={nsp}; lower BENCH_NTL")
    if not quiet:
        lim = device_shared_limit()
        b = {k: v for k, v in {
            "v2": v2_shared_bytes(3, 2, nt_layer, nsp, 512),
            "v3": sighet_shared_bytes(nr, knots, 3, 2, nsp, 0, False),
            "v4-pcr": sighet_shared_bytes(nr, knots, 3, 2, nsp, 0, True),
            "v4-band": sighet_shared_bytes(nr, knots, 3, 2, nsp, 16, True),
            "v5": sighet_v5_shared_bytes(nr, knots, 3, 2, nsp, 16, True),
            "v5-ctl": sighet_v5_shared_bytes(nr, knots, 3, 2, nsp, 16, False),
        }.items() if k in ENGINES}
        print(f"[chunk] Nt_sub={nt_sub} -> chunked-het n_chunks="
              f"{chunked.n_chunks}"
              + ("" if USE_GPU else "  (CPU: one whole chunk expected)"))
        print(f"[grid] Nf={Nf} Nt={nt} nt_layer={nt_layer} | Tobs="
              f"{nt*Nf*dt/86400:.0f} d ({nt*Nf*dt/86400/365.25:.2f} yr), "
              f"sparse stride={max(1, nt//nt_layer)} "
              f"({max(1, nt//nt_layer)*Nf*dt/3600:.0f} h), N_sparse_t={nsp}")
        print(f"[shmem] limit {lim/1024:.0f} KB | " + ", ".join(
            f"{k} {v/1024:.0f}" for k, v in b.items())
            + (" KB  " + ("OK" if max(b.values()) <= lim else "TOO BIG")
               if USE_GPU else " KB"))
        print(f"[nodes] n_r={nr}, K={knots}, half-band={band}")
    return ws, chunked, engines, Nf, dt, t0, nsp, nt_layer



def make_invC(ws, xp, model=None):
    """PHYSICAL inverse sensitivity, so likelihoods are in real lnL units.

    An identity weighting makes d_h/h_h raw waveform quantities (~1e-39 for
    a 1e-22 source), which silently defeats every absolute lnL threshold --
    the tier placement then finds no displacement that moves the likelihood
    by 0.05 and reports no accuracy at all.  Use the analytic XYZ
    sensitivity unless explicitly overridden (BENCH_PSD=identity for a pure
    timing run).
    """
    model = model or os.environ.get("BENCH_PSD", "scirdv1")
    if model == "identity":
        return None            # caller falls back to identity weighting
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    inv = XYZ2SensitivityMatrix(ws, model=model).invC
    # invC already lives on the settings' own backend: cupy under CUDA,
    # numpy on CPU. Only move it when the two disagree -- cupy refuses an
    # implicit host conversion, and xp.asarray on a matching module is a
    # no-op rather than a copy.
    if xp is np and hasattr(inv, "get"):
        inv = inv.get()
    return xp.ascontiguousarray(xp.asarray(inv, dtype=xp.float64))


def make_refs(n):
    rng = np.random.default_rng(19)
    return [np.array([
        10 ** rng.uniform(-22.5, -21.0), rng.uniform(1.5e-3, 1.5e-2),
        rng.uniform(0.0, 3e-16), 0.0, rng.uniform(0, 2 * np.pi),
        np.arccos(rng.uniform(-1, 1)), rng.uniform(0, np.pi),
        rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1))])
        for _ in range(n)]



def wm(v):
    """(worst, median) of a possibly-empty error list.

    A tier can come up empty when a short baseline cannot reach that
    |dlnL| in a given direction before leaving the prior box -- report it
    as missing rather than crashing or, worse, silently plotting zero.
    """
    return (float(np.max(v)), float(np.median(v))) if len(v) else (np.nan,
                                                                   np.nan)


def disp(ref, idx, s):
    p = ref.copy()
    if idx == 0:
        p[0] *= np.exp(s)
    else:
        p[idx] += s
    return p


def timed_and_scored(engine, holder, batch, kw, reps):
    """Time a call AND take the accuracy sample from that same call.

    Accuracy and speed must come from the SAME computational unit, or a
    timing loop could be measuring different work than the one that was
    validated (a degenerate config, a silently-clamped batch, a cached
    path).  So:

      1. one pre-loop call, whose OUTPUT is the accuracy sample;
      2. the timing loop, running the identical call, outputs unused;
      3. re-read the outputs afterwards and verify the last iteration
         reproduces the first EXACTLY -- these kernels are deterministic,
         so any nonzero drift means the timed calls were not the call we
         scored, and the measurement is void.

    Returns (us_per_candidate, deltas_from_first_call, drift).
    """
    def _deltas():
        return (asnp(engine.d_h_out).real.astype(float)
                - 0.5 * asnp(engine.h_h_out).real.astype(float))

    engine.get_ll(holder, batch, phase_maximize=False, **kw)
    sync()
    first = _deltas().copy()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        engine.get_ll(holder, batch, phase_maximize=False, **kw)
        sync()
        ts.append(time.perf_counter() - t)
    last = _deltas()
    n = min(len(first), len(last))
    drift = float(np.max(np.abs(first[:n] - last[:n]))) if n else 0.0
    return min(ts) / batch.shape[0] * 1e6, first, drift



def build_tier_batch(engc, holder, xp, ref, nt, Nf, dt, tiers):
    """Row 0 = the reference point; rows 1.. = candidates placed at each
    tier (true |dlnL| = T) in each direction.

    Scoring this ONE batch gives every engine its delta-from-reference for
    every tier -- so the accuracy sample and the timed call are literally
    the same kernel launch.  The placement uses the chunked reference
    engine, which also supplies the TRUE dlnL each row sits at.
    """
    def delta(e, p):
        z = np.zeros(1, dtype=np.int32)
        e.get_ll(holder, xp.asarray(p)[None, :], phase_maximize=False,
                 data_index=z, noise_index=z, N_vals=None,
                 waveform_kwargs={})
        return (float(asnp(e.d_h_out)[0]) - 0.5 * float(asnp(e.h_h_out)[0]))

    drift = 1.0 / (2 * np.pi * (nt * Nf * dt))
    DIRS = [(1, 0.05 * drift), (0, 0.03), (5, 0.03), (6, 0.03), (7, 0.01)]
    d0 = delta(engc, ref)
    rows, trueT = [ref.copy()], [0.0]
    for idx, s0 in DIRS:
        s, tries = s0, 0
        dl = delta(engc, disp(ref, idx, s)) - d0
        while abs(dl) < 0.05 and tries < 3:
            s *= 10.0
            dl = delta(engc, disp(ref, idx, s)) - d0
            tries += 1
        if abs(dl) < 0.05:
            continue
        for T in tiers:
            sc = s * np.sqrt(T / abs(dl))
            p = disp(ref, idx, sc)
            dT = delta(engc, p) - d0
            if abs(dT) > 0.05:
                sc *= np.sqrt(T / abs(dT))
                p = disp(ref, idx, sc)
                dT = delta(engc, p) - d0
            rows.append(p)
            trueT.append(dT)
    if len(rows) == 1:
        print(f"  [warn] Nt={nt}: no tier candidate could be placed "
              f"(displacements never reached |dlnL| > 0.05). Accuracy will "
              f"be MISSING at this baseline -- it is below the regime the "
              f"tiers are defined for.")
    return np.array(rows), np.array(trueT)


def pad_batch(rows, nb, xp):
    """Grow (or truncate) the tier rows to the requested batch size by
    repeating them; accuracy always reads the first len(rows) entries."""
    if nb <= rows.shape[0]:
        return xp.asarray(rows[:nb]), nb
    reps = int(np.ceil(nb / rows.shape[0]))
    return xp.asarray(np.tile(rows, (reps, 1))[:nb]), rows.shape[0]


def sync():
    if USE_GPU:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()


def main():
    out = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out, exist_ok=True)
    nts = [int(x) for x in os.environ.get(
        "BENCH_NT_LIST",
        "540,1080,2160,4320,8640" if USE_GPU else "1024").split(",")]
    batches = [int(x) for x in os.environ.get(
        "BENCH_BATCHES",
        "16,64,256,1024,4096,16384" if USE_GPU else "8").split(",")]
    Nf = int(os.environ.get("BENCH_NF", "1440" if USE_GPU else "256"))
    n_ref = int(os.environ.get("BENCH_NREF", "4"))
    tiers = [float(x) for x in
             os.environ.get("BENCH_TIERS", "1,10,50,100,1000").split(",")]
    nr = int(os.environ.get("BENCH_NR", "64"))
    knots = int(os.environ.get("BENCH_K", "128"))
    band = int(os.environ.get("BENCH_BAND", "16"))
    reps = int(os.environ.get("BENCH_REPS", "5"))
    do_settings = os.environ.get("BENCH_SETTINGS", "1") == "1"
    T_rep = float(os.environ.get("BENCH_T_REP", "100"))
    T_rep = min(tiers, key=lambda t: abs(t - T_rep))
    refs = make_refs(n_ref)

    speed, err_nt, shm, occ, meta = {}, {}, {}, {}, {}
    settings = {}                       # (nt, nr, K, band) -> cost/med/worst

    for nt in nts:
        print(f"\n=== Nt={nt} " + "=" * 52)
        ws, chunked, engs, Nf_, dt, t0, nsp, ntl = build(
            nt, Nf, nr, knots, band)
        xp = chunked.xp
        ilo, ihi = ws.ind_min_f, ws.ind_max_f + 1
        meta[nt] = dict(Tobs_yr=nt * Nf * dt / 86400.0 / 365.25, nsp=nsp,
                        ntl=ntl,
                        stride_h=max(1, nt // ntl) * Nf * dt / 3600.0)
        ref0 = np.array([1e-22, 7.5e-3, 1e-16, 0.0, 1.2, 0.9, 0.4, 2.0, 0.3])
        href = xp.zeros((3, Nf, nt))
        chunked.fill_global_wdm(xp.asarray(ref0)[None, :], href,
                                convert_to_ra_dec=False)
        h_act = xp.ascontiguousarray(href[:, ilo:ihi, ws.active_slice_t])
        invC_phys = make_invC(ws, xp)
        invC = invC_phys
        if invC is None:
            invC = xp.zeros((3, 3) + h_act.shape[1:])
            for c in range(3):
                invC[c, c] = 1.0
        holder = Holder(xp, h_act, invC)

        # ---- speed AND accuracy from the SAME calls ---------------------
        # For each reference: build one batch whose row 0 is the reference
        # and whose remaining rows sit at each accuracy tier. Every engine
        # scores that batch; the pre-loop call supplies the accuracy sample,
        # the loop supplies the timing, and a post-loop re-read proves the
        # timed calls reproduced the scored call exactly.
        engc = WDMBandLikelihoodEngine(chunked, ws, nchannels=3,
                                       tdi_channel_setup="XYZ")
        ref_ctx = []          # (ref, holder, chunked engine) per reference,
                              # reused by the settings frontier below
        err_nt[nt] = {n: {T: [] for T in tiers} for n in ENGINES[1:]}
        scored = {}          # (ref index, engine) -> accuracy already taken
        drift_max = 0.0
        for ri, ref in enumerate(refs):
            hr = xp.zeros((3, Nf, nt))
            chunked.fill_global_wdm(xp.asarray(ref)[None, :], hr,
                                    convert_to_ra_dec=False)
            ha = xp.ascontiguousarray(hr[:, ilo:ihi, ws.active_slice_t])
            iC = invC_phys
            if iC is None:
                iC = xp.zeros((3, 3) + ha.shape[1:])
                for c in range(3):
                    iC[c, c] = 1.0
            hold_r = Holder(xp, ha, iC)
            engc_r = WDMBandLikelihoodEngine(chunked, ws, nchannels=3,
                                             tdi_channel_setup="XYZ")
            ref_ctx.append((ref, hold_r, engc_r))
            rows, trueT = build_tier_batch(engc_r, hold_r, xp, ref, nt, Nf,
                                           dt, tiers)
            eng_of = {"chunked": engc_r}
            for name, sh in engs.items():
                sh.clear_in_model()
                sh.setup_in_model(hold_r, xp.asarray(ref)[None, :],
                                  np.zeros(1, np.int32))
                eng_of[name] = WDMBandLikelihoodEngine(
                    sh, ws, nchannels=3, tdi_channel_setup="XYZ")
            for nb in batches:
                batch, n_acc = pad_batch(rows, nb, xp)
                z = np.zeros(batch.shape[0], dtype=np.int32)
                kw = dict(data_index=z, noise_index=z, N_vals=None,
                          waveform_kwargs={})
                for name in ENGINES:
                    if name not in eng_of:      # skipped: exceeds shared mem
                        speed[(nt, nb, name)] = np.nan
                        continue
                    try:
                        us, dv, dr = timed_and_scored(eng_of[name], hold_r,
                                                      batch, kw, reps)
                    except Exception as exc:                 # noqa: BLE001
                        print(f"  [{name}] nb={nb} FAILED: "
                              f"{type(exc).__name__}: {exc}")
                        speed[(nt, nb, name)] = np.nan
                        continue
                    drift_max = max(drift_max, dr)
                    if ri == 0:
                        speed[(nt, nb, name)] = us
                    # Accuracy from the first batch this engine scores that
                    # actually CONTAINS tier rows. The arithmetic is identical
                    # at every batch, so one sample suffices -- but keying it
                    # to the largest batch loses accuracy entirely whenever an
                    # engine cannot run there (memory), which is exactly how a
                    # whole column turns into NaN.
                    #
                    # The flag must be set only once a sample is really taken.
                    # pad_batch TRUNCATES when nb < len(rows), so a batch
                    # smaller than the tier count carries only row 0 (the
                    # reference) and the loop below is empty -- marking the
                    # engine scored there would silently forfeit accuracy for
                    # this reference at every LARGER batch too. That is what
                    # emptied every accuracy column as soon as BENCH_BATCHES
                    # started at 1 instead of 256.
                    if name != "chunked":
                        base = dv[0]
                        # scored[] holds the number of tier rows ALREADY
                        # recorded for this (reference, engine), starting at 1
                        # because row 0 is the reference itself. Batches are
                        # ascending and pad_batch truncates, so a small batch
                        # carries only the first few tier rows; picking up
                        # where the last batch stopped accumulates the full
                        # set without ever double-counting a row.
                        start = scored.get((ri, name), 1)
                        stop = min(n_acc, len(dv))
                        for k in range(start, stop):
                            T = min(tiers,
                                    key=lambda t: abs(t - abs(trueT[k])))
                            err_nt[nt][name][T].append(
                                abs((dv[k] - base) - trueT[k]))
                        if stop > start:
                            scored[(ri, name)] = stop
                if ri == 0:
                    print("[speed] batch %6d: " % nb + " ".join(
                        f"{n}={speed[(nt, nb, n)]:.1f}" for n in ENGINES)
                        + " us")
        print(f"[verify] max |first-call - last-call| over every timed "
              f"loop = {drift_max:.3e}  "
              f"({'OK: timed == scored' if drift_max == 0.0 else 'NONZERO -- timing and accuracy are NOT the same call'})")
        nsamp = {n: sum(len(v) for v in err_nt[nt][n].values())
                 for n in ENGINES[1:]}
        print("[acc  ] samples/engine: " + " ".join(
            f"{n}={nsamp[n]}" for n in ENGINES[1:]))
        if not any(nsamp.values()):
            print("  [warn] NO accuracy samples at this baseline -- either no "
                  "tier candidate could be placed, or every engine failed to "
                  "score. The speed columns are still valid.")
        print("[acc  ] worst |dlnL| @T=%d: " % T_rep + " ".join(
            f"{n}={wm(err_nt[nt][n][T_rep])[0]:.2e}" for n in ENGINES[1:]))

        lim = device_shared_limit()
        shm[nt] = {k: v for k, v in {
            "v2": v2_shared_bytes(3, 2, ntl, nsp, 512),
            "v3": sighet_shared_bytes(nr, knots, 3, 2, nsp, 0, False),
            "v4-pcr": sighet_shared_bytes(nr, knots, 3, 2, nsp, 0, True),
            "v4-band": sighet_shared_bytes(nr, knots, 3, 2, nsp, 16, True),
            "v5": sighet_v5_shared_bytes(nr, knots, 3, 2, nsp, 16, True),
            "v5-ctl": sighet_v5_shared_bytes(nr, knots, 3, 2, nsp, 16, False),
        }.items() if k in ENGINES}
        occ[nt] = {k: max(1, int(lim // v)) for k, v in shm[nt].items()}

        # ---- settings frontier ------------------------------------------
        if do_settings:
            nb_s = batches[-1]
            for nr_i in nr_ladder(nt, Nf, dt, base=nr):
                for K_i in (max(64, knots // 2), knots, min(512, knots * 2)):
                    for band_i in (0, band):
                        # A config whose scorer exceeds the device's shared
                        # memory aborts the PROCESS (GPUassert), not the
                        # Python call -- the grid was sized for the default
                        # nodes/knots, so a larger frontier point must be
                        # skipped BEFORE it is launched.
                        need = sighet_shared_bytes(nr_i, K_i, 3, 2, nsp,
                                                   band_i, v4=True)
                        lim_i = device_shared_limit()
                        if USE_GPU and need > lim_i:
                            print(f"[set  ] nr={nr_i:3d} K={K_i:3d} "
                                  f"{'band' if band_i else 'pcr ':>4s}: "
                                  f"SKIP -- needs {need/1024:.0f} KB > "
                                  f"{lim_i/1024:.0f} KB")
                            continue
                        try:
                            sh = GBSignalHetComputations.for_band_engine(
                                chunked, n_sparse_fd=512, n_cp_build=93,
                                nt_layer=ntl, m_active_half_width=2,
                                v3_n_nodes=nr_i, v4_knots=K_i,
                                v4_band=band_i)
                            # Score against the SAME reference battery the
                            # engine comparison uses. A frontier measured on
                            # one benign source understates the error by
                            # orders of magnitude -- the errors are dominated
                            # by specific hard geometries, not by the knobs.
                            e_s, cost = [], None
                            for ref_s, hold_s, engc_s in ref_ctx:
                                sh.clear_in_model()
                                sh.setup_in_model(
                                    hold_s, xp.asarray(ref_s)[None, :],
                                    np.zeros(1, np.int32))
                                en = WDMBandLikelihoodEngine(
                                    sh, ws, nchannels=3,
                                    tdi_channel_setup="XYZ")
                                rows_s, trueT_s = build_tier_batch(
                                    engc_s, hold_s, xp, ref_s, nt, Nf, dt,
                                    [T_rep])
                                b_s, n_as = pad_batch(rows_s, nb_s, xp)
                                z_s = np.zeros(b_s.shape[0], dtype=np.int32)
                                kw_s = dict(data_index=z_s, noise_index=z_s,
                                            N_vals=None, waveform_kwargs={})
                                c_i, dv_s, _ = timed_and_scored(
                                    en, hold_s, b_s, kw_s, max(1, reps // 2))
                                cost = c_i if cost is None else min(cost, c_i)
                                e_s += [abs((dv_s[k] - dv_s[0]) - trueT_s[k])
                                        for k in range(1, min(n_as,
                                                              len(dv_s)))]
                            w_, m_ = wm(e_s)
                            settings[(nt, nr_i, K_i, band_i)] = dict(
                                cost=cost, med=m_, worst=w_)
                            print(f"[set  ] nr={nr_i:3d} K={K_i:3d} "
                                  f"{'band' if band_i else 'pcr ':>4s}: "
                                  f"{cost:7.2f} us worst={w_:.3e}")
                        except Exception as exc:             # noqa: BLE001
                            print(f"[set  ] nr={nr_i} K={K_i} "
                                  f"band={band_i} FAILED: {exc}")

    report(out, nts, batches, tiers, T_rep, speed, err_nt, settings, shm, occ,
           meta, Nf, nr, knots, band)


def report(out, nts, batches, tiers, T_rep, speed, err_nt, settings, shm, occ,
           meta, Nf, nr, knots, band):
    nb = batches[-1]
    # ---------------- tables ---------------------------------------------
    print("\n" + "=" * 78)
    print(f"[SPEED] us per candidate, batch {nb}")
    print("  Tobs[yr]" + "".join(f"{n:>12s}" for n in ENGINES))
    for nt in nts:
        print(f"  {meta[nt]['Tobs_yr']:7.2f} " + "".join(
            f"{speed[(nt, nb, n)]:12.1f}" for n in ENGINES))
    print(f"\n[SPEEDUP] vs chunked-het, batch {nb}")
    print("  Tobs[yr]" + "".join(f"{n:>12s}" for n in ENGINES[1:]))
    for nt in nts:
        print(f"  {meta[nt]['Tobs_yr']:7.2f} " + "".join(
            f"{speed[(nt, nb, 'chunked')]/speed[(nt, nb, n)]:11.1f}x"
            for n in ENGINES[1:]))
    print(f"\n[ACCURACY] worst / median |dlnL| vs chunked   "
          f"(allowed = max(0.1, T/100))")
    print("  Tobs[yr] engine    " + "".join(
        f"{('T=' + str(int(T))):>20s}" for T in tiers))
    for nt in nts:
        for n in ENGINES[1:]:
            row = ""
            for T in tiers:
                v = err_nt[nt][n][T]
                w_, m_ = wm(v)
                row += (f"{w_:9.2e}/{m_:7.1e}"
                        f"{'OK' if w_ <= max(0.1, T/100) else 'OVER':>4s}"
                        if len(v) else f"{'--':>20s}")
            print(f"  {meta[nt]['Tobs_yr']:7.2f}  {n:9s}{row}")
    print("\n[GRID] sparse-time resolution actually used")
    print("  Tobs[yr]  N_sparse_t  stride[h]")
    for nt in nts:
        m = meta[nt]
        print(f"  {m['Tobs_yr']:7.2f}  {m['nsp']:10d}  {m['stride_h']:9.0f}")
    if settings:
        allowed = max(0.1, T_rep / 100.0)
        print(f"\n[RECOMMEND] cheapest configuration meeting the bar at "
              f"T={T_rep:.0f} (allowed {allowed:.2f})")
        for nt in nts:
            c = [(k, v) for k, v in settings.items()
                 if k[0] == nt and np.isfinite(v["worst"])
                 and v["worst"] <= allowed]
            if c:
                key, v = min(c, key=lambda kv: kv[1]["cost"])
                _, nr_i, K_i, b_i = key
                print(f"  {meta[nt]['Tobs_yr']:7.2f} yr: n_r={nr_i} K={K_i} "
                      f"{'banded' if b_i else 'pcr'} -> {v['cost']:.2f} us "
                      f"(worst {v['worst']:.2e})")
            else:
                print(f"  {meta[nt]['Tobs_yr']:7.2f} yr: none passed -- "
                      f"widen the ladder")

    # ---------------- npz -------------------------------------------------
    np.savez(
        os.path.join(out, "gb_engine_benchmark.npz"),
        engines=np.array(ENGINES), tiers=np.array(tiers),
        nt_list=np.array(nts), batches=np.array(batches),
        speed=np.array([[nt, b_, ENGINES.index(n), speed[(nt, b_, n)]]
                        for nt in nts for b_ in batches for n in ENGINES],
                       dtype=float),
        err=np.array([[nt, ENGINES.index(n), T, e] for nt in nts
                      for n in ENGINES[1:] for T in tiers
                      for e in err_nt[nt][n][T]], dtype=float),
        settings=(np.array([[k[0], k[1], k[2], k[3], v["cost"], v["med"],
                             v["worst"]] for k, v in sorted(settings.items())],
                           dtype=float) if settings else np.zeros((0, 7))),
        settings_columns=np.array(["Nt", "nr", "K", "band", "cost_us",
                                   "median_err", "worst_err"]),
        meta=np.array([[nt, meta[nt]["Tobs_yr"], meta[nt]["nsp"],
                        meta[nt]["stride_h"]] for nt in nts], dtype=float))
    try:
        figure(out, nts, batches, tiers, T_rep, speed, err_nt, settings, shm,
               occ, meta, Nf, nr, knots, band)
    except Exception as exc:                                 # noqa: BLE001
        # the measurement is the product; a plotting failure must never
        # discard it -- the npz and the printed tables are already written
        import traceback
        print(f"\n[warn] figure failed ({type(exc).__name__}: {exc}); "
              f"data is saved -- replot from the npz.")
        traceback.print_exc()



def finite_xy(xs, ys):
    """Keep only pairs where BOTH are finite and positive (log axes)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    return xs[m], ys[m]


def logsafe(ax, which="both"):
    """Apply log scaling only if the axis actually holds positive data --
    a run whose accuracy rows are all missing must still produce a figure."""
    try:
        if which in ("x", "both"):
            xs = np.concatenate([np.asarray(l.get_xdata(), float)
                                 for l in ax.get_lines()]) if ax.get_lines() \
                else np.array([])
            if np.isfinite(xs).any() and (xs[np.isfinite(xs)] > 0).any():
                ax.set_xscale("log")
        if which in ("y", "both"):
            ys = np.concatenate([np.asarray(l.get_ydata(), float)
                                 for l in ax.get_lines()]) if ax.get_lines() \
                else np.array([])
            if np.isfinite(ys).any() and (ys[np.isfinite(ys)] > 0).any():
                ax.set_yscale("log")
            else:
                ax.text(0.5, 0.5, "no finite data", transform=ax.transAxes,
                        ha="center", va="center", color=INK2, fontsize=9)
    except Exception:                                        # noqa: BLE001
        pass


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def figure(out, nts, batches, tiers, T_rep, speed, err_nt, settings, shm, occ,
           meta, Nf, nr, knots, band):
    nb = batches[-1]
    yrs = [meta[nt]["Tobs_yr"] for nt in nts]
    fig = plt.figure(figsize=(16.5, 10.4))
    gs = fig.add_gridspec(2, 4, hspace=0.36, wspace=0.30, left=0.05,
                          right=0.985, top=0.885, bottom=0.075)

    ax = fig.add_subplot(gs[0, 0]); style(ax)
    for n in ENGINES:
        ax.plot(yrs, [speed[(nt, nb, n)] for nt in nts], marker=MRK[n],
                color=CLR[n], lw=2, ms=7, ls=LS[n], label=n)
    # logsafe, not a bare set_yscale: an engine that could not run leaves an
    # all-NaN column, and matplotlib only raises "cannot be log-scaled" at
    # savefig -- which throws away the whole figure for one empty panel.
    logsafe(ax, "y")
    ax.set_xlabel("observation time [yr]")
    ax.set_ylabel("per-candidate cost [$\\mu$s]")
    ax.set_title("A · cost vs baseline", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2)

    ax = fig.add_subplot(gs[0, 1]); style(ax)
    for n in ENGINES[1:]:
        y = [speed[(nt, nb, "chunked")] / speed[(nt, nb, n)] for nt in nts]
        ax.plot(yrs, y, marker=MRK[n], color=CLR[n], lw=2, ms=7, ls=LS[n],
                label=n)
        ax.annotate(f"{y[-1]:.0f}$\\times$", (yrs[-1], y[-1]), color=INK,
                    fontsize=8, xytext=(5, 0), textcoords="offset points",
                    va="center")
    ax.axhline(1, color=CLR["chunked"], lw=1.4, ls="--")
    ax.set_xlabel("observation time [yr]"); ax.set_ylabel("speedup vs chunked")
    ax.set_title("B · the payoff grows with baseline", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[0, 2]); style(ax)
    allow = [max(0.1, T / 100.0) for T in tiers]
    ax.fill_between(tiers, 1e-9, allow, color=OKC, alpha=0.10, zorder=0)
    ax.plot(tiers, allow, color=OKC, lw=1.6, ls="--", label="allowed(T)")
    nt_ref = nts[len(nts) // 2]
    for n in ENGINES[1:]:
        ax.plot(tiers, [wm(err_nt[nt_ref][n][T])[1] for T in tiers],
                marker=MRK[n], color=CLR[n], lw=2, ms=6, ls=LS[n], label=n)
        ax.plot(tiers, [wm(err_nt[nt_ref][n][T])[0] for T in tiers],
                color=CLR[n], lw=1, ls=":", alpha=0.85)
    logsafe(ax)
    ax.set_xlabel("true $|\\Delta\\ln L|$ from reference, T")
    ax.set_ylabel("likelihood error")
    ax.set_title(f"C · tiered accuracy @ {meta[nt_ref]['Tobs_yr']:.2f} yr",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="upper left")

    ax = fig.add_subplot(gs[0, 3]); style(ax)
    for n in ENGINES[1:]:
        ax.plot(yrs, [wm(err_nt[nt][n][T_rep])[0] for nt in nts],
                marker=MRK[n], color=CLR[n], lw=2, ms=7, ls=LS[n], label=n)
    ax.axhline(max(0.1, T_rep / 100.0), color=OKC, lw=1.6, ls="--")
    logsafe(ax, "y")
    ax.set_xlabel("observation time [yr]")
    ax.set_ylabel(f"worst error at T={T_rep:.0f}")
    ax.set_title("D · does accuracy hold as the baseline grows?", fontsize=10,
                 loc="left")

    ax = fig.add_subplot(gs[1, 0]); style(ax)
    for n in ENGINES[1:]:
        xs = [speed[(nt, nb, n)] for nt in nts]
        ys = [wm(err_nt[nt][n][T_rep])[0] for nt in nts]
        ax.plot(xs, ys, color=CLR[n], lw=1.5, alpha=0.7, ls=LS[n])
        ax.scatter(xs, ys, s=[26 + 22 * k for k in range(len(xs))],
                   color=CLR[n], marker=MRK[n], edgecolor="white", lw=1.2,
                   zorder=3, label=n)
    ax.axhline(max(0.1, T_rep / 100.0), color=OKC, lw=1.6, ls="--")
    logsafe(ax)
    ax.set_xlabel("cost [$\\mu$s]")
    ax.set_ylabel(f"worst error T={T_rep:.0f}")
    ax.set_title("E · accuracy vs cost (size = baseline)", fontsize=10,
                 loc="left")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="upper right")

    ax = fig.add_subplot(gs[1, 1]); style(ax)
    for n in ENGINES:
        y = np.array([speed[(nts[len(nts)//2], b_, n)] for b_ in batches],
                     float)
        ax.plot(batches, 100 * np.nanmin(y) / y, marker=MRK[n], color=CLR[n],
                lw=2, ms=7, ls=LS[n], label=n)
    ax.axhline(95, color=INK2, lw=1.0, ls="--")
    ax.set_xscale("log", base=2); ax.set_ylim(0, 108)
    ax.set_xlabel("candidates per call"); ax.set_ylabel("% of peak throughput")
    ax.set_title("F · GPU saturation", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[1, 2]); style(ax)
    if settings:
        all_nr = sorted({k[1] for k in settings})
        for (nt, nr_i, K_i, b_i), v in settings.items():
            i = all_nr.index(nr_i)
            col = RAMP[min(int(round(i * (len(RAMP) - 1)
                                     / max(1, len(all_nr) - 1))),
                           len(RAMP) - 1)]
            ax.scatter([v["cost"]], [v["worst"]], s=80, color=col,
                       marker=K_MRK.get(K_i, "o"),
                       edgecolor="white" if b_i else INK2,
                       lw=1.5 if b_i else 0.9, zorder=3)
        ax.axhline(max(0.1, T_rep / 100.0), color=OKC, lw=1.6, ls="--")
        logsafe(ax)
        ax.set_xlabel("cost [$\\mu$s]")
        ax.set_ylabel(f"worst error T={T_rep:.0f}")
    ax.set_title("G · settings frontier (dark = more nodes)", fontsize=10,
                 loc="left")

    ax = fig.add_subplot(gs[1, 3]); style(ax)
    names = [n for n in ["v2", "v3", "v4-pcr", "v4-band", "v5", "v5-ctl"]
             if n in ENGINES]
    vals = [shm[nts[len(nts)//2]][n] / 1024 for n in names]
    bars = ax.bar(range(len(names)), vals, color=[CLR[n] for n in names],
                  width=0.62)
    lim = device_shared_limit() / 1024
    ax.axhline(lim, color=BADC, lw=1.6, ls="--")
    ax.annotate(f"device limit {lim:.0f} KB", (len(names) - 0.5, lim),
                color=BADC, fontsize=8, ha="right", xytext=(0, 4),
                textcoords="offset points")
    for b_, n in zip(bars, names):
        ax.annotate(f"{occ[nts[len(nts)//2]][n]} blk/SM",
                    (b_.get_x() + b_.get_width() / 2, b_.get_height()),
                    color=INK, fontsize=8, ha="center", xytext=(0, 3),
                    textcoords="offset points")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, color=INK2)
    ax.set_ylabel("scorer shared memory [KB]")
    ax.set_title("H · what limits the grid", fontsize=10, loc="left")

    # Global log-scale guard. logsafe() only inspects ax.get_lines(), so a
    # panel drawn with bar/scatter (or one whose series are all NaN) can still
    # be left log-scaled with nothing positive on it -- and matplotlib does
    # not complain until savefig, which then discards the ENTIRE figure over
    # one empty panel. Walk every axis and drop log scaling wherever no
    # positive data exists.
    for _ax in fig.get_axes():
        for _axis, _get, _set in (("x", _ax.get_xscale, _ax.set_xscale),
                                  ("y", _ax.get_yscale, _ax.set_yscale)):
            if _get() != "log":
                continue
            lim = _ax.dataLim
            vals = ([lim.x0, lim.x1] if _axis == "x" else [lim.y0, lim.y1])
            if not any(np.isfinite(v) and v > 0 for v in vals):
                _set("linear")

    fig.suptitle("GB likelihood engines — accuracy and speed, one measurement",
                 fontsize=15, color=INK, x=0.05, ha="left", y=0.962)
    fig.text(0.05, 0.922,
             f"{BACKEND} · WDM Nf={Nf}, dt=10 s, XYZ · n_r={nr} fixed "
             f"(tuned = law n~Tobs/yr), K={knots}, half-band={band} · "
             f"batch={nb} · error vs chunked-het "
             f"(validated to 3e-9 against full time-domain truth), "
             f"delta-vs-delta, tiered bar allowed(T)=max(0.1, T/100)",
             fontsize=8.5, color=INK2, ha="left")
    fp = os.path.join(out, "gb_engine_benchmark.png")
    fig.savefig(fp, dpi=155, facecolor="white")
    print(f"\n[out] {fp}")
    print(f"[out] {os.path.join(out, 'gb_engine_benchmark.npz')}")


if __name__ == "__main__":
    main()
