"""Direct rj_fstat_centers kernel timing at production scale: old vs new.

Basic interactive-node benchmark: reproduces the shape of the centers
workload -- ``GBWDMComputations.get_fstat_ll_wdm`` over hundreds of
thousands of rows in ``GB_FSTAT_CTR_BATCH``-sized batches against a
band slab -- and times three arms on ONE binary:

  prefold     ``fstat_fold=0``: bit-for-bit the pre-2026-08-28 kernel
              (4 independent basis generations). The "old setup".
  fold        ``fstat_fold=1``, orbit cache OFF: the folded path, which on
              this binary also carries the 09-01 per-m fold completion,
              invC hoist and e = C w factorization.
  fold_cache  ``fstat_fold=1`` + ``N_cp_orbit`` from the production stock
              default: the full "new setup" as production runs it.

The v7-era production reference (fold WITHOUT the 09-01 batch) measured a
flat 0.288 ms/row; that state is not reachable from this binary, so use
that number as the middle-era yardstick when reading the table.

Config comes from the INSTALLED stock variant (``erebor.gb_no_fg``
pre-build -- cheap, no data load), so the WDM grid (Nf/Nt/dt), the comp
constructor knobs (Nt_sub/n_pad/N_sparse/N_cp_*) and their env overrides
(TOBS_TARGET, DT, CHUNKED_*) track production automatically.

Run (interactive GPU node; CPU works too, just slow -- shrink ROWS):

    python bench_fstat_centers_scale.py

Env:
  ROWS       total rows per arm (default 500000 -- the per-row rate is flat,
             this is plenty; ROWS=3700000 reproduces the literal v7
             full-propose scale, ~18 min at the old rate)
  BATCH      rows per kernel call (default = GB_FSTAT_CTR_BATCH default 4096)
  ARMS       comma list from {prefold,fold,fold_cache} (default all three)
  BACKEND    force_backend (default: gpu if available, else cpu)
  GB_MIN_FREQ / GB_MAX_FREQ
             f0 band for the slab + rows (default 3e-4 .. 3e-2, the full
             GB band; the default gb_no_fg band is a narrow dev band)
  SEED       row RNG seed (default 20260902)
"""
import os
import time

import numpy as np

import lisatools
from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.globalfit.stock import erebor
from lisatools.utils.utility import get_array_module

import gbgpu
from gbgpu.gbcomps import GBWDMComputations


class _OneSlotHolder:
    """Minimal wdm_holder: one (data slab, invC slab) slot.

    Same interface as the fold test's ``_TwoSlotHolder``: contiguity goes
    through the slabs' OWN array module so the slot arrays stay on-device.
    """

    def __init__(self, data_slab, invC_slab):
        xp = get_array_module(data_slab)
        self.linear_data_arr = [xp.ascontiguousarray(data_slab).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(invC_slab).ravel()]

    def __len__(self):
        return 1


def _sync(xp):
    if hasattr(xp, "cuda"):
        xp.cuda.get_current_stream().synchronize()


def main():
    rows_total = int(float(os.environ.get("ROWS", "500000")))
    batch = int(os.environ.get("BATCH",
                               os.environ.get("GB_FSTAT_CTR_BATCH", "4096")))
    arms = [a.strip() for a in os.environ.get(
        "ARMS", "prefold,fold,fold_cache").split(",") if a.strip()]
    backend = os.environ.get("BACKEND", "")
    if not backend:
        backend = "cpu"
        for cand in ("cuda12x", "cuda13x", "cuda11x"):
            if lisatools.has_backend(cand):
                backend = cand
                break
    f_lo = float(os.environ.get("GB_MIN_FREQ", "3e-4"))
    f_hi = float(os.environ.get("GB_MAX_FREQ", "3e-2"))
    rng = np.random.default_rng(int(os.environ.get("SEED", "20260902")))

    # ---- production-shaped config from the installed stock variant --------
    fit = erebor.gb_no_fg(nwalkers=4)          # pre-build: validation only
    Nf, Nt, _wavelet, Tobs = fit.wdm_grid
    dt = float(fit.general.dt)
    gb = fit.gb
    n_cp_orbit_prod = int(gb.n_cp_orbit)
    print(f"gbgpu {gbgpu.__version__}  lisatools {lisatools.__version__}")
    print(f"backend {backend}  grid Nf={Nf} Nt={Nt} dt={dt} Tobs={Tobs:.4g}")
    print(f"comp Nt_sub={gb.nt_sub} n_pad={gb.n_pad} N_sparse={gb.n_sparse} "
          f"N_cp_sig={gb.n_cp_sig} N_cp_orbit(prod)={n_cp_orbit_prod}")
    print(f"rows {rows_total} x batch {batch}  band {f_lo:.4g}-{f_hi:.4g} Hz "
          f"arms {arms}")

    t_start = 10000.0
    edge = 40
    wdm = WDMSettings(
        int(Nf), int(Nt), dt, t0=t_start,
        min_freq=f_lo, max_freq=f_hi,
        min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
        force_backend=backend,
    )
    comp = GBWDMComputations(
        wdm, t_ref=t_start,
        Nt_sub=int(gb.nt_sub), n_pad=int(gb.n_pad),
        N_sparse=int(gb.n_sparse),
        N_cp_sig=int(gb.n_cp_sig), N_cp_orbit=n_cp_orbit_prod,
        orbits=ESAOrbits(force_backend=backend),
        tdi_config="2nd generation",
        force_backend=backend, tdi_type="XYZ",
    )
    comp.convert_to_ra_dec = False
    xp = comp.xp

    # ---- one band slab: a few hundred injected sources + diag-ones invC ---
    def draw_rows(n):
        p = np.empty((n, 9))
        p[:, 0] = 10.0 ** rng.uniform(-23, -21, n)            # A
        p[:, 1] = rng.uniform(f_lo * 1.02, f_hi * 0.98, n)    # f0
        p[:, 2] = 1e-17 * (p[:, 1] / 3e-3) ** (11.0 / 3.0)    # fdot (GR-ish)
        p[:, 3] = 0.0                                         # fddot
        p[:, 4] = rng.uniform(0, 2 * np.pi, n)                # phi0
        p[:, 5] = np.arccos(rng.uniform(-1, 1, n))            # iota
        p[:, 6] = rng.uniform(0, np.pi, n)                    # psi
        p[:, 7] = rng.uniform(0, 2 * np.pi, n)                # lam
        p[:, 8] = np.arcsin(rng.uniform(-1, 1, n))            # beta
        return p

    n_inj = int(os.environ.get("INJ", "256"))
    print(f"filling the data slab ({n_inj} injected sources) ...", flush=True)
    h = xp.zeros((3, int(Nf), int(Nt)))
    comp.fill_global_wdm(draw_rows(n_inj), h, convert_to_ra_dec=False)
    ilo, ihi = wdm.ind_min_f, wdm.ind_max_f + 1
    h_act = xp.ascontiguousarray(h[:, ilo:ihi, wdm.active_slice_t])
    del h
    nch, nfa, nta = h_act.shape
    invC = xp.zeros((nch, nch, nfa, nta))
    for c in range(nch):
        invC[c, c] = 1.0
    holder = _OneSlotHolder(h_act, invC)
    print(f"slab {nch}x{nfa}x{nta} "
          f"({(h_act.nbytes + invC.nbytes) / 1e6:.0f} MB)", flush=True)

    params = draw_rows(rows_total)
    di = np.zeros(batch, dtype=np.int32)

    # arm -> (fstat_fold, N_cp_orbit passed to the kernel)
    ARM = {
        "prefold":    (0, 0),
        "fold":       (1, 0),
        "fold_cache": (1, n_cp_orbit_prod),
    }
    results, first_NM = {}, {}
    for arm in arms:
        fold, n_cp = ARM[arm]
        comp.N_cp_orbit = n_cp
        comp.fstat_orbit_cache = 1 if n_cp > 0 else 0

        # warmup (JIT/alloc) -- 2 batches, untimed
        for k in range(2):
            b = params[k * batch:(k + 1) * batch]
            comp.get_fstat_ll_wdm(b, holder, data_index=di[:len(b)],
                                  noise_index=di[:len(b)],
                                  convert_to_ra_dec=False, fstat_fold=fold)
        _sync(xp)

        done, t0 = 0, time.perf_counter()
        while done < rows_total:
            b = params[done:done + batch]
            N, M = comp.get_fstat_ll_wdm(
                b, holder, data_index=di[:len(b)], noise_index=di[:len(b)],
                convert_to_ra_dec=False, fstat_fold=fold)
            if done == 0:
                first_NM[arm] = (np.asarray(N.get() if hasattr(N, "get")
                                            else N),
                                 np.asarray(M.get() if hasattr(M, "get")
                                            else M))
            done += len(b)
            if done % (batch * 64) == 0:
                _sync(xp)
                r = (time.perf_counter() - t0) / done * 1e3
                print(f"  [{arm}] {done}/{rows_total}  {r:.4f} ms/row",
                      flush=True)
        _sync(xp)
        el = time.perf_counter() - t0
        results[arm] = el
        print(f"[{arm:10s}] {rows_total} rows in {el:8.1f} s  "
              f"= {el / rows_total * 1e3:.4f} ms/row", flush=True)

    # ---- table + cross-arm value sanity -----------------------------------
    print("\n==== summary " + "=" * 47)
    base = results.get("prefold")
    for arm in arms:
        el = results[arm]
        rate = el / rows_total * 1e3
        vs = f"  {base / el:5.2f}x vs prefold" if base and arm != "prefold" \
             else ""
        print(f"  {arm:10s} {rate:8.4f} ms/row   {el:8.1f} s{vs}")
    print("  (v7 production reference, fold w/o the 09-01 batch: "
          "0.288 ms/row)")
    if len(first_NM) > 1:
        ref_arm = arms[0]
        Nr, Mr = first_NM[ref_arm]
        for arm in arms[1:]:
            Na, Ma = first_NM[arm]
            dN = np.max(np.abs(Na - Nr)) / max(np.max(np.abs(Nr)), 1e-300)
            dM = np.max(np.abs(Ma - Mr)) / max(np.max(np.abs(Mr)), 1e-300)
            same = (Na == Nr).all() and (Ma == Mr).all()
            note = "  ⚠ BITWISE-IDENTICAL (arm knob not engaged?)" if same \
                   else ""
            print(f"  values {arm} vs {ref_arm}: max rel dN {dN:.2e} "
                  f"dM {dM:.2e}{note}")


if __name__ == "__main__":
    main()
