"""Sig-het anchor h_h probe at PRODUCTION shapes -- the corruption's repro.

THE ROOT CAUSE THIS PROBE FOUND (2026-08-19, closing the v4 dissect
capture-replay chain):

* The in-run anchor corruption is CONFINED to h_h: across 171k scored
  sources the d_h ratio (sig-het/exact) is 0.99-1.06 in EVERY corruption
  class while h_h inflates up to 27x, uncorrelated with d_h.
* Mechanism (proved here on CPU, n=64, no GPU involved): the
  make_reference spline reconstruction (``n_cp_build`` control points)
  reproduces each channel to ~0.1%, but the per-channel errors are
  INCOHERENT across X/Y/Z -- they do not preserve the GW template's
  X+Y+Z null cancellation (true null power ~1e-10 of total; recon leaves
  ~1e-5). The near-singular low-f XYZ invC amplifies exactly that
  direction (null eigenvalue 54-7500x the differential ones), so the
  quadratic h_h inflates while the linear d_h stays clean (the residual
  lies in the non-null subspace). Worst for edge-on sources at low f --
  the production offender population exactly.
* Verified fix: n_cp_build 32 -> 256 cuts the null-direction excess
  3e5x -> 148x and the scored anchor error from max |log ratio| 0.31 to
  1.5e-4, at +2.6% setup cost. AUTO now resolves to 256 at 3 months
  (SIGHET_N_CP_SPACING default 0.35 days, GBGPU _resolve_n_cp).

What it does (CPU ~1 min at N_SRC=64; GPU ~2 min at 2048):

1. Builds the production-config engine (grid/knobs from the v4 dissect
   config string) on ``PROBE_BACKEND`` (default cuda12x).
2. N_SRC sources with edge-on-heavy inclinations, low-f heavy f0s, and
   per-slot invC scaled over +-2 decades (the per-walker spread measured
   in the RAW captures).
3. Zero data slabs (h_h is data-independent).
4. Scores every source at ITS OWN anchor twice:
   sig-het (setup_in_model + get_ll) vs chunked task-b (clear + get_ll),
   and reports the per-source hh ratio distribution.

Env: PROBE_BACKEND (cuda12x; cpu reproduces), N_SRC (2048), SEED (7),
N_CP_BUILD (32 shows the corruption, 256 the fix), CHUNK_OVERRIDE.

Run:  N_CP_BUILD=32 python scripts/gb_chunked_het/gb_sighet_bfold_gpu_probe.py
      N_CP_BUILD=256 python scripts/gb_chunked_het/gb_sighet_bfold_gpu_probe.py
"""
import os
import time

import numpy as np

from lisatools.detector import DefaultOrbits
from lisatools.domains import WDMSettings
from lisatools.sensitivity import get_sensitivity, X2TDISens, XY2TDISens
from lisatools import detector as lisa_models
from lisatools.stochastic import HyperbolicTangentGalacticForeground as HTGF

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import make_band_likelihood_engine

GB_MOJITO_T_REF = 97729089.327664
PSD_P = [1.52274518e-11, 2.74762992e-15]
GAL_P = (3.74443460e-44, 5.29868210e-02, 9.26921788e-01, 2.75873152e-03,
         5.70634746e+03)


class Holder:
    def __init__(self, data, invc, xp, slab_min_f, band_slab_Nf):
        self.linear_data_arr = [xp.ascontiguousarray(
            xp.asarray(data, dtype=xp.float64)).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(
            xp.asarray(invc, dtype=xp.float64)).ravel()]
        self.band_slab_Nf = int(band_slab_Nf)
        self.slab_min_f = xp.asarray(np.asarray(slab_min_f, dtype=np.int32))
        self.min_freq_inds = self.slab_min_f
        self._n = int(len(data))

    def __len__(self):
        return self._n


def main():
    backend = os.environ.get("PROBE_BACKEND", "cuda12x")
    n = int(os.environ.get("N_SRC", "2048"))
    seed = int(os.environ.get("SEED", "7"))
    rng = np.random.default_rng(seed)

    Nf, Nt, dt = 1440, 2160, 2.5
    wd = Nf * dt
    wdm = WDMSettings(Nf, Nt, dt, t0=97729939.827664,
                      min_freq=1e-4, max_freq=2.5e-2,
                      min_time=20 * wd, max_time=(Nt - 20) * wd,
                      force_backend=backend)
    orbits = DefaultOrbits(force_backend=backend, frame="icrs")
    chunked = GBWDMComputations(
        wdm, t_ref=GB_MOJITO_T_REF, Nt_sub=256, n_pad=32, N_sparse=256,
        N_cp_sig=48, N_cp_orbit=32, orbits=orbits,
        tdi_config="2nd generation", force_backend=backend,
        d_d=0.0, tdi_type="XYZ")
    xp = chunked.xp

    layer_df = wdm.layer_df
    F = wdm.ind_max_f - wdm.ind_min_f + 1
    T = wdm.ind_max_t - wdm.ind_min_t + 1
    W = 5

    # ---- sources: low-f heavy, edge-on heavy (the corrupt population) ----
    f0 = 10 ** rng.uniform(np.log10(7.5e-4), np.log10(3.2e-3), n)
    cosi = rng.uniform(-1, 1, n)
    cosi[: n // 2] = rng.uniform(-0.2, 0.2, n // 2)     # edge-on half
    params = np.column_stack([
        10 ** rng.uniform(-23, -21, n),                  # amp
        f0,
        10 ** rng.uniform(-18, -16, n),                  # fdot
        np.zeros(n),
        rng.uniform(0, 2 * np.pi, n),                    # phi0
        np.arccos(cosi),                                 # iota
        rng.uniform(0, np.pi, n),                        # psi
        rng.uniform(0, 2 * np.pi, n),                    # alpha
        rng.uniform(-np.pi / 2, np.pi / 2, n),           # delta
    ])

    # ---- per-slot slabs: zero data; XYZ CSD invC with per-slot scatter ---
    m_carrier = np.floor(f0 / layer_df).astype(int)
    slab_lo_abs = np.clip(m_carrier - W // 2, wdm.ind_min_f,
                          wdm.ind_min_f + F - W).astype(np.int32)
    model = lisa_models.LISAModel(PSD_P[0] ** 2, PSD_P[1] ** 2,
                                  lisa_models.DefaultOrbits(), "probe")
    kw = dict(model=model, stochastic_params=tuple(GAL_P),
              stochastic_function=HTGF)
    m_abs = np.arange(wdm.ind_min_f, wdm.ind_max_f + 1)
    f_rows = np.maximum(m_abs * layer_df, 1e-5)
    Cxx = np.asarray(get_sensitivity(f_rows, sens_fn=X2TDISens, **kw), float)
    Cxy = np.asarray(get_sensitivity(f_rows, sens_fn=XY2TDISens, **kw), float)
    C = np.empty((F, 3, 3))
    for a in range(3):
        for b in range(3):
            C[:, a, b] = Cxx if a == b else Cxy
    invC_rows = np.linalg.inv(C)                                   # (F, 3, 3)
    scale = 10 ** rng.uniform(-2, 2, n)                            # walker spread
    invc = np.empty((n, 3, 3, W, T))
    for i in range(n):
        lo = slab_lo_abs[i] - wdm.ind_min_f
        invc[i] = (invC_rows[lo:lo + W].transpose(1, 2, 0)[:, :, :, None]
                   * scale[i])
    data = np.zeros((n, 3, W, T))
    holder = Holder(data, invc, xp, slab_lo_abs, W)
    idx = np.arange(n)

    n_cp = int(os.environ.get("N_CP_BUILD", "32"))
    sig = GBSignalHetComputations.for_band_engine(
        chunked, nt_layer=60, n_sparse_fd=1024, m_active_half_width=2,
        max_r=0.0, n_cp_build=n_cp, v3_n_nodes=64, v4_knots=128, v4_band=16,
        v5=1, tukey_alpha=0.01)
    eng = make_band_likelihood_engine(
        wdm, gb_wdm_comp=sig, nchannels=3, tdi_channel_setup="XYZ")

    co = os.environ.get("CHUNK_OVERRIDE", "")
    if co.strip():
        import gbgpu.gbsignalhetcomputations as _m
        _m._SIGHET_FOLD_MAX_BYTES = int(co) * 2 * 9 * W * T * 16
        print(f"[probe] fold chunk forced to ~{co} sources")

    def _np(a):
        return np.asarray(a.get() if hasattr(a, "get") else a)

    print(f"[probe] backend={backend} n={n} "
          f"fold_host={os.environ.get('GB_SIGHET_FOLD_HOST', '0')}",
          flush=True)
    t0 = time.perf_counter()
    eng.setup_in_model(holder, params, idx)
    t1 = time.perf_counter()
    eng.get_ll(holder, params, data_index=idx, noise_index=idx,
               N_vals=np.full(n, 1024), waveform_kwargs={})
    hh_sig = _np(eng.h_h_out).real.ravel()[:n].copy()
    eng.clear_in_model()
    t2 = time.perf_counter()
    # exact side: chunked task-b kernel (== direct einsum, verified)
    eng.get_ll(holder, params, data_index=idx, noise_index=idx,
               N_vals=np.full(n, 1024), waveform_kwargs={})
    hh_ex = _np(eng.h_h_out).real.ravel()[:n].copy()
    t3 = time.perf_counter()

    ok = hh_ex > 0
    r = hh_sig[ok] / hh_ex[ok]
    q = np.percentile(r, [0, 1, 50, 99, 100])
    print(f"[probe] setup {t1-t0:.1f}s  sighet-score {t2-t1:.1f}s  "
          f"exact-score {t3-t2:.1f}s")
    print(f"[probe] hh ratio sig/exact over {ok.sum()} sources:")
    print(f"        min={q[0]:.6f} p1={q[1]:.6f} MEDIAN={q[2]:.6f} "
          f"p99={q[3]:.6f} max={q[4]:.6f}")
    worst = np.argsort(np.abs(np.log(r)))[::-1][:8]
    iok = np.where(ok)[0]
    for j in worst:
        i = iok[j]
        print(f"        f0={f0[i]*1e3:7.4f} mHz iota={params[i,5]:5.3f} "
              f"invc_scale={scale[i]:8.3f} ratio={r[j]:10.4f}")
    bad = np.abs(np.log(r)).max()
    verdict = ("FAIL -- reconstruction null-coherence error corrupts h_h"
               if bad > 0.01 else
               "PASS -- anchor h_h exact at production shapes")
    print(f"[probe] VERDICT: {verdict} (max |log ratio| = {bad:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
