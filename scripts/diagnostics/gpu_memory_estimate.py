#!/usr/bin/env python
"""Static GPU-memory estimator for the WDM GB global fit (Python-side arrays).

Blowups on the GPU come from a few Python-side ``cupy`` allocations whose
size scales with the WDM grid, the active band, and the walker/temp/band
counts. This script computes those sizes from the ACTUAL array shapes in the
code (no GPU, no data load -- pure arithmetic) and sweeps the knobs so you
can size a run before launching it.

Components modeled (all from the real allocation shapes):

1. **AnalysisContainerArray linear buffers** (analysiscontainer.py):
   per cold-chain walker, a packed residual + inverse-PSD buffer.
     data : data_length * nchannels          * n_acs * 16 B (complex128)
     psd  : data_length * prod(shape_sens)    * n_acs *  8 B (float64)
   with data_length = Nf_active * Nt_active, shape_sens = (nch, nch) for XYZ
   (-> factor 9), n_acs = nwalkers.

2. **GB SubBandBuffer** (globalfit/moves/gbbands.py) -- the dominant term.
   Per active (temp, walker, band) *slot* it stores the FULL active WDM grid.
   With the 2026-07-10 memory defaults (tasks a + c):
     data     : nch     * Nf_active * Nt_active * 8 B  (residual, float64)
     invC     : nch*nch * Nf_active * Nt_active * 8 B  (XYZ cross-channel)
     sens_mat : ZERO-STORAGE by default (keep_sens_mat=False) -- the forward
                matrix is a shape-correct broadcast view; only invC feeds the
                kernel. keep_sens_mat=True restores nch*nch*Nf_active*Nt_active*8.
   Optionally a template twin (use_template_arr, RJ/tempering) adds another
   (data + invC) of the same size.

   PEAK sub-band-buffer memory = n_subbands * per_slot ONCE THE CAP BINDS.
   The BandScheduler holds at most ``n_subbands`` slots and swaps finished
   cells into them, so the peak is ``min(n_subbands, n_cells_total) *
   per_slot`` where n_cells_total = ntemps * nwalkers * n_gb_bands. The
   ensemble counts (ntemps/nwalkers/n_gb_bands) DO NOT scale the peak -- they
   only decide (i) whether the cap binds (n_cells_total > n_subbands) and
   (ii) how many scheduler rounds run (wall-clock). per_slot depends ONLY on
   the grid (Nf_active, Nt_active, nchannels). ``n_subbands`` (task a; GB move
   ``num_band_preload`` / ``GB_N_SUBBANDS``, stock default 1024) is therefore
   THE peak-memory lever. (nwalkers does scale the SEPARATE ACA term above.)
   NOTE: each slot still spans the FULL active band Nf_active (the WDM kernel
   uses a single global [ind_min_f, ind_max_f]); per-band layer slicing (task
   b) would cut Nf_active -> ~n_layers+guard but needs a per-slab layer-origin
   kernel change (native CUDA + JAX). Until then Nf_active is the last big
   lever left in this term.

The ``chunked-het comp`` (gbgpu.gbcomps) internal scratch is NOT modeled
here (it lives in GBGPU and scales with Nt_sub * N_sparse * num_bin); it is
usually far smaller than the two terms above but is flagged in the notes.

Run:  python gpu_memory_estimate.py
Knobs are CLI-free; edit CONFIGS below or import estimate() elsewhere.
"""
from __future__ import annotations

import math

GB = 1024.0**3
MB = 1024.0**2


def _layer_df(Nf: int, dt: float) -> float:
    return 1.0 / (2.0 * Nf * dt)


def active_band(
    Nf: int, Nt: int, dt: float, min_freq: float, max_freq: float, edge_crop: int
) -> tuple[int, int]:
    """(Nf_active, Nt_active) for a WDM grid + analysis band + time-edge crop."""
    ldf = _layer_df(Nf, dt)
    ind_min_f = int(math.floor(min_freq / ldf))
    ind_max_f = int(math.ceil(max_freq / ldf))
    ind_max_f = min(ind_max_f, Nf - 1)
    Nf_active = max(1, ind_max_f - ind_min_f + 1)
    Nt_active = max(1, Nt - 2 * edge_crop)
    return Nf_active, Nt_active


def estimate(
    *,
    tobs_days: float,
    dt: float = 2.5,
    wavelet_duration: float = 3600.0,
    data_min_freq: float = 6e-3,
    data_max_freq: float = 25e-3,
    data_band_layers: int | None = None,
    gb_center_freq: float = 7.5e-3,
    edge_crop: int = 58,
    nchannels: int = 3,
    nwalkers: int = 4,
    ntemps: int = 2,
    n_gb_bands: int = 1,
    tdi_xyz: bool = True,
    n_subbands: int = 1024,  # stock GBSettings default (GB_N_SUBBANDS)
    keep_sens_mat: bool = False,
    use_template_arr: bool = True,
) -> dict:
    """Estimate the dominant Python-side GPU allocations (bytes) for a WDM GB run."""
    Nf = int(round(wavelet_duration / dt))
    Nt = int(round((tobs_days * 86400.0) / (Nf * dt)))
    Nt += Nt % 2  # even
    ldf = _layer_df(Nf, dt)

    # DATA_BAND_LAYERS clips the analysis band to +-N layers around the GB band.
    if data_band_layers is not None:
        k_center = int(math.floor(gb_center_freq / ldf))
        data_min_freq = max(data_min_freq, (k_center - data_band_layers) * ldf)
        data_max_freq = min(data_max_freq, (k_center + 1 + data_band_layers) * ldf)

    Nf_active, Nt_active = active_band(
        Nf, Nt, dt, data_min_freq, data_max_freq, edge_crop
    )
    data_length = Nf_active * Nt_active
    sens_factor = nchannels * nchannels if tdi_xyz else nchannels

    # 1. ACA linear buffers (per cold-chain walker)
    n_acs = nwalkers
    aca_data = data_length * nchannels * n_acs * 16  # complex128
    aca_psd = data_length * sens_factor * n_acs * 8  # float64
    aca_total = aca_data + aca_psd

    # 2. GB SubBandBuffer (per resident temp*walker*band SLOT; full active grid).
    #    num_bands_now is capped by n_subbands (the BandScheduler slot count /
    #    GB move num_band_preload): finished cells swap out for pending ones,
    #    so peak buffer memory scales with the cap, not the total cell count.
    n_cells_total = ntemps * nwalkers * n_gb_bands
    num_bands_now = min(int(n_subbands), n_cells_total)
    per_band_data = nchannels * data_length * 8
    per_band_invC = sens_factor * data_length * 8
    # Forward sensitivity matrix: zero-storage broadcast view by default
    # (keep_sens_mat=False); a real full array only when the caller keeps it.
    per_band_sens = sens_factor * data_length * 8 if keep_sens_mat else 0
    # Template twin (RJ / tempering) mirrors the residual + invC layout.
    per_band_template = (per_band_data + per_band_invC) if use_template_arr else 0
    per_band = per_band_data + per_band_invC + per_band_sens + per_band_template
    buffer_total = num_bands_now * per_band

    total = aca_total + buffer_total
    return dict(
        Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
        data_length=data_length, num_bands_now=num_bands_now,
        n_cells_total=n_cells_total, n_subbands=int(n_subbands),
        keep_sens_mat=keep_sens_mat, use_template_arr=use_template_arr,
        layer_df=ldf, data_band_hz=(data_min_freq, data_max_freq),
        aca_data=aca_data, aca_psd=aca_psd, aca_total=aca_total,
        per_band=per_band, buffer_total=buffer_total, total=total,
    )


def _fmt(nbytes: float) -> str:
    return f"{nbytes / GB:8.3f} GB" if nbytes >= GB else f"{nbytes / MB:8.1f} MB"


# All configs below use the 2026-07-10 memory defaults: invC-only per-band
# storage (keep_sens_mat=False) and the n_subbands peak cap (task a + c). The
# OLD-default rows (keep_sens_mat=True, n_subbands=20000 = uncapped) are shown
# alongside the big-run cases so the win is visible.
CONFIGS = [
    ("gb_no_fg default (90d, band 6-25mHz, 4w/2t, 1 band)",
     dict(tobs_days=90, nwalkers=4, ntemps=2, n_gb_bands=1)),
    ("+ DATA_BAND_LAYERS=10 (clip to GB band)",
     dict(tobs_days=90, nwalkers=4, ntemps=2, n_gb_bands=1, data_band_layers=10)),
    ("bigger 30w/10t, 20 GB bands -- OLD defaults (sens_mat kept, uncapped)",
     dict(tobs_days=90, nwalkers=30, ntemps=10, n_gb_bands=20,
          keep_sens_mat=True, n_subbands=20000)),
    ("bigger 30w/10t, 20 GB bands -- NEW defaults (invC-only, uncapped)",
     dict(tobs_days=90, nwalkers=30, ntemps=10, n_gb_bands=20)),
    ("bigger 30w/10t, 20 GB bands -- NEW + n_subbands=512 cap",
     dict(tobs_days=90, nwalkers=30, ntemps=10, n_gb_bands=20, n_subbands=512)),
    ("1 year, full band 0.1-25mHz, 30w/10t, 100 bands -- OLD defaults",
     dict(tobs_days=365.25, data_min_freq=1e-4, data_max_freq=25e-3,
          nwalkers=30, ntemps=10, n_gb_bands=100,
          keep_sens_mat=True, n_subbands=20000)),
    ("1 year, full band, 30w/10t, 100 bands -- NEW defaults + n_subbands=512",
     dict(tobs_days=365.25, data_min_freq=1e-4, data_max_freq=25e-3,
          nwalkers=30, ntemps=10, n_gb_bands=100, n_subbands=512)),
]


def main():
    print("=" * 100)
    print("WDM GB global-fit GPU-memory estimate (Python-side cupy allocations)")
    print("=" * 100)
    for label, kw in CONFIGS:
        r = estimate(**kw)
        _cap = ("capped" if r['num_bands_now'] < r['n_cells_total'] else "uncapped")
        _sens = ("sens_mat KEPT" if r['keep_sens_mat'] else "invC-only")
        print(f"\n### {label}")
        print(f"    grid Nf={r['Nf']} Nt={r['Nt']} | active {r['Nf_active']}x{r['Nt_active']} "
              f"= {r['data_length']:,} px | band {r['data_band_hz'][0]*1e3:.2f}-"
              f"{r['data_band_hz'][1]*1e3:.2f} mHz")
        print(f"    slots={r['num_bands_now']:,} of {r['n_cells_total']:,} cells "
              f"({_cap}, n_subbands={r['n_subbands']:,}) | {_sens}"
              f"{' | +template twin' if r['use_template_arr'] else ''}")
        print(f"    ACA buffers (per-walker residual+invC) : {_fmt(r['aca_total'])}"
              f"   [data {_fmt(r['aca_data'])} + psd {_fmt(r['aca_psd'])}]")
        print(f"    GB SubBandBuffer (x slots)             : {_fmt(r['buffer_total'])}"
              f"   [{_fmt(r['per_band'])}/slot]")
        print(f"    >>> TOTAL (these two terms)            : {_fmt(r['total'])}")
    print("\n" + "=" * 100)
    print("DOMINANT TERM: the GB SubBandBuffer = num_bands_now (<= n_subbands) x (full active")
    print("grid: residual + 9x cross-channel invC [+ template twin]). Scales as")
    print("min(n_subbands, ntemps*nwalkers*n_gb_bands) * Nf_active*Nt_active.")
    print("")
    print("Defaults as of 2026-07-10 (tasks a + c, now shipped):")
    print("  * n_subbands (GB move num_band_preload / env GB_N_SUBBANDS) is a STOCK knob on")
    print("      GBSettings. It caps resident (temp,walker,band) SLOTS -> the peak-memory")
    print("      lever. Default 20000 (uncapped); lower it to bound peak GPU memory.")
    print("  * keep_sens_mat=False is the SubBandBuffer default: only invC is stored per")
    print("      band (the kernel input); the forward sens_mat is a zero-storage view.")
    print("      Cuts the per-band cross-channel term ~2x. Flip keep_sens_mat=True to keep it.")
    print("")
    print("Other levers you have NOW (no code change):")
    print("  * DATA_BAND_LAYERS=N   clip the analysis band + per-walker ACA slab to +-N WDM")
    print("      layers around the GB band -> cuts Nf_active (e.g. 138 -> 22 = 6x).")
    print("  * nwalkers * ntemps * n_gb_bands   drive n_cells_total; the n_subbands cap")
    print("      bounds the peak regardless.")
    print("")
    print("Remaining code-level win (task b -- NATIVE kernel change, GPU build required):")
    print("  * PER-BAND LAYER SLICING: each slot still spans the FULL Nf_active grid, but a")
    print("      real GB band spans ~n_layers+guard (<=~10) layers. Slicing Nf_active -> ~10")
    print("      cuts the buffer by Nf_active/10 (10-18x). BLOCKED: the shared GBWDMComputations")
    print("      writes each layer at (m - GLOBAL ind_min_f), so per-band slabs need the")
    print("      chunked-het kernels (gb_wdm_het_{get_ll,swap_ll,fill_global}, CUDA + JAX) to")
    print("      take a per-slab layer origin. Cannot be done Python-side.")


if __name__ == "__main__":
    main()
