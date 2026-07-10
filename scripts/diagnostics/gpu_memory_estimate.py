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
   Per active (temp, walker, band) cell it stores the FULL active WDM grid
   FOUR times (band data + template twin + sens_mat + invC):
     data     : nch          * Nf_active * Nt_active * 8 B  (residual, float64)
     template : nch          * Nf_active * Nt_active * 8 B  (template twin)
     sens_mat : nch*nch      * Nf_active * Nt_active * 8 B  (XYZ cross-channel)
     invC     : nch*nch      * Nf_active * Nt_active * 8 B
   times num_bands_now = ntemps * nwalkers * n_gb_bands.
   NOTE: the per-band full-active-grid storage is a documented first-cut
   (the WDM kernel uses a single global [ind_min_f, ind_max_f]); per-band
   layer slicing would cut Nf_active -> ~5 here. Until then this term is the
   one that explodes.

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

    # 2. GB SubBandBuffer (per temp*walker*band cell; full active grid x4)
    num_bands_now = ntemps * nwalkers * n_gb_bands
    per_band_data = nchannels * data_length * 8
    per_band_template = nchannels * data_length * 8
    per_band_sens = sens_factor * data_length * 8
    per_band_invC = sens_factor * data_length * 8
    per_band = per_band_data + per_band_template + per_band_sens + per_band_invC
    buffer_total = num_bands_now * per_band

    total = aca_total + buffer_total
    return dict(
        Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
        data_length=data_length, num_bands_now=num_bands_now,
        layer_df=ldf, data_band_hz=(data_min_freq, data_max_freq),
        aca_data=aca_data, aca_psd=aca_psd, aca_total=aca_total,
        per_band=per_band, buffer_total=buffer_total, total=total,
    )


def _fmt(nbytes: float) -> str:
    return f"{nbytes / GB:8.3f} GB" if nbytes >= GB else f"{nbytes / MB:8.1f} MB"


CONFIGS = [
    ("gb_no_fg default (90d, band 6-25mHz, 4w/2t, 1 band)",
     dict(tobs_days=90, nwalkers=4, ntemps=2, n_gb_bands=1)),
    ("+ DATA_BAND_LAYERS=10 (clip to GB band)",
     dict(tobs_days=90, nwalkers=4, ntemps=2, n_gb_bands=1, data_band_layers=10)),
    ("bigger walkers/temps (30w/10t, 1 band)",
     dict(tobs_days=90, nwalkers=30, ntemps=10, n_gb_bands=1)),
    ("bigger walkers/temps + 20 GB bands",
     dict(tobs_days=90, nwalkers=30, ntemps=10, n_gb_bands=20)),
    ("1 year, full band 0.1-25mHz, 30w/10t, 100 bands",
     dict(tobs_days=365.25, data_min_freq=1e-4, data_max_freq=25e-3,
          nwalkers=30, ntemps=10, n_gb_bands=100)),
    ("1 year, full band, 30w/10t, 100 bands, DATA_BAND_LAYERS=10",
     dict(tobs_days=365.25, data_min_freq=1e-4, data_max_freq=25e-3,
          nwalkers=30, ntemps=10, n_gb_bands=100, data_band_layers=10)),
]


def main():
    print("=" * 100)
    print("WDM GB global-fit GPU-memory estimate (Python-side cupy allocations)")
    print("=" * 100)
    for label, kw in CONFIGS:
        r = estimate(**kw)
        print(f"\n### {label}")
        print(f"    grid Nf={r['Nf']} Nt={r['Nt']} | active {r['Nf_active']}x{r['Nt_active']} "
              f"= {r['data_length']:,} px | band {r['data_band_hz'][0]*1e3:.2f}-"
              f"{r['data_band_hz'][1]*1e3:.2f} mHz | num_bands_now={r['num_bands_now']:,}")
        print(f"    ACA buffers (per-walker residual+invC) : {_fmt(r['aca_total'])}"
              f"   [data {_fmt(r['aca_data'])} + psd {_fmt(r['aca_psd'])}]")
        print(f"    GB SubBandBuffer (x num_bands_now)      : {_fmt(r['buffer_total'])}"
              f"   [{_fmt(r['per_band'])}/band]")
        print(f"    >>> TOTAL (these two terms)            : {_fmt(r['total'])}")
    print("\n" + "=" * 100)
    print("DOMINANT TERM: the GB SubBandBuffer = num_bands_now x (full active grid, 4 copies,")
    print("9x cross-channel invC). Scales as ntemps*nwalkers*n_gb_bands * Nf_active*Nt_active.")
    print("")
    print("Levers you have NOW (largest effect first):")
    print("  * DATA_BAND_LAYERS=N   clip the analysis band to +-N WDM layers around the GB")
    print("      band -> cuts Nf_active (e.g. 138 -> 22 = 6x). The single biggest knob.")
    print("  * n_subbands / num_band_preload  caps how many band cells are buffered per")
    print("      get_buffer call (GB move default 20000 -- far above any real cell count;")
    print("      lower it to bound the peak). Not yet a stock knob.")
    print("  * nwalkers * ntemps    direct multiplier on num_bands_now.")
    print("  * fewer GB bands per run (focused-band runs) -> smaller n_gb_bands.")
    print("")
    print("Code-level fixes (bigger wins, need a change to gbbands.py):")
    print("  * PER-BAND LAYER SLICING: each band stores the FULL Nf_active grid; a real GB")
    print("      band spans ~n_layers+guard (<=~10) layers. Slicing Nf_active -> ~10 cuts the")
    print("      buffer by Nf_active/10 (10-18x here). This is the documented first-cut TODO.")
    print("  * sens_mat AND invC are both allocated per band (2x the 9x cross-channel term);")
    print("      if invC is derived from sens_mat, only one need persist.")


if __name__ == "__main__":
    main()
