# WDM Spline kernel — Python reference + C++/CUDA port notes

## What's here today

This session delivered a **validated Python reference implementation** of
the layer-by-layer WDM transform for spline-based TDI-on-the-fly
waveforms, plus a placeholder C++/CUDA header that lays out the kernel
APIs for the future port.

Files (in `/Users/mkatz/Research/lisa_sprint_2026`):

- [`wdm_spline_helpers.py`](wdm_spline_helpers.py) — Python reference.
  Bluestein FFT for general-even N, 2-stage row-column TD→FD FFT,
  layer-by-layer windowed-IFFT with `(m+n)`-parity extraction on the
  **global** n grid, narrow-window per-(binary, layer) hooks, and
  `fill_global` / `get_ll` / `swap_ll` finishers.
- [`test_fd_spline_wdm.py`](test_fd_spline_wdm.py) — end-to-end test.
  Run with:
  ```sh
  /Users/mkatz/miniconda3/envs/deving/bin/python test_fd_spline_wdm.py
  ```
- [`lisa-on-gpu/src/fastlisaresponse/cutils/WDMSplineHelpers.hh`](lisa-on-gpu/src/fastlisaresponse/cutils/WDMSplineHelpers.hh)
  — header skeleton for the C++/CUDA port (Bluestein and layer-extract
  device routines, kernel-mode enum, host-wrap declarations).  Compiles
  cleanly as a stand-alone header but not wired into the build yet.

## Test result (verified this session)

All 8 cases pass at machine precision against
`lisatools.domains.FDSignal.wdmtransform`:

| Test | Largest rel-err |
| --- | --- |
| Bluestein FFT vs numpy.fft, N ∈ {6, 12, 20, 24, 48, 64} | 3.2e-13 (fwd), 5e-15 (inv) |
| 2-stage row-column FFT vs numpy.fft, (Nf, Nt) ∈ power-of-2 and non-power-of-2 | 8e-15 |
| **Full WDM transform vs lisatools** (Nf, Nt) ∈ {(16,32), (32,64), (16,24)} | **4.7e-15** |
| `fill_global` with signed factors (add - remove) | 0.0 |
| `get_ll` diagonal-noise inner product | 0.0 |
| `swap_ll` five accumulators | 0.0 |
| Narrow-window `(m+n)` parity-flip handling, both parities | 0.0 |
| Narrow == Full at same width / same centre | 0.0 |

## Algorithmic spec (final, validated)

### Bluestein FFT for general even N

- The exact N-point DFT.  The internal padding to a length-M ≥ 2N−1
  power of 2 is **invisible to callers** — the output is the full
  N-point spectrum (verified element-wise against `numpy.fft.fft`).
- Same `chirp_n` and `chirp_M_fft` arrays serve forward and inverse —
  the inverse just conjugates them and divides by N at the end.
- For tiny N ≤ 16 the port may want to fall back to a direct DFT (the
  Bluestein overhead becomes large), but the algorithm is correct at
  any N ≥ 2.

### 2-stage row-column TD→FD FFT (size N = Nf × Nt)

- Layout: `n = a * Nt + b`, with `a ∈ [0, Nf), b ∈ [0, Nt)`.
- Stage A: Nt FFTs of size Nf along the `a` axis (one per column `b`).
- Twiddle `S[a, b] *= exp(-i 2π a b / N)`.
- Stage B: Nf FFTs of size Nt along the `b` axis.
- Output: `X[k_f + Nf * k_t] = fd_2d[k_f, k_t]`.
- This is what the C++ kernel does in shared memory, one row/column at
  a time, with the (Nf × Nt) global scratch as the only per-binary
  intermediate.

### Per-layer WDM extraction

For each layer `m ∈ [0, Nf]` (inclusive of the Nyquist sentinel m=Nf):

1. Gather `Nt_layer` consecutive FD bins around `m * Nt_layer/2` with
   Hermitian wraparound at `k < 0` and `k > N/2`.
2. Multiply by the precomputed `Cmm` window (built from `phitilde` with
   `dOmega_s = π/Nf`).
3. Inverse FFT of size `Nt_layer` (Bluestein for general even).
4. Apply the `(m+n)`-parity selector on the **global** n grid:
   - `n_global = n_global_start + i`
   - `parity   = (m + n_global) & 1`
   - `sign     = ((m + 1) * n_global) & 1 ? -1 : +1`
   - `val      = parity ? imag(IFFT[i]) : real(IFFT[i])`
   - `w_mn[i]  = kappa * sign * val`,  where `kappa = 2 * sqrt(π dt) / Nf`
5. Edge layers (`m == 0` or `m == Nf`) zero out pixels where parity == 1.
6. DC/Nyquist merge after all layers:
   ```
   w_mn_final[:, 1:Nf]      = w_mn_tmp[:, 1:Nf]
   w_mn_final[:, 0, 0::2]   = w_mn_tmp[:, 0,  0::2] / sqrt(2)
   w_mn_final[:, 0, 1::2]   = w_mn_tmp[:, Nf, 0::2] / sqrt(2)
   ```

### Narrow-window per-(binary, layer)

- `narrow_widths[bin, m]` (int, even, > 0) → use `Nt_narrow = narrow_widths[bin, m]`.
- `narrow_centers[bin, m]` (int) → global n-index of the centre pixel.
- `n_global_start = n_center - Nt_narrow / 2`.
- The `(m+n)`-parity uses **`n_global = n_global_start + i`** — **never**
  the local IFFT index `i`.  This is the "evens/odds (m+n)" subtlety
  called out in the spec: when `(m + n_global_start) % 2 == 1`, pixel
  `i=0` lands on an odd-parity slot and the kernel takes `imag(IFFT[0])`
  instead of `real(IFFT[0])`.  The Python tests
  (`test_narrow_window_parity`) exercise both parities and confirm
  bit-identical agreement against an independent reference for each.
- The FD slice for a narrow layer uses `Nt_narrow` bins around
  `m * Nt_narrow / 2`, mirroring `get_shift_map` at the narrower size.
- The Bluestein plan for each distinct `Nt_narrow` is precomputed once
  in `WDMSplineKernelPlan.build(narrow_widths=[...])`.

## C++/CUDA port roadmap

The Python reference is structured so each function maps 1:1 onto a
CUDA-block-cooperative routine:

| Python function | C++/CUDA equivalent | Notes |
| --- | --- | --- |
| `BluesteinPlan.build` | `wdm_spline_build_bluestein_chirps` (host) | already declared in `WDMSplineHelpers.hh` |
| `bluestein_fft` | `wdm_spline_bluestein_fft` (device, block-coop) | uses existing `gbfd_radix2_fft_inplace` for the inner radix-2 FFT of length M |
| `two_stage_fft` | `wdm_spline_synth_and_fft` (device) | TD synth + Stage A + twiddle + Stage B, all in per-block global scratch |
| `extract_layer_wdm` | `wdm_spline_extract_layer` (device) | shared-mem gather slice → window × Bluestein-IFFT → parity |
| `synth_and_transform_one_binary` | per-binary block body inside each kernel | shared scratch budget = `Nt_layer * cmplx + bluestein workspace` |
| `fill_global_wdm` | `fd_spline_wdm_fill_global_kernel` + wrap | atomic-add into `template_fill` |
| `get_ll_wdm` | `fd_spline_wdm_get_ll_kernel` + wrap | uses existing `WDMDomain::add_ip_contrib` ([TDIonTheFly.cu:572](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L572)) |
| `swap_ll_wdm` | `fd_spline_wdm_swap_ll_kernel` + wrap | uses existing `WDMDomain::add_ip_swap_contrib` ([TDIonTheFly.cu:626](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L626)) |

### Concrete next steps for the C++ port (in order)

1. **Wire `WDMSplineHelpers.hh` into the build.**  Add
   `#define WDM_SPLINE_HELPERS_IMPLEMENTATION` then
   `#include "WDMSplineHelpers.hh"` at the bottom of
   [`TDIonTheFly.cu`](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu)
   so the implementation lands in one translation unit.
2. **Replace the placeholder TD synth in `wdm_spline_synth_and_fft`**.
   The current header uses `get_tdi_Xf_single` per sample; the real
   port should match what the existing `FDSplineTDIWaveform::get_tdi`
   does so the kernel produces the same TD as Python's
   `td_synthesizer(t_arr)`.  Verify by running the kernel on a
   single binary, dumping the FD scratch, and comparing against
   `numpy.fft.fft(td)`.
3. **Add the three kernels** following the GB pattern
   ([gb_wdm_fill_global_kernel](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L952),
   [gb_wdm_get_ll_kernel](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L1169),
   [gb_wdm_swap_ll_kernel](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L1349)).
   Each kernel body is ~80 lines: setup → per-binary loop → per-layer
   loop calling the device helpers from the header.
4. **Bindings** in
   [`binding_tof.hpp`](lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.hpp)
   / [`binding_tof.cxx`](lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.cxx)
   — mirror the existing `gb_wdm_*` bindings.  Add the new wrap class
   `FDSplineWDMComputationsWrap` (or fold into `GBComputationGroupWrap`).
5. **Expose in
   [`cutils/__init__.py`](lisa-on-gpu/src/fastlisaresponse/cutils/__init__.py)**
   alongside `FDSplineTDIWaveformWrap`.
6. **Python wrapper** (`splinecomps.py`) — mirror
   [`gbcomps.py`](lisa-on-gpu/src/fastlisaresponse/gbcomps.py).
   The Python reference here is essentially the wrapper's body; just
   replace the in-Python pipeline with backend calls.
7. **CI hookup**: re-run `test_fd_spline_wdm.py` after the C++ kernels
   are in to confirm the port matches the Python reference to ~1e-7
   (rel) on CPU and ~1e-6 on GPU (single-precision arithmetic in some
   GPU paths).

### Memory budget (worst case)

For 3 channels, Nf*Nt = 2^20:

- Per-block global scratch: `Nf * Nt * sizeof(cmplx) * nchannels` = 48 MB
- Shared memory per block (per-layer): `Nt * cmplx + bluestein workspace` =
  `2 * Nt * cmplx + 3 * M * cmplx` ≈ a few KB for Nt ≤ 1024
- For Nt narrow ≤ 256, the layer transform fits comfortably in shared
  memory; full Nt = 4096+ needs the global fallback (or a different
  in-kernel FFT like cuFFTDx).

### Known limitations / follow-ups not addressed

- The Python `extract_layer_wdm` re-builds the Bluestein plan per call
  when none is supplied.  The C++ port needs the plan precomputed on
  the host and copied to device once per launch (header already
  declares `WDMSplineBluesteinTable`).
- TD synthesis in the C++ kernel needs verification that the per-sample
  spline + LISA-response call produces the same TD waveform as Python.
  The simplest validation: call `FDTDIonTheFly.__call__` to get a TD
  reference, then dump the kernel's FD scratch and compare against
  `numpy.fft.fft(td)`.
- Mixed full/narrow scatter in one launch is implemented in Python but
  the C++ kernel needs to handle the per-(binary, layer) dispatch
  (`narrow_widths[bin, m] == 0` → full path; else narrow path).  The
  branch is straightforward but doubles the kernel's complexity; can
  be implemented as two passes (one full, one narrow) if simpler.

## How to use the Python reference today

```python
from wdm_spline_helpers import (
    WDMSplineKernelPlan,
    synth_and_transform_one_binary,
    fill_global_wdm,
    get_ll_wdm,
    swap_ll_wdm,
)

plan = WDMSplineKernelPlan.build(
    Nf=32, Nt=128, data_dt=10.0,
    m_min=0, m_max=31, n_min=0, n_max=127,
    narrow_widths=[32, 64],   # declare once, use many times
)

def my_td(t_arr):
    # Evaluate your spline-based TDI waveform on t_arr.
    # Return shape (nchannels, len(t_arr)) real array.
    ...

w_mn = synth_and_transform_one_binary(plan, my_td, nchannels=3)
# w_mn has shape (3, plan.Nf_active, plan.Nt_active)
```

For 1000s of sources, batch the synthesizers into a list and call
`fill_global_wdm` / `get_ll_wdm` / `swap_ll_wdm`.  The reference does
the sources sequentially (one per Python loop iteration); the C++
port distributes them across CUDA blocks.

## Phase-0 decision (2026-06-02): defer the C++/CUDA port to Phase 3

Per the sprint reorg plan
([`/Users/mkatz/.claude/plans/it-is-time-to-delegated-peach.md`](/Users/mkatz/.claude/plans/it-is-time-to-delegated-peach.md)
finding #3): the missing kernels (Bluestein FFT, 2-stage TD→FD FFT, layer
extraction, `fill_global` / `get_ll` / `swap_ll` finishers) will be
implemented **inside the GBGPU package** after the WDM-spline machinery
moves there in Phase 3, not in lisa-on-gpu. Rationale: implementing the
port in lisa-on-gpu now would require porting the kernels twice (once
here, once after relocation) and would entangle Phase 2's
absorb-lisa-on-gpu work with new physics code. The Python reference
above stays as the canonical source of truth during the port.

Status of the existing skeleton header
[`WDMSplineHelpers.hh`](lisa-on-gpu/src/fastlisaresponse/cutils/WDMSplineHelpers.hh):
move it to GBGPU alongside `GBTDIonTheFly` / `GBComputationGroup` per
the Phase 3 GBGPU plan; do the kernel implementation in the new home.
