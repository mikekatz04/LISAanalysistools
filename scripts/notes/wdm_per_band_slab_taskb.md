# Task (b): per-band WDM slab addressing — completion spec

Goal: shrink each GB SubBandBuffer slab from the full active band
(`Nf_active`, ~138 layers at 6–25 mHz) to just the band's `n_layers + guard`
(~10), cutting the dominant GPU-memory term ~10–18×. Guarded, **default-off =
bit-identical** to today.

## Design (the whole feature in one line)

The chunked-het kernels write/read WDM layer `m` at buffer offset
`m - ind_min_f` with per-slab stride `nchannels*Nf_active*Nt_active`, where
`ind_min_f` is the single GLOBAL analysis-band origin baked into
`cpp_wdm_settings`. Narrow mode replaces that global origin/extent with a
**per-slab** origin + extent:

- `Nf_slab` (int, `<=0` → full `Nf_active`): per-slab frequency extent.
- `slab_min_f` (int array, len = #slabs; `null`/empty → global `ind_min_f`):
  each slab's start WDM layer.

Then per bin: `band_min_f = slab_min_f ? slab_min_f[data_index] : ind_min_f`,
`slab_Nf = Nf_slab>0 ? Nf_slab : Nf_active`, and substitute
`Nf_active→slab_Nf`, `ind_min_f→band_min_f` in per_data/per_invC/g_d/g_inv +
the m-band clamp. Time axis is unchanged (GB spans all `Nt_active`).

## DONE (LAT, `src/lisatools/cutils/lat_chunked_het_kernels.hh`)

Both new args added as trailing defaults (`int Nf_slab=0, const int
*slab_min_f=nullptr`), so every existing caller compiles + behaves identically:

- **Kernels** (all 4): `wdm_het_get_ll_kernel`, `wdm_het_fill_global_kernel`
  (via `_narrow`/`_oNf`/`f_off`), `wdm_het_swap_ll_kernel`,
  `wdm_het_get_fstat_ll_kernel` — per-bin `band_min_f`/`slab_Nf`, m-band clamp,
  `m_act`, and every `g_d`/`g_inv`/`per_data`/`per_invC` stride.
- **Launchers** (all 4 `_impl<SourceT>`): args threaded to all 8 launch sites
  (CUDA `<<<>>>` + CPU). Brace-balanced; no addressing `Nf_active` left.

This LAT change is safe to build/commit now — nothing passes the new args yet.

## DONE — full GB vertical + Python (2026-07-10, default-off = bit-identical)

The GB path is wired end-to-end; narrow mode activates via
`GBSettings.wdm_band_slab_layers` (env `GB_WDM_BAND_SLAB_LAYERS`) → the GB move →
`BandSorter` → `SubBandBuffer` → `chunked_het.py` → the kernels.

- **GBGPU** `gb_tdi_on_the_fly.{cu,hh}`: 4 `..._wrap` thread `int Nf_slab, int
  *slab_min_f` to the launchers.
- **GBGPU** `binding_gbgpu.{cxx,hpp}`: 4 nanobind methods take `int Nf_slab,
  array_type<int> slab_min_f` (REQUIRED — no nanobind array default; `chunked_het.py`
  is the sole caller and always passes them; `fill_global`'s `active_band` default was
  dropped to satisfy C++ ordering). data_d/invC size-check divisor is slab-aware via a
  redefined `Nf_active` local.
- **GBGPU** `gb_likelihood.py` `WDMBandLikelihoodEngine.fill_template`: forwards
  `band_slab_Nf`/`slab_min_f` (sourced from the `SubBandBuffer`, which owns the residual +
  template-twin slabs) to `fill_global_wdm`.
- **LAT** `chunked_het.py`: `_slab_args_from`/`_slab_kernel_args` (return `()` on the JAX
  backend so JAX is untouched; `(0, empty)` off; `(band_slab_Nf, slab_min_f)` on) appended to
  get_ll / swap_ll / get_fstat_ll; `fill_global_wdm` gained `band_slab_Nf`/`slab_min_f` params
  + a narrow `Nfa` size-logic branch.
- **LAT** `gbbands.py`: `wdm_band_slab_layers` ctor flag; `band_slab_Nf` + `slab_min_f`
  (per-slot, band-centered, clamped, tracks cell swaps) properties; narrow
  `_per_band_{data,sens}_shape` via `_per_band_Nf`; `_build_per_band_basis_settings` pins
  per-band `ind_max_f` (Nf_active == slab_Nf); per-slot sliced `_get_fill_buffer_ind_map`;
  `_adjust_via_engine` forwards slab info only when narrow. Threaded through
  `BandSorter`/`get_buffer`, the move (`gbspecialstretch.py`), and base `GBSettings`.

Verified locally: off-path bit-identical (all substitutions reduce to originals when
`Nf_slab=0`/`slab_min_f=null`), `py_compile` clean, 32/32 stock tests, C++ brace-balanced.
NOT compiled (no GPU here).

**SOBBH (BBHx)** = safe-off: `sobbh_tdi_on_the_fly.cu` calls the shared launchers without the
new args → defaults → bit-identical; no change needed to compile/run off. **JAX** = safe-off
(helper returns `()`).

## Sizing default + check script

`wdm_band_slab_layers`: `None` = off; `0` = **auto** = `band_layer_span +
2*(leakage + guard)`; `N>0` = explicit. `leakage = 2`
(`_WDM_SLAB_LEAKAGE_LAYERS` — ideal-WDM tone localizes to ±1 layer measured,
chunked-het effective leakage = `m_band_half_width`=1), `guard =
wdm_slab_guard_layers` (adjustable, default 1). So a 1-layer GB band auto-sizes
to **7** layers.

`scripts/diagnostics/check_wdm_band_slab.py` runs three sections: **leakage**
(measures WDM per-layer energy + prints the recommendation), **logic**
(pure-numpy checks of the real narrow-slab property / index-map code — 11/11
PASS locally, no GPU), **kernel** (numerical narrow-vs-full; SKIPs until the
backend is rebuilt with task-b). Run it before and after the GPU build.

## REMAINING to make narrow mode usable (your side — GPU build + validate)

0. Run `python scripts/diagnostics/check_wdm_band_slab.py` (logic must be 11/11).
1. Rebuild GBGPU cuda12x (+ LAT) on the cluster.
2. **OFF-path**: run the GB match with `wdm_band_slab_layers=None`; must match pre-change to
   machine precision.
3. **ON-path**: set e.g. `GB_WDM_BAND_SLAB_LAYERS=12`; compare mm5/mm2 vs the full-width buffer
   (expect a small documented sub-band-edge spectral-tail truncation; median mm5 ~1e-9 stays in
   band). Ensure the value covers each band's layer span + 2·m_band_half_width.
4. **Memory**: confirm peak drops ~Nf_active/slab_Nf× (add a narrow row to
   `scripts/diagnostics/gpu_memory_estimate.py`).

## (Reference) the original mechanical recipe — now implemented above

### GB vertical — `GBGPU/src/gbgpu/cutils/`
For each of the 4 methods (`gb_wdm_het_{get_ll,swap_ll,fill_global,get_fstat_ll}`):
1. `gb_tdi_on_the_fly.cu` `GBComputationGroup::..._wrap(...)`: add
   `int Nf_slab, int *slab_min_f` params; forward to the launcher call.
2. `gb_tdi_on_the_fly.hh` (class decl): add the 2 params to the `..._wrap` decl.
3. `binding_gbgpu.cxx` `GBComputationGroupWrap::...`: add
   `int Nf_slab, array_type<int> slab_min_f` params; change the data_d/invC
   **size-check divisor** `Nf_active → (Nf_slab>0 ? Nf_slab : Nf_active)`;
   pass `Nf_slab` and `slab_min_f.size()>0 ? return_pointer(slab_min_f) :
   nullptr` to `..._wrap`.
4. `binding_gbgpu.cxx` `.def("gb_wdm_het_...", ...)`: append the 2 args
   (nanobind: `"Nf_slab"_a = 0, "slab_min_f"_a` — pass an empty cupy int array
   from Python when off to avoid array default-arg issues).
5. `binding_gbgpu.hpp`: add the 2 params to the 4 `GBComputationGroupWrap`
   method decls.

### `LISAanalysistools/src/lisatools/chunked_het.py`
`get_ll_wdm` / `get_swap_ll_wdm` / `fill_global_wdm` / `get_fstat_ll_wdm`:
resolve `Nf_slab`/`slab_min_f` from the holder (see gbbands below) and pass
them to `self._kernel(...)`. Default: `Nf_slab=0`, `slab_min_f=` empty array →
off. Read them off `wdm_holder` (e.g. `getattr(holder, "slab_min_f", None)` /
`"band_slab_Nf"`), mirroring how `min_freq_inds` is already read.

### `LISAanalysistools/src/lisatools/globalfit/moves/gbbands.py`
New `wdm_band_slab_layers: Optional[int] = None` ctor flag (None → full-width,
today's behavior; int → narrow). When set, on the WDM path:
- `_per_band_data_shape` / `_per_band_sens_shape`: replace `Nf_active` with
  `slab_Nf = wdm_band_slab_layers + 2*guard` (guard covers m_band_half_width).
- Expose per-slot `band_slab_Nf` (int) + `slab_min_f` (int array, per band =
  `band_center_layer - guard`) that chunked_het reads (like `min_freq_inds`).
- `_get_fill_buffer_ind_map` (WDM): slice the parent ACA
  `[band_m_lo : band_m_lo+slab_Nf]` on the layer axis instead of the full
  `[0:Nf_active]`, with `band_m_lo = slab_min_f[slot] - parent.ind_min_f`.
- `min_freq_inds` (WDM): return each slot's own `slab_min_f` (not the global
  `ind_min_f`).
Wire `wdm_band_slab_layers` through `GBSettings` (like `n_subbands`) → the GB
move → `get_buffer(..., wdm_band_slab_layers=...)`.

### SOBBH parity — `BBHx/src/bbhx/cutils/` (only if SOBBH buffer needs narrowing)
Same 5 edits as the GB vertical in `sobbh_tdi_on_the_fly.{cu,hh}` +
`binding_bbhx.{cxx,hpp}`. Currently SAFE-OFF (SOBBH wraps call the launcher
without the new args → defaults → bit-identical). SOBBH memory is not the
reported problem, so this is optional/deferred.

### JAX mirror — `gbgpu/jax/wdm/heterodyne_kernels.py` (parity rule)
Add per-slab origin to the JAX `gb_wdm_het_{get_ll,swap_ll,fill_global}_jax`.
JAX is validation-only (not the memory-constrained production path); narrow
mode there is a documented follow-up. When off, JAX is untouched.

## Validation (on GPU, after build)
1. **Off-path bit-identity**: run the GB match test with the flag OFF; must
   match pre-change to machine precision.
2. **On-path correctness**: turn `wdm_band_slab_layers` on; compare mm5/mm2
   (canonical narrow-band, `gb_chunked_prior_draws.py:283-340`) against the
   full-width buffer. Expect a small, documented spectral-tail truncation at
   sub-band edges (layers outside the slab are dropped) — validate mm5 stays
   within the acceptance band (median ~1e-9).
3. **Memory**: confirm peak drops ~Nf_active/slab_Nf× via
   `scripts/diagnostics/gpu_memory_estimate.py` (add a narrow row).
