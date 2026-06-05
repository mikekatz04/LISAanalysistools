# Chunked-heterodyne WDM pipeline -- design notes

## Pipeline overview

Phase 2 implements three new CUDA kernels in
`./lisa-on-gpu/src/fastlisaresponse/cutils/TDIonthefly.cu`, all built on
the same chunked-heterodyne primitives:

1. **`fast_wdm_inner_heterodyne`** (CUDA_DEVICE) -- per-chunk slow signal
   + FFT + placement into chunk dense rfft.
2. **`gb_chunk_fd_to_wdm`** (CUDA_DEVICE) -- per-chunk dense FD ->
   chunk WDM coefficients (Nf, Nt_sub).
3. **`gb_wdm_het_fill_global_kernel`** (CUDA_KERNEL) -- assembles
   per-binary stitched WDM via chunked heterodyne.
4. **`gb_wdm_het_get_ll_kernel`** (CUDA_KERNEL) -- per-binary
   `<d|h>`, `<h|h>` via chunked outer loop + shared PSD/data + CUB
   reduce + atomicAdd to per-source output.
5. **`gb_wdm_het_swap_ll_kernel`** -- analogous to get_ll with add/remove
   pairs (`<d|h_add>`, `<d|h_remove>`, `<h_add|h_add>`,
   `<h_remove|h_remove>`, `<h_add|h_remove>`).

Python references for each are in `check_shortened_wdm.py`:

| C++ kernel                       | Python reference                                    |
| -------------------------------- | ----------------------------------------------------|
| `fast_wdm_inner_heterodyne`      | `CachedHeterodyneGenerator.chunk_fd`                |
| `gb_chunk_fd_to_wdm`             | `gb_chunk_fd_to_wdm_python_reference` (Test K)      |
| `gb_wdm_het_fill_global_kernel`  | `_stitched_wdm_from_heterodyne` (+ partial slide)   |
| `gb_wdm_het_get_ll_kernel`       | (not yet -- Phase 2c)                               |
| `gb_wdm_het_swap_ll_kernel`      | (not yet -- Phase 2d)                               |

## Loop structure (CUDA, per the user's plan)

```
gb_wdm_het_get_ll_kernel<<<n_chunks_per_grid, NUM_THREADS>>>(...)
  for chunk j in [BLOCK_START, n_chunks) stride GRID_INCR:        # OUTER
    load PSD[chunk slice]   -> sh_psd[3 * Nf * Nt_sub]            # global -> shared, once
    load data[chunk slice]  -> sh_data[3 * Nf * Nt_sub]
    CUDA_SYNC_THREADS
    for binary i in [0, num_bin):                                 # INNER
      fast_wdm_inner_heterodyne -> chunk_fd_shared
      gb_chunk_fd_to_wdm       -> chunk_wdm_shared
      // each thread accumulates into partial[NUM_THREADS] slots
      partial_dh[THREAD_START] = 0; partial_hh[THREAD_START] = 0
      for (c, m, n_local) striding THREAD_START in chunk's interior:
        dh += sh_data[c, m, n_local] * chunk_wdm[c, m, n_local] * sh_psd[c, m, n_local]
        hh += chunk_wdm[c, m, n_local]^2 * sh_psd[c, m, n_local]
      CUDA_SYNC_THREADS
      #ifdef __CUDACC__
        CUB block-reduce partial_dh, partial_hh -> single doubles
        if (THREAD_START == 0) {
          atomicAdd(&d_h_out[i], dh_chunk);
          atomicAdd(&h_h_out[i], hh_chunk);
        }
      #else
        for k in [1, NUM_THREADS): partial_dh[0] += partial_dh[k]; ...
        d_h_out[i] += partial_dh[0];
        h_h_out[i] += partial_hh[0];
      #endif
```

Atomic contention is collapsed to **one atomicAdd per (chunk, binary)
per source**, not one per WDM pixel.

## NUM_THREADS vs N_sparse / Nt_sub

- `wdm_spline_radix2_fft` and our slow-signal / WDM-pixel loops all
  stride via `THREAD_START / BLOCK_INCR`. So `NUM_THREADS` can be
  picked **independently** of `N_sparse` and `Nt_sub`, subject only to
  `N_sparse >= NUM_THREADS` and `Nt_sub >= NUM_THREADS` (so every
  thread has work in the FFT).
- A reasonable default: `NUM_THREADS = 128`. Both `N_sparse <= 256`
  and `Nt_sub <= 256` satisfy `>= NUM_THREADS`.

## Shared-memory budget (N_sparse=256, Nt_sub=256, nchannels=3)

| Buffer              | Size                                    |
| ------------------- | --------------------------------------- |
| `t_sparse_buf`      | 256 doubles = 2 KB                      |
| `tdi_amp_buf`       | 3 * 256 doubles = 6 KB                  |
| `tdi_phase_buf`     | 3 * 256 doubles = 6 KB                  |
| `phi_ref_buf`       | 256 doubles = 2 KB                      |
| `tdi_channels_buf` | 3 * 256 cmplx = 12 KB                   |
| `slow_buf`          | 3 * 256 cmplx = 12 KB                   |
| **heterodyne sub-total** | **40 KB**                          |
| `chunk_wdm_shared`  | 3 * 64 * 256 doubles = 384 KB ❌ over   |

The chunk WDM block (3 * Nf * Nt_sub doubles) is too big for shared
memory at Nf=64. Options:

(a) Keep `chunk_wdm_shared` in heap-allocated workspace (one slab per
    block), not shared. Per-block heap = ~400 KB.
(b) Stream layers: compute WDM one frequency layer at a time, feed
    into the `<d|h>` / `<h|h>` accumulator immediately, then discard.
    Only one Nt_sub-long buffer in shared at a time (~6 KB / channel).
    This is the right design for get_ll/swap_ll because we never
    actually need all layers in shared at once.

Decision: **stream layers** for get_ll/swap_ll. For fill_global the
output must be assembled, so use heap workspace for that one kernel.

## Partial-slide edge handling

Host pre-computes chunk geometry:

```
starts          : (n_chunks,)   chunk start times (s, absolute)
n_global_lo     : (n_chunks,)   global WDM pixel where this chunk's
                                  writable region starts
keep_lo         : (n_chunks,)   chunk-local lo
keep_hi         : (n_chunks,)   chunk-local hi
```

When the last full slide would over-shoot `Nt`, append one chunk at
`Nt - Nt_sub` (slides less than `step` from the previous chunk).
For this chunk, set `keep_lo = n_pad` and `keep_hi = Nt_sub`, with
`n_global_lo = Nt - (Nt_sub - n_pad)`. The previous full chunk's
`keep_hi` stays `Nt_sub - n_pad` so the overlapped region is not
double-counted.

## Orbits / shared-memory / const-memory / texture-memory

LISA orbit data in the existing kernel: positions x(t) and link
arms L(t, link) interpolated via cubic splines stored in global
memory. Each `LISATDIonTheFly::get_tdi*` call does
`O(N_sparse * n_links)` interpolation lookups, each touching a few
control points in the spline table.

For the chunked workflow the orbit lookup pattern is:

- Outer loop = chunks. Each chunk samples N_sparse <= 256 time
  points within a 1.9-day window. Orbits are smooth over a window
  this short (LISA orbital period ~1 yr, so the window spans ~0.5%
  of the orbit).
- Inner loop = binaries. Every binary in the chunk wants orbits at
  **the same N_sparse time points**.

This is the textbook case for `__constant__` / `__shared__` /
texture-memory orbit caching:

| Option                           | Read cost            | Capacity         | Pros / cons                            |
| -------------------------------- | -------------------- | ---------------- | -------------------------------------- |
| Global mem (current)             | L1 / L2 cached      | huge             | Works; pays cache misses per-call      |
| Constant mem `__constant__`     | broadcast, 1-cycle  | 64 KB total      | Best for small read-only tables shared by all threads |
| Texture mem `tex1D`             | spatial-locality cache | huge          | Hardware linear interpolation; good for splines |
| Shared mem `__shared__`         | 1-cycle, per block  | ~48 KB / SM       | Manual; ideal for "load once, reuse many" pattern |

**Important refinement (per user note 2026-05):** the existing pipeline
evaluates the cubic-spline orbit ONCE and then linearly resamples it
at higher density in the hot loop. The shared-mem cache should hold
the **linear-spline outputs** at the chunk's ``N_sparse`` times --
NOT the cubic control points. That keeps the cache size proportional
to ``N_sparse``, which is what we want.

**Per-chunk orbit-cache budget at N_sparse = 256:**

```
sh_x_rec[N_sparse * 3 (spacecraft) * 3 (xyz)]  =  9 * 256 doubles  = 18 KB
sh_L_arm[N_sparse * 6 (links)]                 =  6 * 256 doubles  = 12 KB
                                                  orbit total    = 30 KB
```

**Combined per-block shared at N_sparse=256, nchannels=3:**

```
heterodyne workspace      ~40 KB
orbit cache               ~30 KB
get_ll partials (dh, hh)   ~4 KB
                ---------------
get_ll total              ~74 KB
swap_ll total (5 partials) ~80 KB
```

**Pre-move:** 74-80 KB EXCEEDS the 48 KB default static-shared budget.

### Mitigation (implemented 2026-05): tdi_channels_buf moved to heap.

`tdi_channels_buf` (12 KB at N_sparse=256, nchannels=3) is a write-once /
read-once scratch buffer written by `get_tdi` and consumed by the
slow-signal loop. There is no random-access pattern, so shared-mem
latency is unnecessary. Moved to a per-chunk slab in heap-allocated
global memory via a new `ws_tdi_channels_all` workspace parameter on the
three kernels (`gb_wdm_het_fill_global_kernel`,
`gb_wdm_het_get_ll_kernel`, `gb_wdm_het_swap_ll_kernel`). The original
single-binary `fast_wdm_inner_heterodyne_dispatch_kernel` test driver
still uses the shared declaration -- it is low-pressure (one block,
one binary) and used only for unit testing.

**Heap budget for `ws_tdi_channels_all`:** sized
`n_chunks * nchannels * N_sparse * sizeof(cmplx)`.

| Config                              | n_chunks | size                |
|-------------------------------------|----------|---------------------|
| Nf=256, Nt_sub=256, Tobs=½yr        | ~25      | 25 * 12 KB = 300 KB |
| Nf=4320, Nt_sub=128, Tobs=1yr       | ~245     | 245 * 12 KB = 2.9 MB |
| Stress (Nf=4320, Nt_sub=64, Tobs=1yr) | ~490   | 490 * 12 KB = 5.9 MB |

All trivial vs. A100's 40-80 GB HBM. No risk of running the device out
of heap.

### Post-move shared budget (N_sparse=256, nchannels=3, with orbit cache):

```
heterodyne workspace   ~28 KB   (was 40 KB; -12 KB for tdi_channels)
orbit cache            ~30 KB
get_ll partials         ~4 KB
                       -------
get_ll total           ~62 KB
swap_ll total          ~68 KB   (5 partials instead of 2)
```

Without the orbit cache (Phase 2 default), get_ll total is **~33 KB**,
under the 48 KB default budget -- no `cudaFuncSetAttribute` opt-in
required.

### A100 occupancy thresholds

A100 (Ampere) shared-mem and occupancy specs:

| Spec                                    | Value                |
|-----------------------------------------|----------------------|
| Combined L1/shared per SM (carveout)    | 192 KB (164 KB max as shared) |
| Default static-shared budget per block  | 48 KB                |
| Max per-block shared (opt-in)           | 163 KB               |
| Max threads per SM                      | 2048                 |
| Max blocks per SM                       | 32                   |
| L1/shared carveouts                     | 0/100/132/164 KB     |

For our `NUM_THREADS=128` choice, the thread limit caps SM occupancy at
`2048 / 128 = 16` blocks/SM. The first occupancy reduction from shared
memory pressure kicks in at `164 KB / 16 blocks = 10.25 KB/block`.

**Shared-mem budget vs. blocks-per-SM at NUM_THREADS=128 (A100):**

| Shared/block | Blocks/SM | Threads/SM | Latency hiding | Verdict |
|--------------|-----------|------------|----------------|---------|
| ≤ 10 KB      | 16        | 2048       | excellent      | full occupancy (thread-limited) |
| ≤ 20 KB      | 8         | 1024       | good           | sweet spot for our kernels |
| ≤ 41 KB      | 4         | 512        | adequate       | post-move get_ll lands here w/o orbit cache |
| ≤ 54 KB      | 3         | 384        | marginal       | -- |
| ≤ 82 KB      | 2         | 256        | poor           | post-move w/ orbit cache lands here |
| ≤ 100 KB     | 1         | 128        | none           | wastes the SM |

**Practical rule for A100 + NUM_THREADS=128:**

- **≤ ~20 KB/block:** no shared-mem pressure, kernel is bound by other
  resources (registers, memory bandwidth).
- **20-41 KB/block:** "starts to potentially slow down" -- still 4
  blocks/SM, but you've cut blocks/SM by 4x relative to the thread
  limit. Acceptable for memory-bandwidth-bound kernels.
- **> 41 KB/block:** occupancy starts to matter; falls to 3 blocks/SM
  then 2.
- **> 82 KB/block:** 1 block/SM, latency is unhidden -- significant
  slowdown expected unless ILP is high.

So the rough lower limit on A100 where shared memory begins to be a
binding resource constraint is **~20-25 KB per block** at our 128-thread
configuration. Below that, increasing shared usage is free; above it,
you pay in occupancy.

### Where this leaves us

| Kernel + orbit-cache                  | Shared (KB) | A100 blocks/SM | Default-budget? |
|---------------------------------------|-------------|----------------|-----------------|
| get_ll, pre-move, no orbit cache      | ~45         | 3              | borderline (45 < 48 OK) |
| get_ll, pre-move, with orbit cache    | ~75         | 2              | needs opt-in    |
| **get_ll, post-move, no orbit cache** | **~33**     | **4**          | **yes ✓**       |
| get_ll, post-move, with orbit cache   | ~63         | 2              | needs opt-in    |
| swap_ll, pre-move, no orbit cache     | ~48         | 3              | borderline      |
| swap_ll, post-move, no orbit cache    | ~36         | 4              | yes ✓           |
| swap_ll, post-move, with orbit cache  | ~66         | 2              | needs opt-in    |

Moving `tdi_channels_buf` to heap buys:
1. ~33 KB get_ll fits the 48 KB default budget -- no opt-in needed on
   any post-Volta GPU, simpler kernel launch.
2. Occupancy goes from 3 → 4 blocks/SM on A100 for the no-orbit-cache
   path (33% boost in achievable warps).
3. Leaves headroom for orbit cache: with cache, total is 63 KB, still
   only 2 blocks/SM but no longer over 82 KB.

### Other buffers ranked by next-largest

Order of further candidates if more shared headroom is wanted:

| Buffer            | Bytes (N_sparse=256, nchan=3) | Movable? |
|-------------------|-------------------------------|----------|
| `slow_buf`        | 12 KB (cmplx)                 | **No** -- FFT butterfly is random-access |
| `tdi_amp_buf`     | 6 KB (double)                 | Maybe -- read-once-per-binary in heterodyne loop, but tight loop |
| `tdi_phase_buf`   | 6 KB (double)                 | Same |

## Baseline regime + on-the-fly source-signal spline cache

**Baseline (2026-05) is half-day wavelets / 1 yr observation / no lookup
table**: ``Nf = 4320, Nt_sub = 128, dt = 10, T_chunk ≈ 32 d, N_sparse =
256``. Validation: ``gb_prior_chunked_Nf4320_*.npz`` shows mm5 median
1.21e-9 over 50 draws spanning 1-25 mHz.

### Orbit-cache density (in-kernel cubic spline)

Plan: at chunk entry, evaluate ``orbits->get_pos`` / ``get_light_travel_time``
at ``N_cp_orbit`` sparse times spanning the chunk window, then fit a cubic
spline cooperatively (PCR from ``GPUBackendTools``) into shared mem.
Replaces ``N_sparse * 32-64`` global-mem orbit linear-interp calls per
(chunk, binary) with one fit + cheap shared-mem cubic evaluation.

Python density study (``dev_orbit_spline_density.py``):

| T_chunk | N_cp_orbit=16 maxL err | N_cp_orbit=24 maxL err | N_cp_orbit=32 maxL err | maxX err at N_cp_orbit=32 |
|---------|------------------------|------------------------|------------------------|---------------------------|
| 0.5 d   | 1.2e-14 s              | 1.2e-14 s              | 2.1e-14 s              | 1.7 m                     |
| 7 d     | 1.6e-13 s              | 6.2e-14 s              | 2.1e-14 s              | 4.2 m                     |
| **30 d** | 3.3e-11 s             | 6.0e-12 s              | **1.7e-12 s**          | **340 m**                 |

Tolerance budget for mm < 1e-9 at f=25 mHz: need L < ~1e-4 s, x < ~1e3 m.

**Baseline choice: N_cp_orbit = 32.** Clears 8 orders on L, 3+ orders on x
even at 30-day chunks. Persistent shared cost: shared ``x_cp[32]`` + 15
scalar series × 4 spline arrays × 32 doubles = ``(60 + 1) × 32 × 8 =``
**15.6 KB**.

### Source-signal spline (within one chunk × binary)

Plan: instead of calling ``gb->get_tdi`` at all ``N_sparse`` times, evaluate
at a much sparser ``N_cp_sig`` set of control points within the chunk,
fit cubic splines through (``tdi_amp[c,t]``, ``tdi_phase[c,t]``,
``phi_ref_het[t] = phi_ref[t] - 2π·f0_grid·t_abs``), and evaluate the
spline at the dense ``N_sparse`` grid. Replaces ``N_sparse`` expensive
get_tdi calls with ``N_cp_sig`` get_tdi + ``N_sparse`` cheap spline evals
per (chunk, binary).

**Critical prereq**: ``LISATDIonTheFly`` must provide a **new method**
``get_tdi_heterodyned(t_arr, params, f0_grid, ...)`` that returns
``(tdi_amp, tdi_phase, phi_ref_het)`` with ``phi_ref_het = phi_ref -
2π·f0_grid·t_abs`` (carrier subtracted in-kernel).

API choice -- **separate method, not a flag**:

* ``get_tdi(t_arr, params, ...) -> (amp, phase, phi_ref)`` -- always
  valid; ``phi_ref`` carries the source's natural fast component.
  Required for sources without a constant carrier (MBHB chirp, EMRI,
  generic spline-defined sources).
* ``get_tdi_heterodyned(t_arr, params, f0_grid, ...) -> (amp, phase,
  phi_ref_het)`` -- carrier-subtracted output. Only meaningful when the
  source has a well-defined constant carrier (GB, SOBBH). Sparse-sample-
  safe (unwrap residual is O(1 rad) over a 30-day chunk).

Why split rather than flag: when a source has no constant ``f0``,
calling ``get_tdi(..., subtract_carrier=True)`` would silently give
garbage. A separate method that doesn't exist on those sources gives a
loud ``AttributeError`` instead, which is the right failure mode.

The chunked-het kernels invoke only ``get_tdi_heterodyned``; if a
source class doesn't override it, the kernel raises a clear error
rather than running on raw ``phi_ref``.

Why the prereq matters: without carrier subtraction, evaluating
``phi_ref`` at sparse samples breaks the kernel's internal phase-unwrap
-- the carrier accumulates ~10⁴ rad over 3 days, so sparse samples span
tens of 2π wraps and cannot be unwrapped. Heterodyning brings the
residual to O(1 rad) over the chunk -> easy to spline.

Python density study (``dev_chunk_signal_spline_density.py``), all
errors are worst case across 4 chunk starts × {6-8 sources}:

| Source / regime          | N_cp_sig=12 mm | N_cp_sig=24 mm | N_cp_sig=32 mm | N_cp_sig=48 mm | Recommendation |
|--------------------------|----------------|----------------|----------------|----------------|----------------|
| GB,   Nf=256  / 3.8 d    | 1e-13          | 3e-16 (FP)     | 3e-16          | 3e-16          | 12             |
| SOBBH, Nf=256  / 3.8 d   | 3e-10          | 3.6e-10        | 3.3e-10        | 2.9e-10        | 16             |
| GB,   Nf=4320 / **32 d** | 2.2e-6         | 1.2e-8         | 2.0e-9         | **3.9e-11**    | **48**         |
| SOBBH, Nf=4320 / **32 d**| 1.4e-5         | 5.8e-7         | 5.1e-8         | **4.3e-9**     | **48** (safe)  |

**Baseline choice: N_cp_sig = 48.** Persistent shared cost:
- 3 channels × (tdi_amp + tdi_phase) splines × 4 arrays × 48 doubles = ``24 × 48 × 8 =`` 9.0 KB
- 1 × dphi_ref spline × 4 arrays × 48 doubles = 1.5 KB
- shared ``x_cp[48]`` = 0.4 KB

**Total source-signal cache: ~11.1 KB.**

### Buffers REMOVED when source-signal cache is active

The signal spline replaces the dense per-sample outputs of get_tdi, so
the existing shared buffers for those go away:

| Buffer removed   | Bytes (N_sparse=256, nch=3) |
|------------------|-----------------------------|
| `tdi_amp_buf`    | 6 KB                        |
| `tdi_phase_buf`  | 6 KB                        |
| `phi_ref_buf`    | 2 KB                        |
| `t_sparse_buf`   | 2 KB (computable inline)    |
| **Removed total** | **16 KB**                  |

So the spline cache (11.1 KB) is a **net 5 KB shared-mem savings** over
the dense-buffer design. ``tdi_channels_buf`` (12 KB) has already moved
to heap separately.

### Baseline post-cache shared budget (N_sparse=256, nch=3)

```
slow_buf (FFT)                    12.0 KB
orbit-spline cache (N_cp_orbit=32) 15.6 KB
source-signal cache (N_cp_sig=48)  11.1 KB
get_ll partials                     2.0 KB
                                  -------
get_ll total                      40.7 KB    <- fits 48 KB default ✓
swap_ll partials (5 slots)          5.0 KB
swap_ll total                     43.7 KB    <- fits 48 KB default ✓
```

PCR scratch buffer (8 × max(N_cp_orbit, N_cp_sig) = 3 KB) is transient
during the spline fits at chunk entry; can share storage with later
buffers, so does not stack.

### A100 / H100 occupancy at baseline

Using ``chunked_het_grid_dim()`` (in ``gb_wdm_het.py``):

| Arch | Shared/SM | get_ll resident blocks/SM | Theoretical occupancy |
|------|-----------|---------------------------|-----------------------|
| A100 | 164 KB    | ``floor(164/41) = 4``     | 25% (4 blocks × 128 threads / 2048) |
| H100 | 228 KB    | ``floor(228/41) = 5``     | 31%                                 |

Launch grid ``= 2 × resident × num_SMs`` (e.g., A100 = 864 blocks),
capped at ``n_chunks``. Grid-stride loop over chunks ensures each block
processes ALL binaries for its assigned chunk -- so the orbit + signal
spline caches are reused ``num_bin`` times per chunk.

### Compound speedup estimate

For the half-day baseline (1 yr, ~245 chunks, num_bin=1000 typical):

| Optimization                                  | Speedup factor on the TDI eval |
|-----------------------------------------------|--------------------------------|
| Orbit cache (replaces global-mem orbit calls) | ~5-10× (depends on cache hit rate) |
| Source-signal cache (N_cp_sig=48 vs 256)      | ~5×                            |
| **Compound (signal-spline within orbit-cache eval)** | **~25-50×**             |

The orbit cache helps every get_tdi call (cuts the inner ~32-64 orbit
lookups per sparse sample to cheap shared-mem reads). The source-signal
cache cuts the total number of get_tdi calls per (chunk, binary) by
~5×. The two compose.

## Future: spline-TDI source plug-in for the chunked path

Currently the chunked-het kernels assume analytic source classes
(``GBTDIonTheFly``, ``SOBBHTDIonTheFly``) that produce ``(amp, phase,
phi_ref_het)`` via the C++ ``get_tdi_heterodyned`` family. The
within-chunk source-signal spline cache (above) IS the abstraction
boundary that lets the kernel be source-agnostic: once the per-(chunk,
binary) ``(amp, dphase, dphi_ref_het)`` splines are populated, the
rest of fill_global / get_ll / swap_ll is identical regardless of how
the splines were built.

So a generalization opens up: instead of ``N_cp_sig`` calls to an
analytic ``get_tdi_heterodyned``, the spline cache can be populated by
**sampling an externally-provided coarse-grid spline** at the chunk's
``N_cp_sig`` control points. The existing ``gb_wdm_spline_*`` and
``gb_wdm_fd_spline_*`` kernels in ``TDIonthefly.cu`` already use this
exact pattern -- they pre-fit ``(amp, dphi, phi_ref)`` splines on a
coarse uniform time grid (``WDM_SPLINE_L`` ≈ 32 points), then evaluate
them densely.

**Templates vs tagged dispatch tradeoff**:

| Aspect                    | Templates (current) | Tagged dispatch (future option) |
|---------------------------|---------------------|----------------------------------|
| Uniform-batch perf        | Inlined; ~0 branch overhead | One branch per chunk; ~0 cost (no warp divergence -- all threads see same tag) |
| Mixed-batch perf (GBs + SOBBHs in one launch) | Not supported -- need separate launch per type | Native; ``source_kind_per_binary[bin_i]`` -- still no warp divergence since binaries iterate serially within a block |
| Binary size               | N_kernels × N_sources entries | 1 kernel handles all sources |
| Code maintenance          | Identical inner code, only the construct line varies | Single inner code path with ``if (kind == ...)`` branches at source-specific points |
| When to use               | Uniform batches per launch (current ask) | Mixed batches in one kernel (future) |

Switching templates -> tagged dispatch later is a clean refactor: add a
``source_kind`` arg, replace ``SourceT src(...)`` with a tagged switch,
collapse the parallel wrappers into one entry point. Reversible. We
keep templates today.

**GPU constraint**: runtime polymorphism via C++ virtuals does NOT work
across the host/device boundary. ``LISATDIonTheFly`` currently has
``virtual get_amp / get_phase / get_f / get_fdot``, and these dispatch
correctly only when the derived object (``GBTDIonTheFly`` etc.) is
constructed INSIDE the kernel body (so the vtable pointer is set on
device). A host-constructed pointer passed to a kernel would carry a
host vtable and crash. The chunked-het kernels currently construct
``GBTDIonTheFly gb(...)`` directly in the kernel body, which is GB-only.

So the source-polymorphism refactor must use **compile-time
polymorphism (templates)** -- e.g., ``template<class Source> kernel
(...)`` with one specialization per source class -- or a **tagged
dispatch** ``if (source_kind == GB) ... else if (SOBBH) ...`` inside
the kernel. **Do not** pass a polymorphic base-class pointer across
the host/device boundary and dispatch through ``vptr``.

**Proposed future architecture** (separate work item; do not block
on this):

```
Source class API (analytic vs spline):

  AnalyticSource (e.g. GBTDIonTheFly, SOBBHTDIonTheFly):
    populate_chunk_spline_cache(cache, chunk_t_start, T_chunk,
                                 params, f0_grid, N_cp_sig)
      -> call get_tdi_heterodyned(...) at N_cp_sig points,
         PCR-fit cubic spline into `cache`.

  SplineSource (e.g. MBHB or EMRI w/ precomputed amp/phase splines):
    populate_chunk_spline_cache(cache, chunk_t_start, T_chunk,
                                 params, f0_grid, N_cp_sig)
      -> evaluate the precomputed amp/phase/phi_ref splines at
         N_cp_sig points (much cheaper than analytic eval),
         PCR-fit into `cache`.

Kernel:
  for chunk j:
    fit orbit-cubic cache (shared)
    for binary i:
      source->populate_chunk_spline_cache(...)   # polymorphic
      evaluate dense at N_sparse from spline cache
      FFT, WDM xform, accumulate
```

This unifies the existing per-pixel ``gb_wdm_spline_*`` path and the
chunked-heterodyne path under the same kernel skeleton, with source
polymorphism at the cache-population step. Implementation is a
sizeable refactor of the source-class hierarchy; deferred until the
analytic chunked-het path is fully validated end-to-end.
| `t_sparse_buf`    | 2 KB                          | Read-once per get_tdi call; could go to heap |
| `phi_ref_buf`     | 2 KB                          | Read in heterodyne loop, keep shared |

`slow_buf` is the natural-stopping-point: anything bigger than its 12 KB
can't easily move out of shared without rewriting the FFT.

Per (chunk, binary):
- Without orbit cache: ~N_sparse * (3 orbit + 6 arm) ~ 9*256 = 2304
  global-mem orbit interpolations, each touching ~4 control points.
- With orbit cache: 0 global orbit reads (all in shared). N_sparse *
  number-of-links worth of shared reads.

**Expected effect.** Orbit eval is currently around 10-30% of GB
TDI cost (measured in earlier C++ profiles I've seen referenced in
the codebase). The chunked loop multiplies that cost by num_bin
(each binary re-pays it). Caching to shared mem in the chunk's outer
loop eliminates ~`num_bin - 1` of the duplicated orbit reads, giving
a `~num_bin` x speedup on the orbit-evaluation portion. For
num_bin = 64 that's a `~64x` reduction in orbit-related global
memory traffic; for the GPU the actual wall-clock saving depends on
how much we were already L1/L2 cached -- realistic guess **10-25%
end-to-end speedup** for medium num_bin.

Constant memory is **not** a great fit here because the orbit table
is ~MB-scale (full mission, dense spline), which doesn't fit in 64
KB. We'd have to copy a per-chunk slice into constant memory at the
top of each chunk's outer-loop iteration, which is the same memcpy
cost as the shared-memory option but with fewer reuse advantages.

Texture memory could be a win for the cubic-spline lookup itself --
the hardware linear interp is "close enough" to a cubic-spline eval
for some uses, but the LISATools orbit pipeline runs a true cubic
spline, so we'd have to either approximate or bypass texture-mode
interp. The simpler win is shared-memory caching of the pre-sampled
orbit values at the N_sparse times.

**Recommendation.** First land Phase 2 with orbits-in-global (current
pattern). Profile. Then drop in shared-memory orbit caching at the
top of the outer chunk loop in get_ll/swap_ll/fill_global as a
follow-up optimisation -- self-contained change, easy to A/B
benchmark, no algorithmic risk.

## Source-class abstraction (SOBBH plug-in)

The chunked heterodyne pipeline only depends on `GBTDIonTheFly`'s
public surface:

- Sparse-grid `get_tdi(t_arr, params, ...) -> tdi_amp, tdi_phase,
  phase_ref`.
- A source-frequency reference `f0` (used to snap `f0_grid`).

Both are abstracted at the `LISATDIonTheFly` base class. To swap in
SOBBH:

1. Subclass `LISATDIonTheFly` (analogous to `GBTDIonTheFly`).
2. Override `ucb_amplitude` / `ucb_phase` / `ucb_f` / `ucb_fdot`
   with the SOBBH analytic frequency model (or a precomputed
   sparse-grid evaluator if there's no closed form).
3. Pass `f0_index` to the heterodyne kernel via the params layout
   (`params[f0_index]`).

No changes needed in `fast_wdm_inner_heterodyne` /
`gb_chunk_fd_to_wdm` / `gb_wdm_het_*_kernel` -- the source object
is templated / passed by pointer in the existing pattern.
