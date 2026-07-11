# LISA Analysis Tools conventions

Canonical home for the rules that apply across all LISA Analysis Tools repos. Referenced from each repo's CLAUDE.md.

### Load-bearing enforcement infra (do not regress)

| Layer | Where |
|---|---|
| **L1** — LISA Analysis Tools–wide nanobind/pybind11 pin | `LISAanalysistools/constraints/sprint.txt`. Export `PIP_CONSTRAINT=<abs-path>/LISAanalysistools/constraints/sprint.txt` before every `pip install`. |
| **L2** — single-registrant rule | `LISATOOLS_IS_WRAPPER_OWNER` macro in `lisatools_header_abi.hpp` + per-TU `static_assert` + the `LISAanalysistools/tools/check_single_registrant.sh` grep gate. Only LAT registers the shared wrapper classes; GBT owns `CubicSplineWrap`. |
| **L3** — OrbitsView POD | `lisatools/cutils/orbits_view.hpp`; layout asserted at every LAT build via `binding_detector.cxx` `static_assert(sizeof + 15 offsetofs)`. |

## Backend implementation hierarchy

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow. When the GPU
   C++ changes, the other backends should be updated to match.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (e.g. `CUDA_SHARED`, `THREAD_START_X`,
   `BLOCK_INCR_X`) to bridge differences. The CPU path exists primarily
   for testing, validation, and use in CPU-only environments. It must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.** When
   validated against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), the CPU C++ output
   should match to machine precision (≤ 1e-15 mismatch) in direct
   modes. Cache/approximation modes have explicit error budgets
   documented per-feature.

4. **JAX may diverge internally** — design it to be as JAX-efficient as
   possible. JAX-CPU and JAX-GPU compilation targets are allowed to
   differ if it improves efficiency on each. JAX should NOT
   mechanically translate the C++ kernel structure (shared memory,
   block tiling, register caches); instead it should use JAX-native
   idioms:
   - `jax.lax.scan` for outer loops, `jax.vmap` for inner batched work
   - `jax.lax.dynamic_slice` / `dynamic_update_slice` with **static
     shapes** (mask out variable regions instead of slicing them)
   - Functional accumulator carries rather than in-place buffers
   - Host-precompute anything that needs `scipy.special` / iterative
     solvers JAX lacks

5. **JAX must match C++ inner-product outputs.** The end-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, the 5 swap_ll terms) must
   match the CPU/GPU C++ to floating-point precision (reldiff ≲ 1e-12)
   on representative test cases. Intermediate quantities (raw
   templates, per-chunk WDM coefficients) may differ at FP precision
   due to summation order — this is acceptable, validate at the
   inner-product level.

**Workflow for a new feature.** Implement on GPU C++ first → mirror to
CPU C++ via `#ifdef` → port to JAX with JAX-native idioms → validate
each backend against the others at the inner-product level before
merging.

## No new global-fit settings files (LISA Analysis Tools–wide rule)

Global-fit run configurations are **installed stock classes**
(`LISAanalysistools`, branch reorg-top-layer, 2026-07-09), not settings
files:

```python
from lisatools.globalfit.stock import erebor
fit = erebor.gb_no_fg(nwalkers=4)      # or get_stock("gb_no_fg", ...); also
fit.gb.center_freq = 8e-3              #   all_sources, full_year_combined
fit.recipe.pop_move("rj_refit")        # named move stacks per recipe stage
fit.build(); fit.run()                 # heavy work only on command
```

A new run variant is a `StockGlobalFit` subclass under
`lisatools/globalfit/stock/erebor/variants/` (knobs = documented dataclass
fields; env vars resolve as field defaults, explicit kwargs win; nothing
heavy in `__init__`; the pre-build fit must pickle/deepcopy). The
`LISAanalysistools/global_fit_input/*.py` files for migrated variants are
compatibility stubs — do not grow them back. Full rules + architecture in
`LISAanalysistools/CLAUDE.md` ("Stock global fits").

## No backend strings as function kwargs (LISA Analysis Tools–wide rule)

When a class or function needs a compute backend (CPU C++, CUDA C++,
JAX, ...), the backend choice MUST be made at object instantiation,
not as a keyword argument on individual methods. Concretely:

- **Allowed:** ``MyClass(force_backend="cpu")`` /
  ``MyClass(force_backend="cuda12x")`` / ``MyClass(force_backend="jax")``.
  Construct subclasses of :class:`FastLISAResponseParallelModule` /
  :class:`LISAToolsParallelModule` etc. and use ``self.backend`` /
  ``self.backend.xp`` / ``self.backend.name`` to dispatch internally.
- **Forbidden in method signatures:** ``backend="jax"`` / ``backend="cpp"`` /
  ``use_cpp=True`` / ``use_jax=True``. Method names MAY carry a backend
  suffix (e.g. ``get_ll_grad_jax``) when the implementation is
  intrinsically tied to that backend, but the choice of which method
  to call belongs to the caller -- not a runtime kwarg.

Rationale: one instance = one backend, so its arrays / kernels /
dispatchers stay consistent. Mixing backends per call leads to
ambiguous ownership of ``xp`` arrays, surprise host↔device copies, and
brittle dispatch logic.

This rule applies to **this folder and all sub-repos** (lisa-on-gpu,
LISAanalysistools, GPUBackendTools, BBHx, FastEMRIWaveforms, GBGPU,
Eryn, plus any repo-root scripts across the LISA Analysis Tools repos).

## Host→device upload of class-wrapper objects (LISA Analysis Tools–wide rule)

Pybind11 wrapper classes in this codebase (`OrbitsWrap`,
`TDIConfigWrap`, `WDMSettingsWrap`, `WDMDomainWrap`, `FDDomainWrap`,
`AnalysisContainerArrayWrap`, …) store their underlying C++ instance
via plain ``new`` on the **host** heap, e.g.

```cpp
class OrbitsWrap : public ReturnPointerBase {
    Orbits *orbits;
    OrbitsWrap(...) {
        orbits = new Orbits(..., _ltt_arr_device_ptr, ...);
        //       ^^^^^^^^^^ host allocation; pointer fields inside
        //                  may already point to device memory.
    }
};
```

The pointer fields inside the struct (e.g. ``Orbits::ltt_arr``,
``WDMDomain::wdm_data``, ``TDIConfig::unit_starts``) are device
pointers extracted from cupy arrays via ``return_pointer_and_check_length``.
But **the struct itself lives on the host**.

A CUDA kernel parameter of type ``Orbits *`` therefore cannot be the
host pointer ``orbits_wrap->orbits`` directly. Dereferencing it from
device code (``orbits->ltt_t0``) reads garbage and triggers an illegal
memory access -- typically with a faulting address in the canonical
Linux PIE/heap range (``0x55555...``) and a sanitizer message of the
form "X bytes after the nearest allocation" with a wildly OOB delta
(tens of TB). That delta is **not** an off-by-one; it means the device
dereferenced a host address.

The required upload pattern (mirrors the canonical implementation in
``LISAanalysistools/src/lisatools/cutils/LISAResponse.cu``):

```cpp
#ifdef __CUDACC__
    Orbits *orbits_gpu = nullptr;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, orbits, sizeof(Orbits),
                         cudaMemcpyHostToDevice));

    TDIConfig *tdi_config_gpu = nullptr;
    gpuErrchk(cudaMalloc(&tdi_config_gpu, sizeof(TDIConfig)));
    gpuErrchk(cudaMemcpy(tdi_config_gpu, tdi_config, sizeof(TDIConfig),
                         cudaMemcpyHostToDevice));

    // ...repeat for every host-side wrapper struct accessed on device:
    //    WDMSettings, WDMDomain, FDDomain, etc.

    my_kernel<<<...>>>(orbits_gpu, tdi_config_gpu, ...);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(orbits_gpu));
    gpuErrchk(cudaFree(tdi_config_gpu));
#else
    // CPU branch keeps the host pointers unchanged.
    my_kernel(orbits, tdi_config, ...);
#endif
```

Rules:

1. **Every** struct constructed via ``new`` on the host that the kernel
   dereferences (i.e. reads scalar fields or pointer fields off of
   ``this``) must be copied to device with ``cudaMalloc`` +
   ``cudaMemcpy(..., cudaMemcpyHostToDevice)`` before the kernel
   launch.
2. The device-side pointer fields *inside* the uploaded struct survive
   the shallow copy; do **not** also try to upload those.
3. Free the device-side struct copies after the kernel sync, before
   returning.
4. The CPU branch (``#else``) does not copy -- it passes the host
   pointer directly into the (host-compiled) kernel.
5. This applies to every CUDA wrapper across the LISA Analysis Tools repos --
   ``LISAanalysistools``, ``GPUBackendTools``, ``BBHx``,
   ``FastEMRIWaveforms``, ``GBGPU``, etc. Existing legacy kernels in
   LAT's ``LISAResponse.cu`` and ``Detector.cu`` already follow it; new
   chunked-het / chunked-FD / WDM impl wrappers must do the same.

When debugging an IMA whose faulting address starts with
``0x55555...`` and whose "nearest allocation" delta is in the TB
range, the first hypothesis should be a missing wrapper upload --
not an indexing bug in the kernel.

## CPU/GPU class-name aliasing (LISA Analysis Tools–wide rule)

Every C++ class that is compiled into **both** the CPU and the GPU
shared object (one per backend wheel) MUST have a per-backend
``#define`` alias at the top of the header that declares it, so the
two builds emit **distinct C++ type names** for the same logical
class. This applies to **two** layers:

**(a) The nanobind wrapper classes** -- anything passed to
``nb::class_<...>(m, "...")``. The alias block lives at the top of each
binding header, e.g. ``GBGPU/src/gbgpu/cutils/binding_gbgpu.hpp`` (GB wraps)
and ``BBHx/src/bbhx/cutils/binding_bbhx.hpp`` (SOBBH/MBH wraps):

```cpp
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#define GBTDIonTheFlyWrap         GBTDIonTheFlyWrapGPU
#define SOBBHTDIonTheFlyWrap      SOBBHTDIonTheFlyWrapGPU
#define FDSplineTDIWaveformWrap   FDSplineTDIWaveformWrapGPU
#define TDSplineTDIWaveformWrap   TDSplineTDIWaveformWrapGPU
#define WaveletLookupTableWrap    WaveletLookupTableWrapGPU
#define WDMSettingsWrap           WDMSettingsWrapGPU
#define WDMDomainWrap             WDMDomainWrapGPU
#define FDDomainWrap              FDDomainWrapGPU
#define GBComputationGroupWrap    GBComputationGroupWrapGPU
#define SOBBHComputationGroupWrap SOBBHComputationGroupWrapGPU
#else
// ...CPU suffixes...
#endif
```

**(b) The underlying C++ classes** that the wrappers hold a pointer
to -- ``Orbits``, ``WDMSettings``, ``WDMDomain``, ``TDIConfig``,
``GBTDIonTheFly``, ``GBComputationGroup``, etc. Same block pattern,
at the top of each header that declares them:

- ``LISAanalysistools/src/lisatools/cutils/Detector.hpp`` aliases
  ``Orbits`` → ``OrbitsGPU`` / ``OrbitsCPU``; LAT's ``domains.hpp`` /
  ``lat_tdi_on_the_fly.hh`` alias ``WDMSettings``, ``WDMDomain``,
  ``FDDomain``, ``LISATDIonTheFly`` and the FD/TD spline classes.
- ``GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.hh`` aliases
  ``GBTDIonTheFly`` + ``GBComputationGroup``;
  ``BBHx/src/bbhx/cutils/sobbh_tdi_on_the_fly.hh`` aliases
  ``SOBBHTDIonTheFly`` + ``SOBBHComputationGroup``.
  (``WaveletLookupTable`` was retired in Phase 3L.3.)

After preprocessing, the GPU build defines ``class WDMSettingsGPU``
and the CPU build defines ``class WDMSettingsCPU``: distinct C++
types with distinct ``typeid``s and distinct mangled symbol names.

Rules:

1. **Every class -- wrapper or underlying -- that ends up in both
   shared objects must appear in the relevant header's ``#define``
   block, with both GPU and CPU branches.** When you add a new
   class to a backend-shared header, add it to the block in the same
   commit.
2. **Both branches of the ``#if/#else`` must have the same set of
   entries.** A missing CPU- or GPU-branch entry (e.g. an alias
   present only on the GPU side) silently produces a backend-asymmetric
   class name, which is exactly the situation the rule prevents.
3. The pybind11 registration line ``py::class_<FooWrap>(m, "FooWrapGPU"
   / "FooWrapCPU")`` in the ``.cxx`` binding source must be guarded by
   the same ``#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)``
   toggle so the Python-visible name tracks the C++ alias.
4. **Inheritance only works through the alias if both the base and
   derived class names are in the ``#define`` block.** Example:
   ``class WDMDomain : public WDMSettings`` works correctly only when
   *both* ``WDMSettings`` and ``WDMDomain`` are aliased; otherwise the
   GPU build links ``WDMDomainGPU`` against the (still-unaliased)
   ``WDMSettings``, while the CPU build links ``WDMDomainCPU`` against
   the same ``WDMSettings`` -- the base type collides across the two
   shared objects even though the derived names differ.
5. Plain helper structs that never escape a single translation unit
   (e.g. a file-static ``OrbitsSplineCache``) do NOT need aliasing --
   only types whose symbols end up in the .so's exported interface
   (held by wrappers, referenced by pybind11, instantiated by
   templates exported from the shared object) need it.

**Rationale: ensures we do not duplicate imported symbols across the
CPU and GPU imports.** Both backends ship as separate plugin wheels
(``lisaanalysistools-cuda12x``, ``-cpu``, …) that load into the same
Python interpreter. If both shared objects declare ``class
WDMSettings``, they emit the same mangled C++ symbols and the same
``typeid``. Effects:

- pybind11's global type registry, keyed by ``typeid``, sees a
  collision: the second registration is rejected or silently shadows
  the first.
- The dynamic linker may resolve one shared object's call to
  ``WDMSettings::method`` against the *other* shared object's vtable,
  producing wrong-arch device calls or stack corruption.
- Inheritance edges registered with pybind11 reference the wrong
  base typeid, breaking ``isinstance`` / downcasts at the Python
  layer.

Aliasing forces every backend-specific symbol to be distinct end to
end, so the CPU and GPU plugin wheels are ABI-independent and
side-loadable in the same process. This is what allows
``has_backend("cpu")`` and ``has_backend("cuda12x")`` to both be true
simultaneously.

## Deepcopy / pickle safety (LISA Analysis Tools–wide rule)

Objects in this codebase routinely travel through ``copy.deepcopy``
(``CurrentInfoGlobalFit`` deepcopies the whole settings tree) and
``pickle`` (MPI workers, multiprocessing, cached states). Any object
graph that transitively holds a raw Python **module** dies in both with
``TypeError: cannot pickle 'module' object``. Rules:

1. **Never store an array module as an instance attribute.**
   ``self.xp = cp`` / ``self._xp = cp if use_cupy else np`` /
   ``self.xp = other.xp`` are all forbidden. Store the *flag or name*
   and expose the module as a **property**:

   ```python
   @property
   def xp(self):
       return cp if self.use_cupy else np      # flag-derived
   # or, for backend-managed classes (preferred):
   @property
   def xp(self):
       return self.backend.xp                  # Backend resolves by name
   ```

   ``LISAToolsParallelModule`` / ``ParallelModuleBase`` subclasses get
   this for free — ``self.backend`` is itself name-resolved
   (``_backend_name`` string) and ``Backend`` pickles as a registry-name
   reference (``Backend.__reduce__``, GBT ``bd6355c``) and
   deepcopies to itself (singleton). Do not bypass that by caching
   ``self.xp = self.backend.xp`` at construction.

2. **``__getattr__`` delegators must guard their storage attribute and
   dunders.** ``copy`` / ``pickle`` probe half-constructed objects for
   hooks (``__deepcopy__``, ``__reduce_ex__``, ``__getstate__``,
   ``__setstate__``) *before* ``__init__`` has run; an unguarded
   ``return getattr(self._inner, name)`` recurses to RecursionError.
   Pattern (see ``_GPUPriorWrapper``):

   ```python
   def __getattr__(self, name):
       if name == "_inner" or (name.startswith("__") and name.endswith("__")):
           raise AttributeError(name)
       return getattr(self._inner, name)
   ```

3. Other known-unpicklable attributes (nanobind wraps like
   ``OrbitsWrap`` / comp objects, open HDF5 files, cupy streams) must
   not live on objects that enter the settings tree or cross process
   boundaries — hold them on runtime-only objects, rebuild them lazily
   (property), or keep the picklable *args* and reconstruct.

When adding a class that will live in a settings tree, prove it with
``pickle.loads(pickle.dumps(copy.deepcopy(obj)))`` in its unit test.

## Narrowband mismatches mm2 / mm5 (chunked-het / WDM validation)

When verifying a chunked-heterodyne or other narrowband WDM template
against a lisatools reference signal, the canonical narrowband
mismatches are:

- **`mm5`** -- "5-layer" mismatch over a 5-m-layer band around the
  carrier `f0`. The band is defined by frequency bounds
  `[f0 - 3*layer_df, f0 + 2*layer_df]` (slightly asymmetric to cover
  the spectral tails on the side where the WDM transform spreads). Use
  this as the **primary** chunked-het accuracy metric -- it captures
  the dominant carrier + first-neighbour m-layers.

- **`mm2`** -- "2-layer" mismatch over just `m_floor` and `m_floor + 1`
  (the two layers that hold the bulk of a near-monochromatic GB
  signal). Band bounds: `[(m_floor - 0.5)*layer_df,
  (m_floor + 1 + 0.5)*layer_df]`. Use this as a tighter check
  isolating the carrier itself; it strips away spectral-tail
  contributions.

Both are **`1 - normalized overlap`**:

```python
mm = 1 - <d|h> / sqrt(<d|d> <h|h>)
```

via `AnalysisContainer.template_inner_product(..., normalize=True)`,
after slicing both `data` and `template` to the same narrow band by
building a per-binary `WDMSettings(min_freq=..., max_freq=...)` and
reusing the parent grid for layer-index alignment.

The canonical implementation lives in
`gb_chunked_prior_draws.py:283-340` (the `mm5` and `mm2` blocks).
SOBBH and other source-class versions should mirror the same band
definition for direct cross-source comparison.

Acceptance thresholds (current chunked-het with N_cp_sig=48,
N_cp_orbit=32, half-day wavelets, full angular prior):
- median mm5 ~ 1e-9, 90% < 8e-9, 99% < 3e-7
- low-frequency (m_floor < 100) sources occasionally show mm5 ~ 1e-7
  due to spectral-tail extension below ind_min_f -- documented
  systematic, not a bug.

## No nested OpenMP in compute kernels (LISA Analysis Tools–wide rule)

Do NOT add OpenMP (or any nested threading) inside C++/CUDA compute kernels
anywhere across the LISA Analysis Tools repos.

**Why:** threading is owned at the RUN level -- `OMP_NUM_THREADS` and the
`AnalysisContainerArray` `n_splits` / `run_threaded` per-split machinery.
Nested kernel threading fights that pool and makes run-level thread budgeting
meaningless.

**How to apply:** when a CPU kernel is slow, fix it algorithmically (restrict
work windows, replace naive DFT loops with FFTs, fold sums analytically) or
move it to the GPU path -- never by adding `#pragma omp` inside the kernel.

## Cross-wheel C++/CUDA sharing: recompile-in-place (LISA Analysis Tools–wide rule)

When a downstream wheel (GBGPU, BBHx, FEW) needs C++/CUDA symbols from an
upstream wheel (GPUBackendTools, LISAanalysistools), **recompile in place
against the upstream's headers -- do NOT link against the upstream's compiled
archive.**

**Why not "install + link":** `nvcc -dlink` doesn't cross `.so` boundaries
cleanly; link-time coupling would force every downstream's (compiler,
libstdc++, CUDA runtime, nanobind) to match the upstream exactly (≈4× CI
matrix); and CPython loads extensions `RTLD_LOCAL`, so an `RTLD_GLOBAL` reload
collides with side-loaded cuda11x/12x/13x plugin wheels. Per-wheel `nvcc` cost
for a duplicated `.cu` is 5–15 s + 100–300 KB -- negligible against a
multi-MB wheel.

**Three hardening points pair with this rule:**
1. **`LISATOOLS_HEADER_ABI_VERSION`** (+ GBT analog) -- monotonic macro in each
   exported public header; downstream binding source `static_assert`s the
   pinned value. Bump on any struct-layout change. First hypothesis for a
   cross-wheel cast failure is a stale downstream built against an older ABI
   version -- rebuild downstream before further diagnosis.
2. **POD `*View` structs as the primary cross-wheel interface** (`OrbitsView`,
   `WDMSettingsView`, …): downstream kernels consume a stable-layout POD by
   const-ref, so **downstream wheels don't recompile `Detector.cu` /
   `LISAResponse.cu` at all -- only LAT compiles them.** Biggest win of the
   scheme; prefer pushing a kernel's cross-wheel API into a `*View` POD.
3. **`ccache` + `CMAKE_CUDA_COMPILER_LAUNCHER=ccache`** in CI so any genuinely
   duplicated `.cu` compiles once per job.

**How to apply:** in a downstream CMakeLists, `target_include_directories(...
${GBT_CUTILS} ${LISATOOLS_CUTILS})`; when a needed host-side symbol lives in a
`.cu` not a header, first try to wrap it behind a `*View` POD, and only if that
doesn't fit, copy the `.cu` into the downstream build (with a comment pointing
back to the canonical LAT/GBT source) and bump the ABI version whenever that
canonical source changes.
