# Multi-GPU audit — the F-stat proposal machinery

**Date:** 2026-08-03
**Status:** AUDIT + PLAN ONLY. Nothing in this document has been implemented;
no source file was modified for it. The plan below is for review.
**Machine:** laptop, no GPU. Everything here is static analysis of the current
working tree (LAT `6c01ddd` + local uncommitted work, GBGPU `9f0fac7`).

---

## Executive summary

| # | Finding | Severity |
|---|---|---|
| 1 | The comb scan **does** see the full residual. Sharding splits *walkers*, never frequency, so a shard holds complete residuals for its own walkers. No correctness bug from partial data. | none (clears a suspicion) |
| 2 | The slab-holder rejection in `route_fstat_ll` is a **correct, load-bearing guard**, not a papered-over bug. Without it the kernel would read per-slot slab origins with intra-shard row indices — a silent wrong answer. | none (guard is right) |
| 3 | The in-fit F-stat **cannot parallelise at all**: every row carries `data_index = walker_ref`, a single walker, so 100 % of the work lands on one shard. Routing is real but never spreads. | perf, but the stage is ~2 % of an iteration |
| 4 | **The live defect:** the GB comps hold device-sticky state (orbits wrap, TDI-config wrap, chunk geometry, WDM window) allocated once on the run's main device. When the reference walker lives on a non-primary shard the F-stat kernel launches under `device_context(1)` and dereferences device-0 pointers. Correct only under P2P; **fails on a non-P2P node**. | **HIGH** |
| 5 | `run_fstat_rj_search.py` builds the birth proposal's cupy tables **before** the run pins its device, so with `GPUS=2` the grid lands on device 0 while everything else runs on device 2. | **HIGH** (script/runbook) |
| 6 | `StackedFStatProposal4D.from_cache(use_cupy=True)` has no way to express a target device. | design gap |

Findings 4 and 5 are the two things that would actually bite a multi-GPU
`GB_MODE=search` run. Finding 3 is real but low-value to fix.

---

## (a) Does the F-stat comb scan see the full residual?

**Yes.** The suspicion does not survive contact with the sharding model.

`AnalysisContainerArray` shards **containers**, not the frequency axis.
The parent residual ACA is built one `AnalysisContainer` per *walker*:

- `globalfit/run.py:911-932` — `acs_tmp = [_build_walker_ac(w) for w in
  range(self.nwalkers)]`, then `AnalysisContainerArray(acs_tmp, gpus=gpus,
  run_threaded=...)`.
- `analysiscontainer.py:1845-1905` — `gpu_splits` is a contiguous
  `np.array_split` over container rows; `split_map[row]` / `gpu_map[row]`
  record ownership.

Each container carries that walker's **entire** residual over the whole
analysis band. So shard 1 holding walkers 8–15 holds eight complete residuals,
not the upper half of a spectrum. A per-walker F-stat scan on shard 1 sees
everything it needs.

The in-fit reference walker is chosen globally and then routed:

- `gbspecialstretch.py:1904-1907` — `_fstat_reference_walker` returns
  `int(np.argmax(_to_numpy(model.analysis_container_arr.likelihood())))`, a
  **global** row id.
- `gbspecialstretch.py:2261-2281` — `_fstat_NM` builds
  `di = xp.full(n, walker_ref)` and calls
  `_RoutedBandEngine.route_fstat_ll(comp_method, holder, params_phys,
  data_index=di, noise_index=di, convert_to_ra_dec=False)`.
- `gbbands.py:700-735` — the router partitions by `split_map[data_index]`,
  translates to intra-shard rows, runs each shard inside
  `device_context(xp, view.device)`, and host-assembles `(N, M)`.

Row-index translation is done by `_partition` (`gbbands.py:388-418`) from
`holder.gpu_splits`, and `_ShardHolderView` (`gbbands.py:205-332`) re-presents
one split as a single-shard holder. That is all correct.

**The standalone grid-prep path never shards at all.**
`plot_fstat_proposal_mojito.py` passes the ACA straight to
`get_fstat_ll_wdm`, and `chunked_het.py:414-424` raises
`NotImplementedError` on any holder with `len(linear_data_arr) != 1`. So a
`GPUS=0,1` grid prep crashes at the first probe rather than producing a
partial-residual grid. Loud, not silent — acceptable, though see (d.4).

## (a′) Is the slab-holder rejection hiding a bug?

**No — it is a correct guard, and it is load-bearing.**

```python
# gbbands.py:709-714
if getattr(holder, "slab_min_f", None) is not None:
    raise NotImplementedError(
        "route_fstat_ll does not support narrow per-band slab holders ...")
```

Why it is necessary:

1. `slab_min_f` / `band_slab_Nf` are **`SubBandBuffer` properties**
   (`gbbands.py:1147` and `gbbands.py:1196`) indexed by *global buffer slot*.
   The parent residual ACA has neither — verified by an unbounded grep across
   `LISAanalysistools/src` and `GBGPU/src`: the only definitions are those two
   properties, the only consumers are `chunked_het.py:334-366`
   (`_slab_kernel_args` / `_slab_args_from`) and
   `_RoutedBandEngine._PER_SLOT_KWARGS` (`gbbands.py:355`).
2. `_ShardHolderView` does **not** override `slab_min_f`. Its `__getattr__`
   (`gbbands.py:325-332`) forwards any non-underscore attribute to the parent,
   so a view would hand back the **full-length, global-slot** array while the
   kernel is being given **intra-shard** `data_index` values. Every binary
   would then read the wrong slab origin.
3. `fill_template` handles exactly this by slicing per-slot kwargs to the
   shard's rows (`gbbands.py:465-473`,
   `kw_s[k] = xp.asarray(host_vals[view.rows])`). `route_fstat_ll` cannot do
   the same because the comp reads the metadata **off the holder**
   (`chunked_het.py:346-351`), not from a kwarg.

So the guard converts a silent-wrong-answer into a loud error, and the
docstring's justification ("F-stat runs on the parent residual ACA, which
carries no slab metadata") checks out. **Keep it.** If anything, the same
guard belongs on `route_information_matrix`, which has no such check — see
(d.5).

---

## (b) Does the F-stat sweep parallelise across devices?

**No. Structurally it cannot, as written.**

`_fstat_NM` scores every candidate against **one** reference walker
(`gbspecialstretch.py:2262`, `di = xp.full(params_phys.shape[0],
int(walker_ref))`). `_partition` therefore puts every row in a single shard's
bucket, and the loop at `gbbands.py:721-731` executes exactly one non-empty
iteration. On N GPUs, N−1 sit idle for the whole F-stat stage. The commit
that added the route recorded this itself ("fstat scores vs ONE reference
walker → all rows one shard → routing not spreading").

**How much work could go wide — quantified.**

The in-fit surface is `_fstat_NM` only, reached from `_fstat_dist_centers`
(`gbspecialstretch.py:2283-2306`) at four call sites: `2521`, `2546`
(RJ birth/death factors) and `2808`, `2825` (the replacement move). All four
are inside the `rj_step` timing span (`gbspecialstretch.py:2124`).

From the run-2 profile (6–8 mHz search, A100, 16w × 6t, 3 036 s/iter mean):
`inmodel_cholesky` 1 739 s (57 %), `get_add_ll` ~31 %, **`rj_step` 57 s
(1.9 %)** — and `rj_step` also covers the RJ proposal draw, the add/remove
scoring and the accept step, so the F-stat share is a *fraction* of 1.9 %.

**A perfect N× on this stage buys well under 2 % of an iteration.** That is
the single most important number in this audit: *do not spend effort making
the F-stat multi-GPU-fast.* Make it multi-GPU-**correct** (section d) and
stop.

Two ways it *could* go wide, recorded for completeness only:

* **Broadcast the reference slab.** Copy the reference walker's residual +
  invC slab to every device once per proposal, then split the binary rows.
  Cost: one broadcast of `nch·Nf_active·Nt_active` complex + the invC block,
  per proposal. Payoff ≲ 2 %. Not worth it.
* **Score each candidate against its own walker** (`di = the candidate's
  walker id`). Rows then spread naturally and the routing already handles it.
  But this is a **science change**, not a perf fix — `_fstat_reference_walker`
  deliberately uses the max-likelihood walker to mirror the serial-search
  move, and the proposal centre would become walker-dependent. Flagging only;
  not recommending.

---

## (c) Device-context bugs analogous to the all_sources ones

**Yes — one whole class of them, and it is the same class that
`project_multigpu_allsources_hardening` fixed for the source generators but
never fixed for the GB comps.** That memory's own "known limits" line already
says "comp-level device-sticky caches are a cluster-smoke watch item". This
section closes that watch item: they are real.

### c.1 The comps pin their device at construction and never record it

`GBWDMComputations` (via `WDMComputationsBase.__init__`) allocates on
whatever CUDA device happens to be current at build time:

| state | site | what it is |
|---|---|---|
| `chunk_t_starts` | `chunked_het.py:190-192` | cupy float64 |
| `chunk_keep_lo` / `chunk_keep_hi` | `chunked_het.py:193-194` | cupy int32 |
| `chunk_n_global_offset` | `chunked_het.py:195-196` | cupy int32 |
| `wdm_window` | `chunked_het.py:197-200` | cupy float64 |
| `cpp_wdm_settings` | `chunked_het.py:219-229` | `WDMSettingsWrap` — scalars only, **safe** |
| `cpp_tdi_config` | `chunked_het.py:245` | `TDIConfigWrap` built from `pytdiconfig_args`, which are six cupy arrays (`response/tdiconfig.py:99-108`) |
| `cpp_orbits` | `chunked_het.py:281` | `OrbitsWrap` built from `pycppdetector_args`, six cupy arrays (`detector.py:464-478`) |

`GBFDComputations` is config-only but holds the same two wraps
(`GBGPU/src/gbgpu/gbcomps.py:189-190`, `:203-204`).

The comp is built **once**, at recipe/variant build time, with no device
context — `recipe.py:1250-1268` (`GBFillGlobalSignalGen._comp`) and the
gb_no_fg variant path — and cached on the settings tree
(`recipe.py:1268`, `si.gb_wdm_comp = comp`).

An exhaustive grep over `chunked_het.py`, `gbcomps.py`,
`gbsignalhetcomputations.py` and `signal_het.py` for `Device(`, `setDevice`,
`device_context`, `gpus`, `device` found **zero** device pinning. The only
hits are `_as_wdm_holder` / `_as_fd_holder` *reading* `arr.device.id` off a
caller's array (`chunked_het.py:429-431`, `gbcomps.py:241-243`) to forward a
`gpus=[...]` kwarg — a call-time inference, not stored state.

### c.2 Why this fires on the F-stat path specifically

`route_fstat_ll` launches inside `device_context(xp, view.device)`
(`gbbands.py:724`). `get_fstat_ll_wdm` then passes
`self.cpp_orbits, self.cpp_tdi_config, self.cpp_wdm_settings`,
`self.xp.asarray(self.chunk_t_starts)`, `self.xp.asarray(self.wdm_window)`
… into the kernel (`chunked_het.py:1125-1147`).

`cupy.asarray` on an array that is *already* a cupy ndarray returns it
unchanged — it does **not** migrate across devices. So a launch on device 1
gets device-0 pointers for the chunk geometry and the window, and the
`OrbitsWrap`/`TDIConfigWrap` structs (uploaded per launch per the LAT-wide
convention) still carry device-0 pointer *fields*.

Consequence: **peer access, or failure.** With P2P enabled the numbers are
right and the traffic is a silent tax; on a node without P2P between the two
devices the launch is an illegal access. This is precisely the failure mode
`project_multigpu_allsources_hardening` warned about for FEW/bbhx generator
tables ("makes run P2P-independent — currently would FAIL on non-P2P nodes").

**Reachability.** `_fstat_reference_walker` is `argmax` over the walker
likelihoods. With `nwalkers` split contiguously across two GPUs, roughly half
of all proposals put the reference on the non-primary shard. This is not an
edge case.

**Why the 2-GPU smokes never caught it.** Those runs were `all_sources` with
`GB_MODE` unset (no `rj_fstat_dist_birth`) — and, at the time, a TEMP override
`_rj_birth_prop = gpu_priors` kept the F-stat container out of the birth path
entirely. That override is **gone** in the current tree (`recipe.py:2439-2448`
now uses `{"gb": _custom_birth}` when set), so the path is live and unexercised
on 2 GPUs.

### c.3 The same defect affects every routed GB call, not only the F-stat

`route_information_matrix` (`gbbands.py:642-678`) and the instance router's
`get_ll` / `get_swap_ll` / `fill_template` (`gbbands.py:451-572`) all launch
under a shard's device context against the same single-device comp. The
2-GPU smoke's bit-identical lnL says the *data plane* is exact; it does not
say the run is P2P-free. Fixing the comp fixes all of them at once — which is
why the recommendation below is comp-level, not F-stat-level.

### c.4 JAX / orbits analogues

No JAX involvement on the GB comp path (grep for `jax` in `chunked_het.py`
returns only the `_BACKEND_PREFIX + "_jax"` backend-name test at
`chunked_het.py:1103`), so no `jax_device_context` is needed here — unlike
MBH. The orbits analogue *is* needed and is exactly what
`source_runtime.py::_device_local_orbits` already implements for the source
generators.

### c.5 A second, independent device bug in the runner script

`scripts/fstat_proposal/run_fstat_rj_search.py`:

- `:387-391` — `birth = build_birth_distribution(fit, ...)`
- `:215-219` — inside it, `StackedFStatProposal4D.from_cache(cache, ...,
  use_cupy=use_cupy)`
- `sampling/fstat_proposal.py:729-732` — `import cupy as _cp; grids =
  _cp.asarray(np.asarray(d["logp_grids"], dtype=float))` — **a bare upload,
  no device argument, no context**
- `:403` — `curr = fit.build()`, which is the first thing that reaches
  `run.py:809`'s `pin_main_device(xp, general_info.gpus)`

So the birth grid is uploaded **before** the run pins its device: it lands on
device 0 regardless of `GPUS`. With `GPUS=2` every `rvs`/`logpdf` during
sampling then mixes device-0 grid tables with device-2 coords.
`_resolve_use_cupy` (`run_fstat_rj_search.py:87-96`) *does* compute
`gpu_index = _gpus[0]`, but it is only forwarded to the optional GMM fit
(`:245-247`) — never to `from_cache`, which is the production default
(`FSTAT_PEAK_SAMPLING=grid`).

`plot_fstat_proposal_mojito.py` is safe by accident: `pin_main_device` runs
inside `gf.setup_acs()` before the comp/holder are built (`:224`, `:266-292`),
and there is no re-pinning afterwards.

---

## (d) Proposed fix — design and patch sketch

Ordered by value. **D1 is the only one that matters; D2–D3 are cheap
hardening; D4–D6 are optional.**

### D1 — Per-device GB comp replicas (the real fix)

Mirror the idiom already proven for the source generators:
`_device_local_orbits` / `_device_local_domain_settings` +
`_DEVICE_*_REPLICAS` module caches in
`globalfit/stock/erebor/source_runtime.py`, and the DCGA `build_cpp_objects`
pattern.

**What gets replicated:** the whole comp object, once per non-primary device.
Everything on it is either a device buffer or a wrap over device buffers, and
it is cheap: the geometry arrays are `n_chunks`-sized and the WDM window is
`Nt_sub`-sized — kilobytes, not the stash.

**What stays shared:** nothing needs to. The primary device (`gpus[0]`) and
CPU **reuse the existing object**, byte-identical to today's behaviour and
zero extra memory — the same "primary reuses, non-primary replicates" rule
`_device_local_orbits` uses.

**Where it lives:** a new `_device_local_gb_comp(comp, xp, primary)` helper
next to `_device_local_orbits`, and a router-side resolver so every routed
call picks the replica for `view.device`.

Sketch — helper (new, `source_runtime.py`):

```python
_DEVICE_GB_COMP_REPLICAS = {}   # (id(comp), device) -> comp replica

def _device_local_gb_comp(comp, xp, device, primary):
    """Per-device replica of a GB comp (chunked or FD).

    Primary device / CPU reuse the shared object: byte-identical to the
    single-GPU path, no extra allocation. A non-primary device gets ONE
    cached replica, constructed inside its own device context so every
    cupy array and every *Wrap's pointer fields are device-local.
    """
    if device is None or primary is None or int(device) == int(primary):
        return comp
    key = (id(comp), int(device))
    hit = _DEVICE_GB_COMP_REPLICAS.get(key)
    if hit is not None:
        return hit
    with device_context(xp, int(device)):
        rep = comp.__class__(*comp.args, **comp.kwargs)   # see note
    _DEVICE_GB_COMP_REPLICAS[key] = rep
    return rep
```

*Note on `comp.args`/`comp.kwargs`:* `GBWDMComputations` does not currently
record its constructor arguments. `_device_local_domain_settings` solves the
identical problem by rebuilding from `settings.args` / `settings.kwargs`, so
the precedent is to **store them** — a two-line addition in
`WDMComputationsBase.__init__`. The `orbits` kwarg must be replaced by the
device-local orbits replica, exactly as `_device_local_domain_settings` drops
the device-resident window from its kwargs so `__init__` rebuilds it on the
owning device.

Sketch — router side (`gbbands.py`), one helper used by all three entry
points:

```python
@staticmethod
def _comp_for(comp, holder, view):
    """The device-local replica of ``comp`` for this shard's device."""
    from ...utils.device import device_context          # noqa: F401
    from ..stock.erebor.source_runtime import _device_local_gb_comp
    gpus = getattr(holder, "gpus", None)
    primary = None if not gpus else int(gpus[0])
    return _device_local_gb_comp(comp, holder.xp, view.device, primary)
```

and at each launch site, inside the existing `device_context`:

```python
# route_fstat_ll -- gbbands.py:721-731
for view, (pos, intra, intra_noise) in zip(views, parts):
    if pos.shape[0] == 0:
        continue
    with device_context(xp, view.device):
        m = getattr(cls._comp_for(comp_owner, holder, view),
                    comp_method.__name__)
        N_s, M_s = m(xp.asarray(params_host[pos]), view, ...)
```

`route_fstat_ll` currently receives a **bound method** (`comp_method`), so it
needs the owning comp too. Cleanest signature change:

```python
-def route_fstat_ll(cls, comp_method, holder, params_phys, *, ...)
+def route_fstat_ll(cls, comp, method_name, holder, params_phys, *, ...)
```

with `_fstat_NM` (`gbspecialstretch.py:2274-2281`) updated to pass
`(wdm_comp, "get_fstat_ll_wdm")` / `(self.gb_fd_comp, "get_fstat_ll_fd")`.
`route_information_matrix` already takes the comp itself, so it needs only the
one-line `comp = cls._comp_for(comp, holder, view)`. The instance router
(`get_ll`, `get_swap_ll`, `fill_template`, `_route_matrix`,
`setup_in_model`) reaches its comp through `self._engine.gb_comps` /
`.gb_fd_comp` and needs an engine-level equivalent — see the sig-het audit,
which needs the same machinery and should land in the same change.

**Memory-lifecycle rule.** Replicas are allocate-once, persist-for-the-run,
attached to a module-level cache — never on the settings tree (pickle safety),
and never freed at proposal teardown. Same contract as
`_DEVICE_ORBITS_REPLICAS`.

**Validation.**
1. CPU fake-shard test in `tests/test_gb_shard_router.py` (which already has
   15 tests and a `RecordingXp`): assert each shard's call receives the comp
   whose recorded construction device matches `view.device`, and that the
   primary shard receives the **same object** (`is`) as the unsharded path.
2. GPU: `GB_MODE=search` on `GPUS=0,1`; initial lnL must be bit-identical to
   `GPUS=0`; peer-access lines for the GB comps must vanish.

### D2 — Assert the comp's device instead of trusting P2P

Even with D1 the class of bug recurs the moment someone adds a new comp-level
buffer. A cheap permanent guard, in `route_fstat_ll` /
`route_information_matrix` / the instance router, right before each launch:

```python
if __debug__ and view.device is not None:
    d = getattr(getattr(comp.wdm_window, "device", None), "id", None)
    if d is not None and int(d) != int(view.device):
        raise RuntimeError(
            f"GB comp buffers live on device {d} but this shard launches on "
            f"device {view.device}: the kernel would read across devices "
            f"(silent P2P tax, or an illegal access on a non-P2P node). "
            f"Use a per-device comp replica.")
```

This is the single highest value-per-line item in the whole audit — it turns
finding 4 from "silently slow, or a mystifying IMA on some clusters" into a
message that names the fix. **It is deliberately NOT applied to the tree**
per the plan-only scope; it is the first thing to land if the plan is
approved.

### D3 — Fix the runner-script device ordering (finding 5)

Two independent one-line-class fixes; do **both**:

1. `run_fstat_rj_search.py`: move `build_birth_distribution(...)` from `:387`
   to **after** `curr = fit.build()` at `:403`, so `pin_main_device` has
   already run. (Check: `fit.gb.rj_birth_distribution = birth` at `:397` must
   move too — verify it is read at build time or later; if `build()` consumes
   it, use option 2 instead.)
2. `fstat_proposal.py::from_cache`: add an explicit `device=None` parameter
   and wrap the upload:

```python
-        if use_cupy:
-            import cupy as _cp
-            grids = _cp.asarray(np.asarray(d["logp_grids"], dtype=float))
+        if use_cupy:
+            import cupy as _cp
+            from ..utils.device import device_context
+            with device_context(_cp, device):
+                grids = _cp.asarray(np.asarray(d["logp_grids"], dtype=float))
```

and have `run_fstat_rj_search.py` pass the `gpu_index` it already computes at
`:87-96`. Option 2 is the more robust of the two because it fixes *any* caller
that builds the container outside the pinned window, which is the general
fragility the auditing agent flagged.

### D4 — Make the grid-prep multi-GPU story explicit

`plot_fstat_proposal_mojito.py` crashes with a `chunked_het.py:414` message
about the shard router if `GPUS` has >1 entry. That message is right but
doesn't tell a grid-prep user what to do. Add an early, explicit check in the
script: grid prep is single-device by design (it scans one walker's residual);
if `len(gpus) > 1`, either error with "grid prep is single-GPU; set GPUS=<one
device>" or silently take `gpus[:1]`. Low effort, prevents a 3-hour comb scan
dying at minute one.

### D5 — `route_information_matrix` and slab holders — CLOSED, but add the guard anyway

Chased to a conclusion during this audit. The chain is:

`route_information_matrix` → `comp.information_matrix`
(`chunked_het.py:1180`) → `_ar()` → `get_swap_ll_wdm`
(`chunked_het.py:1248-1251`) → **`*self._slab_kernel_args(wdm_holder)`
(`chunked_het.py:680`)**.

So a slab holder *would* reach the kernel's slab-origin arguments, and
`_ShardHolderView.__getattr__` would hand back the parent's **global-slot**
`slab_min_f` against **intra-shard** `noise_index` — the exact bug
`route_fstat_ll` guards against.

**It cannot happen today:** the one call site passes
`model.analysis_container_arr` (`gbspecialstretch.py:3374-3377`), the parent
residual ACA, which has no slab metadata. The path is safe by the caller's
choice, not by construction.

Recommendation: copy the four-line `slab_min_f is not None` guard from
`route_fstat_ll` (`gbbands.py:709-714`) into `route_information_matrix`. It
costs nothing and pins an invariant that is currently only maintained by one
call site's habit.

Related: under `GB_SIGHET_INMODEL=1`, `_info_comp` is a
`GBSignalHetComputations` (`gbspecialstretch.py:3369-3373`), whose
`information_matrix` forwards to `self.chunked`
(`gbsignalhetcomputations.py:691-692`). Any `_device_local_gb_comp` helper
must therefore handle the sig-het wrapper class as well as the raw comps —
see the sig-het audit.

### D6 — Router overhead (informational, no action)

Every multi-shard `route_fstat_ll` call does `asnumpy(params)` → host →
`xp.asarray` per shard, and the same for `(N, M)` on the way back
(`gbbands.py:718-735`). With all rows on one shard that is a pure D2H+H2D
round-trip that the single-shard passthrough avoids. At in-fit row counts
(hundreds) it is negligible. It would matter if the standalone sweep ever
routed — it does not.

---

## What was checked and found *absent*

Negative claims below were each verified with an unbounded grep across all
seven repos (`LISAanalysistools`, `GBGPU`, `BBHx`, `Eryn`, `lisa-on-gpu`,
`GPUBackendTools`, `FastEMRIWaveforms`) unless noted.

* `ShardHolderView` / `split_map` / `_run_per_split` / `_split_rows` exist
  **only** in `LISAanalysistools` (60 hits); zero in the other six repos.
* No `device_context`, `cuda.Device`, `setDevice`, `ThreadPool`,
  `concurrent.futures`, or `multiprocessing` anywhere in
  `plot_fstat_proposal_mojito.py`, `run_fstat_rj_search.py`, or
  `sampling/fstat_proposal.py`. The only `mpi4py` use is
  `GlobalFit(curr, comm)` bookkeeping.
* No device pinning of any kind in `chunked_het.py`, `gbcomps.py`,
  `gbsignalhetcomputations.py`, `signal_het.py`.
* `slab_min_f` / `band_slab_Nf` are defined **only** as `SubBandBuffer`
  properties; the parent residual ACA has neither.
* The `recipe.py` TEMP override that discarded the F-stat birth container
  (recorded as open in `project_fstat_proposal_distributions`) is **gone**;
  `recipe.py:2446-2448` now uses the custom container when set. That item can
  be closed.

---

## Suggested order of work

1. **D2** (assert) — one commit, immediate diagnostic value, no behaviour
   change on the working path.
2. **D3** (runner device ordering) — independent, script-level, unblocks a
   `GPUS=<non-zero>` search run today.
3. **D1** (per-device comp replicas) — the real fix; land it together with the
   sig-het per-shard replicas (see `2026-08-03-multigpu-sighet-audit.md`),
   since both need the same `_device_local_gb_comp` helper and the same
   engine-level resolver.
4. **D5** (mirror the slab guard) — four lines, closes an invariant that is
   currently maintained only by one call site's habit.
5. **D4** (grid-prep guard) — nice-to-have.

Do **not** invest in spreading the F-stat sweep across devices (section b):
the whole stage is under 2 % of an iteration.
