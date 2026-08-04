# Multi-GPU correctness for the GB F-stat + sig-het paths — implementation

> **Follow-up review, 2026-08-04 (main session).** Three additions after an
> independent read of the implementation:
>
> * **F-stat audit item D4 closed.** `plot_fstat_proposal_mojito.py` now
>   refuses `len(gpus) > 1` *before* `fit.build()` loads the mojito data,
>   naming the fix. Previously a multi-GPU grid prep sharded the parent ACA
>   and died at the first probe inside `chunked_het` with a shard-router
>   message that is correct but says nothing about grid prep — after the data
>   load. Grid prep is single-device by design (it comb-scans one walker's
>   full residual); multi-GPU applies to the in-fit F-stat, not to this
>   offline step.
> * **New regression test for the PRODUCTION F-stat call shape**
>   (`test_fstat_all_rows_on_one_non_primary_shard`). `_fstat_NM` sends every
>   row to a single walker — `di = xp.full(n, walker_ref)` — so exactly one
>   shard is non-empty, and the failure case is that shard being non-primary.
>   The existing dispatch test uses a *spread* `data_index`, which cannot
>   catch a partition bug that only appears when one shard is populated.
> * **`test_fstat_dispatches_to_device_local_comp` was weaker than it read.**
>   Its `seen = {...for c in comp.calls}` loop inspects only the PROTOTYPE's
>   call log, and a non-primary shard runs on a *replica* — a different
>   object with its own log — so that dict never contained more than the
>   primary shard and the loop was close to vacuous. It now walks
>   `_DEVICE_GB_COMP_REPLICAS` and asserts every shard that owned a row ran on
>   its own comp. The test's first assertion (the device stamped into the
>   output) was always the real proof and still passes unchanged.
>
> Suite: 30 → 31 tests, and 85 across all multi-shard suites, all passing.
> Still nothing run on a GPU.

**Date:** 2026-08-04
**Status:** IMPLEMENTED, in the working tree, **not committed**.
**Machine:** laptop, no GPU. Everything below is CPU-validated; every GPU
claim is UNVERIFIED and listed in §5.
**Implements:** `2026-08-03-multigpu-fstat-audit.md` (D1/D2/D3; D5 resolved
differently — see §6) and `2026-08-03-multigpu-sighet-audit.md` (C1/C3/C5/C6,
L1/L2/L4/L5/L6).

---

## 1. What changed, and why

### 1.1 `_ShardHolderView.slab_min_f` — a silent wrong answer, independent of everything else

`__getattr__` on the shard view forwards any non-underscore attribute to the
parent, so `slab_min_f` came back as the parent's **full-length, global-slot**
array while every index the engines pass is **intra-shard**. The live consumer
is `GBSignalHetComputations.setup_in_model`
(`np.asarray(buffer_aca.slab_min_f)[slots] - ind_min_f`, GBGPU
`gbsignalhetcomputations.py:520-523`), and the chunked-het kernels reach it
through `WDMComputationsBase._slab_kernel_args`
(`chunked_het.py:355-377`) on `get_ll` / `swap_ll` / `fill_global` /
`get_fstat_ll`. A shard-1 source at intra row 0 was being folded against
buffer slot 0's slab origin.

Fixed with an explicit property + a persistent per-shard store refreshed in
place, following the `min_freq_inds` pattern verbatim:

- `gbbands.py:288` — the `slab_min_f` property (docstring explains why it
  cannot delegate).
- `gbbands.py:251` — `_slab_min_f_view = None` initialiser.
- `gbbands.py:344-364` — the in-place refresh inside `refresh_row_metadata`,
  so a cell swap on the parent reaches every view.

`band_slab_Nf` is a scalar extent shared by all slabs, hence shard-invariant,
and keeps delegating (documented in the property docstring).

### 1.2 Per-device comp replicas — the shared fix for both audits

The comps are built once at variant build time
(`variants/gb_no_fg.py:931`, `variants/vgb.py:227/269`) and pin every buffer
they allocate to the device current at that moment. The router launches each
shard inside its own `device_context`, so a non-primary shard dereferenced the
primary device's pointers.

**Recording the reconstruction contract** (mirrors `Orbits.args/kwargs` and
`DomainSettingsBase.args/kwargs`, which is exactly what
`_device_local_domain_settings` already relies on):

- `chunked_het.py:128-155` — `_build_device`, `_ctor_args`, `_ctor_kwargs` in
  `WDMComputationsBase.__init__`; `chunked_het.py:277-292` — the `args` /
  `kwargs` properties.
- `GBGPU/src/gbgpu/gbcomps.py:124-141` — the same for
  `GBFDComputations.__init__`; `gbcomps.py:193-207` — its `args` / `kwargs`.

The recorded objects are ones the comp already holds, so nothing new reaches
the settings tree (pickle-safety rule).

**The replica helpers**, next to `_device_local_orbits` /
`_device_local_domain_settings` and structured identically:

- `source_runtime.py:198-240` — `_DEVICE_TDI_CONFIG_REPLICAS` +
  `_device_local_tdi_config`. **Not in either audit plan**, but required:
  `TDIConfig.__init__` uploads its six link/sign/channel tables at
  construction (`response/tdiconfig.py:99-108`) and the comp's
  `cpp_tdi_config = backend.TDIConfigWrap(*tdi_config.pytdiconfig_args)`
  captures those pointers verbatim. Handing the replica the SHARED TDIConfig
  would have left `cpp_tdi_config` pointing at the primary device — a
  half-fix. Rebuilt from `tdi_config.tdi_combinations`, the constructor's own
  input, so the replica is exact.
- `source_runtime.py:249-300` — `_DEVICE_GB_COMP_REPLICAS` +
  `_device_local_gb_comp(comp, xp, device, primary_device)`.
- `source_runtime.py:302-313` — `_SIGHET_REPLICA_KNOBS`, the
  `for_band_engine` knob name → `_g` key map.
- `source_runtime.py:315-355` — `_build_gb_comp_replica`, the three-way
  dispatch (sig-het wrapper / recorded-ctor comps / anything else returned
  shared).
- `source_runtime.py:357-368` — `_device_local_domain_settings_on`, the
  explicit-device spelling of the existing helper (the source moves call the
  current-device form from inside their shard context; the GB replica builder
  resolves a device it was handed).

Per the brief, the cache **value** is `(prototype, replica)` so a strong
reference keeps `id(prototype)` from being recycled. The existing
`_DEVICE_ORBITS_REPLICAS` was left alone.

The sig-het wrapper needs **no new state on `GBSignalHetComputations`**: every
`for_band_engine` knob is already recorded in `self._g`
(`nt_layer`, `n_sparse_fd`, `m_half`, `max_r`, `n_cp_build`, `v3_n_nodes`,
`v4_knots`, `v4_band`, `v5`), and both value resolutions the factory performs
(the `nt_layer` snap to a divisor of `Nt`, `_resolve_n_cp`) are idempotent, so
a rebuild from `_g` reproduces `_g` exactly. That file was not edited at all.

**Version coverage confirmed by reading (not edited):** `setup_in_model`
(`gbsignalhetcomputations.py:451-682`) contains no version branch — it builds
the same stash for v2/v3/v4, and the `get_ll` dispatch
(`:788-843`) differs only in which kernel consumes it plus which extra device
arrays are passed (`window_full` for v2, `n_sparse_local` for all,
`_v4_band_arrays` for v4). Replicating the comp therefore covers v2/v3/v4 and
the forthcoming v5 through one mechanism. No per-version code path exists in
this change.

### 1.3 Router wiring

- `gbbands.py:415-421` — `_RoutedBandEngine.__init__(engine, engine_factory=None)`
  + the lazily-populated `_engine_by_device` map.
- `gbbands.py:442-459` — `_comp_build_device`: `_build_device`, falling back
  to the sig-het wrapper's chunked delegate, falling back to actual
  `wdm_window` residency.
- `gbbands.py:461-467` — `_engine_comp` (`gb_comps` / `gb_fd_comp`).
- `gbbands.py:469-483` — `_primary_device`. **Departure from both audits:**
  the device a shard may reuse the shared comp on is the comp's OWN build
  device, not blindly `gpus[0]`. A comp constructed before the run pins its
  main device lives on device 0 even when `GPUS=2,3`; keying on the recorded
  value replicates for *every* shard (correct) instead of handing device-0
  pointers to the `gpus[0]` shard (wrong) — and it means the guard below can
  never fire spuriously on that configuration.
- `gbbands.py:485-507` — `_assert_comp_device`, the audit's D2 in Correction
  A's form (recorded build device, type-agnostic, no-ops when nothing is
  recorded, i.e. on CPU).
- `gbbands.py:509-520` — `_comp_for` (raw-comp classmethods).
- `gbbands.py:522-545` — `_engine_for` (instance router).
- `gbbands.py:968-1010` — `make_routed_band_engine`, the one module-level
  helper both construction sites now use. It builds the prototype engine with
  the *identical* arguments as before and adds the factory closure; the two
  sites previously duplicated the `_RoutedBandEngine(make_band_likelihood_engine(...))`
  expression, so this removes duplication rather than adding a layer.

Launch sites switched to the per-shard engine: `fill_template`
(`gbbands.py:~600`), `get_ll` (incl. `d_h_out` / `h_h_out` / `phase_angle` /
`kept_out`, which were being read off `self._engine` and would have returned
the WRONG shard's outputs once replicas existed), `get_swap_ll`,
`_route_matrix`, `setup_in_model`. Construction sites:
`gbbands.py:1305` (`SubBandBuffer`) and `gbspecialstretch.py:895` (the
move-level parent-ACA engine).

### 1.4 The sig-het `NotImplementedError`

`gbbands.py:752-799` — the blanket "multi-shard buffers need per-shard comp
replicas" raise is gone. It is replaced by a *precise* guard: track which
engine object produced each truthy `setup_in_model`, and raise only if two
shards resolved to the **same** engine (i.e. no `engine_factory` was supplied,
or the router could not tell the devices apart). That keeps the landmine shut
for any un-wired construction site while opening the path for the wired ones.

`clear_in_model` (`gbbands.py:801-811`) now fans out to every replica before
clearing the prototype — L4.

### 1.5 F-stat launch-site signature

`route_fstat_ll(cls, comp, method_name, holder, ...)` (`gbbands.py:896`) —
takes the comp object so the shard's replica can be resolved before binding.
`_fstat_NM` updated at `gbspecialstretch.py:2280-2287`, preserving the
existing `getattr(self.gb_wdm_comp, "chunked", self.gb_wdm_comp)` sig-het
unwrap.

### 1.6 Runner-script device ordering (F-stat audit D3)

- `sampling/fstat_proposal.py:721-747` — `StackedFStatProposal4D.from_cache`
  gains `device=None` and wraps the upload in `device_context`.
- `scripts/fstat_proposal/run_fstat_rj_search.py:214-222` — passes the
  `gpu_index` `_resolve_use_cupy` already computes.

I did **not** move `build_birth_distribution` after `fit.build()`. Fixing
`from_cache` is the robust half (it protects any caller built outside the
pinned window), and moving the call would also have to move
`fit.gb.rj_birth_distribution = birth`, which `fit.build()` consumes — a
riskier edit for no extra coverage.

---

## 2. What I did NOT do, and why

- **No attempt to spread the F-stat across devices.** `_fstat_NM` sets
  `di = xp.full(n, walker_ref)`, so every row lands on one shard; the whole
  `rj_step` span is ~1.9% of an iteration and the F-stat is a fraction of
  that. Made correct, not parallel — per the brief.
- **No second `slab_min_f` guard on `route_information_matrix`** (F-stat
  audit D5). See §6.
- **`GBGPU/src/gbgpu/cutils/*` and `gbsignalhetcomputations.py` untouched.**
  Verified: `git status` shows no modification to either.
- **No new settings file, no hand-rolled likelihood, nothing committed.**

---

## 3. CPU test output (verbatim)

Env: `/Users/mkatz/miniconda3/envs/deving/bin/python`, all thread pools
pinned to 1.

Baseline before any edit (15 pre-existing tests):

```
$ python -m unittest tests.test_gb_shard_router
Ran 15 tests in 9.715s

OK
```

After the change — 15 pre-existing tests still pass, 15 new:

```
$ python -m unittest tests.test_gb_shard_router
..............................
Ran 30 tests in 52.583s

OK

$ python -m unittest tests.test_gb_shard_router -v      # names only, trimmed
test_wdm_comp_rebuilds_from_recorded_args           (GBCompReplicaContractTest) ... ok
test_fd_comp_rebuilds_from_recorded_args            (GBCompReplicaContractTest) ... ok
test_sighet_wrapper_rebuilds_from_its_knob_dict     (GBCompReplicaContractTest) ... ok
test_comp_device_assert_fires_on_foreign_buffers    (ShardRouterTest) ... ok
test_comp_replica_reused_across_calls               (ShardRouterTest) ... ok
test_fill_template_routes_and_slices_slab           (ShardRouterTest) ... ok
test_fstat_dispatches_to_device_local_comp          (ShardRouterTest) ... ok
test_fstat_multi_shard_requires_data_index          (ShardRouterTest) ... ok
test_fstat_rejects_slab_holders                     (ShardRouterTest) ... ok
test_fstat_routes_and_reassembles                   (ShardRouterTest) ... ok
test_fstat_single_shard_passthrough                 (ShardRouterTest) ... ok
test_get_ll_params_rows_match_partition             (ShardRouterTest) ... ok
test_get_ll_partition_and_scatter                   (ShardRouterTest) ... ok
test_get_swap_ll_reassembly                         (ShardRouterTest) ... ok
test_information_matrix_dispatches_to_device_local_comp (ShardRouterTest) ... ok
test_information_matrix_routes_and_reassembles      (ShardRouterTest) ... ok
test_information_matrix_single_shard_passthrough    (ShardRouterTest) ... ok
test_min_freq_inds_pointer_identity_across_refresh  (ShardRouterTest) ... ok
test_noop_in_model_routes_multi_shard               (ShardRouterTest) ... ok
test_primary_shard_reuses_the_prototype_engine      (ShardRouterTest) ... ok
test_shard_view_protocol                            (ShardRouterTest) ... ok
test_shard_view_slab_min_f_none_without_parent_metadata (ShardRouterTest) ... ok
test_shard_view_slab_min_f_survives_cell_swap       (ShardRouterTest) ... ok
test_shard_view_slices_slab_min_f                   (ShardRouterTest) ... ok
test_sig_het_clear_in_model_fans_out_to_replicas    (ShardRouterTest) ... ok
test_sig_het_in_model_rejected_multi_shard          (ShardRouterTest) ... ok
test_sig_het_shards_build_independent_references    (ShardRouterTest) ... ok
test_sig_het_without_factory_still_refuses_multi_shard (ShardRouterTest) ... ok
test_single_shard_never_builds_replicas             (ShardRouterTest) ... ok
test_single_shard_passthrough                       (ShardRouterTest) ... ok

Ran 30 tests in 47.909s

OK
```

Neighbouring multi-shard suites, unchanged:

```
$ python -m unittest tests.test_band_view_multi_shard tests.test_multi_gpu_placement tests.test_addremove_multi_shard
Ran 44 tests in 1.320s

OK
```

Engine / F-stat / source-replica / GB-flow suites:

```
$ python -m unittest tests.test_gb_likelihood_engine tests.test_fstat_proposal \
      tests.test_fstat_gridfit tests.test_source_gen_device_replica tests.test_gbspecial_flow
Ran 60 tests in 84.145s
OK (skipped=1)
```

Full suite:

```
$ python -m unittest discover -s tests -t .
Ran 407 tests in 211.050s

FAILED (failures=1, errors=9, skipped=12)
```

All ten are **pre-existing and environmental** — none is in a file this change
touches, and none reaches the router, the replica helpers or the comps'
recorded arguments:

| failing test(s) | cause |
|---|---|
| `test_move_timing` x2, `test_noise_globalfit` x1 | stale HDF backends from the pre-cold-chain layout (`gf_output_blank/blank_fit_testing.h5` dated Jul 29, `gf_output_noise/...` dated Jul 26); `hdfbackend.check_format_version` refuses to resume them by design |
| `test_sobbh_chunked_move` x3 | **stale bbhx binary** — `sobbh_wdm_het_get_ll` is invoked with 41 args (incl. the task-b `band_slab_Nf` + `slab_min_f` pair) against a compiled signature that accepts 39. The documented "bbhx MUST be rebuilt in every env" trap; `sobbhspecialmove.py` is unmodified and my `chunked_het.py` diff is purely additive (`__init__` metadata + two new properties), never touching `get_ll_wdm` or `_slab_kernel_args` |
| `test_stock_globalfit` x2 + `test_global_fit_signal_gen_mojito` x1 | `full_year_combined` `chop_window` validation ("set exactly one of mojito_source_ids non-empty") — env leakage between tests; `full_year_combined.py` is unmodified |
| `test_stock_globalfit.test_lite_kwarg_matches_twin` | `None != False` on a lite-preset field; `stock/base.py` is unmodified |

Note the working tree also carries unrelated uncommitted work
(`plot_fstat_proposal_mojito.py`, `gb_engine_benchmark.py`, `recipe.py`,
`variants/gb_no_fg.py`, and most of the `gbspecialstretch.py` delta), so this
suite was not green before this change either.
```

### 3.1 What the new tests actually pin

Extended `tests/_multishard.py` with `FakeDeviceComp` — a comp stand-in
carrying the `_build_device` / `args` / `kwargs` replica contract whose
outputs stamp the comp's own build device, so a shard running against a
foreign-device comp is visible in the result. Extended
`tests/test_gb_shard_router.py` with `_StubSigHetEngine`, which reproduces the
real sig-het in-model contract (flat intra-shard `_slot_to_ref`, stash keyed
by reference row, single `_in_model` flag that flips the next call to a
mid-block PATCH).

| requirement | test |
|---|---|
| `slab_min_f[shard] == parent.slab_min_f[view.rows]` | `test_shard_view_slices_slab_min_f` |
| ... survives a cell swap | `test_shard_view_slab_min_f_survives_cell_swap` |
| ... `None` on a holder without slab metadata | `test_shard_view_slab_min_f_none_without_parent_metadata` |
| each shard reaches the replica whose recorded device matches `view.device` | `test_fstat_dispatches_to_device_local_comp`, `test_information_matrix_dispatches_to_device_local_comp` |
| replicas are allocate-once (module cache, one per non-prototype device) | `test_comp_replica_reused_across_calls` |
| the permanent device guard fires on un-replicable comps | `test_comp_device_assert_fires_on_foreign_buffers` |
| `len(gpus) <= 1` → same object, nothing allocated | `test_single_shard_never_builds_replicas` |
| primary shard reuses the prototype engine (`is`) | `test_primary_shard_reuses_the_prototype_engine` |
| **two shards, overlapping intra-shard slot ids, both build fresh; neither stash moves** | `test_sig_het_shards_build_independent_references` |
| `clear_in_model` clears every replica | `test_sig_het_clear_in_model_fans_out_to_replicas` |
| un-wired router still refuses multi-shard sig-het | `test_sig_het_without_factory_still_refuses_multi_shard` |
| `route_fstat_ll` routes correctly after the signature change | `test_fstat_routes_and_reassembles`, `test_fstat_single_shard_passthrough`, `test_fstat_multi_shard_requires_data_index`, `test_fstat_rejects_slab_holders` |

Plus `GBCompReplicaContractTest` — the half that CPU fake shards *cannot*
reach. `_device_local_gb_comp` is a no-op on CPU (there is no second device),
so the replica **rebuild** would otherwise only be exercised on the cluster,
where a renamed constructor argument is a `TypeError` at proposal time. These
three tests build REAL `GBWDMComputations` / `GBFDComputations` /
`GBSignalHetComputations` objects and assert
`type(comp)(*comp.args, **comp.kwargs)` reproduces the chunk geometry, window,
Tukey alpha and `t_obs_start`; and that the `_SIGHET_REPLICA_KNOBS` map covers
**every** keyword-only parameter of `for_band_engine` (introspected, so a new
knob fails the test rather than being silently dropped from replicas) and
reproduces `_g` exactly.

---

## 4. Performance notes

- Single-shard / single-GPU is untouched: `_is_multi(holder)` is false, the
  router passes straight through, `_comp_for` / `_engine_for` are never
  reached, and no replica is allocated.
- `refresh_row_metadata` now also does one `asnumpy` of the parent's
  `slab_min_f` (length = active band count, `int32`) plus one per-shard
  upload, on multi-shard holders only, once per routed call. Small next to a
  kernel launch, but it is new traffic and worth watching in the first GPU
  timing run.
- Replica memory: each replica is chunk geometry + window + wraps
  (kilobytes to low MB) plus, under sig-het, that shard's slice of the
  reference stash. Sharding splits *sources*, so the stash total across N
  replicas ≈ today's single-device total.

---

## 5. NOT validated — everything GPU

Nothing in this change has run on a GPU. Specifically unverified:

1. That `type(comp)(*args, **kwargs)` inside `device_context(xp, d)` actually
   places `chunk_t_starts` / `chunk_keep_*` / `wdm_window` and the
   `OrbitsWrap` / `TDIConfigWrap` pointer fields on device `d`. (The
   *reconstruction* is CPU-tested; the *placement* is not.)
2. That `_device_local_orbits` handed an already-configured orbits object
   produces a device-`d`-resident `pycppdetector_args`. The comp's `orbits`
   setter deepcopies and only re-`_configure`s when `not configured`;
   `_device_local_orbits` builds a FRESH `orbits.__class__(*args, **kwargs)`
   (unconfigured, lazy) so the configure — and therefore the
   `self.xp.asarray(...)` uploads at `detector.py:464-478` — should happen
   inside our context. **This is the single most likely place for a residual
   P2P dependency and the first thing to check with the guard below.**
3. That `_assert_comp_device` never fires spuriously in a real run.
4. Bit-identical 1-GPU vs 2-GPU initial lnL on either arm.
5. That the sig-het second shard takes the fresh-build branch on real
   hardware (CPU proves the routing, not the kernel).
6. `from_cache(device=...)` placement.
7. Memory: that replicas do not push a production run over the GPU pool.

---

## 6. Departures from the audit plans

| audit item | what I did instead | why |
|---|---|---|
| F-stat **D2** (assert on `comp.wdm_window.device.id`) | assert on a recorded `_build_device` (`gbbands.py:485`) | the audit's form silently no-ops on `GBFDComputations`, which has no `wdm_window`. Correction A. The `wdm_window` probe is kept only as the last fallback in `_comp_build_device`. |
| F-stat **D5** (copy the `slab_min_f` guard into `route_information_matrix`) | **not added** | Correction B: with `_ShardHolderView.slab_min_f` sliced (§1.1), a slab holder reaching `route_information_matrix` is now *correct*, so a second guard would be a guard against nothing. The existing `route_fstat_ll` guard is **kept** — but its comment is rewritten: it is now a scope assertion (the in-fit F-stat scans one walker's FULL residual by design; a slab holder there means an unvalidated caller), not the last line of defence. |
| both audits' "primary = `gpus[0]`" | primary = the comp's own `_build_device`, `gpus[0]` only as fallback | a comp built before `pin_main_device` lives on device 0 even under `GPUS=2,3`. Keying on `gpus[0]` would hand device-0 pointers to the `gpus[0]` shard *and* make the new assert fire on a configuration that "works" today. |
| sig-het **C5** sketch (`engine_factory is None` → `NotImplementedError` inside `_engine_for`) | `_engine_for` falls back to the shared engine; the raise moved into `setup_in_model` and fires only when two shards resolve to the same engine | the sketch would have broken every multi-shard `get_ll` that works today (three of the pre-existing 15 tests construct `_RoutedBandEngine(engine)` with no factory). |
| sig-het **C6** sketch (record `_replica_kwargs` on `GBSignalHetComputations`) | derived from the existing `self._g` instead | the brief's shared-file rule, and `_g` already records every knob in resolved, idempotent form. `gbsignalhetcomputations.py` was not edited. |
| neither audit | added `_device_local_tdi_config` | `TDIConfig` uploads its tables at construction; without a replica the rebuilt comp's `cpp_tdi_config` still points at the primary device. Both audits list `cpp_tdi_config` in the inventory but neither plan replicates it. |
| neither audit | `make_routed_band_engine` module helper | both construction sites needed the same factory closure; duplicating it twice was the alternative. |

### Audit claims that did not survive contact with the source

Only one, and it is a *gap* rather than an error:

- Both audits' §c.1 / §A4 correctly inventory `cpp_tdi_config` as
  device-resident state, but the D1/C6 patch sketches replicate only
  `orbits`. Rebuilding a comp with the shared `TDIConfig` reproduces the exact
  bug the plan set out to fix, one level down. Closed by
  `_device_local_tdi_config`.

Everything else I checked held: the `_fstat_NM` single-shard concentration,
the `slab_min_f` view delegation bug, the version-agnostic `setup_in_model`,
`SubBandBuffer` really receiving `gpus`, the `from_cache` bare upload, and
`build_birth_distribution` running before `fit.build()`.

---

## 7. GPU checklist for tomorrow, in order

Run each step only if the previous one passed.

1. **Rebuild GBGPU and LAT in the target env.** `gbcomps.py` and
   `chunked_het.py` changed (Python only), but the standing sprint rule is
   that a stale binary produces loud nanobind errors — do it anyway.

2. **Single-GPU no-regression first.** Any GB run you already trust, one
   iteration, `GPUS=<one device>`. Nothing in this change should alter it;
   `_is_multi` is false and the router passes through. If this moves, stop.

3. **The guard, alone.** `GPUS=0,1`, any GB WDM run, one iteration. If
   `_assert_comp_device` fires — `GB comp ... holds buffers on device X but
   this shard launches on device Y` — the replica dispatch missed a comp type;
   report the class name. This is the cheapest possible signal.

4. **The parity gate.**
   ```sh
   GPUS=0,1 OMP_NUM_THREADS=1 NUM_ITERATIONS=2 \
       python scripts/gb_chunked_het/gb_multigpu_parity.py
   ```
   Runs both arms (`sighet` = `erebor.vgb()` with `GB_SIGHET_INMODEL=1`,
   `fstat` = `erebor.gb_no_fg()` with `GB_MODE=search`), each twice in
   separate processes, and requires the pre-move per-walker initial lnL to be
   **bit-identical** (compared as `float.hex()`, so a 1-ulp difference is
   visible). Exits non-zero on the first failing arm and prints the offending
   walker's two values. One arm at a time:
   `--arms sighet` / `--arms fstat`.

5. **P2P off.** The whole point of the change is that a non-P2P node should
   work. If the cluster allows it, disable peer access (e.g.
   `CUDA_VISIBLE_DEVICES` pairing across an unlinked bridge, or a node without
   NVLink) and re-run step 4. A pass here is the real proof; a pass with P2P
   on only proves the numbers, not the independence.

6. **Memory.** `nvidia-smi` watermark on the 2-GPU sig-het arm vs the 1-GPU
   leg. Expect ≈ half the stash per device plus a few MB of fixed replica
   cost. Cross-check `scripts/diagnostics/gpu_memory_estimate.py`.

7. **`from_cache` placement.** A `GPUS=2` (non-zero, single device) F-stat
   search run through `scripts/fstat_proposal/run_fstat_rj_search.py`. Before
   this change the birth grid landed on device 0; after it, `nvidia-smi`
   should show no allocation on device 0 at all.

8. **Then, and only then**, the GPU-count axis on
   `scripts/gb_chunked_het/gb_engine_benchmark.py` (the sig-het audit's
   acceptance item 5). Not before: a "gain vs GPU count" plot drawn while
   correctness is unproven invites exactly the wrong reading.

---

## 8. Files touched

| file | what |
|---|---|
| `src/lisatools/globalfit/moves/gbbands.py` | `_ShardHolderView.slab_min_f`; `_RoutedBandEngine` per-device engines/comps, device guard, `setup_in_model` / `clear_in_model`; `route_fstat_ll` signature; `make_routed_band_engine`; `SubBandBuffer` wiring |
| `src/lisatools/globalfit/moves/gbspecialstretch.py` | move-level engine via `make_routed_band_engine`; `_fstat_NM` passes `(comp, method_name)` |
| `src/lisatools/globalfit/stock/erebor/source_runtime.py` | `_device_local_tdi_config`, `_device_local_gb_comp`, `_build_gb_comp_replica`, `_SIGHET_REPLICA_KNOBS`, `_device_local_domain_settings_on` |
| `src/lisatools/chunked_het.py` | `_build_device` + `args` / `kwargs` on `WDMComputationsBase` |
| `src/lisatools/sampling/fstat_proposal.py` | `from_cache(device=...)` |
| `scripts/fstat_proposal/run_fstat_rj_search.py` | pass `device=gpu_index` |
| `scripts/gb_chunked_het/gb_multigpu_parity.py` | **new** — the 1-vs-N-GPU gate |
| `scripts/diagnostics/multi_gpu_smoke.md` | stale "sig-het raises NotImplementedError" limit removed; parity gate + new guard documented |
| `tests/_multishard.py` | `FakeDeviceComp` |
| `tests/test_gb_shard_router.py` | `_StubSigHetEngine`, 15 new tests, `GBCompReplicaContractTest` |
| `GBGPU/src/gbgpu/gbcomps.py` | `_build_device` + `args` / `kwargs` on `GBFDComputations` |

Not touched: `GBGPU/src/gbgpu/cutils/*`, `GBGPU/src/gbgpu/gbsignalhetcomputations.py`.
