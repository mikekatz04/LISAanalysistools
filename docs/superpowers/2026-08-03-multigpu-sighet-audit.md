# Multi-GPU audit — the sig-het engines (v2 / v3 / v4)

**Date:** 2026-08-03
**Status:** AUDIT + PLAN ONLY. Nothing here has been implemented; no source
file was modified for it. For review.
**Machine:** laptop, no GPU. Static analysis of the current working tree
(LAT `6c01ddd` + local uncommitted work, GBGPU `9f0fac7`).

Companion: `2026-08-03-multigpu-fstat-audit.md`. The two share a fix (the
per-device comp helper) and should land together.

---

## Executive summary

1. **v2, v3 and v4 share the defect completely.** The single-device
   assumption lives entirely in `setup_in_model` and the coefficient stash it
   builds — code that is *version-agnostic*. v3 and v4 differ only in which
   scorer kernel consumes that stash (`gbsignalhetcomputations.py:788-843`).
   There is no v3-specific or v4-specific multi-GPU bug to find.
2. The blocker is **not one thing but three**, and only the first is the one
   the existing `NotImplementedError` describes:
   - a per-device **stash** (7 arrays + `params_ref_all`),
   - a **global slot→reference map** that collides across shards, and
   - a single `_in_model` **flag** that silently turns the second shard's
     fresh build into a mid-block *patch* of the first shard's references.
3. **The third is a genuine correctness landmine** — a reference built on
   shard 0's residual would be overwritten by, or matched against, shard 1's.
   The current guard (`gbbands.py:590-597`) fires *before* it can happen, so
   the landmine is armed but not tripped. Any fix that relaxes the guard
   without fixing the slot map will trip it.
4. Memory cost of the fix is ~neutral: sharding splits the *sources*, so N
   replicas of a 1/N-sized stash ≈ today's total.
5. The **ll reduction across shards is a non-issue.** Each candidate belongs
   to exactly one buffer slot on exactly one shard; the router already
   gathers per-shard `ll` pieces host-side (`gbbands.py:514-524`). There is no
   cross-shard sum to design.
6. Independently of all the above, sig-het inherits the **comp-level
   device-sticky state** documented in the F-stat audit (its own
   `window_full`, `n_sparse_local`, `tdi_wrap`, `_v4_band_arrays`, plus the
   whole chunked delegate). Fixing the stash without fixing these leaves the
   run P2P-dependent.

---

## (a) Where the single-device assumption lives — exact localisation

All citations are `GBGPU/src/gbgpu/gbsignalhetcomputations.py` unless noted.

### A1 — the coefficient stash (the item the error message names)

Built at the end of a fresh `setup_in_model`:

```
:652   self.c0_sparse_all = c0_sparse
:653   self.A0_all  = A0s
:654   self.A1_all  = A1s
:655   self.B0_all  = B0s
:656   self.B1_all  = B1s
:657   self.B0nc_all = B0ncs
:658   self.B1nc_all = B1ncs
:659   self.params_ref_all = refs
```

Shapes: A-blocks `(n, nch, Nf_active, N_sparse_t)` complex128, B-blocks
`(n, nch, nch, Nf_active, N_sparse_t)` complex128 (`_expand_A` / `_expand_B`,
`:605-617`). The arrays are allocated by `xp.zeros` inside those expanders
(`:606`, `:613`) and by `xp.zeros` at `:561-564` for the windowed `c0`, all
on **whatever CUDA device is current when `setup_in_model` runs**. The
comment at `:648-651` states the intent plainly: "built once here (on the
run's device), then reused by every repeat-proposal get_ll".

The mid-block patch path writes into the *same* arrays in place
(`:638-645`).

### A2 — the slot→reference map (the collision)

```
:661   slot_map = np.full(int(slots.max()) + 1, -1, dtype=int)
:662   slot_map[slots] = np.arange(n)
:663   self._slot_to_ref = slot_map          # host
:666   self._slot_to_ref_xp = xp.asarray(slot_map)   # device mirror
```

`slots` comes from `data_index` (`:462-464`). Under the router, `data_index`
is **intra-shard** (`gbbands.py:583-589` passes `intra`, produced by
`_partition` at `gbbands.py:398-402`). So shard 0's slot 3 and shard 1's
slot 3 are different physical cells that map to the *same* entry of a single
`slot_map`. One flat map cannot describe two shards.

The consumer is equally flat: `get_ll_wdm` at `:716-721` does
`ref_raw = self._slot_to_ref_xp[di]` on the intra-shard `di` the router hands
it (`gbbands.py:500-506`).

### A3 — the `_in_model` flag (the landmine)

```
:624   if self._in_model is not None:
           # Mid-block PATCH: re-anchor only the given slots ...
:629       if int(slots.max()) >= len(self._slot_to_ref): raise ...
:633       ref_idx = self._slot_to_ref[slots]
:634       if np.any(ref_idx < 0): raise ...
:638-645   self.c0_sparse_all[ref_idx] = c0_sparse   # etc.
:646       return True
```

`_in_model` is a single boolean-ish flag set at `:667` and cleared only by
`clear_in_model` (`:670-675`). In the router's loop
(`gbbands.py:583-598`) `setup_in_model` is called **once per shard**. The
first non-empty shard takes the fresh-build branch and sets `_in_model = True`;
the *second* would therefore take the **patch** branch and write shard 1's
coefficients into whatever `ref_idx` shard 0's `slot_map` says — i.e. it
would silently corrupt shard 0's references (or raise at `:630`/`:636` if
the slot ids happen not to overlap).

**Today this cannot happen.** The router raises immediately after the first
shard returns truthy:

```python
# gbbands.py:583-598
for view, (pos, intra, _) in zip(views, parts):
    ...
    with device_context(xp, view.device):
        ret = self._engine.setup_in_model(view, ...)
    if ret:
        self._engine.clear_in_model()
        raise NotImplementedError(
            "sig-het in-model references hold single-shard state on the "
            "shared computation object; multi-shard buffers need per-shard "
            "comp replicas (follow-on work). ...")
```

The guard is correct and it calls `clear_in_model()` first so no half-built
state survives. **It is the only thing standing between the current code and
silent cross-shard reference corruption.** That is the headline for (d).

### A4 — construction-time device state (shared by every version)

Set in `for_band_engine` (`:330-435`), on the device current at build:

| state | site | note |
|---|---|---|
| `self.window_full` | `:394-396` | `xp.asarray(...)`; consumed by the **v2** scorer only (`:831`) |
| `self.n_sparse_local` | `:397` | consumed by v2, v3 **and** v4 (`:792`, `:815`, `:831`) |
| `self.tdi_wrap` | `:414` | `GBTDIonTheFly(...).wave_gen`; holds device pointers to orbits/TDI-config splines |
| `self._keep_alive` | `:415-416` | keeps `gb_gen`/`orbits`/`tdi_config` alive |
| `self.chunked` | `:363` | the shared `GBWDMComputations` delegate — itself device-sticky, see the F-stat audit §c.1 |

And lazily, on first scoring call:

| state | site | note |
|---|---|---|
| `self._v4_band_arrays` | `:776-777` via `_make_v4_band_arrays` (`:216-260`) | **v4-only**; two device arrays of cardinal weights |

There is **no** device id anywhere on the class. Exhaustive grep of
`gbsignalhetcomputations.py` for `Device(`, `setDevice`, `device_context`,
`gpus`: zero API hits (only prose comments). The class does not accept a
`gpus=` kwarg and never queries the current device.

### A5 — the engine and the settings tree hold ONE comp

- `WDMBandLikelihoodEngine.__init__` stores `self.gb_comps = gb_comps`
  (`GBGPU/src/gbgpu/gb_likelihood.py:680`); `setup_in_model` /
  `clear_in_model` / `get_ll` all forward to that one object
  (`gb_likelihood.py:757`, `:785-786`, `:788-789`).
- The engine is built once per `SubBandBuffer`
  (`gbbands.py:1033-1045`) and once per move (`gbspecialstretch.py:893-903`),
  both wrapping the same `gb_wdm_comp`.
- That comp is created once at variant build and stored on the settings tree:
  `gb_info.gb_wdm_comp = GBSignalHetComputations.for_band_engine(...)`
  (`globalfit/stock/erebor/variants/gb_no_fg.py:951-960`; the VGB twin at
  `variants/vgb.py:269`).

So one object, one device, every shard.

---

## (b) Do v3 and v4 share the defect, or differ?

**They share it in full.** Concretely:

- `setup_in_model` (`:437-668`) contains **no** version branch. It builds the
  same stash for v2, v3 and v4; the v4 comment at `:785` says so ("Consumes
  the SAME stash as v2/v3"), as does the v3 comment at `:808-810`.
- The three scorers differ only in the kernel called and the extra
  device arrays passed:

  | version | entry | extra device args beyond the stash |
  |---|---|---|
  | v2 | `gb_signal_het_get_ll_in_kernel` (`:827-840`) | `window_full`, `n_sparse_local` |
  | v3 | `gb_signal_het_v3_get_ll` (`:811-823`) | `n_sparse_local` |
  | v4 | `gb_signal_het_v4_get_ll` (`:788-803`) | `n_sparse_local`, `_v4_band_arrays[0..2]` |

- Therefore the **only** v4-specific single-device item is
  `_v4_band_arrays`, and it is the *easiest* thing in the whole audit to
  replicate: the cardinal weights are candidate-independent **and**
  reference-independent (they depend on `Tobs`, `N_sparse_t`, `stride`,
  `ind_min_t`, `Nf`, `dt`, `K`, `band` — see `_make_v4_band_arrays`
  `:236-260`), so a per-device replica is a pure duplicate of an identical
  host computation. No correctness question at all.
- v2's extra item, `window_full`, is likewise a deterministic function of the
  WDM settings.

**Practical consequence for the plan:** design the fix for `setup_in_model` +
the stash + the slot map, and v2/v3/v4 are all fixed at once. Do **not**
build three code paths.

---

## (c) Minimal-diff design — per-shard comp replicas

This mirrors, idiom for idiom, the machinery that already exists:
`_device_local_orbits` / `_device_local_domain_settings` +
`_DEVICE_*_REPLICAS` in
`globalfit/stock/erebor/source_runtime.py`; the router's `_ShardHolderView`
(`gbbands.py:205-332`) which already supplies `.device` and sets
`.gpus = [self.device]` (`:237-240`); and `device_context`
(`utils/device.py:40-49`).

### C1 — what gets replicated

**One `GBSignalHetComputations` per shard device**, each wrapping a
**per-device chunked delegate** (the same `_device_local_gb_comp` helper the
F-stat audit proposes — this is the shared dependency between the two plans).

Replicating the sig-het object automatically gives per-device:

* the whole stash (A1) — because `setup_in_model` runs on the replica;
* the slot map (A2) — each replica gets its own, indexed by **its own**
  intra-shard slots, so the collision disappears by construction;
* the `_in_model` flag (A3) — each replica tracks its own block, so the
  second shard takes the *fresh-build* branch as it should;
* `window_full`, `n_sparse_local`, `tdi_wrap`, `_v4_band_arrays` (A4) —
  because `for_band_engine` re-runs inside the device context.

That is the whole defect list, closed by one mechanism. **Nothing needs to
stay shared**, and the primary device reuses the existing object unchanged
(the N=1 case is byte-identical to today).

### C2 — memory

Sharding partitions **sources** across shards: `_partition`
(`gbbands.py:388-418`) splits the call's rows by owning shard, so shard *s*
sees only its own `n_s` sources and its stash is
`(n_s, nch, ..., N_sparse_t)`. With `sum_s n_s = n`, total device memory
across N replicas ≈ today's single-device total. The *fixed* per-replica cost
is `window_full` + `n_sparse_local` + `_v4_band_arrays` + the `tdi_wrap`
tables — kilobytes to low megabytes.

### C3 — how the reference build is done per shard

**No change to `setup_in_model` itself.** The router already:

1. partitions rows by owning shard (`gbbands.py:580`),
2. enters `device_context(xp, view.device)` (`:586`),
3. presents the shard as a single-shard holder via `_ShardHolderView`, whose
   `linear_data_arr` / `linear_psd_arr` are that split's live buffers
   (`gbbands.py:258-263`) and whose `slab_min_f` / `band_slab_Nf` delegate to
   the parent.

Point 3 is the one place that needs care. `setup_in_model` reads
`buffer_aca.band_slab_Nf` (`:496`) and `buffer_aca.slab_min_f` (`:506`) and
then indexes the latter **by the slot ids it was given** (`:507-509`,
`[slots] - ind_min_f`). Those slots are **intra-shard**, but
`_ShardHolderView.__getattr__` (`gbbands.py:325-332`) returns the parent's
**global-slot** array. **This is a second, independent correctness bug on the
same path** — same family as the one `route_fstat_ll` guards against
(F-stat audit §a′), and it would survive the comp-replica fix.

Fix: give `_ShardHolderView` an explicit `slab_min_f` property that slices to
the shard's rows, exactly as `fill_template` already does for the kwarg form
(`gbbands.py:465-473`):

```python
# gbbands.py, inside _ShardHolderView -- NOT APPLIED, sketch only
@property
def slab_min_f(self):
    """Per-slot slab origins sliced to THIS shard's rows.

    Must be an explicit property: __getattr__ delegation would hand back the
    parent's global-slot array while every index the engines pass is
    intra-shard. Mirrors the per-slot kwarg slice in
    _RoutedBandEngine.fill_template.
    """
    src = getattr(self._parent, "slab_min_f", None)
    if src is None:
        return None
    if self._slab_min_f_view is None or ...:      # cache + refresh in place
        ...
    return self._slab_min_f_view
```

with the refresh hooked into `refresh_row_metadata` (`gbbands.py:285-314`)
so it survives cell swaps like `min_freq_inds` does. `band_slab_Nf` is a
scalar and is shard-invariant — delegation is fine for it.

Once that is in place, each shard's `setup_in_model` builds a reference
against **its own** residual slabs, on **its own** device, with **its own**
slot map. Correct by construction.

### C4 — how the ll reduction is combined

**It already is.** `_RoutedBandEngine.get_ll` (`gbbands.py:479-524`) runs
each shard and host-assembles the full-length outputs with `_assemble`
(`:420-440`), filling `-1e300` for any row no shard produced. Since a
candidate belongs to exactly one buffer slot on exactly one shard, this is a
**gather**, not a sum — there is no cross-shard reduction to design, and the
existing code is right. `d_h_out` / `h_h_out` / `phase_angle` / `kept_out`
are assembled the same way (`:517-523`).

The only change needed is that the per-shard call must be dispatched to the
shard's **replica** rather than to `self._engine`.

### C5 — the router-side patch sketch

`_RoutedBandEngine` currently holds one engine (`gbbands.py:357-358`). Add a
lazily-populated per-device map, built by an `engine_factory` supplied at
construction:

```python
# gbbands.py -- NOT APPLIED, sketch only
class _RoutedBandEngine:
    def __init__(self, engine, engine_factory=None):
        self._engine = engine
        # device -> engine replica. Populated lazily on first multi-shard
        # call; the primary device maps to ``engine`` itself so the N=1 path
        # is byte-identical and allocates nothing.
        self._engine_factory = engine_factory
        self._engine_by_device = {}

    def _engine_for(self, holder, view):
        dev = view.device
        gpus = getattr(holder, "gpus", None)
        primary = None if not gpus else int(gpus[0])
        if dev is None or primary is None or int(dev) == primary:
            return self._engine
        hit = self._engine_by_device.get(int(dev))
        if hit is None:
            if self._engine_factory is None:
                raise NotImplementedError(
                    "multi-shard sig-het needs a per-device engine factory; "
                    "SubBandBuffer must pass engine_factory= ...")
            with device_context(holder.xp, int(dev)):
                hit = self._engine_factory(int(dev))
            self._engine_by_device[int(dev)] = hit
        return hit
```

Then, at every routed launch site, replace `self._engine` with
`self._engine_for(holder, view)` — `fill_template` (`:474`), `get_ll`
(`:500`), `get_swap_ll` (`:552`), `setup_in_model` (`:587`),
`_route_matrix` (`:621`). `clear_in_model` (`:600-601`) must fan out:

```python
    def clear_in_model(self):
        for e in self._engine_by_device.values():
            e.clear_in_model()
        return self._engine.clear_in_model()
```

and the `if ret: raise NotImplementedError(...)` block at `:590-597` is
deleted — but **only after** C3's `slab_min_f` slice lands, or the fix
converts a loud error into a wrong answer.

`_mirror_engine_outputs` (`:442-447`) is passthrough-only, unaffected.

### C6 — the factory

Built where the engine is built today,
`SubBandBuffer.__init__` (`gbbands.py:1033-1045`) and the move-level site
(`gbspecialstretch.py:893-903`). It must construct a **per-device comp**, not
just a per-device engine wrapper — the engine is a thin object over
`gb_comps`:

```python
# gbbands.py -- NOT APPLIED, sketch only
def _make_engine(device):
    with device_context(self.xp, device):
        wdm = _device_local_gb_comp(self.gb_wdm_comp, self.xp, device, primary)
        fd  = _device_local_gb_comp(self.gb_fd_comp,  self.xp, device, primary)
        return make_band_likelihood_engine(
            self._basis_settings, gb=self.gb, gb_fd_comp=fd, gb_wdm_comp=wdm,
            ... )   # the existing kwargs verbatim
self._likelihood_engine = _RoutedBandEngine(
    make_band_likelihood_engine(...), engine_factory=_make_engine)
```

`_device_local_gb_comp` must special-case the sig-het wrapper: it is built
through `for_band_engine(chunked_comp, **knobs)`, not `__class__(*args,
**kwargs)`, so the helper needs a small dispatch —

```python
if isinstance(comp, GBSignalHetComputations):
    return type(comp).for_band_engine(
        _device_local_gb_comp(comp.chunked, xp, device, primary),
        **comp._replica_kwargs)        # stash the knobs at for_band_engine time
```

which means `for_band_engine` should record its knob dict on the instance
(one line, next to `self._g = dict(...)` at `:418-428`) — the same
"remember your constructor arguments so a replica can be rebuilt" pattern
`_device_local_domain_settings` already relies on.

### C7 — per-device orbits / TDI-config / WDM-settings wraps

Required, and this is the trap the source-move work already hit. Because the
replica is constructed **inside** `device_context`, `for_band_engine`'s
`GBTDIonTheFly(...)` (`:410-413`) and the delegate's `OrbitsWrap` /
`TDIConfigWrap` allocate on the owning device automatically — **provided** the
`orbits` object handed in is itself device-local. Reuse
`source_runtime.py::_device_local_orbits` for that; it already implements
"primary reuses the shared object, non-primary gets one cached replica".

No `jax_device_context` is needed here (unlike MBH): grep for `jax` in
`chunked_het.py` returns only the backend-name comparison at
`chunked_het.py:1103`.

---

## (d) Correctness landmines

Ranked. **L1 and L2 are the ones that would produce a wrong answer rather
than an error.**

**L1 — a reference built on one shard used against another shard's
residual.** This is exactly the scenario the task asked about, and it is
real: the flat `_slot_to_ref` map (A2) plus the single `_in_model` flag (A3)
mean that, with the guard removed, shard 1's `setup_in_model` would *patch*
shard 0's reference rows (`:638-645`) and shard 1's `get_ll_wdm` would then
look up references through shard 0's map (`:719`). Slot ids overlap between
shards by construction (both start at 0), so the `< 0` and out-of-range
checks at `:630`/`:636` would **not** catch it. Mitigation: C1 (per-device
replicas) removes the shared map entirely. **Do not relax the
`NotImplementedError` before C1 and C3 are both in.**

**L2 — `slab_min_f` read globally, indexed intra-shard.** Described in C3.
Independent of L1, survives the replica fix, and produces a silently wrong
reference window (each source's fold would use another cell's layer origin).
Must be fixed in the same change.

**L3 — the P2P dependency.** Even with per-shard replicas of the sig-het
object, the *delegate* (`self.chunked`) and its `cpp_orbits` /
`cpp_tdi_config` / `chunk_t_starts` / `wdm_window` are device-sticky unless
`_device_local_gb_comp` is applied to it too (C6). Without that, the run is
correct only where P2P is available and pays peer-access traffic everywhere.
See the F-stat audit §c for the full inventory.

**L4 — `clear_in_model` fan-out.** If C5's fan-out is forgotten, replicas
keep stale `_in_model` state after a block ends, and the next block's first
`setup_in_model` on that device silently takes the patch branch. Cheap to get
wrong, cheap to test (assert every replica's `_in_model is None` after
`clear_in_model`).

**L5 — replica lifetime vs. buffer lifetime.** `_ShardHolderView`s are cached
on the holder and die with it at proposal teardown (the memory-lifecycle rule,
documented at `gbbands.py:225-228`). Engine **replicas** must NOT follow that
rule: rebuilding a `GBTDIonTheFly` + `OrbitsWrap` per proposal would be
ruinous. They belong in a module-level, allocate-once cache keyed by
`(id(prototype), device)`, exactly like `_DEVICE_ORBITS_REPLICAS`. The
*stash*, by contrast, is per-block and is already cleared by
`clear_in_model`.

**L6 — deepcopy/pickle.** Replicas hold nanobind wraps and cupy arrays and
must never reach the settings tree (LAT-wide rule, `docs/conventions.md`
"Deepcopy / pickle safety"). Module-level cache + `attach_runtime_objects`,
not `gb_info.*`.

---

## Acceptance criteria (unchanged from the standing plan)

1. **1-GPU vs 2-GPU sig-het lnL bit-identical** — the bar the all_sources
   work met (`[50785835.71823513, ...]`).
2. Per-device stash memory ≈ 1/N of the single-GPU value.
3. No regression on the single-shard fast path: with `len(gpus) <= 1` the
   router must return the **same object** (`is`) it does today and allocate
   nothing.
4. CPU fake-shard tests (extend `tests/_multishard.py` +
   `tests/test_gb_shard_router.py`, which already has 15 tests and a
   thread-local `RecordingXp`):
   - each shard's `setup_in_model` reaches the replica whose recorded device
     matches `view.device`;
   - two shards with **overlapping intra-shard slot ids** both build fresh
     references and neither's stash changes when the other is built (the L1
     regression test);
   - `slab_min_f` seen by shard *s* equals `parent.slab_min_f[view.rows]`
     (the L2 regression test);
   - `clear_in_model` clears every replica (L4).
5. Only then: add the GPU-count axis to
   `scripts/gb_chunked_het/gb_engine_benchmark.py`. Deliberately not before —
   a "gain vs GPU count" plot with flat sig-het lines invites exactly the
   wrong reading.

---

## Suggested order of work

1. **`_ShardHolderView.slab_min_f`** (L2) — small, self-contained, and a
   prerequisite for anything that relaxes the guard. Testable on CPU today.
2. **`_device_local_gb_comp`** (shared with the F-stat plan) + recording
   constructor kwargs on `WDMComputationsBase` and knob kwargs on
   `GBSignalHetComputations`.
3. **`engine_factory` on `_RoutedBandEngine`** + the per-device dispatch at
   the five launch sites + `clear_in_model` fan-out.
4. **Delete the `NotImplementedError`** (`gbbands.py:590-597`) — last, and
   only with 1–3 green.
5. Cluster: 1-vs-2-GPU parity run with `GB_SIGHET_INMODEL=1`.

Note that steps 2 and 3 are the same work the F-stat plan needs (D1 there).
Landing them once serves both.
