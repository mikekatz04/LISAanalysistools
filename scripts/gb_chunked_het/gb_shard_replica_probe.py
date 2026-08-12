"""Multi-shard GB scoring probe: FILL correctness vs ENGINE-REPLICA parity.

Context (2026-08-12 shard-bug hunt): on 2 GPUs the VGB in-model acceptance
halves UNIFORMLY across walkers (0.35 -> ~0.15 at the same start state), so
the defect is band-cell-side. Two candidates survive every other test:

* the cross-shard BUFFER FILL (``fill_buffer_residual_and_psd_from_acs``)
  puts wrong content into cells on one shard, or
* the per-device ENGINE REPLICA (``_engine_factory`` ->
  ``_device_local_gb_comp``) scores differently from the prototype.

This probe runs ONE stock-vgb iteration (so the move builds its cached
buffers under GB_BUFFER_PERSIST=1), then:

CHECK A -- refill the largest cached buffer from the parent and compare
   EVERY cell's residual + invC content against an independently GATHERED
   parent slab (``gather_data_shaped``: the BandView row-gather path, code-
   disjoint from the fill's tuple-fancy path). Per-shard max |diff| printed;
   nonzero on one shard = the fill is the bug.

CHECK B -- score one synthetic source per cell of each shard view through
   BOTH the prototype engine and that device's replica engine
   (``_engine_for``). Bitwise-equal d_h/h_h = engines equivalent (fill side
   is guilty); differences = the replica construction is the bug.

Run (cluster, 2 GPUs; ~1 min):

    DATA_MODE=mojito MOJITO_DATA_PATH=/shared/home/mlkatz1/mojito_cache/ \
    GPUS=0,1 OMP_NUM_THREADS=1 VERBOSE=1 PROGRESS=0 \
    FILE_STORE_DIR=./replica_probe/ BASE_FILE_NAME=probe \
    python scripts/gb_chunked_het/gb_shard_replica_probe.py

Single-GPU control: same command with GPUS=0 (every check must be clean).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUM_ITERATIONS", "1")

import gc

import numpy as np


def _host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def _nan_drill(np, xp, proto, rep, dh1, n):
    """Localize where a NaN-producing replica diverges from the prototype.

    Prints, for BOTH engines' comps: the orbit configured flag and t-grid
    endpoints (fix-active check: replica grid must equal the prototype's),
    NaN counts + device ids of the comp arrays the kernel consumes, and the
    device ids / NaN counts of every array in ``pycppdetector_args`` (a
    device-0 array feeding a device-1 OrbitsWrap is a finding). Ends with
    the per-slot NaN pattern (all slots vs a subset)."""
    def _comp_of(e):
        c = getattr(e, "gb_comps", None)
        return c if c is not None else getattr(e, "gb_fd_comp", None)

    for label, eng_obj in (("proto", proto), ("replica", rep)):
        c = _comp_of(eng_obj)
        if c is None:
            print(f"[NAN-DRILL] {label}: no comp resolved")
            continue
        orb = getattr(c, "_orbits", None)
        t = getattr(orb, "t", None)
        t_h = None if t is None else _host(np.asarray(t))
        t_desc = ("None" if t_h is None
                  else f"[{t_h[0]:.6g}..{t_h[-1]:.6g}] len={len(t_h)}")
        print(f"[NAN-DRILL] {label} comp={type(c).__name__} "
              f"orbits_configured={getattr(orb, 'configured', None)} "
              f"t={t_desc}")
        for name in ("wdm_window", "chunk_t_starts", "chunk_keep_lo",
                     "chunk_keep_hi", "chunk_n_global_offset"):
            a = getattr(c, name, None)
            if a is None:
                continue
            ah = _host(xp.asarray(a)) if hasattr(a, "__len__") else np.asarray(a)
            dev = getattr(getattr(a, "device", None), "id", None)
            nan = (int(np.isnan(ah).sum())
                   if getattr(ah, "dtype", None) is not None
                   and ah.dtype.kind == "f" else 0)
            print(f"[NAN-DRILL]   {label}.{name}: dev={dev} nan={nan} "
                  f"absmax={float(np.abs(ah).max()):.6g}")
        pargs = getattr(orb, "pycppdetector_args", None)
        if pargs is not None:
            for i, a in enumerate(pargs):
                if hasattr(a, "shape") and getattr(a, "size", 0) > 1:
                    dev = getattr(getattr(a, "device", None), "id", None)
                    ah = _host(a)
                    nan = (int(np.isnan(ah).sum())
                           if ah.dtype.kind == "f" else 0)
                    print(f"[NAN-DRILL]   {label}.orbit_arg[{i}]: dev={dev} "
                          f"shape={tuple(a.shape)} nan={nan}")
    bad = np.where(np.isnan(dh1))[0]
    print(f"[NAN-DRILL] replica NaN d_h slots: {len(bad)}/{n} "
          f"(first 20: {bad[:20].tolist()})")


def main():
    from lisatools.globalfit.stock import erebor
    from lisatools.globalfit.moves.gbspecialstretch import (
        VGBSpecialStretchMove)
    from lisatools.globalfit.moves.gbbands import device_context

    fit = erebor.vgb()
    fit.build()
    try:
        gen = fit.sample(iterations=1)
    except TypeError:
        gen = fit.sample()
    for _model, _state in gen:
        break

    moves = [m for m in gc.get_objects()
             if isinstance(m, VGBSpecialStretchMove)]
    if not moves:
        raise RuntimeError("no VGBSpecialStretchMove found after sampling")
    mv = moves[0]
    cache = getattr(mv, "_prop_buffer_cache", None) or {}
    if not cache:
        raise RuntimeError(
            "no cached buffers (GB_BUFFER_PERSIST must be 1, the default)")
    buf = max(cache.values(), key=lambda b: int(b.num_bands_now))
    acs = buf.parent_acs
    xp = buf.xp
    combos = _host(buf.unique_band_combos)
    n_slots = int(buf.num_bands_now)
    n_shards = len(buf.linear_data_arr)
    split_map = _host(buf.split_map) if hasattr(buf, "split_map") else \
        np.zeros(n_slots, dtype=int)
    print(f"[probe] buffer: {n_slots} cells, {n_shards} shard(s), "
          f"gpus={getattr(buf, 'gpus', None)}")

    # ---------------- CHECK A: fill content vs gathered parent ----------
    inds_fill = xp.arange(n_slots)
    buf.fill_buffer_residual_and_psd_from_acs(acs, inds_fill=inds_fill)
    full_data = _host(acs.gather_data_shaped())
    full_psd = _host(acs.gather_psd_shaped())
    g = buf._basis_settings
    slab_Nf = int(buf.band_slab_Nf)
    slab_lo = _host(buf.slab_min_f).astype(int) - int(g.ind_min_f)
    worst = np.zeros(n_shards)
    worst_psd = np.zeros(n_shards)
    for slot in range(n_slots):
        w = int(combos[slot, 1])
        lo = int(slab_lo[slot])
        exp_d = full_data[w][:, lo:lo + slab_Nf, :]
        got_d = _host(buf.band_buffer[slot])
        s = int(split_map[slot])
        worst[s] = max(worst[s], float(np.abs(exp_d - got_d).max()))
        exp_p = full_psd[w][..., lo:lo + slab_Nf, :]
        got_p = _host(buf.psd_buffer[slot])
        d_p = np.abs(np.asarray(exp_p, dtype=complex)
                     - np.asarray(got_p, dtype=complex)).max()
        worst_psd[s] = max(worst_psd[s], float(d_p))
    for s in range(n_shards):
        print(f"[CHECK A] shard {s}: data max|diff| {worst[s]:.3e}  "
              f"invC max|diff| {worst_psd[s]:.3e}")
    fill_clean = bool((worst < 1e-12).all() and (worst_psd < 1e-12).all())
    print(f"[CHECK A] FILL {'CLEAN' if fill_clean else '** GUILTY **'}")

    # ---------------- CHECK B: prototype vs replica on the same view ----
    eng = buf._likelihood_engine
    proto = eng.wrapped_engine
    views = eng._shard_views(buf)
    band_edges = _host(buf.band_edges)
    any_diff = False
    for si, view in enumerate(views):
        rows = np.asarray(view.rows)
        b = combos[rows, 2].astype(int)
        f0 = 0.5 * (band_edges[b] + band_edges[b + 1])
        n = rows.shape[0]
        params = np.zeros((n, 9))
        params[:, 0] = 1e-22
        params[:, 1] = f0
        params[:, 2] = 1e-17
        params[:, 4] = 0.3
        params[:, 5] = 0.5
        params[:, 6] = 0.2
        params[:, 7] = 1.0
        params[:, 8] = 0.3
        intra = xp.arange(n, dtype=xp.int32)
        engines = {"proto": proto}
        rep = eng._engine_for(buf, view)
        engines["replica" if rep is not proto else "proto(again)"] = rep
        out = {}
        for name, e in engines.items():
            with device_context(xp, view.device):
                e.get_ll(view, xp.asarray(params), data_index=intra,
                         noise_index=intra, N_vals=None,
                         waveform_kwargs={})
                out[name] = (_host(e.d_h_out).copy(),
                             _host(e.h_h_out).copy())
        names = list(out)
        if len(names) == 2:
            dh0, hh0 = out[names[0]]
            dh1, hh1 = out[names[1]]
            ddh = float(np.abs(dh0 - dh1).max())
            dhh = float(np.abs(hh0 - hh1).max())
            n_nan = int(np.isnan(dh1).sum() + np.isnan(hh1).sum())
            print(f"[CHECK B] shard {si} (device {view.device}) "
                  f"{names[0]} vs {names[1]}: max|d_h diff| {ddh:.3e}  "
                  f"max|h_h diff| {dhh:.3e}  "
                  f"(scale h_h ~ {float(np.abs(hh0).max()):.3e}; "
                  f"{n_nan} NaN in {names[1]} outputs)")
            # NaN-safe verdict: NaN > 0 is False, which previously let a
            # NaN-producing replica print as "EQUIVALENT".
            if not (ddh == 0.0 and dhh == 0.0):
                any_diff = True
            if n_nan or not (ddh == 0.0 and dhh == 0.0):
                _nan_drill(np, xp, proto, rep, dh1, n)
        else:
            print(f"[CHECK B] shard {si} (device {view.device}): replica IS "
                  "the prototype (no factory replica in play)")
    print(f"[CHECK B] ENGINES {'** DIFFER — replica guilty **' if any_diff else 'EQUIVALENT'}")
    print("[probe] DONE")


if __name__ == "__main__":
    main()
