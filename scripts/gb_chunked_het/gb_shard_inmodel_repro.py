"""CPU repro hunt for the 2-GPU VGB in-model scoring drift (2026-08 bug).

Fingerprint being chased (GPU evidence): 1-GPU vs 2-GPU VGB run from the
same start state -> in-model stretch acceptance halves, and the
after-proposal incremental-ll check drifts ~1e-2..3e-2 on the SHARD-1
walkers only. Initial lls are IDENTICAL, so the parent build is clean and
the bug engages inside the propose machinery: buffer fill from the parent,
buffer-side scoring through the routed engine, accepted-move write-back,
or the parity open/close fill_template on the parent.

This script fakes the multi-shard layout on CPU (RecordingXp fake device
contexts from ``tests/_multishard.py``) around REAL objects everywhere
else: a real ``GBWDMComputations`` CPU comp, a real ``BandSorter`` +
``SubBandBuffer``, and the real ``_RoutedBandEngine`` (which on the fake
2-shard holder builds a REAL per-device comp replica through
``_device_local_gb_comp`` -- the exact production shard-1 path).

Phases (each compares a fake 2-shard leg against a single-shard control,
bitwise):

  A. replica-comp numerics: prototype vs ``_device_local_gb_comp`` replica
     on identical inputs (fill / get_ll / swap_ll).
  B. buffer FILL from a walker-sharded parent (per-cell residual + psd).
  C. routed SCORING of band-sharded cells (get_ll / phase-max / swap).
  D. one full open -> fill -> remove -> score -> write-back -> close cycle
     on the walker-sharded parent (the parity-unit round trip).

Run:
  OMP_NUM_THREADS=1 python scripts/gb_chunked_het/gb_shard_inmodel_repro.py
"""

from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

# tests/_multishard.py fixtures (not an installed package)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), "tests"))
sys.path.insert(0, os.path.join(_REPO, "tests"))
from _multishard import FakeMultiShardACA  # noqa: E402

from eryn.state import Branch  # noqa: E402
from eryn.utils import TransformContainer  # noqa: E402

from gbgpu.gbcomps import GBWDMComputations  # noqa: E402
from lisatools.domains import WDMSettings  # noqa: E402
from lisatools.analysiscontainer import BandView, band_gpu_assignment  # noqa: E402
from lisatools.globalfit.moves.gbbands import (  # noqa: E402
    BandSorter,
    make_routed_band_engine,
    pack_special_index,
)
from lisatools.globalfit.stock.erebor.source_runtime import (  # noqa: E402
    _device_local_gb_comp,
)

# ----------------------------------------------------------------------
# Fixture constants (GBCompReplicaContractTest scale)
# ----------------------------------------------------------------------
NF, NT, DT = 32, 64, 15.0
NWALKERS = 4
N_BANDS = 6            # bands 1..4 carry sources -> 16 cells
SRC_BANDS = (1, 2, 3, 4)
NCHAN = 3
SEED = 20260812


def f_ms_to_s(x):
    return x * 1e-3


def build_comp():
    wdm = WDMSettings(Nf=NF, Nt=NT, dt=DT, force_backend="cpu")
    comp = GBWDMComputations(
        wdm, t_ref=0.0, Nt_sub=16, n_pad=2, N_sparse=64,
        tdi_config="1st generation", force_backend="cpu",
    )
    return wdm, comp


def build_transform():
    return TransformContainer(
        input_basis=["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam",
                     "sin_beta"],
        output_basis=["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi",
                      "lam", "sin_beta"],
        parameter_transforms={
            "A": np.exp,
            "f0": f_ms_to_s,
            "cos_iota": np.arccos,
            "sin_beta": np.arcsin,
        },
        fill_dict={"fddot": 0.0},
    )


class FakeParentACA(FakeMultiShardACA):
    """Walker-sharded WDM parent stand-in (real settings, fake shards).

    Adds to the base fake the surface ``SubBandBuffer`` /
    ``fill_buffer_residual_and_psd_from_acs`` and the routed parent engine
    consume: ``settings`` (real WDMSettings), ``nchannels``, XYZ
    cross-channel psd buffers (their per-row shape differs from the data's),
    and the ``*_shaped_view`` BandView accessors.
    """

    def __init__(self, wdm_settings, num_shards):
        nfa, nta = int(wdm_settings.Nf_active), int(wdm_settings.Nt_active)
        super().__init__((NCHAN, nfa, nta), NWALKERS, num_shards,
                         layout="blocked", dtype=float)
        self.settings = wdm_settings
        self.nchannels = NCHAN
        self.data_length = nfa * nta
        self.per_psd_shape = (NCHAN, NCHAN, nfa, nta)
        per_psd = int(np.prod(self.per_psd_shape))
        self.linear_psd_arr = [
            np.zeros(len(rows) * per_psd, dtype=float)
            for rows in self.gpu_splits
        ]
        self.min_freq_inds = None

    @property
    def psd_shaped(self):
        return [
            buf.reshape((len(rows),) + self.per_psd_shape)
            for buf, rows in zip(self.linear_psd_arr, self.gpu_splits)
        ]

    def data_shaped_view(self):
        return BandView(self, "data")

    def psd_shaped_view(self):
        return BandView(self, "psd")

    def reference_psd_rows(self):
        out = np.zeros((self.acs_total_entries,) + self.per_psd_shape)
        for s_i, split in enumerate(self.gpu_splits):
            shaped = self.psd_shaped[s_i]
            for intra, ac_i in enumerate(split):
                out[int(ac_i)] = shaped[intra]
        return out


def seed_parent(parent, rng_seed=SEED):
    """Per-GLOBAL-row deterministic distinct residual + invC rows."""
    nfa = parent.per_band_shape[1]
    nta = parent.per_band_shape[2]
    rng = np.random.default_rng(rng_seed)
    data = rng.normal(size=(NWALKERS, NCHAN, nfa, nta)) * 1e-22
    # walker-dependent perturbation of relative size 1e-3
    data *= (1.0 + 1e-3 * np.arange(NWALKERS))[:, None, None, None]
    invc = np.zeros((NWALKERS, NCHAN, NCHAN, nfa, nta))
    row_scale = 1.0 + 0.01 * np.arange(NWALKERS)
    for c in range(NCHAN):
        invc[:, c, c] = 1e44 * row_scale[:, None, None]
    for s_i, rows in enumerate(parent.gpu_splits):
        parent.data_shaped[s_i][...] = data[rows]
        parent.psd_shaped[s_i][...] = invc[rows]
    return data, invc


def build_sources(wdm, rng_seed=SEED + 1):
    """(1, NWALKERS, nleaves, 8) sampling-basis coords: one source per band
    in SRC_BANDS, per-walker 1e-3 multiplicative perturbations (VGB
    truth-null-like but bigger, so wrong-row reads are visible)."""
    rng = np.random.default_rng(rng_seed)
    ldf = float(wdm.layer_df)
    nleaves = len(SRC_BANDS)
    base = np.zeros((nleaves, 8))
    for k, b in enumerate(SRC_BANDS):
        f0_hz = (8.0 + 2.0 * b + 1.0 + 0.13 * (k + 1)) * ldf  # inside band b
        base[k] = [
            np.log(2e-21), f0_hz * 1e3, 1e-12 * (k + 1), 0.3 + 0.2 * k,
            0.1 * (k + 1) - 0.25, 0.2 + 0.1 * k, 0.5 + 0.3 * k,
            0.2 * (k + 1) - 0.5,
        ]
    coords = np.zeros((1, NWALKERS, nleaves, 8))
    for w in range(NWALKERS):
        pert = 1.0 + 1e-3 * rng.standard_normal(base.shape)
        coords[0, w] = base * pert
    inds = np.ones((1, NWALKERS, nleaves), dtype=bool)
    return coords, inds


def build_sorter(wdm, comp, transform, coords, inds, wk):
    ldf = float(wdm.layer_df)
    band_edges = ldf * (8.0 + 2.0 * np.arange(N_BANDS + 1))
    band_N_vals = np.full(N_BANDS, 128, dtype=int)
    branch = Branch(coords, inds=inds)
    sorter = BandSorter(
        branch, band_edges=band_edges, band_N_vals=band_N_vals,
        force_backend="cpu", transform_fn=transform, gb=comp,
        gb_wdm_comp=comp, waveform_kwargs=wk, max_data_store_size=512,
    )
    return sorter, band_edges


def all_specials(sorter):
    sp = np.unique(np.asarray(sorter.special_band_inds))
    return sp


def report(tag, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    d = np.abs(a - b)
    ok = np.array_equal(a, b)
    print(f"  [{tag}] bitwise_equal={ok} max|delta|={d.max() if d.size else 0.0:.3e}")
    return ok, d


# ----------------------------------------------------------------------
# Phase A: replica-comp numerics
# ----------------------------------------------------------------------

def phase_a(wdm, comp, transform, coords):
    print("== Phase A: prototype comp vs _device_local_gb_comp replica ==")
    replica = _device_local_gb_comp(comp, np, 1, 0)
    assert replica is not comp, "replica helper returned the prototype"

    nfa, nta = int(wdm.Nf_active), int(wdm.Nt_active)
    holder = FakeParentACA(wdm, 1)
    seed_parent(holder)
    params = transform.both_transforms(
        coords[0].reshape(-1, 8), xp=np)  # (nsrc, 9)
    di = np.repeat(np.arange(NWALKERS), coords.shape[2]).astype(np.int32)

    ok = True
    # fill_global parity
    t_proto = np.zeros(NWALKERS * NCHAN * nfa * nta)
    t_rep = np.zeros_like(t_proto)
    comp.fill_global_wdm(params, t_proto, data_index=di,
                         factors=np.ones(len(di)))
    replica.fill_global_wdm(params, t_rep, data_index=di,
                            factors=np.ones(len(di)))
    ok &= report("fill_global", t_proto, t_rep)[0]

    ll_p = comp.get_ll_wdm(params, holder, data_index=di, noise_index=di)
    dh_p, hh_p = comp.d_h_out.copy(), comp.h_h_out.copy()
    ll_r = replica.get_ll_wdm(params, holder, data_index=di, noise_index=di)
    ok &= report("get_ll", ll_p, ll_r)[0]
    ok &= report("d_h", dh_p, replica.d_h_out)[0]
    ok &= report("h_h", hh_p, replica.h_h_out)[0]

    out_p = comp.get_swap_ll_wdm(params, np.roll(params, 1, axis=0), holder,
                                 data_index=di, noise_index=di)
    out_r = replica.get_swap_ll_wdm(params, np.roll(params, 1, axis=0),
                                    holder, data_index=di, noise_index=di)
    for i, name in enumerate(("like_add", "like_rem", "d_h_a", "d_h_r",
                              "aa", "rr", "ar")):
        ok &= report(f"swap.{name}", out_p[i], out_r[i])[0]
    print(f"Phase A {'CLEAN' if ok else 'DIVERGES'}\n")
    return ok


# ----------------------------------------------------------------------
# Phase B: buffer fill from a walker-sharded parent
# ----------------------------------------------------------------------

def build_leg(wdm, comp, transform, coords, inds, wk, num_shards):
    parent = FakeParentACA(wdm, num_shards)
    seed_parent(parent)
    sorter, band_edges = build_sorter(wdm, comp, transform, coords, inds, wk)
    specials = all_specials(sorter)
    buf = sorter.get_buffer(parent, specials)
    return parent, sorter, specials, buf


def phase_b(wdm, comp, transform, coords, inds, wk):
    print("== Phase B: buffer fill parity (2-shard parent vs 1-shard) ==")
    _, _, sp1, buf1 = build_leg(wdm, comp, transform, coords, inds, wk, 1)
    _, _, sp2, buf2 = build_leg(wdm, comp, transform, coords, inds, wk, 2)
    assert np.array_equal(np.asarray(sp1), np.asarray(sp2))
    ok, d = report("band_buffer", buf1.band_buffer, buf2.band_buffer)
    if not ok:
        bad = np.where(np.abs(np.asarray(buf1.band_buffer)
                              - np.asarray(buf2.band_buffer))
                       .reshape(len(sp1), -1).max(axis=1) > 0)[0]
        print(f"    differing cells: {bad.tolist()}")
        for c in bad[:8]:
            t, w, b = np.asarray(buf1.unique_band_combos)[c]
            print(f"      cell {c}: temp {t} walker {w} band {b}")
    ok2, _ = report("psd_buffer", buf1.psd_buffer, buf2.psd_buffer)
    print(f"Phase B {'CLEAN' if (ok and ok2) else 'DIVERGES'}\n")
    return ok and ok2, (buf1, buf2)


# ----------------------------------------------------------------------
# Phase C: routed scoring of band-sharded cells
# ----------------------------------------------------------------------

class FakeCellACA(FakeMultiShardACA):
    """Band-sharded cell holder mirroring a multi-shard SubBandBuffer."""

    def __init__(self, wdm_settings, n_cells, gpu_assignment, data, psd):
        nfa, nta = int(wdm_settings.Nf_active), int(wdm_settings.Nt_active)
        num_shards = 1 if gpu_assignment is None else (
            int(np.max(gpu_assignment)) + 1)
        super().__init__((NCHAN, nfa, nta), n_cells, max(num_shards, 1),
                         layout="striped", dtype=float)
        if gpu_assignment is not None:
            self.gpus = sorted(set(int(g) for g in gpu_assignment))
            self.gpu_map = np.asarray(gpu_assignment, dtype=int)
            self.gpu_splits = [np.where(self.gpu_map == g)[0]
                               for g in self.gpus]
            self.split_map = np.zeros(n_cells, dtype=int)
            for s_i, split in enumerate(self.gpu_splits):
                self.split_map[split] = s_i
        self.settings = wdm_settings
        self.nchannels = NCHAN
        self.per_psd_shape = (NCHAN, NCHAN, nfa, nta)
        self.min_freq_inds = None
        self.linear_data_arr = [
            np.ascontiguousarray(data[rows]).ravel()
            for rows in self.gpu_splits
        ]
        self.linear_psd_arr = [
            np.ascontiguousarray(psd[rows]).ravel()
            for rows in self.gpu_splits
        ]

    @property
    def psd_shaped(self):
        return [
            buf.reshape((len(rows),) + self.per_psd_shape)
            for buf, rows in zip(self.linear_psd_arr, self.gpu_splits)
        ]


def make_engine(wdm, comp):
    return make_routed_band_engine(
        wdm, xp=np, gb_wdm_comp=comp, gb_fd_comp=None,
        nchannels=NCHAN, tdi_channel_setup="XYZ",
        df=float(wdm.layer_df), start_freq_inds=None,
        data_length=int(wdm.Nf_active) * int(wdm.Nt_active),
    )


def phase_c(wdm, comp, transform, coords, inds, wk, buf):
    print("== Phase C: routed cell scoring parity (band-sharded cells) ==")
    n_cells = int(buf.num_bands_now)
    data = np.asarray(buf.band_buffer).copy()
    psd = np.asarray(buf.psd_buffer).copy()
    combos = np.asarray(buf.unique_band_combos)
    assign = band_gpu_assignment(n_cells, [0, 1], group_ids=combos[:, 2])
    holder_1 = FakeCellACA(wdm, n_cells, None, data, psd)
    holder_2 = FakeCellACA(wdm, n_cells, assign, data, psd)

    # score each cell's own source params (the picked-source shape)
    sorter, _ = build_sorter(wdm, comp, transform, coords, inds, wk)
    # one source per cell: match each cell's special index
    src_special = np.asarray(sorter.special_band_inds)
    cell_special = np.asarray(buf.special_indices_unique)
    src_rows = np.array([np.where(src_special == s)[0][0]
                         for s in cell_special])
    params_phys = transform.both_transforms(
        np.asarray(sorter.coords)[src_rows], xp=np)
    slots = np.arange(n_cells, dtype=np.int32)

    ok = True
    for label, pm in (("plain", False), ("phase_max", True)):
        eng1 = make_engine(wdm, comp)
        ll1 = eng1.get_ll(holder_1, params_phys, data_index=slots,
                          noise_index=slots, N_vals=None,
                          phase_maximize=pm, waveform_kwargs={})
        dh1, hh1 = np.asarray(eng1.d_h_out), np.asarray(eng1.h_h_out)
        eng2 = make_engine(wdm, comp)
        ll2 = eng2.get_ll(holder_2, params_phys, data_index=slots,
                          noise_index=slots, N_vals=None,
                          phase_maximize=pm, waveform_kwargs={})
        dh2, hh2 = np.asarray(eng2.d_h_out), np.asarray(eng2.h_h_out)
        o1, d = report(f"get_ll[{label}]", ll1, ll2)
        o2, _ = report(f"d_h[{label}]", dh1, dh2)
        o3, _ = report(f"h_h[{label}]", hh1, hh2)
        ok &= o1 and o2 and o3
        if not (o1 and o2 and o3):
            bad = np.where(np.abs(np.asarray(ll1) - np.asarray(ll2)) > 0)[0]
            for c in bad[:8]:
                t, w, b = combos[c]
                print(f"      cell {c}: temp {t} walker {w} band {b} "
                      f"shard {holder_2.split_map[c]} "
                      f"dll {float(ll1[c] - ll2[c]):.3e}")

    # swap parity
    eng1 = make_engine(wdm, comp)
    r1 = eng1.get_swap_ll(holder_1, params_phys,
                          np.roll(params_phys, 1, axis=0),
                          data_index=slots, noise_index=slots, N_vals=None,
                          waveform_kwargs={})
    eng2 = make_engine(wdm, comp)
    r2 = eng2.get_swap_ll(holder_2, params_phys,
                          np.roll(params_phys, 1, axis=0),
                          data_index=slots, noise_index=slots, N_vals=None,
                          waveform_kwargs={})
    for f in ("ll_diff", "d_h_add", "d_h_remove", "hh_add", "hh_remove",
              "hh_cross", "opt_snr_add"):
        v1, v2 = getattr(r1, f), getattr(r2, f)
        if v1 is None and v2 is None:
            continue
        ok &= report(f"swap.{f}", v1, v2)[0]
    print(f"Phase C {'CLEAN' if ok else 'DIVERGES'}\n")
    return ok


# ----------------------------------------------------------------------
# Phase D: full open -> fill -> score -> write-back -> close cycle
# ----------------------------------------------------------------------

def phase_d(wdm, comp, transform, coords, inds, wk):
    print("== Phase D: parity-unit round trip on the walker-sharded parent ==")
    legs = {}
    for num_shards in (1, 2):
        parent = FakeParentACA(wdm, num_shards)
        seed_parent(parent)
        sorter, _ = build_sorter(wdm, comp, transform, coords, inds, wk)
        engine = make_engine(wdm, comp)

        phys = transform.both_transforms(np.asarray(sorter.coords), xp=np)
        walkers = np.asarray(sorter.walker_inds).astype(np.int32)
        N_vals = np.asarray(sorter.N_vals)

        # OPEN: cold-chain sources back into the residual
        engine.fill_template(parent, phys, walkers, N_vals, factor=+1,
                             waveform_kwargs=wk)
        open_rows = parent.reference_rows().copy()

        # FILL buffer from the opened parent (real production path)
        specials = all_specials(sorter)
        buf = sorter.get_buffer(parent, specials)

        # pick one source per cell (every cell), remove from its cell,
        # score ref, perturb, score new, write back the new params
        src_special = np.asarray(sorter.special_band_inds)
        cell_special = np.asarray(buf.special_indices_unique)
        src_rows = np.array([np.where(src_special == s)[0][0]
                             for s in cell_special])
        curr = np.asarray(sorter.coords)[src_rows].copy()
        slots = np.arange(len(cell_special), dtype=np.int32)
        Nv = np.asarray(sorter.N_vals)[src_rows]

        buf.remove_sources_from_band_buffer(curr, slots, Nv)
        ll_ref = np.asarray(buf.get_add_ll(curr, slots, slots, Nv))
        new = curr.copy()
        new[:, 3] += 0.05          # phi0-ish deterministic nudge
        new[:, 0] += 1e-3          # lnA nudge
        new_ll = np.asarray(buf.get_add_ll(new, slots, slots, Nv))
        buf.add_sources_to_band_buffer(new, slots, Nv)

        # write back into the sorter (accepted everywhere)
        sorter.coords[src_rows] = new

        # CLOSE: subtract the UPDATED cold chain from the parent
        phys_new = transform.both_transforms(np.asarray(sorter.coords), xp=np)
        engine.fill_template(parent, phys_new, walkers, N_vals, factor=-1,
                             waveform_kwargs=wk)

        legs[num_shards] = dict(
            open_rows=open_rows, ll_ref=ll_ref, new_ll=new_ll,
            final_rows=parent.reference_rows().copy(),
            band_buffer=np.asarray(buf.band_buffer).copy(),
            combos=np.asarray(buf.unique_band_combos),
        )

    ok = True
    for key in ("open_rows", "ll_ref", "new_ll", "band_buffer",
                "final_rows"):
        o, d = report(key, legs[1][key], legs[2][key])
        ok &= o
        if not o and key in ("open_rows", "final_rows"):
            per_w = np.abs(legs[1][key] - legs[2][key]).reshape(
                NWALKERS, -1).max(axis=1)
            print(f"    per-walker max|delta|: {per_w}")
        if not o and key in ("ll_ref", "new_ll"):
            bad = np.where(np.abs(legs[1][key] - legs[2][key]) > 0)[0]
            for c in bad[:8]:
                t, w, b = legs[1]["combos"][c]
                print(f"      cell {c}: temp {t} walker {w} band {b} "
                      f"delta {float(legs[1][key][c] - legs[2][key][c]):.3e}")
    print(f"Phase D {'CLEAN' if ok else 'DIVERGES'}\n")
    return ok


# ----------------------------------------------------------------------
# Phase E/F: MULTI-SHARD SubBandBuffer (the production shape)
# ----------------------------------------------------------------------

def shardify_buffer(buf, gpu_assignment):
    """Convert a freshly built single-shard SubBandBuffer into a fake
    2-shard one IN PLACE (CPU stand-in for the multi-GPU band-sharded
    buffer). The linear buffers are re-split by ``gpu_assignment`` exactly
    like ``AnalysisContainerArray.__init__`` does with a real
    ``gpu_assignment``; ``xp`` is swapped for a RecordingXp so BandView /
    device contexts run their multi-shard branches."""
    from _multishard import RecordingXp

    n = int(buf.num_bands_now)
    assign = np.asarray(gpu_assignment, dtype=int)
    gpus = sorted(set(int(g) for g in assign))
    splits = [np.where(assign == g)[0] for g in gpus]

    rows_data = buf.linear_data_arr[0].reshape(n, -1)
    rows_psd = buf.linear_psd_arr[0].reshape(n, -1)
    buf.linear_data_arr = [
        np.ascontiguousarray(rows_data[s]).reshape(-1) for s in splits
    ]
    buf.linear_psd_arr = [
        np.ascontiguousarray(rows_psd[s]).reshape(-1) for s in splits
    ]
    buf.gpus = gpus
    buf.gpu_splits = splits
    buf.gpu_map = assign.copy()
    buf.split_map = np.zeros(n, dtype=int)
    for s_i, split in enumerate(splits):
        buf.split_map[split] = s_i

    class _Shardified(type(buf)):
        @property
        def xp(self):
            fx = getattr(self, "_fake_xp", None)
            return fx if fx is not None else self.backend.xp

    _Shardified.__name__ = type(buf).__name__ + "Shardified"
    buf.__class__ = _Shardified
    buf._fake_xp = RecordingXp()
    # drop any cached shard views built against the old layout
    if hasattr(buf, "_shard_holder_views"):
        del buf._shard_holder_views
    return buf


def run_unit(wdm, comp, transform, coords, inds, wk, *, shard_buffer,
             shard_parent, partial_rotate=False):
    """One production-shaped parity unit: open -> (re)fill buffer ->
    remove/score/add per cell -> write back -> close. Returns a dict of
    checkpoint arrays (all in GLOBAL row/cell order)."""
    parent = FakeParentACA(wdm, 2 if shard_parent else 1)
    seed_parent(parent)
    sorter, _ = build_sorter(wdm, comp, transform, coords, inds, wk)
    engine = make_engine(wdm, comp)

    phys = transform.both_transforms(np.asarray(sorter.coords), xp=np)
    walkers = np.asarray(sorter.walker_inds).astype(np.int32)
    N_all = np.asarray(sorter.N_vals)

    engine.fill_template(parent, phys, walkers, N_all, factor=+1,
                         waveform_kwargs=wk)

    specials = all_specials(sorter)
    buf = sorter.get_buffer(parent, specials)
    if shard_buffer:
        combos = np.asarray(buf.unique_band_combos)
        assign = band_gpu_assignment(
            int(buf.num_bands_now), [0, 1], group_ids=combos[:, 2])
        shardify_buffer(buf, assign)
        # production steady state (GB_BUFFER_PERSIST): the cached
        # multi-shard buffer is REBOUND, i.e. refill + reinject through
        # the multi-shard BandView / routed-engine paths.
        sorter.get_buffer(
            parent, specials,
            inds_fill=np.arange(int(specials.shape[0])), buffer_obj=buf,
        )
    filled = dict(
        band_buffer=np.asarray(buf.band_buffer[np.arange(buf.num_bands_now)]
                               if shard_buffer else buf.band_buffer).copy(),
        psd_buffer=np.asarray(buf.psd_buffer[np.arange(buf.num_bands_now)]
                              if shard_buffer else buf.psd_buffer).copy(),
    )

    src_special = np.asarray(sorter.special_band_inds)
    cell_special = np.asarray(buf.special_indices_unique)
    src_rows = np.array([np.where(src_special == s)[0][0]
                         for s in cell_special])
    curr = np.asarray(sorter.coords)[src_rows].copy()
    slots = np.arange(len(cell_special), dtype=np.int32)
    Nv = np.asarray(sorter.N_vals)[src_rows]

    buf.remove_sources_from_band_buffer(curr, slots, Nv)
    ll_ref = np.asarray(buf.get_add_ll(curr, slots, slots, Nv)).copy()
    new = curr.copy()
    new[:, 3] += 0.05
    new[:, 0] += 1e-3
    new_ll = np.asarray(buf.get_add_ll(new, slots, slots, Nv)).copy()
    new_ll_pm = np.asarray(
        buf.get_add_ll(new, slots, slots, Nv, phase_maximize=True)).copy()
    buf.add_sources_to_band_buffer(new, slots, Nv)
    sorter.coords[src_rows] = new

    post_write = np.asarray(
        buf.band_buffer[np.arange(buf.num_bands_now)]
        if shard_buffer else buf.band_buffer).copy()

    rotate = {}
    if partial_rotate:
        # scheduler-advance shape: swap HALF the slots to (temp 0, other
        # walker, same band) cells via the partial-rebind path, refill,
        # then rescore the untouched half (their content must be intact).
        n = int(buf.num_bands_now)
        rot_slots = np.arange(0, n, 2)
        keep_slots = np.arange(1, n, 2)
        t_i, w_i, b_i = (np.asarray(x) for x in
                         buf.get_separate_inds_from_special_index(
                             cell_special))
        new_specials = np.asarray(pack_special_index(
            t_i[rot_slots], (w_i[rot_slots] + 1) % NWALKERS,
            b_i[rot_slots], NWALKERS))
        sorter.get_buffer(parent, new_specials,
                          inds_fill=rot_slots, buffer_obj=buf)
        rotate["post_rotate_buffer"] = np.asarray(
            buf.band_buffer[np.arange(n)]
            if shard_buffer else buf.band_buffer).copy()
        keep_ll = np.asarray(buf.get_add_ll(
            new[keep_slots], keep_slots.astype(np.int32),
            keep_slots.astype(np.int32), Nv[keep_slots])).copy()
        rotate["keep_ll"] = keep_ll

    phys_new = transform.both_transforms(np.asarray(sorter.coords), xp=np)
    engine.fill_template(parent, phys_new, walkers, N_all, factor=-1,
                         waveform_kwargs=wk)

    out = dict(filled=filled, ll_ref=ll_ref, new_ll=new_ll,
               new_ll_pm=new_ll_pm, post_write=post_write,
               final_rows=parent.reference_rows().copy(),
               combos=np.asarray(buf.unique_band_combos).copy(),
               split_map=(np.asarray(buf.split_map).copy()
                          if shard_buffer else None))
    out.update(rotate)
    return out


def phase_e(wdm, comp, transform, coords, inds, wk, partial_rotate=False):
    label = "F (with slot rotation)" if partial_rotate else "E"
    print(f"== Phase {label}: multi-shard BUFFER + multi-shard parent "
          "unit vs single-shard ==")
    ref = run_unit(wdm, comp, transform, coords, inds, wk,
                   shard_buffer=False, shard_parent=False,
                   partial_rotate=partial_rotate)
    two = run_unit(wdm, comp, transform, coords, inds, wk,
                   shard_buffer=True, shard_parent=True,
                   partial_rotate=partial_rotate)
    ok = True
    keys = ["ll_ref", "new_ll", "new_ll_pm", "post_write", "final_rows"]
    if partial_rotate:
        keys += ["post_rotate_buffer", "keep_ll"]
    o, _ = report("fill.band_buffer", ref["filled"]["band_buffer"],
                  two["filled"]["band_buffer"])
    ok &= o
    o, _ = report("fill.psd_buffer", ref["filled"]["psd_buffer"],
                  two["filled"]["psd_buffer"])
    ok &= o
    for key in keys:
        o, d = report(key, ref[key], two[key])
        ok &= o
        if not o and ref[key].ndim == 1 and len(ref[key]) == len(ref["combos"]):
            bad = np.where(np.abs(ref[key] - two[key]) > 0)[0]
            for c in bad[:10]:
                t, w, b = ref["combos"][c]
                sh = (two["split_map"][c]
                      if two["split_map"] is not None else "?")
                print(f"      cell {c}: temp {t} walker {w} band {b} "
                      f"shard {sh} delta {float(ref[key][c] - two[key][c]):.3e}")
        elif not o and key in ("post_write", "final_rows",
                               "post_rotate_buffer"):
            per_row = np.abs(ref[key] - two[key]).reshape(
                ref[key].shape[0], -1).max(axis=1)
            bad = np.where(per_row > 0)[0]
            print(f"    differing rows: {bad.tolist()} "
                  f"max {per_row.max():.3e}")
    print(f"Phase {label} {'CLEAN' if ok else 'DIVERGES'}\n")
    return ok


# ----------------------------------------------------------------------
# Phase G: the GPU-only replica-settings rebuild, with t0 != 0
# ----------------------------------------------------------------------

def phase_g(transform, coords):
    """Reproduce the DEVICE-SIDE settings rebuild the earlier phases could
    not reach on CPU.

    On a real GPU, ``_device_local_gb_comp`` (shard-1 engine replica)
    rebuilds the comp's WDMSettings through ``_device_local_domain_settings``
    -> ``WDMSettings(*settings.args, **settings.kwargs)``. With plain numpy
    ``current_device`` returns None and the SHARED settings are reused
    (t0 preserved -> phases A-F clean); under a RecordingXp the rebuild
    actually runs, exactly like production. The stock VGB/GB builds set
    ``wdm.t0 = data_t0`` before building the comp, so t0 != 0 is the
    production configuration whenever the data does not start at t=0.
    """
    print("== Phase G: shard-1 replica rebuild with t0 != 0 "
          "(the on-GPU path) ==")
    from _multishard import RecordingXp

    T0 = 7776000.0  # 3 months: a mojito-like nonzero data start
    wdm_t0 = WDMSettings(Nf=NF, Nt=NT, dt=DT, t0=T0, force_backend="cpu")
    comp_t0 = GBWDMComputations(
        wdm_t0, t_ref=T0, Nt_sub=16, n_pad=2, N_sparse=64,
        tdi_config="1st generation", force_backend="cpu",
    )
    rxp = RecordingXp()
    replica = _device_local_gb_comp(comp_t0, rxp, 1, 0)
    ok = True
    if replica is comp_t0:
        print("  replica helper returned the prototype (settings shared); "
              "nothing to compare")
    else:
        print(f"  prototype t_obs_start = {comp_t0.t_obs_start!r}")
        print(f"  replica   t_obs_start = {replica.t_obs_start!r}")
        print(f"  prototype wdm_settings.t0 = {comp_t0.wdm_settings.t0!r}")
        print(f"  replica   wdm_settings.t0 = {replica.wdm_settings.t0!r}")
        ok &= report("chunk_t_starts", comp_t0.chunk_t_starts,
                     replica.chunk_t_starts)[0]

        # scoring divergence on identical inputs
        holder = FakeParentACA(wdm_t0, 1)
        seed_parent(holder)
        params = transform.both_transforms(coords[0].reshape(-1, 8), xp=np)
        di = np.repeat(np.arange(NWALKERS), coords.shape[2]).astype(np.int32)
        ll_p = comp_t0.get_ll_wdm(params, holder, data_index=di,
                                  noise_index=di)
        ll_r = replica.get_ll_wdm(params, holder, data_index=di,
                                  noise_index=di)
        ok &= report("get_ll (proto vs replica)", ll_p, ll_r)[0]
        ok &= report("d_h", comp_t0.d_h_out, replica.d_h_out)[0]
    print(f"Phase G {'CLEAN' if ok else 'DIVERGES'}\n")
    return ok


def main():
    wdm, comp = build_comp()
    transform = build_transform()
    coords, inds = build_sources(wdm)
    wk = dict(dt=DT, T=float(wdm.Tobs), tdi_channel_setup="XYZ")

    results = {}
    results["A"] = phase_a(wdm, comp, transform, coords)
    results["B"], (buf1, _buf2) = phase_b(wdm, comp, transform, coords,
                                          inds, wk)
    results["C"] = phase_c(wdm, comp, transform, coords, inds, wk, buf1)
    results["D"] = phase_d(wdm, comp, transform, coords, inds, wk)
    results["E"] = phase_e(wdm, comp, transform, coords, inds, wk)
    results["F"] = phase_e(wdm, comp, transform, coords, inds, wk,
                           partial_rotate=True)
    results["G"] = phase_g(transform, coords)

    print("== SUMMARY ==")
    for k, v in results.items():
        print(f"  Phase {k}: {'CLEAN' if v else 'DIVERGES'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
