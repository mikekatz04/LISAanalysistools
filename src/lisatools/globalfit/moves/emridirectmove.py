"""EMRI add/remove move scored by the direct-to-WDM template (``EMRI_LIKELIHOOD=direct``).

The production EMRI move (stock name ``emri_pe``): the 9mo / 12mo launchers set
``EMRI_LIKELIHOOD=direct``, and the stock default ``auto`` resolves to ``direct`` on a WDM / XYZ
run (``stock.erebor.source_runtime.resolve_emri_direct_cfg``). Built by
``recipe.EMRIDirectMoveBuilder`` from ``source_runtime.build_emri_move_runtime``; it runs in
``source_search``, every cycle of ``gb_search_1`` / ``gb_search_2`` and in ``full_pe``.

:class:`EMRIDirectLikeMove` keeps ALL of :class:`ResidualAddOneRemoveOneMove`'s
choreography (per-leaf expose/fold, in-model repeats, per-leaf tempering, eigen-table
sweeps, replica replays) and swaps ONLY the scoring, as :class:`MBHBatchedLikeMove` and
:class:`SOBBHChunkedLikeMove` do for their branches:

* every chunk of ``batch_max_size`` rows is ONE
  :meth:`~lisatools.sources.emri.direct_signal_gen.EMRIDirectWDMSignalGen.templates` call:
  one FEW call per row, ONE dense TDI-on-the-fly response launch for the chunk, ONE
  table lookup + scatter-add, cropped to the run's active WDM box;
* every row is scored against ITS OWN container (per-walker residual and PSD), one
  batched ``inner_product`` pair per walker in the chunk, one host pull per chunk;
* the convention bridge is the SOBBH/MBH one: ``compute_like`` returns
  ``offset[walker] + <r|h> - 1/2 <h|h>`` with ``offset = acs.likelihood()`` on the
  freshly exposed residual, i.e. the container path's ``-1/2 <r-h|r-h>`` + noise term.

The residual expose/fold runs through the containers' installed ``signal_gen`` (the base
class's per-row path). In the global fit with ``EMRI_LIKELIHOOD=direct`` that generator is
the direct-to-WDM template too (``stock.erebor.source_runtime.SourceSignalGen``), so the
engine's residual rebuilds, the fills and the scoring use one template family and the
residual every other branch sees holds exactly what the sampler scored. (With a production
generator installed instead, as in ``scripts/emri/emri_direct_fit_wiring_check.py``, scoring
is still self-consistent: the leaf's ``prev_logl`` and every proposal are scored by the same
direct generator against the same exposed residual.) The move's own ``waveform_gen`` stays
the PRODUCTION wrap: it owns the cross-check and the fallback.

Two checks therefore carry a per-point TOLERANCE here instead of the base's exact-algebra
gates (``EMRI_CHECK_LL_TOL`` 1 nat plus a template-mismatch term that grows with the point's
``<h|h>``, ``EMRI_CHECK_LL_MM``; :meth:`EMRIDirectLikeMove._point_tolerance`), both on the
COLD rung only:

* :meth:`_verify_prev_logl` (every ``EMRI_CHECK_LL_EVERY``-th visit, default 10):
  direct ``prev_logl`` vs the production container path at the same points;
* :meth:`_verify_entry_vs_acs`: direct ``prev_logl`` on the exposed residual vs the
  pre-expose full lnL (the installed generator's template in the residual: the direct one
  in the fit, so this is then exact up to round-off). The expose-sign class of bug it
  exists for moves lnL by ~SNR^2, far above the tolerance.

A chunk the direct path cannot build (any exception other than a per-row FEW domain
refusal: GPU OOM, a table error, a missing compiled kernel) is scored through the
per-row container path with the production generator (``waveform_gen``), LOUDLY
(warned once per leaf, counted in
``n_batch_fallbacks`` and the ``[EMRI_DIRECT]`` telemetry). A row FEW refuses scores
``-1e300``, as on the container path.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from ...diagnostic import inner_product
from ...domains import WDMSettings, WDMSignal
from ...utils.device import device_context
from ...utils.utility import asnumpy, get_array_module
from .addremovemove import ResidualAddOneRemoveOneMove
from .mbhbatchedmove import MBHBatchedLikeMove

logger = logging.getLogger(__name__)

__all__ = ["EMRIDirectLikeMove"]


class EMRIDirectLikeMove(ResidualAddOneRemoveOneMove):
    """Add/remove move for the EMRI branch scored through the direct-to-WDM template.

    The ``propose(model, state)`` contract is the base class's: the
    :class:`~lisatools.globalfit.moves.walkerfanout.WalkerFanoutMixin` ``propose`` runs
    :meth:`ResidualAddOneRemoveOneMove.propose_local` on every compute rank's walker block;
    per leaf it exposes the cold-chain source into the residual, calls
    :meth:`setup_likelihood_here`, scores ``prev_logl`` and every in-model proposal through
    ``compute_like`` -> :meth:`compute_like_local`, runs the per-leaf ladder swaps, records
    ``<d|h>`` / ``<h|h>`` (:meth:`_record_leaf_inner_products`) and folds the accepted
    source back. The tempered ensemble lives in the ``emri`` sub-state; the base mirrors its
    row 0 into the main state with ``_sync_cold_row``. Only the scoring is replaced here.

    Args:
        *args: Positional arguments of :class:`ResidualAddOneRemoveOneMove`.
            ``waveform_gen`` stays the production generator: it owns the cross-check.
        direct_gen: the :class:`~lisatools.sources.emri.direct_signal_gen.EMRIDirectWDMSignalGen`,
            or a ``DeviceLocalWaveGen`` resolving to one per device. Required (``ValueError``
            when ``None``).
        batch_max_size (int): rows per direct generation (``EMRI_BATCH_MAX_SIZE``, default
            8). Each row holds a full-grid ``(3, Nf, Nt)`` float64 accumulator while it is
            built (~150 MB on the 6-month grid).
        **kwargs: Keyword arguments of the base. ``dcga`` must be ``None`` (``ValueError``
            otherwise).

    Raises:
        ValueError: ``dcga`` given, ``direct_gen`` missing, or the containers' data are
            not on a :class:`~lisatools.domains.WDMSettings` domain.

    Attributes:
        direct_gen: See ``direct_gen``.
        batch_max_size (int): See ``batch_max_size`` (floored at 1).
        n_batch_fallbacks (int): Chunks scored through the container fallback.
        last_batch_error (Exception or None): The last chunk exception that triggered it.
        check_ll_tol (float): ``EMRI_CHECK_LL_TOL`` (default ``1.0`` nat).
        check_ll_mm (float): ``EMRI_CHECK_LL_MM`` (default ``3e-4``), the mismatch term of
            :meth:`_point_tolerance`.

    Note:
        Env knobs read here or by the base with this branch's ``EMRI`` prefix:
        ``EMRI_CHECK_LL`` (``0`` off, ``1`` warn, ``strict`` raise; base default
        ``ADDREMOVE_CHECK_LL`` then ``1``; the launchers set ``0``),
        ``EMRI_CHECK_LL_EVERY`` (class default every 10th visit), ``EMRI_CHECK_LL_TOL``,
        ``EMRI_CHECK_LL_MM`` and ``EMRI_RECORD_DH`` (class default ``1``: the record is
        one direct batch).

    Example:
        Built by the stock layer, not by hand (``stock.erebor.source_runtime``):

        .. code-block:: python

            from lisatools.globalfit.recipe import EMRIDirectMoveBuilder

            _, moves = EMRIDirectMoveBuilder(
                wave_gen=wave_gen,          # production EMRI wrap (fallback + check)
                direct_gen=direct,          # DeviceLocalWaveGen(get_emri_direct_gen, ...)
                batch_max_size=8,
            ).build(None, curr, acs, priors, state)
            emri_pe = moves[0]
    """

    _record_dh_default = "1"
    #: the cross-check re-scores the cold rung through the PRODUCTION generator
    _check_ll_every_default = "10"

    def __init__(self, *args, direct_gen=None, batch_max_size=8, **kwargs):
        if kwargs.get("dcga") is not None:
            raise ValueError(
                "EMRIDirectLikeMove has no DCGA (replica) path: the direct generator scores "
                "against the ACA containers directly, and multi-GPU walker shards are served "
                "by per-shard routing inside compute_like_local. Build it without dcga= "
                "(use_dcga=False)."
            )
        if direct_gen is None:
            raise ValueError("EMRIDirectLikeMove requires direct_gen= (the direct-to-WDM adapter).")
        super().__init__(*args, **kwargs)
        self.direct_gen = direct_gen
        self.batch_max_size = max(1, int(batch_max_size))
        self._wdm = self.acs.acs.flatten()[0].data.settings
        if not isinstance(self._wdm, WDMSettings):
            raise ValueError(
                "EMRI_LIKELIHOOD=direct needs a WDM run domain; the containers carry "
                f"{type(self._wdm).__name__}."
            )
        self._exposed_offset = None
        self._box_checked = False
        self.n_batch_fallbacks = 0
        self.last_batch_error = None
        self._warned_fallback_leaf = None
        self.check_ll_tol = float(os.environ.get(f"{self._dbg_prefix}_CHECK_LL_TOL", "1.0"))
        # template-mismatch budget of the tolerance (see _point_tolerance)
        self.check_ll_mm = float(os.environ.get(f"{self._dbg_prefix}_CHECK_LL_MM", "3e-4"))
        self._gen_kwargs = dict(self.waveform_gen_kwargs or {})
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats():
        """Fresh per-leaf telemetry dict for the ``[EMRI_DIRECT]`` line (:meth:`_flush_stats`)."""
        return dict(leaf=None, rows=0, seconds=0.0, chunks=0, fallbacks=0, refused=0,
                    gen_s=0.0, score_s=0.0)

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------

    def _adapter(self):
        """The direct generator for the CURRENT device (resolves a ``DeviceLocalWaveGen``)."""
        resolve = getattr(self.direct_gen, "_resolve", None)
        return resolve() if callable(resolve) else self.direct_gen

    def _split_rows(self, idx):
        """``[(device, positions)]`` grouping ``idx`` rows by owning walker shard."""
        return MBHBatchedLikeMove._split_rows_static(self.acs, idx)

    def _device_context(self, device):
        """cupy device context on ``device`` (a null context on numpy or ``None``)."""
        return device_context(getattr(self.acs, "xp", None), device)

    # ------------------------------------------------------------------
    # Gram information matrix (EMRI_EIGEN_INFO=gram; the route itself is
    # ResidualAddOneRemoveOneMove._gram_info -- these are its two hooks)
    # ------------------------------------------------------------------

    def _gram_context(self, walker):
        """Enter the device that owns walker ``walker``'s shard."""
        (device, _), = self._split_rows(np.array([int(walker)]))
        return self._device_context(device)

    def _gram_templates(self, coords, walker):
        """The direct templates for the Gram rows: ``(arr, box)`` with one
        row of ``arr`` per waveform-basis row of ``coords``, on the run's
        active WDM box -- the templates the scorer differences against.

        Built in ``batch_max_size`` chunks (the scorer's chunking) and
        concatenated, so the whole ``2 * ndim + 1`` stack is resident at
        once. ``walker`` is unused: the direct template does not depend on
        the walker (its noise enters through the caller's inner products).
        A row the generator refuses raises, and the refresh falls back to
        the likelihood route.
        """
        adapter = self._adapter()
        self._check_box(adapter)
        coords = np.atleast_2d(np.asarray(asnumpy(coords), dtype=np.float64))
        parts = []
        for lo in range(0, coords.shape[0], self.batch_max_size):
            arr, ok = adapter.templates(coords[lo: lo + self.batch_max_size],
                                        **self._gen_kwargs)
            if not np.all(np.asarray(asnumpy(ok), dtype=bool)):
                raise ValueError("EMRI Gram rows refused by the direct generator")
            parts.append(arr)
        xp = get_array_module(parts[0])
        return xp.concatenate(parts), self._wdm

    # ------------------------------------------------------------------
    # likelihood
    # ------------------------------------------------------------------

    def setup_likelihood_here(self, coords):
        """Arm the per-walker exposed-residual offset."""
        self._flush_stats()
        self._exposed_offset = np.asarray(asnumpy(self.acs.likelihood()), dtype=float)
        super().setup_likelihood_here(coords)

    def compute_like_local(self, coords_in, data_index):
        """Rank-local direct-to-WDM scorer (the base ``compute_like`` calls it).

        Rows are grouped by owning walker shard, chunked by ``batch_max_size``
        and scored by :meth:`_score_chunk`. Side outputs ``self._last_d_h`` /
        ``self._last_h_h`` hold the per-row ``<r|h>`` / ``<h|h>`` (NaN where not
        scored).

        Args:
            coords_in (xp.ndarray): Waveform-basis rows, shape ``(n, ndim)``.
            data_index (xp.ndarray): Walker (container) index per row, shape ``(n,)``.

        Returns:
            np.ndarray: ``(n,)`` log-likelihoods in the container convention
            (``offset[walker] + <r|h> - 1/2 <h|h>``); ``-1e300`` for a non-finite
            row or one FEW refuses.

        Raises:
            RuntimeError: Called before :meth:`setup_likelihood_here` armed the
                exposed-residual offset.
        """
        if self._dcga is not None:  # unreachable (ctor guard); keep loud
            raise NotImplementedError("EMRIDirectLikeMove has no DCGA path.")
        if self._exposed_offset is None:
            raise RuntimeError(
                "compute_like_local called before setup_likelihood_here armed the "
                "exposed-residual offset (propose() choreography violated)."
            )
        t_start = time.perf_counter()
        coords = np.atleast_2d(np.asarray(asnumpy(coords_in), dtype=np.float64))
        idx = np.asarray(asnumpy(data_index)).astype(np.int64).reshape(-1)
        n = int(coords.shape[0])
        out = np.full(n, -1e300, dtype=float)
        d_h = np.full(n, np.nan)
        h_h = np.full(n, np.nan)
        self._last_d_h = d_h
        self._last_h_h = h_h
        valid = np.all(np.isfinite(coords), axis=1)
        if not np.any(valid):
            return out
        leaf = self._current_leaf
        if self._stats["leaf"] is None:
            self._stats["leaf"] = leaf
        valid_pos = np.where(valid)[0]
        for device, pos in self._split_rows(idx[valid_pos]):
            rows = valid_pos[pos]
            with self._device_context(device):
                adapter = self._adapter()
                for lo in range(0, rows.size, self.batch_max_size):
                    sel = rows[lo: lo + self.batch_max_size]
                    ll_c, dh_c, hh_c = self._score_chunk(adapter, coords[sel], idx[sel], leaf)
                    out[sel] = ll_c
                    d_h[sel] = dh_c
                    h_h[sel] = hh_c
                    self._stats["chunks"] += 1
        self._stats["rows"] += n
        self._stats["seconds"] += time.perf_counter() - t_start
        return out

    def _check_box(self, adapter):
        """The adapter crops to ITS run-domain copy; scoring labels with the containers'
        settings. They must be the same active box (``WDMSettings.__eq__`` ignores ``t0``)."""
        if self._box_checked:
            return
        dom = getattr(adapter, "domain_settings", None)
        if dom is not None and not (dom == self._wdm):
            raise ValueError(
                "EMRI direct generator crops to a different WDM active box than the "
                f"containers hold: generator f[{dom.ind_min_f}:{dom.ind_max_f}] "
                f"t[{dom.ind_min_t}:{dom.ind_max_t}] vs containers "
                f"f[{self._wdm.ind_min_f}:{self._wdm.ind_max_f}] "
                f"t[{self._wdm.ind_min_t}:{self._wdm.ind_max_t}]."
            )
        self._box_checked = True

    def _score_chunk(self, adapter, coords, idx, leaf):
        """Score one chunk: ``(ll, <r|h>, <h|h>)``, each ``(n,)``.

        One ``adapter.templates`` call; rows it flags as refused keep the
        ``-1e300`` floor (counted as ``refused``). Any exception from the
        direct build goes to :meth:`_fallback`. A box mismatch
        (:meth:`_check_box`) is a configuration error and is not masked.
        """
        n = int(coords.shape[0])
        self._check_box(adapter)                    # a configuration error: never masked
        t_gen = time.perf_counter()
        try:
            arr, ok = adapter.templates(coords, **self._gen_kwargs)
        except Exception as exc:  # noqa: BLE001 - scored exactly by the container path instead
            return self._fallback(coords, idx, leaf, exc, t_gen)
        t_score = time.perf_counter()
        self._stats["gen_s"] += t_score - t_gen
        ll = np.full(n, -1e300, dtype=float)
        dh = np.full(n, np.nan)
        hh = np.full(n, np.nan)
        ok = np.asarray(ok, dtype=bool)
        self._stats["refused"] += int((~ok).sum())
        if np.any(ok):
            pos = np.where(ok)[0]
            xp = get_array_module(arr)
            sub = arr if pos.size == n else arr[xp.asarray(pos)]
            ll[pos], dh[pos], hh[pos] = self._score_templates(sub, idx[pos])
        self._stats["score_s"] += time.perf_counter() - t_score
        return ll, dh, hh

    def _fallback(self, coords, idx, leaf, exc, t_gen):
        """Score a chunk the direct path could not build through the production container path.

        Uses ``waveform_gen`` explicitly (the installed generator may be the
        direct template that just failed); warns once per leaf and counts
        ``n_batch_fallbacks``. Returns ``(ll, nan, nan)`` (no ``<d|h>`` record).
        """
        self._stats["gen_s"] += time.perf_counter() - t_gen
        self.n_batch_fallbacks += 1
        self._stats["fallbacks"] += 1
        self.last_batch_error = exc
        if self._warned_fallback_leaf != leaf:
            self._warned_fallback_leaf = leaf
            logger.warning(
                "[EMRI_DIRECT] leaf %s: direct batch failed (%s: %s); scoring %d rows through "
                "the per-row production container path. Watch n_batch_fallbacks.",
                leaf, type(exc).__name__, exc, int(coords.shape[0]),
            )
        t_score = time.perf_counter()
        # the PRODUCTION generator: the installed one may be the direct template that just failed
        ll = np.real(np.asarray(
            self.compute_acs_like(coords, idx, signal_gen=self.waveform_gen, **self.waveform_like_kwargs),
            dtype=float,
        )).reshape(-1)
        self._stats["score_s"] += time.perf_counter() - t_score
        return ll, np.full(ll.shape, np.nan), np.full(ll.shape, np.nan)

    def _score_templates(self, arr, idx):
        """``(ll, <r|h>, <h|h>)`` for templates ``arr`` (n, nch, Nf_active, Nt_active)."""
        like_kw = {
            k: v for k, v in (self.waveform_like_kwargs or {}).items()
            if k not in ("psd", "complex", "include_psd_info")
        }
        box = self._wdm
        acs_flat = self.acs.acs.flatten()
        xp = get_array_module(arr)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        n = int(idx.size)
        order, dh_dev, hh_dev = [], [], []
        for g in np.unique(idx):
            rows_g = np.where(idx == g)[0]
            h_g = WDMSignal(arr[xp.asarray(rows_g)], box)
            r_box, _, s_box = acs_flat[int(g)]._slice_to_template(h_g)
            dh_dev.append(xp.real(inner_product(r_box, h_g, psd=s_box, **like_kw)).reshape(-1))
            hh_dev.append(xp.real(inner_product(h_g, h_g, psd=s_box, **like_kw)).reshape(-1))
            order.append(rows_g)
        both = asnumpy(xp.stack([xp.concatenate(dh_dev), xp.concatenate(hh_dev)]))
        pos = np.concatenate(order)
        dh = np.empty(n)
        hh = np.empty(n)
        dh[pos] = np.asarray(both[0], dtype=float)
        hh[pos] = np.asarray(both[1], dtype=float)
        ll = self._exposed_offset[idx] + dh - 0.5 * hh
        return ll, dh, hh

    # ------------------------------------------------------------------
    # records, checks, telemetry
    # ------------------------------------------------------------------

    def _record_leaf_inner_products(self, new_state, add_coords_in, leaf):
        """Record cold-chain ``<d|h>``, ``<h|h>`` from the direct scorer (one batch)."""
        if not getattr(self, "record_inner_products", False):
            return
        _sub = (getattr(new_state, "sub_states", None) or {}).get(self.branch_name)
        if _sub is None or getattr(_sub, "d_h", None) is None:
            return
        walker_idx = np.arange(self.nwalkers, dtype=np.int32)
        self.compute_like(add_coords_in, walker_idx)
        _sub.d_h[:, leaf] = self._last_d_h[: self.nwalkers]
        _sub.h_h[:, leaf] = self._last_h_h[: self.nwalkers]

    def _point_tolerance(self, shape):
        """Per-point tolerance of the direct-vs-production checks, shaped ``shape``.

        The two templates differ by ``delta = h_prod - h_direct`` with ``<delta|delta>
        ~ 2 mm <h|h>`` (mismatch ``mm``), so their lnL differ by ``-<r|delta> - 1/2
        <delta|delta>``: a bias ~ ``mm <h|h>`` plus a noise term of standard deviation
        ~ ``sqrt(2 mm <h|h>)``. A fixed tolerance would fire on most visits at high SNR
        (SNR 70, mm 1e-4: ~0.5 nat bias, ~1 nat scatter), so each point gets
        ``EMRI_CHECK_LL_TOL + mm <h|h> + 3 sqrt(2 mm <h|h>)`` with ``mm =
        EMRI_CHECK_LL_MM`` (3e-4: the measured direct-vs-production mismatch, plunges
        included) and ``<h|h>`` the direct scorer's own value for that point (this
        visit's ``prev_logl`` call). An expose-sign or frame bug moves lnL by ~SNR^2,
        far above it. ``EMRI_CHECK_LL_MM=0`` restores the plain tolerance."""
        tol = np.full(shape, self.check_ll_tol, dtype=float)
        hh = getattr(self, "_last_h_h", None)
        if self.check_ll_mm <= 0 or hh is None or np.size(hh) != int(np.prod(shape)):
            return tol
        hh = np.nan_to_num(np.clip(np.asarray(hh, dtype=float).reshape(shape), 0.0, None), nan=0.0)
        return tol + self.check_ll_mm * hh + 3.0 * np.sqrt(2.0 * self.check_ll_mm * hh)

    def _verify_prev_logl(self, prev_logl, old_coords_in, data_index_in, leaf):
        """Direct ``prev_logl`` vs the production container path, COLD rung only.

        The two templates differ (~1e-4 mismatch), so the gate is a per-point tolerance
        (:meth:`_point_tolerance`) on the cold rung; hot rungs sit far from the
        posterior, where any template difference is amplified through ``<r|delta h>``,
        and are reported in the same line without gating."""
        tol = np.atleast_2d(self._point_tolerance(np.shape(prev_logl)))
        acs_like = (
            self.compute_check_like(old_coords_in, data_index_in)
            .reshape(prev_logl.shape)
            .real
        )
        prev2 = np.atleast_2d(prev_logl)
        acs2 = np.atleast_2d(acs_like)
        both = (
            np.isfinite(prev2) & np.isfinite(acs2)
            & (prev2 > -1e299) & (acs2 > -1e299)
        )
        cold = both[0]
        if not np.any(cold):
            return
        diff = prev2[0][cold] - acs2[0][cold]
        max_abs = float(np.abs(diff).max())
        excess = np.abs(diff) - tol[0][cold]
        if float(excess.max()) <= 0.0:
            return
        worst = int(np.argmax(excess))
        hot = both[1:]
        hot_txt = (
            f"{float(np.abs(prev2[1:][hot] - acs2[1:][hot]).max()):.6e} over "
            f"{int(hot.sum())} points" if np.any(hot) else "n/a"
        )
        msg = (
            f"{self.branch_name} leaf {leaf}: direct-to-WDM fast path vs production container "
            f"path disagree beyond the template tolerance on the COLD rung: "
            f"max|diff|={max_abs:.6e} (worst point |diff| {abs(float(diff[worst])):.6e} vs "
            f"tolerance {float(tol[0][cold][worst]):.6e}; {self._dbg_prefix}_CHECK_LL_TOL="
            f"{self.check_ll_tol}, {self._dbg_prefix}_CHECK_LL_MM={self.check_ll_mm}), "
            f"spread={float(diff.max() - diff.min()):.6e} over "
            f"{int(cold.sum())} points (hot rungs, not gating: max|diff|={hot_txt}). A "
            "difference ~ mismatch x SNR^2 is the template accuracy (lookup table, plunge "
            "chunk, mode content); a large one means a time, frame or residual bug."
        )
        if self.check_ll_mode == "strict":
            raise ValueError(msg)
        logger.warning(msg)

    def _verify_entry_vs_acs(self, prev_logl, cold_ref, leaf):
        """The expose invariant with the template tolerance (see the module docstring)."""
        if cold_ref is None:
            return
        tol = np.atleast_2d(self._point_tolerance(np.shape(prev_logl)))[0]
        cold = np.asarray(prev_logl[0], dtype=float)
        ref = np.asarray(cold_ref, dtype=float).reshape(cold.shape)
        both = np.isfinite(cold) & np.isfinite(ref) & (cold > -1e299) & (ref > -1e299)
        if not np.any(both):
            return
        diff = cold[both] - ref[both]
        max_abs = float(np.abs(diff).max())
        excess = np.abs(diff) - tol[both]
        if float(excess.max()) <= 0.0:
            return
        worst = int(np.argmax(excess))
        msg = (
            f"{self.branch_name} leaf {leaf}: EXPOSE INVARIANT VIOLATED -- cold direct "
            f"prev_logl vs pre-expose ACS lnL (the installed generator's template in the residual): "
            f"max|diff| {max_abs:.6e}, median {float(np.median(diff)):.6e} over "
            f"{int(both.sum())} walkers; worst |diff| {abs(float(diff[worst])):.6e} vs its "
            f"tolerance {float(tol[both][worst]):.6e} ({self._dbg_prefix}_CHECK_LL_TOL="
            f"{self.check_ll_tol}, {self._dbg_prefix}_CHECK_LL_MM={self.check_ll_mm}). An "
            "expose-sign bug moves this by ~SNR^2."
        )
        if self.check_ll_mode == "strict":
            raise ValueError(msg)
        logger.warning(msg)

    def _flush_stats(self):
        """Log the finished leaf's ``[EMRI_DIRECT]`` line (if it scored rows) and reset."""
        st = self._stats
        if st["rows"]:
            logger.info(
                "[EMRI_DIRECT] leaf %s: %d rows in %d chunks, %.1f s (%.3f s/row; "
                "generate=%.2f s, score=%.2f s), %d fallbacks, %d rows refused by FEW",
                st["leaf"], st["rows"], st["chunks"], st["seconds"],
                st["seconds"] / st["rows"], st["gen_s"], st["score_s"],
                st["fallbacks"], st["refused"],
            )
        self._stats = self._new_stats()
