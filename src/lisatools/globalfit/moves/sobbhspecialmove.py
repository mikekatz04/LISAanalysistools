"""SOBBH add/remove move scored by a fast WDM likelihood comp (lookup or chunked-heterodyne).

:class:`SOBBHChunkedLikeMove` keeps ALL of :class:`ResidualAddOneRemoveOneMove`'s
choreography (per-leaf expose/fold, in-model repeats, per-leaf tempering,
cold-chain bookkeeping) and swaps the proposal-scoring path: instead of
one full-duration TD waveform + full TD->WDM transform per ``(temp, walker)``
row through the per-container Python loop, every batch is ONE vectorized
``get_ll_wdm`` call of its ``chunked_comp`` directly against the ACA's live WDM
residual buffers.

ONE class, two comps (``SOBBH_LIKELIHOOD``, ``SourceSOBBHSettings.likelihood``):

* ``lookup`` (stock default since 2026-10-03; the 9mo / 12mo runs):
  ``chunked_comp`` is the ``SOBBHLookupRouter`` over per-device
  ``SOBBHLookupComputations`` (batched TDI-on-the-fly response on a sparse grid +
  a WDM lookup table; ``docs/sobbh-wdm-lookup.md``);
* ``chunked``: a ``bbhx.sobbhcomps.SOBBHWDMComputations`` (BBHx ``sobbhcomps`` over
  LAT ``chunked_het``).

There is no lookup-vs-chunked branch in this module: both comps expose the same
``get_ll_wdm`` / ``fill_global_wdm`` / ``d_h_out`` / ``h_h_out`` surface, and
``source_runtime.build_sobbh_move_runtime`` picks the comp
(``SOBBH_LIKELIHOOD=full`` builds the base class instead). Stock name ``sobbh_pe``;
it runs in ``source_search``, every cycle of ``gb_search_1`` / ``gb_search_2`` and in
``full_pe``. The text below says "chunked" for the comp's fast path whichever
comp is installed.

Convention bridge (load-bearing): the container path scores
``-1/2 (d_d + h_h - 2 d_h)`` [+ the per-walker noise term when the run fits a
psd branch], where ``d_d = <r|r>`` of the EXPOSED residual. The chunked call
returns only the source piece ``d_h - 1/2 h_h`` (comp built with ``d_d=0``),
so :meth:`setup_likelihood_here` captures the per-walker offset
``acs.likelihood()`` (= ``-1/2 d_d`` + noise term) on the freshly exposed
residual once per leaf and :meth:`compute_like_local` adds it back. This reproduces
the slow path's numbers on every scoring site (prev_logl, proposal batches,
fancy tempering swap) and keeps the base ``_verify_entry_vs_acs`` expose
invariant meaningful, up to chunked-heterodyne truncation error.

Residual expose/fold ALSO runs chunked since 2026-08-28
(:meth:`SOBBHChunkedLikeMove._apply_cold_chain_sources`): one batched
``fill_global_wdm`` per leaf visit in place of the base's dense
one-row-at-a-time pass, which cost 24 full TD waveform + TD->WDM builds to
expose a leaf and 24 more to fold it back (32 s per build on the 6-mo grid
-- the first leaf never finished inside a preemption window). This DOES put
chunked truncation into the residual, where the move previously kept the
residual bit-identical to the stock path and approximated only scoring; the
trade was measured first (6-mo removal-null: residual 6e-4 of ``<h|h>`` at
the ``Nt_sub=32`` defaults). ``SOBBH_CHUNKED_FILL=0`` restores the base's
per-row pass through the container's installed generator. On the stock path
that is the branch's
:class:`~lisatools.globalfit.stock.erebor.source_runtime.SourceSignalGen`,
i.e. the same fast comp's fill one row at a time, not the exact generator
(``waveform_gen`` serves only a container that lacks the branch). The
built-in fast-vs-slow cross-check (``_verify_prev_logl``)
recomputes through the slow container path at MATCHING convention with a
tolerance knob ``SOBBH_CHECK_LL_TOL``.
"""

import logging
import os
import time

import numpy as np

from ...utils.utility import asnumpy
from .addremovemove import ResidualAddOneRemoveOneMove

logger = logging.getLogger(__name__)

__all__ = ["SOBBHChunkedLikeMove"]


class SOBBHChunkedLikeMove(ResidualAddOneRemoveOneMove):
    """Add/remove move for the SOBBH branch scored through a fast WDM comp.

    The ``propose(model, state)`` contract is the base class's
    (:meth:`ResidualAddOneRemoveOneMove.propose_local` on every compute rank's walker
    block, through the :class:`~lisatools.globalfit.moves.walkerfanout.WalkerFanoutMixin`
    ``propose``): per leaf, expose the cold-chain source (:meth:`_apply_cold_chain_sources`),
    :meth:`setup_likelihood_here`, score ``prev_logl`` and the in-model repeats through
    :meth:`compute_like_local`, swap the per-leaf ladder, record ``<d|h>`` / ``<h|h>``,
    fold back. The tempered ensemble is the ``sobbh`` sub-state; the base writes it and
    mirrors row 0 into the main state (``_sync_cold_row``).

    Args:
        *args: Positional arguments of :class:`ResidualAddOneRemoveOneMove`
            (``branch_name, coords_shape, waveform_gen, ...``). The
            ``waveform_gen`` stays the SLOW exact generator: the
            cross-check/debug paths and the fallback for a container that
            lacks the branch. The ``SOBBH_CHUNKED_FILL=0`` per-row expose/fold
            goes through the container's installed generator instead (on the
            stock path the fast comp's fill).
        chunked_comp: The fast comp. With ``SOBBH_LIKELIHOOD=lookup`` (stock
            default, production) the ``SOBBHLookupRouter``
            (``stock/erebor/source_runtime.get_sobbh_lookup_router``); with
            ``chunked`` a built ``bbhx.sobbhcomps.SOBBHWDMComputations``
            (``get_sobbh_chunked_comp``). Either way on the SAME
            ``WDMSettings`` / orbits / TDI config as the run's data, and its
            ``d_d`` must be 0 (the move adds ``-1/2 <r|r>`` itself). Required.
        m_band_half_width (int): Narrow-band half-width (WDM layers) around each
            chunk's carrier — the chunked scoring path's one live accuracy/speed
            knob (``SOBBH_M_BAND_HALF_WIDTH``; ctor and settings default 1,
            launchers 3). Inert under ``lookup`` (production): the lookup comp
            ignores it and uses its ``lookup_num_m_layers`` band.
        **kwargs: Keyword arguments of :class:`ResidualAddOneRemoveOneMove`.
            ``dcga`` must be ``None`` — the chunked kernel scores against
            the ACA buffers directly (multi-GPU walker shards are handled
            by per-split routing inside :meth:`compute_like_local`, not by the
            DCGA replica machinery).

    Raises:
        ValueError: ``dcga`` given, ``chunked_comp`` missing, or a comp whose
            ``d_d`` is not 0.

    Attributes:
        comp: The ``chunked_comp``.
        m_band_half_width (int): See ``m_band_half_width``.
        check_ll_tol (float): ``SOBBH_CHECK_LL_TOL`` (default ``0.5`` nat).

    Note:
        Env knobs read by this class or by the base with the ``SOBBH`` prefix:
        ``SOBBH_CHUNKED_FILL`` (default ``1``; anything else uses the dense
        per-row expose / fold), ``SOBBH_CHECK_LL`` (base default
        ``ADDREMOVE_CHECK_LL`` then ``1``; launchers ``0``), ``SOBBH_CHECK_LL_EVERY``
        (base default every visit), ``SOBBH_CHECK_LL_TOL``, ``SOBBH_RECORD_DH``
        (class default ``1``) and, in the base's repeat loop,
        ``SOBBH_SINGLE_CALL`` (default ``1``: one full-ensemble scoring call per
        repeat).

    Example:
        Built by the stock layer (``source_runtime.build_sobbh_move_runtime``):

        .. code-block:: python

            from lisatools.globalfit.recipe import SOBBHChunkedMoveBuilder

            comp = get_sobbh_fast_comp(curr.general_info, cfg)   # lookup router
            _, moves = SOBBHChunkedMoveBuilder(
                wave_gen=wave_gen,                 # exact SOBBH wrap
                chunked_comp=comp,
                m_band_half_width=3,
            ).build(None, curr, acs, priors, state)
            sobbh_pe = moves[0]
    """

    #: Column permutation from the stock SOBBH waveform basis
    #: ``(m1, m2, s1, s2, dist[Gpc], inc, f_low, lam, beta, psi, phi0)``
    #: (the already-transformed coords every scoring call receives; sky in
    #: the run/orbits frame — ICRS for stock runs) to the chunked-comp order
    #: ``(m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)``.
    #: ``phi0`` (catalogue TrueAnomaly / reference orbital phase) maps onto
    #: ``phi_c`` — the equivalence is pinned by tests/test_sobbh_chunked_move.
    _CHUNKED_PERM = (0, 1, 2, 3, 4, 6, 10, 5, 9, 7, 8)

    def __init__(self, *args, chunked_comp=None, m_band_half_width=1, **kwargs):
        if kwargs.get("dcga") is not None:
            raise ValueError(
                "SOBBHChunkedLikeMove has no DCGA (replica) path: the "
                "chunked kernel reads the ACA buffers directly, and "
                "multi-GPU walker shards are served by per-split routing "
                "inside compute_like_local. Build it without dcga= (the "
                "SOBBHChunkedMoveBuilder skips the DCGA branch)."
            )
        if chunked_comp is None:
            raise ValueError(
                "SOBBHChunkedLikeMove requires chunked_comp= (a built "
                "bbhx.sobbhcomps.SOBBHWDMComputations with d_d=0)."
            )
        super().__init__(*args, **kwargs)

        self.comp = chunked_comp
        self.m_band_half_width = int(m_band_half_width)

        if float(getattr(self.comp, "d_d", 0.0)) != 0.0:
            raise ValueError(
                "chunked_comp must be built with d_d=0.0 — the move folds the "
                "exposed-residual <r|r> in via its per-walker offset instead."
            )

        # the *_wdm kernels are single-shard by contract (they consume
        # linear_data_arr[0]); multi-shard (multi-GPU walker-shard) ACAs
        # are served by per-split routing in compute_like_local and in the
        # cold-chain fill (gbbands _ShardHolderView + partition, each split
        # under its own device context).
        #
        # DEVICE DISCIPLINE (2026-09-16). Unlike the GB router, which builds
        # a per-device comp REPLICA (_RoutedBandEngine._comp_for ->
        # _device_local_gb_comp, guarded by _assert_comp_device), this move
        # drives ONE SHARED comp under both devices' contexts. Everything it
        # hands a kernel out of ``self`` is therefore a home-device pointer
        # unless something relocates it:
        #   * the five chunk-geometry / WDM-window arrays now go through
        #     WDMComputationsBase._geometry_kernel_args() on EVERY kernel
        #     path (scoring, fill, swap, grads, fstat) — that is the fix for
        #     the null run's illegal access in wdm_het_fill_global_kernel,
        #     which was fed device-0 geometry from a device-1 launch.
        #   * STILL HOME-DEVICE, not yet addressed: self.comp.cpp_orbits /
        #     cpp_tdi_config / cpp_wdm_settings. The C++ impl memcpy's those
        #     host structs to the caller's device per call, but their
        #     POINTER FIELDS (e.g. the orbit spline arrays) still address the
        #     comp's home device. If a cross-device fault survives this fix,
        #     that is the next suspect — the durable answer is a per-device
        #     comp replica here, as GB already does.
        self._n_shards = len(self.acs.linear_data_arr)

        # in-band carrier window from the comp's WDM settings: proposals
        # whose f_low falls outside score as invalid (-1e300) to mirror the
        # slow path's domain_error sentinel (the kernel itself would return
        # d_h = h_h = 0, silently "template = 0")
        ws = self.comp.wdm_settings
        self._f_band_lo = float(ws.ind_min_f) * float(ws.layer_df)
        self._f_band_hi = (float(ws.ind_max_f) + 1.0) * float(ws.layer_df)

        # per-walker exposed-residual offset; armed per leaf in
        # setup_likelihood_here
        self._exposed_offset = None

        # fast-vs-slow tolerance for the overridden _verify_prev_logl
        # (chunked-heterodyne truncation error scales with in-band SNR^2;
        # tighten after the P1.4 A/B numbers are recorded)
        self.check_ll_tol = float(
            os.environ.get(f"{self._dbg_prefix}_CHECK_LL_TOL", "0.5")
        )

    # ------------------------------------------------------------------
    # basis shim
    # ------------------------------------------------------------------

    @staticmethod
    def to_chunked_basis(coords_in: np.ndarray) -> np.ndarray:
        """Waveform-basis rows -> chunked-comp rows (explicit, tested shim).

        Input columns (stock SOBBH waveform basis, what the move's scoring
        sites hand ``compute_like_local`` after the branch transform):
        ``(m1, m2, s1, s2, dist[Gpc], inc, f_low, lam, beta, psi, phi0)``.
        Output columns (``SOBBHTDIonTheFly``/chunked order):
        ``(m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)``.
        Sky angles pass through unchanged (both sides in the orbits frame).

        Args:
            coords_in: Waveform-basis rows, shape ``(n, 11)`` (a single row is
                promoted to ``(1, 11)``).

        Returns:
            np.ndarray: ``(n, 11)`` float64 chunked-basis rows (a copy; distance
            Gpc -> pc).
        """
        coords_in = np.atleast_2d(np.asarray(coords_in, dtype=np.float64))
        out = coords_in[:, SOBBHChunkedLikeMove._CHUNKED_PERM].copy()
        out[:, 4] *= 1e9  # Gpc -> parsec
        return out

    # ------------------------------------------------------------------
    # likelihood seams
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # per-leaf scoring telemetry (2026-09-16). Production measured
    # 190.5 s/leaf = 25 x ~7.35 s compute_like_local calls = 61 ms/row against
    # the in-code job-373 reference of 2.78 ms/row (same Tobs / band
    # half-width / shard count) with LOG-SILENT leaf windows -- a ~22x
    # per-row regression nothing in the log could attribute. One
    # [SOBBH_LL_TIMING] line per leaf window (flushed at the next
    # setup_likelihood_here) splits the wall into host-stage vs kernel
    # and counts calls/rows, so the call-count (extra scorings per
    # repeat?) and the per-row rate are separately visible.
    #
    # RESOLVED 2026-09-18, from 174 leaf windows of the 4-GPU production run
    # (gf_prod_6mo_v8_4gpu, 1 walker/rank x 8 rungs). THERE IS NO PER-ROW
    # REGRESSION: the call wall is a FLAT 1.71-1.73 s and does not move with
    # either batch dimension --
    #
    #     rows/call   6.7 .. 24.2   ->  s/call 1.708 .. 1.831
    #     groups/call 1.0 .. 2.8    ->  s/call 1.708 .. 1.725
    #
    # so the "2.78 ms/row" reference was simply a 288-row call paying the
    # same flat wall, and job 508's 54-row calls "costing 4x" was the same
    # constant divided by fewer rows. The CAUSE is the kernel launch
    # geometry, not the group count: ``wdm_het_get_ll_impl`` sets
    # ``gd_x = num_bin`` -- ONE BLOCK PER ROW, no grid-stride over chunks --
    # and explicitly ``(void)``s binary_perm / group_starts / group_ends /
    # group_m_lo / group_m_hi / n_groups (the narrow band the kernel
    # actually honours comes from ``m_band_half_width`` per binary per
    # chunk). One block's serial sweep over all n_chunks therefore SETS the
    # wall, and every additional row occupies an idle SM for free until the
    # device runs out of blocks -- on an H100 that is ~132, against the 8
    # rows a 1-walker block supplies.
    #
    # Consequences, in order of leverage:
    #   1. rows/call is the lever and it is a LAYOUT knob. One walker per
    #      rank is the worst case for SOBBH: it uses ~6% of the device and
    #      pays full price. More walkers per rank cost SOBBH nothing.
    #   2. calls/leaf = num_repeats + 2, so {BRANCH}_NUM_PROP_REPEATS scales
    #      the branch's cost linearly (20 -> 10 on 2026-09-18).
    #   3. Saturating the device for a small batch needs a (row x chunk)
    #      grid with a cross-block reduction of the partial <d|h>/<h|h>,
    #      i.e. a kernel change in lat_chunked_het_kernels.hh that the GB
    #      branch shares. Not attempted here.
    # The ``n_groups`` figure is kept in the line because the other kernel
    # paths (fill_global / swap_ll / fstat) DO consume the groups -- but for
    # get_ll it is a diagnostic only, never a cost driver.
    # ------------------------------------------------------------------

    #: ``_kernel_ll`` sub-spans accumulated per leaf window, in the order
    #: they are paid. ``dispatch`` = shard-view/partition routing (move
    #: side); ``stage`` = the comp's param staging + index prep + layer
    #: grouping; ``geom`` = its static-geometry re-assert; ``wrap`` = the
    #: per-call ``*ComputationGroupWrap`` construction; ``launch`` = the
    #: kernel invocation; ``sync`` = the device sync that drains an async
    #: launch (so ``launch`` + ``sync`` is the true kernel wall); ``pull``
    #: = the D2H of ll / d_h_out / h_h_out.
    _LL_SPAN_KEYS = ("dispatch", "stage", "geom", "wrap", "launch",
                     "sync", "pull")

    def _ll_stats_reset(self):
        """Start a fresh ``[SOBBH_LL_TIMING]`` accumulation window."""
        self._ll_stats = {"calls": 0, "rows": 0, "host_s": 0.0,
                          "kernel_s": 0.0, "total_s": 0.0,
                          "shard_calls": 0, "groups": 0}
        for key in self._LL_SPAN_KEYS:
            self._ll_stats[key + "_s"] = 0.0

    def _ll_spans_add(self, spans):
        """Fold one ``_kernel_ll`` shard call's sub-spans into the window."""
        st = getattr(self, "_ll_stats", None)
        if st is None:
            self._ll_stats_reset()
            st = self._ll_stats
        st["shard_calls"] += int(spans.get("shard_calls", 0))
        st["groups"] += int(spans.get("n_groups", 0))
        for key in self._LL_SPAN_KEYS:
            st[key + "_s"] = st.get(key + "_s", 0.0) + float(spans.get(key, 0.0))

    def _flush_ll_stats(self):
        """Log the finished leaf window's two ``[SOBBH_LL_TIMING]`` lines, then reset."""
        st = getattr(self, "_ll_stats", None)
        if not st or st["calls"] == 0:
            return
        prefix = getattr(self, "_dbg_prefix", "SOBBH")
        other = st["total_s"] - st["host_s"] - st["kernel_s"]
        logger.info(
            "[%s_LL_TIMING] leaf window: calls=%d rows=%d total=%.2f s "
            "(host_stage=%.2f, kernel=%.2f, other=%.2f) -> %.0f ms/call, "
            "%.2f ms/row",
            prefix, st["calls"], st["rows"],
            st["total_s"], st["host_s"], st["kernel_s"], other,
            1e3 * st["total_s"] / st["calls"],
            1e3 * st["total_s"] / max(st["rows"], 1),
        )
        # companion line: WHICH part of the kernel call the wall went to.
        spans = {k: float(st.get(k + "_s", 0.0)) for k in self._LL_SPAN_KEYS}
        named = sum(spans.values())
        slowest = max(spans, key=spans.get)
        shard_calls = max(st["shard_calls"], 1)
        logger.info(
            "[%s_LL_TIMING] leaf window internals: shard_calls=%d "
            "groups/call=%.1f dispatch=%.2f stage=%.2f geom=%.2f wrap=%.2f "
            "launch=%.2f sync=%.2f pull=%.2f s (named=%.2f of kernel=%.2f) "
            "-> slowest=%s (%.0f%% of named), %.0f ms/call at %.1f rows/call "
            "(kernel grid = ONE BLOCK PER ROW: the wall is one row's serial "
            "chunk sweep, so rows up to the device's block capacity are "
            "~free -- raise rows/call, not groups)",
            prefix, st["shard_calls"], st["groups"] / shard_calls,
            spans["dispatch"], spans["stage"], spans["geom"], spans["wrap"],
            spans["launch"], spans["sync"], spans["pull"],
            named, st["kernel_s"],
            slowest, 100.0 * spans[slowest] / named if named > 0 else 0.0,
            1e3 * (spans["launch"] + spans["sync"]) / shard_calls,
            st["rows"] / shard_calls,
        )
        self._ll_stats_reset()

    def _ll_stats_add(self, rows, t0, t_host, t_kernel):
        """Count one :meth:`compute_like_local` call (``rows`` rows, host / kernel seconds)."""
        st = getattr(self, "_ll_stats", None)
        if st is None:
            self._ll_stats_reset()
            st = self._ll_stats
        st["calls"] += 1
        st["rows"] += int(rows)
        st["host_s"] += t_host
        st["kernel_s"] += t_kernel
        st["total_s"] += time.perf_counter() - t0

    def setup_likelihood_here(self, coords):
        """Capture the per-walker exposed-residual offset for this leaf.

        Called by the base ``propose`` once per leaf, right after the leaf's
        cold-chain sources are exposed into the residual. ``acs.likelihood()``
        here is exactly the ``-1/2 <r|r>`` (+ noise-normalization term when
        configured) that the container scoring path folds into every value —
        the piece the chunked call (built with ``d_d = 0``) leaves out.
        """
        # previous leaf's scoring window ends here -- flush its telemetry
        self._flush_ll_stats()
        self._exposed_offset = np.asarray(asnumpy(self.acs.likelihood()), dtype=float)
        super().setup_likelihood_here(coords)

    def compute_like_local(self, coords_in, data_index):
        """Rank-local chunked-heterodyne scorer for one batch.

        The base ``compute_like`` (:class:`ResidualAddOneRemoveOneMove`)
        scatters rows to this method — the row-fanout scatter seam, not
        an override point.

        Args:
            coords_in (xp.ndarray): ``(N, 11)`` already-transformed waveform-basis
                rows (all walkers x temps under the current leaf).
            data_index (xp.ndarray): ``(N,)`` physical-walker index per row.

        Returns:
            np.ndarray: ``(N,)`` log-likelihoods in the container convention
            (``-1e300`` for invalid rows: non-finite, or ``f_low`` outside the
            comp's band). Side outputs ``self._last_d_h`` / ``self._last_h_h``
            (NaN where not scored).

        Raises:
            RuntimeError: Called before :meth:`setup_likelihood_here` armed the
                exposed-residual offset.
        """
        if self._dcga is not None:  # unreachable (ctor guard); keep loud
            raise NotImplementedError("SOBBHChunkedLikeMove has no DCGA path.")
        if self._exposed_offset is None:
            raise RuntimeError(
                "compute_like_local called before setup_likelihood_here armed the "
                "exposed-residual offset (propose() choreography violated)."
            )

        _t0 = time.perf_counter()
        coords_np = np.atleast_2d(np.asarray(asnumpy(coords_in), dtype=np.float64))
        idx = np.asarray(asnumpy(data_index)).astype(np.int32).reshape(-1)
        _t_host = time.perf_counter() - _t0
        n_rows = coords_np.shape[0]

        params = self.to_chunked_basis(coords_np)
        f_low = params[:, 5]
        valid = (
            np.all(np.isfinite(params), axis=1)
            & (f_low >= self._f_band_lo)
            & (f_low < self._f_band_hi)
        )

        out = np.full(n_rows, -1e300, dtype=float)
        self._last_d_h = np.full(n_rows, np.nan)
        self._last_h_h = np.full(n_rows, np.nan)
        if not np.any(valid):
            self._ll_stats_add(n_rows, _t0, _t_host, 0.0)
            return out

        _t_k = time.perf_counter()
        ll, d_h, h_h = self._kernel_ll(params[valid], idx[valid])
        _t_kernel = time.perf_counter() - _t_k
        out[valid] = ll + self._exposed_offset[idx[valid]]
        self._last_d_h[valid] = d_h
        self._last_h_h[valid] = h_h
        self._ll_stats_add(n_rows, _t0, _t_host, _t_kernel)
        return out

    def _kernel_ll(self, params, idx):
        """One chunked-het scoring pass, shard-routed when the ACA is split.

        Args:
            params (np.ndarray): ``(N, 11)`` chunked-basis host rows (all valid).
            idx (np.ndarray): ``(N,)`` GLOBAL walker indices.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: ``(ll, d_h, h_h)`` host
            arrays in row order; ``ll`` is the source piece ``d_h - 1/2 h_h``
            (the caller adds the exposed-residual offset).
        """
        # TODO(sobbh scoring speed) -- REAL, but for much later. Measured on
        # the 6-mo probe (job 373, 12 temps x 24 walkers x 10 repeats, 2
        # shards, m_band_half_width=3): 8.0 s per leaf = 2880 rows =
        # 2.78 ms/row, i.e. 800 ms per 288-row batched call. That should be
        # beatable.
        # WHY IT IS NOT WORTH DOING NOW: after the chunked fill landed, the
        # whole SOBBH proposal is 48 s of a ~57 min iteration -- 1.4%. MBH
        # (~1430 ms/row) and EMRI (~1040 ms/row) are the other ~98%, both
        # still scoring ONE ROW AT A TIME. Any effort spent here buys at most
        # 1.4% while the same effort on MBH/EMRI batching buys orders of
        # magnitude. Revisit only once those are batched and SOBBH is
        # actually visible in the budget.
        # LEVERS, cheapest first, when that day comes:
        #   1. SOBBH_M_BAND_HALF_WIDTH 3 -> 2. The scoring band is 2m+1
        #      layers, so 7 -> 5 is ~30% off the row cost for a measured
        #      0.1% of <h|h> (m=2 recovered 99.9%, m=3 is converged). Pure
        #      config, no code.
        #   2. Profile the per-CALL overhead: 47% GPU util during the SOBBH
        #      phase with 800 ms calls smells like host-side staging /
        #      launch overhead rather than kernel time. Measure before
        #      optimizing -- the shard routing stages host arrays per call.
        #   3. ``get_swap_ll_wdm`` (already on WDMComputationsBase) scores an
        #      add and a remove in ONE call; the in-model repeat loop
        #      currently pays two separate scoring paths.
        # NB do NOT "batch the repeats": the repeat loop is a sequential MH
        # chain, and batching repeats across sources was VETOED for GB
        # (serial-within-band scheduling policy).
        from ...utils.device import synchronize

        if len(self.acs.linear_data_arr) == 1:
            xp = getattr(self.acs, "xp", None)
            # on the device that owns the (single) shard: the comp resolves
            # on the CURRENT device, so this is not left to whatever device
            # happens to be current (see _shard_context)
            with self._shard_context():
                _t = time.perf_counter()
                ll = self.comp.get_ll_wdm(
                    params, self.acs,
                    data_index=idx, noise_index=idx,
                    m_band_half_width=self.m_band_half_width,
                )
                _t_call = time.perf_counter() - _t
                _t = time.perf_counter()
                synchronize(xp)
                _t_sync = time.perf_counter() - _t
                _t = time.perf_counter()
                out = (
                    np.asarray(asnumpy(ll), dtype=float),
                    np.real(np.asarray(asnumpy(self.comp.d_h_out))),
                    np.real(np.asarray(asnumpy(self.comp.h_h_out))),
                )
            self._ll_spans_add(
                self._shard_spans(_t_call, _t_sync,
                                  time.perf_counter() - _t, 0.0)
            )
            return out

        # multi-GPU walker shards: reuse the GB shard-router primitives —
        # per-split single-shard views + the split partition — and run each
        # split's rows under the owning device context (cross-shard movement
        # is host-routed, matching the ACA conventions)
        from ...utils.device import device_context
        from .gbbands import _RoutedBandEngine

        _t = time.perf_counter()
        holder = self.acs
        views = _RoutedBandEngine._shard_views(holder)
        parts = _RoutedBandEngine._partition(holder, idx)
        xp = holder.xp
        n = params.shape[0]
        ll = np.full(n, -1e300, dtype=float)
        d_h = np.full(n, np.nan)
        h_h = np.full(n, np.nan)
        _t_dispatch = time.perf_counter() - _t
        for view, (pos, intra, _) in zip(views, parts):
            if pos.shape[0] == 0:
                continue
            with device_context(xp, view.device):
                _t = time.perf_counter()
                vals = self.comp.get_ll_wdm(
                    params[pos], view,
                    data_index=np.asarray(intra, dtype=np.int32),
                    noise_index=np.asarray(intra, dtype=np.int32),
                    m_band_half_width=self.m_band_half_width,
                )
                _t_call = time.perf_counter() - _t
                _t = time.perf_counter()
                synchronize(xp)
                _t_sync = time.perf_counter() - _t
                _t = time.perf_counter()
                ll[pos] = np.asarray(asnumpy(vals), dtype=float)
                d_h[pos] = np.real(np.asarray(asnumpy(self.comp.d_h_out)))
                h_h[pos] = np.real(np.asarray(asnumpy(self.comp.h_h_out)))
                self._ll_spans_add(
                    self._shard_spans(_t_call, _t_sync,
                                      time.perf_counter() - _t, _t_dispatch)
                )
                # dispatch is per-CALL, not per-shard -- charge it once
                _t_dispatch = 0.0
        return ll, d_h, h_h

    def _shard_spans(self, t_call, t_sync, t_pull, t_dispatch):
        """One shard call's sub-spans, comp-internal breakdown folded in.

        ``t_call`` is the whole ``get_ll_wdm`` wall; the comp records its
        own split of that on ``last_call_spans`` (stage / geom / wrap /
        launch / total). ``launch`` absorbs whatever ``t_call`` the comp
        left unattributed, so the named spans always sum to the measured
        wall -- including the case of a comp that publishes no breakdown at
        all, where the whole call is charged to ``launch``.
        """
        c = getattr(self.comp, "last_call_spans", None) or {}
        if not c:
            return {
                "dispatch": t_dispatch, "stage": 0.0, "geom": 0.0,
                "wrap": 0.0, "launch": t_call, "sync": t_sync,
                "pull": t_pull, "shard_calls": 1,
            }
        stage = float(c.get("stage", 0.0))
        geom = float(c.get("geom", 0.0))
        wrap = float(c.get("wrap", 0.0))
        launch = float(c.get("launch", 0.0))
        launch += max(t_call - float(c.get("total", t_call)), 0.0)
        return {
            "dispatch": t_dispatch, "stage": stage, "geom": geom,
            "wrap": wrap, "launch": launch, "sync": t_sync, "pull": t_pull,
            "shard_calls": 1, "n_groups": int(c.get("n_groups", 0)),
        }

    #: The chunked record is one cheap vectorized call -> default ON
    #: (SOBBH_RECORD_DH=0 disables).
    _record_dh_default = "1"

    def _record_leaf_inner_products(self, new_state, add_coords_in, leaf):
        """Record cold-chain ``<d|h>``, ``<h|h>`` from the chunked kernel.

        Same sub-state record as the base, at chunked-heterodyne accuracy
        (narrow ``m_band_half_width`` band) — one vectorized ``nwalkers``
        call instead of the base's slow container batch.
        """
        if not getattr(self, "record_inner_products", False):
            return
        _sub = (getattr(new_state, "sub_states", None) or {}).get(self.branch_name)
        if _sub is None or getattr(_sub, "d_h", None) is None:
            return
        walker_idx = np.arange(self.nwalkers, dtype=np.int32)
        self.compute_like(add_coords_in, walker_idx)
        _sub.d_h[:, leaf] = self._last_d_h[: self.nwalkers]
        _sub.h_h[:, leaf] = self._last_h_h[: self.nwalkers]

    # ------------------------------------------------------------------
    # Device ownership (2026-10-10, 9mo job 751). Every SOBBH pass enters
    # the device that owns the walker's shard, as the MBH/EMRI moves do
    # (MBHBatchedLikeMove._split_rows_static), instead of running on
    # whatever device happens to be current. The chunked / lookup comp
    # resolves on the CURRENT device, so a rank whose walker lived on GPU 1
    # built a second lookup comp on GPU 0 for its first Gram refresh and
    # read the GPU-1 residual / PSD across the link (cupy PerformanceWarning
    # at diagnostic.py:248/294 on two ranks -- the family behind the 10-08
    # illegal-memory-access race). The multi-shard branches of _kernel_ll
    # and _apply_cold_chain_sources already entered each view's device;
    # these hooks give the single-shard branches and the Gram the same rule.
    # Not enough on its own (9mo job 754): the lookup comp's PN step runs
    # through JAX, which left GPU 0 current INSIDE these contexts; the comp
    # now pins JAX to the cupy device (wdm_direct.SOBBHBatchedTOF.build).
    # ------------------------------------------------------------------

    def _split_rows(self, idx):
        """``[(device, positions)]`` grouping ``idx`` rows by owning walker shard."""
        from .mbhbatchedmove import MBHBatchedLikeMove

        return MBHBatchedLikeMove._split_rows_static(self.acs, idx)

    def _device_context(self, device):
        """cupy device context on ``device`` (a null context on numpy or ``None``)."""
        from ...utils.device import device_context

        return device_context(getattr(self.acs, "xp", None), device)

    def _shard_context(self, walker=0):
        """Context on the device that owns walker ``walker``'s shard
        (``gpus[0]`` on a single-shard ACA); a null context on numpy or
        without GPUs."""
        (device, _), = self._split_rows(np.array([int(walker)]))
        return self._device_context(device)

    def _gram_context(self, walker):
        """Enter the device that owns walker ``walker``'s shard."""
        return self._shard_context(walker)

    def _gram_templates(self, coords, walker):
        """``SOBBH_EIGEN_INFO=gram`` hook: the move's own templates for the
        Gram rows (see :meth:`ResidualAddOneRemoveOneMove._gram_info`).

        ``coords`` are waveform-basis rows; each is filled into its OWN
        zeroed slab (``data_index = row``) of a scratch buffer shaped like
        walker ``walker``'s data, through ``self.comp.fill_global_wdm`` --
        the same fill the move uses to expose / fold sources, so the Gram
        matrix differentiates the templates the residual actually carries.
        With ``SOBBH_LIKELIHOOD=lookup`` (the stock default) ``comp`` is the
        lookup router, so these are lookup fills; with ``chunked`` they are
        chunked-heterodyne fills.

        Memory: the buffer holds ``2 * ndim + 1`` (23) data-shaped
        active-band slabs at once, plus ``ndim`` half-differences in the
        caller -- ~0.6 GB peak on the 6-month grid (3 x ~177 x ~4200
        float64 per slab), twice that at 1 yr. Not yet run on a GPU
        (2026-10-06).

        Raises (the refresh then falls back to the likelihood route): a
        multi-shard ACA (the fill is single-shard by contract), or a row
        that is non-finite or whose f_low lies outside the comp's band.
        """
        if len(self.acs.linear_data_arr) != 1:
            raise NotImplementedError("SOBBH Gram templates: single-shard ACA only")
        params = self.to_chunked_basis(np.asarray(asnumpy(coords), dtype=np.float64))
        f_low = params[:, 5]
        if not (np.all(np.isfinite(params)) and np.all(f_low >= self._f_band_lo)
                and np.all(f_low < self._f_band_hi)):
            raise ValueError("SOBBH Gram rows outside the comp's band")
        data = self.acs.acs.flatten()[int(walker)].data
        n = int(params.shape[0])
        buf = self.acs.xp.zeros((n,) + tuple(data.arr.shape))
        self.comp.fill_global_wdm(
            params, buf,
            data_index=np.arange(n, dtype=np.int32),
            factors=np.ones(n, dtype=np.float64),
            m_band_half_width=self.m_band_half_width,
        )
        return buf, data.settings

    def _apply_cold_chain_sources(self, coords, sign):
        """Fold the leaf's cold-chain sources in/out through the CHUNKED fill.

        Replaces the base's dense pass -- ``apply_signal_from_params`` ->
        ``build_template``, "ONE ROW AT A TIME", i.e. a full TD waveform
        plus a full TD->WDM transform per walker -- with ONE batched
        ``fill_global_wdm`` call. Measured 2026-08-28 on the 6-mo probe: a
        dense SOBBH build is 32 s on this grid, so a 24-walker leaf visit
        paid 24 builds to expose and 24 to fold back; the first leaf never
        completed inside a 20-25 min preemption window.

        THE SIGN (vocabulary collides -- the numbers do not):
        ``apply_signal_from_params`` documents "+1 adds to the residual
        array, -1 subtracts", so the base calls ``sign=+1`` to EXPOSE a
        source (``r += h``) and ``sign=-1`` to fold it back (``r -= h``).
        ``fill_global_wdm`` accumulates ``factors * h`` into the same
        buffer and calls ``-1`` "remove" -- which is the move's *add_back*.
        Opposite words, identical arithmetic: ``factors = sign``.

        Rows whose template is zero -- non-finite, or ``f_low`` outside the
        comp's active band -- are dropped, mirroring the dense path's
        ``domain_error="skip"``. Skipping is deterministic in the coords,
        so a row skipped on the way out is skipped on the way back.

        ``fill_global_wdm`` is single-shard BY CONTRACT (it raises on a
        split holder), so a multi-GPU ACA is routed per shard with the same
        ``_shard_views``/``_partition`` primitives :meth:`_kernel_ll`
        already uses for scoring -- each view is a single-shard holder.

        ACCURACY: this puts chunked-heterodyne truncation into the RESIDUAL
        itself, where the stock move kept it bit-identical and approximated
        only the scoring. That trade was measured before it was taken (the
        6-mo removal-null sweep: residual 6e-4 of <h|h> at the Nt_sub=32
        defaults). ``SOBBH_CHUNKED_FILL=0`` restores the base's per-row pass,
        which builds each row through the container's installed generator: on
        the stock path the same fast comp's fill
        (:func:`~lisatools.globalfit.stock.erebor.source_runtime.get_sobbh_chunked_signal_gen`),
        not the exact generator.
        """
        _what = "EXPOSE (r += h)" if sign > 0 else "FOLD-BACK (r -= h)"
        if os.environ.get("SOBBH_CHUNKED_FILL", "1").strip() != "1":
            # Say so: this path costs ~32 s PER ROW on the 6-mo grid, and a
            # silent slow path is what made jobs 364/370 un-diagnosable.
            logger.info(
                "[SOBBH_FILL] %s via the DENSE per-row path "
                "(SOBBH_CHUNKED_FILL=0): %d rows",
                _what, int(np.shape(coords)[0]),
            )
            t0 = time.perf_counter()
            out = super()._apply_cold_chain_sources(coords, sign)
            logger.info(
                "[SOBBH_FILL] dense %s done in %.1f s", _what,
                time.perf_counter() - t0,
            )
            return out

        coords_np = np.atleast_2d(np.asarray(asnumpy(coords), dtype=np.float64))
        params = self.to_chunked_basis(coords_np)
        f_low = params[:, 5]
        valid = (
            np.all(np.isfinite(params), axis=1)
            & (f_low >= self._f_band_lo)
            & (f_low < self._f_band_hi)
        )
        if not np.any(valid):
            return

        # GLOBAL walker index per surviving row: the caller passes one row
        # per walker, so the row position IS the walker's residual slab.
        idx = np.arange(params.shape[0], dtype=np.int32)[valid]
        p = params[valid]
        factors = np.full(p.shape[0], float(sign), dtype=np.float64)
        n_shards = len(self.acs.linear_data_arr)
        t0 = time.perf_counter()

        if n_shards == 1:
            # on the device that owns the (single) shard (see _shard_context)
            with self._shard_context():
                self.comp.fill_global_wdm(
                    p, self.acs,
                    data_index=idx,
                    factors=factors,
                    m_band_half_width=self.m_band_half_width,
                )
            logger.info(
                "[SOBBH_FILL] %s via CHUNKED fill_global_wdm: %d rows "
                "(%d skipped) in %.2f s, 1 shard",
                _what, int(p.shape[0]), int(params.shape[0] - p.shape[0]),
                time.perf_counter() - t0,
            )
            return

        from ...utils.device import device_context
        from .gbbands import _RoutedBandEngine

        holder = self.acs
        views = _RoutedBandEngine._shard_views(holder)
        parts = _RoutedBandEngine._partition(holder, idx)
        xp = holder.xp
        for view, (pos, intra, _) in zip(views, parts):
            if pos.shape[0] == 0:
                continue
            with device_context(xp, view.device):
                self.comp.fill_global_wdm(
                    p[pos], view,
                    data_index=np.asarray(intra, dtype=np.int32),
                    factors=factors[pos],
                    m_band_half_width=self.m_band_half_width,
                )
        logger.info(
            "[SOBBH_FILL] %s via CHUNKED fill_global_wdm: %d rows "
            "(%d skipped) in %.2f s, %d shards",
            _what, int(p.shape[0]), int(params.shape[0] - p.shape[0]),
            time.perf_counter() - t0, n_shards,
        )

    def _verify_prev_logl(self, prev_logl, old_coords_in, data_index_in, leaf):
        """Built-in fast-vs-slow A/B at MATCHING convention.

        Recomputes the same points through the slow container path with the
        move's own generator and the SAME convention ``compute_like`` uses
        (NOT ``source_only=True`` — the base's variant would flag the
        per-walker ``d_d``/noise offset as a spurious spread). The residual
        difference is pure chunked-heterodyne truncation error; tolerance is
        ``SOBBH_CHECK_LL_TOL`` (default 0.5), severity/thinning shares the
        base's ``{BRANCH}_CHECK_LL`` / ``_EVERY`` knobs.
        """
        acs_like = (
            self.compute_check_like(old_coords_in, data_index_in)
            .reshape(prev_logl.shape)
            .real
        )
        both = (
            np.isfinite(prev_logl)
            & np.isfinite(acs_like)
            & (prev_logl > -1e299)
            & (acs_like > -1e299)
        )
        if not np.any(both):
            return
        diff = prev_logl[both] - acs_like[both]
        max_abs = float(np.abs(diff).max())
        if max_abs <= self.check_ll_tol:
            return
        spread = float(diff.max() - diff.min())
        msg = (
            f"{self.branch_name} leaf {leaf}: chunked-het fast path vs slow "
            f"container path disagree beyond tol={self.check_ll_tol}: "
            f"max|diff|={max_abs:.6e}, spread={spread:.6e} over "
            f"{int(both.sum())} points. Widen SOBBH_M_BAND_HALF_WIDTH / raise "
            "SOBBH_CHECK_LL_TOL if this is truncation error at high SNR; a "
            "large spread at low SNR means a real scoring/residual bug."
        )
        if self.check_ll_mode == "strict":
            raise ValueError(msg)
        logger.warning(msg)
