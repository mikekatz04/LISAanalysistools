"""SOBBH add/remove move scored by the chunked-heterodyne WDM likelihood.

:class:`SOBBHChunkedLikeMove` keeps ALL of :class:`ResidualAddOneRemoveOneMove`'s
choreography (per-leaf expose/fold, in-model repeats, per-leaf tempering,
cold-chain bookkeeping) and swaps ONLY the proposal-scoring path: instead of
one full-duration TD waveform + full TD->WDM transform per ``(temp, walker)``
row through the per-container Python loop, every batch is ONE vectorized
``SOBBHWDMComputations.get_ll_wdm`` call (BBHx ``sobbhcomps`` over LAT
``chunked_het``) directly against the ACA's live WDM residual buffers.

Convention bridge (load-bearing): the container path scores
``-1/2 (d_d + h_h - 2 d_h)`` [+ the per-walker noise term when the run fits a
psd branch], where ``d_d = <r|r>`` of the EXPOSED residual. The chunked call
returns only the source piece ``d_h - 1/2 h_h`` (comp built with ``d_d=0``),
so :meth:`setup_likelihood_here` captures the per-walker offset
``acs.likelihood()`` (= ``-1/2 d_d`` + noise term) on the freshly exposed
residual once per leaf and :meth:`compute_like` adds it back. This reproduces
the slow path's numbers on every scoring site (prev_logl, proposal batches,
fancy tempering swap) and keeps the base ``_verify_entry_vs_acs`` expose
invariant meaningful, up to chunked-heterodyne truncation error.

Residual expose/fold stays on the exact engine-installed generator
(residual integrity is bit-identical to the stock move); only scoring is
approximated. The built-in fast-vs-slow cross-check (``_verify_prev_logl``)
recomputes through the slow container path at MATCHING convention with a
tolerance knob ``SOBBH_CHECK_LL_TOL``.
"""

import logging
import os

import numpy as np

from ...utils.utility import asnumpy
from .addremovemove import ResidualAddOneRemoveOneMove

logger = logging.getLogger(__name__)

__all__ = ["SOBBHChunkedLikeMove"]


class SOBBHChunkedLikeMove(ResidualAddOneRemoveOneMove):
    """Add/remove move for the SOBBH branch scored via chunked-heterodyne.

    Args:
        *args: Positional arguments of :class:`ResidualAddOneRemoveOneMove`
            (``branch_name, coords_shape, waveform_gen, ...``). The
            ``waveform_gen`` stays the SLOW exact generator — it still owns
            the residual expose/fold and the cross-check/debug paths.
        chunked_comp: A built ``bbhx.sobbhcomps.SOBBHWDMComputations``
            constructed with ``d_d=0.0`` on the SAME ``WDMSettings``/orbits/
            TDI config as the run's data (see
            ``stock/erebor/source_runtime.get_sobbh_chunked_comp``).
        m_band_half_width: Narrow-band half-width (WDM layers) around each
            chunk's carrier — the chunked path's one live accuracy/speed
            knob (``SOBBH_M_BAND_HALF_WIDTH``).
        **kwargs: Keyword arguments of :class:`ResidualAddOneRemoveOneMove`.
            ``dcga`` must be ``None`` — the chunked kernel scores against
            the ACA buffers directly (multi-GPU walker shards are handled
            by per-split routing inside :meth:`compute_like`, not by the
            DCGA replica machinery).
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
                "inside compute_like. Build it without dcga= (the "
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
        # are served by per-split routing in compute_like (gbbands
        # _ShardHolderView + partition, each split under its own device
        # context — the comp re-asserts its geometry arrays per call so
        # they land on the current device)
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
        sites hand ``compute_like`` after the branch transform):
        ``(m1, m2, s1, s2, dist[Gpc], inc, f_low, lam, beta, psi, phi0)``.
        Output columns (``SOBBHTDIonTheFly``/chunked order):
        ``(m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)``.
        Sky angles pass through unchanged (both sides in the orbits frame).
        """
        coords_in = np.atleast_2d(np.asarray(coords_in, dtype=np.float64))
        out = coords_in[:, SOBBHChunkedLikeMove._CHUNKED_PERM].copy()
        out[:, 4] *= 1e9  # Gpc -> parsec
        return out

    # ------------------------------------------------------------------
    # likelihood seams
    # ------------------------------------------------------------------

    def setup_likelihood_here(self, coords):
        """Capture the per-walker exposed-residual offset for this leaf.

        Called by the base ``propose`` once per leaf, right after the leaf's
        cold-chain sources are exposed into the residual. ``acs.likelihood()``
        here is exactly the ``-1/2 <r|r>`` (+ noise-normalization term when
        configured) that the container scoring path folds into every value —
        the piece the chunked call (built with ``d_d = 0``) leaves out.
        """
        self._exposed_offset = np.asarray(asnumpy(self.acs.likelihood()), dtype=float)
        super().setup_likelihood_here(coords)

    def compute_like(self, coords_in, data_index):
        """One vectorized chunked-heterodyne call for the whole batch.

        Args:
            coords_in: ``(N, 11)`` already-transformed waveform-basis rows
                (all walkers x temps under the current leaf).
            data_index: ``(N,)`` physical-walker index per row.

        Returns:
            ``(N,)`` log-likelihoods in the container convention
            (``-1e300`` for invalid rows).
        """
        if self._dcga is not None:  # unreachable (ctor guard); keep loud
            raise NotImplementedError("SOBBHChunkedLikeMove has no DCGA path.")
        if self._exposed_offset is None:
            raise RuntimeError(
                "compute_like called before setup_likelihood_here armed the "
                "exposed-residual offset (propose() choreography violated)."
            )

        coords_np = np.atleast_2d(np.asarray(asnumpy(coords_in), dtype=np.float64))
        idx = np.asarray(asnumpy(data_index)).astype(np.int32).reshape(-1)
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
            return out

        ll, d_h, h_h = self._kernel_ll(params[valid], idx[valid])
        out[valid] = ll + self._exposed_offset[idx[valid]]
        self._last_d_h[valid] = d_h
        self._last_h_h[valid] = h_h
        return out

    def _kernel_ll(self, params, idx):
        """One chunked-het scoring pass, shard-routed when the ACA is split.

        Args:
            params: ``(N, 11)`` chunked-basis host rows (all valid).
            idx: ``(N,)`` GLOBAL walker indices.

        Returns:
            ``(ll, d_h, h_h)`` host arrays in row order.
        """
        if len(self.acs.linear_data_arr) == 1:
            ll = self.comp.get_ll_wdm(
                params, self.acs,
                data_index=idx, noise_index=idx,
                m_band_half_width=self.m_band_half_width,
            )
            return (
                np.asarray(asnumpy(ll), dtype=float),
                np.real(np.asarray(asnumpy(self.comp.d_h_out))),
                np.real(np.asarray(asnumpy(self.comp.h_h_out))),
            )

        # multi-GPU walker shards: reuse the GB shard-router primitives —
        # per-split single-shard views + the split partition — and run each
        # split's rows under the owning device context (cross-shard movement
        # is host-routed, matching the ACA conventions)
        from ...utils.device import device_context
        from .gbbands import _RoutedBandEngine

        holder = self.acs
        views = _RoutedBandEngine._shard_views(holder)
        parts = _RoutedBandEngine._partition(holder, idx)
        xp = holder.xp
        n = params.shape[0]
        ll = np.full(n, -1e300, dtype=float)
        d_h = np.full(n, np.nan)
        h_h = np.full(n, np.nan)
        for view, (pos, intra, _) in zip(views, parts):
            if pos.shape[0] == 0:
                continue
            with device_context(xp, view.device):
                vals = self.comp.get_ll_wdm(
                    params[pos], view,
                    data_index=np.asarray(intra, dtype=np.int32),
                    noise_index=np.asarray(intra, dtype=np.int32),
                    m_band_half_width=self.m_band_half_width,
                )
                ll[pos] = np.asarray(asnumpy(vals), dtype=float)
                d_h[pos] = np.real(np.asarray(asnumpy(self.comp.d_h_out)))
                h_h[pos] = np.real(np.asarray(asnumpy(self.comp.h_h_out)))
        return ll, d_h, h_h

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
            self.compute_acs_like(
                old_coords_in,
                data_index=data_index_in,
                signal_gen=self.waveform_gen,
                **self.waveform_like_kwargs,
            )
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
