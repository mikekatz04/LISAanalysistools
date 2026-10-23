"""MBH add/remove move scored by the batched, windowed grid-aligned likelihood.

The DEFAULT MBH scoring path of the stock erebor fits since 2026-09-30
(``MBH_LIKELIHOOD=auto`` resolves to it on every WDM, legacy-response run
without a conflicting ``MBH_WAVEFORM_DURATION``; see
``stock.erebor.source_runtime.resolve_mbh_batched_cfg``). ``MBH_LIKELIHOOD=full``
keeps the per-row container path. A data span shorter than the window
(3-month runs, the lite smokes) clamps the window to the data
(:func:`mbh_window_layers`) instead of refusing the build.

Production (9mo / 12mo, ``MBH_LIKELIHOOD=batched``): stock name ``mbh_pe``, built by
``recipe.MBHBatchedMoveBuilder`` from ``source_runtime.build_mbh_move_runtime``; it runs
in ``source_search``, every cycle of ``gb_search_1`` / ``gb_search_2`` and in ``full_pe``.
The window and batch sizes come from the ``SourceMBHSettings`` fields
``MBH_BATCH_MAX_SIZE`` (8), ``MBH_WINDOW_BEFORE_DAYS`` (90), ``MBH_WINDOW_AFTER_DAYS``
(10), ``MBH_WINDOW_PAD_DAYS`` (4) and ``MBH_WINDOW_MARGIN_DAYS`` (1), converted to
seconds by ``resolve_mbh_batched_cfg`` (``MBH_WINDOW_DECIMATE``, default 1, launchers 2,
sets the windowed generator's sampling, not the move). Entry points:
:class:`MBHBatchedLikeMove` and :func:`mbh_window_layers`.

:class:`MBHBatchedLikeMove` keeps ALL of :class:`ResidualAddOneRemoveOneMove`'s
choreography (per-leaf expose/fold, in-model repeats, per-leaf tempering,
cold-chain bookkeeping) and swaps the scoring and fill paths, exactly as
:class:`SOBBHChunkedLikeMove` does for SOBBH:

* every chunk of ``batch_max_size`` rows is ONE batched ``compute_tdi_channels``
  launch on the leaf's shared lattice (the ``WindowedGridAlignedMBHWaveform``
  behind a ``MBHWindowedWDMSignalGen``), then a per-row segment WDM transform;
* every row is scored against ITS OWN container: the residual and the PSD are
  sliced to the template's box (``AnalysisContainer._slice_to_template``), so
  per-walker PSD samples are honoured;
* the convention bridge is the SOBBH one: ``compute_like`` returns
  ``offset[walker] + <r|h> - 1/2 <h|h>`` with ``offset = acs.likelihood()`` on
  the freshly exposed residual, which reproduces the container path's
  ``-1/2 <r-h|r-h>`` + noise term on every scoring site;
* the leaf window (kept WDM layers around the leaf's median cold-chain merger
  time) is fixed on expose/setup and locked for the visit; it is rebuilt only
  when the median leaves the margin band, and never on fold-back.
* times: the window is placed with the adapter's ``t0_abs`` (the ABSOLUTE
  time of WDM layer 0, ``general_info.data_t0``), never with a settings
  ``t0``; and every generated template is re-labelled onto a box cut from the
  CONTAINERS' own data settings before scoring and fill. A stock erebor build
  gives the containers ``t0 = 0`` (the WDM factory ignores ``times[0]``) and
  mutates ``general_info.domain_settings.t0`` to the data start when a GB
  comp is built, so neither settings ``t0`` is the data start in general.

A batch the response refuses (``BatchNotLaunchable``) or a domain error in the
chunk falls back to the per-row container path for that chunk, LOUDLY, and is
counted in ``n_batch_fallbacks``. A JAX RUNTIME failure inside the generator
(phentax's (2,2)-coefficient solve not converging for one row: an optimistix
bisection hitting ``max_steps``, an equinox ``error_if``) is a different
animal: it poisons the whole vmapped chunk, and the container path would fail
on the same row. :meth:`_score_chunk` then re-generates the chunk ROW BY ROW
and scores each row that fails at the log-likelihood floor (``-1e300``, the
out-of-domain sentinel the move already uses), logging that row's waveform
parameters so the region can be found, and counting it in
``n_generate_failures`` / the ``[MBH_BATCH]`` ``gen_failed`` field. The 1yr
flagship job 760 (2026-10-14) died on exactly this: one hot-rung row of
MBHB 10 (merger 9.5 d past the data end, inspiral-only, spins and total
mass loosely constrained) took rank 1 down with
``_EquinoxRuntimeError: The maximum number of steps was reached in the
nonlinear solver`` from ``jit__compute_coeffs_22``. Note the solve in
question (``phentax.core.phase.get_time_of_frequency``, the t(f) bisection
for ``f_min`` / ``f_ref``) is one whose RESULT IS DISCARDED here -- lisatools
always passes ``t_min`` and ``t_ref`` -- but phentax selects between the
given and the solved value with ``lax.cond`` under ``jax.vmap``, which
evaluates BOTH branches, so a non-converging solve raises anyway. WHEN it
fails (reproduced, ``scripts/mbh/phentax_tf_bisection_sliver.py``): the
bisection's target ``2 pi M f_min`` lands inside the frequency JUMP that
``time_of_freq`` has at ``t = tEarly`` (phentax shifts ``t -> t - tt0`` below
the PN / pseudo-PN junction), so the interval collapses onto the jump and
``|f|`` never reaches ``atol = 1e-12``. For MBHB 10's mass ratio and spins
that is total mass 4.2224e6..4.2274e6 Msun (a 0.12 % sliver, 1.5x the
truth), i.e. the masses at which the 7e-5 Hz start frequency is reached
exactly ``|tEarly| ~ 2.1e5 M`` before merger; every leaf has such a sliver
at its own ``(eta, s1z, s2z)``. The upstream fix is
``optx.root_find(..., throw=False)`` (or hoisting the ``isnan`` test out of
the vmapped function); here the row is simply rejected.

A row whose IN-DATA merger falls OUTSIDE the locked kept box is rejected at
the same floor (``MBH_REJECT_OUTSIDE_BOX=0`` restores the old truncated
scoring). The window scores the box only, so such a row's template is
TRUNCATED there: a merger 68 d before the box scores ``<h|h> ~ 0`` and the
row is "free" at every rung. Accepting it puts a point in the state whose
template the residual does not carry. The 1yr flagship job 769 (2026-10-16)
did exactly that with MBHB 10 on walker 2 (``t_plunge`` 195 d against a box
pinned to the last 102 d of the data, the source merging after the data
end): the cold-chain record held ``<h|h>`` 0.000 for the point, then 24,917
(SNR 158) for the SAME coords once the window was rebuilt around it, with
the walker's total lnL and its residual unchanged -- and a resume rebuilds
the residual from the coords through the stock full generator, which would
subtract the phantom for real. With the floor, the CURRENT point of a rung
outside the box scores the floor too (``prev_logl`` comes through
:meth:`compute_like`), so the next in-box proposal replaces it, the
tempering swap never pulls it down and the lost-rung guard re-anchors it.
Such rows stay counted in ``outside_box``.
``MBH_BATCHED_FILL=0`` restores the dense
per-row expose/fold. The built-in fast-vs-slow check (``_verify_prev_logl``)
recomputes through the stock generator at matching convention with
``MBH_CHECK_LL_TOL`` (default 0.5 nats), gating on the COLD rung only (hot
rungs reported), every 10th visit unless ``MBH_CHECK_LL_EVERY`` is set.
Generation runs under both the cupy and the JAX device context of the
owning shard (phentax is JAX).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

import numpy as np

from ...analysiscontainer import shard_lookup_maps
from ...diagnostic import inner_product
from ...domains import WDMSettings, WDMSignal
from ...utils.device import device_context, jax_device_context
from ...utils.exceptions import BatchNotLaunchable, WaveformDomainError
from ...utils.utility import asnumpy, get_array_module
from .addremovemove import ResidualAddOneRemoveOneMove

logger = logging.getLogger(__name__)

__all__ = ["MBHBatchedLikeMove", "mbh_window_layers"]

#: waveform-basis column of the merger time (seconds relative to waveform_t0)
_T_PLUNGE_COL = 10
#: the only generator kwargs the grid-aligned generator accepts
_ACCEPTED_GEN_KWARGS = ("start_freq", "ref_freq", "T")
#: clamped geometries already logged by :func:`mbh_window_layers` (log once)
_CLAMP_LOGGED: set = set()
#: per leaf, how many generator-failure rows are logged WITH their parameters
#: before the move falls back to counting them (``gen_failed`` telemetry)
_GEN_FAIL_LOG_CAP = 10


def _is_jax_runtime_failure(exc) -> bool:
    """``True`` for a JAX / equinox RUNTIME failure raised inside the generator.

    These are the errors a compiled program raises while RUNNING -- an
    optimistix solver hitting ``max_steps``, an equinox ``error_if`` firing --
    surfacing as ``jax.errors.JaxRuntimeError`` (``jaxlib`` ``XlaRuntimeError``,
    "INTERNAL: CpuCallback error ...") from a jitted call, or as equinox's own
    ``_EquinoxRuntimeError`` when the program ran eagerly. All subclass
    ``RuntimeError`` and live in the ``jax`` / ``jaxlib`` / ``equinox``
    packages. The test is deliberately narrow (see ``BatchNotLaunchable``'s
    docstring on catching bare ``Exception``): a Python-level bug in the
    generator -- a ``TypeError``, a lisatools ``RuntimeError`` -- still raises.
    No JAX import: the move must not require JAX to be importable.
    """
    if not isinstance(exc, RuntimeError):
        return False
    module = (getattr(type(exc), "__module__", None) or "").split(".")[0]
    return module in ("jax", "jaxlib", "equinox")


def mbh_window_layers(
    wdm, t_merge_abs, window_before, window_after, window_pad, window_margin, t0_abs=None,
):
    """Kept-box and pad geometry, in WDM layers of ``wdm``, for a merger at ``t_merge_abs``.

    ``t0_abs`` is the ABSOLUTE time of layer 0 of ``wdm``'s grid (the data
    start, ``general_info.data_t0``); it defaults to ``wdm.t0`` for callers
    whose settings carry it. A stock erebor build's settings do NOT: the WDM
    factory builds ``t0 = 0`` and a GB comp build later sets it to the data
    start in place, so ``wdm.t0`` is 0 or ``data_t0`` depending on build
    order -- the batched move always passes ``t0_abs`` explicitly.

    Kept layers ``[n_start, n_start + Nt_keep)`` cover ``[t - before - margin,
    t + after + margin]`` snapped OUTWARD to layer boundaries, clamped to the
    data's ACTIVE time box ``[ind_min_t, ind_max_t]``. The transform segment is
    ``[n_start - n_pad_lo, n_start + Nt_keep + n_pad_hi)``, always
    ``Nt_keep + 2 n_pad`` layers long, clamped to the grid ``[0, Nt)``:

    * interior: ``n_pad_lo = n_pad_hi = n_pad``;
    * near a grid edge the segment stops AT the edge, so that side's pad
      shrinks (to 0 when the box touches the edge) and the other side's pad
      grows by the same amount. The kept box still reaches the edge, so a
      merger in the first/last days of the data is covered. The segment
      transform's periodic wrap then touches the far end of the segment --
      pad, where the template is zero (onset inside the box) -- and the edge
      itself is tapered by the data window.

    ``Nt_keep`` and the segment length depend on the durations only (never on
    ``t_merge_abs``), so the shared lattice length -- and phentax's jit cache
    -- is constant across leaves, edges included. ``Nt_keep + 2 n_pad`` is
    made even (``WDMSettings`` needs an even layer count). The adapter's
    even-start parity rule then grows the low pad by one only for an odd
    interior start; a start clamped to layer 0 or to ``Nt - Nt_seg`` is
    already even.

    SHORT DATA SPANS (2026-09-30; a 3-month run, the lite smokes): when the
    configured window does not fit, it is CLAMPED instead of refused --
    ``Nt_keep = min(Nt_keep, active layers)`` and the segment ``min(Nt_keep +
    2 n_pad, Nt)`` (still even; an odd clamped ``Nt_keep`` gets the spare
    layer as pad), so the pads shrink to what fits on each side, possibly 0.
    A window that covers the whole active box AND the whole grid is the stock
    full-grid transform restricted to the active box. Logged once per
    geometry (``[MBH_BATCH]``, INFO). ``n_pad`` keeps the CONFIGURED pad; the
    actual pads are ``n_pad_lo`` / ``n_pad_hi``.

    Args:
        wdm (WDMSettings): The run's WDM grid (``layer_dt``, ``Nt``,
            ``ind_min_t`` / ``ind_max_t``, ``t0``).
        t_merge_abs (float): Absolute merger time, s.
        window_before (float): Kept span before the merger, s.
        window_after (float): Kept span after the merger, s.
        window_pad (float): Transform pad on each side of the kept box, s
            (rounded UP to whole layers).
        window_margin (float): Extra kept span on both sides, s (the
            hysteresis band of :meth:`MBHBatchedLikeMove._leaf_window`).
        t0_abs (float, optional): Absolute time of layer 0, s. Default
            ``None`` uses ``wdm.t0``.

    Returns:
        dict: ``dict(n_start, Nt_keep, n_pad, n_pad_lo, n_pad_hi)``, all ``int``
        layer counts / indices (``n_pad`` is the configured pad).

    Raises:
        ValueError: Only a geometry with no valid segment: an empty active
            box, or a whole-grid segment on an odd-length grid.

    Example:
        .. code-block:: python

            geom = mbh_window_layers(wdm, t_merge_abs=t0 + 1.2e7,
                                     window_before=90 * 86400.0, window_after=10 * 86400.0,
                                     window_pad=4 * 86400.0, window_margin=86400.0,
                                     t0_abs=t0)
            box = slice(geom["n_start"], geom["n_start"] + geom["Nt_keep"])
    """
    layer_dt = float(wdm.layer_dt)
    t0 = float(wdm.t0) if t0_abs is None else float(t0_abs)
    Nt = int(wdm.Nt)
    act_lo = int(wdm.ind_min_t)
    act_hi = int(wdm.ind_max_t) + 1
    n_pad = int(np.ceil(float(window_pad) / layer_dt))
    span = float(window_before) + float(window_after) + 2.0 * float(window_margin)
    Nt_keep = int(np.ceil(span / layer_dt)) + 1
    if (Nt_keep + 2 * n_pad) % 2:
        Nt_keep += 1
    Nt_seg = Nt_keep + 2 * n_pad
    if Nt_seg > Nt or Nt_keep > act_hi - act_lo:
        keep_c = min(Nt_keep, act_hi - act_lo)
        seg_c = keep_c + 2 * n_pad
        if seg_c % 2:  # odd clamped box: the spare layer is pad
            seg_c += 1
        seg_c = min(seg_c, Nt)
        if keep_c < 1 or seg_c % 2:
            raise ValueError(
                f"MBH window cannot be placed on this WDM grid of {Nt} layers (active "
                f"time layers [{act_lo}, {act_hi})) x {layer_dt:.0f} s: the clamped "
                f"segment would be {seg_c} layers around a kept box of {keep_c} (a "
                "segment needs >= 1 kept layer and an even layer count)."
            )
        key = (Nt, act_lo, act_hi, layer_dt, Nt_keep, n_pad, keep_c, seg_c)
        if key not in _CLAMP_LOGGED:
            _CLAMP_LOGGED.add(key)
            logger.info(
                "[MBH_BATCH] window clamped to the data span: configured %.1f d box = "
                "%d kept + 2 x %d pad layers (%d) does not fit the WDM grid of %d layers "
                "(active time layers [%d, %d)) x %.0f s; kept box -> %d layers%s, "
                "segment -> %d layers%s.", span / 86400.0, Nt_keep, n_pad, Nt_seg, Nt,
                act_lo, act_hi, layer_dt, keep_c,
                " (the whole active box)" if keep_c == act_hi - act_lo else "",
                seg_c, " (the whole grid)" if seg_c == Nt else "",
            )
        Nt_keep, Nt_seg = keep_c, seg_c
    lo_abs = float(t_merge_abs) - float(window_before) - float(window_margin)
    n_start = int(np.floor((lo_abs - t0) / layer_dt))
    n_start = min(max(n_start, act_lo), act_hi - Nt_keep)
    s0 = min(max(n_start - n_pad, 0), Nt - Nt_seg)
    n_pad_lo = n_start - s0
    n_pad_hi = s0 + Nt_seg - (n_start + Nt_keep)
    return dict(
        n_start=int(n_start), Nt_keep=int(Nt_keep), n_pad=int(n_pad),
        n_pad_lo=int(n_pad_lo), n_pad_hi=int(n_pad_hi),
    )


class MBHBatchedLikeMove(ResidualAddOneRemoveOneMove):
    """Add/remove move for the MBH branch scored through the batched windowed path.

    The ``propose(model, state)`` contract is the base class's
    (:meth:`ResidualAddOneRemoveOneMove.propose_local` on every compute rank's walker
    block, through the :class:`~lisatools.globalfit.moves.walkerfanout.WalkerFanoutMixin`
    ``propose``): per leaf, expose the cold-chain source, :meth:`setup_likelihood_here`,
    score ``prev_logl`` and the in-model repeats through :meth:`compute_like_local`, swap
    the per-leaf ladder, record ``<d|h>`` / ``<h|h>``, fold back. The tempered ensemble
    is the ``mbh`` sub-state; the base writes it and mirrors row 0 into the main state
    (``_sync_cold_row``). This class replaces the scoring and the expose / fold fill.

    Window hysteresis relies on the walkers' spread in merger time (the
    posterior width, plus hot-rung excursions) staying inside
    ``window_margin`` of the locked window's reference time; a much wider
    spread needs ``MBH_WINDOW_MARGIN_DAYS`` resized to match, or rows land
    outside the locked kept box and are REJECTED at the floor (counted as
    ``outside_box`` in the ``[MBH_BATCH]`` telemetry; the module docstring
    has the production case, ``MBH_REJECT_OUTSIDE_BOX=0`` the old truncated
    scoring).

    Args:
        *args: Positional arguments of :class:`ResidualAddOneRemoveOneMove`.
            ``waveform_gen`` stays the SLOW exact generator (or ``None`` to use
            the containers' installed one): it still owns the cross-check.
        batched_gen: the windowed sub-transform adapter
            (:class:`~lisatools.sources.batching.MBHWindowedWDMSignalGen`), or a
            ``DeviceLocalWaveGen`` resolving to one per device. Must expose
            ``set_window``, ``window_key``, ``waveform_t0``, ``t_plunge_snap``
            and ``t0_abs`` (the absolute time of WDM layer 0). Required.
        batch_max_size (int): rows per generator launch (``MBH_BATCH_MAX_SIZE``).
            Default 8, floored at 1.
        window_before (float): kept span before the leaf's median merger, s.
            Default 90 d.
        window_after (float): kept span after it, s. Default 10 d.
        window_pad (float): segment-transform pad on each side, s. Default 4 d.
        window_margin (float): hysteresis margin, s. Default 1 d.
        **kwargs: Keyword arguments of the base. ``dcga`` must be ``None``.

    Raises:
        ValueError: ``dcga`` given, ``batched_gen`` missing, a non-WDM run domain,
            or a window with no valid geometry on the grid (:func:`mbh_window_layers`).

    Attributes:
        batched_gen: See ``batched_gen``.
        n_batch_fallbacks (int): Chunks the generator refused (container path).
        check_ll_tol (float): ``MBH_CHECK_LL_TOL`` (default ``0.5`` nat).

    Note:
        Env knobs read by this class or by the base with the ``MBH`` prefix:
        ``MBH_REJECT_OUTSIDE_BOX`` (``0``/``false``/``off``/``no`` disables),
        ``MBH_BATCHED_FILL`` (default ``1``; anything else uses the base's dense
        per-row expose / fold), ``MBH_CHECK_LL`` (base default ``ADDREMOVE_CHECK_LL``
        then ``1``; launchers ``0``), ``MBH_CHECK_LL_EVERY`` (class default every 10th
        visit), ``MBH_CHECK_LL_TOL`` and ``MBH_RECORD_DH`` (class default ``1``).

    Example:
        Built by the stock layer (``source_runtime.build_mbh_move_runtime``):

        .. code-block:: python

            from lisatools.globalfit.recipe import MBHBatchedMoveBuilder

            _, moves = MBHBatchedMoveBuilder(
                wave_gen=slow.get_signals_for_residuals,  # stock phentax generator
                batched_gen=batched,  # per-device windowed adapter
                batch_max_size=8,
                window_before=90 * 86400.0, window_after=10 * 86400.0,
                window_pad=4 * 86400.0, window_margin=86400.0,
            ).build(None, curr, acs, priors, state)
            mbh_pe = moves[0]
    """

    _record_dh_default = "1"
    #: The cross-check rebuilds every row through the STOCK generator (a
    #: 90-day waveform per row), so it runs every 10th visit unless
    #: ``MBH_CHECK_LL_EVERY`` is set explicitly.
    _check_ll_every_default = "10"

    def __init__(
        self, *args, batched_gen=None, batch_max_size=8,
        window_before=90 * 86400.0, window_after=10 * 86400.0,
        window_pad=4 * 86400.0, window_margin=86400.0, **kwargs,
    ):
        if kwargs.get("dcga") is not None:
            raise ValueError(
                "MBHBatchedLikeMove has no DCGA (replica) path: the batched "
                "generator scores against the ACA containers directly, and "
                "multi-GPU walker shards are served by per-shard routing inside "
                "compute_like_local. Build it without dcga= (use_dcga=False)."
            )
        if batched_gen is None:
            raise ValueError("MBHBatchedLikeMove requires batched_gen= (the windowed adapter).")
        super().__init__(*args, **kwargs)
        self.batched_gen = batched_gen
        self.batch_max_size = max(1, int(batch_max_size))
        self.window_before = float(window_before)
        self.window_after = float(window_after)
        self.window_pad = float(window_pad)
        self.window_margin = float(window_margin)
        self._wdm = self.acs.acs.flatten()[0].data.settings
        if not isinstance(self._wdm, WDMSettings):
            raise ValueError(
                "MBH_LIKELIHOOD=batched needs a WDM run domain; the containers "
                f"carry {type(self._wdm).__name__}."
            )
        # Fail at BUILD, not at the first leaf, when the window has no valid
        # geometry on this grid (a short data span is CLAMPED, logged once,
        # not refused; the geometry's size is independent of the merger time
        # and of the layer-0 time, so the settings' own t0 serves here).
        mbh_window_layers(
            self._wdm, float(self._wdm.t0), self.window_before, self.window_after,
            self.window_pad, self.window_margin,
        )
        self._exposed_offset = None
        self._leaf_windows = {}
        self.n_batch_fallbacks = 0
        self.last_batch_error = None
        self._warned_fallback_leaf = None
        #: rows whose waveform generation FAILED at runtime (JAX / equinox) and
        #: were scored at the floor; the last such error; per-leaf log tally
        self.n_generate_failures = 0
        self.last_generate_error = None
        self._gen_fail_logged = {}
        self.check_ll_tol = float(
            os.environ.get(f"{self._dbg_prefix}_CHECK_LL_TOL", "0.5")
        )
        #: rows whose in-data merger falls outside the locked kept box are
        #: scored at the floor (rejected) instead of through a truncated
        #: template; on by default, ``{BRANCH}_REJECT_OUTSIDE_BOX=0``
        #: (``MBH_REJECT_OUTSIDE_BOX``) restores the truncated scoring (module
        #: docstring: the job 769 phantom).
        self.reject_outside_box = (
            os.environ.get(f"{self._dbg_prefix}_REJECT_OUTSIDE_BOX", "1").strip().lower()
            not in ("0", "false", "off", "no")
        )
        gen_kwargs = dict(self.waveform_gen_kwargs or {})
        dropped = sorted(set(gen_kwargs) - set(_ACCEPTED_GEN_KWARGS))
        self._gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in _ACCEPTED_GEN_KWARGS}
        if dropped:
            logger.info(
                "[MBH_BATCH] waveform kwargs not forwarded to the grid-aligned "
                "generator (it takes only %s): %s", _ACCEPTED_GEN_KWARGS, dropped,
            )
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats():
        """Fresh per-leaf telemetry dict for the ``[MBH_BATCH]`` line (:meth:`_flush_stats`)."""
        # ``leaf`` is recorded when accumulation STARTS (first scored chunk),
        # so the flush -- which runs at the NEXT leaf's setup -- labels the
        # numbers with the leaf they belong to.
        return dict(
            leaf=None, rows=0, seconds=0.0, chunks=0, fallbacks=0,
            gen_s=0.0, score_s=0.0, outside_box=0, outside_data=0, gen_failed=0,
        )

    # ------------------------------------------------------------------
    # window management
    # ------------------------------------------------------------------

    def _adapter(self):
        """The windowed adapter for the CURRENT device (resolves a ``DeviceLocalWaveGen``)."""
        resolve = getattr(self.batched_gen, "_resolve", None)
        return resolve() if callable(resolve) else self.batched_gen

    @staticmethod
    def _gen_attr(adapter, name):
        """REQUIRED ``adapter.name``, else the wrapped ``adapter.wave_gen.name``.

        Used for ``waveform_t0`` (the SNAPPED, on-lattice epoch the batched
        generator was built with), ``t_plunge_snap`` (snapped minus stock
        epoch) and ``t0_abs`` (absolute time of WDM layer 0, an adapter
        property). ``MBHWindowedWDMSignalGen`` does not forward the first two,
        so the wiring sets them on the adapter (or they live on the wrapped
        generator). There is deliberately NO default: a missing snap read as
        0 would shift every template by up to dt/2, and a layer-0 time read
        off the containers' settings would misplace every window by the data
        start, with no error anywhere.
        """
        if hasattr(adapter, name):
            return getattr(adapter, name)
        inner = getattr(adapter, "wave_gen", None)
        if inner is not None and hasattr(inner, name):
            return getattr(inner, name)
        raise AttributeError(
            f"MBHBatchedLikeMove: batched_gen exposes no {name!r} (neither the "
            f"adapter nor its wave_gen). Set adapter.{name} when wiring the "
            "batched MBH likelihood (waveform_t0 = the snapped epoch the "
            "generator was built with, t_plunge_snap = snapped - stock epoch; "
            "0.0 only if the stock epoch is already on the data lattice; "
            "t0_abs = the absolute data start, MBHWindowedWDMSignalGen(t0_abs=))."
        )

    def _stock_waveform_t0(self):
        """The UNSNAPPED epoch the rows' ``t_plunge`` is relative to.

        Two run constants: resolved ONCE, under the primary shard's device
        contexts (resolving a ``DeviceLocalWaveGen`` outside any context would
        build a generator replica on whatever device happens to be current),
        and cached on the move."""
        cached = getattr(self, "_stock_t0_cache", None)
        if cached is not None:
            return cached
        with self._device_contexts(self._primary_device()):
            adapter = self._adapter()
            val = float(self._gen_attr(adapter, "waveform_t0")) - float(
                self._gen_attr(adapter, "t_plunge_snap")
            )
        self._stock_t0_cache = val
        return val

    def _grid_t0_abs(self):
        """ABSOLUTE time of WDM layer 0: the adapter's REQUIRED ``t0_abs``.

        Never ``self._wdm.t0``: the containers' data settings carry whatever
        ``t0`` they had when the data were poured -- 0 in a stock erebor build
        (``WDMSettings.make_factory`` ignores ``times[0]``) -- while merger
        times are absolute. Like :meth:`_stock_waveform_t0`, a run constant
        resolved once under the primary shard's device contexts."""
        cached = getattr(self, "_grid_t0_cache", None)
        if cached is not None:
            return cached
        with self._device_contexts(self._primary_device()):
            val = float(self._gen_attr(self._adapter(), "t0_abs"))
        self._grid_t0_cache = val
        return val

    def _leaf_window(self, leaf, coords_ws, allow_rebuild):
        """The locked window geometry of ``leaf`` (cached in ``_leaf_windows``).

        The reference time is the stock epoch plus the median ``t_plunge``
        (waveform-basis column 10) of ``coords_ws``. A cached window is kept
        unless ``allow_rebuild`` and the reference moved more than
        ``window_margin``; then it is rebuilt with :func:`mbh_window_layers`
        (logged). ``allow_rebuild`` is True on setup / expose, False on
        fold-back.

        Returns:
            dict: The :func:`mbh_window_layers` geometry plus ``t_ref`` (s).

        Raises:
            RuntimeError: A fold-back reached a leaf no expose has windowed.
        """
        t_ref = self._stock_waveform_t0() + float(np.median(coords_ws[:, _T_PLUNGE_COL]))
        win = self._leaf_windows.get(leaf)
        if win is not None and (
            not allow_rebuild or abs(t_ref - win["t_ref"]) <= self.window_margin
        ):
            return win
        if win is None and not allow_rebuild:
            raise RuntimeError(
                f"MBHBatchedLikeMove: leaf {leaf} fold-back reached before any "
                "expose set its window (propose choreography violated)."
            )
        geom = mbh_window_layers(
            self._wdm, t_ref, self.window_before, self.window_after,
            self.window_pad, self.window_margin, t0_abs=self._grid_t0_abs(),
        )
        geom["t_ref"] = t_ref
        if win is not None:
            logger.info(
                "[MBH_BATCH] leaf %d window rebuilt: median merger moved %.2f h "
                "(layers %d -> %d)", leaf, (t_ref - win["t_ref"]) / 3600.0,
                win["n_start"], geom["n_start"],
            )
        self._leaf_windows[leaf] = geom
        return geom

    @staticmethod
    def _apply_window(adapter, geom):
        """Set ``geom``'s window on ``adapter`` unless its ``window_key`` already matches."""
        key = (geom["n_start"], geom["Nt_keep"], geom["n_pad_lo"], geom["n_pad_hi"])
        if getattr(adapter, "window_key", None) != key:
            adapter.set_window(*key)

    @contextmanager
    def _device_contexts(self, device):
        """cupy AND JAX placement on ``device`` (phentax runs in JAX).

        cupy's :func:`device_context` does not move JAX: without the JAX
        twin every shard's phentax generation lands on JAX's default device
        (gpu0). Same pairing as the stock ``SourceSignalGen``."""
        xp = getattr(getattr(self, "acs", None), "xp", None)
        with device_context(xp, device), jax_device_context(device):
            yield

    def _primary_device(self):
        """``acs.gpus[0]`` as an int, or ``None`` on a CPU / no-GPU ACA."""
        gpus = getattr(getattr(self, "acs", None), "gpus", None)
        return None if gpus is None else int(np.atleast_1d(gpus)[0])

    # ------------------------------------------------------------------
    # shard routing
    # ------------------------------------------------------------------

    @staticmethod
    def _split_rows_static(holder, idx):
        """``[(device, positions)]`` grouping ``idx`` rows by owning walker shard.

        ``device`` is ``None`` when ``holder.gpus`` is ``None``; a single-shard
        holder returns one group with every position. Also used by the EMRI and
        SOBBH moves.
        """
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        n_shards = len(holder.linear_data_arr)
        if holder.gpus is None:
            return [(None, np.arange(idx.size))]
        if n_shards == 1:
            return [(int(holder.gpus[0]), np.arange(idx.size))]
        split_map, _ = shard_lookup_maps(holder)
        owner = np.asarray(split_map)[idx]
        return [
            (int(holder.gpus[s]), np.where(owner == s)[0])
            for s in range(n_shards) if np.any(owner == s)
        ]

    def _split_rows(self, idx):
        """:meth:`_split_rows_static` on this move's ACA."""
        return self._split_rows_static(self.acs, idx)

    # ------------------------------------------------------------------
    # Gram information matrix (MBH_EIGEN_INFO=gram; the route itself is
    # ResidualAddOneRemoveOneMove._gram_info -- these are its two hooks)
    # ------------------------------------------------------------------

    def _gram_context(self, walker):
        """Enter the device(s) that own walker ``walker``'s shard."""
        (device, _), = self._split_rows(np.array([int(walker)]))
        return self._device_contexts(device)

    def _gram_templates(self, coords, walker):
        """The batched windowed templates for the Gram rows: ``(arr, box)``
        with one row of ``arr`` per waveform-basis row of ``coords``.

        Generated in the CURRENT leaf's window (set by
        ``setup_likelihood_here`` / the expose) and relabelled onto the
        containers' box -- exactly what the scorer differences against.
        Built in ``batch_max_size`` chunks and concatenated (every chunk
        shares the one window, hence one ``box``). ``walker`` is unused: the
        template does not depend on the walker (its noise enters through the
        caller's inner products). No window for the leaf, or a batch the
        generator refuses, raises, and the refresh falls back to the
        likelihood route.
        """
        leaf = int(self._current_leaf)
        geom = self._leaf_windows.get(leaf)
        if geom is None:
            raise RuntimeError(f"MBHBatchedLikeMove: leaf {leaf} window not set.")
        adapter = self._adapter()
        self._apply_window(adapter, geom)
        coords = np.atleast_2d(np.asarray(asnumpy(coords), dtype=np.float64))
        parts, box = [], None
        for lo in range(0, coords.shape[0], self.batch_max_size):
            tmpl = self._on_container_box(
                self._generate(adapter, coords[lo: lo + self.batch_max_size]))
            parts.append(tmpl.arr if tmpl.is_batched else tmpl.arr[None])
            box = tmpl.settings if box is None else box
        xp = get_array_module(parts[0])
        return xp.concatenate(parts), box

    # ------------------------------------------------------------------
    # likelihood
    # ------------------------------------------------------------------

    def setup_likelihood_here(self, coords):
        """Arm the per-walker exposed-residual offset and the leaf window."""
        self._flush_stats()
        self._exposed_offset = np.asarray(asnumpy(self.acs.likelihood()), dtype=float)
        coords_np = np.atleast_2d(np.asarray(asnumpy(coords), dtype=np.float64))
        self._leaf_window(int(self._current_leaf), coords_np, allow_rebuild=True)
        super().setup_likelihood_here(coords)

    def compute_like_local(self, coords_in, data_index):
        """Rank-local batched windowed scorer (the base ``compute_like`` calls it).

        Rows whose in-data merger falls outside the leaf's locked kept box are
        counted (``outside_box``) and, with ``reject_outside_box``, left at the
        floor. The rest are grouped by owning shard, chunked by
        ``batch_max_size`` and scored by :meth:`_score_chunk` under the shard's
        cupy + JAX device contexts. Side outputs ``self._last_d_h`` /
        ``self._last_h_h`` hold the per-row ``<r|h>`` / ``<h|h>``.

        Args:
            coords_in (xp.ndarray): Waveform-basis rows, shape ``(n, 11)`` (column
                10 is ``t_plunge`` relative to the stock epoch).
            data_index (xp.ndarray): Walker (container) index per row, shape ``(n,)``.

        Returns:
            np.ndarray: ``(n,)`` log-likelihoods,
            ``offset[walker] + <r|h> - 1/2 <h|h>``; ``-1e300`` for a
            non-finite, rejected or non-generatable row.

        Raises:
            RuntimeError: Called before :meth:`setup_likelihood_here`, or the
                current leaf has no window.
        """
        if self._dcga is not None:  # unreachable (ctor guard); keep loud
            raise NotImplementedError("MBHBatchedLikeMove has no DCGA path.")
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
        leaf = int(self._current_leaf)
        geom = self._leaf_windows.get(leaf)
        if geom is None:
            raise RuntimeError(f"MBHBatchedLikeMove: leaf {leaf} window not set.")
        if self._stats["leaf"] is None:
            self._stats["leaf"] = leaf
        valid_pos = np.where(valid)[0]
        # rows whose IN-DATA merger falls outside the locked kept box: their
        # template would be TRUNCATED there (a merger well before the box
        # scores <h|h> ~ 0, a "free" point at every rung), so they are
        # REJECTED at the floor -- accepting one puts a point in the state
        # whose template the residual does not carry (the module docstring
        # has the job 769 phantom; MBH_REJECT_OUTSIDE_BOX=0 restores the
        # truncated scoring). A steady count means the margin/prior sizing is
        # off -- see the class docstring. A merger outside the data's active
        # time box is clamped to that box's edge first: the data itself ends
        # there, so a row merging after the data end is cut by the box only
        # if the box stops short of the data end (and likewise at the start).
        # Such rows are counted separately, as information.
        layer_dt = float(self._wdm.layer_dt)
        t0_grid = self._grid_t0_abs()
        box_lo = t0_grid + geom["n_start"] * layer_dt
        box_hi = box_lo + geom["Nt_keep"] * layer_dt
        act_lo = t0_grid + int(self._wdm.ind_min_t) * layer_dt
        act_hi = t0_grid + (int(self._wdm.ind_max_t) + 1) * layer_dt
        t_merge = self._stock_waveform_t0() + coords[valid_pos, _T_PLUNGE_COL]
        cut = (np.maximum(t_merge, act_lo) < box_lo) | (np.minimum(t_merge, act_hi) > box_hi)
        self._stats["outside_box"] += int(np.sum(cut))
        self._stats["outside_data"] += int(np.sum((t_merge < act_lo) | (t_merge >= act_hi)))
        if self.reject_outside_box:
            valid_pos = valid_pos[~cut]      # out / d_h / h_h keep the floor / NaN
        for device, pos in (self._split_rows(idx[valid_pos]) if valid_pos.size else ()):
            rows = valid_pos[pos]
            with self._device_contexts(device):
                adapter = self._adapter()
                self._apply_window(adapter, geom)
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

    def _generate(self, adapter, coords):
        """Batched windowed templates for ``coords``, ``t_plunge`` shifted to the snapped epoch."""
        params = np.array(coords, dtype=np.float64, copy=True)
        params[:, _T_PLUNGE_COL] -= float(self._gen_attr(adapter, "t_plunge_snap"))
        return adapter(*params.T, **self._gen_kwargs)

    def _on_container_box(self, tmpl):
        """``tmpl`` re-labelled onto a sub-box of the CONTAINERS' data settings.

        The adapter labels its output with a box cut from ITS settings (a
        device-local copy of ``general_info.domain_settings``); scoring and
        fill slice the containers' data, which carry the ``t0`` their settings
        had when the data were poured. In a stock erebor build the two can
        disagree (factory ``t0 = 0``; a GB comp build later sets
        ``domain_settings.t0 = data_t0`` in place), and
        :meth:`~lisatools.domains.WDMSettings.sub_box_slices` rightly refuses
        any ``t0`` disagreement. The coefficients do not depend on either
        label -- the segment is placed by the adapter's ``t0_abs`` -- so the
        SAME relative layers are re-cut from ``self._wdm``. Only ``t0`` is
        forgiven: the box is located by ``sub_box_slices`` itself against a
        copy of the template's settings carrying the containers' ``t0``, so a
        different wavelet grid or a box outside the data still raises.
        Downstream reads only the label's indices (the window/omega arrays it
        carries are never computed with), so the primary container's settings
        serve every shard."""
        box = tmpl.settings
        probe = WDMSettings(*box.args, **dict(box.kwargs, t0=self._wdm.t0))
        new_box = self._wdm.get_slice(self._wdm.sub_box_slices(probe))
        return WDMSignal(tmpl.arr, new_box)

    def _score_chunk(self, adapter, coords, idx, leaf):
        """Score one chunk: ``(ll, <r|h>, <h|h>)``, each ``(n,)``.

        A refused batch (``BatchNotLaunchable`` / ``WaveformDomainError``) goes
        through the per-row container path (warned once per leaf, counted in
        ``n_batch_fallbacks``, NaN records). A JAX / equinox runtime failure goes
        to :meth:`_score_rows_isolating_failures`. Any other exception raises.
        """
        t_gen = time.perf_counter()
        try:
            tmpl = self._generate(adapter, coords)
        except (BatchNotLaunchable, WaveformDomainError) as exc:
            self._stats["gen_s"] += time.perf_counter() - t_gen
            self.n_batch_fallbacks += 1
            self._stats["fallbacks"] += 1
            self.last_batch_error = exc
            if self._warned_fallback_leaf != leaf:
                self._warned_fallback_leaf = leaf
                logger.warning(
                    "[MBH_BATCH] leaf %d: batch refused (%s); scoring %d rows through "
                    "the per-row container path. Watch n_batch_fallbacks.",
                    leaf, exc, int(coords.shape[0]),
                )
            t_score = time.perf_counter()
            ll = np.real(np.asarray(
                self.compute_acs_like(coords, idx, **self.waveform_like_kwargs), dtype=float
            )).reshape(-1)
            self._stats["score_s"] += time.perf_counter() - t_score
            return ll, np.full(ll.shape, np.nan), np.full(ll.shape, np.nan)
        except Exception as exc:  # noqa: BLE001 -- narrowed on the next line
            if not _is_jax_runtime_failure(exc):
                raise
            # ONE row the waveform model cannot generate poisons the whole
            # vmapped chunk (phentax raises for the batch). Find it, score it
            # at the floor, keep the others. The container path is no fallback
            # here: it runs the same generator on the same row.
            self._stats["gen_s"] += time.perf_counter() - t_gen
            return self._score_rows_isolating_failures(adapter, coords, idx, leaf, exc)
        t_score = time.perf_counter()
        self._stats["gen_s"] += t_score - t_gen
        out = self._score_templates(tmpl, idx)
        self._stats["score_s"] += time.perf_counter() - t_score
        return out

    def _score_rows_isolating_failures(self, adapter, coords, idx, leaf, chunk_exc):
        """Score a chunk ROW BY ROW after its batched generation failed at runtime.

        Each row is generated alone: a row that fails again is scored at the
        log-likelihood floor (``-1e300``, the move's out-of-domain sentinel,
        so the proposal is rejected exactly as a prior violation would be),
        with ``<d|h>`` / ``<h|h>`` NaN, its WAVEFORM-basis parameters logged
        (the first :data:`_GEN_FAIL_LOG_CAP` per leaf) and the row counted in
        ``n_generate_failures`` and the ``[MBH_BATCH]`` ``gen_failed`` field.
        Rows that generate alone are scored normally. A row refused as a batch
        of one (``BatchNotLaunchable`` / ``WaveformDomainError``) takes the
        container path, as a refused chunk does. Any OTHER exception still
        raises: this isolates runtime failures of the waveform model, not bugs.

        Why the floor and not a retry or a repair: the failure seen in
        production (job 760, 2026-10-14) is phentax's t(f) bisection not
        converging for one hot-rung row -- a solve whose result lisatools
        discards (``t_min`` and ``t_ref`` are always given) but which phentax
        evaluates anyway under ``vmap``. The row's template is undefined as
        far as the model is concerned; rejecting it is the same policy the
        move applies to a ``WaveformDomainError``, and far cheaper than an
        MPI abort four iterations into a 1-year run.
        """
        n = int(coords.shape[0])
        ll = np.full(n, -1e300, dtype=float)
        dh = np.full(n, np.nan)
        hh = np.full(n, np.nan)
        self.last_generate_error = chunk_exc
        n_failed_here = 0
        for i in range(n):
            row = coords[i: i + 1]
            row_idx = idx[i: i + 1]
            t_gen = time.perf_counter()
            try:
                tmpl = self._generate(adapter, row)
            except (BatchNotLaunchable, WaveformDomainError):
                self._stats["gen_s"] += time.perf_counter() - t_gen
                t_score = time.perf_counter()
                ll[i] = float(np.real(np.asarray(
                    self.compute_acs_like(row, row_idx, **self.waveform_like_kwargs),
                    dtype=float,
                )).reshape(-1)[0])
                self._stats["score_s"] += time.perf_counter() - t_score
                continue
            except Exception as row_exc:  # noqa: BLE001 -- narrowed on the next line
                if not _is_jax_runtime_failure(row_exc):
                    raise
                self._stats["gen_s"] += time.perf_counter() - t_gen
                n_failed_here += 1
                self._note_generate_failure(leaf, coords[i], int(idx[i]), row_exc)
                continue
            t_score = time.perf_counter()
            self._stats["gen_s"] += t_score - t_gen
            ll_i, dh_i, hh_i = self._score_templates(tmpl, row_idx)
            self._stats["score_s"] += time.perf_counter() - t_score
            ll[i], dh[i], hh[i] = float(ll_i[0]), float(dh_i[0]), float(hh_i[0])
        if n_failed_here == 0:
            logger.warning(
                "[MBH_BATCH] leaf %d: a %d-row batch failed at runtime (%s) but every "
                "row generated alone; scored row by row. If this repeats, the batch "
                "itself is the problem, not a row.", leaf, n,
                str(chunk_exc).strip().splitlines()[0][:200] if str(chunk_exc).strip() else type(chunk_exc).__name__,
            )
        return ll, dh, hh

    def _note_generate_failure(self, leaf, row, walker, exc):
        """Count a row the generator could not produce; log its parameters (capped per leaf)."""
        self.n_generate_failures += 1
        self._stats["gen_failed"] += 1
        self.last_generate_error = exc
        seen = int(self._gen_fail_logged.get(leaf, 0))
        self._gen_fail_logged[leaf] = seen + 1
        text = str(exc).strip()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # the informative line of a wrapped equinox error is the LAST one
        # ("_EquinoxRuntimeError: The maximum number of steps was reached ...")
        reason = next(
            (ln.strip() for ln in reversed(lines) if "Error" in ln or "error" in ln),
            lines[0].strip() if lines else type(exc).__name__,
        )
        if seen < _GEN_FAIL_LOG_CAP:
            logger.warning(
                "[MBH_BATCH] leaf %d walker %d: waveform generation FAILED at runtime "
                "for one row; scored at the log-likelihood floor (-1e300, rejected). "
                "waveform-basis params (m1 m2 s1z s2z dist phi_ref iota psi alpha "
                "delta t_plunge) = %s. %s", leaf, walker,
                np.array2string(np.asarray(row, dtype=float), precision=8,
                                max_line_width=600, separator=", "),
                reason[:300],
            )
        elif seen == _GEN_FAIL_LOG_CAP:
            logger.warning(
                "[MBH_BATCH] leaf %d: %d generator failures logged with parameters; "
                "further ones are counted only (gen_failed in the [MBH_BATCH] line, "
                "n_generate_failures on the move).", leaf, _GEN_FAIL_LOG_CAP,
            )

    def _score_templates(self, tmpl, idx):
        """``(ll, <r|h>, <h|h>)`` for the template stack ``tmpl`` against walkers ``idx``.

        The stack is re-labelled onto the containers' box, then each distinct
        walker gets one batched ``inner_product`` pair against its residual and
        PSD sliced to that box; one host pull per chunk. ``ll`` adds the
        exposed-residual offset of each row's walker.
        """
        like_kw = {
            k: v for k, v in (self.waveform_like_kwargs or {}).items()
            if k not in ("psd", "complex", "include_psd_info")
        }
        tmpl = self._on_container_box(tmpl)
        box = tmpl.settings
        acs_flat = self.acs.acs.flatten()
        arr = tmpl.arr if tmpl.is_batched else tmpl.arr[None]
        xp = get_array_module(arr)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        n = int(idx.size)
        # One BATCHED inner_product pair per distinct walker in the chunk:
        # the walker's rows go in as one template stack (leading source axis)
        # against its sliced residual / PSD. A batched call returns one value
        # per source and stays on device (inner_product only ``.item()``s the
        # UNBATCHED case -- a sync per call, which the per-row loop paid twice
        # per row). <h|h> is the batched stack against itself (equal nbatch:
        # elementwise product, per-source reduction). Everything is gathered
        # on device and pulled ONCE per chunk.
        cache = {}
        order = []
        dh_dev = []
        hh_dev = []
        for g in np.unique(idx):
            rows_g = np.where(idx == g)[0]
            h_g = WDMSignal(arr[rows_g], box)
            if g not in cache:
                r_box, _, s_box = acs_flat[int(g)]._slice_to_template(h_g)
                cache[g] = (r_box, s_box)
            r_box, s_box = cache[g]
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
    # expose / fold
    # ------------------------------------------------------------------

    def _apply_cold_chain_sources(self, coords, sign):
        """Expose (``sign=+1``, ``r += h``) or fold back (``sign=-1``) the leaf's cold rows.

        One row per walker (row position = walker). Batched windowed templates
        in the leaf's window (rebuilt only on expose), applied with
        ``acs.signal_operation``; a refused chunk falls back to the containers'
        per-row ``apply_signal_from_params``. Non-finite rows are skipped.
        ``MBH_BATCHED_FILL`` other than ``1`` uses the base's dense per-row pass.
        One ``[MBH_FILL]`` line per call.
        """
        what = "EXPOSE (r += h)" if sign > 0 else "FOLD-BACK (r -= h)"
        if os.environ.get("MBH_BATCHED_FILL", "1").strip() != "1":
            logger.info(
                "[MBH_FILL] %s via the DENSE per-row path (MBH_BATCHED_FILL=0): %d rows",
                what, int(np.shape(coords)[0]),
            )
            return super()._apply_cold_chain_sources(coords, sign)
        coords_np = np.atleast_2d(np.asarray(asnumpy(coords), dtype=np.float64))
        leaf = int(self._current_leaf)
        geom = self._leaf_window(leaf, coords_np, allow_rebuild=(sign > 0))
        idx_all = np.arange(coords_np.shape[0])
        valid = np.all(np.isfinite(coords_np), axis=1)
        if not np.any(valid):
            return
        t_start = time.perf_counter()
        n_fallback = 0
        valid_idx = idx_all[valid]
        for device, pos in self._split_rows(valid_idx):
            rows = valid_idx[pos]
            with self._device_contexts(device):
                adapter = self._adapter()
                self._apply_window(adapter, geom)
                for lo in range(0, rows.size, self.batch_max_size):
                    sel = rows[lo: lo + self.batch_max_size]
                    try:
                        tmpl = self._generate(adapter, coords_np[sel])
                    except (BatchNotLaunchable, WaveformDomainError) as exc:
                        n_fallback += 1
                        self.last_batch_error = exc
                        self.acs.apply_signal_from_params(
                            sign, {self.branch_name: coords_np[sel]}, index=sel,
                            waveform_kwargs=self._branch_waveform_kwargs(),
                            signal_gen_resolver=self._resolve_signal_gen_override,
                            apply_transform=False, domain_error="skip",
                        )
                        continue
                    self.acs.signal_operation(
                        sign, self._on_container_box(tmpl), data_index=sel
                    )
        logger.info(
            "[MBH_FILL] %s via batched windowed templates: %d rows (%d skipped, "
            "%d chunk fallbacks) in %.2f s", what, int(valid.sum()),
            int((~valid).sum()), n_fallback, time.perf_counter() - t_start,
        )

    # ------------------------------------------------------------------
    # records, checks, telemetry
    # ------------------------------------------------------------------

    def _record_leaf_inner_products(self, new_state, add_coords_in, leaf):
        """Record cold-chain ``<d|h>``, ``<h|h>`` from the batched scorer (free)."""
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
        """Built-in fast-vs-slow A/B at MATCHING convention (see the SOBBH move).

        GATES ON THE COLD RUNG ONLY (temperature index 0 of ``prev_logl``'s
        ``(ntemps, nwalkers)`` shape). Hot rungs sit far from the posterior,
        where any 1e-8 template difference is amplified by a huge ``<r|delta>``
        (mojito id 17: 76 nats between two EXACT stock variants at logL ~ -1e5);
        their max|diff| is reported in the same log line, never gating."""
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
        if max_abs <= self.check_ll_tol:
            return
        spread = float(diff.max() - diff.min())
        hot = both[1:]
        hot_txt = (
            f"{float(np.abs(prev2[1:][hot] - acs2[1:][hot]).max()):.6e} over "
            f"{int(hot.sum())} points" if np.any(hot) else "n/a"
        )
        msg = (
            f"{self.branch_name} leaf {leaf}: batched windowed fast path vs slow "
            f"container path disagree beyond tol={self.check_ll_tol} on the COLD "
            f"rung: max|diff|={max_abs:.6e}, spread={spread:.6e} over "
            f"{int(cold.sum())} points (hot rungs, not gating: max|diff|={hot_txt}). "
            "A small spread at high SNR is sub-transform truncation "
            "(widen MBH_WINDOW_PAD_DAYS / raise MBH_CHECK_LL_TOL); a large spread "
            "means a window, snap or residual bug."
        )
        if self.check_ll_mode == "strict":
            raise ValueError(msg)
        logger.warning(msg)

    def _flush_stats(self):
        """Log the finished leaf's ``[MBH_BATCH]`` line (if it scored rows) and reset."""
        st = self._stats
        if st["rows"]:
            logger.info(
                "[MBH_BATCH] leaf %s: %d rows in %d chunks, %.1f s (%.3f s/row; "
                "generate=%.2f s, score=%.2f s), %d fallbacks, outside_box=%d "
                "(rows whose in-data merger falls outside the kept box%s), "
                "outside_data=%d (rows merging outside the data's active time "
                "box: expected for a source merging near/after a data edge)"
                "%s",
                st["leaf"], st["rows"], st["chunks"], st["seconds"],
                st["seconds"] / st["rows"], st["gen_s"], st["score_s"],
                st["fallbacks"], st["outside_box"],
                (": rejected at the floor" if getattr(self, "reject_outside_box", True)
                 else ": scored truncated"),
                st["outside_data"],
                (f", gen_failed={st['gen_failed']} (rows the waveform model could not "
                 "generate at runtime; scored at the floor)") if st["gen_failed"] else "",
            )
        self._stats = self._new_stats()
