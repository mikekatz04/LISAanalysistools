"""EMRI templates built directly in the WDM domain (lookup table + plunge chunk).

The production EMRI template of the 9mo / 12mo runs (``EMRI_LIKELIHOOD=direct``, reached
through :class:`~lisatools.sources.emri.direct_signal_gen.EMRIDirectWDMSignalGen`; see
``docs/emri-direct-wdm.md``). :class:`EMRIDirectWDM` assembles a template on the run's FULL
wavelet grid without any dense time-domain transform:

1. ONE FEW call per row on the integrator's own knots (``feed="knots"``) gives the kept modes
   and the sparse holder; the trajectory's DOPR853 dense output gives exact harmonic phases.
2. The TDI response: with ``response="dense"`` (production)
   :class:`~lisatools.response.tdionfly.TDDenseTDIonTheFly` on a SPARSE time grid
   (``response_grid="sparse"``: the knots plus a point every ``sparse_dt``), wrapped as
   :class:`ExactPhaseTDIOutput` (exact carrier, splined slow residual); ``response="spline"``
   uses :class:`~lisatools.sources.emri.emritdionfly.EMRITDIonFly`.
3. Per harmonic and channel, the n_ref lookup table
   (:class:`~lisatools.domains.WDMLookupTable`) maps the local ``(amp, phase, f, fdot)`` at each
   pixel centre to the WDM coefficients of ``2 num_m_layers + 1`` frequency layers around the
   carrier (the fused C++/CUDA kernel, :func:`lookup_sum_kernel`, with ``lookup="kernel"``).
4. A harmonic whose local-quadratic model fails (curvature or ``|fdot|`` beyond the table
   axis, :func:`handoff_pixel`) hands off to the plunge chunk: the exact dense TD of the
   channels over even-start chunks of ``Nt_sub`` time layers, transformed with
   :func:`lisatools.wdm_het.wdm_chunk_of_td`.

Indexing conventions: pixel (time layer) ``n`` of the full grid is centred at the absolute time
``data_t0 + n * layer_dt``; frequency layer ``m`` is the global layer index ``0 .. Nf - 1``
(``m * layer_df``); the first and last ``pixel_edge`` time layers stay zero. FEW's clock is
``t - t_start`` (``t_start`` = the catalogue reference epoch REF in mojito mode). Templates
are ``(nch, Nf, Nt)`` (one) or ``(n_rows, nch, n_m, Nt)`` holding layers ``[m_lo, m_hi)`` of
``f_band`` (batches); channels are the TDI ``X, Y, Z`` of ``tdi_config``.

Environment knobs read here (explicit arguments win unless stated): ``EMRI_DIRECT_LOOKUP``
(``kernel`` / ``python``, default ``kernel``), ``EMRI_DIRECT_RESPONSE_GRID`` (``sparse`` /
``pixels``, default ``sparse``), ``EMRI_DIRECT_SPARSE_DT`` (43200 s),
``EMRI_DIRECT_NUM_M_LAYERS`` (2; this one OVERRIDES the ``num_m_layers`` argument),
``EMRI_DIRECT_PLUNGE_TAPER_S`` (120 s; no argument, always read),
``EMRI_DIRECT_HANDOFF_BOUND`` (``1``), ``EMRI_DIRECT_TAIL_BLOCK_ELEMS`` and
``EMRI_DIRECT_MODE_BLOCK_ELEMS`` (2e7), ``EMRI_DIRECT_LOOKUP_CHUNK`` (2e6), ``EMRI_TRACER``
(``analytic``), ``EMRI_TRACER_H_FDOT`` (300 s), ``EMRI_TRACER_MAX_ELEMS`` (6e7). The response
kind comes from the stock knob ``EMRI_DIRECT_RESPONSE`` (``dense``) through the getter.

Harmonic tracks follow FEW's own conventions (few/waveform/base.py:357-364 and
few/summation/directmodesum.py):

* ``xI0 < 0`` is first mapped to ``(a, xI0) = (-a, +1)`` as FEW does; then for ``a > 0``
  the azimuthal phase is multiplied by ``sign(xI0)`` (a no-op after that mapping);
* for ``integrate_backwards`` the knot-end offset ``Phi[-1] + Phi[0]`` is added;
* the ``+m`` harmonic is ``Y_lm A e^{-i Phi}``; its ``-m`` partner (holder built with
  ``include_minus_mkn=True``, ``ylms`` of length ``2 * nmodes``) is
  ``(-1)^l Y_{l,-m} conj(A) e^{+i Phi}``, i.e. a track with ``amp = (-1)^l Y_{l,-m}
  conj(A)`` and ``phase, f, fdot, fddot`` all negated.

Frequencies and their derivatives come ANALYTICALLY from the integrator's derivative
spline (orders 1, 2, 3), never from differencing an unwrapped phase.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np

from ...utils.constants import YRSID_SI
from scipy.interpolate import CubicSpline


@dataclasses.dataclass
class HarmonicTrack:
    """One harmonic at pixel-centre times: h = amp * exp(-1j * phase).

    Attributes:
        lmkn (tuple): ``(l, m, k, n)`` mode indices (negated ``m, k, n`` for a -m partner).
        t (np.ndarray): Times [s], FEW clock, ``(P,)``.
        amp (np.ndarray): Complex amplitude ``Y_lm A`` (or the -m partner's), ``(P,)``.
        phase (np.ndarray): ``m Phi_phi + k Phi_theta + n Phi_r`` [rad], ``(P,)``.
        f (np.ndarray | None): Source-frame harmonic frequency [Hz] (``None`` with
            ``phase_only``).
        fdot (np.ndarray | None): Its first time derivative [Hz/s].
        fddot (np.ndarray | None): Its second time derivative [Hz/s^2].
    """

    lmkn: tuple
    t: np.ndarray
    amp: np.ndarray
    phase: np.ndarray
    f: np.ndarray
    fdot: np.ndarray
    fddot: np.ndarray


def _phase_columns(integrator, t, order):
    """The three fundamental phases ``(Phi_phi, Phi_theta, Phi_r)`` (``order = 0``) or their
    ``order``-th time derivative at times ``t`` from the integrator splines: ``(len(t), 3)``."""
    if order == 0:
        cols = np.asarray(integrator.eval_integrator_spline(t))[:, 3:6]
    else:
        cols = np.asarray(integrator.eval_integrator_derivative_spline(t, order=order))[:, 3:6]
    return np.array(cols, dtype=float, copy=True)


def harmonic_tracks_from_holder(holder, integrator, t_pixels, *, a, xI0, phase_only=False):
    """Per-harmonic tracks at ``t_pixels`` [s, trajectory clock] from a FEW sparse holder.

    Vectorised over modes (one matrix product per derivative order). The amplitudes are cubic
    splines of the holder's Teukolsky amplitudes at its knots times ``Y_lm``; phases and their
    derivatives come from the integrator (the conventions in the module docstring).

    Args:
        holder: FEW ``SparseInfoHolder`` (``return_sparse_holder=True``), host copy.
        integrator: the trajectory integrator (``...inspiral_generator.inspiral_generator``)
            exposing ``eval_integrator_spline`` / ``eval_integrator_derivative_spline``; it
            must still hold ``holder``'s trajectory.
        t_pixels (np.ndarray): evaluation times, same clock as ``holder.t_arr``.
        a (float): spin (retrograde convention: ``xI0 < 0`` maps to ``(-a, +1)``).
        xI0 (float): inclination cosine.
        phase_only (bool, optional): skip the frequency derivatives (``f``, ``fdot``,
            ``fddot`` are None): the response feed needs amplitude and phase only.

    Returns:
        list[HarmonicTrack]: The ``m >= 0`` modes in holder order, then the -m partners of the
        ``m != 0`` modes (when the holder carries ``2 * nmodes`` Ylms).
    """
    t_pixels = np.asarray(t_pixels, dtype=float)
    if xI0 < 0:   # FEW's internal convention first (few/waveform/base.py:208-212)
        a, xI0 = -a, -xI0
    orders = (0,) if phase_only else (0, 1, 2, 3)
    P = [_phase_columns(integrator, t_pixels, k) for k in orders]
    if a > 0:
        for arr in P:
            arr[:, 0] *= np.sign(xI0)
    if holder.integrate_backwards:
        knots = _phase_columns(integrator, np.asarray(holder.t_arr, dtype=float), 0)
        if a > 0:
            knots[:, 0] *= np.sign(xI0)
        P[0] = P[0] + (knots[-1] + knots[0])[None, :]

    t_k = np.asarray(holder.t_arr, dtype=float)
    teuk = np.asarray(holder.teuk_modes)
    nm = teuk.shape[1]
    A = CubicSpline(t_k, teuk.real, axis=0)(t_pixels) + 1j * CubicSpline(t_k, teuk.imag, axis=0)(t_pixels)
    ylms = np.asarray(holder.ylms)
    with_minus = ylms.shape[0] == 2 * nm
    ls, ms, ks, ns = (np.asarray(x) for x in (holder.ls, holder.ms, holder.ks, holder.ns))
    Wm = np.stack([ms, ks, ns]).astype(float)                       # (3, nm)
    D = [Pk @ Wm for Pk in P]                                       # each (T, nm)
    ph = D[0]
    f, fd, fdd = ((D[k] / (2 * np.pi)) for k in (1, 2, 3)) if not phase_only else (None, None, None)
    amp = A * ylms[None, :nm]
    tracks = []
    for j in range(nm):
        lmkn = (int(ls[j]), int(ms[j]), int(ks[j]), int(ns[j]))
        tracks.append(HarmonicTrack(lmkn, t_pixels, amp[:, j], ph[:, j],
                                    None if f is None else f[:, j], None if fd is None else fd[:, j],
                                    None if fdd is None else fdd[:, j]))
    minus = []
    if with_minus:
        jm = np.flatnonzero(ms != 0)
        amp_m = ((-1.0) ** ls[jm])[None, :] * ylms[None, nm + jm] * np.conj(A[:, jm])
        for c, j in enumerate(jm):
            minus.append(HarmonicTrack((int(ls[j]), -int(ms[j]), -int(ks[j]), -int(ns[j])), t_pixels,
                                       amp_m[:, c], -ph[:, j], None if f is None else -f[:, j],
                                       None if fd is None else -fd[:, j], None if fdd is None else -fdd[:, j]))
    return tracks + minus


# ----------------------------------------------------------------------
# Handoff from the lookup to the plunge chunk
# ----------------------------------------------------------------------

#: Half-support of the WDM wavelet in units of layer_dt used by the curvature trigger.
#: Calibrated in Task A6 (scripts/emri/emri_single_harmonic_gate.py): the exact local-chirp
#: model error (1.9e-4 median over the inspiral) passes the table-interpolation error
#: (~3-5e-3) near 0.9 t_plunge, where (pi/3)|fddot| layer_dt^3 ~ 1.2e-3 rad; a 0.1 rad
#: trigger there needs tau ~ 4 layer_dt (the WDM wavelet's tails are wide).
WDM_HALF_SUPPORT_LAYERS = 4.0
def handoff_pixel(track, layer_dt, layer_df, fdot_axis_max, tol_rad=0.1):
    """First pixel index where the local-quadratic lookup stops being valid.

    Trips on the cubic phase term ``(pi/3) |fddot| tau^3 > tol_rad`` (tau = the wavelet
    half-support, :data:`WDM_HALF_SUPPORT_LAYERS` ``* layer_dt``) OR on ``|fdot|`` leaving the
    table's fdot axis, whichever comes first. Never on fdot alone: the dominant harmonic can
    stay inside the fdot axis and still fail on curvature.

    Args:
        track (HarmonicTrack): The harmonic (needs ``fdot`` and ``fddot``).
        layer_dt (float): WDM layer duration [s].
        layer_df (float): WDM layer bandwidth [Hz]; unused by the test.
        fdot_axis_max (float): Largest ``|fdot|`` of the table axis [Hz/s].
        tol_rad (float, optional): Cubic-phase tolerance [rad]. Default ``0.1``.

    Returns:
        int: Index into ``track.t`` of the first tripping sample, or ``len(track.t)`` if it
        never trips.
    """
    tau = WDM_HALF_SUPPORT_LAYERS * layer_dt
    cubic = (np.pi / 3.0) * np.abs(np.asarray(track.fddot)) * tau ** 3
    trip = (cubic > tol_rad) | (np.abs(np.asarray(track.fdot)) > fdot_axis_max)
    idx = np.flatnonzero(trip)
    return int(idx[0]) if idx.size else int(np.asarray(track.t).size)


# TODO check: keep/remove/document -- not reached by the 9mo/12mo runs (accumulate_harmonic_batch plans its chunks with tail_chunk_plan); only caller is tests/test_wdm_chunk_splice.py
def plunge_chunk_wdm(td_tail_fn, n_h, Nt, Nf, dt, Nt_sub=128, n_end=None, ind_max_t=None, backend="cpu"):
    """WDM of the plunge tail ``[n_h, n_end)`` from ONE even-start chunk.

    Args:
        td_tail_fn: ``(start_sample, n_samples) -> (nch, n_samples)`` dense TD of the
            signal over the chunk window only (no window/taper).
        n_h: first pixel handed off to the chunk; n_end: one past the last pixel to
            fill (default ``Nt``).
        ind_max_t: the production time crop's last kept pixel + 1; a chunk reaching
            past it would carry a tapered tail, so this raises instead.
    Returns ``(chunk, n0, keep_lo, keep_hi)`` for :func:`lisatools.wdm_het.splice_chunk`.
    """
    from ...wdm_het import chunk_start_for_pixels, wdm_chunk_of_td

    n_end = int(Nt if n_end is None else n_end)
    # Edge contamination decays algebraically from each chunk edge (Nt_sub=128: 0.2 at
    # the edge, 5e-5 at 16 px, 6e-6 at 24 px; a Tukey does not help) -> discard a
    # quarter of the chunk on each side (tests/test_wdm_chunk_splice.py measures it).
    n_pad = Nt_sub // 4
    n0 = chunk_start_for_pixels(n_h, n_end, Nt, Nt_sub, n_pad)
    if ind_max_t is not None and n0 + Nt_sub > int(ind_max_t):
        raise ValueError(
            f"plunge chunk [{n0}, {n0 + Nt_sub}) runs into the production time crop "
            f"(ind_max_t={ind_max_t}); it would include tapered/cropped samples")
    td = td_tail_fn(n0 * int(Nf), int(Nf) * int(Nt_sub))
    chunk = wdm_chunk_of_td(td, 0, Nf, Nt_sub, dt, backend=backend)
    return chunk, n0, int(n_h) - n0, n_end - n0


# ----------------------------------------------------------------------
# Per-channel tracer from the TDI-on-the-fly output, and the full assembly
# ----------------------------------------------------------------------

def _spline_vals(out, t, subs):
    """``out.eval_spline_vals(t)`` (``(amp, tdi_phase, phase_ref)``), for the subset ``subs``
    when given."""
    if subs is None:
        return out.eval_spline_vals(t)
    return out.eval_spline_vals(t, subs=subs)


def _total_phase(out, t, subs=None):
    """Channel phase ``(S, nch, len(t))`` of a response output at times ``t``."""
    if hasattr(out, "eval_phase"):
        return out.eval_phase(t) if subs is None else out.eval_phase(t, subs=subs)
    _, tph, pref = _spline_vals(out, t, subs)
    xp = getattr(out, "xp", np)
    return xp.asarray(tph) + xp.asarray(pref)[:, None, :]


def tracer_from_tof_output(out, t_pixels, h=30.0, h_fdot=None, max_elems=None, subs=None, method=None):
    """Per-sub, per-channel ``(amp, phase, f, fdot)`` at ``t_pixels`` [absolute s].

    The channel signal is ``Re[amp exp(-i phase)]`` with ``phase = tdi_phase + phase_ref``
    (``TDTDIOutput.eval_tdi``; an :class:`ExactPhaseTDIOutput` adds its exact carrier instead).
    ``f`` and ``fdot`` are the first two derivatives of that CONTINUOUS phase over 2 pi, so they
    carry the Doppler shift of the channel (the source-frame f is off by ~1e-4 f, a ~1% pixel
    error). ``method``: ``"analytic"`` (default for an output with ``eval_tracer``, i.e.
    :class:`SplinedTDIOutput` / :class:`ExactPhaseTDIOutput`; env ``EMRI_TRACER``) differentiates
    the splines and the dense-output carrier exactly, one evaluation per pixel; ``"stencil"``
    takes central differences (steps ``h``, ``h_fdot``) of the phase (any output). A
    negative-frequency sub (a -m partner) is mirrored to positive frequency: cos is even, so
    ``phase -> -phase``, ``f -> -f``, ``fdot -> -fdot``.

    ``subs`` (host integer array, optional): only these harmonics (rows of the output), for a
    mode-batched caller (needs an output with subset evaluation, :class:`SplinedTDIOutput`).

    Memory: evaluated in blocks of pixels so no temporary exceeds ``max_elems`` (env
    ``EMRI_TRACER_MAX_ELEMS``, default 6e7) elements per ``(S, nch, block)`` array, and the
    amplitude is read at the pixel centres only.

    Args:
        out: Response output (:class:`SplinedTDIOutput`, :class:`ExactPhaseTDIOutput` or a
            ``TDTDIOutput`` with splines).
        t_pixels: Pixel-centre times [s, absolute], ``(P,)``.
        h (float, optional): Stencil step [s] for ``f``. Default ``30.0``.
        h_fdot (float, optional): Stencil step [s] for ``fdot``; ``None`` reads
            ``EMRI_TRACER_H_FDOT`` (300).
        max_elems (float, optional): Block budget; ``None`` reads ``EMRI_TRACER_MAX_ELEMS``.
        subs (np.ndarray, optional): Rows (harmonics) to evaluate; ``None`` = all.
        method (str, optional): ``"analytic"`` or ``"stencil"``; ``None`` reads
            ``EMRI_TRACER`` (``"analytic"``).

    Returns:
        tuple: ``(amp, phase, f, fdot)``, each ``(num_sub, nch, P)`` on ``out.xp``.

    Raises:
        ValueError: For an unknown ``method``.
    """
    xp = getattr(out, "xp", np)
    t = xp.asarray(t_pixels, dtype=float)
    hd = float(os.environ.get("EMRI_TRACER_H_FDOT", "300")) if h_fdot is None else float(h_fdot)
    P = int(t.size)
    try:
        S = int(np.size(subs)) if subs is not None else int(np.shape(out.x)[0])
        nch = int(getattr(out, "nch", 0) or np.shape(out.tdi_amp)[1])
    except AttributeError:                                  # a minimal output: one pass
        S = nch = 1
    budget = float(os.environ.get("EMRI_TRACER_MAX_ELEMS", 6e7)) if max_elems is None else float(max_elems)
    block = max(1, int(budget // max(1, S * nch)))
    if method is None:
        method = os.environ.get("EMRI_TRACER", "analytic")
    if method not in ("analytic", "stencil"):
        raise ValueError("tracer method must be 'analytic' or 'stencil'")
    analytic = method == "analytic" and hasattr(out, "eval_tracer")
    parts = []
    for p0 in range(0, P, block):
        tb = t[p0:p0 + block]
        if analytic:
            parts.append(out.eval_tracer(tb, subs=subs))
            continue
        amp, tph, pref = _spline_vals(out, tb, subs)
        amp = xp.asarray(amp)
        ph0 = xp.asarray(tph) + xp.asarray(pref)[:, None, :]
        del tph, pref
        f = (_total_phase(out, tb + h, subs) - _total_phase(out, tb - h, subs)) / (2 * h) / (2 * np.pi)
        fdot = (_total_phase(out, tb + hd, subs) - 2 * ph0 + _total_phase(out, tb - hd, subs)) / hd ** 2 / (2 * np.pi)
        #                                     phase ~1e5 rad: a short step turns roundoff into fdot noise
        neg = f < 0
        parts.append((amp, xp.where(neg, -ph0, ph0), xp.where(neg, -f, f), xp.where(neg, -fdot, fdot)))
        del ph0, f, fdot, neg
    if len(parts) == 1:
        return parts[0]
    return tuple(xp.concatenate([p[k] for p in parts], axis=-1) for k in range(4))


def _scatter_add(xp, acc, idx, vals):
    """acc[idx] += vals with repeated indices accumulated (numpy add.at / cupyx scatter_add)."""
    if xp is np:
        np.add.at(acc, idx, vals)
    else:
        import cupyx
        cupyx.scatter_add(acc, idx, vals)


def accumulate_harmonic_batch(acc, table, tracks, tracer, n_ok, tail_td, *, Nf, Nt, dt, layer_dt,
                              layer_df, t0, Nt_sub=128, num_m_layers=2, fdot_axis_max=np.inf,
                              pixel_edge=8, backend="cpu", sub_row=None, lookup_chunk=None, m_lo=0,
                              num_sub=None, lookup_kernel=None, common_handoff=False):
    """Vectorised :func:`_accumulate_harmonic_batch_loop`: ONE table evaluation for every
    (sub, channel, pixel) of the batch and one scatter-add, on the array module of ``acc``
    (numpy or cupy). Same arguments, same result up to summation order. Adds in place into
    ``acc``: the lookup part over the pixels before each sub's handoff (skipping pixels whose
    channel ``|fdot|`` leaves the table or whose ``f <= 2 layer_df``), then the plunge chunks
    (``tail_chunk_plan`` windows ending a quarter chunk past the last tracked pixel, capped at
    ``Nt - pixel_edge``; ``wdm_chunk_of_td`` of ``tail_td`` per window and keep range).

    The common arguments are those of :func:`_accumulate_harmonic_batch_loop`: ``acc``
    (``(nch, n_m, Nt)``, or ``(n_templates, nch, n_m, Nt)`` with ``sub_row``), ``table``
    (:class:`~lisatools.domains.WDMLookupTable`), ``tracer``, ``n_ok`` (ascending absolute
    pixel indices of the tracer samples), ``tail_td`` (``ts -> (num_sub, nch, ts.size)``, or
    ``(nch, ts.size)`` when ``tail_td.summed``), the grid ``Nf``, ``Nt``, ``dt``,
    ``layer_dt``, ``layer_df``, ``t0`` (absolute time of pixel 0), ``Nt_sub``,
    ``num_m_layers``, ``fdot_axis_max``, ``pixel_edge`` and ``backend``.

    ``tracks``: the subs' :class:`HarmonicTrack` (their handoff pixels), or ``None`` when no
    sub hands off inside ``n_ok`` (then ``num_sub`` or the tracer gives the number of subs).
    ``sub_row`` (optional, length num_sub): the template each sub belongs to; ``acc`` is then
    ``(n_templates, nch, n_m, Nt)`` and many templates accumulate in one call (no plunge
    chunks allowed in that mode).
    ``m_lo``: ``acc`` holds layers ``m_lo .. m_lo + n_m - 1`` (``n_m = acc.shape[-2]``) only;
    the rest is dropped.
    ``lookup_chunk``: max (sub, channel, pixel) entries per table call (bounds the lookup's
    temporaries: ~6 arrays of (entries, 2 num_m_layers + 1) doubles); default from env
    EMRI_DIRECT_LOOKUP_CHUNK or 2,000,000.
    ``lookup_kernel``: ``n_stop -> stats`` doing the lookup in the fused kernel
    (:func:`lookup_sum_kernel`; ``n_stop`` = each sub's first pixel past its handoff); the
    tracer is then unused (pass None). The plunge chunks stay here.
    ``common_handoff``: every sub hands off at the EARLIEST sub's handoff pixel (the chunk's
    time series is exact, so earlier is never worse), so each chunk window has one keep range
    and one transform. Implied by a summed ``tail_td`` (``tail_td.summed``: it returns the sum
    over all subs, ``(nch, n)``, which only one shared keep range can use).

    Returns:
        dict: ``lookup_pixels``, ``chunk_pixels``, ``dropped_pixels`` (plus the kernel's
        counters with ``lookup_kernel``).

    Raises:
        NotImplementedError: If a plunge chunk is needed with ``sub_row`` (many templates).
    """
    from ...wdm_het import tail_chunk_plan, wdm_chunk_of_td
    from ...utils.utility import get_array_module

    xp = get_array_module(acc)
    n_ok_h = np.asarray(n_ok.get() if hasattr(n_ok, "get") else n_ok)
    P = int(n_ok_h.size)
    if tracks is not None:
        S = len(tracks)
    elif num_sub is not None:
        S = int(num_sub)
    else:
        S = int(np.shape(tracer[0])[0])
    summed = bool(getattr(tail_td, "summed", False))
    if tracks is None:
        k_h = np.full(S, P)
    else:
        k_h = np.array([handoff_pixel(tr, layer_dt, layer_df, fdot_axis_max) for tr in tracks], dtype=int)
        if (common_handoff or summed) and k_h.size:
            k_h[:] = k_h.min()
    n_m = int(acc.shape[-2])
    stats = dict(lookup_pixels=0, chunk_pixels=0, dropped_pixels=0)
    if lookup_kernel is not None:
        n_stop = np.where(k_h < P, n_ok_h[np.minimum(k_h, P - 1)], n_ok_h[-1] + 1) if P else np.zeros(S, int)
        stats.update(lookup_kernel(n_stop))
    else:
        amp, phase, f, fdot = (xp.asarray(a) for a in tracer)
        nch = amp.shape[1]
        n_ok_x = xp.asarray(n_ok_h)
        before = xp.arange(P)[None, :] < xp.asarray(k_h)[:, None]                       # (S, P)
        sel = before[:, None, :] & (xp.abs(fdot) <= fdot_axis_max) & (f > 2 * layer_df)  # (S, C, P)
        stats["dropped_pixels"] = int(nch * int(before.sum()) - int(sel.sum()))
        s_all, c_all, p_all = xp.nonzero(sel)
        chunk = int(lookup_chunk or os.environ.get("EMRI_DIRECT_LOOKUP_CHUNK", 2_000_000))
        sub_row_x = None if sub_row is None else xp.asarray(sub_row)
        for j0 in range(0, int(s_all.size), chunk):
            s_i, c_i, p_i = s_all[j0:j0 + chunk], c_all[j0:j0 + chunk], p_all[j0:j0 + chunk]
            co, mm = table.get_wdm_coeffs(amp[s_i, c_i, p_i], phase[s_i, c_i, p_i], f[s_i, c_i, p_i],
                                          fdot[s_i, c_i, p_i], n_ok_x[p_i],
                                          num_m_layers=num_m_layers, out_of_support="zero")
            co, mm = xp.asarray(co), xp.asarray(mm)
            n_e = n_ok_x[p_i]
            r_i = None if sub_row_x is None else sub_row_x[s_i]
            for c in range(co.shape[1]):
                good = (mm[:, c] >= m_lo) & (mm[:, c] < m_lo + n_m)
                idx = (c_i[good], mm[good, c] - m_lo, n_e[good])
                if r_i is not None:
                    idx = (r_i[good],) + idx
                _scatter_add(xp, acc, idx, co[good, c])
            del co, mm
        stats["lookup_pixels"] = int(s_all.size)
    windows = {}
    for s in range(S):
        if k_h[s] < n_ok_h.size:
            n_h = int(n_ok_h[k_h[s]])
            n_end = min(int(n_ok_h[-1]) + 1 + Nt_sub // 4, Nt - pixel_edge)
            for n0, klo, khi in tail_chunk_plan(n_h, n_end, Nt, Nt_sub):
                windows.setdefault(n0, []).append((s, klo, khi))
    if windows and sub_row is not None:
        raise NotImplementedError("plunge chunks are not batched across templates; build that template alone")
    for n0, items in windows.items():
        ts = t0 + (n0 * Nf + xp.arange(Nf * Nt_sub)) * dt
        td_all = xp.asarray(tail_td(ts))
        groups = {}
        for s, klo, khi in items:                  # the transform is linear: one per keep range
            groups.setdefault((klo, khi), []).append(s)
        for (klo, khi), subs in groups.items():
            if summed:
                td = td_all
            else:
                td = td_all[subs[0]] if len(subs) == 1 else td_all[xp.asarray(np.asarray(subs))].sum(axis=0)
            chunk = xp.asarray(wdm_chunk_of_td(td, 0, Nf, Nt_sub, dt, backend=backend))
            hi = min(m_lo + n_m, chunk.shape[-2])                     # chunk rows = global layers
            acc[:, :hi - m_lo, n0 + klo:n0 + khi] += chunk[:, m_lo:hi, klo:khi]
            stats["chunk_pixels"] += (khi - klo) * len(subs)
    return stats


def table_kernel_view(table):
    """The lookup table as the fused kernel reads it (cached on the table), or ``None`` when the
    kernel cannot reproduce its evaluation: needs ``INTERP_METHOD == "spline"`` (the prefiltered
    uniform cubic B-spline, :class:`lisatools.domains._UniformCubicSpline`), an ``n_ref`` build
    and the ``quarter_turn`` basis cycle. Coefficients: the cos table (``Re(table_cx)``) and the
    sin table with the build's ``(-1)^block`` bake undone (``WDMLookupTable._sin_unbaked_coeffs``).

    Args:
        table (WDMLookupTable): The lookup table (its interpolators are read; the view is
            cached on it as ``_kernel_view``, keyed on the interpolator identities).

    Returns:
        dict | None: Flat float64 ``coeff_c`` / ``coeff_s`` on ``table.xp``, the fdot and f
        axes (``FD``, ``FF``, ``fdot0``, ``dfdot``, ``f0``, ``df``), the normalised f range
        (``f_lo``, ``f_hi``), the reference parity ``ref_odd = (m_ref + n_ref) % 2`` and
        ``layer_df``; ``None`` when the kernel cannot reproduce the table.
    """
    if (getattr(table, "INTERP_METHOD", None) != "spline"
            or getattr(table, "build_kind", None) not in ("n_ref_only", "n_ref_complex")
            or getattr(table, "BASIS_CYCLE", None) != "quarter_turn"):
        return None
    xp = table.xp
    ci = table.table_cx_re_interpolate if table.build_kind == "n_ref_complex" else table.table_cos_interpolate
    if getattr(table, "_sin_unbaked_interp", None) is None:
        table._sin_unbaked_coeffs(xp.zeros(1), xp.zeros(1))          # builds it (lazily, once)
    si = table._sin_unbaked_interp
    key = (id(ci), id(si))
    view = getattr(table, "_kernel_view", None)
    if view is not None and view["key"] == key:
        return view
    if ci.two_d:
        (fd0, dfd, FD), (f0, df, FF) = ci.axes
    else:
        (f0, df, FF), = ci.axes
        fd0, dfd, FD = 0.0, 1.0, 1
    fv = table.f_vals_norm
    view = dict(key=key, coeff_c=xp.ascontiguousarray(ci.coeffs, dtype=xp.float64).reshape(-1),
                coeff_s=xp.ascontiguousarray(si.coeffs, dtype=xp.float64).reshape(-1),
                FD=int(FD), FF=int(FF), fdot0=float(fd0), dfdot=float(dfd), f0=float(f0), df=float(df),
                f_lo=float(fv.min()), f_hi=float(fv.max()), ref_odd=int((table.m_ref + table.n_ref) % 2),
                layer_df=float(table.layer_df))
    table._kernel_view = view
    return view


def lookup_sum_kernel(backend, resp, acc, table, *, sub_row, n_stop, n_lo, n_hi, t0, layer_dt, Nf, m_lo,
                      num_m_layers, fdot_axis_max, f_min):
    """The fused lookup: ``acc[sub_row[s], ch, m - m_lo, n] +=`` every harmonic ``s``'s n_ref
    lookup over pixels ``n_lo <= n < min(n_hi, n_stop[s])``, straight from the sparse response
    ``resp`` (:class:`SplinedTDIOutput` / :class:`ExactPhaseTDIOutput`) -- the C++/CUDA
    ``wdm_lookup_sum`` of the backend. Same result as :func:`tracer_from_tof_output` (analytic) +
    :func:`accumulate_harmonic_batch`'s table lookup, up to summation order; nothing per pixel
    is materialised.

    Args:
        backend: The lisatools backend object (has ``wdm_lookup_sum`` and ``xp``).
        resp (SplinedTDIOutput): The sparse response (its :meth:`~SplinedTDIOutput.kernel_inputs`).
        acc (xp.ndarray): C-contiguous float64 ``(n_rows, nch, n_m, Nt)`` on the backend;
            added to in place.
        table (WDMLookupTable): A spline-interpolated n_ref table (:func:`table_kernel_view`).
        sub_row (array-like): Template row of each harmonic, ``(num_sub,)`` int.
        n_stop (array-like): Per harmonic, the first pixel NOT looked up (its handoff),
            ``(num_sub,)`` int.
        n_lo (int): First pixel of the contiguous range.
        n_hi (int): One past the last pixel.
        t0 (float): Absolute time of pixel 0 [s].
        layer_dt (float): Layer duration [s].
        Nf (int): Frequency layers of the full grid.
        m_lo (int): First layer held in ``acc``.
        num_m_layers (int): Layers each side of the carrier layer.
        fdot_axis_max (float): Table fdot bound [Hz/s] (beyond it a pixel is dropped).
        f_min (float): Lowest channel frequency looked up [Hz].

    Returns:
        dict: ``lookup_pixels`` and ``dropped_pixels``.

    Raises:
        ValueError: For a wrongly laid out ``acc``, a non-kernel table or a channel-count
            mismatch.
    """
    xp = backend.xp
    if acc.ndim != 4 or acc.dtype != np.float64 or not acc.flags.c_contiguous:
        raise ValueError("lookup_sum_kernel: acc must be a C-contiguous float64 (n_rows, nch, n_m, Nt) array")
    n_rows, nch, n_m, Nt = (int(v) for v in acc.shape)
    ki = resp.kernel_inputs()
    tv = table_kernel_view(table)
    if tv is None:
        raise ValueError("lookup_sum_kernel: the table is not a spline-interpolated n_ref table")
    if ki["nch"] != nch:
        raise ValueError(f"lookup_sum_kernel: response has {ki['nch']} channels, acc {nch}")

    def i32(a):
        """``a`` as a flat C-contiguous int32 array on ``xp``."""
        return xp.ascontiguousarray(xp.asarray(np.asarray(a, dtype=np.int32).reshape(-1)))

    no_d = xp.zeros(0, dtype=xp.float64)
    no_i = xp.zeros(0, dtype=xp.int32)
    counts = xp.zeros(2, dtype=xp.uint64)
    backend.wdm_lookup_sum(
        acc.reshape(-1), n_rows, counts,
        int(ki["carrier"]), int(ki.get("n_temp", 0)), int(ki.get("K", 0)), ki.get("t_knots", no_d),
        ki.get("n_knots", no_i), ki.get("C", no_d),
        int(ki["num_sub"]), ki.get("sub_temp", no_i), i32(sub_row), ki.get("sub_mkn", no_i), i32(n_stop),
        int(ki["N"]), nch, ki["x"], *ki["amp"], *ki["res"],
        int(n_lo), int(n_hi), float(t0), float(layer_dt), float(tv["layer_df"]), int(Nf), Nt, int(m_lo),
        int(m_lo) + n_m, int(num_m_layers), float(fdot_axis_max), float(f_min),
        tv["coeff_c"], tv["coeff_s"], tv["FD"], tv["FF"], tv["fdot0"], tv["dfdot"], tv["f0"], tv["df"],
        tv["f_lo"], tv["f_hi"], tv["ref_odd"])
    c = np.asarray(counts.get() if hasattr(counts, "get") else counts)
    return dict(lookup_pixels=int(c[0]), dropped_pixels=int(c[1]))


# TODO check: keep/remove/document -- per-sub reference loop of accumulate_harmonic_batch; not reached by the 9mo/12mo runs; only caller is tests/test_emri_wdm_direct.py
def _accumulate_harmonic_batch_loop(acc, table, tracks, tracer, n_ok, tail_td, *, Nf, Nt, dt, layer_dt,
                                    layer_df, t0, Nt_sub=128, num_m_layers=2, fdot_axis_max=np.inf,
                                    pixel_edge=8, backend="cpu"):
    """Add one batch of harmonics to ``acc`` (``(nch, Nf, Nt)``, global layers x pixels).

    Per sub ``s`` (``tracks[s]``, tracer row ``s``): the n_ref lookup on pixels before its
    handoff pixel, then even-start chunks of its own dense TD over ``[n_h, n_end)``
    (``n_end`` = a quarter chunk past the last tracked pixel: the abrupt stop at plunge
    rings ~24 px). The tail TD is evaluated ONCE per chunk window for all subs that need it
    (``tail_td(ts) -> (num_sub, nch, ts.size)``, zero outside the response support) and
    chunks are added in place over their kept slice (no full-grid temporaries).

    Args:
        tracer: ``(amp, phase, f, fdot)`` from :func:`tracer_from_tof_output`,
            each ``(num_sub, nch, n_ok.size)``.
        n_ok: pixel indices of the tracer/track samples (ascending).
        t0: absolute time of pixel 0.
    Returns a stats dict (lookup pixels, chunk pixels, pixels dropped before the handoff
    because the channel fdot left the table or f was below 2 layers).
    """
    from ...wdm_het import tail_chunk_plan, wdm_chunk_of_td

    amp, phase, f, fdot = tracer
    nch = amp.shape[1]
    n_ok = np.asarray(n_ok)
    stats = dict(lookup_pixels=0, chunk_pixels=0, dropped_pixels=0)
    windows = {}
    for s, tr in enumerate(tracks):
        k_h = handoff_pixel(tr, layer_dt, layer_df, fdot_axis_max)
        before = np.arange(n_ok.size) < k_h
        for ch in range(nch):
            sel = before & (np.abs(fdot[s, ch]) <= fdot_axis_max) & (f[s, ch] > 2 * layer_df)
            stats["dropped_pixels"] += int(np.sum(before & ~sel))
            if np.any(sel):
                co, mm = table.get_wdm_coeffs(amp[s, ch, sel], phase[s, ch, sel], f[s, ch, sel],
                                              fdot[s, ch, sel], n_ok[sel], num_m_layers=num_m_layers,
                                              out_of_support="zero")
                co, mm = np.asarray(co), np.asarray(mm)
                for c in range(co.shape[1]):
                    good = mm[:, c] >= 0
                    np.add.at(acc[ch], (mm[good, c], n_ok[sel][good]), co[good, c])
                stats["lookup_pixels"] += int(sel.sum())
        if k_h < n_ok.size:
            n_h = int(n_ok[k_h])
            n_end = min(int(n_ok[-1]) + 1 + Nt_sub // 4, Nt - pixel_edge)
            for n0, klo, khi in tail_chunk_plan(n_h, n_end, Nt, Nt_sub):
                windows.setdefault(n0, []).append((s, klo, khi))
    for n0, items in windows.items():
        ts = t0 + (n0 * Nf + np.arange(Nf * Nt_sub)) * dt
        td_all = np.asarray(tail_td(ts))                     # one evaluation per window
        for s, klo, khi in items:
            chunk = np.asarray(wdm_chunk_of_td(td_all[s], 0, Nf, Nt_sub, dt, backend=backend))
            acc[:, :chunk.shape[-2], n0 + klo:n0 + khi] += chunk[:, :, klo:khi]   # chunk rows = global layers
            stats["chunk_pixels"] += khi - klo
    return stats


def dense_inputs_from_holder(holder, integrator, *, a, xI0):
    """Inputs of :class:`lisatools.response.tdionfly.TDDenseTDIonTheFly` for one template.

    Returns ``(t_k (K,), phase_coeffs (K-1, 3, 8), mkn (S, 3), amp_re (S, K-1, 4),
    amp_im (S, K-1, 4))`` with the harmonics in :func:`harmonic_tracks_from_holder`'s order
    (m >= 0 modes, then the -m partners). The DOPR853 dense-output coefficients carry every
    convention of ``harmonic_tracks_from_holder`` (massratio scaling, the integrator's backwards
    adjustment, sign(xI0) on Phi_phi for a > 0, the holder's backwards offset), so the dense
    polynomial IS the track phase; the amplitude splines are the tracks' (scipy CubicSpline over
    the knots, monomial coefficients in ``t - t_knot``).

    Args:
        holder: Host FEW sparse holder of the template (knots feed).
        integrator: The trajectory integrator, still holding ``holder``'s trajectory.
        a (float): Spin.
        xI0 (float): Inclination cosine (``< 0`` maps to ``(-a, +1)``).

    Returns:
        tuple: ``(t_k, phase_coeffs, mkn, amp_re, amp_im)``; ``t_k`` on the FEW clock.
    """
    if xI0 < 0:
        a, xI0 = -a, -xI0
    t_int = np.asarray(integrator.integrator_t_cache, dtype=float).copy()
    t_k = np.asarray(holder.t_arr, dtype=float).copy()     # amplitude knots (FEW cuts the last at T)
    C = np.array(integrator.integrator_spline_coeff, dtype=float)[:, 3:6, :] / integrator.massratio
    if integrator.integrate_backwards:              # eval_integrator_spline's adjustment
        traj = np.asarray(integrator.trajectory)
        C[:, :, 0] += (traj[0, 4:7] + traj[-1, 4:7])[None, :]
    if a > 0:
        C[:, 0, :] *= np.sign(xI0)
    if holder.integrate_backwards:                  # base.py:361-364 (as harmonic_tracks_from_holder)
        knots = _phase_columns(integrator, t_k, 0)
        if a > 0:
            knots[:, 0] *= np.sign(xI0)
        C[:, :, 0] += (knots[-1] + knots[0])[None, :]
    C = _dense_on_grid(t_int, C, t_k)                        # phases re-expressed on the amplitude knots
    teuk = np.asarray(holder.teuk_modes)
    nm = teuk.shape[1]
    cr = CubicSpline(t_k, teuk.real, axis=0).c[::-1]         # (4, K-1, nm), a0..a3
    ci = CubicSpline(t_k, teuk.imag, axis=0).c[::-1]
    ylms = np.asarray(holder.ylms)
    ls, ms, ks, ns = (np.asarray(x) for x in (holder.ls, holder.ms, holder.ks, holder.ns))
    mkn, are, aim = [], [], []
    for j in range(nm):                                      # A * Ylm
        y = ylms[j]
        are.append((y.real * cr[:, :, j] - y.imag * ci[:, :, j]).T)
        aim.append((y.real * ci[:, :, j] + y.imag * cr[:, :, j]).T)
        mkn.append((ms[j], ks[j], ns[j]))
    if ylms.shape[0] == 2 * nm:                              # (-1)^l Y_{l,-m} conj(A), phase -Phi
        for j in np.flatnonzero(ms != 0):
            y = ((-1.0) ** ls[j]) * ylms[nm + j]
            are.append((y.real * cr[:, :, j] + y.imag * ci[:, :, j]).T)
            aim.append((-y.real * ci[:, :, j] + y.imag * cr[:, :, j]).T)
            mkn.append((-ms[j], -ks[j], -ns[j]))
    return t_k, C, np.array(mkn, dtype=np.int32), np.array(are), np.array(aim)


def _dense_basis(s):
    """The DOPR853 nested dense-output basis at s (n,) -> (n, 8): the polynomial is basis @ r."""
    s1 = 1.0 - s
    return np.stack([np.ones_like(s), s, s * s1, s ** 2 * s1, s ** 2 * s1 ** 2, s ** 3 * s1 ** 2,
                     s ** 3 * s1 ** 3, s ** 4 * s1 ** 3], axis=-1)


def _dense_on_grid(t_int, C, t_new):
    """DOPR853 coefficients ``C`` (n_int - 1, P, 8) on knots ``t_int`` re-expressed on knots
    ``t_new`` (a subset of ``t_int`` plus a cut inside a segment, e.g. FEW's final point at T).

    Segments shared by both grids keep their coefficients; any other is refit EXACTLY (the 8
    nested basis functions span all degree-7 polynomials) from 16 Chebyshev points of the
    original polynomial."""
    t_new = np.asarray(t_new, dtype=float)
    out = np.empty((t_new.size - 1,) + C.shape[1:])
    j = np.clip(np.searchsorted(t_int, t_new[:-1], side="right") - 1, 0, t_int.size - 2)
    same = (np.abs(t_int[j] - t_new[:-1]) <= 1e-9 * np.abs(t_new[:-1]).max()) & \
           (np.abs(t_int[j + 1] - t_new[1:]) <= 1e-9 * np.abs(t_new[1:]).max())
    out[same] = C[j[same]]
    x = 0.5 * (1 - np.cos(np.pi * (np.arange(16) + 0.5) / 16))
    B = _dense_basis(x)
    for q in np.flatnonzero(~same):
        ts = t_new[q] + x * (t_new[q + 1] - t_new[q])
        vals = dense_phase_eval(t_int, C, ts)                # (16, P)
        out[q] = np.linalg.lstsq(B, vals, rcond=None)[0].T
    return out


def dense_eval_derivs(c, s):
    """The nested DOPR853 polynomial and its first two ``s``-derivatives.

    ``c`` (..., 8) coefficients, ``s`` broadcastable to ``c[..., 0]``. Horner with forward-mode
    derivatives through ``r1 + s(r2 + s1(r3 + s(r4 + s1(r5 + s(r6 + s1(r7 + s r8))))))``,
    ``s1 = 1 - s``: a step ``q = c + s p`` gives ``q' = p + s p'``, ``q'' = 2 p' + s p''``; a step
    ``q = c + s1 p`` gives ``q' = -p + s1 p'``, ``q'' = -2 p' + s1 p''``. Returns ``(P, dP/ds,
    d2P/ds2)``; divide by ``h`` and ``h**2`` for time derivatives. The C++ lookup-sum kernel runs
    the same recursion (cutils/wdm_lookup_kernels.hh: wdm_dense_phase_derivs)."""
    s1 = 1.0 - s
    v = c[..., 7]
    d1 = 0.0 * v
    d2 = 0.0 * v
    for j, use_s in ((6, True), (5, False), (4, True), (3, False), (2, True), (1, False), (0, True)):
        if use_s:
            d2, d1, v = 2.0 * d1 + s * d2, v + s * d1, c[..., j] + s * v
        else:
            d2, d1, v = -2.0 * d1 + s1 * d2, -v + s1 * d1, c[..., j] + s1 * v
    return v, d1, d2


def dense_phase_eval(t_k, C, t):
    """Evaluate DOPR853 dense-output coefficients ``C`` (K-1, P, 8) on knots ``t_k`` at ``t``.

    Returns ``(len(t), P)``; times outside the knots use the end segments. Used by
    :func:`_dense_on_grid` and as the reference in tests.
    """
    t = np.asarray(t, dtype=float)
    seg = np.clip(np.searchsorted(t_k, t, side="right") - 1, 0, t_k.size - 2)
    s = ((t - t_k[seg]) / np.diff(t_k)[seg])[:, None]
    s1 = 1.0 - s
    c = C[seg]
    return c[..., 0] + s * (c[..., 1] + s1 * (c[..., 2] + s * (c[..., 3] + s1 * (c[..., 4] + s * (c[..., 5] + s1 * (c[..., 6] + s * c[..., 7]))))))


#: Fewest points a template's response grid may have (:meth:`EMRIDirectWDM._dense_response`
#: refuses fewer, as a domain error). The response splines are fitted on the run's backend;
#: on the GPU that is GBT's ``CubicSplineInterpolant`` -> cuSPARSE ``gtsv2StridedBatch``,
#: which rejects a system of fewer than 3 points, and GBT then calls ``exit(-1)``: no
#: traceback, the MPI job torn down (9mo jobs 795, 798 and 808, 2026-10-20/22, a hot-rung row
#: whose trajectory ended within the delay margins of its start, so
#: :func:`sparse_response_grid` returned its 2-point fallback).
MIN_RESPONSE_POINTS = 4


def sparse_response_grid(knots, a, b, sparse_dt, margin):
    """Sparse response times over ``[a, b]``, kept where the response is COMPLETE.

    The knots inside, plus a point every ``sparse_dt`` at most, restricted to ``[knots[0] +
    margin, knots[-1] - margin]``: at a time within ``margin`` (the TDI delay margin) of a
    trajectory end some delayed source times fall outside the trajectory and the channel
    amplitude drops to zero there. A cubic spline over half-day intervals across that drop
    rings back into the interior (2 % in amplitude, 0.3 rad in phase two intervals in), so the
    sparse grid must never contain such a point; an in-window plunge is resolved by the plunge
    chunk's exact dense time series instead.

    Args:
        knots (np.ndarray): Trajectory knot times [s, FEW clock].
        a (float): Window start [s, FEW clock].
        b (float): Window end [s, FEW clock].
        sparse_dt (float): Largest spacing [s].
        margin (float): TDI delay margin [s].

    Returns:
        np.ndarray: Sorted grid times; ``[lo, lo + 1]`` when the restricted window is empty
        (a trajectory shorter than about two margins: too few points for the response
        splines, refused by :meth:`EMRIDirectWDM._dense_response`).
    """
    knots = np.asarray(knots, dtype=float)
    lo = max(float(a), float(knots[0]) + margin)
    hi = min(float(b), float(knots[-1]) - margin)
    if hi <= lo:
        return np.array([lo, lo + 1.0])
    inner = knots[(knots > lo) & (knots < hi)]
    return np.union1d(np.append(np.arange(lo, hi, sparse_dt), hi), inner)


def pad_grid(g, N):
    """``g`` grown to ``N`` points by halving its longest intervals (stays inside ``g``'s span:
    a template's grid is padded to the batch's common length without leaving the region where
    its response is complete). Returns an ``np.ndarray`` (sorted when grown; the input as a
    float array when it already has ``>= N`` points)."""
    g = np.asarray(g, dtype=float)
    while g.size < N:
        d = np.diff(g)
        k = min(N - g.size, d.size)
        widest = np.argsort(d)[::-1][:k]
        g = np.sort(np.concatenate([g, g[widest] + 0.5 * d[widest]]))
    return g


class SplinedTDIOutput:
    """A dense TDI response's amplitude and phase splined over its output grid, evaluable for
    any SUBSET of harmonics (a mode-batched caller keeps every temporary at a block's size).

    The channel signal of harmonic ``s`` is ``amp(t) exp(-i (carrier_s(t) + r(t)))``; here the
    carrier is the kernel's ``phase_ref`` and ``r`` its ``tdi_phase``, both splined (the
    original construction: the grid must resolve the whole phase). Duck-types what
    :func:`tracer_from_tof_output` and the plunge tail read from a ``TDTDIOutput``: ``x``,
    ``xp``, ``nch``, ``eval_spline_vals(t, subs=None)`` (``(amp, r, carrier)``),
    ``eval_phase(t, subs=None)`` and ``eval_tdi(t)``. Used with ``response_grid="pixels"``.

    Args:
        out (TDTDIOutput): Raw dense-kernel output (``return_spline=False``): ``x``
            ``(num_sub, N)``, ``tdi_amp`` / ``tdi_phase`` ``(num_sub, nch, N)``, ``phase_ref``
            ``(num_sub, N)``, ``build_spline``.
        sub_mkn (np.ndarray): ``(num_sub, 3)`` harmonic ``(m, k, n)``.
        sub_temp (np.ndarray): ``(num_sub,)`` template index of each harmonic.

    Attributes:
        x: Response times ``(num_sub, N)`` [s, absolute].
        nch (int): Channels.
        num_sub (int): Harmonics (all templates).
    """

    def __init__(self, out, sub_mkn, sub_temp):
        self.xp = out.xp
        self.x = out.x
        self.nch = int(out.tdi_amp.shape[1])
        self.num_sub = int(out.tdi_amp.shape[0])
        self._temp = np.asarray(sub_temp, dtype=np.int64).reshape(-1)
        self._mkn_host = np.asarray(sub_mkn, dtype=float).reshape(-1, 3)
        self._amp_spl = out.build_spline(out.x, out.tdi_amp)
        self._res_spl = out.build_spline(out.x, self._residual(out))
        self._car_spl = self._carrier_spline(out)

    def _residual(self, out):
        """The splined channel phase: ``tdi_phase`` here (the whole phase minus the carrier)."""
        return out.tdi_phase

    def _carrier_spline(self, out):
        """Spline of the kernel's ``phase_ref`` (the carrier)."""
        return out.build_spline(out.x, out.phase_ref)

    def _subs(self, subs):
        """``subs`` as a flat int64 host array (all harmonics for ``None``)."""
        return np.arange(self.num_sub) if subs is None else np.asarray(subs, dtype=np.int64).reshape(-1)

    def _eval_chan(self, spl, t, subs, derivative=0):
        """A per-(harmonic, channel) spline at the shared times ``t``: ``(len(subs), nch, M)``."""
        xp = self.xp
        t = xp.asarray(t, dtype=float)
        n = subs.size
        ind = xp.asarray(subs[:, None] * self.nch + np.arange(self.nch)[None, :])
        return spl(xp.tile(t, (n, self.nch, 1)), ind_interps=ind, derivative=derivative)

    def carrier(self, t, subs=None):
        """Carrier phase ``(len(subs), len(t))``."""
        xp = self.xp
        subs = self._subs(subs)
        t = xp.asarray(t, dtype=float)
        return self._car_spl(xp.tile(t, (subs.size, 1)), ind_interps=xp.asarray(subs))

    def carrier_derivs(self, t, subs=None):
        """Carrier phase and its first two time derivatives, each ``(len(subs), len(t))``."""
        xp = self.xp
        subs = self._subs(subs)
        tt = xp.tile(xp.asarray(t, dtype=float), (subs.size, 1))
        ind = xp.asarray(subs)
        return tuple(self._car_spl(tt, ind_interps=ind, derivative=d) for d in (0, 1, 2))

    def eval_tracer(self, t, subs=None):
        """``(amp, phase, f, fdot)`` at times ``t`` with ANALYTIC derivatives, one evaluation
        per time: ``f = (carrier' + r') / 2 pi`` and ``fdot = (carrier'' + r'') / 2 pi`` (the
        residual spline's own derivatives, the carrier's exact ones). A negative-frequency sub
        (a -m partner) is mirrored as in :func:`tracer_from_tof_output`. Each ``(len(subs),
        nch, len(t))``."""
        xp = self.xp
        subs = self._subs(subs)
        car, car1, car2 = self.carrier_derivs(t, subs)
        amp = self._eval_chan(self._amp_spl, t, subs)
        ph = self._eval_chan(self._res_spl, t, subs) + car[:, None, :]
        f = (self._eval_chan(self._res_spl, t, subs, derivative=1) + car1[:, None, :]) / (2 * np.pi)
        fdot = (self._eval_chan(self._res_spl, t, subs, derivative=2) + car2[:, None, :]) / (2 * np.pi)
        neg = f < 0
        return amp, xp.where(neg, -ph, ph), xp.where(neg, -f, f), xp.where(neg, -fdot, fdot)

    def eval_spline_vals(self, t, subs=None):
        """``(amp, r, carrier)`` at the shared times ``t``: ``(S, nch, M)``, ``(S, nch, M)``,
        ``(S, M)`` for ``S = len(subs)``."""
        subs = self._subs(subs)
        return (self._eval_chan(self._amp_spl, t, subs), self._eval_chan(self._res_spl, t, subs),
                self.carrier(t, subs))

    def eval_phase(self, t, subs=None):
        """Channel phase ``r + carrier`` only (no amplitude): ``(len(subs), nch, len(t))``."""
        subs = self._subs(subs)
        return self._eval_chan(self._res_spl, t, subs) + self.carrier(t, subs)[:, None, :]

    def eval_tdi(self, t, subs=None):
        """Channel time series ``Re[amp exp(-i (r + carrier))]``, ``(len(subs), nch, len(t))``."""
        amp, res, car = self.eval_spline_vals(t, subs)
        return self.xp.real(amp * self.xp.exp(-1j * (res + car[:, None, :])))

    def kernel_inputs(self):
        """Arrays of the fused lookup-sum kernel (:func:`lookup_sum_kernel`): the amplitude and
        phase splines as GBT CubicSplineInterpolant flats (ninterps ``num_sub * nch``, row
        ``s * nch + ch``) and the carrier. Here the carrier spline is folded into the phase
        spline (a cubic spline is linear in its data on fixed knots: the spline of
        ``tdi_phase + phase_ref`` has the summed coefficients) and the kernel runs without one."""
        xp = self.xp
        S, nch = self.num_sub, self.nch
        res = []
        for name in ("y_flat", "c1_flat", "c2_flat", "c3_flat"):
            r = getattr(self._res_spl, name).reshape(S, nch, -1)
            c = getattr(self._car_spl, name).reshape(S, 1, -1)
            res.append(xp.ascontiguousarray(r + c).reshape(-1))
        return self._kernel_common(dict(carrier=0, res=res))

    def _kernel_common(self, d):
        """Add the amplitude spline flats and the sizes (``N``, ``nch``, ``num_sub``, ``x``) to
        the kernel-input dict ``d`` and return it."""
        xp = self.xp
        a = self._amp_spl
        d.update(N=int(a.length), nch=self.nch, num_sub=self.num_sub,
                 x=xp.ascontiguousarray(a.x_flat, dtype=xp.float64),
                 amp=[xp.ascontiguousarray(getattr(a, k), dtype=xp.float64) for k in ("y_flat", "c1_flat", "c2_flat", "c3_flat")])
        return d


class ExactPhaseTDIOutput(SplinedTDIOutput):
    """A dense TDI response computed on a SPARSE time grid, evaluated anywhere with the
    harmonic's EXACT carrier phase.

    The channel signal of harmonic ``s`` is ``amp(t) exp(-i (Phi_s(t) + r(t)))``. ``Phi_s =
    m Phi_phi + k Phi_theta + n Phi_r`` is the DOPR853 dense-output carrier, exact at any time
    from the same coefficients the kernel uses (held at the trajectory end outside it, as the
    kernel holds its reference phase). Everything else is slow -- the TDI amplitude and the
    residual ``r = tdi_phase + phase_ref - Phi_s`` (the Doppler delay to spacecraft 1, the
    polarisation and transfer-function phase, the mode amplitude's own phase) -- so only
    ``amp`` and ``r`` are splined over the response grid. The response therefore needs only
    the trajectory's own knots plus a spacing cap, not one sample per pixel. The carrier is
    evaluated on the output's array module (the GPU on a GPU run), per template. This is the
    production response output (``response="dense"``, ``response_grid="sparse"``).

    Args:
        out (TDTDIOutput): Raw dense-kernel output on the sparse grid (see
            :class:`SplinedTDIOutput`).
        t_knots (list[np.ndarray]): Per template, the absolute knot times [s] of its DOPR853
            dense output.
        coeffs (list[np.ndarray]): Per template, the dense-output coefficients
            ``(K - 1, 3, 8)``.
        sub_mkn (np.ndarray): ``(num_sub, 3)`` harmonic ``(m, k, n)``.
        sub_temp (np.ndarray): ``(num_sub,)`` template index of each harmonic.
    """

    def __init__(self, out, t_knots, coeffs, sub_mkn, sub_temp):
        xp = out.xp
        self._tk = [xp.asarray(np.asarray(tk, dtype=float)) for tk in t_knots]   # absolute, per template
        self._C = [xp.asarray(np.asarray(c, dtype=float)) for c in coeffs]
        self._mkn = xp.asarray(np.asarray(sub_mkn, dtype=float).reshape(-1, 3))
        super().__init__(out, sub_mkn, sub_temp)

    def _residual(self, out):
        """The slow residual ``tdi_phase + phase_ref - Phi_s`` on the response grid."""
        xp = self.xp
        carrier = xp.empty(out.phase_ref.shape)
        x = out.x
        for b in np.unique(self._temp):
            rows = np.flatnonzero(self._temp == b)
            carrier[xp.asarray(rows)] = self._mkn[xp.asarray(rows)] @ self._P3(int(b), x[int(rows[0])]).T
        return out.tdi_phase + (out.phase_ref - carrier)[:, None, :]

    def _carrier_spline(self, out):
        """No carrier spline: the carrier is exact (:meth:`carrier`)."""
        return None

    def _P3(self, b, t):
        """Template ``b``'s three fundamental phases at the 1-D times ``t``: ``(len(t), 3)``."""
        xp = self.xp
        tk, C = self._tk[b], self._C[b]
        t = xp.clip(xp.asarray(t, dtype=float), tk[0], tk[-1])
        seg = xp.clip(xp.searchsorted(tk, t, side="right") - 1, 0, tk.size - 2)
        s = ((t - tk[seg]) / (tk[seg + 1] - tk[seg]))[:, None]
        s1 = 1.0 - s
        c = C[seg]
        return c[..., 0] + s * (c[..., 1] + s1 * (c[..., 2] + s * (c[..., 3] + s1 * (c[..., 4] + s * (
            c[..., 5] + s1 * (c[..., 6] + s * c[..., 7]))))))

    def _P3_derivs(self, b, t):
        """``_P3`` and its first two time derivatives, each ``(len(t), 3)``; outside the
        trajectory the phase is held at the nearest end, so both derivatives are zero there."""
        xp = self.xp
        tk, C = self._tk[b], self._C[b]
        t_in = xp.asarray(t, dtype=float)
        t = xp.clip(t_in, tk[0], tk[-1])
        held = ((t_in < tk[0]) | (t_in > tk[-1]))[:, None]
        seg = xp.clip(xp.searchsorted(tk, t, side="right") - 1, 0, tk.size - 2)
        h = (tk[seg + 1] - tk[seg])[:, None]
        v, d1, d2 = dense_eval_derivs(C[seg], ((t - tk[seg]) / (tk[seg + 1] - tk[seg]))[:, None])
        return v, xp.where(held, 0.0, d1 / h), xp.where(held, 0.0, d2 / h ** 2)

    def carrier_derivs(self, t, subs=None):
        """Exact carrier ``Phi_s`` and its first two time derivatives at ``t``, each
        ``(len(subs), len(t))`` (zero derivatives outside the trajectory)."""
        xp = self.xp
        subs = self._subs(subs)
        t = xp.asarray(t, dtype=float)
        outs = [xp.empty((subs.size, t.size)) for _ in range(3)]
        temp = self._temp[subs]
        for b in np.unique(temp):
            pos = np.flatnonzero(temp == b)
            mk = self._mkn[xp.asarray(subs[pos])]
            for o, P in zip(outs, self._P3_derivs(int(b), t)):
                o[xp.asarray(pos)] = mk @ P.T
        return tuple(outs)

    def kernel_inputs(self):
        """Kernel arrays (see :meth:`SplinedTDIOutput.kernel_inputs`) with the EXACT carrier: the
        templates' knots padded to a common stride ``K`` (``(n_temp, K)``, the last knot
        repeated), ``n_knots``, the DOPR853 coefficients ``(n_temp, K - 1, 3, 8)``, ``sub_temp``
        and ``sub_mkn``; the phase spline is the residual."""
        xp = self.xp
        n_temp = len(self._tk)
        K = max(int(tk.size) for tk in self._tk)
        t_knots = xp.empty((n_temp, K))
        C = xp.zeros((n_temp, K - 1, 3, 8))
        for b, (tk, Cb) in enumerate(zip(self._tk, self._C)):
            t_knots[b, :tk.size] = tk
            t_knots[b, tk.size:] = tk[-1]
            C[b, :tk.size - 1] = Cb
        r = self._res_spl
        return self._kernel_common(dict(
            carrier=1, n_temp=n_temp, K=K, t_knots=t_knots.reshape(-1), C=C.reshape(-1),
            n_knots=xp.asarray(np.array([int(tk.size) for tk in self._tk], dtype=np.int32)),
            sub_temp=xp.asarray(self._temp.astype(np.int32)),
            sub_mkn=xp.asarray(np.rint(self._mkn_host).astype(np.int32).reshape(-1)),
            res=[xp.ascontiguousarray(getattr(r, k), dtype=xp.float64) for k in ("y_flat", "c1_flat", "c2_flat", "c3_flat")]))

    def carrier(self, t, subs=None):
        """``Phi_s(t)`` for harmonics ``subs`` at the shared times ``t``: ``(len(subs), len(t))``."""
        xp = self.xp
        subs = self._subs(subs)
        t = xp.asarray(t, dtype=float)
        out = xp.empty((subs.size, t.size))
        temp = self._temp[subs]
        for b in np.unique(temp):
            pos = np.flatnonzero(temp == b)
            out[xp.asarray(pos)] = self._mkn[xp.asarray(subs[pos])] @ self._P3(int(b), t).T
        return out


def feed_from_tracks(tracks, amp_factor):
    """Per-sub (amp, phase) with h = sum amp exp(-i phase) from harmonic tracks on the feed grid.

    A track term is ``track.amp * exp(-i track.phase)`` (FEW's mode sum, -m partners included),
    so ``amp = |track.amp| * amp_factor`` and ``phase = track.phase - unwrap(arg(track.amp))``.
    Returns two ``(num_sub, n)`` ``np.ndarray`` (the spline-response knots feed)."""
    amp = np.stack([np.abs(tr.amp) for tr in tracks]) * amp_factor
    phase = np.stack([tr.phase - np.unwrap(np.angle(tr.amp)) for tr in tracks])
    return amp, phase


def slice_holder(H, idx):
    """The host holder restricted to modes ``idx`` (m >= 0 rows; the -m partners' Ylm follow).

    Returns a ``types.SimpleNamespace`` with the holder fields (``t_arr``, ``phases`` and
    ``freqs`` shared, not copied). Used for the mode batches of :meth:`EMRIDirectWDM.__call__`.
    """
    import types

    idx = np.asarray(idx)
    nm = len(H.ls)
    ylms = np.asarray(H.ylms)
    yl = np.concatenate([ylms[idx], ylms[nm + idx]]) if ylms.shape[0] == 2 * nm else ylms[idx]
    return types.SimpleNamespace(t_arr=H.t_arr, teuk_modes=np.asarray(H.teuk_modes)[:, idx], phases=H.phases,
                                 freqs=H.freqs, ylms=yl, ls=np.asarray(H.ls)[idx], ms=np.asarray(H.ms)[idx],
                                 ks=np.asarray(H.ks)[idx], ns=np.asarray(H.ns)[idx],
                                 integrate_backwards=H.integrate_backwards)


def track_rows(H, idx):
    """Rows of ``harmonic_tracks_from_holder(H, ...)`` for modes ``idx``, in the order the same
    function (and ``EMRITDIonFly.mode_amp_phase``) gives for ``slice_holder(H, idx)``: the m >= 0
    modes, then the -m partners of those with m != 0."""
    ms = np.asarray(H.ms)
    nm = ms.size
    with_minus = np.asarray(H.ylms).shape[0] == 2 * nm
    minus_row = np.full(nm, -1)
    minus_row[ms != 0] = nm + np.arange(int(np.sum(ms != 0)))
    rows = list(np.asarray(idx))
    if with_minus:
        rows += [int(minus_row[i]) for i in idx if ms[i] != 0]
    return rows


def subset_track(tr, pos):
    """A HarmonicTrack restricted to sample positions ``pos`` (needs ``f``, ``fdot``, ``fddot``)."""
    return HarmonicTrack(tr.lmkn, tr.t[pos], tr.amp[pos], tr.phase[pos], tr.f[pos], tr.fdot[pos], tr.fddot[pos])


class EMRIDirectWDM:
    """EMRI template built directly in the WDM domain.

    Per harmonic: the n_ref lookup up to its handoff pixel (curvature or fdot range),
    then even-start chunks of its own dense TDI-on-the-fly TD over the plunge tail.
    On the ``feed="fine"`` / fixed-``n_fine`` path of :meth:`__call__`, harmonics are processed
    in batches of ``mode_batch`` so the per-harmonic response splines never all live at once
    (the scale-up constraint); the knots path builds all harmonics in one response.
    Production builds it in ``globalfit.stock.erebor.source_runtime.get_emri_direct_gen`` on
    the run's FULL grid with ``response="dense"`` (``EMRI_DIRECT_RESPONSE``), the default
    ``feed="knots"``, ``lookup="kernel"`` and ``response_grid="sparse"``, and
    :class:`~lisatools.sources.emri.direct_signal_gen.EMRIDirectWDMSignalGen` sets ``f_band``.

    Args:
        few_gen: FEW generator (the production one; its inspiral options are restored
            after every call).
        table (WDMLookupTable): :class:`lisatools.domains.WDMLookupTable` built on the same Nf,
            dt. NOTE: ``interp`` switches this table's interpolators in place.
        wdm_set (WDMSettings): target :class:`~lisatools.domains.WDMSettings` (the full grid;
            pixel ``n`` is centred at ``data_t0 + n * layer_dt``).
        orbits (Orbits): as for :class:`~lisatools.sources.emri.emritdionfly.EMRITDIonFly`
            (``frame="icrs"`` orbits: the template uses the ``icrs_special`` frame).
        tdi_config (TDIConfig): the TDI configuration (``nchannels`` sets the output channels).
        t_start (float): FEW reference epoch (trajectory clock origin), e.g.
            ``MOJITO_REFERENCE_TIME``.
        data_t0 (float): absolute time of WDM pixel 0.
        Nt_sub (int, optional): time layers per plunge chunk. Default ``128``.
        n_fine (int, optional): fixed number of fine trajectory points over the window.
            Default ``None``: one per ``fine_dt`` (3600 s: direct-vs-production unchanged from
            80 s to 3600 s; 6 months: 1800 and 3600 s identical, mm 1e-7..4e-7 on CD1L EMRI 1,
            16 d), densified to ``fine_dt_plunge`` from a chunk before a handoff to the end
            (uniform at ``fine_dt_plunge`` with explicit modes on the ``feed="fine"`` path).
        fine_dt (float, optional): coarse fine-grid step [s]. Default ``3600.0``.
        fine_dt_plunge (float, optional): fine-grid step [s] near the plunge. Default ``80.0``.
        mode_batch (int, optional): harmonics per response call on the non-knots paths;
            ``None`` = all at once.
        pixel_edge (int, optional): pixels left zero at each grid end (response spline
            support). Default ``8``.
        num_m_layers (int, optional): layers each side of the carrier layer per pixel. Default
            ``2``; ``EMRI_DIRECT_NUM_M_LAYERS`` overrides it when set.
        interp (str, optional): table interpolation (``"spline"`` default: uniform cubic
            B-spline, CPU and GPU; ``"cubic"`` scipy-only; ``None`` keeps the table's).
        force_backend (str, optional): backend name. Default ``"cpu"``.
        feed (str, optional): ``"knots"`` (default: FEW amplitudes at the integrator knots,
            phases from the dense output; one cheap FEW call) or ``"fine"`` (FEW on the fine
            grid).
        response (str, optional): ``"spline"`` (class default:
            :class:`~lisatools.sources.emri.emritdionfly.EMRITDIonFly`) or ``"dense"``
            (:class:`~lisatools.response.tdionfly.TDDenseTDIonTheFly`; exact dense-output
            phases, many templates per launch; needs ``feed="knots"``).
        response_grid (str, optional): with ``response="dense"``, where the response is
            evaluated: ``"sparse"`` (default; env ``EMRI_DIRECT_RESPONSE_GRID``) the
            integrator knots plus a point every ``sparse_dt`` s at most, with the exact
            dense-output carrier added at the pixels (:class:`ExactPhaseTDIOutput`);
            ``"pixels"`` one sample per ``fine_dt`` with the whole phase splined. Forced to
            ``"pixels"`` with ``response="spline"``.
        sparse_dt (float, optional): sparse-grid spacing cap [s]; ``None`` reads
            ``EMRI_DIRECT_SPARSE_DT`` (43200).
        lookup (str, optional): ``"kernel"`` (the fused C++/CUDA lookup, falling back to Python
            with one warning when unavailable) or ``"python"``; ``None`` reads
            ``EMRI_DIRECT_LOOKUP`` (``"kernel"``).
        f_band (tuple[int, int], optional): layers ``[m_lo, m_hi)`` :meth:`batch` returns;
            ``None`` = all ``Nf``.

    Attributes:
        fdot_axis_max (float): Largest ``|fdot|`` of the table's axis [Hz/s].
        plunge_taper_s (float): Source-frame stop taper [s] (``EMRI_DIRECT_PLUNGE_TAPER_S``,
            default 120).
        last_stats (dict): Counters of the last call (``modes``, ``lookup_pixels``,
            ``chunk_pixels``, ``dropped_pixels``, ...).
        last_failed_rows (list[int]): Rows of the last :meth:`batch` FEW refused.

    Raises:
        ValueError: For an unknown ``feed``, ``response``, ``response_grid`` or ``lookup``, or
            ``response="dense"`` with ``feed="fine"``.

    Example:
        .. code-block:: python

            from lisatools.domains import WDMLookupTable, WDMSettings
            from lisatools.response.tdiconfig import TDIConfig
            from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

            table = WDMLookupTable.from_file(table_path, force_backend="cuda13x")
            direct = EMRIDirectWDM(
                few_gen, table, WDMSettings(Nf, Nt, dt, force_backend="cuda13x"),
                orbits=icrs_orbits, tdi_config=TDIConfig("2nd generation",
                                                         force_backend="cuda13x"),
                t_start=ref_epoch, data_t0=data_t0, force_backend="cuda13x",
                response="dense",
            )
            sig = direct(*few_params)            # WDMSignal (nch, Nf, Nt) on the full grid
            arr = direct.batch(rows, chunk_rows=8)
    """

    #: TDI delay margin [s]: a source time contributes to response times up to this much later
    DELAY_MARGIN = 600.0

    def __init__(self, few_gen, table, wdm_set, *, orbits, tdi_config, t_start, data_t0,
                 Nt_sub=128, n_fine=None, fine_dt=3600.0, fine_dt_plunge=80.0, mode_batch=None,
                 pixel_edge=8, num_m_layers=2, interp="spline", force_backend="cpu", feed="knots",
                 response="spline", response_grid=None, sparse_dt=None, lookup=None, f_band=None):
        self.few_gen, self.table, self.wdm = few_gen, table, wdm_set
        self.orbits, self.tdi_config = orbits, tdi_config
        self.t_start, self.data_t0 = float(t_start), float(data_t0)
        self.Nt_sub, self.pixel_edge = int(Nt_sub), int(pixel_edge)
        self.mode_batch = None if mode_batch is None else int(mode_batch)   # None: all modes in one batch
        # layers m-num_m_layers..m+num_m_layers per pixel: a chirping carrier near a layer edge
        # puts ~2.5e-4 of its energy two layers away (A8 gate); needs table support [-2, 3) df
        self.num_m_layers = int(os.environ.get("EMRI_DIRECT_NUM_M_LAYERS", num_m_layers))
        span = wdm_set.Nt * wdm_set.layer_dt
        if feed not in ("knots", "fine"):
            raise ValueError("feed must be 'knots' or 'fine'")
        # "knots": FEW amplitudes at the integrator knots, splined to the response grid, phases
        # from the integrator's dense output (production's construction; the cheap FEW call).
        # "fine": FEW evaluates amplitudes at every fine point (the earlier feed).
        self.feed = feed
        if response not in ("spline", "dense"):
            raise ValueError("response must be 'spline' or 'dense'")
        # "dense": TDDenseTDIonTheFly (exact dense-output phases, geometry shared across the
        # harmonics of a template, many templates per launch); needs feed="knots" and a backend
        # module built with TDDenseTDIWaveformWrap. "spline": EMRITDIonFly -> TDTDIonTheFly.
        self.response = response
        if response == "dense" and feed != "knots":
            raise ValueError("response='dense' needs feed='knots'")
        # response="dense" only: "sparse" evaluates the response on the integrator knots plus
        # a spacing cap (sparse_dt) and adds the exact dense-output carrier at the pixels
        # (ExactPhaseTDIOutput); "pixels" evaluates it on the fine_dt grid and splines the
        # whole channel phase (the original construction)
        # (explicit argument > env EMRI_DIRECT_RESPONSE_GRID / EMRI_DIRECT_SPARSE_DT > default)
        if response_grid is None:
            response_grid = os.environ.get("EMRI_DIRECT_RESPONSE_GRID", "sparse")
        if response_grid not in ("sparse", "pixels"):
            raise ValueError("response_grid must be 'sparse' or 'pixels'")
        self.response_grid = response_grid if response == "dense" else "pixels"
        if sparse_dt is None:
            sparse_dt = os.environ.get("EMRI_DIRECT_SPARSE_DT", 43200.0)
        self.sparse_dt = float(sparse_dt)
        self.n_fine_fixed = int(n_fine) if n_fine is not None else None
        self.fine_dt, self.fine_dt_plunge = float(fine_dt), float(fine_dt_plunge)
        self.n_fine = self.n_fine_fixed or max(64, int(span / self.fine_dt_plunge))   # set per call
        self.force_backend = force_backend
        from ...utils.utility import asnumpy
        from ... import get_backend
        self.xp = get_backend(force_backend).xp
        self.fdot_axis_max = float(np.max(np.abs(asnumpy(table.fdot_vals))))
        # cubic table interpolation: linear left a ~2.5e-4 amplitude deficit (A9, EMRI 1).
        # NOTE: this switches the passed table's interpolators (set_interp_method).
        if interp is not None and getattr(table, "INTERP_METHOD", None) != interp:
            table.set_interp_method(interp)
        # "kernel": the fused C++/CUDA lookup sum (lookup_sum_kernel) for a dense response,
        # nothing per pixel materialised; "python": analytic tracer + get_wdm_coeffs +
        # scatter-add in blocks of harmonics. A kernel request falls back to Python (warned
        # once) when the backend module, the response or the table cannot run it.
        if lookup is None:
            lookup = os.environ.get("EMRI_DIRECT_LOOKUP", "kernel")
        if lookup not in ("kernel", "python"):
            raise ValueError("lookup must be 'kernel' or 'python'")
        self.lookup = lookup
        self._kernel_fallback_warned = False
        # batch() returns layers [m_lo, m_hi) only (None: all Nf): the run's active band
        # (EMRIDirectWDMSignalGen sets it); a 6-month template keeps ~1/8 of the grid
        self.f_band = None if f_band is None else (int(f_band[0]), int(f_band[1]))
        # source-frame taper [s] over the last stretch of EMISSION time before the trajectory's
        # end, on every delay term of the plunge chunk's channels (_dense_tail_fn; env
        # EMRI_DIRECT_PLUNGE_TAPER_S, 0 = the hard stop)
        self.plunge_taper_s = float(os.environ.get("EMRI_DIRECT_PLUNGE_TAPER_S", 120.0))
        self.last_stats = {}
        self.last_failed_rows = []

    def _band(self):
        """``(m_lo, m_hi)`` of ``f_band`` (``(0, Nf)`` when unset); raises ``ValueError`` if it
        falls outside the grid."""
        if self.f_band is None:
            return 0, int(self.wdm.Nf)
        m_lo, m_hi = self.f_band
        if not 0 <= m_lo < m_hi <= int(self.wdm.Nf):
            raise ValueError(f"f_band {self.f_band} outside [0, {self.wdm.Nf}]")
        return m_lo, m_hi

    def _kernel_backend(self, out):
        """The backend whose fused lookup kernel serves the response ``out``, or ``None`` (the
        Python lookup runs)."""
        if self.lookup != "kernel":
            return None
        from ... import get_backend

        be = get_backend(self.force_backend)
        why = None
        if getattr(be, "wdm_lookup_sum", None) is None:
            why = "the backend module was built without wdm_lookup_sum"
        elif not hasattr(out, "kernel_inputs"):
            why = "the response output has no kernel inputs (response='spline')"
        elif table_kernel_view(self.table) is None:
            why = "the table is not a spline-interpolated n_ref quarter_turn table"
        if why is None:
            return be
        if not self._kernel_fallback_warned:
            import warnings

            warnings.warn(f"EMRIDirectWDM: lookup='kernel' unavailable ({why}); using the Python lookup")
            self._kernel_fallback_warned = True
        return None

    def _lookup_fn(self, be, out, acc4, sub_row, n_ok, m_lo):
        """``n_stop -> stats`` running the fused kernel into ``acc4`` (n_rows, nch, n_m, Nt; layers
        from ``m_lo``) over the contiguous pixels ``n_ok``."""
        n_ok = np.asarray(n_ok)
        if n_ok.size and not np.all(np.diff(n_ok) == 1):
            raise ValueError("the fused lookup needs a contiguous pixel range")

        def run(n_stop):
            if not n_ok.size:
                return dict(lookup_pixels=0, dropped_pixels=0)
            return lookup_sum_kernel(
                be, out, acc4, self.table, sub_row=sub_row, n_stop=n_stop, n_lo=int(n_ok[0]),
                n_hi=int(n_ok[-1]) + 1, t0=self.data_t0, layer_dt=self.wdm.layer_dt, Nf=self.wdm.Nf,
                m_lo=m_lo, num_m_layers=self.num_m_layers, fdot_axis_max=self.fdot_axis_max,
                f_min=2 * self.wdm.layer_df)
        return run

    def _few_holder(self, few_args, few_kwargs, new_t, mode_selection=None):
        """ONE FEW call on the fine grid ``new_t`` (FEW clock) -> host sparse holder.

        Mode selection by ``few_kwargs['mode_selection_threshold']`` unless ``mode_selection``
        is given. The generator's inspiral_kwargs are restored afterwards (FEW merges
        call-time ones permanently)."""
        from few.utils.utility import get_viewing_angles

        from .emritdionfly import host_holder

        m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = few_args[:14]
        th, ph = get_viewing_angles(qS, phiS, qK, phiK)
        kw = {k: v for k, v in few_kwargs.items() if k != "inspiral_kwargs"}
        if mode_selection is not None:
            kw.pop("mode_selection_threshold", None)
            kw["mode_selection"] = [tuple(int(v) for v in md) for md in mode_selection]
        new_t = np.asarray(new_t, dtype=float)
        saved = dict(self.few_gen.inspiral_kwargs)
        try:
            H = self.few_gen(m1, m2, a, p0, e0, x0, th, ph, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
                             Phi_r0=Pr, T=float(new_t[-1]) / YRSID_SI, dt=self.wdm.data_dt,
                             return_sparse_holder=True, include_minus_mkn=True,
                             inspiral_kwargs={"upsample": True, "fix_t": True, "new_t": new_t}, **kw)
        finally:
            self.few_gen.inspiral_kwargs.clear()
            self.few_gen.inspiral_kwargs.update(saved)
        return host_holder(H)                               # GPU generator: cupy -> host

    def _few_knots(self, few_args, few_kwargs, mode_selection=None):
        """ONE FEW call on the integrator's own knots (no upsampling: amplitudes at ~10^2-10^3
        knots, as the production mode sum) -> host sparse holder; mode selection by the call's
        threshold unless ``mode_selection`` is given."""
        from few.utils.utility import get_viewing_angles

        from .emritdionfly import host_holder

        m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = few_args[:14]
        th, ph = get_viewing_angles(qS, phiS, qK, phiK)
        kw = {k: v for k, v in few_kwargs.items() if k != "inspiral_kwargs"}
        if mode_selection is not None:
            kw.pop("mode_selection_threshold", None)
            kw["mode_selection"] = [tuple(int(v) for v in md) for md in mode_selection]
        span = self.wdm.Nt * self.wdm.layer_dt
        T = (self.data_t0 - self.t_start + span + 2000.0) / YRSID_SI
        saved = dict(self.few_gen.inspiral_kwargs)
        try:
            H = self.few_gen(m1, m2, a, p0, e0, x0, th, ph, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
                             Phi_r0=Pr, T=T, dt=self.wdm.data_dt, return_sparse_holder=True,
                             include_minus_mkn=True, **kw)
        finally:
            self.few_gen.inspiral_kwargs.clear()
            self.few_gen.inspiral_kwargs.update(saved)
        return host_holder(H)

    def _pixel_times(self, H):
        """Pixel indices (and times, FEW clock) inside the holder's trajectory span."""
        t_arr = np.asarray(H.t_arr)
        lo = self.data_t0 - self.t_start
        n_all = np.arange(self.pixel_edge, self.wdm.Nt - self.pixel_edge)
        t_rel = lo + n_all * self.wdm.layer_dt
        inside = (t_rel >= t_arr[0]) & (t_rel <= t_arr[-1])
        return n_all[inside], t_rel[inside]

    def _mode_list(self, few_args, few_kwargs):
        """Modes kept at the call's threshold and the earliest in-window handoff time.

        One FEW call (on the integrator knots with ``feed="knots"``, else on the coarse fine
        grid); the holder, its harmonic tracks (``None`` when no harmonic can hand off) and
        their pixels are kept (``_last_holder``, ``_last_tracks``, ``_last_track_n``) so the
        template reuses them. Returns ``(modes, chunk_start)`` with ``modes`` a list of
        ``(l, m, k, n)`` and ``chunk_start`` from :meth:`_chunk_start`."""
        if self.feed == "knots":
            H = self._few_knots(few_args, few_kwargs, mode_selection=few_kwargs.get("mode_selection"))
        else:
            H = self._few_holder(few_args, few_kwargs, self._fine_grid(None))
        modes = [(int(l), int(m), int(k), int(n)) for l, m, k, n in zip(H.ls, H.ms, H.ks, H.ns)]
        n_in, t_rel = self._pixel_times(H)
        tracks = self._maybe_tracks(H, few_args, t_rel)
        self._last_holder, self._last_tracks, self._last_track_n = H, tracks, n_in
        return modes, self._chunk_start(H, few_args, tracks=tracks, t_rel=t_rel)

    def _handoff_may_trip(self, H, t_rel):
        """Whether ANY harmonic of ``H`` can hand off to the plunge chunk at the times ``t_rel``
        (FEW clock), from a bound over the three fundamentals: ``|fdot_s| <= sum_j M_j
        |Omega'_j| / 2 pi`` and ``|fddot_s| <= sum_j M_j |Omega''_j| / 2 pi`` with ``M_j`` the
        largest ``|(m, k, n)_j|`` of the holder's modes (the -m partners have the same). False
        means :func:`handoff_pixel` cannot trip for any of them (no tracks needed); True means
        it may (the exact per-harmonic tracks decide). Env ``EMRI_DIRECT_HANDOFF_BOUND=0``
        always answers True (exact tracks every call)."""
        if os.environ.get("EMRI_DIRECT_HANDOFF_BOUND", "1") == "0":
            return True
        integ = self.few_gen.inspiral_generator.inspiral_generator
        M = np.max(np.abs(np.stack([np.asarray(H.ms), np.asarray(H.ks), np.asarray(H.ns)])), axis=1).astype(float)
        fdot_b = np.abs(_phase_columns(integ, t_rel, 2)) @ M / (2 * np.pi)
        fddot_b = np.abs(_phase_columns(integ, t_rel, 3)) @ M / (2 * np.pi)
        tau = WDM_HALF_SUPPORT_LAYERS * self.wdm.layer_dt
        return bool(np.any((np.pi / 3.0) * fddot_b * tau ** 3 > 0.1) or np.any(fdot_b > self.fdot_axis_max))

    def _maybe_tracks(self, H, few_args, t_rel):
        """Per-harmonic tracks at ``t_rel`` when a harmonic may hand off there, else ``None``
        (no handoff: the bound of :meth:`_handoff_may_trip` -- the tracks are needed only to
        find handoff pixels)."""
        if not np.size(t_rel):
            return []
        if not self._handoff_may_trip(H, t_rel):
            return None
        integ = self.few_gen.inspiral_generator.inspiral_generator
        return harmonic_tracks_from_holder(H, integ, t_rel, a=few_args[2], xI0=few_args[5])

    def _chunk_start(self, H, few_args, tracks="compute", t_rel=None):
        """Earliest time (FEW clock) at which any harmonic of the holder ``H`` hands off to the
        plunge chunk inside the window, or ``None`` if none does. The chunk reads the dense
        response, so the fine grid must be dense from there on. ``tracks``: the holder's tracks
        at ``t_rel``, ``None`` (proved not to hand off) or ``"compute"``."""
        t_arr = np.asarray(H.t_arr)
        lo = self.data_t0 - self.t_start
        if isinstance(tracks, str):
            _, t_rel = self._pixel_times(H)
            tracks = self._maybe_tracks(H, few_args, t_rel)
        first = np.inf
        for tr in tracks or []:
            k = handoff_pixel(tr, self.wdm.layer_dt, self.wdm.layer_df, self.fdot_axis_max)
            if k < t_rel.size:
                first = min(first, float(t_rel[k]))
        span = self.wdm.Nt * self.wdm.layer_dt
        if t_arr[-1] < lo + span - 1.0:                      # stops in the window
            first = min(first, float(t_arr[-1]) - self.Nt_sub * self.wdm.layer_dt)
        return None if not np.isfinite(first) else first

    def _fine_grid(self, chunk_start):
        """Fine trajectory times (FEW clock): ``fine_dt`` over the window, ``fine_dt_plunge``
        from a chunk length before ``chunk_start`` to the end (non-uniform is fine: the
        response splines auto-detect general spacing)."""
        lo = self.data_t0 - self.t_start
        span = self.wdm.Nt * self.wdm.layer_dt
        pad = 600.0 + 2.0 * self.fine_dt                     # delay margin + spline support
        a = max(0.0, lo - pad)
        b = lo + span + 2000.0                                # = the integration span (T_traj)
        coarse = np.append(np.arange(a, b, self.fine_dt), b)
        if chunk_start is None:
            return coarse
        d0 = max(a, chunk_start - self.Nt_sub * self.wdm.layer_dt)
        dense = np.append(np.arange(d0, b, self.fine_dt_plunge), b)
        return np.union1d(coarse[coarse < d0], dense)

    def _orbit_span(self):
        """Absolute times ``(t_lo, t_hi)`` the orbit tables cover (spacecraft positions AND
        light travel times: the C++ response zeroes any time outside either)."""
        span = getattr(self, "_orbit_span_cache", None)
        if span is None:
            args = getattr(self.orbits, "pycppdetector_args", None) if self.orbits is not None else None
            if args is None:
                span = (-np.inf, np.inf)
            else:
                sc_t0, sc_dt, sc_N, ltt_t0, ltt_dt, ltt_N = (float(v) for v in args[:6])
                span = (max(sc_t0, ltt_t0), min(sc_t0 + (sc_N - 1) * sc_dt, ltt_t0 + (ltt_N - 1) * ltt_dt))
            self._orbit_span_cache = span
        return span

    def _sparse_grid(self, H, chunk_start):
        """Response times (FEW clock) for ``response_grid="sparse"``: the holder's integrator
        knots over the window (plus margins) and every ``sparse_dt`` at most in between (the
        Doppler residual turns by up to ~1.4 rad/day at 25 mHz), ending a delay margin inside the
        trajectory. ``chunk_start`` adds nothing: the plunge chunk reads its time series from
        the dense kernel at the sample times (:meth:`_dense_td`), not from these splines (an
        80 s segment from there to past the stop used to be added, 11000 response points for a
        720-day window ending in a plunge)."""
        lo = self.data_t0 - self.t_start
        span = self.wdm.Nt * self.wdm.layer_dt
        a = max(0.0, lo - (self.DELAY_MARGIN + 2.0 * self.sparse_dt))
        b = lo + span + 2000.0                                # = the integration span (T_traj)
        # ... and inside the orbit tables: before/after them the kernel ZEROES the channel, and
        # a half-day spline across that edge rings into the window (measured: 4e-4 rad at the
        # first pixels on a laptop-trimmed L1 table)
        o_lo, o_hi = self._orbit_span()
        a = max(a, o_lo - self.t_start + self.DELAY_MARGIN)
        b = min(b, o_hi - self.t_start - self.DELAY_MARGIN)
        return sparse_response_grid(H.t_arr, a, b, self.sparse_dt, self.DELAY_MARGIN)

    def _knots_feed(self, H, few_args, t_fine, fly):
        """Response feed on ``t_fine`` from the knots holder ``H``: amplitudes splined over the
        knots, phases from the integrator's dense output (call right after H's FEW call: the
        integrator holds H's trajectory). Returns ``fly.prepare_feed_arrays(...)``."""
        from .emritdionfly import EMRITDIonFly

        t_end = float(np.asarray(H.t_arr)[-1])
        t_src = t_fine[t_fine <= t_end]
        if t_src[-1] < t_end - 1e-6:
            t_src = np.append(t_src, t_end)
        integ = self.few_gen.inspiral_generator.inspiral_generator
        tr = harmonic_tracks_from_holder(H, integ, t_src, a=few_args[2], xI0=few_args[5], phase_only=True)
        amp, ph = feed_from_tracks(tr, EMRITDIonFly.AMP_FACTOR)
        return fly.prepare_feed_arrays(t_src, amp, ph)

    def _dense_kernel(self, items, t_abs):
        """The TDDenseTDIonTheFly of ``items`` = [(dense inputs, (psi, lam, beta)), ...] at the
        absolute times ``t_abs`` ``(n_temp, N)``; returns ``(dense, params, sub_mkn, t_knots_abs,
        coeffs)``."""
        from ...response.tdionfly import TDDenseTDIonTheFly
        from .emritdionfly import EMRITDIonFly

        K = max(it[0][0].size for it in items)
        n_temp = len(items)
        t_k = np.empty((n_temp, K))
        C = np.zeros((n_temp, K - 1, 3, 8))
        n_k = np.zeros(n_temp, dtype=np.int32)
        mkn, are, aim, offs, par, tk_abs, Cs = [], [], [], [0], [], [], []
        for b, ((tk, Cb, mk, ar, ai), (psi, lam, beta)) in enumerate(items):
            nk = tk.size
            tk = self.t_start + tk                                 # FEW clock -> absolute (the response's clock)
            t_k[b, :nk], t_k[b, nk:] = tk, tk[-1]
            C[b, :nk - 1] = Cb
            n_k[b] = nk
            pad = ((0, 0), (0, K - nk), (0, 0))
            mkn.append(mk)
            are.append(np.pad(ar, pad))
            aim.append(np.pad(ai, pad))
            offs.append(offs[-1] + mk.shape[0])
            par.append((0.0, psi, lam, beta))
            tk_abs.append(tk)
            Cs.append(Cb)
        dense = TDDenseTDIonTheFly(
            t_abs, np.array(offs), np.concatenate(mkn),
            t_k, n_k, C, np.concatenate(are), np.concatenate(aim), amp_factor=EMRITDIonFly.AMP_FACTOR,
            tdi_config=self.tdi_config, orbits=self.orbits, force_backend=self.force_backend)
        return dense, np.array(par), np.concatenate(mkn), tk_abs, Cs

    def _dense_response(self, items, t_grid):
        """ONE TDDenseTDIonTheFly call for ``items`` = [(dense inputs, (psi, lam, beta)), ...].

        ``t_grid`` (FEW clock): one array shared by every template, or one array per template
        (padded to a common length inside its span, :func:`pad_grid`). With ``response_grid="sparse"`` the result is
        an :class:`ExactPhaseTDIOutput`, else the kernel's splined ``TDTDIOutput``.

        Raises:
            WaveformDomainError: When the (common) grid has fewer than
                :data:`MIN_RESPONSE_POINTS` points, before any kernel or spline runs."""
        n_temp = len(items)
        if isinstance(t_grid, (list, tuple)):
            N = max(int(np.size(g)) for g in t_grid)
        else:
            N = int(np.size(t_grid))
        if N < MIN_RESPONSE_POINTS:
            from ...utils.exceptions import WaveformDomainError

            raise WaveformDomainError(
                f"EMRI direct response grid of {N} point(s) (fewer than {MIN_RESPONSE_POINTS}): "
                "the trajectory is too short against the TDI delay margin for the response "
                "splines")
        if isinstance(t_grid, (list, tuple)):
            t_eval = np.stack([pad_grid(g, N) for g in t_grid])
        else:
            t_eval = np.tile(np.asarray(t_grid, dtype=float), (n_temp, 1))
        dense, par, mkn, tk_abs, Cs = self._dense_kernel(items, self.t_start + t_eval)
        # outside the trajectory the kernel holds the reference phase at the trajectory end (no jump)
        out = dense(par, return_spline=False)
        if self.response_grid != "sparse":
            return SplinedTDIOutput(out, mkn, dense.sub_temp_host)
        return ExactPhaseTDIOutput(out, tk_abs, Cs, mkn, dense.sub_temp_host)

    def _dense_td(self, item, ts, sum_subs=False, stop_taper=0.0):
        """The EXACT channel time series of one template's harmonics at the absolute sample times
        ``ts``: ``(num_sub, nch, ts.size)``, the real part of the dense kernel's raw TDI channels.

        The plunge chunk's input. An amplitude/phase spline cannot represent the channel where the
        signal stops (an in-window plunge: each TDI delay term drops out at its own time, a
        staircase over ~4 arm lengths); splined, it leaks broadband power whose low-frequency part
        the TDI combination would have suppressed -- under the instrument noise that was a
        template SNR of ~3900 against 56 for a 720-day window holding EMRI 1's plunge. Evaluated in
        blocks of at most EMRI_DIRECT_TAIL_BLOCK_ELEMS (sub, channel, sample) entries (2e7).
        ``sum_subs``: the sum over the harmonics, ``(nch, ts.size)``, summed inside the kernel.
        ``stop_taper``: the source-frame stop taper [s] (``TDDenseTDIonTheFly.channels``)."""
        xp = self.xp
        nch = self.tdi_config.nchannels
        S = 1 if sum_subs else int(item[0][2].shape[0])
        ts = xp.asarray(ts, dtype=float)
        out = xp.empty((S, nch, int(ts.size)))
        blk = max(1, int(float(os.environ.get("EMRI_DIRECT_TAIL_BLOCK_ELEMS", 2e7)) // (S * nch)))
        for b0 in range(0, int(ts.size), blk):
            tb = ts[b0:b0 + blk]
            dense, par = self._dense_kernel([item], tb[None, :])[:2]
            out[:, :, b0:b0 + tb.size] = xp.real(dense.channels(par, sum_subs=sum_subs, stop_taper=stop_taper))
        return out[0] if sum_subs else out

    def _dense_tail_fn(self, item, t_end_abs):
        """``tail_td(ts) -> (nch, ts.size)`` for the plunge chunk: the exact channels summed over
        the template's harmonics (:meth:`_dense_td`, ``sum_subs``; ``tail_td.summed`` makes every
        harmonic hand off at the same pixel, :func:`accumulate_harmonic_batch`), with the source
        amplitude tapered over the last ``plunge_taper_s`` of EMISSION time before the trajectory's
        end ``t_end_abs`` (``tail_td.taper``; ``stop_taper`` of ``TDDenseTDIonTheFly.channels``).

        Why: the trajectory stops abruptly at the plunge. Every delay term of a channel is the same
        truncated function, so the stop has no net T-like area, but SAMPLED, each term's cut falls
        at its own place between samples and the channel keeps an area that aliases into the
        lowest frequencies: EMRI 1's plunge scored SNR 2.7 in 0.25-1 mHz (dt 5 s; 90 at 20 s
        without any taper) where production, whose Lagrange interpolation band-limits the stop,
        has 2e-4. A taper on the emission time is the same for every term (the cancellation
        holds) and makes each term smooth at its end. A trajectory that outlasts the window is
        tapered past its end, where no pixel of the window looks."""
        taper = float(self.plunge_taper_s)

        def tail_td(ts):
            return self._dense_td(item, ts, sum_subs=True, stop_taper=taper)
        tail_td.taper = taper
        tail_td.summed = True
        return tail_td

    def _call_knots(self, few_args, few_kwargs, modes, reuse=None):
        """One template on the knots feed. ``reuse=(modes, chunk_start)``: the caller's
        :meth:`_mode_list` for these parameters was the LAST FEW call (its holder, tracks and
        the integrator's trajectory are current) -- skip the second FEW call."""
        from ...domains import WDMSignal
        from .emritdionfly import EMRITDIonFly

        wdm = self.wdm
        Nf, Nt, dt, ldt, ldf = wdm.Nf, wdm.Nt, wdm.data_dt, wdm.layer_dt, wdm.layer_df
        span = Nt * ldt
        n_all = np.arange(self.pixel_edge, Nt - self.pixel_edge)
        t_pix = self.data_t0 + n_all * ldt
        T_traj = self.data_t0 - self.t_start + span + 2000.0
        xp = self.xp
        acc = xp.zeros((self.tdi_config.nchannels, Nf, Nt))
        kw = dict(few_kwargs)
        if modes is not None:
            kw["mode_selection"] = modes
        if reuse is None:
            modes, chunk_start = self._mode_list(few_args, kw)
        else:
            modes, chunk_start = reuse
        H, tracks_pix, n_tr = self._last_holder, self._last_tracks, self._last_track_n
        if float(np.asarray(H.t_arr)[-1]) < self.data_t0 - self.t_start - self.DELAY_MARGIN:
            # the inspiral ends before the window (plus the TDI delay margin): no pixel sees
            # it, the production template is zero there too, and the response grid below
            # would be empty
            self.last_stats = dict(modes=len(modes), n_fine=0, feed="knots", chunk_start=chunk_start,
                                   lookup_pixels=0, chunk_pixels=0, dropped_pixels=0,
                                   ended_before_window=1)
            return WDMSignal(acc, wdm)
        t_fine = self._fine_grid(chunk_start)
        self.n_fine, self.last_t_fine = int(t_fine.size), t_fine
        fly = EMRITDIonFly(self.few_gen, self.orbits, self.tdi_config, dt, T_traj, self.t_start,
                           frame="icrs_special", t_fine_window=(self.data_t0, self.data_t0 + span), t_fine=t_fine)
        _, _, psi, lam, beta = fly.sky(few_args[7], few_args[8], few_args[9], few_args[10])
        if self.response == "dense":
            integ = self.few_gen.inspiral_generator.inspiral_generator
            din = dense_inputs_from_holder(H, integ, a=few_args[2], xI0=few_args[5])
            t_end = float(np.asarray(H.t_arr)[-1])
            t_resp = self._sparse_grid(H, chunk_start) if self.response_grid == "sparse" else t_fine
            if t_end < t_resp[-1] - 1.0:
                # stops in the window: end the response grid two delay margins after the stop
                # (as the spline feed's zero-amplitude continuation does); beyond it the channel
                # is exactly zero and its extracted phase meaningless, so it must not be splined
                t_resp = t_resp[t_resp <= t_end + 2 * 600.0 + 2 * self.fine_dt_plunge]
            self.n_fine = int(t_resp.size)
            out = self._dense_response([(din, (psi, lam, beta))], t_resp)
        else:
            feed = self._knots_feed(H, few_args, t_fine, fly)
            out = fly.run_response([tuple(feed) + (psi, lam, beta)])
        x = np.asarray(out.x.get() if hasattr(out.x, "get") else out.x)
        t_traj_end = float(np.asarray(H.t_arr)[-1])
        ok_t = (t_pix > x[:, 0].max()) & (t_pix < x[:, -1].min()) & (t_pix - self.t_start <= t_traj_end)
        n_ok, tt = n_all[ok_t], t_pix[ok_t]
        totals = dict(lookup_pixels=0, chunk_pixels=0, dropped_pixels=0)
        be = self._kernel_backend(out)
        if n_ok.size:
            tracks = None
            if tracks_pix is not None:                     # a harmonic may hand off (bound)
                pos = np.searchsorted(n_tr, n_ok)
                assert np.array_equal(n_tr[pos], n_ok)
                tracks = [subset_track(t, pos) for t in tracks_pix]
            nch = self.tdi_config.nchannels
            num_sub = int(np.shape(x)[0])
            if be is None:
                tracer = tracer_from_tof_output(out, xp.asarray(tt))
                assert tracks is None or tracer[0].shape[0] == len(tracks), (tracer[0].shape, len(tracks))
                kern = None
            else:
                tracer = None
                kern = self._lookup_fn(be, out, acc[None], np.zeros(num_sub, dtype=np.int32), n_ok, 0)

            if self.response == "dense":
                # the plunge chunk's TD straight from the dense kernel at the sample times: exact
                # through the stop (_dense_td), not the amplitude/phase splines; the source
                # tapered over the last plunge_taper_s of emission time (_dense_tail_fn)
                tail_td = self._dense_tail_fn((din, (psi, lam, beta)), self.t_start + t_traj_end)
            else:
                def tail_td(ts, out=out, x_lo=float(x[:, 0].max()), x_hi=float(x[:, -1].min())):
                    live = (ts > x_lo) & (ts < x_hi)
                    td = xp.zeros((x.shape[0], nch, ts.size))
                    if bool(xp.any(live)):
                        td[:, :, live] = xp.asarray(out.eval_tdi(ts[live]))
                    return td

            totals = accumulate_harmonic_batch(
                acc, self.table, tracks, tracer, n_ok, tail_td, Nf=Nf, Nt=Nt, dt=dt, layer_dt=ldt,
                layer_df=ldf, t0=self.data_t0, Nt_sub=self.Nt_sub, num_m_layers=self.num_m_layers,
                fdot_axis_max=self.fdot_axis_max, pixel_edge=self.pixel_edge, backend=self.force_backend,
                num_sub=num_sub, lookup_kernel=kern)
        self.last_stats = dict(modes=len(modes), n_fine=self.n_fine, feed="knots", chunk_start=chunk_start,
                               lookup="python" if be is None else "kernel", **totals)
        return WDMSignal(acc, wdm)

    def __call__(self, *few_args, **few_kwargs):
        """ONE template on the full WDM grid.

        With ``feed="knots"`` and no fixed ``n_fine`` (the default) this is
        :meth:`_call_knots`: one FEW call on the knots, one response, the lookup and (when a
        harmonic hands off) the plunge chunk. Otherwise FEW runs on the fine grid and the
        harmonics go through :class:`~lisatools.sources.emri.emritdionfly.EMRITDIonFly` in
        ``mode_batch`` blocks. Updates :attr:`last_stats`.

        Args:
            *few_args: The 14 FEW waveform-basis parameters ``(M, mu, a, p0, e0, xI0, dist,
                qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)``.
            **few_kwargs: FEW call kwargs (``mode_selection_threshold``, or an explicit
                ``mode_selection`` list of ``(l, m, k, n)``).

        Returns:
            WDMSignal: ``(nch, Nf, Nt)`` on ``wdm_set`` (all layers, ignoring ``f_band``).

        Note:
            The mode-list / holder FEW calls (:meth:`_mode_list`, :meth:`_few_holder`) are not
            wrapped in :func:`~lisatools.sources.emri.domain.few_domain_guard` here, so an
            out-of-domain row can raise FEW's own ``ValueError`` / ``AssertionError``;
            :meth:`batch` (and so
            :class:`~lisatools.sources.emri.direct_signal_gen.EMRIDirectWDMSignalGen`) wraps
            them into ``WaveformDomainError``.
        """
        from ...domains import WDMSignal
        from .emritdionfly import EMRITDIonFly

        if self.feed == "knots" and self.n_fine_fixed is None:
            kw = dict(few_kwargs)
            return self._call_knots(few_args, kw, kw.pop("mode_selection", None))

        wdm = self.wdm
        Nf, Nt, dt, ldt, ldf = wdm.Nf, wdm.Nt, wdm.data_dt, wdm.layer_dt, wdm.layer_df
        span = Nt * ldt
        n_all = np.arange(self.pixel_edge, Nt - self.pixel_edge)
        t_pix = self.data_t0 + n_all * ldt
        T_traj = self.data_t0 - self.t_start + span + 2000.0
        xp = self.xp
        acc = xp.zeros((self.tdi_config.nchannels, Nf, Nt))
        few_kwargs = dict(few_kwargs)
        modes = few_kwargs.pop("mode_selection", None)      # explicit (l, m, k, n) list, e.g. gates
        totals = dict(lookup_pixels=0, chunk_pixels=0, dropped_pixels=0)
        H = tracks_all = n_tr = None
        t_fine, chunk_start = None, "unknown"
        if modes is None and self.n_fine_fixed is None:
            # ONE FEW call on the coarse grid gives the modes, the holder the response is fed
            # from and the harmonic tracks; a second (explicit-mode) call only when a harmonic
            # hands off to the plunge chunk inside the window and the grid must be densified
            modes, chunk_start = self._mode_list(few_args, few_kwargs)
            H, tracks_all, n_tr = self._last_holder, self._last_tracks, self._last_track_n
            t_fine = self._fine_grid(chunk_start)
            if chunk_start is not None:
                H = self._few_holder(few_args, few_kwargs, t_fine, mode_selection=modes)
                n_tr, t_rel = self._pixel_times(H)
                integ = self.few_gen.inspiral_generator.inspiral_generator
                tracks_all = harmonic_tracks_from_holder(H, integ, t_rel, a=few_args[2], xI0=few_args[5])
            self.n_fine = int(t_fine.size)
        else:                                               # explicit modes / fixed n_fine: FEW per batch
            if modes is None:
                modes, chunk_start = self._mode_list(few_args, few_kwargs)
            self.n_fine = self.n_fine_fixed if self.n_fine_fixed is not None else max(64, int(span / self.fine_dt_plunge))
        self.last_t_fine = t_fine                             # None: uniform n_fine over the window
        few_kwargs.pop("mode_selection_threshold", None)

        nm = len(modes)
        batch = nm if (self.mode_batch is None or self.mode_batch <= 0) else int(self.mode_batch)
        for j in range(0, nm, batch):
            idx = np.arange(j, min(j + batch, nm))
            fly = EMRITDIonFly(self.few_gen, self.orbits, self.tdi_config, dt, T_traj, self.t_start,
                               frame="icrs_special", n_fine=None if t_fine is not None else self.n_fine,
                               t_fine_window=(self.data_t0, self.data_t0 + span), t_fine=t_fine)
            if H is not None:
                Hb = H if idx.size == nm else slice_holder(H, idx)
                out = fly(*few_args, holder=Hb, **few_kwargs)
            else:
                out = fly(*few_args, mode_selection=[modes[i] for i in idx], **few_kwargs)
                Hb = fly.last_holder
            x = np.asarray(out.x.get() if hasattr(out.x, "get") else out.x)
            t_traj_end = float(np.asarray(Hb.t_arr)[-1])
            ok_t = (t_pix > x[:, 0].max()) & (t_pix < x[:, -1].min()) & (t_pix - self.t_start <= t_traj_end)
            n_ok, tt = n_all[ok_t], t_pix[ok_t]
            if n_ok.size == 0:
                continue
            if tracks_all is not None:
                pos = np.searchsorted(n_tr, n_ok)
                assert np.array_equal(n_tr[pos], n_ok)
                tracks = [subset_track(tracks_all[i], pos) for i in track_rows(H, idx)]
            else:
                integ = self.few_gen.inspiral_generator.inspiral_generator
                tracks = harmonic_tracks_from_holder(Hb, integ, tt - self.t_start, a=few_args[2], xI0=few_args[5])
            tracer = tracer_from_tof_output(out, xp.asarray(tt))
            assert tracer[0].shape[0] == len(tracks), (tracer[0].shape, len(tracks))
            x_lo, x_hi = x[:, 0][:, None, None], x[:, -1][:, None, None]

            def tail_td(ts, out=out, x_lo=float(x_lo.max()), x_hi=float(x_hi.min())):
                live = (ts > x_lo) & (ts < x_hi)
                td = xp.zeros((x.shape[0], tracer[0].shape[1], ts.size))
                if bool(xp.any(live)):
                    td[:, :, live] = xp.asarray(out.eval_tdi(ts[live]))
                return td

            st = accumulate_harmonic_batch(
                acc, self.table, tracks, tracer, n_ok, tail_td, Nf=Nf, Nt=Nt, dt=dt, layer_dt=ldt,
                layer_df=ldf, t0=self.data_t0, Nt_sub=self.Nt_sub, num_m_layers=self.num_m_layers,
                fdot_axis_max=self.fdot_axis_max, pixel_edge=self.pixel_edge, backend=self.force_backend)
            for key in totals:
                totals[key] += st[key]
            del out, fly
        self.last_stats = dict(modes=nm, n_fine=self.n_fine,
                               chunk_start=None if chunk_start == "unknown" else chunk_start, **totals)
        return WDMSignal(acc, wdm)

    def batch(self, rows, chunk_rows=16, consume=None, skip_domain_errors=False, **few_kwargs):
        """Templates for many parameter rows, ``chunk_rows`` per response call (bounds GPU memory:
        one 6-month production-grid template is ~150 MB of WDM coefficients).

        ``consume(row_indices, arr)`` (optional) receives each chunk's ``(n, nch, n_m, Nt)`` array
        and nothing is kept; otherwise the full ``(n_rows, nch, n_m, Nt)`` array is returned
        (``n_m = m_hi - m_lo`` with ``f_band``, else ``Nf``). ``last_stats`` sums the chunks' stats.

        FEW's out-of-domain failures are raised as
        :class:`~lisatools.utils.exceptions.WaveformDomainError` (``few_domain_guard``). With
        ``skip_domain_errors=True`` such a row is left all-zero instead, the others are built,
        and its index (into ``rows``) is listed in ``last_failed_rows``.

        Args:
            rows (iterable): FEW waveform-basis rows (14 parameters each).
            chunk_rows (int, optional): Rows per :meth:`_batch_chunk` call. Default ``16``.
            consume (callable, optional): ``consume(row_indices, arr)`` per chunk.
            skip_domain_errors (bool, optional): Zero-fill FEW-refused rows instead of raising.
                Default ``False``.
            **few_kwargs: FEW call kwargs for every row.

        Returns:
            xp.ndarray | None: ``(n_rows, nch, n_m, Nt)`` float64 on the backend, or ``None``
            with ``consume``.

        Raises:
            WaveformDomainError: For a FEW-refused row when ``skip_domain_errors`` is ``False``.
        """
        rows = list(rows)
        parts, tot, failed = [], {}, []
        for i in range(0, len(rows), max(1, int(chunk_rows))):
            idx = list(range(i, min(i + int(chunk_rows), len(rows))))
            arr = self._batch_chunk([rows[k] for k in idx], skip_domain_errors=skip_domain_errors,
                                    **few_kwargs)
            failed += [idx[k] for k in self.last_failed_rows]
            for k, v in self.last_stats.items():
                if k == "n_response":                         # a per-call grid length, not a count
                    tot[k] = max(tot.get(k, 0), v)
                else:
                    tot[k] = tot.get(k, 0) + v if isinstance(v, (int, float)) else v
            if consume is not None:
                consume(idx, arr)
                del arr
            else:
                parts.append(arr)
        self.last_stats = tot
        self.last_failed_rows = failed
        if consume is not None:
            return None
        return parts[0] if len(parts) == 1 else self.xp.concatenate(parts, axis=0)

    def _batch_chunk(self, rows, skip_domain_errors=False, **few_kwargs):
        """Many templates with ONE TDI-on-the-fly response call, ONE tracer and ONE lookup.

        ``rows``: parameter rows (as for ``__call__``). FEW runs once per row (it is a
        single-template generator); every row whose harmonics stay on the lookup feeds one
        response kernel launch (num_sub x n_rows blocks fill the GPU), then ONE fused lookup-sum
        kernel (``lookup="kernel"``) or the Python tracer + table call + scatter-add in blocks of
        harmonics, into ``(n_rows, nch, n_m, Nt)`` (layers ``f_band``). A row that hands off
        to the plunge chunk is built alone with ``__call__``. Returns the array (backend xp).
        A row FEW refuses (WaveformDomainError) is left zero and listed in ``last_failed_rows``
        when ``skip_domain_errors``; otherwise the error propagates.
        """
        from ...utils.exceptions import WaveformDomainError
        from .domain import few_domain_guard
        from .emritdionfly import EMRITDIonFly

        self.last_failed_rows = []
        wdm = self.wdm
        Nf, Nt, dt, ldt, ldf = wdm.Nf, wdm.Nt, wdm.data_dt, wdm.layer_dt, wdm.layer_df
        span = Nt * ldt
        xp = self.xp
        n_all = np.arange(self.pixel_edge, Nt - self.pixel_edge)
        t_pix = self.data_t0 + n_all * ldt
        T_traj = self.data_t0 - self.t_start + span + 2000.0
        m_lo, m_hi = self._band()
        out_arr = xp.zeros((len(rows), self.tdi_config.nchannels, m_hi - m_lo, Nt))
        few_kwargs = dict(few_kwargs)
        t_fine = self._fine_grid(None)
        fly = EMRITDIonFly(self.few_gen, self.orbits, self.tdi_config, dt, T_traj, self.t_start,
                           frame="icrs_special", t_fine_window=(self.data_t0, self.data_t0 + span), t_fine=t_fine)
        feeds, n_subs, rows_in, t_ends, grids = [], [], [], [], []
        stats = dict(rows=len(rows), alone=0, subs=0, failed=0)
        for r, p in enumerate(rows):
            try:
                with few_domain_guard():
                    modes, chunk_start = self._mode_list(p, few_kwargs)
                    if chunk_start is not None:              # plunge chunk: build this one alone
                        if self.feed == "knots" and self.n_fine_fixed is None:
                            # the _mode_list above was the last FEW call: reuse it
                            kw = dict(few_kwargs)                    # as __call__ passes them
                            one = self._call_knots(p, kw, kw.pop("mode_selection", None),
                                                   reuse=(modes, chunk_start))
                        else:
                            one = self(*p, **few_kwargs)
                        out_arr[r] = xp.asarray(one.arr)[:, m_lo:m_hi]
                        stats["alone"] += 1
                        continue
            except WaveformDomainError:
                if not skip_domain_errors:
                    raise
                self.last_failed_rows.append(r)
                stats["failed"] += 1
                continue
            H = self._last_holder
            _, _, psi, lam, beta = fly.sky(p[7], p[8], p[9], p[10])
            if self.response == "dense":
                integ = self.few_gen.inspiral_generator.inspiral_generator
                din = dense_inputs_from_holder(H, integ, a=p[2], xI0=p[5])
                feeds.append((din, (psi, lam, beta)))
                n_subs.append(int(din[2].shape[0]))
                if self.response_grid == "sparse":
                    grids.append(self._sparse_grid(H, None))
            else:
                if self.feed == "knots":
                    t_in, amp, ph, t_tdi = self._knots_feed(H, p, t_fine, fly)
                else:
                    t_in, amp, ph, t_tdi = fly.prepare_feed(H, True)
                feeds.append((t_in, amp, ph, t_tdi, psi, lam, beta))
                n_subs.append(int(np.shape(amp)[0]))
            t_ends.append(float(np.asarray(H.t_arr)[-1]))
            rows_in.append(r)
        if not feeds:
            self.last_stats = stats
            return out_arr
        if self.response == "dense":
            out = self._dense_response(feeds, grids if self.response_grid == "sparse" else t_fine)
            stats["n_response"] = int(np.shape(out.x)[-1])
        else:
            out = fly.run_response(feeds)
        x = np.asarray(out.x.get() if hasattr(out.x, "get") else out.x)
        ok_t = (t_pix > x[:, 0].max()) & (t_pix < x[:, -1].min()) & (t_pix - self.t_start <= min(t_ends))
        n_ok, tt = n_all[ok_t], t_pix[ok_t]
        # every row here has no harmonic handing off anywhere inside its trajectory's pixels
        # (_chunk_start is None), so none does on n_ok: the whole range is looked up
        sub_row = np.repeat(np.asarray(rows_in), n_subs)
        S, P = int(sub_row.size), int(tt.size)
        nch = self.tdi_config.nchannels
        kw = dict(Nf=Nf, Nt=Nt, dt=dt, layer_dt=ldt, layer_df=ldf, t0=self.data_t0, Nt_sub=self.Nt_sub,
                  num_m_layers=self.num_m_layers, fdot_axis_max=self.fdot_axis_max,
                  pixel_edge=self.pixel_edge, backend=self.force_backend, m_lo=m_lo)
        tot = dict(lookup_pixels=0, chunk_pixels=0, dropped_pixels=0)
        be = self._kernel_backend(out)
        if be is not None:
            # ONE fused kernel for every harmonic of every row: nothing per pixel materialised
            st = accumulate_harmonic_batch(out_arr, self.table, None, None, n_ok, None, sub_row=sub_row,
                                           num_sub=S, lookup_kernel=self._lookup_fn(be, out, out_arr, sub_row, n_ok, m_lo),
                                           **kw)
            tot.update({k: st[k] for k in tot})
            stats.update(subs=S, mode_blocks=1, lookup="kernel", **tot)
            self.last_stats = stats
            return out_arr
        # Python: tracer + lookup in blocks of harmonics (modes): every per-block array is (block,
        # nch, pixels), whatever the number of rows and the duration
        # (EMRI_DIRECT_MODE_BLOCK_ELEMS elements per such array, default 2e7 = 160 MB)
        if isinstance(out, SplinedTDIOutput):
            budget = float(os.environ.get("EMRI_DIRECT_MODE_BLOCK_ELEMS", 2e7))
            blk = max(1, int(budget // max(1, nch * P)))
        else:                                                # spline response: no subset evaluation
            blk = S
        for s0 in range(0, S, blk):
            subs = np.arange(s0, min(S, s0 + blk))
            tracer = tracer_from_tof_output(out, xp.asarray(tt), subs=subs if blk < S else None)
            assert tracer[0].shape[0] == subs.size, (tracer[0].shape, subs.size)
            st = accumulate_harmonic_batch(out_arr, self.table, None, tracer, n_ok, None, sub_row=sub_row[subs], **kw)
            for k in tot:
                tot[k] += st[k]
            del tracer
        stats.update(subs=S, mode_blocks=-(-S // blk), lookup="python", **tot)
        self.last_stats = stats
        return out_arr
