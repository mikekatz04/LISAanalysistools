"""Observable-basis adapter for the warm-start pipeline.

ONE place the fitter (:mod:`.fit_from_store`), the referee
(:mod:`.match_referee`), the SNR gate (:mod:`.opt_snr`), the consensus filter
(:func:`.self_refit.consensus_filter`), the para-search epoch writer and the
proposal (:class:`.proposal.WarmStartComponents`) all get their
:class:`~lisatools.sampling.gb_observable_basis.GBObservableFiberBasis` from,
so the stages can never drift apart on Tobs, shear or the fiber coordinate.
A fitted npz records the map in its ``meta["map_params"]``
(:func:`map_params_from_map`) and every reader rebuilds it from there
(:func:`build_map_from_stored_params`).

The map is a 9-to-9 bijection (see the class docstring); indices 3..7 pass
through unchanged, so the pipeline's ``CIRCULAR_COLS`` / ``COS_IOTA_COL``
constants are valid in BOTH bases::

    index   sampling (astro)      internal (observable)
    0       dist                  lnA
    1       f0                    f_mid
    2       Mc                    fdot
    3..7    phi0 cos_iota psi alpha sin_delta   (identical)
    8       fdot_astro_ratio      Mc   (the fiber)

Units: the sampling ``f0`` is in mHz, the observable ``f_mid`` in Hz and
``fdot`` in Hz/s; the sky columns are ICRS (``alpha`` = RA, ``sin_delta``).

**Measure convention.** ``GBObservableFiberBasis.log_jacobian(y)`` is
``ln|dy/dz|`` evaluated at a SAMPLING point ``y``.  A density fitted in
``z`` therefore transports to the sampling basis as::

    ln q_y(y) = ln q_z(z(y)) - log_jacobian(y)

-- a MINUS, because ``q_y = q_z * |dz/dy|``.  Every consumer goes through
:func:`log_density_to_sampling` rather than writing the sign out by hand.
"""

from __future__ import annotations

import numpy as np

from lisatools.sampling.gb_observable_basis import (
    GB_INTERNAL_BASIS,
    GBObservableFiberBasis,
)

#: Observable column names, in map order.
OBSERVABLE_COLUMN_NAMES = list(GB_INTERNAL_BASIS)


def build_map(transform_container, *, Tobs, shear=0.5, fiber_coord="Mc"):
    """Build the observable map for a run.

    ``Tobs`` is the run's ``1.0 / df`` -- NOT ``basis_settings.Tobs``, which
    does not exist on ``FDSettings`` and has broken every FD-domain GB flow
    once. This mirrors the in-model caller in
    ``globalfit/moves/gbspecialstretch.py``.

    Args:
        transform_container: The GB transform container (or a
            :class:`BasisContainer`); only its ``input_basis`` (the sampling
            column names) is read.
        Tobs (float): Observation time [s] the map's ``f_mid`` / ``fdot``
            are defined at (for a warm-start fit: the SOURCE run's Tobs).
        shear (float, optional): The map's shear. Default ``0.5``.
        fiber_coord (str, optional): The fiber coordinate. Default ``"Mc"``.

    Returns:
        GBObservableFiberBasis: The map (``to_internal`` / ``from_internal`` /
        ``log_jacobian``).
    """
    return GBObservableFiberBasis(
        transform_container, Tobs=float(Tobs), shear=float(shear),
        fiber_coord=str(fiber_coord),
    )


def map_params_from_map(m) -> dict:
    """The parameters needed to rebuild ``m`` without the source store.

    Args:
        m (GBObservableFiberBasis): A built map.

    Returns:
        dict: ``{"Tobs", "shear", "fiber_coord", "input_basis"}``, JSON-safe;
        stored as ``meta["map_params"]`` of every observable-basis npz.
    """
    return {
        "Tobs": float(m.Tobs),
        "shear": float(m.shear),
        "fiber_coord": str(m.fiber_coord),
        "input_basis": list(m.input_basis),
    }


def build_map_from_params(transform_container, params: dict):
    """Rebuild a map from :func:`map_params_from_map` output.

    Refuses when the container's sampling basis differs from the one the
    params were written with -- a silent mismatch would mis-index every
    column rather than fail.

    Args:
        transform_container: The run's GB transform container (its
            ``input_basis`` is checked against ``params["input_basis"]``).
        params: :func:`map_params_from_map` output. ``shear`` and
            ``fiber_coord`` default to ``0.5`` / ``"Mc"`` when absent.

    Returns:
        GBObservableFiberBasis: The rebuilt map.

    Raises:
        ValueError: When the two sampling bases differ.
    """
    want = list(params["input_basis"])
    got = list(getattr(transform_container, "input_basis", []) or [])
    if got != want:
        raise ValueError(
            f"map_params were written for input_basis {want} but this "
            f"container has {got}; refusing to build a mis-indexed map."
        )
    return build_map(
        transform_container,
        Tobs=params["Tobs"],
        shear=params.get("shear", 0.5),
        fiber_coord=params.get("fiber_coord", "Mc"),
    )


class BasisContainer:
    """Stand-in transform container carrying just ``input_basis``.

    ``GBObservableFiberBasis`` pins nothing per leaf for GB, so this is the
    only attribute it reads. Stored ``map_params`` carry their own
    ``input_basis``, which makes :func:`build_map_from_stored_params`
    possible without a live run object.

    Args:
        input_basis (sequence of str): The sampling column names.

    Attributes:
        input_basis (list of str): The sampling column names.
    """

    def __init__(self, input_basis):
        self.input_basis = list(input_basis)


def build_map_from_stored_params(params: dict):
    """Rebuild the map from stored ``map_params`` alone.

    The params carry the sampling basis they were written against, so no
    run object is needed. Pass a REAL container to
    :func:`build_map_from_params` instead when one is available -- that
    additionally checks the run's basis against the file's.

    Args:
        params: A stored ``meta["map_params"]`` dict.

    Returns:
        GBObservableFiberBasis: The rebuilt map.
    """
    return build_map_from_params(BasisContainer(params["input_basis"]),
                                 params)


def component_means_sampling(d, meta: dict, transform_container=None):
    """Component means in the SAMPLING basis, whichever format ``d`` is.

    This is the seam the SNR gate uses (the referee does the same mapping
    inline): both build real waveforms, which need astro columns whatever
    basis the fit ran in.

    Args:
        d (mapping): An open fitter / refereed npz.
        meta: Its decoded ``meta`` JSON. ``meta["basis"] != "observable"``
            (or absent) = a legacy sampling-basis set.
        transform_container (optional): A real GB transform container; when
            given, the map is rebuilt with the basis check of
            :func:`build_map_from_params`. Default ``None`` = from the stored
            params alone.

    Returns:
        tuple[np.ndarray, GBObservableFiberBasis | None]: ``(means_sampling,
        obs_map)``. ``means_sampling`` is ``(n_components, 9)`` (the
        per-component ``gmm_means`` mapped back, or the legacy ``means``);
        ``obs_map`` is ``None`` for a legacy set.
    """
    import numpy as _np

    if str(meta.get("basis", "sampling")) != "observable":
        return _np.asarray(d["means"], dtype=float), None
    params = meta["map_params"]
    m = (build_map_from_stored_params(params) if transform_container is None
         else build_map_from_params(transform_container, params))
    z = _np.asarray(d["gmm_means"], dtype=float)
    return _np.asarray(m.from_internal(z), dtype=float), m


def log_density_to_sampling(log_q_z, x_sampling, obs_map, leaf_inds=None):
    """Transport an observable-basis log density to the sampling basis.

    ``log_q_z`` is ``ln q_z(z)`` at ``z = obs_map.to_internal(x_sampling)``;
    the return is ``ln q_y(x_sampling)`` for the SAME distribution.

    The sign is the whole reason this function exists.  ``log_jacobian`` is
    ``ln|dy/dz|``, and a density transforms with the determinant of the
    INVERSE map, so the correction SUBTRACTS::

        q_y(y) = q_z(z(y)) * |dz/dy| = q_z(z(y)) / |dy/dz|

    Getting it backwards leaves ``rvs`` and ``logpdf`` inconsistent by
    ``2 * log_jacobian``, which varies across a component and so biases the
    RJ birth/death factors rather than cancelling as a constant would.

    DISPATCHES ON THE INPUTS (2026-09-18). This used to force both terms
    through ``np.asarray``, which raises ``TypeError: Implicit conversion
    to a NumPy array is not allowed`` the moment either side is a cupy
    array -- and on the production path both are: ``BandSorter`` scores
    ``rj_prop.logpdf`` on device-resident band coordinates, and
    ``log_jacobian`` is itself ``xp``-generic, so it hands back cupy. The
    laptop tests never saw it because there is no cupy there.

    Args:
        log_q_z (array-like): ``(n,)`` observable-basis log density.
        x_sampling (array-like): ``(n, 9)`` the same points in the sampling
            basis.
        obs_map (GBObservableFiberBasis): The map the density was fitted in.
        leaf_inds (optional): Forwarded to ``obs_map.log_jacobian``. Default
            ``None``.

    Returns:
        ``xp.ndarray``: ``(n,)`` float64 ``ln q_y``; cupy when either input
        is a cupy array, else numpy.
    """
    from ...utils.utility import get_array_module

    # cupy wins if EITHER side is on device: a host->device promotion is
    # free, the reverse is the exception above. get_array_module RAISES on
    # a list/scalar, which is a legitimate input here -- those are host.
    xp = np
    for a in (log_q_z, x_sampling):
        try:
            mod = get_array_module(a)
        except ValueError:
            continue
        if mod is not np:
            xp = mod
            break
    return (xp.asarray(log_q_z, dtype=xp.float64)
            - xp.asarray(obs_map.log_jacobian(x_sampling, leaf_inds),
                         dtype=xp.float64))
