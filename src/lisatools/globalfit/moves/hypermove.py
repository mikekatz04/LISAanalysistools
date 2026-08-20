"""Trans-model moves for population inference inside the global fit.

What this module is for
-----------------------
:class:`HyperMove` samples a discrete model index :math:`M` -- which of several
competing astrophysical population models the galactic binaries and the galactic
foreground are drawn from -- jointly with the source parameters
:math:`\\{\\vec\\theta\\}_{k_1}` and the stochastic parameters :math:`\\Sigma`. It is the
move to use when several populations are supplied as alternative *priors*
(``HyperGBConfig``/``HyperGalForConfig`` with one normalising-flow configuration per
model) and the question is which of them the data prefer. It is not needed, and should
not be added, when a single population prior is in use.

Use it inside a :class:`~lisatools.globalfit.moves.globalfitmove.GFCombineMove` together
with the in-model moves, as ``build_hyper_moves`` sets up: this move only ever changes
the model index, and relies on the other moves to sample
:math:`(\\{\\vec\\theta\\}_{k_1}, \\Sigma)`.

What it computes
----------------
From the dependent-thinning form of the population posterior, everything that depends
on :math:`M` at fixed :math:`(\\{\\vec\\theta\\}_{k_1}, \\Sigma, k_1)` is

.. math::

    \\ell_m = \\underbrace{\\sum_{i=1}^{k_1} \\ln \\mathcal N(\\vec\\theta_i \\mid m)}_{\\rm intensity}
              \\; - \\; \\underbrace{\\hat N_1(\\Sigma, m)}_{\\rm expected\\ count}
              \\; + \\; \\underbrace{\\ln p(\\Sigma \\mid m)}_{\\rm stochastic}
              \\; + \\; \\ln p(m),
    \\qquad
    \\mathcal N(\\vec\\theta \\mid m) = \\hat N_{\\rm tot}(m)\\, p_{\\rm pop}(\\vec\\theta \\mid m),

and the acceptance ratio of a symmetric model proposal is
:math:`\\ln\\alpha = \\ell_{m'} - \\ell_m`. :meth:`HyperMove.compute_model_terms`
evaluates :math:`\\ell_m` and each of its parts for *all* models at once, so the ratio
is a difference of two entries of one array and the whole decomposition is available
for diagnostics.

Assumptions
-----------
* The likelihood :math:`p(\\vec d \\mid \\{\\vec\\theta\\}_{k_1}, \\Sigma)` and the
  :math:`1/k_1!` factor carry no model dependence, hence do not appear in
  :math:`\\ell_m`. This is what makes the move cheap: no waveform is regenerated.
* Resolvability :math:`\\alpha(\\vec\\theta \\mid \\Sigma)` is a fixed detector and
  pipeline property (:math:`\\rho_{\\rm thr}`, :math:`\\sigma` model independent), so it
  cancels in the ratio and is absent by construction.
* Every leaf of a walker shares one model index, stored in ``coords["hyper"][..., 0, 0]``.
* :math:`\\hat N_{\\rm tot}(m)` is the size of catalogue ``m`` and is fixed at
  construction; :math:`\\hat N_1(\\Sigma, m)` is recomputed from the walkers' current
  PSDs at every proposal, with a hard SNR cut (see :meth:`compute_expected_resolved_counts`).
* ``acs`` holds one PSD per walker, so :math:`\\hat N_1` has no temperature dependence.
* Eryn tempers the likelihood only, and the model index enters through the prior alone.
  Untempered, the ratio above is therefore the correct Metropolis-Hastings ratio for the
  tempered target *and* is identical at every temperature, so the ladder provides no help
  whatever in crossing a model barrier. Passing ``tempering`` weights the model-dependent
  part of the prior by :math:`\\beta` as well, which is what makes the hot temperatures
  indifferent between models; see :meth:`HyperMove.temper_model_terms`,
  :meth:`HyperMove.run_hyper_tempering` and ``_dev/prior_tempering.md``.

Swapping between temperatures
-----------------------------
With tempering on, :meth:`HyperMove.run_hyper_tempering` exchanges the model, resolved
and stochastic branches between adjacent temperatures so that what the hot temperatures
find reaches the cold chain. The instrument-noise branch stays where it is, so the
likelihood does not travel with the exchange and has to be recomputed --
:class:`TemperatureLikelihood` is the evaluator that does it, and the reason this is the
one move that needs a likelihood at parameters drawn from more than one temperature.

A swap that reaches temperature 0 replaces the state the *shared* analysis containers'
residual was built from, so that residual is rebuilt before the move returns. This is
the only way the move is visible to the rest of the pipeline; everything else it does is
confined to the state it was handed.

Diagnostics
-----------
Pass ``diagnostics_file`` to append :math:`\\ell_m`, decomposed into intensity,
expected count and stochastic parts, for every model, walker, temperature and
iteration; :mod:`lisatools.globalfit.hyperdiagnostics` reads such a file back and
turns it into per-source numbers and a Rao-Blackwellised model posterior. Because
:math:`\\ln\\alpha` scales as :math:`k_1` times a per-source term, a zero acceptance
rate is expected whenever the populations are well separated at production
:math:`k_1`; the recording is what tells a correct rejection from a broken move. See
``_dev/why_the_model_index_does_not_jump.md``.
"""

import numpy as np
import logging
from copy import deepcopy
from typing import Dict, Any, List, Sequence
from eryn.moves import Move
from .globalfitmove import GlobalFitMove
from ..state import GFState
from ..hyperdiagnostics import HyperMoveRecorder
from ..priors.tempering import PopulationTempering
from ...analysiscontainer import AnalysisContainerArray
from ..priors.sourceconfigs import BaseSourceConfig
from ...utils.typing import NDArrayLike
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_N

logger = logging.getLogger(__name__)


class TemperatureLikelihood:
    """:math:`\\ln L` at an arbitrary combination of branch states and temperatures.

    Why this exists, and why it lives here
    --------------------------------------
    The shared analysis containers hold **one residual per walker**, built from the
    coldest temperature's noise parameters, and every temperature reads that same
    residual and that same spectrum -- the supplemental index that reaches them maps
    :math:`(t, w) \\mapsto w` for every :math:`t`. The temperature dependence of the
    galactic-binary likelihood therefore enters only through which sources a temperature
    holds and where, never through the data they are compared against.

    That is sufficient for every move that works *within* a temperature, which is all of
    them except this one. :class:`HyperMove` alone proposes exchanging states *between*
    temperatures, and to score such a proposal it needs
    :math:`\\ln L(\\{\\vec\\theta\\}, \\Sigma, \\Pi)` at combinations of resolved
    parameters, foreground parameters and instrument-noise parameters that come from
    different temperatures. No such evaluator exists in the engine: the object handed to
    a move carries the container array, a map function and the random state, and no
    likelihood function at all.

    This class is that evaluator, and it is deliberately owned by the model move rather
    than built into the shared containers. Rebuilding those at
    :math:`(n_{\\rm temps}, n_{\\rm walkers})` would change the likelihood seen by the
    galactic-binary and foreground moves at every hot temperature, making every existing
    chain incomparable, for a capability only this move uses. See section 7 of
    ``_dev/prior_tempering.md``.

    How it works
    ------------
    1. **Recover the data.** The shared containers hold the residual, i.e. the data minus
       the coldest temperature's templates. Adding those templates back recovers the data
       itself. In this configuration the galactic binaries are the only signal branch, so
       nothing else has been subtracted. Done once per proposal by :meth:`refresh_data`.
    2. **Form the residual** for a requested state by generating its templates and
       subtracting them from that data.
    3. **Form the spectrum** for the requested noise parameters through the same
       sensitivity backend call the foreground move makes.
    4. **Evaluate**, reusing a scratch container array so that the working set stays at
       ``batch_size`` residuals rather than the whole grid.
    5. **Put the shared residual back in step** when a swap has replaced the coldest
       temperature's state, via :meth:`restore_residual`. The shared containers are what
       every *other* move reads, so this is the one way this move is visible outside
       itself, and it is not optional.

    Both sides of a swap ratio must be evaluated through this object. Taking the current
    value from the stored likelihood instead would mix the shared-residual convention
    with a self-consistent one, and their difference would not be a ratio of anything.

    Args:
        acs: The shared analysis containers, one per walker.
        wave_gen: Waveform generator for the resolved sources.
        waveform_kwargs: Runtime kwargs for ``wave_gen``.
        resolved_transform: Transform from the sampling basis of the resolved branch to
            the waveform basis; ``None`` if the coordinates are already in it.
        sensitivity_backend: Called as ``backend(name, psd_params, galfor_params=...)``
            to build a spectrum, exactly as the foreground move calls it.
        psd_transform: Transform for the instrument-noise parameters, or ``None``.
        galfor_transform: Transform for the foreground parameters, or ``None``.
        batch_size: Configurations evaluated at once. Defaults to twice the number of
            walkers, which is one adjacent pair of temperatures.
    """

    def __init__(
        self,
        acs: AnalysisContainerArray,
        wave_gen: GBGPU,
        waveform_kwargs: Dict[str, Any],
        resolved_transform: Any = None,
        sensitivity_backend: Any = None,
        psd_transform: Any = None,
        galfor_transform: Any = None,
        batch_size: int | None = None,
    ):
        self.acs = acs
        self.wave_gen = wave_gen
        self.waveform_kwargs = dict(waveform_kwargs)
        self.resolved_transform = resolved_transform
        self.sensitivity_backend = sensitivity_backend
        self.psd_transform = psd_transform
        self.galfor_transform = galfor_transform

        self.nwalkers = len(acs)
        self.batch_size = int(batch_size) if batch_size is not None else 2 * self.nwalkers

        if acs.gpus is not None and len(acs.gpus) > 1:
            raise NotImplementedError(
                "TemperatureLikelihood assumes the analysis containers occupy a single "
                f"device, but they are split over {len(acs.gpus)}. The block arithmetic "
                "that copies one walker's data into a scratch slot would have to follow "
                "the split map before this can be lifted."
            )

        # ``data_splits`` selects, per template, which device that template's buffer
        # lives on -- the waveform generator keeps only the templates whose entry equals
        # the device it is currently filling. A zero here is not a "no split" marker: it
        # is device 0, so on any other device every template would be silently dropped
        # and _inject would write nothing at all.
        self._device = 0 if acs.gpus is None else int(acs.gpus[0])

        generator_gpus = getattr(wave_gen, "gpus", None)
        if generator_gpus is not None and list(generator_gpus) != [self._device]:
            raise ValueError(
                "TemperatureLikelihood needs the waveform generator on the same single "
                f"device as the analysis containers. The containers are on {self._device} "
                f"and the generator on {list(generator_gpus)}, so the templates would be "
                "generated on one device and written into a buffer on another."
            )

        # The block arithmetic treats one container as nchannels * data_length
        # contiguous entries and passes end_shape[0] as the generator's data_length,
        # which is the same convention the galactic-binary move uses. Both hold for a
        # frequency-domain layout, where end_shape is (data_length,), and neither holds
        # for the time-frequency one, where end_shape is (m, n) and data_length is m * n.
        if len(acs.end_shape) != 1:
            raise NotImplementedError(
                "TemperatureLikelihood assumes a one-dimensional data layout, but the "
                f"containers report end_shape {acs.end_shape}. The slot arithmetic and "
                "the data_length handed to the waveform generator would both need to "
                "follow the two-dimensional layout first."
            )

        self._base_data = None  # the data, i.e. residual + coldest temperature templates
        self._scratch: AnalysisContainerArray | None = None

    # ------------------------------------------------------------------
    # block arithmetic on the linear buffers
    # ------------------------------------------------------------------

    @property
    def _block(self) -> int:
        """Entries one container occupies in a linear data buffer."""
        return int(self.acs.nchannels * self.acs.data_length)

    def _slot(self, array, index: int):
        """View of container ``index`` inside a linear data buffer."""
        block = self._block
        return array[index * block : (index + 1) * block]

    # ------------------------------------------------------------------
    # the data
    # ------------------------------------------------------------------

    def refresh_data(
        self, cold_coords: NDArrayLike, cold_inds: NDArrayLike
    ) -> None:
        """Recover the data from the residual, once per proposal.

        Args:
            cold_coords: Resolved coordinates of the coldest temperature, shape
                ``(nwalkers, nleaves_max, ndim)``, in the sampling basis.
            cold_inds: Which leaves are alive, shape ``(nwalkers, nleaves_max)``.
        """
        xp = self.acs.xp
        residual = self.acs.linear_data_arr[0]

        if self._base_data is None:
            self._base_data = xp.empty_like(residual)

        self._base_data[:] = residual[:]
        # factors +1 adds a template back into a residual, which is what recovers the
        # data; this is the same convention the galactic-binary move uses when it
        # temporarily restores the coldest temperature's sources.
        self._inject(self._base_data, cold_coords, cold_inds, factor=+1.0)

    def restore_residual(
        self, cold_coords: NDArrayLike, cold_inds: NDArrayLike
    ) -> None:
        """Rewrite the shared residual for a *new* coldest-temperature state.

        The inverse of :meth:`refresh_data`, and the reason it has to exist. The shared
        containers hold one residual per walker, built by subtracting the coldest
        temperature's templates, and every other move in the pipeline reads it -- the
        galactic-binary move computes its band-level likelihood differences straight out
        of it. A swap that moves a state *into* temperature 0 therefore invalidates it:
        ``branches[resolved].coords[0]`` no longer describes what was subtracted.

        Rebuilding from :attr:`_base_data` rather than incrementally undoing the old
        templates keeps this exact rather than accumulating a difference of two
        cancelling injections, and the galactic-binary move rebuilds the residual from
        scratch at the end of its own proposal anyway, so any drift is bounded by one
        iteration.

        Args:
            cold_coords: Resolved coordinates now held by the coldest temperature,
                shape ``(nwalkers, nleaves_max, ndim)``, in the sampling basis.
            cold_inds: Which of those leaves are alive, ``(nwalkers, nleaves_max)``.
        """
        if self._base_data is None:
            raise RuntimeError(
                "restore_residual needs the data, which refresh_data recovers from the "
                "residual; call it before any swap is attempted."
            )
        residual = self.acs.linear_data_arr[0]
        residual[:] = self._base_data[:]
        self._inject(residual, cold_coords, cold_inds, factor=-1.0)

    def _inject(
        self,
        target,
        coords: NDArrayLike,
        inds: NDArrayLike,
        factor: float,
    ) -> None:
        """Add (``factor = +1``) or subtract (``-1``) every active leaf's template.

        ``coords`` and ``inds`` are ``(nconfig, nleaves_max, ...)``; each configuration
        writes into its own block of ``target``.

        No ``N`` is passed, so the generator sizes each template from
        ``get_N(A, f0, T, oversample)`` per source, whereas the galactic-binary move
        passes the per-band ``N_vals`` it carries in the branch supplemental. The two
        agree to within the band quantisation, which is why the residual recovered here
        is the same data the rest of the pipeline works against.
        """
        xp = self.acs.xp
        nconfig = int(inds.shape[0])

        inds_host = np.asarray(inds.get() if hasattr(inds, "get") else inds)
        if not inds_host.any():
            return

        coords_host = np.asarray(coords.get() if hasattr(coords, "get") else coords)
        active = coords_host[inds_host]
        config_of_leaf = np.repeat(np.arange(nconfig), inds_host.sum(axis=-1))

        params = active
        if self.resolved_transform is not None:
            params = self.resolved_transform.both_transforms(params)

        group_index = xp.asarray(config_of_leaf, dtype=xp.int32)
        factors = xp.full(params.shape[0], factor, dtype=xp.float64)

        self.wave_gen.generate_global_template(
            xp.asarray(params),
            group_index,
            [target],
            data_length=self.acs.end_shape[0],
            factors=factors,
            data_splits=np.full(nconfig, self._device, dtype=int),
            **self.waveform_kwargs,
        )

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def _ensure_scratch(self, size: int) -> AnalysisContainerArray:
        if self._scratch is None or len(self._scratch) < size:
            template = self.acs[0] if self.acs.acs.ndim == 1 else self.acs.acs.flatten()[0]
            self._scratch = AnalysisContainerArray(
                [deepcopy(template) for _ in range(size)], gpus=self.acs.gpus
            )
        return self._scratch

    def evaluate(
        self,
        resolved_coords: NDArrayLike,
        resolved_inds: NDArrayLike,
        psd_coords: NDArrayLike,
        galfor_coords: NDArrayLike | None,
        walker_of_config: np.ndarray,
        subtract_resolved: bool = True,
    ) -> np.ndarray:
        """:math:`\\ln L` for each of ``nconfig`` requested states.

        Args:
            resolved_coords: ``(nconfig, nleaves_max, ndim)`` in the sampling basis.
            resolved_inds: ``(nconfig, nleaves_max)`` bool.
            psd_coords: ``(nconfig, ndim_psd)`` in the sampling basis.
            galfor_coords: ``(nconfig, ndim_galfor)``, or ``None`` when the foreground is
                not sampled.
            walker_of_config: Which walker's data each configuration belongs to, shape
                ``(nconfig,)``. The data is the same for every temperature of a walker;
                it is the residual that differs, and that is rebuilt here.
            subtract_resolved: When ``True``, the residual is formed by subtracting this
                configuration's own templates from the data -- the self-consistent value
                the swap ratio needs. When ``False``, the shared residual is used
                unchanged, which reproduces the convention the rest of the pipeline
                stores in ``state.log_like``: the walker's residual, carrying the
                *coldest* temperature's sources, against this configuration's noise
                parameters. Use it to keep the stored likelihood in the pipeline's own
                convention after a swap, never to score one.

        Returns:
            ``(nconfig,)`` log-likelihoods.
        """
        if subtract_resolved and self._base_data is None:
            raise RuntimeError(
                "refresh_data must be called before evaluate: the data is recovered "
                "from the residual once per proposal."
            )

        nconfig = int(
            resolved_inds.shape[0] if subtract_resolved else psd_coords.shape[0]
        )
        out = np.zeros(nconfig)

        for start in range(0, nconfig, self.batch_size):
            stop = min(start + self.batch_size, nconfig)
            size = stop - start
            scratch = self._ensure_scratch(self.batch_size)

            # the data each configuration is compared against
            source = self._base_data if subtract_resolved else self.acs.linear_data_arr[0]
            for slot, config in enumerate(range(start, stop)):
                walker = int(walker_of_config[config])
                self._slot(scratch.linear_data_arr[0], slot)[:] = self._slot(source, walker)

            if subtract_resolved:
                # subtract this configuration's own templates
                self._inject(
                    scratch.linear_data_arr[0],
                    resolved_coords[start:stop],
                    resolved_inds[start:stop],
                    factor=-1.0,
                )

            # and give it its own spectrum
            psd_here = np.asarray(
                psd_coords[start:stop].get()
                if hasattr(psd_coords, "get")
                else psd_coords[start:stop]
            )
            if self.psd_transform is not None:
                psd_here = self.psd_transform.both_transforms(psd_here)

            if galfor_coords is not None:
                galfor_here = np.asarray(
                    galfor_coords[start:stop].get()
                    if hasattr(galfor_coords, "get")
                    else galfor_coords[start:stop]
                )
                if self.galfor_transform is not None:
                    galfor_here = self.galfor_transform.both_transforms(galfor_here)
            else:
                galfor_here = None

            for slot in range(size):
                scratch[slot].sens_mat = self.sensitivity_backend(
                    f"hyper_swap_{slot}",
                    psd_here[slot],
                    galfor_params=None if galfor_here is None else galfor_here[slot],
                )
            scratch.reset_linear_psd_arr()

            values = scratch.likelihood()
            values = np.asarray(values.get() if hasattr(values, "get") else values)
            out[start:stop] = values.flatten()[:size]

        return out


class HyperMove(GlobalFitMove, Move):
    """
    A model informed reversible jump move to jump between discrete models.
    The setup:
        This move handles the changing in model index. This model index is propagated to other relevant
        branches in the model using the ``model_index`` key in the branch supplemental information. This move is
        designed to be used with a custom proposal that proposes a change in the model index. The branch supplemental
        informatin of each breach is propagated to prior calculation such that the correct population prior can be used.

    The model dependent part of the log target is written in the
    inhomogeneous-Poisson-process grouping (§1.2 of
    ``_dev/why_the_model_index_does_not_jump.md``),

    .. math::

        \\ell_m = \\sum_{i=1}^{k_1} \\ln \\mathcal N(\\vec\\theta_i \\mid m)
                  - \\hat N_1(\\Sigma, m)
                  + \\ln p(\\Sigma \\mid m)
                  + \\ln p(m),

    with the intensity
    :math:`\\mathcal N(\\vec\\theta \\mid m) = \\hat N_{\\rm tot}(m) p_{\\rm pop}(\\vec\\theta \\mid m)`.
    The likelihood and the :math:`1/k_1!` factor carry no model dependence and are
    therefore absent. :meth:`compute_model_terms` evaluates :math:`\\ell_m` for *every*
    model at once -- at no extra cost, since the expensive expected-count loop already
    ran over all catalogues -- so the acceptance ratio is a difference of two entries
    of one array, and the full decomposition can be recorded for every walker,
    temperature and iteration (diagnostic D1) by passing ``diagnostics_file``.

    The resolvability :math:`\\alpha(\\vec\\theta \\mid \\Sigma)` is deliberately absent
    from :math:`\\ell_m`: :math:`\\rho_{\\rm thr}` and :math:`\\sigma` are fixed detector
    and pipeline properties, so :math:`\\alpha` depends on :math:`(\\vec\\theta, \\Sigma)`
    only and cancels identically in the ratio.

    Args:
        acs: Analysis containers, one per walker; supplies the PSD used for
            :math:`\\hat N_1`.
        wave_gen: Waveform generator used for the optimal SNRs of the catalogues.
        waveform_kwargs: Runtime kwargs for ``wave_gen``.
        source_setups: ``dict`` with keys ``"resolved"`` and ``"stochastic"`` holding
            the corresponding :class:`BaseSourceConfig` objects.
        branch_name_map: Maps ``"resolved"``/``"stochastic"`` onto branch names.
        catalogues: One catalogue per model, shape ``(N_tot, 9)``.
        snr_threshold: :math:`\\rho_{\\rm thr}` for :math:`\\hat N_1`.
        num_repeats: Number of repeated proposals per call.
        model_log_prior: :math:`\\ln p(m)`, one entry per model. Defaults to a uniform
            model prior, which cancels in the acceptance ratio but is needed for the
            Rao-Blackwellised model posterior (R4).
        diagnostics_file: If given, every proposal appends the full decomposition of
            :math:`\\ell_m` to this HDF5 file (diagnostic D1).
        diagnostics_overwrite: Truncate ``diagnostics_file`` instead of appending.
        tempering: :class:`~lisatools.globalfit.priors.tempering.PopulationTempering`
            weighting the model-dependent prior by :math:`\\beta`, so that hot temperatures are
            indifferent between models. ``None`` -- the default -- leaves the move
            untempered and bit-identical to before. See :meth:`temper_model_terms` and
            ``_dev/prior_tempering.md``.
    """

    #: Class-level defaults so that a move built without ``__init__`` -- which the term
    #: tests do, to exercise the arithmetic with no GPU -- is simply untempered rather
    #: than missing attributes.
    tempering: "PopulationTempering | None" = None
    temperature_likelihood: "TemperatureLikelihood | None" = None

    def __init__(
        self,
        acs: AnalysisContainerArray,
        wave_gen: GBGPU, # can be any waveform generator
        waveform_kwargs: Dict[str, Any],
        source_setups: Dict[str, BaseSourceConfig],
        branch_name_map: Dict[str, str],
        catalogues: List[NDArrayLike],
        snr_threshold: float = 7.0,
        num_repeats: int = 1,
        model_log_prior: Sequence[float] | None = None,
        diagnostics_file: str | None = None,
        diagnostics_overwrite: bool = False,
        tempering: PopulationTempering | None = None,
        **kwargs
    ):
        Move.__init__(self, is_rj=True, **kwargs)

        self.acs = acs
        self.wave_gen = wave_gen
        self.waveform_kwargs = waveform_kwargs
        self.source_setups = source_setups
        self.branch_name_map = branch_name_map
        self.catalogues = catalogues
        self.nmodels = len(catalogues)

        self.N_tot_model = {}
        for model in range(self.nmodels):
            assert self.catalogues[model].shape[0] > self.catalogues[model].shape[1], (
                "The number of sources should be the first axis of the catalog."
            )
            self.N_tot_model[model] = self.catalogues[model].shape[0]

        self.snr_threshold = snr_threshold
        self.num_repeats = num_repeats

        self.first_catalogue_itteration = True

        # N_tot is fixed by the catalogues and enters the intensity as k_1 ln N_tot
        self.N_tot_array = np.array(
            [self.N_tot_model[m] for m in range(self.nmodels)], dtype=np.float64
        )
        self.ln_N_tot_array = np.log(self.N_tot_array)

        if model_log_prior is None:
            # p(M) is uniform, so it cancels in the acceptance ratio; it is kept
            # explicitly because ell_m itself is used in the Rao-Blackwell estimator.
            self.model_log_prior = np.full(self.nmodels, -np.log(self.nmodels))
        else:
            self.model_log_prior = np.asarray(model_log_prior, dtype=np.float64)
            if self.model_log_prior.shape != (self.nmodels,):
                raise ValueError(
                    f"model_log_prior must have one entry per model ({self.nmodels})."
                )

        self.resolved_priors = self.source_setups["resolved"].priors[
            self.branch_name_map["resolved"]
        ]
        self.stochastic_priors = self.source_setups["stochastic"].priors[
            self.branch_name_map["stochastic"]
        ]

        self.diagnostics_file = diagnostics_file
        self.diagnostics_overwrite = diagnostics_overwrite
        self.recorder: HyperMoveRecorder | None = None

        self.tempering = tempering
        # built by build_hyper_moves when the pieces it needs are available; without it
        # the move still tempers its own acceptance ratio, but cannot swap between
        # temperatures, so the hot temperatures have no way to feed the cold chain
        self.temperature_likelihood = None
        if tempering is not None and tempering.enabled:
            logger.info(
                "HyperMove: population prior tempering is ON (%s), ladder %s. The "
                "acceptance ratio at temperature t is beta_t * (ell_m' - ell_m).",
                tempering.scheme,
                np.array2string(tempering.betas, precision=4),
            )
            self._check_prior_support()

    def _check_prior_support(self) -> None:
        """Grade the competing densities on shared support, before the run starts.

        Section 4.5 of ``_dev/prior_tempering.md``. The ``geometric`` reference averages
        :math:`\\ln p_{m'}` over the models finite at a point, so where only one model has
        support the reference collapses onto that model and the annealing switches itself
        off *exactly* at the points that make a barrier infinite. That is well defined and
        it is useless, so it has to be established before a run rather than inferred from
        one afterwards -- which is why this runs at construction, from the training boxes
        alone, with no flow evaluated and no device touched.

        Each model-dependent family is graded separately: a container holds several
        priors and only some of them carry competing models, so flattening them together
        would compare boxes that were never meant to be comparable.
        """
        for label, container in (
            ("resolved population priors", self.resolved_priors),
            ("stochastic (foreground) priors", self.stochastic_priors),
        ):
            for _, prior in getattr(container, "priors", []):
                sub_models = list(getattr(prior, "models", []))
                if len(sub_models) < 2:
                    continue
                verdict = self.tempering.check_common_support(
                    sub_models, label=f"{label} ({type(prior).__name__})"
                )
                logger.info(
                    "HyperMove: %s graded %r for the geometric reference.",
                    label,
                    verdict,
                )

    @staticmethod
    def _to_host(array: NDArrayLike) -> np.ndarray:
        return np.asarray(array.get() if hasattr(array, "get") else array)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, coords):

        if self.first_catalogue_itteration:
            logger.info("Starting first snr loop of catalogues")

            max_logl_walker = np.argmax(self.acs.likelihood()).item()
            acs_max_logl = AnalysisContainerArray([deepcopy(self.acs[max_logl_walker])], gpus=self.acs.gpus)
            xp = acs_max_logl.xp
            f_min_global = acs_max_logl.settings.f_arr.min().get()
            df = acs_max_logl.settings.df
            oversample = self.waveform_kwargs["oversample"] if "oversample" in self.waveform_kwargs else 1
            f_min_filter = f_min_global + get_N(1e-30, f_min_global, 1/df, oversample) * df

            self.catalogue_sizes_filtered = np.zeros(self.nmodels, dtype=np.int64)
            for i, catalogue in enumerate(self.catalogues):

                # filter for out_of_bounds; currently assume f0_idx = 1
                mask = catalogue[:,1] > f_min_filter
                catalogue = catalogue[mask, :]

                ncat = catalogue.shape[0]
                data_index = xp.asarray(
                    np.repeat(0, ncat), dtype=np.int32
                )

                self.wave_gen.get_ll(
                    catalogue,
                    acs_max_logl.linear_data_arr, # we are only interested in h_h contribution for opt_snr, so this can be anything
                    acs_max_logl.linear_psd_arr,
                    data_index=data_index,
                    noise_index=data_index,
                    data_length=acs_max_logl.data_length,
                    data_splits=np.array([0]),
                    phase_marginalize=False,
                    **self.waveform_kwargs
                )
                h_h_raw = self.wave_gen.h_h
                h_h = h_h_raw.get() if hasattr(h_h_raw, "get") else h_h_raw

                opt_snrs = np.sqrt(h_h.real)
                catalogue_filtered = catalogue[opt_snrs > 0.5] # TODO cut adjustable?
                logger.info(f"For catalogue {i}, {catalogue_filtered.shape[0]}/{ncat} binaries are above SNR=0.5")
                self.catalogues[i] = catalogue_filtered
                self.catalogue_sizes_filtered[i] = catalogue_filtered.shape[0]

            del acs_max_logl, data_index
            self.first_catalogue_itteration = False

        return

    # ------------------------------------------------------------------
    # the model dependent terms of the log target
    # ------------------------------------------------------------------

    def compute_resolved_contribution(
        self,
        resolved_coords: NDArrayLike,
        resolved_inds: NDArrayLike,
    ) -> np.ndarray:
        """:math:`\\sum_i \\ln p_{\\rm pop}(\\vec\\theta_i \\mid m)` over resolved sources.

        The sum runs over the active leaves only; the model independent factors of the
        prior (the angular parameters) are included, and being identical for every
        model they cancel in the ratio.

        Returns:
            Array of shape ``(nmodels, ntemps, nwalkers)``.
        """
        xp = self.acs.xp
        ntemps, nwalkers, nleaves_max, ndim = resolved_coords.shape
        nflat = ntemps * nwalkers * nleaves_max

        coords_flat = xp.asarray(resolved_coords.reshape(-1, ndim))
        inds_flat = xp.asarray(resolved_inds.reshape(-1))
        active = coords_flat[inds_flat, :]
        nactive = int(active.shape[0])

        out = np.zeros((self.nmodels, ntemps, nwalkers))
        if nactive == 0:
            return out

        for model in range(self.nmodels):
            model_index = xp.full(nactive, model, dtype=xp.int32)
            values = xp.asarray(
                self.resolved_priors.logpdf(active, model_index=model_index)
            )
            filled = xp.zeros(nflat, dtype=xp.float64)
            filled[inds_flat] = values
            out[model] = self._to_host(
                filled.reshape((ntemps, nwalkers, nleaves_max)).sum(axis=-1)
            )

        return out

    def compute_stochastic_contribution(
        self, stochastic_coords: NDArrayLike
    ) -> np.ndarray:
        """:math:`\\ln p(\\Sigma \\mid m)`, shape ``(nmodels, ntemps, nwalkers)``."""
        xp = self.acs.xp
        
        ntemps, nwalkers, nleaves, ndim = stochastic_coords.shape
        nflat = ntemps * nwalkers * nleaves

        coords_flat = xp.asarray(stochastic_coords.reshape(-1, ndim))

        out = np.zeros((self.nmodels, ntemps, nwalkers))
        for model in range(self.nmodels):
            model_index = xp.full(nflat, model, dtype=xp.int32)
            values = xp.asarray(
                self.stochastic_priors.logpdf(coords_flat, model_index=model_index)
            )
            out[model] = self._to_host(
                values.reshape((ntemps, nwalkers, nleaves)).sum(axis=-1)
            ) # psd only has one leaf

        return out

    def compute_expected_resolved_counts(self) -> np.ndarray:
        """:math:`\\hat N_1(\\Sigma, m)`, shape ``(nmodels, nwalkers)``.

        The count is the number of catalogue sources above ``snr_threshold`` at each
        walker's PSD -- a hard step, whereas the formalism and ``ResolvabilityPrior``
        use an erf of width ``sigma_resolv`` (defect B6, quantified by diagnostic D2
        before anything is changed here). Only ``nwalkers`` PSDs exist in ``acs``, so
        the count carries no temperature dependence.
        """
        xp = self.acs.xp
        nwalkers = len(self.acs)

        counts = np.zeros((self.nmodels, nwalkers))
        for i, catalogue in enumerate(self.catalogues):
            ncat = catalogue.shape[0]

            coords_in = xp.asarray(
                np.broadcast_to(catalogue, (nwalkers,)+catalogue.shape).reshape(-1, 9)
            )
            data_index = xp.asarray(
                np.repeat(np.arange(nwalkers), ncat), dtype=np.int32
            )
            # we are only interested in h_h contribution for opt_snr, so this can be anything
            self.wave_gen.get_ll(
                coords_in,
                self.acs.linear_data_arr,
                self.acs.linear_psd_arr,
                data_index=data_index,
                noise_index=data_index,
                data_length=self.acs.data_length,
                data_splits=np.array([0]),
                phase_marginalize=False,
                **self.waveform_kwargs
            )
            h_h_raw = self.wave_gen.h_h
            h_h = h_h_raw.get() if hasattr(h_h_raw, "get") else h_h_raw

            opt_snrs = np.sqrt(h_h.real).reshape(nwalkers, ncat)
            counts[i] = np.sum(opt_snrs > self.snr_threshold, axis=-1)

        return counts

    def compute_model_terms(
        self,
        resolved_coords: NDArrayLike,
        resolved_inds: NDArrayLike,
        stochastic_coords: NDArrayLike,
    ) -> Dict[str, np.ndarray]:
        """Evaluate :math:`\\ell_m` and its parts for every model.

        Returns:
            A dictionary of host arrays. ``resolved``, ``intensity``, ``n1_expected``,
            ``stochastic`` and ``ell`` have shape ``(nmodels, ntemps, nwalkers)``;
            ``num_resolved`` has shape ``(ntemps, nwalkers)``.
        """
        ntemps, nwalkers = resolved_coords.shape[:2]

        num_resolved = np.asarray(resolved_inds.sum(axis=-1))

        resolved = self.compute_resolved_contribution(resolved_coords, resolved_inds)
        stochastic = self.compute_stochastic_contribution(stochastic_coords)

        counts = self.compute_expected_resolved_counts()
        n1_expected = np.broadcast_to(
            counts[:, None, :], (self.nmodels, ntemps, nwalkers)
        ).copy()

        intensity = (
            resolved
            + num_resolved[None, :, :] * self.ln_N_tot_array[:, None, None]
        )
        ell = (
            intensity
            - n1_expected
            + stochastic
            + self.model_log_prior[:, None, None]
        )

        return dict(
            resolved=resolved,
            intensity=intensity,
            n1_expected=n1_expected,
            stochastic=stochastic,
            ell=ell,
            ell_tempered=self.temper_model_terms(ell),
            num_resolved=num_resolved,
        )

    def temper_model_terms(self, ell: np.ndarray) -> np.ndarray:
        """Weight :math:`\\ell_m` by its own inverse temperature.

        The model index enters the target through the prior alone, which the sampler does
        not temper, so without this the acceptance ratio :math:`\\ell_{m'} - \\ell_m` is
        identical at every temperature and the ladder is, with respect to :math:`M`, many copies
        of the same problem. Weighting by :math:`\\beta_t` makes the hot temperatures indifferent
        between models: at :math:`\\beta = 10^{-4}` a barrier of :math:`10^{3}` nats
        becomes :math:`0.1` and the model index randomises freely.

        Under both schemes the model-independent reference cancels from the difference,
        so at temperature :math:`t`

        .. math::

            \\ln\\alpha_t = \\beta_t\\,(\\ell_{m'} - \\ell_m).

        Args:
            ell: Shape ``(nmodels, ntemps, nwalkers)``, the untempered values.

        Returns:
            The tempered array, of the same shape; ``ell`` itself when tempering is off,
            so that the untempered pipeline is reproduced exactly.

        Note:
            The **raw** ``ell`` is what the Rao-Blackwellised model posterior needs, since
            a tempered one would describe a different target. Both arrays are therefore
            returned by :meth:`compute_model_terms` and both are recorded.
        """
        if self.tempering is None or not self.tempering.enabled:
            return ell

        ntemps = ell.shape[1]
        if self.tempering.ntemps != ntemps:
            raise ValueError(
                f"The population tempering ladder has {self.tempering.ntemps} temperatures but "
                f"the state has {ntemps}. One ladder, set in the settings file, must "
                "describe the whole run."
            )
        # through beta_for, not off the array: that method is the one sanctioned
        # conversion from a temperature index to a beta, and a consumer that reaches
        # past it is a consumer that can quietly acquire a different ladder
        beta = self.tempering.beta_for(np.arange(ntemps))[:, None]
        return self.tempering.temper_ell(ell, beta)

    @staticmethod
    def gather_model(terms: np.ndarray, model_indices: np.ndarray) -> np.ndarray:
        """Select, per (temperature, walker), the entry of ``terms`` for its model."""
        return np.take_along_axis(terms, model_indices[None, ...], axis=0)[0]

    # ------------------------------------------------------------------
    # proposal and acceptance
    # ------------------------------------------------------------------

    def get_proposal(self, coords, random, supps=None, branch_supps=None):
        if self.nmodels <= 1:
            raise ValueError("nmodels must be strictly greater than 1 to propose a change.")

        ntemps, nwalkers, _, _ = coords.shape

        # all leaves of each walker and temperature have the same model
        current_indices = coords[..., 0, 0].copy().astype(int)

        proposed_indices = random.randint(1, self.nmodels, size=(ntemps, nwalkers))

        new_indices = (current_indices + proposed_indices) % self.nmodels

        new_coords = coords.copy()
        new_coords[..., 0, 0] = new_indices

        factors = np.zeros((ntemps, nwalkers))

        return new_coords, factors

    def accept_delta(
        self,
        delta_logp: np.ndarray,
        random: np.random.RandomState,
        current_indices: np.ndarray,
        proposed_indices: np.ndarray,
    ) -> np.ndarray:
        """Metropolis-Hastings decision with the undefined cases separated out.

        ``-inf`` is a genuine rejection and ``+inf`` a genuine acceptance, but
        ``nan`` -- which is what a bare comparison silently turns into when both
        models return ``-inf``, e.g. for a source outside a flow's training box --
        means the ratio is undefined. Those walkers are rejected *and* reported,
        so that "correctly rejected" is never confused with "numerically undefined"
        (defect B3).
        """
        undefined = np.isnan(delta_logp)
        if np.any(undefined):
            temps, walkers = np.nonzero(undefined)
            logger.error(
                "HyperMove: undefined acceptance ratio (nan) for %d/%d walkers, "
                "rejected. First occurrences (temp, walker, model, proposed): %s. "
                "Both models most likely returned -inf, e.g. a source outside a "
                "flow's training box.",
                undefined.sum(),
                undefined.size,
                [
                    (int(t), int(w), int(current_indices[t, w]), int(proposed_indices[t, w]))
                    for t, w in zip(temps[:5], walkers[:5])
                ],
            )

        accepted = np.zeros(delta_logp.shape, dtype=bool)
        finite_or_inf = ~undefined
        accepted[finite_or_inf] = delta_logp[finite_or_inf] > np.log(
            random.rand(*delta_logp.shape)
        )[finite_or_inf]

        impossible_start = np.isposinf(delta_logp)
        if np.any(impossible_start):
            logger.warning(
                "HyperMove: %d walker(s) had zero prior mass under their current "
                "model and were moved unconditionally.",
                int(impossible_start.sum()),
            )
        return accepted

    def swap_branch_names(self, state) -> List[str]:
        """Branches the tempering swap exchanges: the model, resolved and stochastic.

        Every other branch -- in this configuration the instrument noise -- stays where
        it is, which is what makes the likelihood not travel with the exchange.
        """
        names = ["hyper"] + list(self.branch_name_map.values())
        return [name for name in names if name in state.branches]

    @staticmethod
    def swap_log_ratio_from_terms(
        beta_hot: float,
        beta_cold: float,
        logl_new_hot: np.ndarray,
        logl_new_cold: np.ndarray,
        logl_hot: np.ndarray,
        logl_cold: np.ndarray,
        u_hot: np.ndarray,
        u_cold: np.ndarray,
    ) -> np.ndarray:
        """The swap ratio itself, equation (15), given every quantity it needs.

        Kept apart from :meth:`hyper_swap_log_ratio` for one reason: that method has to
        call the likelihood evaluator, which needs a populated sampler state on a GPU,
        and the arithmetic here does not. Separating them is what lets the sign
        conventions, the orientation of :math:`\\Delta\\beta` and the collapse to the
        standard form be pinned by a test today rather than inferred from a live run.

        With :math:`i` the hotter of the pair and
        :math:`u_t = \\ell_{M_t}(x_t) - R(x_t)`,

        .. math::

            \\ln\\alpha = \\beta_i\\,\\big[\\ln L^{\\rm new}_i - \\ln L^{\\rm old}_i\\big]
                        + \\beta_{i-1}\\,\\big[\\ln L^{\\rm new}_{i-1} - \\ln L^{\\rm old}_{i-1}\\big]
                        + (\\beta_{i-1} - \\beta_i)\\,\\big(u_i - u_{i-1}\\big).

        Had the instrument-noise branch been exchanged as well, the pair *would* be a
        permutation, the likelihoods would travel with the states, and this would
        collapse to Eryn's familiar :math:`\\Delta\\beta\\,[A(x_i) - A(x_{i-1})]` with
        :math:`A = \\ln L + \\ell_M - R`. That identity is the sharpest available check
        on the algebra and is asserted by the tests.
        """
        return (
            beta_hot * (logl_new_hot - logl_hot)
            + beta_cold * (logl_new_cold - logl_cold)
            + (beta_cold - beta_hot) * (u_hot - u_cold)
        )

    #: Branches that carry no subtracted template, so the residual does not depend on
    #: them. Everything else in a state is a signal branch as far as this move is
    #: concerned, and :meth:`TemperatureLikelihood.refresh_data` would recover the wrong
    #: data in its presence.
    NON_SIGNAL_BRANCHES = ("hyper", "psd", "galfor")

    def _assert_resolved_is_the_only_signal(self, state) -> None:
        """Refuse a configuration in which the residual carries more than one signal.

        :meth:`TemperatureLikelihood.refresh_data` recovers the data by adding the
        coldest temperature's *resolved* templates back into the residual. That is only
        the data if nothing else was ever subtracted from it. In this configuration the
        galactic binaries are the only signal branch, and the note has said so from the
        beginning -- but as an observation about the configuration, not as anything the
        code checked. Adding, say, a massive-black-hole branch would leave the recovered
        "data" short by every one of its templates, and the evaluator would return
        confident, finite, wrong likelihoods rather than failing.
        """
        known = set(self.NON_SIGNAL_BRANCHES) | set(self.branch_name_map.values())
        extra = sorted(set(state.branches) - known)
        if extra:
            raise NotImplementedError(
                f"HyperMove's temperature swap recovers the data by adding back the "
                f"resolved branch's templates alone, but this state also carries "
                f"{extra}. If any of those subtracts a template from the shared "
                "residual, the recovered data is short by it and every swap likelihood "
                "is wrong. Add them to HyperMove.NON_SIGNAL_BRANCHES if they do not, or "
                "extend refresh_data to restore them if they do."
            )

    def model_swap_terms(self, ell: np.ndarray, state) -> np.ndarray:
        """:math:`u_t = \\ell_{M_t}(x_t) - R(x_t)`, one entry per temperature and walker.

        Computed **once** per call of :meth:`run_hyper_tempering` rather than once per
        adjacent pair, and thereafter carried alongside ``ell`` through the cascade. Both
        :math:`\\ell` and its reference depend on the resolved and stochastic parameters
        and the model index alone, and an accepted swap exchanges all three together, so
        :math:`u` travels with the state and needs no re-evaluation.

        Recomputing it per pair was not wrong, but it charged
        :meth:`PopulationTempering.reference` eleven extra passes an iteration, and that
        method keeps the census behind ``partial_reference_fraction`` -- the one
        measurement a live run makes of how weak the annealing really was. Counting the
        same states a dozen times over made it a statement about the swap loop rather
        than about the sampled volume.
        """
        reference = self.tempering.reference(ell)
        if np.isscalar(reference):
            reference = np.zeros(ell.shape[1:])
        models = state.branches_coords["hyper"][..., 0, 0].astype(int)
        return np.take_along_axis(ell, models[None, ...], axis=0)[0] - reference

    def hyper_swap_log_ratio(
        self,
        ell: np.ndarray,
        state,
        hot: int,
        logl_self: np.ndarray,
        u: np.ndarray,
    ) -> tuple:
        """Acceptance ratio for exchanging two adjacent temperatures' states.

        The exchange carries the model, resolved and stochastic branches; the
        instrument-noise branch stays where it is, so the states of the two temperatures
        are *not* simply permuted and the likelihood does not travel with them. Writing
        :math:`i` for the hotter of the pair, :math:`x_t` for its state and
        :math:`u_t = \\ell_{M_t}(x_t) - R(x_t)`,

        .. math::

            \\ln\\alpha = \\beta_i\\,\\big[\\ln L^{\\rm new}_i - \\ln L^{\\rm old}_i\\big]
                        + \\beta_{i-1}\\,\\big[\\ln L^{\\rm new}_{i-1} - \\ln L^{\\rm old}_{i-1}\\big]
                        + \\Delta\\beta\\,\\big(u_i - u_{i-1}\\big).

        The model term reduces to the standard :math:`\\Delta\\beta\\,\\Delta u` because
        :math:`\\ell` and its reference depend on the resolved and stochastic parameters
        only -- the expected count reads the walker's spectrum, not this branch -- so
        both move with the exchange and the model-independent parts cancel between the
        two temperatures. Every entry of :math:`u` is already in ``ell``.

        Args:
            ell: Raw ``(nmodels, ntemps, nwalkers)`` terms.
            state: The current state.
            hot: Index :math:`i` of the hotter temperature of the pair.
            logl_self: ``(ntemps, nwalkers)`` self-consistent likelihoods, i.e. each
                temperature's own parameters against its own residual and spectrum.
            u: ``(ntemps, nwalkers)`` model terms from :meth:`model_swap_terms`, kept in
                step with the exchange by the caller.

        Returns:
            ``(log_alpha, logl_new_hot, logl_new_cold)``.
        """
        cold = hot - 1
        beta_hot, beta_cold = self.tempering.beta_for(np.array([hot, cold]))
        nwalkers = ell.shape[2]
        walkers = np.arange(nwalkers)

        resolved = self.branch_name_map["resolved"]
        stochastic = self.branch_name_map["stochastic"]

        def gather(branch, temperature):
            return state.branches[branch].coords[temperature]

        # the proposal: each temperature takes the other's exchanged branches and keeps
        # its own instrument-noise parameters
        logl_new = self.temperature_likelihood.evaluate(
            np.concatenate([gather(resolved, cold), gather(resolved, hot)]),
            np.concatenate(
                [
                    state.branches[resolved].inds[cold],
                    state.branches[resolved].inds[hot],
                ]
            ),
            np.concatenate(
                [gather("psd", hot)[:, 0], gather("psd", cold)[:, 0]]
            ),
            np.concatenate(
                [gather(stochastic, cold)[:, 0], gather(stochastic, hot)[:, 0]]
            )
            if stochastic in state.branches
            else None,
            walker_of_config=np.concatenate([walkers, walkers]),
        )
        logl_new_hot, logl_new_cold = logl_new[:nwalkers], logl_new[nwalkers:]

        log_alpha = self.swap_log_ratio_from_terms(
            beta_hot,
            beta_cold,
            logl_new_hot,
            logl_new_cold,
            logl_self[hot],
            logl_self[cold],
            u[hot],
            u[cold],
        )
        return log_alpha, logl_new_hot, logl_new_cold

    def run_hyper_tempering(
        self, state, ell: np.ndarray, stochastic_terms: np.ndarray | None = None
    ):
        """Exchange states between adjacent temperatures, carrying the model index.

        Tempering the model term makes the hot temperatures indifferent between models,
        but that is only useful if what they find can reach the cold chain. This is the
        move that carries it there. It exchanges the model, resolved and stochastic
        branches, cascading from the hottest temperature downwards, and leaves every
        other branch -- in this configuration the instrument noise -- in place.

        The likelihood therefore does not travel with the exchange and is recomputed
        through :class:`TemperatureLikelihood`; see :meth:`hyper_swap_log_ratio` for the
        ratio and section 7 of ``_dev/prior_tempering.md`` for why that evaluator has to
        exist and why it belongs to this move.

        Walkers are never permuted. A walker keeps its index throughout, because the data
        a configuration is compared against is that walker's, and permuting would compare
        a state against another walker's residual.

        The cascade ends at the pair ``(1, 0)``, and an acceptance there is the only
        effect this move has outside itself: it replaces the state the shared residual
        was built from. The residual is therefore rebuilt through
        :meth:`TemperatureLikelihood.restore_residual` before this method returns.
        Without that, the galactic-binary move would spend its next whole proposal
        computing band-level likelihood differences against data that no longer matches
        ``branches[resolved].coords[0]``, and would only discover it at the very end,
        after every accept/reject decision had already been made against the wrong
        residual.

        Args:
            state: The state after the within-temperature proposal.
            ell: The **raw** ``(nmodels, ntemps, nwalkers)`` terms from
                :meth:`compute_model_terms`. They remain valid here: the within-
                temperature proposal changes only the model index, and ``ell`` holds
                every model's value already.
            stochastic_terms: :math:`\\ln p(\\Sigma \\mid m)` for every model, same shape,
                also from :meth:`compute_model_terms`. Used to keep ``state.log_prior``
                in step with the exchange; ``None`` skips that correction and leaves the
                stored prior stale, which is what the first version of this move did.

        Returns:
            The state, with the exchanged branches swapped where accepted.
        """
        if self.tempering is None or not self.tempering.enabled:
            return state
        if getattr(self, "prevent_swaps", False):
            return state
        if self.temperature_likelihood is None:
            logger.warning(
                "HyperMove: population tempering is on but no likelihood evaluator was "
                "supplied, so no swaps between temperatures can be attempted and the "
                "hot temperatures cannot feed the cold chain. Pass the sensitivity "
                "backend and the transforms to build_hyper_moves."
            )
            return state

        _, ntemps, nwalkers = ell.shape
        if ntemps <= 1:
            return state

        names = self.swap_branch_names(state)
        resolved = self.branch_name_map["resolved"]
        stochastic = self.branch_name_map["stochastic"]
        walkers = np.arange(nwalkers)

        self._assert_resolved_is_the_only_signal(state)

        # the data every configuration is measured against, recovered once
        self.temperature_likelihood.refresh_data(
            state.branches[resolved].coords[0], state.branches[resolved].inds[0]
        )

        # each temperature's own parameters against its own residual and spectrum; the
        # stored likelihood cannot be used here, since it carries the coldest
        # temperature's sources for every temperature
        flat = lambda branch: np.concatenate(
            [state.branches[branch].coords[t] for t in range(ntemps)]
        )
        logl_self = self.temperature_likelihood.evaluate(
            flat(resolved),
            np.concatenate([state.branches[resolved].inds[t] for t in range(ntemps)]),
            np.concatenate([state.branches["psd"].coords[t][:, 0] for t in range(ntemps)]),
            np.concatenate(
                [state.branches[stochastic].coords[t][:, 0] for t in range(ntemps)]
            )
            if stochastic in state.branches
            else None,
            walker_of_config=np.tile(walkers, ntemps),
        ).reshape(ntemps, nwalkers)

        self.swaps_accepted = np.zeros(ntemps - 1)
        self.swaps_proposed = np.zeros(ntemps - 1)
        cold_chain_changed = False

        # Which entries of the stored likelihood the exchange invalidates. A swap pairs
        # each temperature's own instrument noise with the other's foreground, so both
        # entries of an accepted pair go stale; nothing else does, and on an iteration
        # where nothing is accepted the stored value must be left exactly as the
        # foreground move computed it.
        stale = np.zeros((ntemps, nwalkers), dtype=bool)

        # u = ell_M - R, once: it travels with the exchange like ell does
        u = self.model_swap_terms(ell, state)

        # the foreground prior is the part of log_prior that travels with the exchange
        if stochastic_terms is not None:
            models_before = state.branches_coords["hyper"][..., 0, 0].astype(int)
            log_prior_stochastic_before = self.gather_model(
                stochastic_terms, models_before
            )

        for hot in range(ntemps - 1, 0, -1):
            cold = hot - 1
            log_alpha, logl_new_hot, logl_new_cold = self.hyper_swap_log_ratio(
                ell, state, hot, logl_self, u
            )
            accepted = self.accept_delta(
                log_alpha[None, :],
                self.current_model.random,
                state.branches_coords["hyper"][hot, :, 0, 0].astype(int)[None, :],
                state.branches_coords["hyper"][cold, :, 0, 0].astype(int)[None, :],
            )[0]

            if np.any(accepted):
                for name in names:
                    branch = state.branches[name]
                    hot_coords = branch.coords[hot, accepted].copy()
                    branch.coords[hot, accepted] = branch.coords[cold, accepted]
                    branch.coords[cold, accepted] = hot_coords
                    if branch.inds is not None:
                        hot_inds = branch.inds[hot, accepted].copy()
                        branch.inds[hot, accepted] = branch.inds[cold, accepted]
                        branch.inds[cold, accepted] = hot_inds

                logl_self[hot, accepted] = logl_new_hot[accepted]
                logl_self[cold, accepted] = logl_new_cold[accepted]

                # ell follows its state: the exchanged branches carry every quantity it
                # depends on, so the columns swap rather than needing re-evaluation.
                # The foreground term is carried for the same reason, and is what tells
                # log_prior below how much of itself moved.
                arrays = (ell,) if stochastic_terms is None else (ell, stochastic_terms)
                for array in arrays:
                    hot_column = array[:, hot, accepted].copy()
                    array[:, hot, accepted] = array[:, cold, accepted]
                    array[:, cold, accepted] = hot_column

                # u carries the model index, which the exchange moved, so it travels too
                hot_u = u[hot, accepted].copy()
                u[hot, accepted] = u[cold, accepted]
                u[cold, accepted] = hot_u

                stale[hot, accepted] = True
                stale[cold, accepted] = True

                # the last pair of the cascade is (1, 0), and an acceptance there is the
                # one thing this move can do that the rest of the pipeline notices
                if cold == 0:
                    cold_chain_changed = True

            self.swaps_accepted[cold] = accepted.sum()
            self.swaps_proposed[cold] = nwalkers

        # A swap into temperature 0 replaces the state the shared residual was built
        # from, and every other move reads that residual. Put it back in step before
        # anything -- including the stored likelihood just below -- reads it again.
        if cold_chain_changed:
            self.temperature_likelihood.restore_residual(
                state.branches[resolved].coords[0], state.branches[resolved].inds[0]
            )
            # every temperature is scored against the shared residual, so rebuilding it
            # invalidates the stored likelihood everywhere, not only where a swap landed
            stale[:] = True
            logger.debug(
                "HyperMove: %d cold-chain swap(s) accepted; shared residual rebuilt.",
                int(self.swaps_accepted[0]),
            )

        if stale.any():
            self._restore_stored_likelihood(state, stale)

        # log_prior travels with the exchange too. Under the convention the foreground
        # move establishes, what the array holds is the instrument-noise and foreground
        # priors; the instrument noise stays put, so the whole change is the change in
        # ln p(Sigma | M), and both Sigma and M moved together. Every value needed is
        # already in the term array, so this costs no evaluation.
        if stochastic_terms is not None:
            models_after = state.branches_coords["hyper"][..., 0, 0].astype(int)
            delta = (
                self.gather_model(stochastic_terms, models_after)
                - log_prior_stochastic_before
            )
            if np.any(~np.isfinite(delta)):
                logger.warning(
                    "HyperMove: %d entries of the foreground prior changed by a "
                    "non-finite amount across the swap; log_prior left unchanged there.",
                    int(np.sum(~np.isfinite(delta))),
                )
                delta = np.where(np.isfinite(delta), delta, 0.0)
            state.log_prior[:] = state.log_prior + delta

        logger.debug(
            "HyperMove swaps accepted per temperature pair: %s / %d",
            self.swaps_accepted.astype(int),
            nwalkers,
        )
        return state

    def _restore_stored_likelihood(self, state, stale: np.ndarray) -> None:
        """Recompute the stale entries of ``state.log_like`` in the pipeline's convention.

        The stored likelihood is not the self-consistent one: it is the walker's shared
        residual -- which carries the *coldest* temperature's sources whatever the
        temperature -- evaluated against that entry's own noise parameters. That is
        exactly what the foreground move computes
        (``PSDMove.psd_log_like`` scores ``acs.linear_data_arr[0]`` against the given
        temperature's noise parameters), and the foreground move runs immediately before
        this one, so this reproduces the convention in force rather than introducing one.

        A swap moves the foreground branch while leaving the instrument-noise branch
        behind, so an exchanged entry's stored value is stale in a way that no exchange
        of entries can repair, and a rebuilt residual makes every entry stale at once.
        Only those entries are touched: an untouched entry must keep the number the
        foreground move produced, not a re-derivation of it, because this reaches the
        quantity through ``AnalysisContainer.likelihood`` while the foreground move
        reaches it through the sensitivity backend, and the two paths have historically
        disagreed at the level ``PSDMove``'s own CHECK2 exists to catch.

        No waveform is generated, since no templates are subtracted.

        Args:
            state: The state whose ``log_like`` is to be brought up to date.
            stale: ``(ntemps, nwalkers)`` bool; which entries to recompute.
        """
        stochastic = self.branch_name_map["stochastic"]
        temps, walkers = np.nonzero(stale)
        if temps.size == 0:
            return

        psd_coords = state.branches["psd"].coords[temps, walkers, 0]
        galfor_coords = (
            state.branches[stochastic].coords[temps, walkers, 0]
            if stochastic in state.branches
            else None
        )

        state.log_like[temps, walkers] = self.temperature_likelihood.evaluate(
            None,
            None,
            psd_coords,
            galfor_coords,
            walker_of_config=walkers,
            subtract_resolved=False,
        )

    # ------------------------------------------------------------------
    # diagnostics (D1)
    # ------------------------------------------------------------------

    def _flow_metadata(self) -> Dict[str, Any]:
        """Checkpoint provenance and support-floor settings of the flows.

        The floor's ``epsilon`` and ``ln|B|`` are recorded because they are what makes a
        recording reversible: the stochastic term is one evaluation per state, so
        ``unfloor_log_density`` recovers the raw flow density from it exactly and the
        epsilon-sensitivity of the model comparison can be measured after the fact,
        without a rerun. See D14 and section 10.6 of
        ``_dev/why_the_model_index_does_not_jump.md``.
        """
        metadata: Dict[str, Any] = {}
        for label, container in (
            ("resolved", self.resolved_priors),
            ("stochastic", self.stochastic_priors),
        ):
            sub_models = [
                sub_model
                for _, prior in container.priors
                for sub_model in getattr(prior, "models", [])
            ]
            checkpoints = [
                str(getattr(sub_model, "checkpoint_path", "unknown"))
                for sub_model in sub_models
            ]
            if checkpoints:
                metadata[f"{label}_checkpoints"] = checkpoints

            # apply_common_support_floor gives every model of a family the same epsilon
            # and the same broad box, so one pair of numbers describes the family. If
            # that ever stops being true the recording must not claim otherwise.
            epsilons = {
                getattr(sub_model, "support_floor_epsilon", None)
                for sub_model in sub_models
            }
            log_uniforms = {
                getattr(sub_model, "support_floor_log_uniform", None)
                for sub_model in sub_models
            }
            if len(epsilons) == 1 and len(log_uniforms) == 1:
                epsilon = epsilons.pop()
                log_uniform = log_uniforms.pop()
                if epsilon is not None and log_uniform is not None:
                    metadata[f"{label}_support_floor_epsilon"] = float(epsilon)
                    metadata[f"{label}_support_floor_log_uniform"] = float(log_uniform)
            elif sub_models:
                logger.warning(
                    "%s priors do not share one support floor (epsilons %s). The "
                    "recording cannot describe it, so the epsilon-sensitivity of this "
                    "run (D14) will not be measurable offline.",
                    label,
                    sorted(str(value) for value in epsilons),
                )
        return metadata

    def _ensure_recorder(self, ntemps: int, nwalkers: int) -> None:
        if self.recorder is not None or self.diagnostics_file is None:
            return

        metadata: Dict[str, Any] = dict(
            snr_threshold=float(self.snr_threshold),
            branch_name_map=[
                f"{key}={value}" for key, value in self.branch_name_map.items()
            ],
        )
        if hasattr(self, "catalogue_sizes_filtered"):
            metadata["catalogue_sizes_snr_filtered"] = self.catalogue_sizes_filtered

        # the scheme and the ladder, so a recording says what target it describes rather
        # than leaving a later reader to infer it; the reader's raw_ell check needs both
        metadata["population_tempering_scheme"] = (
            "off" if self.tempering is None else self.tempering.scheme
        )
        if self.tempering is not None and self.tempering.betas is not None:
            metadata["population_tempering_betas"] = np.asarray(
                self.tempering.betas, dtype=np.float64
            )
        try:
            metadata.update(self._flow_metadata())
        except Exception:  # pragma: no cover - provenance must never break a run
            logger.warning("Could not collect flow checkpoint metadata.", exc_info=True)

        self.recorder = HyperMoveRecorder(
            self.diagnostics_file,
            nmodels=self.nmodels,
            ntemps=ntemps,
            nwalkers=nwalkers,
            n_tot=self.N_tot_array,
            model_log_prior=self.model_log_prior,
            metadata=metadata,
            overwrite=self.diagnostics_overwrite,
        )

    def record_diagnostics(
        self,
        terms: Dict[str, np.ndarray],
        current_indices: np.ndarray,
        proposed_indices: np.ndarray,
        accepted: np.ndarray,
        iteration: int | None = None,
    ) -> None:
        """Append one row of the D1 recording; a no-op without ``diagnostics_file``.

        Args:
            terms: Output of :meth:`compute_model_terms`.
            current_indices: Model index of each walker before the proposal.
            proposed_indices: Model index proposed for each walker.
            accepted: Outcome of the Metropolis-Hastings decision.
            iteration: Label stored with the row; defaults to the move's proposal
                counter. Replaying a stored chain should pass the chain iteration.
        """
        self._ensure_recorder(*accepted.shape)
        if self.recorder is None:
            return
        row = {name: terms[name] for name in HyperMoveRecorder.PER_MODEL_TERMS}
        # the tempered terms beside the raw ones, so that the acceptance ratio the move
        # actually used is recoverable from the recording; the model posterior still
        # takes the raw ones, which is what the reader's raw_ell enforces
        for name in HyperMoveRecorder.OPTIONAL_PER_MODEL_TERMS:
            if name in terms:
                row[name] = terms[name]
        row["num_resolved"] = terms["num_resolved"]
        row["model_current"] = current_indices
        row["model_proposed"] = proposed_indices
        row["accepted"] = accepted
        try:
            self.recorder.record(
                self.num_proposals if iteration is None else iteration, row
            )
        except Exception:  # pragma: no cover - diagnostics must never kill a run
            logger.warning("Failed to record HyperMove diagnostics.", exc_info=True)

    # ------------------------------------------------------------------
    # the move
    # ------------------------------------------------------------------

    def propose(self, model, state):

        self.setup(state.branches_coords)

        all_branch_names = list(state.branches.keys())

        # setup supplemental information
        if not np.all(
            np.asarray(list(state.branches_supplemental.values())) == None
        ):
            new_branch_supps = deepcopy(state.branches_supplemental)
        else:
            new_branch_supps = None

        if state.supplemental is not None:
            new_supps = deepcopy(state.supplemental)
        else:
            new_supps = None

        self.current_model = model
        self.current_state = state

        resolved_inds = state.branches[self.branch_name_map["resolved"]].inds[:]
        old_coords_model = deepcopy(state.branches_coords["hyper"])
        coords_resolved = deepcopy(state.branches_coords[self.branch_name_map["resolved"]])
        coords_stochastic = deepcopy(state.branches_coords[self.branch_name_map["stochastic"]])

        # ell_m for every model: the catalogue loop of the expected count already
        # visits all models, so the full set costs no more than the two states
        terms = self.compute_model_terms(
            coords_resolved, resolved_inds, coords_stochastic
        )
        # the acceptance ratio uses the tempered values, so that temperature t sees
        # beta_t * (ell_m' - ell_m); with tempering off this is terms["ell"] itself
        ell = terms["ell_tempered"]

        current_indices = old_coords_model[..., 0, 0].astype(int)
        logp_prev = self.gather_model(ell, current_indices)

        # get new model coords
        new_coords_model, factors = self.get_proposal(
            old_coords_model,
            model.random,
            supps=new_supps,
            branch_supps=new_branch_supps,
        )
        proposed_indices = new_coords_model[..., 0, 0].astype(int)
        logp_curr = self.gather_model(ell, proposed_indices)

        # acceptance fraction
        delta_logp = factors + logp_curr - logp_prev
        accepted = self.accept_delta(
            delta_logp, model.random, current_indices, proposed_indices
        )

        logger.debug(f"Old model coords are     {old_coords_model[0,:,0,0]}")
        logger.debug(f"Proposed model coords are {new_coords_model[0,:,0,0]}")
        model_coords_tmp = deepcopy(old_coords_model)
        model_coords_tmp[accepted] = new_coords_model[accepted]
        logger.debug(f"New model coords are     {model_coords_tmp[0,:,0,0]}")

        new_state = GFState(state, copy=True)
        new_state.branches_coords["hyper"][accepted] = new_coords_model[accepted]
        assert new_state.log_prior is not None
        # Only the accepted walkers change their model dependent prior.
        #
        # The *raw* difference, not the tempered one that decided the acceptance:
        # ``log_prior`` is by convention the untempered log prior, and the whole ladder
        # is built on forming ``beta * logl + logp`` from it. Writing a beta-weighted
        # increment here would leave the hot temperatures holding neither the prior nor
        # the tempered target, and every other consumer of the array -- the foreground
        # move's swaps among them -- would then be mixing two conventions. With tempering
        # off the two are the same array and this is a no-op.
        delta_logp_raw = (
            factors
            + self.gather_model(terms["ell"], proposed_indices)
            - self.gather_model(terms["ell"], current_indices)
        )
        update = np.where(accepted & np.isfinite(delta_logp_raw), delta_logp_raw, 0.0)
        if np.any(accepted & ~np.isfinite(delta_logp_raw)):
            logger.warning(
                "HyperMove: accepted %d walker(s) with a non-finite delta log prior; "
                "log_prior left unchanged for those walkers.",
                int(np.sum(accepted & ~np.isfinite(delta_logp_raw))),
            )
        new_state.log_prior[:] = new_state.log_prior + update

        self.record_diagnostics(terms, current_indices, proposed_indices, accepted)

        # the raw terms, not the tempered ones: the swap ratio applies the temperatures
        # itself, and needs each model's untempered value
        new_state = self.run_hyper_tempering(
            new_state, terms["ell"].copy(), terms["stochastic"].copy()
        )

        # add to move-specific accepted information
        self.accepted += accepted
        self.num_proposals += 1

        return new_state, accepted
