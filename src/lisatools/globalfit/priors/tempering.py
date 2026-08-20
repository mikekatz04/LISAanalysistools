"""Tempering of the model-dependent population prior (R3).

Why this exists
---------------
Eryn tempers the **likelihood** only. The population model index :math:`M` enters the
target through the **prior** alone, so the acceptance ratio of
:class:`~lisatools.globalfit.moves.hypermove.HyperMove`,

.. math::

    \\ln\\alpha = \\ell_{m'} - \\ell_m,

is identical at every temperature of the ladder. The ladder therefore provides
*zero* help in crossing a model barrier, and with
:math:`|\\Delta_{\\rm per\\ source}| \\approx 0.5` nats at :math:`k_1 \\approx 4\\times10^3`
that barrier is :math:`\\sim10^3` nats: the measured acceptance rate is exactly zero,
which is what a *correct* move and a *broken* move both produce. See
``_dev/why_the_model_index_does_not_jump.md``, section 4 ("Not a defect") and section 3.1.

R3 raises the model-dependent part of the prior to the power :math:`\\beta`, so that hot
temperatures are indifferent between models and swaps carry model indices down to the cold
chain.

The two schemes
---------------
Write :math:`p_m(\\vec\\theta) \\equiv p_{\\rm pop}(\\vec\\theta \\mid m)` and let
:math:`\\bar L(\\vec\\theta) = \\frac1n \\sum_{m'} \\ln p_{m'}(\\vec\\theta)` be the
model-independent reference.

``"off"``
    :math:`\\ln p^{(\\beta)}_m = \\ln p_m`. Unchanged, bit-for-bit. The default.

``"flat"``
    :math:`\\ln p^{(\\beta)}_m = \\beta \\ln p_m`. The textbook power prior. One flow pass.
    At :math:`\\beta \\to 0` the prior is uniform over the training box, so the hot temperatures
    lose every astrophysical structure in :math:`\\vec\\theta` and the reversible-jump
    birth ratio goes as :math:`1/p_m`.

``"geometric"``
    :math:`\\ln p^{(\\beta)}_m = \\bar L + \\beta[\\ln p_m - \\bar L]`, a geometric bridge
    from the common reference :math:`\\bar L` to the true :math:`p_m`. Costs :math:`n`
    flow passes. Only the *model-discriminating* part of the log-density is annealed;
    the model-common part stays at full strength, so at :math:`\\beta \\to 0` the model
    index is free while :math:`\\vec\\theta` still follows a physically structured
    population density.

Both satisfy :math:`\\beta = 1 \\Rightarrow` the exact posterior, so **the cold chain is
untouched in every scheme**. For two models ``geometric`` gives

.. math::

    \\ln p^{(\\beta)}_0 - \\ln p^{(\\beta)}_1 = \\beta\\,(\\ln p_0 - \\ln p_1),
    \\qquad
    \\ln p^{(\\beta)}_0 + \\ln p^{(\\beta)}_1 = \\ln p_0 + \\ln p_1 ,

i.e. the discrimination is annealed and the constraint on :math:`\\vec\\theta` is not.

The target, and what the swap ratio must be
-------------------------------------------
Temperature :math:`t` targets :math:`\\pi_t \\propto \\exp[\\beta_t A + B]` with

.. math::

    A = \\ln L + \\ell_M - R, \\qquad B = R + \\ln p_{\\rm indep},

where :math:`R` is the model-independent reference of :math:`\\ell` (:math:`\\bar\\ell`
for ``geometric``, :math:`0` for ``flat``). For a whole-state swap between adjacent temperatures
the :math:`B` terms cancel exactly and

.. math::

    \\ln\\alpha_{\\rm swap} = \\Delta\\beta\\,[A(x_t) - A(x_{t-1})],

which is Eryn's standard form with :math:`\\ln L` replaced by :math:`\\ln L + \\ell_M - R`.
That substitution is the "combined tempered quantity" R3 requires, and it is implemented
in :meth:`~lisatools.globalfit.moves.hypermove.HyperMove.hyper_swap_log_ratio`.
Note that the swap does *not* carry every branch: the instrument-noise parameters stay
where they are, so the likelihood does not travel with the exchange and is recomputed.
See section 5 of ``_dev/prior_tempering.md`` for the form that then applies.

Which :math:`\\beta`
--------------------
The population-prior temperature is indexed by the walker's **temperature only, never
by band**. This is a modelling choice, and it is worth being explicit about why it
needs to be one: the moves in this pipeline do not share a ladder. Measured directly off
the acceptance ratios:

============================  ==================================================
move                          what multiplies ``logl`` in its acceptance ratio
============================  ==================================================
``GBSpecialRJPriorMove``      ``band_temps[band, temp]`` -- **per band**, and
                              *adapted* during the run
                              (``gbspecialstretch.py``, ``delta_logP = curr_beta *
                              ll_diff + delta_logp``, and the ladder adaptation in
                              ``run_tempering``)
``PSDMove``                   ``temperature_control.betas``, i.e.
                              ``make_ladder(ndim, ntemps, Tmax)``
``HyperMove``                 nothing; the model index was untempered
============================  ==================================================

So there is no single :math:`\\beta` in the codebase to reuse, and no choice makes the
per-temperature target exactly coherent across every move. This is a pre-existing property of
the pipeline rather than something R3 introduces, but R3 makes it *matter*, because the
prior now carries :math:`\\beta` too. One array is therefore nominated here, held by this
object, and passed by reference to every consumer.

The construction, the measurements behind it and the verification programme are in
``_dev/prior_tempering.md``; the diagnosis of the zero acceptance rate that motivates it
is in ``_dev/why_the_model_index_does_not_jump.md``.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ...utils.typing import NDArrayLike

logger = logging.getLogger(__name__)

__all__ = [
    "PopulationTempering",
    "SCHEMES",
    "DEFAULT_SCHEME",
    "resolve_scheme",
    "population_tempering_from_settings",
    "tempered_model_logpdf",
    "match_index_shape",
]

#: The schemes :class:`PopulationTempering` understands.
SCHEMES = ("off", "flat", "geometric")

#: What ``population_tempering = True`` means in the settings.
DEFAULT_SCHEME = "geometric"


def resolve_scheme(value: bool | str | None) -> str:
    """Turn a settings value into a scheme name.

    The settings field is a flag first and a choice second: ``False`` and ``True`` cover
    the two cases a run normally wants, and the scheme names remain available for the
    comparison of section 4.4 of ``_dev/prior_tempering.md``, in which the conventional
    power prior is the control against which the geometric bridge is measured.

    ============  ==========================================================
    value         scheme
    ============  ==========================================================
    ``False``     ``"off"`` -- the untempered pipeline, bit for bit
    ``True``      ``"geometric"``
    a string      that scheme, one of ``"off"``, ``"flat"``, ``"geometric"``
    ============  ==========================================================
    """
    if value is None or value is False:
        return "off"
    if value is True:
        return DEFAULT_SCHEME
    scheme = str(value).lower()
    if scheme not in SCHEMES:
        raise ValueError(
            f"Unknown population tempering scheme {value!r}; expected a bool or one of "
            f"{SCHEMES}."
        )
    return scheme


class PopulationTempering:
    """The :math:`\\beta`-weighting applied to the model-dependent population prior.

    One instance is built from the settings and handed *by reference* to every consumer
    of a model-dependent prior, so that a run has exactly one ladder and one scheme.
    :meth:`combine` is the single place the schemes differ, which keeps the consumers
    scheme-agnostic.

    Args:
        scheme: One of ``"off"``, ``"flat"``, ``"geometric"``. ``"off"`` is the default
            and reproduces the untempered pipeline bit-for-bit.
        betas: Inverse temperatures, shape ``(ntemps,)``, ordered cold to hot. Required
            unless the scheme is ``"off"``.
        allow_partial_support: Downgrade the common-support requirement of the
            ``"geometric"`` scheme from an error to a warning. See
            :meth:`check_common_support`.
    """

    def __init__(
        self,
        scheme: str = "off",
        betas: ArrayLike | None = None,
        allow_partial_support: bool = False,
    ):
        scheme = str(scheme).lower()
        if scheme not in SCHEMES:
            raise ValueError(
                f"Unknown population tempering scheme {scheme!r}; expected one of "
                f"{SCHEMES}."
            )
        self.scheme = scheme
        self.allow_partial_support = bool(allow_partial_support)

        if betas is None:
            if scheme != "off":
                raise ValueError(
                    f"The {scheme!r} population tempering scheme needs a betas array."
                )
            self.betas = None
        else:
            self.betas = np.asarray(betas, dtype=np.float64).ravel()
            if self.betas.size == 0:
                raise ValueError("betas must not be empty.")
            if np.any(self.betas < 0.0) or np.any(self.betas > 1.0):
                raise ValueError(
                    "Every beta must lie in [0, 1], and the coldest temperature must be "
                    "exactly 1, so that the cold chain samples the true posterior. Got "
                    f"{self.betas}."
                )
            if scheme != "off" and self.betas[0] != 1.0:
                raise ValueError(
                    "betas[0] must be exactly 1.0, otherwise the cold chain does not "
                    f"sample the posterior. Got {self.betas[0]}."
                )

        # so that a missed consumer is reported once rather than silently untempered
        self._warned_missing_beta = False
        # running census of where the geometric reference could not average every model
        self._reference_points = 0
        self._reference_points_partial = 0

        if self.enabled:
            logger.info(
                "PopulationTempering: scheme=%s, ntemps=%d, betas=%s. The population "
                "prior is now temperature dependent; every consumer must pass the "
                "matching beta or that temperature is not invariant with respect to anything.",
                self.scheme,
                self.ntemps,
                np.array2string(self.betas, precision=4),
            )

    # ------------------------------------------------------------------
    # description
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """``True`` unless the scheme is ``"off"``."""
        return self.scheme != "off"

    @property
    def needs_all_models(self) -> bool:
        """``True`` when :meth:`combine` requires every model's log-density.

        Only ``"geometric"`` does; the other schemes keep the existing single-model fast
        path, which for the flows is the difference between one pass and :math:`n`.
        """
        return self.scheme == "geometric"

    @property
    def ntemps(self) -> int:
        return 0 if self.betas is None else int(self.betas.size)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PopulationTempering(scheme={self.scheme!r}, "
            f"ntemps={self.ntemps}, betas={self.betas})"
        )

    def __deepcopy__(self, memo: dict) -> "PopulationTempering":
        """Survive a deep copy as *the same object*.

        ``build_gb_moves`` obtains its device-side priors by deep-copying the ones the
        rest of the pipeline holds. This object is immutable configuration -- a scheme
        name and a ladder -- so copying it would serve no purpose and would defeat the
        one guarantee that matters, that a run has exactly one ladder. Returning ``self``
        keeps the identity check in the tests meaningful.
        """
        memo[id(self)] = self
        return self

    def beta_for(self, temperature_indices: ArrayLike, xp: Any = np) -> NDArrayLike:
        """The inverse temperature of each entry of ``temperature_indices``.

        This is the only sanctioned way to turn a temperature index into a
        :math:`\\beta`, so that a consumer cannot quietly pick up a different ladder.
        """
        if self.betas is None:
            raise ValueError("This PopulationTempering has no ladder.")
        indices = xp.asarray(temperature_indices)
        return xp.asarray(self.betas)[indices.astype(xp.int32)]

    # ------------------------------------------------------------------
    # the weighting itself
    # ------------------------------------------------------------------

    def reference(self, log_p_all: NDArrayLike, xp: Any = None) -> NDArrayLike | float:
        """The model-independent reference :math:`R`, given every model's log-density.

        Args:
            log_p_all: Shape ``(nmodels, ...)``; ``log_p_all[m]`` is
                :math:`\\ln p_m` at every point.

        Returns:
            :math:`0.0` for ``"off"`` and ``"flat"``; for ``"geometric"``, the mean of
            :math:`\\ln p_m` over the models that are **finite** at each point, and
            :math:`-\\infty` at points no model can reach. Restricting the mean to the
            finite models keeps :math:`R` a well-defined function of :math:`\\vec x`
            alone -- it does not depend on which model is selected -- so the target
            stays well defined even where the boxes disagree (defect B9).
        """
        if self.scheme != "geometric":
            return 0.0

        xp = _module_of(log_p_all) if xp is None else xp
        finite = xp.isfinite(log_p_all)
        count = finite.sum(axis=0)
        total = xp.where(finite, log_p_all, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ref = xp.where(count > 0, total / xp.maximum(count, 1), -xp.inf)

        n_models = log_p_all.shape[0]
        self._reference_points += int(count.size)
        self._reference_points_partial += int((count < n_models).sum())
        return ref

    @property
    def partial_reference_fraction(self) -> float:
        """Fraction of evaluated points whose reference averaged fewer than all models.

        Zero when the competing densities share a support. Anything else is the fraction
        of the sampled volume over which the ``"geometric"`` annealing is weaker than
        equation (9) of ``_dev/prior_tempering.md`` describes, and it belongs beside any
        acceptance rate measured from such a run.
        """
        if self._reference_points == 0:
            return 0.0
        return self._reference_points_partial / self._reference_points

    def combine(
        self,
        log_p_selected: NDArrayLike,
        beta: NDArrayLike | float | None,
        log_p_all: NDArrayLike | None = None,
        reference: NDArrayLike | float | None = None,
    ) -> NDArrayLike:
        """Apply the scheme: :math:`\\ln p_m \\mapsto \\ln p^{(\\beta)}_m`.

        Args:
            log_p_selected: :math:`\\ln p_m` for each point's own model.
            beta: The inverse temperature of each point, or a scalar. ``None`` means
                :math:`\\beta = 1` and returns ``log_p_selected`` unchanged, bit-for-bit
                -- which is what makes every call site that has not been updated behave
                exactly as before. A warning is emitted once if that happens while
                tempering is enabled, because a consumer left behind is a consumer whose
                temperature is not invariant with respect to its target.
            log_p_all: Shape ``(nmodels, ...)``. Required for ``"geometric"`` unless
                ``reference`` is supplied.
            reference: A precomputed :math:`R`, to avoid recomputing it when the caller
                already has one.

        Returns:
            The tempered log-density, same shape as ``log_p_selected``.
        """
        if self.scheme == "off":
            return log_p_selected

        if beta is None:
            if not self._warned_missing_beta:
                self._warned_missing_beta = True
                logger.warning(
                    "PopulationTempering (%s) was asked to combine without a beta, so "
                    "this evaluation is untempered. Some consumer of the "
                    "model-dependent prior has not been updated; its temperature "
                    "is not invariant with respect to the tempered target.",
                    self.scheme,
                )
            return log_p_selected

        xp = _module_of(log_p_selected)
        dtype = getattr(log_p_selected, "dtype", np.float64)
        beta = xp.asarray(beta, dtype=dtype) if not np.isscalar(beta) else beta

        if self.scheme == "flat":
            # beta * (-inf) is -inf, so a point outside the model's box stays a clean
            # rejection. At beta == 0 it is nan instead, which the outer where discards
            # -- but numpy still evaluates the whole product before discarding anything,
            # so the errstate is what keeps that from printing a warning per call.
            with np.errstate(invalid="ignore"):
                out = xp.where(
                    xp.asarray(beta) == 1.0, log_p_selected, beta * log_p_selected
                )
            return xp.where(
                xp.isfinite(log_p_selected), out, xp.full_like(log_p_selected, -xp.inf)
            )

        # geometric
        if reference is None:
            if log_p_all is None:
                raise ValueError(
                    "The 'geometric' scheme needs every model's log-density (or a "
                    "precomputed reference) to build the model-independent reference."
                )
            reference = self.reference(log_p_all, xp=xp)

        selected_finite = xp.isfinite(log_p_selected)
        reference_finite = xp.isfinite(reference)
        safe_selected = xp.where(selected_finite, log_p_selected, 0.0)
        safe_reference = xp.where(reference_finite, reference, 0.0)

        out = safe_reference + beta * (safe_selected - safe_reference)
        # R + 1*(ln p - R) is ln p only up to rounding, and the coldest temperature is
        # the one the results are read from: make it exactly the untempered value, so
        # that switching the scheme on cannot perturb the cold chain's arithmetic.
        out = xp.where(xp.asarray(beta) == 1.0, safe_selected, out)
        # A model cannot reach this point: impossible at any beta > 0, and at beta == 0
        # the prior is the reference, which every model shares by construction.
        out = xp.where(
            selected_finite,
            out,
            xp.where(xp.asarray(beta) == 0.0, safe_reference, -xp.inf),
        )
        # No model can reach this point.
        return xp.where(reference_finite, out, xp.full_like(out, -xp.inf))

    def temper_ell(
        self, ell: NDArrayLike, beta: NDArrayLike | float | None
    ) -> NDArrayLike:
        """Temper a full ``(nmodels, ntemps, nwalkers)`` array of :math:`\\ell_m`.

        The counterpart of :meth:`combine` for
        :meth:`~lisatools.globalfit.moves.hypermove.HyperMove.compute_model_terms`,
        which already holds every model's value. The reference is taken over the model
        axis, so the acceptance ratio :math:`\\ell^{(\\beta)}_{m'} - \\ell^{(\\beta)}_m`
        reduces to :math:`\\beta\\,(\\ell_{m'} - \\ell_m)` under **both** schemes.
        """
        if self.scheme == "off" or beta is None:
            if self.enabled and beta is None and not self._warned_missing_beta:
                self._warned_missing_beta = True
                logger.warning(
                    "PopulationTempering (%s): temper_ell called without a beta.",
                    self.scheme,
                )
            return ell

        xp = _module_of(ell)
        reference = self.reference(ell, xp=xp)
        beta = xp.asarray(beta)
        out = xp.empty_like(ell)
        for model in range(ell.shape[0]):
            out[model] = self.combine(ell[model], beta, reference=reference)
        return out

    # ------------------------------------------------------------------
    # the guard the geometric scheme needs
    # ------------------------------------------------------------------

    def check_common_support(
        self, models: Sequence[Any], label: str = "priors"
    ) -> str:
        """Grade a family of competing densities on whether they share a support.

        Why the ``"geometric"`` scheme cares. Its reference is the mean of
        :math:`\\ln p_{m'}` over the models that are finite at a point. Where every model
        is finite, that is the geometric mean of equation (6) of
        ``_dev/prior_tempering.md`` and the annealing behaves as documented. Where only
        one model has support, the reference collapses onto that model's own
        log-density, so

        .. math::

            \\ln p^{(\\beta)}_m = R + \\beta[\\ln p_m - R] = \\ln p_m ,

        and **the annealing switches itself off exactly there**. It is not wrong -- the
        target is still well defined -- but it is inert at precisely the points that make
        a barrier infinite, so a family with a large unshared region buys nothing from
        this scheme. That is what has to be measured before a run rather than discovered
        after one.

        The grading, from the training boxes alone (no flow evaluation, no GPU):

        ``"shared"``
            Identical training boxes, or every model carrying the same active support
            floor **over the same broad box**, which makes every density positive on the
            same region. An equal :math:`\\epsilon` over *different* boxes is refused
            outright rather than graded, since it is a configuration error rather than a
            property of the fits.
        ``"partial"``
            Boxes overlap in every coordinate but are not identical. The annealing is
            inhomogeneous over the unshared fraction, which is reported. Warned, not
            refused: the population priors of the current configuration sit here, with
            boxes agreeing to better than one part in a hundred.
        ``"disjoint"``
            Some coordinate has no overlap at all. The scheme is inert wherever the
            models disagree, which is everywhere that matters. Refused unless
            ``allow_partial_support`` is set.

        Args:
            models: The per-model density objects, each exposing ``param_min`` and
                ``param_max``, and optionally ``support_floor_epsilon``.
            label: Name of the family, used in the messages.

        Returns:
            One of ``"shared"``, ``"partial"``, ``"disjoint"``, or ``"unknown"`` when the
            boxes cannot be read.

        Raises:
            ValueError: On ``"disjoint"`` when ``allow_partial_support`` is not set.
        """
        if self.scheme != "geometric" or len(models) < 2:
            return "shared"

        floors = [getattr(m, "support_floor_epsilon", None) for m in models]
        if all(floor is not None for floor in floors):
            if len(set(floors)) != 1:
                raise ValueError(
                    f"{label}: the models carry different support floors {floors}, so "
                    "they do not share a broad box, and the geometric reference would "
                    "average densities defined on different regions. Give them a common "
                    "floor (apply_common_support_floor)."
                )
            # An equal epsilon is not on its own a shared support: set_support_floor is
            # public and takes a box per model, so two models can be floored to the same
            # weight over different regions. What makes the support shared is the broad
            # box, which is what apply_common_support_floor equalises.
            broad = [getattr(m, "support_floor_box", None) for m in models]
            if any(box is None for box in broad):
                logger.warning(
                    "%s: the models report a support floor but not the box it acts over, "
                    "so whether they really share a support could not be established. "
                    "Falling back to comparing the training boxes.",
                    label,
                )
            elif all(
                np.array_equal(box[0], broad[0][0]) and np.array_equal(box[1], broad[0][1])
                for box in broad
            ):
                logger.info(
                    "%s: support is shared through a common floor (epsilon = %.3e) over "
                    "one broad box, so the geometric reference averages every model "
                    "everywhere.",
                    label,
                    floors[0],
                )
                return "shared"
            else:
                raise ValueError(
                    f"{label}: the models carry the same support floor epsilon "
                    f"{floors[0]:.3e} but over different broad boxes, so each density is "
                    "positive on a different region and the geometric reference would "
                    "average densities defined on different supports. Give them one box "
                    "(apply_common_support_floor)."
                )

        boxes = []
        for model in models:
            low = getattr(model, "param_min", None)
            high = getattr(model, "param_max", None)
            if low is None or high is None:
                logger.warning(
                    "%s: cannot read a training box off %s, so whether the geometric "
                    "reference is well posed could not be checked before the run.",
                    label,
                    type(model).__name__,
                )
                return "unknown"
            boxes.append((_as_numpy(low).ravel(), _as_numpy(high).ravel()))

        # The intersection of every model's box, against the union, per coordinate.
        low_shared = np.max([box[0] for box in boxes], axis=0)
        high_shared = np.min([box[1] for box in boxes], axis=0)
        low_union = np.min([box[0] for box in boxes], axis=0)
        high_union = np.max([box[1] for box in boxes], axis=0)

        overlap = high_shared - low_shared
        union = np.maximum(high_union - low_union, 1e-300)
        shared_fraction = np.clip(overlap / union, 0.0, 1.0)

        if all(
            np.array_equal(box[0], boxes[0][0]) and np.array_equal(box[1], boxes[0][1])
            for box in boxes
        ):
            return "shared"

        worst_coord = int(np.argmin(shared_fraction))
        worst = float(shared_fraction[worst_coord])

        if np.any(overlap <= 0.0):
            empty = int(np.argmin(overlap))
            message = (
                f"{label}: the models' training boxes do not overlap in coordinate "
                f"{empty}, so under the geometric scheme the reference is a single "
                "model there and the annealing is inert wherever the models actually "
                "differ. Give the densities a common support first -- a shared support "
                "floor, or a retraining on a common box -- or set "
                "allow_partial_support to proceed anyway."
            )
            if self.allow_partial_support:
                logger.warning("%s Continuing: allow_partial_support is set.", message)
                return "disjoint"
            raise ValueError(message)

        logger.warning(
            "%s: the training boxes overlap but are not identical; the worst coordinate "
            "is %d, where only %.4f of the union is shared. Over the unshared %.2f%% "
            "the geometric reference falls back to fewer models and the annealing is "
            "correspondingly weaker. Proceeding.",
            label,
            worst_coord,
            worst,
            100.0 * (1.0 - worst),
        )
        return "partial"


def match_index_shape(index: NDArrayLike, shape: tuple, xp: Any) -> NDArrayLike:
    """Bring a model-index or inverse-temperature array onto the output shape.

    .. warning::
       **Not yet called by anything, and therefore not exercised by any test.** It
       belongs to the deferred step of section 6.1 of ``_dev/prior_tempering.md``, in
       which the weighting is extended to the conditional prior classes. It is kept
       because it encodes the shape handling those classes already rely on, but it should
       be treated as a draft until that step lands and tests it.

    The conditional priors accept indices in several shapes -- one per point, one per
    walker with a trailing singleton leaf axis, or a scalar -- and previously resolved
    that with a ``squeeze`` applied to the per-model comparison mask. Doing it once, to
    the index itself, gives the same masks while also making the array usable for the
    gather the geometric scheme needs.
    """
    if index.shape == shape:
        return index
    target_size = 1
    for extent in shape:
        target_size *= extent
    if index.size == target_size:
        return index.reshape(shape)
    return xp.broadcast_to(index, shape)


def tempered_model_logpdf(
    tempering: PopulationTempering | None,
    n_models: int,
    model_index: NDArrayLike,
    beta: NDArrayLike | float | None,
    evaluate: Any,
    xp: Any,
    shape: tuple,
) -> NDArrayLike:
    """Evaluate a model-indexed log-density and apply the tempering scheme.

    .. warning::
       **Not yet called by anything, and therefore not exercised by any test.** The
       tempering currently lives entirely in the model move, which holds every model's
       :math:`\\ell_m` already and so needs no dispatch. This function is the dispatch the
       *conditional prior classes* would need under the deferred step of section 6.1 of
       ``_dev/prior_tempering.md``. It is kept because it records how that step should
       work -- one evaluation of every model when the reference is needed, the existing
       single-model path otherwise -- but it should be treated as a draft until that step
       lands and tests it.

    The dispatch is shared by every conditional prior -- the population prior over
    resolved sources, the foreground prior and the Poisson prior on the number of
    sources -- because they differ only in how a single model is evaluated, not in how
    the models are selected or combined.

    Args:
        tempering: The scheme in force; ``None`` behaves as ``"off"``.
        n_models: Number of competing models.
        model_index: Integer model index per point, already broadcast to ``shape``.
        beta: Inverse temperature per point, or a scalar, or ``None`` for
            :math:`\\beta = 1`.
        evaluate: ``evaluate(model, mask)`` returning :math:`\\ln p_{\\rm model}` at the
            points selected by ``mask``; ``mask is None`` means every point.
        xp: The array module the result should live in.
        shape: Shape of the output.

    Returns:
        The tempered log-density, shape ``shape``.

    Notes:
        When the scheme needs the model-independent reference, every model is evaluated
        at every point -- for a normalising flow that is :math:`n` passes instead of one.
        Otherwise the per-model mask loop is used unchanged, so the ``"off"`` and
        ``"flat"`` schemes cost exactly what the untempered pipeline costs.
    """
    valid = (model_index >= 0) & (model_index < n_models)

    if tempering is not None and tempering.needs_all_models:
        log_p_all = xp.stack(
            [xp.asarray(evaluate(model, None)).reshape(shape) for model in range(n_models)]
        )
        safe_index = xp.clip(model_index, 0, n_models - 1).astype(xp.int64)
        selected = xp.take_along_axis(log_p_all, safe_index[None, ...], axis=0)[0]
        selected = xp.where(valid, selected, -xp.inf)
        return tempering.combine(selected, beta, log_p_all=log_p_all)

    out = xp.full(shape, -np.inf, dtype=xp.float64)
    for model in range(n_models):
        mask = valid & (model_index == model)
        if not mask.any():
            continue
        out[mask] = xp.asarray(evaluate(model, mask))

    if tempering is None:
        return out
    return tempering.combine(out, beta)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _module_of(array: Any) -> Any:
    """``numpy`` or ``cupy``, whichever owns ``array``; ``numpy`` for anything else."""
    module = getattr(type(array), "__module__", "")
    if module.startswith("cupy"):
        import cupy

        return cupy
    return np


def _as_numpy(values: Any) -> np.ndarray:
    """A numpy view of a numpy array, a CuPy array or a torch buffer alike."""
    detach = getattr(values, "detach", None)  # torch
    if callable(detach):
        return detach().cpu().numpy()
    getter = getattr(values, "get", None)  # CuPy
    return np.asarray(getter() if callable(getter) else values)


def population_tempering_from_settings(
    population_tempering: bool | str | None,
    betas: ArrayLike | None,
    allow_partial_support: bool = False,
) -> PopulationTempering:
    """Build the object from the settings fields.

    Args:
        population_tempering: The settings flag; see :func:`resolve_scheme`.
        betas: The ladder, set in the settings file exactly as the galactic-binary
            ladder is. Ignored when the flag resolves to ``"off"``.
        allow_partial_support: See
            :meth:`PopulationTempering.check_common_support`.
    """
    scheme = resolve_scheme(population_tempering)
    return PopulationTempering(
        scheme=scheme,
        betas=None if scheme == "off" else betas,
        allow_partial_support=allow_partial_support,
    )
