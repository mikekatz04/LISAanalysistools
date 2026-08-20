"""Instrumentation of the model-index terms of :class:`~lisatools.globalfit.moves.hypermove.HyperMove`.

This module implements diagnostic **D1** of
``_dev/why_the_model_index_does_not_jump.md``: log

.. math::

    \\ell_m = \\ln p(\\Sigma \\mid m)
            + \\sum_{i=1}^{k_1} \\ln \\mathcal N(\\vec\\theta_i \\mid m)
            - \\hat N_1(\\Sigma, m)
            + \\ln p(m)

for every model :math:`m`, walker, temperature and iteration, decomposed into the
inhomogeneous-Poisson-process grouping of §1.2 of that note,

* ``intensity``   : :math:`\\sum_i \\ln \\mathcal N(\\vec\\theta_i \\mid m)`, with
  :math:`\\mathcal N = \\hat N_{\\rm tot}(m)\\, p_{\\rm pop}(\\vec\\theta \\mid m)`,
* ``n1_expected`` : :math:`\\hat N_1(\\Sigma, m)`, entering :math:`\\ell_m` with a minus sign,
* ``stochastic``  : :math:`\\ln p(\\Sigma \\mid m)`, the galactic-foreground term.

Both halves live here: :class:`HyperMoveRecorder` is the append-only writer used by
the move, :class:`HyperMoveDiagnostics` reads a recording back and turns it into the
§1.2 table, the Rao-Blackwellised model posterior of R4 and the per-source numbers
that set the requirements of §3.

The two terms that do *not* depend on the model index -- the likelihood
:math:`p(\\vec d \\mid \\{\\vec\\theta\\}_{k_1}, \\Sigma)` and the :math:`1/k_1!`
combinatorial factor -- are omitted from :math:`\\ell_m`, exactly as in R4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    """A stable ``log(sum(exp(values)))`` that leaves an all-``-inf`` slice at ``-inf``."""
    largest = np.max(values, axis=axis, keepdims=True)
    largest = np.where(np.isfinite(largest), largest, 0.0)
    return np.squeeze(
        largest + np.log(np.sum(np.exp(values - largest), axis=axis, keepdims=True)),
        axis=axis,
    )


def _effective_sample_size(log_weights: np.ndarray) -> float:
    """Kish's ESS of self-normalised log weights, ``(sum w)^2 / sum w^2``.

    Reported alongside every reweighted quantity because it is the honest measure of
    whether the reweighting means anything. Undoing a support floor drives it towards
    one: the states the floored target allowed are precisely the states the unfloored
    target assigns no mass to.
    """
    finite = log_weights[np.isfinite(log_weights)]
    if finite.size == 0:
        return 0.0
    shifted = finite - finite.max()
    total = np.sum(np.exp(shifted))
    return float(total**2 / np.sum(np.exp(2.0 * shifted)))


def _log_epsilon_slope(rows: Sequence[Mapping[str, Any]]) -> float:
    """:math:`\\partial \\ln B_{10} / \\partial \\ln\\epsilon`, by least squares.

    An integer, in the regimes that matter: the number of floored density factors by
    which the two models differ. Zero says the floor is inert and the Bayes factor is
    the analysis's; one says it is :math:`\\epsilon`'s.
    """
    points = [
        (np.log(row["epsilon"]), row["ln_posterior_odds"])
        for row in rows
        if row["epsilon"] is not None and np.isfinite(row["ln_posterior_odds"])
    ]
    if len(points) < 2:
        return float("nan")
    x, y = np.asarray(points, dtype=np.float64).T
    if np.ptp(x) == 0.0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


class HyperMoveRecorder:
    """Append-only HDF5 writer for the per-model log-target terms of ``HyperMove``.

    One row is appended per call to :meth:`record`, i.e. per proposal of the move.
    Writes are flushed immediately: the recording of a job that is killed mid-run is
    still readable, and the file is small (a few tens of kB per iteration).

    Args:
        filename: Destination HDF5 file. Created if absent, appended to otherwise.
        nmodels: Number of discrete models.
        ntemps: Number of temperatures.
        nwalkers: Number of walkers.
        n_tot: Catalogue sizes :math:`\\hat N_{\\rm tot}(m)`, shape ``(nmodels,)``.
        model_log_prior: :math:`\\ln p(m)`, shape ``(nmodels,)``.
        metadata: Extra scalar/array attributes stored on the group, e.g. the flow
            configuration files and the SNR threshold, so a recording is
            self-describing.
        overwrite: If ``True``, truncate an existing file instead of appending.
    """

    #: Terms carrying a model axis, shape ``(nmodels, ntemps, nwalkers)`` per row.
    PER_MODEL_TERMS: tuple[str, ...] = (
        "resolved",
        "intensity",
        "n1_expected",
        "stochastic",
        "ell",
    )

    #: Per-model terms recorded when the move supplies them, and tolerated as absent.
    #:
    #: ``ell_tempered`` is the beta-weighted model term the acceptance ratio actually
    #: used. It is recorded *beside* the raw ``ell`` rather than in place of it, because
    #: the Rao-Blackwellised model posterior of section 6.4 of
    #: ``_dev/prior_tempering.md`` needs the raw values -- a tempered one describes a
    #: different target. Optional rather than required so that a recording written
    #: before this existed stays readable and appendable.
    OPTIONAL_PER_MODEL_TERMS: tuple[str, ...] = ("ell_tempered",)

    #: Integer bookkeeping, shape ``(ntemps, nwalkers)`` per row.
    INDEX_TERMS: dict[str, Any] = {
        "model_current": np.int16,
        "model_proposed": np.int16,
        "num_resolved": np.int32,
    }

    #: Boolean bookkeeping, shape ``(ntemps, nwalkers)`` per row.
    FLAG_TERMS: tuple[str, ...] = ("accepted",)

    GROUP = "hyper_move"

    def __init__(
        self,
        filename: str | Path,
        nmodels: int,
        ntemps: int,
        nwalkers: int,
        n_tot: Sequence[float],
        model_log_prior: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ):
        self.path = Path(filename)
        self.filename = str(self.path)
        self.nmodels = int(nmodels)
        self.ntemps = int(ntemps)
        self.nwalkers = int(nwalkers)

        self.path.parent.mkdir(parents=True, exist_ok=True)

        mode = "w" if overwrite else "a"
        with h5py.File(self.filename, mode) as f:
            group = f.require_group(self.GROUP)
            if "iteration" not in group:
                self._create_datasets(group)
                group.attrs["nmodels"] = self.nmodels
                group.attrs["ntemps"] = self.ntemps
                group.attrs["nwalkers"] = self.nwalkers
                group.attrs["n_tot"] = np.asarray(n_tot, dtype=np.float64)
                group.attrs["ln_n_tot"] = np.log(np.asarray(n_tot, dtype=np.float64))
                group.attrs["model_log_prior"] = np.asarray(
                    model_log_prior, dtype=np.float64
                )
                for key, value in (metadata or {}).items():
                    group.attrs[key] = value
            else:
                stored = (
                    int(group.attrs["nmodels"]),
                    int(group.attrs["ntemps"]),
                    int(group.attrs["nwalkers"]),
                )
                if stored != (self.nmodels, self.ntemps, self.nwalkers):
                    raise ValueError(
                        f"{self.filename} was written with (nmodels, ntemps, nwalkers)"
                        f" = {stored}, cannot append {(self.nmodels, self.ntemps, self.nwalkers)}."
                        " Pass overwrite=True or a new filename."
                    )
            self._nrows = int(group["iteration"].shape[0])

        logger.info(
            "HyperMove diagnostics (D1) -> %s (%d row(s) already present)",
            self.filename,
            self._nrows,
        )

    @property
    def nrows(self) -> int:
        """Number of rows recorded so far."""
        return self._nrows

    def _create_datasets(self, group: h5py.Group) -> None:
        model_shape = (self.nmodels, self.ntemps, self.nwalkers)
        walker_shape = (self.ntemps, self.nwalkers)

        group.create_dataset(
            "iteration", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(64,)
        )
        for name in self.PER_MODEL_TERMS + self.OPTIONAL_PER_MODEL_TERMS:
            group.create_dataset(
                name,
                shape=(0,) + model_shape,
                maxshape=(None,) + model_shape,
                dtype=np.float64,
                chunks=(1,) + model_shape,
            )
        for name, dtype in self.INDEX_TERMS.items():
            group.create_dataset(
                name,
                shape=(0,) + walker_shape,
                maxshape=(None,) + walker_shape,
                dtype=dtype,
                chunks=(8,) + walker_shape,
            )
        for name in self.FLAG_TERMS:
            group.create_dataset(
                name,
                shape=(0,) + walker_shape,
                maxshape=(None,) + walker_shape,
                dtype=bool,
                chunks=(8,) + walker_shape,
            )

    def record(self, iteration: int, terms: Mapping[str, np.ndarray]) -> None:
        """Append one row.

        Args:
            iteration: Iteration (or proposal) counter stored alongside the row.
            terms: One entry per dataset name in :attr:`PER_MODEL_TERMS`,
                :attr:`INDEX_TERMS` and :attr:`FLAG_TERMS`. Shapes are validated;
                CuPy arrays are moved to the host.
        """
        model_shape = (self.nmodels, self.ntemps, self.nwalkers)
        walker_shape = (self.ntemps, self.nwalkers)

        expected: dict[str, tuple[int, ...]] = {}
        expected.update({name: model_shape for name in self.PER_MODEL_TERMS})
        expected.update({name: walker_shape for name in self.INDEX_TERMS})
        expected.update({name: walker_shape for name in self.FLAG_TERMS})

        missing = set(expected) - set(terms)
        if missing:
            raise KeyError(f"Missing terms for the D1 recording: {sorted(missing)}.")

        row: dict[str, np.ndarray] = {}
        for name, shape in expected.items():
            value = terms[name]
            value = value.get() if hasattr(value, "get") else np.asarray(value)
            if value.shape != shape:
                raise ValueError(
                    f"Term '{name}' has shape {value.shape}, expected {shape}."
                )
            row[name] = value

        with h5py.File(self.filename, "a") as f:
            group = f[self.GROUP]

            # Optional terms are skipped when the *file* has nowhere to put them, so a
            # recording written before they existed can still be appended to instead of
            # raising halfway through a run. But once a dataset exists it is written on
            # every row, filled with nan where the caller supplied nothing: a dataset
            # that grows more slowly than the others would silently stop lining up with
            # them, and a misaligned row is worse than a missing one.
            for name in self.OPTIONAL_PER_MODEL_TERMS:
                if name not in group:
                    continue
                if name not in terms:
                    row[name] = np.full(model_shape, np.nan)
                    continue
                value = terms[name]
                value = value.get() if hasattr(value, "get") else np.asarray(value)
                if value.shape != model_shape:
                    raise ValueError(
                        f"Term '{name}' has shape {value.shape}, expected {model_shape}."
                    )
                row[name] = value

            index = int(group["iteration"].shape[0])
            group["iteration"].resize(index + 1, axis=0)
            group["iteration"][index] = int(iteration)
            for name, value in row.items():
                group[name].resize(index + 1, axis=0)
                group[name][index] = value
            self._nrows = index + 1


class HyperMoveDiagnostics:
    """Reader and post-processor for a :class:`HyperMoveRecorder` file.

    Everything is loaded into memory; a recording of a few thousand iterations is a
    few tens of MB at most.

    Args:
        filename: HDF5 file written by :class:`HyperMoveRecorder`.
        discard: Number of leading rows to drop (burn-in).
    """

    def __init__(self, filename: str | Path, discard: int = 0):
        self.path = Path(filename)
        self.filename = str(self.path)
        self.discard = int(discard)

        self.data: dict[str, np.ndarray] = {}
        with h5py.File(self.filename, "r") as f:
            group = f[HyperMoveRecorder.GROUP]
            self.attrs = {key: value for key, value in group.attrs.items()}
            for name, dataset in group.items():
                self.data[name] = np.asarray(dataset[self.discard :])

        self.nmodels = int(self.attrs["nmodels"])
        self.ntemps = int(self.attrs["ntemps"])
        self.nwalkers = int(self.attrs["nwalkers"])
        self.n_tot = np.asarray(self.attrs["n_tot"], dtype=np.float64)
        self.ln_n_tot = np.asarray(self.attrs["ln_n_tot"], dtype=np.float64)
        self.model_log_prior = np.asarray(
            self.attrs["model_log_prior"], dtype=np.float64
        )
        self.niterations = int(self.data["iteration"].shape[0])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[name]

    def term(self, name: str, temperature: int | None = 0) -> np.ndarray:
        """Return a recorded term, optionally restricted to one temperature.

        Args:
            name: Dataset name.
            temperature: Temperature index to select, or ``None`` for all of them.
                The default, ``0``, is the cold chain.
        """
        value = self.data[name]
        if temperature is None:
            return value
        # per-model terms carry the model axis at position 1
        per_model = (
            HyperMoveRecorder.PER_MODEL_TERMS
            + HyperMoveRecorder.OPTIONAL_PER_MODEL_TERMS
        )
        axis = 2 if name in per_model else 1
        return np.take(value, temperature, axis=axis)

    def raw_ell(self, temperature: int | None = 0) -> np.ndarray:
        """:math:`\\ell_m` as equation (1) defines it, never the tempered weighting.

        Section 6.4 of ``_dev/prior_tempering.md``: the Rao-Blackwellised model posterior
        and the Bayes factor are statements about the *target*, so they need the raw
        model terms. Substituting :math:`\\beta_t \\ell_m` would return the model
        posterior of a different distribution -- and would do it silently, since the two
        arrays have the same shape, the same sign and the same rough magnitude at the
        cold end.

        This is where that is checked rather than assumed. When a recording carries both
        arrays and says tempering was on, the two *must* differ away from :math:`\\beta =
        1`; if they do not, the tempered array was written under the raw name and every
        number downstream is a number about the wrong target.
        """
        ell = self.term("ell", temperature=temperature)

        tempered = self.data.get("ell_tempered")
        scheme = str(self.attrs.get("population_tempering_scheme", "off"))
        betas = np.asarray(self.attrs.get("population_tempering_betas", [1.0]))
        if tempered is None or scheme == "off" or np.all(betas == 1.0):
            return ell

        # compare where the weighting must bite: the hottest temperature of the ladder
        hottest = int(np.argmin(betas))
        raw_hot = np.take(self.data["ell"], hottest, axis=2)
        tempered_hot = np.take(tempered, hottest, axis=2)
        finite = np.isfinite(raw_hot) & np.isfinite(tempered_hot)
        if finite.any() and np.allclose(raw_hot[finite], tempered_hot[finite]):
            raise ValueError(
                f"{self.filename} records scheme {scheme!r} with beta = "
                f"{betas[hottest]:.3e} at temperature {hottest}, yet 'ell' and "
                "'ell_tempered' agree there. The tempered terms were written under the "
                "raw name, so the model posterior computed from this file describes the "
                "tempered target rather than the posterior. See section 6.4 of "
                "_dev/prior_tempering.md."
            )
        return ell

    def delta_ln_alpha(
        self, model_from: int = 0, model_to: int = 1, temperature: int | None = 0
    ) -> np.ndarray:
        """:math:`\\ell_{m'} - \\ell_m`, the log acceptance ratio of the move.

        For a symmetric model proposal with :math:`p(m)` uniform this is exactly the
        Metropolis-Hastings ratio evaluated by ``HyperMove``.
        """
        ell = self.raw_ell(temperature=temperature)
        return ell[:, model_to] - ell[:, model_from]

    def rao_blackwell(self, temperature: int | None = 0) -> np.ndarray:
        """Rao-Blackwellised model posterior (R4).

        Averages the per-state softmax of :math:`\\ell_m` over iterations and
        walkers, which is exact provided the :math:`(\\vec\\theta, \\Sigma)` chain
        samples the :math:`M`-marginalised posterior.

        A model with :math:`\\ell_m = -\\infty` -- a state outside that model's flow
        support -- contributes a weight of exactly zero, which is correct and common
        here. A state is skipped only when its weights are undefined: some
        :math:`\\ell_m` is ``nan`` or :math:`+\\infty`, or every model is impossible.

        Returns:
            Array of shape ``(nmodels,)`` summing to one.
        """
        return self._rao_blackwell_from_ell(self.raw_ell(temperature=temperature))

    def _rao_blackwell_from_ell(
        self, ell: np.ndarray, log_weights: np.ndarray | None = None
    ) -> np.ndarray:
        """The R4 estimator on an arbitrary :math:`\\ell_m` array.

        Split out of :meth:`rao_blackwell` so that the same estimator can be applied to
        an :math:`\\ell_m` re-evaluated at another support-floor :math:`\\epsilon`
        (:meth:`ell_at_epsilon`), which is the whole of D14.

        Args:
            ell: Shape ``(niterations, nmodels, nwalkers)``, the model axis at 1.
            log_weights: Optional per-state log importance weights, shape
                ``(niterations, nwalkers)``, for reweighting a chain that was run at a
                different target. Self-normalised here.
        """
        # move the model axis last, then a stable softmax over it
        ell = np.moveaxis(ell, 1, -1)

        largest = np.max(ell, axis=-1)
        usable = np.isfinite(largest) & ~np.any(
            np.isnan(ell) | np.isposinf(ell), axis=-1
        )
        if log_weights is not None:
            usable &= np.isfinite(log_weights)
        if not np.any(usable):
            logger.warning(
                "No state has a well defined set of ell_m; the Rao-Blackwell "
                "estimator is undefined for this recording."
            )
            return np.full(self.nmodels, np.nan)

        ell = ell[usable].reshape(-1, self.nmodels)
        weights = np.exp(ell - ell.max(axis=-1, keepdims=True))
        weights /= np.sum(weights, axis=-1, keepdims=True)
        if log_weights is None:
            return np.asarray(weights.mean(axis=0))

        state = np.exp(log_weights[usable] - np.max(log_weights[usable]))
        total = float(np.sum(state))
        if not total > 0.0:
            return np.full(self.nmodels, np.nan)
        return np.asarray(state @ weights / total)

    # ------------------------------------------------------------------
    # D14: what the support floor is doing to the model comparison
    # ------------------------------------------------------------------

    @property
    def stochastic_support_floor(self) -> tuple[float, float] | None:
        """``(epsilon, -ln|B|)`` of the galfor floor, or ``None`` if it was off.

        Read from the attributes ``HyperMove._flow_metadata`` writes. A recording made
        before those attributes existed reports ``None``, which is indistinguishable
        from a run with the floor off -- correctly, since in both cases the recorded
        ``stochastic`` term is the raw flow density.
        """
        keys = ("stochastic_support_floor_epsilon", "stochastic_support_floor_log_uniform")
        if not all(key in self.attrs for key in keys):
            return None
        return float(self.attrs[keys[0]]), float(self.attrs[keys[1]])

    def stochastic_floor_margin(self, temperature: int | None = 0) -> np.ndarray:
        """:math:`\\ln p_\\epsilon(\\Sigma\\mid m) - (\\ln\\epsilon - \\ln|B|)`, in nats.

        The number that decides whether the recorded stochastic term is a statement
        about the population or a statement about :math:`\\epsilon`. Large margin, the
        flow is speaking; margin near zero, the floor is. ``nan`` when the floor was off.
        """
        floor = self.stochastic_support_floor
        stochastic = self.term("stochastic", temperature=temperature)
        if floor is None:
            return np.full_like(stochastic, np.nan)
        epsilon, log_uniform = floor
        return stochastic - (np.log(epsilon) + log_uniform)

    def stochastic_at_epsilon(
        self, epsilon: float | None, temperature: int | None = 0
    ) -> np.ndarray:
        """The stochastic term as it would have been at another :math:`\\epsilon`.

        The floored galfor density is one flow evaluation per state, so it inverts
        exactly: undo the recorded floor, then re-apply the requested one. No rerun and
        no flow evaluation. ``epsilon=None`` returns the unfloored
        :math:`\\ln p_{\\rm flow}(\\Sigma\\mid m)`, which is :math:`-\\infty` wherever
        the floor was carrying the state -- that being the honest answer, see §10.6.

        The same trick does *not* work for the ``resolved`` term: that is a sum of
        :math:`k_1` logs, and a sum of logs cannot be un-mixed term by term. Use
        ``_testing/diagnose_support_floor.py`` for the GB side.
        """
        from .priors.network import refloor_log_density, unfloor_log_density

        stochastic = self.term("stochastic", temperature=temperature)
        floor = self.stochastic_support_floor
        if floor is None:
            if epsilon is None:
                return stochastic
            raise ValueError(
                "This recording carries no support-floor attributes, so its stochastic "
                "term is already the raw flow density and there is no recorded ln|B| to "
                "re-floor it with. Re-evaluate the priors instead (D14 check 3)."
            )

        recorded_epsilon, log_uniform = floor
        raw = unfloor_log_density(stochastic, recorded_epsilon, log_uniform)
        if epsilon is None:
            return raw
        return refloor_log_density(raw, epsilon, log_uniform)

    def ell_at_epsilon(
        self, epsilon: float | None, temperature: int | None = 0
    ) -> np.ndarray:
        """:math:`\\ell_m` with the galfor floor moved to another :math:`\\epsilon`.

        Only the stochastic term changes: the floor enters ``ell`` through (S) and (R),
        and (T) is computed from the catalogues, not from any flow.
        """
        ell = self.term("ell", temperature=temperature)
        recorded = self.term("stochastic", temperature=temperature)
        return ell - recorded + self.stochastic_at_epsilon(epsilon, temperature)

    def epsilon_scan(
        self,
        epsilons: Sequence[float | None],
        model_from: int = 0,
        model_to: int = 1,
        temperature: int | None = 0,
        reweight: bool = True,
    ) -> dict[str, Any]:
        """D14: how the model comparison moves when :math:`\\epsilon` moves.

        In the regime the floor was introduced for, :math:`p_{\\rm flow}(\\Sigma\\mid M_1)`
        lies far below :math:`\\epsilon/|B|` and the mixture returns
        :math:`\\ln\\epsilon - \\ln|B|` regardless of the population, so

        .. math::
            \\frac{\\partial \\ln B_{10}}{\\partial \\ln\\epsilon}
            = n_{\\rm floored}(M_1) - n_{\\rm floored}(M_0),

        which is :math:`1` when the floor is carrying :math:`M_1` alone and :math:`0`
        when the floor is inert. **A slope of one means the reported Bayes factor is a
        restatement of** :math:`\\epsilon`. That is the number this method exists to
        produce; everything else it returns is context for it.

        ``reweight`` controls what the model posterior at a new :math:`\\epsilon` means.
        The chain was run at the recorded :math:`\\epsilon`, so a plain average of the
        softmax answers "what would R4 have reported from these states", while the
        importance-weighted average answers "what would a run at this
        :math:`\\epsilon` have reported". The weights are
        :math:`w = \\sum_m e^{\\ell_m^{\\rm new}} / \\sum_m e^{\\ell_m^{\\rm rec}}`, and
        their effective sample size is returned because it collapses in exactly the
        regime that matters: the states the floor allowed are the ones the unfloored
        target gives no mass to.

        Args:
            epsilons: The grid, ``None`` meaning the unfloored density.
            model_from: Denominator of the Bayes factor.
            model_to: Numerator.
            temperature: Temperature index, cold chain by default.
            reweight: Importance-reweight to the new target, as above.
        """
        recorded_ell = self.term("ell", temperature=temperature)
        rows: list[dict[str, Any]] = []
        for epsilon in epsilons:
            ell = self.ell_at_epsilon(epsilon, temperature=temperature)
            log_weights = None
            ess = float(ell.shape[0] * ell.shape[-1])
            if reweight:
                log_weights = _logsumexp(ell, axis=1) - _logsumexp(recorded_ell, axis=1)
                ess = _effective_sample_size(log_weights)
            posterior = self._rao_blackwell_from_ell(ell, log_weights=log_weights)
            delta = ell[:, model_to] - ell[:, model_from]
            with np.errstate(divide="ignore", invalid="ignore"):
                ln_odds = float(
                    np.log(posterior[model_to]) - np.log(posterior[model_from])
                )
            rows.append(
                dict(
                    epsilon=epsilon,
                    posterior=posterior,
                    ln_posterior_odds=ln_odds,
                    mean_delta_ell=float(np.mean(delta[np.isfinite(delta)]))
                    if np.any(np.isfinite(delta))
                    else -np.inf,
                    finite_fraction=float(np.mean(np.isfinite(delta))),
                    effective_sample_size=ess,
                )
            )

        return dict(
            rows=rows,
            model_from=model_from,
            model_to=model_to,
            recorded_floor=self.stochastic_support_floor,
            slope=_log_epsilon_slope(rows),
        )

    def per_source(self, name: str, temperature: int | None = 0) -> np.ndarray:
        """A term divided by :math:`k_1`, the quantity §3 states requirements on."""
        value = self.term(name, temperature=temperature)
        k1 = self.term("num_resolved", temperature=temperature).astype(np.float64)
        if name in HyperMoveRecorder.PER_MODEL_TERMS:
            k1 = k1[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(k1 > 0, value / k1, np.nan)

    def summary(self, temperature: int | None = 0) -> dict[str, Any]:
        """Exact numbers replacing the ``(fig)`` entries of §1 of the D1 note.

        Returns a dictionary with, per model, the mean over iterations and walkers of
        every term of :math:`\\ell_m`, and, for each pair of models, the difference
        and the per-source difference.
        """
        k1 = self.term("num_resolved", temperature=temperature).astype(np.float64)
        k1_mean = float(np.mean(k1))

        out: dict[str, Any] = {
            "filename": self.filename,
            "niterations": self.niterations,
            "temperature": temperature,
            "nmodels": self.nmodels,
            "n_tot": self.n_tot,
            "k1_mean": k1_mean,
            "acceptance_fraction": float(
                np.mean(self.term("accepted", temperature=temperature))
            ),
            "model_occupancy": np.asarray(
                [
                    float(
                        np.mean(
                            self.term("model_current", temperature=temperature) == model
                        )
                    )
                    for model in range(self.nmodels)
                ]
            ),
            "rao_blackwell": self.rao_blackwell(temperature=temperature),
            "terms": {},
            "deltas": {},
        }

        names = ("intensity", "n1_expected", "stochastic", "resolved", "ell")
        for name in names:
            value = self.term(name, temperature=temperature)
            out["terms"][name] = np.asarray(
                [float(np.mean(value[:, model])) for model in range(self.nmodels)]
            )
        out["terms"]["model_log_prior"] = self.model_log_prior

        for model_from in range(self.nmodels):
            for model_to in range(self.nmodels):
                if model_from == model_to:
                    continue
                key = (model_from, model_to)
                delta = {}
                for name in names:
                    value = self.term(name, temperature=temperature)
                    difference = value[:, model_to] - value[:, model_from]
                    if name == "n1_expected":
                        # enters ell with a minus sign
                        difference = -difference
                    delta[name] = float(np.mean(difference))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        per_source = np.where(
                            k1 > 0, difference / np.where(k1 > 0, k1, 1.0), np.nan
                        )
                    delta[name + "_per_source"] = float(np.nanmean(per_source))
                out["deltas"][key] = delta

        return out

    def summary_text(self, temperature: int | None = 0) -> str:
        """Human-readable version of :meth:`summary`, in the §1.2 grouping."""
        summary = self.summary(temperature=temperature)
        lines: list[str] = []
        lines.append(f"D1 recording: {summary['filename']}")
        lines.append(
            f"iterations = {summary['niterations']}, temperature = {summary['temperature']}, "
            f"mean k_1 = {summary['k1_mean']:.1f}"
        )
        lines.append(
            "N_tot = " + ", ".join(f"{value:.6g}" for value in summary["n_tot"])
        )
        lines.append(
            "model occupancy = "
            + ", ".join(f"{value:.4f}" for value in summary["model_occupancy"])
            + f" | move acceptance fraction = {summary['acceptance_fraction']:.3e}"
        )
        lines.append(
            "Rao-Blackwell P(M|d) = "
            + ", ".join(f"{value:.6e}" for value in summary["rao_blackwell"])
        )
        lines.append("")
        header = f"{'term':<22}" + "".join(
            f"{'M_' + str(model):>18}" for model in range(summary["nmodels"])
        )
        lines.append(header)
        lines.append("-" * len(header))
        labels = {
            "intensity": "intensity",
            "n1_expected": "N_1 (enters as -N_1)",
            "stochastic": "stochastic",
            "resolved": "  of which sum ln p_pop",
            "ell": "ell_m (total)",
        }
        delta_labels = {
            "intensity": "intensity",
            "n1_expected": "-N_1",
            "stochastic": "stochastic",
            "resolved": "sum ln p_pop",
            "ell": "ell_m (total)",
        }
        for name, label in labels.items():
            values = summary["terms"][name]
            lines.append(
                f"{label:<22}" + "".join(f"{value:>18.6g}" for value in values)
            )
        lines.append("")
        for (model_from, model_to), delta in summary["deltas"].items():
            lines.append(f"M_{model_from} -> M_{model_to}   (d = M_{model_to} - M_{model_from})")
            for name, label in delta_labels.items():
                lines.append(
                    f"  d {label:<16}{delta[name]:>16.6g}"
                    f"   per source {delta[name + '_per_source']:>12.6g}"
                )
        return "\n".join(lines)

    def epsilon_scan_text(
        self,
        epsilons: Sequence[float | None],
        model_from: int = 0,
        model_to: int = 1,
        temperature: int | None = 0,
        reweight: bool = True,
    ) -> str:
        """Human-readable :meth:`epsilon_scan`, with its verdict spelled out."""
        scan = self.epsilon_scan(
            epsilons,
            model_from=model_from,
            model_to=model_to,
            temperature=temperature,
            reweight=reweight,
        )
        floor = scan["recorded_floor"]
        lines: list[str] = []
        lines.append(f"D14: support-floor sensitivity of B_{model_to}{model_from}")
        lines.append(f"recording {self.filename}, temperature {temperature}")
        if floor is None:
            lines.append(
                "The recording carries no support-floor attributes: either the floor "
                "was off, or the run predates the attributes. Nothing to scan."
            )
            return "\n".join(lines)

        epsilon, log_uniform = floor
        lines.append(
            f"recorded floor: epsilon = {epsilon:.3e}, ln|B| = {-log_uniform:.4f}, "
            f"floor level = {np.log(epsilon) + log_uniform:.4f} nats"
        )
        margin = self.stochastic_floor_margin(temperature=temperature)
        for model in range(self.nmodels):
            values = margin[:, model]
            values = values[np.isfinite(values)]
            if values.size == 0:
                lines.append(f"  M_{model}: no finite margin")
                continue
            lines.append(
                f"  M_{model}: floor margin median {np.median(values):10.4g} nats, "
                f"floor-dominated (< 1 nat) in {np.mean(values < 1.0):6.2%} of states"
            )
        lines.append("")
        header = (
            f"{'epsilon':>12}{'ln B':>14}{'mean d ell':>16}"
            f"{'finite frac':>14}{'ESS':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in scan["rows"]:
            label = "unfloored" if row["epsilon"] is None else f"{row['epsilon']:.3e}"
            lines.append(
                f"{label:>12}{row['ln_posterior_odds']:>14.4f}"
                f"{row['mean_delta_ell']:>16.6g}"
                f"{row['finite_fraction']:>14.4f}{row['effective_sample_size']:>12.1f}"
            )
        lines.append("")
        slope = scan["slope"]
        lines.append(f"d ln B / d ln epsilon = {slope:.4f}")
        if not np.isfinite(slope):
            lines.append(
                "  UNDETERMINED: fewer than two grid points gave a finite Bayes factor."
            )
        elif abs(slope) < 0.05:
            lines.append(
                "  INERT: the floor does not move the model comparison. The Bayes "
                "factor is the analysis's to quote."
            )
        else:
            lines.append(
                f"  FLOOR-DRIVEN: {abs(slope):.2f} density factor(s) of the comparison "
                "are floored, so ln B moves one-for-one with ln epsilon. The value is a "
                "restatement of epsilon and must be reported with this slope beside it, "
                "never as a bare Bayes factor. See section 10.6 of "
                "_dev/why_the_model_index_does_not_jump.md."
            )
        return "\n".join(lines)
