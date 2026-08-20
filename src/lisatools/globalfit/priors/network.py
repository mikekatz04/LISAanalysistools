import logging
import numpy as np
from typing import Sequence, Callable, Tuple, Any

from numpy.typing import ArrayLike
from ...utils.typing import NDArrayLike

from lisaflow.experiments.rvs.galaxy import GalaxyFlow
from lisaflow.experiments.rvs.galfor import GalForFlow
from .joint import JointPrior
from ...utils.utility import get_array_module

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

def _warn_on_duplicate_configs(class_name: str, config_files: Sequence[str]) -> None:
    """Warn when two model indices are handed the same flow configuration.

    Two identical configurations make the corresponding models indistinguishable, so
    their log-density ratio is identically zero. This is exactly defect B1, which went
    unnoticed for a full production run.
    """
    seen: dict[str, int] = {}
    for index, config_file in enumerate(config_files):
        key = str(config_file)
        if key in seen:
            logger.warning(
                "%s: model %d and model %d were given the same configuration file %s. "
                "Those models share one flow, so their log-density ratio is "
                "identically zero and they cannot be told apart.",
                class_name,
                seen[key],
                index,
                key,
            )
        else:
            seen[key] = index


#! ``NormalizingFlowPrior``, a generic flow prior applying the min-max normalisation
#! Jacobian itself, was removed here (defect B8): it was unused -- ``FullGalaxyPrior``
#! and ``GalForNFPrior`` wrap the lisaflow classes, which own that Jacobian -- and its
#! ``logpdf`` referenced a ``_norm_log_jacobian`` that only existed inside a
#! commented-out block, so instantiating it raised ``AttributeError``.


class SupportFloor:
    """Regularise a flow's tails so a model comparison is never :math:`-\\infty`.

    This is method 2 of R5. A flow in a logit-transformed coordinate has *no* support
    outside the ``param_min``/``param_max`` box recorded in its checkpoint, and those
    bounds are nothing but the extent of the training samples plus a 0.1% buffer. When
    two models' boxes differ, the log-density ratio at a state one model cannot reach is
    :math:`\\pm\\infty` -- defect B9 -- which is immune to tempering, to reducing
    :math:`k_1` and to any improvement of the fits.

    The floor replaces the flow density by the mixture

    .. math::
        p_\\epsilon(\\vec x \\mid M) = (1 - \\epsilon)\\, p_{\\rm flow}(\\vec x \\mid M)
                                     + \\epsilon\\, \\mathcal U(\\vec x \\mid B),

    with :math:`B` a broad box covering every competing model's own box -- in practice
    their union, the smallest choice that works. Two properties make this a modelling
    choice rather than a fudge:

    * **It stays a proper density.** Each component integrates to one over :math:`B`,
      and every model's box lies inside :math:`B`, so :math:`\\int_B p_\\epsilon = 1`
      exactly. No normalisation is broken and no probability mass is invented.
    * **It changes nothing where the comparison was already meaningful.** With the
      galfor boxes of ``medium_int``/``medium_weak_int`` the union volume is
      :math:`\\ln|B| = 4.07`, so at :math:`\\epsilon = 10^{-6}` the broad component
      contributes :math:`-17.9` nats where the flow reports :math:`+15`. The mixture is
      the flow value to fourteen decimal places, except exactly where the old answer
      was :math:`-\\infty`.

    The price is explicit. The mixture cannot report a log-density below
    :math:`\\ln\\epsilon - \\ln|B|`, so the evidence a single state may carry is capped:
    the floor *floors* the tail rather than modelling it. :math:`\\epsilon` is therefore
    a modelling parameter, not a numerical tolerance, and it changes the target
    distribution -- which is why it is off by default and must be switched on explicitly
    in the settings. The honest fix remains training both models on a common support
    (method 1 of R5); this one buys measurability, not mixing.

    ``rvs`` draws from the same mixture. That matters because these priors are also the
    reversible-jump birth proposal, where reporting one density and sampling another
    would silently corrupt the acceptance ratio.

    See ``_dev/why_the_model_index_does_not_jump.md``, section 9.
    """

    #: mixture weight of the broad component; ``None`` leaves the flow untouched
    _floor_epsilon: float | None = None

    def set_support_floor(
        self, epsilon: float, box_min: ArrayLike, box_max: ArrayLike
    ) -> None:
        """Switch the floor on, with a broad box in the flow's own coordinates.

        Args:
            epsilon: Mixture weight of the broad component, in ``(0, 1)``.
            box_min: Lower corner of the broad box. Must cover this flow's own box.
            box_max: Upper corner of the broad box.
        """
        if not 0.0 < epsilon < 1.0:
            raise ValueError(
                f"The support floor epsilon must lie in (0, 1), got {epsilon}."
            )

        box_min = _to_numpy(box_min).astype(np.float64)
        box_max = _to_numpy(box_max).astype(np.float64)
        own_min = _to_numpy(self.param_min).astype(np.float64)
        own_max = _to_numpy(self.param_max).astype(np.float64)
        if np.any(box_min > own_min) or np.any(box_max < own_max):
            raise ValueError(
                "The broad box of a support floor must cover the flow's own training "
                f"box, otherwise the mixture is improper. Got broad [{box_min}, "
                f"{box_max}] against own [{own_min}, {own_max}]."
            )

        self._floor_epsilon = float(epsilon)
        self._floor_min_host = box_min
        self._floor_max_host = box_max
        self._floor_box_cache: dict[str, tuple] = {}
        #: log-density of the broad component, constant inside the broad box
        self._floor_log_uniform = float(-np.sum(np.log(box_max - box_min)))
        logger.info(
            "%s: support floor ON, epsilon = %.3e, ln|B| = %.4f, so no state can "
            "report less than %.2f nats and the per-state model separation is capped.",
            type(self).__name__,
            self._floor_epsilon,
            -self._floor_log_uniform,
            np.log(self._floor_epsilon) + self._floor_log_uniform,
        )

    @property
    def support_floor_epsilon(self) -> float | None:
        """The mixture weight in use, or ``None`` when the floor is off."""
        return self._floor_epsilon

    @property
    def support_floor_log_uniform(self) -> float | None:
        """:math:`-\\ln|B|`, the log-density of the broad component, or ``None``.

        Together with :attr:`support_floor_epsilon` this is everything a consumer needs
        to undo the floor exactly (:func:`unfloor_log_density`), so a recording that
        stores the pair is self-describing and the :math:`\\epsilon`-sensitivity of a
        finished run can be measured offline. See D14.
        """
        return self._floor_log_uniform if self._floor_epsilon is not None else None

    @property
    def support_floor_level(self) -> float | None:
        """:math:`\\ln\\epsilon - \\ln|B|`, the smallest log-density the mixture can report."""
        if self._floor_epsilon is None:
            return None
        return float(np.log(self._floor_epsilon) + self._floor_log_uniform)

    @property
    def support_floor_box(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The broad box :math:`B`, as ``(min, max)``, or ``None`` when the floor is off.

        Two models share a support only if they carry the same floor **over the same
        box**; :math:`\\epsilon` alone says nothing, since :meth:`set_support_floor` is
        public and can be handed a different box per model.
        :func:`apply_common_support_floor` gives every model the union box, which is the
        case this accessor is meant to let a consumer verify rather than assume.
        """
        if self._floor_epsilon is None:
            return None
        return self._floor_min_host.copy(), self._floor_max_host.copy()

    def unfloor_logpdf(self, log_floored: NDArrayLike) -> NDArrayLike:
        """Recover :math:`\\ln p_{\\rm flow}` from a value this object returned.

        The inverse of :meth:`_apply_support_floor`, in the coordinate the floor acts in.
        A no-op when the floor is off. See :func:`unfloor_log_density` for the caveat
        that makes this lossy in the tail.
        """
        if self._floor_epsilon is None:
            return log_floored
        return unfloor_log_density(
            log_floored, self._floor_epsilon, self._floor_log_uniform, xp=self.xp
        )

    def _floor_box(self) -> tuple[NDArrayLike, NDArrayLike]:
        """The broad box, in whichever array module ``self.xp`` currently is."""
        key = self.xp.__name__
        if key not in self._floor_box_cache:
            self._floor_box_cache[key] = (
                self.xp.asarray(self._floor_min_host),
                self.xp.asarray(self._floor_max_host),
            )
        return self._floor_box_cache[key]

    def _apply_support_floor(
        self, log_flow: NDArrayLike, x: NDArrayLike
    ) -> NDArrayLike:
        """Mix the broad component into a flow log-density.

        Args:
            log_flow: ``ln p_flow(x)``, which is ``-inf`` outside the flow's own box.
            x: The points, in the coordinates the box is expressed in.
        """
        if self._floor_epsilon is None:
            return log_flow

        floor_min, floor_max = self._floor_box()
        inside = self.xp.all((x >= floor_min) & (x <= floor_max), axis=-1)
        log_broad = self.xp.where(
            inside, np.log(self._floor_epsilon) + self._floor_log_uniform, -np.inf
        )
        # logaddexp leaves -inf + -inf = -inf: a point outside the *broad* box still has
        # zero density. The floor widens the support, it does not abolish the notion.
        return self.xp.logaddexp(
            np.log1p(-self._floor_epsilon) + log_flow, log_broad
        )

    def _draw_support_floor(self, x: NDArrayLike) -> NDArrayLike:
        """Replace an :math:`\\epsilon` fraction of draws by uniforms over the box."""
        if self._floor_epsilon is None:
            return x

        floor_min, floor_max = self._floor_box()
        from_broad = self.xp.random.random(x.shape[:-1]) < self._floor_epsilon
        count = int(from_broad.sum())
        if count == 0:
            return x
        draws = floor_min + self.xp.random.random((count, x.shape[-1])) * (
            floor_max - floor_min
        )
        # ``_tensor_to_xp`` can hand back a zero-copy view of torch memory, so copy
        x = x.copy()
        x[from_broad] = draws
        return x


def _to_numpy(values: ArrayLike) -> np.ndarray:
    """A numpy view of a numpy array, a CuPy array or a torch buffer alike."""
    if torch is not None and isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    getter = getattr(values, "get", None)  # CuPy
    return np.asarray(getter() if callable(getter) else values)


def support_floor_margin(
    log_floored: NDArrayLike, epsilon: float, log_uniform: float, xp=np
) -> NDArrayLike:
    """How far a floored log-density sits above the floor, in nats.

    :math:`\\ln p_\\epsilon - (\\ln\\epsilon - \\ln|B|)`, the single number that says
    which regime a state is in. A margin of many nats means the flow is speaking and the
    floor is decoration; a margin near zero means the value is a statement about
    :math:`\\epsilon` and not about the population. D14 reports its distribution.
    """
    return log_floored - (np.log(epsilon) + log_uniform)


def unfloor_log_density(
    log_floored: NDArrayLike, epsilon: float, log_uniform: float, xp=np
) -> NDArrayLike:
    """Undo the support floor: :math:`\\ln p_{\\rm flow}` from :math:`\\ln p_\\epsilon`.

    The mixture is invertible pointwise,

    .. math::
        p_{\\rm flow} = \\frac{p_\\epsilon - \\epsilon/|B|}{1 - \\epsilon},

    so a floored run can be un-floored exactly after the fact, with no rerun and no
    further flow evaluations, *provided the stored quantity is a single evaluation*.
    A recorded sum over :math:`k_1` sources is a sum of logs and is not invertible; the
    galfor term is one :math:`\\Sigma` per state and is.

    The inversion is lossy in exactly the place the floor matters. The subtraction
    cancels ``margin`` nats of significance (:func:`support_floor_margin`), so a state
    whose margin is below about :math:`10^{-8}` returns :math:`-\\infty`: the floor has
    destroyed what the flow said there, which is the honest report, not a failure of the
    arithmetic. That is the same statement as "the floored value carries no information
    about the population at this state".

    Args:
        log_floored: :math:`\\ln p_\\epsilon`, as the prior returned it.
        epsilon: The mixture weight that was in use.
        log_uniform: :math:`-\\ln|B|` of the broad box that was in use.
        xp: Array module, so this works on the host and on the device alike.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must lie in (0, 1), got {epsilon}.")

    log_floored = xp.asarray(log_floored, dtype=xp.float64)
    log_broad = float(np.log(epsilon) + log_uniform)

    # ``margin <= 0`` covers three cases at once and all three answer -inf: a state
    # outside the broad box (log_floored = -inf), a state the floor fully dominates,
    # and float64 noise pushing a floor-dominated value a hair below the floor.
    margin = log_floored - log_broad
    safe = xp.where(margin > 0.0, margin, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recovered = log_floored + xp.log(-xp.expm1(-safe)) - np.log1p(-epsilon)
    return xp.where(margin > 0.0, recovered, -xp.inf)


def refloor_log_density(
    log_flow: NDArrayLike, epsilon: float, log_uniform: float, xp=np
) -> NDArrayLike:
    """Apply a floor to a raw log-density, outside any prior object.

    The counterpart of :func:`unfloor_log_density`, so that a stored recording can be
    re-evaluated at an :math:`\\epsilon` it was not run at:
    ``refloor(unfloor(x, eps), eps')``. This is what makes the D14 sensitivity scan free.

    Unlike :meth:`SupportFloor._apply_support_floor` this has no box to test against, so
    it assumes every point lies inside :math:`B`. That holds for recorded values, which
    were produced by a prior that had already applied the box.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must lie in (0, 1), got {epsilon}.")
    log_flow = xp.asarray(log_flow, dtype=xp.float64)
    log_broad = float(np.log(epsilon) + log_uniform)
    return xp.logaddexp(np.log1p(-epsilon) + log_flow, log_broad)


def apply_common_support_floor(models: Sequence[SupportFloor], epsilon: float | None) -> None:
    """Give every model of a hyper prior the union of all their boxes as broad box.

    The union is the smallest box that covers all the competing models, so it is the
    weakest regularisation that removes every :math:`\\pm\\infty` from the model
    comparison. ``epsilon = None`` leaves the flows exactly as they are.
    """
    if epsilon is None:
        return
    mins = np.stack([_to_numpy(model.param_min) for model in models])
    maxs = np.stack([_to_numpy(model.param_max) for model in models])
    union_min, union_max = mins.min(axis=0), maxs.max(axis=0)
    logger.info(
        "Applying a common support floor to %d models, broad box [%s, %s].",
        len(models),
        np.array2string(union_min, precision=4),
        np.array2string(union_max, precision=4),
    )
    for model in models:
        model.set_support_floor(epsilon, union_min, union_max)


class FullGalaxyPrior(SupportFloor, JointPrior, GalaxyFlow):
    """
    5D Normalizing Flow prior for Galactic Binaries.

    Network Fit Basis (NFB): [logA, logf0_Hz, -sign(fdot)*log|C*fdot|, ra, sin_dec]
    Sampling Basis (SB):     [logA, f0_mHz, fdot, ra, sin_dec]
    Physical Basis (PB):     [A, f0_Hz, fdot, ra, dec]

    The optional support floor of :class:`SupportFloor` acts in the NFB, where the
    training box lives; the Jacobian to the SB is applied to the mixture afterwards.
    """
    
    def __init__(
        self, 
        config_file: str, 
        use_cupy: bool = False,
        return_gpu: bool = False
    ):

        GalaxyFlow.__init__(self, config_file)

        # provenance: which checkpoint of which fit is actually in use (defect B4).
        # The epoch is *not* selected by any criterion here -- see diagnostic D5.
        self.config_file = str(config_file)
        self.checkpoint_path = (
            self.config["saving"]["save_root"] + self.config["training"]["checkpoints"]
        )
        self.load_fit(self.checkpoint_path)
        logger.info(
            "FullGalaxyPrior: loaded %s (config %s)",
            self.checkpoint_path,
            self.config_file,
        )

        param_names = ("logA", "f0_mHz", "fdot", "ra", "sin_dec")
        param_names_phys = ("A", "f0", "fdot", "ra", "dec")
        latex_labels = (r"\log \mathcal{A}", r"f_{0, {\rm mHz}}", r"\dot{f}", r"\alpha", r"\sin \delta")

        JointPrior.__init__(
            self,
            names=param_names,
            names_phys=param_names_phys,
            latex_labels=latex_labels,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            inverse_transform_nd=self._sb_to_pb  # sampling to physical
        )

    def _sb_to_pb(self, logA, f0_mHz, fdot, ra, sin_dec, **kwargs):
        """Maps: Sampling Basis (SB) -> Physical Basis (PB)"""
        xp = get_array_module(logA)
        A = xp.exp(logA)
        f0 = f0_mHz * 1e-3
        dec = xp.arcsin(sin_dec)
        return A, f0, fdot, ra, dec

    def _fdot_rescale_factor(self, f0_max = 0.029) -> float:
        return 3200 * self.xp.cbrt(2) * self.xp.sqrt(5/3) / f0_max**(9/2)
    
    def _nfb_to_sb(self, nfb_samples):
        """Maps: Network Fit Basis (NFB) -> Sampling Basis (SB)"""
        logA = nfb_samples[..., 0]
        logf0 = nfb_samples[..., 1]
        mlogfdot = nfb_samples[..., 2]
        ra = nfb_samples[..., 3]
        sin_dec = nfb_samples[..., 4]

        f0_mHz = self.xp.exp(logf0) * 1e3
        
        # y = -sign(fdot) * log|C*fdot|
        # Since log|fdot| is negative (fdot ~ 1e-16), |y| = -log|C*fdot|.
        # Thus |fdot| = exp(-|y|)/C and sign of y matches sign of fdot.
        rescale_factor = self._fdot_rescale_factor()
        fdot = self.xp.sign(mlogfdot) * self.xp.exp(-self.xp.abs(mlogfdot)) / rescale_factor

        return self.xp.stack([logA, f0_mHz, fdot, ra, sin_dec], axis=-1)
    
    def _sb_to_nfb(self, sb_samples):
        """Maps: Sampling Basis (SB) -> Network Fit Basis (NFB)"""
        logA = sb_samples[..., 0]
        f0_mHz = sb_samples[..., 1]
        fdot = sb_samples[..., 2]
        ra = sb_samples[..., 3]
        sin_dec = sb_samples[..., 4]

        # f0_mHz -> log(f0_Hz)
        logf0 = self.xp.log(f0_mHz * 1e-3)
        
        # fdot -> -sign(fdot) * log|C*fdot|
        # Safe guard against absolute 0 to prevent log(0) = -inf causing NaN gradients
        safe_fdot = self.xp.where(fdot != 0, fdot, 1e-30)
        rescale_factor = self._fdot_rescale_factor()
        mlogfdot = -self.xp.sign(safe_fdot) * self.xp.log(self.xp.abs(safe_fdot * rescale_factor)) 
        return self.xp.stack([logA, logf0, mlogfdot, ra, sin_dec], axis=-1)

    def _tensor_to_xp(self, tensor: torch.Tensor):
        """Zero-copy transfer from PyTorch Tensor to CuPy/NumPy array."""
        if self.use_cupy and tensor.is_cuda:
            from torch.utils.dlpack import to_dlpack
            return self.xp.from_dlpack(to_dlpack(tensor))
        return self.xp.asarray(tensor.detach().cpu().numpy())
    
    def rvs(self, size: int | Tuple[int, ...] = (1,), **kwargs: Any) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
            
        num_samples = int(np.prod(size))
        
        nfb_samples = self._tensor_to_xp(self.sample(num_samples=num_samples))
        nfb_samples = nfb_samples.reshape(*size, self.num_vars)
        # the density this class reports is the density it samples from (R5 method 2)
        nfb_samples = self._draw_support_floor(nfb_samples)

        sb_samples = self._nfb_to_sb(nfb_samples)

        return self._to_device(sb_samples)

    def logpdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        x_sb = self.xp.asarray(x)
        original_shape = x_sb.shape

        x_nfb = self._sb_to_nfb(x_sb)
        x_nfb_flat = x_nfb.reshape(-1, self.num_vars)

        logpdf_nfb_flat = self._tensor_to_xp(self.log_prob(x_nfb_flat))
        logpdf_nfb_flat = self._apply_support_floor(logpdf_nfb_flat, x_nfb_flat)
        logpdf_nfb = logpdf_nfb_flat.reshape(original_shape[:-1])
        
        # d(NFB_f0) / d(SB_f0) = d(ln(f0_mHz*1e-3)) / d(f0_mHz) = 1 / f0_mHz
        # d(NFB_fdot) / d(SB_fdot) = d(-sign(fdot)*ln|C*fdot|) / d(fdot) = 1 / |fdot|
        # Log Determinant = -ln(f0_mHz) - ln(|fdot|)
        f0_mHz = x_sb[..., 1]
        fdot = x_sb[..., 2]
        
        safe_fdot = self.xp.where(fdot != 0, self.xp.abs(fdot), 1e-30)
        
        log_jacobian = -self.xp.log(f0_mHz) - self.xp.log(safe_fdot)
        
        # Final probability in the Sampling Basis
        logpdf_sb = logpdf_nfb + log_jacobian
        
        return self._to_device(logpdf_sb)


class HyperGalaxyPrior(JointPrior):
    """
    Conditional Normalizing Flow Prior for Model Selection.
    
    Evaluates p(theta | M_i) by routing the parameter batch `x` to the 
    corresponding Normalizing Flow network based on `model_index`.
    """

    def __init__(
        self,
        config_files: Sequence[str],  # One config file per model
        use_cupy: bool = False,
        return_gpu: bool = False,
        support_floor: float | None = None,
    ):
        self.n_models = len(config_files)
        _warn_on_duplicate_configs(type(self).__name__, config_files)

        # Initialize the list of underlying FullGalaxyPriors
        self.models = [
            FullGalaxyPrior(
                config_file=cfg,
                use_cupy=use_cupy,
                return_gpu=True, # Keep strictly on GPU internally
            ) for cfg in config_files
        ]

        apply_common_support_floor(self.models, support_floor)

        names = self.models[0].names
        names_phys = self.models[0].names_phys
        latex_labels = self.models[0].latex_label

        super().__init__(
            names=names,
            names_phys=names_phys,
            latex_labels=latex_labels,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            inverse_transform_nd=self.models[0].inverse_transform_nd
        )

    def logpdf(self, x: ArrayLike, model_index: ArrayLike, **kwargs: Any) -> NDArrayLike:
        """
        Evaluate p(x | M_i).
        
        Args:
            x: GB parameters. Shape (nwalkers, nleaves, ndim).
            model_index: Model indicators. Shape (nwalkers,) or (nwalkers, 1).
        """
        x_sb = self.xp.asarray(x)
        mod_idx = self.xp.asarray(model_index).astype(self.xp.int32)
        
        if mod_idx.ndim < x_sb.ndim - 1:
            mod_idx = self.xp.expand_dims(mod_idx, tuple(range(mod_idx.ndim, x_sb.ndim - 1)))
        
        if mod_idx.shape[:-1] != x_sb.shape[:-1]:
            mod_idx = self.xp.broadcast_to(mod_idx, x_sb.shape[:-1])
        
        out = self.xp.full(x_sb.shape[:-1], -np.inf, dtype=self.xp.float64)

        for i in range(self.n_models):
            mask = (mod_idx == i).squeeze()
            
            if not mask.any():
                continue
            
            x_model_subset = x_sb[mask]
            logp_subset = self.models[i].logpdf(x_model_subset)
            out[mask] = logp_subset

        return self._to_device(out)

    def rvs(self, size: int | Tuple[int, ...] = (1,), model_index: ArrayLike = 0, **kwargs: Any) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        
        mod_idx = self.xp.asarray(model_index).astype(self.xp.int32)
        
        if mod_idx.shape != size:
            mod_idx = self.xp.broadcast_to(mod_idx, size)
            
        out_samples = self.xp.empty((*size, self.num_vars), dtype=self.xp.float64)
        
        for i in range(self.n_models):
            mask = (mod_idx == i)
            n_samples_model = int(mask.sum())
            if n_samples_model == 0:
                continue
                
            out_samples[mask] = self.models[i].rvs(size=n_samples_model)

        return self._to_device(out_samples)
    
    
# TODO change to GalForFlow

class GalForNFPrior(SupportFloor, JointPrior, GalForFlow):
    """
    5D Normalizing Flow prior for the Galactic Foreground.

    Network Fit Basis (NFB) == Sampling Basis (SB):
        [log10_Amp, alpha, log10_f1, log10_fknee, log10_f2]

    Physical Basis (PB):
        [Amp, alpha, f1, fknee, f2]

    This is the density defect B9 lives in: its training box is the extent of a
    posterior chain, so two populations' boxes need not overlap where the sampler
    goes. See :class:`SupportFloor`.
    """
    
    def __init__(
        self, 
        config_file: str, 
        use_cupy: bool = False,
        return_gpu: bool = False
    ):
        GalForFlow.__init__(self, config_file)

        self.config_file = str(config_file)
        self.checkpoint_path = (
            self.config["saving"]["save_root"] + self.config["training"]["checkpoints"]
        )
        self.load_fit(self.checkpoint_path)
        logger.info(
            "GalForNFPrior: loaded %s (config %s)",
            self.checkpoint_path,
            self.config_file,
        )

        param_names = ("log10_Amp", "alpha", "log10_f1", "log10_fknee", "log10_f2")
        param_names_phys = ("Amp", "alpha", "f1", "fknee", "f2")
        latex_labels = (
            r"\log_{10} A_{\rm gal}", 
            r"\alpha_{\rm gal}", 
            r"\log_{10} f_1", 
            r"\log_{10} f_{\rm knee}", 
            r"\log_{10} f_2"
        )

        JointPrior.__init__(
            self,
            names=param_names,
            names_phys=param_names_phys,
            latex_labels=latex_labels,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            inverse_transform_nd=self._sb_to_pb
        )

    def _sb_to_pb(self, log10_Amp, alpha, log10_f1, log10_fknee, log10_f2, **kwargs):
        """Maps: Sampling Basis (SB) -> Physical Basis (PB)"""
        Amp = 10.0 ** log10_Amp
        f1 = 10.0 ** log10_f1
        fknee = 10.0 ** log10_fknee
        f2 = 10.0 ** log10_f2
        return Amp, alpha, f1, fknee, f2

    def _tensor_to_xp(self, tensor: torch.Tensor):
        """Zero-copy transfer from PyTorch Tensor to CuPy/NumPy array."""
        if self.use_cupy and tensor.is_cuda:
            from torch.utils.dlpack import to_dlpack
            return self.xp.from_dlpack(to_dlpack(tensor))
        return self.xp.asarray(tensor.detach().cpu().numpy())
    
    def rvs(self, size: int | Tuple[int, ...] = (1,), **kwargs: Any) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
            
        num_samples = int(np.prod(size))
        
        # SB == NFB, so we sample directly and reshape
        sb_samples_torch = self.sample(num_samples=num_samples)
        sb_samples = self._tensor_to_xp(sb_samples_torch).reshape(*size, self.num_vars)
        # the density this class reports is the density it samples from (R5 method 2)
        sb_samples = self._draw_support_floor(sb_samples)

        return self._to_device(sb_samples)

    def logpdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        x_sb = self.xp.asarray(x)
        original_shape = x_sb.shape

        x_sb_flat = x_sb.reshape(-1, self.num_vars)

        # no jacobians needed because sb=nfb
        logpdf_sb_flat = self._tensor_to_xp(self.log_prob(x_sb_flat))
        logpdf_sb_flat = self._apply_support_floor(logpdf_sb_flat, x_sb_flat)
        logpdf_sb = logpdf_sb_flat.reshape(original_shape[:-1])
        
        return self._to_device(logpdf_sb)
    

class HyperGalForPrior(JointPrior):
    """
    Conditional Normalizing Flow Prior for Galactic Foreground Model Selection.
    """

    def __init__(
        self,
        config_files: Sequence[str],
        use_cupy: bool = False,
        return_gpu: bool = False,
        support_floor: float | None = None,
    ):
        self.n_models = len(config_files)
        _warn_on_duplicate_configs(type(self).__name__, config_files)

        self.models = [
            GalForNFPrior(
                config_file=cfg,
                use_cupy=use_cupy,
                return_gpu=True, # who is going to sample networks on cpu?
            ) for cfg in config_files
        ]

        # R5 method 2, off unless asked for. This is the density B9 lives in: without a
        # floor, ln p(Sigma | M_1) is -inf at every state the sampler reaches.
        apply_common_support_floor(self.models, support_floor)

        names = self.models[0].names
        names_phys = self.models[0].names_phys
        latex_labels = self.models[0].latex_label

        super().__init__(
            names=names,
            names_phys=names_phys,
            latex_labels=latex_labels,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            inverse_transform_nd=self.models[0].inverse_transform_nd
        )

    def logpdf(self, x: ArrayLike, model_index: ArrayLike, **kwargs: Any) -> NDArrayLike:
        x_sb = self.xp.asarray(x)
        mod_idx = self.xp.asarray(model_index).astype(self.xp.int32)
        
        if mod_idx.ndim < x_sb.ndim - 1:
            mod_idx = self.xp.expand_dims(mod_idx, tuple(range(mod_idx.ndim, x_sb.ndim - 1)))
        
        mod_idx = self.xp.broadcast_to(mod_idx, x_sb.shape[:-1])
        out = self.xp.full(x_sb.shape[:-1], -np.inf, dtype=self.xp.float64)

        for i in range(self.n_models):
            mask = (mod_idx == i)
            if not mask.any():
                continue
            out[mask] = self.models[i].logpdf(x_sb[mask])

        return self._to_device(out)

    def rvs(self, size: int | Tuple[int, ...] = (1,), model_index: ArrayLike = 0, **kwargs: Any) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)
        
        mod_idx = self.xp.asarray(model_index).astype(self.xp.int32)
        if mod_idx.shape != size:
            mod_idx = self.xp.broadcast_to(mod_idx, size)
            
        out_samples = self.xp.empty((*size, self.num_vars), dtype=self.xp.float64)
        
        for i in range(self.n_models):
            mask = (mod_idx == i)
            n_samples_model = int(mask.sum())
            if n_samples_model == 0:
                continue
            out_samples[mask] = self.models[i].rvs(size=n_samples_model)

        return self._to_device(out_samples)