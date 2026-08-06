"""MCMC move that updates the LISA noise model (psd / galfor / sgwb branches).

Noise-move split (2026-07): ONE parameterized move class. Each instance
samples ``sampled_branches`` under its OWN temperature ladder while the
likelihood always evaluates the full noise model, with non-sampled noise
branches frozen at their cold-chain rows. ``sampled_branches=None``
reproduces the historical joint psd+galfor+sgwb move exactly.

stft_tof merge (2026-06): hybrid likelihood routing. The move keeps the
dev-side domain-agnostic structure (per-walker sensitivity matrices installed
on the AnalysisContainers, scored via ``acs.likelihood()``), and adds the
stft_tof C++ XYZ-sensitivity kernel (``psd_likelihood_wrap`` shared-memory
sums) as a fast path when the data basis is FD or STFT and the ACA holds a
single shard. The WDM basis falls back to the ACA route until the WDM
counterpart kernels land in domains.cu.
"""

import time
import warnings
from copy import deepcopy
import logging

import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp
from eryn.model import Model
from eryn.moves import RedBlueMove, StretchMove
from eryn.state import BranchSupplemental
from eryn.state import State as eryn_State
from eryn.utils.transform import TransformContainer

from tqdm import tqdm

from ...analysiscontainer import AnalysisContainerArray
from ...domaincomputation import DomainComputationGroupArray
from ...domains import FDSettings, STFTSettings
from ...sensitivity import XYZSensitivityBackend
from ...utils.utility import asnumpy
from ..state import GFState
from .globalfitmove import GlobalFitMove

logger = logging.getLogger(__name__)

DEBUG_MODE = False


class PSDMove(GlobalFitMove, StretchMove):
    """
    Noise-model move: samples ``sampled_branches`` of the noise model
    (``psd``, ``galfor``, ``sgwb``) while the likelihood always evaluates the
    FULL noise model — non-sampled noise branches contribute their cold-chain
    coordinates (per walker), the same "temper your own ensemble against the
    cold-chain rest" structure the add/remove source branches use. With the
    default ``sampled_branches=None`` every noise branch present in the state
    is sampled jointly (the historical single-joint-move behavior).

    Args:
        acs: AnalysisContainerArray containing the data and sensitivity information.
        priors: dictionary of priors for the parameters.
        *args: additional arguments for the Move class.
        sampled_branches: which noise branches THIS move samples (subset of
            ``["psd", "galfor", "sgwb"]``). ``None`` samples all present
            branches jointly. Each move instance carries its own
            ``temperature_control`` ladder for its sampled block.
        num_repeats: number of times to repeat the move before returning.
        max_logl_mode: if True, will keep running the move until the maximum log likelihood does not change for a certain number of checks. This is useful for finding the maximum likelihood point.
        psd_kwargs: additional keyword arguments for the psd_log_like function.
        sensitivity_backend: instance of XYZSensitivityBackend to use for computing the likelihood.
        psd_transform_fn: TransformContainer for transforming the PSD parameters.
        galfor_transform_fn: TransformContainer for transforming the galactic
            foreground parameters. NOTE the long-standing order convention is
            preserved as-is: the galfor prior/sampling basis is labelled
            ``[amp, knee, alpha, Slope1, Slope2]`` while the kernel unpack
            reads ``[Amp, alpha, f_1, kn, f_2]`` (see
            :meth:`prepare_likelihood_inputs`; history in the commented
            reorder block in ``run.py``). Do not reorder here.
        sgwb_transform_fn: TransformContainer hook for the SGWB parameters
            (plumbed for symmetry; SGWB coordinates are currently consumed
            untransformed on the ACA route, matching historical behavior).
        permute_every: number of repeats after which to permute the walkers during a temperature swap. This helps with the mixing of the chains.
        tolerance: minimum allowed distance between spline knot positions in the sensitivity model.
        **kwargs: additional keyword arguments for the Move class.
    """

    # canonical noise-model branch order (psd is mandatory in the model;
    # galfor/sgwb optional)
    NOISE_BRANCHES = ("psd", "galfor", "sgwb")

    # per-iteration acceptance deltas; allocated by propose() at the module
    # ladder shape, so run_move() skips the tally when called on its own.
    _tally_in_model_proposed = None
    _tally_in_model_accepted = None
    _tally_swaps_proposed = None
    _tally_swaps_accepted = None

    def __init__(
        self,
        acs: AnalysisContainerArray,
        priors,
        *args,
        sampled_branches: list = None,
        num_repeats: int = 1,
        max_logl_mode: bool = False,
        psd_kwargs: dict = {},
        sensitivity_backend: XYZSensitivityBackend = None,
        psd_transform_fn: TransformContainer = None,
        galfor_transform_fn: TransformContainer = None,
        sgwb_transform_fn: TransformContainer = None,
        permute_every: int = 20,
        tolerance: float = 0.0,
        dcga: DomainComputationGroupArray = None,
        run_async: bool = False,
        run_threaded: bool = False,
        **kwargs,
    ):

        GlobalFitMove.__init__(self, *args, **kwargs)
        StretchMove.__init__(self, *args, **kwargs)
        # ONE move structure (2026-07 multi-GPU pass): pass ``dcga=`` to run
        # the kernel fast path through per-device replicas — how many GPUs
        # the move uses is set by the DCGA it is built with. Without a DCGA
        # the move runs its single-device path. Same object, same propose.
        self._dcga = dcga
        self._run_async = run_async
        self._run_threaded = run_threaded
        if dcga is not None:
            if acs is None:
                acs = dcga.acs
            if sensitivity_backend is None:
                sensitivity_backend = (
                    dcga.computation_groups[0].sensitivity_backend
                )
        self.acs = acs
        self.psd_kwargs = psd_kwargs
        self.priors = priors
        self.num_repeats = num_repeats
        self.max_logl_mode = max_logl_mode
        self.starting_now = True

        if sampled_branches is not None:
            unknown = [b for b in sampled_branches if b not in self.NOISE_BRANCHES]
            if unknown:
                raise ValueError(
                    f"Unknown noise branch(es) {unknown}; sampled_branches must be a "
                    f"subset of {list(self.NOISE_BRANCHES)}."
                )
            # canonical order regardless of input order
            sampled_branches = [b for b in self.NOISE_BRANCHES if b in sampled_branches]
        self.sampled_branches = sampled_branches
        # cold-row snapshot of the non-sampled noise branches, refreshed at
        # every propose(); empty when this move samples the whole model
        self._fixed_noise_coords = {}

        self.sensitivity_backend = sensitivity_backend
        self.psd_transform_fn = psd_transform_fn
        self.galfor_transform_fn = galfor_transform_fn
        self.sgwb_transform_fn = sgwb_transform_fn

        self.permute_every = permute_every
        self.tolerance = tolerance

        # Debug escape hatch (see the note in the legacy MultiGPUPSDMove):
        # when True, psd_log_like bypasses the DCGA routing entirely.
        self._force_parent_path = False

    @property
    def dcga(self) -> DomainComputationGroupArray:
        return self._dcga

    @property
    def run_async(self) -> bool:
        return self._run_async

    @property
    def run_threaded(self) -> bool:
        return self._run_threaded

    # ------------------------------------------------------------------
    # sampled-vs-fixed noise-branch resolution
    # ------------------------------------------------------------------

    def _noise_model_branches(self, state) -> list:
        """Noise branches present in ``state`` (the full noise MODEL)."""
        return [key for key in self.NOISE_BRANCHES if key in state.branches_coords]

    def _resolve_sampled(self, state) -> list:
        """Branches THIS move samples; validated against the state's model."""
        model = self._noise_model_branches(state)
        if self.sampled_branches is None:
            return model
        missing = [b for b in self.sampled_branches if b not in model]
        if missing:
            raise ValueError(
                f"PSDMove configured to sample {self.sampled_branches} but branch(es) "
                f"{missing} are not in the state (noise model: {model})."
            )
        return list(self.sampled_branches)

    def _merged_noise_rows(self, coords, logp_keep, walker_inds_keep) -> dict:
        """Per-row full-noise-model coordinates (sampling basis).

        Sampled branches come from the proposal ``coords`` dict (rows
        surviving the prior cut); fixed branches come from the cold-row
        snapshot taken at ``propose`` entry, indexed by each row's physical
        walker. This is the single seam both likelihood routes consume.
        """
        merged = {}
        for key in self.NOISE_BRANCHES:
            if key in coords:
                merged[key] = coords[key][logp_keep][:, 0]
            elif key in self._fixed_noise_coords:
                merged[key] = self._fixed_noise_coords[key][walker_inds_keep]
        return merged

    def _cold_noise_log_prior(self, state) -> np.ndarray:
        """Sum of ALL noise-branch priors at the cold-row coordinates.

        Written into the engine ``log_prior[0]`` so the cold prior does not
        flip-flop between per-branch partial priors as split noise moves run
        in sequence within a stage.
        """
        lp = None
        for key in self._noise_model_branches(state):
            coords_b = np.asarray(state.branches_coords[key][0])
            ndim = coords_b.shape[-1]
            vals = (
                self.priors[key]
                .logpdf(coords_b.reshape(-1, ndim))
                .reshape(coords_b.shape[:2])
                .sum(axis=-1)
            )
            lp = vals if lp is None else lp + vals
        return lp

    # ------------------------------------------------------------------
    # stft_tof kernel fast path
    # ------------------------------------------------------------------

    def transform_coords(self, coords: list, return_cupy: bool = False, xp=None):
        """
        Prepare the coordinates for the move. This can include transforming the parameters if necessary.

        Args:
            coords: list of numpy arrays containing the parameters for the move. The first element should be the PSD parameters, and the second element (if present) should be the galactic foreground parameters.
            return_cupy: if True, will return the transformed coordinates as cupy arrays for use on the GPU. If False, will return numpy arrays.
            xp: the array library to use (numpy or cupy).
        Returns:
            A tuple containing the transformed PSD parameters and galactic foreground parameters (if present) in the target array library.
        """
        if xp is None:
            xp = self.acs.xp  # Use the array library from the analysis container

        if self.psd_transform_fn is not None:
            psd_pars = self.psd_transform_fn.both_transforms(coords[0])
        else:
            psd_pars = coords[0]

        if len(coords) == 1:
            galfor_pars = np.tile(np.array([1e-200, 1e-3, 1.0, 1.0, 1.0]), (psd_pars.shape[0], 1))
        else:
            if self.galfor_transform_fn is not None:
                galfor_pars = self.galfor_transform_fn.both_transforms(coords[1])
            else:
                galfor_pars = coords[1]

        if return_cupy:
            psd_pars = xp.asarray(psd_pars)
            galfor_pars = xp.asarray(galfor_pars)

        return psd_pars, galfor_pars

    def prepare_likelihood_inputs(self, psd_pars, galfor_pars) -> tuple:
        """
        Prepare the inputs for the likelihood computation. This can include placing the data on the correct device and formatting the parameters as needed.
        """

        xp = self.acs.xp  # Use the appropriate array library (numpy or cupy)

        # Use .copy() (not xp.ascontiguousarray) to guarantee each kernel input
        # owns an independent allocation. ascontiguousarray is a no-op when the
        # slice is already contiguous — which is the case for a column of a
        # Fortran-order 2D array. In that case Soms_d_in_all and Sa_a_in_all
        # would alias psd_pars's buffer, and the CUDA kernel would read through
        # those aliased pointers — exposing it to cupy memory-pool reuse
        # hazards while the kernel runs asynchronously. .copy() forces a
        # fresh, contiguous, owning allocation.
        Soms_d_in_all = psd_pars[:, 0].copy()
        Sa_a_in_all = psd_pars[:, 1].copy()

        if self.sensitivity_backend.use_splines:
            knots_positions = xp.asarray(psd_pars[:, 3::2])
            knots_amplitudes = xp.asarray(psd_pars[:, 2:-1:2])
            half = knots_positions.shape[1] // 2  # Get the mid of the array
            # put the 2 noise levels on the batch axis
            spline_knots_amplitude = xp.stack(
                (knots_amplitudes[:, :half], knots_amplitudes[:, half:])
            )
            spline_knots_position = xp.stack(
                (knots_positions[:, :half], knots_positions[:, half:])
            )

            # Sort
            sort_indices = xp.argsort(spline_knots_position, axis=2)
            # Apply the same indices to both arrays
            spline_knots_position = xp.take_along_axis(
                spline_knots_position, sort_indices, axis=2
            )
            spline_knots_amplitude = xp.take_along_axis(
                spline_knots_amplitude, sort_indices, axis=2
            )
            # now check if any knot position is not too close together
        else:
            spline_knots_position = None
            spline_knots_amplitude = None

        # Same defensive copy for the galactic-foreground components. The
        # previous code unpacked `*xp.ascontiguousarray(galfor_pars.T)` which,
        # for an already-contiguous input, would unpack 5 views into the same
        # base array — same aliasing hazard as Soms/Sa. Copying each row
        # independently gives 5 owning 1D arrays.
        galfor_pars_T = galfor_pars.T  # view; transpose is always free
        Amp_all = galfor_pars_T[0].copy()
        alpha_all = galfor_pars_T[1].copy()
        f_1_all = galfor_pars_T[2].copy()
        kn_all = galfor_pars_T[3].copy()
        f_2_all = galfor_pars_T[4].copy()

        likelihood_args = (
            Soms_d_in_all,
            Sa_a_in_all,
            Amp_all,
            alpha_all,
            f_1_all,
            kn_all,
            f_2_all,
            spline_knots_position,
            spline_knots_amplitude,
        )

        return likelihood_args

    def psd_log_like(self, x: list, supps=None, **sens_kwargs) -> np.ndarray:
        """
        Internal method to compute the log likelihood for the PSD parameters via the C++ XYZ sensitivity kernel (stft_tof fast path).

        Args:
            x: list of numpy arrays containing the parameters for the move. The first element should be the PSD parameters, and the second element (if present) should be the galactic foreground parameters.
            supps: supplemental information for the likelihood computation, such as walker indices.
            **sens_kwargs: additional keyword arguments to pass to the sensitivity backend for likelihood computation.
        Returns:
            A numpy array containing the log likelihood values for each set of parameters.
        """
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        if self._dcga is not None and not getattr(self, "_force_parent_path", False):
            # GPU-spread path: per-device replicas via the DCGA this move was
            # constructed with (one move structure; see __init__).
            return self._psd_log_like_dcga(x, supps=supps, **sens_kwargs)

        xp = self.acs.xp  # Use the appropriate array library (numpy or cupy)

        wi = supps["walker_inds"]
        data_index_all = xp.asarray(wi).astype(np.int32)

        psd_pars, galfor_pars = self.transform_coords(x, return_cupy=True, xp=xp)

        likelihood_args = self.prepare_likelihood_inputs(psd_pars, galfor_pars)

        ll = self.sensitivity_backend.compute_log_like(
            self.acs.linear_data_arr[0], data_index_all, *likelihood_args
        )

        if likelihood_args[-2] is not None and likelihood_args[-1] is not None:
            invalid_knots = xp.any(
                xp.diff(10 ** likelihood_args[-2], axis=2) < self.tolerance, axis=(0, 2)
            )
        else:
            invalid_knots = xp.zeros(psd_pars.shape[0], dtype=bool)

        ll[invalid_knots] = -1e300

        return ll.get() if hasattr(ll, "get") else ll

    def _kernel_fast_path_available(self, has_sgwb: bool = False) -> bool:
        """True when the C++ XYZ sensitivity kernel can score this proposal.

        Conditions: a sensitivity backend is configured, the data basis is FD
        or STFT (the kernel's two domains — the WDM counterpart is planned in
        domains.cu), the ACA holds a single shard (the kernel reads
        ``linear_data_arr[0]`` directly), and the noise MODEL carries no sgwb
        branch — the kernel has no SGWB component, so scoring an sgwb-bearing
        model through it (sampled or fixed) would silently drop that term.
        """
        if has_sgwb:
            return False
        if self.sensitivity_backend is None:
            return False
        basis = self.sensitivity_backend.basis_settings
        if not isinstance(basis, (FDSettings, STFTSettings)):
            return False
        # A DCGA-configured move scores each shard on its own replica, so
        # the kernel path stays available on multi-shard ACAs.
        return self._dcga is not None or len(self.acs.linear_data_arr) == 1

    # ------------------------------------------------------------------
    # dev ACA path + hybrid dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _to_physical(transform_fn, params):
        """One branch's sampling-basis row -> the physical basis the model wants.

        ``both_transforms`` can hand back a leading axis for a single row, so
        squeeze it the way :meth:`SensitivityBackendBase.__call__` does.
        """
        if transform_fn is None or params is None:
            return params
        out = transform_fn.both_transforms(np.asarray(params, dtype=float))
        return np.atleast_1d(np.asarray(out).squeeze())

    def _build_sensitivity_for_walker(
        self, walker_index: int, psd_params, galfor_params, sgwb_params=None
    ):
        """Build the per-walker sensitivity matrix for the given parameters.

        Takes every branch's row in ITS SAMPLING BASIS and applies that
        branch's transform here, so a transformed branch behaves the same on
        this route as on the kernel fast path (:meth:`transform_coords`) --
        the backend only ever knows about the psd transform, so a log-sampled
        galfor/sgwb would otherwise reach the model unexponentiated.

        Routes through the configured :class:`XYZSensitivityBackend`. The
        returned :class:`SensitivityMatrix` is what will be installed on the
        AnalysisContainer for the matching walker when we accept proposals
        (see :meth:`propose`).
        """
        # ``sgwb_params`` is only forwarded when present so the legacy
        # XYZSensitivityBackend (whose __call__ has no sgwb kwarg) keeps
        # working for runs without an sgwb branch.
        sgwb_params = self._to_physical(self.sgwb_transform_fn, sgwb_params)
        extra = {} if sgwb_params is None else dict(sgwb_params=sgwb_params)
        return self.sensitivity_backend(
            f"walker_{walker_index}",
            self._to_physical(self.psd_transform_fn, psd_params),
            galfor_params=self._to_physical(self.galfor_transform_fn, galfor_params),
            **extra,
        )

    def compute_log_like(self, coords, inds=None, logp=None, supps=None, branch_supps=None):
        """Compute the PSD/galfor branch log-likelihood.

        Hybrid routing (stft_tof merge): when the C++ XYZ kernel fast path is
        available (FD/STFT basis, single shard), score the proposals directly
        through ``XYZSensitivityBackend.compute_log_like``. Otherwise fall
        back to the dev-side, domain-agnostic route: install the proposed PSD
        (and optional galactic foreground) into each walker's
        :class:`AnalysisContainer` ``sens_mat`` and read the per-walker
        likelihood from the shared :class:`AnalysisContainerArray` (works for
        FD / STFT / WDM alike).

        Args:
            coords: Branch-keyed dict of coordinates from the current state.
            inds: Branch-keyed dict of leaf occupancy flags (unused here).
            logp: Optional pre-computed log prior; computed if ``None``.
            supps: Eryn supplemental information; ``walker_inds`` selects
                which physical walker each row maps to.
            branch_supps: Branch-supplemental dict (unused here).

        Returns:
            Tuple ``(logl, blobs)`` matching the eryn signature.
        """
        if logp is None:
            logp = self.compute_log_prior(coords, inds=inds, supps=supps, branch_supps=branch_supps)

        assert logp is not None
        logl = np.full_like(logp, -1e300)

        logp_keep = ~np.isinf(logp)
        if not np.any(logp_keep):
            warnings.warn("All points entering likelihood have a log prior of minus inf.")
            return logl, None

        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        # ``walker_inds`` is broadcast (ntemps, nwalkers) — flatten and mask
        # to the rows that survived the prior cut. ``BranchSupplemental``'s
        # ``__getitem__`` expects an integer/slice index, so reach into
        # ``.holder`` to fetch the named entry.
        walker_inds_all = np.asarray(supps.holder["walker_inds"]).reshape(logp.shape)
        walker_inds_keep = walker_inds_all[logp_keep].astype(int)

        # full noise model per row: sampled branches from the proposal,
        # fixed branches from the cold-row snapshot (per walker)
        merged = self._merged_noise_rows(coords, logp_keep, walker_inds_keep)
        psd_coords = merged["psd"]
        has_galfor = "galfor" in merged
        galfor_coords = merged.get("galfor")
        has_sgwb = "sgwb" in merged
        sgwb_coords = merged.get("sgwb")

        if self._kernel_fast_path_available(has_sgwb=has_sgwb):
            # stft_tof fast path: C++ shared-memory kernel.
            if has_galfor:
                input_args = [psd_coords, galfor_coords]
            else:
                input_args = [psd_coords]

            supps_keep = supps[logp_keep]
            logl[logp_keep] = self.psd_log_like(input_args, supps=supps_keep, **self.psd_kwargs)

            self.prev_logl = logl.copy()
            return logl, None

        # Cache and restore the per-walker sensitivity matrix so we don't
        # corrupt the state seen by other moves. After all proposals are
        # scored we put each AC's original sens_mat back; the caller (the
        # ``propose`` loop) reinstalls the accepted PSD onto the ACs.
        # There are ntemps*nwalkers proposal rows but only nwalkers ACs, and
        # ``walker_inds`` maps every rung of walker w onto acs[w]. Scoring all
        # rows in one pass would let each rung overwrite the previous one's
        # sens_mat, and the single batched ``likelihood()`` would then hand
        # every rung of a walker the SAME value (the last one written) --
        # the cold rung scored with a hot rung's parameters, so the Metropolis
        # ratio compares unrelated likelihoods and accepts nearly everything.
        # Score in batches instead, each holding a walker at most once.
        original_sens = {}
        tmp_logl = np.empty(walker_inds_keep.shape[0], dtype=float)
        try:
            remaining = np.arange(walker_inds_keep.shape[0])
            while remaining.size:
                # first remaining row for each distinct walker -> one batch
                _, first = np.unique(walker_inds_keep[remaining], return_index=True)
                batch = remaining[np.sort(first)]
                for row in batch:
                    w = int(walker_inds_keep[row])
                    if w not in original_sens:
                        original_sens[w] = self.acs[w].sens_mat
                    galfor_here = None if not has_galfor else galfor_coords[row]
                    sgwb_here = None if not has_sgwb else sgwb_coords[row]
                    self.acs[w].sens_mat = self._build_sensitivity_for_walker(
                        w, psd_coords[row], galfor_here, sgwb_here
                    )
                self.acs.reset_linear_psd_arr()
                walker_ll = asnumpy(np.asarray(self.acs.likelihood()))
                tmp_logl[batch] = walker_ll[walker_inds_keep[batch].astype(int)]
                remaining = np.setdiff1d(remaining, batch)
            logl[logp_keep] = tmp_logl
        finally:
            for w, sens in original_sens.items():
                self.acs[w].sens_mat = sens
            self.acs.reset_linear_psd_arr()

        self.prev_logl = logl.copy()

        return logl, None

    def compute_log_prior(self, branches_coords, *args, **kwargs):
        """Sum the per-branch log priors over THIS move's sampled branches.

        Fixed (non-sampled) noise branches are constants during this move's
        Metropolis-Hastings updates, so their priors cancel in the acceptance
        ratio and must not be summed here (a split move using the joint prior
        would double-count them against another noise move).

        Args:
            branches_coords: Branch-keyed dict of coordinates.

        Returns:
            ``(ntemps, nwalkers)`` array of prior log probabilities.
        """
        # wait to get ntemps, nwalkers
        logp = None
        for key in self.NOISE_BRANCHES:
            if key not in branches_coords:
                continue
            if self.sampled_branches is not None and key not in self.sampled_branches:
                continue
            ntemps, nwalkers, _, ndim = branches_coords[key].shape
            if logp is None:
                logp = np.zeros((ntemps, nwalkers))

            logp[:] += (
                self.priors[key]
                .logpdf(branches_coords[key].reshape(-1, ndim))
                .reshape(ntemps, nwalkers)
            )
        return logp

    def run_move(self, move_i, model, state):
        """Run one stretch-move iteration and (optionally) a tempering swap.

        Args:
            move_i: Iteration index used to schedule the periodic
                :attr:`permute_every` temperature swap.
            model: Eryn ``Model`` object.
            state: Current sampler state.

        Returns:
            Tuple ``(new_state, accepted)``.
        """
        new_state, accepted = super(PSDMove, self).propose(model, state)

        # in-model bookkeeping: eryn returns (ntemps, nwalkers) acceptances and
        # every walker is proposed once per call, so the per-temperature deltas
        # are the row sums / the walker count. Tallies are reset per propose()
        # and written into the sub-states there (see delta_counter_names).
        acc = np.asarray(accepted)
        acc = acc.reshape(acc.shape[0], -1)
        if self._tally_in_model_accepted is not None:
            self._tally_in_model_accepted += acc.sum(axis=-1).astype(int)
            self._tally_in_model_proposed += acc.shape[-1]

        if move_i % self.permute_every == 0:
            x = new_state.branches_coords
            logl = new_state.log_like
            logp = new_state.log_prior
            branch_supps = new_state.branches_supplemental
            supps = new_state.supplemental

            logP = self.compute_log_posterior(logl, logp)
            x, logP, logl, logp, inds, blobs, supps, branch_supps = (
                self.temperature_control.temperature_swaps(
                    x,
                    logP,
                    logl,
                    logp,
                    supps=supps,
                    branch_supps=branch_supps,
                    compute_log_like=self.compute_log_like,
                    compute_log_prior=self.compute_log_prior,
                    fancy_swap=True,
                    permute_here=True,
                )
            )

            for name in x:
                new_state.branches[name].coords[:] = x[name][:]
                new_state.branches[name].branch_supplemental = branch_supps[name]

            new_state.log_like[:] = logl[:]
            new_state.log_prior[:] = logp[:]
            new_state.supplemental = supps

            # swap bookkeeping: TemperatureControl refreshes swaps_accepted on
            # every call and holds swaps_proposed fixed at nwalkers per rung.
            tc = self.temperature_control
            if self._tally_swaps_accepted is not None:
                sa = getattr(tc, "swaps_accepted", None)
                sp = getattr(tc, "swaps_proposed", None)
                if sa is not None:
                    self._tally_swaps_accepted += np.asarray(sa).ravel().astype(int)
                if sp is not None:
                    self._tally_swaps_proposed += np.asarray(sp).ravel().astype(int)

        return new_state, accepted

    def run_move_for_loop(self, model, state, num_repeats):
        """Run :meth:`run_move` ``num_repeats`` times sequentially."""
        for i in tqdm(
            range(num_repeats), desc="psd update",
            disable=not getattr(self, "progress", False),
        ):
            state, accepted = self.run_move(i, model, state)
        return state, accepted

    def run_move_max_likelihood(self, model, state):
        """Repeat :meth:`run_move_for_loop` until the max log-like plateaus.

        Used in search-style runs (``max_logl_mode=True``). The loop counts
        consecutive iterations during which the cold-chain max log-likelihood
        no longer increases and exits after ``num_checks`` such iterations.
        """
        num_checks = 5
        num_so_far = 0
        max_logl = -np.inf
        changed_once = False
        while num_so_far < num_checks:
            state, accepted = self.run_move_for_loop(model, state, self.num_repeats)

            if state.log_like[0].max() != max_logl and not np.isinf(max_logl):
                changed_once = True

            if state.log_like[0].max() > max_logl:
                max_logl = state.log_like[0].max()
                num_so_far = 0
            else:
                if changed_once:
                    num_so_far += 1

        return state, accepted

    def propose(self, model, state):
        """Propose a noise-model update and refresh per-walker sensitivity matrices.

        Builds a temporary :class:`GFState` containing only THIS move's
        sampled noise branches (non-sampled noise branches are frozen at
        their cold rows and merged into every likelihood evaluation), runs
        the inner stretch-move loop, then writes the accepted coordinates
        back into a copy of ``state`` and refreshes each walker's
        sensitivity matrix in :attr:`acs`.

        Returns:
            Tuple ``(new_state, accepted)``.
        """
        model_branches = self._noise_model_branches(state)
        noise_branches = self._resolve_sampled(state)

        # cold-row agreement between the main state and the noise-model
        # branches' sub-states — fixed inputs included, they feed the
        # likelihood too (GF_SUBSTATE_CHECK=0 disables)
        self._check_substate_consistency(state, model_branches)
        engine_ntemps = state.log_like.shape[0]

        # freeze the non-sampled noise branches at their cold rows for the
        # duration of this proposal block (consumed by _merged_noise_rows)
        self._fixed_noise_coords = {
            key: np.asarray(state.branches_coords[key][0, :, 0]).copy()
            for key in model_branches
            if key not in noise_branches
        }

        # The working ensembles are the SUB-STATES' tempered branches (this
        # move's module ladder); the main state carries only the engine's
        # cold chain.
        tmp_branches_coords = {
            key: self._work_branch(state, key).coords for key in noise_branches
        }

        shapes = {key: tmp_branches_coords[key].shape[:2] for key in noise_branches}
        if len(set(shapes.values())) != 1:
            raise ValueError(
                "Noise branches sampled by ONE move must share (ntemps, nwalkers); "
                f"got {shapes}. Match the branches' ntemps knobs "
                "(PSD_NTEMPS / GALFOR_NTEMPS / SGWB_NTEMPS) or sample them "
                "with separate moves."
            )

        # move-local supplemental at the MODULE ladder shape (the main
        # state's supplemental is engine-shaped)
        nt_mod, nwalkers_mod = tmp_branches_coords[noise_branches[0]].shape[:2]

        # per-iteration acceptance deltas, accumulated across the repeat block
        # by run_move and written into each sampled branch's sub-state below.
        self._tally_in_model_proposed = np.zeros(nt_mod, dtype=int)
        self._tally_in_model_accepted = np.zeros(nt_mod, dtype=int)
        self._tally_swaps_proposed = np.zeros(max(nt_mod - 1, 0), dtype=int)
        self._tally_swaps_accepted = np.zeros(max(nt_mod - 1, 0), dtype=int)
        tmp_supps = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(nwalkers_mod), (nt_mod, 1))},
            base_shape=(nt_mod, nwalkers_mod),
            copy=True,
        )

        # module-ladder acceptance tracking (the stage combine-move setter
        # presets an engine-shaped array, so re-shape on first use)
        _acc = getattr(self, "_accepted", None)
        if _acc is None or np.shape(_acc) != (nt_mod, nwalkers_mod):
            self.accepted = np.zeros((nt_mod, nwalkers_mod))

        tmp_state = GFState(tmp_branches_coords, copy=True, supplemental=tmp_supps)

        # ensuring it is up to date. Should not change anything.
        before_vals = model.analysis_container_arr.likelihood().copy()

        tmp_state.log_prior = self.compute_log_prior(tmp_branches_coords)
        tmp_state.log_like = self.compute_log_like(
            tmp_branches_coords, logp=tmp_state.log_prior, supps=tmp_state.supplemental
        )[0]
        self.starting_now = False

        tmp_model = Model(
            state,
            self.compute_log_like,
            self.compute_log_prior,
            self.temperature_control,
            model.map_fn,
            model.random,
        )

        if self.max_logl_mode:
            tmp_state, accepted = self.run_move_max_likelihood(tmp_model, tmp_state)

        else:
            tmp_state, accepted = self.run_move_for_loop(tmp_model, tmp_state, self.num_repeats)

        # CHECK THIS STATE SETUP
        new_state = GFState(state, copy=True)

        for key in noise_branches:
            if key not in tmp_state.branches:
                continue
            # full-ladder write into the sub-state (shared-memory view),
            # cold row mirrored into the main (engine) state. Only the
            # SAMPLED branches are written: each noise branch's sub-state is
            # owned by the move that samples it (full-noise-model tempered
            # lnL of ITS ensemble + ITS move's adapted ladder).
            self._work_branch(new_state, key).coords[:] = tmp_state.branches[key].coords[:]
            self._sync_cold_row(new_state, key)
            sub = (new_state.sub_states or {}).get(key)
            if sub is not None and getattr(sub, "tempered_initialized", False):
                sub.log_like[:] = tmp_state.log_like[:]
                sub.log_prior[:] = tmp_state.log_prior[:]
                if hasattr(sub, "betas") and sub.betas is not None:
                    sub.betas[:] = self.temperature_control.betas
                # acceptance deltas for this iteration (the backend zeroes
                # them after each save, so these are per-iteration counts).
                # rj_* stay zero: this move never changes leaf count.
                sub.in_model_proposed[:] = self._tally_in_model_proposed
                sub.in_model_accepted[:] = self._tally_in_model_accepted
                if sub.swaps_proposed.size:
                    sub.swaps_proposed[:] = self._tally_swaps_proposed
                    sub.swaps_accepted[:] = self._tally_swaps_accepted

        # the engine state keeps only the cold row (row 0 rewritten from the
        # refreshed ACS below). The cold prior is the FULL noise-model sum so
        # it doesn't flip-flop between partial priors as split moves run.
        new_state.log_like[0] = tmp_state.log_like[0]
        new_state.log_prior[0] = self._cold_noise_log_prior(new_state)

        # TODO: check speed of this? (needed?)
        nwalkers = len(self.acs)
        for w in range(nwalkers):
            psd_params = new_state.branches_coords["psd"][0, w, 0]
            if "galfor" in new_state.branches_coords:
                galfor_params = new_state.branches_coords["galfor"][0, w, 0]
            else:
                galfor_params = None
            if "sgwb" in new_state.branches_coords:
                sgwb_params = new_state.branches_coords["sgwb"][0, w, 0]
            else:
                sgwb_params = None

            new_sens = self._build_sensitivity_for_walker(
                w, psd_params, galfor_params, sgwb_params
            )
            self.acs[w].sens_mat = new_sens

        self.acs.reset_linear_psd_arr()
        after_vals = self.acs.likelihood()

        new_state.log_like[0] = after_vals
        # eryn-facing acceptance uses the engine (cold-chain) shape
        return new_state, np.asarray(accepted)[:engine_ntemps]


    # ------------------------------------------------------------------
    # DCGA (multi-device) kernel path — absorbed from the legacy
    # MultiGPUPSDMove in the 2026-07 multi-GPU pass (one move structure).
    # ------------------------------------------------------------------

    def _psd_log_like_dcga(self, x: list, supps=None, **sens_kwargs):
        """Kernel fast path through the DCGA's per-device replicas."""
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        wi = supps["walker_inds"]

        psd_pars, galfor_pars = self.transform_coords(x, return_cupy=False)

        data_index_all = np.asarray(wi).astype(np.int32)

        positions_per_split, data_intra_index_per_split, _ = self.dcga.unpack_indices(data_index_all)
        coords_per_split = self.dcga.unpack_coords(positions_per_split, (psd_pars, galfor_pars))

        data_intra_index_per_split, coords_per_split = self.dcga.place_on_device(
            items=(data_intra_index_per_split, coords_per_split)
        )

        likelihood_args_per_split = self.dcga._loop_operation(
            operation=self.prepare_likelihood_inputs,
            operation_args_per_split=coords_per_split,
            positions_per_split=positions_per_split,
        )

        ll = self.dcga.compute_psd_likelihood(
            positions_per_split,
            data_intra_index_per_split,
            data_intra_index_per_split,
            likelihood_args_per_split,
            likelihood_kwargs={'run_async': self.run_async},
            run_threaded=self.run_threaded,
        )

        # now check if any knot position is not too close together
        if self.sensitivity_backend.use_splines:
            invalid_knots_mask = np.zeros(psd_pars.shape[0], dtype=bool)
            for i, likelihood_args in enumerate(likelihood_args_per_split):
                if likelihood_args is None:
                    continue
                spline_knots_position = (
                    likelihood_args[-2].get()
                    if hasattr(likelihood_args[-2], "get")
                    else likelihood_args[-2]
                )

                invalid_knots_mask[positions_per_split[i]] = np.any(
                    np.diff(10**spline_knots_position, axis=2) < self.tolerance, axis=(0, 2)
                )
            ll[invalid_knots_mask] = -1e300

        self.dcga.free_gpu_memory()
        return ll


class MultiGPUPSDMove(PSDMove):
    """Deprecated alias — :class:`PSDMove` is ONE structure that takes
    ``dcga=`` directly (2026-07 multi-GPU pass). This shim maps the legacy
    ``MultiGPUPSDMove(dcga, priors, ...)`` signature onto it.
    """

    def __init__(self, dcga: DomainComputationGroupArray, priors, *args,
                 run_async: bool = False, run_threaded: bool = False,
                 **kwargs):
        warnings.warn(
            "MultiGPUPSDMove is deprecated: construct "
            "PSDMove(acs, priors, dcga=...) directly (one move structure).",
            DeprecationWarning, stacklevel=2,
        )
        PSDMove.__init__(
            self, dcga.acs, priors, *args, dcga=dcga,
            run_async=run_async, run_threaded=run_threaded, **kwargs,
        )
