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
from eryn.state import State as eryn_State
from eryn.utils.transform import TransformContainer

# from ..utils import new_sens_mat
from tqdm import tqdm

from .globalfitmove import GlobalFitMove
from .multigpumove import MultiGPUMoveBase
from ..state import GFState
from ...analysiscontainer import AnalysisContainerArray
from ...domaincomputation import DomainComputationGroupArray
from ...sensitivity import XYZSensitivityBackend

logger = logging.getLogger(__name__)

DEBUG_MODE = False

class PSDMove(GlobalFitMove, StretchMove):
    """
    Move for sampling over PSD parameters. Can also include galactic foreground parameters if desired.

    Args:
        acs: AnalysisContainerArray containing the data and sensitivity information.
        priors: dictionary of priors for the parameters.
        *args: additional arguments for the Move class.
        num_repeats: number of times to repeat the move before returning.
        max_logl_mode: if True, will keep running the move until the maximum log likelihood does not change for a certain number of checks. This is useful for finding the maximum likelihood point.
        psd_kwargs: additional keyword arguments for the psd_log_like function.
        sensitivity_backend: instance of XYZSensitivityBackend to use for computing the likelihood.
        psd_transform_fn: TransformContainer for transforming the PSD parameters.
        galfor_transform_fn: TransformContainer for transforming the galactic foreground parameters.
        permute_every: number of repeats after which to permute the walkers during a temperature swap. This helps with the mixing of the chains.
        tolerance: minimum allowed distance between spline knot positions in the sensitivity model. 
        **kwargs: additional keyword arguments for the Move class.
    """

    def __init__(
        self,
        acs: AnalysisContainerArray,
        priors,
        *args,
        num_repeats: int = 1,
        max_logl_mode: bool = False,
        psd_kwargs: dict = {},
        sensitivity_backend: XYZSensitivityBackend = None,
        psd_transform_fn: TransformContainer = None,
        galfor_transform_fn: TransformContainer = None,
        permute_every: int = 20,
        tolerance: float = 0.0,
        **kwargs,
    ):

        GlobalFitMove.__init__(self, *args, **kwargs)
        StretchMove.__init__(self, *args, **kwargs)
        self.acs = acs
        self.psd_kwargs = psd_kwargs
        self.priors = priors
        self.num_repeats = num_repeats
        self.max_logl_mode = max_logl_mode
        self.starting_now = True

        self.sensitivity_backend = sensitivity_backend
        self.psd_transform_fn = psd_transform_fn
        self.galfor_transform_fn = galfor_transform_fn

        self.permute_every = permute_every
        self.tolerance = tolerance

        self._propose_call_count = 0
        self._sanity_atol = 1e-1

    def transform_coords(self, coords: list[np.ndarray], return_cupy: bool = False, xp=None) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
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

    def prepare_likelihood_inputs(
        self, psd_pars: np.ndarray | cp.ndarray, galfor_pars: np.ndarray | cp.ndarray
    ) -> tuple:
        """
        Prepare the inputs for the likelihood computation. This can include placing the data on the correct device and formatting the parameters as needed.
        """

        xp = self.acs.xp  # Use the appropriate array library (numpy or cupy)

        # Use .copy() (not xp.ascontiguousarray) to guarantee each kernel input
        # owns an independent allocation. ascontiguousarray is a no-op when the
        # slice is already contiguous — which is the case for a column of a
        # Fortran-order 2D array (as produced by DCGA's unpack_coords on the
        # MultiGPU path). In that case Soms_d_in_all and Sa_a_in_all would
        # alias psd_pars's buffer, and the CUDA kernel would read through those
        # aliased pointers — exposing it to cupy memory-pool reuse hazards
        # while the kernel runs asynchronously. .copy() forces a fresh,
        # contiguous, owning allocation.
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

    def psd_log_like(self, x: list[np.ndarray], supps=None, **sens_kwargs) -> np.ndarray:
        """
        Internal method to compute the log likelihood for the PSD parameters. This is called by the compute_log_like method after preparing the coordinates and priors. It uses the sensitivity backend to compute the likelihood based on the current parameters and the data.

        Args:
            x: list of numpy arrays containing the parameters for the move. The first element should be the PSD parameters, and the second element (if present) should be the galactic foreground parameters.
            supps: supplemental information for the likelihood computation, such as walker indices.
            **sens_kwargs: additional keyword arguments to pass to the sensitivity backend for likelihood computation.
        Returns:
            A numpy array containing the log likelihood values for each set of parameters.
        """
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

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

    def compute_log_like(self, coords, inds=None, logp=None, supps=None, branch_supps=None):
        if logp is None:
            logp = self.compute_log_prior(coords, inds=inds, supps=supps, branch_supps=branch_supps)

        assert logp is not None
        logl = np.full_like(logp, -1e300)

        logp_keep = ~np.isinf(logp)
        if not np.any(logp_keep):
            warnings.warn("All points entering likelihood have a log prior of minus inf.")
            return logl, None
        psd_coords = coords["psd"][logp_keep][:, 0]
        if "galfor" in coords:
            galfor_coords = coords["galfor"][logp_keep][:, 0]
            input_args = [psd_coords, galfor_coords]
        else:
            input_args = [psd_coords]

        supps = supps[logp_keep]

        tmp_logl = self.psd_log_like(input_args, supps=supps, **self.psd_kwargs)

        logl[logp_keep] = tmp_logl

        self.prev_logl = logl.copy()

        return logl, None

    def compute_log_prior(self, branches_coords, *args, **kwargs):
        # wait to get ntemps, nwalkers
        logp = None
        for key in ["psd", "galfor"]:
            if key not in branches_coords:
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
        new_state, accepted = super(PSDMove, self).propose(model, state)

        if move_i % self.permute_every == 0 and move_i > 0:
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

        return new_state, accepted

    def run_move_for_loop(self, model, state, num_repeats):
        for i in tqdm(range(num_repeats), desc="psd update"):
            state, accepted = self.run_move(i, model, state)
        return state, accepted

    def run_move_max_likelihood(self, model, state):

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

            logger.info(f"Max log-likelihood: {max_logl}. For {num_so_far} out of {num_checks} checks the log-likelihood has not changed.")

        # logger.debug the max log-likelihood and the corresponding parameters
        logger.debug(f"Max log-likelihood found: {max_logl}")
        logger.debug(f"Corresponding PSD parameters: {state.branches_coords['psd'][0, np.argmax(state.log_like[0]), :]}")
        if "galfor" in state.branches_coords:
            logger.debug(f"Corresponding galfor parameters: {state.branches_coords['galfor'][0, np.argmax(state.log_like[0]), :]}")
        
        return state, accepted

    def propose(self, model, state):
        # setup model framework for passing necessary
        # self.priors["all_models_together"].full_state = state
        xp = self.acs.xp  # Use the appropriate array library (numpy or cupy)

        tmp_branches_coords = {
            key: state.branches_coords[key]
            for key in ["psd", "galfor"]
            if key in state.branches_coords
        }

        tmp_state = GFState(tmp_branches_coords, copy=True, supplemental=state.supplemental)

        # ensuring it is up to date. Should not change anything.
        # eryn_state_in = eryn_State(state.branches_coords, inds=state.branches_inds, supplemental=state.supplemental, branch_supplemental=state.branches_supplemental, betas=state.betas, log_like=state.log_like, log_prior=state.log_prior, copy=True)
        before_vals = model.analysis_container_arr.likelihood().copy()

        # TODO: check this
        # if self.starting_now:
        tmp_state.log_prior = self.compute_log_prior(tmp_branches_coords)
        tmp_state.log_like = self.compute_log_like(
            tmp_branches_coords, logp=tmp_state.log_prior, supps=tmp_state.supplemental
        )[0]
        self.starting_now = False

        self._propose_call_count += 1

        # CHECK 1: stored vs freshly-recomputed log_like at propose entry.
        # Fires when the (params, log_like) coming IN to propose is inconsistent —
        # i.e., something between iterations is corrupting the stored log_like
        # relative to the stored coords. Most likely suspect: the line ~400
        # `new_state.log_like[0] = after_vals` overwrite at the end of the
        # previous propose.
        if DEBUG_MODE:
            diff_in = np.abs(tmp_state.log_like - state.log_like)
            mask_in = diff_in > self._sanity_atol
            if mask_in.any():
                bad = np.where(mask_in)
                logger.warning(
                    "[CHECK1 propose#%d entry] stored vs fresh log_like diverge "
                    "at %s: stored=%s, fresh=%s, max_diff=%.3e",
                    self._propose_call_count, list(zip(*bad)),
                    state.log_like[bad], tmp_state.log_like[bad],
                    float(diff_in.max()),
                )
        # if np.any(np.abs(before_vals - tmp_state.log_like[0]) > 1e-4) :
        #     breakpoint()

        # breakpoint()
        # logp = model.compute_log_prior_fn(state.branches_coords, inds=state.branches_inds, supps=state.supplemental)
        # logl_test = self.compute_log_like(state.branches_coords, inds=state.branches_inds, supps=state.supplemental, logp=logp)
        tmp_coords_check = state.branches["psd"].coords[0, :, 0].copy()
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

        for key in ["psd", "galfor"]:
            if key not in tmp_state.branches:
                continue
            new_state.branches[key].coords[:] = tmp_state.branches[key].coords[:]

        new_state.log_like[:] = tmp_state.log_like[:]
        new_state.log_prior[:] = tmp_state.log_prior[:]

        # TODO: check speed of this? (needed?)
        nwalkers = len(self.acs)
        for w in range(nwalkers):
            psd_params = new_state.branches_coords["psd"][0, w, 0]
            psd_params = self.psd_transform_fn.both_transforms(psd_params) if self.psd_transform_fn is not None else psd_params
            if "galfor" in new_state.branches_coords:
                galfor_params = new_state.branches_coords["galfor"][0, w, 0]
                galfor_params = self.galfor_transform_fn.both_transforms(galfor_params) if self.galfor_transform_fn is not None else galfor_params
            else:
                galfor_params = None

            gpu = self.acs.gpu_map[w]
            if self.acs.gpus is not None:
                with xp.cuda.Device(gpu):
                    new_sens = self.sensitivity_backend(
                        f"walker_{w}",
                        psd_params,
                        galfor_params=galfor_params,
                    )
            else:
                new_sens = self.sensitivity_backend(
                    f"walker_{w}",
                    psd_params,
                    galfor_params=galfor_params,
                    transform_fn=self.psd_transform_fn,
                )

            self.acs[w].sens_mat = new_sens

        self.acs.reset_linear_psd_arr()
        after_vals = self.acs.likelihood()

        # CHECK 2: acs.likelihood() (cached sens_mat path) vs compute_log_like
        # (MCMC's stateless path) for the cold chain. The MCMC accepted/rejected
        # using compute_log_like values; if `after_vals` disagrees, the value
        # we're about to store and hand back to the backend is NOT the value
        # the MCMC dynamics used — producing apparent "spikes" without actually
        # breaking the chain. Divergence here points at the per-walker
        # sens_mat installation loop above (lines ~370-395) or stale acs cache.
        if DEBUG_MODE:
            mcmc_logl = self.compute_log_like(
                new_state.branches_coords, supps=new_state.supplemental
            )[0]
            diff_paths = np.abs(after_vals - mcmc_logl[0])
            mask_paths = diff_paths > self._sanity_atol
            if mask_paths.any():
                bad_w = np.where(mask_paths)[0]
                logger.warning(
                    "[CHECK2 propose#%d exit] acs.likelihood vs compute_log_like "
                    "diverge at walkers=%s: acs=%s, mcmc=%s, "
                    "psd_coords=%s, max_diff=%.3e",
                    self._propose_call_count, bad_w.tolist(),
                    after_vals[bad_w], mcmc_logl[0][bad_w],
                    new_state.branches_coords["psd"][0][bad_w].tolist(),
                    float(diff_paths.max()),
                )

        new_state.log_like[0] = after_vals
        return new_state, accepted


class MultiGPUPSDMove(PSDMove, MultiGPUMoveBase):
    def __init__(
        self,
        dcga: DomainComputationGroupArray,
        priors,
        *args,
        num_repeats: int = 1,
        max_logl_mode: bool = False,
        psd_kwargs: dict = {},
        psd_transform_fn: TransformContainer = None,
        galfor_transform_fn: TransformContainer = None,
        permute_every: int = 20,
        tolerance: float = 0.0,
        run_async: bool = False,
        run_threaded: bool = False,
        **kwargs,
    ):

        PSDMove.__init__(
            self,
            dcga.acs,
            priors,
            *args,
            num_repeats=num_repeats,
            max_logl_mode=max_logl_mode,
            psd_kwargs=psd_kwargs,
            sensitivity_backend=dcga.computation_groups[0].sensitivity_backend,
            psd_transform_fn=psd_transform_fn,
            galfor_transform_fn=galfor_transform_fn,
            permute_every=permute_every,
            tolerance=tolerance,
            **kwargs,
        )
        MultiGPUMoveBase.__init__(self, dcga, run_async=run_async, run_threaded=run_threaded)

        # TEST FLAG: when True, MultiGPUPSDMove.psd_log_like delegates to the
        # parent PSDMove.psd_log_like, completely bypassing DCGA's unpack/place/
        # loop_operation machinery. Only meaningful on single-GPU setups.
        # If flipping this to True makes CHECK1/CHECK2 stop firing, the bug is
        # localized to the DCGA path (unpack_coords/place_on_device/
        # _loop_operation/_compute_group_likelihood). Set from the settings
        # file via `psd_move._force_parent_path = True` after move construction.
        self._force_parent_path = False

    def psd_log_like(self, x: list[np.ndarray], supps=None, **sens_kwargs):
        """ """
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        # Single-GPU debug path: skip DCGA entirely and run the parent's direct
        # sensitivity_backend.compute_log_like call. This isolates whether the
        # bug is in the MultiGPU routing (DCGA unpack/place/loop) or elsewhere.
        if getattr(self, "_force_parent_path", False):
            return PSDMove.psd_log_like(self, x, supps=supps, **sens_kwargs)

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
