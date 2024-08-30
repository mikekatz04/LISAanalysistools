import numpy as np
from mpi4py import MPI
import os
import warnings
from copy import deepcopy
from lisatools.globalfit.state import State
from lisatools.globalfit.hdfbackend import HDFBackend as GBHDFBackend
from eryn.backends import HDFBackend
from eryn.moves import Move
from eryn.state import BranchSupplimental
from eryn.ensemble import EnsembleSampler
from global_fit_input.global_fit_settings import get_global_fit_settings
from lisatools.utils.multigpudataholder import MultiGPUDataHolder
import cupy as cp
from lisatools.sampling.prior import GBPriorWrap
from lisatools.globalfit.psdglobal import log_like as psd_log_like
from lisatools.globalfit.psdglobal import PSDwithGBPriorWrap
from lisatools.globalfit.moves import MBHSpecialMove
from eryn.model import Model
from eryn.state import State as eryn_State
from eryn.ensemble import _FunctionWrapper
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.state import GFBranchInfo, AllGFBranchInfo
from lisatools.globalfit.moves import GlobalFitMove
cp.cuda.runtime.setDevice(2)

from lisatools.globalfit.moves import GBSpecialStretchMove

class PlaceHolder(Move):
    def __init__(self, *args, **kwargs):
        super(PlaceHolder, self).__init__(*args, **kwargs)

    def propose(self, model, state):
        accepted = np.zeros_like(state.log_like, dtype=int)
        self.temperature_control.swaps_accepted = np.zeros(
            self.temperature_control.ntemps - 1
        )
        return state, accepted
from eryn.model import Model
# ModelIncludingPSD(Model):


from eryn.moves import StretchMove
class PSDMove(GlobalFitMove, StretchMove):
    def __init__(self, gb, mgh, priors, gpu_priors, *args, psd_kwargs={}, **kwargs):
        super(PSDMove, self).__init__(*args, **kwargs)
        self.mgh = mgh
        self.gb = gb
        self.psd_kwargs = psd_kwargs
        self.priors = priors
        self.gpu_priors = gpu_priors

        
    def compute_log_like(
        self, coords, inds=None, logp=None, supps=None, branch_supps=None
    ):
        assert logp is not None
        logl = np.full_like(logp, -1e300)

        logp_keep = ~np.isinf(logp)
        if not np.any(logp_keep):
            warnings.warn("All points entering likelihood have a log prior of minus inf.")
            return logl
        psd_coords = coords["psd"][logp_keep][:, 0]
        galfor_coords = coords["galfor"][logp_keep][:, 0]

        data_tmp = self.mgh.data_shaped
        data = [data_tmp[0][0].copy().flatten(), data_tmp[1][0].copy().flatten()]

        supps = supps[logp_keep]

        tmp_logl = psd_log_like([psd_coords, galfor_coords], cp.asarray(self.mgh.fd), data, self.gb, self.mgh.df, self.mgh.data_length, supps=supps, **self.psd_kwargs)

        logl[logp_keep] = tmp_logl
        return logl, None

    # def compute_log_prior(self, coords, inds=None, supps=None, branch_supps=None):

    #     ntemps, nwalkers, _, _ = coords[list(coords.keys())[0]].shape
 
    #     logp = np.zeros((ntemps, nwalkers))
    #     for key in ["galfor", "psd"]:
    #         ntemps, nwalkers, nleaves_max, ndim = coords[key].shape
    #         if nleaves_max > 1:
    #             raise NotImplementedError

    #         logp_contrib = self.priors[key].logpdf(coords[key].reshape(-1, ndim)).reshape(ntemps, nwalkers, nleaves_max).sum(axis=-1)
    #         logp[:] += logp_contrib

    #     # now that everything is lined up
    #     breakpoint()
    #     nleaves_max_gb = inds["gb"].shape[-1]
    #     gb_inds_tiled = cp.tile(cp.asarray(inds["gb"][0][None, :]), (ntemps, 1, 1))
    #     gb_coords = cp.tile(cp.asarray(coords["gb"][0]), (ntemps, 1, 1, 1))[gb_inds_tiled]
    #     walker_inds = cp.repeat(cp.arange(ntemps * nwalkers)[:, None], nleaves_max_gb, axis=-1).reshape(ntemps, nwalkers, nleaves_max_gb)[gb_inds_tiled]
    #     logp_per_bin = cp.zeros((ntemps, nwalkers, nleaves_max_gb))
    #     logp_per_bin[gb_inds_tiled] = self.gpu_priors["gb"].logpdf(gb_coords, psds=self.mgh.lisasens_list[0][0],walker_inds=walker_inds)
    #     logp[:] += logp_per_bin.sum(axis=-1).get()
        

    #     cp.get_default_memory_pool().free_all_blocks()
    #     return logp
    
    def propose(self, model, state):
        # setup model framework for passing necessary 
        self.priors["all_models_together"].full_state = state

        # ensuring it is up to date. Should not change anything.
        self.mgh = state.mgh
        eryn_state_in = eryn_State(state.branches_coords, inds=state.branches_inds, supplimental=state.supplimental, branch_supplimental=state.branches_supplimental, betas=state.betas, log_like=state.log_like, log_prior=state.log_prior, copy=True)
        before_vals = state.mgh.get_ll(include_psd_info=True).copy()
        
        tmp_model = Model(
            state,
            self.compute_log_like,
            self.priors["all_models_together"].logpdf,  # self.compute_log_prior,
            model.temperature_control,
            model.map_fn,
            model.random,
        )

        tmp_state, accepted = super(PSDMove, self).propose(tmp_model, state)
        new_state = State(
            tmp_state.branches_coords,
            inds=tmp_state.branches_inds,
            log_like=tmp_state.log_like,
            log_prior=tmp_state.log_prior,
            betas=tmp_state.betas,
            betas_all=state.betas_all,
            band_info=state.band_info,
            supplimental=tmp_state.supplimental,
            branch_supplimental=tmp_state.branches_supplimental,
            random_state=tmp_state.random_state,
            mgh=state.mgh
        )

        new_state.mgh.set_psd_vals(
            new_state.branches["psd"].coords[0, :, 0], 
            overall_inds=np.arange(new_state.branches["psd"].shape[1]), 
            foreground_params=new_state.branches["galfor"].coords[0, :, 0]
        )
        after_vals = new_state.mgh.get_ll(include_psd_info=True)
        return new_state, accepted


class BasicResidualMGHLikelihood:
    def __init__(self, mgh):
        self.mgh = mgh

    def __call__(self, *args, supps=None, **kwargs):
        ll_temp = self.mgh.get_ll()
        overall_inds = supps["overal_inds"]
        breakpoint()
        return ll_temp[overall_inds]

class GlobalFit:
    def __init__(self, curr, comm):
        self.comm = comm
        self.curr = curr
        self.rank = comm.Get_rank()

        self.all_ranks = list(range(self.comm.Get_size()))
        self.used_ranks = []
        self.head_rank = self.curr.rank_info["head_rank"]
        self.main_rank = self.curr.rank_info["main_rank"]
        self.used_ranks.append(self.head_rank)
        self.used_ranks.append(self.main_rank)

        self.ranks_to_give =  deepcopy(self.all_ranks)
        if self.head_rank in self.ranks_to_give:
            self.ranks_to_give.remove(self.head_rank)
        self.ranks_to_give.remove(self.main_rank)

        if comm.Get_size() < 3:
            self.results_rank = self.main_rank
        else:
            self.results_rank = self.ranks_to_give.pop()
            self.used_ranks.append(self.results_rank)
        
    def run_global_fit(self):
        
        if self.rank == self.curr.settings_dict["rank_info"]["main_rank"]: 
            general_info = self.curr.settings_dict["general"]
            gb_info = self.curr.settings_dict["gb"]
            mbh_info = self.curr.settings_dict["mbh"]
            psd_info = self.curr.settings_dict["psd"]

            gf_branch_information = GFBranchInfo("mbh", 11, 15, 15) + GFBranchInfo("gb", 8, 15000, 0) + GFBranchInfo("galfor", 5, 1, 1) + GFBranchInfo("psd", 4, 1, 1)

            branch_names = gf_branch_information.branch_names
            ndims = gf_branch_information.ndims
            nleaves_max = gf_branch_information.nleaves_max
            nleaves_min = gf_branch_information.nleaves_min
            nwalkers = gb_info["pe_info"]["nwalkers"]
            band_edges = gb_info["band_edges"]
            betas = gb_info["pe_info"]["betas"]
            ntemps = len(betas)

            band_temps = np.tile(np.asarray(betas), (len(band_edges) - 1, 1))

            supps_base_shape = (ntemps, nwalkers)
            walker_vals = np.tile(np.arange(nwalkers), (ntemps, 1))
            supps = BranchSupplimental({"walker_inds": walker_vals}, base_shape=supps_base_shape, copy=True)

            if os.path.exists("test_new.h5"):
                state = GBHDFBackend("test_new.h5").get_a_sample(0)

            else:
                coords = {key: np.zeros((ntemps, nwalkers, nleaves_max[key], ndims[key])) for key in branch_names}
                inds = {key: np.ones((ntemps, nwalkers, nleaves_max[key]), dtype=bool) for key in branch_names}
                inds["gb"][:] = False
                state = State(coords, inds=inds, random_state=np.random.get_state())
                state.initialize_band_information(nwalkers, ntemps, band_edges, band_temps)

            state.supplimental = supps

            # gb_backend = GBHDFBackend("global_fit_output/eighth_run_through_parameter_estimation_gb.h5")
            # psd_backend = HDFBackend("global_fit_output/eighth_run_through_parameter_estimation_psd.h5")
            # mbh_backend = HDFBackend("global_fit_output/eighth_run_through_parameter_estimation_mbh.h5")

            # last_gb = gb_backend.get_last_sample()
            # last_psd = psd_backend.get_last_sample()
            # last_mbh = mbh_backend.get_last_sample()

            # state.branches["gb"] = deepcopy(last_gb.branches["gb"])
            # state.branches["psd"].coords[:] = last_psd.branches["psd"].coords[0, :nwalkers]
            # # order of call function changed for galfor 
            # galfor_coords_orig = last_psd.branches["galfor"].coords[0, :nwalkers]
            # galfor_coords = np.zeros_like(galfor_coords_orig)
            # galfor_coords[:, :, 0] = galfor_coords_orig[:, :, 0]
            # galfor_coords[:, :, 1] = galfor_coords_orig[:, :, 3]
            # galfor_coords[:, :, 2] = galfor_coords_orig[:, :, 1]
            # galfor_coords[:, :, 3] = galfor_coords_orig[:, :, 2]
            # galfor_coords[:, :, 4] = galfor_coords_orig[:, :, 4]
            # state.branches["galfor"].coords[:] = galfor_coords
            # state.branches["mbh"].coords[:] = last_mbh.branches["mbh"].coords[0, :nwalkers]

            # FOR TESTING
            # state.branches["gb"].coords[:] = state.branches["gb"].coords[0, 0][None, None, :, :]
            # state.branches["gb"].inds[:] = state.branches["gb"].inds[0, 0][None, None, :]
            # state.branches["mbh"].coords[:] = state.branches["mbh"].coords[0, 0][None, None, :, :]
            # state.branches["psd"].coords[:] = state.branches["psd"].coords[0, 0][None, None, :, :]
            # state.branches["galfor"].coords[:] = state.branches["galfor"].coords[0, 0][None, None, :, :]

            from lisatools.globalfit.generatefuncs import GenerateCurrentState

            A_inj = general_info["A_inj"].copy()
            E_inj = general_info["E_inj"].copy()

            generate = GenerateCurrentState(A_inj, E_inj)

            accepted = np.zeros((ntemps, nwalkers), dtype=int)
            swaps_accepted = np.zeros((ntemps - 1,), dtype=int)
            state.log_like = np.zeros((ntemps, nwalkers))
            state.log_prior = np.zeros((ntemps, nwalkers))
            state.betas = np.ones((ntemps,))

            generated_info = generate(state, self.curr.settings_dict, include_gbs=False, include_psd=True, include_lisasens=True, include_ll=True, include_source_only_ll=True, n_gen_in=nwalkers, return_prior_val=False, fix_val_in_gen=["gb", "psd", "mbh"])
            generated_info_with_gbs = generate(state, self.curr.settings_dict, include_psd=True, include_lisasens=True, include_ll=True, include_source_only_ll=True, n_gen_in=nwalkers, return_prior_val=False, fix_val_in_gen=["gb", "psd", "mbh"])

            data = generated_info["data"]
            psd = generated_info["psd"]
            lisasens = generated_info["lisasens"]

            df = general_info["df"]
            A_going_in = np.repeat(data[0], 2, axis=0).reshape(nwalkers, 2, general_info["data_length"]).transpose(1, 0, 2)
            E_going_in = np.repeat(data[1], 2, axis=0).reshape(nwalkers, 2, general_info["data_length"]).transpose(1, 0, 2)

            A_psd_in = np.repeat(psd[0], 1, axis=0).reshape(nwalkers, 1, general_info["data_length"]).transpose(1, 0, 2)
            E_psd_in = np.repeat(psd[1], 1, axis=0).reshape(nwalkers, 1, general_info["data_length"]).transpose(1, 0, 2)

            A_lisasens_in = np.repeat(lisasens[0], 1, axis=0).reshape(nwalkers, 1, general_info["data_length"]).transpose(1, 0, 2)
            E_lisasens_in = np.repeat(lisasens[1], 1, axis=0).reshape(nwalkers, 1, general_info["data_length"]).transpose(1, 0, 2)

            gpus = [2]
            mgh = MultiGPUDataHolder(
                gpus,
                A_going_in,
                E_going_in,
                A_going_in, # store as base
                E_going_in, # store as base
                A_psd_in,
                E_psd_in,
                A_lisasens_in,
                E_lisasens_in,
                df,
                base_injections=[general_info["A_inj"], general_info["E_inj"]],
                base_psd=None,  # [psd.copy(), psd.copy()]
            )

            state.mgh = mgh

            priors = {**mbh_info["priors"], **psd_info["priors"], "gb": gb_info["priors"]}
            periodic = {**mbh_info["periodic"], "gb": gb_info["periodic"]}

            gpu_priors_in = deepcopy(priors["gb"].priors_in)
            for key, item in gpu_priors_in.items():
                item.use_cupy = True

            from eryn.prior import ProbDistContainer
            gpu_priors = {"gb": GBPriorWrap(gb_info["pe_info"]["ndim"], ProbDistContainer(gpu_priors_in, use_cupy=True))}


            from gbgpu.gbgpu import GBGPU
            gb = GBGPU(use_gpu=True)

            priors["all_models_together"] = PSDwithGBPriorWrap(
                nwalkers, 
                gb, 
                priors
            )

            waveform_kwargs = gb_info["pe_info"]["pe_waveform_kwargs"].copy()
            if "N" in waveform_kwargs:
                waveform_kwargs.pop("N")

            if state.branches["gb"].inds[0].sum() > 0:
                
                # from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR
                # L = 2.5e9
                # amp_transform = AmplitudeFromSNR(L, general_info['Tobs'], fd=general_info["fd"])

                # walker_inds = np.repeat(np.arange(nwalkers)[:, None], ntemps * nleaves_max_gb, axis=-1).reshape(nwalkers, ntemps, nleaves_max_gb).transpose(1, 0, 2)[state.branches["gb"].inds]
                
                # coords_fix = state.branches["gb"].coords[state.branches["gb"].inds]
                # coords_fix[:, 0], _ = amp_transform(coords_fix[:, 0], coords_fix[:, 1] / 1e3, psds=psd[0], walker_inds=walker_inds)
                
                # state.branches["gb"].coords[state.branches["gb"].inds, 0] = coords_fix[:, 0]

                coords_out_gb = state.branches["gb"].coords[0,
                    state.branches["gb"].inds[0]
                ]

                nleaves_max_gb = state.branches["gb"].shape[-2]

                walker_inds = np.repeat(np.arange(nwalkers)[:, None], nleaves_max_gb, axis=-1)[state.branches["gb"].inds[0]]
                
                check = priors["gb"].logpdf(coords_out_gb, psds=lisasens[0], walker_inds=walker_inds)

                if np.any(np.isinf(check)):
                    raise ValueError("Starting priors are inf.")

                coords_out_gb[:, 3] = coords_out_gb[:, 3] % (2 * np.pi)
                coords_out_gb[:, 5] = coords_out_gb[:, 5] % (1 * np.pi)
                coords_out_gb[:, 6] = coords_out_gb[:, 6] % (2 * np.pi)
                
                coords_in_in = gb_info["transform"].both_transforms(coords_out_gb)

                band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1

                walker_vals = np.tile(
                    np.arange(nwalkers), (nleaves_max_gb, 1)
                ).transpose((1, 0))[state.branches["gb"].inds[0]]

                data_index_1 = ((band_inds % 2) + 0) * nwalkers + walker_vals

                data_index = cp.asarray(data_index_1).astype(
                    cp.int32
                )

                # goes in as -h
                factors = -cp.ones_like(data_index, dtype=cp.float64)


                band_mean_f = (band_edges[1:] + band_edges[:-1]) / 2
                from gbgpu.utils.utility import get_N

                band_N_vals = cp.asarray(get_N(np.full_like(band_mean_f, 1e-30), band_mean_f, waveform_kwargs["T"], waveform_kwargs["oversample"]))

                N_vals = band_N_vals[band_inds]

                print("before global template")
                # TODO: add test to make sure that the genertor send in the general information matches this one
                gb.gpus = mgh.gpus
                gb.generate_global_template(
                    coords_in_in,
                    data_index,
                    mgh.data_list,
                    data_length=mgh.data_length,
                    factors=factors,
                    data_splits=mgh.gpu_splits,
                    N=N_vals,
                    **waveform_kwargs,
                )
                print("after global template")
                del data_index
                del factors
                cp.get_default_memory_pool().free_all_blocks()

            psd_move = PSDMove(
                gb, mgh, priors, gpu_priors, live_dangerously=True, 
                gibbs_sampling_setup=[{
                    "psd": np.ones((1, ndims["psd"]), dtype=bool),
                    "galfor": np.ones((1, ndims["galfor"]), dtype=bool)
                }]
            )

            like_mix = BasicResidualMGHLikelihood(None)

            backend = GBHDFBackend(
                self.curr.settings_dict["general"]["file_information"]["fp_main"],
                compression="gzip",
                compression_opts=9,
                comm=self.comm,
                save_plot_rank=self.results_rank
            )

            if not backend.initialized:
                backend.reset(
                    nwalkers,
                    ndims,
                    nleaves_max=nleaves_max,
                    ntemps=ntemps,
                    branch_names=branch_names,
                    nbranches=len(branch_names),
                    rj=True,
                    moves=None,
                    num_bands=len(band_edges) - 1,
                    band_edges=band_edges
                )


            # backend.grow(1, None)
            # backend.save_step(state, accepted, rj_accepted=accepted, swaps_accepted=swaps_accepted)
            # exit()
            from bbhx.waveformbuild import BBHWaveformFD

            wave_gen = BBHWaveformFD(
                **mbh_info["initialize_kwargs"]
            )

            from eryn.moves.tempering import TemperatureControl, make_ladder

            if hasattr(state, "betas_all") and state.betas_all is not None:
                    betas_all = state.betas_all
            else:
                betas_all = np.tile(make_ladder(mbh_info["pe_info"]["ndim"], ntemps=ntemps), (mbh_info["pe_info"]["nleaves_max"], 1))

            # to make the states work 
            betas = betas_all[0]
            state.betas_all = betas_all

            temperature_controls = [None for _ in range(mbh_info["pe_info"]["nleaves_max"])]
            for leaf in range(mbh_info["pe_info"]["nleaves_max"]):
                temperature_controls[leaf] = TemperatureControl(
                    mbh_info["pe_info"]["ndim"],
                    nwalkers,
                    betas=betas_all[leaf],
                    permute=False,
                    skip_swap_branches=["psd", "gb", "galfor"]
                )

            inner_moves = mbh_info["pe_info"]["inner_moves"]
            mbh_move = MBHSpecialMove(wave_gen, mgh, mbh_info["pe_info"]["num_prop_repeats"], mbh_info["transform"], priors, mbh_info["waveform_kwargs"].copy(), inner_moves, df, temperature_controls)

            ########### GB

            band_edges = gb_info["band_edges"]
            num_sub_bands = len(band_edges)
            betas_gb = gb_info["pe_info"]["betas"]

            adjust_temps = False

            if hasattr(state, "band_info"):
                band_info_check = deepcopy(state.band_info)
                adjust_temps = True
            #    del state.band_info

            band_temps = np.tile(np.asarray(betas_gb), (len(band_edges) - 1, 1))
            state.initialize_band_information(nwalkers, ntemps, band_edges, band_temps)
            if adjust_temps:
                state.band_info["band_temps"][:] = band_info_check["band_temps"][0, :]

            band_inds_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)
            N_vals_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)

            if state.branches["gb"].inds.sum() > 0:
                f_in = state.branches["gb"].coords[state.branches["gb"].inds][:, 1] / 1e3
                band_inds_in[state.branches["gb"].inds] = np.searchsorted(band_edges, f_in, side="right") - 1
                N_vals_in[state.branches["gb"].inds] = band_N_vals.get()[band_inds_in[state.branches["gb"].inds]]

            branch_supp_base_shape = (ntemps, nwalkers, nleaves_max_gb)
            state.branches["gb"].branch_supplimental = BranchSupplimental(
                {"N_vals": N_vals_in, "band_inds": band_inds_in}, base_shape=branch_supp_base_shape, copy=True
            )

            gb_kwargs = dict(
                waveform_kwargs=waveform_kwargs,
                parameter_transforms=gb_info["transform"],
                provide_betas=True,
                skip_supp_names_update=["group_move_points"],
                random_seed=general_info["random_seed"],
                use_gpu=True,
                nfriends=nwalkers,
                phase_maximize=gb_info["pe_info"]["in_model_phase_maximize"],
                ranks_needed=5,
                gpus_needed=0,
                **gb_info["pe_info"]["group_proposal_kwargs"]
            )

            fd = general_info["fd"].copy()

            gb_args = (
                gb,
                priors,
                general_info["start_freq_ind"],
                mgh.data_length,
                mgh,
                np.asarray(fd),
                band_edges,
                gpu_priors,
            )

            gb_move = GBSpecialStretchMove(
                *gb_args,
                **gb_kwargs,
            )

            # add the other
            gb_move.gb.gpus = gpus

            rj_moves_in = []

            rj_moves_in_frac = []

            gb_kwargs_rj = dict(
                waveform_kwargs=waveform_kwargs,
                parameter_transforms=gb_info["transform"],
                search=False,
                provide_betas=True,
                skip_supp_names_update=["group_move_points"],
                random_seed=general_info["random_seed"],
                use_gpu=True,
                rj_proposal_distribution=gpu_priors,
                name="rj_prior",
                use_prior_removal=gb_info["pe_info"]["use_prior_removal"],
                nfriends=nwalkers,
                phase_maximize=gb_info["pe_info"]["rj_phase_maximize"],
                ranks_needed=0,
                gpus_needed=0,
                **gb_info["pe_info"]["group_proposal_kwargs"]  # needed for it to work
            )

            gb_args_rj = (
                gb,
                priors,
                general_info["start_freq_ind"],
                mgh.data_length,
                mgh,
                np.asarray(fd),
                band_edges,
                gpu_priors,
            )

            rj_move_prior = GBSpecialStretchMove(
                *gb_args_rj,
                **gb_kwargs_rj,
            )

            rj_moves_in.append(rj_move_prior)
            rj_moves_in_frac.append(gb_info["pe_info"]["rj_prior_fraction"])

            total_frac = sum(rj_moves_in_frac)

            rj_moves = [(rj_move_i, rj_move_frac_i / total_frac) for rj_move_i, rj_move_frac_i in zip(rj_moves_in, rj_moves_in_frac)]

            for rj_move in rj_moves:
                rj_move[0].gb.gpus = gpus

            in_model_moves = [mbh_move, psd_move, gb_move]

            rank_instructions = {}
            for move in in_model_moves:
                if not isinstance(move, GlobalFitMove):
                    raise ValueError("All moves must be a subclass of GlobalFitMove.")
                move.comm = self.comm
                if move.ranks_needed > 0:
                    tmp_ranks = []
                    for _ in range(move.ranks_needed):
                        try:
                            tmp_ranks.append(self.ranks_to_give.pop())
                        except IndexError:
                            raise ValueError("Not enough MPI ranks to give.")
                    self.used_ranks += tmp_ranks
                    move.assign_ranks(tmp_ranks)
                    for rank in tmp_ranks:
                        rank_instructions[rank] = move.get_rank_function()
            breakpoint()

            # stop unneeded processes
            for rank in self.all_ranks:
                if rank in self.used_ranks:
                    continue
                self.comm.send("stop", dest=rank)
                
            for rank, function in rank_instructions.items():
                self.comm.send({"rank": rank, "function": function}, dest=rank)
            
            breakpoint()
            # permute False is there for the PSD sampling for now
            sampler_mix = EnsembleSampler(
                nwalkers,
                ndims,  # assumes ndim_max
                like_mix,
                priors,
                tempering_kwargs={"betas": betas, "permute": False, "skip_swap_branches": ["mbh", "gb"]},
                nbranches=len(branch_names),
                nleaves_max=nleaves_max,
                nleaves_min=nleaves_min,
                moves=in_model_moves,
                rj_moves=rj_moves,
                kwargs=None,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
                backend=backend,
                vectorize=True,
                periodic=periodic,  # TODO: add periodic to proposals
                branch_names=branch_names,
                # update_fn=update,  # stop_converge_mix,
                # update_iterations=gb_info["pe_info"]["update_iterations"],
                provide_groups=True,
                provide_supplimental=True,
                track_moves=False,
                # stopping_fn=stopping_fn,
                # stopping_iterations=stopping_iterations,
            )

            state.log_prior = sampler_mix.compute_log_prior(state.branches_coords, inds=state.branches_inds, supps=supps)
            state.log_like = psd_move.compute_log_like(state.branches_coords, logp=state.log_prior, inds=state.branches_inds, supps=supps)[0]

            sampler_mix.run_mcmc(state, 10, progress=True, store=False)

        elif self.rank == self.results_rank:
            print("RESULTS RANK")
        else:
            info = self.comm.recv(source=self.main_rank)
            if isinstance(info, dict):
                launch_rank = info["rank"]
                assert launch_rank == self.rank
                launch_function = info["function"]
                launch_function(self.comm)

            print(f"Process {self.rank} finished.")
            

if __name__ == "__main__":
    curr = CurrentInfoGlobalFit(get_global_fit_settings())
    gf = GlobalFit(curr, MPI.COMM_WORLD)
    gf.run_global_fit()