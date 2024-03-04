import numpy as np
from copy import deepcopy
import warnings

from lisatools.sensitivity import get_sensitivity
from lisatools.globalfit.hdfbackend import HDFBackend as GBHDFBackend

from bbhx.waveformbuild import BBHWaveformFD

from gbgpu.gbgpu import GBGPU

from eryn.backends import HDFBackend

import os
import pickle

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError) as e:
    pass


class GetPSDModel:

    def __init__(self, psd_kwargs):
        self.psd_kwargs = psd_kwargs

    def __call__(self, psd_info, general_info, only_max_ll=False, n_gen_in=None, return_lisasens=False, return_prior_val=False):

        if "use_gpu" in self.psd_kwargs and self.psd_kwargs["use_gpu"]:
            xp = cp
            use_gpu = True
        else:
            xp = np
            use_gpu = False

        if only_max_ll:
            best = psd_info["cc_ll"].argmax()
            psd_params_A = psd_info["cc_A"][best:best + 1]
            psd_params_E = psd_info["cc_E"][best:best + 1]
            psd_foreground_params = psd_info["cc_foreground_params"][best:best + 1]

        else:
            num_psd_walkers = psd_info["cc_A"].shape[0]

            inds = np.arange(num_psd_walkers)

            # adjust if looking for less
            if n_gen_in is not None and isinstance(n_gen_in, int):
                if num_psd_walkers > n_gen_in:
                    inds = np.random.choice(np.arange(num_psd_walkers), n_gen_in, replace=False)
                
            psd_params_A = psd_info["cc_A"][inds].copy()
            psd_params_E = psd_info["cc_E"][inds].copy()
            psd_foreground_params = psd_info["cc_foreground_params"][inds].copy()

        num_psds = len(psd_params_A)
        num_freqs = len(general_info["fd"])
        A_psd = np.zeros((num_psds, num_freqs))
        E_psd = np.zeros((num_psds, num_freqs))
        freqs = xp.asarray(general_info["fd"])
        psd_kwargs_in = self.psd_kwargs.copy()
        if return_lisasens:
            psd_kwargs_in["sens_fn"] = "lisasens"

        for i, (A_params, E_params, foreground_params) in enumerate(zip(psd_params_A, psd_params_E, psd_foreground_params)):
            A_tmp1 = get_sensitivity(freqs, model=A_params, foreground_params=foreground_params, **psd_kwargs_in)
            E_tmp1 = get_sensitivity(freqs, model=E_params, foreground_params=foreground_params, **psd_kwargs_in)

            if use_gpu:
                A_tmp2 = A_tmp1.get()
                E_tmp2 = E_tmp1.get()
            else:
                A_tmp2 = A_tmp1
                E_tmp2 = E_tmp1

            A_psd[i] = A_tmp2
            E_psd[i] = E_tmp2

        del freqs, A_tmp1, E_tmp1
        if use_gpu:
            xp.get_default_memory_pool().free_all_blocks()

        A_psd[:, 0] = A_psd[:, 1]
        E_psd[:, 0] = E_psd[:, 1]

        return_tuple = (A_psd, E_psd)
        if return_prior_val:
            psd_prior = psd_info["priors"]["psd"].logpdf(np.concatenate([A_params, E_params], axis=-1))
            galfor_prior = psd_info["priors"]["galfor"].logpdf(psd_foreground_params)
            all_prior = psd_prior + galfor_prior
            return_tuple += (all_prior,)
        else:
            return_tuple += (None,)
        return return_tuple


class GetMBHTemplates:

    def __init__(self, initialization_kwargs, runtime_kwargs):
        self.initialization_kwargs = initialization_kwargs
        self.runtime_kwargs = runtime_kwargs

    def __call__(self, mbh_info, general_info, only_max_ll=False, n_gen_in=None, return_prior_val=False):

        if "use_gpu" in self.initialization_kwargs and self.initialization_kwargs["use_gpu"]:
            xp = cp
            use_gpu = True
        else:
            xp = np
            use_gpu = False

        mbh_gen = BBHWaveformFD(**self.initialization_kwargs)

        num_mbh = len(mbh_info["cc_params"])
        num_freqs = len(general_info["fd"])
    
        if only_max_ll:
            best = mbh_info["cc_ll"].argmax()
            mbh_params = mbh_info["cc_params"][best:best + 1]

        else:
            num_mbh_walkers = mbh_info["cc_params"].shape[0]
            
            inds = np.arange(num_mbh_walkers)

            # adjust if looking for less
            if n_gen_in is not None and isinstance(n_gen_in, int):
                if num_mbh_walkers > n_gen_in:
                    inds = np.random.choice(np.arange(num_mbh_walkers), n_gen_in, replace=False)
                
            mbh_params = mbh_info["cc_params"][inds].copy()

        freqs = xp.asarray(general_info["fd"])

        out = np.zeros((mbh_params.shape[0], 3, general_info["fd"].shape[0]), dtype=complex)

        for i in range(mbh_params.shape[0]):
            for leaf in range(mbh_params.shape[1]):
                mbh_params_in = mbh_info["transform"].both_transforms(mbh_params[i, leaf:leaf + 1].reshape(-1, mbh_info["pe_info"]["ndim"]))
            
                AET = mbh_gen(*mbh_params_in.T, freqs=freqs, fill=True, direct=False, **self.runtime_kwargs)
            
                tmp1 = AET.reshape((3, -1))
                
                if use_gpu:
                    tmp2 = tmp1.get()
                else:
                    tmp2 = tmp1

                out[i] += tmp2

        del mbh_gen, freqs, AET, tmp2

        if use_gpu:
            xp.get_default_memory_pool().free_all_blocks()
        
        return_tuple = (out[:, 0], out[:, 1])
        # TODO: delete GPU data if needed?

        if return_prior_val:
            prior_val = mbh_info["priors"]["mbh"].logpdf(mbh_params.reshape(-1, mbh_params.shape[-1])).reshape(mbh_params.shape[:-1]).sum(axis=-1)
            return_tuple += (prior_val,)
        else:
            return_tuple += (None,)
        return return_tuple

class GetGBTemplates:

    def __init__(self, initialization_kwargs, runtime_kwargs):
        self.initialization_kwargs = initialization_kwargs
        self.runtime_kwargs = runtime_kwargs

    def __call__(self, gb_info, general_info, only_max_ll=False, n_gen_in=None, return_prior_val=False, lisasens=None):

        gb_gen = GBGPU(**self.initialization_kwargs)

        use_gpu = "use_gpu" in self.initialization_kwargs and self.initialization_kwargs["use_gpu"]

        if use_gpu:
            xp = cp
        else:
            xp = np

        if only_max_ll:
            best = gb_info["cc_ll"].argmax()
            gb_params = gb_info["cc_params"][best:best + 1]
            gb_inds = gb_info["cc_inds"][best:best + 1]

        else:
            gb_params = gb_info["cc_params"].copy()
            gb_inds = gb_info["cc_inds"].copy()

        num_gb = len(gb_params)
        num_freqs = len(general_info["fd"])

        gb_params_flat = gb_params[gb_inds]

        group_index = xp.asarray(np.repeat(np.arange(num_gb)[:, None], repeats=gb_params.shape[1], axis=-1)[gb_inds].astype(np.int32))

        gb_params_in = gb_info["transform"].both_transforms(gb_params_flat)
        
        templates_in_tmp = xp.zeros((num_gb, 2, num_freqs), dtype=complex)

        gb_gen.generate_global_template(gb_params_in, group_index, templates_in_tmp, **self.runtime_kwargs)
        
        if use_gpu:
            templates_in = templates_in_tmp.get()

        else:
            templates_in = templates_in_tmp

        del gb_gen, templates_in_tmp, group_index

        if use_gpu:
            xp.get_default_memory_pool().free_all_blocks()
        
        # TODO: delete GPU data if needed?

        return_tuple = (templates_in[:, 0], templates_in[:, 1])

        if return_prior_val:
            self.gb_inds_store = gb_inds.copy()
            self.gb_params_store = gb_params.copy()
        return return_tuple

    def get_gb_prior(self, gb_info, lisasens, sampler_walker_inds):
        nwalkers, nleaves_max = self.gb_inds_store.shape
        gb_params_flat = self.gb_params_store[sampler_walker_inds][self.gb_inds_store[sampler_walker_inds]]
        each_source_walker_inds = np.repeat(sampler_walker_inds[:, None], repeats=nleaves_max, axis=-1)[self.gb_inds_store[sampler_walker_inds]]
        tmp = np.zeros_like(self.gb_inds_store[sampler_walker_inds], dtype=float)

        tmp[self.gb_inds_store[sampler_walker_inds]] = gb_info["priors"].logpdf(gb_params_flat, psds=lisasens, walker_inds=each_source_walker_inds)
        prior_out = tmp.sum(axis=-1)
        return prior_out

def get_ll_source(data, psd, df):
    inner_here = -1. / 2. * df * 4 * np.sum(np.asarray([data_i.conj() * data_i / psd_i for data_i, psd_i in zip(data, psd)]).transpose(1, 0, 2), axis=(1, 2)).real
    return inner_here

def get_psd_val(psd):
    psd_term_here = np.sum(np.log(np.asarray(psd).transpose(1, 0, 2)), axis=(1, 2))
    return psd_term_here

def get_ll(data, psd, df, return_source_only_ll=False):
    ll_source = get_ll_source(data, psd, df)
    psd_val = get_psd_val(psd)
    ll_total = ll_source - psd_val
    if return_source_only_ll:
        return ll_total, ll_source
    else:
        return ll_total

class GenerateCurrentState:
    def __init__(self, A_inj, E_inj):
        self.A_inj, self.E_inj = A_inj, E_inj

    def __call__(self, general_info, include_mbhs=True, include_gbs=True, include_psd=True, include_lisasens=True, only_max_ll=False, n_gen_in=None, include_ll=False, include_source_only_ll=False, return_prior_val=False, fix_val_in_gen=None):
        
        info_dict = {}
        n_gen_check_it = []
        if include_mbhs and "cc_params" in general_info.mbh_info:
            A_mbh, E_mbh, prior_vals = general_info.mbh_info["get_templates"](general_info.mbh_info, general_info.general_info, only_max_ll=only_max_ll, n_gen_in=n_gen_in, return_prior_val=return_prior_val)
            n_mbh = A_mbh.shape[0]
            info_dict["mbh"] = {"n": n_mbh, "A": A_mbh, "E": E_mbh, "prior": prior_vals}
            n_gen_check_it.append(n_mbh)
        elif include_mbhs and "cc_params" not in general_info.mbh_info:
            include_mbhs = False
            
        if include_gbs and "cc_params" in general_info.gb_info:
            A_gb, E_gb = general_info.gb_info["get_templates"](general_info.gb_info, general_info.general_info, only_max_ll=only_max_ll, return_prior_val=True)
            n_gb = A_gb.shape[0]
            info_dict["gb"] = {"n": n_gb, "A": A_gb, "E": E_gb, "prior": None}
            n_gen_check_it.append(n_gb)

        elif include_gbs and "cc_params" not in general_info.gb_info:
            include_gbs = False

        if include_psd and "cc_A" in general_info.psd_info:
            A_psd, E_psd, psd_prior_val = general_info.psd_info["get_psd"](general_info.psd_info, general_info.general_info, only_max_ll=only_max_ll, return_lisasens=False, return_prior_val=return_prior_val)
            n_psd = A_psd.shape[0]
            info_dict["psd"] = {"n": n_psd, "A": A_psd, "E": E_psd, "prior": psd_prior_val}
            n_gen_check_it.append(n_psd)
        elif include_psd and "cc_A" not in general_info.psd_info:
            include_psd = False

        if include_lisasens and "cc_A" in general_info.psd_info:
            A_lisasens, E_lisasens, _ = general_info.psd_info["get_psd"](general_info.psd_info, general_info.general_info, only_max_ll=only_max_ll, return_lisasens=True, return_prior_val=False)
            n_lisasens = A_lisasens.shape[0]
            info_dict["lisasens"] = {"n": n_lisasens, "A": A_lisasens, "E": E_lisasens, "prior": None}
            n_gen_check_it.append(n_lisasens)
        elif include_lisasens and "cc_A" not in general_info.psd_info:
            include_lisasens = False

        if n_gen_in is None and len(n_gen_check_it) > 0:
            n_gen = np.max(n_gen_check_it)

        elif isinstance(n_gen_in, int):
            n_gen = n_gen_in

        elif isinstance(n_gen_in, str):
            if n_gen_in == "mbh":
                n_gen = n_mbh

            elif n_gen_in == "psd":
                n_gen = n_psd

            elif n_gen_in == "gb":
                n_gen = n_gb

            else:
                raise ValueError("n_gen_in must be None, 'mbh', 'psd', or 'gb'.")

        else:
            breakpoint()
            raise ValueError("n_gen_in must be None, string, or integer.")
        
        for which in info_dict.keys():
            if which != "lisasens":
                n_gen_check, A, E, prior_vals = info_dict[which]["n"], info_dict[which]["A"], info_dict[which]["E"], info_dict[which]["prior"]

                if fix_val_in_gen is None or (fix_val_in_gen is not None and which not in fix_val_in_gen):
                    # less available than needed
                    # randomly pick with replacement
                    inds_keep = np.random.choice(np.arange(n_gen_check), n_gen, replace=True)
                    info_dict[which]["walker_inds"] = inds_keep.copy()

                # rest is in fix_val_in_gen
                elif n_gen_check == n_gen:
                    info_dict[which]["walker_inds"] = np.arange(n_gen)
                    continue
                elif n_gen_check > n_gen:
                    # more available than needed
                    # randomly pick
                    inds_keep = np.random.choice(np.arange(n_gen_check), n_gen, replace=False)
                    info_dict[which]["walker_inds"] = inds_keep.copy()

                elif n_gen_check < n_gen:
                    if less_fill_style == "better_marg":
                        # not enough
                        # check how many copies needed
                        copies = n_gen // n_gen_check
                        inds_keep = np.arange(n_gen_check)
                        n_gen_curr = len(inds_keep)
                        while n_gen_curr < n_gen:
                            if n_gen_curr + n_gen_check < n_gen:
                                inds_keep = np.concatenate([inds_keep, np.arange(n_gen_check)])
                                
                            else:
                                # randomly choose remaining additions
                                inds_keep = np.concatenate([inds_keep, np.random.choice(np.arange(n_gen_check), n_gen - n_gen_curr, replace=False)])
                    elif less_fill_style == "random":
                        # less available than needed
                        # randomly pick with replacement
                        inds_keep = np.random.choice(np.arange(n_gen_check), n_gen, replace=True)
                        info_dict[which]["walker_inds"] = inds_keep.copy()

                    else:
                        raise ValueError("less_fill_style must be in ['random', 'better_marg']")

                    n_gen_curr = len(inds_keep)

                    info_dict[which]["walker_inds"] = inds_keep.copy()
            
            else:
                # ensure lisasens matches psd
                inds_keep = info_dict["psd"]["walker_inds"].copy()
                info_dict[which]["walker_inds"] = inds_keep.copy()
                n_gen_check, A, E, prior_vals = info_dict[which]["n"], info_dict[which]["A"], info_dict[which]["E"], info_dict[which]["prior"]
                if n_gen_check == n_gen:
                    continue

            A = A[inds_keep]
            E = E[inds_keep]

            if prior_vals is not None:
                prior_vals = prior_vals[inds_keep]

            info_dict[which]["A"] = A
            info_dict[which]["E"] = E
            info_dict[which]["prior"] = prior_vals

        data_A = np.tile(self.A_inj, (n_gen, 1))
        data_E = np.tile(self.E_inj, (n_gen, 1))

        if include_mbhs:
            data_A -= info_dict["mbh"]["A"]
            data_E -= info_dict["mbh"]["E"]

        if include_gbs:
            data_A -= info_dict["gb"]["A"]
            data_E -= info_dict["gb"]["E"]

            # calculate prior
            if return_prior_val:
                if "lisasens" in info_dict and "A" in info_dict["lisasens"] and info_dict["lisasens"]["A"] is not None and "walker_inds" in info_dict["gb"]:
                    info_dict["gb"]["prior"] = general_info.gb_info["get_templates"].get_gb_prior(general_info.gb_info, info_dict["lisasens"]["A"], info_dict["gb"]["walker_inds"])

        if only_max_ll:
            data_A = data_A.squeeze()
            data_E = data_E.squeeze()
            
        output = {
            "data": [data_A, data_E],
            "psd": None,
            "lisasens": None,
            "ll": None,
            "ll_source": None
        }

        if include_psd:
            output["psd"] = [info_dict["psd"]["A"], info_dict["psd"]["E"]]
            
        if include_lisasens:
            output["lisasens"] = [info_dict["lisasens"]["A"], info_dict["lisasens"]["E"]]

        if return_prior_val:
            total_prior_val = np.zeros(data_A.shape[0])
        for name in ["psd", "mbh", "gb", "lisasens"]:
            if return_prior_val:
                if name in info_dict and "prior" in info_dict[name] and info_dict[name]["prior"] is not None:
                    output[f"{name}_prior_vals"] = info_dict[name]["prior"].copy()
                    total_prior_val += info_dict[name]["prior"].copy()
            if name in info_dict and "walker_inds" in info_dict[name]:
                output[f"{name}_inds"] = info_dict[name]["walker_inds"].copy()

        if return_prior_val:
            output["total_prior_val"] = total_prior_val
        if include_ll or include_source_only_ll:
            if not include_psd:
                warnings.UserWarning("If requesting ll and/or inner product, include_psd must be True.")
            else:
                if include_source_only_ll and not include_ll:
                    output["ll"] = get_ll_source(output["data"], output["psd"], general_info.general_info["df"])
                elif not include_source_only_ll and include_ll:
                    output["ll"] = get_ll(output["data"], output["psd"], general_info.general_info["df"], return_source_only_ll=False)
                else:
                    # get both
                    output["ll"], output["ll_source"] = get_ll(output["data"], output["psd"], general_info.general_info["df"], return_source_only_ll=True)

        if only_max_ll:
            output["data"] = [output["data"][0].squeeze(), output["data"][1].squeeze()]
            if include_psd:
                output["psd"] = [output["psd"][0].squeeze(), output["psd"][1].squeeze()]
            if include_lisasens:
                output["lisasens"] = [output["lisasens"][0].squeeze(), output["lisasens"][1].squeeze()]
            
        return output

