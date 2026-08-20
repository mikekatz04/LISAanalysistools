from __future__ import annotations

import logging
import multiprocessing as mp
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

try:
    import cupy as xp
    GPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    import numpy as xp
    GPU_AVAILABLE = False

from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_N
from lisatools.utils.constants import YRSID_SI

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from eryn.utils import TransformContainer
    from ..sensitivity import XYZSensitivityBackend, SensitivityMatrixBase
    from .hdfbackend import GFHDFBackend


def gather_gb_samples(
    fd: FloatArray,
    transform_fn: TransformContainer | None,
    gb: GBGPU,
    waveform_kwargs: Dict[str, Any],
    band_edges: FloatArray,
    band_N_vals: IntArray,
    reader: GFHDFBackend,
    sens_mat: XYZSensitivityBackend | SensitivityMatrixBase,
    gb_samples: Optional[FloatArray] = None,
    gb_inds: Optional[NDArray[np.bool_]] = None,
    num_compare_samples: int = 1,
    samples_keep: int = 1,
    thin_by: int = 1,
    snr_lim_first_cut: float = 6.0,
    snr_lim_second_cut: float = 5.0,
    overlap_lim: float = 0.5,
    snr_diff_lim: float = 20.0,
    use_representative: bool = False,
) -> List[FloatArray]:
    """
    Cluster GB samples using strict frequency/SNR bounds and flexible consolidation.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available, cannot gather GB samples.")

    assert gb.gpus is not None, "GBGPU instance must have a GPU attribute."
    getattr(gb, "backend").set_cuda_device(gb.gpus[0])

    fake_data = [
        xp.zeros((len(waveform_kwargs["tdi_channel_setup"]), fd.shape[0]), dtype=xp.complex128)
    ]
    psd_in = [xp.asarray(sens_mat.invC.copy())]

    if gb_samples is None or gb_inds is None:
        assert isinstance(reader.iteration, int)
        gb_chain = reader.get_chain(
            branch_names=["gb"], 
            temp_index=0, 
            discard=reader.iteration - samples_keep, 
            thin=thin_by
        )
        assert gb_chain is not None
        gb_samples = gb_chain["gb"]
        
        gb_inds_reader = reader.get_inds(
            branch_names=["gb"], 
            temp_index=0, 
            discard=reader.iteration - samples_keep, 
            thin=thin_by
        )
        assert gb_inds_reader is not None
        gb_inds = gb_inds_reader["gb"] # type: ignore

    assert isinstance(gb_samples, np.ndarray) and isinstance(gb_inds, np.ndarray)
    gb_samples = gb_samples.reshape(-1, gb_samples.shape[-2], gb_samples.shape[-1])
    gb_inds = gb_inds.reshape(-1, gb_inds.shape[-1])

    test_bins_for_snr = gb_samples[gb_inds]
    if transform_fn is not None:
        test_bins_for_snr_in = transform_fn.both_transforms(test_bins_for_snr)
    else:
        test_bins_for_snr_in = test_bins_for_snr
    setattr(gb, "d_d", 0.0)

    waveform_kwargs_no_n = waveform_kwargs.copy()
    waveform_kwargs_no_n.pop("N", None)

    _ = gb.get_ll(test_bins_for_snr_in, fake_data, psd_in, **waveform_kwargs_no_n)

    optimal_snr = gb.h_h.real ** (1 / 2)
    gb_snrs = np.full(gb_inds.shape, -1e10)
    gb_snrs[gb_inds] = optimal_snr.get()
    gb_inds_tmp = gb_inds.copy()

    keep_groups: List[FloatArray] = []
    
    actual_num_compare = min(num_compare_samples, len(gb_samples))
    random_samples = np.random.choice(np.arange(len(gb_samples)), len(gb_samples) - 1, replace=False)
    
    for samp_i in tqdm(range(actual_num_compare), desc="Comparing samples"):
        assert gb_samples
        first_sample = gb_samples[random_samples[samp_i]].reshape(-1, 8)
        first_sample_snrs = gb_snrs[random_samples[samp_i]].flatten()
        inds_keep_i = np.delete(np.arange(gb_samples.shape[0]), random_samples[: samp_i + 1]) #? why up to samp_i + 1 and not only keep out samp_i?
        
        gb_samples_in = gb_samples[inds_keep_i]
        gb_inds_in = gb_inds_tmp[inds_keep_i]
        gb_snrs_in = gb_snrs[inds_keep_i]

        keep_map, binaries_for_test, binaries_base_sample = [], [], []
        num_so_far = 0
        keep_going_in = []

        for i, binary in enumerate(first_sample):
            if first_sample_snrs[i] < snr_lim_first_cut: # we do not match low SNR samples to higher SNR samples, but do allow for the reverse
                continue
            freq_dist = np.abs(binary[1] - gb_samples_in[:, :, 1])
            snr_dist = np.abs(first_sample_snrs[i] - gb_snrs_in)

            keep_going_in.append(i)
            keep_i = np.where( #? in general we could improve this
                (freq_dist < 1e-4) # TODO double this check criteria
                & (snr_dist < snr_diff_lim)
                & (gb_snrs_in >= snr_lim_second_cut)
                & gb_inds_in
            )

            keep_map.append([num_so_far + np.arange(len(keep_i[0])), keep_i])
            binaries_for_test.append(gb_samples_in[keep_i])
            binaries_base_sample.append(np.tile(binary, (len(keep_i[0]), 1)))
            num_so_far += len(keep_i[0])
        
        if not binaries_for_test:
            continue

        binaries_for_test = np.concatenate(binaries_for_test, axis=0)
        binaries_base_sample = np.concatenate(binaries_base_sample, axis=0)
        
        band_inds = np.searchsorted(band_edges.get(), binaries_for_test[:, 1] / 1e3, side="right") - 1
        n_vals = band_N_vals[band_inds]

        batch_size = int(1e7)
        fake_data = [xp.zeros((len(waveform_kwargs["tdi_channel_setup"]), fd.shape[0]), dtype=xp.complex128)]
        psd_in = [xp.asarray(sens_mat.invC.copy())]

        inds_split = np.arange(0, binaries_for_test.shape[0] + batch_size, batch_size)
        overlap = np.zeros(binaries_for_test.shape[0])
        
        for start_ind, end_ind in zip(inds_split[:-1], inds_split[1:]):
            waveform_kwargs["N"] = xp.asarray(n_vals[start_ind:end_ind])
            
            if transform_fn is not None:
                bin_test_batch = transform_fn.both_transforms(binaries_for_test[start_ind:end_ind])
                bin_base_batch = transform_fn.both_transforms(binaries_base_sample[start_ind:end_ind])
            else:
                bin_test_batch = binaries_for_test[start_ind:end_ind]
                bin_base_batch = binaries_base_sample[start_ind:end_ind]

            _ = gb.swap_likelihood_difference(
                bin_test_batch, bin_base_batch, fake_data, psd_in, phase_marginalize=True, **waveform_kwargs
            )
            overlap[start_ind:end_ind] = (
                gb.add_remove.real / np.sqrt(gb.add_add.real * gb.remove_remove.real)
            ).get()

        for i, (keep_inds, keep_map_back) in enumerate(keep_map):
            if len(keep_inds) == 0:
                continue

            overlap_i = overlap[keep_inds]
            mismatch = np.abs(1.0 - overlap_i)
            indicator = keep_map_back[0] * 1e6 + mismatch
            
            group_sort = np.argsort(indicator)
            indicator = indicator[group_sort]
            keep_inds = keep_inds[group_sort]
            keep_map_back = (keep_map_back[0][group_sort], keep_map_back[1][group_sort])
            mismatch = mismatch[group_sort]

            _, uni_sample_index = np.unique(keep_map_back[0], return_index=True)
            group_test = mismatch[uni_sample_index] < np.abs(1.0 - overlap_lim)
            keep_group_test = uni_sample_index[group_test]
            
            if len(keep_group_test) == 0:
                continue

            in_here = keep_going_in[i]
            ind1 = inds_keep_i[keep_map_back[0][keep_group_test]]
            ind2 = keep_map_back[1][keep_group_test]

            if np.any(~gb_inds_tmp[ind1, ind2]):
                keep_group_test = keep_group_test[gb_inds_tmp[ind1, ind2]]
                if len(keep_group_test) == 0:
                    continue
                ind1 = inds_keep_i[keep_map_back[0][keep_group_test]]
                ind2 = keep_map_back[1][keep_group_test]

            gb_inds_tmp[ind1, ind2] = False
            group = np.concatenate(
                [first_sample[in_here][None, :], gb_samples_in[keep_map_back][keep_group_test]], axis=0
            )
            
            if len(group) <= 1:
                logger.warning("Group formulated with <= 1 element; skipping.")
                continue

            keep_groups.append(group)

    # Consolidation Loop
    current_number = len(keep_groups)
    final_number = -1
    logger.info(f"Initial number of groups: {current_number}")
    
    while current_number != final_number:
        current_number = len(keep_groups)
        
        group_min_f = np.asarray([g[:, 1].min() for g in keep_groups])
        group_max_f = np.asarray([g[:, 1].max() for g in keep_groups])

        diffs_min = np.abs(
            np.array([
                group_max_f[:, None] - group_max_f[None, :],
                group_min_f[:, None] - group_min_f[None, :],
                group_min_f[:, None] - group_max_f[None, :],
            ])
        )

        # Ignore self-comparisons by forcing them to infinity
        for i in range(3):
            diffs_min[i, np.arange(len(keep_groups)), np.arange(len(keep_groups))] = 1e100

        _inds1 = diffs_min.argmin(axis=-1)
        inds2 = np.take_along_axis(diffs_min, _inds1[:, :, None], axis=-1)[:, :, 0].argmin(axis=0)
        inds1 = _inds1.T[(np.arange(len(inds2)), inds2)]

        if use_representative:
            base_bins = np.stack([
                group[np.argsort(group[:, 1])[
                    len(group) // 2
                ]] for g in keep_groups
            ], axis=0)
            
            test_bins = base_bins[inds1]
            new_group_map = np.arange(len(keep_groups))
            old_group_map = inds1.copy()
        else:
            base_bins_list, test_bins_list, new_group_map_list, old_group_map_list = [], [], [], []
            for i, (group, closest_group) in enumerate(zip(keep_groups, inds1)):
                _b, _t = [tmp.flatten() for tmp in np.meshgrid(np.arange(len(group)), np.arange(len(keep_groups[closest_group])))]
                base_bins_list.append(group[_b])
                test_bins_list.append(keep_groups[closest_group][_t])
                new_group_map_list.append(np.full_like(_t, i))
                old_group_map_list.append(np.full_like(_t, closest_group))
            
            base_bins = np.concatenate(base_bins_list, axis=0)
            test_bins = np.concatenate(test_bins_list, axis=0)
            new_group_map = np.concatenate(new_group_map_list, axis=0)
            old_group_map = np.concatenate(old_group_map_list, axis=0)

        band_inds = np.searchsorted(band_edges.get(), base_bins[:, 1] / 1e3, side="right") - 1
        n_vals = band_N_vals[band_inds]

        batch_size = int(1e7)
        fake_data = [xp.zeros((len(waveform_kwargs["tdi_channel_setup"]), fd.shape[0]), dtype=xp.complex128)]
        psd_in = [xp.asarray(sens_mat.invC.copy())]

        inds_split = np.arange(0, base_bins.shape[0] + batch_size, batch_size)
        overlap = np.zeros(base_bins.shape[0])
        snr1 = np.zeros(base_bins.shape[0])
        snr2 = np.zeros(base_bins.shape[0])
        
        for start_ind, end_ind in zip(inds_split[:-1], inds_split[1:]):
            waveform_kwargs["N"] = xp.asarray(n_vals[start_ind:end_ind])
            
            if transform_fn is not None:
                base_bins_in = transform_fn.both_transforms(base_bins[start_ind:end_ind])
                test_bins_in = transform_fn.both_transforms(test_bins[start_ind:end_ind])
            else:
                base_bins_in = base_bins[start_ind:end_ind]
                test_bins_in = test_bins[start_ind:end_ind]

            _ = gb.swap_likelihood_difference(
                base_bins_in, 
                test_bins_in, 
                fake_data, 
                psd_in, 
                phase_marginalize=True, 
                **waveform_kwargs
            )
            overlap[start_ind:end_ind] = (
                gb.add_remove.real / np.sqrt(
                    gb.add_add.real * gb.remove_remove.real
                )
            ).get()
            snr1[start_ind:end_ind] = np.sqrt(gb.add_add.real).get()
            snr2[start_ind:end_ind] = np.sqrt(gb.remove_remove.real).get()
            
        keep = (overlap > overlap_lim) & (np.abs(snr2 - snr1) < snr_diff_lim)
        
        overlap_keep = overlap[keep]
        new_group_keep = new_group_map[keep]
        old_group_keep = old_group_map[keep]
        
        old_group_update = np.argsort(old_group_keep * int(1e3) + overlap_keep)[::-1]
        _, uni_index = np.unique(old_group_keep[old_group_update], return_index=True)
        
        new_group_final = new_group_keep[old_group_update[uni_index]]
        old_group_final = old_group_keep[old_group_update[uni_index]]

        new_groups_2 = -np.ones(len(keep_groups), dtype=int)
        for new_group, old_group in zip(new_group_final, old_group_final):
            new_groups_2[old_group] = new_group

        new_groups_3 = new_groups_2.copy()
        for i in range(len(new_groups_3)):
            n_grp = new_groups_3[i]
            if n_grp == -1:
                continue
            keep_groups[n_grp] = np.concatenate([keep_groups[n_grp], keep_groups[i]])
            new_groups_3[new_groups_3 == i] = n_grp
            new_groups_3[i] = -1
            keep_groups[i] = None

        keep_groups = [tmp for tmp in keep_groups if tmp is not None]
        final_number = len(keep_groups)
        
    logger.info(f"Final number of groups after consolidation: {len(keep_groups)}")
    logger.info(f"Number of samples in each group: {[len(g) for g in keep_groups]}")
    return keep_groups