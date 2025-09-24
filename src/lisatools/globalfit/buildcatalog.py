import time
import cupy as xp
from gbgpu.utils.utility import get_N
from gbgpu.gbgpu import GBGPU
import numpy as np
from gbgpu.utils.constants import *
from datetime import datetime
import pandas as pd
import os

from bbhx.utils.transform import LISA_to_SSB
from .gathergalaxy import gather_gb_samples_cat


def build_gb_catalog(current_info, gpu, **kwargs):

    if gpu is None:
        raise ValueError("When building a gb catalog, must add a GPU index. `gpu` is currently None.")

    xp.cuda.runtime.setDevice(gpu)
    gb_reader = current_info.gb_info["reader"]

    generated_info = current_info.get_data_psd(include_gbs=False, include_mbhs=False, include_lisasens=False, only_max_ll=True)  # , include_ll=True, include_source_only_ll=True)
    psd = xp.asarray(generated_info["psd"])
    output_information = gather_gb_samples_cat(current_info, gb_reader, psd, gpu, **kwargs)
    
    info_together = np.concatenate([output_information.groups[:, None], output_information.confidence[:, None], output_information.snr[:, None], output_information.params], axis=1)
    
    # keep only
    info_together = info_together[output_information.confidence >= 0.25]
    kept_groups = np.unique(output_information.groups[output_information.confidence >= 0.25])

    median_inds_out = np.zeros(kept_groups.shape[0], dtype=int)
    for i, group in enumerate(kept_groups):
        params_group = output_information.get_group(group)
        median_ind = np.argsort(params_group[:, 1])[int(len(params_group) / 2)]
        tmp = np.where(info_together[:, 4] == params_group[median_ind, 1])[0][0]  # 4 should be frequency
        median_inds_out[i] = tmp  # tmp is a reference ot order in info_together after removed groups for low confidence

    keys = [
        "Group ID",
        "Confidence",
        "SNR",
        "Amplitude",
        "Frequency",
        "Frequency Derivative",
        "Initial Phase",
        "cosinc",
        "Polarization",
        "Ecliptic Longitude",
        "sinlat",
    ]

    df = pd.DataFrame({key: item for key, item in zip(keys, info_together.T)})

    df["Ecliptic Latitude"] = np.arcsin(df["sinlat"])
    df["Frequency"] = df["Frequency"] / 1e3
    df["coslat"] = np.cos(np.pi / 2. - df["Ecliptic Latitude"])
    df["Inclination"] = np.arccos(df["cosinc"])
    df["Parent"] = [None for _ in range(df.shape[0])]
    year_val = datetime.now().year
    month_val = datetime.now().month
    day_val = datetime.now().day
    
    df_main = df.iloc[median_inds_out]

    name_index = [f"EREBOR_GB_{year_val}_{month_val}_{day_val}_{i:08d}" for i in range(df_main.shape[0])]
    
    chain_file_index = 0
    store = 500
    chain_file_list = []
    for i in range(len(name_index)):
        chain_file_list.append(current_info.general_info["file_information"]["gb_all_chain_file"].split("/")[-1][:-3] + f"_{chain_file_index:04d}.h5")
        
        if (i + 1) %store == 0:
            chain_file_index += 1

    df_main["chain file"] = chain_file_list

    df_main.index = name_index
    df_main.to_hdf(current_info.general_info["file_information"]["gb_main_chain_file"], "detections", mode="w", complevel=9)
    
    dirname = os.path.dirname(current_info.general_info["file_information"]["gb_main_chain_file"])
    if dirname[-1] != "/":
        dirname += "/"
    
    # metadata
    df_meta = pd.DataFrame({"Observation Time": [current_info.general_info["Tobs"]], "parent": [None], "Build Time": [datetime.now()]})
    df_meta.to_hdf(current_info.general_info["file_information"]["gb_main_chain_file"], "metadata", mode="a", complevel=9)

    assert len(name_index) == len(median_inds_out)
    # exit(0)
    for i, (sub_df_name, index, store_file) in enumerate(zip(name_index, median_inds_out, chain_file_list)):
        sub_df = df[df["Group ID"].astype(int) == int(df.iloc[index]["Group ID"])]
        sub_df.to_hdf(dirname + store_file, sub_df_name + "_chain", mode="a", complevel=9)
        if (i + 1) % 100 == 0:
            print(f"Sub hdf: {i + 1} of {len(median_inds_out)}")

def build_mbh_catalog(current_info, gpu, **kwargs):
    
    mbh_reader = current_info.mbh_info["reader"]
    
    ntemps, nwalkers, mbh_leaves, mbh_ndim  = mbh_reader.shape["mbh"]

    # TODO: adjust when time evolving is added 
    
    keys = ['Log Likelihood', 'Mass 1', 'Mass 2', 'Spin 1', 'Spin 2',
       'Merger Phase', 'Barycenter Merge Time', 'Luminosity Distance',
       'cos ecliptic colatitude', 'Ecliptic Longitude', 'Polarization',
       'cos inclination', 'Detector Merger Time', 'Ecliptic Latitude']

    samp_keys = ["ln total mass", "Mass Ratio", "Spin 1", "Spin 2", "Luminosity Distance",
        "Merger Phase", "cos inclination", "Ecliptic Longitude (L-frame)", "sin ecliptic latitude (L-Frame)",
        "Polarization (L-frame)", "Detector Merger Time"]

    year_val = datetime.now().year
    month_val = datetime.now().month
    day_val = datetime.now().day

    name_index = [f"EREBOR_MBH_{year_val}_{month_val}_{day_val}_{i + 1:04d}" for i in range(mbh_leaves)]
    output_for_main = []
    for leaf_i in range(mbh_leaves):

        mbh_samples = mbh_reader.get_chain(**kwargs)["mbh"][:, 0, :, leaf_i].reshape(-1, mbh_ndim)
        
        df = pd.DataFrame({key: item for key, item in zip(samp_keys, mbh_samples.T)})

        df["Log Likelihood"] = mbh_reader.get_log_like(**kwargs)[:, 0].flatten()

        # massess
        df["Total Mass"] = np.exp(df['ln total mass'])
        df["Mass 1"] = df["Total Mass"] / (1.0 + df["Mass Ratio"])
        df["Mass 2"] = df["Total Mass"] * df["Mass Ratio"] / (1.0 + df["Mass Ratio"])
        assert np.allclose(df["Mass 1"] + df["Mass 2"], df["Total Mass"])

        # sky transform
        # TODO: Jacobian?
        df["Ecliptic Latitude (L-Frame)"] = np.arcsin(df["sin ecliptic latitude (L-Frame)"])
        tL, lamL, betaL, psiL = df["Detector Merger Time"], df["Ecliptic Longitude (L-frame)"], df["Ecliptic Latitude (L-Frame)"], df["Polarization (L-frame)"]
        tSSB, lamSSB, betaSSB, psiSSB = LISA_to_SSB(tL, lamL, betaL, psiL)

        df["Barycenter Merge Time"] = tSSB
        df["Ecliptic Longitude"] = lamSSB
        df["Ecliptic Latitude"] = betaSSB
        df["Polarization"] = psiSSB
        df["cos ecliptic colatitude"] = np.cos(np.pi / 2 - betaSSB) 

        # store samples
        df.to_hdf(current_info.general_info["file_information"]["mbh_main_chain_file"], name_index[leaf_i] + "_chain", mode="a", complevel=9)
        
        # determine median merger time
        median_merger_time_detector_ind_keep = np.argsort(df["Detector Merger Time"])[int(len(df["Detector Merger Time"]) / 2) + int(len(df["Detector Merger Time"]) % 2)]

        median_merger = df.iloc[median_merger_time_detector_ind_keep]

        output_for_main.append(median_merger)

    # setup main file group
    df_main = pd.concat(output_for_main, axis=1).T
    df_main.index = name_index 
    df_main["chain file"] = [tmp + "_chain" for tmp in name_index]
    df_main["Parent"] = [None for tmp in name_index]

    df_main = df_main.sort_values("Detector Merger Time")
    df_main.to_hdf(current_info.general_info["file_information"]["mbh_main_chain_file"], "detections", mode="a", complevel=9)
    
    # metadata
    df_meta = pd.DataFrame({"observation week": [52], "parent": [None], "creation date": [datetime.now()], "author": ["Michael Katz"]})
    
    df_meta.index = ["EREBOR_MBHcatalog_week052"]
    df_meta.to_hdf(current_info.general_info["file_information"]["mbh_main_chain_file"], "metadata", mode="a", complevel=9)


catalog_generate_funcs = {
    "gb": build_gb_catalog,
    "mbh": build_mbh_catalog
}

def build_catalog(current_info, gpu=None, catalogs=["gb", "mbh"], cat_kwargs={}, **kwargs):
     
    if isinstance(catalogs, str):
        catalogs = [catalogs]

    for catalog_type in catalogs:
        if catalog_type not in cat_kwargs:
            cat_kwargs[catalog_type] = {}

        catalog_generate_funcs[catalog_type](current_info, gpu, **cat_kwargs[catalog_type])