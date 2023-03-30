import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from lisatools.glitch import tdi_glitch_XYZ1
import corner
from lisatools.utils.utility import AET

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.likelihood import Likelihood as MBHLikelihood
from bbhx.likelihood import HeterodynedLikelihood
from bbhx.utils.constants import *
from bbhx.utils.transform import *

from eryn.moves import StretchMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.sampling.stopping import SearchConvergeStopping

from lisatools.sensitivity import get_sensitivity
from lisatools.sampling.utility import HeterodynedUpdate

from eryn.utils import TransformContainer

from full_band_global_fit_settings import *
fd1 = fd.copy()

xp.cuda.runtime.setDevice(3)
use_gpu = True

mbh_kwargs = {
    "modes": [(2,2)],
    "length": 1024,
    "t_obs_start": 0.0,
    "t_obs_end": 1.0,
    "shift_t_limits": True
}

# for transforms
fill_dict = {
    "ndim_full": 12,
    "fill_values": np.array([0.0]),
    "fill_inds": np.array([6]),
}

# transforms from pe to waveform generation
parameter_transforms = {
    0: np.exp,
    4: lambda x: x * PC_SI * 1e9,  # Gpc
    7: np.arccos,
    9: np.arcsin,
    (0, 1): mT_q,
    (11, 8, 9, 10): LISA_to_SSB,
}

transform_fn = TransformContainer(
    parameter_transforms=parameter_transforms,
    fill_dict=fill_dict,
)

# wave generating class
wave_gen = BBHWaveformFD(
    amp_phase_kwargs=dict(run_phenomd=True, initial_t_val=0.0),
    response_kwargs=dict(TDItag="AET"),
    use_gpu=use_gpu
)

psd_reader = HDFBackend(fp_psd_search_initial)
best_psd_ind = psd_reader.get_log_like().flatten().argmax()
best_psd_params = psd_reader.get_chain()["psd"].reshape(-1, 4)[best_psd_ind]

A_psd = get_sensitivity(fd, sens_fn="noisepsd_AE", model=best_psd_params[:2])
E_psd = get_sensitivity(fd, sens_fn="noisepsd_AE", model=best_psd_params[2:])

A_psd[0] = A_psd[1]
E_psd[0] = E_psd[1]

try:
    del like_mbh
    del like
    del sampler
    del data_channels
    del psd
except NameError:
    pass

data_channels = xp.asarray([A_inj, E_inj, np.zeros_like(E_inj)])

psd = xp.asarray([A_psd, E_psd, np.full_like(A_psd, 1e10)])

# initialize Likelihood
like_mbh = MBHLikelihood(
    wave_gen, xp.asarray(fd), data_channels, psd, use_gpu=use_gpu
)

# this is a parent likelihood class that manages the parameter transforms
like = Likelihood(
    like_mbh,
    3,
    dt=None,
    df=None,
    f_arr=fd,
    parameter_transforms={"mbh": transform_fn},
    use_gpu=use_gpu,
    vectorized=True,
    transpose_params=True,
    subset=100,
)

output_A = np.zeros_like(A_inj)
output_E = np.zeros_like(E_inj)

ndim = 11
# fd = np.asarray(fd1)
mbh_found_points = np.load("found_mbh_points_after_search.npy")
output_points_into_mix = []
start_ll_check = (-1/2 * 4 * df * np.sum(data_channels.conj() * data_channels / psd) - xp.sum(xp.log(xp.asarray(psd)))).get()
print(start_ll_check)
last_ll = start_ll_check
for mbh_inj_point in mbh_found_points:

    like_tmp = like(mbh_inj_point[None, :], phase_marginalize=False, **mbh_kwargs)
    # print(mbh_inj_point, like.template_model.d_h)
    phase_change = np.angle(like.template_model.d_h)
    mbh_inj_point[5] = (mbh_inj_point[5] + 1/ 2 * phase_change) % (2 * np.pi)
    output_points_into_mix.append(mbh_inj_point)
    check_like_tmp = like(mbh_inj_point[None, :], phase_marginalize=False, **mbh_kwargs)
    # print(like.template_model.d_h)

    # breakpoint()
    injection_in = transform_fn.both_transforms(mbh_inj_point[None, :], return_transpose=True)

    # get XYZ
    data_channels_AET = wave_gen(*injection_in, freqs=xp.asarray(fd),
        modes=[(2,2)], direct=False, fill=True, squeeze=True, length=1024
    )[0]

    A_inj -= data_channels_AET[0].get()
    E_inj -= data_channels_AET[1].get()

    data_channels = xp.asarray([A_inj, E_inj, np.zeros_like(E_inj)])

    start_ll_check = (-1/2 * 4 * df * np.sum(data_channels.conj() * data_channels / psd) - xp.sum(xp.log(xp.asarray(psd)))).get()
    print(start_ll_check, start_ll_check - last_ll)
    last_ll = start_ll_check
    output_A += data_channels_AET[0].get()
    output_E += data_channels_AET[1].get()

np.save("mbh_injection_points_after_initial_search", np.asarray(output_points_into_mix))
np.save(fp_mbh_template_search, np.asarray([np.tile(output_A, (100, 1)), np.tile(output_E, (100, 1))]).transpose(1, 0, 2))
np.save(fp_psd_residual_search, np.asarray([np.tile(A_psd, (100, 1)), np.tile(E_psd, (100, 1))]).transpose(1, 0, 2))

