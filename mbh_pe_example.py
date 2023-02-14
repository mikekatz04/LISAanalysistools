import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist

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

from lisatools.sensitivity import get_sensitivity
from lisatools.sampling.utility import HeterodynedUpdate

from eryn.utils import TransformContainer

np.random.seed(111222)

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(7)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

# whether you are using 
use_gpu = False

if use_gpu is False:
    xp = np

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

# function call
def run_mbh_pe(
    mbh_injection_params, 
    Tobs,
    dt,
    fp,
    ntemps,
    nwalkers,
    mbh_kwargs={}
):

    # sets the proper number of points and what not

    N_obs = int(Tobs / dt) # may need to put "- 1" here because of real transform
    Tobs = N_obs * dt
    t_arr = xp.arange(N_obs) * dt

    # frequencies
    freqs = xp.fft.rfftfreq(N_obs, dt)

    # wave generating class
    wave_gen = BBHWaveformFD(
        amp_phase_kwargs=dict(run_phenomd=True),
        response_kwargs=dict(TDItag="AET"),
        use_gpu=use_gpu
    )

    # for transforms
    fill_dict = {
        "ndim_full": 12,
        "fill_values": np.array([0.0]),
        "fill_inds": np.array([6]),
    }

    (
        M, 
        q,
        a1, 
        a2,
        dist,
        phi_ref,
        inc,
        lam,
        beta,
        psi,
        t_ref,
    ) = mbh_injection_params

    # get the right parameters
    mbh_injection_params[0] = np.log(mbh_injection_params[0])
    mbh_injection_params[6] = np.cos(mbh_injection_params[6]) 
    mbh_injection_params[8] = np.sin(mbh_injection_params[8])

    # priors
    priors = {
        "mbh": ProbDistContainer(
            {
                0: uniform_dist(np.log(1e5), np.log(1e8)),
                1: uniform_dist(0.01, 0.999999999),
                2: uniform_dist(-0.99999999, +0.99999999),
                3: uniform_dist(-0.99999999, +0.99999999),
                4: uniform_dist(0.01, 1000.0),
                5: uniform_dist(0.0, 2 * np.pi),
                6: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                7: uniform_dist(0.0, 2 * np.pi),
                8: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                9: uniform_dist(0.0, np.pi),
                10: uniform_dist(0.0, Tobs + 3600.0),
            }
        ) 
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

    # sampler treats periodic variables by wrapping them properly
    periodic = {
        "mbh": {5: 2 * np.pi, 7: np.pi, 8: np.pi}
    }

    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(mbh_injection_params[None, :], return_transpose=True)

    # get XYZ
    data_channels_AET = wave_gen(*injection_in, freqs=freqs,
        modes=[(2,2)], direct=False, fill=True, squeeze=True, length=1024
    )[0]

    # time domain
    data_channels_AET_TD = np.fft.irfft(data_channels_AET,axis=-1)

    # convert to AET
    data_channels = data_channels_AET
    
    reference_params = injection_in.copy()   

    # how many frequencies to use in heterodyning. 
    # may need more with higher modes
    length_f_het = 128
    
    # initialize Likelihood
    like_het = HeterodynedLikelihood(
        wave_gen,
        freqs,
        data_channels,
        reference_params,
        length_f_het,
        use_gpu=use_gpu,
    )

    update = HeterodynedUpdate({}, set_d_d_zero=True)
    
    like_het.reference_d_d = 0.0
    # this is a parent likelihood class that manages the parameter transforms
    like = Likelihood(
        like_het,
        3,
        dt=None,
        df=None,
        f_arr=freqs,
        parameter_transforms={"mbh": transform_fn},
        use_gpu=use_gpu,
        vectorized=True,
        transpose_params=True,
    )

    ndim = 11

    # generate starting points
    factor = 1e-5
    cov = np.ones(ndim) * 1e-3
    cov[0] = 1e-5
    cov[-1] = 1e-5

    start_like = np.zeros((nwalkers * ntemps))
    
    iter_check = 0
    max_iter = 1000
    while np.std(start_like) < 5.0:
        
        logp = np.full_like(start_like, -np.inf)
        tmp = np.zeros((ntemps * nwalkers, ndim))
        fix = np.ones((ntemps * nwalkers), dtype=bool)
        while np.any(fix):
            tmp[fix] = (mbh_injection_params[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, ndim)))[fix]

            tmp[:, 5] = tmp[:, 5] % (2 * np.pi)
            tmp[:, 7] = tmp[:, 7] % (2 * np.pi)
            tmp[:, 9] = tmp[:, 9] % (1 * np.pi)
            logp = priors["mbh"].logpdf(tmp)

            fix = np.isinf(logp)
            if np.all(fix):
                breakpoint()

        start_like = like(tmp, **mbh_kwargs)

        iter_check += 1
        factor *= 1.5

        print(np.std(start_like))

        if iter_check > max_iter:
            raise ValueError("Unable to find starting parameters.")

    start_params = tmp.copy()

    # position things around the sky
    sky_related_coords = start_params[:, 6:10]
    # get arccos and arcsin
    sky_related_coords[:, 0] = np.arccos(sky_related_coords[:, 0])
    sky_related_coords[:, 2] = np.arcsin(sky_related_coords[:, 2])
    ind_map = dict(inc=0, lam=1, beta=2, psi=3)
    out_sky_related_coords = mbh_sky_mode_transform(sky_related_coords, ind_map=ind_map, inplace=True, kind="both", cos_i=False)

    out_sky_related_coords[:, 0] = np.cos(out_sky_related_coords[:, 0])
    out_sky_related_coords[:, 2] = np.sin(out_sky_related_coords[:, 2])

    start_params[:, 6:10] = out_sky_related_coords
    
    # get_ll and not __call__ to work with lisatools
    start_like = like(start_params, **mbh_kwargs)
    start_prior = priors["mbh"].logpdf(start_params)

    # start state
    start_state = State(
        {"mbh": start_params.reshape(ntemps, nwalkers, 1, ndim)}, 
        log_like=start_like.reshape(ntemps, nwalkers), 
        log_prior=start_prior.reshape(ntemps, nwalkers)
    )

    # MCMC moves (move, percentage of draws)
    moves = [
        (SkyMove(which="both"), 0.04),
        (SkyMove(which="long"), 0.02),
        (SkyMove(which="lat"), 0.02),
        (StretchMove(), 0.94)
    ]

    # prepare sampler
    sampler = EnsembleSampler(
        nwalkers,
        [ndim],  # assumes ndim_max
        like,
        priors,
        tempering_kwargs={"ntemps": ntemps, "Tmax": np.inf},
        moves=moves,
        kwargs=mbh_kwargs,
        backend=fp,
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        update_fn=update,
        update_iterations=5,
        branch_names=["mbh"],
    )

    nsteps = 50
    out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=100, burn=500)

    # get samples
    samples = sampler.get_chain(discard=0, thin=1)["mbh"][:, 0].reshape(-1, ndim)
    
    # plot
    fig = corner.corner(samples, levels=1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2), truths=mbh_injection_params)
    fig.savefig(fp[:-3] + "_corner.png", dpi=150)
    return

if __name__ == "__main__":
    # set parameters
    f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
    phi_ref = 0.5 # phase at f_ref
    m1 = 1e6
    m2 = 5e5
    M = m1 + m2
    q = m2 / m1  # m2 less than m1
    a1 = 0.2
    a2 = 0.4
    dist = 100.0  #   * PC_SI * 1e6 # 3e3 in Gpc
    inc = np.pi/3.
    beta = np.pi/4.  # ecliptic latitude
    lam = np.pi/5.  # ecliptic longitude
    psi = np.pi/6.  # polarization angle
    
    t_ref = 0.1 * YRSID_SI + 500.0  # t_ref  (in the SSB reference frame)

    Tobs = 0.5 * YRSID_SI
    dt = 1. / 0.3
    fp = "test_mbh.h5"

    mbh_injection_params = np.array([
        M, 
        q,
        a1, 
        a2,
        dist,
        phi_ref,
        inc,
        lam,
        beta,
        psi,
        t_ref
    ])

    # for base comparison

    ntemps = 10
    nwalkers = 1024

    waveform_kwargs = {
        "modes": [(2,2)],
    }

    print("start", fp)
    run_mbh_pe(
        mbh_injection_params, 
        Tobs,
        dt,
        fp,
        ntemps,
        nwalkers,
        mbh_kwargs=waveform_kwargs
    )
    print("end", fp)
    # frequencies to interpolate to
    
        