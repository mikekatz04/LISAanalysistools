from typing import Any
import numpy as np
from lisatools.diagnostic import snr as snr_func
from lisatools.sensitivity import get_sensitivity
from abc import ABC
from typing import Union, Tuple

from few.waveform import GenerateEMRIWaveform
from fastlisaresponse import ResponseWrapper

from gbgpu.gbgpu import GBGPU

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.constants import *


class AETTemplateGen(ABC):
    # @classmethod
    # @property
    # def domain_variables(self) -> dict:
    #     breakpoint()
    #     return {"dt": self.dt, "f_arr": self.f_arr, "df": self.df}

    @property
    def dt(self) -> float:
        return None

    @property
    def f_arr(self) -> float:
        return None

    @property
    def df(self) -> float:
        return None


class BBHTemplateGen(AETTemplateGen):
    def __init__(self, amp_phase_kwargs: dict = dict(run_phenomd=True)) -> None:
        # wave generating class
        self.wave_gen = BBHWaveformFD(
            amp_phase_kwargs=amp_phase_kwargs,
            response_kwargs=dict(TDItag="AET"),
        )

    @property
    def f_arr(self) -> np.ndarray:
        return self._f_arr

    @f_arr.setter
    def f_arr(self, f_arr: np.ndarray) -> None:
        self._f_arr = f_arr

    def __call__(
        self, *params: Any, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m1 = params[0]
        m2 = params[1]

        min_f = 1e-4 / (MTSUN_SI * (m1 + m2))
        max_f = 0.6 / (MTSUN_SI * (m1 + m2))

        self.f_arr = np.logspace(np.log10(min_f), np.log10(max_f), 1024)
        AET = self.wave_gen(
            *params,
            modes=[(2, 2)],
            direct=True,
            combine=True,
            freqs=self.f_arr,
            **kwargs,
        )[0]

        return (AET[0], AET[1], AET[2])


class GBTemplateGen(AETTemplateGen):
    def __init__(self) -> None:
        # wave generating class
        self.wave_gen = GBGPU()

    @property
    def f_arr(self) -> np.ndarray:
        return self._f_arr

    @f_arr.setter
    def f_arr(self, f_arr: np.ndarray) -> None:
        self._f_arr = f_arr

    def __call__(
        self, *params: Any, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.wave_gen.run_wave(
            *params,
            **kwargs,
        )

        self.f_arr = self.wave_gen.freqs[0]
        A = self.wave_gen.A[0]
        E = self.wave_gen.E[0]
        T = self.wave_gen.X[0]

        return (A, E, T)


class EMRITemplateGen(AETTemplateGen):
    def __init__(
        self, waveform: str, orbit_file: str, dt: float = 15.0, Tobs: float = 1.0
    ) -> None:
        # sets the proper number of points and what not
        N_obs = int(
            Tobs * YRSID_SI / dt
        )  # may need to put "- 1" here because of real transform
        Tobs = (N_obs * dt) / YRSID_SI
        t_arr = np.arange(N_obs) * dt
        self.dt = dt

        few_gen = GenerateEMRIWaveform(
            waveform,
            sum_kwargs=dict(pad_output=True),
        )

        orbit_kwargs = dict(orbit_file=orbit_file)
        tdi_gen = "1st generation"  # fixed and fine for basic comps
        order = 25  # interpolation order (should not change the result too much)
        tdi_kwargs = dict(
            orbit_kwargs=orbit_kwargs,
            order=order,
            tdi=tdi_gen,
            tdi_chan="AET",
        )  # could do "AET"
        index_lambda = 8
        index_beta = 7
        # with longer signals we care less about this
        t0 = 20000.0  # throw away on both ends when our orbital information is weird
        self.wave_gen = ResponseWrapper(
            few_gen,
            Tobs,
            dt,
            index_lambda,
            index_beta,
            t0=t0,
            flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
            is_ecliptic_latitude=False,  # False if using polar angle (theta)
            remove_garbage="zero",  # removes the beginning of the signal that has bad information
            **tdi_kwargs,
        )

    @property
    def dt(self) -> np.ndarray:
        return self._dt

    @dt.setter
    def dt(self, dt: np.ndarray) -> None:
        self._dt = dt

    def __call__(
        self, *params: Any, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        AET = self.wave_gen(
            *params,
            **kwargs,
        )
        return (AET[0], AET[1], AET[2])


class CalculateSNR:
    def __init__(self, aet_template_gen: AETTemplateGen, psd_kwargs: dict) -> None:
        self.aet_template_gen = aet_template_gen
        self.psd_kwargs = psd_kwargs

    def __call__(self, *params: np.ndarray | list, **kwargs: Any) -> float:
        a_chan, e_chan, t_chan = self.aet_template_gen(*params, **kwargs)

        # ignore t channel for snr computation
        opt_snr = snr_func(
            [a_chan, e_chan],
            PSD="noisepsd_AE",
            PSD_kwargs=self.psd_kwargs,
            dt=self.aet_template_gen.dt,
            f_arr=self.aet_template_gen.f_arr,
            df=self.aet_template_gen.df,
        )
        self.f_arr = self.aet_template_gen.f_arr
        self.last_output = (a_chan, e_chan)

        return opt_snr


if __name__ == "__main__":
    bbh = BBHTemplateGen()
    psd_kwargs = dict(
        model="SciRDv1",  # can be SciRDv1, MRDv1, sangria
        includewd=1,  # Years of observation so far
    )
    gb = GBTemplateGen()
    emri = EMRITemplateGen(
        "FastSchwarzschildEccentricFlux",
        "../../lisa-on-gpu/orbit_files/esa-trailing-orbits.h5",
    )

    bbh_snr_calc = CalculateSNR(bbh, psd_kwargs)
    gb_snr_calc = CalculateSNR(gb, psd_kwargs)
    emri_snr_calc = CalculateSNR(emri, psd_kwargs)

    f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
    phi_ref = 0.5  # phase at f_ref
    m1 = 1e6
    m2 = 5e5
    M = m1 + m2
    q = m2 / m1  # m2 less than m1
    a1 = 0.2
    a2 = 0.4
    dist = 10.0 * PC_SI * 1e9  # in m
    inc = np.pi / 3.0
    beta = np.pi / 4.0  # ecliptic latitude
    lam = np.pi / 5.0  # ecliptic longitude
    psi = np.pi / 6.0  # polarization angle

    t_ref = 0.1 * YRSID_SI + 500.0  # t_ref  (in the SSB reference frame)

    mbh_injection_params = np.array(
        [m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref]
    )

    amp = 1e-22
    f0 = 4e-3
    fdot = 1e-18
    fddot = 0.0
    phi0 = 0.5
    inc = 0.2
    psi = 0.6
    lam = 0.9
    beta = -0.2
    gb_injection_params = np.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta])

    # data_channels_AET = bbh(
    #     *mbh_injection_params,
    # )[0]

    snr_val = bbh_snr_calc(*mbh_injection_params)

    gb_snr_val = gb_snr_calc(*gb_injection_params)

    M = 1e6
    mu = 1e1
    a = 0.2
    p0 = 12.0
    e0 = 0.2
    x0 = 0.1
    dist = 3.0
    phiS = 0.4423
    qS = 0.523023
    qK = 0.8923123
    phiK = 0.1221209312
    Phi_phi0 = 0.1231232
    Phi_theta0 = 4.234903824
    Phi_r0 = 3.230923

    emri_injection_params = np.array(
        [M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
    )

    print("begin")
    emri_snr_val = emri_snr_calc(*emri_injection_params)
    breakpoint()
