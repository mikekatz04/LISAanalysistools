"""GB JAX TDI-on-the-fly likelihood, gradient, and tempered NUTS MCMC.

Pieces wired together:

* ``gbgpu.jax`` + ``lisatools.jax.response`` -- TDI-on-the-fly GB JAX
  waveform driven by :class:`gbgpu.jax.sources.JaxUCBSource`, projected
  through :func:`gbgpu.jax.tdi_on_the_fly.gb_run_wave_tdi`. The TDI
  projection uses an :class:`OrbitsWrapJAX` and
  :class:`lisatools.jax.response.TDIConfigWrapJAX` built from LISAtools'
  ``EqualArmlengthOrbits`` and ``lisatools.response.tdiconfig.TDIConfig``
  on the jax backend.
* ``lisatools`` -- :class:`AnalysisContainer` + :class:`DataResidualArray`
  + :class:`SensitivityMatrix` package the data, the PSD, and the signal
  generator on the lisatools side. The actual likelihood and its
  gradient are evaluated in pure :mod:`jax.numpy` (so we can differentiate
  through them).
* ``eryn.moves.NUTSMove`` (with tempering) -- the gradient produced here
  is fed to a parallel-tempered NUTS chain that targets the GB posterior.

Run it as::

    python LISAanalysistools/examples/gb_jax_likelihood_grad.py

The numbers are deliberately small (TDI 1, short observation, low ``N``,
few walkers/temps/steps) so the example compiles and runs in a few
minutes on CPU. Bigger problems are a straightforward scale-up.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

# Make the repo-root scripts importable when the example is run from
# inside ``LISAanalysistools/examples`` without installing them.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import lisatools  # noqa: F401  -- registers the lisatools_jax backend
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import EqualArmlengthOrbits
from lisatools.diagnostic import inner_product
from lisatools.domains import FDSettings, TDSettings
from lisatools.sensitivity import (
    SensitivityMatrix,
    X1TDISens,
    Y1TDISens,
    Z1TDISens,
)
from lisatools.stochastic import StochasticContribution

# GB JAX TDI-on-the-fly (carved out of lisa-on-gpu at Phase 3L.7j into
# lisatools.jax.response / gbgpu.jax).
from lisatools.jax.response import TDIConfigWrapJAX
from gbgpu.jax.sources import JaxUCBSource
from gbgpu.jax.tdi_on_the_fly import gb_run_wave_tdi
from lisatools.response.tdiconfig import TDIConfig

# Eryn pieces for the MCMC stage
from eryn.ensemble import EnsembleSampler
from eryn.moves import NUTSMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State


PARAM_NAMES = (
    "amp", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta",
)
N_PARAMS = len(PARAM_NAMES)


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------

class _ZeroStochastic(StochasticContribution):
    """Stochastic add-on stub so ``Sensitivity.get_Sn`` is non-divergent."""

    @classmethod
    def get_Sh(cls, f, **kwargs):  # noqa: D401
        return np.zeros_like(np.asarray(f, dtype=float))


def build_orbits_and_tdi(
    tdi_gen: str = "1st generation",
    obs_window: tuple = (0.0, 3600.0),
    margin: float = 7200.0,
    orbit_dt: float = 600.0,
):
    """Return ``(orbits_jax, tdi_config_jax)`` ready for the JAX kernel.

    The orbit interpolation table is trimmed to a small custom time array
    that covers ``obs_window`` plus a ``margin`` on either side -- this
    keeps the JAX gather tables tiny so eager calls aren't dominated by
    the cost of dispatching ops over the full 5-year LISA mission grid.

    Args:
        tdi_gen: ``"1st generation"`` or ``"2nd generation"``.
        obs_window: ``(t_start, t_end)`` of the actual observation, in
            seconds since the orbits' ``t0``.
        margin: Padding on either side of ``obs_window`` so the TDI
            delays + advances stay inside the interpolation grid.
        orbit_dt: Spacing of the trimmed orbit grid. ~10 minutes is
            plenty for LISA's slow geometry.
    """
    orbits = EqualArmlengthOrbits(force_backend="jax")
    t_lo = max(0.0, obs_window[0] - margin)
    t_hi = obs_window[1] + margin
    t_arr_orbit = np.arange(t_lo, t_hi + 0.5 * orbit_dt, orbit_dt)
    orbits.configure(t_arr=t_arr_orbit)
    orbits_jax = orbits.pycppdetector  # OrbitsWrapJAX

    tdi = TDIConfig(tdi_gen, force_backend="jax")
    tdi_jax = TDIConfigWrapJAX(*tdi.pytdiconfig_args)
    return orbits_jax, tdi_jax


def build_psd(f_arr: np.ndarray, channels=(X1TDISens, Y1TDISens, Z1TDISens)) -> np.ndarray:
    """Compute the per-channel XYZ TDI-1 PSD on ``f_arr``.

    Returns an array of shape ``(nchannels, len(f_arr))``. ``f = 0`` is
    masked out by sending it to ``inf`` (the bin is dropped at inner
    product time anyway).
    """
    f_safe = np.where(f_arr > 0, f_arr, f_arr[1] if len(f_arr) > 1 else 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        psd = np.stack(
            [ch.get_Sn(f_safe, stochastic_function=_ZeroStochastic) for ch in channels],
            axis=0,
        )
    psd[:, f_arr == 0] = np.inf
    return psd


# ----------------------------------------------------------------------
# JAX likelihood factory
# ----------------------------------------------------------------------

def gb_template_real_td(theta, t_arr_jax, t_ref, orbits_jax, tdi_jax):
    """Pure GB TDI-on-the-fly template (real-valued, time-domain).

    Standalone helper so we can build the injection up-front and feed it
    to the :class:`AnalysisContainer` before constructing the likelihood.
    """
    M, _amp, _phase, _pref = gb_run_wave_tdi(
        theta[None, :], t_arr_jax[None, :], t_ref, orbits_jax, tdi_jax,
    )
    return jnp.real(M[0])                 # (nchannels, N) real


def make_likelihood(
    analysis: "AnalysisContainer",
    orbits_jax,
    tdi_jax,
    t_arr_jax: jnp.ndarray,
    t_ref: float,
    dt: float,
    fd_settings: "FDSettings",
):
    """Build ``(template_one, loglike_one, grad_loglike_one)`` closures.

    The :class:`AnalysisContainer` is the source of truth for the FD
    injection and the (jax-converted) sensitivity matrix; the closures
    just delegate to :meth:`AnalysisContainer.template_likelihood`,
    which evaluates ``log L = -0.5 * <d-h|d-h>`` through
    :func:`lisatools.diagnostic.inner_product`. The whole graph is
    ``jax.grad``-differentiable because ``get_array_module`` is
    jax-aware and ``inner_product`` no longer concretises its return
    value via ``.item()``.

    The container caches:

    * the FD-domain injection (``analysis.data_res_arr``),
    * the PSD / inverse-PSD (``analysis.sens_mat``, with
      ``sens_mat.invC`` already converted to ``jnp``),

    so neither is rebuilt inside the likelihood.

    Args:
        analysis: pre-built :class:`AnalysisContainer` with the data +
            sensitivity arrays already loaded.
        orbits_jax: :class:`OrbitsWrapJAX` instance.
        tdi_jax: :class:`TDIConfigWrapJAX` instance.
        t_arr_jax: ``(N,)`` time array.
        t_ref: GB reference time (seconds).
        dt: Time sample spacing (seconds), needed for rFFT scaling.
        fd_settings: :class:`FDSettings` matching ``analysis.data_res_arr``
            (must compare equal so :meth:`template_likelihood` takes the
            fast no-slice path).
    """

    def template_one(theta):
        return gb_template_real_td(theta, t_arr_jax, t_ref, orbits_jax, tdi_jax)

    def template_fd_arr(theta):
        """JAX template wrapped as a :class:`DataResidualArray`."""
        h_fd = jnp.fft.rfft(template_one(theta), axis=-1) * dt
        return DataResidualArray(h_fd, input_signal_domain=fd_settings)

    def loglike_one(theta):
        # AnalysisContainer drives the computation: it owns the cached
        # injection and PSD and dispatches to ``inner_product`` for
        # ``<d|d>``, ``<h|h>``, and ``<d|h>``. The whole call is
        # traceable by jax.grad.
        return analysis.template_likelihood(template_fd_arr(theta))

    grad_loglike_one = jax.grad(loglike_one)

    return template_one, loglike_one, grad_loglike_one


# ----------------------------------------------------------------------
# Demo runner
# ----------------------------------------------------------------------

def main():
    # ----- problem size --------------------------------------------------
    # Short observation: enough to validate the pipeline; keeps the
    # 24-unit TDI-1 JAX kernel within a manageable compile budget.
    N = 64
    dt = 30.0
    T = N * dt                       # seconds of observation
    t_ref = T / 2
    df = 1.0 / T                     # FD bin spacing of the FFT'd residual
    Nf = N // 2 + 1                  # rfft length
    f_arr = np.arange(Nf) * df

    tdi_gen = "1st generation"
    print("Setup: N={0}, dt={1:.1f}s, T={2:.1f}s, Nf={3}, df={4:.3e} Hz, tdi={5}".format(
        N, dt, T, Nf, df, tdi_gen,
    ))

    # ----- backends ------------------------------------------------------
    # Trim orbits to a window covering the observation + delay margin so
    # the JAX gather tables stay small (otherwise the lookup tables span
    # the full 5-year LISA mission and dominate every eager call).
    t0 = time.time()
    orbits_jax, tdi_jax = build_orbits_and_tdi(
        tdi_gen,
        obs_window=(0.0, T),
        margin=2.0 * T,
        orbit_dt=300.0,
    )
    print(
        "Built orbits + tdi_config (jax backend) in {0:.1f}s, "
        "orbit table size: {1}".format(time.time() - t0, orbits_jax.x.shape[0])
    )

    # ----- lisatools containers ------------------------------------------
    # The frequency-domain grid + sensitivity matrix are built once
    # (numpy/cpu side). We then convert ``sens_mat.invC`` to a JAX array
    # so the LISAtools ``inner_product`` runs end-to-end in jax inside
    # the likelihood.
    td_settings = TDSettings(N=N, dt=dt, force_backend="cpu")
    fd_settings = FDSettings(N=Nf, df=df, force_backend="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sens_mat = SensitivityMatrix(
            fd_settings, [X1TDISens, Y1TDISens, Z1TDISens],
            stochastic_function=_ZeroStochastic,
        )
    sens_mat.invC = jnp.asarray(sens_mat.invC)
    # Mask DC bin (f=0) -- some PSD models go to inf there.
    sens_mat.invC = sens_mat.invC.at[:, 0].set(0.0)

    # ----- inject data at theta_true -------------------------------------
    theta_true = jnp.array(
        [2.0e-22,    # amp -- large, so the band has measurable signal
         2.35e-3,    # f0 (Hz)
         1.0e-17,    # fdot
         0.0,        # fddot
         0.7,        # phi0
         0.9,        # iota
         0.4,        # psi
         1.7,        # lam
         0.3,        # beta
         ],
        dtype=jnp.float64,
    )
    source = JaxUCBSource(t_ref=t_ref)
    print("Source: {0} ({1} params)".format(type(source).__name__, source.n_params))

    # Generate the injection with the standalone GB JAX TDI-on-the-fly
    # template helper (no AnalysisContainer needed yet).
    print("Generating injection ...", flush=True)
    t0 = time.time()
    t_arr_jax = jnp.arange(N, dtype=jnp.float64) * dt
    h_true_td = gb_template_real_td(
        theta_true, t_arr_jax, t_ref, orbits_jax, tdi_jax,
    )
    h_true_td.block_until_ready()
    print("  template(theta_true): {0:.2f}s".format(time.time() - t0))
    data_fd_jnp = jnp.fft.rfft(h_true_td, axis=-1) * dt

    # ----- AnalysisContainer = single source of truth --------------------
    # The injection (FD) and the PSD (jax-converted invC) are cached on
    # the AnalysisContainer here, BEFORE the likelihood factory runs.
    # Downstream likelihood / gradient / MCMC all read from this object.
    data_arr = DataResidualArray(data_fd_jnp, input_signal_domain=fd_settings)

    def signal_gen(*theta, **_kwargs):
        """FD-domain template. Returns a JAX array so the AnalysisContainer's
        vectorized likelihood path (``eryn_likelihood_wrap(..., use_vmap=True)``)
        can batch it under :func:`jax.vmap`."""
        h_td = gb_template_real_td(
            jnp.asarray(theta, dtype=jnp.float64),
            t_arr_jax, t_ref, orbits_jax, tdi_jax,
        )
        return jnp.fft.rfft(h_td, axis=-1) * dt

    analysis = AnalysisContainer(
        data_res_arr=data_arr, sens_mat=sens_mat, signal_gen=signal_gen,
    )

    # Hand the cached container off to the likelihood factory; the
    # closures it returns delegate every <d|d>, <h|h>, <d|h> to
    # ``analysis.template_likelihood``.
    template_one, loglike_one, grad_loglike_one = make_likelihood(
        analysis=analysis,
        orbits_jax=orbits_jax, tdi_jax=tdi_jax, t_arr_jax=t_arr_jax,
        t_ref=t_ref, dt=dt, fd_settings=fd_settings,
    )
    print("AnalysisContainer: nchannels={0}, sens.shape={1}".format(
        analysis.data_res_arr.nchannels, analysis.sens_mat.shape,
    ))

    # ----- likelihood / gradient at theta_true and an offset -------------
    t0 = time.time()
    ll0 = float(loglike_one(theta_true))
    print("loglike(theta_true) = {0:+.3e}   ({1:.2f}s)".format(ll0, time.time() - t0))
    t0 = time.time()
    g0 = np.asarray(grad_loglike_one(theta_true))
    print("||grad loglike(theta_true)||_inf = {0:.3e}   ({1:.2f}s)".format(
        float(np.max(np.abs(g0))), time.time() - t0,
    ))

    theta_off = theta_true.at[1].add(1.0 / T * 0.05)  # 5% of a bin
    ll_off = float(loglike_one(theta_off))
    g_off = np.asarray(grad_loglike_one(theta_off))
    print("\nlog L(theta + 0.05/T) = {0:+.3e}".format(ll_off))
    print("Per-parameter gradient at the offset point:")
    for name, gi in zip(PARAM_NAMES, g_off):
        print("  d/d{0:<6s} = {1:+.6e}".format(name, gi))

    # ----- FD vs autodiff sanity check on phi0 ---------------------------
    eps = 1e-4
    idx = PARAM_NAMES.index("phi0")
    fd = (
        float(loglike_one(theta_off.at[idx].add(+eps)))
        - float(loglike_one(theta_off.at[idx].add(-eps)))
    ) / (2.0 * eps)
    print("\nFinite-diff d/dphi0 at theta_off = {0:+.6e}".format(fd))
    print("Autodiff    d/dphi0 at theta_off = {0:+.6e}".format(g_off[idx]))

    # ====================================================================
    # Tempered NUTS MCMC
    # ====================================================================
    print("\n" + "=" * 70)
    print(" Tempered NUTS MCMC on the JAX TDI-on-the-fly GB likelihood")
    print("=" * 70)

    # ----- Eryn-side vectorized likelihood -------------------------------
    # We use ``analysis.eryn_likelihood_wrap(..., use_vmap=True)`` as
    # the Eryn log-like function. The AnalysisContainer drives the
    # computation: per-walker signal generation (via ``signal_gen``)
    # and ``template_likelihood`` are batched together through
    # :func:`jax.vmap`. NUTS' per-walker gradient is built the same
    # way -- a jax.grad on the single-walker loglike then vmap'd.
    def log_like_eryn(theta_batch, *_args):
        return np.asarray(
            analysis.eryn_likelihood_wrap(
                jnp.asarray(theta_batch, dtype=jnp.float64), use_vmap=True,
            )
        )

    grad_batch_jax = jax.vmap(grad_loglike_one)

    def grad_log_like_eryn(theta_batch):
        """Gradient callable for :class:`NUTSMove`: per-walker grad of loglike_one."""
        return np.asarray(grad_batch_jax(jnp.asarray(theta_batch, dtype=jnp.float64)))

    # ----- priors --------------------------------------------------------
    # Uniform priors centred on theta_true with widths chosen so the
    # injection is well inside the support but the chain still has room
    # to move.
    half_widths = np.array([
        1.0e-22,    # amp
        1.0e-7,     # f0   (about a few FFT bins for T=1 hour)
        1.0e-16,    # fdot
        1.0e-22,    # fddot
        np.pi,      # phi0
        np.pi / 2.0,  # iota
        np.pi,      # psi
        np.pi,      # lam
        np.pi / 2.0,  # beta
    ])
    theta_true_np = np.asarray(theta_true)
    lo = theta_true_np - half_widths
    hi = theta_true_np + half_widths

    priors = ProbDistContainer({i: uniform_dist(float(lo[i]), float(hi[i])) for i in range(N_PARAMS)})

    # ----- per-dimension scaling -----------------------------------------
    # The GB parameters span ~22 orders of magnitude (``amp ~ 1e-22`` vs
    # ``phi0 ~ 1``) -- NUTS would be unusable with an identity metric.
    # ``scale=`` builds ``metric = diag(1/scale**2)`` for us, so the
    # natural NUTS step in dimension ``i`` is ``scale[i]``. Using the
    # prior half-widths as the scale is a robust default.
    nuts_scale = half_widths.copy()

    # ----- temperature ladder --------------------------------------------
    # The TDI-on-the-fly graph is eager (no jit) so each gradient eval is
    # several seconds; we deliberately keep ntemps, nwalkers, tree depth
    # and step count tiny so the demo finishes in minutes, not hours.
    betas = np.array([1.0, 0.3])
    ntemps = len(betas)

    # ----- walkers -------------------------------------------------------
    nwalkers = 2
    # Initialize tightly around theta_true. Clip per-parameter using
    # ``half_widths`` so the nudge is scaled to each parameter rather
    # than using an absolute floor that wrecks tiny-scale params like
    # ``amp``.
    scatter = 0.05 * half_widths
    coords = theta_true_np + scatter * np.random.RandomState(0).randn(
        ntemps, nwalkers, N_PARAMS,
    )
    safe_eps = 1e-4 * half_widths
    coords = np.clip(coords, lo + safe_eps, hi - safe_eps)

    nuts = NUTSMove(
        grad_log_like_fn=grad_log_like_eryn,
        ndim=N_PARAMS,
        scale=nuts_scale,
        step_size=0.6,
        max_tree_depth=2,   # 2^2 = 4 leapfrog steps per proposal
    )

    ensemble = EnsembleSampler(
        nwalkers,
        N_PARAMS,
        log_like_eryn,
        priors,
        moves=nuts,
        tempering_kwargs={"betas": betas},
        vectorize=True,
    )

    nsteps = 4
    burn = 1
    print(
        "Sampler: nwalkers={0}, ntemps={1}, nsteps={2} (burn {3}), "
        "step_size={4}".format(nwalkers, ntemps, nsteps, burn, nuts.step_size)
    )
    t0 = time.time()
    state = State(coords)
    last_sample = ensemble.run_mcmc(state, nsteps, burn=burn, progress=False)
    print("MCMC walltime: {0:.1f}s".format(time.time() - t0))
    print("NUTS acceptance fraction (per-temp mean): {0}".format(
        np.mean(ensemble.acceptance_fraction, axis=-1)
    ))

    chain = ensemble.get_chain()["model_0"]   # (nsteps, ntemps, nwalkers, 1, ndim)
    chain_cold = chain[:, 0].reshape(-1, N_PARAMS)
    print("\nCold-chain posterior summary (mean +/- std over kept samples):")
    for i, name in enumerate(PARAM_NAMES):
        mean_i = float(np.mean(chain_cold[:, i]))
        std_i = float(np.std(chain_cold[:, i]))
        true_i = float(theta_true_np[i])
        print(
            "  {0:<6s}: true={1:+.4e}  mean={2:+.4e}  std={3:.3e}".format(
                name, true_i, mean_i, std_i,
            )
        )


if __name__ == "__main__":
    main()
