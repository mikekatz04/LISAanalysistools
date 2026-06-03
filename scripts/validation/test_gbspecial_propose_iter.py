#!/usr/bin/env python
"""CPU smoke test: call GBSpecialRJPriorMove.propose() once.

This script is the next step up from test_gbspecial_engine_iter.py: instead of
exercising the WDM likelihood engine in isolation, it constructs the minimum
fixture needed to invoke ``GBSpecialRJPriorMove.propose(model, state)`` once,
on the CPU backend.

The goal is to drive the propose() path through far enough to find the next
CPU/cupy incompatibility, fix it, and re-run.  Errors are expected; the script
is meant to be re-run repeatedly as the move is made CPU-clean.

Run from the repo root:

    /Users/mkatz/miniconda3/envs/deving/bin/python test_gbspecial_propose_iter.py
"""

from __future__ import annotations

import os
import sys
import traceback
from copy import deepcopy

import numpy as np

from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_N, get_fdot

from eryn.moves.tempering import TemperatureControl
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State as ErynState
from eryn.utils import PeriodicContainer, TransformContainer

from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.datacontainer import DataResidualArray
from lisatools.domains import FDSettings
from lisatools.sensitivity import AE1SensitivityMatrix
from lisatools.globalfit.engine import GlobalFitInfo
from lisatools.globalfit.moves.gbspecialstretch import GBSpecialRJPriorMove
from lisatools.globalfit.state import GBState, GFState
from lisatools.utils.constants import YRSID_SI


BACKEND = "cpu"
NTEMPS = 2
NWALKERS = 4
NLEAVES_MAX = 4
NDIM = 8


def f_ms_to_s(x):
    return x * 1e-3


def _build_gb_transform() -> TransformContainer:
    input_basis = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]
    output_basis = ["A", "f0", "fdot", "fddot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]
    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms={
            "A": np.exp,
            "f0": f_ms_to_s,
            "cos_iota": np.arccos,
            "sin_beta": np.arcsin,
        },
        fill_dict={"fddot": 0.0},
    )


def _build_gb_priors():
    f0_lo, f0_hi = 2.9, 3.1            # mHz
    A_lo, A_hi = 1e-23, 1e-20
    fdot_lo, fdot_hi = -1e-13, 1e-13
    priors_in = {
        0: uniform_dist(np.log(A_lo), np.log(A_hi)),
        1: uniform_dist(f0_lo, f0_hi),     # f0 in mHz before transform
        2: uniform_dist(fdot_lo, fdot_hi),
        3: uniform_dist(0.0, 2 * np.pi),
        4: uniform_dist(-1, 1),
        5: uniform_dist(0.0, np.pi),
        6: uniform_dist(0.0, 2 * np.pi),
        7: uniform_dist(-1, 1),
    }
    return {"gb": ProbDistContainer(priors_in)}


def _build_band_structure(start_freq: float, end_freq: float, Tobs: float, df: float,
                          oversample: int = 4, extra_buffer: int = 5, N_fixed: int = 128):
    """Hand-rolled band structure with fixed N (avoids get_N which currently
    needs a DomainSettings object the smoke fixture doesn't have)."""
    band_edges_rev = [end_freq]
    band_N_rev = [N_fixed]
    current_freq = end_freq
    last_freq = end_freq
    while current_freq > start_freq:
        current_freq = last_freq - (N_fixed * 2 + extra_buffer) * df
        band_edges_rev.append(current_freq)
        band_N_rev.append(N_fixed)
        last_freq = current_freq
    band_edges_rev.append(last_freq - (N_fixed * 2 + extra_buffer) * df)
    band_edges = np.asarray(band_edges_rev)[::-1]
    band_N_vals = np.asarray(band_N_rev)[::-1]
    return band_edges, band_N_vals


def _build_fd_ac(fd: np.ndarray, df: float):
    """Two-channel (A,E) frequency-domain analysis container on CPU.

    Note: FDSettings owns the full ``np.arange(N) * df`` grid; ``min_freq``
    only marks the active slice. So ``N`` is sized to cover [0, fd[-1]] and
    the active slice maps to our (shorter) fd window.
    """
    nchan = 2
    N_full = int(np.ceil(fd[-1] / df)) + 1
    data_full = np.zeros((nchan, N_full), dtype=np.complex128)
    ind_lo = int(np.round(fd[0] / df))
    fd_settings = FDSettings(
        N=N_full, df=df,
        min_freq=float(fd[0]), max_freq=float(fd[-1]),
        force_backend=BACKEND,
    )
    drs = DataResidualArray(data_full, input_signal_domain=fd_settings)
    # DataResidualArray.__init__ doesn't auto-call _store_time_and_frequency_information;
    # f_arr/df/dt accessors below assume those caches are populated. Set them by hand
    # so move.propose() (which reads acs.f_arr) works.
    drs._f_arr = fd_settings.f_arr
    drs._df = df
    drs._dt = None
    drs._Tobs = None
    drs._fmax = float(fd_settings.f_arr.max())
    sens = AE1SensitivityMatrix(drs.settings)
    return AnalysisContainer(drs, sens), N_full, ind_lo


def main() -> int:
    np.random.seed(0)

    # ------------------------------------------------------------------
    # Common scales (kept small for a CPU iteration)
    # ------------------------------------------------------------------
    dt = 10.0
    Tobs = 0.25 * YRSID_SI                  # ~3 months
    df = 1.0 / Tobs
    f0_lo = 2.9e-3
    f0_hi = 3.1e-3
    n_pad = 1024
    data_length = int(np.ceil((f0_hi - f0_lo) / df)) + 2 * n_pad
    f_start = f0_lo - n_pad * df
    fd = f_start + np.arange(data_length) * df
    print(f"[setup] dt={dt}, Tobs={Tobs:.3e}, df={df:.3e}, data_length={data_length}, fd[0..-1]=({fd[0]:.6e},{fd[-1]:.6e})")

    # ------------------------------------------------------------------
    # GBGPU on CPU
    # ------------------------------------------------------------------
    gb = GBGPU(force_backend=BACKEND)
    gb.gpus = None  # CPU path: no GPU device list
    print(f"[setup] GBGPU backend={gb.backend.name}, xp={gb.backend.xp.__name__}")

    # ------------------------------------------------------------------
    # Band structure (covers our fd window)
    # ------------------------------------------------------------------
    # Hand-rolled bands inside [fd[0], fd[-1]] so the buffer start indices stay positive
    # relative to the acs start_freq_ind, AND each band's buffer (max_data_store_size
    # wide) fits within acs.data_length once shifted by buffer_start_index.
    N_per_band = 64
    band_width = (2 * N_per_band + 5) * df
    n_bands = 4
    band_edges = fd[0] + n_pad // 2 * df + np.arange(n_bands + 1) * band_width
    band_N_vals = np.full(n_bands, N_per_band)
    print(f"[setup] band_edges shape={band_edges.shape}, band_N_vals shape={band_N_vals.shape}")

    # ------------------------------------------------------------------
    # Priors, transform, waveform kwargs
    # ------------------------------------------------------------------
    priors = _build_gb_priors()
    gpu_priors = priors  # same on CPU
    transform = _build_gb_transform()
    waveform_kwargs = dict(dt=dt, T=Tobs, tdi_channel_setup="AE", use_c_implementation=True)

    # ------------------------------------------------------------------
    # AnalysisContainerArray (single AC, FD path, gpus=None)
    # ------------------------------------------------------------------
    # One AC per walker (per-walker residual buffer is what the move's
    # walker-indexed kernel reads). The move drives only the cold-chain ACs,
    # so we replicate to NWALKERS entries (one per cold walker).
    acs_list = []
    for _ in range(NWALKERS):
        ac_i, N_full, ind_lo = _build_fd_ac(fd, df)
        acs_list.append(ac_i)
    acs = AnalysisContainerArray(acs_list, gpus=None)
    print(f"[setup] acs.data_length={acs.data_length}, acs.nchannels={acs.nchannels}, acs.acs_total_entries={acs.acs_total_entries}, acs.gpus={acs.gpus}")
    fd_full = np.arange(N_full) * df

    # ------------------------------------------------------------------
    # Initial coords / inds / log_like for the GB branch
    # All zeros (no sources yet -> RJ birth proposals).
    # ------------------------------------------------------------------
    gb_coords = np.zeros((NTEMPS, NWALKERS, NLEAVES_MAX, NDIM))
    gb_inds = np.zeros((NTEMPS, NWALKERS, NLEAVES_MAX), dtype=bool)
    log_like = np.zeros((NTEMPS, NWALKERS))
    log_prior = np.zeros((NTEMPS, NWALKERS))
    eryn_state = ErynState(
        {"gb": gb_coords},
        inds={"gb": gb_inds},
        log_like=log_like,
        log_prior=log_prior,
    )

    state = GFState(
        eryn_state,
        is_eryn_state_input=True,
        sub_state_bases={"gb": GBState},
    )

    # Initialize band_info on the GB sub-state.
    betas = np.array([1.0, 0.5])[:NTEMPS]
    band_temps = np.tile(betas, (len(band_edges) - 1, 1))
    state.sub_states["gb"].initialize_band_information(NWALKERS, NTEMPS, band_edges, band_temps)
    print(f"[setup] band_info initialized, band_temps shape={band_temps.shape}")

    # ------------------------------------------------------------------
    # Build the move
    # ------------------------------------------------------------------
    move = GBSpecialRJPriorMove(
        gb, priors,
        ind_lo,                      # start_freq_ind (where active band begins in full grid)
        acs.data_length,
        acs,                         # mgh
        fd_full,
        band_edges,
        band_N_vals,
        gpu_priors,
        rj_proposal_distribution=gpu_priors,
        max_data_store_size=512,
        name="rj_prior_cpu_smoke",
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform,
        is_rj_prop=True,
        run_swaps=True,
        nfriends=NWALKERS,
        force_backend=BACKEND,
        provide_betas=True,
        random_seed=0,
        periodic=PeriodicContainer({"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}}),
    )
    move.temperature_control = TemperatureControl(NDIM, NWALKERS, betas=betas)
    move.time = 0
    print(f"[ok] Move constructed. backend.uses_cupy={move.backend.uses_cupy}")

    # ------------------------------------------------------------------
    # Build the model (eryn-style namedtuple)
    # ------------------------------------------------------------------
    model = GlobalFitInfo(
        analysis_container_arr=acs,
        map_fn=map,
        random=np.random.RandomState(0),
    )

    # ------------------------------------------------------------------
    # Run propose() once
    # ------------------------------------------------------------------
    print("\n[run] Calling move.propose(model, state)...")
    try:
        new_state, accepted = move.propose(model, state)
    except Exception as exc:
        print("[FAIL] propose() raised:")
        traceback.print_exc()
        return 1

    print("[ok] propose() returned without error.")
    print(f"     accepted shape={getattr(accepted, 'shape', None)} sum={int(np.asarray(accepted).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
