# Direct Fast Fresnel Response for the STFT galactic-binary likelihood

This folder documents the integration of the Direct Fast Fresnel Response (DFFR) into the
short-time Fourier transform (STFT) likelihood of galactic binaries in lisaanalysistools (LAT)
and GBGPU. The work is in progress on the branch `feat-dffr` of both repositories. Nothing
described here can be selected yet.

| Document | Content |
| --- | --- |
| `README.md` | What DFFR is, why it is integrated, where its pieces will live and how it will be selected. |
| [`plan.md`](plan.md) | The plan, with the status of each step. |
| [`tests.md`](tests.md) | The tests, regression tests and validation runs of each step, with their results and the commands that reproduce them. |

The scripts behind `tests.md` are in [`scripts/dffr/`](../../../scripts/dffr/).

## The response today and with DFFR

`STFTGBComputations` (GBGPU, `gbgpu/gbcomps.py`) evaluates the likelihood of galactic binaries
on an STFT grid of one-day segments. Its kernels sample the TDI response on the fly three times
per segment and channel, and evaluate a windowed Fresnel transform of seven terms per pixel.

DFFR computes the same pixels in another way. Within one segment, the signal that reaches each
spacecraft is a linear chirp under a slowly varying envelope, so the windowed transform of the
segment has a closed form in Fresnel integrals. DFFR evaluates the three spacecraft reception
states of each pixel from a table of spacecraft kinematics at the segment midpoints, which does
not depend on the source, and folds them with eighteen TDI weights per segment. The cost per
frequency bin does not depend on the sampling rate of an underlying time series.

## Why integrate it

Measured with the standalone package, on its CPU backend unless stated. The first row was
measured for this integration and is recorded in `tests.md`; the others come from the package's
validation document.

| Comparison | Result |
| --- | --- |
| Against a time-domain TDI 2.0 truth built with lisagwresponse, lisainstrument (every noise off) and pytdi: 30 one-day segments on the ESA orbits that LAT ships, four sources from 3 to 40 mHz at two sky positions each, Tukey window 0.116 | mismatch 6.5e-10 to 1.2e-9; median error per segment 3.2e-5 to 4.7e-5 |
| Time per source on one H100 NVL, a year of one-day segments, a band of 41 bins, 1000 to 10 000 sources | template 6.5 µs; likelihood 8.7 to 10 µs |
| CUDA backend against CPU backend, same inputs | at most 6.8e-14 relative |

The aim of the integration is a second response of `STFTGBComputations`, selected with one flag
and built into the existing STFT infrastructure: computation groups, `STFTDomain`, window starts,
per-GPU replicas and the likelihood kernels. It must lose neither accuracy nor speed against
today's response, and its source model must be able to carry higher-order modes, as the
TDI-on-the-fly family does.

## Where the pieces will live

The pieces of steps 1 and 2 exist on `feat-dffr`; the others are planned. The final layout is recorded
here once the steps in `plan.md` are done.

| Repository | Piece | Content | Status |
| --- | --- | --- | --- |
| LAT | `cutils/Detector.{hpp,cu}` | Five-point derivatives of the orbit tables (`get_velocity`, `get_acceleration`, `get_ltt_rate`, `get_ltt_acceleration`) and `Orbits::get_kinematics_table`, which fills the source-independent kinematics table at the segment midpoints. The existing response takes its spacecraft acceleration from `get_acceleration`. | built (step 1) |
| LAT | `cutils/binding_detector.{hpp,cxx}` | `KinematicsTableWrap`, the table over arrays owned by Python. | built (step 1) |
| LAT | `domaincomputation.py` | `STFTComputationGroup.cpp_kinematics`: builds the table on the group's device on first use and keeps it. | built (step 1) |
| LAT | `cutils/lat_fresnel.hh` | The Fresnel integrals, their auxiliary functions and the derivatives of these, with DFFR's numerics, shared by `STFTFresnel` and DFFR. The header records its branches, their accuracy and its assumptions. `STFTFresnelWrap.compute_fresnel_functions` evaluates it for tests. | built (step 2) |
| LAT | `cutils/lat_stft_kernels.hh` | Likelihood kernels written against a tile of pixels, so that both responses feed the same consumers. | planned |
| LAT | `cutils/lat_dffr_*.hh` | DFFR settings, source-model contract, segment table, TDI weights, windowed transform, DFFR tile and Fisher kernel. | planned |
| GBGPU | `cutils/gb_dffr_source.hh` | The galactic-binary source model of DFFR, sharing its phase and amplitude formulas with `GBTDIonTheFly`. | planned |
| GBGPU | `gbcomps.py` | The `response` flag of `STFTGBComputations` and the dispatch to the DFFR kernels. | planned |

## How it will be selected

Planned for step 5 of `plan.md`:

```python
comp = STFTGBComputations(stft_comps, T, ..., response="dffr")  # default: "tdi_on_the_fly"
```

DFFR adds two settings, `first_order_delays` and `chirp_rate_floor`. It supports the XYZ channels
only and raises on an AET domain. Its gradients and Fisher matrix follow the `linear_envelope`
setting of the computation group.

## The standalone package

DFFR was developed and validated as a standalone package:
<https://github.com/coen-rondeel/direct-fast-fresnel-response>. The integration ports its
compiled kernels as of commit `01530a5`.

- [Theory and derivation](https://github.com/coen-rondeel/direct-fast-fresnel-response/blob/main/docs/theory.md)
- [Conventions](https://github.com/coen-rondeel/direct-fast-fresnel-response/blob/main/docs/conventions.md):
  source parameters, time frames, STFT and likelihood conventions, and the mapping to
  lisagwresponse and lisatools.
- [Validation](https://github.com/coen-rondeel/direct-fast-fresnel-response/blob/main/docs/validation.md):
  the tests and comparisons behind the timing and backend numbers above.

Author: Coen Rondeel.
