# Direct Fast Fresnel Response for the STFT galactic-binary likelihood

This folder documents the integration of the Direct Fast Fresnel Response (DFFR) into the
short-time Fourier transform (STFT) likelihood of galactic binaries in lisaanalysistools (LAT)
and GBGPU. The work is in progress on the branch `feat-dffr` of both repositories. Since step 5,
`STFTGBComputations(..., response="dffr")` selects DFFR (see "How to select it" below), and since
part 4c it has a Fisher matrix (see "The Fisher matrix" below). Since 2026-10-20 every step has run
on a full A100 as well; the validation of step 6 is done, and the review of step 7 follows.

| Document | Content |
| --- | --- |
| `README.md` | What DFFR is, why it is integrated, where its pieces live and how to select it. |
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

## Where the pieces live

The pieces of steps 1 to 5 and of part 4c, the DFFR Fisher matrix, exist on `feat-dffr`. The
final layout is recorded here once the steps in `plan.md` are done.

| Repository | Piece | Content | Status |
| --- | --- | --- | --- |
| LAT | `cutils/Detector.{hpp,cu}` | Five-point derivatives of the orbit tables (`get_velocity`, `get_acceleration`, `get_ltt_rate`, `get_ltt_acceleration`) and `Orbits::get_kinematics_table`, which fills the source-independent kinematics table at the segment midpoints. The existing response takes its spacecraft acceleration from `get_acceleration`. | built (step 1) |
| LAT | `cutils/binding_detector.{hpp,cxx}` | `KinematicsTableWrap`, the table over arrays owned by Python. | built (step 1) |
| LAT | `domaincomputation.py` | `STFTComputationGroup.cpp_kinematics`: builds the table on the group's device on first use and keeps it. | built (step 1) |
| LAT | `cutils/lat_fresnel.hh` | The Fresnel integrals, their auxiliary functions and the derivatives of these, with DFFR's numerics, shared by `STFTFresnel` and DFFR. The header records its branches, their accuracy and its assumptions. `STFTFresnelWrap.compute_fresnel_functions` evaluates it for tests. | built (step 2) |
| LAT | `cutils/lat_stft_kernels.hh` | Likelihood kernels written against a tile of pixels, so that both responses feed the same consumers (see "The tile seam" below). `ColumnTile` feeds them the existing response; the Fisher kernel keeps its own loop. Step 4 adds the band-bin-first pixel map and the launch path above 48 kB of shared memory. | built (steps 3 and 4) |
| LAT | `cutils/lat_dffr_kernels.hh` | DFFR settings, source-model contract, windowed transform, closed-form TDI weights of both generations, segment table, the DFFR tile, two diagnostics of a batch of sources (compact templates, segment table), and the DFFR Fisher matrix. The header records the definitions, conventions, accuracy and assumptions. | built (step 4, Fisher in part 4c) |
| LAT | `cutils/lat_dffr_diagnostics.{hh,cu}` | Three source-independent diagnostics for `tests/test_dffr.py`: the bounded sub-interval integral, the windowed transform and the closed-form TDI coefficients, as methods of `STFTFresnelWrap` and `KinematicsTableWrap`. | built (step 4); removed in step 6 (validation item 6), a copy kept outside the repository |
| LAT | `response/tdiconfig.py` | `TDIConfig.generation`: 1 or 2 for the stock combinations, `None` otherwise. | built (step 4) |
| GBGPU | `cutils/gb_physics.hh` | The GB phase, frequency and amplitude formulas, shared by `GBTDIonTheFly` and the DFFR source model. | built (step 4) |
| GBGPU | `cutils/gb_dffr_source.hh` | `GBDFFRSource`, the galactic-binary source model of DFFR (see "Source model and components" below), with the parameters that leave the reception states unchanged. | built (step 4, that list in part 4c) |
| GBGPU | `cutils/gb_dffr.{hh,cu}` | The DFFR entry points, bound as the `gb_dffr_*` methods of `GBComputationGroupWrap` next to the `gb_stft_*` methods of the existing response. | built (step 4, `gb_dffr_information_matrix` in part 4c) |
| GBGPU | `gbcomps.py` | The `response` flag of `STFTGBComputations`, the DFFR settings and the dispatch of its six likelihood methods and of `information_matrix` to the DFFR kernels. | built (step 5, Fisher in part 4c) |
| LAT | `globalfit/recipe.py` | Optional `response`, `first_order_delays` and `chirp_rate_floor` in `gb_stft_kwargs`. | built (step 5) |

## The tile seam

The likelihood kernels of `lat_stft_kernels.hh` are split into a producer of pixel values, the
tile, and the consumers that use them: `get_ll`, `fill_global`, `swap_ll`, the F-statistic and
the two gradients. A consumer runs over the STFT segments of a source in passes. It keeps the
window mask of its cell, the factor 0.5 of the real signal, the fill factor and every sum, and
asks the tile for the values of the three channels at each pixel. A second response is then a
second tile, and the consumers do not change.

A tile type provides:

| Member | Role |
| --- | --- |
| `Inputs` | The launch inputs of the response, a plain struct that the kernel receives by value and the consumers pass on to `Template::build`. |
| `BlockShared` | The response's static shared memory in a block. |
| `table_bytes()` | The dynamic shared memory of one template. A launch reserves it once per template: twice for `swap_ll`, its gradient and the F-statistic, which hold two templates. |
| `Map` | The pixel map: which thread evaluates which pixel of a pass. |
| `Context` | Built by every thread at the start of a kernel; for the existing response, the source object and the link tables of the spacecraft. `rows_per_pass()` gives the number of segments built per pass. |
| `Template` | One template. `load` reads the source parameters once per evaluation, so that the gradients and the F-statistic can change them between evaluations; `build` prepares the segments of a pass; `carrier_bin` places the band of a segment, the carrier bin ± `n_side_bins`; `values` returns the three channels at one pixel. `swap_ll` sums both templates over the union of their two bands. |

`ColumnTile<ColumnT>` adapts the column policy of the existing response, `FresnelColumn`, to this
contract. A pass holds one segment per thread, and the map gives each thread the band of its own
segment, walked in increasing frequency. Each thread thus evaluates the same segments, bins and
channels in the same order as the per-column kernels did, with the same calls and arguments, and
the outputs stay bit-identical: on the CPU every output of the seven kernels equals that of the
step 2 code bit for bit; the GPU comparison waits for a GPU node. The seven entry points
`stft_*_impl<SourceT>` keep their signatures and use this tile by default, so GBGPU compiles
unchanged. The Fisher kernel keeps its own loop over the column policy: it shares one Fresnel
kernel between perturbed templates that have the same chirp, which needs the split of a pixel
value into a prefactor and a kernel that a tile does not expose. The DFFR Fisher matrix is
likewise a kernel of its own (see "The Fisher matrix" below).

DFFR builds its segment tables in shared memory, 32 segments per pass and with all threads of a
block, and maps one band bin to each warp. The 32 is the warp size of NVIDIA GPUs: with 32
segments per pass, the 32 threads of a warp take the 32 segments of one band bin. Which branch of
the Fresnel functions an endpoint takes depends on its distance from the carrier, which is about
the same for one band bin in every segment, so one band bin per warp keeps that choice the same
within a warp. The table fits as well: one segment row holds 1012 bytes (three reception states
of five complex numbers and six doubles, six links of three complex numbers and one double,
eighteen complex TDI weights and one integer), so 32 rows take 32.4 kB, under the 48 kB that a
launch gets without raising its limit on dynamic shared memory. Two tables, as `swap_ll` needs,
take 64.8 kB. Step 4 built that second pixel map, `StftBandBinMap`, and the launch path above 48
kB, which raises a kernel's limit within what the device allows a block. One DFFR template, the
table with the source constants and the settings, takes 32 672 bytes at 32 rows. The consumers
that hold two templates are built for 32 rows and for 16. `STFTGBComputations` uses 32: on a full
A100, 16 rows are 5 % to 17 % slower at 10 000 sources (`tests.md`, step 6).

## Source model and components

`GBDFFRSource` (GBGPU) is the galactic-binary source model of DFFR. Per source it loads the
constants once: the sky basis and polarisation, the amplitude, the frequency and its two
derivatives, the initial phase, and the plus and cross weights of the inclination. Per component it
gives the quadratic phase expansion at a retarded time, the amplitude ratio and its log-slope. The
phase, frequency and amplitude formulas are those of `GBTDIonTheFly`, from `gb_physics.hh`.

The contract speaks in components, so that a source with several harmonics fits it: each
component would carry its own phase, amplitude and polarisation weights, and the components of
one source would add in each pixel, so that `(h|h)` keeps their cross terms. The DFFR tile
evaluates component 0 only and requires a source with one component. The reason is the band. A
template covers `2 n_side_bins + 1` bins about one carrier, while the harmonics of a binary lie
f0 / df bins apart: 86 bins at 1 mHz and 864 at 10 mHz on one-day segments. Each harmonic needs a
band of its own, so it belongs in a template of its own.

## How to select it

```python
from gbgpu.gbcomps import STFTGBComputations

comp = STFTGBComputations(
    stft_comps, T, t_ref=t_ref, orbits=orbits, tdi_config="2nd generation",
    n_side_bins=10, response="dffr",  # default: "tdi_on_the_fly"
)
```

In a global-fit recipe, `gb_stft_kwargs` takes the same three optional keys, `response`,
`first_order_delays` and `chirp_rate_floor`. Without them the existing response runs, as before.
The comp's methods, outputs and attributes are the same for both responses, so the band engine
and the device replicas need no change; a replica keeps the response and its settings.

| Setting | Default | Meaning for DFFR |
| --- | --- | --- |
| `response` | `"tdi_on_the_fly"` | `"dffr"` selects DFFR. |
| `first_order_delays` | `True` | Carries every light travel time to first order across a segment. |
| `chirp_rate_floor` | `1e-22` Hz/s | Sign-preserving floor on the local chirp rate of a reception state; the default is the floor of the existing response. It changes the phase by at most π floor (dt / 2)²: 6e-13 rad on a one-day segment at the default. |

What DFFR reads from the bound computation group:

- the taper `window_alpha` of its `STFTFresnel`, and the grid, data and inverse covariance of its
  `STFTDomain`;
- its `linear_envelope`;
- its kinematics table. The group builds the table from the comp's orbits on the first DFFR call
  and keeps it (`STFTComputationGroup.cpp_kinematics`). Every GPU split of the residual container
  and of each band buffer has a group of its own, so each builds its table once.

Settings that DFFR does not read:

- `T`;
- `freq_from_tdi_phase`, which picks how the existing response estimates its chirp, and for the
  same reason the `astro_track` argument of `information_matrix`;
- `use_midpoint`: DFFR always expands the signal about the segment midpoint;
- `window_factor`. This setting scales the existing response's templates when `window_alpha` is
  0, that is, when its model assumes a boxcar segment. It approximates the power that a taper of
  the data removes, by the mean window weight. DFFR models the window exactly through
  `window_alpha`, so tapered data need `window_alpha`, not this factor. With `window_alpha` 0 and
  a `window_factor` other than 1, the two responses differ by that factor.

Limits:

- XYZ channels only: a group or domain of another TDI set raises `ValueError`.
- The combinations of `TDIConfig("1st generation")` or `("2nd generation")` only, in any order;
  others raise `ValueError`.
- The Fisher matrix uses 4 segment rows per pass on the GPU; the CPU uses one.
- The two-template kernels (swap, its gradient, the F-statistic) use 32 segment rows per pass on
  the GPU; the CPU uses one.

The linear envelope. The likelihood, swap, F-statistic and fill methods follow the group's
`linear_envelope`. The gradients take `linear_envelope=RESPONSE_DEFAULT`: the group's setting for
DFFR, and `False` for the existing response, as before. The existing response turns the envelope
off in its derivatives because its slope comes from its TDI stencil, whose rounding would enter
the differences; DFFR's slopes are closed forms in the source parameters. Measured in step 5 on
three production fixtures: with the envelope on, DFFR's finite-difference noise in f0, ḟ, f̈, φ0,
λ and β is unchanged, to two digits (`tests.md`). `True`, `False` and `None` (the group's
setting) apply to both responses.

## Accuracy against a time-domain truth

Measured in step 6 (`tests.md`, item 1): both responses of `STFTGBComputations` against the STFT of
a time-domain truth (lisagwresponse and pytdi, X2, Y2 and Z2 every 5 s) over a year of one-day
segments on the ESA orbits. Eight sources at 0.5 and 20 mHz: at each frequency the two ḟ bounds of
the global fit's GB prior, each at two sky positions. Production settings but 60 side bins, and
data windowed with the periodic Tukey window. Ranges over the four sources of each frequency.

| | 0.5 mHz, DFFR | 0.5 mHz, existing | 20 mHz, DFFR | 20 mHz, existing |
| --- | --- | --- | --- | --- |
| Flat mismatch, template pixels | 2.7e-8 | 2.7e-8 | 4.8e-10 to 6.8e-10 | 7.6e-10 to 1.2e-9 |
| Median error per segment | 1.0e-4 | 1.0e-4 | 2.2e-5 to 3.2e-5 | 1.4e-5 to 1.9e-5 |
| Largest error per segment | 2.5e-4 to 9.0e-4 | 3.2e-4 to 6.2e-3 | 4.0e-4 to 4.7e-4 | 4.1e-3 to 4.6e-3 |
| Noise-weighted mismatch | 1.3e-7 | 7.2e-6 to 1.4e-5 | 2.4e-8 to 3.6e-8 | 2.5e-8 to 3.7e-8 |

- On the eight cases of the standalone package's truth comparison (the first row of the table in
  "Why integrate it"), with its chain and grid over 30 days, DFFR in `STFTGBComputations`
  reproduces those numbers to two or three digits (`tests.md`, step 6, item 2).
- The error per segment is max |h - d| / max |d| per segment and channel. The largest errors fall
  where one channel's signal is weak.
- At 0.5 mHz both responses share a floor from the signal's negative-frequency image, which
  neither computes and which 60 side bins reach near the start of the production band. The
  existing response's larger noise-weighted mismatch there comes from the combination X + Y + Z,
  which the likelihood weights heavily at low frequency.
- The global fit windows the data with LAT's symmetric Tukey window, while both responses
  integrate the periodic one. That adds a mismatch of 3.6e-8 and 6.3e-5 per segment. Production
  keeps it, because its 10 side bins leave 1.3e-3 to 1.7e-3 of the power outside the template, a
  mismatch of 6.4e-4 to 8.8e-4 (`plan.md`, Risks and observations).

Likelihood surfaces against the same truth (`tests.md`, step 6, item 3): the eight sources at an
SNR of 100, each of four waveforms in its own domain with X, Y, Z and the full CSD, one parameter
scanned at a time over ±4σ of DFFR's Fisher matrix. DFFR puts every peak within 0.002σ of the
truth, with widths equal to its Fisher matrix to three digits, and loses at most 1.3e-3 in ln L at
the truth. TDI on the fly stays within 0.010σ in the STFT and 0.016σ in the frequency domain
(`GBFDComputations`). FastGB is off by up to 2.2σ at 20 mHz and 6.6σ at 0.5 mHz, where it lacks
the combination X + Y + Z.

On the Mojito-light data (`tests.md`, step 6, item 4): the five galactic binaries of highest
frequency (15.6 to 20.4 mHz) at the catalogue's amplitudes (SNR 67 to 223), the galaxy and the
instrument noise preprocessed as the global fit does over 1.92 years, each of six waveforms in its
own domain. Against the noiseless galaxy with each source's neighbours removed, DFFR reaches a
mismatch of 3.1e-8 to 5.8e-8 (up to 4.5e-7 beside a louder binary), like TDI on the fly in the
STFT, and puts every peak in f0, ḟ, φ0, ψ, λ and β within 0.003σ of the truth. A and ι move by
up to 0.18σ: the preprocessing's low-pass filter lowers the data by 0.07 % at 20 mHz, and the
Mojito binaries have no amplitude evolution. FastGB and jaxgb's JaxGB and JaxGBaccurate miss by
6.8e-4 to 2.5e-3 and lie up to 1.3σ off, because they form TDI with one arm length; jaxgb's link
responses, combined by pytdi with the time-varying light travel times as the Mojito pipeline
does, reproduce the data to rounding.

In the global fit (`tests.md`, step 6, item 5): the galactic-binary block runs on the same
Mojito-light data, over the 713 days the preprocessing leaves, once with each response, for the
56 catalogue binaries from 9.6 to 20.4 mHz, started at their catalogue values, with 32 walkers and
24 temperatures on one A100 each, for 1.45 days. With identical settings an iteration takes 17
minutes with DFFR and 32 with TDI on the fly; DFFR evaluates the in-model likelihoods 3.4 times
faster. Over the 67 iterations both runs reached, the two give the same posteriors within the
sampling noise of the chains: widths within 1 % at the median, means within 0.24σ. The walkers,
all started at the injection, were still spreading in f0, ḟ, φ0 and ψ when the runs stopped, a
property of the GB move, so these parameters are not yet converged posteriors.

## The Fisher matrix

`information_matrix` of `STFTGBComputations(response="dffr")` returns the Fisher matrices of the
DFFR templates, F_ij = 4 df Re sum conj(dh_i) invC dh_j, with the estimator of the existing
response:

- four-point central differences of the template (two-point with `easy_central_difference`) over
  the parameters in `inds`, at the steps of `_information_matrix_steps`;
- a step of zero or less freezes a parameter (zero row and column);
- the four templates of one derivative share the union of their bands, so a carrier that the step
  moves across a bin does not inflate the matrix.

Each stencil template equals the template that `fill_global_stft` writes for its shifted
parameters, bit for bit. The tests compare the kernel with `gbgpu.testing`'s reference Fisher,
which differences those templates: they agree to 3.3e-15 of sqrt(F_ii F_jj) at most.

The kernel, `dffr_information_matrix_kernel` in `lat_dffr_kernels.hh`, keeps for each stencil
template what its pixels read from a segment table. It builds these tables with the build of the
tile (`dffr_build_rows`), straight into the Fisher tables, a wave of templates at a time, with all
threads of a block. The parameters that change no reception state share the windowed sums of the
unperturbed source at each pixel; for a galactic binary these are the amplitude, the phase offset,
the inclination and the polarisation, which `GBDFFRSource` lists in `reception_invariant_params`.
On the 8 parameters of the global fit, a pixel evaluates 17 sets of sums for 32 templates. On the
GPU a pass holds 4 segment rows: 66 kB of shared memory on 8 parameters, 77 kB on 9.

- `linear_envelope` defaults to the group's setting, as for the gradients, and `astro_track` does
  not apply.
- The steps are those of the existing response. On the production fixtures the DFFR Fisher changes
  by at most 1.1e-5 of sqrt(F_ii F_jj) between them and tenfold steps, outside the row and column
  of f̈.
- On the CPU with the installed build, the DFFR Fisher takes 0.82 to 1.13 of the time of the
  existing one: it costs per pixel, the existing one per segment. With GBGPU's CPU module compiled
  at `-O3`, it takes 0.30 to 0.42.
- On a full A100 it takes 0.51 to 0.63 of the time of the existing Fisher, from 1 to 10 000 sources
  on a year of one-day segments (`tests.md`, step 6).

The questions left open, and observations on the accuracy of the Fisher matrix, which a separate
validation takes up, are listed in `plan.md` (Risks and observations).

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
