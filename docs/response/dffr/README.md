# Direct Fast Fresnel Response for the STFT galactic-binary likelihood

The Direct Fast Fresnel Response (DFFR) is the default response of `STFTGBComputations`, the
short-time Fourier transform (STFT) likelihood of galactic binaries in GBGPU and lisaanalysistools
(LAT); `response="tdi_on_the_fly"` selects the response that existed before. DFFR was built into
both packages on the branch `feat-dffr` in October 2026, in steps that each kept the existing
response bit-identical, and then validated against the standalone package, a time-domain truth,
the Mojito-light data and the galactic-binary block of the global fit.

| Document | Content |
| --- | --- |
| `README.md` | What DFFR computes and gains, how it is built in, how to select it, which settings matter, and which choices are tuned to an A100. |
| [`results.md`](results.md) | The measurements: regression of the existing response, accuracy and speed of DFFR, the global-fit comparison, kernel resources, tests, and the commands that reproduce them. |

The scripts behind `results.md` are in [`scripts/dffr/`](../../../scripts/dffr/), with one driver,
`run_dffr.sh`. The full record of the integration, step by step with every intermediate
measurement, is in the git history of this folder up to commit 8f56f00e (`plan.md`, `tests.md`).
Author: Coen Rondeel.

## What DFFR computes

`STFTGBComputations` evaluates the likelihood of galactic binaries on an STFT grid of one-day
segments. The existing response samples the TDI response on the fly three times per segment and
channel and evaluates a windowed Fresnel transform of seven terms per pixel.

DFFR computes the same pixels in another way. Within one segment the signal that reaches each
spacecraft is a linear chirp under a slowly varying envelope, so the windowed transform of the
segment has a closed form in Fresnel integrals. DFFR evaluates three spacecraft reception states
per pixel from a table of spacecraft kinematics at the segment midpoints, which does not depend on
the source, and folds them with eighteen TDI weights per segment, closed forms for the 1st and
the 2nd generation combinations with the light travel times carried to first order across the
segment. The cost per frequency bin does not depend on the sampling rate of an underlying time
series. The theory is in the standalone package (last section).

## What it gains

Measured on one full A100 and on the CPU of SURF Spider; the details and the conditions are in
`results.md`.

| Quantity | DFFR against the existing response |
| --- | --- |
| Speed of the likelihood methods on the A100 (a year of one-day segments, 1000 to 10 000 sources) | 2.1 to 8.8 times faster; `get_ll` 7.7 µs per source instead of 63.5 at 10 000 sources |
| Speed of the Fisher matrix on the A100 | 1.6 to 2.0 times faster |
| Speed of the galactic-binary block of the global fit (56 binaries, 32 walkers, 24 temperatures, one A100) | an iteration in 17.2 min instead of 31.6; the in-model likelihood evaluations 3.4 times faster |
| Speed on the CPU, one thread | 0.28 to 0.40 of the time in the likelihood methods |
| Accuracy against a year of a time-domain truth (lisagwresponse and pytdi), 20 mHz | flat mismatch 4.8e-10 to 6.8e-10, as the standalone package; the existing response 7.6e-10 to 1.2e-9 |
| The same at 0.5 mHz, noise-weighted | mismatch 1.3e-7; the existing response 7.2e-6 to 1.4e-5, from its error in the combination X + Y + Z |
| Likelihood surfaces against the truth at SNR 100, eight parameters | every peak within 0.002σ of the truth with the widths of its Fisher matrix; the existing response within 0.010σ |
| Six waveforms on the Mojito-light data, five binaries of 15.6 to 20.4 mHz | mismatch 3.1e-8 to 5.8e-8 and peaks within 0.003σ, like the existing response; FastGB and jaxgb 6.8e-4 to 2.5e-3 and up to 1.3σ |
| Posteriors of the global fit, 56 binaries, equal iterations | the same within the sampling noise of the chains: widths within 1 % at the median, means within 0.24σ |
| Agreement with the standalone package on identical inputs | bit-identical templates on the CPU; inner products within 6.3e-15 of their peaks |
| The existing response after the integration | every output bit-identical on the CPU, within 1.2e-16 of its peak for the atomic `fill` on the GPU; timings within the run-to-run noise |

## Where the pieces live

| Repository | Piece | Content |
| --- | --- | --- |
| LAT | `cutils/Detector.{hpp,cu}` | Five-point derivatives of the orbit tables (`get_velocity`, `get_acceleration`, `get_ltt_rate`, `get_ltt_acceleration`, a step of whole cells closest to 2000 s) and `Orbits::get_kinematics_table`, which fills the source-independent kinematics table at the segment midpoints. The existing response takes its spacecraft acceleration from `get_acceleration`. |
| LAT | `cutils/binding_detector.{hpp,cxx}` | `KinematicsTableWrap`, the table over arrays owned by Python; a required field of `LISAToolsBackendMethods`, which every downstream loader must pass (GBGPU does; BBHx needs the one line). |
| LAT | `domaincomputation.py` | `STFTComputationGroup.cpp_kinematics(cpp_orbits)` builds the table on the group's device on first use and keeps it, one per orbits object; `kinematics_fields` returns its arrays. |
| LAT | `cutils/lat_fresnel.hh` | The Fresnel integrals, their auxiliary functions and the derivatives of these, with DFFR's numerics, shared by `STFTFresnel` and DFFR. The header records its branches, their accuracy and its assumptions. |
| LAT | `cutils/lat_stft_kernels.hh` | The likelihood kernels, written against a tile of pixels so that both responses feed the same consumers (next section). `ColumnTile` feeds them the existing response; `StftBandBinMap` and the launch path above 48 kB of shared memory serve DFFR. The Fisher kernel of the existing response keeps its own loop. |
| LAT | `cutils/lat_dffr_kernels.hh` | DFFR settings, the source-model contract, the windowed transform, the closed-form TDI weights of both generations, the segment table, `DFFRTile`, and the DFFR Fisher matrix. The header records the definitions, conventions, accuracy and assumptions. |
| LAT | `response/tdiconfig.py` | `TDIConfig.generation`: 1 or 2 for the stock combinations, `None` otherwise. |
| LAT | `globalfit/recipe.py` | The optional keys `response` and `first_order_delays` of `gb_stft_kwargs`. |
| GBGPU | `cutils/gb_physics.hh` | The GB phase, frequency and amplitude formulas, shared by `GBTDIonTheFly` and the DFFR source model. |
| GBGPU | `cutils/gb_dffr.{hh,cu}` | `GBDFFRSource`, the galactic-binary source model of DFFR, with the list of parameters that leave the reception states unchanged, and the DFFR entry points, bound as the `gb_dffr_*` methods of `GBComputationGroupWrap` next to the `gb_stft_*` methods of the existing response. Nothing DFFR enters `gb_tdi_on_the_fly.{hh,cu}`. |
| GBGPU | `gbcomps.py` | The `response` flag of `STFTGBComputations`, the DFFR settings, the checks at construction and the dispatch of the six likelihood methods and of `information_matrix`. |

## How it is built in

**Kinematics table.** A computation group builds the table from the comp's orbits on the first
DFFR call and keeps it: segment midpoints, spacecraft positions, velocities and accelerations,
light travel times with their rates and accelerations, and the link vectors. Every GPU split of the
residual container and of each band buffer has a group of its own, so each builds its table once
(0.35 ms at 32 segments on the CPU, 2.4 ms at 365; later calls find it in a few µs). The
derivatives are five-point central differences of the linearly interpolated orbit tables, with
samples that share their position within the table cells, so that the interpolation error cancels.

**The tile seam.** The likelihood kernels of `lat_stft_kernels.hh` are split into a producer of
pixel values, the tile, and the consumers that use them: `get_ll`, `fill_global`, `swap_ll`, the
F-statistic and the two gradients. A consumer runs over the STFT segments of a source in passes,
keeps the window mask of its cell, the factor 0.5 of the real signal, the fill factor and every
sum, and asks the tile for the values of the three channels at each pixel. A tile type provides:

| Member | Role |
| --- | --- |
| `Inputs` | The launch inputs of the response, a plain struct that the kernel receives by value and the consumers pass on to `Template::build`. |
| `BlockShared` | The response's static shared memory in a block. |
| `table_bytes()` | The dynamic shared memory of one template; a launch reserves it once per template, twice for `swap_ll`, its gradient and the F-statistic, which hold two. |
| `Map` | The pixel map: which thread evaluates which pixel of a pass. |
| `Context` | Built by every thread at the start of a kernel; `rows_per_pass()` gives the number of segments built per pass. |
| `Template` | One template: `load` reads the source parameters once per evaluation, `build` prepares the segments of a pass, `carrier_bin` places the band of a segment, `values` returns the three channels at one pixel. |

`ColumnTile<ColumnT>` adapts the column policy of the existing response to this contract: a pass
holds one segment per thread, each thread walks the band of its own segment in increasing
frequency, with the same calls and arguments as before, so the outputs stay bit-identical.
`DFFRTile` builds its segment tables in shared memory with all threads of a block, 32 segments per
pass, and maps one band bin to each warp. The 32 is the warp size of NVIDIA GPUs: the 32 threads of
a warp take the 32 segments of one band bin, so that the branch of the Fresnel functions, which
depends on the distance from the carrier, stays the same within a warp. One segment row holds 1012
bytes, so one template takes 32 672 bytes with its source constants and settings, under the 48 kB a
launch gets by default; the two templates of the swap kernels take 65 344 bytes, and the launch
path raises the kernel's limit within what the device allows a block. On the CPU a pass holds one
segment row.

**Source model.** `GBDFFRSource` loads the constants of a binary once (sky basis and
polarisation, amplitude, frequency and its two derivatives, initial phase, the plus and cross
weights of the inclination) and gives, per component, the quadratic phase expansion at a retarded
time, the amplitude ratio and its log-slope, with the formulas of `gb_physics.hh`. The contract
speaks in components so that a source with several harmonics fits it, but the tile evaluates one
component and requires a source with one: the harmonics of a binary lie f0 / df bins apart, 86
bins at 1 mHz on one-day segments, far outside a template of `2 n_side_bins + 1` bins, so each
belongs in a template of its own.

**Call path.** `comp.get_ll_stft` asks its group for the kinematics table
(`stft_comps.cpp_kinematics(cpp_orbits)`) and calls `GBComputationGroupWrap.gb_dffr_get_ll` with
the data index, the noise index, the parameters, `t_ref`, `n_side_bins`, the TDI generation, the
envelope, the delays and the window starts; the entry point launches
`stft_get_ll_launch<DFFRTile<GBDFFRSource, 32>>`. The window settings come from the bound
`STFTFresnel`, the grid and data from `STFTDomain`.

**The Fisher matrix.** `dffr_information_matrix_kernel` is a kernel of its own with the estimator
of the existing response: four-point central differences of the template (two-point with
`easy_central_difference`) over the parameters in `inds` at the steps of
`_information_matrix_steps`, a step of zero or less freezing a parameter, and the four templates
of one derivative sharing the union of their bands. The parameters that change no reception state
share the windowed sums of the unperturbed source at each pixel; for a galactic binary these are A,
φ0, ι and ψ, so a pixel evaluates 17 sets of sums for the 32 stencil templates of the 8 parameters
of the global fit. The tables of a pass are built in waves straight into the Fisher tables, through
the build the tile uses (`dffr_build_rows`); every stencil template equals the template that
`fill_global_stft` writes for its shifted parameters, bit for bit, and the kernel agrees with a
reference that differences those templates to 3.3e-15 of sqrt(F_ii F_jj). On the GPU a pass holds
4 segment rows (66 kB of shared memory on 8 parameters, 77 kB on 9); the CPU uses one.

## How to select it

```python
from gbgpu.gbcomps import STFTGBComputations

comp = STFTGBComputations(
    stft_comps, T, t_ref=t_ref, orbits=orbits, tdi_config="2nd generation",
    n_side_bins=10, response="dffr",  # the default; "tdi_on_the_fly" is the other response
)
```

In a global-fit recipe, `gb_stft_kwargs` takes the same optional keys. The comp's methods, outputs
and attributes are the same for both responses, so the band engine and the device replicas need no
change; a replica keeps the response and its settings, which are part of the recorded constructor
arguments.

| Setting | Default | Meaning for DFFR |
| --- | --- | --- |
| `response` | `"dffr"` | `"tdi_on_the_fly"` selects the response that existed before. |
| `first_order_delays` | `False` | `True` carries every light travel time to first order across a segment. The validation numbers of `results.md` were taken with `True`; without it, the templates of the production fixtures change by up to 1.0e-4 of their peak and (d\|h) by 8.5e-5, and the accuracy against a truth is not measured. |
| `linear_envelope` | `True` | Must equal the `linear_envelope` of the bound computation group, `True` by default as well since the review of this effort; the group's value is the one the kernels use (the amplitude and antenna slopes within a segment). An explicit mismatch raises `ValueError`. |
| `linear_envelope` of the gradients and the Fisher | `RESPONSE_DEFAULT` | The group's setting for DFFR, `False` for the existing response, as before. The existing response turns the envelope off in its derivatives because its slope comes from its TDI stencil, whose rounding would enter the differences; DFFR's slopes are closed forms, and with the envelope on its finite-difference noise in f0, ḟ, f̈, φ0, λ and β is unchanged. `True`, `False` and `None` apply to both responses. |

The floor on the local chirp rate of a reception state is not a setting: DFFR applies
`STFT_FDOT_FLOOR`, 1e-22 Hz/s, the sign-preserving floor of the existing response, which changes
the phase by at most π floor (dt / 2)², 6e-13 rad on a one-day segment.

DFFR reads from the bound computation group the taper `window_alpha` of its `STFTFresnel`, the
grid, data and inverse covariance of its `STFTDomain`, its `linear_envelope` and its kinematics
table. It does not read `T`, `freq_from_tdi_phase` (how the existing response estimates its chirp)
and for the same reason the `astro_track` argument of `information_matrix`, `use_midpoint` (DFFR
always expands about the segment midpoint), and `window_factor`, which scales the templates of
the existing response when `window_alpha` is 0 to approximate the power a taper removes; DFFR
models the window through `window_alpha`, so tapered data need `window_alpha`, not this factor.

Limits, checked at construction with `ValueError`: XYZ channels only, so a group or domain of
another TDI set raises; the combinations of `TDIConfig("1st generation")` or `("2nd generation")`
only, in any order, also when the `TDIConfig` is set later. A group without `cpp_kinematics`
raises `TypeError`.

## Settings that set the accuracy and the speed

These apply to both responses and were measured in the validation.

- Side bins (`n_side_bins`). With the production taper of 0.1157 (1e4 s on a one-day segment), the
  10 side bins of production leave 1.3e-3 (20 mHz) to 1.7e-3 (0.5 mHz) of a source's noise-weighted
  power outside the template: a mismatch of 6.4e-4 to 8.8e-4 against time-domain data and a loss
  of 6 to 9 in ln L at the true parameters for a source of SNR 100. A tone keeps 1.3e-6 to 5.8e-6
  of its power beyond 30 bins and 4e-8 to 1e-7 beyond 60; the validation used 60. More side bins
  cost time in proportion to the band, 2 n_side_bins + 1 bins per segment.
- The window taper. A taper of 1/(14 mHz) = 71 s, as the tracked high-frequency settings had,
  leaves up to 1.9e-2 of a tone's power outside 10 side bins; 1e4 s leaves 1.4e-3. The taper also
  sets the width of the store windows and so the memory per cell of the GB block.
- The window convention. The global fit windows the data with LAT's symmetric Tukey window, both
  responses integrate the periodic one; the difference is a flat mismatch of 3.6e-8 and 6.3e-5 per
  segment, four orders of magnitude below the side bins above. Production keeps its window.
- `linear_envelope`. With it, DFFR and the existing response agree to a median mismatch of 5.6e-10
  on the production fixtures; without it each holds the amplitude constant over a segment in its
  own way, and they differ by about 1e-5. The TDI generation changes nothing between them. Since
  the review that closed the effort, both `STFTComputationGroup` and the comp default to the
  envelope on, as production runs it.
- Batches of the GB block. A cell of the GB move, one (temperature, walker, band), holds its band's
  store window of the residual, its own copy of the inverse 3 × 3 CSD and a template twin: about
  153 MiB for a window of 555 bins over 713 days. With 32 walkers (15 GiB) the A100's 40 GiB held 64
  cells per proposal batch and 2 rows per tempering batch, where the settings file's 512 and 128
  ran out of memory. Since each binary is one thread block and the A100 holds 216 to 324 at once, a
  likelihood call of 64 binaries costs about as much as one of 200 to 300, so the batch size sets
  most of the time per iteration, for both responses.

## Choices tuned to an A100

Every GPU timing and resource gate was taken on one A100-PCIE-40GB (sm_80: 108 SMs, 65 536
registers and 164 kB of shared memory per SM, 163 kB of dynamic shared memory per block after the
opt-in). These choices follow from it and may change on another device:

- 32 segment rows per pass in the DFFR tile: the warp size, and 32 rows of 1012 bytes fit the 48 kB
  a launch gets by default. The two-template kernels (swap, its gradient, the F-statistic) are
  built for 32 and for 16 rows and take `rows_per_pass`; `STFTGBComputations` passes 32
  (`_DFFR_ROWS_PER_PASS`): on the A100, 16 rows are 5 % to 17 % slower at 10 000 sources, since
  the registers hold both at 2 blocks per SM.
- 4 segment rows per pass in the DFFR Fisher (`_DFFR_FISHER_ROWS_PER_PASS`); 8 rows are built as
  well. At 4 rows two blocks share an SM, at 8 one block fills its shared memory. 8 rows are faster
  while a batch has fewer sources than the device has SMs (0.54 to 0.93 of the 4-row time) and
  slower from 1000 sources (1.25 to 1.32). A choice by batch size would need the SM count at launch.
- Blocks per SM, from the linked registers: the one-template DFFR kernels (154 to 164 registers) run
  3 blocks of 128 threads per SM, the two-template ones (220 to 254) and every kernel of the
  existing response (244) run 2, the DFFR Fisher (176) 2. A register cap of 168 on `domains.cu`
  gives the existing response a third block and 4 % to 15 % more speed, but slows two DFFR methods
  by up to 1 % through spills and changes rounding; it is left out. Device link-time optimisation
  would give the DFFR two-template kernels a third block, but needs `libnvvm` next to nvlink, which
  the pip toolkit does not provide.
- One block per source: a batch needs about 430 sources to occupy the A100 at 4 blocks per SM (530
  on an H100 with 132 SMs); below that a call costs about as much as for one source.
- The launch path takes the shared-memory limit from the device
  (`cudaDevAttrMaxSharedMemoryPerBlockOptin`), so larger tables launch on an H100 (227 kB per
  block); the block size, 128 threads (`NUM_THREADS_HERE`), is one value for every device.
- H100 (sm_90). The install script compiles machine code for Spider's A100s and PTX of that
  generation, so an H100 runs translated code unless 90 is added to `CUDA_ARCH`. The same source
  gives other register counts there (222 instead of 244 for the existing kernels), so
  `kernel_resources.py --arch 90` goes with `--arch 80` for any change of the kernels, and speed
  claims need the benchmarks repeated on an H100. What could pay there: the two-template DFFR
  kernels fit 3 blocks per SM by shared memory, the 64 FP64 units per SM (32 on the A100) want more
  work in flight per SM, and clusters of blocks with distributed shared memory could share the add
  and remove tables of swap. None of this was measured.
- GPU memory: the GB-block batch sizes above are set by the 40 GiB of this A100; a device with more
  memory takes larger batches and fewer, cheaper likelihood calls per iteration.

## Open items

The items left open at the end of the effort are tracked by the author outside the repository.
Their themes: the build and checks on H100 nodes, multi-device runs (unchecked), the `-Os` of
GBGPU's CPU module, two CUDA tests that fail by design of the GPU build, the open questions of the
DFFR Fisher matrix and its validation, properties of LAT's orbit classes and of `pyResponseTDI`
found on the way, the side bins and the window taper of the production settings, the amplitude
bias of the preprocessing's low-pass filter, and the observations on the galactic-binary block
from the global-fit runs (memory per cell, prior births whose tracks leave their store window, the
edge of the sampled f0 range, the leaf order, the slow spread of the phase parameters). BBHx needs
the one-line change that passes `KinematicsTableWrap` in its loader before it loads against this
branch.

## The standalone package

DFFR was developed and validated as a standalone package:
<https://github.com/coen-rondeel/direct-fast-fresnel-response>. The integration ports its compiled
kernels as of commit `01530a5`, and the package gained a CUDA 13 backend for the GPU comparison.

- [Theory and derivation](https://github.com/coen-rondeel/direct-fast-fresnel-response/blob/main/docs/theory.md)
- [Conventions](https://github.com/coen-rondeel/direct-fast-fresnel-response/blob/main/docs/conventions.md):
  source parameters, time frames, STFT and likelihood conventions, and the mapping to
  lisagwresponse and lisatools.
- [Validation](https://github.com/coen-rondeel/direct-fast-fresnel-response/blob/main/docs/validation.md):
  the package's own tests and comparisons.
