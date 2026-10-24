# DFFR integration: results

The checks behind the Direct Fast Fresnel Response (DFFR) in `STFTGBComputations`, by theme: how
the existing response was kept unchanged while DFFR was built in, how DFFR agrees with the
standalone package and with the existing response, how both compare with a time-domain truth and
with the Mojito-light data, what they do in the galactic-binary block of the global fit, and what
DFFR costs on the CPU and on an A100. `README.md` describes the response; the commands that
reproduce each check are at the end.

Conventions:

- A relative difference is the largest absolute difference over an array divided by the largest
  absolute value of the reference array. "Bit-identical" means `numpy.array_equal`. A bound is a
  threshold that a test asserts; where none exists, the text says what a number is compared with.
- CPU runs use one OpenMP and one BLAS thread, on `wn-la-01` of SURF Spider (AMD EPYC 9754). GPU
  runs use one full A100 (A100-PCIE-40GB) on `wn-gb-02`, CUDA toolkit 13.2.2, CuPy 14.2.0. The
  global-fit runs used one A100 each on `wn-gb-01`.
- The integration started from `l2d-dev` (LAT 7c12a36d, GBGPU ed22789) and ends at LAT 8f56f00e
  and GBGPU 11d5159 on `feat-dffr`. The standalone package is direct-fast-fresnel-response at
  01530a5, plus a CUDA 13 backend (2b61cdd) added for the GPU comparison.
- Every DFFR number below was taken with `first_order_delays=True`, the default of the comp during
  the effort. Since the review that closed it, the default is `False`, which holds each light
  travel time constant across a segment; the scripts pass `True` explicitly, so they reproduce
  the numbers recorded here. The accuracy with `False` is not measured.
- Outputs (`.npz`, `.json`, logs, figures) are kept outside git, in the author's work folder,
  `$OUT`. The full record of every step, with each intermediate capture and the steps' own
  drivers, is in the git history of this folder up to 8f56f00e (`plan.md`, `tests.md`).

## Method

- Fixtures (`scripts/dffr/stft_fixtures.py`). Seven `STFTGBComputations` fixtures with seeded data,
  inverse covariances and 24 sources, carriers placed next to the edges of their windows: three
  short equal-arm fixtures (16 six-hour segments, 61 bins, 4 side bins; without a window, with a
  Tukey window, and on the astrophysical frequency track) that repeat the configurations of
  `scripts/validation/stft_column_policy_oracle.py`; three production fixtures on the ESA orbits
  (32 one-day segments, taper 0.1157, 10 side bins, two store windows of 64 bins at about 3 and
  15 mHz; TDI 2nd generation with the linear envelope, TDI 1st generation without it, and an AET
  domain with a diagonal covariance); and the production fixture on `L1Orbits` from a Mojito L1
  file (two orbit grids at absolute times). DFFR runs on the three XYZ production fixtures.
- Golden outputs (`stft_golden_outputs.py`). Every output of the seven STFT methods (`get_ll`,
  `get_swap_ll`, `get_fstat_ll`, `fill_global`, both gradients, `information_matrix`) on every
  fixture: 144 arrays for the existing response on the six single-grid fixtures and 26 on
  `L1Orbits`, 50 and 25 for DFFR. On the CPU two captures in separate processes are bit-identical.
  On the GPU every output is bit-identical between captures except `fill_global_stft`, which adds
  with `atomicAdd` in an order that changes from run to run; it differs by up to 1.1e-16 of its
  peak, so its gate on the GPU is 1e-15 of its peak, and every other output must stay
  bit-identical.
- Frozen environments. A comparison with the code before a change comes from a second
  environment built from the earlier commits, accepted only when it reproduces that code's CPU
  golden outputs bit for bit. Every environment of this effort did.
- Timings (`stft_benchmarks.py`, `compare_benchmarks.py`). One warm-up and five timed calls per
  method and batch size, three runs in separate processes; a change is read against the larger
  run-to-run spread of the two sets. On the GPU the seven environments of steps 0 to 5 and part 4c
  were timed in one interleaved round (a year of one-day segments, batches 1 to 10 000), so that
  drift of the node reaches them alike. Runs of identical code hours apart differ by up to 1 % on
  the CPU, which bounds what such comparisons resolve.
- Gradients and Fisher matrices. At their default steps these finite differences carry rounding
  noise (per gradient component, relative to its scale: up to 0.25 in f0, 1.8 in ḟ and 13 in f̈;
  per Fisher entry, relative to sqrt(F_ii F_jj), up to 18 in the f̈ entries and 5.2e-4 elsewhere).
  A change of rounding in the templates is therefore judged per component and per entry against
  each code's own noise, the difference between the default and the tenfold step.
- Tests. The suites of LAT (`python -m unittest discover`) and GBGPU (`pytest tests`) ran after
  every step on the CPU and on the A100, and the outcome of every test was compared by name with
  the step before.

## The existing response through the integration

The existing response (TDI on the fly) had to keep its outputs and its speed while the DFFR
pieces were built in. Outputs are against the code before each step; timings are the ratio of
the median times. The accuracy test `GBGPU/tests/test_stft_gb_accuracy.py` printed the same
numbers after every step (end-to-end mismatch 9.049e-3 at 10 side bins, 3.950e-3 at 20, on-bin
mismatch 1.409e-5), and the 54 arrays of the column-policy oracle stayed bit-identical to the
step 0 capture.

| Change | Outputs of the existing response | CPU timing | A100 timing |
| --- | --- | --- | --- |
| Orbits cache: five-point derivatives of the orbit tables replace the three-point acceleration of the TDI-phase estimator | templates and inner products change by at most 6e-11 of their peaks, gradients 2e-9, Fisher matrices on the TDI track 2e-8 (the f̈ entries by up to 1.9e-5 of sqrt(F_ii F_jj) on the four-day fixtures); arrays that never read the acceleration are bit-identical. The size of the removed truncation error, 1.3e-8 of the annual acceleration, carried into a segment's phase | within the noise on the ESA fixture (0.998 to 1.006), up to 1 % slower on `L1Orbits` | 1.1 % to 2.1 % slower from 1000 sources, gradients the most |
| Fresnel split: the special functions move to `lat_fresnel.hh` with DFFR's numerics | outputs computed directly from templates change by at most 1.1e-14 of their peaks (gate 1e-12); gradients and Fisher matrices within each code's own finite-difference noise (62 of 63 components, 4526 of 4536 entries; the rest are terms both codes compute almost exactly) | within the noise (0.985 to 1.010) | 3 % to 6 % faster in every method but the Fisher matrix |
| Tile seam: the kernels become templates on a tile type | bit-identical, swap kernels included (144 and 26 arrays) | within the noise at 27 of 30 points, the other 3 within 0.7 % | within the noise at 27 of 35 points, the other 8 within 0.7 %; drift accepted |
| DFFR kernels, wiring, DFFR Fisher | bit-identical at every step (CPU machine code of the existing kernels identical instruction for instruction, two functions aside that now call `gb_physics.hh` with the same bits) | within the noise or faster | 33 to 35 of 35 points within the noise, the rest within 0.6 % |
| Fisher optimisation and occupancy A/B of step 6 | bit-identical (the register cap that would have changed rounding was left out) | | unchanged |
| The source-independent DFFR diagnostics leave LAT | bit-identical on the CPU, `fill` within 1.2e-16 on the GPU; every test keeps its outcome | | |

On the GPU, `fill` differed between captures of different steps by at most 8.6e-17 of its peak,
within its gate. From the final DFFR code to the removal of the diagnostics the whole suite kept
its outcomes: LAT 292 tests (246 passed, 7 failures, 12 errors and 27 skips on the A100, every
failure and error from before this effort), GBGPU 84 tests (72 passed, 6 failures, 6 errors: the 4
failures and 6 errors from before this effort, and the two CUDA tests of DFFR that fail by design
of the GPU build, listed under the unit tests below).

## Orbits and kinematics

DFFR reads a table of spacecraft kinematics at the segment midpoints, built once per computation
group by `Orbits::get_kinematics_table` from five-point central differences of the orbit tables
(`get_velocity`, `get_acceleration`, `get_ltt_rate`, `get_ltt_acceleration`; a step of whole
cells closest to 2000 s). `tests/test_orbit_kinematics.py` (13 tests) checks them on synthetic
orbits and on the ESA file; every bound holds on the CPU and on the A100, where the CUDA table
equals the CPU table bit for bit.

| Check | Bound | Measured |
| --- | --- | --- |
| Quadratic positions and light travel times against the exact derivatives | rounding bound: velocity 2.4e-7 m/s, acceleration 4.3e-10 m/s², light-travel-time rate 1.1e-17, its acceleration 2.0e-20 1/s | 1.4e-8 m/s, 3.1e-11 m/s², 8.5e-19, 1.3e-21 1/s |
| Annual circle against the analytic values, relative | position 2e-9, velocity 1e-8, acceleration 5e-8, light travel time 1e-15, its rate 1e-9, its acceleration 2e-5, link vector and rate 5e-8 | 1.2e-9, 1.2e-9, 4.7e-9, 2.1e-16, 1.2e-10, 1.1e-6, 4.9e-9 |
| ESA orbits: receiver velocities against the file velocities, relative | 2e-6 | 2.6e-7 to 2.8e-7 |
| Coverage and arguments | grids 1 s outside the tables raise `ValueError`; wrong lengths and dtypes raise | as expected |

Against standalone dffr on the same ESA orbits (365 segments): with dffr's C builder on the same
stencil, receivers and light travel times are bit-identical, senders differ by 2.4e-14 relative
(dffr rounds the emission time after subtracting the file's t0); with its lisaorbits builder,
positions differ by 3.2e-9, velocities 2.8e-7 and accelerations 8.4e-5 relative, the error of a
cubic spline through position samples 1.87 days apart, which is how LAT represents the orbit. At
20 mHz those differences shift the phase by about 2e-7 rad. On `L1Orbits` the receiver velocities
differ from the file velocities by 8.1e-6 relative (samples 5.8 days apart). On the CPU a group
builds its table in 0.35 ms at 32 segments and 2.4 ms at 365; later calls find it in 2 to 3 µs.

Three properties of the orbit classes found on the way, left as they are: `Orbits::get_window`
accepts the last cell of a table, where `interpolate` reads one row past the array;
`Orbits._configure` appends the file's last sample after its uniform grid, so the last cell of the
ESA file spans 74.7 s instead of 50 s; and the base classes and `L1Orbits` pass a cubic spline
through the file's positions without using the file velocities. The new methods stay away from
the last cells. The derivative helpers of GBT's `CubicSplineSegment` return other polynomials than
the derivatives; nothing calls them.

## Fresnel functions

`lat_fresnel.hh` holds the Maclaurin series of C and S (|x| ≤ 1.6), minimax rational fits of the
auxiliary functions f and g and their derivatives (1.6 < |x| < 8) and a five-term asymptotic
series above, with DFFR's numerics; `STFTFresnel` delegates to it. `tests/test_fresnel.py`
(9 tests) checks it against `scipy.special.fresnel` and quadratures of the integral
representations of f and g.

| Check | Bound | Measured |
| --- | --- | --- |
| C and S, 24 806 arguments with \|x\| ≤ 1000, absolute | series 1e-15; rational fits 2e-10 (C) and 1e-10 (S); asymptotic series 2e-13 | 4.4e-16 and 3.3e-16; 1.6e-10 and 7.0e-11; 1.0e-13 and 8.5e-14 |
| f, g, f', g', 882 arguments, relative (g' on the rational branch absolute) | rational fits 1e-9, 2.5e-9, 2.5e-9, 1e-9; asymptotic series 1e-14, 2.5e-13, 2.5e-13, 2e-12 | 8.6e-10, 2.2e-9, 2.2e-9, 8.4e-10; 5.8e-15, 1.3e-13, 1.3e-13, 9.6e-13 |
| Windowed phase kernel and its moment against adaptive quadrature, the 16 cases of dffr's tests | 2e-10 of the segment length; 5e-8 of the half-length squared | 7.2e-11; 2.9e-8 |
| CUDA against CPU, A100 | 1e-14 | C 2.2e-16, S 1.1e-16, f, g and their derivatives bit-identical, kernel 9.0e-17, moment 3.0e-16 |

## DFFR against the standalone package

- Exchange on identical inputs (ESA orbits, 32 one-day segments, the production band as one
  domain, 24 production sources), standalone dffr recomputing LAT's inputs in its own environment:
  on the CPU the compact templates, the `fill_global` grid and the whole segment table are
  bit-identical, (d|h) and (h|h) differ by 1.6e-15 and 6.3e-15 of their peaks (gate 1e-13), the
  order of their sums. On the A100 the templates differ by 1.7e-14, `fill_global` 1.2e-14, (d|h)
  5.0e-16 and (h|h) 1.5e-16; the segment table is identical except the TDI weights (2.7e-16).
- Over a year and 10 000 sources on the A100, the `fill_global` grids of the two codes differ by
  1.0e-12 of their peak while (d|h) and (h|h) agree to 1.4e-15. The difference is systematic
  (1.04e-12 in two pairs of runs, against 2.4e-16 between runs of one code), does not grow with
  time (per segment 1.3e-14 to 1.2e-13, rising and falling with the orbit's configuration over
  the year) and grows with the square root of the source count: the two GPU builds round the pixel
  evaluation differently, and the roundings add up over overlapping sources. On the CPU the two
  codes agree bit for bit.
- The standalone package's own suite passes on the A100, 51 of 51, with the CUDA 13 backend added
  for this comparison.
- The eight cases of the standalone package's time-domain comparison (3, 20 and 40 mHz and
  20 mHz with ḟ = 1e-13 Hz/s, two sky positions, 30 one-day segments, lisagwresponse,
  lisainstrument with every noise off and pytdi): DFFR in `STFTGBComputations`, on the GPU from
  LAT's kinematics table, reproduces the standalone CPU path to two or three digits in every case
  (flat mismatch 6.5e-10 to 1.2e-9, median segment error 3.2e-5 to 4.7e-5). The existing response
  has lower median errors (1.4e-5 to 3.0e-5) but an 8 times larger mismatch at 20 mHz in one sky
  position (1.04e-8), from the segments where the Y channel is weak.

## DFFR against the existing response

On the 24 production sources and the exchange fixture, the two templates of each source over the
whole band, with unit weights over the three channels. There is no threshold; the distance of
each from a time-domain truth is in the next section.

| Setting | Median mismatch 1 - Re(a\|b)/sqrt((a\|a)(b\|b)) | Largest | Largest phase of (a\|b) | Norm ratio |
| --- | --- | --- | --- | --- |
| TDI 2nd generation, linear envelope (production) | 5.6e-10 | 2.6e-8 | 2.2e-5 rad | 0.99998 to 1.00008 |
| TDI 1st generation, linear envelope | 5.6e-10 | 2.6e-8 | 2.2e-5 rad | 0.99998 to 1.00008 |
| either generation, constant envelope | 8.6e-6 | 4.3e-5 | 2.7e-5 rad | 0.99995 to 1.00006 |

The TDI generation changes nothing; the envelope sets the distance, since each response holds the
amplitude constant over a segment in its own way. The same numbers come out on the CPU and on the
A100.

## Accuracy against a time-domain truth

Truth: X2, Y2 and Z2 every 5 s from lisagwresponse (strain at the exact retarded times) and pytdi
on the ESA orbits that LAT ships, a year of one-day segments from day 10 of the orbit file. Eight
sources: 0.5 and 20 mHz, at each the two ḟ bounds of the global fit's GB prior (1.1e-18 and
-6.6e-20 Hz/s at 0.5 mHz, 8.2e-13 and -2.3e-11 Hz/s at 20 mHz), each at two sky positions.
Templates with the production settings but 60 side bins, on data windowed with the periodic Tukey
window of taper 0.1157 that both responses integrate. Ranges over the four sources of each
frequency; the segment error is max |h - d| / max |d| per segment and channel.

| Frequency | Response | Flat mismatch, template pixels | Median segment error | Largest segment error | Noise-weighted mismatch |
| --- | --- | --- | --- | --- | --- |
| 0.5 mHz | DFFR | 2.7e-8 | 1.0e-4 | 2.5e-4 to 9.0e-4 | 1.3e-7 |
| 0.5 mHz | existing | 2.7e-8 | 1.0e-4 | 3.2e-4 to 6.2e-3 | 7.2e-6 to 1.4e-5 |
| 20 mHz | DFFR | 4.8e-10 to 6.8e-10 | 2.2e-5 to 3.2e-5 | 4.0e-4 to 4.7e-4 | 2.4e-8 to 3.6e-8 |
| 20 mHz | existing | 7.6e-10 to 1.2e-9 | 1.4e-5 to 1.9e-5 | 4.1e-3 to 4.6e-3 | 2.5e-8 to 3.7e-8 |

- At 20 mHz DFFR matches the truth as well as the standalone package did; the largest segment
  errors of both responses fall where one channel's signal is weak (6 % to 7 % of its median over
  the year). The noise-weighted mismatch of both is set by the power beyond 60 bins, 4.6e-8 to
  7.0e-8 of (d|d).
- At 0.5 mHz the band of 60 side bins reaches the start of the production band, where the data
  hold the leakage of the signal's negative-frequency image, which neither response computes: a
  floor of 1.0e-4 in the segment error and 2.7e-8 in the flat mismatch of both. The noise-weighted
  mismatch of the existing response is 50 to 100 times DFFR's: its error sits in the combination
  X + Y + Z (2.6e-3 to 3.7e-3 of its size against 3.4e-4 for DFFR), which carries 0.9 % of the
  signal of A there and far less noise, so the likelihood weights it heavily.
- The window convention. The global fit windows the data with LAT's symmetric Tukey window, which
  divides by N - 1 and is zero at the last sample; both responses integrate the periodic one. The
  two data transforms differ by a flat mismatch of 3.6e-8 and a median segment error of 6.3e-5,
  the same for every source and segment; in ln L at the true parameters it costs SNR² × 3.6e-8.
  Production keeps its window, since its 10 side bins cost four orders of magnitude more (next
  point); the validation ran on data with the periodic window.
- Side bins. With the 10 side bins of production and the taper 0.1157, 1.3e-3 (20 mHz) to 1.7e-3
  (0.5 mHz) of (d|d) lies outside the template, for both responses: a noise-weighted mismatch of
  6.4e-4 to 8.8e-4, and a loss of 6 to 9 in ln L at the true parameters for a source of SNR 100.
  A tone keeps 1.3e-6 to 5.8e-6 of its power beyond 30 bins and 4e-8 to 1e-7 beyond 60. The
  validation therefore used 60 side bins.
- LAT's full time-domain response `pyResponseTDI` was the first injection. Its fractional-delay
  factor A(e) comes from a linearly interpolated table of 1001 values, which underestimates it by
  up to 2.1e-7; TDI 2 amplifies this at low frequency to 3.0e-3 of the peak at 0.5 mHz (4.3e-5 at
  2 mHz, 3.4e-7 at 20 mHz). Recorded only; the truth chain above replaced it.

## Likelihood surfaces against the truth

The same eight sources at an optimal SNR of 100 (SNR² is the STFT (d|d)). Each of four waveforms
runs in its own domain against the transform of the same noiseless data, on X, Y and Z with the
full complex 3 × 3 CSD of `XYZSensitivityBackend` at the global fit's default noise levels: DFFR
and TDI on the fly in the STFT (`get_ll_stft`, one CSD per segment), TDI on the fly in the
frequency domain (`GBFDComputations`, the year on N points with the carrier shifted out) and
FastGB (`gbgpu.gbgpu.GBGPU`), the FD codes on dt rfft of the year with the global fit's FD taper
of 1e4 s and one CSD with the transfer functions averaged over the orbit. Each of the eight
parameters of the global fit is scanned over ±4σ in 81 points with the others at the truth, σ the
conditional width 1/sqrt(F_ii) of DFFR's Fisher matrix; ln L = -(d - h|d - h)/2, so an exact
waveform gives 0 at the truth.

| Sources | Waveform | Largest peak offset [σ] | Width / σ | ln L at the truth | Mismatch at the truth |
| --- | --- | --- | --- | --- | --- |
| 0.5 mHz (4) | DFFR | 0.002 | 1.000 | -1.3e-3 | 1.3e-7 |
| | TDI on the fly, STFT | 0.010 | 0.998 to 1.000 | -0.07 to -0.14 | 7.2e-6 to 1.4e-5 |
| | TDI on the fly, FD | 0.016 | 0.70 to 0.86 | -0.49 to -0.77 | 2.8e-5 to 4.4e-5 |
| | FastGB | 6.6 | 0.89 to 0.99 | -3.1e3 to -3.2e3 | 0.20 to 0.21 |
| 20 mHz, largest ḟ (2) | DFFR | 0.002 | 1.000 | -2.4e-4 to -2.7e-4 | 2.4e-8 to 2.7e-8 |
| | TDI on the fly, STFT | 0.001 | 1.000 | -2.5e-4 to -2.8e-4 | 2.5e-8 to 2.8e-8 |
| | TDI on the fly, FD | 0.010 | 0.962 to 0.965 | -0.34 to -0.44 | 3.2e-5 to 4.1e-5 |
| | FastGB | 2.2 | 0.965 to 0.978 | -36 | 3.3e-3 to 3.4e-3 |
| 20 mHz, smallest ḟ (2) | DFFR | 0.002 | 1.000 | -3.6e-4 | 3.6e-8 |
| | TDI on the fly, STFT | 0.001 | 1.000 | -3.6e-4 to -3.7e-4 | 3.6e-8 to 3.7e-8 |
| | TDI on the fly, FD | 0.009 | 0.961 to 0.965 | -0.30 to -0.39 | 2.8e-5 to 3.6e-5 |
| | FastGB | 5.7 | 0.971 to 0.988 | -45 to -49 | 4.1e-3 to 4.5e-3 |

- DFFR puts every peak within 0.002σ of the truth with the widths of its Fisher matrix to three
  digits; its model error costs at most 1.3e-3 in ln L. TDI on the fly stays within 0.010σ in the
  STFT and 0.016σ in the FD.
- The FD widths differ from the STFT ones by the information each domain holds, not by the
  waveforms: at 20 mHz 0.963 = sqrt(1 - 5α/8), the power the STFT's taper removes; at 0.5 mHz
  0.70 to 0.86, because the FD CSD weights T more than the CSD per segment (FD SNR 132 against
  100). The FD losses at the truth come from the data window, which no FD template models, and at
  20 mHz with the largest ḟ from the FD heterodyne's own sampling (0.34 to 0.44 at N = 8192,
  0.0011 to 0.0015 at 16N).
- FastGB is off by up to 6.6σ, mostly in φ0, ψ, f0, ḟ and the sky. At 0.5 mHz it produces no
  combination T = (X + Y + Z)/√3, where the truth's is 0.9 % of A; at 20 mHz its T is 11 % low
  with the opposite sign, its amplitude 0.6 % to 1.8 % low and its phase 0.02 to 0.06 rad off.
- `GBFDComputations` contracts with the real part of C⁻¹ only, which moves (h|h) by up to 2.9e-3 at
  0.5 mHz against the full CSD; its constructor requires t_start = t_ref. The default FD CSD of
  `XYZSensitivityBackend`, at the orbit-averaged arm lengths, gives an FD SNR of 182 at 0.5 mHz
  against 132 with the transfer functions averaged over the orbit. Recorded for the owners of
  these codes.

## Six waveforms on the Mojito-light data

The galaxy and the instrument noise of Mojito light, read and preprocessed as the global fit does
(the settings of `mojito_input/gb_high_freq_global_fit_settings.py`: high-pass at 1e-5 Hz,
low-pass at 0.1 Hz, downsampling to 5 s, 200 h trimmed at each end) over 1.92 years (701 one-day
segments). The five catalogue binaries of highest frequency, at the catalogue's amplitudes and
with the parameters mapped as the global fit maps them (φ0 = +TrueAnomaly, sky and polarisation
in ICRS): 20.380, 19.668, 16.716, 15.652 and 15.626 mHz, SNR 223, 95, 163, 67 and 193. The last
two lie 1.9 bins apart, inside each other's templates. STFT: the periodic Tukey window of taper
1e4 s, 60 side bins, t_ref the catalogue's reference time; FD: the global fit's FD taper on a band
of 8192 bins about each carrier. The data of the scans are the galaxy with each source's
neighbours removed (as DFFR templates in the STFT, as `GBFDComputations` templates in the FD, both
with the gain of the preprocessing filters), the galaxy alone, and the galaxy plus the noise.

| Waveform | Mismatch at the truth | Largest peak offset in f0, ḟ, φ0, ψ, λ, β [σ] |
| --- | --- | --- |
| DFFR | 3.1e-8 to 5.8e-8; 1.8e-7 and 4.5e-7 beside a louder binary | 0.003 |
| TDI on the fly, STFT | the same to 2 % | 0.003 |
| TDI on the fly, FD | 1.4e-5 to 1.1e-4 (3.3e-6 to 5.5e-5 on 16 times the samples) | 0.010 |
| FastGB | 7.6e-4 to 2.5e-3 | 1.32 |
| JaxGB | 7.6e-4 to 2.5e-3 | 1.33 |
| JaxGBaccurate | 6.8e-4 to 2.4e-3 | 1.27 |

- Beside a louder binary the mismatch of the STFT responses rises to the neighbour's own template
  error (3e-8 to 5e-8 of its power) times (SNR of the neighbour / SNR)².
- Amplitude. The low-pass filter of the preprocessing scales a source by 0.99932 at 20.4 mHz and
  0.99977 at 15.6 mHz, which no waveform models, and the Mojito binaries keep a constant amplitude
  where the model of LAT and GBGPU evolves it as f^(2/3) (3e-5 to 1e-4 over the span). Together
  they move the peak of A, and of ι through its correlation, by up to 0.18σ for the source of SNR
  223; the amplitude ratio predicts the shift to three digits.
- FastGB, JaxGB and JaxGBaccurate form X, Y and Z with one arm length, while the Mojito pipeline
  combines the links with the time-varying light travel times. A check with pytdi on the raw
  data showed that the Mojito galaxy is JaxGBaccurate's link responses (mismatch 7.8e-16 to
  3.9e-12 against the file's η_ij) combined by pytdi's TDI 2.0 (0 to 5.6e-12 against the file's
  X2, Y2, Z2), while jaxgb's own `get_tdi` gives 9.4e-4 to 2.4e-3. The light travel times move
  within ±0.08 s of their mean over the span, up to 1.0e-2 rad of delay phase at 20.4 mHz.
- The pair: with the neighbour left in the noiseless galaxy, the peaks move by at most 0.031σ in
  the STFT and 0.033σ in the FD, and the A of the weaker binary by 0.115σ. With the noise the
  peaks move by up to 2.5σ, alike for all six waveforms; the CSD at its default levels describes
  the Mojito noise to 0.7 %.
- The taper of the tracked settings file, `window_taper_duration = 1/start_freq` = 71 s for its
  band from 14 mHz, leaves up to 1.9e-2 of a tone's power outside 10 side bins and 2.9e-3 outside
  60, against 1.4e-3 and 1.1e-7 with the taper of 1e4 s used here.

## The galactic-binary block of the global fit with each response

The parameter estimation of the GB block on the Mojito-light data, started at the injected
values, run once with TDI on the fly and once with DFFR on the same data, settings and random seed,
each for 1.45 days of wall clock on one A100 (`scripts/dffr/gb_high_freq_globalfit_settings.py`).
Data: all 713 one-day segments the trims leave; LAT's symmetric window of taper 1e4 s, as
production; 16 side bins; fixed noise weights. Grid from 9.514 to 23.0 mHz: 38 bands, 36 sampled
(f0 from 9.578 to 22.898 mHz), chosen so that every catalogue binary above 9.75 mHz, the two of
highest frequency included, lies inside the sampled range. 56 catalogue binaries started at their
catalogue values in every walker and temperature, 64 leaf slots with the reversible jump of the
production move (`GBSpecialRJPriorMove`: one birth or death proposal per slot and 50 Fisher jumps
per live binary per iteration, then tempering swaps), 32 walkers, 24 temperatures, 64 cells per
proposal batch and 2 rows per tempering batch (the settings file's 512 and 128 ran out of the
A100's 40 GiB). Medians per iteration:

| | TDI on the fly | DFFR |
| --- | --- | --- |
| Iterations in 1.45 days | 67 | 122 |
| Time per iteration | 31.6 min | 17.2 min |
| In-model likelihood evaluations | 1001 s | 298 s |
| In-model Fisher matrices | 22.4 s | 8.5 s |
| Reversible-jump step | 27.0 s | 10.2 s |
| Tempering | 454 s | 390 s |
| GPU memory, peak | 31.1 GiB | 28.7 GiB |
| Injected binaries alive in every cold walker | 56 of 56 at every iteration | 56 of 56 at every iteration |
| In-model acceptance, cold chain | 0.463 | 0.463 |

The two posteriors over the first 67 iterations of each run (the first 25 % dropped, cold chain,
leaves matched to the injected binaries by f0), over the 56 binaries; differences in σ of TDI on
the fly:

| Parameter | ln A | f0 | ḟ | φ0 | cos ι | ψ | α | sin δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Difference of the means, median | 0.033 | 0.065 | 0.075 | 0.059 | 0.040 | 0.037 | 0.026 | 0.023 |
| Difference of the means, largest | 0.20 | 0.23 | 0.24 | 0.22 | 0.19 | 0.20 | 0.08 | 0.10 |
| Width ratio DFFR / TDI on the fly, median | 0.998 | 0.993 | 0.989 | 0.998 | 1.003 | 1.003 | 0.990 | 0.996 |
| Dispersion of (mean - injected)/σ over the binaries, TDI on the fly / DFFR | 0.99 / 1.05 | 0.69 / 0.81 | 0.61 / 0.73 | 0.74 / 0.80 | 0.95 / 1.00 | 0.59 / 0.59 | 0.92 / 0.94 | 1.05 / 1.04 |

- At equal iterations the two responses give the same posteriors within the sampling noise of the
  chains: widths within 1 % at the median, means within 0.24σ. Both runs follow the same walker
  spread curves iteration by iteration. The deterministic difference of the two responses is far
  below this: the surfaces above put their peaks within 0.003σ of each other.
- ln A, cos ι and the sky angles settle within about 30 iterations. In f0, ḟ, φ0 and ψ the walkers,
  started at one point, still spread when the runs stopped (by 2 % to 8 % over the last quarter of
  the first 67 iterations), and their posterior means sit closer to the injection than converged
  posteriors do (dispersion 0.59 to 0.81 instead of 1). The comparison therefore shows that DFFR
  gives the sampler the same likelihood as TDI on the fly, not converged posteriors of these
  parameters. A first look over each run's own iterations made DFFR 13 % to 15 % wider in f0, ḟ
  and φ0, an effect of its larger iteration count that vanishes at equal iterations.
- Speed: with identical settings an iteration takes 1.84 times less with DFFR, its in-model
  likelihood evaluations 3.4 times less and its Fisher matrices 2.6 times less. Tempering, mostly
  the building of the swap buffers, gains 1.16. Each binary is one thread block and the A100 holds
  216 to 324 of them at once, so a likelihood call of 64 binaries costs about as much as one of 200
  to 300: the batch size, set by the memory per cell (a cell holds its band's store window of the
  residual, 18.1 MiB, its own copy of the inverse 3 × 3 CSD, 54.3 MiB, and a template twin,
  18.1 MiB, about 153 MiB measured), sets most of the time per iteration.
- Observations on the GB block, alike in both runs: prior births in hot chains with large negative
  ḟ have tracks that leave their band's store window (1.1 % of the evaluated templates), and
  tempering swaps carry some into cold walkers (up to 4 and 3 in one walker, at the prior's lower
  amplitude edge), after which the move's rebuilt ln L differs from its bookkeeping by up to 233;
  the move writes the leaves back in band order, so a slot is not a binary; the sampled f0 range
  ends at the last band edge below the top of the grid, and with bands 3 to 11 mHz wide above
  17 mHz one bin more at the start of the grid moves that end from 22.90 to 19.92 mHz.

## Speed on the A100 and on the CPU

DFFR against the existing response on the A100 (a year of one-day segments, the ESA production
fixture; `L1Orbits` within 3 % except at batch 1), as the ratio of the existing time to the DFFR
time, with the final Fisher kernel:

| Method | Batch 1 | 10 | 100 | 1000 | 10 000 | Per source at 10 000, existing / DFFR [µs] |
| --- | --- | --- | --- | --- | --- | --- |
| `fill_global_stft` | 5.2 | 5.7 | 5.8 | 7.9 | 8.8 | 57.9 / 6.6 |
| `get_ll_stft` | 7.2 | 5.0 | 5.1 | 7.2 | 8.2 | 63.5 / 7.7 |
| `get_ll_grad_stft` | 7.5 | 4.9 | 4.3 | 4.9 | 4.9 | 998 / 206 |
| `get_swap_ll_stft` | 4.7 | 3.4 | 3.4 | 4.0 | 4.1 | 179 / 43.7 |
| `get_fstat_ll_stft` | 4.5 | 4.9 | 4.8 | 4.7 | 4.0 | 1077 / 273 |
| `get_swap_ll_grad_stft` | 6.4 | 2.8 | 2.2 | 2.2 | 2.1 | 5457 / 2584 |
| `information_matrix`, 9 parameters | 1.6 | 1.9 | 1.9 | 1.8 | 2.0 | 606 / 309 |
| `information_matrix`, 8 parameters | 1.8 | 1.9 | 2.0 | 1.8 | 1.9 | 513 / 266 |

- The gate of the effort, DFFR no slower than the existing response in any method the global fit
  calls, holds at every batch size. The likelihood methods are 2.1 to 8.8 times faster from 1000
  sources on; up to 100 sources a call costs about the same as for one, since the device is not
  filled.
- The DFFR Fisher matrix of the first build was 1.3 to 1.5 times slower than the existing one from
  1000 sources on nine parameters: its final reduction, staged in a static array, left one block
  per SM, and its table build ran one stencil template at a time with 12 to 24 of a block's 128
  threads, half of the kernel's time. Two changes brought it to 0.51 to 0.63 of the existing time:
  the reduction staged in the dynamic shared memory that is free after a source's last pass, and
  the tables built in waves straight into the Fisher tables with all threads of the block, through
  the build the tile uses (`dffr_build_rows`). The build now takes 12 % to 15 % of the kernel and
  the pixel loop 85 % to 87 %. 8 segment rows per pass instead of 4 are 0.54 to 0.93 times the
  time up to 100 sources and 1.25 to 1.32 from 1000, where a second block per SM wins;
  `STFTGBComputations` keeps 4 rows for every batch.
- Against standalone dffr on the same inputs at 10 000 sources, DFFR in the comp takes 4 % to 7 %
  more time per source in `fill_global` and 6 % to 13 % more in `get_ll`, with the standalone's
  (d|d) step removed.
- Occupancy. Capping the registers of `domains.cu` at 168 makes every method of the existing
  response 4 % to 15 % faster (3 blocks per SM instead of 2) but the DFFR swap and swap gradient
  0.5 % to 1 % slower (spills in `add_ip_swap_contrib`), and it changes rounding (12 fused
  multiply-adds split in `add_ip_contrib`: inner products move by up to 4e-16 of their peaks,
  gradients by 5.4e-10). The author left it out. 16 rows per pass for the two-template DFFR
  consumers are 5 % to 17 % slower than 32 at 10 000 sources without the cap; they stay at 32.
  Device link-time optimisation would give the DFFR two-template kernels a third block per SM but
  needs `libnvvm` next to nvlink, which the pip toolkit does not provide; not timed.
- CPU, one thread, 32 segments (`-Os`, the optimisation level of GBGPU's CPU module): DFFR takes
  0.28 to 0.40 of the time of the existing response in the likelihood methods, and its Fisher
  matrix 0.82 (batch 1) to 1.13 (batch 64), since it costs per pixel where the existing Fisher
  costs per segment. Compiled at `-O3`, which the module's build does not apply, the DFFR code runs
  2.3 to 2.9 times faster and its outputs move at rounding level (templates within 8e-15 of their
  peaks, gradients 6e-11); the DFFR Fisher then takes 0.30 to 0.42 of the existing one.

## Kernel resources on the A100 and the H100

Registers, stack and shared memory of the kernels as GBGPU links them (`kernel_resources.py`;
with relocatable device code a kernel gets the largest register count among itself and the
functions it calls). Blocks of 128 threads per SM follow from 65 536 registers per SM (4 blocks up
to 128 registers, 3 up to 168, 2 up to 255) and from the shared memory (164 kB per SM on the A100,
228 kB on the H100). One DFFR template holds 32 segment rows of 1012 bytes, the source constants
and the settings: 32 672 bytes.

| Kernel | Linked registers, A100 / H100 | Blocks per SM by registers | Blocks per SM by shared memory, A100 / H100 |
| --- | --- | --- | --- |
| existing response, every STFT kernel and its Fisher (set by `STFTFresnelGPU::get_fourier_value`) | 244 / 222 | 2 | 16 and more |
| DFFR `get_ll`, `fill_global`, `get_ll` gradient | 154 to 164 | 3 | 4 / 6 |
| DFFR swap, F-statistic (set by `STFTDomainGPU::add_ip_swap_contrib`) | 220 / 222 | 2 | 2 / 3 at 32 rows, 3 / 5 at 16 |
| DFFR swap gradient (`Template::values` and `build` out of line) | 254 / 254 | 2 | 2 / 3 |
| DFFR Fisher, 4 rows per pass, 8 / 9 parameters (66 272 / 77 088 bytes of dynamic shared memory) | 176 / 183 | 2 | 2 / 3 and 2 / 2 |

The two-template DFFR kernels take 65 344 bytes of dynamic shared memory at 32 rows, above the
48 kB a launch gets by default; the launch path raises a kernel's limit within the device's opt-in
limit (163 kB per block on the A100, 227 kB on the H100). The H100 numbers come from compiling the
module for sm_90 with nvcc; no H100 timing was made.

## Unit tests

- LAT `tests/test_orbit_kinematics.py` (13 tests) and `tests/test_fresnel.py` (9 tests): the
  tables above; CPU and A100.
- GBGPU `tests/test_stft_gb_dffr.py` (38 tests on the A100, 27 on the CPU, where the CUDA classes
  are skipped). The six likelihood methods of `STFTGBComputations(response="dffr")` on a fixture
  of 24 six-hour segments: `fill_global` followed by the inner product equals `get_ll` (rtol
  1e-10); swap(A, A) equals get_ll(A) and swap on the union band equals the inner products of
  compact templates; the F-statistic's N and M equal the get_ll and swap terms of its filters
  (rtol 1e-9); both gradients equal host central differences (rtol 1e-6); each method equals its
  entry point called with the expected arguments bit for bit, in two settings that differ in
  every argument the comp passes; the envelope default follows the group; a replica rebuilt
  through LAT's replica builder gives the same outputs bit for bit; DFFR against the existing
  response on the fixture (overlap above 0.99, norm ratio in (0.9, 1.1)); the Fisher matrix
  against a reference built from `fill_global` templates to 1e-13 of sqrt(F_ii F_jj) (measured
  2.3e-15 to 3.3e-15), its exact amplitude and phase entries (F_AA A² = (h|h) to 3.9e-14, F_φ0φ0 to
  2.1e-7, set by the rounding of the 1e-6 rad phase step), a carrier crossing a bin inside the
  stencil, a reception chirp changing sign, the parameters that leave the reception states
  unchanged; argument checks. On the A100 the CUDA classes compare CUDA with CPU within 1e-12 of
  each peak and the Fisher within 1e-7 of sqrt(F_ii F_jj) (measured 9.3e-6 on nine parameters,
  where the f̈ entries carry their finite-difference noise, 2.8e-8 on f0, ḟ, λ, β), and 16-row
  with 32-row tiles and 4-row with 8-row Fisher tables. Two CUDA tests fail by design of the GPU
  build and are left as they are: the compact template against the `fill_global` grid asks for
  bit equality, and on the A100 375 of 1080 elements differ by at most 3.8e-15 (two kernels
  compiled separately); the swap gradient against host differences at rtol 1e-6 misses in f̈ of
  the added source by 5.2e-4, a component whose step changes the phase less than rounding.
- GBGPU `tests/test_stft_gb.py`: the TDI-on-the-fly response with the DFFR setting it must ignore
  gives every output bit for bit; its exact amplitude and phase Fisher entries hold to 1.4e-6.
- LAT `tests/test_gb_likelihood_engine.py`: the STFT band engine fills, evaluates, swaps and
  removes with both responses on a container of two bands.

## Changes after the validation

At the review that closed the effort (2026-10-23), after every number above was taken:

- The floor on the local chirp rate of a reception state is `STFT_FDOT_FLOOR` (1e-22 Hz/s), the
  floor of the TDI-on-the-fly response, instead of a setting that every caller passed with that
  value; `GBDFFRSource` moved from `gb_dffr_source.hh` into `gb_dffr.hh`.
- `STFTGBComputations` defaults to `response="dffr"`, `first_order_delays=False`,
  `linear_envelope=True` and `n_side_bins=12`, and `STFTComputationGroup` to
  `linear_envelope=True`. The tests and scripts of the TDI-on-the-fly response name it, and the
  DFFR scripts pass `first_order_delays=True`, the value of the validation.

Checks of the rebuilt environment against the captures of the removal of the diagnostics
(`run_dffr.sh` parts `golden` and `against_existing`, both suites, the outcome of every test by
name):

- The golden outputs of both responses are bit-identical on the CPU (144, 50 and 25 arrays) and
  on the GPU but for `fill` (within 1.2e-16 of its peak, the order of its atomic additions). DFFR
  against the TDI-on-the-fly response is unchanged (median mismatch 5.6e-10 with the envelope,
  8.6e-6 without).
- LAT (292 tests) and GBGPU (84 tests) give the same outcome for every test, with the envelope
  on in the groups that the tests build without the keyword.
- Every part of `run_dffr.sh` ran once after the fold of the step drivers, with short spans where a
  part has one (3 days of truth, the fixtures' batch sizes): the golden outputs as above, the
  other parts with the numbers of this document, and the GB block for three iterations with each
  response on the rebuilt module, 32.2 minutes per iteration with TDI on the fly and 17.4 with
  DFFR, as in the runs of the validation.
- The effect of `first_order_delays=False`, measured when the fixtures first ran with the new
  default: on the production fixtures the DFFR templates change by up to 1.0e-4 of their peak,
  (d|h) by up to 8.5e-5 and (h|h) by up to 8.0e-6 relative, and the swap cross terms by up to
  6.5e-4. Against a time-domain truth it is not measured.

## Reproducing the checks

`scripts/dffr/run_dffr.sh --device cpu|gpu OUTPUT_ROOT [PART ...]`, from the LAT root, with
`PYTHON` the Python of the environment. Parts:

| Part | What it runs | Environment |
| --- | --- | --- |
| `golden` | the golden outputs of both responses on every fixture, compared with an earlier capture | `GOLDEN_REFERENCE` (a folder of earlier captures), `MOJITO_L1` (the Mojito L1 file, for the `L1Orbits` fixture) |
| `against_existing` | DFFR against the existing response on the production fixtures | |
| `benchmarks` | three timing runs per response, DFFR against the existing response and each against an earlier folder; on the GPU with `BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000"` | `BENCH_REFERENCE`, `MOJITO_L1`, `BENCH_ARGS` |
| `resources` | registers, stack and shared memory of the linked kernels for sm_80 and sm_90 (nvcc, no GPU) | `GBGPU_BUILD_DIR`, `REFERENCE_BUILD_DIR` |
| `accuracy`, `truth_chains`, `accuracy_response`, `step0` | the truth of the eight validation sources (pytdi, in the dffr environment) and both responses against it; the two truth chains over 2 days; `pyResponseTDI` as the injection; the eight cases of the standalone package | `DFFR_PYTHON`, `DAYS`, `N_SIDE_BINS` |
| `surfaces` | the likelihood surfaces of four waveforms against that truth | `SURFACE_PROCESSES` |
| `mojito` | six waveforms on the Mojito-light data | `MOJITO_FOLDER` |
| `globalfit_smoke`, `globalfit_<response>`, `submit_<response>` | the GB block with one response for a few iterations; one full run on the GPU at hand; the same submitted to Slurm on one A100 | `GLOBAL_FIT_RUNS_DIR`, `MOJITO_FOLDER` |

The report of two global-fit runs: `python scripts/dffr/globalfit_report.py --runs RUN_A RUN_B
--output FOLDER`. The test suites: `python -m unittest discover -v` from the LAT root and
`python -m pytest tests -rA --continue-on-collection-errors` from the GBGPU root. The standalone
dffr environment for the truth: `uv sync --extra compare` in the dffr repository.
