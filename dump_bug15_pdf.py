"""Dump Bug #15 (WDM phitilde betainc NaN at boundary) to a PDF."""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUT_PATH = "/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools/bug15_wdm_phitilde_nan.pdf"


PAGES = [
    (
        "Bug #15 — WDM phitilde betainc NaN at boundary",
        r"""SYMPTOM
-------
Initial log_like = NaN in the WDM gb_and_foreground smoke run.
Probing showed the WDM data array was 100% NaN, while the TD/FD
inputs were clean.

ISOLATION
---------
Reduced to debug_wdm_transform_nan.py operating on a saved
td_sig.npy (shape (3, 1555200), no NaN, max |a| ~ 5.27e-20):

  WDM window: shape (2160,), nan = 2 at indices [270, 1890]
              omega at NaN = +/- 3.272492e-3 = +/- (A + B) = +/- pi/960
  WDM signal (post-transform): nan = 100% across all channels
  Lookup table (table_cos, table_sin): 12000 / 12000 NaN
""",
    ),
    (
        "Root cause",
        r"""LOCATION
--------
WDMSettings.phitilde in src/lisatools/domains.py around L1528:

    A = self.A
    B = dOmega - 2 * A
    z = self.xp.zeros(omega.shape[0])
    beta_inc_calc = (np.abs(omega) >= A) & (np.abs(omega) <= A + B)
    x = (np.abs(omega[beta_inc_calc]) - A) / B
    y = special.betainc(WAVELET_FILTER_CONSTANT,
                        WAVELET_FILTER_CONSTANT, x)
    z[beta_inc_calc] = insDOM * np.cos(y * np.pi / 2.0)

WHY IT FAILS
------------
At omega = +/- (A + B), the mathematical value of x is exactly 1.0,
but float arithmetic gives 1.0 + 1 ULP = 1.0000000000000002.
scipy.special.betainc(4, 4, x) is defined only on x in [0, 1] and
returns NaN for any x > 1.0:

    betainc(4, 4, 1.0)               = 1.0
    betainc(4, 4, 1.0000000000000002) = nan

PROPAGATION
-----------
1. Two NaN entries land in WDMSettings.window.
2. In FDSignal.wdmtransform the FD-then-IFFT path multiplies
   before_ifft *= base_window[None, None, :], seeding 2 * (Nf + 1)
   = 1442 NaN cells per channel.
3. np.fft.ifft along the last axis mixes all bins, spreading NaN
   over the entire (nchannels, Nf + 1, Nt) array (100% NaN).
4. The same path built the lookup table, so table_cos and table_sin
   were 100% NaN as well.
""",
    ),
    (
        "Fix and follow-up",
        r"""FIX (one-line, minimal)
-----------------------
Clip x into [0, 1] before betainc. The mathematical boundary
value at x = 1 is z = insDOM * cos(pi / 2) = 0, so clipping is
exact at the boundary and only suppresses the ULP overshoot:

    x = np.clip((np.abs(omega[beta_inc_calc]) - A) / B, 0.0, 1.0)
    y = special.betainc(WAVELET_FILTER_CONSTANT,
                        WAVELET_FILTER_CONSTANT, x)

POST-FIX VERIFICATION (debug_wdm_transform_nan.py)
--------------------------------------------------
  WDM window:                nan = 0
  before_ifft (post-window): nan = 0
  after_ifft:                nan = 0
  wdm.arr (active slice):    nan = 0, max |a| ~ 9e-20

FOLLOW-UP
---------
Rebuild wdm_lookup_n_ref_NF720_NT2160_3mo.h5 (build script
build_wdm_lookup_3mo.py). The cached table on disk was poisoned
by the same bug and must be regenerated for the fix to flow
through GB likelihood and swap_ll kernels.
""",
    ),
]


def main() -> None:
    with PdfPages(OUT_PATH) as pdf:
        for title, body in PAGES:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.94, title, fontsize=16, fontweight="bold")
            fig.text(
                0.06,
                0.05,
                body.strip(),
                fontsize=9.5,
                fontfamily="monospace",
                verticalalignment="bottom",
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
