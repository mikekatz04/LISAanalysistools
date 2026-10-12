"""The monitor page carries the foreground-ratchet diagnosis section.

User request 2026-10-02 (6mo job 675, after the first ratchet cycle released
and the foreground climbed back): "add diagnostic plots that help us diagnose
this to the html ... whitened (included foreground) residuals ... check where
leaves added". The generator is a script module (it raises SystemExit on
import by design), so this test reads its source the way
test_submit_scripts_layout reads the launchers: every panel the section
promises must be rendered somewhere, the section must be in the page body and
reachable from the nav, and the whitening must be per walker against that
walker's OWN noise (the whitening its likelihood used), never one shared
curve.

Layout since 2026-10-11 (user request: succinct first, detail grouped and
collapsed): the foreground panels that do not need the ratchet (the whitened
residual per band and its zooms, the unresolved-source scorecard, the truth
decomposition) sit in the noise group right after the residual spectrum; the
ratchet-only panels (timeline, where the leaves went) form their own group,
rendered only when the store ran the ratchet; the RJ-traffic panel joined the
search section.
"""

import os
import re
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEN = os.path.join(_HERE, os.pardir, "src", "lisatools", "globalfit", "monitor",
                    "_generator.py")

PANELS = ("fg_ratchet_timeline", "fg_whitened_bands", "fg_whitened_zoom",
          "fg_resid_psd_zoom", "fg_leaf_delta", "fg_leaf_delta_zoom",
          "fg_missed_vs_row", "fg_truth_decomp", "fg_rj_per_band")


class RatchetPanelsLayoutTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(_GEN, encoding="utf-8") as fh:
            cls.src = fh.read()

    def test_every_promised_panel_is_rendered_and_placed(self):
        for key in PANELS:
            self.assertRegex(self.src, rf'fig_b64\(fig, "{key}"',
                             f"{key} is never rendered")
            self.assertRegex(self.src, rf'img\("{key}"', f"{key} is not placed on the page")

    def test_the_section_exists_and_is_in_the_nav(self):
        self.assertIn('<section id="ratchet">', self.src)
        self.assertIn('<a href="#ratchet">', self.src)

    def test_the_foreground_panels_follow_the_residual_in_the_noise_group(self):
        grp = self.src[self.src.index('<details class="group" id="grp-noise">'):]
        grp = grp[:grp.index("</details>")]
        i_res = grp.index('<section id="resid">')
        i_fg = grp.index('<section id="foreground">')
        self.assertLess(i_res, i_fg)
        for key in ("fg_whitened_bands", "fg_whitened_zoom", "fg_resid_psd_zoom",
                    "fg_missed_vs_row", "fg_truth_decomp"):
            self.assertIn(f'img("{key}"', grp[i_fg:], key)

    def test_the_ratchet_only_panels_are_their_own_group_when_the_ratchet_ran(self):
        i_if = self.src.index('if "fg_ratchet_timeline" in IMGS:')
        blk = self.src[i_if:self.src.index('html = f"""<title>')]
        self.assertIn('<details class="group" id="grp-ratchet">', blk)
        for key in ("fg_ratchet_timeline", "fg_leaf_delta", "fg_leaf_delta_zoom"):
            self.assertIn(f'img("{key}"', blk, key)
        self.assertIn("{RATCHET_GROUP}", self.src)
        srch = self.src[self.src.index('<section id="search">'):]
        self.assertIn('img("fg_rj_per_band"', srch[:srch.index("</section>")])

    def test_the_captions_render_even_when_the_panels_cannot(self):
        # defaults exist before any try block can fail, so the f-string page
        # template never hits a NameError on a store without ratchet rows
        self.assertIn('cap_fgA = cap_fgB = cap_fgC = ""', self.src)
        self.assertIn("FGW = {}", self.src)
        self.assertIn("RATCHET = {}", self.src)

    def test_each_walker_is_whitened_by_its_own_noise(self):
        blk = self.src[self.src.index("FOREGROUND DIAGNOSIS: the whitened residual"):
                       self.src.index('fig_b64(fig, "fg_whitened_bands")')]
        # the loop over walkers builds THAT walker's instrument + foreground
        self.assertRegex(blk, r"for _w, _wc, _wl in _WSHOW:")
        self.assertRegex(blk, r"psd_cold\[-1, _w\]")
        self.assertRegex(blk, r"gal_cold_phys\[-1, _w\]")
        # and the Gaussian reference marks are the Exp(1) ones
        self.assertIn("1.0 / np.log(2.0)", blk)
        self.assertIn("4.605", blk)

    def test_the_noise_panels_show_the_max_and_min_lnl_walkers_not_a_mean(self):
        # user request 2026-10-02: "rather than showing the mean of the 4, show the
        # max logL and min logL from the cold chain last sample"
        blk = self.src[self.src.index("FOREGROUND DIAGNOSIS: the whitened residual"):
                       self.src.index('fig_b64(fig, "fg_resid_psd_zoom")')]
        self.assertIn("WWORST = int(np.argmin(ll[-1]))", blk)
        self.assertIn('_WSHOW = ((WBEST, CYAN, "max lnL"), (WWORST, AMBER, "min lnL"))', blk)
        self.assertIn("WBEST = int(np.argmax(ll[-1]))", self.src)
        for arr in ("_rat_tot", "_rat_inst", "_rat_est", "_mm", "_tail", "_share",
                    "_psd_b", "_Stot_b", "_Sinst_b"):
            self.assertNotIn(f"np.nanmean({arr}, axis=0)", blk, arr)
        dec = self.src[self.src.index("(E) what the fitted foreground is made of"):
                       self.src.index('fig_b64(fig, "fg_truth_decomp")')]
        self.assertIn("_WSH_R = (int(np.argmax(ll[-1])), int(np.argmin(ll[-1])))", dec)
        self.assertNotIn("_share_r.mean(axis=0)", dec)

    def test_the_scorecard_rows_share_one_y_axis(self):
        blk = self.src[self.src.index('_tot7_r = {'):
                       self.src.index('fig_b64(fig, "fg_missed_vs_row")')]
        self.assertIn("sharex=True, sharey=True", blk)

    def test_the_reference_row_is_the_one_before_the_first_nudge(self):
        blk = self.src[self.src.index("FOREGROUND RATCHET: the timeline"):
                       self.src.index('fig_b64(fig, "fg_ratchet_timeline")')]
        self.assertIn('s == "noise_ratchet_search"', blk)
        self.assertIn("REF = _gate_rows[0] - 1", blk)

    def test_failures_report_to_missing_not_the_page(self):
        for tag in ("whitened residual per band unavailable",
                    "ratchet timeline unavailable", "leaf-delta panel unavailable"):
            self.assertRegex(self.src, rf'MISSING\.append\(f"{re.escape(tag)}')


if __name__ == "__main__":
    unittest.main()
