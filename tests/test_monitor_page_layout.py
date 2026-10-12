"""The full monitor page's layout: the at-a-glance block first, then the detail
groups, every one collapsed.

User request 2026-10-11: "We really want the plots and info that are most
important/succinct in capturing information appearing first (this should be
the 'short' html as well). Then after grouped by type with drop down options
defaulting the more detailed information (all info other than the important
and succinct info) to 'collapsed' so the user can see the section title and
expand it if they want." and "stack the other sources closer to the top".

The generator is a script module (it raises SystemExit on import by design),
so, like test_monitor_ratchet_panels, this reads its source: the page template
is one f-string, and its order is the page's order.
"""

import os
import re
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEN = os.path.join(_HERE, os.pardir, "src", "lisatools", "globalfit", "monitor",
                    "_generator.py")

GROUPS = ("grp-sources", "grp-gbrec", "grp-noise", "grp-gbsearch", "grp-method",
          "grp-mech", "grp-notes")


class FullPageLayoutTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(_GEN, encoding="utf-8") as fh:
            cls.src = fh.read()
        a = cls.src.index('html = f"""<title>')
        cls.tpl = cls.src[a:cls.src.index('open(OUT, "w").write(html)')]
        cls.main = cls.tpl[cls.tpl.index("<main>"):cls.tpl.index("</main>")]

    def test_the_glance_block_is_the_short_pages_and_comes_first(self):
        self.assertIn("from lisatools.globalfit.monitor._short import GLANCE_CSS, build_glance",
                      self.src)
        i_g = self.main.index('<div id="glance">')
        self.assertIn("{GLANCE_HTML}", self.main[i_g:self.main.index("</div>", i_g)])
        for gid in GROUPS:
            self.assertLess(i_g, self.main.index(f'id="{gid}"'), gid)
        self.assertIn("{GLANCE_CSS}", self.tpl)

    def test_the_groups_are_in_order_and_start_collapsed(self):
        pos = [self.main.index(f'<details class="group" id="{gid}">') for gid in GROUPS]
        self.assertEqual(pos, sorted(pos))
        self.assertEqual(GROUPS[0], "grp-sources", "the other sources open the details")
        opened = re.findall(r'<details class="group"[^>]*\bopen\b', self.src)
        self.assertEqual(opened, [], "a detail group renders open")

    def test_every_detail_section_lives_inside_a_group(self):
        grp_spans = []
        for m in re.finditer(r'<details class="group" id="([^"]+)">', self.main):
            grp_spans.append((m.start(), self.main.index("</details>\n", m.start())))
        for m in re.finditer(r'<section id="([^"]+)">', self.main):
            self.assertTrue(any(a < m.start() < b for a, b in grp_spans), m.group(1))

    def test_every_nav_link_has_a_target(self):
        nav = self.tpl[self.tpl.index("<nav>"):self.tpl.index("</nav>")]
        targets = set(re.findall(r'id="([^"]+)"', self.src))
        targets |= {"params", "gates"}            # NAV_PARAMS / NAV_GATES: their own blocks
        targets |= {"topline", "lnl", "leaves", "sources", "overlap", "noise-now"}   # the glance
        for href in re.findall(r'href="#([^"]+)"', nav):
            self.assertIn(href, targets, href)
        self.assertIn('id="grp_open"', nav)
        self.assertIn('id="grp_close"', nav)

    def test_a_link_into_a_closed_group_opens_it(self):
        js = self.src[self.src.index("// ---- the collapsed detail groups"):]
        self.assertIn('el.closest("details")', js)
        self.assertIn("hashchange", js)
        self.assertIn("localStorage", js)
        self.assertIn("catch (e)", js, "storage access must be guarded")

    def test_the_glance_and_page_ids_do_not_collide(self):
        from lisatools.globalfit.monitor import _short as sh

        with open(sh.__file__, encoding="utf-8") as fh:
            short_src = fh.read()
        glance = short_src[short_src.index('    html = f"""<section id="topline">'):]
        glance_ids = set(re.findall(r'<section id="([^"]+)">', glance[:glance.index('"""\n')]))
        self.assertEqual(glance_ids, {"topline", "lnl", "leaves", "sources", "overlap",
                                      "noise-now"})
        self.assertEqual(set(re.findall(r'id="([^"]+)"', self.src)) & glance_ids, set())
        self.assertIn("table.src", sh.GLANCE_CSS)

    def test_the_completeness_axis_is_not_capped_below_the_curve(self):
        blk = self.src[self.src.index("# ---- F2: completeness AND purity"):
                       self.src.index('fig_b64(fig, "f2_progress")')]
        self.assertNotIn("set_ylim(0, 60)", blk)
        self.assertIn("ax.set_ylim(0, 100)", blk)

    def test_every_figure_is_shrunk_before_it_is_embedded(self):
        fb = self.src[self.src.index("def fig_b64("):]
        fb = fb[:fb.index("\ndef ")]
        self.assertIn("_shrink_png(", fb)
        leaf = self.src[self.src.index("def _src_make_leaf_png("):]
        leaf = leaf[:leaf.index("\n\n\n")]
        self.assertIn("_shrink_png(", leaf)


class ShrinkPngTest(unittest.TestCase):

    def _png(self):
        import io

        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4, 3))
        ax = fig.subplots()
        x = np.linspace(0, 1, 200)
        ax.imshow(np.outer(x, x), cmap="viridis")
        ax.plot(x * 199, x * 199, color="#4FD8EB")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()

    def test_it_shrinks_and_stays_the_same_picture(self):
        import io

        from PIL import Image

        from lisatools.globalfit.monitor._png import shrink_png

        raw = self._png()
        out = shrink_png(raw)
        self.assertLess(len(out), len(raw))
        a = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), int)
        b = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), int)
        self.assertEqual(a.shape, b.shape)
        self.assertLessEqual(int(np.abs(a - b).max()), 32)
        self.assertLess(float(np.abs(a - b).mean()), 1.0)

    def test_the_knob_and_bad_input_return_the_original(self):
        from unittest import mock

        from lisatools.globalfit.monitor._png import shrink_png

        raw = self._png()
        with mock.patch.dict(os.environ, {"GF_MONITOR_PNG_QUANTIZE": "0"}):
            self.assertIs(shrink_png(raw), raw)
        self.assertEqual(shrink_png(b"not a png"), b"not a png")


if __name__ == "__main__":
    unittest.main()
