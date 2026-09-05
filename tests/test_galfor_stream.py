"""Tests for the GALFOR whole-stream source type and the joint noise variant.

``GALFOR`` is mojito's galactic-confusion stream (``data/GALFOR/L1``). Like
``NOISE`` and ``COMBINED`` it is a STREAM, not a source class: no catalogue,
no source ids, and a brick name (``GALFOR_731d_2.5s_L1.h5``) that carries no
``source{id}_`` infix, so :func:`find_file` cannot resolve it. These tests pin
the resolver, the allow-list, and the ``noise_galfor_mojito`` variant's shape.

Everything here is hermetic except :class:`RealBrickTest`, which is skipped
unless the mojito-light GALFOR and NOISE bricks are both on the machine.
"""

import copy
import os
import pickle
import unittest

from lisatools.globalfit.preprocessing import ALLOWED_SOURCES, find_stream_file

MOJITO_CACHE = os.path.expanduser(
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/data"
)


class FindStreamFileTest(unittest.TestCase):
    """The folder-based resolver for whole-stream bricks."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        # find_stream_file reads the env var by name; make sure a stray
        # override from the developer's shell cannot leak into the tests.
        self._saved = os.environ.pop("TEST_STREAM_FILE", None)

        def _restore():
            if self._saved is not None:
                os.environ["TEST_STREAM_FILE"] = self._saved
            else:
                os.environ.pop("TEST_STREAM_FILE", None)

        self.addCleanup(_restore)

    def _touch(self, name):
        path = os.path.join(self.folder, name)
        open(path, "w").close()
        return path

    def test_single_file_resolves_without_source_infix(self):
        """The real failure mode: right prefix, NO ``source0_`` infix."""
        want = self._touch("GALFOR_731d_2.5s_L1.h5")
        self.assertEqual(
            find_stream_file(self.folder, "GALFOR", "TEST_STREAM_FILE"), want
        )

    def test_ignores_non_h5_and_dotfiles(self):
        want = self._touch("GALFOR_731d_2.5s_L1.h5")
        self._touch("notes.txt")
        self._touch(".GALFOR_partial.h5")
        self.assertEqual(
            find_stream_file(self.folder, "GALFOR", "TEST_STREAM_FILE"), want
        )

    def test_two_files_disambiguated_by_prefix(self):
        want = self._touch("GALFOR_731d_2.5s_L1.h5")
        self._touch("something_else.h5")
        self.assertEqual(
            find_stream_file(self.folder, "GALFOR", "TEST_STREAM_FILE"), want
        )

    def test_two_prefixed_files_refuse_to_guess(self):
        self._touch("GALFOR_a.h5")
        self._touch("GALFOR_b.h5")
        with self.assertRaises(ValueError) as cm:
            find_stream_file(self.folder, "GALFOR", "TEST_STREAM_FILE")
        self.assertIn("TEST_STREAM_FILE", str(cm.exception))

    def test_env_override_absolute_and_basename(self):
        self._touch("GALFOR_a.h5")
        other = self._touch("GALFOR_b.h5")
        os.environ["TEST_STREAM_FILE"] = other
        self.assertEqual(
            find_stream_file(self.folder, "GALFOR", "TEST_STREAM_FILE"), other
        )
        os.environ["TEST_STREAM_FILE"] = "GALFOR_b.h5"
        self.assertEqual(
            find_stream_file(self.folder, "GALFOR", "TEST_STREAM_FILE"), other
        )

    def test_env_override_missing_raises(self):
        self._touch("GALFOR_a.h5")
        os.environ["TEST_STREAM_FILE"] = "nope.h5"
        with self.assertRaises(FileNotFoundError):
            find_stream_file(self.folder, "GALFOR", "TEST_STREAM_FILE")

    def test_missing_folder_and_empty_folder_raise(self):
        with self.assertRaises(FileNotFoundError):
            find_stream_file(
                os.path.join(self.folder, "nope"), "GALFOR", "TEST_STREAM_FILE"
            )
        with self.assertRaises(FileNotFoundError):
            find_stream_file(self.folder, "GALFOR", "TEST_STREAM_FILE")

    def test_galfor_is_an_allowed_source(self):
        self.assertIn("GALFOR", ALLOWED_SOURCES)


class NoiseGalForVariantTest(unittest.TestCase):
    """``noise_galfor_mojito``: shape only — construction stays cheap."""

    def setUp(self):
        from lisatools.globalfit.stock import erebor

        self.erebor = erebor
        self.fit = erebor.get_stock("noise_galfor_mojito")

    def test_registered(self):
        names = [n for n, _ in self.erebor.get_stock_options()]
        self.assertIn("noise_galfor_mojito", names)

    def test_two_branches_psd_and_galfor(self):
        branches = self.fit.default_branches()
        self.assertEqual(sorted(branches), ["galfor", "psd"])
        self.assertEqual(branches["psd"].ndim, 2)
        self.assertEqual(branches["galfor"].ndim, 5)

    def test_recipe_has_both_moves_on_independent_ladders(self):
        stages = self.fit.default_recipe().stages
        self.assertEqual(len(stages), 1)
        self.assertEqual(
            [m.name for m in stages[0].moves], ["psd_pe", "galfor_pe"]
        )
        self.assertIs(
            stages[0].combine_kwargs["share_temperature_control"], False
        )

    def test_joint_noise_move_collapses_to_one_move(self):
        """JOINT_NOISE_MOVE must not leave the recipe asking for an unbuilt name."""
        fit = self.erebor.get_stock("noise_galfor_mojito")
        fit.general.joint_noise_move = True
        self.assertEqual(
            [m.name for m in fit.default_recipe().stages[0].moves], ["noise_pe"]
        )

    def test_source_types_include_noise_and_galfor(self):
        self.assertEqual(
            [s.upper() for s in self.fit.general.source_types], ["NOISE", "GALFOR"]
        )

    def test_refuses_without_galfor_stream(self):
        gs = self.fit.general
        gs.source_types = ("NOISE",)
        with self.assertRaises(ValueError) as cm:
            self.fit.set_default_processor(gs)
        self.assertIn("GALFOR", str(cm.exception))
        self.assertIn("noise_mojito", str(cm.exception))

    def test_refuses_without_noise_stream(self):
        """The parent's NOISE check still applies."""
        gs = self.fit.general
        gs.source_types = ("GALFOR",)
        with self.assertRaises(ValueError) as cm:
            self.fit.set_default_processor(gs)
        self.assertIn("NOISE", str(cm.exception))

    def test_store_defaults_differ_from_noise_mojito(self):
        """A resume across a branch-set change dies with a bare KeyError."""
        other = self.erebor.get_stock("noise_mojito")
        self.assertNotEqual(
            self.fit.general.base_file_name, other.general.base_file_name
        )
        self.assertNotEqual(
            self.fit.general.file_store_dir, other.general.file_store_dir
        )

    def test_grid_matches_the_instrument_only_variant(self):
        """Same WDM grid + band as the control, so the two are comparable."""
        other = self.erebor.get_stock("noise_mojito")
        for field in ("dt", "nf", "nt", "min_freq", "max_freq",
                      "window_tukey_alpha", "edge_crop_wavelets"):
            self.assertEqual(
                getattr(self.fit.general, field),
                getattr(other.general, field),
                msg=f"{field} differs from noise_mojito",
            )

    def test_pickle_and_deepcopy(self):
        """LISA Analysis Tools-wide rule: the pre-build fit must survive both."""
        clone = pickle.loads(pickle.dumps(copy.deepcopy(self.fit)))
        self.assertEqual(clone.option_name, "noise_galfor_mojito")
        self.assertEqual(sorted(clone.default_branches()), ["galfor", "psd"])


@unittest.skipUnless(
    os.path.isdir(os.path.join(MOJITO_CACHE, "GALFOR", "L1"))
    and os.path.isdir(os.path.join(MOJITO_CACHE, "INSTRUMENT", "L1")),
    "mojito-light GALFOR + NOISE bricks not present",
)
class RealBrickTest(unittest.TestCase):
    """The loader SUMS the two streams — they must share a grid."""

    def test_galfor_brick_resolves_and_matches_the_noise_grid(self):
        import h5py

        from lisatools.globalfit.preprocessing import find_file

        galfor = find_stream_file(
            os.path.join(MOJITO_CACHE, "GALFOR", "L1"), "GALFOR",
            "MOJITO_GALFOR_FILE",
        )
        noise = find_file(
            os.path.join(MOJITO_CACHE, "INSTRUMENT", "L1"), "NOISE", 0
        )
        with h5py.File(galfor, "r") as fg, h5py.File(noise, "r") as fn:
            g = dict(fg["tdis/sampling"].attrs)
            n = dict(fn["tdis/sampling"].attrs)
            for key in ("t0", "dt", "size", "duration"):
                self.assertAlmostEqual(
                    float(g[key]), float(n[key]), places=6,
                    msg=f"GALFOR/NOISE {key} mismatch — the loader sums these",
                )
            # The brick is the unresolved-GB residual, not an analytic draw;
            # the readout depends on this, so pin the provenance.
            self.assertIn("subtracted", str(fg.attrs.get("derived_by", "")))

    def test_find_file_cannot_resolve_the_galfor_brick(self):
        """Why find_stream_file exists: no ``source{id}_`` infix in the name."""
        from lisatools.globalfit.preprocessing import find_file

        with self.assertRaises(FileNotFoundError):
            find_file(os.path.join(MOJITO_CACHE, "GALFOR", "L1"), "GALFOR", 0)


if __name__ == "__main__":
    unittest.main()
