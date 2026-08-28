"""`band_slab_Nf` is a RUN constant -- its cache must survive a rebind.

Production audit (2026-08-28): `fill_indmap_data` cost 129.9 s/iteration
(~6.6% of a ~1954 s iteration) and was almost entirely
`band_support_halfwidths` -- a 1232-iteration python loop over gbgpu's
`get_N` -- re-running ~2,100x/iteration. Cause: every buffer rebind
called `_invalidate_slab_metadata_cache`, which dropped
`_band_slab_Nf_cached` along with the two caches that genuinely DO
consume the live binding.

The smoking gun was a twin-call gap: `fill_indmap_psd` invokes the same
function with the same inputs (only a bool differs) at 0.195 ms/chunk
against 56 ms for the data call -- 300x, which is cold-vs-warm cache and
not index arithmetic.

`_compute_band_slab_Nf` reads only `_wdm_band_slab_layers`,
`_basis_settings`, `df` and `band_edges` -- all fixed at construction, so
the value cannot change across a rebind (a rebind changes which CELLS a
buffer holds, not the band grid). Keeping it cached is therefore
bit-identical.

`_slab_min_f_cached` and `_min_freq_inds_wdm_cached` DO depend on the
live binding and must still be dropped -- that is the distinction this
test pins.
"""

import unittest

from lisatools.globalfit.moves.gbbands import SubBandBuffer


class _Stub:
    """Minimal stand-in: only the cache protocol is under test."""

    _invalidate_slab_metadata_cache = (
        SubBandBuffer._invalidate_slab_metadata_cache)


class InvalidateSlabMetadataCacheTest(unittest.TestCase):
    def _primed(self):
        s = _Stub()
        s._band_slab_Nf_cached = 5
        s._slab_min_f_cached = 17
        s._min_freq_inds_wdm_cached = 42
        return s

    def test_band_slab_nf_survives_a_rebind(self):
        s = self._primed()
        s._invalidate_slab_metadata_cache()
        self.assertIn("_band_slab_Nf_cached", s.__dict__)
        self.assertEqual(s._band_slab_Nf_cached, 5)

    def test_bind_dependent_caches_are_still_dropped(self):
        s = self._primed()
        s._invalidate_slab_metadata_cache()
        self.assertNotIn("_slab_min_f_cached", s.__dict__)
        self.assertNotIn("_min_freq_inds_wdm_cached", s.__dict__)

    def test_repeated_invalidation_is_safe_when_nothing_is_cached(self):
        s = _Stub()
        s._invalidate_slab_metadata_cache()
        s._invalidate_slab_metadata_cache()
        self.assertNotIn("_slab_min_f_cached", s.__dict__)

    def test_deepcopy_idiom_preserved(self):
        # Cache presence is tracked by __dict__ membership -- no sentinel
        # objects -- so deepcopy/pickle stay safe (repo-wide rule).
        s = self._primed()
        s._invalidate_slab_metadata_cache()
        for v in s.__dict__.values():
            self.assertNotIn(type(v).__name__, ("_Sentinel", "sentinel"))


class ComputeBandSlabNfInputsTest(unittest.TestCase):
    """The value is a run constant: guard the reason the cache is safe."""

    def test_compute_reads_no_live_binding_state(self):
        # Match real ATTRIBUTE ACCESS (`self.x`), not bare names -- several
        # bind-scoped names appear in this method's prose comments, which
        # is not a read.
        import inspect
        import re
        src = inspect.getsource(SubBandBuffer._compute_band_slab_Nf)
        reads = set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)", src))
        for bind_scoped in ("special_indices_unique", "special_index",
                            "inds_main_band_sorter", "band_buffer",
                            "slab_min_f", "min_freq_inds"):
            self.assertNotIn(
                bind_scoped, reads,
                f"_compute_band_slab_Nf now reads bind-scoped "
                f"self.{bind_scoped}; caching it across rebinds is no "
                f"longer safe.")
        # Positive statement of what it MAY read -- all fixed at
        # construction. A new name here is a prompt to re-audit, not an
        # automatic failure.
        self.assertTrue(
            reads <= {"_wdm_band_slab_layers", "_basis_settings", "df",
                      "band_edges", "xp", "_wdm_slab_guard_layers"},
            f"_compute_band_slab_Nf reads an unexpected attribute: "
            f"{sorted(reads)} -- re-audit before trusting the cache.")


if __name__ == "__main__":
    unittest.main()
