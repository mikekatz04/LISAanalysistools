"""SOBBH residual add/remove goes through the CHUNKED fill, not dense builds.

Measured 2026-08-28 (6-mo sources probe, jobs 359/364): the SOBBH branch
scored through the vectorized chunked kernel but folded sources in and out
of the residual through ``acs.apply_signal_from_params`` -> dense
``build_template``, explicitly "one-at-a-time ... never more than one live
template per split". That is 24 dense builds to expose a leaf and 24 more
to fold it back -- 48 per leaf visit, each a full TDI-on-the-fly waveform
plus a complete WDM transform (32 s apiece on this grid, measured from the
injection stage). The first leaf never finished inside a 20-25 min
preemption window.

``WDMComputationsBase.fill_global_wdm`` already does this the cheap way:
batched over rows, per-row ``data_index`` slabs, and a ``factors`` sign, so
one call replaces the whole serial pass.

THE SIGN (the thing to get right -- the vocabulary collides):

* the move's ``sign``: ``apply_signal_from_params`` documents "+1 adds to
  the residual array, -1 subtracts", and the move calls
  ``remove_cold_chain_sources`` -> ``sign=+1`` (EXPOSE the source: r += h)
  and ``add_back_in_cold_chain_sources`` -> ``sign=-1`` (fold it back into
  the fit: r -= h).
* ``fill_global_wdm``'s ``factors``: "Default +1; pass -1 to remove",
  accumulating ``factors * h`` into the buffer.

So fill's "remove" means SUBTRACT FROM THE BUFFER, which is the move's
*add_back*. The words are opposites; the NUMBERS are identical. Hence
``factors = sign``, pinned by these tests in both directions.
"""

import os
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.sobbhspecialmove import SOBBHChunkedLikeMove

# stock SOBBH waveform basis:
# (m1, m2, s1, s2, dist[Gpc], inc, f_low, lam, beta, psi, phi0)
F_LO, F_HI = 0.010, 0.020          # the comp's active band in these tests
IN_BAND, OUT_OF_BAND = 0.015, 0.050


def _coords(f_lows):
    rows = []
    for f in f_lows:
        rows.append([30.0, 25.0, 0.1, 0.2, 1.0, 0.5, f, 1.0, 0.3, 0.4, 0.6])
    return np.array(rows, dtype=np.float64)


class _FakeComp:
    """Records fill_global_wdm calls; stands in for the CUDA kernel."""

    def __init__(self):
        self.fill_calls = []

    def fill_global_wdm(self, params, templates, **kwargs):
        self.fill_calls.append(
            {"params": np.asarray(params), "templates": templates, **kwargs}
        )


class _FakeACA:
    """Single-shard ACA: fill_global_wdm's documented contract."""

    def __init__(self, nshards=1, dense_calls=None):
        self.linear_data_arr = [object() for _ in range(nshards)]
        self.xp = np
        self.dense_calls = dense_calls if dense_calls is not None else []

    def apply_signal_from_params(self, sign, params, **kwargs):
        """The DENSE path this change exists to stop using."""
        self.dense_calls.append((sign, params, kwargs))


def _move(comp, acs, dense_calls=None):
    """A REAL SOBBHChunkedLikeMove with __init__ bypassed.

    ``__new__`` rather than a SimpleNamespace so the instance is a genuine
    subclass instance: the zero-argument ``super()`` inside the override
    (the dense fallback) resolves, and the class's own ``to_chunked_basis``
    is used rather than a stand-in. The heavy ctor (chunked comp, WDM
    settings, eryn move plumbing) is exactly what we do not want here.
    """
    move = SOBBHChunkedLikeMove.__new__(SOBBHChunkedLikeMove)
    move.comp = comp
    move.acs = acs
    move.branch_name = "sobbh"
    move.m_band_half_width = 3
    move._f_band_lo = F_LO
    move._f_band_hi = F_HI
    # consumed only by the dense fallback; acs records that it was reached
    move._branch_waveform_kwargs = lambda: {}
    move._resolve_signal_gen_override = lambda ac: None
    return move


def _apply(move, coords, sign):
    return SOBBHChunkedLikeMove._apply_cold_chain_sources(move, coords, sign)


class ChunkedFillReplacesDenseBuildsTest(unittest.TestCase):

    def setUp(self):
        self.comp = _FakeComp()
        self.dense = []
        self.acs = _FakeACA(dense_calls=self.dense)
        self.move = _move(self.comp, self.acs)

    def test_one_batched_call_for_all_walkers(self):
        """THE regression: 24 walkers must cost ONE fill, not 24 builds."""
        _apply(self.move, _coords([IN_BAND] * 24), +1)
        self.assertEqual(len(self.comp.fill_calls), 1)
        self.assertEqual(self.comp.fill_calls[0]["params"].shape, (24, 11))

    def test_remove_sign_maps_to_factors_plus_one(self):
        """move sign=+1 (expose: r += h) -> fill factors=+1."""
        _apply(self.move, _coords([IN_BAND] * 4), +1)
        np.testing.assert_array_equal(
            np.asarray(self.comp.fill_calls[0]["factors"]),
            np.full(4, 1.0),
        )

    def test_add_back_sign_maps_to_factors_minus_one(self):
        """move sign=-1 (fold in: r -= h) -> fill factors=-1.

        NB fill's own docs call -1 "remove"; that is the move's add_back.
        """
        _apply(self.move, _coords([IN_BAND] * 4), -1)
        np.testing.assert_array_equal(
            np.asarray(self.comp.fill_calls[0]["factors"]),
            np.full(4, -1.0),
        )

    def test_data_index_is_the_walker_row(self):
        """Each walker's template lands in ITS OWN residual slab."""
        _apply(self.move, _coords([IN_BAND] * 5), +1)
        np.testing.assert_array_equal(
            np.asarray(self.comp.fill_calls[0]["data_index"]), np.arange(5)
        )

    def test_writes_into_the_aca_residual(self):
        _apply(self.move, _coords([IN_BAND] * 3), +1)
        self.assertIs(self.comp.fill_calls[0]["templates"], self.acs)

    def test_params_are_chunked_basis(self):
        """dist Gpc->pc and the column permutation must be applied."""
        _apply(self.move, _coords([IN_BAND]), +1)
        got = self.comp.fill_calls[0]["params"]
        want = SOBBHChunkedLikeMove.to_chunked_basis(_coords([IN_BAND]))
        np.testing.assert_allclose(got, want)
        self.assertAlmostEqual(got[0, 4], 1.0e9)   # 1 Gpc in parsec
        self.assertAlmostEqual(got[0, 5], IN_BAND)  # f_low column

    def test_m_band_half_width_is_forwarded(self):
        _apply(self.move, _coords([IN_BAND]), +1)
        self.assertEqual(self.comp.fill_calls[0]["m_band_half_width"], 3)


class ObservabilityTest(unittest.TestCase):
    """Which residual path ran must be visible in the log.

    Jobs 364 and 370 both sat ~14-16 min in SOBBH with no leaf and no
    error, and the logs could not say whether the chunked fill was even
    deployed -- the path was silent. One line per fold removes that
    ambiguity for good.
    """

    def test_chunked_fill_logs_path_and_row_count(self):
        comp = _FakeComp()
        move = _move(comp, _FakeACA())
        with self.assertLogs(
            "lisatools.globalfit.moves.sobbhspecialmove", level="INFO"
        ) as cm:
            _apply(move, _coords([IN_BAND] * 7), +1)
        line = "\n".join(cm.output)
        self.assertIn("chunked", line.lower())
        self.assertIn("7", line)          # rows actually filled

    def test_dense_fallback_says_so(self):
        """SOBBH_CHUNKED_FILL=0 must announce the slow path, not run it
        silently -- that is the configuration that costs 32 s per row."""
        comp = _FakeComp()
        dense = []
        move = _move(comp, _FakeACA(dense_calls=dense))
        with mock.patch.dict(os.environ, {"SOBBH_CHUNKED_FILL": "0"}):
            with self.assertLogs(
                "lisatools.globalfit.moves.sobbhspecialmove", level="INFO"
            ) as cm:
                _apply(move, _coords([IN_BAND] * 3), +1)
        self.assertIn("dense", "\n".join(cm.output).lower())
        self.assertEqual(len(dense), 1)   # took the base path
        self.assertEqual(comp.fill_calls, [])


class ValidityMaskingTest(unittest.TestCase):
    """Out-of-band / non-finite rows have a ZERO template: they must be
    skipped, exactly as the dense path's ``domain_error='skip'`` does."""

    def setUp(self):
        self.comp = _FakeComp()
        self.move = _move(self.comp, _FakeACA())

    def test_out_of_band_rows_are_dropped(self):
        _apply(self.move, _coords([IN_BAND, OUT_OF_BAND, IN_BAND]), +1)
        call = self.comp.fill_calls[0]
        self.assertEqual(call["params"].shape[0], 2)
        # kept rows keep their ORIGINAL walker slabs (0 and 2, not 0 and 1)
        np.testing.assert_array_equal(
            np.asarray(call["data_index"]), np.array([0, 2])
        )

    def test_non_finite_rows_are_dropped(self):
        c = _coords([IN_BAND, IN_BAND])
        c[1, 0] = np.nan
        _apply(self.move, c, +1)
        self.assertEqual(self.comp.fill_calls[0]["params"].shape[0], 1)

    def test_no_valid_rows_makes_no_call(self):
        _apply(self.move, _coords([OUT_OF_BAND, OUT_OF_BAND]), +1)
        self.assertEqual(self.comp.fill_calls, [])


class RoundTripSignTest(unittest.TestCase):
    """remove(+1) then add_back(-1) must cancel exactly.

    The residual is simulated as an accumulator so the test asserts the
    NET effect of the two passes, which is what a sign error breaks.
    """

    def test_expose_then_fold_back_cancels(self):
        acc = {"v": 0.0}

        class _Accumulating(_FakeComp):
            def fill_global_wdm(self, params, templates, **kw):
                super().fill_global_wdm(params, templates, **kw)
                # each row deposits factors * h with h == 1.0 here
                acc["v"] += float(np.sum(np.asarray(kw["factors"])))

        comp = _Accumulating()
        move = _move(comp, _FakeACA())
        coords = _coords([IN_BAND] * 6)
        _apply(move, coords, +1)
        self.assertEqual(acc["v"], 6.0)     # exposed
        _apply(move, coords, -1)
        self.assertEqual(acc["v"], 0.0)     # folded back -> residual restored


class MultiShardTest(unittest.TestCase):
    """fill_global_wdm is SINGLE-SHARD by contract (it raises on a
    multi-shard ACA). The probe runs 2 GPUs, so the override must route
    per shard the way _kernel_ll already does -- never hand it the split
    holder."""

    def test_multi_shard_does_not_pass_the_split_holder(self):
        comp = _FakeComp()
        acs = _FakeACA(nshards=2)
        move = _move(comp, acs)
        try:
            _apply(move, _coords([IN_BAND] * 4), +1)
        except NotImplementedError:
            self.fail("override handed the split ACA straight to fill_global_wdm")
        except Exception:
            pass  # shard routing needs real views; only the contract matters
        for call in comp.fill_calls:
            self.assertIsNot(
                call["templates"], acs,
                msg="multi-shard ACA passed to a single-shard-only kernel",
            )


if __name__ == "__main__":
    unittest.main()
