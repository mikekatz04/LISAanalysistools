"""rj_para_search_pe through the REAL stock fit (gated: ``RUN_GF_PARA_INTEGRATION=1``).

``erebor.gb_no_fg(debug=True)`` (3-day Tobs, synthetic data, one loud GB
injection), single process (``fit.sample()``), ``rj_para_search_pe`` added to
its ``gb_pe`` stage, a tiny search config and a 2-iteration cadence. This is
the one test that runs the real constructor (``recipe.build_gb_moves``), the
real ``setup()`` in the real propose prologue, the move-side scorer through
the band-engine routers (sig-het F-stat, parent-engine full likelihood) on the
run's own residual and priors, the head merge + GMM + npz, the holder's
adoption and the RJ births drawn from it. ~2 GB, ~2 min on the laptop CPU.
Pinned: epoch 0 built at the move's first entry and NOT drawn from by that
propose (the one-draw lag); the next draw adopts it and proposes births; the
cadence rebuilds; every build is complete.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

ENV = dict(
    USE_GPU="0", MAKE_DIAGNOSTIC_PLOTS="0", MPLBACKEND="Agg", DATA_BAND_LAYERS="8",
    GB_DEBUG="0", GB_PARA_SEARCH="1", GB_PARA_SEARCH_NTEMPS="2",
    GB_PARA_SEARCH_NWALKERS="4", GB_PARA_SEARCH_MIN_STEPS="8",
    GB_PARA_SEARCH_MAX_STEPS="16", GB_PARA_SEARCH_CONVERGE_WINDOW="4",
    GB_PARA_SEARCH_STAGE2_STEPS="6", GB_PARA_SEARCH_STAGE2_NTEMPS="2",
    GB_PARA_SEARCH_EVERY_PE="2", GB_PARA_SEARCH_INIT_NSKY_MIN="4",
    GB_PARA_SEARCH_INIT_NSKY_MAX="4",
)


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        msg = record.getMessage()
        if "GB_PARA_SEARCH" in msg or "rj-split rj_para_search_pe" in msg:
            self.lines.append(msg)


@unittest.skipUnless(os.environ.get("RUN_GF_PARA_INTEGRATION") == "1",
                     "set RUN_GF_PARA_INTEGRATION=1 (builds a real fit, ~2 GB)")
class StockFitIntegrationTest(unittest.TestCase):
    def test_build_lag_adopt_and_births(self):
        d = tempfile.mkdtemp(prefix="para_int_")
        self.addCleanup(shutil.rmtree, d, True)
        with mock.patch.dict(os.environ, ENV, clear=False):
            from lisatools.globalfit.stock import erebor

            fit = erebor.gb_no_fg(debug=True, nwalkers=2, ntemps=2, data_mode="synthetic",
                                  file_store_dir=d, make_diagnostic_plots=False)
            fit.general.random_seed = 7
            fit.general.gb_injection_params = [[3e-21, 7.5e-3, 1e-16, 0.0, 1.2, 0.9, 1.0,
                                                4.0, -0.6]]
            fit.recipe.add_move("rj_para_search_pe", stage="gb_pe", branch="gb")
            h = _Lines()
            lg = logging.getLogger("lisatools")
            old_level = lg.level
            lg.setLevel(logging.INFO)
            self.addCleanup(lg.setLevel, old_level)
            lg.addHandler(h)
            try:
                per_it = []
                for _model, _state in fit.sample(iterations=4):
                    per_it.append(list(h.lines))
                    h.lines.clear()
            finally:
                lg.removeHandler(h)
        root = os.path.join(d, "warmstart", "parasearch")
        it1, it2, it3, it4 = per_it
        self.assertTrue(any(re.search(r"epoch 0 built: 2 walkers", m) for m in it1), it1)
        self.assertTrue(any("draws nothing" in m for m in it1), it1)
        self.assertTrue(any("adopted epoch 0" in m for m in it2), it2)
        births = [int(re.search(r"viable (\d+)", m).group(1)) for m in it2
                  if "rj-split rj_para_search_pe" in m]
        self.assertTrue(births and births[0] > 0, it2)
        self.assertTrue(any(re.search(r"epoch 1 built", m) for m in it3), it3)
        self.assertTrue(any("used epoch 0" in m for m in it3), it3)       # the lag
        self.assertTrue(any("adopted epoch 1" in m for m in it4), it4)
        for k in (0, 1):
            self.assertTrue(os.path.exists(os.path.join(root, f"epoch_{k:04d}", "DONE.json")))
            self.assertTrue(os.path.exists(os.path.join(root, f"epoch_{k:04d}",
                                                        "parasearch.npz")))


if __name__ == "__main__":
    unittest.main()
