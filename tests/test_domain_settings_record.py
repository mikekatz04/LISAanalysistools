"""The store's ``domain_settings`` record must read back from a GPU run's store.

Before 2026-10-19 ``_write_domain_settings`` stringified anything that was not
a numpy array or a scalar. On a GPU run that is the WDM ``window`` / ``omega``
(cupy arrays) and the ``force_backend`` backend object, so every production
store carried ``window = "[0. 0. 0. ... 0. 0. 0.]"`` and a backend repr, and
``GFHDFBackend.read_domain_settings()`` raised on it (the 12-month flagship
store, job 760). The reader now drops those legacy strings (WDMSettings
rebuilds window / omega from the grid) and the writer stores host arrays and
no backend.
"""

import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from lisatools.domains import WDMSettings
from lisatools.globalfit.hdfbackend import _read_domain_settings, _write_domain_settings


def _settings():
    return WDMSettings(64, 128, 5.0, min_freq=1e-3, max_freq=2e-2, force_backend="cpu")


class DomainSettingsRecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.path = os.path.join(self.tmp, "store.h5")

    def test_legacy_gpu_record_reads_back(self):
        ref = _settings()
        with h5py.File(self.path, "w") as f:
            _write_domain_settings(f, ref)
            kw = f["domain_settings/kwargs"]
            for key in ("window", "omega"):
                del kw[key]
                kw.attrs[key] = "[0. 0. 0. ... 0. 0. 0.]"
            kw.attrs["force_backend"] = "<lisatools.cutils.LISAToolsCuda13xBackend object at 0x1>"
        with h5py.File(self.path, "r") as f:
            out = _read_domain_settings(f, force_backend="cpu")
        self.assertEqual(out.basis_shape_active, ref.basis_shape_active)
        np.testing.assert_array_equal(np.asarray(out.window), np.asarray(ref.window))
        np.testing.assert_array_equal(np.asarray(out.omega), np.asarray(ref.omega))

    def test_writer_stores_no_backend(self):
        with h5py.File(self.path, "w") as f:
            _write_domain_settings(f, _settings())
            self.assertNotIn("force_backend", f["domain_settings/kwargs"].attrs)
            self.assertIn("window", f["domain_settings/kwargs"])


if __name__ == "__main__":
    unittest.main()
