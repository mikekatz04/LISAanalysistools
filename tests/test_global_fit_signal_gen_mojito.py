"""Match + likelihood of the global-fit engine ``signal_gen`` vs mojito data.

This imports the **exact** full-year global-fit settings module
(``global_fit_input/full_year_combined_global_fit_settings.py``) the way
``scripts/run_global.py`` does, builds the ``CurrentInfoGlobalFit``, and
compares each active branch's *registered* ``signal_gen`` template to the
loaded mojito L1 data — in the run (WDM) domain and in FD bands. It is the
end-to-end check that the assembled global-fit infrastructure reproduces the
TDI-on-the-fly debug-script matches (``scripts/{mbh,sobbh}/*_likelihood_compare.py``).

* MBH and SOBBH use the validated TDI-on-the-fly path -> asserted tight.
* EMRI uses the legacy ResponseWrapper path -> report-only (its frame issue
  leaves ``|O| ~ 0.87`` per ``scripts/sobbh/emri_tdionfly_validate.py``).

One source per run (the settings module reads the source from the env at
import). Select with ``GF_TEST_SOURCE`` in {``mbh``, ``sobbh``, ``emri``,
``all``}; default ``mbh``. Auto-skips when ``bbhx`` / ``phentax`` / ``mojito``
or the mojito cache are unavailable, so CI never fails on missing deps.

    GF_TEST_SOURCE=mbh   python -m unittest tests.test_global_fit_signal_gen_mojito
    GF_TEST_SOURCE=sobbh python -m unittest tests.test_global_fit_signal_gen_mojito
    GF_TEST_SOURCE=all   python -m unittest tests.test_global_fit_signal_gen_mojito  # slow

Override ``MOJITO_DATA_PATH`` for a non-default L1 folder; ``TOBS_TARGET`` to
shrink the snippet for a faster smoke.
"""
import importlib.util
import os
import sys
import unittest

import numpy as np

# Settings module path (sibling of run_global.py's settings files).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETTINGS_DIR = os.path.join(_REPO, "global_fit_input")
_SETTINGS_FILE = os.path.join(_SETTINGS_DIR, "full_year_combined_global_fit_settings.py")

DEFAULT_CACHE = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MOJITO_DATA_PATH = os.environ.get("MOJITO_DATA_PATH", DEFAULT_CACHE)
GF_TEST_SOURCE = os.environ.get("GF_TEST_SOURCE", "mbh").lower()

SENS_MODEL = "scirdv1"
NCH = 3

# Per-class catalogue id used for the single-source snippet (matches the
# debug scripts: MBHB_ID=0, SOBBH_SRC=0; EMRI uses id 1 per the L1 indexing).
_SOURCE_ENV = {
    "mbh":   {"MBHB_IDS": "0", "EMRI_IDS": "", "SOBHB_IDS": ""},
    "sobbh": {"MBHB_IDS": "",  "EMRI_IDS": "", "SOBHB_IDS": "0"},
    "emri":  {"MBHB_IDS": "",  "EMRI_IDS": "1", "SOBHB_IDS": ""},
}
_BRANCH = {"mbh": "mbh", "sobbh": "sobbh", "emri": "emri"}

# Acceptance thresholds. TDI-on-the-fly (MBH/SOBBH) is validated to mm~1e-6
# (narrowband) / ~1e-5 (full band); generous headroom below. EMRI legacy is
# report-only with a loose floor (known frame discrepancy ~|O| 0.87).
_THRESH = {
    "mbh":   {"wdm_full": 1e-3, "fd_full": 1e-3, "fd_narrow": 5e-6, "narrow_band": (1e-3, 2.5e-2)},
    "sobbh": {"wdm_full": 1e-3, "fd_full": 5e-3, "fd_narrow": 1e-3, "narrow_band": None},  # narrow set per-f0
}


def _deps_available():
    try:
        import bbhx.mbhtdionfly  # noqa: F401
        import bbhx.sobbhtdionfly  # noqa: F401
        import phentax.waveform  # noqa: F401
        import mojito  # noqa: F401
    except Exception:
        return False, "bbhx / phentax / mojito not importable"
    if not os.path.isdir(MOJITO_DATA_PATH):
        return False, f"mojito cache not found at {MOJITO_DATA_PATH}"
    return True, ""


_DEPS_OK, _SKIP_REASON = _deps_available()


def _load_settings(source):
    """Set the per-source env, then importlib-load a FRESH settings module."""
    os.environ.update(_SOURCE_ENV[source])
    os.environ["CHOP_WINDOW"] = "1"
    os.environ["USE_TDIONFLY"] = "1"
    os.environ["MOJITO_DATA_PATH"] = MOJITO_DATA_PATH
    if _SETTINGS_DIR not in sys.path:
        sys.path.insert(0, _SETTINGS_DIR)
    # Fresh module each call so the module-level source/env logic re-runs.
    mod_name = f"_gf_settings_{source}"
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(mod_name, _SETTINGS_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    curr = mod.get_global_fit_settings()
    return mod, curr


def _attr(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    if hasattr(obj, "settings") and hasattr(obj.settings, name):
        return getattr(obj.settings, name)
    raise AttributeError(name)


def _data_td(gi):
    dp = gi.data_processor
    if getattr(dp, "data", None) is not None:
        return np.asarray(dp.data)[:NCH]
    return np.asarray(dp.td_signal[:])[:NCH]


def _band_mm(data_td, tmpl_td, td_set, n, dt, flo, fhi):
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, TDSignal
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from scipy.signal.windows import tukey

    win = tukey(n, 0.05)
    fd = FDSettings(N=n // 2 + 1, df=1.0 / (n * dt), min_freq=flo, max_freq=fhi,
                    force_backend="cpu")
    d = TDSignal(data_td, td_set).transform(fd, window=win)
    t = TDSignal(tmpl_td, td_set).transform(fd, window=win)
    ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=SENS_MODEL))
    O = ac.template_inner_product(t, normalize=True, complex=True)
    logL = float(np.real(ac.template_likelihood(t)))
    _, det = ac.template_snr(t)
    return dict(mm=1.0 - abs(O), reO=O.real, logL=logL, snr_det=float(det))


def evaluate(source):
    """Build the engine, run the registered signal_gen, return a metrics dict."""
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import TDSignal
    from lisatools.sensitivity import XYZ2SensitivityMatrix

    mod, curr = _load_settings(source)
    gi = curr.general_info
    key = _BRANCH[source]
    assert key in curr.source_info, f"branch {key!r} not in source_info"
    setup = curr.source_info[key]
    signal_gen = setup.signal_gen
    row = np.asarray(_attr(setup, "injection"), dtype=float)[0]

    dom = gi.domain_settings
    td_set = gi.data_td_settings
    data_td = _data_td(gi)
    n, dt = data_td.shape[1], gi.dt

    # --- primary: run-domain (WDM) match via the EXACT engine signal_gen ---
    t_run = signal_gen(*row)
    data_run = TDSignal(data_td, td_set).transform(dom)
    ac = AnalysisContainer(data_run, XYZ2SensitivityMatrix(dom, model=SENS_MODEL))
    O = ac.template_inner_product(t_run, normalize=True, complex=True)
    out = {
        "source": source, "n": n, "dt": dt,
        "wdm_full": dict(mm=1.0 - abs(O), reO=O.real,
                         logL=float(np.real(ac.template_likelihood(t_run))),
                         snr_det=float(ac.template_snr(t_run)[1])),
    }

    # --- FD-band cross-checks (TDI-on-the-fly branches expose raw_td) ---
    if source in ("mbh", "sobbh"):
        params_in = _attr(setup, "transform").both_transforms(row)
        wrap = (mod._get_mbh_tdionfly_wave_wrap(gi) if source == "mbh"
                else mod._get_sobbh_wave_wrap(gi))
        tmpl_td = np.asarray(wrap.raw_td(*params_in))[:NCH]
        out["fd_full"] = _band_mm(data_td, tmpl_td, td_set, n, dt, 1e-4, 2.5e-2)
        if source == "mbh":
            out["fd_narrow"] = _band_mm(data_td, tmpl_td, td_set, n, dt, 1e-3, 2.5e-2)
        else:  # SOBBH: carrier +-2 mHz from f_low (full-basis slot 6)
            f0 = float(params_in[6])
            out["fd_narrow"] = _band_mm(data_td, tmpl_td, td_set, n, dt,
                                        max(1e-4, f0 - 2e-3), f0 + 2e-3)
            out["f0"] = f0
    return out


def _report(m):
    lines = [f"[{m['source']}] N={m['n']} dt={m['dt']}"]
    for band in ("wdm_full", "fd_full", "fd_narrow"):
        if band in m:
            b = m[band]
            lines.append(f"  {band:10s} mm={b['mm']:.3e}  Re(O)={b['reO']:+.5f}  "
                         f"logL={b['logL']:+.4e}  SNRdet={b['snr_det']:+.2f}")
    return "\n".join(lines)


@unittest.skipUnless(_DEPS_OK, _SKIP_REASON)
class GlobalFitSignalGenMojitoTest(unittest.TestCase):
    """Drive the exact engine signal_gen vs mojito data per source class."""

    @unittest.skipUnless(GF_TEST_SOURCE in ("mbh", "all"), "GF_TEST_SOURCE != mbh/all")
    def test_mbh(self):
        m = evaluate("mbh")
        print("\n" + _report(m), flush=True)
        th = _THRESH["mbh"]
        self.assertLess(m["wdm_full"]["mm"], th["wdm_full"], "MBH WDM full-band mm")
        self.assertLess(m["fd_full"]["mm"], th["fd_full"], "MBH FD full-band mm")
        self.assertLess(m["fd_narrow"]["mm"], th["fd_narrow"], "MBH FD >1mHz mm")
        self.assertGreater(m["fd_narrow"]["reO"], 0.99, "MBH FD >1mHz Re(O)")

    @unittest.skipUnless(GF_TEST_SOURCE in ("sobbh", "all"), "GF_TEST_SOURCE != sobbh/all")
    def test_sobbh(self):
        m = evaluate("sobbh")
        print("\n" + _report(m), flush=True)
        th = _THRESH["sobbh"]
        self.assertLess(m["wdm_full"]["mm"], th["wdm_full"], "SOBBH WDM full-band mm")
        self.assertLess(m["fd_full"]["mm"], th["fd_full"], "SOBBH FD full-band mm")
        self.assertLess(m["fd_narrow"]["mm"], th["fd_narrow"], "SOBBH FD carrier mm")

    @unittest.skipUnless(GF_TEST_SOURCE in ("emri", "all"), "GF_TEST_SOURCE != emri/all")
    def test_emri_reportonly(self):
        # EMRI is the legacy path with a known frame discrepancy: report mm /
        # logL and assert only a loose floor so this never gates CI.
        m = evaluate("emri")
        print("\n" + _report(m), flush=True)
        self.assertGreater(1.0 - m["wdm_full"]["mm"], 0.70,
                           "EMRI legacy |O| floor (known frame issue)")


if __name__ == "__main__":
    unittest.main()