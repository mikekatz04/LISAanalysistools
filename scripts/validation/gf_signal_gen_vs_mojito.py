"""Manual runner: print the global-fit ``signal_gen`` vs mojito mm/logL table.

Thin, assert-free wrapper over
``tests/test_global_fit_signal_gen_mojito.py``. Imports the EXACT full-year
settings module, builds the engine, drives each branch's registered
``signal_gen`` against the loaded mojito data, and prints the
mm / Re(O) / logL / SNR table (the interactive analogue of the
``scripts/{mbh,sobbh}/*_likelihood_compare.py`` debug scripts, but going
through the assembled global-fit infrastructure).

Run with the ``deving`` interpreter:

    python scripts/validation/gf_signal_gen_vs_mojito.py              # all sources
    GF_TEST_SOURCE=mbh python scripts/validation/gf_signal_gen_vs_mojito.py
    MOJITO_DATA_PATH=/path/to/L1 TOBS_TARGET=2.6e6 python scripts/validation/gf_signal_gen_vs_mojito.py
"""
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TEST_FILE = os.path.join(_REPO, "tests", "test_global_fit_signal_gen_mojito.py")

_spec = importlib.util.spec_from_file_location("_gf_sig_test", _TEST_FILE)
_t = importlib.util.module_from_spec(_spec)
sys.modules["_gf_sig_test"] = _t
_spec.loader.exec_module(_t)


def main():
    if not _t._DEPS_OK:
        print(f"SKIP: {_t._SKIP_REASON}")
        return
    which = os.environ.get("GF_TEST_SOURCE", "all").lower()
    sources = ["mbh", "sobbh", "emri"] if which == "all" else [which]
    print("=" * 78)
    print(f"global-fit signal_gen vs mojito data  (sources={sources}, "
          f"sens={_t.SENS_MODEL}, cache={_t.MOJITO_DATA_PATH})")
    print("=" * 78)
    for s in sources:
        try:
            m = _t.evaluate(s)
            print(_t._report(m), flush=True)
        except Exception as exc:  # noqa: BLE001 — manual diagnostic runner
            import traceback
            print(f"[{s}] ERROR: {exc}")
            traceback.print_exc()
    print("\nDONE.")


if __name__ == "__main__":
    main()