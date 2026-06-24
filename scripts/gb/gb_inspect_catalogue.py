"""Run FIRST once the GB mojito data is downloaded.

Loads the full-galaxy GB data + catalogue via L1ProcessingStep, prints the load
interface that works (source_ids), the catalogue structure + exact field names,
the data shape/dt/t0, and the TOP highest-frequency sources (candidates for the
band-pass-in-isolation comparison at ~15-21 mHz). Use the printed field names to
finalize gb_mojito_match.py.

GB load takes NO source_ids (whole-galaxy stream); falls back to {"gb": 0}.
"""
import os
import numpy as np
from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import L1ProcessingStep

PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
BACKEND = "cpu"
TOP = int(os.environ.get("GB_TOP", "8"))


def try_load():
    kw = dict(L1_folder=PATH, source_types=["gb"], orbits_class=L1Orbits,
              orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"), verbose=True)
    for sid in (None, {"gb": 0}, {"gb": [0]}):
        try:
            print(f"\n[try] source_ids={sid}", flush=True)
            return L1ProcessingStep(**({**kw, "source_ids": sid} if sid is not None else kw))
        except Exception as e:
            print(f"   failed: {type(e).__name__}: {e}", flush=True)
    raise SystemExit("could not load GB data with any source_ids form")


def main():
    loader = try_load()
    print("\n=== data ===", flush=True)
    times = np.asarray(loader.times); data = np.asarray(loader.data)
    print(f"  data shape={data.shape}  dt={float(loader.dt)}  t0={float(times[0])}", flush=True)

    print("\n=== catalogue keys ===", flush=True)
    print(f"  {list(loader.catalogue.keys())}", flush=True)
    for k in loader.catalogue:
        entry = loader.catalogue[k]
        print(f"\n  catalogue['{k}'] type={type(entry).__name__} "
              f"len={len(entry) if hasattr(entry, '__len__') else 'n/a'}", flush=True)
        # first source's fields
        first = entry[0] if hasattr(entry, "__getitem__") else entry
        if isinstance(first, dict):
            print("  fields:", flush=True)
            for fk in sorted(first):
                v = first[fk]
                try: vv = float(np.asarray(v))
                except Exception: vv = v
                print(f"    {fk:34s} = {vv}", flush=True)

    # frequency sort -> top sources (try common freq field names)
    gbkey = "GB" if "GB" in loader.catalogue else list(loader.catalogue.keys())[0]
    cat = loader.catalogue[gbkey]
    freq_field = None
    for cand in ("GW22FrequencySSBFrame", "GW22FrequencySourceFrame", "Frequency", "f0"):
        if isinstance(cat[0], dict) and cand in cat[0]:
            freq_field = cand; break
    print(f"\n=== top {TOP} highest-frequency '{gbkey}' sources (field '{freq_field}') ===", flush=True)
    if freq_field is not None:
        freqs = np.array([float(np.asarray(c[freq_field])) for c in cat])
        order = np.argsort(freqs)[::-1][:TOP]
        for rank, i in enumerate(order):
            print(f"  #{rank}: row {i}  f={freqs[i]*1e3:.6f} mHz", flush=True)
        # nearest-neighbour spacing of the top sources (isolation check)
        topf = np.sort(freqs)[::-1][:max(TOP, 30)]
        print(f"\n  highest f = {topf[0]*1e3:.4f} mHz; "
              f"gap to 2nd = {(topf[0]-topf[1])*1e6:.3f} uHz", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
