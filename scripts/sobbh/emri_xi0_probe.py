"""Fast probe: inspect the EMRI injection -> full-basis transform (xI0 slot)
without generating a waveform. Just builds curr and prints the fill_values,
the stored injection sampling coords, and the transformed full basis."""
import os
os.environ.setdefault("DATA_PROCESSOR", "mojito")
os.environ.setdefault(
    "MOJITO_DATA_PATH",
    "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/",
)
os.environ["EMRI_IDS"] = "1"
os.environ["MBHB_IDS"] = ""
os.environ["SOBHB_IDS"] = ""
os.environ.setdefault("TOBS_TARGET", "7889537.440886401")
os.environ.setdefault("NWALKERS", "1")
os.environ.setdefault("NTEMPS", "1")

import importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
settings_path = os.path.join(
    os.path.dirname(os.path.dirname(here)),
    "global_fit_input", "full_year_combined_global_fit_settings.py",
)
spec = importlib.util.spec_from_file_location("fy_settings", settings_path)
fy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fy)

curr = fy.get_global_fit_settings()
setup = curr.source_info["emri"]
inj = np.asarray(setup.injection)
print("[probe] setup.fill_values =", np.asarray(setup.fill_values), flush=True)
print("[probe] injection (sampling) =", inj, flush=True)
coords = inj[0] if inj.ndim == 2 else inj
full = np.asarray(setup.transform.both_transforms(np.asarray([coords], dtype=float)))
print("[probe] full-basis shape =", full.shape, flush=True)
print("[probe] full-basis row =", full.reshape(-1), flush=True)
fr = full.reshape(-1)
names = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS", "qK",
         "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
for n, v in zip(names, fr):
    print(f"    {n:11s} = {v}", flush=True)
print(f"\n[probe] >>> xI0 (index 5) = {fr[5]}   (FEW requires exactly 1.0)", flush=True)
