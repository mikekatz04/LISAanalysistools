"""Combined final plot: at-injection mismatch (1-|O|) and |logL| for MBHB id0 +
SOBHB src0/src1, legacy pyResponse vs TDI-on-the-fly (scirdv1).

Consumes /tmp/{mbh,sobbh}_injection_summary.npz written by
mbh_injection_final.py / sobbh_injection_final.py. -> /tmp/injection_final.png
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

C_LEG, C_FLY = "#c0392b", "#2471a3"
mbh = np.load("/tmp/mbh_injection_summary.npz", allow_pickle=True)["summary"].item()
sob = np.load("/tmp/sobbh_injection_summary.npz", allow_pickle=True)["summary"].item()

# (label, mm_inj_leg, mm_inj_fly, logL_leg, logL_fly)
rows = []
def add(label, leg, fly):
    rows.append((label, leg["mm_inj"], fly["mm_inj"], abs(leg["logL_inj"]), abs(fly["logL_inj"])))
add("MBHB id0\nfull",  mbh[("scirdv1", "full[1e-4,2.5e-2]", "legacy")], mbh[("scirdv1", "full[1e-4,2.5e-2]", "on-fly")])
add("MBHB id0\n>1mHz", mbh[("scirdv1", ">1mHz", "legacy")],            mbh[("scirdv1", ">1mHz", "on-fly")])
add("SOBHB src0\n[10,30]mHz", sob[(0, "scirdv1", "legacy")], sob[(0, "scirdv1", "on-fly")])
add("SOBHB src1\n[10,30]mHz", sob[(1, "scirdv1", "legacy")], sob[(1, "scirdv1", "on-fly")])

labels = [r[0] for r in rows]; x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(2, 1, figsize=(11, 9))

mmL = [r[1] for r in rows]; mmF = [r[2] for r in rows]
b1 = ax[0].bar(x - w/2, mmL, w, color=C_LEG, label="legacy pyResponse")
b2 = ax[0].bar(x + w/2, mmF, w, color=C_FLY, label="TDI-on-the-fly")
ax[0].set_yscale("log"); ax[0].set_ylabel("1 - |O|  (at injection)")
ax[0].set_title("Mismatch vs mojito data at injection parameters (scirdv1)")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=9); ax[0].legend()
ax[0].grid(True, which="both", axis="y", alpha=.25)

llL = [r[3] for r in rows]; llF = [r[4] for r in rows]
b3 = ax[1].bar(x - w/2, llL, w, color=C_LEG, label="legacy pyResponse")
b4 = ax[1].bar(x + w/2, llF, w, color=C_FLY, label="TDI-on-the-fly")
ax[1].set_yscale("log"); ax[1].set_ylabel("|logL| = 1/2 <d-h|d-h>  (lower=better)")
ax[1].set_title("Likelihood vs mojito data at injection (scirdv1)")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=9); ax[1].legend()
ax[1].grid(True, which="both", axis="y", alpha=.25)

for bars in (b1, b2, b3, b4):
    for r in bars:
        ax_ = bars[0].axes
        ax_.annotate(f"{r.get_height():.1e}", (r.get_x()+r.get_width()/2, r.get_height()),
                     ha="center", va="bottom", fontsize=7, rotation=90)

fig.suptitle("Legacy pyResponse vs TDI-on-the-fly at injection parameters, vs mojito data", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("/tmp/injection_final.png", dpi=120); plt.close(fig)
print("DONE -> /tmp/injection_final.png")
