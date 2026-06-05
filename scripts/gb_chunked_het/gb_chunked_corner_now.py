"""Snapshot corner plot from the in-progress gb_chunked_test_script.py
HDF backend. Safe to run while the sampler is still writing -- eryn
flushes incrementally and we just read the current chain.

Outputs: ``gb_chunked_posterior_snapshot.png``.

Env:
    MCMC_BACKEND_PATH  : .h5 file (default test_gb_chunked_pe_small.h5)
    DISCARD            : absolute # of leading stored samples to drop
                         (default 0; takes precedence over DISCARD_FRAC).
    DISCARD_FRAC       : fractional discard (default 0.25 if DISCARD unset).
    CORNER_PATH        : output PNG path
    NF, NT, DT         : WDM grid; needed to compute the injection
                         truth values from the test script's recipe.
                         Defaults match the optimal-config run.
    F_FRAC             : which entry of the test script's F_FRACS sweep
                         (default 0.95 -- last iteration is the one
                         the MCMC consumes).
    M_OFFSET, SOURCE_FDOT, BETA : direct passthroughs of the test
                         script's env knobs so truths line up.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
from eryn.backends import HDFBackend


# Sampled basis the test script's TransformContainer consumes.
SAMPLED_BASIS = ["amp", "f0", "fdot0", "phi0", "cosinc", "psi", "lam", "sinbeta"]


def _injection_truths():
    """Reproduce the injection params from gb_chunked_test_script.py
    in the sampled basis. Has to match the script's defaults for the
    truth markers to line up with the chain.
    """
    NF = int(os.environ.get("NF", "1460"))
    DT = float(os.environ.get("DT", "10.0"))
    f_frac = float(os.environ.get("F_FRAC", "0.95"))
    m_offset = int(os.environ.get("M_OFFSET", "0"))
    layer_df = 1.0 / (2.0 * NF * DT)
    m_ref_source = int(3e-3 / layer_df)
    f0_inj   = (m_ref_source + m_offset + f_frac) * layer_df
    amp_inj  = 8.0e-22
    fdot_inj = float(os.environ.get("SOURCE_FDOT", "1e-17"))
    phi0_inj = 2.09802430298
    inc_inj  = 0.23984234
    psi_inj  = 1.234019814
    lam_inj  = 4.09808143
    beta_inj = float(os.environ.get("BETA", "0.04"))
    return np.array([
        amp_inj,
        f0_inj,
        fdot_inj,
        phi0_inj,
        np.cos(inc_inj),
        psi_inj,
        lam_inj,
        np.sin(beta_inj),
    ])


def main():
    backend_path = os.environ.get(
        "MCMC_BACKEND_PATH", "test_gb_chunked_pe_small.h5"
    )
    out_png = os.environ.get(
        "CORNER_PATH", "gb_chunked_posterior_snapshot.png"
    )

    print(f"[snapshot] reading {backend_path}", flush=True)
    backend = HDFBackend(backend_path)
    chain_full = backend.get_chain()["gb"]
    print(f"[snapshot] chain shape (ns, ntemps, nwalkers, nleaves, ndim): "
          f"{chain_full.shape}", flush=True)

    n_stored = chain_full.shape[0]
    if n_stored == 0:
        print("[snapshot] no samples yet, nothing to plot")
        return

    if "DISCARD" in os.environ:
        discard = int(os.environ["DISCARD"])
    else:
        discard_frac = float(os.environ.get("DISCARD_FRAC", 0.25))
        discard = int(n_stored * discard_frac)
    discard = max(0, min(discard, n_stored - 1))
    chain = chain_full[discard:]
    print(f"[snapshot] discarding first {discard}/{n_stored} samples", flush=True)

    # Cold chain (temperature index 0).
    cold = chain[:, 0]                                  # (ns, nwalkers, nleaves, ndim)
    samples = cold.reshape(-1, cold.shape[-1])
    print(f"[snapshot] samples flat: {samples.shape}", flush=True)

    truths = _injection_truths()
    print(f"[snapshot] truth values: {dict(zip(SAMPLED_BASIS, truths))}",
          flush=True)

    fig = corner.corner(
        samples, labels=SAMPLED_BASIS,
        truths=truths,
        show_titles=True, title_fmt=".4e",
        quantiles=[0.16, 0.5, 0.84],
        truth_color="red",
    )
    fig.suptitle(
        f"{os.path.basename(backend_path)}  |  "
        f"discard={discard}/{n_stored}  |  samples={samples.shape[0]}",
        fontsize=10, y=1.0,
    )
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[snapshot] wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
