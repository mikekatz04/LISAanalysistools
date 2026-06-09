"""Rebuild the WDM lookup table for Sangria-shaped data (dt=5s, ~1 yr).

Layer geometry from the existing ``wdm_lookup_per_n_NF365_NT10240_fix.h5`` is:

    Nf=365, Nt=10240, data_dt=10s
    -> layer_dt = Nf*data_dt = 3650 s   (pixel width in time)
       layer_df = 1/(2*Nf*data_dt) = 1/7300 ≈ 1.3699e-4 Hz   (pixel width in freq)
       Tobs = Nf*Nt*data_dt = 37,376,000 s ≈ 1.18 yr

Sangria has data_dt=5s and ~30 Ms of usable data after preprocessing. To
keep both pixel widths identical we double Nf so that ``Nf*data_dt`` stays
at 3650 s; ``Nt`` is then the largest even integer whose ``Nf*Nt*data_dt``
fits inside the Sangria window:

    Nf=730, Nt=8244, data_dt=5s
    -> layer_dt = 730*5 = 3650 s        (unchanged)
       layer_df = 1/(2*730*5) = 1/7300  (unchanged)
       Tobs = 730*8244*5 = 30,090,600 s  (~0.953 yr; fits inside ~30.1 Ms data)

Other build knobs (eps_freq=1e-3, num_layers_diff=5, fdot_vals=[0.0],
m_ref=21, nchannels=3) mirror the original file's per-layer table shape
``(Nt, 1, 12000)``.
"""

import os

import numpy as np

# fastlisaresponse registers the ``fastlisaresponse_*`` backends that
# ``DomainSettingsBase.supported_backends()`` resolves against when no explicit
# ``force_backend`` is threaded through (the WDMLookupTable build does that
# internally for its sub_settings).
# Phase 3L.7l: import fastlisaresponse removed (no longer registers backends).
# If a script needs the backend registry, import lisatools / gbgpu / bbhx as appropriate.
from lisatools.domains import WDMLookupTable, WDMSettings


def main():
    settings = WDMSettings(
        Nf=730,
        Nt=8244,
        dt=5.0,
        oversample=16,
        min_freq=1e-4,
        max_freq=3.5e-2,
        force_backend="cpu",
    )

    print(f"layer_dt = {settings.layer_dt} s (target 3650)")
    print(f"layer_df = {settings.layer_df} Hz (target {1.0 / 7300:.6e})")
    print(f"Tobs     = {settings.Tobs} s (target ~30.1e6)")
    print(f"ind_min_f/ind_max_f = {settings.ind_min_f}/{settings.ind_max_f}")

    # Match the existing per_n file's table shape: (Nt, num_fdot=1, num_f=12000).
    eps_freq = 1e-3
    num_layers_diff = 5
    norm_freq_single_layer, m_diffs, m_ref = WDMLookupTable.apply_eps_frequency(
        eps_freq, settings, m_ref=21, num_layers_diff=num_layers_diff
    )
    assert len(norm_freq_single_layer) == 1000, len(norm_freq_single_layer)
    assert len(m_diffs) == 12, len(m_diffs)

    fdot_vals = np.array([0.0])

    out_path = "wdm_lookup_per_n_NF730_NT8244_sangria.h5"
    if os.path.exists(out_path):
        os.remove(out_path)

    print(f"Building {out_path} (per_n, Nt={settings.Nt}, num_f={len(norm_freq_single_layer)*len(m_diffs)}) ...")

    table = WDMLookupTable(
        settings,
        nchannels=3,
        m_ref=m_ref,
        norm_freq_single_layer=norm_freq_single_layer,
        m_diffs=m_diffs,
        fdot_vals=fdot_vals,
        store_path=out_path,
        batch_size_gen=20,
        build_kind="per_n",
        verbose=True,
    )

    print("done. table_cos shape:", table.table_cos.shape,
          "table_sin shape:", table.table_sin.shape)


if __name__ == "__main__":
    main()
