"""Build the WDM lookup table for the 3-month Sangria smoke test.

Grid (matches what gb_and_foreground_global_fit_settings.py resolves):

    Nf=720, Nt=2160, data_dt=5s
    layer_dt = Nf*data_dt = 3600 s   (1-hour wavelets in time)
    layer_df = 1/(2*Nf*data_dt) ≈ 1.389e-4 Hz
    Tobs = Nf*Nt*data_dt = 7,776,000 s ≈ 90 days

Uses ``build_kind="n_ref_only"`` (Plan A) which is the fast, no-windowing
build that the GB C code now consumes.
"""

import os

import numpy as np

# Registering ``fastlisaresponse_*`` backends — needed because
# DomainSettingsBase.supported_backends() resolves against them when
# ``WDMLookupTable._build_lookup_table_*`` constructs its sub_settings
# without an explicit force_backend.
# Phase 3L.7l: import fastlisaresponse removed (no longer registers backends).
# If a script needs the backend registry, import lisatools / gbgpu / bbhx as appropriate.
from lisatools.domains import WDMLookupTable, WDMSettings


def main() -> None:
    settings = WDMSettings(
        Nf=720,
        Nt=2160,
        dt=5.0,
        oversample=16,
        min_freq=1e-4,
        max_freq=2.5e-2,
        force_backend="cpu",
    )

    print(f"layer_dt = {settings.layer_dt} s (target 3600)")
    print(f"layer_df = {settings.layer_df} Hz")
    print(f"Tobs     = {settings.Tobs} s")
    print(f"ind_min_f/ind_max_f = {settings.ind_min_f}/{settings.ind_max_f}")

    # Build the sub-layer frequency offsets and layer-index offsets directly
    # rather than via ``apply_eps_frequency`` — floating-point in ``np.arange``
    # there produces an off-by-one length that breaks the uniform-spacing
    # invariant ``WDMLookupTable.f_vals`` asserts. Match the original table's
    # granularity: 1000 sub-layer offsets, num_layers_diff=5 (m_diffs len 12).
    num_norm = 1000
    num_layers_diff = 5
    norm_freq_single_layer = np.linspace(
        0.0, settings.layer_df, num_norm, endpoint=False
    )
    m_diffs = (np.arange(2 * num_layers_diff + 2) - (num_layers_diff + 1)).astype(np.int32)
    m_ref = 21
    print(f"norm_freq_single_layer: {len(norm_freq_single_layer)} entries")
    print(f"m_diffs: {len(m_diffs)} entries, m_ref={m_ref}")

    fdot_vals = np.array([0.0])

    out_path = "wdm_lookup_n_ref_NF720_NT2160_3mo.h5"
    if os.path.exists(out_path):
        os.remove(out_path)

    print(
        f"Building {out_path} (n_ref_only, Nt={settings.Nt}, "
        f"num_f={len(norm_freq_single_layer)*len(m_diffs)}) ..."
    )

    # ``time_layers`` shrinks the synthetic FFT used during build (this is the
    # n_ref_only acceleration knob — only one time pixel is sampled from the
    # synthetic transform, so the build cost scales linearly in this length).
    # 256 mirrors the previous table's build setting.
    table = WDMLookupTable(
        settings,
        nchannels=3,
        m_ref=m_ref,
        norm_freq_single_layer=norm_freq_single_layer,
        m_diffs=m_diffs,
        fdot_vals=fdot_vals,
        store_path=out_path,
        batch_size_gen=20,
        build_kind="n_ref_only",
        time_layers=256,
        verbose=True,
    )

    print(
        "done. table_cos shape:",
        table.table_cos.shape,
        "table_sin shape:",
        table.table_sin.shape,
    )


if __name__ == "__main__":
    main()
