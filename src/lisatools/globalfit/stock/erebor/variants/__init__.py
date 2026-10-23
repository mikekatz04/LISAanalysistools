"""Registered Erebor stock run variants.

One module per variant family, each a
:class:`~lisatools.globalfit.stock.erebor.fit.EreborFit` subclass plus its
module-level ``setup_recipe`` (the run's ``setup_function``):

- :mod:`.all_sources` -- ``all_sources``, THE production variant of the 9mo /
  12mo runs (gb, vgb, psd, galfor, mbh, emri, sobbh);
- :mod:`.gb_no_fg` -- ``gb_no_fg`` (GB only, fixed PSD) and the shared GB
  branch helpers (``prepare_gb_branch``, ``setup_gb_moves``) that
  ``all_sources`` reuses;
- :mod:`.vgb` -- ``vgb`` (verification binaries) and ``setup_vgb_moves``;
- :mod:`.full_year_combined` -- ``full_year_combined`` (MBH + EMRI + SOBBH,
  fixed noise; not the production variant);
- :mod:`.noise` -- ``noise_only`` / ``noise_sgwb``;
- :mod:`.noise_mojito` -- ``noise_mojito`` / ``noise_galfor_mojito``;
- :mod:`.blank` -- ``blank``, the zero-branch canvas;
- :mod:`.lite` -- the ``*_lite`` laptop-smoke twins and preset tables.

The classes are registered, and one default instance of each is built, in
:mod:`lisatools.globalfit.stock.erebor`; use ``erebor.<name>(...)`` or
``erebor.get_stock("<name>", ...)``.
"""
