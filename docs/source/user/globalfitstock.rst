Stock Global Fits
~~~~~~~~~~~~~~~~~~~~~

Run configurations are **installed classes, not settings files**. Pick a stock
fit, turn knobs as plain attributes, build, run::

    from lisatools.globalfit.stock import erebor

    erebor.get_stock_options()             # [(name, description), ...]
    fit = erebor.all_sources(nwalkers=4)   # or erebor.get_stock("all_sources", nwalkers=4)
    fit.general.num_iterations = 50        # plain attribute access on the blocks
    fit.emri.ntemps = 12                   # per-branch field (env EMRI_NTEMPS)
    fit.remove_branch("galfor")            # compose whole branches out (drops galfor_pe too)
    fit.pop_move("sobbh_pe")               # named moves per recipe stage (here full_pe)
    print(fit.list_moves())
    fit.build()                            # the heavy stage, on command; returns fit
    fit.run()                              # MPI run: launch under mpiexec

``all_sources`` is the production variant: the 9-month and 12-month runs
(``scripts/fstat_proposal/submit_gf_9mo_v9.sh``, ``scripts/flagship/submit_gf_12mo.sh``)
start ``scripts/fstat_proposal/run_combined_staged.py``, which clones
``erebor.all_sources(nwalkers=4)`` and replaces its recipe with the staged search + PE
recipe. ``erebor.<name>(**overrides)`` clones the module-level default instance;
``erebor.get_stock(name, **overrides)`` constructs a fresh one. ``fit.sample()`` is the
single-process generator mode (``for model, state in fit.sample(iterations=3): ...``), and
every full variant except ``noise_mojito`` / ``noise_galfor_mojito`` has a ``*_lite``
laptop-smoke twin (``erebor.gb_no_fg_lite``, ...; ``blank`` is already minimal).

Two rules make this work: nothing heavy happens in ``__init__`` (data loads,
waveform builds and HDF backends are deferred to ``.build()``), and the
pre-build fit must pickle/deepcopy — so runtime-only objects attach after the
deepcopy via ``attach_runtime_objects``.

Environment knobs are named for the field they seed: ``general.data_mode`` →
``DATA_MODE``, ``general.num_iterations`` → ``NUM_ITERATIONS``; per-branch
blocks prefix the branch (``gb.min_freq`` → ``GB_MIN_FREQ``). Resolution order
is *explicit kwarg > env var > lite preset > hard default*. Renames go through
``ENV_ALIASES`` (honored with a ``DeprecationWarning``) because an unrecognised
env var is silently ignored — a hard rename would quietly downgrade a runbook.

The building blocks
+++++++++++++++++++++++

.. automodule:: lisatools.globalfit.stock.base
    :members:
    :show-inheritance:
    :ignore-module-all:

The erebor family
+++++++++++++++++++++++

.. automodule:: lisatools.globalfit.stock.erebor

.. automodule:: lisatools.globalfit.stock.erebor.fit
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.common
    :members:
    :show-inheritance:

Per-branch settings
+++++++++++++++++++++++

Each source class owns its parameter basis and its own knob block.

.. automodule:: lisatools.globalfit.stock.erebor.gb
    :members:
    :show-inheritance:
    :exclude-members: get_n_based_band_edges

.. automodule:: lisatools.globalfit.stock.erebor.vgb
    :members:
    :show-inheritance:
    :exclude-members: vgb_preset_band_edges

.. automodule:: lisatools.globalfit.stock.erebor.mbh
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.emri
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.sobbh
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.noise
    :members:
    :show-inheritance:
    :exclude-members: get_galfor_erebor_settings

.. automodule:: lisatools.globalfit.stock.erebor.stochastic
    :members:
    :show-inheritance:
    :exclude-members: get_sgwb_erebor_settings

Transforms, wrappers and runtime glue
++++++++++++++++++++++++++++++++++++++++

.. automodule:: lisatools.globalfit.stock.erebor.transforms
    :members:
    :show-inheritance:
    :exclude-members: mbh_dist_trans, m1_m2_to_mT_q

.. automodule:: lisatools.globalfit.stock.erebor.wrappers
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.source_runtime
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.injections
    :members:
    :show-inheritance:
    :exclude-members: emri_full_to_sampling, sobbh_full_to_sampling, SyntheticEMRIProcessingStep, SyntheticSOBBHProcessingStep, mbh_phentax_injection_full_basis, AnnualModulatedGalacticForeground

The variants
+++++++++++++++++++++++

A variant's data pipeline swaps with one knob (``general.data_mode``: ``"mojito"`` by
default, ``"synthetic"`` for in-process streams, and the legacy ``"sangria"`` on
``all_sources`` only; ``blank`` supports ``"synthetic"`` only; ``noise_mojito`` /
``noise_galfor_mojito`` support ``"mojito"`` only and read it from the legacy env name
``DATA_PROCESSOR``).
``all_sources`` is the production variant; ``full_year_combined`` is a separate,
non-production variant. The six ``*_lite``
laptop-smoke twins live in :mod:`~lisatools.globalfit.stock.erebor.variants.lite`
(``noise_mojito``, ``noise_galfor_mojito`` and ``blank`` have none).

.. automodule:: lisatools.globalfit.stock.erebor.variants.all_sources
    :members:
    :show-inheritance:
    :exclude-members: active_source

.. automodule:: lisatools.globalfit.stock.erebor.variants.gb_no_fg
    :members:
    :show-inheritance:
    :exclude-members: t0, oversample

.. automodule:: lisatools.globalfit.stock.erebor.variants.vgb
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.variants.noise
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.variants.noise_mojito
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.variants.full_year_combined
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.variants.blank
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.variants.lite
    :members:
    :show-inheritance:
