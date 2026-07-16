Stock Global Fits
~~~~~~~~~~~~~~~~~~~~~

Run configurations are **installed classes, not settings files**. Pick a stock
fit, turn knobs as plain attributes, build, run::

    from lisatools.globalfit.stock import erebor

    erebor.get_stock_options()             # [(name, description), ...]
    fit = erebor.gb_no_fg(nwalkers=4)      # or erebor.get_stock("gb_no_fg", ...)
    fit.gb.center_freq = 8e-3              # plain attribute access on the blocks
    fit.recipe.pop_move("rj_refit")        # named move stacks per recipe stage
    fit.remove_branch("galfor")            # compose whole objects in and out
    curr = fit.build()                     # the heavy stage, on command
    fit.run()

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

The erebor family
+++++++++++++++++++++++

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

.. automodule:: lisatools.globalfit.stock.erebor.stochastic
    :members:
    :show-inheritance:

Transforms, wrappers and runtime glue
++++++++++++++++++++++++++++++++++++++++

.. automodule:: lisatools.globalfit.stock.erebor.transforms
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.wrappers
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.source_runtime
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.injections
    :members:
    :show-inheritance:

The variants
+++++++++++++++++++++++

Every variant's data pipeline swaps with one knob (``general.data_mode``), and
each has a ``*_lite`` laptop-smoke twin.

.. automodule:: lisatools.globalfit.stock.erebor.variants.gb_no_fg
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.variants.noise
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.variants.full_year_combined
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.variants.all_sources
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.stock.erebor.variants.lite
    :members:
    :show-inheritance:
