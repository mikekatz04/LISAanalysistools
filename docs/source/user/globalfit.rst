The Global Fit — engine and run layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``lisatools.globalfit`` builds and runs the LISA global fit. A run is a
configured :class:`~lisatools.globalfit.stock.base.StockGlobalFit` (see
:doc:`globalfitstock`), turned into a built
:class:`~lisatools.globalfit.run.GlobalFitSetup` by ``.build()`` and handed to
:class:`~lisatools.globalfit.run.GlobalFit` to sample.

Three peer configuration blocks describe a fit — ``general`` (run-wide), one
block per source class, and the ``recipe`` — and only the build-time resolution
orders them: general first, then the branches (which inherit unset
``Tobs``/``dt``/domain), then the recipe.

The 9-month and 12-month production runs build the ``all_sources`` variant
(:mod:`lisatools.globalfit.stock.erebor.variants.all_sources`): the driver
``scripts/fstat_proposal/run_combined_staged.py`` clones
``erebor.all_sources(nwalkers=4)``, replaces its recipe with the staged search + PE
recipe, then calls ``fit.build()`` and ``fit.run(comm)``. The moves are on
:doc:`globalfitmoves`; the rank layout, checkpoints, telemetry and priors on
:doc:`globalfit_internals`; the warm starts on :doc:`globalfitwarmstart`; the
monitor page on :doc:`globalfitmonitor`.

Settings and Setup
+++++++++++++++++++++++

The per-branch knob block (``Settings``) and its built twin (``Setup``), the
run-wide ``GeneralSettings`` / ``GeneralSetup``, and the eryn engine
(``GlobalFitEngine``). Every source class extends the ``Settings`` / ``Setup``
pair.

.. automodule:: lisatools.globalfit.engine
    :members:
    :show-inheritance:
    :ignore-module-all:

Running a fit
+++++++++++++++++++++++

``GlobalFitSetup`` is the built configuration + live state (the name
``CurrentInfoGlobalFit`` remains as an alias); ``GlobalFit`` is the runner that
holds one and drives the sampler.

.. automodule:: lisatools.globalfit.run
    :members:
    :show-inheritance:
    :exclude-members: gpu_assignments

Recipes — stages and moves
+++++++++++++++++++++++++++++

The recipe is a plan, not just a move stack: it is handed to the sampler as its
stopping function and consulted every iteration, so a completed stage can
reconfigure the *live* sampler for the next one (the in-run search → PE
hand-off).

.. automodule:: lisatools.globalfit.recipe
    :members:
    :show-inheritance:
    :exclude-members: subtract_initial_signal, apply_shutoff_floor_to_state, build_psd_moves, GBWaveformDict, GBMoveBuilder, VGBMoveBuilder, PSDMoveBuilder

Search legs
+++++++++++++++++++++++

One stored row per leg of a GB search cycle (``GB_SEARCH_LEGS=1`` in the production
launchers).

.. automodule:: lisatools.globalfit.legs
    :members:
    :show-inheritance:
    :exclude-members: end_early_at

State
+++++++++++++++++++++++

The sampler state, extended per branch to carry each source class's own
bookkeeping (GB a per-*band* temperature ladder, the rest a per-*leaf* one).

.. automodule:: lisatools.globalfit.state
    :members:
    :show-inheritance:
    :exclude-members: cap_divisor_from_edges, reseed_cold_into_hottest, update_band_information, AllGFBranchInfo, GFBranchInfo

HDF backend
+++++++++++++++++++++++

Persistence for the run: chains, leaf occupation, and the recipe's progress
(so a resume picks up mid-recipe).

.. automodule:: lisatools.globalfit.hdfbackend
    :members:
    :show-inheritance:

Data in — preprocessing
+++++++++++++++++++++++++++

The data layer. ``general.data_mode`` (env ``DATA_MODE``) swaps the whole
pipeline: ``"mojito"`` (the default and the production setting: the mojito L1
bricks), ``"synthetic"`` (every stream built in-process) and, on
``all_sources`` only, the legacy ``"sangria"``. Downstream, nothing knows which
one ran.

.. automodule:: lisatools.globalfit.preprocessing
    :members:
    :show-inheritance:
    :exclude-members: dump_catalogue, dump_individual_timeseries

Results out — postprocessing
+++++++++++++++++++++++++++++++

End-of-run products (submission writer, plotter, detection criteria). The
production runs do not reach them: ``general.submission_parent_folder`` is
unset and the multi-rank run skips the submission. ``SubmissionWriter`` does not
yet handle the ``vgb``, ``sobbh`` and ``emri`` branches of ``all_sources`` (see
its docstring).

.. automodule:: lisatools.globalfit.postprocessing
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.gathergalaxy
    :members:
    :show-inheritance:
    :exclude-members: GBGrouping, gather_gb_samples_cat, para_func

Diagnostics and plotting
+++++++++++++++++++++++++++

Plot producers for a run's chains; the running fit itself plots through eryn's
``PlotContainer``. The legacy ``lisatools.globalfit.plot`` module is listed in
:doc:`legacy`.

.. automodule:: lisatools.globalfit.diagnosticplot
    :members:
    :show-inheritance:

Run bookkeeping
+++++++++++++++++++++++

.. automodule:: lisatools.globalfit.loginfo
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.utils
    :members:
    :show-inheritance:
    :exclude-members: NewSensitivityMatrix, new_sens_mat, AllSetupInfoTransfer, SetupInfoTransfer
