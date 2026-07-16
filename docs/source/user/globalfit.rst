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

Settings and Setup
+++++++++++++++++++++++

The per-branch knob block (``Settings``) and its built twin (``Setup``). Every
source class extends this pair.

.. automodule:: lisatools.globalfit.engine
    :members:
    :show-inheritance:

Running a fit
+++++++++++++++++++++++

``GlobalFitSetup`` is the built configuration + live state (the name
``CurrentInfoGlobalFit`` remains as an alias); ``GlobalFit`` is the runner that
holds one and drives the sampler.

.. automodule:: lisatools.globalfit.run
    :members:
    :show-inheritance:

Recipes — stages and moves
+++++++++++++++++++++++++++++

The recipe is a plan, not just a move stack: it is handed to the sampler as its
stopping function and consulted every iteration, so a completed stage can
reconfigure the *live* sampler for the next one (the in-run search → PE
hand-off).

.. automodule:: lisatools.globalfit.recipe
    :members:
    :show-inheritance:

State
+++++++++++++++++++++++

The sampler state, extended per branch to carry each source class's own
bookkeeping (GB a per-*band* temperature ladder, the rest a per-*leaf* one).

.. automodule:: lisatools.globalfit.state
    :members:
    :show-inheritance:

HDF backend
+++++++++++++++++++++++

Persistence for the run: chains, leaf occupation, and the recipe's progress
(so a resume picks up mid-recipe).

.. automodule:: lisatools.globalfit.hdfbackend
    :members:
    :show-inheritance:

Data in — preprocessing
+++++++++++++++++++++++++++

The data layer. ``general.data_mode`` swaps the whole pipeline
(``"mojito"`` / ``"synthetic"`` / ``"sangria"``); downstream, nothing knows
which one ran.

.. automodule:: lisatools.globalfit.preprocessing
    :members:
    :show-inheritance:

Results out — postprocessing
+++++++++++++++++++++++++++++++

.. automodule:: lisatools.globalfit.postprocessing
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.gathergalaxy
    :members:
    :show-inheritance:

Diagnostics and plotting
+++++++++++++++++++++++++++

.. automodule:: lisatools.globalfit.diagnosticplot
    :members:
    :show-inheritance:

.. automodule:: lisatools.globalfit.plot
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
