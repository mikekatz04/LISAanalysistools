Stochastic Signals
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: lisatools.stochastic.get_stock_gb_stochastic_options

.. autofunction:: lisatools.stochastic.get_default_stochastic_from_str

.. autofunction:: lisatools.stochastic.check_stochastic

.. autofunction:: lisatools.stochastic.get_stock_sgwb_options


Stochastic Base Class
--------------------------

.. autoclass:: lisatools.stochastic.StochasticContribution
    :members:
    :show-inheritance:

Combinatorial Stochastic Container
--------------------------------------

.. autoclass:: lisatools.stochastic.StochasticContributionContainer
    :members:
    :show-inheritance:

Stock Stochastic Models
--------------------------

.. autoclass:: lisatools.stochastic.HyperbolicTangentGalacticForeground
    :members:
    :show-inheritance:

.. autoclass:: lisatools.stochastic.FittedHyperbolicTangentGalacticForeground
    :members:
    :show-inheritance:

Stock SGWB Models
--------------------------

Stochastic-background spectra for the optional ``sgwb`` branch of the stock fits
(``FIT_SGWB=1``, off in the 9-month and 12-month runs; the ``noise_sgwb`` variant fits one).

.. autoclass:: lisatools.stochastic.PowerLawSGWB
    :members:
    :show-inheritance:

.. autoclass:: lisatools.stochastic.LogNormalSGWB
    :members:
    :show-inheritance:

.. autoclass:: lisatools.stochastic.PhaseTransitionSGWB
    :members:
    :show-inheritance: