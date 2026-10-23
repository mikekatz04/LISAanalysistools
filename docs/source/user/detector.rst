Detector Information
~~~~~~~~~~~~~~~~~~~~~~~~~~

The detector information contains the sensitivity and orbits for a given LISA configuration.

LISA Models
---------------------

.. autoclass:: lisatools.detector.LISAModel
    :members:
    :show-inheritance:

.. autoclass:: lisatools.detector.LISAModelSettings
    :members:
    :show-inheritance:

.. autoclass:: lisatools.detector.ExtendedLISAModel
    :members:
    :show-inheritance:

.. autoclass:: lisatools.detector.ExtendedLISAModelSettings
    :members:
    :show-inheritance:

LISA Models Stock Options
****************************

.. autofunction:: lisatools.detector.get_available_default_lisa_models

.. autofunction:: lisatools.detector.get_default_lisa_model_from_str

.. autofunction:: lisatools.detector.check_lisa_model

    

.. autoclass:: lisatools.detector.CurrentNoises
    :members:
    :show-inheritance:

The stock :class:`~lisatools.detector.LISAModel` instances are module attributes:
``scirdv1``, ``proposal``, ``mrdv1`` and ``sangria`` (and the
:class:`~lisatools.detector.ExtendedLISAModel` ``sangria_v2``).

Orbits
-----------

.. autoclass:: lisatools.detector.Orbits
    :members:
    :show-inheritance:

Mojito L1 Orbits (global fit)
*******************************

The global-fit production runs load their orbits from the Mojito ``INSTRUMENT/L1`` NOISE file
with :class:`~lisatools.detector.L1Orbits` and ``frame="icrs"`` (the run frame).

.. autoclass:: lisatools.detector.L1Orbits
    :members:
    :show-inheritance:

.. autofunction:: lisatools.detector.icrs_to_ecliptic

Stock Orbit Options
*************************

The packaged orbit files hold SSB-ecliptic positions, while the inherited default label is
``frame="icrs"`` (see the Warning in :class:`~lisatools.detector.Orbits`); pass
``frame="ecliptic"`` to make the label match the data.

.. autoclass:: lisatools.detector.EqualArmlengthOrbits
    :members:
    :show-inheritance:

.. autoclass:: lisatools.detector.DefaultOrbits
    :members:
    :show-inheritance:

.. autoclass:: lisatools.detector.ESAOrbits
    :members:
    :show-inheritance:

