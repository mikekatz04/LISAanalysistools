Sensitivity Information
~~~~~~~~~~~~~~~~~~~~~~~~~~

The analytic sensitivity code is heavily based on an original code by Stas Babak, Antoine Petiteau for the LDC team.

The module has four layers: single-channel PSD classes (:class:`~lisatools.sensitivity.Sensitivity`
subclasses), sensitivity (covariance) matrices, the composite noise model of the global fit, and
the native XYZ backend. The 9-month and 12-month global-fit runs use the composite model
(:class:`~lisatools.sensitivity.CompositeSensitivityBackend`) on XYZ TDI 2.0 channels with the
unequal-arm instrument noise (:class:`~lisatools.sensitivity.UnequalArmInstrumentNoise`,
``UNEQUAL_ARM=1``) on the WDM domain (``WDM_PSD_METHOD=layer_calibrated``). The analytic
equal-arm curves and matrices below are the quick-study building blocks; the global-fit noise
model is the unequal-arm composite one.

References for noise models:
  * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
  * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
  * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
  * 'sangria': Model used for LDC2A data generation.

.. autofunction:: lisatools.sensitivity.get_sensitivity

.. autofunction:: lisatools.sensitivity.get_stock_sensitivity_options

.. autofunction:: lisatools.sensitivity.get_stock_sensitivity_matrix_options

.. autofunction:: lisatools.sensitivity.get_stock_sensitivity_from_str

.. autofunction:: lisatools.sensitivity.check_sensitivity

Sensitivity Matrix
---------------------
The sensitivity matrix is designed to house sensitivity information that will enter into the Likelihood function. 
It is a matrix because the 3 channels in LISA will be correlated. Therefore, in XYZ channels, this is a 3x3 matrix that includes correlated cross-terms between the channels. 
When working with AET channels (uncorrelated in idealized noise situations), then the sensitivity matrix will be an array of length 2 for AE or 3 for AET. 
The :func:`lisatools.diagnostic.inner_product` will adjust its computation based on the shape of the sensitivity matrix input. 
The user does not have to do anything special for this change to work. It happens under the hood.

.. autoclass:: lisatools.sensitivity.SensitivityMatrixBase
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.SensitivityMatrix
    :members:
    :show-inheritance:

Stock Sensitivity Matrices
******************************

.. autoclass:: lisatools.sensitivity.XYZ2SensitivityMatrix
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.XYZ1SensitivityMatrix
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.AET2SensitivityMatrix
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.AET1SensitivityMatrix
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.AE2SensitivityMatrix
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.AE1SensitivityMatrix
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.LISASensSensitivityMatrix
    :members:
    :show-inheritance:


Sensitivity Base Class
--------------------------

.. autoclass:: lisatools.sensitivity.Sensitivity
    :members:
    :show-inheritance:

Stock Sensitivity Models
*************************

.. autoclass:: lisatools.sensitivity.LISASens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.CornishLISASens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.FlatPSDFunction
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.X1TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.Y1TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.Z1TDISens
    :members:
    :show-inheritance:  

.. autoclass:: lisatools.sensitivity.XY1TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.YZ1TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.ZX1TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.A1TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.E1TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.T1TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.X2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.Y2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.Z2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.XY2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.YZ2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.ZX2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.A2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.E2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.T2TDISens
    :members:
    :show-inheritance:

Unequal-arm TDI 2.0 elements
*****************************

The XYZ TDI 2.0 covariance elements for six independent link light travel times (the
production instrument model, through :class:`~lisatools.sensitivity.UnequalArmInstrumentNoise`).

.. autoclass:: lisatools.sensitivity.UnequalArmXX2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.UnequalArmYY2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.UnequalArmZZ2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.UnequalArmXY2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.UnequalArmXZ2TDISens
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.UnequalArmYZ2TDISens
    :members:
    :show-inheritance:

Composite Noise Model (global fit)
------------------------------------

A :class:`~lisatools.sensitivity.CompositeSensitivityMatrix` sums noise components: the
instrument (:class:`~lisatools.sensitivity.UnequalArmInstrumentNoise` with its
:class:`~lisatools.sensitivity.LinkDelayTable` in production), the galactic foreground
(:class:`~lisatools.sensitivity.GalacticForeground`, optionally time-modulated by
:class:`~lisatools.sensitivity.GalForTimeModulation`), an optional
:class:`~lisatools.sensitivity.SGWB` and the tabulated
:class:`~lisatools.sensitivity.MojitoNoiseEstimates`.
:class:`~lisatools.sensitivity.CompositeSensitivityBackend` builds one such matrix per walker
from the sampled ``psd`` and ``galfor`` parameters.

.. autoclass:: lisatools.sensitivity.SensitivityBackendBase
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.CompositeSensitivityBackend
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.CompositeSensitivityMatrix
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.MojitoNoiseSensitivityMatrix
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.NoiseComponent
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.SeparableComponent
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.InstrumentNoise
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.UnequalArmInstrumentNoise
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.LinkDelayTable
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.GalacticForeground
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.GalForTimeModulation
    :members:
    :show-inheritance:

.. autodata:: lisatools.sensitivity.GALFOR_MODULATION_TABLE

.. autofunction:: lisatools.sensitivity.packaged_galfor_modulation_path

.. autofunction:: lisatools.sensitivity.modulation_from_elements

.. autoclass:: lisatools.sensitivity.SGWB
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.MojitoNoiseEstimates
    :members:
    :show-inheritance:

.. autofunction:: lisatools.sensitivity.estimate_noise_params_from_file

.. autofunction:: lisatools.sensitivity.tdi_generation_from_channel

Native XYZ Backend
----------------------

An alternative ``sensitivity_backend`` computed by the C++/CUDA ``SensitivityMatrixWrap``
kernel; the production runs use :class:`~lisatools.sensitivity.CompositeSensitivityBackend`.

.. autoclass:: lisatools.sensitivity.XYZSensitivityBackend
    :members:
    :show-inheritance:

Synthetic Noise Generation
-----------------------------

Time-domain noise and foreground realizations, used by the stock fits' synthetic data streams
(:mod:`lisatools.globalfit.preprocessing`, :mod:`lisatools.globalfit.stock.erebor.injections`).
The instrument-noise draw uses the equal-arm :class:`~lisatools.sensitivity.InstrumentNoise`
covariance.

.. autofunction:: lisatools.sensitivity.generate_correlated_instrument_noise_td

.. autofunction:: lisatools.sensitivity.generate_foreground_td

.. autofunction:: lisatools.sensitivity.annual_amplitude_envelope

.. autofunction:: lisatools.sensitivity.annual_modulation_matrix
