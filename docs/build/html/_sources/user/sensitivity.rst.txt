Sensitivity Information
~~~~~~~~~~~~~~~~~~~~~~~~~~

The sensitivity code is heavily based on an original code by Stas Babak, Antoine Petiteau for the LDC team.

References for noise models:
  * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
  * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
  * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
  * 'sangria': Model used for LDC2A data generation.

.. autofunction:: lisatools.sensitivity.get_sensitivity

.. autofunction:: lisatools.sensitivity.get_stock_sensitivity_options

.. autofunction:: lisatools.sensitivity.get_stock_sensitivity_matrix_options

Sensitivity Matrix
---------------------
The sensitivity matrix is designed to house sensitivity information that will enter into the Likelihood function. 
It is a matrix because the 3 channels in LISA will be correlated. Therefore, in XYZ channels, this is a 3x3 matrix that includes correlated cross-terms between the channels. 
When working with AET channels (uncorrelated in idealized noise situations), then the sensitivity matrix will be an array of length 2 for AE or 3 for AET. 
The :func:`lisatools.diagnostic.inner_product` will adjust its computation based on the shape of the sensitivity matrix input. 
The user does not have to do anything special for this change to work. It happens under the hood.

.. autoclass:: lisatools.sensitivity.SensitivityMatrix
    :members:
    :show-inheritance:

Stock Sensitivity Matrices
******************************

.. autoclass:: lisatools.sensitivity.XYZ1SensitivityMatrix
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.AET1SensitivityMatrix
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
