Sampling Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~

MCMC pieces built on ``eryn``: priors, the GB F-statistic birth machinery and proposal bases,
batched Gaussian-mixture fits, per-leaf ladders, stopping criteria and custom inner moves.
Nothing is re-exported by ``lisatools.sampling``; import the modules directly. The legacy
modules ``sampling.likelihood``, ``sampling.utility`` and ``sampling.warmstart_proposal`` are
listed in :doc:`legacy`.

Priors
---------------

.. automodule:: lisatools.sampling.prior
    :members:
    :show-inheritance:
    :exclude-members: AmplitudeFrequencySNRPrior, AmplitudeFromSNR, GBPriorWrap, map_input, log_det_J

.. automodule:: lisatools.sampling.f0_mchirp_prior
    :members:
    :show-inheritance:

GB F-statistic birth proposals
-------------------------------

The grid fit (:mod:`lisatools.sampling.fstat_gridfit`) and the proposal distributions built
from it (:mod:`lisatools.sampling.fstat_proposal`), used by the GB ``rj_fstat_*`` moves.

.. automodule:: lisatools.sampling.fstat_gridfit
    :members:
    :show-inheritance:
    :ignore-module-all:
    :exclude-members: union_stacked_npz, union_cell_keys

.. automodule:: lisatools.sampling.fstat_proposal
    :members:
    :show-inheritance:
    :ignore-module-all:
    :exclude-members: pe_extrinsic_sigma, pe_extrinsic_logpdf, pe_extrinsic_rvs, rvs_per_box, ColumnPermutedProposal, fit_gmm_to_stacked, build_peak_gmm

GB proposal bases
------------------

The observable basis of the GB in-model proposals (``GB_INMODEL_PROPOSAL=observable``) and the
ridge-Gibbs fiber move for the ``(Mc, ratio, distance)`` degeneracy (``GB_RIDGE_GIBBS=1``).

.. automodule:: lisatools.sampling.gb_observable_basis
    :members:
    :show-inheritance:
    :exclude-members: f_mid_from_f0, f0_from_f_mid, fdot_shear_hz, sky_doppler_alias_jump, sky_doppler_alias_jump_batch

.. automodule:: lisatools.sampling.ridge_fiber
    :members:
    :show-inheritance:

Gaussian mixtures
------------------

.. automodule:: lisatools.sampling.gmm
    :members:
    :show-inheritance:
    :exclude-members: aic, predict_proba, general_way_logpdf

Ladders and stopping
---------------------

.. automodule:: lisatools.sampling.snrladder
    :members:
    :show-inheritance:

.. automodule:: lisatools.sampling.stopping
    :members:
    :show-inheritance:
    :exclude-members: NLeavesSearchStopping, SearchConvergeStopping2, EvidenceStopping, MPICommunicateStopping

Inner moves (``lisatools.sampling.moves``)
-------------------------------------------

The inner moves of the MBH / EMRI / SOBBH add/remove moves: the eigen-axis move with a
wider-step share, the cross-walker difference jump, and the MBH sky-mode hops with the
LISA-frame proposal map (``MBH_PROPOSAL_FRAME=lisa`` in the production launchers).

.. automodule:: lisatools.sampling.moves.eigenjumpmix
    :members:
    :show-inheritance:

.. automodule:: lisatools.sampling.moves.walkerdiffjump
    :members:
    :show-inheritance:

.. automodule:: lisatools.sampling.moves.skymodehop
    :members:
    :show-inheritance:
    :ignore-module-all:
