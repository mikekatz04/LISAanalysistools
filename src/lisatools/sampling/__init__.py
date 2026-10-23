"""Sampling utilities for LISA analyses built on top of ``eryn``.

Nothing is re-exported; import the modules directly: :mod:`~lisatools.sampling.prior`
(priors), :mod:`~lisatools.sampling.gmm` (batched GMM fits),
:mod:`~lisatools.sampling.f0_mchirp_prior` (the GB ``(f0, Mc)`` prior),
:mod:`~lisatools.sampling.snrladder` (SNR-keyed ladders),
:mod:`~lisatools.sampling.stopping` (stopping criteria), the GB F-statistic
proposal / grid-fit / observable-basis / ridge-fiber modules, and the custom
moves in :mod:`lisatools.sampling.moves`. :mod:`~lisatools.sampling.likelihood`,
:mod:`~lisatools.sampling.utility` and :mod:`~lisatools.sampling.warmstart_proposal`
are legacy / back-compat.
"""
