"""Custom ``eryn`` MCMC moves used by the LISA global fit.

A move IS a proposal: every stage of a :class:`~lisatools.globalfit.recipe.Recipe` runs its moves
inside one :class:`~lisatools.globalfit.moves.globalfitmove.GFCombineMove`, once per global-fit
iteration. This package re-exports the classes the recipe builders
(:mod:`lisatools.globalfit.recipe`, :mod:`lisatools.globalfit.stock`) construct:

* recipe layer (:mod:`~lisatools.globalfit.moves.globalfitmove`,
  :mod:`~lisatools.globalfit.moves.functionmove`): ``Move``, ``MoveBuildContext``,
  ``FunctionMove``, the ``GlobalFitMove`` mixin and the per-stage ``GFCombineMove``;
* GB / VGB (:mod:`~lisatools.globalfit.moves.gbspecialstretch`): ``GBSpecialStretchMove``
  (``in_model*``), ``VGBSpecialStretchMove`` (``vgb_pe``), ``GBSpecialRJPriorMove``
  (``rj_prior_*``, ``rj_warm_*``), ``GBSpecialRJFStatGridMove`` (``rj_fstat_*``),
  ``GBSpecialRJSerialSearchMCMC`` (``rj_fstat_mcmc*``), ``GBSpecialRJRefitMove`` (``rj_refit*``);
* noise (:mod:`~lisatools.globalfit.moves.psdmove`): ``PSDMove`` (``psd_pe`` / ``galfor_pe`` /
  ``sgwb_pe`` and their ``*_search`` twins, one instance per noise branch);
* single sources (:mod:`~lisatools.globalfit.moves.addremovemove` and its subclasses):
  ``ResidualAddOneRemoveOneMove``, ``MBHBatchedLikeMove`` (``mbh_pe``), ``EMRIDirectLikeMove``
  (``emri_pe``), ``SOBBHChunkedLikeMove`` (``sobbh_pe``).

Legacy names kept in the import surface but built by no recipe builder: ``MBHSpecialMove`` and
``TDMBHSpecialMove`` (:mod:`~lisatools.globalfit.moves.mbhspecialmove`; the MBH branch runs
``MBHBatchedLikeMove`` or the base ``ResidualAddOneRemoveOneMove``) and ``MultiGPUPSDMove``
(a deprecated alias of ``PSDMove(acs, priors, dcga=...)``).
"""

from .addremovemove import ResidualAddOneRemoveOneMove
from .functionmove import FunctionMove
from .gbspecialstretch import (
    GBSpecialRJPriorMove,
    GBSpecialRJFStatGridMove,
    GBSpecialRJRefitMove,
    GBSpecialRJSerialSearchMCMC,
    GBSpecialStretchMove,
    VGBSpecialStretchMove,
)
from .globalfitmove import GFCombineMove, GlobalFitMove, Move, MoveBuildContext
from .mbhspecialmove import MBHSpecialMove, TDMBHSpecialMove
from .psdmove import PSDMove, MultiGPUPSDMove
from .sobbhspecialmove import SOBBHChunkedLikeMove
from .mbhbatchedmove import MBHBatchedLikeMove
from .emridirectmove import EMRIDirectLikeMove
