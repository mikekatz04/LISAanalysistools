"""Extreme mass-ratio inspiral (EMRI) waveform generators.

Production (``EMRI_LIKELIHOOD=direct``, through ``auto`` the library default on a WDM / XYZ
run): the direct-to-WDM lookup template
:class:`~lisatools.sources.emri.wdm_direct.EMRIDirectWDM` behind the batched adapter
:class:`~lisatools.sources.emri.direct_signal_gen.EMRIDirectWDMSignalGen`, with the
TDI-on-the-fly helper :class:`~lisatools.sources.emri.emritdionfly.EMRITDIonFly`. Standby: the
legacy FEW + ``ResponseWrapper`` path
(:func:`~lisatools.sources.emri.response.get_emri_response_wrapper`,
:class:`~lisatools.sources.emri.response.EMRIWaveWrap`), used for the cross-check, the fallback
and synthetic injections. Stand-alone: :class:`~lisatools.sources.emri.waveform.EMRITDIWaveform`.

Re-exports the domain-of-validity helpers (:mod:`~lisatools.sources.emri.domain`), the catalogue
glue :func:`~lisatools.sources.emri.waveform.emri_catalogue_to_waveform_basis`, and the stock FEW
kwargs constants. Importing it requires FEW (``few``).
"""

from .domain import (  # noqa: F401
    FEW_DOMAIN_ERROR_PATTERNS,
    emri_kerr_domain_mask,
    few_domain_guard,
)
from .waveform import EMRITDIWaveform, emri_catalogue_to_waveform_basis  # noqa: F401
from .emritdionfly import EMRITDIonFly
from .response import (  # noqa: F401
    EMRIWaveWrap,
    get_emri_response_wrapper,
    EMRI_INSPIRAL_KWARGS,
    EMRI_SUM_KWARGS,
    EMRI_MODE_SELECTOR_KWARGS,
)
