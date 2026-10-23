"""Definition of LISA Analysis Tools package common exceptions.

All package exceptions derive from :class:`LISAToolsException`. The two raised and caught
on the production path are :class:`WaveformDomainError` (parameters outside a waveform
model's validity domain, scored at the log-likelihood floor) and :class:`BatchNotLaunchable`
(a batched launch refuses a batch, the caller falls back to per-row evaluation). The
remaining classes mirror ``few.utils.exceptions`` and are not raised anywhere in
``lisatools``.
"""

try:
    from exceptiongroup import ExceptionGroup
except (ImportError, ModuleNotFoundError):
    # Python >= 3.11: the builtin ExceptionGroup.
    ExceptionGroup = ExceptionGroup


class LISAToolsException(Exception):
    """Base class for LISA Analysis Tools package exceptions."""

    pass


# TODO check: keep/remove/document -- never raised or caught in lisatools; copied from few.utils.exceptions
class CudaException(LISAToolsException):
    """Base class for CUDA-related exceptions."""

    pass


# TODO check: keep/remove/document -- never raised or caught in lisatools; copied from few.utils.exceptions
class CuPyException(LISAToolsException):
    """Base class for CuPy-related exceptions."""

    pass


# TODO check: keep/remove/document -- never raised in lisatools (jax/backend.py raises gpubackendtools' MissingDependency)
class MissingDependency(LISAToolsException):
    """Exception raised when a required dependency is missing."""

    pass


class BatchNotLaunchable(LISAToolsException):
    """Raised when a batch of sources cannot be evaluated as ONE launch.

    A REFUSAL IS NOT AN ERROR. It means the geometry of this particular batch
    is not expressible in one call -- sub-sample alignments that differ across
    sources, merger times spread wider than a shared evaluation window --
    and the caller should evaluate the rows separately instead. That is
    designed control flow, and it is expected wherever a walker cloud is wide
    (burn-in especially).

    It exists so sampling machinery can catch refusals NARROWLY. Catching
    bare ``Exception`` around a batched launch conflates "this batch is an
    awkward shape" with "this code is wrong", and the second one then hides
    inside the first: a mis-wired call site raising ``TypeError`` looks
    exactly like a refusal, falls back to the serial loop, returns correct
    numbers, and the batched path is never exercised again.

    Mirrors :class:`WaveformDomainError`'s purpose for the same reason.
    """


# TODO check: keep/remove/document -- never raised or caught in lisatools; copied from few.utils.exceptions
class InvalidInputFile(LISAToolsException):
    """Exception raised when the content of an input file does not match expectations."""


# TODO check: keep/remove/document -- never raised; lisatools has no configuration system (copied from few.utils.exceptions)
class ConfigurationError(LISAToolsException):
    """Exception raised when configuration setup fails."""


# TODO check: keep/remove/document -- never raised; lisatools has no configuration system (copied from few.utils.exceptions)
class ConfigurationMissing(ConfigurationError):
    """Exception raised when an expected configuration entry is missing."""


# TODO check: keep/remove/document -- never raised; lisatools has no configuration system (copied from few.utils.exceptions)
class ConfigurationValidationError(ConfigurationError):
    """Exception raised when a configuration entry is invalid."""


# TODO check: keep/remove/document -- never raised; lisatools has no FileManager (copied from few.utils.exceptions)
class FileManagerException(LISAToolsException):
    """Exception raised by the FileManager."""


# TODO check: keep/remove/document -- never raised; lisatools has no FileManager (copied from few.utils.exceptions)
class FileNotInRegistry(FileManagerException):
    """Exception raised when a requested file is not in file registry."""


# TODO check: keep/remove/document -- never raised; lisatools has no FileManager (copied from few.utils.exceptions)
class FileNotFoundLocally(FileManagerException):
    """Exception raised when file not found locally but expected to be."""


# TODO check: keep/remove/document -- never raised; lisatools has no FileManager (copied from few.utils.exceptions)
class FileInvalidChecksum(FileManagerException):
    """Exception raised when file has invalid checksum."""


# TODO check: keep/remove/document -- never raised; lisatools has no FileManager (copied from few.utils.exceptions)
class FileDownloadException(FileManagerException):
    """Exception raised if file download failed."""


# TODO check: keep/remove/document -- never raised; lisatools has no FileManager (copied from few.utils.exceptions)
class FileDownloadNotFound(FileDownloadException):
    """Exception raised if file is not found at expected URL."""


# TODO check: keep/remove/document -- never raised; lisatools has no FileManager (copied from few.utils.exceptions)
class FileDownloadConnectionError(FileDownloadException):
    """Exception raised in case of connection error during download."""


# TODO check: keep/remove/document -- never raised; lisatools has no FileManager (copied from few.utils.exceptions)
class FileDownloadIntegrityError(FileDownloadException):
    """Exception raised in case of integrity error after download."""


# TODO check: keep/remove/document -- never raised; lisatools has no FileManager (copied from few.utils.exceptions)
class FileManagerDisabledAccess(FileManagerException):
    """Exception raised when trying to access a file whose tags are disabled"""

    disabled_tag: str
    file_name: str

    def __init__(self, /, *args, disabled_tag: str, file_name: str, **kwargs):
        self.disabled_tag = disabled_tag
        self.file_name = file_name
        super().__init__(*args, **kwargs)


### Waveform-related exceptions
class WaveformDomainError(LISAToolsException):
    """Exception raised when source parameters fall outside a waveform model's domain of validity.

    Raised (in place of the model's bare ``ValueError``/``AssertionError``) so
    sampling machinery can catch it narrowly and score the proposal at the
    log-likelihood floor instead of crashing the run.
    """


### Trajectory-related exceptions
# TODO check: keep/remove/document -- never raised; sources/emri/domain.py catches few.utils.exceptions.TrajectoryOffGridException instead
class TrajectoryOffGridException(Exception):
    """Exception raised when a trajectory goes off-grid (except for the lower boundary in p)."""
