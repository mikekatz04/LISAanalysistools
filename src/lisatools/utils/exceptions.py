"""Definition of LISA Analysis Tools package common exceptions"""

try:
    from exceptiongroup import ExceptionGroup
except (ImportError, ModuleNotFoundError):
    ExceptionGroup = ExceptionGroup


class LISAToolsException(Exception):
    """Base class for LISA Analysis Tools package exceptions."""

    pass


class CudaException(LISAToolsException):
    """Base class for CUDA-related exceptions."""

    pass


class CuPyException(LISAToolsException):
    """Base class for CuPy-related exceptions."""

    pass


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


class InvalidInputFile(LISAToolsException):
    """Exception raised when the content of an input file does not match expectations."""


class ConfigurationError(LISAToolsException):
    """Exception raised when configuration setup fails."""


class ConfigurationMissing(ConfigurationError):
    """Exception raised when an expected configuration entry is missing."""


class ConfigurationValidationError(ConfigurationError):
    """Exception raised when a configuration entry is invalid."""


class FileManagerException(LISAToolsException):
    """Exception raised by the FileManager."""


class FileNotInRegistry(FileManagerException):
    """Exception raised when a requested file is not in file registry."""


class FileNotFoundLocally(FileManagerException):
    """Exception raised when file not found locally but expected to be."""


class FileInvalidChecksum(FileManagerException):
    """Exception raised when file has invalid checksum."""


class FileDownloadException(FileManagerException):
    """Exception raised if file download failed."""


class FileDownloadNotFound(FileDownloadException):
    """Exception raised if file is not found at expected URL."""


class FileDownloadConnectionError(FileDownloadException):
    """Exception raised in case of connection error during download."""


class FileDownloadIntegrityError(FileDownloadException):
    """Exception raised in case of integrity error after download."""


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
class TrajectoryOffGridException(Exception):
    """Exception raised when a trajectory goes off-grid (except for the lower boundary in p)."""
