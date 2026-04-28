# Coding Conventions

**Analysis Date:** 2026-04-28

## Naming Patterns

**Files:**
- Module files use `snake_case`: `analysiscontainer.py`, `datacontainer.py`, `diagnostic.py`
- C++/Cython files in `cutils/` use `PascalCase` for CUDA kernels (`Detector.cu`, `Detector.hpp`) and `snake_case` for Python bindings (`binding.cxx`, `binding.hpp`)
- Test files follow the pattern `test_<module>.py`: `test_sensitivity.py`, `test_detector.py`
- Global fit stock configurations live under `globalfit/stock/` (e.g., `erebor.py`)

**Classes:**
- All classes use `PascalCase`: `DataResidualArray`, `AnalysisContainer`, `SensitivityMatrix`, `DomainSettingsBase`
- Abstract base classes are suffixed with nothing special but always inherit from `ABC`: `Sensitivity(ABC)`, `Orbits(LISAToolsParallelModule, ABC)`
- Settings / configuration dataclasses end with `Settings` or `Info`: `GeneralSettings`, `GlobalFitSettings`, `RankInfo`, `EngineInfo`
- Matrix subclasses name their TDI channel type and generation inline: `AET1SensitivityMatrix`, `XYZ2SensitivityMatrix`, `AE1SensitivityMatrix`

**Functions:**
- Public methods and free functions use `snake_case`: `get_sensitivity`, `get_array_module`, `inner_product`, `get_normal_unit_vec`
- Private/internal helpers are prefixed with a single underscore: `_seconds_to_l3c_datetime`, `_infer_tdi_channels`, `_trim_and_shift_times`
- Factory/getter functions follow `get_<thing>` pattern: `get_sensitivity`, `get_array_module`, `get_pos`, `get_light_travel_times`

**Variables:**
- Array module variable universally named `xp` throughout the codebase (numpy or cupy depending on backend)
- Physical/numerical parameters use short scientific names: `Tobs`, `dt`, `df`, `f0`, `fdot`, `Sn`
- Constants in `utils/constants.py` are imported via wildcard `from .utils.constants import *` and use `UPPER_SNAKE_CASE`: `YRSID_SI`, `LINEAR_INTERP_TIMESTEP`
- Private instance variables use underscore prefix: `self._armlength`, `self._data_res_arr`, `self._cold_chains`

**Types and Type Hints:**
- Union types written with `|` syntax (Python 3.10+): `float | np.ndarray`, `str | None`, `HDFBackend | str`
- `Optional[X]` used where applicable but often replaced by `X | None` directly
- `from __future__ import annotations` is present at the top of every major module to enable forward references
- Common type aliases imported from `typing`: `Any`, `List`, `Optional`, `Tuple`, `Dict`, `Callable`

## Docstring Style

Docstrings follow the **NumPy/SciPy convention** throughout the package.

**Structure:**
```python
def get_Sn(
    cls,
    f: float | np.ndarray,
    model: Optional[LISAModel | str] = sangria,
    **kwargs: dict,
) -> float | np.ndarray:
    """Calculate the PSD

    Args:
        f: Frequency array.
        model: Noise model. Object of type :class:`LISAModel`.
            It can also be a string corresponding to one of the stock models.
            ...
        **kwargs: For interoperability.

    Returns:
        PSD values.

    """
```

- Top-level description on the first line, no blank line before it
- `Args:` section with indented parameter descriptions; multi-line descriptions are indented an extra level
- `Returns:` section for return value description
- `:class:`, `:func:`, `:meth:` Sphinx cross-references used inline in descriptions
- Math notation is included via reStructuredText `.. math::` blocks in longer docstrings (see `diagnostic.py` `inner_product`)
- Module-level docstrings use triple-quoted strings describing the module's purpose (e.g., `postprocessing.py` lines 1-9)

## Type Annotation Usage

- All public method signatures include type hints for parameters and return types
- `-> None` is written explicitly where applicable
- Properties annotate their return type in the `@property` signature: `def xp(self) -> object`, `def armlength(self) -> float`
- Setter type hints are included: `def armlength(self, armlength: float) -> None`
- Abstract methods are typed even when they raise `NotImplementedError`
- Type hints are NOT exhaustive in older code; newer modules (e.g., `postprocessing.py`) are more thoroughly annotated

## GPU/CPU Backend Pattern

The universal pattern for backend-agnostic code is:

```python
from lisatools.utils.utility import get_array_module

xp = get_array_module(data)   # Returns numpy or cupy
result = xp.sum(data)          # Works transparently on CPU or GPU
```

`get_array_module` is defined in `src/lisatools/utils/utility.py` (line 196):
```python
def get_array_module(arr: np.ndarray | cp.ndarray) -> object:
    if isinstance(arr, np.ndarray):
        return np
    elif isinstance(arr, cp.ndarray):
        return cp
    else:
        raise ValueError("arr must be a numpy or cupy array.")
```

**CuPy graceful fallback pattern** — used at the top of every module that supports GPU:
```python
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
except (ModuleNotFoundError, ImportError):
    import numpy as cp          # cp silently becomes np
    cp_gaussian_filter1d = np_gaussian_filter1d
```
This is used in `sensitivity.py`, `detector.py`, `domains.py`, `analysiscontainer.py`, `diagnostic.py`, `datacontainer.py`.

**Backend selection** is managed by `LISAToolsParallelModule` (in `src/lisatools/utils/parallelbase.py`), which wraps `gpubackendtools.ParallelModuleBase`. Classes that support GPU/CPU switching inherit from it:
- `Orbits(LISAToolsParallelModule, ABC)` in `detector.py`
- `DomainSettingsBase(LISAToolsParallelModule)` in `domains.py`
- `Sensitivity` delegates to `get_array_module` rather than inheriting `LISAToolsParallelModule`

The `force_backend` parameter is passed as a string (`"cpu"`, `"gpu"`, `"cuda12x"`) to constructors. In `LISAToolsParallelModule.__init__`, the string is wrapped as `("lisatools", force_backend)` before passing to `gpubackendtools`.

In tests, the `xp` variable is set at runtime:
```python
xp = cp if gpu_available else np
```

## Pydantic Model Usage

Pydantic is listed as a dependency in `pyproject.toml` (for "citations and references with advanced dataclasses") but is **not used directly** in the main source tree as of this analysis — no `BaseModel` subclasses were found. The dependency is present for future use or for citation/reference utilities not yet traced.

## Configuration Patterns

**Primary configuration mechanism is Python `dataclasses`**, not Pydantic. Configurations are structured as frozen or regular dataclasses:

- `@dataclasses.dataclass` for mutable config objects: `Settings`, `GeneralSettings`, `EngineInfo` in `engine.py`
- `@dataclasses.dataclass(frozen=True)` for immutable metadata: `ParameterInfo` in `postprocessing.py`
- `@dataclass` (bare import from `dataclasses`) in `detector.py` and `state.py`

**GlobalFit settings pattern** (`src/lisatools/globalfit/engine.py`):
- `Settings` is a base dataclass with fields for `Tobs`, `dt`, `transform`, `priors`, etc.
- `GeneralSettings(Settings)` adds fields for `file_store_dir`, `gpu_backend`, `nwalkers`, `ntemps`, etc.
- `Setup` wraps a `Settings` instance and reflects its fields to `self` using `dataclasses.fields()`
- `GeneralSetup(Setup, GeneralSettings)` is the concrete runtime class

**Physical constants** are accessed via wildcard import from `src/lisatools/utils/constants.py`:
```python
from .utils.constants import *   # provides YRSID_SI, etc.
```

**File registry/external data** is managed with PyYAML and `jsonschema` validation.

## Import Organization

**Module import order** (from top of files):
1. `from __future__ import annotations` (always first where used)
2. Standard library: `os`, `math`, `warnings`, `copy`, `abc`, `typing`, `dataclasses`, `logging`
3. Third-party scientific: `numpy`, `scipy`, `matplotlib`
4. Optional GPU: `cupy` in a `try/except (ModuleNotFoundError, ImportError)` block
5. LISA-specific third-party: `eryn`, `h5py`, `lisaconstants`, `gpubackendtools`, `cudakima`
6. Internal absolute imports: `from lisatools.utils.utility import get_array_module`
7. Internal relative imports: `from . import domains`, `from .detector import L1Orbits`

**Barrel file** (`src/lisatools/__init__.py`) exports the primary public API:
- `AnalysisContainer`, `DataResidualArray`, `SensitivityMatrix`, `get_sensitivity`
- `cutils`, `utils` subpackages
- `gpubackendtools` backend utilities (`get_backend`, `has_backend`)

**Star imports** are used cautiously for constants only:
- `from .utils.constants import *` — in almost every module
- `from .domains import *` — in `datacontainer.py`

## GlobalFit Subpackage Patterns

Location: `src/lisatools/globalfit/`

**Key patterns specific to this subpackage:**

- **MPI-aware**: `run.py` imports `from mpi4py import MPI` and operates rank-aware
- **Logging via `logging` module**: Uses `getLogger(__name__)` and `init_logger` helper from `loginfo.py`. In `postprocessing.py`: `logger = getLogger(__name__)` at module level. In `run.py`: `self.logger = init_logger(...)` per instance
- **HDF5 state persistence**: `GFHDFBackend` (wrapping `eryn.backends.HDFBackend`) is used for checkpointing MCMC state
- **Registry dicts for parameter metadata**: Module-level `Dict[str, ParameterInfo]` constants map short parameter names to L3C-compliant names, LaTeX labels, and units (e.g., `_GB_PARAM_INFO`, `_MBH_PARAM_INFO` in `postprocessing.py`)
- **Source type keys**: Source branches are identified by lowercase string keys: `"gb"` (galactic binary), `"mbh"` (massive black hole), `"emri"`, `"psd"`. These keys index `curr.source_info` dicts throughout the pipeline.
- **Recipe pattern**: `Recipe` and `recipe_steps.py` implement a step-based execution pipeline for the global fit engine, where each step is a callable or object with a defined interface
- **Stock configurations**: Predefined setups live in `globalfit/stock/erebor.py` and use `@dataclasses.dataclass` for `Settings` subclasses (lines 33, 256, 445, 639, 705 of `erebor.py`)

## Error Handling Conventions

- **Primary exception type**: `ValueError` for invalid arguments (most common; used throughout `detector.py`, `datacontainer.py`, `sensitivity.py`)
- **`RuntimeError`** for "must call setup method first" guard conditions: `detector.py` lines 1134, 1162, 1191 (`"Must call configure() before get_pos()"`)
- **`NotImplementedError`** for abstract methods and unimplemented transforms in base classes (`DomainBase.transform`, `DomainSettingsBase.get_slice`)
- **`AttributeError`** for lazy-init guards: `BackendConsumer.curr` raises `AttributeError` if accessed before `_curr` is set
- **`warnings.warn`** (standard library) for non-fatal issues like overlapping frequency ranges (`analysiscontainer.py` lines 866, 888, 936, 978), not logging
- **Bare `except:`** clause exists at `sensitivity.py` line 147 (followed by `breakpoint()`) — this is a debugging artifact that should be replaced
- **`assert`** used for input validation in setters: `assert isinstance(data_res_arr, DataResidualArray)` in `analysiscontainer.py` — not suitable for production validation in library code
- Error messages are formatted as f-strings with the offending value included for debugging

---

*Convention analysis: 2026-04-28*
