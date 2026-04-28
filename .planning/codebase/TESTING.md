# Testing Patterns

**Analysis Date:** 2026-04-28

## Test Framework

**Runner:**
- `pytest` (primary, used in `test_fresnel.py` via `@pytest.mark.parametrize`)
- `unittest` (used in `test_detector.py`, `test_orbits.py`, `test_sensitivity.py`, `test_sources_utils.py`, `test_get_amp_phase_shifting.py`)
- No `pytest.ini` or `conftest.py` at project root — pytest discovers tests via standard conventions
- Config: `pyproject.toml` (`[tool.coverage]` section)

**Assertion Library:**
- `unittest.TestCase` assertion methods (`assertFalse`, `assertAlmostEqual`, `assertEqual`, `assertIsInstance`)
- `numpy.testing` (`np.testing.assert_allclose`, `np.testing.assert_array_equal`) for numerical assertions

**Run Commands:**
```bash
uv run pytest tests/                    # Run all tests
uv run pytest tests/test_fresnel.py     # Run specific file
uv run coverage run -m pytest tests/    # With coverage
```

## Test File Organization

**Location:**
- All tests in `tests/` directory at project root (flat, not co-located with source)

**Naming:**
- Pattern: `test_<module_or_feature>.py`
- Examples: `test_sensitivity.py`, `test_detector.py`, `test_orbits.py`, `test_fresnel.py`, `test_sources_utils.py`, `test_get_amp_phase_shifting.py`

**Structure:**
```
tests/
├── __init__.py
├── test_detector.py
├── test_orbits.py
├── test_sensitivity.py
├── test_fresnel.py
├── test_get_amp_phase_shifting.py
├── test_sources_utils.py
└── fresnel_plots/         # Output directory for diagnostic plots
```

## Test Structure

**`unittest`-style suite organization:**
```python
import unittest
import numpy as np

class SensitivityTest(unittest.TestCase):
    def test_get_sen(self):
        xp = cp if gpu_available else np
        frqs = xp.logspace(-5., 0., 1000)
        Sn = get_sensitivity(frqs, sens_fn="X1TDISens", model=lisa.sangria)
        self.assertFalse(xp.any(xp.isnan(Sn)))

    def _test_sens_mat(self, sens_mat_class, model):
        # Helper method (prefixed with _ to avoid auto-discovery)
        ...

    def test_sensitivity_matrix_AET1(self):
        self._test_sens_mat(AET1SensitivityMatrix, lisa.sangria)
```

**`pytest`-style suite organization (used for parametrized tests):**
```python
@pytest.mark.parametrize("force_backend", backends)
class TestFresnelVsScipy:
    def test_single_binary(self, force_backend):
        ...
```

**Patterns:**
- No `setUp`/`tearDown` methods found — tests are stateless and self-contained
- Helper methods prefixed with `_` to prevent auto-discovery by test runners
- Numerical tolerance constants defined at module level (`rtol=2e-2`, `atol=3e-2`)

## Mocking

**Framework:** None detected — no `unittest.mock` or `pytest-mock` usage in the test suite.

**Patterns:**
- Instead of mocking, tests use real objects with minimal configurations
- GPU availability is detected at import time and tests adapt accordingly:

```python
try:
    import cupy as cp
    gpu_available = True
except (ImportError, ModuleNotFoundError):
    import numpy as xp
    gpu_available = False

# In test body:
xp = cp if gpu_available else np
```

**What NOT to Mock:**
- Do not mock `numpy`/`cupy` array operations — tests validate numerical correctness
- Do not mock `lisatools` module internals — tests validate integration of real components

## Fixtures and Factories

**Test Data:**
```python
# Pattern: inline synthetic data construction via helper methods
def _build_synthetic_batch(self):
    """Build a synthetic batch of 3 sources with different valid lengths."""
    dt = 0.5
    N = 10
    num_modes = 2
    Nbatch = 3
    # ... construct numpy arrays directly
    return times, mask, amplitude, phase, dt

# Used in each test:
def test_output_shapes(self):
    times, mask, amplitude, phase, dt = self._build_synthetic_batch()
    ...
```

```python
# Pattern: inline creation for simple objects
frqs = xp.logspace(-5., 0., 1000)
Sn = get_sensitivity(frqs, sens_fn="X1TDISens", model=lisa.sangria)
```

**Location:**
- No separate fixtures directory — test data is constructed inline within test classes
- `tests/fresnel_plots/` stores diagnostic plot outputs (not fixtures)

## Coverage

**Requirements:** Not enforced — no minimum threshold configured in `pyproject.toml`

**Configuration (`pyproject.toml`):**
```toml
[tool.coverage]
paths.source = ["src/", "**/site-packages/"]
report.omit = [
    "*/lisatools/_version.py",
    "*/lisatools/tests/*.py",
    "*/lisatools/git_version.py",
]
```

**View Coverage:**
```bash
uv run coverage run -m pytest tests/
uv run coverage html --include="src/lisatools/*" --directory=tests/results/coverage
```

## Test Types

**Unit Tests:**
- Scope: individual functions and classes in isolation
- Examples: `test_sources_utils.py` (pure coordinate transforms), `test_get_amp_phase_shifting.py` (standalone logic mirrored from production class)
- Pattern: construct minimal inputs, assert output properties

**Integration Tests:**
- Scope: full module interactions including detector orbits + sensitivity + C++ extensions
- Examples: `test_detector.py`, `test_orbits.py`, `test_sensitivity.py`, `test_fresnel.py`
- Pattern: instantiate real objects, exercise full compute path, check for NaN-free outputs

**E2E Tests:**
- Not present as a dedicated category

## Common Patterns

**Dual-backend (CPU/GPU) testing:**
```python
# Backend detection at module level
try:
    import cupy as cp
    cp.cuda.runtime.setDevice(0)
    gpu_available = True
except (ImportError, ModuleNotFoundError):
    gpu_available = False

# Backend selection per test
xp = cp if gpu_available else np
force_backend = "cuda12x" if gpu_available else "cpu"
```

**Parametrized backend testing with pytest:**
```python
try:
    import cupy as cp
    backends = ["cpu", "gpu"]
except ImportError:
    backends = ["cpu"]

@pytest.mark.parametrize("force_backend", backends)
class TestFresnelVsScipy:
    def test_single_binary(self, force_backend):
        fresnel_wrap, xp = _make_fresnel_wrap(dt_window, df, force_backend)
        if force_backend == "gpu":
            output = output.get()  # transfer back to CPU for numpy assertions
```

**NaN-free numerical output assertion:**
```python
self.assertFalse(xp.any(xp.isnan(result)))
```

**Numerical tolerance assertions:**
```python
np.testing.assert_allclose(np.abs(output), np.abs(h_scipy), rtol=2e-2,
                           err_msg="Amplitude mismatch: C++ vs scipy")
phase_diff = np.angle(output * np.conj(h_scipy))
np.testing.assert_allclose(phase_diff, 0.0, atol=3e-2,
                           err_msg="Phase mismatch: C++ vs scipy")
```

**Standalone logic mirroring for hard-to-instantiate classes:**
```python
# When production class requires heavy deps (phentax, JAX), mirror the logic
# as a standalone function for testability — see test_get_amp_phase_shifting.py
def _trim_and_shift_times(times, mask, dt):
    """Standalone mirror of PhenomTHMTDIOnFlyWaveform.trim_and_shift_times"""
    ...
```

**GPU result retrieval:**
```python
if force_backend == "gpu":
    output = output.get()  # cupy → numpy transfer before numpy assertions
```

---

*Testing analysis: 2026-04-28*
