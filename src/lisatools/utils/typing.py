from typing import Union, TYPE_CHECKING
import numpy as np
from types import ModuleType

try:
    from typing import TypeAlias
except ImportError:
    # Fallback for Python versions < 3.10 if needed
    from typing_extensions import TypeAlias 

if TYPE_CHECKING:
    # Strict typing context: Pylance will use this branch.
    try:
        import cupy as cp
        ArrayLike: TypeAlias = Union[np.ndarray, cp.ndarray]
    except ImportError:
        ArrayLike: TypeAlias = np.ndarray
else:
    # Runtime context: Python executes this branch without loading cupy.
    # Set to np.ndarray so it evaluates as a valid Type at runtime without crashing.
    ArrayLike: TypeAlias = np.ndarray  

ArrayModule: TypeAlias = ModuleType