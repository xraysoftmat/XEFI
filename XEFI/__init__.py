"""
XEFI provides tools for simulating and fitting X-ray Electric Field Intensity profiles.
"""

# import XEFI.fitting as fitting
import importlib.metadata

from XEFI import models, utils
from XEFI.models import (
    BasicResult,
    BasicRoughResult,
    SlicedResult,
    XEF_Basic,
    XEF_Sliced,
)
from XEFI.results import (
    XEF_method,
)

# Calculate the __version__ from the pyproject.toml file
__version__ = importlib.metadata.version("XEFI")
del importlib

__all__ = [
    # Modules:
    "models",
    "utils",
    # "fitting",
    # Enumerates:
    "XEF_method",
    # Calculation Methods:
    "XEF_Basic",
    "XEF_Sliced",
    # Result classes:
    "BasicResult",
    "BasicRoughResult",
    "SlicedResult",
    # Properties
    "__version__",
]
