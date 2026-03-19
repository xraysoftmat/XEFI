"""
XEFI provides tools for simulating and fitting X-ray Electric Field Intensity profiles.
"""

from XEFI import models
from XEFI.models import (
    XEF_Basic,
    XEF_Sliced,
    SlicedResult,
    BasicResult,
    BasicRoughResult,
)
from XEFI.results import (
    XEF_method,
)
import XEFI.fitting as fitting

import importlib.metadata

# Calculate the __version__ from the pyproject.toml file
__version__ = importlib.metadata.version("XEFI")
del importlib

__all__ = [
    # Modules:
    models,
    fitting,
    # Enumerates:
    XEF_method,
    # Calculation Methods:
    XEF_Basic,
    XEF_Sliced,
    # Result classes:
    BasicResult,
    BasicRoughResult,
    SlicedResult,
]
