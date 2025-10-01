"""
XEFI provides tools for simulating and fitting X-ray Electric Field Intensity profiles.
"""

from XEFI import models
from XEFI.models import (
    XEF_Basic,
    XEF_Sliced,
    BasicResult,
    BasicRoughResult,
    SlicedResult,
)
from XEFI.results import XEF_method
import XEFI.fitting as fitting


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
