"""
Models and result classes for XEFI calculation.
"""

import XEFI.models.sliced as sliced
import XEFI.models.basic as basic
from XEFI.models.sliced import SlicedResult, XEF_Sliced
from XEFI.models.basic import BasicResult, BasicRoughResult, XEF_Basic

__all__ = [
    # Modules:
    sliced,
    basic,
    # Methods:
    XEF_Basic,
    XEF_Sliced,
    # Result Classes
    BasicResult,
    BasicRoughResult,
    SlicedResult,
]
