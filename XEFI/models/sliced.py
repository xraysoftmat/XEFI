"""
Module for the XEFI calculation of a sliced set of layers, decomposed into a specified thickness.
"""

from XEFI.models.results import BaseResult
import kkcalc as kkc


class SlicedResult(BaseResult):
    """
    Result class for the sliced XEFI model, inheriting from BaseResult.

    This class extends the BaseResult to include additional properties specific to the sliced model.

    Notably, `z_interfaces` is the list of interfaces prior to slicing, and `z` is the list of interfaces
    after slicing. `layer_names` now corresponds to the layers separated by `z_interfaces`, not `z`.

    Attributes
    ----------
    """
    
    
