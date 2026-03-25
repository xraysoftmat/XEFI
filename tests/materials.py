
from typing import TypedDict

class Material(TypedDict):
    formula: str
    density: float

P3HT: Material = {
    "formula": "C10H14S",
    "density": 1.33,
}

PS: Material = {
    "formula": "C8H8",
    "density": 1.05,
}

Si: Material = {
    "formula": "Si",
    "density": 2.329,
}

Air: Material = {
    "formula": "N78O20Ar1", # A very rough approximation of air.
    "density": 1.225e-3,
}

MATERIALS: dict[str, Material] = {
    "P3HT": P3HT,
    "PS": PS,
    "Si": Si,
    "Air": Air
}