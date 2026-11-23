"""Plane-stress finite elements and primal component-mode synthesis."""

from substructuring.coupling import CoupledModel, ReducedModel, couple
from substructuring.craig_bampton import craig_bampton
from substructuring.fem import Material, Model, assemble_plane_stress, rectangle_mesh, structural_model
from substructuring.rubin import rubin

__all__ = [
    "CoupledModel",
    "Material",
    "Model",
    "ReducedModel",
    "assemble_plane_stress",
    "couple",
    "craig_bampton",
    "rectangle_mesh",
    "rubin",
    "structural_model",
]
