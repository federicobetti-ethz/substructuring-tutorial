Reference
=========

The package is imported from the repository root. It is not installed as a
distribution: ``package-mode`` is off, and the scripts insert the root on
``sys.path``.

Finite element model
--------------------

.. automodule:: substructuring.fem
   :no-members:
   :no-index:

.. autoclass:: substructuring.fem.Material
   :exclude-members: __init__, __new__
.. autoclass:: substructuring.fem.Model
   :exclude-members: __init__, __new__
.. autofunction:: substructuring.fem.rectangle_mesh
.. autofunction:: substructuring.fem.assemble_plane_stress
.. autofunction:: substructuring.fem.structural_model
.. autofunction:: substructuring.fem.vibration_modes

Component-mode synthesis
------------------------

.. autofunction:: substructuring.static_condensation.static_condensation
.. autofunction:: substructuring.craig_bampton.craig_bampton
.. autofunction:: substructuring.rubin.rubin

.. autoclass:: substructuring.coupling.ReducedModel
   :exclude-members: __init__, __new__
.. autoclass:: substructuring.coupling.CoupledModel
   :exclude-members: __init__, __new__
.. autoclass:: substructuring.coupling.ModeShape
   :exclude-members: __init__, __new__
.. autofunction:: substructuring.coupling.couple

Mode figures
------------

.. autofunction:: substructuring.visualize.plot_modes
.. autofunction:: substructuring.visualize.plot_vibration_modes
.. autofunction:: substructuring.visualize.plot_static_condensation
.. autofunction:: substructuring.visualize.plot_tip
.. autofunction:: substructuring.visualize.plot_model
.. autofunction:: substructuring.visualize.plot_assembly
.. autofunction:: substructuring.visualize.rigid_body_modes
.. autofunction:: substructuring.visualize.mid_interface_column
