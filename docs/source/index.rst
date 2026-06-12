MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.2.1** improves the N-layer Green's-function workflow for sparse
emitter-separation grids and spectral-density validation. Green's-function CLIs
can now preserve flexible Rx grids such as ``[0, d, 2d, ...]`` in HDF5, the
N-layer solver has an opt-in fixed-grid path that reuses Bessel-free Sommerfeld
kernels across many Rx values, and the DDI builder accepts sparse separation
grids with tolerant physical-value lookup. Spectral-density plots can select
curves by physical ``Rx_nm`` values, display ``J(ω)`` in either eV or SI
``s^-1``, and apply scientific y-axis formatting for large SI rates.

.. image:: _static/workflow_diagram.png
   :width: 600
   :align: center
   :alt: MQED-QD Workflow

Key Features
------------

- Dyadic Green's functions via Sommerfeld integrals and N-layer planar stacks
- Resonance energy transfer (RET) and field enhancement (FE) analysis
- Open-system dynamics: Lindblad master equation & NHSE
- Boundary Element Method (BEM) for arbitrary geometries
- Hydra-based configuration with Joblib, MPI, and SGE workflow support

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   getting-started

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   configuration
   api/index

.. seealso::

   `GitHub Repository <https://github.com/MQED-QD/MQED-QD#readme>`_
      Project overview, installation quick-start, latest updates, and citation info.
