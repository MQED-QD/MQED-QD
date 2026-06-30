MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.3.1** refines the N-layer Green's-function workflow for
plasmonic stacks. The N-layer solver now includes singularity-aware quadrature,
pole diagnostics, pole-subtracted direct integration, and a validation-gated
``pole_aware_hybrid_dcim`` path for high-q tail acceleration. DCIM-family methods
automatically route ``Rx = 0`` local calculations through ``singularity_aware``,
and q-window controls can be written as dimensionless multiples of ``|k0|``.
Spectral-density plotting also supports multi-file comparison curves with
file-level styles and per-selected separation/pair overrides.

The Mie Green's-function and emission-spectrum workflows are still experimental
and should be treated as development APIs until they are benchmarked against
literature results.

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
