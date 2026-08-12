MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.4.3** adds a compact ``ring_circulant`` Mie layout for
symmetry-compatible spherical emitter rings. It stores one dipole-projected
cyclic Green-function row and supports emission spectra, spectral densities,
pair-style plotting, projected DDI construction, and stationary Lindblad
dynamics with periodic coupling filters.

N-layer MPI jobs can now distribute horizontal-distance chunks across ranks
when a single- or few-frequency sweep would otherwise leave most ranks idle.
Whole-energy batching remains in place for ``fixed_grid`` integration so its
sampled Sommerfeld kernels are still reused across the complete Rx grid.
Annotated sphere-ring and single-frequency DBP/DBR examples document both
workflows.

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
