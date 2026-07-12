MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.4.0** turns the N-layer and Mie Green's-function workflows into
documented spectral-density pipelines. Scan-indexed Mie HDF5 files now work with
``mqed_calc_spec_dens`` and ``mqed_plot_spec_dens``, preserving explicit
source/observer positions and selecting scan curves by physical distance. The
release also adds curated N-layer reference HDF5 files, comparison plot configs,
and reference figures for direct quadrature, singularity-aware integration, and
pole-aware hybrid DCIM.

New tutorials cover N-layer planar stacks and Mie core-shell spherical cavities,
while new theory pages summarize N-layer Sommerfeld integrals, singularity-aware
pole extraction, hybrid DCIM acceleration, and Mie Green's functions. The Mie
workflow remains experimental until benchmark comparisons against literature
results are completed.

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
