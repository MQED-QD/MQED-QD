MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.4.4** adds a shared flattened energy/Rx scheduler for two-layer
Sommerfeld and N-layer Green-tensor calculations. One- or few-frequency jobs
can distribute contiguous horizontal-distance chunks across Joblib workers or
MPI ranks, while backward-compatible energy-row defaults and N-layer
``fixed_grid`` kernel reuse remain intact.

Integration warnings now include energy, wavelength, and position context, and
the new ``mqed_validate_gf_h5`` command checks separation-indexed HDF5 tensors
for valid shapes, finite values, metadata, and component consistency. Material
loading also supports labeled and headerless spreadsheets and includes bundled
fused-silica optical data.

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
