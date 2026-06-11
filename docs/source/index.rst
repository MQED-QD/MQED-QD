MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.2.0** adds N-layer dyadic Green's functions for finite planar
dielectric stacks. The release includes the new ``mqed_GF_NLayer`` CLI,
five-layer Ag/spacer example configs, DCIM and hybrid direct/DCIM integration
tools, MPI execution for HPC jobs, and API reference pages for the new layered
Green's-function modules.

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
