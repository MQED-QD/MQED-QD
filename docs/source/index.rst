MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.1.4** adds reproducible, literature-connected tutorials for the
Sommerfeld Green's-function and spectral-density workflows. The documentation
now includes complete single-frequency and multi-frequency planar Ag examples,
bundled HDF5 example data, and a spectral-density walkthrough that reproduces a
Figure-2C-style result from Chuang *et al.*

.. image:: _static/workflow_diagram.png
   :width: 600
   :align: center
   :alt: MQED-QD Workflow

Key Features
------------

- Dyadic Green's functions via Sommerfeld integrals
- Resonance energy transfer (RET) and field enhancement (FE) analysis
- Open-system dynamics: Lindblad master equation & NHSE
- Boundary Element Method (BEM) for arbitrary geometries
- Hydra-based configuration for reproducible workflows

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
