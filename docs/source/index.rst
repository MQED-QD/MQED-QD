MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.3.2** refines the experimental Mie Green's-function workflow for
core-shell spherical-cavity studies. Mie scan and pair outputs now use the shared
dyadic-Green HDF5 utilities, scan files store explicit source/observer positions,
and generated filenames follow the N-layer-style ``.hdf5`` parameter suffixes.
The default Mie config documents scan vs pair semantics, segmented energy grids,
and the ``0``, ``2``, and ``20`` nm core-cavity positions used for spectral-density
literature reproduction. A new SGE launcher, ``gf_mie_single_job.sh``, provides a
documented MPI run path for many-frequency Mie calculations.

The Mie Green's-function workflow is still experimental and should be treated as
a development API until benchmark comparisons against literature results are
completed.

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
