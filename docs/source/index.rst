MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.4.1** stabilizes DBP-in-DBR Green-tensor calculations and adds direct
physical-coupling analysis. The N-layer workflow now keeps the DBP emitter in the
intended zero-based source layer, rejects non-finite tensors before caching, and
avoids algebraic high-q overflow for off-center source positions.

The new ``mqed_plot_dbr_couplings`` command projects separation-indexed Green
tensors onto donor and acceptor orientations and plots signed or absolute
``V_ij`` and ``hbarGamma_ij`` values without vacuum-normalized enhancement
ratios. Ordered ``Rx_nm`` selection can combine dense near-field samples with
sparse far-field DBR points, using strict or nearest-grid matching with complete
HDF5, CSV, and PNG provenance.

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
