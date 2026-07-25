MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.4.2** adds optional TE/TM-resolved scattering output to the N-layer
solver while preserving the existing total/vacuum-only schema by default.
Stored structure tensors satisfy ``structure = TE + TM`` and can be selected
independently by downstream collective-emission analysis.

Emission workflows now support Varguet-effective and renormalized-total Green
matrices for separation-indexed chains as well as pair-indexed geometries.  The
verified sphere-ring workflow persists its emitter positions and orientations,
and emission maps follow the literature convention with transition energy on
the horizontal axis and emission energy on the vertical axis. Personal or
unpublished Hydra configurations can remain under the ignored ``local/`` tree.

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
