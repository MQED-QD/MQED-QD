MQED-QD Documentation
=====================

**MQED-QD** is a Python toolkit for simulating exciton-polariton transport
near plasmonic interfaces using macroscopic quantum electrodynamics.

Latest Update
-------------

**Version 1.4.6** improves separation-indexed emission-spectrum calculations:
when multiple ``Rx_nm`` samples fall within the configured tolerance for an
emitter-pair separation, the nearest sample is selected. Mie material loading
also accepts Rakić gold settings supplied as a plain YAML mapping. Both changes
have regression coverage. The Mie and emission-spectrum workflows remain
experimental and have not been literature-validated.

See :doc:`the spectral-density tutorial <tutorials/spectral_density>` for
Green-tensor layouts and ``CHANGELOG.md`` for the full release notes.

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
