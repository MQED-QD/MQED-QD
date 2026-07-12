.. _tutorials:

=========
Tutorials
=========

Step-by-step guides that walk you through the MQED-QD pipeline — from
computing dyadic Green's functions to visualising quantum-dynamics results.

The tutorials are designed to be followed in order, but each one ships with
bundled example data under ``data/example/`` so you can jump straight to any
topic.

Green-function solver quick choice
==================================

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Tutorial
     - Use when
     - Output layout
   * - :ref:`tutorial-gf-sommerfeld`
     - You need a two-layer planar interface.
     - ``separation`` indexed by horizontal distance ``Rx``.
   * - :ref:`tutorial-gf-nlayer`
     - You need a finite planar stack with several layers.
     - ``separation`` indexed by horizontal distance ``Rx``.
   * - :ref:`tutorial-gf-mie`
     - You need a sphere, spherical cavity, or core-shell sphere.
     - ``scan`` for fixed-source observer scans, or ``pair`` for emitter lists.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   GF_Sommerfeld
   GF_NLayer
   GF_Mie
   BEM-Vacuum
   BEM-Reconstruct-GF
   BEM-Nanorod
   field_enhancement
   spectral_density
   quantum_dynamics
   plotting
