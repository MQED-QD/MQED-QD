.. _getting-started:

===============
Getting Started
===============

This page introduces the core simulation workflow and the command-line tools
that ship with MQED-QD.  Make sure you have completed :doc:`/installation`
before proceeding.

Simulation workflow
-------------------

A typical MQED-QD study follows a three-stage pipeline:

.. image:: /_static/workflow_diagram.png
   :width: 600
   :align: center
   :alt: MQED-QD simulation workflow: Green's function → Quantum dynamics → Post-processing

|

Each stage reads its configuration from a YAML file under ``configs/`` and
writes results into a timestamped Hydra output directory under ``outputs/``.

CLI commands
------------

Installing the package registers the following entry points:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``mqed_GF_Sommerfeld``
     - Compute the dyadic Green's function for a planar layered system
       (Sommerfeld integration).
   * - ``mqed_RET``
     - Resonance energy transfer analysis.
   * - ``mqed_FE``
     - Field-enhancement analysis.
   * - ``mqed_lindblad``
     - Time evolution with Lindblad master equation.
   * - ``mqed_nhse``
     - Time evolution with the non-Hermitian Schrödinger equation (faster
       for large systems).
   * - ``mqed_nhse_disorder``
     - Disorder-averaged NHSE sweep.
   * - ``mqed_plot_msd``
     - Plot mean-squared displacement.
   * - ``mqed_plot_sqrt_msd``
     - Plot root-mean-squared displacement.
   * - ``mqed_plot_IPR``
     - Plot inverse participation ratio.
   * - ``mqed_plot_PR``
     - Plot participation ratio.
   * - ``mqed_BEM_compute_peff``
     - Compute effective dipole-moment intensity (BEM).
   * - ``mqed_BEM_reconstruct_GF``
     - Reconstruct dyadic Green's function from BEM simulation.
   * - ``mqed_BEM_compare_silver``
     - Compare BEM and Fresnel results for a silver planar interface.

Quick first run
---------------

The fastest way to verify everything works is to run the Green's function
simulation with default settings:

.. code-block:: bash

   mqed_GF_Sommerfeld

This computes the dyadic Green's function for a silver half-space at 1.0 eV
with donor and acceptor 5 nm above the surface.  When it finishes you will
see a log line like:

.. code-block:: text

   Simulation complete. Output saved to: /…/outputs/Dyadic_GF_Sommerfeld/…/result_Ag_5_nm.hdf5

To override a parameter from the command line:

.. code-block:: bash

   mqed_GF_Sommerfeld simulation.energy_eV=1.864

See the :ref:`tutorial-gf-sommerfeld` tutorial for a full walkthrough
including multi-energy runs and downstream usage.

Hydra configuration
-------------------

Every command reads a YAML config from the ``configs/`` directory.
You can inspect the active configuration for any command with the ``--cfg job``
flag:

.. code-block:: bash

   mqed_GF_Sommerfeld --cfg job

Override any key on the command line using dot-notation:

.. code-block:: bash

   mqed_lindblad simulation.Nmol=50 simulation.d_nm=5.0

Hydra automatically:

- writes all outputs to a timestamped directory under ``outputs/``,
- saves a copy of the resolved config alongside the results, and
- supports multi-run sweeps via the ``-m`` flag.

See the `Hydra documentation <https://hydra.cc/docs/intro/>`_ for details on
overrides, config groups, and sweeps.

What's next?
------------

.. list-table::
   :widths: 40 60

   * - :ref:`tutorial-gf-sommerfeld`
     - Compute Green's functions for a planar two-layer system.
   * - :doc:`/tutorials/field_enhancement`
     - Analyse field enhancement from cached Green's functions.
   * - :doc:`/tutorials/quantum_dynamics`
     - Run Lindblad or NHSE time evolution.
   * - :doc:`/tutorials/plotting`
     - Plot MSD, RMSD, and participation ratios.
