.. _tutorial-gf-sommerfeld:

===================================================
Dyadic Green's Function via Sommerfeld Integrals
===================================================

Goal
----

In this tutorial you will compute the **dyadic Green's function**
:math:`\overleftrightarrow{G}(\mathbf{r}_D, \mathbf{r}_A, \omega)` for a
two-layer planar system (vacuum above a semi-infinite substrate) using the
Fresnel-coefficient / Sommerfeld-integral method implemented in MQED-QD.

By the end you will know how to:

- run the ``mqed_GF_Sommerfeld`` command with default and custom parameters,
- specify single-frequency, multi-frequency (list), and swept-frequency (dict)
  inputs,
- accelerate multi-frequency sweeps with **joblib** (multicore) or **MPI**
  (cluster) parallelism,
- locate and interpret the HDF5 output file.

.. seealso::

   :ref:`theory-two-layer` for the mathematical derivation of the Sommerfeld
   integrals and Fresnel reflection coefficients used by this solver.

Prerequisites
-------------

Make sure you have installed the package and activated the environment as
described in :doc:`/installation`.


Quick start
-----------

Run from the repository root with all defaults:

.. code-block:: bash

   mqed_GF_Sommerfeld

This uses the YAML configuration at ``configs/Dyadic_GF/GF_Sommerfeld.yaml``.
The default settings compute the Green's function for a **single energy
(1.0 eV)** at a height of **5 nm** above a silver (Ag) surface.


Specifying the emitter frequency
--------------------------------

You can override any configuration key on the command line using
`Hydra <https://hydra.cc/>`_ syntax.

**Single energy**

.. code-block:: bash

   mqed_GF_Sommerfeld simulation.energy_eV=1.864

**List of energies**

.. code-block:: bash

   mqed_GF_Sommerfeld simulation.energy_eV=[1.5,1.8,2.0]

**Swept range (dict input)**

.. code-block:: bash

   mqed_GF_Sommerfeld simulation.energy_eV.min=1.0 \
       simulation.energy_eV.max=2.0 \
       simulation.energy_eV.points=11

This generates 11 linearly spaced energies from 1.0 to 2.0 eV.

.. tip::

   You can use wavelength instead of energy by setting
   ``simulation.spectral_param=wavelength_nm`` and providing
   ``simulation.wavelength_nm`` values in the same single / list / dict
   format.


Using a custom configuration file
----------------------------------

Every MQED command loads its default YAML from the ``configs/`` directory.
Hydra provides two flags that let you swap the entire configuration file
without editing code:

**Use a different YAML in the same config directory**

.. code-block:: bash

   mqed_GF_Sommerfeld --config-name=my_GF

This loads ``configs/Dyadic_GF/my_GF.yaml`` instead of the default
``GF_Sommerfeld.yaml``.  Your custom file must live in the same
``configs/Dyadic_GF/`` directory.

**Use a YAML from an arbitrary directory**

.. code-block:: bash

   mqed_GF_Sommerfeld --config-dir=/path/to/my/configs --config-name=my_GF

This tells Hydra to look for ``my_GF.yaml`` in ``/path/to/my/configs/``
instead of the default config directory.

.. tip::

   You can combine ``--config-name`` (or ``--config-dir``) with individual
   parameter overrides:

   .. code-block:: bash

      mqed_GF_Sommerfeld --config-name=my_GF simulation.energy_eV=2.0


Parallel execution
------------------

When computing the Green's function over many frequencies (e.g. a 200-point
spectral density sweep), the sequential loop can take tens of minutes.
Because each energy point is **completely independent**, the frequency axis
can be parallelized with no inter-worker communication.

MQED-QD supports two parallel backends, selectable via the ``parallel``
configuration group.

**Joblib (shared-memory multiprocessing)**

Best for laptops and single-node workstations.  Uses Python's
``multiprocessing`` under the hood via `joblib <https://joblib.readthedocs.io>`_.

.. code-block:: bash

   # Use all available CPU cores
   mqed_GF_Sommerfeld \
       simulation.energy_eV='{min: 0.5, max: 4.0, points: 200}' \
       parallel.backend=joblib

   # Limit to 4 workers
   mqed_GF_Sommerfeld \
       simulation.energy_eV='{min: 0.5, max: 4.0, points: 200}' \
       parallel.backend=joblib parallel.n_jobs=4

Setting ``parallel.n_jobs=-1`` (the default) uses all available cores.

.. tip::

   On a machine with 8 cores, a 200-energy sweep that takes ~30 min
   sequentially will finish in ~4 min with ``parallel.backend=joblib``.

**MPI (distributed parallelism)**

Best for HPC clusters with multi-node allocation.  Requires
`mpi4py <https://mpi4py.readthedocs.io>`_ and an MPI runtime (e.g.
OpenMPI or MPICH).

*Auto-launch mode* (default) — the script detects it is not inside an MPI
communicator and re-launches itself under ``mpiexec``:

.. code-block:: bash

   mqed_GF_Sommerfeld \
       simulation.energy_eV='{min: 0.5, max: 4.0, points: 200}' \
       parallel.backend=mpi parallel.mpi_nproc=16

*Manual launch* — submit from a job scheduler (SLURM, PBS, etc.) with
``mpi_auto_launch=false``:

.. code-block:: bash

   # Inside a SLURM job script, for example:
   srun -n 32 mqed_GF_Sommerfeld \
       simulation.energy_eV='{min: 0.5, max: 4.0, points: 200}' \
       parallel.backend=mpi parallel.mpi_auto_launch=false

Energy points are scattered round-robin across MPI ranks.  Only rank 0
writes the final HDF5 file.

.. list-table:: Parallel configuration reference
   :header-rows: 1
   :widths: 35 45 20

   * - Parameter
     - Description
     - Default
   * - ``parallel.backend``
     - Execution mode: ``sequential``, ``joblib``, or ``mpi``
     - ``sequential``
   * - ``parallel.n_jobs``
     - Number of joblib workers (``-1`` = all cores)
     - ``-1``
   * - ``parallel.mpi_nproc``
     - Number of MPI ranks for auto-launch
     - ``4``
   * - ``parallel.mpi_auto_launch``
     - Re-launch under ``mpiexec`` automatically
     - ``true``
   * - ``parallel.mpi_exec``
     - Path to MPI launcher binary
     - ``mpiexec``


Configuration reference
-----------------------

The full default configuration file
(``configs/Dyadic_GF/GF_Sommerfeld.yaml``) is reproduced below with
annotations.

.. code-block:: yaml

   # ── Material ────────────────────────────────────────────
   material:
     source_type: excel          # 'excel' (dispersive) or 'constant'
     constant_value: 9.0+0.0j   # used only when source_type = 'constant'
     excel_config:
       filepath: "DielectricFunction/dielectric function.xlsx"
       sheet_name: "Ag_BEM"

   # ── Simulation ──────────────────────────────────────────
   simulation:
     material: "Ag"             # label used in the output filename
     spectral_param: "energy_eV"  # 'energy_eV' or 'wavelength_nm'

     energy_eV:                 # single value, dict {min, max, points}, or list
       min: 1.0
       max: 1.0
       points: 1

     wavelength_nm:             # same three formats as energy_eV
       min: 500.0
       max: 500.0
       points: 1

     position:
       zD: 5.0e-9              # donor height above surface (m)
       zD_nm: 5                # donor height label for filename (nm)
       zA: 5.0e-9              # acceptor height above surface (m)
       Rx_nm:                  # horizontal donor–acceptor distances
         start: 0.0
         stop: 300.0
         points: 301           # 0, 1, 2, …, 300 nm

     integration:              # numerical quadrature controls
       qmax: null              # finite cutoff for evanescent tail; null → ∞
       epsabs: 1.0e-10         # absolute tolerance
       epsrel: 1.0e-10         # relative tolerance
       limit: 400              # max sub-intervals for quad_vec
       split_propagating: true # split at k₀ for propagating / evanescent

   # ── Output ──────────────────────────────────────────────
   output:
     filename: "Fresnel_GF_planar_${simulation.material}_height_${simulation.position.zD_nm}nm.hdf5"

   # ── Parallel execution ──────────────────────────────────
   parallel:
     backend: sequential        # 'sequential', 'joblib', or 'mpi'
     n_jobs: -1                 # joblib workers; -1 = all cores
     mpi_nproc: 4               # MPI ranks for auto-launch
     mpi_auto_launch: true      # re-launch under mpiexec if needed
     mpi_exec: mpiexec          # path to MPI launcher

.. list-table:: Key parameters at a glance
   :header-rows: 1
   :widths: 30 50 20

   * - Parameter
     - Description
     - Default
   * - ``material.source_type``
     - Source of the dielectric function (``excel`` or ``constant``)
     - ``excel``
   * - ``simulation.spectral_param``
     - Which spectral unit to use (``energy_eV`` or ``wavelength_nm``)
     - ``energy_eV``
   * - ``simulation.position.zD``
     - Donor height above the surface in **meters**
     - ``5.0e-9``
   * - ``simulation.position.zA``
     - Acceptor height above the surface in **meters**
     - ``5.0e-9``
   * - ``simulation.position.Rx_nm``
     - Horizontal distance grid (``start``, ``stop``, ``points``) in **nm**
     - 0 – 300 nm, 301 pts
   * - ``simulation.integration.epsabs``
     - Absolute tolerance for Sommerfeld quadrature
     - ``1.0e-10``
   * - ``simulation.integration.limit``
     - Maximum sub-intervals for adaptive quadrature
     - ``400``


Expected output
---------------

After the simulation finishes, a success message is printed to the terminal
(and to the Hydra log file
``outputs/Dyadic_GF_Sommerfeld/.../Dyadic_GF_Sommerfeld.log``):

.. code-block:: text

   2025-10-24 11:40:02.818 | SUCCESS | mqed.Dyadic_GF.main:run_simulation:114
   - Simulation complete. Output saved to:
     /.../MacroscopicQED/outputs/Dyadic_GF_Sommerfeld/Y-M-D/H-M-S/YOUR_NAME.hdf5

The output is an **HDF5** file that stores:

- the 9-component dyadic Green's function (total and vacuum) as
  4-D complex arrays of shape ``[nE, nR, 3, 3]``,
- the energy grid (``energy_eV``),
- the horizontal distance grid (``Rx_nm``),
- the donor and acceptor heights (``zD``, ``zA``).


Using the output in downstream analyses
----------------------------------------

Many MQED-QD workflows read a cached Green's function file as their input.
The recommended workflow is:

1. Create a cache directory under the project root (if it does not exist):

   .. code-block:: bash

      mkdir -p data/GF_cache

2. Copy (or symlink) the HDF5 result into the cache:

   .. code-block:: bash

      cp outputs/Dyadic_GF_Sommerfeld/<date>/<time>/YOUR_NAME.hdf5 \
         data/GF_cache/

3. Update the ``greens.h5_path`` or ``input_file`` key in the downstream
   configuration to point to this cached file.

.. seealso::

   - :ref:`tutorial-field-enhancement` -- compute the field enhancement using
     the Green's function produced here.
   - :ref:`tutorial-quantum-dynamics` -- run Lindblad or NHSE dynamics with
     this Green's function as input.
