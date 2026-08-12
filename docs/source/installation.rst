.. _installation:

============
Installation
============

Requirements
------------

- Python >= 3.10
- A conda-compatible package manager: `conda <https://docs.conda.io/>`_,
  `mamba <https://mamba.readthedocs.io/>`_, or
  `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_
  (recommended for the full scientific stack)
- Git

.. note::

   MQED-QD depends on compiled libraries (MEEP, MPB, QuTiP, GSL, FFTW, etc.)
   that are easiest to obtain through conda-forge.
   A pip-only install is possible but you must provide these libraries yourself.

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/MQED-transport/Macroscopic-Quantum-Electrodynamics.git
   cd MacroscopicQED

Conda / Mamba install (recommended)
------------------------------------

.. code-block:: bash

   # Use conda, mamba, or micromamba — the commands are interchangeable
   conda env create -f environment.yaml
   conda activate mqed
   pip install -e .          # install the package in editable mode

This creates the ``mqed`` environment with all dependencies (NumPy, SciPy,
QuTiP, Hydra, Matplotlib, etc.) and registers the CLI entry points listed in
:doc:`/getting-started`.

Pip-only install
----------------

If you prefer not to use conda:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

.. warning::

   The pip path installs only the Python dependencies declared in ``setup.py``.
   Compiled solver backends (MEEP, MPB, harminv, GSL, FFTW) will **not** be
   available unless you install them separately.
   Use the conda path if you need BEM or eigenmode features.

Verify the installation
-----------------------

After installing, check that the CLI entry points are available and the test
suite passes:

.. code-block:: bash

   # Should print the Hydra help / default config
   mqed_GF_Sommerfeld --help

   # Run the test suite
   pytest

If ``mqed_GF_Sommerfeld`` is not found, make sure your environment is
activated and ``pip install -e .`` completed without errors.

Dielectric-function data
------------------------

The default configuration reads material data from an Excel file shipped with
the repository::

   DielectricFunction/dielectric function.xlsx

Sheet ``Ag_BEM`` contains a fitted dielectric function for silver.
To use a different material, either:

- add a new sheet to the existing workbook and update
  ``material.excel_config.sheet_name`` in the YAML config, or
- supply a separate ``.xlsx`` file and point
  ``material.excel_config.filepath`` to it, or
- bypass the file entirely by setting ``material.source_type: constant`` and
  providing a complex value in ``material.constant_value``.

Data directories
----------------

Several workflows cache intermediate results that downstream steps consume.
You may need to create these directories under the project root before your
first run:

.. code-block:: bash

   mkdir -p data/GF_cache data/QDyn_cache

===================  ===============================================
Directory            Purpose
===================  ===============================================
``data/GF_cache``    Cached Green's function HDF5 files
``data/QDyn_cache``  Cached quantum-dynamics HDF5 files
``outputs/``         Created automatically by Hydra at runtime
===================  ===============================================

.. seealso::

   :doc:`/getting-started` for a quick walkthrough of the core simulation
   workflow.
