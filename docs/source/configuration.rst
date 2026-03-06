.. _configuration:

Configuration Reference
========================

All MQED-QD simulations are configured through YAML files managed by
`Hydra <https://hydra.cc/>`_. The configuration tree lives under
``configs/`` and is organised into five directories, each corresponding to a
stage of the workflow.

.. code-block:: text

   configs/
   ├── Dyadic_GF/      # Green's function computation
   ├── Lindblad/        # Quantum dynamics solver
   ├── analysis/        # Post-processing (FE & RET)
   ├── BEM/             # Boundary-element-method utilities
   └── plots/           # Observable plotting

Every CLI command loads its default config automatically. You can override any
parameter on the command line using Hydra's dotlist syntax:

.. code-block:: bash

   # Override a single key
   mqed_GF simulation.energy_eV=2.0

   # Override a nested key
   mqed_GF simulation.integration.qmax=100

   # Use a different config file
   mqed_GF --config-name=My_GF

See the `Hydra documentation <https://hydra.cc/docs/intro/>`_ for full
override syntax including sweeps and multirun.


.. _config-dyadic-gf:

``Dyadic_GF/`` — Green's Function Computation
-----------------------------------------------

**Files:** ``GF_Sommerfeld.yaml`` (default), ``My_GF.yaml``

Used by the ``mqed_GF`` command (:doc:`/tutorials/GF_Sommerfeld`).

``material``
^^^^^^^^^^^^

Defines the dielectric function of the substrate.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``source_type``
     - ``str``
     - ``"excel"`` to load experimental data, or ``"constant"`` for a
       frequency-independent dielectric constant.
   * - ``constant_value``
     - ``complex``
     - Dielectric constant when ``source_type: constant``
       (e.g. ``-11.0+1.0j``).
   * - ``excel_config.filepath``
     - ``str``
     - Path to the Excel file containing the tabulated dielectric function.
   * - ``excel_config.sheet_name``
     - ``str``
     - Sheet name inside the Excel file.

``simulation``
^^^^^^^^^^^^^^

Controls the physical parameters and numerical integration.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``material``
     - ``str``
     - Human-readable material label (used in output filenames).
   * - ``spectral_param``
     - ``str``
     - ``"energy_eV"`` or ``"wavelength_nm"`` — selects which spectral
       parameterisation to use.
   * - ``energy_eV``
     - ``float | dict``
     - A single value (e.g. ``1.864``), or a range
       ``{start, stop, points}`` for a sweep.
   * - ``wavelength_nm``
     - ``float | dict``
     - Same as ``energy_eV`` but in nanometers.
   * - ``position.zD_nm``
     - ``float``
     - Donor height above the interface (nm).
   * - ``position.zA``
     - ``float``
     - Acceptor height (same units as ``zD``).
   * - ``position.Rx_nm``
     - ``dict``
     - Lateral donor–acceptor separation sweep:
       ``{start, stop, points}``.
   * - ``integration.qmax``
     - ``float``
     - Upper cutoff for the Sommerfeld integral in-plane momentum.
   * - ``integration.epsabs``
     - ``float``
     - Absolute tolerance for numerical quadrature.
   * - ``integration.epsrel``
     - ``float``
     - Relative tolerance.
   * - ``integration.limit``
     - ``int``
     - Maximum number of subintervals.
   * - ``integration.split_propagating``
     - ``bool``
     - Whether to split the integral at the propagating/evanescent
       boundary for improved accuracy.

``output``
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``filename``
     - ``str``
     - Output HDF5 filename template. Supports Hydra interpolation
       (e.g. ``${simulation.material}_GF.h5``).


.. _config-lindblad:

``Lindblad/`` — Quantum Dynamics
----------------------------------

**Files:** ``quantum_dynamics.yaml`` (default),
``quantum_dynamics_disorder.yaml``, ``quantum_dynamics_nhse.yaml``

Used by the ``mqed_QD`` command (:doc:`/tutorials/quantum_dynamics`).

``greens``
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``h5_path``
     - ``str``
     - Path to the cached Green's function HDF5 file produced by ``mqed_GF``.

``simulation``
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``t_ps.start``
     - ``float``
     - Start time (ps).
   * - ``t_ps.stop``
     - ``float``
     - End time (ps).
   * - ``t_ps.output_step``
     - ``float``
     - Time step for saved output (ps).
   * - ``Nmol``
     - ``int``
     - Number of molecules in the 1-D chain.
   * - ``d_nm``
     - ``float``
     - Lattice spacing (nm).
   * - ``lambda_nm``
     - ``float``
     - Transition wavelength (nm).
   * - ``gf_method``
     - ``str``
     - ``"Fresnel"`` or ``"BEM"`` — which Green's function to use.
   * - ``mode``
     - ``str``
     - ``"stationary"`` or ``"disorder"``.
   * - ``coupling_limit.enable``
     - ``bool``
     - If ``true``, truncate the coupling matrices at a finite range.
   * - ``coupling_limit.V_hop_radius``
     - ``int``
     - Maximum hopping range (sites) for the DDI matrix
       :math:`V_{\alpha\beta}`.
   * - ``coupling_limit.Gamma_hop_radius``
     - ``int``
     - Maximum hopping range (sites) for the dissipation matrix
       :math:`\Gamma_{\alpha\beta}`.
   * - ``coupling_limit.Gamma_rule``
     - ``str``
     - Strategy for truncating :math:`\Gamma` (e.g. ``"match_V"``).
   * - ``coupling_limit.keep_V_on_site``
     - ``bool``
     - Always keep the diagonal (:math:`\alpha = \beta`) DDI entries.
   * - ``coupling_limit.keep_Gamma_on_site``
     - ``bool``
     - Always keep the diagonal dissipation entries.
   * - ``dipole.mu_D_debye``
     - ``float``
     - Donor transition dipole moment (Debye).
   * - ``dipole.mu_A_debye``
     - ``float``
     - Acceptor transition dipole moment (Debye).
   * - ``dipole.theta_deg``
     - ``float``
     - Polar angle of the dipole orientation (degrees).
   * - ``dipole.phi_deg``
     - ``float``
     - Azimuthal angle of the dipole orientation (degrees).

``observables``
^^^^^^^^^^^^^^^

A list of observables to compute at each output time step. Each entry has:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Description
   * - ``name``
     - ``str``
     - Observable label (used in output files).
   * - ``kind``
     - ``str``
     - One of ``root_MSD``, ``X_shift``, ``X_shift2``, ``IPR_site``,
       ``pop_site``, ``X_shift_cond``, ``X_shift2_cond``.
   * - ``enabled``
     - ``bool``
     - Whether to compute this observable.
   * - ``params``
     - ``dict``
     - Observable-specific parameters (e.g. ``{site: 0}`` for
       ``pop_site``).

``initial_state``
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``site_index``
     - ``int``
     - Index of the initially excited site.

``solver``
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``method``
     - ``str``
     - ``"Lindblad"`` (full master equation) or ``"NonHermitian"``
       (non-Hermitian Hamiltonian approximation).

Disorder and NHSE Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``quantum_dynamics_disorder.yaml`` extends the base config with:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``disorder.n_realizations``
     - ``int``
     - Number of disorder realisations.
   * - ``disorder.seed``
     - ``int``
     - Random seed for reproducibility.
   * - ``disorder.n_jobs``
     - ``int``
     - Parallel workers (``-1`` for all cores).
   * - ``disorder.save_each``
     - ``bool``
     - Save each realisation individually.
   * - ``disorder_sigma_phi_deg``
     - ``float``
     - Standard deviation of orientational disorder (degrees).

``quantum_dynamics_nhse.yaml`` adds a ``height`` parameter and uses
``gf_method: BEM`` with ``coupling_limit.enable: false`` (long-range
coupling) for studying the non-Hermitian skin effect.


.. _config-analysis:

``analysis/`` — Post-Processing
----------------------------------

**Files:** ``FE.yaml``, ``RET.yaml``

Used by ``mqed_FE`` and ``mqed_RET``
(:doc:`/tutorials/field_enhancement`).

Both configs share the same structure. The main difference is that ``FE.yaml``
plots the real and imaginary components
(:math:`V_{\alpha\beta}/V_{0,\alpha\beta}` and
:math:`\Gamma_{\alpha\beta}/\Gamma_{0,\alpha\beta}`) separately, while
``RET.yaml`` plots the single enhancement factor :math:`\gamma`.

``input_file``
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``input_file``
     - ``str``
     - Path to the cached Green's function HDF5 file.

``orientations``
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``donor.theta_deg``
     - ``float``
     - Donor dipole polar angle (degrees).
   * - ``donor.phi_deg``
     - ``float | str``
     - Donor dipole azimuthal angle. Use ``"magic"`` for the magic angle.
   * - ``acceptor.theta_deg``
     - ``float``
     - Acceptor dipole polar angle.
   * - ``acceptor.phi_deg``
     - ``float | str``
     - Acceptor dipole azimuthal angle.

``plot_settings``
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``save_plot``
     - ``bool``
     - Write the figure to disk.
   * - ``dpi``
     - ``int``
     - Figure resolution.
   * - ``x_range_nm``
     - ``list``
     - ``[min, max]`` lateral distance range to plot (nm).
   * - ``components``
     - ``list``
     - ``["real", "imag"]`` (FE only). Omitted in RET.
   * - ``xlabel``, ``ylabel``
     - ``str``
     - Axis labels.
   * - ``title_template``
     - ``str``
     - Title with ``{energy:.3f}`` placeholder.
   * - ``legend.real_label``
     - ``str``
     - Legend entry for the real component (FE).
   * - ``legend.imag_label``
     - ``str``
     - Legend entry for the imaginary component (FE).
   * - ``legend.label``
     - ``str``
     - Single legend entry (RET).
   * - ``xscale``, ``yscale``
     - ``str``
     - ``"linear"`` or ``"log"``.
   * - ``xlim``, ``ylim``
     - ``list``
     - Axis limits ``[min, max]``.
   * - ``filename_prefix``
     - ``str``
     - Output filename prefix.


.. _config-bem:

``BEM/`` — Boundary Element Method
-------------------------------------

**Files:** ``compare_bem_dyadic.yaml``, ``compare.yaml``,
``compare_silver.yaml``, ``compare_enhancement.yaml``,
``compute_peff.yaml``, ``reconstruct_GF.yaml``

These configs drive the BEM validation and reconstruction utilities.

``compute_peff.yaml``
^^^^^^^^^^^^^^^^^^^^^

Compute the effective dipole moment from BEM near-field data.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``io.xlsx_path``
     - ``str``
     - Path to the BEM Excel export.
   * - ``io.output_csv``
     - ``str``
     - Output CSV with fitted effective dipoles.
   * - ``sim.lambdas_nm``
     - ``float | list | dict``
     - Wavelengths to process (single, list, or ``{start, stop, points}``).
   * - ``sim.Rx_min_nm``
     - ``float``
     - Minimum lateral distance for fitting.
   * - ``dipole.p_test_Cm``
     - ``float``
     - Test dipole moment (C·m).
   * - ``fit.drop_small_E0``
     - ``bool``
     - Exclude low-amplitude points from the fit.

``reconstruct_GF.yaml``
^^^^^^^^^^^^^^^^^^^^^^^^

Reconstruct the full dyadic Green's function from BEM data and effective
dipoles.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``parameters.energy_eV``
     - ``float``
     - Photon energy (eV).
   * - ``parameters.lambda_nm``
     - ``float``
     - Corresponding wavelength (nm).
   * - ``parameters.zD_nm``, ``zA_nm``
     - ``float``
     - Donor / acceptor heights (nm).
   * - ``io.xlsx_path``
     - ``str``
     - BEM Excel export.
   * - ``io.peff_path``
     - ``str``
     - Effective-dipole CSV from ``compute_peff``.
   * - ``io.output_file``
     - ``str``
     - Output HDF5 with reconstructed Green's function.

``compare*.yaml``
^^^^^^^^^^^^^^^^^^

Comparison and validation configs (``compare.yaml``,
``compare_silver.yaml``, ``compare_bem_dyadic.yaml``,
``compare_enhancement.yaml``) share a common pattern:

- **paths / io** — Input files (BEM Excel, Fresnel HDF5, effective-dipole CSV).
- **dipoles / dipole_frequency** — Orientation and transition frequency.
- **test** — Numerical comparison tolerances (``rtol``, ``atol``,
  ``enabled``).
- **plot** — Full Matplotlib configuration: ``figsize``, ``dpi``,
  ``rcParams``, series definitions with labels, colours, and line styles.


.. _config-plots:

``plots/`` — Observable Plotting
----------------------------------

**Files:** ``msd.yaml``, ``sqrt_msd.yaml``, ``pr.yaml``, ``ipr.yaml``

Used by ``mqed_plot_msd``, ``mqed_plot_sqrt_msd``, ``mqed_plot_pr``, and
``mqed_plot_ipr`` (:doc:`/tutorials/plotting`).

All four configs follow the same pattern.

``curves``
^^^^^^^^^^

A list of datasets to overlay. Each entry:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``label``
     - ``str``
     - Legend label.
   * - ``use_latest_glob``
     - ``str``
     - Glob pattern to find the input HDF5 file (most recent match is
       used).
   * - ``style``
     - ``str``
     - Matplotlib line style (e.g. ``"-"``, ``"--"``).
   * - ``lw``
     - ``float``
     - Line width.
   * - ``color``
     - ``str``
     - Line colour.

``plot_settings``
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Type
     - Description
   * - ``save_plot``
     - ``bool``
     - Write figure to disk.
   * - ``dpi``
     - ``int``
     - Figure resolution.
   * - ``figsize``
     - ``list``
     - ``[width, height]`` in inches.
   * - ``x_range_ps``
     - ``list``
     - ``[min, max]`` time range (ps).
   * - ``xlabel``, ``ylabel``, ``title``
     - ``str``
     - Axis and title labels.
   * - ``xscale``, ``yscale``
     - ``str``
     - ``"linear"`` or ``"log"``.
   * - ``xlim``, ``ylim``
     - ``list``
     - Axis limits.
   * - ``shade_std``
     - ``bool``
     - If ``true``, plot shaded standard-deviation band (for disorder
       averaging).
   * - ``x_scale_factor``
     - ``float``
     - Multiply x-axis values (e.g. for unit conversion).
   * - ``font``
     - ``dict``
     - Font settings: ``family``, ``labelsize``, ``ticksize``,
       ``legendsize``, ``titlesize``.
   * - ``y_sci``
     - ``dict``
     - Scientific notation formatting for the y-axis
       (``enabled``, ``scilimits``, ``use_math_text``).
