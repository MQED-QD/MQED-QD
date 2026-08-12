# Changelog

## 1.4.3 - 2026-08-12

### Compact Mie ring-circulant workflows

- Added an opt-in `ring_circulant` Mie output layout for symmetry-compatible spherical emitter rings, storing one dipole-projected cyclic Green-function row instead of the full pair-indexed Cartesian tensor.
- Added strict projected-representation metadata, finite-value and shape validation, required emitter geometry provenance, and configurable memory guards for loading or expanding compact ring data.
- Extended emission-spectrum and spectral-density analysis to reconstruct scalar pair matrices from ring-circulant rows, and routed spectral-density plotting through the existing pair-index selection workflow.
- Added stationary Lindblad/DDI support for projected circulant couplings, including periodic ring-distance filtering, while explicitly rejecting unsupported reorientation and non-stationary workflows.

### Rx-aware N-layer MPI scheduling

- Improved N-layer MPI utilization for single- or few-frequency sweeps by distributing contiguous Rx chunks when energy-level scheduling cannot occupy all ranks.
- Preserved whole-energy batching for `fixed_grid` integration so sampled Sommerfeld kernels remain reusable across the complete Rx grid.
- Added strict root-side validation for worker indices, tensor shapes, complete coverage, duplicate work, and optional TE/TM component assembly without changing the saved HDF5 result layout.
- Added an annotated single-frequency DBP/DBR MPI example for a 0--120 nm, 121-point horizontal-distance sweep.

### Documentation and validation

- Added Mie ring-circulant and spectral-density tutorial guidance, example configuration, and expanded Sphinx API reference coverage for Mie, analysis, plotting, and emitter-geometry modules.
- Added regression coverage for compact Mie output, HDF5 validation, ring reconstruction, emission and spectral analysis, projected DDI construction, periodic coupling filters, and Rx-aware MPI scheduling and assembly.

## 1.4.2 - 2026-07-25

### Polarization-resolved N-layer Green tensors

- Added opt-in TE- and TM-resolved scattering tensors to the N-layer solver without duplicating Sommerfeld integrations.
- Added optional `green_function_structure`, `green_function_scattering_te`, and `green_function_scattering_tm` HDF5 datasets while preserving the legacy total/vacuum-only schema by default.
- Enforced `structure = TE + TM` and `total = vacuum + structure` before saving polarized output, with consistent support across sequential, Joblib, MPI, direct, and fixed-grid execution paths.
- Added support for sources and observers in the semi-infinite top exterior while retaining the complete multilayer reflection response below the source region.

### Effective collective-emission analysis

- Extended the Varguet-effective convention to separation-indexed linear chains, using structure-only self terms and adding the real vacuum tensor only for nonzero separations.
- Added a renormalized-total convention that keeps structure-only self terms and the full complex total Green tensor off diagonal.
- Added explicit full, TE, and TM structure-channel selection with validation and provenance; TE/TM spectra are independent channel calculations rather than additive intensities.
- Added strict chain-size, spacing, separation-grid, tensor-shape, finite-value, and metadata validation.
- Updated emission maps to place transition energy on the horizontal axis and emission energy on the vertical axis.

### Verified sphere-ring workflow

- Added reusable sphere-ring geometry generation with configurable emitter count and orientation.
- Persisted Mie pair-layout emitter positions and orientations so emission analysis consumes the exact simulated geometry.
- Added and verified ring-oriented HPC and emission-analysis configurations, including out-of-plane molecular orientations.

### Configuration, privacy, and validation

- Added automatic local Hydra overlays and ignored `local/` at repository level so personal or unpublished configurations remain private while retaining normal `--config-name` usage.
- Documented the optional polarized HDF5 schema and top-exterior N-layer coordinate convention.
- Added comprehensive regression coverage for polarization decomposition, integration reuse, HDF5 compatibility, chain reconstruction, effective conventions, sphere-ring geometry, plotting axes, and configuration composition.
- Verified the complete test suite with **187 passing tests** before release preparation.

## 1.4.1 - 2026-07-24

### DBP-in-DBR Green tensors

- Corrected DBP source-layer selection for the N-layer DBR workflow so the emitter remains in the intended zero-based layer index.
- Added validation that rejects non-finite Green tensors before invalid values can be stored or passed to downstream analysis.
- Stabilized off-center N-layer Sommerfeld kernels against algebraic high-q overflow while retaining the established on-axis behavior.

### Physical DBR coupling analysis

- Added `mqed_plot_dbr_couplings` to project separation-indexed Green tensors onto independent donor and acceptor orientations and plot the physical coherent coupling $V_{ij}$ and dissipative energy $\hbar\Gamma_{ij}$.
- Added total, vacuum, structure, and scattered Green-component selection, with stored structure tensors preferred and a total-minus-vacuum fallback.
- Added ordered `Rx_nm` selection by source indices or physical values so one output can combine near-field samples with sparse far-field DBR separations.
- Added strict tolerance-based matching by default, optional nearest-grid matching, and provenance for requested values, selected values, source indices, and matching deltas.
- Added signed and absolute coupling arrays, decay rates in inverse seconds, and HDF5, CSV, and PNG outputs without vacuum-normalized enhancement ratios.

### Documentation and validation

- Documented the DBR coupling command, Hydra configuration, physical formulas, separation selection, and output artifacts in the README and Sphinx documentation.
- Added regression coverage for coupling signs and units, Green-component selection, separation-only layout validation, non-finite input rejection, output provenance, nested output paths, and disjoint near-/far-field selection.
- Verified the complete test suite, installed Hydra command, and synthetic near-/far-field CLI workflow.

## 1.4.0 - 2026-07-12

### Scan-layout spectral-density analysis

- Added scan-layout support to `mqed_calc_spec_dens`, enabling fixed-source Mie scan HDF5 files to produce spectral-density curves for explicit observer positions.
- Preserved scan metadata in spectral-density outputs, including source positions, observer positions, and observer distances in nanometers.
- Added scan-layout plotting support to `mqed_plot_spec_dens`, including curve selection by scan distance, scan indices, scan labels, and per-curve scale factors.

### N-layer comparison examples and plotting configs

- Renamed the spectral-density plotting configs into method-comparison workflows for direct-vs-singularity-aware and singularity-aware-vs-pole-aware hybrid DCIM plots.
- Added bundled N-layer Green-tensor and spectral-density reference HDF5 files for sparse 0, 12, and 120 nm separations.
- Added reference figures for N-layer spectral-density comparisons and the Mie spherical-cavity scan example.

### Tutorials and theory documentation

- Added focused tutorials for N-layer planar stacks and Mie core-shell spherical cavities, including command-line workflows, HDF5 schema notes, spectral-density handoff, and HPC launcher guidance.
- Updated the spectral-density tutorial for separation, scan, and pair layouts, including single-file plotting overrides and comparison-plot configs.
- Added concise theory pages for N-layer Sommerfeld integrals and Mie Green's functions, with notes on singularity-aware pole extraction and hybrid DCIM acceleration.
- Updated the getting-started and tutorial index pages so users can find the N-layer, Mie, and spectral-density workflows directly.

### Tests and validation

- Added regression tests for scan-layout spectral-density calculation, scan metadata persistence, scan plotting, and Hydra plot-output configuration.
- Verified changed Python files with `py_compile`, focused pytest coverage, plot CLI smoke tests, and a Sphinx HTML documentation build.

## 1.3.2 - 2026-07-10

### Mie Green's-function HDF5 workflow

- Moved Mie scan and pair output writing onto the shared `mqed.utils.dgf_data` helpers, removing the duplicate local `save_hdf5` and `save_pair_hdf5` implementations from `main_mie.py`.
- Added a shared scan-indexed HDF5 layout for fixed-source, explicit-observer Green tensors, including canonical datasets, compatibility aliases, Mie structure terms, wavelength metadata, projected scalars, Purcell factors, and explicit source/observer position datasets.
- Extended the shared loader to read scan-indexed Mie files while preserving pair and separation layout support.

### Core-shell cavity configuration and launch scripts

- Updated `configs/Dyadic_GF/GF_Mie.yaml` with detailed annotations for scan vs pair layouts, source/observer semantics, segmented energy grids, supported core-region scans, and the `[0, 2, 20]` nm positions used for the current core-shell cavity spectral-density reproduction.
- Changed Mie output paths to use the N-layer-style YAML prefix plus `.hdf5` parameter suffixes.
- Added `mqed/Dyadic_GF/gf_mie_single_job.sh`, an annotated SGE/MPI launcher for many-frequency Mie jobs.

### Tests and validation

- Updated Mie output tests to cover the shared scan schema, loader compatibility, and generated `.hdf5` filename convention.
- Verified the annotated Mie config parses to a scan layout with 122 segmented energy points and observer positions at 0, 2, and 20 nm.

## 1.3.1 - 2026-06-30

### N-layer pole-aware integration

- Added `pole_subtracted_direct`, a Stage-1 Bessel-form pole-subtraction reference path that subtracts detected simple-pole residue models from the real-axis Sommerfeld kernels, integrates the smooth remainder and pole model separately, then validates against `singularity_aware` with mixed absolute/relative tolerances.
- Added `pole_aware_hybrid_dcim`, a Stage-2 hybrid path that fits only the pole-subtracted smooth high-q tail with DCIM, adds the pole model back in the same Bessel convention, and falls back to `singularity_aware` when validation rejects the approximation.
- Added diagnostics/report fields for pole-aware methods, including whether the returned result is the approximation or the `singularity_aware` fallback.

### DCIM safety and q-window controls

- Routed `Rx = 0` calculations for DCIM-family methods (`dcim`, `hybrid_dcim`, `branch_cut_dcim`, and `pole_aware_hybrid_dcim`) through `singularity_aware`, avoiding Hankel/DCIM singular behavior for local LDOS/Purcell-style calculations.
- Added dimensionless `|k0|`-scaled q-window factors for DCIM and hybrid DCIM ranges while preserving the previous absolute SI q cutoffs for backward compatibility.
- Expanded `configs/Dyadic_GF/GF_five_layer_example_multi_freq.yaml` with method-choice annotations, q-window unit guidance, pole-search range comments, and warnings for experimental branch-cut/pole-aware methods.

### Spectral-density plotting

- Added multi-file spectral-density comparison plotting, matching the `plot_msd.py` / `plot_pr.py` style of a `curves` list with `path` or `use_latest_glob`.
- Added file-level and per-selected separation/pair styling, so comparison plots can use color for the input file or numerical method and linestyle/marker for individual `Rx` values or emitter pairs.
- Updated spectral-density plot YAML files with tutorial-style examples for direct, `singularity_aware`, and pole-aware/hybrid comparison plots.

### Tests and validation

- Added regression tests for pole-subtracted direct integration, pole-aware hybrid fallback/reporting, `Rx = 0` DCIM routing, q-window factor conversion, multi-file spectral-density plotting, and per-separation/per-pair style precedence.

## 1.3.0 - 2026-06-29

### N-layer singularity diagnostics

- Added `mqed.Dyadic_GF.sommerfeld_singularities` with argument-principle pole search, residue extraction, and vertical-wavenumber branch-cut sampling/integration helpers.
- Extended `NLayerGreenFunction` with `singularity_aware` quadrature, pole diagnostics, branch-cut diagnostics, and an experimental `branch_cut_dcim` mode that validates against `singularity_aware` and falls back by default when the fitted branch/pole decomposition is not accurate enough.
- Added Hydra pass-through options for pole search, branch-cut diagnostics, and `branch_cut_dcim` validation controls in the N-layer runner.

### Experimental Mie and emission-spectrum workflows

- Added an experimental Mie Green's-function workflow (`GF_Mie.py`, `main_mie.py`, and `mqed_GF_Mie`) with example Hydra configs for spherical geometries.
- Added experimental emission-spectrum calculation and plotting entry points with Hydra configs.
- Important: `GF_Mie.py` and `emission_spectrum.py` have not yet been verified against literature results. They are included for development/testing and should not be used as validated production workflows until benchmark comparisons are completed.

### Pair-layout analysis and plotting

- Added pair-indexed Green tensor handling for Lindblad/DDI construction and downstream spectral-density plotting.
- Added spectral-density plot selection by pair separation values and configurable pair labels.

### Tests and validation

- Added regression tests for pole search, residue extraction, branch-cut diagnostics, `singularity_aware`, `branch_cut_dcim` validation/fallback, pair-layout dynamics, Mie output shape handling, emission-spectrum calculation, and pair spectral-density plotting.

## 1.2.1 - 2026-06-12

### N-layer Green's-function workflow

- Added flexible Rx-grid handling for both Sommerfeld and N-layer Green's-function CLIs. Legacy `{start, stop, points}` configs still work, while scalar, list, `{values: [...]}`, `{min, max, points}`, and segmented grids can now preserve sparse physical separations in HDF5.
- Added an opt-in `fixed_grid` N-layer integration mode that samples the Bessel-free Sommerfeld kernels once on a finite q grid and reuses them across many Rx values. The existing `direct`, `dcim`, and `hybrid_dcim` paths remain unchanged.
- Restored the default `mqed_GF_NLayer` config name with `configs/Dyadic_GF/GF_NLayer_five_layer.yaml`, pointing to the bundled five-layer example.

### Downstream DDI and plotting

- Updated the DDI matrix builder to resolve required separations `0, d, 2d, ...` with a strict floating-point tolerance, so sparse Rx grids match equivalent dense grids without exact-float lookup failures.
- Added spectral-density plotting by physical separation values through `plot_settings.separation_values_nm`, while preserving the previous `separation_indices` behavior.
- Added plot-time spectral-density unit selection with `plot_settings.spectral_density_unit`, supporting the stored eV values and SI `s^-1` display for literature comparison.
- Added `plot_settings.y_sci` to spectral-density plots so large SI y-axis values can use scientific notation formatting like the MSD plotter.

### Tests and validation

- Added regression coverage showing sparse Rx grids produce the same DDI matrices as equivalent dense grids.
- Verified the N-layer CLI, fixed-grid sparse-Rx smoke path, spectral-density calculation, value-selected plotting, SI-unit plotting, scientific y-axis formatting, and the full pytest suite.

## 1.2.0 - 2026-06-11

### New features

- Added N-layer dyadic Green's-function support for finite planar stacks through
  `mqed.Dyadic_GF.GF_NLayer`, including recursive N-layer Fresnel reflection,
  same-layer source/observer amplitudes, and Sommerfeld kernels for total and
  vacuum Green's-function output.
- Added the `mqed_GF_NLayer` Hydra CLI and example five-layer Ag/spacer configs
  for single-frequency and multi-frequency layered-media calculations.
- Added DCIM utilities in `mqed.Dyadic_GF.dcim` and a conservative hybrid
  direct/DCIM integration path for testing accelerated Sommerfeld tails.
- Added MPI execution support to the N-layer CLI so large energy grids can be
  distributed across HPC ranks, plus an SGE single-job launcher for cluster runs.

### Documentation

- Added API reference pages for `mqed.Dyadic_GF.GF_NLayer` and
  `mqed.Dyadic_GF.dcim`.
- Updated the README and documentation landing page to describe the N-layer
  Green's-function workflow and the new `mqed_GF_NLayer` command.

### Tests and validation

- Added pytest coverage for N-layer Green's-function construction and numerical
  behavior.
- Verified the N-layer CLI with sequential, MPI smoke, and Sphinx documentation
  builds during release preparation.

## 1.1.4 - 2026-05-23

### Documentation

- Expanded the Sommerfeld dyadic Green's-function tutorial with two complete,
  reproducible examples: a single-frequency molecular-aggregate setup and a
  multi-frequency Drude-model sweep for spectral-density calculations.
- Added a literature-reproduction spectral-density tutorial that walks from the
  bundled Green's-function HDF5 data through spectral-density calculation and
  plotting of a Figure-2C-style comparison for Chuang *et al.*
- Added clearer guidance for MPI/HPC execution, custom Hydra configs, dielectric
  model selection, segmented energy grids, output HDF5 contents, and downstream
  handoff between Green's-function and spectral-density workflows.

### Examples and reproducibility

- Added shared example configs for the Sommerfeld single-frequency and
  multi-frequency workflows under `configs/Dyadic_GF/`.
- Added spectral-density analysis and plotting example configs under
  `configs/analysis/` and `configs/plots/`.
- Bundled precomputed Green's-function and spectral-density HDF5 example data
  under `data/example/` so users can reproduce the spectral-density tutorial
  without first running a long Green's-function sweep.

### Maintenance

- Fixed RST formatting and Sphinx rendering issues in the new tutorial pages so
  the documentation builds cleanly.

## 1.1.3 - 2026-04-17

### New features

- Added configurable dielectric-source models in
  `mqed.Dyadic_GF.data_provider.DataProvider` with YAML-driven `source_type`
  support for `excel`, `constant`, `Drude`, and `Drude-Lorentz`.
- Added model-parameter parsing for both `*_eV` and `*_rad_s` keys, plus
  oscillator support for Drude-Lorentz fits.
- Added nonuniform segmented spectral grids in `build_grid` for Sommerfeld and
  shared BEM workflows (`segments: [{min, max, points}, ...]`).
- Added spectral-density overlay controls for per-curve multipliers, colors,
  and linestyles in `mqed.plotting.plot_spectral_density`.

### Documentation

- Updated `docs/source/tutorials/GF_Sommerfeld.rst` with segmented energy-sweep
  examples and dielectric-model usage (`Drude` / `Drude-Lorentz`).
- Expanded `configs/Dyadic_GF/GF_Sommerfeld.yaml` annotations to document model
  equations and parameter-unit conventions.
- Added local example configs under `local/configs/Dyadic_GF/` for nonuniform
  grids and Drude-Lorentz material setup.

### Tests

- Added `test/test_data_provider_models.py` to validate source-type selection,
  Drude/Drude-Lorentz formulas, unit-key validation, and error handling.
- Added tests for segmented grid handling and spectral-density plotting
  multipliers/styles in `test/test_build_grid.py` and
  `test/test_plot_spectral_density.py`.

## 1.1.2 - 2026-04-16

### New features

- Added local Hydra config discovery through `mqed.utils.hydra_local` so
  personal YAML files under `local/configs/<group>/` can be used with the same
  CLI `--config-name=...` workflow as shared configs.

### CLI and usability

- Updated all Hydra-powered console scripts to resolve shared and personal
  config trees together, covering plotting, BEM, Sommerfeld Green's functions,
  analysis, Lindblad dynamics, and disorder workflows.
- Restored terminal help for installed commands after the local-config wiring so
  users can inspect `--help`, `--hydra-help`, `--cfg`, and `--info` directly in
  the conda environment.

### Documentation

- Updated the README to document the `local/configs/` workflow, clarify that
  tutorials remain the main user guide, and add a short CLI-help reference for
  quick terminal discovery.

## 1.1.1 - 2026-04-15

### New features

- Update the way to name the dyadic Green's function (`mqed.Dyadic_GF.main`) 
and spectral density (`mqed.analysis.spectral_density`) so that the output 
file contains the info of height and multiple energy points during simulation.

### Bug fixes

- Fixed the previous bugs in the `mqed.Dyadic_GF.main`: it only took `dict` 
as input so that SGE script parameter was incompatible with previous program.
Now we added `DictConfig` solving this issue.
- Fixed the Lindblad collapse-operator positive-semidefinite validation in
  `mqed.Lindblad.quantum_dynamics` so that only numerically tiny negative
  eigenvalues are clipped, while genuinely invalid decay matrices still raise a
  `ValueError`.
- Fixed the initial-state site resolution in
  `mqed.Lindblad.run_quantum_dynamics` so that `center_site` correctly takes
  precedence when Gaussian initial states are configured through Hydra.

### Packaging and infrastructure

- Consolidated package metadata into `pyproject.toml`, kept `setup.py` as a
  lightweight compatibility shim, and aligned the package version metadata with
  `mqed.__version__`.
- Included Hydra YAML configuration files in built distributions so installed
  wheels retain the same CLI configuration support as editable source checkouts.
- Added GitHub Actions pytest CI for Python 3.10 and 3.11 to exercise the test
  suite automatically on code, config, packaging, and workflow changes.
- Tightened dependency bounds in `environment.yaml` and
  `environment_windows.yaml` to better match the validated packaging stack.

## 1.1.0 - 2026-04-14

### New features

- Added spectral density analysis module (`mqed.analysis.spectral_density`) for
  computing and visualising the photonic spectral density from dyadic Green's
  function data.
- Added multi-frequency simulation support for the Sommerfeld dyadic Green's
  function with MPI and Joblib parallel backends.
- Added SGE job-array script and TSV parameter file for batch Sommerfeld
  Green's function sweeps on HPC clusters
  (`mqed/Dyadic_GF/gf_sommerfeld_jobarray.sh`).
- Added four-dimensional `(N, N, 3, 3)` storage format for dyadic Green's
  function data.

### Bug fixes

- Fixed MSD definition across the codebase: the previous formula `<x²> − <x>²`
  computed the *variance* of displacement, not the mean square displacement.
  MSD is now correctly computed as `<(x−x₀)²>` (the second moment alone).
  Affected files: `run_quantum_dynamics.py`, `plot_msd.py`,
  `plot_sqrt_msd.py`, `run_disorder.py`, `run_disorder_nn.py`, and
  `nn_compare_analytical.py`.  The variance is still available as a separate
  saved key (`variance_mean`) where applicable.

### Documentation

- Added a BEM nanorod tutorial (`docs/source/tutorials/BEM-Nanorod.rst`) with
  beginner-friendly annotations, a "What is BEM?" introduction, convergence
  testing guidance, and a troubleshooting section.
- Polished existing BEM tutorials for clarity and RST correctness.

### Compatibility notes

- MSD-related output keys now contain the true MSD.  Scripts that relied on the
  previous (variance) values should be updated.  A new `variance_mean` key is
  saved alongside `msd_mean` where both values are available.
- The dyadic Green's function storage format now supports four-dimensional
  arrays; older two-dimensional xlsx-based workflows are still supported.

## 1.0.0 - 2026-03-16

This release marks the first stable MQED-QD milestone. The main focus is a much
more complete BEM workflow, clearer transport observables, and more practical
plotting and disorder-simulation tooling.

### BEM highlights

- Added a full vacuum-calibration tutorial covering the MNPBEM setup,
  `mqed_BEM_compute_peff`, expected outputs, and configuration guidance.
- Added a full reconstruction tutorial for dyadic Green's functions from BEM
  field data, including planar validation and calibration-accuracy guidance.
- Added bundled BEM tutorial figures under `docs/source/_static/tutorials/bem/`
  to make the workflow easier to follow.
- Added API reference coverage for `mqed.BEM.accuracy_plot`.
- Improved the BEM validation workflow with the newer dyadic-comparison flow,
  verification script support, and related documentation updates.
- Renamed `Frensel` references to `Fresnel` in BEM resources and paths.

### Transport and plotting updates

- Clarified the distinction between position, second moment `<x^2>`, true MSD,
  and RMSD across the NN-disorder and Lindblad output pipeline.
- Updated MSD/RMSD plotting to compute the true MSD consistently from
  `<x^2> - <x>^2` when needed.
- Added optional analytical MSD/RMSD overlays for NN-chain comparisons.
- Added configurable time-axis handling in plotting (`fs`, `ps`, and `s`).
- Added examples and switches for analytical plotting in YAML configs.

### Disorder and simulation updates

- Added MPI support for NN-chain disorder averaging with YAML-controlled
  backend selection and rank-aware realization splitting.
- Added config support for controlling legacy aliases in saved output files.
- Improved naming and storage conventions for observables to reduce ambiguity in
  downstream analysis.

### Compatibility notes

- BEM resources and path names were standardized from `Frensel` to `Fresnel`.
- Transport outputs now distinguish position, second moment `<x^2>`, and true
  MSD more explicitly; downstream scripts that assumed the older naming may need
  small updates.
- Legacy aliases are still supported in output files, with config options to
  control whether they are saved.

### Documentation and configuration updates

- Expanded tutorial coverage for BEM workflows and refreshed tutorial index
  entries.
- Updated getting-started and configuration guidance to better explain Hydra
  config usage and custom YAML workflows.
- Added example configuration files for BEM reconstruction and plotting.

### Notes on versioning

- The project version moved from `0.1.1` to `1.0.0` to reflect a stable public
  workflow spanning Sommerfeld Green's functions, BEM reconstruction, quantum
  dynamics, disorder averaging, and plotting.
