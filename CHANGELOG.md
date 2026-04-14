# Changelog

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
