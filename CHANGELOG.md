# Changelog

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
