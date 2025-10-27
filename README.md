<div align="center">

# MacroscopicQED

[![python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python\&logoColor=white)](#)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.x-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-TBD-lightgrey)](#)

A Python package for macroscopic QED simulations (Dyadic Green’s functions, RET analysis, and open‑system dynamics via Lindblad / NHSE), with Hydra-based configuration and small CLI wrappers for common workflows.

</div>

## Table of Contents

* [Features](#features)
* [Installation](#installation)

  * [Clone](#clone)
  * [Conda/Mamba/Micromamba (recommended)](#condamambamicromamba-recommended)
  * [Pip only](#pip-only)
  * [MPI notes (optional)](#mpi-notes-optional)
* [Quick Start](#quick-start)

  * [Console commands](#console-commands)
  * [Examples](#examples)
* [Configuration (Hydra)](#configuration-hydra)
* [Project Layout](#project-layout)
* [Troubleshooting](#troubleshooting)
* [Documentation](#documentation)
* [License](#license)
* [Third‑party notices](#third-party-notices)

---

## Features

* Dyadic Green’s function simulations.
* Resonance energy transfer (RET) analysis.
* Lindblad and non‑Hermitian skin effect (NHSE) dynamics.
* Disorder sweeps (single process or MPI).
* Plotting utilities for MSD and √MSD.
* Reproducible runs via Hydra configs and on‑disk caching.

---

## Installation

### Clone

```bash
# clone project
git clone <YOUR_REPO_URL>.git
cd MacroscopicQED
```

### Conda/Mamba/Micromamba (recommended)

```bash
# choose one of: conda | mamba | micromamba
mamba env create -f environmental.yaml   # create environment
mamba activate mqed                      # activate environment
pip install -e .                         # install as editable package
```

### Pip only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                         # install package
```

### MPI notes (optional)

To use the MPI‑based disorder sweeps you need an MPI implementation and `mpi4py` inside the same Python env.

```bash
# Install an MPI runtime first (OpenMPI or MPICH), then:
pip install mpi4py
# quick check
mpirun --version
```

> **Tip:** If system MPI headers are unavailable on your machine, install via your package manager (e.g. `brew install open-mpi`, `apt install libopenmpi-dev openmpi-bin`).

---

## Quick Start

### Console commands

This package installs the following command‑line tools (from `setup.py` entry points):

| Command              | What it does                                     |
| -------------------- | ------------------------------------------------ |
| `mqed_GF`            | Run Dyadic Green’s function simulation           |
| `mqed_RET`           | Run RET analysis                                 |
| `mqed_lindblad`      | Time evolution with Lindblad dynamics            |
| `mqed_nhse`          | Time evolution with NHSE model                   |
| `mqed_nhse_disorder` | Disorder sweep;                                  |
| `mqed_plot_msd`      | Plot mean‑squared displacement from results      |
| `mqed_plot_sqrt_msd` | Plot square‑root MSD from results                |

> All commands are configured via Hydra using YAML files under `configs/`. You can edit those files or override any key from the CLI, e.g. `t.max=1000`.

### Examples

Run a Dyadic GF simulation with defaults:

```bash
mqed_GF
```

Override parameters inline (Hydra style):

```bash
mqed_GF geometry=Dyadic_GF/example.yaml output_dir=outputs/gf_run1
```

Lindblad dynamics:

```bash
mqed_lindblad system=Lindblad/default.yaml t.max=1000 dt=0.1 save=true
```

NHSE dynamics:

```bash
mqed_nhse system=Lindblad/nhse_default.yaml t.max=500 save=true
```

Disorder sweep (single process):

```bash
mqed_nhse_disorder disorder.n_samples=50 seed=1234
```

Disorder sweep with MPI (8 ranks):

```bash
mpirun -n 8 mqed_nhse_disorder disorder.n_samples=400
```

Plot MSD and √MSD:

```bash
mqed_plot_msd input_dir=outputs/nhse_run1 save=true
mqed_plot_sqrt_msd input_dir=outputs/nhse_run1 save=true
```

---

## Configuration (Hydra)

* Base config directory: `configs/`
* Notable groups:

  * `configs/Lindblad/` — system and solver settings
  * `configs/Dyadic_GF/` — geometry/material settings
  * `configs/analysis/` — RET and plotting parameters
  * Top‑level: `msd.yaml`, `sqrt_msd.yaml` for plotting defaults

Override any key from the CLI:

```bash
mqed_lindblad +experiment=my_note t.max=2000 solver.method=expm save=true
```

Hydra organizes outputs under `outputs/` and keeps copies of the used configs for reproducibility. Heavy intermediates may be cached under `data/`.

---

## Project Layout

```
MacroscopicQED/
├─ configs/                 # Hydra configs (Dyadic_GF, Lindblad, analysis, plotting)
├─ data/                    # caches (e.g., GF_cache, QDyn_cache)
├─ mqed/                    # package source
│  ├─ Dyadic_GF/
│  ├─ Lindblad/
│  ├─ analysis/
│  ├─ plotting/
│  └─ utils/
├─ outputs/                 # run outputs (created at runtime)
├─ plots/                   # generated figures (created at runtime)
├─ environmental.yaml       # environment specification
├─ pyproject.toml           # build metadata
└─ setup.py                 # entry points and packaging
```

---

## Troubleshooting

* **Command not found** after install: make sure the env is activated and `pip install -e .` completed without errors.
* **MPI errors**: verify `mpirun` exists on PATH and `mpi4py` is installed in the active env.
* **Missing config**: ensure the specified YAML exists under `configs/` or list available options in that folder.
* **Plot scripts**: check that `input_dir` points to a completed run directory containing the expected logs/data.

---

## Documentation

Documentation build is not set up yet. When ready, add a `docs/` tree and wire a `Makefile` target:

```bash
make docs           # build docs
# make docs-clean   # optional clean rebuild
```

For now, this README is the primary user guide.

---

## License

**TBD.** Choose a license (MIT, BSD‑3‑Clause, or Apache‑2.0 are common). Add a `LICENSE` file at the repository root and update the badge above.

---

## Third‑party notices

At present this repository does **not** vendor code from third‑party projects. If you later copy/adapt external source files, list each project and include its license text here or in a `third_party_licenses/` folder. Depending only on libraries via pip/conda does not typically require reproducing their licenses here.
