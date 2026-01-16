<div align="center">

# MQEDTP

[![python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python\&logoColor=white)](#)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.x-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-TBD-lightgrey)](#)

A Python package for Macroscopic QED simulations (Dyadic Green’s functions, RET and FE analysis, and open‑system dynamics via Lindblad / NHSE) studying exciton-polariton and plasmon-polariton under the dielectric environment, with Hydra-based configuration and small CLI wrappers for common workflows.

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
* Resonance energy transfer (RET) anf field enhancement (FE) analysis.
* Lindblad and non‑Hermitian skin effect (NHSE) dynamics.
* Disorder sweeps (single process or MPI, will be implemented in the future).
* Plotting utilities for MSD and root-MSD.
* Reproducible runs via Hydra configs and on‑disk caching.

---

## Installation

### Clone

```bash
# clone project
git clone https://github.com/MQED-transport/Macroscopic-Quantum-Electrodynamics.git
cd MacroscopicQED
```

### Conda/Mamba/Micromamba (recommended)

```bash
# choose one of: conda | mamba | micromamba
conda env create -f environmental.yaml   # create environment
conda activate mqed                      # activate environment
pip install -e .                         # install as editable package
```

### Pip only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                         # install package
```

<!-- ### MPI notes (optional)

To use the MPI‑based disorder sweeps you need an MPI implementation and `mpi4py` inside the same Python env.

```bash
# Install an MPI runtime first (OpenMPI or MPICH), then:
pip install mpi4py
# quick check
mpirun --version
``` -->

> **Tip:** If system MPI headers are unavailable on your machine, install via your package manager (e.g. `brew install open-mpi`, `apt install libopenmpi-dev openmpi-bin`).

---

## Quick Start

### Console commands

This package installs the following command‑line tools (from `setup.py` entry points):

| Command              | What it does                                       |
| -------------------- | ---------------------------------------------------|
| `mqed_GF_Sommerfeld` | Run Dyadic Green’s function simulation on planar.  |
| `mqed_RET`           | Run RET analysis                                   |
| `mqed_FE`            | Run Field Enhancement analysis                     |
| `mqed_lindblad`      | Time evolution with Lindblad dynamics              |
| `mqed_nhse`          | Time evolution with NHSE model                     |
<!-- | `mqed_nhse_disorder` | Disorder sweep;                                    | -->
| `mqed_plot_msd`      | Plot mean‑squared displacement from results        |
| `mqed_plot_sqrt_msd` | Plot square‑root MSD from results                  |

> All commands are configured via Hydra using YAML files under `configs/`. You can edit those files or override any key from the CLI.

### Examples

Run a **Dyadic Green's function simulation on planar surface** with defaults:

```bash
mqed_GF_Sommerfeld
```

Override parameters inline (Hydra style):

```bash
mqed_GF_Sommerfeld simulation.energy_eV=1.864
```
If you want to simulate multiple frequencies, you can choose List or Dict input:
```bash
mqed_GF_Sommerfeld simulation.energy_eV.min=1.0 simulation.energy_eV.max=2.0 simulation.energy_eV.points=11
# This will simulate 11 energy sources betweeen (1.0, 1.1, 1.2, ... ,2.0 )eV
```
or:
```bash
mqed_GF_Sommerfeld simulation.energy_eV=[1.0,1.5,2.0]
#This will simulate 3 energy points as (1.0,1.5,2.0) eV
```
You can also change other simulation parameters in the `configs/Dyadic_GF/GF_Sommerfeld.yaml`:
```bash
    position:
        zD: 2.0e-9 # The height of donor at z-axis
        zD_nm: 2 # Key for name
        zA: 2.0e-9 # The height of acceptor at z-axis, default same height with donor in simulation.
        Rx_nm:  # This section defines the range of horizontal distances between donor and acceptor
        start: 1.0
        stop: 500.0
        points: 501  # This will give you points 1, 2, 3, ..., 500, total 501 points.
output:
    filename: "result_${simulation.material}_${simulation.position.zD_nm}_nm.hdf5" 
    #Or your own file name.
```
<!-- The output file will be saved to outputs/Dyadic_GF_analytical/%%Year-Month-Day/%%Hour-Min-S/.. -->
After simulation, either in terminal or in the `outputs/Dyadic_GF_Sommerfeld/.../Dyadic_GF_Sommerfeld.log` you will see:
```bash
2025-10-24 11:40:02.818 | SUCCESS  | mqed.Dyadic_GF.main:run_simulation:114 - Simulation complete. Output saved to: /.../MacroscopicQED/outputs/Dyadic_GF_Sommerfeld/Y-M-D/H-M-S/YOUR_NAME.hdf5

```
For post-process (Simulate QED or RET), the default path of Green's function is: `/data/GF_cache/result_Ag_2_nm_latest.hdf5`

So you can create a subdirectory `data/GF_cache/` and copy-paste the hdf5 file from the path `/.../MacroscopicQED/outputs/Dyadic_GF_Sommerfeld/Y-M-D/H-M-S/YOUR_NAME.hdf5`.

**Lindblad dynamics:**

```bash
mqed_lindblad simulation.t_ps.start=0.0 simulation.t_ps.stop=150.0 simulation.t_ps.output_step=2e-3
```
You can also change the config files directly in `configs/Lindblad/quantum_dynamics.yaml` :
```bash
greens:
  h5_path: ${oc.env:MQED_ROOT,${oc.env:PWD}}/data/GF_cache/YOUR_NAME.hdf5 # update as needed


# Simulation controls
simulation:
# time grid in picoseconds
  t_ps:
    start: 0.0
    stop: 150.0
    output_step: 5e-3


# system
  Nmol: 100 # number of sites
  d_nm: 3.0 # lattice spacing (nm)


# dipoles / orientation
  mu_D_debye: 3.8
  mu_A_debye: 3.8
  theta_deg: 90.0
  phi_deg: magic # or a number
  mode: stationary # or 'disorder'
```
The program will read Green's function data from `/data/GF_cache/YOUR_NAME.hdf5` for calculating dipole-dipole interaction matrix. However, you can also overrite the path for the parameter you are interested by command line:
```bash
mqed_lindblad greens.h5_path=YOUR_PATH
```
Here the path directly comes from the absolute path after simulation `/YOUR_PATH/YOUR_NAME.hdf5` as mentioned in Dyadic Green's function simulation. Or you can manually overwrite the yaml file in `configs/Lindblad/quantum_dynamics.yaml` file.

**NHSE dynamics:**
The equivalent **Non-Hermitian Schodinger equation(NHSE)** is implemented here which gives identical result with Lindblad dynamic as we tested. **We recommend NHSE for large simulation** since it is **much faster** than simulate density matrix in general.
You can overwrite the simulation parameter as:
```bash
mqed_nhse simulation.t_ps.start=0.0 simulation.t_ps.stop=100.0 simulation.t_ps.output_step=2e-3
```
You can also change the config files directly in `configs/Lindblad/quantum_dynamics.yaml`.
```bash
simulation:
# time grid in picoseconds
  t_ps:
    start: 0.0
    stop: 150.0
    output_step: 5e-3


# system
  Nmol: 100 # number of sites
  d_nm: 3.0 # lattice spacing (nm)


# dipoles / orientation
  mu_D_debye: 3.8
  mu_A_debye: 3.8
  theta_deg: 90.0
  phi_deg: magic # or a number
  mode: stationary # or 'disorder'
  # disorder_sigma_phi_deg: 15.0 # uncomment for disordered orientations
```

Disorder sweep (multi process):

```bash
mqed_nhse_disorder simulation.disorder_sigma_phi_deg=8.0 initial_state.site_index=51
# Give the std of azimuthal angle as 8.0 and the initial excitation at the middle of 100 molecules.
```

Disorder sweep with MPI (8 ranks):
Not test yet.
<!-- ```bash
mpirun -n 8 mqed_nhse_disorder disorder.n_samples=400
``` -->

Plot MSD and RMSD:

```bash
mqed_plot_msd 
mqed_plot_sqrt_msd 
```
The `configs/plots/sqrt_msd.yaml` and `configs/plots/msd.yaml` are configure file for **root mean square displacement(RMSD)** and **mean square displacement(MSD)**:
```bash
curves:
  - label: "Magic-Angle"
    use_latest_glob: "${oc.env:MQED_ROOT,${oc.env:PWD}}/data/QDyn_cache/YOUR_NAME.hdf5"  
    style: "-"         # matplotlib line format
    lw: 1.5
  - label: "σ=10"
    use_latest_glob: "${oc.env:MQED_ROOT,${oc.env:PWD}}/data/QDyn_cache/YOUR_NAME.hdf5" 
    style: "-"         # e.g., "C1-"
    lw: 1.5
```
This part gives the curve you want to plot, `use_latest_glob` gives the path of the file, `label` gives the label in the final plot, `lw` is the line-width parameter from plot, `style` is the line style. The users can change the parameter as they prefer, see: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html for line style. 

If the users want to plot single/multiple lines, they can delete/add from the template. 
```bash
curves:
  - label: "YOUR LABEL"
    use_latest_glob: "YOUR PATH"  
    style: "YOUR STYLE"         # matplotlib line format
    lw: 1.5
# This one is single plot

```bash
curves:
  - label: "YOUR LABEL"
    use_latest_glob: "YOUR PATH"  
    style: "-"         # matplotlib line format
    lw: 1.5
  - label: "YOUR LABEL"
    use_latest_glob: "YOUR PATH" 
    style: "-"         # e.g., "C1-"
    lw: 1.5
  - label: "YOUR LABEL"
    use_latest_glob: "YOUR PATH" 
    style: "YOUR STYLE"         # e.g., "C1-"
    lw: 1.5
    #This one is for multiple curves plot
```

The plot settings can also be changed by users' preferences: (See matplotlib documentation for instruction: https://matplotlib.org/stable/users/index)
```bash
plot_settings:
  save_plot: true
  filename: "YOUR_NAME.png"
  dpi: 400
  show: false
  tight_layout: true
  grid: true
  legend: true
  figsize: [7.5, 5.0]

  # x selection (time). Choose one: x_range_ps or x_index_range
  x_range_ps: [0.0, 150.0]   # in picoseconds
  # x_index_range: [0, 500]

  # axis labels, scales, limits
  xlabel: "t (ps)"            # or "t (s)" if you keep in s
  ylabel: "$\\Delta x$ (nm)"
  title: "Silver; Middle Excitation"
  xscale: linear             # or "log"
  yscale: linear
  xlim: null
  ylim: null

  # optional multiply time axis (e.g., to show ×10^-10 s visually):
  x_scale_factor: 1.0        # keep 1.0 if you already saved 't_ps' in seconds; use 1e-12 to convert ps→s
  lw: 1.5
```


---

## Configuration (Hydra)

* Base config directory: `configs/`
* Notable groups:

  * `configs/Lindblad/` — quantum dynamics solver settings, user can change simulation method (`Lindblad`,`NonHermitian`), evaluation time steps, intermolecular distance(`d_nm`), dipole orientations(`theta_deg`, `phi_deg`, dipole strength of donor and acceptor.). The stationary orientation simulation is controlled by 'quantum_dynamics.yaml' and disorder-orientation simulation is controlled by 'quantum_dynamics_disorder.yaml'. 

  -Warning: The number of molecules (`Nmol`) times the intermolecular distance (`d_nm`) has to be less than the Dyadic Green's function simulation of horizental distance, i.e, if user's Dyadic Green's function simulates the horizental distance (`Rx_nm`) as 501 points starting from 0.0 to 500nm, (which is `Rx_nm.start=0.0, Rx_nm.stop=500.0, Rx_nm.points=501`) but user inputs 100 molecules(`Nmol=100`) of distance 6nm(`d_nm=6.0`), the program will **abort** since 100*6=600>= 500 (in Dyadic Green's function).
  * `configs/Dyadic_GF/` — geometry/material settings for Dyadic Green's function simulation on planar surface. User can change the dipole source frequency as **single value**, **dictionary of a range of values** or **list of some values** for multi-frequency simulation. The position is referred to **z-axis height** of donor and acceptor on the surface. `Rx_nm` is the *horizental distance* between donor and acceptor, i.e, source and response. For the material of user's interest, user may fit a dielectric function by him- or herself and append to the current excel sheet or create new .xlsx file to overwrite the default path and sheet name for the simulation.
  * `configs/analysis/` — RET and plotting parameters. User can change the parameter of donor and acceptor to evaluate the system of interest. Plot-setting parts can be customized by user's preference with the reference of matplotlib.
  * `configs/plots/`: `msd.yaml`, `sqrt_msd.yaml` - The two plot settings for the **mean square displacement (MSD)** or **root mean square displacement (RMSD)**. User can change the part of `curves` to customize the curves of interest with corresponding labels and data path. The plot settings can also be customized by user's preference.

Override any key from the CLI:

```bash
mqed_lindblad +experiment=my_note simulation.t_ps.stop=10.0 simulation.t_ps.output_step=1e-3
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
│  ├─ BEM/                  # Run Boundary Element Method(BEM) for electric field simulation.
│  └─ utils/
├─ outputs/                 # run outputs (created at runtime)
├─ environmental.yaml       # environment specification
├─ pyproject.toml           # build metadata
└─ setup.py                 # entry points and packaging
```

---

## Troubleshooting

* **Command not found** after install: make sure the env is activated and `pip install -e .` completed without errors.
* **MPI errors**: verify `mpirun` exists on PATH and `mpi4py` is installed in the active env.
* **Missing config**: ensure the specified YAML exists under `configs/` or list available options in that folder.
* **Plot scripts**: check that `curves` points to a completed run directory containing the expected logs/data.

---

## Beta Test:
* **Prerequisite:** Install the package as introduced in **Installation**.
* **Step1:** After install the package, run `mqed_GF_Sommerfeld simulation.energy_eV=1.864` in the ternimal, it will generate `result_Ag_2nm.hdf5` file under subdirectory `outputs/Dyadic_GF_Sommerfeld/Y-M-D/H-M-S/`. Create a new subdirectory named `data/GF_cache` under the root directory (See project layout), copy-paste the hdf5 file into `data/GF_cache/` and **rename it** as `Frensel_Ag_2_nm_665nm.hdf5`, which means simulation of donor on the height 2nm of silver planar surface and emitting 665nm photon.
* **Step2:** Run `mqed_FE` in the terminal, it will generate `enhancement_magic_angle_1.864eV.png` file under subdirectory `outputs/FE/Y-M-D/H-M-S/`. This result is the enhancement electric field of dipole emitting energy with value **1.864eV** with the azimuthal angle of both donor and acceptor is magic-angle(**arcos(1/sqrt(3))**). The X-axis is the horizental distance between donor and acceptor. You should get same result as `enhancement_magic_angle_1.864eV.png` under subdirectory `Beta_Test/`.
Here is the reference plot for this step:
  <p align="center">
    <img src="Beta_Test/enhancement_magic_angle_1.864eV.png" alt="Reference result for Step 3" width="500">
  </p>

<!-- * **Step3:** Run `mqed_nhse initial_state.site_index=51` in the terminal, it will generate `silver_stationary_latest.hdf5` file under subdirectory `outputs/Lindblad/Y-M-D/H-M-S/`. This result represents the quantum dynamics of molecular aggregate aligned on the silver surface with their azimuthal angle at magic angle. Copy-paste the`silver_stationary_latest.hdf5` file into subdirectory `data/QDyn_cache/`(**create this subdirectory first**) without rename the file.

* **Step4** Run multiple different commands:

```bash
mqed_nhse_disorder simulation.disorder_sigma_phi_deg=10
mqed_nhse_disorder simulation.disorder_sigma_phi_deg=30
mqed_nhse_disorder simulation.disorder_sigma_phi_deg=50
```

After those commands, there will be multiple files name as `silver_sigma${disorder_sigma_phi_deg}_avg_latest.hdf5` (The **disorder_sigma_phi_deg** is the value of your input, i.e, **3,10,30,50**)generated under the subdirectories `outputs/NHSE/Y-M-D/H-M-S/`. The file path will show up in the terminal or you can find them under the file `outputs/NHSE/Y-M-D/H-M-S/NHSE_disorder.log`. Copy-paste those `silver_sigma${disorder_sigma_phi_deg}_avg_latest.hdf5` into subdirectory `data/QDyn_cache/`.
* **Step5:** After **Step4**, you should have `silver_stationary_latest.hdf5` and multiple `silver_sigma${disorder_sigma_phi_deg}_avg_latest.hdf5` files under the `data/QDyn_cache/`. Make sure you have following `curves` part in the `configs/plots/sqrt_msd.yaml`:

```bash
curves:
  - label: "Magic-Angle"
    use_latest_glob: "${oc.env:MQED_ROOT,${oc.env:PWD}}/data/QDyn_cache/silver_stationary_latest.hdf5"  # or using "outputs/Lindblad/.../qdyn_result.hdft"
    style: "-"         # matplotlib line format
    lw: 1.5
  - label: "σ=10"
    use_latest_glob: "${oc.env:MQED_ROOT,${oc.env:PWD}}/data/QDyn_cache/silver_sigma10_avg_latest.hdf5" # or using "outputs/NHSE_disorder/.../qdyn_disorder_avg.hdft"
    style: "-"         # e.g., "C1-"
    lw: 1.5
  - label: "σ=30"
    use_latest_glob: "${oc.env:MQED_ROOT,${oc.env:PWD}}/data/QDyn_cache/silver_sigma30_avg_latest.hdf5" # or using "outputs/NHSE_disorder/.../qdyn_disorder_avg.hdft"
    style: "-"         # e.g., "C1-"
    lw: 1.5
  - label: "σ=50"
    use_latest_glob: "${oc.env:MQED_ROOT,${oc.env:PWD}}/data/QDyn_cache/silver_sigma50_avg_latest.hdf5" # or using "outputs/NHSE_disorder/.../qdyn_disorder_avg.hdft"
    style: "-"         # e.g., "C1-"
    lw: 1.5
```

Run `mqed_plot_sqrt_msd` and you should get a file `silver_middle_sqrt_msd.png` under the subdirectory `outputs/plot_sqrt_msd/Y-M-D/H-M-S/`. You should get the same result under the `Beta_Test/silver_middle_sqrt_msd.png`.

Here is the reference of the figure:
  <p align="center">
    <img src="Beta_Test/silver_middle_sqrt_msd.png" alt="Reference result for Step 3" width="500">
  </p> -->

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
