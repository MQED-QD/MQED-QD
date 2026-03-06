<div align="center">

# MQED-QD

[![python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](#)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.x-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![docs](https://img.shields.io/badge/Docs-GitHub_Pages-blue?logo=github)](https://mqed-qd.github.io/Macroscopic-Quantum-Electrodynamics/index.html)

A Python toolkit for simulating exciton transport
near plasmonic interfaces using macroscopic quantum electrodynamics (MQED).

</div>

---

## Features

| Category | Capability |
|----------|------------|
| **Green's functions** | Dyadic Green's functions via Sommerfeld integrals (planar) and BEM (arbitrary nanostructures) |
| **Energy transfer** | Resonance energy transfer (RET) and field enhancement (FE) analysis |
| **Quantum dynamics** | Lindblad master equation and non-Hermitian Schr&ouml;dinger equation (NHSE) solvers |
| **Transport studies** | Disorder sweeps for orientation-averaged transport; MSD, RMSD, IPR, and participation ratio |
| **Reproducibility** | Hydra YAML configs with automatic output versioning and bundled example data |

---

## Installation

```bash
git clone https://github.com/MQED-transport/Macroscopic-Quantum-Electrodynamics.git
cd Macroscopic-Quantum-Electrodynamics
conda env create -f environment.yaml
conda activate mqed
pip install -e .
```

<details>
<summary><strong>MPI support (optional)</strong></summary>

If you need MPI parallelism for large-scale BEM or dynamics runs, install
`mpi4py` after activating the environment:

```bash
conda install -c conda-forge mpi4py openmpi
```

</details>

---

## Quick Start

```bash
# 1. Compute dyadic Green's function for an Ag planar interface
mqed_GF_Sommerfeld simulation.energy_eV=1.864

# 2. Run non-Hermitian dynamics
mqed_nhse

# 3. Plot mean-squared displacement
mqed_plot_msd
```

Override any parameter from the CLI:

```bash
mqed_nhse simulation.Nmol=50 simulation.d_nm=4.0
```

For step-by-step walkthroughs, see the
**[Tutorials](https://mqed-qd.github.io/Macroscopic-Quantum-Electrodynamics/tutorials/index.html)**.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `mqed_GF_Sommerfeld` | Dyadic Green's function (planar geometry) |
| `mqed_RET` | Resonance energy transfer analysis |
| `mqed_FE` | Field enhancement analysis |
| `mqed_lindblad` | Lindblad master-equation dynamics |
| `mqed_nhse` | Non-Hermitian dynamics (recommended for large systems) |
| `mqed_plot_msd` | Plot mean-squared displacement |
| `mqed_plot_sqrt_msd` | Plot root-mean-squared displacement |
| `mqed_plot_IPR` | Plot inverse participation ratio |
| `mqed_plot_PR` | Plot participation ratio |
| `mqed_BEM_compute_peff` | BEM effective dipole moment |
| `mqed_BEM_reconstruct_GF` | Reconstruct Green's function from BEM |
| `mqed_BEM_compare_silver` | Compare BEM vs Fresnel (Ag) |

---

## Configuration

All commands use [Hydra](https://hydra.cc/) YAML configs under `configs/`.

| Config directory | Purpose |
|------------------|---------|
| `configs/Dyadic_GF/` | Geometry, materials, frequency grids |
| `configs/Lindblad/` | Quantum dynamics solver parameters |
| `configs/analysis/` | RET and FE analysis settings |
| `configs/BEM/` | BEM geometry and comparison settings |
| `configs/plots/` | MSD, RMSD, PR, IPR plot settings |

See the [Configuration Reference](https://mqed-qd.github.io/Macroscopic-Quantum-Electrodynamics/configuration.html) for full documentation.

---

## Project Layout

```
Macroscopic-Quantum-Electrodynamics/
├── configs/           # Hydra YAML configurations
├── data/
│   └── example/       # Bundled example data (tracked in git)
│       ├── GF_data/   # Pre-computed Green's function caches
│       └── QD_data/   # Pre-computed quantum dynamics results
├── mqed/              # Package source
│   ├── Dyadic_GF/     # Fresnel / Sommerfeld integrals
│   ├── Lindblad/      # Lindblad & NHSE solvers
│   ├── analysis/      # RET, FE calculations
│   ├── plotting/      # MSD, IPR, RMSD, PR plots
│   ├── BEM/           # Boundary element method
│   └── utils/         # Shared helpers (units, HDF5 I/O, logging)
├── docs/              # Sphinx documentation source
├── test/              # Pytest test suite
└── environment.yaml   # Conda environment specification
```

---

## Documentation

**Full documentation:**
[https://mqed-transport.github.io/Macroscopic-Quantum-Electrodynamics/](https://mqed-qd.github.io/Macroscopic-Quantum-Electrodynamics/index.html)

Build locally:

```bash
cd docs && make html
open build/html/index.html        # macOS
# xdg-open build/html/index.html  # Linux
```

---

## Citation

If you use MQED-QD in your research, please cite:

```bibtex
@article{liu2025mqed,
  title  = {MQED-QD: ...},
  author = {Liu, Guangming and ...},
  year   = {2025},
  doi    = {10.xxxx/...}
}
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Support

- **Issues:** [GitHub Issues](https://github.com/MQED-transport/Macroscopic-Quantum-Electrodynamics/issues)
- **Contact:** gliu8@nd.edu
