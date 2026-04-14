"""Nearest-neighbour 1D chain dynamics with diagonal + off-diagonal disorder.

Physics
-------
Tight-binding Hamiltonian on N sites with nearest-neighbour coupling:
.. math::
    H_{nn'} = \epsilon_n \delta_{nn'} + J_n \delta_{n,n'+1} + J_n^* \delta_{n,n'-1}

Disorder is Gaussian:
.. math::
    \epsilon_n = \epsilon_0 + \delta\epsilon_n,    \delta\epsilon_n ~ N(0, \sigma_{\epsilon}^2)
    J_n   = J_0   + \delta J_n,    \delta J_n ~ N(0, \sigma_J^2)

MSD analytical solution for off-diagonal disorder for local excitation:
.. math::
    \langle x^2(t) \\rangle = 2 a^2 \frac{J_0^2 + \delta J_n^2}{\hbar^2} t^2

For Gaussian wave excitation with wavevector k_parallel:
.. math::
    c_n(0) = N e^{i k_{parallel} n a} \exp(-\frac{(n - n_0)^2}{2 \omega_{0}^2})
MSD analytical solution for off-diagonal disorder for Gaussian wave excitation:
.. math::
    \langle x^2(t) \\rangle = a^2 (\frac{\omega_{0}^2}{2} +  \frac{2 \sigma_J^2}{\hbar^2} t^2)

This program is used to verify if our analytical formula for MSD with off-diagonal disorder matches the numerical solution.
"""

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import h5py
from matplotlib import pyplot as plt

current_dir = Path(__file__).parent.resolve()

print(f"Current directory: {current_dir}")

project_root = current_dir.parent.parent
print(f"Project root: {project_root}")
# breakpoint()

root_path = os.path.dirname(os.path.abspath(__file__))
# plane_wave_numerical_data_path = os.path.join(root_path, "NN_cache/nn_chain_sigma_eps0.0_sigma_J0.05_avg.hdf5")

USE_PLANE_WAVE_PHASE = True  # set to False to test the local excitation formula instead
if USE_PLANE_WAVE_PHASE:
    print("Comparing with Gaussian wave excitation analytical formula.")
    numerical_data_path = os.path.join(root_path, "NN_cache/nn_chain_sigma_eps0.0_sigma_J0.01_avg_rlz_3000.hdf5")
else:    
    print("Comparing with local excitation analytical formula.")
    numerical_data_path = os.path.join(root_path, "NN_cache/nn_chain_sigma_eps0.0_sigma_J0.05_local_avg.hdf5")
# ---- load numerical data for comparison ----
with h5py.File(numerical_data_path, "r") as f:
    # scalar parameters are stored as file-level attributes
    J_0_eV = float(f.attrs["J_0_eV"])
    sigma_J_eV = float(f.attrs["sigma_J_eV"])
    eps_0_eV = float(f.attrs["eps_0_eV"])
    sigma_eps_eV = float(f.attrs["sigma_eps_eV"])
    t_total_fs = float(f.attrs["t_total_fs"])
    n_steps = int(f.attrs["n_steps"])
    k_parallel = float(f.attrs["k_parallel"])
    sigma_sites = float(f.attrs["sigma_sites"])
    # time axis is a top-level dataset in picoseconds
    t_array = np.array(f["t_ps"][:])
    # breakpoint()
    # MSD lives inside the "expectations" group
    msd_mean = np.array(f["expectations/msd_mean"][:])
    position_mean = np.array(f["expectations/position_mean"][:])


def msd_analytical_formula_offdiag_local_excitation(a, hbar, J_0_eV, sigma_J_eV, t_fs):
    """Analytical formula for MSD with off-diagonal disorder for local excitation. 
    <x>=0 for local excitation, so only the second moment contributes."""
    J_eff_squared = J_0_eV**2 + sigma_J_eV**2
    prefactor = 2 * a**2 * J_eff_squared / hbar**2
    return prefactor * t_fs**2

def x_square_analytical_formula_offdiag_gaussian_wave_excitation(a, hbar, J_0_eV, sigma_J_eV, t_fs, k_parallel, omega_0):
    """Analytical formula for second moment of position with 
    off-diagonal disorder for Gaussian wave excitation."""
    term1 = omega_0**2 / 2
    term2 = (4 * J_0_eV**2 / hbar**2) * np.sin(k_parallel * a)**2
    term3 = (2 * sigma_J_eV**2) / hbar**2
    prefactor = a**2 * (term1 + ( term2 + term3) * t_fs**2)
    return prefactor

def msd_analytical_formula_offdiag_gaussian_wave_excitation(a, hbar, J_0_eV, sigma_J_eV, t_fs, k_parallel, omega_0):
    """Analytical formula for MSD with off-diagonal disorder for Gaussian wave excitation.

    MSD = <(x-x0)^2> = x2 (second moment of displacement from initial site).
    Note: this is NOT the variance <(x-x0)^2> - <x-x0>^2.
    """
    x_square = x_square_analytical_formula_offdiag_gaussian_wave_excitation(a, hbar, J_0_eV, sigma_J_eV, t_fs, k_parallel, omega_0)
    # MSD is the full second moment x2, not x2 - <x>^2 (which is variance).
    return x_square

def position_analytical_formula_offdiag_gaussian_wave_excitation(a, hbar, J_0_eV, t_fs, k_parallel):
    """Analytical formula for mean position with off-diagonal disorder for Gaussian wave excitation."""
    term1 = (2 * -J_0_eV / hbar) * np.sin(k_parallel * a)
    prefactor = a * term1 * t_fs
    return prefactor

def RMSD_analytical_formula_offdiag_gaussian_wave_excitation(a, hbar, J_0_eV, sigma_J_eV, t_fs, k_parallel, omega_0):
    """Analytical formula for root mean square displacement (RMSD) with
    off-diagonal disorder for Gaussian wave excitation.

    RMSD = sqrt(MSD) = sqrt(<(x-x0)^2>) = sqrt(x2).
    Note: this is NOT sqrt(variance) = sqrt(x2 - <x>^2).
    """
    x_square = x_square_analytical_formula_offdiag_gaussian_wave_excitation(a, hbar, J_0_eV, sigma_J_eV, t_fs, k_parallel, omega_0)
    # RMSD = sqrt(MSD) = sqrt(x2), not sqrt(x2 - <x>^2) (which is std dev).
    return np.sqrt(np.maximum(0.0, x_square))

def RMSD_analytical_formula_offdiag_local_excitation(a, hbar, J_0_eV, sigma_J_eV, t_fs):
    """Analytical formula for root mean square displacement (RMSD) with off-diagonal disorder for local excitation."""
    msd = msd_analytical_formula_offdiag_local_excitation(a, hbar, J_0_eV, sigma_J_eV, t_fs)
    return np.sqrt(msd)

def compare_x_square_with_analytical_gaussian_wave_excitation(t_fs, msd_numerical, a, hbar, J_0_eV, sigma_J_eV, k_parallel, omega_0):
    """Compare numerical MSD with analytical formula."""
    if USE_PLANE_WAVE_PHASE:
        x_square_analytical = x_square_analytical_formula_offdiag_gaussian_wave_excitation(
            a=a,
            hbar=hbar,
            J_0_eV=J_0_eV,
            sigma_J_eV=sigma_J_eV,
            t_fs=t_fs,
            k_parallel=k_parallel,
            omega_0=omega_0
        )
        
    else:
        x_square_analytical = msd_analytical_formula_offdiag_local_excitation(
            a=a,
            hbar=hbar,
            J_0_eV=J_0_eV,
            sigma_J_eV=sigma_J_eV,
            t_fs=t_fs
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(t_fs, msd_numerical, label="Numerical MSD", color="blue", marker="o", linestyle="--")
    ax.plot(t_fs, x_square_analytical, label="Analytical X^2", color="red", linestyle="-")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("$<X^2>$")
    # ax.set_xlim(0, 5)
    # ax.set_ylim(48, 54)
    ax.legend()
    if USE_PLANE_WAVE_PHASE:
        ax.set_title("Gaussian wave excitation: Numerical vs Analytical MSD")
    else:
        ax.set_title("Local excitation: Numerical vs Analytical MSD")
    plt.show()

def compare_position_with_analytical_gaussian_wave_excitation(t_fs, position_numerical, a, hbar, J_0_eV, sigma_J_eV, k_parallel, omega_0):
    """Compare numerical mean position with analytical formula."""
    position_analytical = position_analytical_formula_offdiag_gaussian_wave_excitation(
        a=a,
        hbar=hbar,
        J_0_eV=J_0_eV,
        sigma_J_eV=sigma_J_eV,
        t_fs=t_fs,
        k_parallel=k_parallel,
        omega_0=omega_0
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(t_fs, position_numerical, label="Numerical Mean Position", color="blue", marker="o", linestyle="--")
    ax.plot(t_fs, position_analytical, label="Analytical Mean Position", color="red", linestyle="-")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Mean Position (site index)")
    ax.legend()
    ax.set_title("Gaussian wave excitation: Numerical vs Analytical Mean Position")
    plt.show()

hbar_eVfs = 0.6582119514  # ℏ in eV·fs
a = 1.0                    # lattice constant (lattice units)
omega_0 = sigma_sites      # Gaussian wavepacket width σ_sites (lattice units)

t_fs = t_array* 1e3        # time axis in fs

# compare_x_square_with_analytical_gaussian_wave_excitation(
#     t_fs=t_fs,
#     msd_numerical=msd_mean,
#     a=a,
#     hbar=hbar_eVfs,
#     J_0_eV=J_0_eV,
#     sigma_J_eV=sigma_J_eV,
#     k_parallel=k_parallel,
#     omega_0=omega_0,
# )

compare_position_with_analytical_gaussian_wave_excitation(
    t_fs=t_fs,
    position_numerical=position_mean,
    a=a,
    hbar=hbar_eVfs,
    J_0_eV=J_0_eV,
    sigma_J_eV=sigma_J_eV,
    k_parallel=k_parallel,
    omega_0=omega_0,
)