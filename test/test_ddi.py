'''
This is the script to test our Fresnel implementation generated matrix element in agreement with 
previous MATLAB generated matrix element (which is benchmark). Test results confirm our implementation is correct.
'''
import numpy as np
import h5py
from pathlib import Path
from scipy.io import loadmat, savemat
import pytest
from matplotlib import pyplot as plt
from scipy.stats import norm
import os

from mqed.utils.SI_unit import eV_to_J, c, eps0, hbar,D2CMM
from mqed.Lindblad.ddi_matrix import build_ddi_matrix_from_Gslice, _phi_wrapped_normal_deg
from mqed.utils.dgf_data import load_gf_h5
from mqed.utils.orientation import resolve_angle_deg, spherical_to_cartesian_dipole

dir_path = os.path.dirname(os.path.abspath(__file__))
# Substitute with your own path.
dgf_data_path = os.path.join(dir_path, 'GF_Sommerfeld_data/Fresnel_GF_planar_Ag_height_2nm_665nm.hdf5')
matlab_path = os.path.join(dir_path,'matlab_data/Parameter_Set1.mat')
matlab_disorder_matrix = os.path.join(dir_path,'matlab_data/Parameter_Set2.mat')
matlab_angle_disorder = os.path.join(dir_path,'matlab_data/Angle_Set2.mat')

data = load_gf_h5(dgf_data_path)   # {"G_total","G_vac","energy_eV","Rx_nm","zD","zA"}
Gtot  = data["G_total"]             # (M,N,3,3)
E_eV  = data["energy_eV"]            # (M,)
Rx_nm = data["Rx_nm"] 
N_mol = 100                # (N,) --- IGNORE ---
d_nm = 3
mu_d_debye = 3.8

phi_donor   = resolve_angle_deg('magic')
phi_acceptor  = resolve_angle_deg('magic')
theta_donor = float(90.0)
theta_acceptor= float(90.0)
# Convert angles to Cartesian vectors
p_donor = spherical_to_cartesian_dipole(theta_donor,
                                        phi_donor)
p_acceptor = spherical_to_cartesian_dipole(theta_acceptor,
                                        phi_acceptor)

# breakpoint()

Gamma_ab_matlab = np.array(loadmat(matlab_path)['Gamma_ab'])  # (N_mol,N_mol)
V_ab_matlab = np.array(loadmat(matlab_path)['Vab'])            # (

Gamma_ab_matlab_disorder = np.array(loadmat(matlab_disorder_matrix)['Gamma_ab'])
V_ab_matlab_disorder = np.array(loadmat(matlab_disorder_matrix)['Vab'])
phi_matlab = np.array(loadmat(matlab_angle_disorder)['wrapped_angles'])

def test_stationary():
    """
    Test the data from implementation with data from the Matlab.
    The data from Matlab is the benchmark.
    """
    V_ab_test, Gamma_ab_test = build_ddi_matrix_from_Gslice(
        G_slice= Gtot[0],
        Rx_nm = Rx_nm,
        energy_emitter= E_eV[0],
        N_mol= N_mol,
        d_nm=d_nm,
        mu_D_debye= mu_d_debye,
        uA= p_acceptor,
        uD= p_donor
    )

    assert V_ab_test.shape == (N_mol, N_mol)
    assert Gamma_ab_test.shape == (N_mol, N_mol)
    assert np.allclose(V_ab_test, V_ab_matlab), "V_ab matrix does not match MATLAB benchmark"
    assert np.allclose(Gamma_ab_test, Gamma_ab_matlab), "Gamma_ab matrix does not match MATLAB benchmark"

def plot_disorder():
    phi = np.rad2deg(np.arccos(1/np.sqrt(3)))

    sigma = 8.0

    wrapped_angles = _phi_wrapped_normal_deg(N_mol, phi, sigma, seed=None)

    plt.figure(figsize=(8,6))
    plt.hist(
    wrapped_angles,
    bins=200,                # like MATLAB's "70"
    density=True,           # "Normalization","pdf"
    range=(0, 360),         # xlim([0,360])
    )

    x_pdf = np.linspace(-100, 460, 500)
    y_pdf = norm.pdf(x_pdf, loc=phi, scale=sigma)
    plt.plot(x_pdf, y_pdf, "r--", lw=1.5, label="Original Normal PDF")


    plt.title("Wrapped Normal Distribution of Angles (0–360 degrees)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Probability Density")
    plt.xlim(0, 360)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def test_vectorize():
    # scalar
    v = spherical_to_cartesian_dipole(90.0, 0.0)
    assert v.shape == (3,)
    # array phi, scalar theta
    phi = np.array([0, 90, 180, 270])
    U = spherical_to_cartesian_dipole(90.0, phi)
    assert U.shape == (4,3)
    # known directions
    assert np.allclose(U[0], [1,0,0], atol=1e-12)
    assert np.allclose(U[1], [0,1,0], atol=1e-12)
    assert np.allclose(U[2], [-1,0,0], atol=1e-12)
    assert np.allclose(U[3], [0,-1,0], atol=1e-12)

def test_disorder_matrix():
    theta = 90.0
    pos_donor = spherical_to_cartesian_dipole(theta,phi_matlab)
    pos_acceptor = spherical_to_cartesian_dipole(theta, phi_matlab)

    V_ab_test, Gamma_ab_test = build_ddi_matrix_from_Gslice(
        G_slice= Gtot[0],
        Rx_nm = Rx_nm,
        energy_emitter= E_eV[0],
        N_mol= N_mol,
        d_nm=d_nm,
        mu_D_debye= mu_d_debye,
        U_list=pos_donor,
        mode = 'disorder',
    )

    assert V_ab_test.shape == (N_mol, N_mol)
    assert Gamma_ab_test.shape == (N_mol, N_mol)
    assert np.allclose(V_ab_test, V_ab_matlab_disorder), "V_ab matrix does not match MATLAB benchmark"
    assert np.allclose(Gamma_ab_test, Gamma_ab_matlab_disorder), "Gamma_ab matrix does not match MATLAB benchmark"

if __name__ == "__main__":
    # test_stationary()
    plot_disorder()
