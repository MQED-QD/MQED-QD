# A test program for the run_quantum_dynamics module
import numpy as np
import hydra

from mqed.Lindblad.run_quantum_dynamics import app_run
from mqed.utils.dgf_data import load_gf_h5
from mqed.Lindblad.ddi_matrix import build_ddi_matrix_from_Gslice
from mqed.utils.orientation import resolve_angle_deg, spherical_to_cartesian_dipole
from mqed.Lindblad.quantum_dynamics import SimulationConfig, LindbladDynamics, NonHermitianSchDynamics
from mqed.Lindblad.quantum_operator import msd_operator

from omegaconf import DictConfig

# Test the result of Lindblad and non-Hermitian dynamics for MQED setup
# The test is to ensure that both methods give consistent results for the same physical setup
def test_lindblad_vs_nonhermitian():
    # Load Green's function data:
    data = load_gf_h5('/home/guangmingliu/Documents/Notre_Dame/2025Fall/MacroscopicQED/outputs/Dyadic_GF_analytical/2025-10-13/11-29-07/result_Ag_2_nm.hdf5')   # {"G_total","G_vac","energy_eV","Rx_nm","zD","zA"}
    G_slice  = data["G_total"]             # (M,N,3,3)
    E_eV  = data["energy_eV"]            # (M,)
    Rx_nm = data["Rx_nm"]
    Z_D = data["zD"]

    # Build a simple simulation config
    sim_cfg = SimulationConfig(
        tlist=np.arange(0, 1, 5e-3),  # 0 to 1 ps
        emitter_frequency=E_eV[0],
        Nmol=30,
        Rx_nm=Rx_nm,
        d_nm=3.0,
        mu_D_debye=3.8,
        mu_A_debye= None,
        theta_deg=90.0,
        phi_deg='magic',
        mode = 'stationary',
    )

    # Initialize both dynamics solvers
    lindblad_dyn = LindbladDynamics(sim_cfg, G_slice[0])
    nonherm_dyn = NonHermitianSchDynamics(sim_cfg, G_slice[0])

    # Initial state: excitation on the first molecule
    from qutip import fock_dm, fock
    rho_init = fock_dm(sim_cfg.Nmol + 1, 1)  # Density matrix for Lindblad
    psi_init = fock(sim_cfg.Nmol + 1, 1)     # State vector for Non-Hermitian

    # Evolve both systems
    e_ops = {'MSD_nm2': msd_operator(dim=sim_cfg.Nmol + 1,
                                    d_nm=sim_cfg.d_nm,
                                    Nmol=sim_cfg.Nmol,
                                    init_site_index=1)}
    
    options = {'atol': 1e-9, 'rtol': 1e-6}

    lindblad_result = lindblad_dyn.evolve(rho_init, e_ops=e_ops, options=options)
    nonherm_result = nonherm_dyn.evolve(psi_init, e_ops=e_ops, options=options)

    # Compare the mean-square displacement results
    msd_lindblad = lindblad_result.expectations['MSD_nm2']
    msd_nonherm = nonherm_result.expectations['MSD_nm2']

    # Assert that the results are close
    assert np.allclose(msd_lindblad, msd_nonherm, atol=1e-2), "Lindblad and Non-Hermitian results differ significantly!"

