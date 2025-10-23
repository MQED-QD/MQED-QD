import hydra
from omegaconf import  DictConfig
import numpy as np
import h5py
from loguru import logger
from pathlib import Path
from qutip import fock, fock_dm
from typing import Optional

from hydra import compose

from mqed.Lindblad.quantum_dynamics import SimulationConfig, LindbladDynamics, NonHermitianSchDynamics
from mqed.utils.dgf_data import load_gf_h5
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from hydra.core.hydra_config import HydraConfig
from mqed.Lindblad.quantum_operator import msd_operator, excitation_population_operator, position_operator
from mqed.utils.save_hdf5 import save_dx_h5

def app_run(cfg:DictConfig, output_dir: Optional[Path]=None):
    if output_dir is None:
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_loggers_hydra_aware()

    logger.info("--- Starting quantum dynamics simulation---")
# 1) Load Green's function slice at the emitter energy
    data = load_gf_h5(cfg.greens.h5_path)   # {"G_total","G_vac","energy_eV","Rx_nm","zD","zA"}
    G_slice  = data["G_total"]             # (M,N,3,3)
    E_eV  = data["energy_eV"]            # (M,)
    Rx_nm = data["Rx_nm"] 

    G_slice = G_slice[0] # use the first energy slice for now


    # 2) Build SimulationConfig (unify with your abstractions)
    # breakpoint()
    tlist = np.arange(cfg.simulation.t_ps.start, cfg.simulation.t_ps.stop, cfg.simulation.t_ps.output_step)


    sim_cfg = SimulationConfig(
        tlist=tlist,
        emitter_frequency=E_eV[0],
        Nmol=cfg.simulation.Nmol,
        Rx_nm=Rx_nm,
        d_nm=cfg.simulation.d_nm,
        mu_D_debye=cfg.simulation.mu_D_debye,
        mu_A_debye=cfg.simulation.mu_A_debye,
        theta_deg=cfg.simulation.theta_deg,
        phi_deg=cfg.simulation.phi_deg,
        disorder_sigma_phi_deg=cfg.simulation.get('disorder_sigma_phi_deg', None),
        mode=cfg.simulation.mode,
    )


    # 3) Choose dynamics backend
    method = cfg.solver.method
    if method == 'Lindblad':
        dyn = LindbladDynamics(sim_cfg, G_slice)
        # Lindblad expects density matrix
        rho_or_psi = fock_dm(sim_cfg.Nmol + 1, cfg.initial_state.site_index)
    elif method == 'NonHermitian':
        dyn = NonHermitianSchDynamics(sim_cfg, G_slice)
        # Non-Hermitian Sch. expects ket
        rho_or_psi = fock(sim_cfg.Nmol + 1, cfg.initial_state.site_index)
    else:
        raise ValueError(f"Unknown solver.method = {method}")


    
    excitation_population_ops = excitation_population_operator(dim =sim_cfg.Nmol + 1,
                                                            Nmol=sim_cfg.Nmol)


    e_ops = {"X_shift": position_operator(sim_cfg.Nmol + 1, sim_cfg.d_nm, sim_cfg.Nmol, cfg.initial_state.site_index),
            "X_shift2": msd_operator(sim_cfg.Nmol + 1, sim_cfg.d_nm, sim_cfg.Nmol, cfg.initial_state.site_index)}


    # 5) Evolve
    result = dyn.evolve(rho_or_psi, e_ops=e_ops, options=None)

    #calculate dx from expectations
    ex1 = np.asarray(result.expectations["X_shift"])
    ex2 = np.asarray(result.expectations["X_shift2"])
    dx = np.sqrt(np.maximum(0.0, ex2 - ex1**2))


    # 6) Save to HDF5
    outfile = output_dir / cfg.output.filename
    # states = getattr(result, 'states', None)
    save_dx_h5(
    outfile=outfile,
    t_ps=result.tlist,
    dx_mean_nm=dx,
    dx_std_nm=np.zeros_like(dx),      # single run → zero spread
    method=method,
    mode="stationary",
    n_realizations=1,
    expectations=result.expectations, # keeps raw expectations if you want
    )


    logger.success(f"Simulation complete. Output saved to: {outfile.absolute()}")

@hydra.main(config_path="../../configs/Lindblad", config_name="quantum_dynamics", version_base=None)
def mqed_lindblad(cfg:DictConfig):
    cfg.solver.method = "Lindblad"
    app_run(cfg)

@hydra.main(config_path="../../configs/Lindblad", config_name="quantum_dynamics", version_base=None)
def mqed_nhse(cfg:DictConfig):
    cfg.solver.method = "NonHermitian"
    app_run(cfg)

if __name__ == "__main__":
    mqed_nhse()