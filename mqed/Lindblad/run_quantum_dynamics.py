import hydra
from omegaconf import  DictConfig
import numpy as np
import h5py
from loguru import logger
from pathlib import Path
from qutip import fock, fock_dm
from typing import Optional

from mqed.Lindblad.quantum_dynamics import SimulationConfig, LindbladDynamics, NonHermitianSchDynamics
from mqed.utils.dgf_data import load_gf_h5
from hydra.core.hydra_config import HydraConfig
from mqed.Lindblad.quantum_operator import msd_operator, excitation_population_operator

def save_qdyn_h5(outfile: Path, method: str, t_ps: np.ndarray, expectations: dict):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(outfile, 'w') as f:
        f.attrs['method'] = method
        f.create_dataset('t_ps', data=np.asarray(t_ps))


        # # Save states (density matrices or kets)
        # if states is not None and len(states) > 0:
        #     first = states[0]
        # if hasattr(first, 'isket') and first.isket:
        #     arr = np.stack([s.full().ravel() for s in states], axis=0)
        #     f.create_dataset('state_vectors_flat', data=arr)
        #     f.attrs['state_format'] = 'ket_flat'
        # else:
        #     arr = np.stack([s.full() for s in states], axis=0)
        #     f.create_dataset('density_matrices', data=arr)
        #     f.attrs['state_format'] = 'density_matrix'


        # Save expectations
        exp_grp = f.create_group('expectations')
        for name, val in expectations.items():
            exp_grp.create_dataset(name, data=np.asarray(val))


        logger.success(f"Saved quantum dynamics results → {outfile}")

def app_run(cfg:DictConfig, output_dir: Optional[Path]=None):
    if output_dir is None:
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- Starting quantum dynamics simulation---")
# 1) Load Green's function slice at the emitter energy
    data = load_gf_h5(cfg.greens.h5_path)   # {"G_total","G_vac","energy_eV","Rx_nm","zD","zA"}
    G_slice  = data["G_total"]             # (M,N,3,3)
    E_eV  = data["energy_eV"]            # (M,)
    Rx_nm = data["Rx_nm"] 

    G_slice = G_slice[0] # use the first energy slice for now


    # 2) Build SimulationConfig (unify with your abstractions)
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


    # 4) e_ops: mean-square displacement (and optional extras)
    msd_op = msd_operator(dim=sim_cfg.Nmol + 1,
        d_nm=sim_cfg.d_nm,
        Nmol=sim_cfg.Nmol,
        init_site_index=cfg.initial_state.site_index)
    
    excitation_population_ops = excitation_population_operator(dim =sim_cfg.Nmol + 1,
                                                            Nmol=sim_cfg.Nmol)


    e_ops = {'MSD_nm2': msd_op, 'Excitation_Populations': excitation_population_ops}


    # 5) Evolve
    result = dyn.evolve(rho_or_psi, e_ops=e_ops, options=None)


    # 6) Save to HDF5
    outfile = output_dir / cfg.output.filename
    # states = getattr(result, 'states', None)
    save_qdyn_h5(outfile, method=method, t_ps=result.tlist, expectations=result.expectations)


    logger.info("Done.")

@hydra.main(config_path="../../configs/Lindblad", config_name="quantum_dynamics", version_base=None)
def run_quantum_dynamics(cfg:DictConfig):
    app_run(cfg)

if __name__ == "__main__":
    run_quantum_dynamics()