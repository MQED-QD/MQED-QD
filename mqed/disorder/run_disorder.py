from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np

from loguru import logger

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from joblib import Parallel, delayed
from tqdm import tqdm

# your modules
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.Lindblad.quantum_operator import msd_operator, position_operator
from mqed.Lindblad.quantum_dynamics import (
    SimulationConfig,
    NonHermitianSchDynamics,
)
from mqed.utils.dgf_data import load_gf_h5  # you already call load_gf_h5 somewhere
from mqed.utils.save_hdf5 import save_dx_h5
from mqed.utils.joblib_track import tqdm_joblib

# ---------- helpers ----------

def _time_grid(cfg) -> np.ndarray:
    t0, t1, dt = cfg.simulation.t_ps.start, cfg.simulation.t_ps.stop, cfg.simulation.t_ps.output_step
    return np.arange(t0, t1+dt, dt)

def _dx_from_expects(expect: Dict[str, np.ndarray]) -> np.ndarray:
    ex1 = np.asarray(expect["X_shift"])
    ex2 = np.asarray(expect["X_shift2"])
    return np.sqrt(np.maximum(0.0, ex2 - ex1**2))

def _load_greens_slice(h5_path: str, energy_eV: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (G_slice, energy_eV_list, Rx_nm) at either closest energy to `energy_eV`
    or (if None) simply the 0th energy slice.
    """
    data = load_gf_h5(h5_path)  # -> {"G_total","energy_eV","Rx_nm",...}
    G_all = data["G_total"]        # (nE, K, 3,3)
    energies = data["energy_eV"]   # (nE,)
    rx_nm = data["Rx_nm"]          # (K,)
    # breakpoint()
    if energy_eV is None:
        idx = 0
    else:
        idx = int(np.argmin(np.abs(energies - energy_eV)))
    if G_all.ndim != 4 or G_all.shape[2:] != (3,3):
        raise ValueError(f"Unexpected G_all shape: {G_all.shape}, expected (nE, K, 3, 3)")
    logger.info(f"Using GF slice at E={energies[idx]:.3f} eV")
    return G_all[idx], energies[idx], rx_nm

def _run_one(seed: int, 
             base_cfg: DictConfig, 
             G_slice: np.ndarray,
             energy_eV: Union[float, np.ndarray],
             Rx_nm: np.ndarray, 
             tlist: np.ndarray) -> np.ndarray:
    # Deep-copy cfg for isolation
    # logger.info(f" starting realization {seed+1}")
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))


    # pack sim config (NHSE)
    sim_cfg = SimulationConfig(
        tlist=tlist,
        emitter_frequency= energy_eV if isinstance(energy_eV, float) else energy_eV[0],
        Nmol=cfg.simulation.Nmol,
        Rx_nm=Rx_nm,
        d_nm=cfg.simulation.d_nm,
        mu_D_debye=cfg.simulation.mu_D_debye,
        mu_A_debye=cfg.simulation.mu_A_debye,
        theta_deg=cfg.simulation.theta_deg,
        phi_deg=cfg.simulation.phi_deg,
        disorder_sigma_phi_deg=cfg.simulation.disorder_sigma_phi_deg,
        mode=cfg.simulation.mode,
    )

    # dynamics object
    dyn = NonHermitianSchDynamics(sim_cfg, GreensFunction= G_slice, seed= seed)

    # initial state |site_index>
    from qutip import fock
    psi0 = fock(sim_cfg.Nmol + 1, cfg.initial_state.site_index)

    # two expectations to compute Δx
    e_ops = {"X_shift": position_operator(sim_cfg.Nmol + 1, sim_cfg.d_nm, sim_cfg.Nmol, cfg.initial_state.site_index),
             "X_shift2": msd_operator(sim_cfg.Nmol + 1, sim_cfg.d_nm, sim_cfg.Nmol, cfg.initial_state.site_index)}

    res = dyn.evolve(psi0, e_ops=e_ops)
    dx = _dx_from_expects(res.expectations)  # (T,)
    return dx

# ---------- Hydra entrypoint ----------

@hydra.main(config_path="../../configs/Lindblad", config_name="quantum_dynamics_disorder", version_base=None)
def run_disorder(cfg: DictConfig) -> None:
    outdir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()

    logger.info("— NHSE disorder ensemble —")
    # data = load_gf_h5(cfg.greens.h5_path)
    G_slice, energies, Rx_nm = _load_greens_slice(cfg.greens.h5_path, cfg.simulation.get("emitter_frequency_eV", None))
    # breakpoint()
    tlist = _time_grid(cfg)

    n = int(cfg.disorder.n_realizations)
    base = int(cfg.disorder.seed)
    seeds = [base + i for i in range(n)]

    # # parallel across realizations
    n_jobs = int(cfg.disorder.n_jobs) if cfg.disorder.n_jobs is not None else -1
    logger.info(f"Running {n} realizations (n_jobs={n_jobs})")

    worker = delayed(_run_one)
    # tqdm will update when each job finishes
    with tqdm_joblib(
        tqdm(
            total=len(seeds),
            desc="NHSE disorder realizations",
        )
    ) as progress_bar:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            worker(s, cfg, G_slice, energies, Rx_nm, tlist)
            for s in seeds
        )
    # shape: (n, T)
    stack = np.stack(results, axis=0)
    dx_mean = np.mean(stack, axis=0)
    dx_std = np.std(stack, axis=0)
    # breakpoint()

    outfile = outdir / cfg.output.filename
    save_dx_h5(
    outfile=outfile,
    t_ps=tlist,
    dx_mean_nm=dx_mean,
    dx_std_nm=dx_std,
    method="NonHermitian",
    mode=cfg.simulation.mode,
    n_realizations=int(cfg.disorder.n_realizations),
    extra_attrs={
        "sigma_phi_deg": float(cfg.simulation.disorder_sigma_phi_deg),
        "seed_base": int(cfg.disorder.seed),
    },
    )
    logger.success(f"Simulation complete. Output saved to: {outfile.absolute()}")

if __name__ == "__main__":
    run_disorder()
