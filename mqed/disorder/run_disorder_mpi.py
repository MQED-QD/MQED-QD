from __future__ import annotations
from pathlib import Path
from typing import Dict, Union
import numpy as np
import h5py
from loguru import logger

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from mpi4py import MPI

# your imports
from mqed.Lindblad.quantum_operator import msd_operator, position_operator
from mqed.Lindblad.quantum_dynamics import SimulationConfig, NonHermitianSchDynamics
from mqed.utils.dgf_data import load_gf_h5
from qutip import Qobj, fock

def _time_grid(cfg) -> np.ndarray:
    t0, t1, dt = cfg.simulation.t_ps.start, cfg.simulation.t_ps.stop, cfg.simulation.t_ps.output_step
    return np.arange(t0, t1+dt, dt)

def _build_X_ops(dim: int, d_nm: float, Nmol: int, init_site_index: int)-> Dict[str, Qobj]:
    Xs = position_operator(dim, d_nm, Nmol, init_site_index)
    return {"X_shift": Xs, "X_shift2": Xs**2}

def _dx_from_expect(expect):
    ex1 = np.asarray(expect["X_shift"])
    ex2 = np.asarray(expect["X_shift2"])
    return np.sqrt(np.maximum(0.0, ex2 - ex1**2))

def _run_one(seed: int, 
             base_cfg: DictConfig, 
             G_slice: np.ndarray, 
             energy_eV: Union[float, np.ndarray],
             Rx_nm: np.ndarray, 
             tlist) -> np.ndarray:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

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
        mode="disorder",
    )

    dyn = NonHermitianSchDynamics(sim_cfg, GreensFunction=G_slice, seed=seed)
    psi0 = fock(sim_cfg.Nmol + 1, cfg.initial_state.site_index)
    e_ops = _build_X_ops(sim_cfg.Nmol + 1, sim_cfg.d_nm, sim_cfg.Nmol, cfg.initial_state.site_index)
    res = dyn.evolve(psi0, e_ops=e_ops)
    return _dx_from_expect(res.expectations)  # shape (T,)

def _save_avg(outfile: Path, t_ps: np.ndarray, dx_mean: np.ndarray, dx_std: np.ndarray, cfg: DictConfig):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(outfile, "w") as f:
        f.create_dataset("t_ps", data=t_ps)
        f.create_dataset("dx_mean_nm", data=dx_mean)
        f.create_dataset("dx_std_nm", data=dx_std)
        # f.attrs["method"] = "NonHermitian"
        f.attrs["n_realizations"] = int(cfg.disorder.n_realizations)
        f.attrs["seed_base"] = int(cfg.disorder.seed)
        f.attrs["sigma_phi_deg"] = float(cfg.simulation.disorder_sigma_phi_deg)
    logger.success(f"Wrote ensemble average → {outfile}")

@hydra.main(config_path="../../configs/NHSE", config_name="quantum_dynamics_disorder", version_base=None)
def run_disorder_mpi(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Hydra run directory (rank 0 will write)
    outdir = Path(HydraConfig.get().runtime.output_dir)

    # Load GF *once* on rank 0, then broadcast arrays
    if rank == 0:
        data = load_gf_h5(cfg.greens.h5_path)
        G_all = data["G_total"]
        energies = data["energy_eV"]
        Rx_nm = data["Rx_nm"]
        idx = 0 if cfg.simulation.get("emitter_frequency_eV", None) is None \
              else int(np.argmin(np.abs(energies - cfg.simulation.emitter_frequency_eV)))
        G_slice = G_all[idx]
        energies = energies[idx]
    else:
        G_slice = None
        Rx_nm = None

    G_slice = comm.bcast(G_slice, root=0)
    Rx_nm   = comm.bcast(Rx_nm, root=0)

    tlist = _time_grid(cfg)
    n = int(cfg.disorder.n_realizations)
    base = int(cfg.disorder.seed)
    seeds = np.arange(base, base + n, dtype=int)

    # block distribution of seeds
    local_seeds = seeds[rank::size]

    # compute local sum and sumsq for mean/std
    local_sum = None
    local_sumsq = None
    for s in local_seeds:
        dx = _run_one(int(s), cfg, G_slice, energies, Rx_nm, tlist)  # (T,)
        if local_sum is None:
            local_sum   = dx.copy()
            local_sumsq = dx*dx
        else:
            local_sum   += dx
            local_sumsq += dx*dx

    # make sure arrays exist even if local_seeds is empty
    if local_sum is None:
        # Create zeros shaped like a sample dx
        sample = _run_one(int(seeds[0]), cfg, G_slice, Rx_nm, tlist)
        local_sum   = np.zeros_like(sample)
        local_sumsq = np.zeros_like(sample)

    global_sum   = np.zeros_like(local_sum)
    global_sumsq = np.zeros_like(local_sumsq)
    comm.Reduce(local_sum,   global_sum,   op=MPI.SUM, root=0)
    comm.Reduce(local_sumsq, global_sumsq, op=MPI.SUM, root=0)

    # also reduce counts (seeds per rank can differ by <=1)
    local_count = np.array([len(local_seeds)], dtype=np.int64)
    global_count = np.array([0], dtype=np.int64)
    comm.Reduce(local_count, global_count, op=MPI.SUM, root=0)

    if rank == 0:
        N = int(global_count[0])
        mean = global_sum / N
        var  = np.maximum(0.0, global_sumsq / N - mean**2)
        std  = np.sqrt(var)
        _save_avg(outdir / cfg.output.filename, tlist, mean, std, cfg)

if __name__ == "__main__":
    run_disorder_mpi()
