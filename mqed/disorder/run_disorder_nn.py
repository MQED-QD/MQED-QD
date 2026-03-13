"""Hydra CLI for disorder-averaged NN-chain quantum dynamics.

Usage
-----
    mqed_nn_disorder                                  # default config
    mqed_nn_disorder chain.N_emit=500 disorder.n_realizations=100
    mqed_nn_disorder disorder.sigma_eps_eV=0.01 disorder.sigma_J_eV=0.005

The ensemble-averaging pattern mirrors ``mqed.disorder.run_disorder`` (NHSE),
adapted for the lightweight NN-chain propagator.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from loguru import logger

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from joblib import Parallel, delayed
from tqdm import tqdm

from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.utils.save_hdf5 import save_dx_h5
from mqed.utils.joblib_track import tqdm_joblib

from mqed.disorder.nn_chain_dynamics import NNChainConfig, NNChainDynamics, NNChainResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_config(cfg: DictConfig) -> NNChainConfig:
    """Translate the Hydra DictConfig into a frozen NNChainConfig."""
    center = cfg.initial_state.get("center_site", None)
    return NNChainConfig(
        N_emit=int(cfg.chain.N_emit),
        eps_0_eV=float(cfg.chain.eps_0_eV),
        J_0_eV=float(cfg.chain.J_0_eV),
        sigma_eps_eV=float(cfg.disorder.sigma_eps_eV),
        sigma_J_eV=float(cfg.disorder.sigma_J_eV),
        t_total_fs=float(cfg.time.t_total_fs),
        n_steps=int(cfg.time.n_steps),
        initial_state_type=str(cfg.initial_state.type),
        sigma_sites=float(cfg.initial_state.sigma_sites),
        k_parallel=float(cfg.initial_state.k_parallel),
        center_site=int(center) if center is not None else None,
        obs_msd=bool(cfg.observables.msd),
        obs_populations=bool(cfg.observables.populations),
    )


def _run_one(seed: int, nn_cfg: NNChainConfig) -> NNChainResult:
    """Worker function: run a single disorder realisation.

    Args:
        seed: RNG seed for this realisation.
        nn_cfg: frozen chain configuration (safe across processes).

    Returns:
        :class:`NNChainResult` with the requested observables.
    """
    dyn = NNChainDynamics(nn_cfg)
    return dyn.evolve(seed=seed)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../../configs/disorder_nn",
    config_name="nn_chain",
    version_base=None,
)
def run_disorder_nn(cfg: DictConfig) -> None:
    """Run disorder-averaged NN-chain simulation."""

    outdir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()

    logger.info("— NN-chain disorder ensemble —")
    nn_cfg = _build_config(cfg)
    logger.info(
        f"Chain: N={nn_cfg.N_emit}, ε₀={nn_cfg.eps_0_eV} eV, "
        f"J₀={nn_cfg.J_0_eV} eV"
    )
    logger.info(
        f"Disorder: σ_ε={nn_cfg.sigma_eps_eV} eV, σ_J={nn_cfg.sigma_J_eV} eV"
    )
    logger.info(
        f"Time: {nn_cfg.t_total_fs} fs, {nn_cfg.n_steps} steps"
    )
    logger.info(
        f"Initial state: {nn_cfg.initial_state_type}"
    )

    # ---- seeds ----
    n = int(cfg.disorder.n_realizations)
    base_seed = int(cfg.disorder.seed)
    seeds = [base_seed + i for i in range(n)]

    # ---- parallel ensemble ----
    n_jobs = int(cfg.disorder.n_jobs) if cfg.disorder.n_jobs is not None else -1
    logger.info(f"Running {n} realisations (n_jobs={n_jobs})")

    worker = delayed(_run_one)
    with tqdm_joblib(
        tqdm(total=n, desc="NN-chain disorder realisations")
    ) as progress_bar:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            worker(s, nn_cfg) for s in seeds
        )
    assert results is not None, "Parallel returned None"
    results_typed = list(results)  # type: list[NNChainResult]

    # ---- aggregate ----
    t_fs = results_typed[0].t_fs
    # convert to ps for the save utility
    t_ps = t_fs * 1.0e-3

    # MSD
    msd_mean: Optional[np.ndarray] = None
    msd_std: Optional[np.ndarray] = None
    if nn_cfg.obs_msd:
        msd_stack = np.stack([r.msd for r in results_typed if r.msd is not None], axis=0)
        msd_mean = np.mean(msd_stack, axis=0)
        msd_std = np.std(msd_stack, axis=0)
        logger.info(
            f"MSD(t_final) mean={float(np.asarray(msd_mean).flat[-1]):.4f}, "
            f"std={float(np.asarray(msd_std).flat[-1]):.4f}"
        )
    
    # Position 
    position_mean: Optional[np.ndarray] = None
    position_std: Optional[np.ndarray] = None
    if nn_cfg.obs_position:
        position_stack = np.stack([r.position for r in results_typed if r.position is not None], axis=0)
        position_mean = np.mean(position_stack, axis=0)
        position_std = np.std(position_stack, axis=0)
        logger.info(
            f"Position(t_final) mean={float(np.asarray(position_mean).flat[-1]):.4f}, "
            f"std={float(np.asarray(position_std).flat[-1]):.4f}"
        )

    # Populations
    pop_mean: Optional[np.ndarray] = None
    pop_std: Optional[np.ndarray] = None
    if nn_cfg.obs_populations:
        pop_stack = np.stack(
            [r.populations for r in results_typed if r.populations is not None], axis=0
        )
        pop_mean = np.mean(pop_stack, axis=0)   # (N, T)
        pop_std = np.std(pop_stack, axis=0)     # (N, T)

    # ---- save ----
    outfile = outdir / cfg.output.filename
    expectations = {}
    if msd_mean is not None:
        expectations["msd_mean"] = msd_mean
    if msd_std is not None:
        expectations["msd_std"] = msd_std
    if pop_mean is not None:
        expectations["populations_mean"] = pop_mean
    if pop_std is not None:
        expectations["populations_std"] = pop_std
    if position_mean is not None:
        expectations["position_mean"] = position_mean
    if position_std is not None:
        expectations["position_std"] = position_std

    save_dx_h5(
        outfile=outfile,
        t_ps=t_ps,
        dx_mean_nm=None,       # not a displacement in nm for this model
        method="expm_multiply",
        mode="nn_chain_disorder",
        n_realizations=n,
        expectations=expectations,
        extra_attrs={
            "N_emit": nn_cfg.N_emit,
            "eps_0_eV": nn_cfg.eps_0_eV,
            "J_0_eV": nn_cfg.J_0_eV,
            "sigma_eps_eV": nn_cfg.sigma_eps_eV,
            "sigma_J_eV": nn_cfg.sigma_J_eV,
            "t_total_fs": nn_cfg.t_total_fs,
            "n_steps": nn_cfg.n_steps,
            "initial_state_type": nn_cfg.initial_state_type,
            "sigma_sites": nn_cfg.sigma_sites,
            "k_parallel": nn_cfg.k_parallel,
            "seed_base": base_seed,
        },
    )
    logger.success(f"Simulation complete. Output saved to: {outfile.absolute()}")


if __name__ == "__main__":
    run_disorder_nn()
