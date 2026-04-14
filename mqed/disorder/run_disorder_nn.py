"""Hydra CLI for disorder-averaged NN-chain quantum dynamics.

Usage
-----
    mqed_nn_disorder                                  # default config (joblib backend)
    mqed_nn_disorder chain.N_emit=500 disorder.n_realizations=100
    mqed_nn_disorder disorder.sigma_eps_eV=0.01 disorder.sigma_J_eV=0.005
    mpirun -np 8 mqed_nn_disorder disorder.backend=mpi

The ensemble-averaging pattern mirrors ``mqed.disorder.run_disorder`` (NHSE),
adapted for the lightweight NN-chain propagator.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from loguru import logger

import hydra
from omegaconf import DictConfig
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
        obs_position=bool(cfg.observables.get("position", True)),
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


def _is_mpi_parent_process() -> bool:
    mpi_env_keys = (
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_RANK",
        "PMI_SIZE",
        "PMIX_RANK",
        "MV2_COMM_WORLD_SIZE",
        "MPI_LOCALNRANKS",
        "SLURM_PROCID",
        "SLURM_NTASKS",
    )
    return any(key in os.environ for key in mpi_env_keys)


def _maybe_auto_launch_mpi(cfg: DictConfig) -> None:
    backend = str(cfg.disorder.get("backend", "joblib")).lower()
    auto_launch = bool(cfg.disorder.get("mpi_auto_launch", False))
    if backend != "mpi" or not auto_launch:
        return
    if os.environ.get("MQED_MPI_AUTO_LAUNCHED") == "1":
        return
    if _is_mpi_parent_process():
        return

    mpi_nproc = int(cfg.disorder.get("mpi_nproc", 1))
    if mpi_nproc <= 1:
        logger.warning(
            "MPI auto-launch requested with mpi_nproc <= 1; continuing without mpirun."
        )
        return

    mpi_exec = str(cfg.disorder.get("mpi_exec", "mpirun"))
    mpi_cmd = shutil.which(mpi_exec)
    if mpi_cmd is None:
        raise FileNotFoundError(
            f"MPI launcher '{mpi_exec}' was not found in PATH. "
            "Set disorder.mpi_exec to a valid launcher or disable disorder.mpi_auto_launch."
        )

    cmd = [mpi_cmd, "-np", str(mpi_nproc), sys.executable, "-m", "mqed.disorder.run_disorder_nn"]
    cmd.extend(sys.argv[1:])
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv[1:]):
        cmd.append(f"hydra.run.dir={HydraConfig.get().runtime.output_dir}")

    env = os.environ.copy()
    env["MQED_MPI_AUTO_LAUNCHED"] = "1"

    logger.info(
        f"Auto-launching MPI job via {mpi_exec} with {mpi_nproc} ranks."
    )
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"MPI launcher exited with code {completed.returncode}."
        )
    raise SystemExit(0)


def _run_ensemble_joblib(seeds: List[int], nn_cfg: NNChainConfig, n_jobs: int) -> List[NNChainResult]:
    logger.info(f"Running {len(seeds)} realisations with joblib (n_jobs={n_jobs})")
    worker = delayed(_run_one)
    with tqdm_joblib(
        tqdm(total=len(seeds), desc="NN-chain disorder realisations")
    ) as progress_bar:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            worker(s, nn_cfg) for s in seeds
        )
    assert results is not None, "Parallel returned None"

    results_typed: List[NNChainResult] = []
    for result in results:
        if result is None:
            raise RuntimeError("A disorder realization returned None.")
        results_typed.append(result)
    return results_typed


def _compute_aggregates_from_results(
    results_typed: List[NNChainResult],
    nn_cfg: NNChainConfig,
) -> Dict[str, Optional[np.ndarray]]:
    if not results_typed:
        raise ValueError("No results to aggregate.")

    out: Dict[str, Optional[np.ndarray]] = {
        "t_fs": results_typed[0].t_fs,
        "x2_mean": None,
        "x2_std": None,
        "msd_mean": None,
        "msd_std": None,
        "position_mean": None,
        "position_std": None,
        "pop_mean": None,
        "pop_std": None,
    }

    x2_stack: Optional[np.ndarray] = None
    position_stack: Optional[np.ndarray] = None

    if nn_cfg.obs_msd:
        x2_stack = np.stack([r.msd for r in results_typed if r.msd is not None], axis=0)
        out["x2_mean"] = np.mean(x2_stack, axis=0)
        out["x2_std"] = np.std(x2_stack, axis=0)

    if nn_cfg.obs_position:
        position_stack = np.stack([r.position for r in results_typed if r.position is not None], axis=0)
        out["position_mean"] = np.mean(position_stack, axis=0)
        out["position_std"] = np.std(position_stack, axis=0)

    if x2_stack is not None:
        # MSD = <(x-x0)^2> = x2 (second moment of displacement).
        # Note: x2 - position^2 would be the variance, not the MSD.
        out["msd_mean"] = out["x2_mean"]
        out["msd_std"] = out["x2_std"]

    if nn_cfg.obs_populations:
        pop_stack = np.stack(
            [r.populations for r in results_typed if r.populations is not None], axis=0
        )
        out["pop_mean"] = np.mean(pop_stack, axis=0)
        out["pop_std"] = np.std(pop_stack, axis=0)

    return out


def _run_ensemble_mpi(
    seeds: List[int],
    nn_cfg: NNChainConfig,
) -> tuple[Optional[Dict[str, Optional[np.ndarray]]], int, int]:
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "MPI backend requested but mpi4py is not installed. "
            "Install mpi4py or switch disorder.backend=joblib."
        ) from exc

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_seeds = seeds[rank::size]
    t_fs = np.linspace(0.0, nn_cfg.t_total_fs, nn_cfg.n_steps)
    T = t_fs.size

    local_x2_sum = np.zeros(T, dtype=float)
    local_x2_sumsq = np.zeros(T, dtype=float)
    local_msd_sum = np.zeros(T, dtype=float)
    local_msd_sumsq = np.zeros(T, dtype=float)
    local_pos_sum = np.zeros(T, dtype=float)
    local_pos_sumsq = np.zeros(T, dtype=float)
    local_pop_sum = np.zeros((nn_cfg.N_emit, T), dtype=float)
    local_pop_sumsq = np.zeros((nn_cfg.N_emit, T), dtype=float)

    for seed in local_seeds:
        result = _run_one(seed, nn_cfg)
        if nn_cfg.obs_msd and result.msd is not None:
            local_x2_sum += result.msd
            local_x2_sumsq += result.msd ** 2
            # MSD = <(x-x0)^2> = x2 (second moment of displacement).
            # Previously this subtracted position^2 (giving variance, not MSD).
            msd_realization = result.msd
            local_msd_sum += msd_realization
            local_msd_sumsq += msd_realization ** 2
        if nn_cfg.obs_position and result.position is not None:
            local_pos_sum += result.position
            local_pos_sumsq += result.position ** 2
        if nn_cfg.obs_populations and result.populations is not None:
            local_pop_sum += result.populations
            local_pop_sumsq += result.populations ** 2

    local_count = np.array([len(local_seeds)], dtype=np.int64)
    global_count = np.array([0], dtype=np.int64)
    comm.Reduce(local_count, global_count, op=MPI.SUM, root=0)

    global_x2_sum = np.zeros_like(local_x2_sum)
    global_x2_sumsq = np.zeros_like(local_x2_sumsq)
    global_msd_sum = np.zeros_like(local_msd_sum)
    global_msd_sumsq = np.zeros_like(local_msd_sumsq)
    global_pos_sum = np.zeros_like(local_pos_sum)
    global_pos_sumsq = np.zeros_like(local_pos_sumsq)
    global_pop_sum = np.zeros_like(local_pop_sum)
    global_pop_sumsq = np.zeros_like(local_pop_sumsq)

    if nn_cfg.obs_msd:
        comm.Reduce(local_x2_sum, global_x2_sum, op=MPI.SUM, root=0)
        comm.Reduce(local_x2_sumsq, global_x2_sumsq, op=MPI.SUM, root=0)
        comm.Reduce(local_msd_sum, global_msd_sum, op=MPI.SUM, root=0)
        comm.Reduce(local_msd_sumsq, global_msd_sumsq, op=MPI.SUM, root=0)
    if nn_cfg.obs_position:
        comm.Reduce(local_pos_sum, global_pos_sum, op=MPI.SUM, root=0)
        comm.Reduce(local_pos_sumsq, global_pos_sumsq, op=MPI.SUM, root=0)
    if nn_cfg.obs_populations:
        comm.Reduce(local_pop_sum, global_pop_sum, op=MPI.SUM, root=0)
        comm.Reduce(local_pop_sumsq, global_pop_sumsq, op=MPI.SUM, root=0)

    n_total = int(global_count[0])
    if rank != 0:
        return None, rank, size
    if n_total <= 0:
        raise ValueError("No realizations were assigned across MPI ranks.")

    out: Dict[str, Optional[np.ndarray]] = {
        "t_fs": t_fs,
        "x2_mean": None,
        "x2_std": None,
        "msd_mean": None,
        "msd_std": None,
        "position_mean": None,
        "position_std": None,
        "pop_mean": None,
        "pop_std": None,
    }

    if nn_cfg.obs_msd:
        x2_mean = global_x2_sum / n_total
        x2_var = np.maximum(0.0, global_x2_sumsq / n_total - x2_mean ** 2)
        out["x2_mean"] = x2_mean
        out["x2_std"] = np.sqrt(x2_var)

    if nn_cfg.obs_position:
        pos_mean = global_pos_sum / n_total
        pos_var = np.maximum(0.0, global_pos_sumsq / n_total - pos_mean ** 2)
        out["position_mean"] = pos_mean
        out["position_std"] = np.sqrt(pos_var)

    if out["x2_mean"] is not None:
        msd_mean = global_msd_sum / n_total
        msd_var = np.maximum(0.0, global_msd_sumsq / n_total - msd_mean ** 2)
        out["msd_mean"] = msd_mean
        out["msd_std"] = np.sqrt(msd_var)

    if nn_cfg.obs_populations:
        pop_mean = global_pop_sum / n_total
        pop_var = np.maximum(0.0, global_pop_sumsq / n_total - pop_mean ** 2)
        out["pop_mean"] = pop_mean
        out["pop_std"] = np.sqrt(pop_var)

    return out, rank, size


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

    _maybe_auto_launch_mpi(cfg)

    backend = str(cfg.disorder.get("backend", "joblib")).lower()
    rank = 0
    size = 1
    requested_size = int(cfg.disorder.get("mpi_nproc", 1))
    if backend == "mpi":
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError(
                "MPI backend requested but mpi4py is not installed. "
                "Install mpi4py or set disorder.backend=joblib."
            ) from exc
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()

    if backend != "mpi" or rank == 0:
        setup_loggers_hydra_aware()
        if backend == "mpi" and size != requested_size:
            logger.warning(
                f"MPI world size is {size}, while disorder.mpi_nproc={requested_size}. "
                "Using the actual launcher-provided world size."
            )

    outdir = Path(HydraConfig.get().runtime.output_dir)

    if backend != "mpi" or rank == 0:
        logger.info("— NN-chain disorder ensemble —")
    nn_cfg = _build_config(cfg)
    save_legacy_aliases = bool(cfg.output.get("save_legacy_aliases", True))
    if backend != "mpi" or rank == 0:
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
        if nn_cfg.initial_state_type == "gaussian":
            logger.info(
                f"Initial state: Gaussian wavepacket with σ_sites={nn_cfg.sigma_sites} sites, "
                f"k_parallel={nn_cfg.k_parallel}"
            )
        else:
            logger.info(
                f"Initial state: {nn_cfg.initial_state_type}"
            )

    # ---- seeds ----
    n = int(cfg.disorder.n_realizations)
    base_seed = int(cfg.disorder.seed)
    seeds = [base_seed + i for i in range(n)]

    if backend == "joblib":
        n_jobs = int(cfg.disorder.n_jobs) if cfg.disorder.n_jobs is not None else -1
        stats = _compute_aggregates_from_results(_run_ensemble_joblib(seeds, nn_cfg, n_jobs), nn_cfg)
    elif backend == "mpi":
        if rank == 0:
            logger.info(f"Running {n} realisations with MPI (size={size})")
        stats, rank, size = _run_ensemble_mpi(seeds, nn_cfg)
        if rank != 0 or stats is None:
            return
    else:
        raise ValueError(f"Unsupported disorder.backend '{backend}'. Use 'joblib' or 'mpi'.")

    t_fs = stats["t_fs"]
    if t_fs is None:
        raise RuntimeError("Missing time grid in aggregated stats.")
    t_ps = np.asarray(t_fs) * 1.0e-3

    x2_mean = stats["x2_mean"]
    x2_std = stats["x2_std"]
    msd_mean = stats["msd_mean"]
    msd_std = stats["msd_std"]
    pop_mean = stats["pop_mean"]
    pop_std = stats["pop_std"]
    position_mean = stats["position_mean"]
    position_std = stats["position_std"]

    if x2_mean is not None and x2_std is not None:
        logger.info(
            f"<x^2>(t_final) mean={float(np.asarray(x2_mean).flat[-1]):.4f}, "
            f"std={float(np.asarray(x2_std).flat[-1]):.4f}"
        )
    if position_mean is not None and position_std is not None:
        logger.info(
            f"Position(t_final) mean={float(np.asarray(position_mean).flat[-1]):.4f}, "
            f"std={float(np.asarray(position_std).flat[-1]):.4f}"
        )
    if msd_mean is not None and msd_std is not None:
        logger.info(
            f"MSD(t_final) mean={float(np.asarray(msd_mean).flat[-1]):.4f}, "
            f"std={float(np.asarray(msd_std).flat[-1]):.4f}"
        )

    # ---- save ----
    outfile = outdir / cfg.output.filename
    expectations = {}
    if x2_mean is not None:
        expectations["x2_mean"] = x2_mean
        if save_legacy_aliases:
            expectations["X_shift2"] = x2_mean
    if x2_std is not None:
        expectations["x2_std"] = x2_std
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
        if save_legacy_aliases:
            expectations["X_shift"] = position_mean
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
            "save_legacy_aliases": save_legacy_aliases,
            "parallel_backend": backend,
            "mpi_size": size,
        },
    )
    logger.success(f"Simulation complete. Output saved to: {outfile.absolute()}")


if __name__ == "__main__":
    run_disorder_nn()
