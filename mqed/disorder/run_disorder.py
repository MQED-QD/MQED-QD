"""Hydra CLI for disorder-averaged NHSE dynamics.

This runner executes many disorder realizations and aggregates observables
using either joblib multiprocessing or MPI reduction.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from joblib import Parallel, delayed
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mqed.Lindblad.quantum_dynamics import NonHermitianSchDynamics, SimulationConfig
from mqed.Lindblad.run_quantum_dynamics import _build_observables, build_initial_state
from mqed.utils.dgf_data import load_gf_h5
from mqed.utils.joblib_track import tqdm_joblib
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.utils.hydra_local import prepare_hydra_config_path
from mqed.utils.save_hdf5 import save_dx_h5


@dataclass(frozen=True)
class DisorderPayload:
    """Immutable worker payload shared across realizations."""

    sim_cfg_kwargs: Dict[str, Any]
    g_slice: np.ndarray
    initial_state_cfg: Dict[str, Any]
    observables_cfg: List[Dict[str, Any]]
    method: str
    init_site: int


def _default_observables_cfg() -> List[Dict[str, Any]]:
    """Fallback observables when none are explicitly configured."""
    return [
        {"name": "root_MSD", "kind": "derived", "enabled": True},
        {"name": "X_shift", "kind": "operator"},
        {"name": "X_shift2", "kind": "operator"},
    ]


def _as_real_array(values: Any) -> np.ndarray:
    """Convert solver output to a real-valued NumPy array."""
    arr = np.asarray(values)
    return np.asarray(np.real_if_close(arr), dtype=float)


def _is_mpi_parent_process() -> bool:
    """Return True when current environment already looks MPI-launched."""
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
    """Auto-spawn mpirun for MPI backend when requested by config."""
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

    cmd = [mpi_cmd, "-np", str(mpi_nproc), sys.executable, "-m", "mqed.disorder.run_disorder"]
    cmd.extend(sys.argv[1:])
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv[1:]):
        cmd.append(f"hydra.run.dir={HydraConfig.get().runtime.output_dir}")

    env = os.environ.copy()
    env["MQED_MPI_AUTO_LAUNCHED"] = "1"

    logger.info(f"Auto-launching MPI job via {mpi_exec} with {mpi_nproc} ranks.")
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"MPI launcher exited with code {completed.returncode}.")
    raise SystemExit(0)


def _build_payload(cfg: DictConfig) -> tuple[DisorderPayload, bool, np.ndarray]:
    """Load once-per-run inputs and package worker configuration.

    Returns:
        A tuple of (payload, compute_root_msd_flag, time_grid_ps).
    """
    data = load_gf_h5(cfg.greens.h5_path)
    g_slice = np.asarray(data["G_total"][0])
    emitter_ev = float(np.asarray(data["energy_eV"])[0])
    rx_nm = np.asarray(data["Rx_nm"])
    tlist = np.arange(
        float(cfg.simulation.t_ps.start),
        float(cfg.simulation.t_ps.stop),
        float(cfg.simulation.t_ps.output_step),
    )

    observables_cfg = OmegaConf.to_container(
        cfg.get("observables", _default_observables_cfg()),
        resolve=True,
    )
    if not observables_cfg:
        observables_cfg = _default_observables_cfg()

    initial_state_cfg = OmegaConf.to_container(cfg.initial_state, resolve=True)
    method = str(cfg.solver.method)
    _, init_site = build_initial_state(initial_state_cfg, method=method, nmol=int(cfg.simulation.Nmol))
    _, compute_root_msd = _build_observables(
        observables_cfg,
        dim=int(cfg.simulation.Nmol) + 1,
        d_nm=float(cfg.simulation.d_nm),
        Nmol=int(cfg.simulation.Nmol),
        init_site=init_site,
    )

    sim_cfg_kwargs: Dict[str, Any] = {
        "tlist": tlist,
        "emitter_frequency": emitter_ev,
        "Nmol": int(cfg.simulation.Nmol),
        "Rx_nm": rx_nm,
        "d_nm": float(cfg.simulation.d_nm),
        "mu_D_debye": float(cfg.simulation.mu_D_debye),
        "mu_A_debye": float(cfg.simulation.mu_A_debye),
        "theta_deg": float(cfg.simulation.theta_deg),
        "phi_deg": cfg.simulation.phi_deg,
        "disorder_sigma_phi_deg": cfg.simulation.get("disorder_sigma_phi_deg", None),
        "mode": str(cfg.simulation.mode),
        "coupling_limit": cfg.simulation.get("coupling_limit", None),
    }

    return (
        DisorderPayload(
            sim_cfg_kwargs=sim_cfg_kwargs,
            g_slice=g_slice,
            initial_state_cfg=initial_state_cfg,
            observables_cfg=observables_cfg,
            method=method,
            init_site=init_site,
        ),
        compute_root_msd,
        tlist,
    )


def _run_one(seed: int, payload: DisorderPayload) -> Dict[str, np.ndarray]:
    """Run one disorder realization and return raw expectation arrays."""
    sim_cfg = SimulationConfig(**payload.sim_cfg_kwargs)
    dyn = NonHermitianSchDynamics(sim_cfg, payload.g_slice, seed=seed)
    rho_or_psi, init_site = build_initial_state(
        payload.initial_state_cfg,
        method=payload.method,
        nmol=sim_cfg.Nmol,
    )
    e_ops, _ = _build_observables(
        payload.observables_cfg,
        dim=sim_cfg.Nmol + 1,
        d_nm=sim_cfg.d_nm,
        Nmol=sim_cfg.Nmol,
        init_site=init_site,
    )
    result = dyn.evolve(rho_or_psi, e_ops=e_ops, options=None)
    return {key: _as_real_array(val) for key, val in result.expectations.items()}


def _aggregate_samples(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Aggregate realization samples into mean/std arrays per observable."""
    if not samples:
        raise ValueError("No realization samples were produced.")
    keys = sorted(samples[0].keys())
    stats: Dict[str, np.ndarray] = {}
    for key in keys:
        stack = np.stack([s[key] for s in samples], axis=0)
        stats[f"{key}_mean"] = np.mean(stack, axis=0)
        stats[f"{key}_std"] = np.std(stack, axis=0)
    return stats


def _append_transport_moments(stats: Dict[str, np.ndarray]) -> None:
    """Add position/x2/MSD aliases expected by downstream plotting/output."""
    if "X_shift_mean" in stats:
        stats["position_mean"] = stats["X_shift_mean"]
    if "X_shift_std" in stats:
        stats["position_std"] = stats["X_shift_std"]

    if "X_shift2_mean" in stats:
        x2_mean = stats["X_shift2_mean"]
        stats["x2_mean"] = x2_mean
        stats["x2_std"] = stats.get("X_shift2_std", np.zeros_like(x2_mean))

    if "X_shift_mean" in stats and "X_shift2_mean" in stats:
        x_mean = stats["X_shift_mean"]
        x2_mean = stats["X_shift2_mean"]
        # MSD = <(x-x0)^2> = x2 (second moment of displacement).
        # Note: x2 - <x>^2 is the *variance*, not the MSD.
        stats["msd_mean"] = np.maximum(0.0, x2_mean)
        # Also store variance separately for reference.
        stats["variance_mean"] = np.maximum(0.0, x2_mean - x_mean ** 2)


def _root_msd_from_sample(sample: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Compute sqrt(MSD) from one sample.

    MSD = <(x-x0)^2> = X_shift2 (the second moment of displacement).
    Note: previously this subtracted <x>^2, giving sqrt(variance) instead.
    """
    if "X_shift2_cond" in sample:
        return np.sqrt(np.maximum(0.0, sample["X_shift2_cond"]))
    if "X_shift2" in sample:
        return np.sqrt(np.maximum(0.0, sample["X_shift2"]))
    return None


def _compute_root_msd(samples: List[Dict[str, np.ndarray]]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Aggregate sqrt(MSD) mean/std across all samples."""
    root_vals: List[np.ndarray] = []
    for sample in samples:
        root_series = _root_msd_from_sample(sample)
        if root_series is None:
            return None, None
        root_vals.append(root_series)
    root_stack = np.stack(root_vals, axis=0)
    return np.mean(root_stack, axis=0), np.std(root_stack, axis=0)


def _run_ensemble_joblib(
    seeds: List[int],
    payload: DisorderPayload,
    n_jobs: int,
) -> List[Dict[str, np.ndarray]]:
    """Evaluate all realizations via joblib multiprocessing."""
    worker = delayed(_run_one)
    with tqdm_joblib(
        tqdm(total=len(seeds), desc="NHSE disorder realizations")
    ):
        results = Parallel(n_jobs=n_jobs, prefer="processes")(worker(seed, payload) for seed in seeds)
    return results


def _run_ensemble_mpi(
    seeds: List[int],
    payload: DisorderPayload,
    compute_root_msd: bool,
) -> tuple[Optional[Dict[str, np.ndarray]], int, int, int]:
    """Evaluate realizations with MPI and reduce moments to rank 0."""
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

    sim_cfg = SimulationConfig(**payload.sim_cfg_kwargs)
    e_ops, _ = _build_observables(
        payload.observables_cfg,
        dim=sim_cfg.Nmol + 1,
        d_nm=sim_cfg.d_nm,
        Nmol=sim_cfg.Nmol,
        init_site=payload.init_site,
    )
    keys = list(e_ops.keys())
    t_len = int(np.asarray(sim_cfg.tlist).size)
    has_conditional_moments = "X_shift_cond" in keys and "X_shift2_cond" in keys
    has_raw_moments = "X_shift" in keys and "X_shift2" in keys
    can_compute_root_msd = compute_root_msd and (has_conditional_moments or has_raw_moments)

    local_sum: Dict[str, np.ndarray] = {key: np.zeros(t_len, dtype=float) for key in keys}
    local_sumsq: Dict[str, np.ndarray] = {key: np.zeros(t_len, dtype=float) for key in keys}
    local_root_sum = np.zeros(t_len, dtype=float)
    local_root_sumsq = np.zeros(t_len, dtype=float)

    for seed in local_seeds:
        sample = _run_one(seed, payload)
        for key in keys:
            local_sum[key] += sample[key]
            local_sumsq[key] += sample[key] ** 2
        if can_compute_root_msd:
            root_series = _root_msd_from_sample(sample)
            if root_series is None:
                raise ValueError(
                    "root_MSD requested, but required moments are missing from sampled observables."
                )
            local_root_sum += root_series
            local_root_sumsq += root_series ** 2

    local_count = np.array([len(local_seeds)], dtype=np.int64)
    global_count = np.array([0], dtype=np.int64)
    comm.Reduce(local_count, global_count, op=MPI.SUM, root=0)

    global_sum: Dict[str, np.ndarray] = {}
    global_sumsq: Dict[str, np.ndarray] = {}
    for key in keys:
        global_sum[key] = np.zeros_like(local_sum[key])
        global_sumsq[key] = np.zeros_like(local_sumsq[key])
        comm.Reduce(local_sum[key], global_sum[key], op=MPI.SUM, root=0)
        comm.Reduce(local_sumsq[key], global_sumsq[key], op=MPI.SUM, root=0)

    global_root_sum = np.zeros_like(local_root_sum)
    global_root_sumsq = np.zeros_like(local_root_sumsq)
    if can_compute_root_msd:
        comm.Reduce(local_root_sum, global_root_sum, op=MPI.SUM, root=0)
        comm.Reduce(local_root_sumsq, global_root_sumsq, op=MPI.SUM, root=0)

    if rank != 0:
        return None, rank, size, int(global_count[0])

    n_total = int(global_count[0])
    if n_total <= 0:
        raise ValueError("No realizations were assigned across MPI ranks.")

    stats: Dict[str, np.ndarray] = {}
    for key in keys:
        mean = global_sum[key] / n_total
        var = np.maximum(0.0, global_sumsq[key] / n_total - mean ** 2)
        stats[f"{key}_mean"] = mean
        stats[f"{key}_std"] = np.sqrt(var)

    if can_compute_root_msd:
        root_mean = global_root_sum / n_total
        root_var = np.maximum(0.0, global_root_sumsq / n_total - root_mean ** 2)
        stats["root_MSD_mean"] = root_mean
        stats["root_MSD_std"] = np.sqrt(root_var)

    return stats, rank, size, n_total


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("Lindblad", __file__)

@hydra.main(
    config_path=HYDRA_CONFIG_PATH,
    config_name="quantum_dynamics_disorder",
    version_base=None,
)
def run_disorder(cfg: Optional[DictConfig] = None) -> None:
    """Hydra entrypoint for NHSE disorder ensemble simulation."""
    if cfg is None:
        raise ValueError("Hydra did not provide configuration.")

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
                "Using launcher-provided world size."
            )

    if str(cfg.solver.method) != "NonHermitian":
        raise ValueError("disorder runner supports solver.method=NonHermitian only.")

    if backend != "mpi" or rank == 0:
        logger.info("--- NHSE disorder ensemble ---")
        logger.info(f"Parallel backend: {backend}")
        if bool(cfg.disorder.get("save_each", False)):
            logger.warning("disorder.save_each is currently not implemented in run_disorder.")

    payload, compute_root_msd, tlist = _build_payload(cfg)
    n = int(cfg.disorder.n_realizations)
    base_seed = int(cfg.disorder.seed)
    seeds = [base_seed + i for i in range(n)]

    dx_mean: Optional[np.ndarray] = None
    dx_std: Optional[np.ndarray] = None
    n_total = n

    if backend == "joblib":
        n_jobs = int(cfg.disorder.n_jobs) if cfg.disorder.n_jobs is not None else -1
        samples = _run_ensemble_joblib(seeds, payload, n_jobs=n_jobs)
        stats = _aggregate_samples(samples)
        if compute_root_msd:
            dx_mean, dx_std = _compute_root_msd(samples)
            if dx_mean is None:
                raise ValueError(
                    "root_MSD requested, but required moments are missing from sampled observables."
                )
            stats["root_MSD_mean"] = dx_mean
            stats["root_MSD_std"] = dx_std if dx_std is not None else np.zeros_like(dx_mean)
    elif backend == "mpi":
        if rank == 0:
            logger.info(f"Running {n} realizations with MPI (size={size})")
        stats, rank, size, n_total = _run_ensemble_mpi(seeds, payload, compute_root_msd=compute_root_msd)
        if rank != 0 or stats is None:
            return
        if compute_root_msd:
            if "root_MSD_mean" not in stats:
                raise ValueError(
                    "root_MSD requested, but required moments are missing from sampled observables."
                )
            dx_mean = stats["root_MSD_mean"]
            dx_std = stats["root_MSD_std"]
    else:
        raise ValueError(f"Unsupported disorder.backend '{backend}'. Use 'joblib' or 'mpi'.")

    _append_transport_moments(stats)

    save_legacy_aliases = bool(cfg.output.get("save_legacy_aliases", True))
    if not save_legacy_aliases:
        stats.pop("X_shift_mean", None)
        stats.pop("X_shift_std", None)
        stats.pop("X_shift2_mean", None)
        stats.pop("X_shift2_std", None)

    outdir = Path(HydraConfig.get().runtime.output_dir)
    outfile = outdir / str(cfg.output.filename)

    save_dx_h5(
        outfile=outfile,
        t_ps=np.asarray(tlist),
        dx_mean_nm=dx_mean,
        dx_std_nm=dx_std,
        method="NonHermitian",
        mode="disorder",
        n_realizations=n_total,
        expectations=stats,
        extra_attrs={
            "parallel_backend": backend,
            "mpi_size": size,
            "seed_base": base_seed,
            "save_legacy_aliases": save_legacy_aliases,
            "initial_state_type": str(cfg.initial_state.get("type", "single_site")),
            "initial_state_site_index": int(payload.init_site),
            "sigma_sites": float(cfg.initial_state.get("sigma_sites", 1.0)),
            "k_parallel": float(cfg.initial_state.get("k_parallel", 0.0)),
            "initial_state_sigma_sites": float(cfg.initial_state.get("sigma_sites", 1.0)),
            "initial_state_k_parallel": float(cfg.initial_state.get("k_parallel", 0.0)),
            "root_MSD_enabled": compute_root_msd,
        },
    )

    logger.success(f"Simulation complete. Output saved to: {outfile.absolute()}")


if __name__ == "__main__":
    run_disorder()
