"""
Sommerfeld Green's function simulation — multi-frequency, multi-Rx grid.

Execution Model
---------------
The frequency axis is embarrassingly parallel: each energy point requires
an independent ``Greens_function_analytical`` instance (with its own
ε(ω)) and loops over all Rx positions.  Three execution backends are
supported, controlled by ``parallel.backend`` in the Hydra config:

  * ``sequential`` — plain for-loop (default, zero dependencies).
  * ``joblib``     — multiprocessing via joblib (shared-memory, good
                     for laptops/workstations).
  * ``mpi``        — distributed via mpi4py (good for HPC clusters).

For ``joblib``, the worker function :func:`_compute_one_energy` is
called via ``Parallel(delayed(...))``.  ``DataProvider`` is re-created
inside each worker to avoid pickling issues with interpolation objects.

For ``mpi``, energies are scattered round-robin across ranks; each rank
writes its chunk and rank 0 gathers and saves.
"""

import sys
import subprocess

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import numpy as np
from loguru import logger
from pathlib import Path
from tqdm import tqdm

from mqed.Dyadic_GF.data_provider import DataProvider
from mqed.Dyadic_GF.GF_Sommerfeld import Greens_function_analytical
from mqed.utils.SI_unit import eV_to_J, hbar, c
from mqed.utils.dgf_data import save_gf_h5
from hydra.core.hydra_config import HydraConfig
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.utils.hydra_local import prepare_hydra_config_path


# ─────────────────────────────────────────────────────────────────────
#  Grid builder (unchanged)
# ─────────────────────────────────────────────────────────────────────

def build_grid(config):
    """Builds a 1-D numpy array from flexible Hydra config input.

    Accepted formats:
        - Single value:  ``2.0``              →  ``[2.0]``
        - List:          ``[1.0, 2.0, 3.0]``  →  as-is
        - Dict/linspace: ``{min: 1.0, max: 3.0, points: 5}``
    """
    if isinstance(config, (float, int)):
        return np.array([config], dtype=float)
    elif isinstance(config, (list, ListConfig)):
        return np.array(config, dtype=float)
    elif isinstance(config, (dict, DictConfig)):
        return np.linspace(config.min, config.max, config.points, dtype=float)
    else:
        raise TypeError(f"Unsupported spectral config type: {type(config)}")


# ─────────────────────────────────────────────────────────────────────
#  Per-energy worker (top-level function so it can be pickled by joblib)
# ─────────────────────────────────────────────────────────────────────

def _compute_one_energy(
    idx: int,
    energy_eV: float,
    lambda_m: float,
    rx_values_m: np.ndarray,
    zD: float,
    zA: float,
    material_cfg,
    integ_cfg,
) -> tuple:
    """Compute Green's function for all Rx at a single energy.

    This is the atomic unit of work for parallelization.  Each call is
    completely independent — no shared state with other workers.

    ``DataProvider`` is re-instantiated inside the worker because
    scipy's ``interp1d`` objects are not always pickle-safe across
    processes.

    Args:
        idx:          Energy index (used to place results back in the
                      output array).
        energy_eV:    Energy value in eV (for logging only).
        lambda_m:     Corresponding wavelength in meters.
        rx_values_m:  1-D array of Rx positions in meters.
        zD:           Source z-position (donor height) in meters.
        zA:           Observer z-position (acceptor height) in meters.
        material_cfg: OmegaConf subtree for material (re-creates
                      ``DataProvider`` inside worker).
        integ_cfg:    OmegaConf subtree for integration parameters.

    Returns:
        (idx, total, vacuum): idx is the energy index; total and vacuum
        are ``(nR, 3, 3)`` complex arrays.
    """
    omega = 2 * np.pi * c / lambda_m

    # Re-create DataProvider per worker (avoids pickling interp1d).
    data_provider = DataProvider(material_cfg)
    epsilon = data_provider.get_epsilon(omega)

    calculator = Greens_function_analytical(
        omega=omega,
        metal_epsi=epsilon,
        qmax=None if integ_cfg is None else integ_cfg.qmax,
        epsabs=1e-10 if integ_cfg is None else integ_cfg.epsabs,
        epsrel=1e-10 if integ_cfg is None else integ_cfg.epsrel,
        limit=400 if integ_cfg is None else int(integ_cfg.limit),
        split_propagating=True if integ_cfg is None else bool(integ_cfg.split_propagating),
    )

    nR = len(rx_values_m)
    total = np.zeros((nR, 3, 3), dtype=complex)
    vacuum = np.zeros((nR, 3, 3), dtype=complex)

    for j, rx_m in enumerate(rx_values_m):
        total[j] = calculator.calculate_total_Green_function(
            x=rx_m, y=0, z1=zD, z2=zA,
        )
        vacuum[j] = calculator.vacuum_component(
            x=rx_m, y=0, z1=zD, z2=zA,
        )

    return idx, total, vacuum


# ─────────────────────────────────────────────────────────────────────
#  Backend dispatchers
# ─────────────────────────────────────────────────────────────────────

def _run_sequential(
    energy_eV_array, target_lambdas_m, rx_values_m, sim_params, material_cfg
):
    """Sequential execution — simple for-loop, no parallelism."""
    nE = len(energy_eV_array)
    nR = len(rx_values_m)
    results_total = np.zeros((nE, nR, 3, 3), dtype=complex)
    results_vacuum = np.zeros((nE, nR, 3, 3), dtype=complex)
    integ_cfg = getattr(sim_params, "integration", None)

    for i in tqdm(range(nE), desc="Energies", ncols=100):
        logger.info(f"Energy {i+1}/{nE}: {energy_eV_array[i]:.3f} eV")
        _, tot, vac = _compute_one_energy(
            idx=i,
            energy_eV=energy_eV_array[i],
            lambda_m=target_lambdas_m[i],
            rx_values_m=rx_values_m,
            zD=sim_params.position.zD,
            zA=sim_params.position.zA,
            material_cfg=material_cfg,
            integ_cfg=integ_cfg,
        )
        results_total[i] = tot
        results_vacuum[i] = vac

    return results_total, results_vacuum


def _run_joblib(
    energy_eV_array, target_lambdas_m, rx_values_m, sim_params, material_cfg, n_jobs
):
    """Joblib backend — parallelize over energy axis.

    Each energy is dispatched as an independent task to a pool of
    ``n_jobs`` processes.  Communication is via return values (no
    shared memory needed).
    """
    from joblib import Parallel, delayed
    from mqed.utils.joblib_track import tqdm_joblib

    nE = len(energy_eV_array)
    nR = len(rx_values_m)
    integ_cfg = getattr(sim_params, "integration", None)

    # OmegaConf containers are not always pickle-friendly — convert to
    # plain dicts/primitives for cross-process transfer.
    material_cfg_plain = OmegaConf.to_container(material_cfg, resolve=True)
    integ_cfg_plain = (
        OmegaConf.to_container(integ_cfg, resolve=True)
        if integ_cfg is not None
        else None
    )
    # Wrap the plain dict back into a DictConfig so DataProvider sees
    # the same attribute-access interface it expects.
    material_cfg_dc = OmegaConf.create(material_cfg_plain)
    integ_cfg_dc = OmegaConf.create(integ_cfg_plain) if integ_cfg_plain is not None else None

    logger.info(f"Joblib backend: dispatching {nE} energies across {n_jobs} workers")

    with tqdm_joblib(tqdm(total=nE, desc="Energies (joblib)", ncols=100)):
        raw_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_compute_one_energy)(
                idx=i,
                energy_eV=energy_eV_array[i],
                lambda_m=target_lambdas_m[i],
                rx_values_m=rx_values_m,
                zD=sim_params.position.zD,
                zA=sim_params.position.zA,
                material_cfg=material_cfg_dc,
                integ_cfg=integ_cfg_dc,
            )
            for i in range(nE)
        )

    results_total = np.zeros((nE, nR, 3, 3), dtype=complex)
    results_vacuum = np.zeros((nE, nR, 3, 3), dtype=complex)
    for idx, tot, vac in raw_results:
        results_total[idx] = tot
        results_vacuum[idx] = vac

    return results_total, results_vacuum


def _maybe_auto_launch_mpi(parallel_cfg):
    """Re-launch this script under mpiexec if not already running in MPI.

    Same pattern as ``mqed.disorder.run_disorder._maybe_auto_launch_mpi``.
    Returns True if we re-launched (caller should ``sys.exit``).
    """
    if not parallel_cfg.get("mpi_auto_launch", True):
        return False

    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:
            return False
    except ImportError:
        pass

    nproc = parallel_cfg.get("mpi_nproc", 4)
    mpi_exec = parallel_cfg.get("mpi_exec", "mpiexec")
    cmd = [mpi_exec, "-n", str(nproc)] + sys.argv
    logger.info(f"Auto-launching MPI: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def _run_mpi(
    energy_eV_array, target_lambdas_m, rx_values_m, sim_params, material_cfg, parallel_cfg
):
    """MPI backend — scatter energies round-robin across ranks.

    Rank 0 gathers all partial results and returns the full arrays.
    Non-root ranks return ``(None, None)``.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nE = len(energy_eV_array)
    nR = len(rx_values_m)
    integ_cfg = getattr(sim_params, "integration", None)

    # Round-robin assignment: rank k handles energies k, k+size, k+2*size, ...
    local_indices = list(range(rank, nE, size))

    if rank == 0:
        logger.info(f"MPI backend: {size} ranks, {nE} energies ({len(local_indices)} per rank avg)")

    local_results = []
    for i in local_indices:
        if rank == 0:
            logger.info(f"[rank {rank}] Energy {i+1}/{nE}: {energy_eV_array[i]:.3f} eV")
        idx, tot, vac = _compute_one_energy(
            idx=i,
            energy_eV=energy_eV_array[i],
            lambda_m=target_lambdas_m[i],
            rx_values_m=rx_values_m,
            zD=sim_params.position.zD,
            zA=sim_params.position.zA,
            material_cfg=material_cfg,
            integ_cfg=integ_cfg,
        )
        local_results.append((idx, tot, vac))

    # Gather all partial results on rank 0
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        results_total = np.zeros((nE, nR, 3, 3), dtype=complex)
        results_vacuum = np.zeros((nE, nR, 3, 3), dtype=complex)
        for rank_results in all_results:
            for idx, tot, vac in rank_results:
                results_total[idx] = tot
                results_vacuum[idx] = vac
        return results_total, results_vacuum

    return None, None


# ─────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────

HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("Dyadic_GF", __file__)

@hydra.main(config_path=HYDRA_CONFIG_PATH, config_name="GF_Sommerfeld", version_base=None)
def run_simulation(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()

    logger.info("--- Starting Green's Function Simulation ---")

    sim_params = cfg.simulation
    material_cfg = cfg.material
    parallel_cfg = cfg.get("parallel", {})
    backend = parallel_cfg.get("backend", "sequential") if parallel_cfg else "sequential"

    # ── MPI auto-launch guard ──
    if backend == "mpi":
        _maybe_auto_launch_mpi(parallel_cfg)

    # ── Build energy grid ──
    kind = sim_params.spectral_param
    logger.info(f"Spectral parameter kind: {kind}")

    if kind == "energy_eV":
        energy_ev_array = build_grid(sim_params.energy_eV)
    elif kind == "wavelength_nm":
        lambda_nm = build_grid(sim_params.wavelength_nm)
        energy_ev_array = 2 * np.pi * hbar * c / (lambda_nm * 1e-9 * eV_to_J)
    else:
        raise ValueError(f"Unknown spectral_param: {kind}")

    energy_J = energy_ev_array * eV_to_J
    target_lambdas_m = 2 * np.pi * hbar * c / energy_J

    # ── Build Rx grid ──
    rx_values_nm = np.linspace(
        sim_params.position.Rx_nm.start,
        sim_params.position.Rx_nm.stop,
        sim_params.position.Rx_nm.points,
    )
    rx_values_m = rx_values_nm * 1e-9
    logger.info(
        f"Grid: {len(energy_ev_array)} energies × {len(rx_values_m)} Rx points  |  backend={backend}"
    )

    # ── Dispatch to backend ──
    if backend == "sequential":
        results_total, results_vacuum = _run_sequential(
            energy_ev_array, target_lambdas_m, rx_values_m, sim_params, material_cfg,
        )
    elif backend == "joblib":
        n_jobs = parallel_cfg.get("n_jobs", -1)
        results_total, results_vacuum = _run_joblib(
            energy_ev_array, target_lambdas_m, rx_values_m, sim_params, material_cfg, n_jobs,
        )
    elif backend == "mpi":
        results_total, results_vacuum = _run_mpi(
            energy_ev_array, target_lambdas_m, rx_values_m, sim_params, material_cfg, parallel_cfg,
        )
        # Only rank 0 saves output
        try:
            from mpi4py import MPI
            if MPI.COMM_WORLD.Get_rank() != 0:
                return
        except ImportError:
            pass
    else:
        raise ValueError(
            f"Unknown parallel.backend: '{backend}'. "
            f"Choose from: sequential, joblib, mpi"
        )

    # ── Save results ──
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dynamic output filename:
    #   Hydra config provides the prefix (e.g. "Fresnel_GF_planar_Ag_height_5nm")
    #   Python appends the energy range, points, and Rx info at runtime.
    output_prefix = cfg.output.prefix
    E_min = energy_ev_array[0]
    E_max = energy_ev_array[-1]
    n_energy = len(energy_ev_array)
    Rx_max = rx_values_nm[-1]
    n_rx = len(rx_values_nm)
    output_fname = (
        f"{output_prefix}"
        f"_Emin_{E_min:.2f}_Emax_{E_max:.2f}_{n_energy}pts"
        f"_Rx_{Rx_max:.0f}nm_{n_rx}pts.hdf5"
    )
    output_file = output_dir / output_fname

    save_gf_h5(
        output_file,
        results_total,
        results_vacuum,
        energy_J / eV_to_J,
        rx_values_nm,
        sim_params.position.zD,
        sim_params.position.zA,
    )

    logger.success(f"Simulation complete. Output saved to: {output_file.absolute()}")


if __name__ == "__main__":
    run_simulation()
