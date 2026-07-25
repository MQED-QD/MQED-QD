from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mqed.Dyadic_GF.GF_NLayer import LayerSpec, NLayerGreenFunction
from mqed.Dyadic_GF.data_provider import DataProvider
from mqed.Dyadic_GF.main import (
    _deduplicate_sorted_grid,
    _maybe_auto_launch_mpi,
    build_grid,
    build_position_grid,
)
from mqed.utils.SI_unit import c, eV_to_J, hbar
from mqed.utils.dgf_data import save_gf_h5
from mqed.utils.hydra_local import prepare_hydra_config_path
from mqed.utils.logging_utils import setup_loggers_hydra_aware


def _thickness_m(layer_cfg) -> float | None:
    if "thickness_m" in layer_cfg:
        value = layer_cfg.thickness_m
        return None if value is None else float(value)
    if "thickness_nm" in layer_cfg:
        value = layer_cfg.thickness_nm
        return None if value is None else float(value) * 1e-9
    return None


def _position_m(position_cfg, meter_key: str, nm_key: str) -> float:
    if meter_key in position_cfg:
        return float(position_cfg[meter_key])
    if nm_key in position_cfg:
        return float(position_cfg[nm_key]) * 1e-9
    raise ValueError(f"simulation.position must define '{meter_key}' or '{nm_key}'.")


def _material_config_for_layer(layer_cfg, materials_cfg):
    if "material" in layer_cfg:
        key = str(layer_cfg.material)
        if key not in materials_cfg:
            raise KeyError(f"Layer material '{key}' is not defined under materials.")
        return materials_cfg[key]
    if "material_config" in layer_cfg:
        return layer_cfg.material_config
    raise ValueError("Each stack layer must define material or material_config.")


def build_layers(stack_cfg, materials_cfg, omega: float) -> list[LayerSpec]:
    layers = []
    for index, layer_cfg in enumerate(stack_cfg.layers):
        material_cfg = _material_config_for_layer(layer_cfg, materials_cfg)
        epsilon = DataProvider(material_cfg).get_epsilon(omega)
        layers.append(
            LayerSpec(
                epsilon=epsilon,
                thickness_m=_thickness_m(layer_cfg),
                name=str(layer_cfg.get("name", f"layer_{index}")),
            )
        )
    return layers


def _require_finite_green_results(
    total: np.ndarray,
    vacuum: np.ndarray,
    energy_eV: float,
    rx_values_m: np.ndarray,
    scattering_te: np.ndarray | None = None,
    scattering_tm: np.ndarray | None = None,
) -> None:
    datasets = [("total", total), ("vacuum", vacuum)]
    if scattering_te is not None:
        datasets.append(("scattering_te", scattering_te))
    if scattering_tm is not None:
        datasets.append(("scattering_tm", scattering_tm))
    for dataset_name, values in datasets:
        invalid = np.argwhere(~np.isfinite(values))
        if invalid.size == 0:
            continue
        rx_index, row, column = invalid[0]
        raise FloatingPointError(
            f"Non-finite {dataset_name} Green tensor at energy {energy_eV:.6f} eV, "
            f"Rx index {rx_index} ({rx_values_m[rx_index] * 1e9:.6g} nm), "
            f"component ({row}, {column}). Check stack.source_layer and integration settings."
        )


def _require_polarization_identities(
    total: np.ndarray,
    vacuum: np.ndarray,
    scattering_te: np.ndarray,
    scattering_tm: np.ndarray,
    energy_eV: float,
) -> np.ndarray:
    structure = scattering_te + scattering_tm
    if total.shape != vacuum.shape or total.shape != structure.shape:
        raise ValueError(
            "Polarization output arrays must match total/vacuum shape; got "
            f"total {total.shape}, vacuum {vacuum.shape}, TE {scattering_te.shape}, "
            f"TM {scattering_tm.shape}."
        )
    if not np.allclose(total, vacuum + structure, rtol=1e-8, atol=1e-12):
        mismatch = float(np.max(np.abs(total - vacuum - structure)))
        raise FloatingPointError(
            f"TE/TM algebraic identity failed at energy {energy_eV:.6f} eV; "
            f"max |total-vacuum-TE-TM|={mismatch:.3e}."
        )
    return structure


def _save_polarization_components(cfg) -> bool:
    output_cfg = cfg.get("output", {})
    return bool(output_cfg.get("save_polarization_components", False)) if output_cfg else False


def _energy_grid(sim_params):
    kind = sim_params.spectral_param
    if kind == "energy_eV":
        energy_ev_array = build_grid(sim_params.energy_eV)
    elif kind == "wavelength_nm":
        lambda_nm = build_grid(sim_params.wavelength_nm)
        energy_ev_array = 2 * np.pi * hbar * c / (lambda_nm * 1e-9 * eV_to_J)
        sort_idx = np.argsort(energy_ev_array)
        energy_ev_array = energy_ev_array[sort_idx]
        lambda_nm = lambda_nm[sort_idx]
        energy_ev_array, _ = _deduplicate_sorted_grid(energy_ev_array, lambda_nm)
    else:
        raise ValueError(f"Unknown spectral_param: {kind}")
    target_lambdas_m = 2 * np.pi * hbar * c / (energy_ev_array * eV_to_J)
    return energy_ev_array, target_lambdas_m


def _compute_one_energy(
    idx: int,
    energy_eV: float,
    lambda_m: float,
    rx_values_m: np.ndarray,
    z_observer: float,
    z_source: float,
    stack_cfg,
    materials_cfg,
    integ_cfg,
    rx_progress_desc: str | None = None,
    save_polarization_components: bool = False,
):
    omega = 2 * np.pi * c / lambda_m
    integration_method = (
        "direct"
        if integ_cfg is None
        else str(integ_cfg.get("method", "direct")).strip().lower()
    )
    if save_polarization_components and integration_method == "componentwise":
        raise ValueError(
            "output.save_polarization_components=true is not supported with "
            "simulation.integration.method=componentwise. Componentwise integrates assembled "
            "tensor entries and cannot safely return same-run TE/TM scattering components."
        )
    layers = build_layers(stack_cfg, materials_cfg, omega)
    calculator = NLayerGreenFunction(
        layers=layers,
        source_layer=int(stack_cfg.source_layer),
        omega=omega,
        qmax=None if integ_cfg is None else integ_cfg.qmax,
        qmax_factor=None if integ_cfg is None else integ_cfg.get("qmax_factor", None),
        epsabs=1e-9 if integ_cfg is None else float(integ_cfg.epsabs),
        epsrel=1e-9 if integ_cfg is None else float(integ_cfg.epsrel),
        limit=400 if integ_cfg is None else int(integ_cfg.limit),
        split_propagating=False if integ_cfg is None else bool(integ_cfg.split_propagating),
        integration_method=integration_method,
        dcim_q_start=0.0 if integ_cfg is None else float(integ_cfg.get("dcim_q_start", 0.0)),
        dcim_q_stop=None if integ_cfg is None else integ_cfg.get("dcim_q_stop", integ_cfg.qmax),
        dcim_q_start_factor=None if integ_cfg is None else integ_cfg.get("dcim_q_start_factor", None),
        dcim_q_stop_factor=None if integ_cfg is None else integ_cfg.get("dcim_q_stop_factor", None),
        dcim_sample_count=128 if integ_cfg is None else int(integ_cfg.get("dcim_sample_count", 128)),
        dcim_image_count=16 if integ_cfg is None else int(integ_cfg.get("dcim_image_count", 16)),
        hybrid_direct_q_stop=None if integ_cfg is None else integ_cfg.get("hybrid_direct_q_stop", None),
        hybrid_tail_q_stop=None if integ_cfg is None else integ_cfg.get("hybrid_tail_q_stop", None),
        hybrid_direct_q_stop_factor=None if integ_cfg is None else integ_cfg.get("hybrid_direct_q_stop_factor", None),
        hybrid_tail_q_stop_factor=None if integ_cfg is None else integ_cfg.get("hybrid_tail_q_stop_factor", None),
        hybrid_sample_count=None if integ_cfg is None else integ_cfg.get("hybrid_sample_count", None),
        hybrid_image_count=None if integ_cfg is None else integ_cfg.get("hybrid_image_count", None),
        hybrid_validation_rtol=5e-2 if integ_cfg is None else float(integ_cfg.get("hybrid_validation_rtol", 5e-2)),
        hybrid_validate_tail=True if integ_cfg is None else bool(integ_cfg.get("hybrid_validate_tail", True)),
        hybrid_fallback_to_direct=True if integ_cfg is None else bool(integ_cfg.get("hybrid_fallback_to_direct", True)),
        fixed_grid_points=2048 if integ_cfg is None else int(integ_cfg.get("fixed_grid_points", 2048)),
        pole_search_real_min_factor=0.0 if integ_cfg is None else float(integ_cfg.get("pole_search_real_min_factor", 0.0)),
        pole_search_real_max_factor=6.0 if integ_cfg is None else float(integ_cfg.get("pole_search_real_max_factor", 6.0)),
        pole_search_imag_min_factor=-2.0 if integ_cfg is None else float(integ_cfg.get("pole_search_imag_min_factor", -2.0)),
        pole_search_imag_max_factor=1e-3 if integ_cfg is None else float(integ_cfg.get("pole_search_imag_max_factor", 1e-3)),
        pole_search_max_depth=10 if integ_cfg is None else int(integ_cfg.get("pole_search_max_depth", 10)),
        pole_search_contour_points=24 if integ_cfg is None else int(integ_cfg.get("pole_search_contour_points", 24)),
        pole_search_residual_tol=1e-6 if integ_cfg is None else float(integ_cfg.get("pole_search_residual_tol", 1e-6)),
        pole_search_branch_guard_factor=1e-4 if integ_cfg is None else float(integ_cfg.get("pole_search_branch_guard_factor", 1e-4)),
        pole_residue_radius_factor=1e-4 if integ_cfg is None else float(integ_cfg.get("pole_residue_radius_factor", 1e-4)),
        pole_residue_points=96 if integ_cfg is None else int(integ_cfg.get("pole_residue_points", 96)),
        branch_cut_t_limit_factor=8.0 if integ_cfg is None else float(integ_cfg.get("branch_cut_t_limit_factor", 8.0)),
        branch_cut_side_offset_factor=1e-6 if integ_cfg is None else float(integ_cfg.get("branch_cut_side_offset_factor", 1e-6)),
        branch_cut_layers="auto" if integ_cfg is None else integ_cfg.get("branch_cut_layers", "auto"),
        branch_cut_sample_count=128 if integ_cfg is None else int(integ_cfg.get("branch_cut_sample_count", 128)),
        branch_cut_image_count=16 if integ_cfg is None else int(integ_cfg.get("branch_cut_image_count", 16)),
        branch_cut_validation_rtol=5e-2 if integ_cfg is None else float(integ_cfg.get("branch_cut_validation_rtol", 5e-2)),
        branch_cut_validate=True if integ_cfg is None else bool(integ_cfg.get("branch_cut_validate", True)),
        branch_cut_fallback_to_singularity_aware=True if integ_cfg is None else bool(integ_cfg.get("branch_cut_fallback_to_singularity_aware", True)),
        branch_cut_include_poles=True if integ_cfg is None else bool(integ_cfg.get("branch_cut_include_poles", True)),
        branch_cut_prefactor=1.0 + 0.0j if integ_cfg is None else complex(integ_cfg.get("branch_cut_prefactor", 1.0 + 0.0j)),
        branch_cut_jump_sign=1.0 if integ_cfg is None else float(integ_cfg.get("branch_cut_jump_sign", 1.0)),
        branch_cut_use_hankel=True if integ_cfg is None else bool(integ_cfg.get("branch_cut_use_hankel", True)),
        pole_subtraction_validate=True if integ_cfg is None else bool(integ_cfg.get("pole_subtraction_validate", True)),
        pole_subtraction_validation_rtol=5e-2 if integ_cfg is None else float(integ_cfg.get("pole_subtraction_validation_rtol", 5e-2)),
        pole_subtraction_validation_atol=1e-12 if integ_cfg is None else float(integ_cfg.get("pole_subtraction_validation_atol", 1e-12)),
        pole_subtraction_fallback_to_singularity_aware=True if integ_cfg is None else bool(integ_cfg.get("pole_subtraction_fallback_to_singularity_aware", True)),
        pole_subtraction_include_poles=True if integ_cfg is None else bool(integ_cfg.get("pole_subtraction_include_poles", True)),
        pole_subtraction_residue_method="contour" if integ_cfg is None else str(integ_cfg.get("pole_subtraction_residue_method", "contour")),
    )

    total = np.zeros((len(rx_values_m), 3, 3), dtype=complex)
    vacuum = np.zeros_like(total)
    scattering_te = np.zeros_like(total) if save_polarization_components else None
    scattering_tm = np.zeros_like(total) if save_polarization_components else None
    if integration_method == "fixed_grid":
        if save_polarization_components:
            total[:], vacuum[:], scattering_te[:], scattering_tm[:] = (
                calculator.calculate_Green_function_components_for_x_values(
                    rx_values_m,
                    y=0.0,
                    z_observer=z_observer,
                    z_source=z_source,
                )
            )
        else:
            total[:] = calculator.calculate_total_Green_functions_for_x_values(
                rx_values_m,
                y=0.0,
                z_observer=z_observer,
                z_source=z_source,
            )
            for rx_index, rx_m in enumerate(rx_values_m):
                vacuum[rx_index] = calculator.vacuum_component(
                    x=rx_m,
                    y=0.0,
                    z_observer=z_observer,
                    z_source=z_source,
                )
        _require_finite_green_results(total, vacuum, energy_eV, rx_values_m, scattering_te, scattering_tm)
        if save_polarization_components:
            structure = _require_polarization_identities(
                total,
                vacuum,
                scattering_te,
                scattering_tm,
                energy_eV,
            )
            return idx, total, vacuum, structure, scattering_te, scattering_tm
        return idx, total, vacuum

    rx_iter = enumerate(rx_values_m)
    if rx_progress_desc is not None:
        rx_iter = enumerate(tqdm(rx_values_m, desc=rx_progress_desc, ncols=100, leave=False))

    for rx_index, rx_m in rx_iter:
        if save_polarization_components:
            (
                total[rx_index],
                vacuum[rx_index],
                scattering_te[rx_index],
                scattering_tm[rx_index],
            ) = calculator.calculate_Green_function_components(
                x=rx_m,
                y=0.0,
                z_observer=z_observer,
                z_source=z_source,
            )
        else:
            total[rx_index] = calculator.calculate_total_Green_function(
                x=rx_m,
                y=0.0,
                z_observer=z_observer,
                z_source=z_source,
            )
            vacuum[rx_index] = calculator.vacuum_component(
                x=rx_m,
                y=0.0,
                z_observer=z_observer,
                z_source=z_source,
            )

    _require_finite_green_results(total, vacuum, energy_eV, rx_values_m, scattering_te, scattering_tm)
    if save_polarization_components:
        structure = _require_polarization_identities(
            total,
            vacuum,
            scattering_te,
            scattering_tm,
            energy_eV,
        )
        return idx, total, vacuum, structure, scattering_te, scattering_tm
    return idx, total, vacuum


def _run_sequential(energy_ev_array, target_lambdas_m, rx_values_m, cfg):
    results_total = np.zeros((len(energy_ev_array), len(rx_values_m), 3, 3), dtype=complex)
    results_vacuum = np.zeros_like(results_total)
    save_components = _save_polarization_components(cfg)
    results_structure = np.zeros_like(results_total) if save_components else None
    results_scattering_te = np.zeros_like(results_total) if save_components else None
    results_scattering_tm = np.zeros_like(results_total) if save_components else None
    sim_params = cfg.simulation
    z_source = _position_m(sim_params.position, "zD", "zD_nm")
    z_observer = _position_m(sim_params.position, "zA", "zA_nm")
    show_rx_progress = len(energy_ev_array) == 1 and len(rx_values_m) > 1

    for energy_index in tqdm(range(len(energy_ev_array)), desc="Energies", ncols=100):
        logger.info("N-layer energy {}/{}: {:.3f} eV", energy_index + 1, len(energy_ev_array), energy_ev_array[energy_index])
        result = _compute_one_energy(
            idx=energy_index,
            energy_eV=energy_ev_array[energy_index],
            lambda_m=target_lambdas_m[energy_index],
            rx_values_m=rx_values_m,
            z_observer=z_observer,
            z_source=z_source,
            stack_cfg=cfg.stack,
            materials_cfg=cfg.materials,
            integ_cfg=getattr(sim_params, "integration", None),
            rx_progress_desc="Rx points" if show_rx_progress else None,
            save_polarization_components=save_components,
        )
        if save_components:
            _, total, vacuum, structure, scattering_te, scattering_tm = result
            results_structure[energy_index] = structure
            results_scattering_te[energy_index] = scattering_te
            results_scattering_tm[energy_index] = scattering_tm
        else:
            _, total, vacuum = result
        results_total[energy_index] = total
        results_vacuum[energy_index] = vacuum

    if save_components:
        return results_total, results_vacuum, results_structure, results_scattering_te, results_scattering_tm
    return results_total, results_vacuum


def _run_joblib(energy_ev_array, target_lambdas_m, rx_values_m, cfg, n_jobs: int):
    from joblib import Parallel, delayed
    from mqed.utils.joblib_track import tqdm_joblib

    sim_params = cfg.simulation
    save_components = _save_polarization_components(cfg)
    z_source = _position_m(sim_params.position, "zD", "zD_nm")
    z_observer = _position_m(sim_params.position, "zA", "zA_nm")
    stack_plain = OmegaConf.create(OmegaConf.to_container(cfg.stack, resolve=True))
    materials_plain = OmegaConf.create(OmegaConf.to_container(cfg.materials, resolve=True))
    integ_cfg = getattr(sim_params, "integration", None)
    integ_plain = OmegaConf.create(OmegaConf.to_container(integ_cfg, resolve=True)) if integ_cfg else None

    with tqdm_joblib(tqdm(total=len(energy_ev_array), desc="Energies (joblib)", ncols=100)):
        raw_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_compute_one_energy)(
                idx=energy_index,
                energy_eV=energy_ev_array[energy_index],
                lambda_m=target_lambdas_m[energy_index],
                rx_values_m=rx_values_m,
                z_observer=z_observer,
                z_source=z_source,
                stack_cfg=stack_plain,
                materials_cfg=materials_plain,
                integ_cfg=integ_plain,
                save_polarization_components=save_components,
            )
            for energy_index in range(len(energy_ev_array))
        )

    results_total = np.zeros((len(energy_ev_array), len(rx_values_m), 3, 3), dtype=complex)
    results_vacuum = np.zeros_like(results_total)
    if save_components:
        results_structure = np.zeros_like(results_total)
        results_scattering_te = np.zeros_like(results_total)
        results_scattering_tm = np.zeros_like(results_total)
        for idx, total, vacuum, structure, scattering_te, scattering_tm in raw_results:
            results_total[idx] = total
            results_vacuum[idx] = vacuum
            results_structure[idx] = structure
            results_scattering_te[idx] = scattering_te
            results_scattering_tm[idx] = scattering_tm
        return results_total, results_vacuum, results_structure, results_scattering_te, results_scattering_tm
    for idx, total, vacuum in raw_results:
        results_total[idx] = total
        results_vacuum[idx] = vacuum
    return results_total, results_vacuum


def _run_mpi(energy_ev_array, target_lambdas_m, rx_values_m, cfg):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_energy = len(energy_ev_array)
    n_rx = len(rx_values_m)
    sim_params = cfg.simulation
    save_components = _save_polarization_components(cfg)
    z_source = _position_m(sim_params.position, "zD", "zD_nm")
    z_observer = _position_m(sim_params.position, "zA", "zA_nm")
    stack_plain = OmegaConf.create(OmegaConf.to_container(cfg.stack, resolve=True))
    materials_plain = OmegaConf.create(OmegaConf.to_container(cfg.materials, resolve=True))
    integ_cfg = getattr(sim_params, "integration", None)
    integ_plain = OmegaConf.create(OmegaConf.to_container(integ_cfg, resolve=True)) if integ_cfg else None

    local_indices = list(range(rank, n_energy, size))
    if rank == 0:
        logger.info(
            "MPI backend: {} ranks, {} energies (rank 0 handles {} energies)",
            size,
            n_energy,
            len(local_indices),
        )

    local_results = []
    for energy_index in local_indices:
        logger.info(
            "[rank {}] N-layer energy {}/{}: {:.3f} eV",
            rank,
            energy_index + 1,
            n_energy,
            energy_ev_array[energy_index],
        )
        local_results.append(
            _compute_one_energy(
                idx=energy_index,
                energy_eV=energy_ev_array[energy_index],
                lambda_m=target_lambdas_m[energy_index],
                rx_values_m=rx_values_m,
                z_observer=z_observer,
                z_source=z_source,
                stack_cfg=stack_plain,
                materials_cfg=materials_plain,
                integ_cfg=integ_plain,
                save_polarization_components=save_components,
            )
        )

    all_results = comm.gather(local_results, root=0)
    if rank != 0:
        return None, None

    results_total = np.zeros((n_energy, n_rx, 3, 3), dtype=complex)
    results_vacuum = np.zeros_like(results_total)
    if save_components:
        results_structure = np.zeros_like(results_total)
        results_scattering_te = np.zeros_like(results_total)
        results_scattering_tm = np.zeros_like(results_total)
        for rank_results in all_results:
            for idx, total, vacuum, structure, scattering_te, scattering_tm in rank_results:
                results_total[idx] = total
                results_vacuum[idx] = vacuum
                results_structure[idx] = structure
                results_scattering_te[idx] = scattering_te
                results_scattering_tm[idx] = scattering_tm
        return results_total, results_vacuum, results_structure, results_scattering_te, results_scattering_tm
    for rank_results in all_results:
        for idx, total, vacuum in rank_results:
            results_total[idx] = total
            results_vacuum[idx] = vacuum
    return results_total, results_vacuum


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("Dyadic_GF", __file__)


@hydra.main(config_path=HYDRA_CONFIG_PATH, config_name="GF_NLayer_five_layer", version_base=None)
def run_simulation(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()
    logger.info("--- Starting N-layer Green's Function Simulation ---")

    sim_params = cfg.simulation
    integ_cfg = getattr(sim_params, "integration", None)
    integration_method = "direct" if integ_cfg is None else str(integ_cfg.get("method", "direct"))
    save_components = _save_polarization_components(cfg)
    if save_components and integration_method.strip().lower() == "componentwise":
        raise ValueError(
            "output.save_polarization_components=true is not supported with "
            "simulation.integration.method=componentwise. Use a non-componentwise method."
        )
    if integration_method.strip().lower() not in {
        "direct",
        "dcim",
        "hybrid_dcim",
        "fixed_grid",
        "componentwise",
        "singularity_aware",
        "branch_cut_dcim",
        "pole_subtracted_direct",
        "pole_aware_hybrid_dcim",
    }:
        raise ValueError(
            "simulation.integration.method must be 'direct', 'dcim', 'hybrid_dcim', "
            "'fixed_grid', 'componentwise', 'singularity_aware', 'branch_cut_dcim', "
            "'pole_subtracted_direct', or 'pole_aware_hybrid_dcim'."
        )
    energy_ev_array, target_lambdas_m = _energy_grid(sim_params)
    rx_values_nm = build_position_grid(sim_params.position.Rx_nm)
    rx_values_m = rx_values_nm * 1e-9
    parallel_cfg = cfg.get("parallel", {})
    backend = parallel_cfg.get("backend", "sequential") if parallel_cfg else "sequential"
    logger.info("Grid: {} energies × {} Rx points | backend={}", len(energy_ev_array), len(rx_values_m), backend)

    if backend == "mpi":
        _maybe_auto_launch_mpi(parallel_cfg)

    if backend == "sequential":
        run_results = _run_sequential(
            energy_ev_array,
            target_lambdas_m,
            rx_values_m,
            cfg,
        )
    elif backend == "joblib":
        run_results = _run_joblib(
            energy_ev_array,
            target_lambdas_m,
            rx_values_m,
            cfg,
            int(parallel_cfg.get("n_jobs", -1)),
        )
    elif backend == "mpi":
        run_results = _run_mpi(
            energy_ev_array,
            target_lambdas_m,
            rx_values_m,
            cfg,
        )
        try:
            from mpi4py import MPI

            if MPI.COMM_WORLD.Get_rank() != 0:
                return
        except ImportError:
            pass
    else:
        raise ValueError("N-layer runner supports parallel.backend values: sequential, joblib, mpi.")

    if save_components:
        results_total, results_vacuum, results_structure, results_scattering_te, results_scattering_tm = run_results
    else:
        results_total, results_vacuum = run_results
        results_structure = None
        results_scattering_te = None
        results_scattering_tm = None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = cfg.output.get("filename", None)
    if output_filename is None:
        output_filename = (
            f"{cfg.output.prefix}"
            f"_Emin_{energy_ev_array[0]:.2f}_Emax_{energy_ev_array[-1]:.2f}_"
            f"{len(energy_ev_array)}pts"
            f"_Rx_{rx_values_nm[-1]:.0f}nm_{len(rx_values_nm)}pts.hdf5"
        )
    output_file = output_dir / str(output_filename)
    save_gf_h5(
        output_file,
        results_total,
        results_vacuum,
        energy_ev_array,
        rx_values_nm,
        _position_m(sim_params.position, "zD", "zD_nm"),
        _position_m(sim_params.position, "zA", "zA_nm"),
        Gstructure=results_structure,
        G_scattering_te=results_scattering_te,
        G_scattering_tm=results_scattering_tm,
        attrs={"save_polarization_components": 1} if save_components else None,
    )
    logger.success("N-layer simulation complete. Output saved to: {}", output_file.absolute())


if __name__ == "__main__":
    run_simulation()
