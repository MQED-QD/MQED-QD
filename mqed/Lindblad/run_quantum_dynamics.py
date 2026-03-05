import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from qutip import fock, fock_dm
from typing import Any, Dict, Optional, Tuple

from loguru import logger
from omegaconf import DictConfig

from mqed.Lindblad.quantum_dynamics import LindbladDynamics, NonHermitianSchDynamics, SimulationConfig
from mqed.Lindblad.quantum_operator import ipr_callable, msd_operator, position_operator, site_population_operator, x_shift_conditional_callable, x_shift2_conditional_callable
from mqed.utils.dgf_data import load_gf_h5
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.utils.save_hdf5 import save_dx_h5

def build_observable(item: Dict[str, Any], *, dim: int, d_nm: float, Nmol: int, init_site: int) -> Tuple[str, object]:
    """
    Turn one YAML 'observables' item into (key, obj_or_callable) for QuTiP.
    - If 'kind' == 'operator'  -> return Qobj
    - If 'kind' == 'callable'  -> return f(t, state) callable
    """
    name   = str(item["name"])
    kind   = str(item.get("kind", "operator"))
    params = dict(item.get("params", {}) or {})
    if name in {"X_shift_cond", "X_shift2_cond", "IPR_site"} and kind != "callable":
        logger.warning(f"{name} is usually used with kind='callable' (got {kind!r})")


    if name == "X_shift":
        logger.info(f"Adding observable: position operator (X_shift)")
        return name, position_operator(dim, d_nm, Nmol, init_site)
    if name == "X_shift2":
        logger.info(f"Adding observable: mean squared displacement operator (X_shift2)")
        return name, msd_operator(dim, d_nm, Nmol, init_site)
    if name == "pop_site":
        logger.info(f"Adding observable: population operator at specified site")
        site = int(params.get("site", 1))
        return f"pop_site_{site}", site_population_operator(dim, site)  # label includes site
    if name == "IPR_site":
        # bind Nmol (required); any extra params are ignored safely
        logger.info(f"Adding observable: Inverse Participation Ratio (requires Nmol param)")
        Nmol_local = int(params.get("Nmol", Nmol))
        return name, (lambda t, st: ipr_callable(t, st, Nmol=Nmol_local))

    if name == "X_shift_cond":
        logger.info("Adding observable: conditional position expectation (X_shift_cond)")
        Nmol_local = int(params.get("Nmol", Nmol))
        X_shift_op = position_operator(dim, d_nm, Nmol, init_site)  # fixed operator, normalization in callable
        return name, (
            lambda t, st, X=X_shift_op, N=Nmol_local:
                x_shift_conditional_callable(t, st, X_shift=X, Nmol=N)
        )

    if name == "X_shift2_cond":
        logger.info("Adding observable: conditional MSD moment (X_shift2_cond)")
        Nmol_local = int(params.get("Nmol", Nmol))
        X_shift2_op = msd_operator(dim, d_nm, Nmol, init_site)
        return name, (
            lambda t, st, X2=X_shift2_op, N=Nmol_local:
                x_shift2_conditional_callable(t, st, X_shift2=X2, Nmol=N)
        )

    logger.error(f"Unknown observable name: {name!r}")
    raise ValueError(f"Unknown observable name: {name!r}")


def _build_observables(
    obs_cfg,
    *,
    dim: int,
    d_nm: float,
    Nmol: int,
    init_site: int,
):
    e_ops = {}
    compute_root_msd = False

    for item in obs_cfg:
        name = str(item.get("name", ""))
        if name == "root_MSD":
            compute_root_msd = bool(item.get("enabled", True))
            logger.info(f"root_MSD output enabled: {compute_root_msd}")
            continue

        key, obj = build_observable(
            item,
            dim=dim,
            d_nm=d_nm,
            Nmol=Nmol,
            init_site=init_site,
        )
        e_ops[key] = obj

    return e_ops, compute_root_msd


def app_run(cfg: DictConfig, output_dir: Optional[Path] = None):
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
    
    logger.info(f"Dipole heights (nm): zD={data['zD']}  zA={data['zA']}")
    logger.info(f"inter-site distance (nm): {cfg.simulation.d_nm}")

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
        coupling_limit=cfg.simulation.get('coupling_limit', None),
    )


    # 3) Choose dynamics backend
    method = cfg.solver.method
    if method == 'Lindblad':
        logger.info("Using Lindblad master equation solver.")
        dyn = LindbladDynamics(sim_cfg, G_slice)
        # Lindblad expects density matrix
        rho_or_psi = fock_dm(sim_cfg.Nmol + 1, cfg.initial_state.site_index)
    elif method == 'NonHermitian':
        logger.info("Using Non-Hermitian Schrodinger Equation solver.")
        dyn = NonHermitianSchDynamics(sim_cfg, G_slice)
        # Non-Hermitian Sch. expects ket
        rho_or_psi = fock(sim_cfg.Nmol + 1, cfg.initial_state.site_index)
    else:
        logger.error(f"Unknown solver.method = {method}")
        raise ValueError(f"Unknown solver.method = {method}")


    # e_ops = {"X_shift": position_operator(sim_cfg.Nmol + 1, sim_cfg.d_nm, sim_cfg.Nmol, cfg.initial_state.site_index),
    #         "X_shift2": msd_operator(sim_cfg.Nmol + 1, sim_cfg.d_nm, sim_cfg.Nmol, cfg.initial_state.site_index),
    #         "IPR_site": lambda t, st: ipr_callable(t, st, Nmol=sim_cfg.Nmol),}

    obs_cfg = getattr(cfg, "observables", []) or []
    e_ops, compute_root_msd = _build_observables(
        obs_cfg,
        dim=sim_cfg.Nmol + 1,
        d_nm=sim_cfg.d_nm,
        Nmol=sim_cfg.Nmol,
        init_site=cfg.initial_state.site_index,
    )


    # 5) Evolve
    result = dyn.evolve(rho_or_psi, e_ops=e_ops, options=None)

    #calculate dx from expectations
    # breakpoint()
    # ex1 = np.asarray(result.expectations["X_shift"])
    # ex2 = np.asarray(result.expectations["X_shift2"])
    # dx = np.sqrt(np.maximum(0.0, ex2 - ex1**2))

    dx = None
    dx_std = None
    if compute_root_msd:
        # Prefer conditional moments if they exist (best for NHSE / fair comparisons)
        if "X_shift_cond" in result.expectations and "X_shift2_cond" in result.expectations:
            logger.info("Computing root_MSD from conditional moments: X_shift_cond, X_shift2_cond")
            x = np.asarray(result.expectations["X_shift_cond"])
            x2 = np.asarray(result.expectations["X_shift2_cond"])

        # Fallback to raw moments
        elif "X_shift" in result.expectations and "X_shift2" in result.expectations:
            logger.info("Computing root_MSD from raw moments: X_shift, X_shift2")
            x = np.asarray(result.expectations["X_shift"])
            x2 = np.asarray(result.expectations["X_shift2"])

        else:
            raise ValueError(
                "root_MSD requested, but need either "
                "('X_shift_cond','X_shift2_cond') or ('X_shift','X_shift2')."
            )

        dx = np.sqrt(np.maximum(0.0, x2 - x**2))
        dx_std = np.zeros_like(dx)


    # 6) Save to HDF5
    outfile = output_dir / cfg.output.filename
    # states = getattr(result, 'states', None)
    logger.info(f"Saving results to {outfile.absolute()}")
    mode = str(cfg.simulation.get("mode", "stationary"))
    n_realizations = 1
    if mode == "disorder" and hasattr(cfg, "disorder"):
        n_realizations = int(cfg.disorder.get("n_realizations", 1))

    save_dx_h5(
        outfile=outfile,
        t_ps=result.tlist,
        dx_mean_nm=dx,
        dx_std_nm=dx_std,
        method=method,
        mode=mode,
        n_realizations=n_realizations,
        expectations=result.expectations,
        extra_attrs={"root_MSD_enabled": compute_root_msd},
    )


    logger.success(f"Simulation complete. Output saved to: {outfile.absolute()}")

@hydra.main(config_path="../../configs/Lindblad", config_name="quantum_dynamics", version_base=None)
def mqed_lindblad(cfg: DictConfig):
    cfg.solver.method = "Lindblad"
    app_run(cfg)

@hydra.main(config_path="../../configs/Lindblad", config_name="quantum_dynamics_nhse", version_base=None)
def mqed_nhse(cfg: DictConfig):
    cfg.solver.method = "NonHermitian"
    app_run(cfg)

if __name__ == "__main__":
    mqed_nhse()
