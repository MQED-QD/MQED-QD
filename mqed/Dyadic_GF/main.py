import hydra
from omegaconf import DictConfig
import numpy as np
import h5py
from loguru import logger
from pathlib import Path
from tqdm import tqdm

# Import our custom classes
from mqed.Dyadic_GF.data_provider import DataProvider
from mqed.Dyadic_GF.GF_Sommerfeld import Greens_function_analytical
from mqed.utils.SI_unit import eV_to_J, hbar, c
from mqed.utils.dgf_data import save_gf_h5
from hydra.core.hydra_config import HydraConfig# Physical constants
from mqed.utils.logging_utils import setup_loggers_hydra_aware

def build_grid(config):
    """Builds a grid based on the configuration input.
    The config can be a single value, a list of values, or a dictionary
    specifying min, max, and number of points for a linear space.
    Examples:
        - Single value: 2.0
        - List of values: [1.0, 2.0, 3.0]
        - Dictionary for linspace: {'min': 1.0, 'max': 3.0, 'points': 5}
    Returns:
        A numpy array of the constructed grid.
    """
    if isinstance(config, (float, int)):
        return np.array([config], dtype=float)
    elif isinstance(config, list):
        return np.array(config, dtype=float)
    elif isinstance(config, dict):
        return np.linspace(config.min, config.max, config.points, dtype=float)
    else:
        raise TypeError(f"Unsupported spectral config type: {type(config)}")

def compute_gf_grid(energy_J, target_lambdas_m, rx_values_m, sim_params, data_provider):
    """Computes the Green's function grid over specified energies and Rx values.
    1. Initializes result arrays for total and vacuum Green's functions.
    2. Loops over each energy to compute the corresponding Green's functions.
    3. Stores results in pre-allocated arrays.
    Returns:
        results_total: 4D numpy array [M,N,3,3] of total Green's functions.
        results_vacuum: 4D numpy array [M,N,3,3] of vacuum Green's functions.
    """
    nE = len(energy_J)
    nR = len(rx_values_m)
    results_total = np.zeros((nE, nR, 3, 3), dtype=complex)
    results_vacuum = np.zeros((nE, nR, 3, 3), dtype=complex)

    for i, lambda_m in enumerate(tqdm(target_lambdas_m, desc="Energies", ncols=100)):
        omega = 2 * np.pi * c / lambda_m
        epsilon = data_provider.get_epsilon(omega)
        logger.info(f"Energy {i+1}/{nE}: {(energy_J[i]/eV_to_J):.3f} eV")

        calculator = Greens_function_analytical(omega=omega, metal_epsi=epsilon)

        for j, rx_m in enumerate(
            tqdm(rx_values_m, desc=f"Rx @ E[{i+1}/{nE}]", leave=False, ncols=100)
        ):
            g_tensor = calculator.calculate_total_Green_function(
                x=rx_m,
                y=0,
                z1=sim_params.position.zD,
                z2=sim_params.position.zA,
            )
            g_vacuum = calculator.vacuum_component(
                x=rx_m,
                y=0,
                z1=sim_params.position.zD,
                z2=sim_params.position.zA,
            )
            results_total[i, j] = g_tensor
            results_vacuum[i, j] = g_vacuum

    return results_total, results_vacuum


@hydra.main(config_path="../../configs/Dyadic_GF", config_name="GF_Sommerfeld", version_base=None)
def run_simulation(cfg: DictConfig) -> None:
    # --- Logging Setup ---
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()

    logger.info("--- Starting Green's Function Simulation ---")
    
    # 1. Initialize the data provider (same as before)
    data_provider = DataProvider(cfg.material)
    
    # 2. Set up the simulation parameters from the config
    # breakpoint()
    sim_params = cfg.simulation
    kind = sim_params.spectral_param
    logger.info(f"Spectral parameter kind: {kind}")

    # 3. Build the energy/wavelength grid based on user choice
    if kind == "energy_eV":
        cfg_e = sim_params.energy_eV
        logger.info(f"Spectral parameter: energy {cfg_e} (eV)")
        energy_ev_array = build_grid(cfg_e)

    elif kind == "wavelength_nm":
        cfg_l = sim_params.wavelength_nm
        logger.info(f"Spectral parameter: wavelength {cfg_l} (nm)")
        lambda_nm = build_grid(cfg_l)
        # hc in eV·nm (or use your existing eV_to_J and hbar,c if you prefer)
        
        energy_ev_array = 2*np.pi * hbar * c / (lambda_nm*1e-9* eV_to_J) 

    else:
        raise ValueError(f"Unknown spectral_param: {kind}")

    # Convert energy array to Joules and then to wavelengths
    energy_J = energy_ev_array * eV_to_J
    target_lambdas_m = 2 * np.pi * hbar * c / energy_J
    
    # Create the array of Rx values from the config
    rx_values_nm = np.linspace(
        sim_params.position.Rx_nm.start,
        sim_params.position.Rx_nm.stop,
        sim_params.position.Rx_nm.points
    )
    rx_values_m = rx_values_nm * 1e-9 # Convert to meters
    logger.info(f"Will calculate for {len(rx_values_m)} Rx distances from {rx_values_nm[0]} nm to {rx_values_nm[-1]} nm.")

    # 4. Compute the Green's function grid
    results_total, results_vacuum = compute_gf_grid(
        energy_J, target_lambdas_m, rx_values_m, sim_params, data_provider
    )

    # 5. Save the multi-dimensional results to HDF5
    output_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist
    output_file = output_dir/cfg.output.filename  #Use the full path

    save_gf_h5(output_file, results_total, results_vacuum, energy_J / eV_to_J, rx_values_nm, sim_params.position.zD, sim_params.position.zA)


    logger.success(f"Simulation complete. Output saved to: {output_file.absolute()}")

if __name__ == "__main__":
    run_simulation()