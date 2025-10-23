import hydra
from omegaconf import DictConfig
import numpy as np
import h5py
from loguru import logger
from pathlib import Path
from tqdm import tqdm

# Import our custom classes
from mqed.Dyadic_GF.data_provider import DataProvider
from mqed.Dyadic_GF.GF_analytical import Greens_function_analytical
from mqed.utils.SI_unit import eV_to_J, hbar, c
from mqed.utils.dgf_data import save_gf_h5
from hydra.core.hydra_config import HydraConfig# Physical constants
from mqed.utils.logging_utils import setup_loggers_hydra_aware


@hydra.main(config_path="../../configs/Dyadic_GF", config_name="GF_analytical", version_base=None)
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
    energy_config = sim_params.energy_eV
    
    if isinstance(energy_config, (float, int)) :
        # Case 1: A single energy value
        logger.info(f"Running for single energy point: {energy_config} eV")
        energy_ev_array = np.array([energy_config])
    elif isinstance(energy_config, list) :
        # Case 2: A list of specific energy values
        logger.info(f"Running for a list of {len(energy_config)} energy points.")
        energy_ev_array = np.array(energy_config)
    elif isinstance(energy_config, dict) :
        # Case 3: A range of energies (min, max, points)
        logger.info(f"Running for a range of {energy_config.points} energy points from {energy_config.min} to {energy_config.max} eV.")
        energy_ev_array = np.linspace(energy_config.min, energy_config.max, energy_config.points)
    else:
        raise TypeError(f"Unknown type for energy_eV configuration: {type(energy_config)}")

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

    # 3. Prepare a larger results array to store all the data
    # Dimensions: (num_energies, num_rx_points, 3, 3)
    results_total = np.zeros((len(energy_J), len(rx_values_m), 3, 3), dtype=complex)
    results_vacuum = np.zeros((len(energy_J), len(rx_values_m), 3, 3), dtype=complex)
    
    # 4. Loop through each energy and then each donor-acceptor distance
    for i, lambda_m in enumerate(tqdm(target_lambdas_m, desc="Energies", ncols=100)):
        # breakpoint()
        omega = 2 * np.pi * c / lambda_m
        epsilon = data_provider.get_epsilon(omega)
        
        logger.info(f"Running Energy Point {i+1}/{len(energy_J)} (E={(energy_J[i]/eV_to_J):.3f} eV)")
        
        # Create the calculator for 
        calculator = Greens_function_analytical(omega=omega, metal_epsi=epsilon)
        
        # Inner loop for the Rx parameter sweep 
        for j, rx_m in enumerate(tqdm(rx_values_m, desc=f"Rx @ E[{i+1}/{len(target_lambdas_m)}]", leave=False, ncols=100)):
            g_tensor = calculator.calculate_total_Green_function(
                x=rx_m,
                y=0,
                z1=sim_params.position.zD,
                z2=sim_params.position.zA
            )
            g_vacuum = calculator.vacuum_component(
                x = rx_m,
                y = 0,
                z1=sim_params.position.zD,
                z2=sim_params.position.zA
            )
            results_total[i, j, :, :] = g_tensor
            results_vacuum[i, j, :, :] = g_vacuum

    # 5. Save the multi-dimensional results to HDF5
    # from hydra.core.hydra_config import HydraConfig
    # output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist
    output_file = output_dir/cfg.output.filename  #Use the full path

    save_gf_h5(output_file, results_total, results_vacuum, energy_J / eV_to_J, rx_values_nm, sim_params.position.zD, sim_params.position.zA)

    # with h5py.File(output_file, 'w') as f:
    #     logger.info(f"Saving results to {output_file}")
    #     f.create_dataset('green_function_total', data=results_total)
    #     f.create_dataset('green_function_vacuum',data= results_vacuum) #later
    #     f.create_dataset('energy_eV', data=(energy_J / eV_to_J))
    #     f.create_dataset('Rx_nm', data=rx_values_nm) # Save the Rx values used
        
    #     pos_group = f.create_group('position_fixed')
    #     pos_group.attrs['zD_meters'] = sim_params.position.zD
    #     pos_group.attrs['zA_meters'] = sim_params.position.zA

    logger.success(f"Simulation complete. Output saved to: {output_file.absolute()}")

if __name__ == "__main__":
    run_simulation()