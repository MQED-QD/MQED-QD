import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import h5py
import matplotlib
# Force Matplotlib to use a non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
from mqed.utils.orientation import spherical_to_cartesian_dipole, resolve_angle_deg # NEW IMPORT
from mqed.utils.dgf_data import load_gf_h5 # NEW IMPORT
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from pathlib import Path
from hydra.utils import get_original_cwd # <-- IMPORT THIS
from hydra.core.hydra_config import HydraConfig

HBAR_EVS = 6.582119569e-16 # Planck's constant in eV*s
C_CMS = 2.99792458e10      # Speed of light in cm/s
eV_to_J = 1.60218e-19 # eV to Joules conversion

def compute_enhancement(p_donor,p_acceptor,g_total, g_vac):
    """
    Computes the enhancement factor for Resonance Energy Transfer (RET)
    given donor and acceptor dipole orientations and Green's functions.

    Args:
        p_donot (np.ndarray): Donor dipole orientation (3-element array).
        p_acceptor (np.ndarray): Acceptor dipole orientation (3-element array).
        g_total (np.ndarray): Total Green's function array of shape (M, N, 3, 3).
        g_vac (np.ndarray): Vacuum Green's function array of shape (M, N, 3, 3).
    Returns:
        gamma (np.ndarray): Enhancement factor array of shape (M, N).
        E_enhance_real (np.ndarray): Real part of field enhancement array of shape (M, N).
        E_enhance_imag (np.ndarray): Imaginary part of field enhancement array of shape (M, N).
    """
    # Project the Green's functions onto the dipole orientations
    # This is the NumPy equivalent of the MATLAB code
    # g_da = p_A^T * G * p_D
    g_da_total = np.einsum('i,...ij,j->...', p_acceptor, g_total, p_donor)
    g_da_vac = np.einsum('i,...ij,j->...', p_acceptor, g_vac, p_donor)

    # Calculate enhancement factor
    gamma = np.abs(g_da_total / g_da_vac)**2
    E_enhance_real = np.real(g_da_total)/np.real(g_da_vac)
    E_enhance_imag = np.imag(g_da_total)/np.imag(g_da_vac)
    return gamma, E_enhance_real, E_enhance_imag


def calculate_enhancement(cfg: DictConfig):
    """
    Loads simulation data and calculates the Resonance Energy Transfer (RET)
    enhancement factor.
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()
    

    input_path = Path(cfg.input_file)

    # Make the path absolute if it's not already
    if not input_path.is_absolute():
        input_path = Path(get_original_cwd()) / input_path
    
    logger.info(f"Attempting to load data from: {input_path}")

    try:
        # with h5py.File(input_path, 'r') as f:
        #     # Load the required datasets
            
        #     g_total = f['green_function_total'][:]
        #     g_vac = f['green_function_vacuum'][:]
        #     energy_ev = f['energy_eV'][:]
        #     rx_nm = f['Rx_nm'][:]
        g_total, g_vac, energy_ev, rx_nm, zD, zA = load_gf_h5(str(input_path)).values()
        logger.success("Successfully loaded data file.")

    except FileNotFoundError:
        # This block runs ONLY if the file doesn't exist at the path.
        logger.warning(f"File not found at the specified path: {input_path}")
        logger.warning("Please check the 'input_file' path in your config or ensure the simulation was run first.")
        return # Exit the function gracefully to prevent further errors.
    
    except Exception as e:
        # Catch any other potential errors during file reading (e.g., corrupted file)
        logger.error(f"An unexpected error occurred while reading the HDF5 file: {e}")
        return
    
    phi_donor   = resolve_angle_deg(cfg.orientations.donor.phi_deg)
    phi_acceptor  = resolve_angle_deg(cfg.orientations.acceptor.phi_deg)
    theta_donor = float(cfg.orientations.donor.theta_deg)
    theta_acceptor= float(cfg.orientations.acceptor.theta_deg)
    # Convert angles to Cartesian vectors
    p_donor = spherical_to_cartesian_dipole(theta_donor,
                                            phi_donor)
    p_acceptor = spherical_to_cartesian_dipole(theta_acceptor,
                                            phi_acceptor)

    # Project the Green's functions onto the dipole orientations
    # This is the NumPy equivalent of the MATLAB code
    # g_da = p_A^T * G * p_D
    # g_da_total = np.einsum('i,...ij,j->...', p_acceptor, g_total, p_donor)
    # g_da_vac = np.einsum('i,...ij,j->...', p_acceptor, g_vac, p_donor)

    # Calculate enhancement factor
    # enhancement_factor = np.abs(g_da_total / g_da_vac)**2
    # enhancement_field_real = np.real(g_da_total)/np.real(g_da_vac)
    # enhancement_field_imag = np.imag(g_da_total)/np.imag(g_da_vac)
    enhancement_factor, enhancement_field_real, enhancement_field_imag = compute_enhancement(p_donor, p_acceptor, g_total, g_vac)
    

    # Loop through each energy point to create a separate plot for each
    ps = cfg.plot_settings

    for i, energy in enumerate(energy_ev):
        enhancement_slice = enhancement_factor[i, :]  # if you still need it
            # -------- choose x-range --------
    # Option A: pick by indices (inclusive slice semantics like [0, 9])
        if hasattr(ps, "x_index_range") and ps.x_index_range:
            i0, i1 = int(ps.x_index_range[0]), int(ps.x_index_range[1])
            sel = np.zeros_like(rx_nm, dtype=bool)
            sel[i0:i1+1] = True
        # Option B: pick by values in nm (inclusive)
        elif hasattr(ps, "x_range_nm") and ps.x_range_nm:
            xmin, xmax = float(ps.x_range_nm[0]), float(ps.x_range_nm[1])
            sel = (rx_nm >= xmin) & (rx_nm <= xmax)
        else:
            sel = np.ones_like(rx_nm, dtype=bool)

        x = rx_nm[sel]
        y_real = enhancement_field_real[i, sel]
        y_imag = enhancement_field_imag[i, sel]

        # -------- choose components to plot --------
        # components can be: ["real"], ["imag"], or ["real","imag"]
        components = getattr(ps, "components", ["real", "imag"])

        fig, ax = plt.subplots(figsize=(8, 6))

        if "real" in components:
            ax.plot(x, y_real, getattr(ps, "real_style", "r--"),
                lw=getattr(ps, "lw", 1),
                label=ps.legend.real_label)
        if "imag" in components:
            ax.plot(x, y_imag, getattr(ps, "imag_style", "b--"),
                lw=getattr(ps, "lw", 1),
                label=ps.legend.imag_label)
        # ax.plot(rx_nm, enhancement_field_real[i, :], 'r--', lw=1, label=ps.legend.real_label)
        # ax.plot(rx_nm, enhancement_field_imag[i, :], 'b--', lw=1, label=ps.legend.imag_label)

        ax.set_xlabel(ps.xlabel)
        ax.set_ylabel(ps.ylabel)

        # Title string supports {energy:.3f} formatting
        ax.set_title(ps.title_template.format(energy=energy))

        if getattr(ps, "xscale", "linear"):
            ax.set_xscale(ps.xscale)
        if getattr(ps, "yscale", "linear"):
            ax.set_yscale(ps.yscale)

        # Optional axis limits (override selection if you want)
        if hasattr(ps, "xlim") and ps.xlim: ax.set_xlim(ps.xlim[0], ps.xlim[1])
        if hasattr(ps, "ylim") and ps.ylim: ax.set_ylim(ps.ylim[0], ps.ylim[1])

        if getattr(ps, "grid", True):
            ax.grid(True, which="both", ls="--")

        ax.legend()

        if ps.save_plot:
            plot_filename = ps.filename_prefix + f"_{energy:.3f}eV.png"
            plot_filepath = output_dir / plot_filename
            fig.savefig(plot_filepath, dpi=ps.dpi, bbox_inches="tight")
            logger.info(f"Plot saved to {plot_filepath}")
        plt.close(fig)
    logger.success(f"RET enhancement calculation complete. Logs saved to: {output_dir.absolute()}")

@hydra.main(config_path="../../configs/analysis", config_name="RET", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. Get the output directory managed by Hydra
    calculate_enhancement(cfg)


if __name__ == "__main__":
    main()