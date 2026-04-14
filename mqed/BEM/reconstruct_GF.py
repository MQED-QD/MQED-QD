'''Reconstruct the dyadic Green's function from BEM electric field data.

Two reconstruction modes are supported:

**Separation-indexed** (planar / translational symmetry):
    BEM provides the electric field for a single dipole at various
    separations Rx.  Translational symmetry is assumed — all emitter
    pairs at the same separation share the same Green's function.  A
    single self-term (Purcell factor at Rx = 0) is used for all sites.
    Output: HDF5 with ``gf_layout = "separation"``.

**Pair-indexed** (nanorod / arbitrary geometry):
    BEM provides the electric field for dipoles placed at each emitter
    site independently.  Each emitter has its own Purcell factor and
    inter-site coupling.  No translational symmetry assumed.
    Output: HDF5 with ``gf_layout = "pair"``.

The :func:`build_and_save` function handles separation-indexed data.
The :func:`build_and_save_pair` function handles pair-indexed data.
'''

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig

from loguru import logger
from mqed.utils.BEM_tools import read_bem_dyadic, read_peff, read_purcell_sheet
from mqed.utils.dgf_data import save_gf_h5, save_gf_pair_h5
from mqed.BEM.compute_peff import omega_from_lambda_nm
from mqed.utils.SI_unit import c, hbar, eV_to_J
from mqed.Dyadic_GF.GF_Sommerfeld import Greens_function_analytical
from mqed.utils.logging_utils import setup_loggers_hydra_aware

def build_and_save(
    xlsx_path: str,
    out_h5: str,
    zD_m: float,
    zA_m: float,
    energy_eV: float,
    p_eff_path: str,   # if your Excel dyadic is (G * p_eff), divide by p_eff here
):
    logger.info(f"Reading BEM dyadic Green's function from {xlsx_path}...")
    rx_nm_pos, G_pos = read_bem_dyadic(xlsx_path, "DyadicG")
    lam_nm, Fx, Fy, Fz = read_purcell_sheet(xlsx_path, "G_self")
    omega = omega_from_lambda_nm(lam_nm)

    logger.info(f"Reading p_eff from {p_eff_path}...")
    p_eff = read_peff(p_eff_path, lambda_nm=lam_nm)  
    
    # Verify if the BEM scaling matches with GSI units 
    # eps0 = 8.8541878128e-12;   # F/m
    # c0   = 299792458;          # m/s
    # lambda_m = 665 * 1e-9;  # nm
    # omega = 2*np.pi*c0 / lambda_m;   # s^-1
    # pref = eps0 * c0**2 / omega**2; #prefactor of calculating GF from electric field.
    # p_bem =pref/p_eff/np.pi
    # p_GSI = 2.998e11
    # print(f"p_eff from BEM: {p_bem:.3e}, p_eff from GSI: {p_GSI:.3e}")
    # breakpoint()

    # Convert dyadic from stored (G * p_eff) -> G, then optional s-correction
    G_pos = (G_pos / p_eff) 

    logger.info("Computing self term at Rx=0 from Purcell factors...")
    # --- self term at Rx=0 ---
    G_self = np.zeros((3,3), dtype=np.complex128)
    pref_self = omega/(6*np.pi*c)
    G_self[0,0] = 1j * pref_self * Fx
    G_self[1,1] = 1j * pref_self * Fy
    G_self[2,2] = 1j * pref_self * Fz
    # off-diagonals kept 0

    # --- vacuum dyadic (optional) ---
    # If you don't have vacuum for Rx>0, you can set it to zeros or compute analytically elsewhere.
    
    logger.info("Preparing total Green's function array...")
    # breakpoint()
    if rx_nm_pos[0] != 0.0:
        logger.info("The first position is not zero, shift positions accordingly.")
        rx_nm_pos = rx_nm_pos - rx_nm_pos[0] +1
    logger.info(f"Final positions (nm): {rx_nm_pos}")
    # --- prepend Rx=0 ---
    Rxnm = np.concatenate(([0.0], rx_nm_pos))
    Gtot = np.zeros((1, len(Rxnm), 3, 3), dtype=np.complex128)
    Gvac = np.zeros((1, len(Rxnm), 3, 3), dtype=np.complex128)

    Gtot[0,0,:,:] = G_self
    Gtot[0,1:,:,:] = G_pos

    logger.info('Calculate Vacuum components from Greens_function_analytical...')
    calculator = Greens_function_analytical(metal_epsi=1.0 + 0.0j, omega=omega)
    for j in range(len(Rxnm)):
        g_vacuum = calculator.vacuum_component(
                x = Rxnm[j]*1e-9,  # convert nm to m
                y = 0,
                z1=zD_m,
                z2=zA_m
            )
        Gvac[0,j,:,:] = g_vacuum

    E = np.array([float(energy_eV)], dtype=float)
    save_gf_h5(out_h5, Gtot, Gvac, E, Rxnm, zD_m, zA_m)
    

def build_and_save_pair(
    xlsx_paths: list,
    out_h5: str,
    zD_m: float,
    zA_m: float,
    energy_eV: float,
    p_eff_path: str,
    emitter_positions_nm: np.ndarray,
):
    r"""Reconstruct pair-indexed Green's function from per-emitter BEM data.

    For geometries without translational symmetry (e.g. nanorods), each
    emitter must be used as a dipole source in a separate BEM simulation.
    This function reads one Excel file per emitter, reconstructs the full
    ``(N, N, 3, 3)`` Green's function tensor, and saves it in the
    pair-indexed HDF5 format.

    The BEM workflow for pair-indexed reconstruction:

    1. For each emitter site *i*, run a BEM simulation with a point dipole
       at position ``r_i``.  Record the electric field at all other sites
       ``r_j`` and the Purcell factor at ``r_i``.
    2. Collect all results into one Excel file per source emitter *i*.
    3. Pass the list of Excel files (one per emitter) to this function.

    Args:
        xlsx_paths:          List of ``N`` Excel file paths, one per
                             source emitter.  ``xlsx_paths[i]`` contains
                             the BEM fields from a dipole at site *i*.
        out_h5:              Output HDF5 file path.
        zD_m:                Source z-position in meters.
        zA_m:                Observer z-position in meters.
        energy_eV:           Emitter transition energy (eV).
        p_eff_path:          Path to p_eff calibration file.
        emitter_positions_nm: Shape ``(N, 3)`` array of emitter positions
                             in nm.  Needed for vacuum GF computation.

    Output:
        HDF5 file with ``gf_layout = "pair"`` and datasets
        ``green_function_total(1, N, N, 3, 3)`` and
        ``green_function_vacuum(1, N, N, 3, 3)``.
    """
    N = len(xlsx_paths)
    emitter_positions_nm = np.asarray(emitter_positions_nm, dtype=float)
    if emitter_positions_nm.shape != (N, 3):
        raise ValueError(
            f"emitter_positions_nm shape {emitter_positions_nm.shape} "
            f"does not match {N} emitters."
        )

    Gtot = np.zeros((1, N, N, 3, 3), dtype=np.complex128)
    Gvac = np.zeros((1, N, N, 3, 3), dtype=np.complex128)

    lam_nm_ref = hbar * c / (energy_eV * eV_to_J) * 2 * np.pi * 1e9
    p_eff = read_peff(p_eff_path, lambda_nm=lam_nm_ref)

    for i, xlsx in enumerate(xlsx_paths):
        logger.info(f"Reading BEM data for source emitter {i+1}/{N}: {xlsx}")

        # Off-diagonal: fields at other sites from dipole at site i
        rx_nm_raw, G_raw = read_bem_dyadic(xlsx, "DyadicG")
        G_raw = G_raw / p_eff

        # Diagonal: Purcell factor (self-term) from dipole at site i
        lam_nm, Fx, Fy, Fz = read_purcell_sheet(xlsx, "G_self")
        omega = omega_from_lambda_nm(lam_nm)
        pref_self = omega / (6 * np.pi * c)
        G_self = np.zeros((3, 3), dtype=np.complex128)
        G_self[0, 0] = 1j * pref_self * Fx
        G_self[1, 1] = 1j * pref_self * Fy
        G_self[2, 2] = 1j * pref_self * Fz

        Gtot[0, i, i, :, :] = G_self

        # Map BEM output positions to emitter indices j ≠ i
        # (implementation depends on your BEM output format — this is
        # a placeholder showing the expected structure)
        for j_idx, G_tensor in enumerate(G_raw):
            j = j_idx if j_idx < i else j_idx + 1
            if j < N:
                Gtot[0, i, j, :, :] = G_tensor

    logger.info("Computing vacuum Green's function for all emitter pairs...")
    omega = energy_eV * eV_to_J / hbar
    calculator = Greens_function_analytical(metal_epsi=1.0 + 0.0j, omega=omega)
    for i in range(N):
        for j in range(N):
            ri = emitter_positions_nm[i] * 1e-9
            rj = emitter_positions_nm[j] * 1e-9
            dx = ri[0] - rj[0]
            dy = ri[1] - rj[1]
            Gvac[0, i, j, :, :] = calculator.vacuum_component(
                x=dx, y=dy, z1=ri[2], z2=rj[2],
            )

    E = np.array([float(energy_eV)], dtype=float)
    save_gf_pair_h5(out_h5, Gtot, Gvac, E, emitter_positions_nm, zD_m, zA_m)
    logger.success(f"Pair-indexed GF saved: {out_h5} ({N} emitters)")


@hydra.main(config_path="../../configs/BEM", config_name="reconstruct_GF",version_base=None)
def main(cfg: DictConfig):
    '''
    Main function to run the reconstruction of the Green's function.
    '''
    setup_loggers_hydra_aware()
    xlsx_path = Path(cfg.io.xlsx_path)
    output_file = Path(cfg.io.output_file)
    p_eff_path = Path(cfg.io.peff_path)
    
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    if not xlsx_path.exists():
        logger.error(f"Input file {xlsx_path} does not exist.")
        return

    if not p_eff_path.exists():
        logger.error(f"p_eff file {p_eff_path} does not exist.")
        return
    
    output_path = output_dir / output_file
    build_and_save(
        xlsx_path=str(xlsx_path),
        out_h5=str(output_path),
        zD_m=cfg.parameters.zD_nm * 1e-9,
        zA_m=cfg.parameters.zA_nm * 1e-9,
        energy_eV=cfg.parameters.energy_eV,
        p_eff_path=str(p_eff_path)
    )
    
    logger.success(f"Green's function reconstruction completed and saved to {output_path.absolute()}.")
    

if __name__ == "__main__":
    main()

