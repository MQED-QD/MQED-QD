'''This module provides functions to reconstruct the dyadic Green's function from BEM electric field data.
It includes reading BEM data, Purcell factors, p_eff values, and reconstructing the Green's function.
'''

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig

from loguru import logger
from mqed.utils.BEM_tools import read_bem_dyadic, read_peff, read_purcell_sheet
from mqed.utils.dgf_data import save_gf_h5
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

