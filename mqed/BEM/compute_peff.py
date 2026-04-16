import numpy as np
import pandas as pd
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path

from mqed.utils.SI_unit import c,D2CMM,eps0 
from mqed.Dyadic_GF.GF_Sommerfeld import Greens_function_analytical
from mqed.Dyadic_GF.main import build_grid
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.utils.hydra_local import prepare_hydra_config_path
from loguru import logger


def omega_from_lambda_nm(lam_nm: float) -> float:
    '''Convert wavelength in nm to angular frequency in rad/s
    '''
    return 2*np.pi*c/(lam_nm*1e-9)

def read_bem_fieldline_xlsx(xlsx_path: str, sheet: str) -> tuple[np.ndarray, np.ndarray]:
    '''Read BEM electric field data from an Excel sheet.
    Returns positions in nm and complex electric field vectors.
    Args:
        xlsx_path: Path to the Excel file.
        sheet: Sheet name in the Excel file.
    Return:
        x_nm: Positions in nm as a 1D numpy array.
        E: Complex electric field vectors as a (N,3) numpy array.
    '''
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    x_nm = df["x_nm"].to_numpy(dtype=float)
    Ex = df["Re_Ex"].to_numpy(float) + 1j*df["Im_Ex"].to_numpy(float)
    Ey = df["Re_Ey"].to_numpy(float) + 1j*df["Im_Ey"].to_numpy(float)
    Ez = df["Re_Ez"].to_numpy(float) + 1j*df["Im_Ez"].to_numpy(float)

    E = np.stack([Ex, Ey, Ez], axis=1)  # (N,3)
    return x_nm, E

def fit_complex_scalar_ls(Ea: np.ndarray, Eb: np.ndarray) -> complex:
    """
    Fit s minimizing ||Eb - s Ea||^2 for complex arrays.
    Ea, Eb are 1D complex vectors.
    Args:
        Ea: 1D complex numpy array.
        Eb: 1D complex numpy array. 
    Return:
        s: Complex scalar minimizing ||Eb - s Ea||^2
    """
    num = np.vdot(Ea, Eb)   # Ea^H Eb
    den = np.vdot(Ea, Ea)   # Ea^H Ea
    return num / den

def compute_E0_from_vacuum_G0(calculator, omega, x_m, y_m, zD_m, zA_m, pvec_Cm):
    """
    Build analytic vacuum fields E0 at points (x_m, y_m, zA_m) from a dipole at zD_m.
    Returns E0 as (N,3) complex.
    

    """
    pref = (omega**2) / (eps0 * c**2)

    E0 = np.zeros((len(x_m), 3), dtype=np.complex128)
    for j, xm in enumerate(x_m):
        G0 = calculator.vacuum_component(
            x=xm, y=y_m, z1=zD_m, z2=zA_m
        )  # (3,3) SI
        E0[j, :] = pref * (G0 @ pvec_Cm)
    return E0

HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("BEM", __file__)

@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name="compute_peff")
def main(cfg: DictConfig):
    setup_loggers_hydra_aware()
    # ---- read BEM field data ----
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    
    logger.info("Reading BEM field data...")
    xlsx_path = Path(cfg.io.xlsx_path)
    x_nm, E_bem = read_bem_fieldline_xlsx(xlsx_path, cfg.io.sheet)
    x_m = x_nm * 1e-9
    y_m = float(cfg.sim.y_nm) * 1e-9

    zD_m = float(cfg.sim.zD_nm) * 1e-9
    zA_m = float(cfg.sim.zA_nm) * 1e-9

    pdir = np.array(cfg.sim.pdir, dtype=float)
    pdir = pdir / np.linalg.norm(pdir)

    # choose p_test in C*m
    if "p_test_debye" in cfg.dipole:
        p_test_Cm = float(cfg.dipole.p_test_debye) * D2CMM
    else:
        p_test_Cm = float(cfg.dipole.p_test_Cm)

    # stack BEM E into 1D vector (3N,) for LS
    Eb_stack = E_bem.reshape(-1)

    rows = []

    logger.info("Computing analytical vacuum fields and fitting p_eff...")
    
    lambdas_nm = build_grid(cfg.sim.lambdas_nm)
    logger.info(f"Using wavelengths (nm): {lambdas_nm}")
    for lam_nm in lambdas_nm:
        omega = omega_from_lambda_nm(float(lam_nm))

        # create your analytic calculator
        # For vacuum_component, metal_epsi is irrelevant; pass anything sensible.

        calculator = Greens_function_analytical(omega=omega, metal_epsi=1.0 + 0.0j)

        pvec = p_test_Cm * pdir  # (3,) C*m

        # analytic vacuum field for p_test
        E0 = compute_E0_from_vacuum_G0(calculator, omega, x_m, y_m, zD_m, zA_m, pvec)
        Ea_stack = E0.reshape(-1)

        # optional masking where analytic field is tiny
        if bool(cfg.fit.drop_small_E0):
            thr = float(cfg.fit.drop_threshold_rel) * np.max(np.abs(Ea_stack))
            mask = np.abs(Ea_stack) > thr
            Ea_use = Ea_stack[mask]
            Eb_use = Eb_stack[mask]
        else:
            Ea_use, Eb_use = Ea_stack, Eb_stack

        # fit s such that Eb ≈ s * Ea; then p_eff = s * p_test
        s = fit_complex_scalar_ls(Ea_use, Eb_use)
        p_eff = s * p_test_Cm

        rows.append({
            "lambda_nm": float(lam_nm),
            "p_test_Cm": p_test_Cm,
            "p_eff_Cm_real": np.real(p_eff),
            "p_eff_Cm_imag": np.imag(p_eff),
            "p_eff_Cm_abs":  np.abs(p_eff),
            "s_real": np.real(s),
            "s_imag": np.imag(s),
            "s_abs":  np.abs(s),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / cfg.io.output_csv, index=False)
    logger.success(f"Saved p_eff results to {(output_dir / cfg.io.output_csv).absolute()}")
    print(out)

if __name__ == "__main__":
    main()
