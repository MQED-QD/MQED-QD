import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from mqed.utils.dgf_data import load_gf_h5
from hydra.core.hydra_config import HydraConfig
from mqed.utils.hydra_local import prepare_hydra_config_path
from pathlib import Path

from mqed.utils.BEM_tools import read_bem_dyadic, read_peff

from loguru import logger

HC_EV_NM = 1239.841984  # eV*nm


def pick_fresnel_at_lambda(fresnel_h5: str, lambda_nm: float, drop_zero_rx: bool = True):
    """Return Rx_nm (N,) and G_total at nearest energy (N,3,3). Optionally drop Rx=0."""
    d = load_gf_h5(fresnel_h5)
    E_eV = d["energy_eV"]
    Rx = d["Rx_nm"]
    Gtot = d["G_total"]  # (M,N,3,3)

    target_E = HC_EV_NM / lambda_nm
    m = int(np.argmin(np.abs(E_eV - target_E)))

    G = Gtot[m]  # (N,3,3)

    if drop_zero_rx:
        # robust: drop the entry whose Rx is exactly 0 (or extremely close)
        zero_mask = np.isclose(Rx, 0.0, atol=1e-12)
        Rx = Rx[~zero_mask]
        G  = G[~zero_mask, :, :]

    return Rx, G


def interp_dyadic(rx_src, G_src, rx_tgt):
    """Linear interpolate complex dyadic from rx_src -> rx_tgt. Shapes: (N,3,3)."""
    G_out = np.zeros((len(rx_tgt), 3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            re = np.interp(rx_tgt, rx_src, np.real(G_src[:, i, j]))
            im = np.interp(rx_tgt, rx_src, np.imag(G_src[:, i, j]))
            G_out[:, i, j] = re + 1j * im
    return G_out


def ij_of(comp: str):
    """Map string like 'Gzz' -> indices (2,2)."""
    m = {"x": 0, "y": 1, "z": 2}
    return m[comp[1]], m[comp[2]]


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("BEM", __file__)

@hydra.main(config_path=HYDRA_CONFIG_PATH, config_name="compare_bem_dyadic", version_base=None)
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    
    lam = float(cfg.sim.lambda_nm)

    # 1) Load inputs
    logger.info("Loading inputs...")
    p_eff = read_peff(cfg.io.peff_csv, lam)
    rx_bem, G_bem_times_p = read_bem_dyadic(cfg.io.xlsx_path, cfg.io.sheet)
    rx_fr, G_fr = pick_fresnel_at_lambda(cfg.io.fresnel_h5, lam, drop_zero_rx=True)


    # 2) Rescale BEM: (G * p_eff) / p_eff = G_SI
    G_bem = G_bem_times_p / p_eff

    # 3) Put Fresnel on BEM grid if needed
    if bool(cfg.sim.interpolate_rx) and not np.allclose(rx_fr, rx_bem):
        G_fr = interp_dyadic(rx_fr, G_fr, rx_bem)

    # 4) Build output table
    logger.info("Building comparison table...")
    out = {"Rx_nm": rx_bem}

    for comp in cfg.compare.components:
        i, j = ij_of(comp)
        bem = G_bem[:, i, j]
        fr  = G_fr[:, i, j]

        out[f"Re_{comp}_BEM"] = np.real(bem)
        out[f"Im_{comp}_BEM"] = np.imag(bem)
        out[f"Re_{comp}_Fresnel"] = np.real(fr)
        out[f"Im_{comp}_Fresnel"] = np.imag(fr)

    df_out = pd.DataFrame(out)
    df_out.to_csv(output_dir / cfg.io.out_csv, index=False)

    logger.info(f"lambda = {lam} nm")
    logger.info(f"p_eff  = {p_eff} C*m")
    logger.success(f"Saved comparison CSV -> {(output_dir / cfg.io.out_csv).absolute()}")


if __name__ == "__main__":
    main()
