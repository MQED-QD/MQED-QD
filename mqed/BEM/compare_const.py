import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from hydra.core.hydra_config import HydraConfig
from loguru import logger

from mqed.utils.orientation import spherical_to_cartesian_dipole
from mqed.utils.dgf_data import load_gf_h5
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.utils.hydra_local import prepare_hydra_config_path


def _apply_rcparams(rcparams: dict):
    # rcparams is a normal dict in YAML: {"font.size": 18, ...}
    plt.rcParams.update(rcparams or {})


def _apply_axes_style(ax, cfg):
    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)
    if cfg.get("title"):
        ax.set_title(cfg.title)

    if cfg.get("xlim") is not None:
        ax.set_xlim(cfg.xlim[0], cfg.xlim[1])
    if cfg.get("ylim") is not None:
        ax.set_ylim(cfg.ylim[0], cfg.ylim[1])

    ax.tick_params(
        direction=cfg.ticks.direction,
        top=cfg.ticks.top,
        right=cfg.ticks.right,
        labelsize=cfg.ticks.labelsize,
        length=cfg.ticks.length,
        width=cfg.ticks.width,
    )

    if cfg.get("grid", False):
        ax.grid(True, alpha=cfg.get("grid_alpha", 0.25))

    for spine in ax.spines.values():
        spine.set_linewidth(cfg.get("spine_width", 1.2))


def _plot_series(ax, x, y, s):
    """
    s is one series config dict from YAML.
    """
    ax.plot(
        x, y,
        label=s.label,
        linestyle=s.get("linestyle", "-"),
        linewidth=s.get("linewidth", 2.5),
        marker=s.get("marker", None),
        markersize=s.get("markersize", 7),
        markerfacecolor=s.get("markerfacecolor", "none"),
        markeredgewidth=s.get("markeredgewidth", 1.5),
        color=s.get("color", None),
    )


# def test_BEM_comparison(cfg, impl_real, impl_imag, bem_real, bem_imag):
#     rtol = cfg.test.rtol
#     atol = cfg.test.atol

#     ok_r = np.allclose(impl_real, bem_real, rtol=rtol, atol=atol)
#     ok_i = np.allclose(impl_imag, bem_imag, rtol=rtol, atol=atol)

#     if not ok_r:
#         diff = impl_real - bem_real
#         raise AssertionError(
#             f"Real mismatch: max_abs={np.max(np.abs(diff)):.3e}, "
#             f"max_rel={np.max(np.abs(diff)/(np.abs(bem_real)+1e-30)):.3e}"
#         )
#     if not ok_i:
#         diff = impl_imag - bem_imag
#         raise AssertionError(
#             f"Imag mismatch: max_abs={np.max(np.abs(diff)):.3e}, "
#             f"max_rel={np.max(np.abs(diff)/(np.abs(bem_imag)+1e-30)):.3e}"
#         )


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("BEM", __file__)

@hydra.main(config_path=HYDRA_CONFIG_PATH, config_name="compare", version_base=None)
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    _apply_rcparams(cfg.plot.rcParams)
    setup_loggers_hydra_aware()

    # IMPORTANT with Hydra: use to_absolute_path or get_original_cwd()
    dgf_data_path = Path(cfg.paths.dgf_h5)
    bem_plane_path = Path(cfg.paths.bem_plane_xlsx)
    bem_vacuum_path = Path(cfg.paths.bem_vacuum_xlsx)
    for p in [dgf_data_path, bem_plane_path, bem_vacuum_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")



    data = load_gf_h5(dgf_data_path)
    Gtot = data["G_total"]       # (M,N,3,3)
    Gvac = data["G_vac"]         # (M,N,3,3)
    x_nm = data[cfg.data.x_key]  # e.g. "Rx_nm"

    # Dipoles
    p_donor = spherical_to_cartesian_dipole(cfg.dipoles.donor.theta_deg, cfg.dipoles.donor.phi_deg)
    p_acc   = spherical_to_cartesian_dipole(cfg.dipoles.acceptor.theta_deg, cfg.dipoles.acceptor.phi_deg)

    # Pick energy index (your current code uses [0])
    m = cfg.data.energy_index
    g_vac = np.einsum("i,...ij,j->...", p_acc, Gvac[m], p_donor)
    g_tot = np.einsum("i,...ij,j->...", p_acc, Gtot[m], p_donor)

    # Your enhancement definition (kept consistent with your code)
    impl_real = np.real(g_tot) / np.real(g_vac)
    impl_imag = np.imag(g_tot) / np.imag(g_vac)

    # Load BEM
    bem_plane = pd.read_excel(bem_plane_path, sheet_name=cfg.bem.sheet)
    bem_vac   = pd.read_excel(bem_vacuum_path, sheet_name=cfg.bem.sheet)

    bem_real = bem_plane[cfg.bem.re_col].to_numpy() / bem_vac[cfg.bem.re_col].to_numpy()
    bem_imag = bem_plane[cfg.bem.im_col].to_numpy() / bem_vac[cfg.bem.im_col].to_numpy()

    # Optional: run test with tolerances
    # if cfg.test.enabled:
    #     test_BEM_comparison(cfg, impl_real, impl_imag, bem_real, bem_imag)

    # Plot in “screenshot 2” style: one axes, markers for BEM, lines for Analytical/Implementation
    fig, ax = plt.subplots(figsize=tuple(cfg.plot.figsize), dpi=cfg.plot.get("dpi", 120))

    _apply_axes_style(ax, cfg.plot.axes)
    
    title = cfg.plot.axes.get("title")
    if title:
        ax.set_title(title.format(dipole_frequency=cfg.dipole_frequency))


    # Series order controlled by YAML
    for key in cfg.plot.series_order:
        s = cfg.plot.series[key]
        if s.source == "impl_real":
            _plot_series(ax, x_nm, impl_real, s)
        elif s.source == "impl_imag":
            _plot_series(ax, x_nm, impl_imag, s)
        elif s.source == "bem_real":
            _plot_series(ax, x_nm, bem_real, s)
        elif s.source == "bem_imag":
            _plot_series(ax, x_nm, bem_imag, s)

    leg = ax.legend(
        loc=cfg.plot.legend.loc,
        frameon=cfg.plot.legend.frameon,
        fancybox=cfg.plot.legend.fancybox,
        framealpha=cfg.plot.legend.framealpha,
        fontsize=cfg.plot.legend.fontsize,
    )
    leg.get_frame().set_edgecolor(cfg.plot.legend.get("edgecolor", "0.4"))

    fig.tight_layout()

    if cfg.plot.get("save", False):
        plot_filename= Path(cfg.plot.save_path)
        plot_filepath = output_dir / plot_filename
        fig.savefig(plot_filepath, bbox_inches="tight")
        logger.info(f"Plot saved to {plot_filepath}")
    plt.close(fig)
    logger.success(f"BEM comparison complete. Logs saved to: {output_dir.absolute()}")
    # plt.show()


if __name__ == "__main__":
    main()
