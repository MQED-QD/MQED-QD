from pathlib import Path

import h5py
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


# ---- MEEP HDF5 loader ---------------------------------------------------
# The MEEP script (planar_example.py) saves:
#   /E_field   : (Npts, 3, 3) complex128  — E_a at point i for dipole orient b
#   /x_nm      : (Npts,) observation x-coordinates
# Column mapping:  orient b={0:x, 1:y, 2:z},  component a={0:x, 1:y, 2:z}
#
# To match BEM Excel columns like "Re_Ez" with a z-oriented dipole:
#   E_z component (a=2) for z-oriented dipole (b=2) → E_field[:, 2, 2]
# -------------------------------------------------------------------------

# Maps human-readable component names to (field_index, orient_index) pairs.
# These match the BEM Excel column convention: "{Re|Im}_{component}"
# with the dipole oriented along the same axis as the monitored component
# (z-oriented dipole → Ez, x-oriented dipole → Ex, etc.)
_MEEP_COMPONENT_MAP = {
    "Ex": (0, 0),  # Ex component, x-oriented dipole
    "Ey": (1, 1),  # Ey component, y-oriented dipole
    "Ez": (2, 2),  # Ez component, z-oriented dipole
}


def _load_meep_h5(h5_path: Path) -> dict:
    """Load MEEP planar simulation HDF5.

    Args:
        h5_path: Path to MEEP output (e.g., MEEP_planar_silver_665nm_2D.h5)

    Returns:
        dict with keys:
            E_field : (Npts, 3, 3) complex128
            x_nm    : (Npts,) float64
    """
    with h5py.File(h5_path, "r") as f:
        E_field = f["E_field"][:]   # (Npts, 3, 3) complex128
        x_nm = f["x_nm"][:]        # (Npts,)
    return {"E_field": E_field, "x_nm": x_nm}


def _meep_enhancement(meep_plane: dict, meep_vac: dict,
                      component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute E(silver)/E(vacuum) enhancement from MEEP data.

    Args:
        meep_plane: MEEP data dict for silver substrate
        meep_vac:   MEEP data dict for vacuum
        component:  Which diagonal component to compare ("Ex", "Ey", or "Ez")

    Returns:
        x_nm:     observation x-coordinates (from silver run)
        enh_real: Re(E_silver) / Re(E_vacuum) — analogous to BEM Re enhancement
        enh_imag: Im(E_silver) / Im(E_vacuum) — analogous to BEM Im enhancement

    Note:
        The x-grids from the two runs may differ (e.g., different --x-min).
        We interpolate the vacuum data onto the silver x-grid for consistency.
    """
    if component not in _MEEP_COMPONENT_MAP:
        raise ValueError(
            f"Unknown component '{component}'. "
            f"Choose from: {list(_MEEP_COMPONENT_MAP.keys())}"
        )
    ifield, iorient = _MEEP_COMPONENT_MAP[component]

    E_plane = meep_plane["E_field"][:, ifield, iorient]  # complex (Npts,)
    E_vac   = meep_vac["E_field"][:, ifield, iorient]    # complex (Npts_vac,)

    x_plane = meep_plane["x_nm"]
    x_vac   = meep_vac["x_nm"]

    # If x-grids match exactly, use directly; otherwise interpolate vacuum
    if len(x_plane) == len(x_vac) and np.allclose(x_plane, x_vac):
        E_vac_interp = E_vac
    else:
        logger.info(
            f"MEEP x-grids differ (silver: {len(x_plane)} pts, "
            f"vacuum: {len(x_vac)} pts). Interpolating vacuum onto silver grid."
        )
        # Interpolate real and imaginary parts separately
        E_vac_interp = (
            np.interp(x_plane, x_vac, np.real(E_vac))
            + 1j * np.interp(x_plane, x_vac, np.imag(E_vac))
        )

    enh_real = np.real(E_plane) / np.real(E_vac_interp)
    enh_imag = np.imag(E_plane) / np.imag(E_vac_interp)

    return x_plane, enh_real, enh_imag


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


@hydra.main(config_path="../../configs/BEM", config_name="compare_silver", version_base=None)
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    _apply_rcparams(cfg.plot.rcParams)
    setup_loggers_hydra_aware()

    # ---- Fresnel / analytical reference (always loaded) -------------------
    dgf_data_path = Path(cfg.paths.dgf_h5)
    if not dgf_data_path.exists():
        raise FileNotFoundError(f"Missing Fresnel data: {dgf_data_path}")

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

    # ---- Numerical simulation data (BEM or MEEP) --------------------------
    # Controlled by cfg.simulation_source: "bem" (default) or "meep"
    sim_source = cfg.get("simulation_source", "bem")
    logger.info(f"Simulation data source: {sim_source}")

    if sim_source == "meep":
        # --- MEEP HDF5 path loading ---
        meep_plane_path = Path(cfg.paths.meep_plane_h5)
        meep_vac_path   = Path(cfg.paths.meep_vacuum_h5)
        for p in [meep_plane_path, meep_vac_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing MEEP file: {p}")

        meep_plane = _load_meep_h5(meep_plane_path)
        meep_vac   = _load_meep_h5(meep_vac_path)

        # Which field component to compare (default: "Ez" for z-oriented dipole)
        meep_component = cfg.get("meep", {}).get("component", "Ez")
        logger.info(f"MEEP component: {meep_component}")

        x_sim, sim_real, sim_imag = _meep_enhancement(
            meep_plane, meep_vac, meep_component
        )

    else:
        # --- BEM Excel loading (original behavior) ---
        bem_plane_path = Path(cfg.paths.bem_plane_xlsx)
        bem_vacuum_path = Path(cfg.paths.bem_vacuum_xlsx)
        for p in [bem_plane_path, bem_vacuum_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing BEM file: {p}")

        bem_plane = pd.read_excel(bem_plane_path, sheet_name=cfg.bem.sheet)
        bem_vac   = pd.read_excel(bem_vacuum_path, sheet_name=cfg.bem.sheet)

        sim_real = bem_plane[cfg.bem.re_col].to_numpy() / bem_vac[cfg.bem.re_col].to_numpy()
        sim_imag = bem_plane[cfg.bem.im_col].to_numpy() / bem_vac[cfg.bem.im_col].to_numpy()

        # BEM x-coordinates come from the Excel x column
        x_sim = bem_plane[cfg.bem.get("x_col", "x_nm")].to_numpy()

    # ---- Plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=tuple(cfg.plot.figsize), dpi=cfg.plot.get("dpi", 120))

    _apply_axes_style(ax, cfg.plot.axes)
    
    title = cfg.plot.axes.get("title")
    if title:
        ax.set_title(title.format(dipole_frequency=cfg.dipole_frequency))
    yscale = cfg.get("yscale", "log")
    ax.set_yscale(yscale)
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=cfg.get("linthresh", 1e-3))

    # Series order controlled by YAML
    for key in cfg.plot.series_order:
        s = cfg.plot.series[key]
        source = s.source
        if source == "impl_real":
            _plot_series(ax, x_nm, impl_real, s)
        elif source == "impl_imag":
            _plot_series(ax, x_nm, impl_imag, s)
        elif source in ("bem_real", "sim_real"):
            _plot_series(ax, x_sim, sim_real, s)
        elif source in ("bem_imag", "sim_imag"):
            _plot_series(ax, x_sim, sim_imag, s)

    leg = ax.legend(
        loc=cfg.plot.legend.loc,
        bbox_to_anchor=tuple(cfg.plot.legend.bbox_to_anchor) if "bbox_to_anchor" in cfg.plot.legend else None,
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
