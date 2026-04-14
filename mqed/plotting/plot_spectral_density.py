"""
Plot Spectral Density
=====================

Plots the generalized spectral density :math:`J_{\\alpha\\beta}(\\omega)` as a
function of energy (eV) for user-selected emitter pairs or separations.

Reads spectral density data from the HDF5 file produced by
:mod:`mqed.analysis.spectral_density`.

Usage::

    python -m mqed.plotting.plot_spectral_density

Configuration via ``configs/plots/spectral_density.yaml``.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import h5py
import hydra
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf

from mqed.utils.logging_utils import setup_loggers_hydra_aware


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------


def _load_spectral_density_h5(filepath: str) -> dict:
    """Load spectral density data from HDF5.

    Returns:
        Dictionary with keys: J_eV, energy_eV, gf_layout, and either
        Rx_nm (separation layout) or emitter_positions_nm (pair layout).
    """
    data = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]
        for key in f.attrs:
            data[key] = f.attrs[key]
    return data


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


def _apply_font_config(cfg):
    """Apply font configuration from config, following plot_pr.py conventions."""
    font_cfg = cfg.get("font", {})
    plt.rcParams.update({
        "font.family": font_cfg.get("family", "Arial"),
        "axes.labelsize": font_cfg.get("labelsize", 18),
        "xtick.labelsize": font_cfg.get("ticksize", 16),
        "ytick.labelsize": font_cfg.get("ticksize", 16),
        "legend.fontsize": font_cfg.get("legendsize", 14),
        "axes.titlesize": font_cfg.get("titlesize", 18),
        "axes.labelweight": font_cfg.get("labelweight", "bold"),
        "axes.titleweight": font_cfg.get("titleweight", "bold"),
    })


def _plot_separation_layout(J_eV, energy_eV, Rx_nm, cfg):
    """Plot J(ω) for separation-indexed data.

    Produces one curve per selected separation Rx.
    """
    ps = cfg.plot_settings

    # Select which separations to plot
    sep_indices = list(ps.get("separation_indices", [0]))
    if not sep_indices:
        sep_indices = [0]

    fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))

    for idx in sep_indices:
        if idx >= len(Rx_nm):
            logger.warning(f"Separation index {idx} out of range "
                           f"(max {len(Rx_nm) - 1}), skipping.")
            continue

        label = ps.get("label_template", "Rx = {Rx:.1f} nm").format(Rx=Rx_nm[idx])
        ax.plot(
            energy_eV,
            J_eV[idx, :],
            lw=ps.get("lw", 1.5),
            label=label,
        )

    ax.set_xlabel(ps.get("xlabel", r"Energy (eV)"))
    ax.set_ylabel(ps.get("ylabel", r"$J(\omega)$ (eV)"))

    title_template = ps.get("title", r"Spectral Density $J(\omega)$")
    ax.set_title(title_template)

    if ps.get("yscale", "linear") == "log":
        ax.set_yscale("log")
    if ps.get("xscale", "linear") == "log":
        ax.set_xscale("log")

    x_range = ps.get("x_range_eV", None)
    if x_range is not None:
        ax.set_xlim(x_range)

    y_range = ps.get("y_range", None)
    if y_range is not None:
        ax.set_ylim(y_range)

    if ps.get("grid", True):
        ax.grid(True, alpha=0.3)

    ax.legend()
    fig.tight_layout()
    return fig


def _plot_pair_layout(J_eV, energy_eV, cfg):
    """Plot J_αβ(ω) for pair-indexed data.

    Produces one curve per selected (α, β) pair.
    """
    ps = cfg.plot_settings

    # Select which pairs to plot: list of [alpha, beta] pairs
    # Default: self-term of emitter 0
    pair_indices = list(ps.get("pair_indices", [[0, 0]]))
    if not pair_indices:
        pair_indices = [[0, 0]]

    fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))

    N = J_eV.shape[0]
    for pair in pair_indices:
        alpha, beta = int(pair[0]), int(pair[1])
        if alpha >= N or beta >= N:
            logger.warning(f"Pair ({alpha}, {beta}) out of range "
                           f"(N={N}), skipping.")
            continue

        label = ps.get(
            "label_template",
            r"$J_{{\alpha={a},\beta={b}}}(\omega)$"
        ).format(a=alpha, b=beta)
        ax.plot(
            energy_eV,
            J_eV[alpha, beta, :],
            lw=ps.get("lw", 1.5),
            label=label,
        )

    ax.set_xlabel(ps.get("xlabel", r"Energy (eV)"))
    ax.set_ylabel(ps.get("ylabel", r"$J_{\alpha\beta}(\omega)$ (eV)"))

    title_template = ps.get(
        "title", r"Spectral Density $J_{\alpha\beta}(\omega)$"
    )
    ax.set_title(title_template)

    if ps.get("yscale", "linear") == "log":
        ax.set_yscale("log")
    if ps.get("xscale", "linear") == "log":
        ax.set_xscale("log")

    x_range = ps.get("x_range_eV", None)
    if x_range is not None:
        ax.set_xlim(x_range)

    y_range = ps.get("y_range", None)
    if y_range is not None:
        ax.set_ylim(y_range)

    if ps.get("grid", True):
        ax.grid(True, alpha=0.3)

    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
#  Hydra CLI entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="../../configs/plots",
    config_name="spectral_density",
    version_base=None,
)
def plot_spectral_density(cfg) -> None:
    """Plot spectral density from pre-computed HDF5 data.

    This is the Hydra CLI entry point.  Configuration is loaded from
    ``configs/plots/spectral_density.yaml``.
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()

    logger.info("Plotting spectral density")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Resolve input ---
    input_path = Path(cfg.input_file)
    if not input_path.is_absolute():
        input_path = Path(hydra.utils.get_original_cwd()) / input_path
    logger.info(f"Loading spectral density from: {input_path}")

    data = _load_spectral_density_h5(str(input_path))
    J_eV = data["J_eV"]
    energy_eV = data["energy_eV"]
    gf_layout = data["gf_layout"]

    logger.info(f"GF layout: {gf_layout}, J shape: {J_eV.shape}")

    # --- Apply font config ---
    _apply_font_config(cfg)

    # --- Plot ---
    ps = cfg.plot_settings
    if gf_layout == "separation":
        Rx_nm = data["Rx_nm"]
        fig = _plot_separation_layout(J_eV, energy_eV, Rx_nm, cfg)
    elif gf_layout == "pair":
        fig = _plot_pair_layout(J_eV, energy_eV, cfg)
    else:
        raise ValueError(f"Unknown GF layout: {gf_layout}")

    # --- Save ---
    filename = ps.get("filename", "spectral_density.png")
    dpi = ps.get("dpi", 300)
    plot_filepath = output_dir / filename

    if ps.get("save_plot", True):
        fig.savefig(plot_filepath, dpi=dpi, bbox_inches="tight")
        logger.success(f"Saved plot to: {plot_filepath}")
    else:
        logger.info("save_plot=False; plot not saved.")

    plt.close(fig)


if __name__ == "__main__":
    plot_spectral_density()
