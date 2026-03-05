from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from loguru import logger
import pandas as pd

from mqed.utils.file_utils import _resolve_input_path
from mqed.utils.logging_utils import setup_loggers_hydra_aware
from mqed.utils.dgf_data import load_gf_h5
from mqed.utils.orientation import spherical_to_cartesian_dipole


def _clip_xy(x: np.ndarray, y: np.ndarray, xlim) -> tuple[np.ndarray, np.ndarray]:
    if xlim is None:
        return x, y
    xmin, xmax = float(xlim[0]), float(xlim[1])
    m = (x >= xmin) & (x <= xmax)
    return x[m], y[m]


def _drop_nonfinite(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _apply_fonts(ax, ps):
    font = getattr(ps, "font", None)
    if not font:
        return

    if getattr(font, "family", None):
        plt.rcParams["font.family"] = str(font.family)

    labelsize = int(getattr(font, "labelsize", 12))
    ticksize = int(getattr(font, "ticksize", 12))
    titlesize = int(getattr(font, "titlesize", 12))
    legendsize = int(getattr(font, "legendsize", 12))
    labelweight = str(getattr(font, "labelweight", "normal"))
    legendweight = str(getattr(font, "legendweight", "normal"))

    ax.set_xlabel(ps.xlabel, fontsize=labelsize, fontweight=labelweight)
    ax.set_ylabel(ps.ylabel, fontsize=labelsize, fontweight=labelweight)
    if getattr(ps, "title", None):
        ax.set_title(ps.title, fontsize=titlesize, fontweight=labelweight)

    ax.tick_params(axis="both", which="both", labelsize=ticksize)

    if getattr(ps, "legend", True):
        leg = ax.legend(fontsize=legendsize)
        for txt in leg.get_texts():
            txt.set_fontweight(legendweight)


def _compute_enhancement_from_h5(
    h5_path: Path,
    x_key: str,
    energy_index: int,
    donor_theta_deg: float,
    donor_phi_deg: float,
    acc_theta_deg: float,
    acc_phi_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute enhancement curves from stored Green's tensors.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(x_nm, enh_real, enh_imag)`` with
        ``enh_real = Re(g_tot)/Re(g_vac)`` and ``enh_imag = Im(g_tot)/Im(g_vac)``, where
        ``g = p_acc^T G p_donor``.
    """
    data = load_gf_h5(h5_path)
    Gtot = np.asarray(data["G_total"])  # (M,N,3,3)
    Gvac = np.asarray(data["G_vac"])    # (M,N,3,3)
    x_nm = np.asarray(data[x_key]).ravel()
    # breakpoint()

    p_donor = spherical_to_cartesian_dipole(donor_theta_deg, donor_phi_deg)
    p_acc = spherical_to_cartesian_dipole(acc_theta_deg, acc_phi_deg)

    m = int(energy_index)
    g_vac = np.einsum("i,...ij,j->...", p_acc, Gvac[m], p_donor)
    g_tot = np.einsum("i,...ij,j->...", p_acc, Gtot[m], p_donor)

    enh_real = np.real(g_tot) / np.real(g_vac)
    enh_imag = np.imag(g_tot) / np.imag(g_vac)
    return x_nm, enh_real, enh_imag


@hydra.main(config_path="../../configs/BEM", config_name="compare_enhancement", version_base=None)
def main(cfg: DictConfig) -> None:
    outdir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()

    ps = cfg.plot_settings
    fig, ax = plt.subplots(figsize=tuple(ps.figsize) if getattr(ps, "figsize", None) else (11, 6))

    # optional global font family
    if getattr(ps, "font", None) and getattr(ps.font, "family", None):
        plt.rcParams["font.family"] = str(ps.font.family)

    # global GF settings (can be overridden per curve)
    gf_global = getattr(cfg, "gf_settings", None)

    for curve in cfg.curves:
        path = _resolve_input_path(curve)

        gf = getattr(curve, "gf", None)
        if gf is None and gf_global is None:
            raise ValueError("Need either cfg.gf_settings or curve.gf for GF parameters (x_key, dipoles, etc.).")

        # merge: curve.gf overrides global gf_settings
        x_key = str(getattr(gf, "x_key", getattr(gf_global, "x_key", "Rx_nm")))
        energy_index = int(getattr(gf, "energy_index", getattr(gf_global, "energy_index", 0)))

        donor_cfg = getattr(gf, "dipoles", None).donor if gf and getattr(gf, "dipoles", None) else gf_global.dipoles.donor
        acc_cfg   = getattr(gf, "dipoles", None).acceptor if gf and getattr(gf, "dipoles", None) else gf_global.dipoles.acceptor

        # compute both components once per file
        x_nm, enh_real, enh_imag = _compute_enhancement_from_h5(
            path,
            x_key=x_key,
            energy_index=energy_index,
            donor_theta_deg=float(donor_cfg.theta_deg),
            donor_phi_deg=float(donor_cfg.phi_deg),
            acc_theta_deg=float(acc_cfg.theta_deg),
            acc_phi_deg=float(acc_cfg.phi_deg),
        )

        # per-curve plotting choices
        want = getattr(curve, "components", ["real", "imag"])  # list
        base_label = getattr(curve, "label", path.stem)

        # styles (per-curve defaults)
        lw_real = float(getattr(curve, "lw_real", getattr(curve, "lw", getattr(ps, "lw", 2.5))))
        lw_imag = float(getattr(curve, "lw_imag", getattr(curve, "lw", getattr(ps, "lw", 2.5))))
        style_real = getattr(curve, "style_real", getattr(curve, "style", "-"))
        style_imag = getattr(curve, "style_imag", getattr(curve, "style", "--"))
        color_real = getattr(curve, "color_real", getattr(curve, "color", None))
        color_imag = getattr(curve, "color_imag", getattr(curve, "color", None))

        # optional label suffixes
        real_suffix = getattr(ps, "real_label_suffix", " $V/V^{0}$")
        imag_suffix = getattr(ps, "imag_label_suffix", " $\\Gamma/\\Gamma^{0}$")

        xlim = getattr(ps, "xlim", None)

        if "real" in want:
            x, y = _clip_xy(x_nm, enh_real, xlim)
            x, y = _drop_nonfinite(x, y)
            ax.plot(x, y, style_real, lw=lw_real, label=base_label + real_suffix, color=color_real)

        if "imag" in want:
            x, y = _clip_xy(x_nm, enh_imag, xlim)
            x, y = _drop_nonfinite(x, y)
            ax.plot(x, y, style_imag, lw=lw_imag, label=base_label + imag_suffix, color=color_imag)

        logger.info(f"Plotted {base_label} from {path.name} (components={want})")

    # add vlines:
    vcfg = getattr(ps, "vlines", None)
    if vcfg and getattr(vcfg, "enabled", False):
        logger.info("Add a verticle line.")
        xs = list(getattr(vcfg, "xs",[]))
        
        # either a list of colors or one fallback
        colors = getattr(vcfg, "colors", None)
        default_color = getattr(vcfg, "color", "k")
        
        for i, xv in enumerate(xs):
            col = colors[i] if colors is not None and i < len(colors) else default_color
            ax.axvline(
                xv,
                linestyle = getattr(vcfg, "style", "--"),
                linewidth = float(getattr(vcfg, "lw", 1.5)),
                alpha = float(getattr(vcfg, "alpha", 0.8)),
                color=col,
            )
    # axis style
    ax.set_xlabel(ps.xlabel)
    ax.set_ylabel(ps.ylabel)
    if getattr(ps, "title", None):
        ax.set_title(ps.title)

    if getattr(ps, "xscale", None):
        ax.set_xscale(ps.xscale)
    if getattr(ps, "yscale", None):
        ax.set_yscale(ps.yscale)

    if getattr(ps, "xlim", None):
        ax.set_xlim(ps.xlim[0], ps.xlim[1])
    if getattr(ps, "ylim", None):
        ax.set_ylim(ps.ylim[0], ps.ylim[1])

    if getattr(ps, "grid", False):
        ax.grid(True, which="both", ls="--", alpha=0.5)

    if getattr(ps, "legend", True):
        ax.legend()

    _apply_fonts(ax, ps)

    if getattr(ps, "tight_layout", True):
        plt.tight_layout()

    if getattr(ps, "save_plot", True):
        name = getattr(ps, "filename", "gf_enhancement.png")
        figpath = outdir / name
        fig.savefig(figpath, dpi=getattr(ps, "dpi", 300), bbox_inches="tight")
        logger.success(f"Saved plot → {figpath}")
        df = pd.DataFrame({"x_nm": x_nm, "enh_real": enh_real, "enh_imag": enh_imag})
        data_path = figpath.with_suffix(".csv")
        df.to_csv(data_path, index=False)
        logger.success(f"Saved data → {data_path}")
    


    if getattr(ps, "show", False):
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
