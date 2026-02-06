from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from loguru import logger

from mqed.utils.file_utils import _resolve_input_path

from mqed.utils.logging_utils import setup_loggers_hydra_aware

def _load_dx_and_time(h5_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
        t_ps: (T,)
        dx_nm: (T,)  (mean if available; otherwise single-run Δx)
        meta: dict with info about what we loaded
    Supports:
      - datasets: 'dx_mean_nm' (preferred), 'dx_nm', or expectations: X_shift, X_shift2 (compute Δx)
    """
    logger.info(f"Loading Δx data from {h5_path}")
    meta = {}
    with h5py.File(str(h5_path), "r") as f:
        # time
        if "t_ps" not in f:
            raise ValueError(f"{h5_path} has no 't_ps' dataset.")
        t_ps = np.asarray(f["t_ps"][...]).ravel()
        # breakpoint()

        # 1) direct MSD dataset?
        if "msd_nm2" in f:
            msd = np.asarray(f["msd_nm2"][...]).ravel()
            meta["source"] = "msd_nm2"

        # 2) expectations group
        elif "expectations" in f and "X_shift2" in f["expectations"]:
            msd = np.asarray(f["expectations"]["X_shift2"][...]).ravel()
            meta["source"] = "expectations/X_shift2"

        # 3) last resort: square of sqrt-MSD (if that file only saved dx)
        elif "dx_mean_nm" in f:
            dx = np.asarray(f["dx_mean_nm"][...]).ravel()
            msd = dx**2
            meta["source"] = "dx_mean_nm**2"

        else:
            raise ValueError(
                f"{h5_path} does not contain 'msd_nm2', "
                "'expectations/X_shift2', or 'dx_mean_nm'."
            )

        # carry over a few helpful attributes if present
        for k in ("method", "mode", "n_realizations", "sigma_phi_deg", "seed_base"):
            if k in f.attrs:
                meta[k] = f.attrs[k]

    if msd.shape != t_ps.shape:
        raise ValueError(f"{h5_path} msd shape {msd.shape} and t_ps shape {t_ps.shape} mismatch.")

    return t_ps, msd, meta


def _select_x(t_ps: np.ndarray, cfg_ps) -> np.ndarray:
    """Return boolean mask for x selection by index or by time value (ps)."""
    if hasattr(cfg_ps, "x_index_range") and cfg_ps.x_index_range:
        i0, i1 = int(cfg_ps.x_index_range[0]), int(cfg_ps.x_index_range[1])
        sel = np.zeros_like(t_ps, dtype=bool)
        sel[max(0, i0): min(len(t_ps), i1 + 1)] = True
        return sel
    if hasattr(cfg_ps, "x_range_ps") and cfg_ps.x_range_ps:
        xmin, xmax = float(cfg_ps.x_range_ps[0]), float(cfg_ps.x_range_ps[1])
        return (t_ps >= xmin) & (t_ps <= xmax)
    return np.ones_like(t_ps, dtype=bool)


@hydra.main(config_path="../../configs/plots", config_name="msd", version_base=None)
def main(cfg: DictConfig) -> None:
    outdir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()

    ps = cfg.plot_settings
    fig, ax = plt.subplots(figsize=(ps.figsize[0], ps.figsize[1]) if getattr(ps, "figsize", None) else (8, 6))

    # set global font sizes
    font = getattr(ps, "font", None)

    # optional: set global family (affects everything)
    if font and getattr(font, "family", None):
        plt.rcParams["font.family"] = str(font.family)


    for curve in cfg.curves:
        path = _resolve_input_path(curve)
        t_ps, msd, meta = _load_dx_and_time(path)

        sel = _select_x(t_ps, ps)
        x = t_ps[sel] * getattr(ps, "x_scale_factor", 1.0)   # keep default 1.0; you can set 1e-12 if you want 10^-10 s scaling, etc.
        y = msd[sel]
        # style
        style = getattr(curve, "style", "-")
        lw = getattr(curve, "lw", ps.get("lw", 1.5))
        label = getattr(curve, "label", path.stem)

        ax.plot(x, y, style, lw=lw, label=label)

        logger.info(f"Plotted {label} from {path.name} (source={meta.get('source','?')})")

    # labels and title
    if font:
        labelsize  = int(getattr(font, "labelsize", 12))
        titlesize  = int(getattr(font, "titlesize", 12))
        ticksize   = int(getattr(font, "ticksize", 12))
        legendsize = int(getattr(font, "legendsize", 12))
        labelweight = str(getattr(font, "labelweight", "normal"))
        legendweight = str(getattr(font, "legendweight", "normal"))
    else:
        labelsize = titlesize = 12
        ticksize = 12
        legendsize = 12
        labelweight = "normal"
        legendweight = "normal"

    ax.set_xlabel(ps.xlabel, fontsize=labelsize, fontweight=labelweight)
    ax.set_ylabel(ps.ylabel, fontsize=labelsize, fontweight=labelweight)

    if getattr(ps, "title", None):
        ax.set_title(ps.title, fontsize=titlesize, fontweight=labelweight)

    # ticks
    ax.tick_params(axis="both", which="both", labelsize=ticksize)
    
    # NEW: bold tick labels if requested (fallback to labelweight if tickweight not set)
    tickweight = str(getattr(font, "tickweight", labelweight)) if font else labelweight
    for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
        ticklabel.set_fontweight(tickweight)

    # legend
    if getattr(ps, "legend", True):
        leg = ax.legend(fontsize=legendsize)
        # make legend text bold if requested
        for txt in leg.get_texts():
            txt.set_fontweight(legendweight)


    # scales
    if getattr(ps, "xscale", None): ax.set_xscale(ps.xscale)
    if getattr(ps, "yscale", None): ax.set_yscale(ps.yscale)

    # limits
    if getattr(ps, "xlim", None): ax.set_xlim(ps.xlim[0], ps.xlim[1])
    if getattr(ps, "ylim", None): ax.set_ylim(ps.ylim[0], ps.ylim[1])

    if getattr(ps, "grid", True):
        ax.grid(True, which="both", ls="--", alpha=0.5)


    if getattr(ps, "tight_layout", True):
        plt.tight_layout()

    if getattr(ps, "save_plot", True):
        name = getattr(ps, "filename", "sqrt_msd.png")
        figpath = outdir / name
        fig.savefig(figpath, dpi=getattr(ps, "dpi", 300), bbox_inches="tight")
        logger.success(f"Saved plot → {figpath}")

    if getattr(ps, "show", False):
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
