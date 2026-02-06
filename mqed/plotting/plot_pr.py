from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from loguru import logger

from mqed.utils.file_utils import _resolve_input_path
from mqed.utils.logging_utils import setup_loggers_hydra_aware


def _load_ipr_and_time(h5_path: Path, *, nmol_hint: int | None = None) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """
    Load t_ps and IPR(t). Tries, in order:
      1) top-level 'ipr_mean' (and optional 'ipr_std')
      2) expectations['IPR_site']
      3) derive from expectations['Excitation_Populations'] (renormalized to excited manifold)

    Returns:
      t_ps : (T,)
      ipr  : (T,)
      meta : dict
      ipr_std : (T,) or None
    """
    meta, ipr_std = {}, None
    with h5py.File(str(h5_path), "r") as f:
        if "t_ps" not in f:
            raise ValueError(f"{h5_path} has no 't_ps'.")
        t_ps = np.asarray(f["t_ps"][...]).ravel()

        if "ipr_mean" in f:
            ipr = np.asarray(f["ipr_mean"][...]).ravel()
            meta["source"] = "ipr_mean"
            if "ipr_std" in f:
                ipr_std = np.asarray(f["ipr_std"][...]).ravel()

        elif "expectations" in f and "IPR_site" in f["expectations"]:
            ipr = np.asarray(f["expectations"]["IPR_site"][...]).ravel()
            meta["source"] = "expectations/IPR_site"

        elif "expectations" in f and "Excitation_Populations" in f["expectations"]:
            # Build IPR from populations (drop ground, renormalize)
            pops = np.asarray(f["expectations"]["Excitation_Populations"][...])  # (T, N+1) or (T,N)
            if pops.ndim != 2:
                raise ValueError(f"Excitation_Populations has shape {pops.shape}, expected 2D.")
            # infer Nmol: try nmol_hint else from width-1
            if nmol_hint is None:
                nmol_guess = pops.shape[1] - 1 if pops.shape[1] > 1 else pops.shape[1]
            else:
                nmol_guess = nmol_hint
            if pops.shape[1] == nmol_guess + 1:
                pop_exc = pops[:, 1:]      # drop ground
            else:
                pop_exc = pops
            s = pop_exc.sum(axis=1, keepdims=True)
            s = np.where(s == 0.0, 1.0, s)  # avoid div/0
            q = pop_exc / s
            ipr = np.einsum("ti,ti->t", q, q)
            meta["source"] = "derived_from_populations"

        else:
            raise ValueError(
                f"{h5_path} does not contain 'ipr_mean', 'expectations/IPR_site', "
                "or 'expectations/Excitation_Populations'."
            )

        for k in ("method", "mode", "n_realizations", "sigma_phi_deg", "seed_base"):
            if k in f.attrs:
                meta[k] = f.attrs[k]

    if ipr.shape != t_ps.shape:
        raise ValueError(f"{h5_path} IPR shape {ipr.shape} vs t_ps {t_ps.shape}")
    return t_ps, ipr, meta, ipr_std


def _select_x(t_ps: np.ndarray, cfg_ps) -> np.ndarray:
    """
    Build a boolean mask for choosing an x-range (time window).

    Examples:
      - x_index_range: [0, 100]  -> selects indices 0..100 (first 101 points)
      - x_range_ps:    [0.0, 15] -> selects times between 0.0 and 15 ps (inclusive)
    """
    if hasattr(cfg_ps, "x_index_range") and cfg_ps.x_index_range:
        i0, i1 = int(cfg_ps.x_index_range[0]), int(cfg_ps.x_index_range[1])
        if i0 > i1: i0, i1 = i1, i0
        i0 = max(0, min(i0, len(t_ps) - 1))
        i1 = max(0, min(i1, len(t_ps) - 1))
        mask = np.zeros_like(t_ps, dtype=bool)
        mask[i0:i1 + 1] = True
        return mask
    if hasattr(cfg_ps, "x_range_ps") and cfg_ps.x_range_ps:
        xmin, xmax = float(cfg_ps.x_range_ps[0]), float(cfg_ps.x_range_ps[1])
        if xmin > xmax: xmin, xmax = xmax, xmin
        return (t_ps >= xmin) & (t_ps <= xmax)
    return np.ones_like(t_ps, dtype=bool)


@hydra.main(config_path="../../configs/plots", config_name="pr", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_loggers_hydra_aware()
    outdir = Path(HydraConfig.get().runtime.output_dir)

    ps = cfg.plot_settings
    fig, ax = plt.subplots(
        figsize=(ps.figsize[0], ps.figsize[1]) if getattr(ps, "figsize", None) else (8, 6)
    )

    # set global font sizes
    font = getattr(ps, "font", None)

    # optional: set global family (affects everything)
    if font and getattr(font, "family", None):
        plt.rcParams["font.family"] = str(font.family)


    for curve in cfg.curves:
        path = _resolve_input_path(curve)
        logger.info(f"Using file: {path}")
        t_ps, ipr, meta, ipr_std = _load_ipr_and_time(path)

        sel = _select_x(t_ps, ps)
        x = t_ps[sel] * getattr(ps, "x_scale_factor", 1.0)
        y = 1.0/ipr[sel]

        style = getattr(curve, "style", "-")
        lw = getattr(curve, "lw", ps.get("lw", 1.6))
        label = getattr(curve, "label", path.stem)

        ax.plot(x, y, style, lw=lw, label=label)

        # Optional shading if ipr_std exists
        if ipr_std is not None and getattr(ps, "shade_std", True):
            ystd = ipr_std[sel]
            ax.fill_between(x, y - ystd, y + ystd, alpha=0.25, linewidth=0)

        logger.info(f"Plotted {label} (source={meta.get('source','?')})")

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

    if getattr(ps, "xlim", None): ax.set_xlim(ps.xlim[0], ps.xlim[1])
    if getattr(ps, "ylim", None): ax.set_ylim(ps.ylim[0], ps.ylim[1])
    if getattr(ps, "grid", True):  ax.grid(True, which="both", ls="--", alpha=0.5)
    if getattr(ps, "tight_layout", True): plt.tight_layout()

    if getattr(ps, "save_plot", True):
        name = getattr(ps, "filename", "ipr.png")
        fig.savefig(outdir / name, dpi=getattr(ps, "dpi", 400), bbox_inches="tight")
        logger.success(f"Saved plot → {outdir / name}")

    if getattr(ps, "show", False):
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
