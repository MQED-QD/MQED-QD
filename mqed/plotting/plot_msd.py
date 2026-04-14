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


def _to_plot_time(t_ps: np.ndarray, cfg_ps) -> np.ndarray:
    unit = str(getattr(cfg_ps, "time_unit", "ps")).lower()
    if unit == "ps":
        return t_ps
    if unit == "fs":
        return t_ps * 1.0e3
    if unit == "s":
        return t_ps * 1.0e-12
    raise ValueError(f"Unsupported plot_settings.time_unit='{unit}'. Use 'fs', 'ps', or 's'.")


def _msd_analytical_local(t_fs: np.ndarray, a: float, hbar_eV_fs: float, j_0_eV: float, sigma_j_eV: float) -> np.ndarray:
    j_eff_sq = j_0_eV ** 2 + sigma_j_eV ** 2
    return 2.0 * a ** 2 * j_eff_sq * t_fs ** 2 / (hbar_eV_fs ** 2)


def _x_square_analytical_gaussian(
    t_fs: np.ndarray,
    a: float,
    hbar_eV_fs: float,
    j_0_eV: float,
    sigma_j_eV: float,
    k_parallel: float,
    sigma_sites: float,
) -> np.ndarray:
    term1 = sigma_sites ** 2 / 2.0
    term2 = (4.0 * j_0_eV ** 2 / hbar_eV_fs ** 2) * np.sin(k_parallel * a) ** 2
    term3 = 2.0 * sigma_j_eV ** 2 / hbar_eV_fs ** 2
    return a ** 2 * (term1 + (term2 + term3) * t_fs ** 2)


def _position_analytical_gaussian(t_fs: np.ndarray, a: float, hbar_eV_fs: float, j_0_eV: float, k_parallel: float) -> np.ndarray:
    velocity_prefactor = (-2.0 * j_0_eV / hbar_eV_fs) * np.sin(k_parallel * a)
    return a * velocity_prefactor * t_fs


def _nn_msd_analytical(t_fs: np.ndarray, model: str, params: dict) -> np.ndarray:
    """Analytical MSD for nearest-neighbour chain models.

    MSD = <(x-x0)^2>  (second moment of displacement from initial site).
    Note: this is NOT the variance <(x-x0)^2> - <x-x0>^2.
    """
    a = float(params.get("a", 1.0))
    hbar_eV_fs = float(params.get("hbar_eV_fs", 0.6582119514))
    j_0_eV = float(params["J_0_eV"])
    sigma_j_eV = float(params["sigma_J_eV"])
    if model == "local_excitation":
        return _msd_analytical_local(t_fs, a, hbar_eV_fs, j_0_eV, sigma_j_eV)
    if model == "gaussian_wave":
        k_parallel = float(params["k_parallel"])
        sigma_sites = float(params["sigma_sites"])
        # MSD = <(x-x0)^2> = x2 (the full second moment, not x2 - <x>^2)
        x2 = _x_square_analytical_gaussian(t_fs, a, hbar_eV_fs, j_0_eV, sigma_j_eV, k_parallel, sigma_sites)
        return x2
    raise ValueError(f"Unsupported analytical model '{model}'. Use 'local_excitation' or 'gaussian_wave'.")

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
        t_ps_ds = f.get("t_ps")
        if not isinstance(t_ps_ds, h5py.Dataset):
            raise ValueError(f"{h5_path} has no 't_ps' dataset.")
        t_ps = np.asarray(t_ps_ds[...]).ravel()
        # breakpoint()

        msd_ds = f.get("msd_nm2")
        ex_group = f.get("expectations")
        dx_mean_ds = f.get("dx_mean_nm")

        # 1) direct MSD dataset?
        if isinstance(msd_ds, h5py.Dataset):
            msd = np.asarray(msd_ds[...]).ravel()
            meta["source"] = "msd_nm2"

        # 2) expectations group
        elif isinstance(ex_group, h5py.Group) and "x2_mean" in ex_group and "position_mean" in ex_group:
            x2_ds = ex_group.get("x2_mean")
            x_ds = ex_group.get("position_mean")
            if not isinstance(x2_ds, h5py.Dataset) or not isinstance(x_ds, h5py.Dataset):
                raise ValueError(f"{h5_path} has invalid expectations/x2_mean or position_mean dataset.")
            x2 = np.asarray(x2_ds[...]).ravel()
            # MSD = <(x-x0)^2> = x2 (second moment of displacement).
            # Note: x2 - <x>^2 would be the variance, not the MSD.
            msd = x2
            meta["source"] = "expectations/x2_mean,position_mean"
        elif isinstance(ex_group, h5py.Group) and "X_shift2" in ex_group and "X_shift" in ex_group:
            x2_ds = ex_group.get("X_shift2")
            x_ds = ex_group.get("X_shift")
            if not isinstance(x2_ds, h5py.Dataset) or not isinstance(x_ds, h5py.Dataset):
                raise ValueError(f"{h5_path} has invalid expectations/X_shift2 or X_shift dataset.")
            x2 = np.asarray(x2_ds[...]).ravel()
            # MSD = <(x-x0)^2> = x2 (second moment of displacement).
            msd = x2
            meta["source"] = "expectations/X_shift2,X_shift"
        elif isinstance(ex_group, h5py.Group) and "msd_mean" in ex_group:
            msd_mean_ds = ex_group.get("msd_mean")
            if not isinstance(msd_mean_ds, h5py.Dataset):
                raise ValueError(f"{h5_path} has invalid expectations/msd_mean dataset.")
            msd = np.asarray(msd_mean_ds[...]).ravel()
            meta["source"] = "expectations/msd_mean"
        elif isinstance(ex_group, h5py.Group) and "x2_mean" in ex_group:
            x2_ds = ex_group.get("x2_mean")
            if not isinstance(x2_ds, h5py.Dataset):
                raise ValueError(f"{h5_path} has invalid expectations/x2_mean dataset.")
            msd = np.asarray(x2_ds[...]).ravel()
            meta["source"] = "expectations/x2_mean"
        elif isinstance(ex_group, h5py.Group) and "X_shift2" in ex_group:
            x2_ds = ex_group.get("X_shift2")
            if not isinstance(x2_ds, h5py.Dataset):
                raise ValueError(f"{h5_path} has invalid expectations/X_shift2 dataset.")
            msd = np.asarray(x2_ds[...]).ravel()
            meta["source"] = "expectations/X_shift2"

        # 3) last resort: square of sqrt-MSD (if that file only saved dx)
        elif isinstance(dx_mean_ds, h5py.Dataset):
            dx = np.asarray(dx_mean_ds[...]).ravel()
            msd = dx**2
            meta["source"] = "dx_mean_nm**2"

        else:
            raise ValueError(
                f"{h5_path} does not contain 'msd_nm2', "
                "'expectations/msd_mean', {'x2_mean','position_mean'}, {'X_shift2','X_shift'}, "
                "'expectations/x2_mean', 'expectations/X_shift2', or 'dx_mean_nm'."
            )

        # carry over a few helpful attributes if present
        for k in (
            "method",
            "mode",
            "n_realizations",
            "sigma_phi_deg",
            "seed_base",
            "J_0_eV",
            "sigma_J_eV",
            "k_parallel",
            "sigma_sites",
            "eps_0_eV",
        ):
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
    if hasattr(cfg_ps, "x_range") and cfg_ps.x_range:
        t_plot = _to_plot_time(t_ps, cfg_ps)
        xmin, xmax = float(cfg_ps.x_range[0]), float(cfg_ps.x_range[1])
        return (t_plot >= xmin) & (t_plot <= xmax)
    if hasattr(cfg_ps, "x_range_ps") and cfg_ps.x_range_ps:
        xmin, xmax = float(cfg_ps.x_range_ps[0]), float(cfg_ps.x_range_ps[1])
        return (t_ps >= xmin) & (t_ps <= xmax)
    if hasattr(cfg_ps, "x_range_fs") and cfg_ps.x_range_fs:
        t_fs = t_ps * 1.0e3
        xmin, xmax = float(cfg_ps.x_range_fs[0]), float(cfg_ps.x_range_fs[1])
        return (t_fs >= xmin) & (t_fs <= xmax)
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


    loaded_curves = []
    for curve in cfg.curves:
        path = _resolve_input_path(curve)
        t_ps, msd, meta = _load_dx_and_time(path)

        sel = _select_x(t_ps, ps)
        x = _to_plot_time(t_ps[sel], ps) * getattr(ps, "x_scale_factor", 1.0)
        y = msd[sel]
        # style
        linestyle = getattr(curve, "linestyle", getattr(curve, "style", "-"))
        if isinstance(linestyle, str) and linestyle.lower() == "none":
            linestyle = "None"

        lw = getattr(curve, "lw", ps.get("lw", 1.5))
        label = getattr(curve, "label", path.stem)
        color = getattr(curve, "color", None)

        marker = getattr(curve, "marker", None)
        markersize = getattr(curve, "markersize", None)
        markerfacecolor = getattr(curve, "markerfacecolor", None)
        markeredgecolor = getattr(curve, "markeredgecolor", color)
        markeredgewidth = getattr(curve, "markeredgewidth", None)
        markevery = getattr(curve, "markevery", None)
        alpha = getattr(curve, "alpha", None)
        zorder = getattr(curve, "zorder", None)

        plot_kwargs = {
            "lw": lw,
            "label": label,
            "color": color,
            "linestyle": linestyle,
        }

        if marker is not None:
            plot_kwargs["marker"] = marker
        if markersize is not None:
            plot_kwargs["markersize"] = markersize
        if markerfacecolor is not None:
            plot_kwargs["markerfacecolor"] = markerfacecolor
        if markeredgecolor is not None:
            plot_kwargs["markeredgecolor"] = markeredgecolor
        if markeredgewidth is not None:
            plot_kwargs["markeredgewidth"] = markeredgewidth
        if markevery is not None:
            plot_kwargs["markevery"] = markevery
        if alpha is not None:
            plot_kwargs["alpha"] = alpha
        if zorder is not None:
            plot_kwargs["zorder"] = zorder

        ax.plot(x, y, **plot_kwargs)
        loaded_curves.append({"t_ps": t_ps, "sel": sel, "meta": meta})

        logger.info(f"Plotted {label} from {path.name} (source={meta.get('source','?')})")

    if bool(getattr(ps, "enable_analytical_curves", True)):
        for curve in getattr(cfg, "analytical_curves", []):
            if not loaded_curves:
                raise ValueError("analytical_curves requires at least one numerical curve to define time grid.")

            from_curve_index = int(getattr(curve, "from_curve_index", 0))
            if from_curve_index < 0 or from_curve_index >= len(loaded_curves):
                raise ValueError(
                    f"analytical from_curve_index={from_curve_index} out of range [0, {len(loaded_curves)-1}]."
                )

            ref = loaded_curves[from_curve_index]
            t_ps_ref = np.asarray(ref["t_ps"])
            sel = np.asarray(ref["sel"], dtype=bool)
            meta = dict(ref["meta"])
            params = dict(getattr(curve, "params", {}) or {})

            for key in ("J_0_eV", "sigma_J_eV", "k_parallel", "sigma_sites"):
                if key not in params and key in meta:
                    params[key] = meta[key]

            model = str(getattr(curve, "model", "gaussian_wave"))
            t_fs = t_ps_ref[sel] * 1.0e3
            y = _nn_msd_analytical(t_fs, model, params)
            x = _to_plot_time(t_ps_ref[sel], ps) * getattr(ps, "x_scale_factor", 1.0)

            style = getattr(curve, "style", "-")
            lw = getattr(curve, "lw", ps.get("lw", 1.5))
            label = getattr(curve, "label", f"Analytical ({model})")
            color = getattr(curve, "color", None)
            ax.plot(x, y, style, lw=lw, label=label, color=color)
            logger.info(
                f"Plotted analytical curve: {label} (model={model}, from_curve_index={from_curve_index})"
            )

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


    ysc = getattr(ps, "y_sci", None)
    if ysc and getattr(ysc, "enabled", False):
        logger.info("Use scientific visualization on y axis.")
        if getattr(ysc, "style", "sci") == "sci":
            ax.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(
                    int(getattr(ysc, "scilimits", (-2, 2))[0]),
                    int(getattr(ysc, "scilimits", (-2, 2))[1]),
                ),
                useMathText=bool(getattr(ysc, "use_math_text", True)),
                
            )
            off = ax.yaxis.get_offset_text()
            off.set_fontsize(int(getattr(ysc,"offset_text_size",ticksize)))
        else:
            ax.ticklabel_format(axis="y", style="plain")

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
