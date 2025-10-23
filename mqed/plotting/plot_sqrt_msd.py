from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from loguru import logger


def _resolve_path(p: str) -> Path:
    """Expand env vars and ~, return absolute Path."""
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()


def _find_newest(pattern: str) -> Path | None:
    """Return newest file matching a glob pattern, or None."""
    from glob import glob
    hits = sorted(glob(pattern), key=lambda s: Path(s).stat().st_mtime, reverse=True)
    return Path(hits[0]).resolve() if hits else None


def _load_dx_and_time(h5_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
        t_ps: (T,)
        dx_nm: (T,)  (mean if available; otherwise single-run Δx)
        meta: dict with info about what we loaded
    Supports:
      - datasets: 'dx_mean_nm' (preferred), 'dx_nm', or expectations: X_shift, X_shift2 (compute Δx)
    """
    meta = {}
    with h5py.File(str(h5_path), "r") as f:
        # time
        if "t_ps" not in f:
            raise ValueError(f"{h5_path} has no 't_ps' dataset.")
        t_ps = np.asarray(f["t_ps"][...]).ravel()
        # breakpoint()

        # Δx directly?
        if "dx_mean_nm" in f:
            dx = np.asarray(f["dx_mean_nm"][...]).ravel()
            meta["source"] = "dx_mean_nm"
        elif "dx_nm" in f:
            dx = np.asarray(f["dx_nm"][...]).ravel()
            meta["source"] = "dx_nm"
        else:
            # try expectations group
            if "expectations" in f and "X_shift" in f["expectations"] and "X_shift2" in f["expectations"]:
                ex1 = np.asarray(f["expectations"]["X_shift"][...]).ravel()
                ex2 = np.asarray(f["expectations"]["X_shift2"][...]).ravel()
                dx = np.sqrt(np.maximum(0.0, ex2 - ex1**2))
                meta["source"] = "expectations"
            else:
                raise ValueError(
                    f"{h5_path} does not contain 'dx_mean_nm'/'dx_nm' "
                    "or expectations/['X_shift','X_shift2']."
                )

        # Optional attrs
        for k in ("method", "n_realizations", "sigma_phi_deg", "seed_base", "state_format"):
            if k in f.attrs:
                meta[k] = f.attrs[k]

    if dx.shape != t_ps.shape:
        raise ValueError(f"{h5_path} Δx shape {dx.shape} and t_ps shape {t_ps.shape} mismatch.")

    return t_ps, dx, meta


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


def _resolve_input_path(curve_cfg) -> Path:
    """
    Either use an absolute 'path', or if 'use_latest_glob' is set,
    choose newest file matching the glob (relative to MQED_ROOT/PWD if present).
    """
    if getattr(curve_cfg, "path", None):
        return _resolve_path(curve_cfg.path)
    if getattr(curve_cfg, "use_latest_glob", None):
        base = os.environ.get("MQED_ROOT", os.environ.get("PWD", "."))
        newest = _find_newest(os.path.join(base, curve_cfg.use_latest_glob))
        if newest is None:
            raise FileNotFoundError(f"No files for pattern: {curve_cfg.use_latest_glob}")
        return newest
    raise ValueError("Each curve needs either 'path' or 'use_latest_glob'.")


@hydra.main(config_path="../../configs/plots", config_name="sqrt_msd", version_base=None)
def main(cfg: DictConfig) -> None:
    outdir = Path(HydraConfig.get().runtime.output_dir)

    ps = cfg.plot_settings
    fig, ax = plt.subplots(figsize=(ps.figsize[0], ps.figsize[1]) if getattr(ps, "figsize", None) else (7, 5))

    for curve in cfg.curves:
        path = _resolve_input_path(curve)
        t_ps, dx_nm, meta = _load_dx_and_time(path)

        sel = _select_x(t_ps, ps)
        x = t_ps[sel] * getattr(ps, "x_scale_factor", 1.0)   # keep default 1.0; you can set 1e-12 if you want 10^-10 s scaling, etc.
        y = dx_nm[sel]

        # style
        style = getattr(curve, "style", "-")
        lw = getattr(curve, "lw", ps.get("lw", 1.5))
        label = getattr(curve, "label", path.stem)

        ax.plot(x, y, style, lw=lw, label=label)

        logger.info(f"Plotted {label} from {path.name} (source={meta.get('source','?')})")

    # labels and title
    ax.set_xlabel(ps.xlabel)
    ax.set_ylabel(ps.ylabel)
    if getattr(ps, "title", None):
        ax.set_title(ps.title)

    # scales
    if getattr(ps, "xscale", None): ax.set_xscale(ps.xscale)
    if getattr(ps, "yscale", None): ax.set_yscale(ps.yscale)

    # limits
    if getattr(ps, "xlim", None): ax.set_xlim(ps.xlim[0], ps.xlim[1])
    if getattr(ps, "ylim", None): ax.set_ylim(ps.ylim[0], ps.ylim[1])

    if getattr(ps, "grid", True):
        ax.grid(True, which="both", ls="--", alpha=0.5)

    if getattr(ps, "legend", True):
        ax.legend()

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
