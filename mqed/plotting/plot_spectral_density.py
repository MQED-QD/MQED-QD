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

from ast import literal_eval
from typing import Any

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
from mqed.utils.hydra_local import prepare_hydra_config_path


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


def _parse_index_selection(raw_selection: Any, default_value: Any, selection_name: str) -> Any:
    """Normalize Hydra index selection values into plain Python objects."""
    if raw_selection is None:
        return default_value

    if isinstance(raw_selection, str):
        stripped = raw_selection.strip()
        if not stripped:
            return default_value
        try:
            return literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Invalid {selection_name}: {raw_selection!r}. "
                "Use Python-style list syntax such as [0, 3] or [[0, 0], [0, 3]]."
            ) from exc

    if hasattr(raw_selection, "__iter__") and not isinstance(raw_selection, (bytes, bytearray)):
        return list(raw_selection)

    return raw_selection


def _normalize_separation_indices(raw_selection: Any) -> list[int]:
    """Return a validated list of separation indices."""
    normalized = _parse_index_selection(
        raw_selection,
        default_value=[0],
        selection_name="plot_settings.separation_indices",
    )

    if isinstance(normalized, (int, float)):
        return [int(normalized)]

    if not isinstance(normalized, list):
        raise ValueError(
            "plot_settings.separation_indices must be an integer or a list of integers."
        )

    if not normalized:
        return [0]

    return [int(idx) for idx in normalized]


def _is_nested_index_collection(value: Any) -> bool:
    """Return True when a selection entry is itself a collection of indices."""
    return hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray))


def _normalize_pair_indices(raw_selection: Any) -> list[list[int]]:
    """Return a validated list of [alpha, beta] pair indices."""
    normalized = _parse_index_selection(
        raw_selection,
        default_value=[[0, 0]],
        selection_name="plot_settings.pair_indices",
    )

    if not isinstance(normalized, list):
        raise ValueError(
            "plot_settings.pair_indices must be a pair like [0, 3] or a list of pairs."
        )

    if not normalized:
        return [[0, 0]]

    if len(normalized) == 2 and all(not _is_nested_index_collection(item) for item in normalized):
        first = normalized[0]
        second = normalized[1]
        return [[int(first), int(second)]]

    pair_indices = []
    for pair in normalized:
        if not _is_nested_index_collection(pair) or len(pair) != 2:
            raise ValueError(
                "Each pair in plot_settings.pair_indices must have exactly two entries."
            )
        pair_indices.append([int(pair[0]), int(pair[1])])

    return pair_indices


def _normalize_curve_scales(raw_scales: Any, count: int, setting_name: str) -> list[float]:
    """Return one multiplicative scale factor per plotted curve."""
    normalized = _parse_index_selection(
        raw_scales,
        default_value=[1.0] * count,
        selection_name=setting_name,
    )

    if isinstance(normalized, (int, float)):
        return [float(normalized)] * count

    if not isinstance(normalized, list):
        raise ValueError(f"{setting_name} must be a number or a list of numbers.")

    if not normalized:
        return [1.0] * count

    if len(normalized) == 1 and count > 1:
        return [float(normalized[0])] * count

    if len(normalized) != count:
        raise ValueError(
            f"{setting_name} must contain {count} value(s) to match the selected curves."
        )

    return [float(scale) for scale in normalized]


def _normalize_curve_styles(raw_styles: Any, count: int, setting_name: str) -> list[Any]:
    """Return one optional style entry per plotted curve."""
    if raw_styles is None:
        return [None] * count

    if isinstance(raw_styles, str):
        return [raw_styles] * count

    if not isinstance(raw_styles, list):
        raw_styles = list(raw_styles)

    if not raw_styles:
        return [None] * count

    if len(raw_styles) == 1 and count > 1:
        return [raw_styles[0]] * count

    if len(raw_styles) != count:
        raise ValueError(
            f"{setting_name} must contain {count} value(s) to match the selected curves."
        )

    return list(raw_styles)


def _resolve_curve_multipliers(ps, primary_key: str, legacy_key: str, count: int) -> list[float]:
    """Load curve multipliers with a backwards-compatible fallback key."""
    raw_multipliers = ps.get(primary_key, None)
    if raw_multipliers is None:
        raw_multipliers = ps.get(legacy_key, None)

    return _normalize_curve_scales(raw_multipliers, count=count, setting_name=f"plot_settings.{primary_key}")


def _format_scaled_label(base_label: str, scale_factor: float) -> str:
    """Append a multiplier annotation to the legend label when needed."""
    if scale_factor == 1.0:
        return base_label

    if base_label.startswith("$") and base_label.endswith("$"):
        return f"{base_label[:-1]}\\,\\times\\,{scale_factor:g}$"

    return f"{base_label} ×{scale_factor:g}"


def _validate_curve_multiplier(scale_factor: float, yscale: str, setting_name: str) -> None:
    """Reject invalid multipliers for the active y-axis scale."""
    if yscale == "log" and scale_factor <= 0:
        raise ValueError(f"{setting_name} values must be positive when plot_settings.yscale is 'log'.")


def _resolve_curve_styles(ps, prefix: str, count: int) -> tuple[list[Any], list[Any]]:
    """Load optional per-curve colors and linestyles for one plot layout."""
    colors = _normalize_curve_styles(
        ps.get(f"{prefix}_colors", None),
        count=count,
        setting_name=f"plot_settings.{prefix}_colors",
    )
    linestyles = _normalize_curve_styles(
        ps.get(f"{prefix}_linestyles", None),
        count=count,
        setting_name=f"plot_settings.{prefix}_linestyles",
    )
    return colors, linestyles


def _plot_separation_layout(J_eV, energy_eV, Rx_nm, cfg):
    """Plot J(ω) for separation-indexed data.

    Produces one curve per selected separation Rx.
    """
    ps = cfg.plot_settings

    # Select which separations to plot
    sep_indices = _normalize_separation_indices(ps.get("separation_indices", [0]))
    yscale = ps.get("yscale", "linear")
    scale_factors = _resolve_curve_multipliers(
        ps,
        primary_key="separation_multipliers",
        legacy_key="separation_scale_factors",
        count=len(sep_indices),
    )
    colors, linestyles = _resolve_curve_styles(ps, prefix="separation", count=len(sep_indices))

    fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))

    for idx, scale_factor, color, linestyle in zip(
        sep_indices,
        scale_factors,
        colors,
        linestyles,
    ):
        if idx >= len(Rx_nm):
            logger.warning(f"Separation index {idx} out of range "
                           f"(max {len(Rx_nm) - 1}), skipping.")
            continue

        _validate_curve_multiplier(
            scale_factor,
            yscale=yscale,
            setting_name="plot_settings.separation_multipliers",
        )

        label = ps.get("label_template", "Rx = {Rx:.1f} nm").format(Rx=Rx_nm[idx])
        label = _format_scaled_label(label, scale_factor)
        ax.plot(
            energy_eV,
            scale_factor * J_eV[idx, :],
            lw=ps.get("lw", 1.5),
            label=label,
            color=color,
            linestyle=linestyle,
        )

    ax.set_xlabel(ps.get("xlabel", r"Energy (eV)"))
    ax.set_ylabel(ps.get("ylabel", r"$J(\omega)$ (eV)"))

    title_template = ps.get("title", r"Spectral Density $J(\omega)$")
    ax.set_title(title_template)

    if yscale == "log":
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

    if ax.lines:
        ax.legend()
    else:
        logger.warning("No valid separation indices were plotted.")
    fig.tight_layout()
    return fig


def _plot_pair_layout(J_eV, energy_eV, cfg):
    """Plot J_αβ(ω) for pair-indexed data.

    Produces one curve per selected (α, β) pair.
    """
    ps = cfg.plot_settings

    # Select which pairs to plot: list of [alpha, beta] pairs
    # Default: self-term of emitter 0
    pair_indices = _normalize_pair_indices(ps.get("pair_indices", [[0, 0]]))
    yscale = ps.get("yscale", "linear")
    scale_factors = _resolve_curve_multipliers(
        ps,
        primary_key="pair_multipliers",
        legacy_key="pair_scale_factors",
        count=len(pair_indices),
    )
    colors, linestyles = _resolve_curve_styles(ps, prefix="pair", count=len(pair_indices))

    fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))

    N = J_eV.shape[0]
    for pair, scale_factor, color, linestyle in zip(
        pair_indices,
        scale_factors,
        colors,
        linestyles,
    ):
        alpha, beta = int(pair[0]), int(pair[1])
        if alpha >= N or beta >= N:
            logger.warning(f"Pair ({alpha}, {beta}) out of range "
                           f"(N={N}), skipping.")
            continue

        _validate_curve_multiplier(
            scale_factor,
            yscale=yscale,
            setting_name="plot_settings.pair_multipliers",
        )

        label = ps.get(
            "label_template",
            r"$J_{{\alpha={a},\beta={b}}}(\omega)$"
        ).format(a=alpha, b=beta)
        label = _format_scaled_label(label, scale_factor)
        ax.plot(
            energy_eV,
            scale_factor * J_eV[alpha, beta, :],
            lw=ps.get("lw", 1.5),
            label=label,
            color=color,
            linestyle=linestyle,
        )

    ax.set_xlabel(ps.get("xlabel", r"Energy (eV)"))
    ax.set_ylabel(ps.get("ylabel", r"$J_{\alpha\beta}(\omega)$ (eV)"))

    title_template = ps.get(
        "title", r"Spectral Density $J_{\alpha\beta}(\omega)$"
    )
    ax.set_title(title_template)

    if yscale == "log":
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

    if ax.lines:
        ax.legend()
    else:
        logger.warning("No valid pair indices were plotted.")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
#  Hydra CLI entry point
# ---------------------------------------------------------------------------


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("plots", __file__)

@hydra.main(
    config_path=HYDRA_CONFIG_PATH,
    config_name="plt_spec_dens",
    version_base=None,
)
def plot_spectral_density(cfg=None) -> None:
    """Plot spectral density from pre-computed HDF5 data.

    This is the Hydra CLI entry point.  Configuration is loaded from
    ``configs/plots/plt_spec_dens.yaml``.
    """
    if cfg is None:
        raise ValueError("Hydra did not provide a plotting configuration.")

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
