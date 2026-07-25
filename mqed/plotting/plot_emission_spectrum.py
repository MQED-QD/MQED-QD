from ast import literal_eval
from pathlib import Path
from typing import Any

import h5py
import hydra
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf

from mqed.utils.hydra_local import prepare_hydra_config_path
from mqed.utils.logging_utils import setup_loggers_hydra_aware


def _load_emission_h5(filepath: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    with h5py.File(filepath, "r") as h5:
        for key in h5.keys():
            data[key] = h5[key][()]
        for key in h5.attrs:
            data[key] = h5.attrs[key]
    return data


def _apply_font_config(cfg) -> None:
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


def _parse_selection(raw_selection: Any, default_value: Any, selection_name: str) -> Any:
    if raw_selection is None:
        return default_value
    if isinstance(raw_selection, str):
        stripped = raw_selection.strip()
        if not stripped:
            return default_value
        try:
            return literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid {selection_name}: {raw_selection!r}.") from exc
    if hasattr(raw_selection, "__iter__") and not isinstance(raw_selection, (bytes, bytearray)):
        return list(raw_selection)
    return raw_selection


def _normalize_indices(raw_selection: Any, default_value: list[int]) -> list[int]:
    normalized = _parse_selection(raw_selection, default_value, "plot_settings.transition_indices")
    if isinstance(normalized, (int, float)):
        return [int(normalized)]
    if not isinstance(normalized, list):
        raise ValueError("plot_settings.transition_indices must be an integer or list of integers.")
    if not normalized:
        return default_value
    return [int(value) for value in normalized]


def _normalize_transition_values(raw_selection: Any) -> list[float]:
    normalized = _parse_selection(raw_selection, [], "plot_settings.transition_values_eV")
    if isinstance(normalized, (int, float)):
        return [float(normalized)]
    if not isinstance(normalized, list):
        raise ValueError("plot_settings.transition_values_eV must be a number or list of numbers.")
    return [float(value) for value in normalized]


def _resolve_transition_indices(ps, transition_energy_eV: np.ndarray) -> tuple[list[int], list[float]]:
    values_eV = _normalize_transition_values(ps.get("transition_values_eV", []))
    if not values_eV:
        indices = _normalize_indices(ps.get("transition_indices", [0]), [0])
        return indices, [float(transition_energy_eV[idx]) for idx in indices]

    tolerance_eV = float(ps.get("transition_value_tolerance_eV", 1e-9))
    indices: list[int] = []
    values: list[float] = []
    for value in values_eV:
        matches = np.where(np.isclose(transition_energy_eV, value, rtol=0.0, atol=tolerance_eV))[0]
        if matches.size == 0:
            nearest = int(np.argmin(np.abs(transition_energy_eV - value)))
            logger.warning(
                "Transition energy {:.6g} eV not found within {:.3g} eV; nearest is "
                "index {} at {:.6g} eV, skipping.",
                value,
                tolerance_eV,
                nearest,
                float(transition_energy_eV[nearest]),
            )
            continue
        idx = int(matches[0])
        indices.append(idx)
        values.append(float(transition_energy_eV[idx]))
    return indices, values


def _plot_map(spectrum: np.ndarray, emission_energy_eV: np.ndarray, transition_energy_eV: np.ndarray, cfg):
    ps = cfg.plot_settings
    fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [7, 5])))
    image = ax.imshow(
        spectrum.T,
        origin="lower",
        aspect="auto",
        extent=[
            float(transition_energy_eV[0]),
            float(transition_energy_eV[-1]),
            float(emission_energy_eV[0]),
            float(emission_energy_eV[-1]),
        ],
        cmap=ps.get("cmap", "viridis"),
    )
    ax.set_xlabel(ps.get("map_xlabel", ps.get("xlabel", r"Transition energy $\omega_0$ (eV)")))
    ax.set_ylabel(ps.get("map_ylabel", ps.get("ylabel", r"Emission energy $\omega$ (eV)")))
    ax.set_title(ps.get("title", "Emission spectrum"))
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(ps.get("colorbar_label", "D(ω)"))
    fig.tight_layout()
    return fig


def _plot_curves(spectrum: np.ndarray, emission_energy_eV: np.ndarray, transition_energy_eV: np.ndarray, cfg):
    ps = cfg.plot_settings
    indices, values = _resolve_transition_indices(ps, transition_energy_eV)
    fig, ax = plt.subplots(figsize=tuple(ps.get("figsize", [8, 5])))
    for idx, value in zip(indices, values):
        if idx < 0 or idx >= spectrum.shape[0]:
            logger.warning("Transition index {} out of range; skipping.", idx)
            continue
        label = ps.get("label_template", "ω0 = {omega0:.3f} eV").format(
            omega0=value,
            index=idx,
        )
        ax.plot(emission_energy_eV, spectrum[idx], lw=ps.get("lw", 1.5), label=label)
    ax.set_xlabel(ps.get("xlabel", "Emission energy (eV)"))
    ax.set_ylabel(ps.get("curve_ylabel", ps.get("ylabel", "D(ω)")))
    ax.set_title(ps.get("title", "Emission spectrum"))
    if ps.get("xscale", "linear") == "log":
        ax.set_xscale("log")
    if ps.get("yscale", "linear") == "log":
        ax.set_yscale("log")
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
    fig.tight_layout()
    return fig


HYDRA_CONFIG_PATH: str = prepare_hydra_config_path("plots", __file__)


@hydra.main(config_path=HYDRA_CONFIG_PATH, config_name="plt_emission_spectrum", version_base=None)
def plot_emission_spectrum(cfg=None) -> None:
    if cfg is None:
        raise ValueError("Hydra did not provide a plotting configuration.")
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_loggers_hydra_aware()
    logger.info("Plotting emission spectrum")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    input_path = Path(cfg.input_file)
    if not input_path.is_absolute():
        input_path = Path(hydra.utils.get_original_cwd()) / input_path
    data = _load_emission_h5(str(input_path))
    spectrum = data["emission_spectrum"]
    emission_energy_eV = data["emission_energy_eV"]
    transition_energy_eV = data["transition_energy_eV"]
    _apply_font_config(cfg)

    plot_type = str(cfg.plot_settings.get("plot_type", "map")).strip().lower()
    if plot_type == "map":
        fig = _plot_map(spectrum, emission_energy_eV, transition_energy_eV, cfg)
    elif plot_type in {"curve", "curves"}:
        fig = _plot_curves(spectrum, emission_energy_eV, transition_energy_eV, cfg)
    else:
        raise ValueError("plot_settings.plot_type must be 'map' or 'curves'.")

    if cfg.plot_settings.get("save_plot", True):
        filename = cfg.plot_settings.get("filename", "emission_spectrum.png")
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=int(cfg.plot_settings.get("dpi", 300)), bbox_inches="tight")
        logger.success(f"Saved plot to: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    plot_emission_spectrum()
