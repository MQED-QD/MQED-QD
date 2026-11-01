"""Plot real-q reflection diagnostics saved by ``compute_mode_diagnostic``.

Bright reflection features and complex-q pole candidates are diagnostics, not
proof of a bound BSW/SPP mode; identification also requires localization,
dispersion, and branch/sheet checks. The horizontal coordinate is q/k0, where
k0 depends on energy, not a fixed physical momentum q.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def _edges(centers: np.ndarray) -> np.ndarray:
    if centers.size == 1:
        half_width = max(abs(float(centers[0])) * 0.02, 0.02)
        return np.array([centers[0] - half_width, centers[0] + half_width])
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0] - (midpoints[0] - centers[0]), midpoints,
                 centers[-1] + (centers[-1] - midpoints[-1])]


def _source_light_line(data, energy: np.ndarray) -> np.ndarray | None:
    for key in ("source_epsilon", "source_layer_epsilon"):
        if key not in data:
            continue
        epsilon = np.asarray(data[key], dtype=complex)
        if epsilon.shape == ():
            epsilon = np.full(energy.shape, epsilon.item(), dtype=complex)
        if epsilon.shape != energy.shape:
            raise ValueError(f"{key} must be a scalar or have shape {energy.shape}.")
        allowed = (np.isfinite(epsilon) & (epsilon.real > 0) & (epsilon.imag == 0))
        if np.any(allowed):
            line = np.full(energy.shape, np.nan)
            line[allowed] = np.sqrt(epsilon.real[allowed])
            return line
    return None


def plot_mode_diagnostic(input_path: Path, output_path: Path, direction: str = "down") -> None:
    """Save a two-panel TE/TM reflection map from a diagnostic NPZ.

    Args:
        input_path: NPZ produced by ``compute_mode_diagnostic``.
        output_path: Image destination (format inferred from extension).
        direction: Saved reflection direction, usually ``down`` or ``up``.
    """
    with np.load(input_path, allow_pickle=False) as data:
        energy = np.asarray(data["energy_eV"], dtype=float)
        momentum = np.asarray(data["q_over_k0"], dtype=float)
        directions = list(np.asarray(data["directions"]).astype(str))
        polarizations = list(np.asarray(data["polarizations"]).astype(str))
        reflection = np.asarray(data["reflection"])
        valid = np.asarray(data["reflection_valid"], dtype=bool)
        source_name = str(np.asarray(data["source_layer_name"]).item())
        if (energy.ndim != 1 or momentum.ndim != 1 or not energy.size or not momentum.size
                or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(momentum))
                or np.any(np.diff(energy) <= 0) or np.any(np.diff(momentum) <= 0)):
            raise ValueError("energy_eV and q_over_k0 must be nonempty, finite strictly ascending grids.")
        expected = (energy.size, momentum.size, len(directions), len(polarizations))
        if reflection.shape != expected or valid.shape != expected:
            raise ValueError(f"reflection and reflection_valid must have shape {expected}.")
        if direction not in directions:
            raise ValueError(f"Direction {direction!r} not saved; available: {directions}.")
        if not {"s", "p"}.issubset(polarizations):
            raise ValueError("Both s (TE) and p (TM) polarizations are required.")

        direction_index = directions.index(direction)
        maps = []
        for pol in ("s", "p"):
            samples = reflection[:, :, direction_index, polarizations.index(pol)]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                values = np.log10(np.abs(samples))
            maps.append(np.ma.masked_where(~valid[:, :, direction_index,
                                                polarizations.index(pol)] | ~np.isfinite(values), values))

        finite = np.concatenate([item.compressed() for item in maps])
        if finite.size:
            lower, upper = float(finite.min()), float(finite.max())
            if lower == upper:
                lower -= 0.5
                upper += 0.5
        else:
            lower, upper = 0.0, 1.0

        source_line = _source_light_line(data, energy)
        candidates = None
        if {"pole_energy_eV", "pole_q_over_k0", "pole_polarization"}.issubset(data.files):
            candidate_energy = float(np.asarray(data["pole_energy_eV"]).item())
            candidate_q = np.atleast_1d(np.asarray(data["pole_q_over_k0"], dtype=complex))
            candidate_pol = np.atleast_1d(np.asarray(data["pole_polarization"]).astype(str))
            if candidate_q.shape != candidate_pol.shape:
                raise ValueError("pole_q_over_k0 and pole_polarization must have matching shapes.")
            candidates = (candidate_energy, candidate_q, candidate_pol)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True,
                             constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    norm = Normalize(vmin=lower, vmax=upper)
    for ax, pol, values in zip(axes, ("s", "p"), maps):
        image = ax.pcolormesh(_edges(momentum), _edges(energy), values,
                              cmap=cmap, norm=norm, shading="flat")
        ax.axvline(1, color="white", linestyle="--", linewidth=1.5,
                   label=r"Air light line $q/k_0=1$")
        if source_line is not None:
            ax.plot(source_line, energy, color="cyan", linestyle=":", linewidth=1.5,
                    label="Source-layer light line")
        if candidates is not None:
            candidate_energy, candidate_q, candidate_pol = candidates
            selected = ((candidate_pol == pol) & np.isfinite(candidate_q)
                        & np.isfinite(candidate_energy))
            if np.any(selected):
                ax.scatter(candidate_q[selected].real,
                           np.full(np.count_nonzero(selected), candidate_energy),
                           s=65, marker="x", color="red", linewidths=1.8,
                           label=r"Pole candidate ($\operatorname{Re} q/k_0$)", zorder=5)
        ax.set_title(f"{'TE (s)' if pol == 's' else 'TM (p)'} — {direction} reflection")
        ax.set_xlabel(r"In-plane momentum $q/k_0$ (energy-dependent $k_0$)")
        ax.legend(loc="best", fontsize="small", facecolor="white", framealpha=0.85)
        if not values.count():
            ax.text(0.5, 0.5, "No finite nonzero reflection samples", transform=ax.transAxes,
                    ha="center", va="center", bbox={"facecolor": "white", "alpha": 0.85})
    axes[0].set_ylabel("Energy (eV)")
    fig.colorbar(image, ax=axes, label=r"$\log_{10}|r|$ (amplitude reflection)", shrink=0.85)
    fig.suptitle(f"Source layer: {source_name} | {direction} generalized reflection")
    fig.supxlabel("Reflection peaks and pole candidates alone do not establish a bound mode; "
                 "check localization, dispersion and branch conditions.", fontsize=9)
    try:
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and save a headless diagnostic figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ from mode_diagnostic")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument("--direction", choices=("down", "up"), default="down",
                        help="Reflection direction (default: down)")
    args = parser.parse_args(argv)
    plot_mode_diagnostic(args.input, args.output, args.direction)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
