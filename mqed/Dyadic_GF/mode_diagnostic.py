"""Real-q optical-mode diagnostics for a bottom-to-top N-layer stack.

Run with ``python -m mqed.Dyadic_GF.mode_diagnostic --config STACK.yaml
--output /path/to/diagnostic.npz``. A bright reflection feature or a small
real-axis characteristic is not, by itself, evidence of a Bloch surface wave
(BSW) or a surface plasmon polariton (SPP). Such identification also needs
field localization, polarization, dispersion and branch/sheet checks.
In the DBR/DBP example, source layer 9 is the finite DBP spacer between
mirrors, while source layer 10 is top air looking down onto the full stack.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from mqed.Dyadic_GF.GF_NLayer import NLayerGreenFunction
from mqed.Dyadic_GF.data_provider import DataProvider
from mqed.Dyadic_GF.main_nlayer import build_layers
from mqed.Dyadic_GF.sommerfeld_singularities import PoleSearchConfig
from mqed.utils.SI_unit import c, eV_to_J, hbar


POLARIZATIONS = ("s", "p")
DIRECTIONS = ("down", "up")


def _grid(values, name: str, positive: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty, finite one-dimensional grid.")
    if np.any(result < 0) or (positive and np.any(result <= 0)):
        raise ValueError(f"{name} must contain {'positive' if positive else 'nonnegative'} values.")
    if np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must be in strictly ascending order.")
    return result


def _material_providers(cfg):
    providers = {}
    layer_keys = []
    for index, layer in enumerate(cfg.stack.layers):
        if "material" in layer:
            key = str(layer.material)
            material = cfg.materials[key]
        else:
            key = f"inline_{index}"
            material = layer.material_config
        if key not in providers:
            providers[key] = DataProvider(material)
        layer_keys.append(key)
    return providers, layer_keys


def _calculator(cfg, energy: float, source_layer: int, template, providers, layer_keys):
    omega = energy * eV_to_J / hbar
    layers = [
        replace(layer, epsilon=providers[key].get_epsilon(omega))
        for layer, key in zip(template, layer_keys)
    ]
    if not all(np.isfinite(layer.epsilon) for layer in layers):
        raise ValueError(f"Material permittivity is non-finite at {energy:g} eV.")
    return NLayerGreenFunction(layers=layers, source_layer=source_layer, omega=omega)


def _finite_complex(values) -> np.ndarray:
    result = np.asarray(values, dtype=complex)
    return np.where(np.isfinite(result), result, complex(np.nan, np.nan))


def compute_mode_diagnostic(
    cfg,
    energy_eV,
    q_over_k0,
    source_layer: int | None = None,
    pole_energy_eV: float | None = None,
) -> dict[str, np.ndarray]:
    """Sample generalized mirrors and the source-layer modal characteristic.

    Args:
        cfg: N-layer GF YAML configuration (OmegaConf object).
        energy_eV: Ascending positive energy grid in electronvolts.
        q_over_k0: Ascending nonnegative real in-plane momentum grid.
        source_layer: Zero-based incident/source layer; defaults to cfg.stack.source_layer.
        pole_energy_eV: Optional positive energy for a bounded complex-q search.

    Returns:
        NPZ-ready arrays. ``reflection`` has axes (energy, q, direction, pol),
        with direction (down, up) and pol (s=TE, p=TM). ``characteristic``
        has axes (energy, q, pol). For finite source layers this is the Airy
        denominator; for top exterior sources it is the downward recursive
        reflection denominator, NOT a finite-cavity Airy denominator. Boundary
        directions without a neighboring layer are NaN, and ``reflection_valid``
        and ``characteristic_valid`` mark finite samples. Optional pole arrays
        contain candidates, not confirmed BSW/SPP modes; residuals use the
        stack's ratio-free, normalized pole-search characteristic.
    """
    energies = _grid(energy_eV, "energy_eV", positive=True)
    momenta = _grid(q_over_k0, "q_over_k0")
    source = int(cfg.stack.source_layer if source_layer is None else source_layer)
    count = len(cfg.stack.layers)
    if not 0 <= source < count:
        raise ValueError(f"source_layer must be between 0 and {count - 1}.")
    if pole_energy_eV is not None and (not np.isfinite(pole_energy_eV) or pole_energy_eV <= 0):
        raise ValueError("pole_energy_eV must be finite and positive.")

    first_omega = energies[0] * eV_to_J / hbar
    template = build_layers(cfg.stack, cfg.materials, first_omega)
    providers, layer_keys = _material_providers(cfg)
    if template[source].thickness_m is None and source not in (0, count - 1):
        raise ValueError("A semi-infinite internal source layer has no Airy characteristic.")
    if template[source].thickness_m is None and source == 0:
        raise ValueError("The N-layer stack supports finite or top-exterior source layers only.")

    reflection = np.full((len(energies), len(momenta), 2, 2), np.nan + 1j * np.nan)
    characteristic = np.full((len(energies), len(momenta), 2), np.nan + 1j * np.nan)
    source_epsilon = np.empty(len(energies), dtype=complex)
    k0 = energies * eV_to_J / (hbar * c)
    for index, energy in enumerate(energies):
        gf = _calculator(cfg, energy, source, template, providers, layer_keys)
        source_epsilon[index] = gf.eps[source]
        q_m_inv = momenta * gf.k0
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for pol_index, pol in enumerate(POLARIZATIONS):
                for direction_index, direction in enumerate(DIRECTIONS):
                    if (direction == "down" and source == 0) or (
                        direction == "up" and source == count - 1
                    ):
                        continue
                    reflection[index, :, direction_index, pol_index] = _finite_complex(
                        gf.reflection_coefficient(q_m_inv, direction, pol)
                    )
                characteristic[index, :, pol_index] = _finite_complex(
                    gf.airy_denominator(q_m_inv, pol)
                )

    result = {
        "energy_eV": energies,
        "q_over_k0": momenta,
        "k0_m_inv": k0,
        "q_m_inv": k0[:, None] * momenta[None, :],
        "source_layer": np.asarray(source),
        "source_layer_name": np.asarray(template[source].name),
        "source_epsilon": source_epsilon,
        "directions": np.asarray(DIRECTIONS),
        "polarizations": np.asarray(POLARIZATIONS),
        "reflection": reflection,
        "reflection_valid": np.isfinite(reflection),
        "characteristic": characteristic,
        "characteristic_valid": np.isfinite(characteristic),
        "characteristic_kind": np.asarray(
            "top_exterior_reflection_denominator"
            if template[source].thickness_m is None else "finite_source_airy_denominator"
        ),
    }
    if pole_energy_eV is not None:
        gf = _calculator(cfg, pole_energy_eV, source, template, providers, layer_keys)
        real_min = float(momenta[0])
        real_max = float(momenta[-1])
        if real_min == real_max:
            raise ValueError("A pole search needs a nonzero q_over_k0 interval.")
        pole_cfg = PoleSearchConfig.from_k0(
            gf.k0, real_min_factor=real_min, real_max_factor=real_max,
            imag_min_factor=1e-6, imag_max_factor=0.2,
            contour_points_per_side=16, max_depth=8, max_boxes=2000,
            skip_branch_points=tuple(gf.branch_points()),
        )
        poles = gf.find_poles(config=pole_cfg)
        result.update({
            "pole_energy_eV": np.asarray(pole_energy_eV),
            "pole_q_m_inv": np.asarray([pole.q for pole in poles], dtype=complex),
            "pole_q_over_k0": np.asarray([pole.q / gf.k0 for pole in poles], dtype=complex),
            "pole_polarization": np.asarray([pole.polarization for pole in poles], dtype="U1"),
            "pole_residual": np.asarray([pole.residual for pole in poles], dtype=float),
        })
    return result


def main(argv: list[str] | None = None) -> None:
    """Run a finite-aware NPZ diagnostic without generating a plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing N-layer GF YAML path")
    parser.add_argument("--energy-min", type=float)
    parser.add_argument("--energy-max", type=float)
    parser.add_argument("--energy-points", type=int)
    parser.add_argument("--q-min", type=float, default=0.0, help="Minimum q/k0 (default: 0)")
    parser.add_argument("--q-max", type=float, default=5.0, help="Maximum q/k0 (default: 5)")
    parser.add_argument("--q-points", type=int, default=101)
    parser.add_argument("--source-layer", type=int, help=(
        "Zero-based incident layer; DBR example: 9=finite DBP spacer, "
        "10=top air looking down at the stack (default: stack.source_layer)"
    ))
    parser.add_argument("--output", type=Path, required=True, help="NPZ output path")
    parser.add_argument("--pole-energy", type=float, help="Optional complex-q search energy (eV)")
    args = parser.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    energy_cfg = cfg.simulation.energy_eV
    minimum = float(energy_cfg.min if args.energy_min is None else args.energy_min)
    maximum = float(energy_cfg.max if args.energy_max is None else args.energy_max)
    points = int(energy_cfg.points if args.energy_points is None else args.energy_points)
    if (points < 1 or args.q_points < 1 or not np.isfinite([minimum, maximum,
            args.q_min, args.q_max]).all() or minimum <= 0 or maximum < minimum
            or args.q_min < 0 or args.q_max < args.q_min):
        parser.error("Energy must be positive, q/k0 nonnegative, max >= min, and point counts >= 1.")
    result = compute_mode_diagnostic(
        cfg, np.linspace(minimum, maximum, points),
        np.linspace(args.q_min, args.q_max, args.q_points),
        source_layer=args.source_layer, pole_energy_eV=args.pole_energy,
    )
    np.savez_compressed(args.output, **result)
    valid = result["reflection_valid"]
    print(f"Saved {args.output}: layer {result['source_layer']} "
          f"({result['source_layer_name']}), {points} energies x {args.q_points} q/k0; "
          f"finite reflections {valid.sum()}/{valid.size}; "
          f"finite characteristics {result['characteristic_valid'].sum()}/"
          f"{result['characteristic_valid'].size}.")
    if args.pole_energy is not None:
        print(f"Complex-q pole candidates at {args.pole_energy:g} eV: "
              f"{len(result['pole_q_m_inv'])} (q/k0, normalized residual, TE/TM):")
        for q, residual, pol in zip(result["pole_q_over_k0"], result["pole_residual"],
                                    result["pole_polarization"]):
            print(f"  {q:.8g}  {residual:.3g}  {'TE' if pol == 's' else 'TM'}")
    print("Reflection peaks, denominator minima and candidate poles alone do not establish "
          "a BSW or SPP; check field localization, dispersion and branch conditions.")


if __name__ == "__main__":
    main()
