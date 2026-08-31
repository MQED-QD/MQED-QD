"""Validate separation-indexed Green-tensor HDF5 output.

The production solver already rejects non-finite values. This module adds a
post-run spectral continuity check because adaptive quadrature can return a
finite value after emitting an ``IntegrationWarning``. Such values are valid
floating-point data but may still be unconverged numerical outliers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode_layout(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _spike_records(
    values: np.ndarray,
    energy_eV: np.ndarray,
    rx_nm: np.ndarray,
    ratio_threshold: float,
) -> list[dict[str, Any]]:
    """Return isolated componentwise spectral excursions.

    A point is flagged when its magnitude exceeds both adjacent magnitudes by
    ``ratio_threshold``. Endpoints are compared with their sole neighbor. A
    small per-component magnitude floor prevents ratios between numerical zeros
    from being reported without allowing unrelated tensor components to set the
    comparison scale.
    """
    magnitudes = np.abs(values)
    magnitude_floor = np.zeros(magnitudes.shape[1:], dtype=float)
    for rx_index, row, column in np.ndindex(magnitude_floor.shape):
        trace = magnitudes[:, rx_index, row, column]
        nonzero = trace[trace > 0.0]
        if nonzero.size:
            magnitude_floor[rx_index, row, column] = np.median(nonzero) * 1e-6

    if not np.any(magnitude_floor):
        return []

    def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.zeros_like(numerator, dtype=float)
        np.divide(numerator, denominator, out=result, where=denominator > 0.0)
        return result

    records: list[dict[str, Any]] = []

    comparisons: list[tuple[int, np.ndarray]] = []
    if energy_eV.size >= 2:
        comparisons.append(
            (0, safe_ratio(magnitudes[0], np.maximum(magnitudes[1], magnitude_floor)))
        )
        comparisons.append(
            (-1, safe_ratio(magnitudes[-1], np.maximum(magnitudes[-2], magnitude_floor)))
        )
    if energy_eV.size >= 3:
        neighbor_scale = np.maximum(magnitudes[:-2], magnitudes[2:])
        comparisons.extend(
            (index + 1, ratio)
            for index, ratio in enumerate(
                safe_ratio(magnitudes[1:-1], np.maximum(neighbor_scale, magnitude_floor))
            )
        )

    for energy_index, ratios in comparisons:
        resolved_index = energy_index % energy_eV.size
        for rx_index, row, column in np.argwhere(
            (ratios >= ratio_threshold) & (magnitudes[resolved_index] >= magnitude_floor)
        ):
            value = values[resolved_index, rx_index, row, column]
            records.append(
                {
                    "energy_index": int(resolved_index),
                    "energy_eV": float(energy_eV[resolved_index]),
                    "rx_index": int(rx_index),
                    "rx_nm": float(rx_nm[rx_index]),
                    "component": (int(row), int(column)),
                    "ratio": float(ratios[rx_index, row, column]),
                    "magnitude": float(abs(value)),
                    "value": complex(value),
                }
            )

    records.sort(key=lambda record: record["ratio"], reverse=True)
    return records


def _report(
    path: Path,
    spike_ratio: float,
    *,
    errors: list[str] | None = None,
    nonfinite_total: int = 0,
    nonfinite_vacuum: int = 0,
    nonfinite_scattering: int = 0,
    spikes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report_errors = errors or []
    report_spikes = spikes or []
    return {
        "path": str(path),
        "valid": not report_errors and not report_spikes,
        "errors": report_errors,
        "nonfinite_total": nonfinite_total,
        "nonfinite_vacuum": nonfinite_vacuum,
        "nonfinite_scattering": nonfinite_scattering,
        "spike_ratio": float(spike_ratio),
        "spikes": report_spikes,
    }


def validate_separation_gf_h5(
    h5_path: str | Path,
    *,
    spike_ratio: float = 10.0,
) -> dict[str, Any]:
    """Validate one separation-indexed Green-tensor file.

    Args:
        h5_path: HDF5 file produced by the planar Green-function solvers.
        spike_ratio: Minimum center-to-neighbor magnitude ratio used to flag an
            isolated spectral excursion.

    Returns:
        A report containing structural errors, finite-value counts, and
        componentwise outlier coordinates for the scattering tensor.
    """
    if not np.isfinite(spike_ratio) or spike_ratio <= 1.0:
        raise ValueError("spike_ratio must be finite and greater than 1.")

    path = Path(h5_path)
    errors: list[str] = []
    try:
        with h5py.File(path, "r") as h5:
            layout = _decode_layout(h5.attrs.get("gf_layout", "separation"))
            if layout != "separation":
                errors.append(f"Expected gf_layout='separation'; got {layout!r}.")

            required = {
                "energy_eV",
                "Rx_nm",
                "green_function_total",
                "green_function_vacuum",
            }
            missing = sorted(required.difference(h5.keys()))
            if missing:
                errors.append(f"Missing required datasets: {missing}.")
                return _report(path, spike_ratio, errors=errors)

            energy_eV = np.asarray(h5["energy_eV"], dtype=float)
            rx_nm = np.asarray(h5["Rx_nm"], dtype=float)
            total = np.asarray(h5["green_function_total"])
            vacuum = np.asarray(h5["green_function_vacuum"])
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"Could not read Green-tensor data: {error}")
        return _report(path, spike_ratio, errors=errors)

    expected_shape = (energy_eV.size, rx_nm.size, 3, 3)
    if energy_eV.ndim != 1 or energy_eV.size == 0:
        errors.append("energy_eV must be a non-empty one-dimensional array.")
    elif not np.all(np.isfinite(energy_eV)):
        errors.append("energy_eV contains non-finite values.")
    elif not np.all(np.diff(energy_eV) > 0.0):
        errors.append("energy_eV must be strictly increasing.")
    if rx_nm.ndim != 1 or rx_nm.size == 0 or not np.all(np.isfinite(rx_nm)):
        errors.append("Rx_nm must be a non-empty finite one-dimensional array.")
    if total.shape != expected_shape:
        errors.append(f"green_function_total has shape {total.shape}; expected {expected_shape}.")
    if vacuum.shape != expected_shape:
        errors.append(f"green_function_vacuum has shape {vacuum.shape}; expected {expected_shape}.")

    nonfinite_total = int(total.size - np.count_nonzero(np.isfinite(total)))
    nonfinite_vacuum = int(vacuum.size - np.count_nonzero(np.isfinite(vacuum)))
    if nonfinite_total:
        errors.append(f"green_function_total contains {nonfinite_total} non-finite values.")
    if nonfinite_vacuum:
        errors.append(f"green_function_vacuum contains {nonfinite_vacuum} non-finite values.")

    nonfinite_scattering = 0
    spikes: list[dict[str, Any]] = []
    if not errors:
        with np.errstate(over="ignore", invalid="ignore"):
            scattering = total - vacuum
        nonfinite_scattering = int(
            scattering.size - np.count_nonzero(np.isfinite(scattering))
        )
        if nonfinite_scattering:
            errors.append(
                f"green_function_total - green_function_vacuum contains "
                f"{nonfinite_scattering} non-finite values."
            )
        elif energy_eV.size >= 2:
            spikes = _spike_records(scattering, energy_eV, rx_nm, spike_ratio)

    return _report(
        path,
        spike_ratio,
        errors=errors,
        nonfinite_total=nonfinite_total,
        nonfinite_vacuum=nonfinite_vacuum,
        nonfinite_scattering=nonfinite_scattering,
        spikes=spikes,
    )


def _print_report(report: dict[str, Any], max_spikes: int) -> None:
    status = "PASS" if report["valid"] else "SUSPECT"
    print(f"{status}: {report['path']}")
    for error in report["errors"]:
        print(f"  error: {error}")
    spikes = report["spikes"]
    if spikes:
        print(
            f"  isolated scattering spikes: {len(spikes)} "
            f"(ratio >= {report['spike_ratio']:g})"
        )
        for spike in spikes[:max_spikes]:
            print(
                "  "
                f"E={spike['energy_eV']:.9g} eV, Rx={spike['rx_nm']:.9g} nm, "
                f"component={spike['component']}, ratio={spike['ratio']:.6g}, "
                f"|Gscat|={spike['magnitude']:.6g}"
            )


def main() -> int:
    """Run Green-tensor HDF5 validation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="HDF5 files to validate.")
    parser.add_argument(
        "--spike-ratio",
        type=float,
        default=10.0,
        help="Flag isolated scattering values this many times larger than neighbors.",
    )
    parser.add_argument(
        "--max-spikes",
        type=int,
        default=20,
        help="Maximum spike records printed per file.",
    )
    args = parser.parse_args()
    if not np.isfinite(args.spike_ratio) or args.spike_ratio <= 1.0:
        parser.error("--spike-ratio must be finite and greater than 1.")
    if args.max_spikes < 0:
        parser.error("--max-spikes must be non-negative.")

    reports = [
        validate_separation_gf_h5(path, spike_ratio=args.spike_ratio) for path in args.paths
    ]
    for report in reports:
        _print_report(report, args.max_spikes)
    return 0 if all(report["valid"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
