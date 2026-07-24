"""Reusable emitter geometry generators."""

from typing import Mapping, Tuple

import numpy as np


SUPPORTED_RING_ORIENTATIONS = ("orthoradial", "radial")


def normalize_orientation_vectors(
    vectors: np.ndarray,
    expected_count: int,
    allow_single_vector: bool = False,
) -> np.ndarray:
    """Validate and normalize orientation vectors without norm overflow.

    Each row is first scaled by its largest absolute component before its
    Euclidean norm is evaluated. This keeps very large or very small finite
    vectors numerically stable while preserving their direction.

    Args:
        vectors: Orientation vector or array of vectors.
        expected_count: Required number of output rows.
        allow_single_vector: If true, a single shape-``(3,)`` vector is repeated
            for every emitter.

    Returns:
        Finite unit vectors with shape ``(expected_count, 3)``.

    Raises:
        ValueError: If the shape is invalid or any vector is non-finite or zero.
    """

    orientations = np.asarray(vectors, dtype=float)
    if allow_single_vector and orientations.shape == (3,):
        orientations = np.tile(orientations, (expected_count, 1))
    if orientations.shape != (expected_count, 3):
        suffix = " or (3,)" if allow_single_vector else ""
        raise ValueError(
            f"Emitter orientations must have shape ({expected_count}, 3){suffix}; "
            f"got {orientations.shape}."
        )
    if not np.all(np.isfinite(orientations)):
        raise ValueError("Emitter orientations must be finite.")

    row_scales = np.max(np.abs(orientations), axis=1)
    if np.any(row_scales == 0.0):
        raise ValueError("Emitter orientations must be nonzero vectors.")
    scaled = orientations / row_scales[:, np.newaxis]
    norms = np.linalg.norm(scaled, axis=1)
    normalized = scaled / norms[:, np.newaxis]
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Emitter orientations could not be normalized to finite vectors.")
    return normalized


def equatorial_ring_positions_orientations_nm(
    emitter_count: int,
    emitter_radius_nm: float,
    z_nm: float = 0.0,
    phase_offset_deg: float = 0.0,
    orientation: str = "orthoradial",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate positions and dipole orientations for an equatorial ring.

    Emitters are placed in the xy plane with angles
    ``phase_offset_deg + 360*j/emitter_count`` for ``j = 0..N-1``. The endpoint
    is excluded, so arbitrary ``N`` produces evenly spaced points without a
    duplicate at 360 degrees.

    Args:
        emitter_count: Number of emitters in the ring. Must be a positive integer.
        emitter_radius_nm: Ring radius in nanometers. Must be finite and positive.
        z_nm: Shared z coordinate in nanometers. Must be finite.
        phase_offset_deg: Angular phase offset in degrees. Must be finite.
        orientation: Orientation mode. ``"orthoradial"`` gives local azimuthal
            unit vectors ``[-sin(phi), cos(phi), 0]``. ``"radial"`` gives
            outward radial unit vectors ``[cos(phi), sin(phi), 0]``.

    Returns:
        Tuple ``(positions_nm, orientations)`` where both arrays have shape
        ``(emitter_count, 3)`` and dtype ``float``.

    Raises:
        ValueError: If any parameter is invalid or the orientation mode is unsupported.
    """

    count = _validate_emitter_count(emitter_count)
    radius_nm = _validate_finite_float(emitter_radius_nm, "emitter_radius_nm")
    if radius_nm <= 0.0:
        raise ValueError("emitter_radius_nm must be positive.")
    z_value_nm = _validate_finite_float(z_nm, "z_nm")
    phase_deg = _validate_finite_float(phase_offset_deg, "phase_offset_deg")
    orientation_mode = str(orientation).strip().lower()
    if orientation_mode not in SUPPORTED_RING_ORIENTATIONS:
        supported = ", ".join(SUPPORTED_RING_ORIENTATIONS)
        raise ValueError(f"Unsupported ring orientation {orientation!r}; expected one of {supported}.")

    angles_rad = np.deg2rad(phase_deg + 360.0 * np.arange(count, dtype=float) / count)
    cos_phi = np.cos(angles_rad)
    sin_phi = np.sin(angles_rad)
    positions_nm = np.column_stack(
        [radius_nm * cos_phi, radius_nm * sin_phi, np.full(count, z_value_nm)]
    )
    if orientation_mode == "orthoradial":
        orientations = np.column_stack([-sin_phi, cos_phi, np.zeros(count)])
    else:
        orientations = np.column_stack([cos_phi, sin_phi, np.zeros(count)])
    return positions_nm.astype(float), orientations.astype(float)


def equatorial_ring_nearest_neighbor_chord_nm(
    emitter_count: int,
    emitter_radius_nm: float,
) -> float:
    """Return the nearest-neighbor chord distance for an equatorial ring.

    Args:
        emitter_count: Number of emitters in the ring. Must be a positive integer.
        emitter_radius_nm: Ring radius in nanometers. Must be finite and positive.

    Returns:
        Nearest-neighbor chord distance in nanometers. For one emitter, returns
        ``0.0`` because there is no distinct neighbor.
    """

    count = _validate_emitter_count(emitter_count)
    radius_nm = _validate_finite_float(emitter_radius_nm, "emitter_radius_nm")
    if radius_nm <= 0.0:
        raise ValueError("emitter_radius_nm must be positive.")
    if count == 1:
        return 0.0
    return float(2.0 * radius_nm * np.sin(np.pi / count))


def resolve_equatorial_ring_radius_nm(ring_config: Mapping[str, object]) -> float:
    """Resolve an equatorial-ring radius from explicit or sphere-gap settings.

    ``emitter_radius_nm`` may be supplied directly. Alternatively,
    ``sphere_radius_nm`` plus ``emitter_surface_gap_nm`` derives the radius as
    ``sphere_radius_nm + emitter_surface_gap_nm``. If both forms are supplied,
    they must agree. A positive exterior gap is required whenever sphere
    geometry is used because emitters exactly on the boundary are delicate for
    strict-region Mie calculations.

    Args:
        ring_config: Mapping with ``emitter_radius_nm`` or both
            ``sphere_radius_nm`` and ``emitter_surface_gap_nm``.

    Returns:
        Ring radius in nanometers.

    Raises:
        ValueError: If radius settings are missing, non-finite, non-positive, or
            inconsistent.
    """

    explicit_radius = ring_config.get("emitter_radius_nm")
    sphere_radius = ring_config.get("sphere_radius_nm")
    gap = ring_config.get("emitter_surface_gap_nm")

    resolved_radius = None
    if explicit_radius is not None:
        resolved_radius = _validate_finite_float(explicit_radius, "emitter_radius_nm")
        if resolved_radius <= 0.0:
            raise ValueError("emitter_radius_nm must be positive.")

    derived_radius = None
    if sphere_radius is not None or gap is not None:
        if sphere_radius is None or gap is None:
            raise ValueError(
                "Ring radius derived from sphere geometry requires both "
                "sphere_radius_nm and emitter_surface_gap_nm."
            )
        sphere_radius_nm = _validate_finite_float(sphere_radius, "sphere_radius_nm")
        gap_nm = _validate_finite_float(gap, "emitter_surface_gap_nm")
        if sphere_radius_nm <= 0.0:
            raise ValueError("sphere_radius_nm must be positive.")
        if gap_nm <= 0.0:
            raise ValueError("emitter_surface_gap_nm must be positive for exterior sphere rings.")
        derived_radius = sphere_radius_nm + gap_nm
        if resolved_radius is not None and not np.isclose(
            resolved_radius,
            derived_radius,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "emitter_radius_nm is inconsistent with sphere_radius_nm + "
                "emitter_surface_gap_nm."
            )
        if resolved_radius is not None and resolved_radius <= sphere_radius_nm:
            raise ValueError("emitter_radius_nm must be outside the sphere radius.")

    if resolved_radius is None:
        if derived_radius is None:
            raise ValueError(
                "Define emitter_radius_nm or both sphere_radius_nm and emitter_surface_gap_nm."
            )
        resolved_radius = derived_radius
    return float(resolved_radius)


def generate_equatorial_ring_from_config(
    ring_config: Mapping[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate equatorial-ring positions and orientations from a config mapping.

    Args:
        ring_config: Mapping containing ``emitter_count``, radius settings, and
            optional ``z_nm``, ``phase_offset_deg``, and ``orientation`` keys.

    Returns:
        Tuple ``(positions_nm, orientations)`` with shape ``(N, 3)``.
    """

    if "emitter_count" not in ring_config:
        raise ValueError("emitter_ring requires emitter_count.")
    radius_nm = resolve_equatorial_ring_radius_nm(ring_config)
    return equatorial_ring_positions_orientations_nm(
        emitter_count=ring_config["emitter_count"],
        emitter_radius_nm=radius_nm,
        z_nm=float(ring_config.get("z_nm", 0.0)),
        phase_offset_deg=float(ring_config.get("phase_offset_deg", 0.0)),
        orientation=str(ring_config.get("orientation", "orthoradial")),
    )


def _validate_emitter_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("emitter_count must be a positive integer.")
    count = int(value)
    if count <= 0:
        raise ValueError("emitter_count must be a positive integer.")
    return count


def _validate_finite_float(value: object, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number
