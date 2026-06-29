"""Generalized Mie dyadic Green's function for concentric spherical dielectrics.

This module is the spherical analogue of ``GF_Sommerfeld.py`` in the MQED-QD
project.  Instead of evaluating Sommerfeld integrals for a planar stack, it
expands the electromagnetic Green tensor in normalized vector spherical
functions and uses Mie coefficients to enforce boundary conditions at one or two
spherical interfaces.

Implemented geometries
----------------------
``sphere``
    Region 0 is the exterior background and region 1 is a homogeneous sphere.
    The source dipole is assumed to be in region 0, matching the MATLAB
    generalized-Mie implementation.

``coreshell``
    Region 0 is exterior, region 1 is shell, region 2 is core.  The exterior
    scattered Green tensor is implemented for exterior sources.  For finite
    spherical-shell cavities, source and observer points in the core use inner
    reflection coefficients from both shell interfaces.  Shell observations and
    cross-region transmission are not implemented.

``simplecavity``
    Region 0 is the exterior medium and region 1 is a spherical cavity.  The
    source dipole is assumed to be in the cavity.

Coordinate convention
---------------------
The sphere/cavity is centered at the origin.  Public Green-tensor methods accept
Cartesian source and observer positions in meters and return 3x3 Cartesian
Green tensors.  Internally, the Mie expansion is evaluated in the spherical
basis at the observer and then rotated back to Cartesian coordinates.

The code follows the notation of the uploaded MATLAB implementation:

* ``alpha`` and ``beta`` are exterior TM/TE reflection coefficients.
* ``delta`` and ``gamma`` are interior TM/TE transmission coefficients.
* ``p``/``q`` and ``r``/``s`` are source expansion coefficients for exterior and
  cavity sources respectively.

The returned tensor has the same normalization as the existing Sommerfeld code:
``E = k^2 G p / eps0`` in SI conventions.  For convenience, ``field_for_dipole``
returns ``G @ dipole_direction`` or ``G @ dipole_moment`` depending on what you
pass in.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional, Sequence
import warnings

import numpy as np
from scipy.special import spherical_jn, spherical_yn

try:  # project constants when used inside mqed
    from mqed.utils.SI_unit import c, hbar, eV_to_J  # type: ignore
except Exception:  # standalone fallback
    c = 2.99792458e8
    hbar = 1.054571817e-34
    eV_to_J = 1.602176634e-19

GeometryName = Literal["sphere", "coreshell", "simplecavity"]
OrderName = Literal["normal", "reversed"]
RadialKind = Literal["bessel", "hankel1"]
SourceKind = Literal["green", "dipole"]


@dataclass(frozen=True)
class RadialFunctions:
    """Radial Bessel/Hankel data for all multipoles n=1..nmax.

    Attributes:
        z: Spherical Bessel/Hankel function ``j_n(z)`` or ``h_n^(1)(z)``.
        riccati: Riccati-Bessel/Hankel function ``z*j_n(z)`` or ``z*h_n^(1)(z)``.
        d_riccati: Derivative of ``riccati`` with respect to its argument.
        d_riccati_over_z: ``d_riccati / z``.  The small-argument Bessel limit is
            handled explicitly for n=1.
        kind: ``"bessel"`` for regular waves or ``"hankel1"`` for outgoing waves.
    """

    z: np.ndarray
    riccati: np.ndarray
    d_riccati: np.ndarray
    d_riccati_over_z: np.ndarray
    kind: RadialKind


@dataclass(frozen=True)
class AngularFunctions:
    """Normalized angular functions used by the vector spherical functions."""

    ntau: np.ndarray
    npi: np.ndarray
    np_func: np.ndarray
    m: np.ndarray
    mask: np.ndarray
    order: OrderName


@dataclass(frozen=True)
class VectorSphericalFunctions:
    """Vector spherical functions in the local spherical basis.

    Arrays have shape ``(nmax, 2*nmax + 1, 3)``.  The last index is the spherical
    component ``(r, theta, phi)``.
    """

    M: np.ndarray
    N: np.ndarray


@dataclass(frozen=True)
class MieCoefficients:
    """Mie coefficients for a spherical boundary problem."""

    alpha: np.ndarray
    beta: np.ndarray
    gamma: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SourceCoefficients:
    """Expansion coefficients of an electric point-dipole source."""

    p: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    s: Optional[np.ndarray] = None


@dataclass(frozen=True)
class MieResult:
    """Container returned by high-level field helpers."""

    total: np.ndarray
    vacuum: np.ndarray
    structure: np.ndarray
    observer_region: int
    source_region: int


# -----------------------------------------------------------------------------
#  Coordinate utilities
# -----------------------------------------------------------------------------


def cartesian_to_spherical(position: Sequence[float]) -> np.ndarray:
    """Convert a Cartesian position to ``(r, theta, phi)``.

    ``theta`` is the polar angle measured from +z and ``phi`` is the azimuthal
    angle returned by ``atan2(y, x)``.  At the origin, both angles are set to 0.
    """

    p = np.asarray(position, dtype=float).reshape(3)
    x, y, z = p
    r = float(np.linalg.norm(p))
    if r == 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    theta = float(np.arccos(np.clip(z / r, -1.0, 1.0)))
    phi = float(np.arctan2(y, x))
    return np.array([r, theta, phi], dtype=float)


def spherical_basis(theta: float, phi: float) -> np.ndarray:
    """Return a matrix whose columns are ``e_r``, ``e_theta``, and ``e_phi``.

    Multiplying this matrix by spherical components returns Cartesian
    components.  Its transpose converts Cartesian vector components to the local
    spherical basis.
    """

    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    e_r = np.array([st * cp, st * sp, ct], dtype=float)
    e_theta = np.array([ct * cp, ct * sp, -st], dtype=float)
    e_phi = np.array([-sp, cp, 0.0], dtype=float)
    return np.column_stack([e_r, e_theta, e_phi])


def vector_cartesian_to_spherical(vector: Sequence[complex], theta: float, phi: float) -> np.ndarray:
    """Convert vector components from Cartesian to the local spherical basis."""

    return spherical_basis(theta, phi).T @ np.asarray(vector, dtype=complex).reshape(3)


def vector_spherical_to_cartesian(vector: Sequence[complex], theta: float, phi: float) -> np.ndarray:
    """Convert vector components from the local spherical basis to Cartesian."""

    return spherical_basis(theta, phi) @ np.asarray(vector, dtype=complex).reshape(3)


def _normalize_vector(vector: Sequence[float | complex], *, name: str = "vector") -> np.ndarray:
    v = np.asarray(vector, dtype=complex).reshape(3)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError(f"{name} must be non-zero.")
    return v / norm


# -----------------------------------------------------------------------------
#  Special functions: Wigner d, normalized angular functions, radial functions
# -----------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _wigner_d_cached(j: int, theta_rounded: float) -> np.ndarray:
    """Small Wigner-d matrix using the same diagonalization idea as MATLAB.

    The rows and columns are ordered by ``m=-j,...,+j``.  ``theta_rounded`` is a
    rounded float only to make caching useful and deterministic.
    """

    theta = float(theta_rounded)
    m = np.arange(-j, j, dtype=float)
    # J_+ in the |j,m_z> basis.  MATLAB uses diag(...,-1).
    J_plus = np.diag(np.sqrt((j - m) * (j + m + 1.0)), k=-1).astype(complex)
    Jy = (J_plus - J_plus.conj().T) / (2j)
    eigvals, eigvecs = np.linalg.eigh(Jy)
    d = eigvecs @ np.diag(np.exp(-1j * theta * eigvals)) @ eigvecs.conj().T
    if np.max(np.abs(np.imag(d))) > 1e-10:
        warnings.warn("Wigner_d has non-negligible imaginary residuals.", RuntimeWarning)
    return np.real(d)


def wigner_d(j: int, theta: float) -> np.ndarray:
    """Return the Wigner small-d matrix ``d^j(theta)``.

    A light rounding is used only for cache keys; the error is well below the
    numerical tolerance of the Mie sums.
    """

    return _wigner_d_cached(int(j), round(float(theta), 15)).copy()


def m_table(nmax: int, order: OrderName) -> tuple[np.ndarray, np.ndarray]:
    """Build the padded m-table used by the MATLAB code.

    ``normal`` rows are ``m=-n,...,+n``.  ``reversed`` rows are ``m=+n,...,-n``.
    Invalid padded entries are masked out and assigned ``m=0``.
    """

    width = 2 * nmax + 1
    m = np.zeros((nmax, width), dtype=int)
    mask = np.zeros((nmax, width), dtype=bool)
    for row, n in enumerate(range(1, nmax + 1)):
        values = np.arange(-n, n + 1, dtype=int)
        if order == "reversed":
            values = values[::-1]
        elif order != "normal":
            raise ValueError("order must be 'normal' or 'reversed'.")
        m[row, : values.size] = values
        mask[row, : values.size] = True
    return m, mask


def normalized_tau_pi_p(nmax: int, theta: float, order: OrderName = "normal") -> AngularFunctions:
    """Compute normalized ``tau_nm``, ``pi_nm``, and ``P_nm`` tables.

    This follows the uploaded MATLAB function ``NormTauPiP``: it obtains the
    angular functions from Wigner-d matrix elements instead of directly
    differentiating associated Legendre polynomials, which is more stable at
    high multipole order.
    """

    nmax = int(nmax)
    width = 2 * nmax + 1
    ntau = np.zeros((nmax, width), dtype=float)
    npi = np.zeros((nmax, width), dtype=float)
    np_func = np.zeros((nmax, width), dtype=float)
    m, mask = m_table(nmax, order)

    for row, n in enumerate(range(1, nmax + 1)):
        dn = wigner_d(n, theta)
        # columns correspond to m=-n,...,+n; hence -1,0,+1 are n-1,n,n+1.
        d_m_plus_1 = dn[:, n + 1]
        d_m_0 = dn[:, n]
        d_m_minus_1 = dn[:, n - 1]
        if order == "reversed":
            d_m_plus_1 = d_m_plus_1[::-1]
            d_m_0 = d_m_0[::-1]
            d_m_minus_1 = d_m_minus_1[::-1]

        norm_tau_pi = np.sqrt((2 * n + 1) / 8.0)
        norm_p = np.sqrt((2 * n + 1) / (2.0 * n * (n + 1)))
        cols = slice(0, 2 * n + 1)
        npi[row, cols] = -norm_tau_pi * (d_m_plus_1 + d_m_minus_1)
        ntau[row, cols] = -norm_tau_pi * (d_m_plus_1 - d_m_minus_1)
        np_func[row, cols] = norm_p * d_m_0

    npi[np.abs(npi) < 1e-15] = 0.0
    ntau[np.abs(ntau) < 1e-15] = 0.0
    np_func[np.abs(np_func) < 1e-15] = 0.0
    return AngularFunctions(ntau=ntau, npi=npi, np_func=np_func, m=m, mask=mask, order=order)


def azimuthal_table(nmax: int, phi: float, order: OrderName) -> np.ndarray:
    """Return the normalized azimuthal factors ``(2π)^-1/2 exp(i m phi)``."""

    m, mask = m_table(nmax, order)
    out = np.sqrt(1.0 / (2.0 * np.pi)) * np.exp(1j * m * float(phi))
    out[~mask] = 0.0
    return out


def _riccati_from_spherical(z_arg: complex, n: np.ndarray, kind: RadialKind) -> RadialFunctions:
    """Radial functions built from SciPy's spherical Bessel functions."""

    z_arg = complex(z_arg)
    if abs(z_arg) < 1e-60:
        values = np.zeros(n.size, dtype=complex)
        riccati = np.zeros(n.size, dtype=complex)
        d_riccati = np.zeros(n.size, dtype=complex)
        d_over_z = np.zeros(n.size, dtype=complex)
        if kind == "bessel" and n.size:
            # j_1(z)/z -> 1/3 and psi_1'(z)/z -> 2/3.
            d_over_z[0] = 2.0 / 3.0
        elif kind == "hankel1":
            # The true limit is singular; keep a large finite value to make the
            # failure mode explicit but avoid NaNs from immediate division.
            values[:] = -1j * 1e300
            riccati[:] = -1j * 1e300
            d_riccati[:] = 1j * 1e300
        return RadialFunctions(values, riccati, d_riccati, d_over_z, kind)

    j = spherical_jn(n, z_arg)
    dj = spherical_jn(n, z_arg, derivative=True)
    if kind == "bessel":
        values = j.astype(complex)
        derivative = dj.astype(complex)
    elif kind == "hankel1":
        y = spherical_yn(n, z_arg)
        dy = spherical_yn(n, z_arg, derivative=True)
        values = j + 1j * y
        derivative = dj + 1j * dy
    else:
        raise ValueError("kind must be 'bessel' or 'hankel1'.")

    riccati = z_arg * values
    d_riccati = values + z_arg * derivative
    return RadialFunctions(
        z=values,
        riccati=riccati,
        d_riccati=d_riccati,
        d_riccati_over_z=d_riccati / z_arg,
        kind=kind,
    )


def spherical_radial_functions(z_arg: complex, nmax: int, kind: RadialKind) -> RadialFunctions:
    """Return radial functions for all orders n=1..nmax."""

    n = np.arange(1, int(nmax) + 1, dtype=int)
    return _riccati_from_spherical(z_arg, n, kind)


def vector_spherical_functions(
    kr: complex,
    nmax: int,
    radial: RadialFunctions,
    angular: AngularFunctions,
    emphi: np.ndarray,
) -> VectorSphericalFunctions:
    """Generate normalized vector spherical functions ``M_nm`` and ``N_nm``.

    The implementation mirrors MATLAB ``VectSphFunc``.  The output components
    are in the local spherical basis at the evaluation point.
    """

    width = 2 * nmax + 1
    M = np.zeros((nmax, width, 3), dtype=complex)
    N = np.zeros((nmax, width, 3), dtype=complex)
    n = np.arange(1, nmax + 1, dtype=float)[:, None]
    z = radial.z[:, None]
    d_over_z = radial.d_riccati_over_z[:, None]

    if abs(complex(kr)) < 1e-60 and radial.kind == "bessel":
        z_over_kr = np.zeros((nmax, 1), dtype=complex)
        z_over_kr[0, 0] = 1.0 / 3.0
    else:
        z_over_kr = z / complex(kr)

    M[:, :, 1] = 1j * z * angular.npi * emphi
    M[:, :, 2] = -z * angular.ntau * emphi

    N[:, :, 0] = z_over_kr * n * (n + 1.0) * angular.np_func * emphi
    N[:, :, 1] = d_over_z * angular.ntau * emphi
    N[:, :, 2] = 1j * d_over_z * angular.npi * emphi

    # Invalid padded m slots should remain zero even if broadcasting introduced
    # tiny numerical values.
    M[~angular.mask, :] = 0.0
    N[~angular.mask, :] = 0.0
    return VectorSphericalFunctions(M=M, N=N)


def _contract_vsf_with_vector(vsf_component: np.ndarray, vector_sph: np.ndarray) -> np.ndarray:
    """Contract a ``(..., 3)`` VSF array with a spherical-basis vector."""

    return np.einsum("nmj,j->nm", vsf_component, vector_sph)


# -----------------------------------------------------------------------------
#  Mie coefficients
# -----------------------------------------------------------------------------


def _safe_equal(a: complex, b: complex, rtol: float = 1e-12) -> bool:
    return bool(np.isclose(a, b, rtol=rtol, atol=rtol))


def dlog_riccati(z: complex, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Logarithmic derivatives ``D1=psi'/psi`` and ``D3=xi'/xi``.

    This is a direct Python translation of the recurrence used in the uploaded
    MATLAB ``Dlog`` function.  It is used for the core/shell and simple-cavity
    formulas, where direct ratios can be poorly conditioned.
    """

    z = complex(z)
    nmax = int(nmax)
    if abs(z) < 1e-60:
        # Fall back to direct SciPy ratios with tiny offset; zero arguments are
        # singular for the logarithmic derivative anyway.
        z = 1e-60 + 0j
    nex = nmax + int(np.floor(abs(1.0478 * z + 18.692)))
    nex = max(nex, nmax + 16)
    D1 = np.zeros(nex, dtype=complex)
    D3 = np.zeros(nmax, dtype=complex)
    for nn in range(nex, 1, -1):
        # MATLAB index D1(nn-1), D1(nn) -> Python D1[nn-2], D1[nn-1]
        D1[nn - 2] = nn / z - 1.0 / (nn / z + D1[nn - 1])

    # First-order logarithmic derivatives.  These are the formulas used by the
    # updated MATLAB file rather than the older PDF appendix.
    D1[0] = (z**2 * np.tan(z) + z - np.tan(z)) / (-z**2 + z * np.tan(z))
    D3[0] = (1j * z**2 - z - 1j) / (z**2 + 1j * z)
    for nn in range(2, nmax + 1):
        D3[nn - 1] = -nn / z + 1.0 / (nn / z - D3[nn - 2])
    return D1[:nmax], D3


def mie_single(refractive_indices: Sequence[complex], k0_radius: float, nmax: int) -> MieCoefficients:
    """Mie coefficients for one homogeneous sphere."""

    nr = np.asarray(refractive_indices, dtype=complex).reshape(-1)
    if nr.size != 2:
        raise ValueError("mie_single expects refractive_indices=[n_exterior, n_sphere].")
    n0, n1 = nr
    n0kr1 = n0 * k0_radius
    n1kr1 = n1 * k0_radius

    n0_bessel = spherical_radial_functions(n0kr1, nmax, "bessel")
    n0_hankel = spherical_radial_functions(n0kr1, nmax, "hankel1")
    n1_bessel = spherical_radial_functions(n1kr1, nmax, "bessel")

    n0psi, n0dpsi = n0_bessel.riccati, n0_bessel.d_riccati
    n0xi, n0dxi = n0_hankel.riccati, n0_hankel.d_riccati
    n1psi, n1dpsi = n1_bessel.riccati, n1_bessel.d_riccati

    alpha = -(
        n1 * n0dpsi * n1psi - n0 * n0psi * n1dpsi
    ) / (
        n1 * n0dxi * n1psi - n0 * n0xi * n1dpsi
    )
    beta = -(
        n0 * n0dpsi * n1psi - n1 * n0psi * n1dpsi
    ) / (
        n0 * n0dxi * n1psi - n1 * n0xi * n1dpsi
    )
    gamma = n1 * (n0dpsi * n0xi - n0psi * n0dxi) / (
        n1 * n1dpsi * n0xi - n0 * n1psi * n0dxi
    )
    delta = n1 * (n0dpsi * n0xi - n0psi * n0dxi) / (
        n0 * n1dpsi * n0xi - n1 * n1psi * n0dxi
    )
    return MieCoefficients(alpha=alpha, beta=beta, gamma=gamma, delta=delta)


def mie_simple_cavity(refractive_indices: Sequence[complex], k0_radius: float, nmax: int) -> MieCoefficients:
    """Mie coefficients for a spherical cavity embedded in an exterior medium."""

    nr = np.asarray(refractive_indices, dtype=complex).reshape(-1)
    if nr.size != 2:
        raise ValueError("mie_simple_cavity expects refractive_indices=[n_exterior, n_cavity].")
    n0, n1 = nr
    n0kr1 = n0 * k0_radius
    n1kr1 = n1 * k0_radius

    n0_hankel = spherical_radial_functions(n0kr1, nmax, "hankel1")
    n1_bessel = spherical_radial_functions(n1kr1, nmax, "bessel")
    n1_hankel = spherical_radial_functions(n1kr1, nmax, "hankel1")

    n0xi = n0_hankel.riccati
    n1psi = n1_bessel.riccati
    n1xi = n1_hankel.riccati
    n1D1, n1D3 = dlog_riccati(n1kr1, nmax)
    _, n0D3 = dlog_riccati(n0kr1, nmax)

    alpha = (n0 * n1D3 - n0 * n1D1) / (n1 * n0D3 - n0 * n1D1) * n1xi / n0xi
    beta = (n0 * n1D3 - n0 * n1D1) / (n0 * n0D3 - n1 * n1D1) * n1xi / n0xi
    gamma = -(
        n1 * n1D3 - n0 * n0D3
    ) / (
        n1 * n1D1 - n0 * n0D3
    ) * n1xi / n1psi
    delta = -(
        n0 * n1D3 - n1 * n0D3
    ) / (
        n0 * n1D1 - n1 * n0D3
    ) * n1xi / n1psi
    return MieCoefficients(alpha=alpha, beta=beta, gamma=gamma, delta=delta)


def mie_coreshell(
    refractive_indices: Sequence[complex], k0_radii: Sequence[float], nmax: int
) -> MieCoefficients:
    """Mie coefficients for a core/shell sphere.

    Args:
        refractive_indices: ``[n_exterior, n_shell, n_core]``.
        k0_radii: ``[k0 * outer_radius, k0 * core_radius]``.

    ``alpha`` and ``beta`` are exterior-source scattering coefficients. ``gamma``
    and ``delta`` are finite-shell inner-cavity reflection coefficients for a
    source and observer both in the core.
    """

    nr = np.asarray(refractive_indices, dtype=complex).reshape(-1)
    ks = np.asarray(k0_radii, dtype=float).reshape(-1)
    if nr.size != 3 or ks.size != 2:
        raise ValueError("mie_coreshell expects nr=[n0,n1,n2] and k0_radii=[k0*r_outer,k0*r_core].")
    n0, n1, n2 = nr
    if _safe_equal(n0, n1):
        n1 = n1 + 1e-7

    n0kr1 = n0 * ks[0]
    n1kr1 = n1 * ks[0]
    n1kr2 = n1 * ks[1]
    n2kr2 = n2 * ks[1]

    n0kr1psi = spherical_radial_functions(n0kr1, nmax, "bessel").riccati
    n0kr1xi = spherical_radial_functions(n0kr1, nmax, "hankel1").riccati
    n1kr1xi = spherical_radial_functions(n1kr1, nmax, "hankel1").riccati
    n1kr2xi = spherical_radial_functions(n1kr2, nmax, "hankel1").riccati
    n1kr1psi = spherical_radial_functions(n1kr1, nmax, "bessel").riccati
    n1kr2psi = spherical_radial_functions(n1kr2, nmax, "bessel").riccati

    n1kr2D1, n1kr2D3 = dlog_riccati(n1kr2, nmax)
    n1kr1D1, n1kr1D3 = dlog_riccati(n1kr1, nmax)
    n2kr2D1, _ = dlog_riccati(n2kr2, nmax)
    n0kr1D1, n0kr1D3 = dlog_riccati(n0kr1, nmax)

    f1 = n1kr2xi / n1kr2psi
    f2 = n1kr1xi / n1kr1psi
    f3 = n0kr1psi / n0kr1xi

    A = (n2 * n1kr2D3 - n1 * n2kr2D1) / (n1 * n2kr2D1 - n2 * n1kr2D1) * f1
    B = (n2 * n2kr2D1 - n1 * n1kr2D3) / (n1 * n1kr2D1 - n2 * n2kr2D1) * f1
    A1 = (n1 * n0kr1D1 - n0 * n1kr1D3) / (n0 * n1kr1D1 - n1 * n0kr1D1) * f2
    A2 = (n0 * n1kr1D3 - n1 * n0kr1D3) / (n1 * n0kr1D3 - n0 * n1kr1D1) * f2
    B1 = (n1 * n1kr1D3 - n0 * n0kr1D1) / (n0 * n0kr1D1 - n1 * n1kr1D1) * f2
    B2 = (n0 * n0kr1D3 - n1 * n1kr1D3) / (n1 * n1kr1D1 - n0 * n0kr1D3) * f2

    alpha = (A1 - A) / (A2 - A) * f3 * (
        n0 * n1kr1D1 - n1 * n0kr1D1
    ) / (
        n1 * n0kr1D3 - n0 * n1kr1D1
    )
    beta = (B1 - B) / (B2 - B) * f3 * (
        n0 * n0kr1D1 - n1 * n1kr1D1
    ) / (
        n1 * n1kr1D1 - n0 * n0kr1D3
    )
    gamma, delta = mie_coreshell_core_reflection(nr, ks, nmax)
    return MieCoefficients(alpha=alpha, beta=beta, gamma=gamma, delta=delta)


def mie_coreshell_core_reflection(
    refractive_indices: Sequence[complex], k0_radii: Sequence[float], nmax: int
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-cavity reflection coefficients for a finite core/shell sphere.

    The source-generated outgoing wave in the core has unit amplitude.  The
    unknown reflected regular wave in the core is solved together with regular
    and outgoing shell waves and an outgoing exterior wave.  This enforces
    tangential-field continuity at the core/shell and shell/exterior interfaces.

    Returns:
        ``(gamma, delta)`` where ``gamma`` multiplies TE/M source coefficients
        and ``delta`` multiplies TM/N source coefficients.
    """

    nr = np.asarray(refractive_indices, dtype=complex).reshape(-1)
    ks = np.asarray(k0_radii, dtype=float).reshape(-1)
    if nr.size != 3 or ks.size != 2:
        raise ValueError(
            "mie_coreshell_core_reflection expects nr=[n0,n1,n2] and "
            "k0_radii=[k0*r_outer,k0*r_core]."
        )
    n0, n1, n2 = nr
    outer_kr, core_kr = ks

    n0_outer_hankel = spherical_radial_functions(n0 * outer_kr, nmax, "hankel1")
    n1_outer_bessel = spherical_radial_functions(n1 * outer_kr, nmax, "bessel")
    n1_outer_hankel = spherical_radial_functions(n1 * outer_kr, nmax, "hankel1")
    n1_core_bessel = spherical_radial_functions(n1 * core_kr, nmax, "bessel")
    n1_core_hankel = spherical_radial_functions(n1 * core_kr, nmax, "hankel1")
    n2_core_bessel = spherical_radial_functions(n2 * core_kr, nmax, "bessel")
    n2_core_hankel = spherical_radial_functions(n2 * core_kr, nmax, "hankel1")

    gamma = np.zeros(nmax, dtype=complex)
    delta = np.zeros(nmax, dtype=complex)
    for idx in range(nmax):
        psi2a = n2_core_bessel.riccati[idx]
        dpsi2a = n2_core_bessel.d_riccati[idx]
        xi2a = n2_core_hankel.riccati[idx]
        dxi2a = n2_core_hankel.d_riccati[idx]

        psi1a = n1_core_bessel.riccati[idx]
        dpsi1a = n1_core_bessel.d_riccati[idx]
        xi1a = n1_core_hankel.riccati[idx]
        dxi1a = n1_core_hankel.d_riccati[idx]

        psi1b = n1_outer_bessel.riccati[idx]
        dpsi1b = n1_outer_bessel.d_riccati[idx]
        xi1b = n1_outer_hankel.riccati[idx]
        dxi1b = n1_outer_hankel.d_riccati[idx]

        xi0b = n0_outer_hankel.riccati[idx]
        dxi0b = n0_outer_hankel.d_riccati[idx]

        te_matrix = np.array(
            [
                [psi2a, -psi1a, -xi1a, 0.0],
                [n2 * dpsi2a, -n1 * dpsi1a, -n1 * dxi1a, 0.0],
                [0.0, psi1b, xi1b, -xi0b],
                [0.0, n1 * dpsi1b, n1 * dxi1b, -n0 * dxi0b],
            ],
            dtype=complex,
        )
        te_rhs = np.array([-xi2a, -n2 * dxi2a, 0.0, 0.0], dtype=complex)
        gamma[idx] = np.linalg.solve(te_matrix, te_rhs)[0]

        tm_matrix = np.array(
            [
                [psi2a, -psi1a, -xi1a, 0.0],
                [dpsi2a / n2, -dpsi1a / n1, -dxi1a / n1, 0.0],
                [0.0, psi1b, xi1b, -xi0b],
                [0.0, dpsi1b / n1, dxi1b / n1, -dxi0b / n0],
            ],
            dtype=complex,
        )
        tm_rhs = np.array([-xi2a, -dxi2a / n2, 0.0, 0.0], dtype=complex)
        delta[idx] = np.linalg.solve(tm_matrix, tm_rhs)[0]

    return gamma, delta


# -----------------------------------------------------------------------------
#  Green tensor class
# -----------------------------------------------------------------------------


class MieGreenFunction:
    r"""Dyadic Green tensor for spherical Mie geometries.

    Args:
        refractive_indices: Relative refractive indices by region.  Use
            ``[n0, n1]`` for ``sphere`` or ``simplecavity`` and
            ``[n0, n1, n2]`` for ``coreshell``.
        radii_m: Spherical boundary radii in meters.  For ``sphere`` and
            ``simplecavity`` give one radius.  For ``coreshell`` give
            ``[outer_radius, core_radius]``.
        omega: Angular frequency in rad/s.
        nmax: Highest multipole order in the Mie sum.
        geometry: ``"sphere"``, ``"coreshell"``, or ``"simplecavity"``.
        strict_regions: If true, invalid source-region placements raise an
            error.  If false, the calculation proceeds but the result is not
            guaranteed to match the derivation.
    """

    def __init__(
        self,
        refractive_indices: Sequence[complex],
        radii_m: Sequence[float] | float,
        omega: float,
        nmax: int,
        geometry: GeometryName = "sphere",
        strict_regions: bool = True,
    ):
        self.nr = np.asarray(refractive_indices, dtype=complex).reshape(-1)
        self.radii_m = np.atleast_1d(np.asarray(radii_m, dtype=float)).reshape(-1)
        self.omega = float(omega)
        self.k0 = self.omega / c
        self.nmax = int(nmax)
        self.geometry: GeometryName = geometry
        self.strict_regions = bool(strict_regions)
        self._mie_cache: Optional[MieCoefficients] = None
        self._warned_unsupported_coreshell_inside = False
        self._warned_unsupported_coreshell_cross_region = False

        self._validate_initialization()

    def _validate_initialization(self) -> None:
        if self.nmax < 1:
            raise ValueError("nmax must be at least 1.")
        if self.geometry in {"sphere", "simplecavity"}:
            if self.nr.size != 2 or self.radii_m.size != 1:
                raise ValueError(
                    f"{self.geometry} expects 2 refractive indices and one radius."
                )
            if self.radii_m[0] <= 0:
                raise ValueError("Sphere/cavity radius must be positive.")
        elif self.geometry == "coreshell":
            if self.nr.size != 3 or self.radii_m.size != 2:
                raise ValueError("coreshell expects 3 refractive indices and [outer, core] radii.")
            if not (self.radii_m[0] > self.radii_m[1] > 0):
                raise ValueError("coreshell radii must satisfy outer_radius > core_radius > 0.")
        else:
            raise ValueError("geometry must be 'sphere', 'coreshell', or 'simplecavity'.")

    @property
    def k0_radii(self) -> np.ndarray:
        return self.k0 * self.radii_m

    def mie_coefficients(self) -> MieCoefficients:
        """Return and cache the Mie coefficients for the active frequency."""

        if self._mie_cache is not None:
            return self._mie_cache
        if self.geometry == "sphere":
            self._mie_cache = mie_single(self.nr, self.k0_radii[0], self.nmax)
        elif self.geometry == "simplecavity":
            self._mie_cache = mie_simple_cavity(self.nr, self.k0_radii[0], self.nmax)
        elif self.geometry == "coreshell":
            self._mie_cache = mie_coreshell(self.nr, self.k0_radii, self.nmax)
        else:  # pragma: no cover - guarded by initialization
            raise ValueError(self.geometry)
        return self._mie_cache

    # ------------------------------------------------------------------
    # Region logic
    # ------------------------------------------------------------------

    def region_of(self, position_m: Sequence[float]) -> int:
        """Return the concentric-region index of a Cartesian position."""

        r = float(np.linalg.norm(np.asarray(position_m, dtype=float).reshape(3)))
        if self.geometry in {"sphere", "simplecavity"}:
            return 0 if r >= self.radii_m[0] else 1
        if self.geometry == "coreshell":
            if r >= self.radii_m[0]:
                return 0
            if r >= self.radii_m[1]:
                return 1
            return 2
        raise ValueError(self.geometry)

    def expected_source_region(self) -> int:
        """Default source region implemented by the uploaded MATLAB theory/code."""

        return 1 if self.geometry == "simplecavity" else 0

    def _check_source_region(self, source_position_m: Sequence[float]) -> int:
        region = self.region_of(source_position_m)
        if self.geometry == "coreshell":
            if region in {0, 2}:
                return region
            if self.strict_regions:
                raise ValueError(
                    "For coreshell, the source dipole must be outside the shell or inside the core. "
                    "Sources in the shell region are not implemented."
                )
            return region
        expected = self.expected_source_region()
        if region != expected and self.strict_regions:
            if self.geometry == "simplecavity":
                raise ValueError("For simplecavity, the source dipole must be inside the cavity.")
            raise ValueError("For sphere/coreshell, the source dipole must be outside the sphere/shell.")
        return region

    # ------------------------------------------------------------------
    # Source and structure terms
    # ------------------------------------------------------------------

    def source_coefficients(
        self,
        source_position_m: Sequence[float],
        source_orientation_cart: Sequence[complex],
        kind: SourceKind = "green",
    ) -> SourceCoefficients:
        """Expansion coefficients of an electric point dipole.

        ``kind='green'`` gives the coefficient normalization used to assemble
        the Green tensor.  ``kind='dipole'`` gives the MATLAB Gaussian-unit
        electric-field prefactor.
        """

        source_sph = cartesian_to_spherical(source_position_m)
        source_region = self._check_source_region(source_position_m)
        ni = self.nr[source_region]
        kr = ni * self.k0 * source_sph[0]
        is_interior_cavity_source = self.geometry == "simplecavity" or (
            self.geometry == "coreshell" and source_region == 2
        )
        radial_kind: RadialKind = "bessel" if is_interior_cavity_source else "hankel1"
        radial = spherical_radial_functions(kr, self.nmax, radial_kind)
        angular = normalized_tau_pi_p(self.nmax, source_sph[1], "reversed")
        emphi = azimuthal_table(self.nmax, source_sph[2], "reversed")
        vsf = vector_spherical_functions(kr, self.nmax, radial, angular, emphi)
        dipole_sph = vector_cartesian_to_spherical(source_orientation_cart, source_sph[1], source_sph[2])
        sign = np.where((angular.m % 2) == 0, 1.0, -1.0)
        if kind == "green":
            prefactor = 1j * (ni * self.k0) * sign
        elif kind == "dipole":
            prefactor = 4.0 * np.pi * 1j * (ni * self.k0) ** 3 * sign
        else:
            raise ValueError("kind must be 'green' or 'dipole'.")
        prefactor = prefactor * angular.mask
        Nproj = _contract_vsf_with_vector(vsf.N, dipole_sph)
        Mproj = _contract_vsf_with_vector(vsf.M, dipole_sph)
        if is_interior_cavity_source:
            return SourceCoefficients(r=prefactor * Nproj, s=prefactor * Mproj)
        return SourceCoefficients(p=prefactor * Nproj, q=prefactor * Mproj)

    def _warn_unsupported_coreshell_inside(self) -> None:
        if not self._warned_unsupported_coreshell_inside:
            warnings.warn(
                "Core/shell shell-region observation is not implemented; returning zero structure "
                "contribution for region 1.",
                RuntimeWarning,
            )
            self._warned_unsupported_coreshell_inside = True

    def _warn_unsupported_coreshell_cross_region(self) -> None:
        if not self._warned_unsupported_coreshell_cross_region:
            warnings.warn(
                "Core/shell cross-region transmission is not implemented; returning zero structure "
                "contribution for this source/observer region pair.",
                RuntimeWarning,
            )
            self._warned_unsupported_coreshell_cross_region = True

    def _observer_vsf(self, observer_position_m: Sequence[float], region: int) -> VectorSphericalFunctions:
        observer_sph = cartesian_to_spherical(observer_position_m)
        if region == 0:
            radial_kind: RadialKind = "hankel1"
        else:
            radial_kind = "bessel"
        ni = self.nr[region]
        kr = ni * self.k0 * observer_sph[0]
        radial = spherical_radial_functions(kr, self.nmax, radial_kind)
        angular = normalized_tau_pi_p(self.nmax, observer_sph[1], "normal")
        emphi = azimuthal_table(self.nmax, observer_sph[2], "normal")
        return vector_spherical_functions(kr, self.nmax, radial, angular, emphi)

    @staticmethod
    def _sum_mie_field(vsf: VectorSphericalFunctions, coeff_M: np.ndarray, coeff_N: np.ndarray) -> np.ndarray:
        """Sum TE/M and TM/N contributions to one spherical-basis vector."""

        field_M = np.sum(vsf.M * coeff_M[:, :, None], axis=(0, 1))
        field_N = np.sum(vsf.N * coeff_N[:, :, None], axis=(0, 1))
        return field_M + field_N

    def structure_field_for_orientation(
        self,
        observer_position_m: Sequence[float],
        source_position_m: Sequence[float],
        source_orientation_cart: Sequence[complex],
    ) -> np.ndarray:
        """Return the structure-induced ``G_sc @ orientation`` vector in Cartesian components.

        For exterior points this is the scattered field; for points across the
        boundary it is the transmitted Green-tensor contribution.
        """

        observer_sph = cartesian_to_spherical(observer_position_m)
        observer_region = self.region_of(observer_position_m)
        source_region = self._check_source_region(source_position_m)

        if self.geometry == "coreshell":
            if source_region == 1 or observer_region == 1:
                self._warn_unsupported_coreshell_inside()
                return np.zeros(3, dtype=complex)
            if source_region != observer_region:
                self._warn_unsupported_coreshell_cross_region()
                return np.zeros(3, dtype=complex)
            if source_region not in {0, 2}:
                self._warn_unsupported_coreshell_cross_region()
                return np.zeros(3, dtype=complex)

        if self.geometry == "coreshell" and observer_region != 0 and source_region != 2:
            self._warn_unsupported_coreshell_inside()
            return np.zeros(3, dtype=complex)

        coeff = self.mie_coefficients()
        source = self.source_coefficients(source_position_m, source_orientation_cart, "green")
        vsf = self._observer_vsf(observer_position_m, observer_region)

        if observer_region == 0:
            if self.geometry == "simplecavity":
                if source.r is None or source.s is None:
                    raise RuntimeError("Missing simplecavity source coefficients.")
                coeff_N = source.r * coeff.alpha[:, None]
                coeff_M = source.s * coeff.beta[:, None]
            else:
                if source.p is None or source.q is None:
                    raise RuntimeError("Missing exterior source coefficients.")
                coeff_N = source.p * coeff.alpha[:, None]
                coeff_M = source.q * coeff.beta[:, None]
        else:
            if coeff.gamma is None or coeff.delta is None:
                raise RuntimeError("Interior/transmission coefficients are unavailable.")
            if self.geometry == "simplecavity" or (
                self.geometry == "coreshell" and source_region == 2 and observer_region == 2
            ):
                if source.r is None or source.s is None:
                    raise RuntimeError("Missing interior source coefficients.")
                coeff_N = source.r * coeff.delta[:, None]
                coeff_M = source.s * coeff.gamma[:, None]
            else:
                if source.p is None or source.q is None:
                    raise RuntimeError("Missing exterior source coefficients.")
                coeff_N = source.p * coeff.delta[:, None]
                coeff_M = source.q * coeff.gamma[:, None]

        field_sph = self._sum_mie_field(vsf, coeff_M=coeff_M, coeff_N=coeff_N)
        return vector_spherical_to_cartesian(field_sph, observer_sph[1], observer_sph[2])

    def structure_component(
        self,
        observer_position_m: Sequence[float],
        source_position_m: Sequence[float],
    ) -> np.ndarray:
        """Return the 3x3 Cartesian structure Green tensor.

        The tensor is assembled column-by-column by exciting x-, y-, and
        z-oriented unit dipoles.
        """

        Gs = np.zeros((3, 3), dtype=complex)
        for col in range(3):
            u = np.zeros(3, dtype=complex)
            u[col] = 1.0
            Gs[:, col] = self.structure_field_for_orientation(
                observer_position_m, source_position_m, u
            )
        return Gs

    # ------------------------------------------------------------------
    # Vacuum and total Green tensor
    # ------------------------------------------------------------------

    def _direct_medium_index(self, source_region: int, observer_region: int) -> Optional[int]:
        """Return medium index for the direct homogeneous term, or ``None``."""

        if self.geometry in {"sphere", "coreshell"}:
            if source_region == 0 and observer_region == 0:
                return 0
            if self.geometry == "coreshell" and source_region == 2 and observer_region == 2:
                return 2
            return None
        if self.geometry == "simplecavity":
            return 1 if source_region == 1 and observer_region == 1 else None
        return None

    def vacuum_component(
        self,
        observer_position_m: Sequence[float],
        source_position_m: Sequence[float],
        refractive_index: Optional[complex] = None,
    ) -> np.ndarray:
        r"""Homogeneous-space dyadic Green tensor.

        .. math::

            G_0 = \frac{e^{ikR}}{4\pi R k^2}
            \left[k^2(I-\hat R\hat R)+(3\hat R\hat R-I)/R^2
            + i k (I-3\hat R\hat R)/R\right].

        The coincident-point value returns the regular imaginary part
        ``i k/(6π) I``.
        """

        obs = np.asarray(observer_position_m, dtype=float).reshape(3)
        src = np.asarray(source_position_m, dtype=float).reshape(3)
        n = self.nr[0] if refractive_index is None else complex(refractive_index)
        k = n * self.k0
        R_vec = obs - src
        R = float(np.linalg.norm(R_vec))
        if R < 1e-12:
            return 1j * k / (6.0 * np.pi) * np.eye(3, dtype=complex)
        eR = R_vec / R
        I3 = np.eye(3, dtype=complex)
        RR = np.outer(eR, eR)
        term1 = (I3 - RR) * k**2
        term2 = (3.0 * RR - I3) / R**2
        term3 = (I3 - 3.0 * RR) * (1j * k / R)
        return np.exp(1j * k * R) / (4.0 * np.pi * R * k**2) * (term1 + term2 + term3)

    def calculate_components(
        self,
        observer_position_m: Sequence[float],
        source_position_m: Sequence[float],
    ) -> MieResult:
        """Return total, homogeneous, and structure tensors for one point pair."""

        source_region = self._check_source_region(source_position_m)
        observer_region = self.region_of(observer_position_m)
        structure = self.structure_component(observer_position_m, source_position_m)
        medium = self._direct_medium_index(source_region, observer_region)
        if medium is None:
            vacuum = np.zeros((3, 3), dtype=complex)
        else:
            vacuum = self.vacuum_component(
                observer_position_m, source_position_m, refractive_index=self.nr[medium]
            )
        return MieResult(
            total=vacuum + structure,
            vacuum=vacuum,
            structure=structure,
            observer_region=observer_region,
            source_region=source_region,
        )

    def calculate_total_Green_function(
        self,
        observer_position_m: Sequence[float],
        source_position_m: Sequence[float],
    ) -> np.ndarray:
        """Return the total 3x3 Cartesian Green tensor."""

        return self.calculate_components(observer_position_m, source_position_m).total

    def calculate_total_Green_functions_for_points(
        self,
        observer_positions_m: np.ndarray,
        source_position_m: Sequence[float],
    ) -> np.ndarray:
        """Vectorized convenience wrapper over observer positions."""

        positions = np.asarray(observer_positions_m, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, 3)
        return np.array(
            [self.calculate_total_Green_function(pos, source_position_m) for pos in positions],
            dtype=complex,
        )

    def field_for_dipole(
        self,
        observer_position_m: Sequence[float],
        source_position_m: Sequence[float],
        dipole_vector_cart: Sequence[complex],
    ) -> np.ndarray:
        """Return ``G(observer, source) @ dipole_vector``."""

        return self.calculate_total_Green_function(observer_position_m, source_position_m) @ np.asarray(
            dipole_vector_cart, dtype=complex
        ).reshape(3)

    def projected_green(
        self,
        observer_position_m: Sequence[float],
        source_position_m: Sequence[float],
        observer_orientation_cart: Sequence[complex],
        source_orientation_cart: Sequence[complex],
    ) -> complex:
        """Return ``e_A · G(r_A,r_D) · e_D`` for oriented dipoles."""

        eA = _normalize_vector(observer_orientation_cart, name="observer_orientation")
        eD = _normalize_vector(source_orientation_cart, name="source_orientation")
        return complex(eA @ (self.calculate_total_Green_function(observer_position_m, source_position_m) @ eD))

    def projected_im_green_at_source(
        self,
        source_position_m: Sequence[float],
        orientation_cart: Sequence[complex],
    ) -> float:
        """Return ``Im[e · G(r,r) · e]`` with the homogeneous regular term included."""

        e = _normalize_vector(orientation_cart, name="orientation")
        result = self.calculate_components(source_position_m, source_position_m)
        return float(np.imag(e @ (result.total @ e)))

    def purcell_factor(
        self,
        source_position_m: Sequence[float],
        orientation_cart: Sequence[complex],
        reference_k: Optional[float] = None,
    ) -> float:
        """Return the orientation-resolved Purcell factor.

        The default reference uses ``k0`` to match the uploaded MATLAB code,
        whose examples place the coincident source in a vacuum-like region.  Pass
        ``reference_k=abs(n_region*k0)`` if a different homogeneous reference is
        desired.
        """

        k_ref = self.k0 if reference_k is None else float(reference_k)
        return float(6.0 * np.pi / k_ref * self.projected_im_green_at_source(source_position_m, orientation_cart))

    def electric_field_si(
        self,
        observer_position_m: Sequence[float],
        source_position_m: Sequence[float],
        dipole_moment_Cm: Sequence[complex],
        medium_region_for_prefactor: Optional[int] = None,
    ) -> np.ndarray:
        """Return the SI electric field generated by a point dipole moment.

        This helper applies ``E = k^2 G p / eps0`` using the direct medium index
        when available.  For transmitted-only cases, pass ``medium_region_for_prefactor``
        explicitly if you want a different convention.
        """

        eps0 = 8.8541878128e-12
        source_region = self.region_of(source_position_m)
        observer_region = self.region_of(observer_position_m)
        if medium_region_for_prefactor is None:
            medium = self._direct_medium_index(source_region, observer_region)
            medium_region_for_prefactor = observer_region if medium is None else medium
        k = self.nr[medium_region_for_prefactor] * self.k0
        return (k**2 / eps0) * (
            self.calculate_total_Green_function(observer_position_m, source_position_m)
            @ np.asarray(dipole_moment_Cm, dtype=complex).reshape(3)
        )


__all__ = [
    "AngularFunctions",
    "GeometryName",
    "MieCoefficients",
    "MieGreenFunction",
    "MieResult",
    "RadialFunctions",
    "SourceCoefficients",
    "VectorSphericalFunctions",
    "azimuthal_table",
    "cartesian_to_spherical",
    "dlog_riccati",
    "mie_coreshell",
    "mie_simple_cavity",
    "mie_single",
    "normalized_tau_pi_p",
    "spherical_radial_functions",
    "vector_cartesian_to_spherical",
    "vector_spherical_functions",
    "vector_spherical_to_cartesian",
    "wigner_d",
]
