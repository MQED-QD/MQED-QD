"""Pole and branch-cut utilities for N-layer Sommerfeld Green functions.

The routines in this module are intentionally independent of the concrete
``NLayerGreenFunction`` class.  They work with callables such as
``denominator(q)`` or ``kernel(q)`` so they can be reused by planar multilayer,
Sommerfeld-interface, or transmission-line Green-function implementations.

The most useful entry points are

* :func:`find_poles_by_winding` - locate zeros of an Airy/modal denominator by
  recursively applying the argument principle on rectangular boxes.
* :func:`residue_vector_by_contour` - estimate vector residues of a spectral
  kernel after a pole location has been found.
* :func:`branch_cut_integral_kz` - evaluate a diagnostic branch-cut contour in a
  vertical-wavenumber variable.  This is useful for validating branch-cut DCIM
  ideas before making them the production solver.

All lengths are SI, so a pole in the q plane has units of 1/m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import root
from scipy.special import hankel1, jv

ArrayLikeComplex = complex | np.ndarray


@dataclass(frozen=True)
class ComplexBox:
    """Axis-aligned rectangular search box in the complex q plane.

    Args:
        real_min: Lower bound for ``Re(q)`` in SI units.
        real_max: Upper bound for ``Re(q)`` in SI units.
        imag_min: Lower bound for ``Im(q)`` in SI units.
        imag_max: Upper bound for ``Im(q)`` in SI units.
        depth: Recursion depth used by the winding-number subdivision search.
    """

    real_min: float
    real_max: float
    imag_min: float
    imag_max: float
    depth: int = 0

    @property
    def width(self) -> float:
        return float(self.real_max - self.real_min)

    @property
    def height(self) -> float:
        return float(self.imag_max - self.imag_min)

    @property
    def center(self) -> complex:
        return complex(0.5 * (self.real_min + self.real_max), 0.5 * (self.imag_min + self.imag_max))

    @property
    def diameter(self) -> float:
        return float(np.hypot(self.width, self.height))

    def subdivide(self) -> tuple["ComplexBox", "ComplexBox", "ComplexBox", "ComplexBox"]:
        mid_r = 0.5 * (self.real_min + self.real_max)
        mid_i = 0.5 * (self.imag_min + self.imag_max)
        d = self.depth + 1
        return (
            ComplexBox(self.real_min, mid_r, self.imag_min, mid_i, d),
            ComplexBox(mid_r, self.real_max, self.imag_min, mid_i, d),
            ComplexBox(self.real_min, mid_r, mid_i, self.imag_max, d),
            ComplexBox(mid_r, self.real_max, mid_i, self.imag_max, d),
        )


@dataclass(frozen=True)
class PoleSearchConfig:
    r"""Controls recursive pole search by the argument principle.

    The search locates zeros of a scalar denominator ``D(q)`` by sampling boxes
    in the complex ``q`` plane and applying

    .. math::

       N_Z-N_P = \frac{1}{2\pi i}\oint_{\partial B}\frac{D'(q)}{D(q)}dq
       = \frac{\Delta \arg D(q)}{2\pi}.

    For the Airy denominators used by :class:`NLayerGreenFunction`, the relevant
    singularities are zeros rather than poles of ``D``.  Nonzero winding boxes
    are subdivided until they are small enough for a local complex root solve.
    The default region is intended for outgoing-wave poles with positive
    ``Re(q)`` and negative ``Im(q)``.  Use :meth:`from_k0` to specify the region
    in dimensionless ``q/k0`` units.

    Args:
        real_min: Lower search bound for ``Re(q)`` in SI units.
        real_max: Upper search bound for ``Re(q)`` in SI units.
        imag_min: Lower search bound for ``Im(q)`` in SI units.
        imag_max: Upper search bound for ``Im(q)`` in SI units.
        contour_points_per_side: Number of denominator samples on each box side.
        max_depth: Maximum recursive subdivision depth.
        min_box_size: Minimum box width/height before accepting a candidate.
        winding_tol: Absolute winding below this threshold is treated as zero.
        denominator_floor: Near-zero denominator threshold that forces subdivision.
        root_tol: Tolerance passed to the local root solver.
        residual_tol: Maximum accepted ``abs(D(q_p))`` after refinement.
        dedup_tol: Distance below which two refined roots are treated as one pole.
        max_boxes: Safety bound on the number of boxes tested.
        skip_branch_points: Branch points to guard against false pole detections.
        branch_point_guard: Exclusion radius around branch points in SI units.
    """

    real_min: float
    real_max: float
    imag_min: float
    imag_max: float
    contour_points_per_side: int = 24
    max_depth: int = 10
    min_box_size: float = 1e-6
    winding_tol: float = 0.35
    denominator_floor: float = 1e-12
    root_tol: float = 1e-10
    residual_tol: float = 1e-6
    dedup_tol: float = 1e-5
    max_boxes: int = 20000
    skip_branch_points: Sequence[complex] = field(default_factory=tuple)
    branch_point_guard: float = 0.0

    @classmethod
    def from_k0(
        cls,
        k0: float,
        real_min_factor: float = 0.0,
        real_max_factor: float = 6.0,
        imag_min_factor: float = -2.0,
        imag_max_factor: float = 1e-3,
        **kwargs,
    ) -> "PoleSearchConfig":
        k0_abs = float(abs(k0))
        if k0_abs <= 0:
            raise ValueError("k0 must be nonzero.")
        return cls(
            real_min=real_min_factor * k0_abs,
            real_max=real_max_factor * k0_abs,
            imag_min=imag_min_factor * k0_abs,
            imag_max=imag_max_factor * k0_abs,
            min_box_size=float(kwargs.pop("min_box_size_factor", 1e-6)) * k0_abs,
            dedup_tol=float(kwargs.pop("dedup_tol_factor", 1e-5)) * k0_abs,
            branch_point_guard=float(kwargs.pop("branch_point_guard_factor", 1e-4)) * k0_abs,
            **kwargs,
        )


@dataclass(frozen=True)
class SommerfeldPole:
    """One located Sommerfeld pole/root.

    Args:
        q: Complex in-plane pole wave number in SI units.
        polarization: Polarization label, normally ``"s"`` or ``"p"``.
        residual: Absolute denominator residual at ``q``.
        winding_number: Winding number of the terminal box that produced the pole.
        box: Terminal search box associated with this pole, when available.
        derivative: Numerical derivative of the searched denominator at ``q``.
    """

    q: complex
    polarization: str
    residual: float
    winding_number: int = 1
    box: ComplexBox | None = None
    derivative: complex | None = None

@dataclass(frozen=True)
class PoleResidue:
    r"""Residue of the seven Bessel-free kernels at one pole.

    If a Bessel-free kernel has a simple pole at ``q_p``,

    .. math::

       F_m(q)=\frac{A_m}{q-q_p}+F_m^{\mathrm{reg}}(q),

    then ``residues[m]`` stores ``A_m`` before multiplication by Bessel or Hankel
    functions.
    """

    pole: SommerfeldPole
    residues: np.ndarray
    contour_radius: float
    method: str = "contour"


@dataclass(frozen=True)
class BranchCutConfig:
    r"""Controls branch-cut diagnostic integration in a ``kz`` variable.

    ``branch_layer`` selects the layer whose light-line branch point
    ``sqrt(eps_l) k0`` is used.  The path is parameterized by the vertical
    wavenumber

    .. math::

       k_{z,l}=t \pm i\eta,\qquad
       q_\pm(t)=\sqrt{k_l^2-k_{z,l}^2},\qquad
       \frac{dq_\pm}{dt}=-\frac{k_{z,l}}{q_\pm}.

    For lossy stacks and arbitrary branch-cut conventions this should be treated
    as a diagnostic/validation tool rather than a universal production solver.

    Args:
        branch_layer: Layer index whose light-line branch point is sampled.
        t_limit: Integration half-width for ``t`` in SI units.
        side_offset: Offset ``eta`` separating the two branch-cut sides.
        epsabs: Absolute quadrature tolerance.
        epsrel: Relative quadrature tolerance.
        limit: Maximum number of ``quad_vec`` subintervals.
        use_hankel: Use outgoing Hankel functions instead of Bessel functions.
        include_two_sides: Return the two-side jump when true; otherwise sample
            only the ``+`` side.
    """

    branch_layer: int
    t_limit: float
    side_offset: float = 0.0
    epsabs: float = 1e-8
    epsrel: float = 1e-8
    limit: int = 200
    use_hankel: bool = True
    include_two_sides: bool = True

    @classmethod
    def from_k0(
        cls,
        k0: float,
        branch_layer: int,
        t_limit_factor: float = 8.0,
        side_offset_factor: float = 1e-6,
        **kwargs,
    ) -> "BranchCutConfig":
        k0_abs = float(abs(k0))
        if k0_abs <= 0:
            raise ValueError("k0 must be nonzero.")
        return cls(
            branch_layer=branch_layer,
            t_limit=t_limit_factor * k0_abs,
            side_offset=side_offset_factor * k0_abs,
            **kwargs,
        )


def _as_1d_complex(value: ArrayLikeComplex) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def contour_points(box: ComplexBox, points_per_side: int) -> np.ndarray:
    """Counter-clockwise contour points around ``box`` without duplicate corners.

    Args:
        box: Rectangular complex-q domain.
        points_per_side: Number of samples per side, excluding the next corner.

    Returns:
        Closed contour samples with the starting point repeated once at the end.
    """

    n = int(points_per_side)
    if n < 2:
        raise ValueError("points_per_side must be at least 2.")
    r0, r1 = box.real_min, box.real_max
    i0, i1 = box.imag_min, box.imag_max
    bottom = np.linspace(r0 + 1j * i0, r1 + 1j * i0, n, endpoint=False)
    right = np.linspace(r1 + 1j * i0, r1 + 1j * i1, n, endpoint=False)
    top = np.linspace(r1 + 1j * i1, r0 + 1j * i1, n, endpoint=False)
    left = np.linspace(r0 + 1j * i1, r0 + 1j * i0, n, endpoint=False)
    return np.concatenate((bottom, right, top, left, np.array([r0 + 1j * i0], dtype=complex)))


def winding_number(values: np.ndarray, denominator_floor: float = 1e-12) -> int:
    r"""Integer winding number of complex samples around the origin.

    The discrete estimate is

    .. math::

       w \approx \operatorname{round}\left[\frac{\Delta\arg D}{2\pi}\right].

    If a contour sample passes too close to the origin, the function returns a
    nonzero value to force subdivision instead of accidentally discarding a pole.
    """

    vals = np.asarray(values, dtype=complex)
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return 0
    if np.min(np.abs(vals)) < denominator_floor:
        # The contour probably passes too close to a zero/pole.  Returning a
        # nonzero value forces subdivision, which is safer than discarding it.
        return 1
    phases = np.unwrap(np.angle(vals))
    return int(np.rint((phases[-1] - phases[0]) / (2 * np.pi)))


def _near_branch_point(box: ComplexBox, branch_points: Sequence[complex], guard: float) -> bool:
    if guard <= 0 or not branch_points:
        return False
    c = box.center
    expanded_real = (box.real_min - guard, box.real_max + guard)
    expanded_imag = (box.imag_min - guard, box.imag_max + guard)
    for bp in branch_points:
        if expanded_real[0] <= np.real(bp) <= expanded_real[1] and expanded_imag[0] <= np.imag(bp) <= expanded_imag[1]:
            return True
        if abs(c - bp) <= guard:
            return True
    return False


def _box_winding(denominator: Callable[[complex], complex], box: ComplexBox, config: PoleSearchConfig) -> int:
    pts = contour_points(box, config.contour_points_per_side)
    try:
        vals = np.array([denominator(q) for q in pts], dtype=complex)
    except Exception:
        return 0
    if not np.all(np.isfinite(vals)):
        return 0
    return winding_number(vals, config.denominator_floor)


def _refine_root(
    denominator: Callable[[complex], complex],
    start: complex,
    tol: float,
) -> tuple[complex, float, bool]:
    def f_xy(xy: np.ndarray) -> np.ndarray:
        z = complex(float(xy[0]), float(xy[1]))
        value = denominator(z)
        return np.array([np.real(value), np.imag(value)], dtype=float)

    sol = root(f_xy, np.array([np.real(start), np.imag(start)], dtype=float), tol=tol)
    q = complex(float(sol.x[0]), float(sol.x[1]))
    res = float(abs(denominator(q)))
    return q, res, bool(sol.success)


def complex_derivative(func: Callable[[complex], complex], z: complex, step: float) -> complex:
    r"""Symmetric finite-difference derivative in the complex plane.

    .. math::

       f'(z) \approx \frac{f(z+h)-f(z-h)}{2h}.
    """

    h = float(step)
    if h <= 0:
        h = max(1e-9, 1e-7 * max(1.0, abs(z)))
    return (func(z + h) - func(z - h)) / (2.0 * h)


def find_poles_by_winding(
    denominator: Callable[[complex], complex],
    polarization: str,
    config: PoleSearchConfig,
) -> list[SommerfeldPole]:
    """Find denominator zeros in a rectangle using recursive winding numbers.

    The algorithm implements the spirit of the pole-locating method described in
    the FIPWA literature: use contour information to find all boxes that contain
    singularities/zeros, recursively subdivide them, then refine the candidates
    with a local nonlinear root solve.
    """

    root_box = ComplexBox(config.real_min, config.real_max, config.imag_min, config.imag_max, 0)
    queue: list[ComplexBox] = [root_box]
    candidates: list[tuple[ComplexBox, int]] = []
    boxes_tested = 0

    while queue:
        box = queue.pop()
        boxes_tested += 1
        if boxes_tested > config.max_boxes:
            break
        if _near_branch_point(box, config.skip_branch_points, config.branch_point_guard):
            # Branch points are not modal poles.  They are handled by branch-cut
            # diagnostics, so do not chase winding generated by the square root.
            continue
        w = _box_winding(denominator, box, config)
        if abs(w) < config.winding_tol:
            continue
        if box.depth >= config.max_depth or max(box.width, box.height) <= config.min_box_size:
            candidates.append((box, int(np.sign(w)) if w != 0 else 1))
            continue
        queue.extend(box.subdivide())

    poles: list[SommerfeldPole] = []
    for box, w in candidates:
        q0 = box.center
        q_refined, residual, success = _refine_root(denominator, q0, config.root_tol)
        if not success and residual > config.residual_tol:
            # Keep the box center only if it is already a plausible denominator minimum.
            q_refined = q0
            residual = float(abs(denominator(q_refined)))
        if residual > config.residual_tol:
            continue
        if not (config.real_min <= np.real(q_refined) <= config.real_max):
            continue
        if not (config.imag_min <= np.imag(q_refined) <= config.imag_max):
            continue
        if _near_branch_point(
            ComplexBox(np.real(q_refined), np.real(q_refined), np.imag(q_refined), np.imag(q_refined)),
            config.skip_branch_points,
            config.branch_point_guard,
        ):
            continue
        if any(abs(q_refined - pole.q) <= config.dedup_tol for pole in poles):
            continue
        deriv = complex_derivative(denominator, q_refined, max(config.min_box_size, 1e-7 * max(1.0, abs(q_refined))))
        poles.append(
            SommerfeldPole(
                q=q_refined,
                polarization=polarization,
                residual=residual,
                winding_number=w,
                box=box,
                derivative=deriv,
            )
        )

    poles.sort(key=lambda p: (np.real(p.q), np.imag(p.q), p.polarization))
    return poles


def residue_vector_by_contour(
    kernel: Callable[[complex], ArrayLikeComplex],
    pole: SommerfeldPole,
    radius: float,
    points: int = 96,
) -> PoleResidue:
    r"""Estimate vector residues by a circular contour integral.

    .. math::

       A_m = \frac{1}{2\pi i}\oint_{|q-q_p|=r} F_m(q)\,dq.

    Args:
        kernel: Callable returning one or more Bessel-free kernel values.
        pole: Located pole around which the contour is drawn.
        radius: Positive contour radius in SI units.
        points: Number of endpoint-free trapezoid samples on the circle.
    """

    if radius <= 0:
        raise ValueError("radius must be positive.")
    n = int(points)
    if n < 16:
        raise ValueError("points must be at least 16.")
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    q = pole.q + radius * np.exp(1j * theta)
    dq_dtheta = 1j * radius * np.exp(1j * theta)
    samples = np.array([_as_1d_complex(kernel(qq)) for qq in q], dtype=complex)
    # Periodic trapezoid rule on endpoint-free samples.  Using ``np.trapezoid``
    # with endpoint-free theta would omit the closing segment.
    dtheta = 2.0 * np.pi / n
    integral = dtheta * np.sum(samples * dq_dtheta[:, None], axis=0)
    residues = integral / (2.0j * np.pi)
    return PoleResidue(pole=pole, residues=residues, contour_radius=radius, method="contour")


def residue_vector_by_limit(
    kernel: Callable[[complex], ArrayLikeComplex],
    pole: SommerfeldPole,
    step: float,
) -> PoleResidue:
    r"""Estimate residues from the simple-pole limit.

    .. math::

       A_m = \lim_{q\to q_p} (q-q_p)F_m(q).

    Four axial approach directions are averaged to reduce directional bias.
    """

    if step <= 0:
        raise ValueError("step must be positive.")
    directions = np.array([1.0, -1.0, 1.0j, -1.0j], dtype=complex)
    samples = []
    for direction in directions:
        q = pole.q + step * direction
        samples.append(step * direction * _as_1d_complex(kernel(q)))
    residues = np.mean(np.array(samples, dtype=complex), axis=0)
    return PoleResidue(pole=pole, residues=residues, contour_radius=step, method="limit")


def pole_integral_contribution(
    residue: PoleResidue,
    rho: float,
    orders: Sequence[int],
    prefactor: complex = 1j * np.pi,
    use_hankel: bool = True,
) -> np.ndarray:
    """Convert kernel residues to scalar Sommerfeld-integral pole terms.

    The default prefactor ``i*pi`` is the common half-line/Hankel contour factor
    for outgoing waves, but sign conventions differ.  Treat this routine as an
    explicit diagnostic unless you have validated the prefactor against a known
    analytic case for your exact Green-function normalization.
    """

    q = residue.pole.q
    values = []
    for res, order in zip(residue.residues, orders):
        special = hankel1(order, q * rho) if use_hankel else jv(order, q * rho)
        values.append(prefactor * res * special)
    return np.asarray(values, dtype=complex)


def _sqrt_physical(value: complex) -> complex:
    root_value = np.lib.scimath.sqrt(value)
    if np.imag(root_value) < 0 or (abs(np.imag(root_value)) < 1e-18 and np.real(root_value) < 0):
        root_value = -root_value
    return complex(root_value)


def branch_cut_integral_kz(
    kernel: Callable[[complex], ArrayLikeComplex],
    branch_wavenumber: complex,
    rho: float,
    orders: Sequence[int],
    config: BranchCutConfig,
) -> np.ndarray:
    """Diagnostic branch-cut integral parameterized by a layer vertical wavenumber.

    This evaluates

        integral [K(q_+) B(q_+ rho) dq_+/dt - K(q_-) B(q_- rho) dq_-/dt] dt

    on ``t in [-t_limit, t_limit]`` if ``include_two_sides`` is true.  With one
    side it evaluates only the ``+`` path.  The result is a vector with the same
    length as ``orders`` and the kernel vector.
    """

    orders = list(orders)
    tmax = float(config.t_limit)
    offset = float(config.side_offset)
    if tmax <= 0:
        raise ValueError("t_limit must be positive.")

    def mapped_path(t: float, sign: float) -> tuple[complex, complex]:
        kz = complex(float(t), sign * offset)
        q = _sqrt_physical(branch_wavenumber**2 - kz**2)
        if abs(q) < 1e-30:
            dqdt = 0.0 + 0.0j
        else:
            dqdt = -kz / q
        return q, dqdt

    def side_value(t: float, sign: float) -> np.ndarray:
        q, dqdt = mapped_path(t, sign)
        kvec = _as_1d_complex(kernel(q))
        if kvec.size != len(orders):
            raise ValueError("kernel vector length must match the number of Bessel/Hankel orders.")
        special = np.array(
            [hankel1(order, q * rho) if config.use_hankel else jv(order, q * rho) for order in orders],
            dtype=complex,
        )
        return kvec * special * dqdt

    def integrand(t: float) -> np.ndarray:
        plus = side_value(t, +1.0)
        if not config.include_two_sides:
            return plus
        minus = side_value(t, -1.0)
        return plus - minus

    value, _ = quad_vec(
        integrand,
        -tmax,
        tmax,
        epsabs=config.epsabs,
        epsrel=config.epsrel,
        limit=config.limit,
    )
    return np.asarray(value, dtype=complex)


def branch_cut_samples_kz(
    kernel: Callable[[complex], ArrayLikeComplex],
    branch_wavenumber: complex,
    t_values: np.ndarray,
    side_offset: float = 0.0,
    side: int = +1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a kernel along a kz-parameterized branch-cut side.

    Returns ``(q_values, samples)``.  This is intended for matrix-pencil fitting
    experiments on a Sommerfeld branch cut.
    """

    t_values = np.asarray(t_values, dtype=float)
    q_values = np.empty_like(t_values, dtype=complex)
    samples = []
    for idx, t in enumerate(t_values):
        kz = complex(float(t), float(side) * float(side_offset))
        q = _sqrt_physical(branch_wavenumber**2 - kz**2)
        q_values[idx] = q
        samples.append(_as_1d_complex(kernel(q)))
    return q_values, np.asarray(samples, dtype=complex)
