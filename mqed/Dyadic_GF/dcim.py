from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import jv


@dataclass(frozen=True)
class ComplexImageFit:
    coefficients: np.ndarray
    exponents: np.ndarray
    q_origin: float = 0.0
    relative_residual: float | None = None
    singular_values: np.ndarray | None = None


def _svd_solve(matrix: np.ndarray, vector: np.ndarray, rcond: float):
    u_matrix, singular_values, vh_matrix = np.linalg.svd(matrix, full_matrices=False)
    cutoff = float(rcond) * singular_values[0] if singular_values.size else 0.0
    inverse_singular = np.zeros_like(singular_values, dtype=complex)
    keep = singular_values > cutoff
    inverse_singular[keep] = 1.0 / singular_values[keep]
    solution = vh_matrix.conj().T @ (inverse_singular * (u_matrix.conj().T @ vector))
    return solution, singular_values


def fit_exponentials(
    q_values: np.ndarray,
    samples: np.ndarray,
    image_count: int,
    rcond: float = 1e-10,
    q_origin: float = 0.0,
) -> ComplexImageFit:
    q_values = np.asarray(q_values, dtype=float)
    samples = np.asarray(samples, dtype=complex)
    if q_values.ndim != 1 or samples.ndim != 1 or q_values.size != samples.size:
        raise ValueError("q_values and samples must be matching 1-D arrays.")
    if q_values.size < 2 * image_count + 1:
        raise ValueError("sample_count must be at least 2 * image_count + 1.")

    dq_values = np.diff(q_values)
    dq = float(dq_values[0])
    if dq <= 0 or not np.allclose(dq_values, dq, rtol=1e-6, atol=1e-15):
        raise ValueError("GPOF samples must be uniformly spaced in q.")

    row_count = samples.size - image_count
    predictor = np.empty((row_count, image_count), dtype=complex)
    target = np.empty(row_count, dtype=complex)
    for row in range(row_count):
        predictor[row] = samples[row + image_count - 1:row - 1 if row else None:-1]
        target[row] = -samples[row + image_count]

    polynomial_tail, predictor_singular_values = _svd_solve(predictor, target, rcond)
    roots = np.roots(np.concatenate(([1.0 + 0.0j], polynomial_tail)))
    roots = roots[np.isfinite(roots)]
    roots = roots[(np.abs(roots) > 1e-14) & (np.abs(roots) < 1.0 - 1e-12)]
    if roots.size == 0:
        return ComplexImageFit(
            coefficients=np.array([], dtype=complex),
            exponents=np.array([], dtype=complex),
            q_origin=float(q_origin),
            relative_residual=np.inf,
            singular_values=np.array([], dtype=float),
        )

    roots = roots[np.argsort(np.abs(roots))[::-1]][:image_count]
    exponents = -np.log(roots) / dq
    shifted_q = q_values - float(q_origin)
    basis = np.exp(-np.outer(shifted_q, exponents))
    coefficients, basis_singular_values = _svd_solve(basis, samples, rcond)
    fitted = basis @ coefficients
    denominator = np.linalg.norm(samples)
    if denominator < 1e-30:
        relative_residual = float(np.linalg.norm(fitted - samples))
    else:
        relative_residual = float(np.linalg.norm(fitted - samples) / denominator)
    return ComplexImageFit(
        coefficients=coefficients,
        exponents=exponents,
        q_origin=float(q_origin),
        relative_residual=relative_residual,
        singular_values=np.concatenate((predictor_singular_values, basis_singular_values)),
    )


def evaluate_complex_images(fit: ComplexImageFit, q_values: np.ndarray) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=float)
    if fit.coefficients.size == 0:
        return np.zeros_like(q_values, dtype=complex)
    shifted_q = q_values - fit.q_origin
    basis = np.exp(-np.outer(shifted_q, fit.exponents))
    return basis @ fit.coefficients


def laplace_bessel_transform(exponent: complex, rho: float, order: int) -> complex:
    rho = float(abs(rho))
    if rho < 1e-15:
        if order == 0:
            return 1 / exponent
        return 0.0 + 0.0j

    radius = np.sqrt(exponent**2 + rho**2)
    if order == 0:
        return 1 / radius
    if order == 1:
        return (radius - exponent) / (rho * radius)
    if order == 2:
        return (radius - exponent) ** 2 / (rho**2 * radius)
    raise ValueError("Only Bessel orders 0, 1, and 2 are used by the DCIM kernels.")


def integrate_complex_images(fit: ComplexImageFit, rho: float, order: int) -> complex:
    if fit.coefficients.size == 0:
        return 0.0 + 0.0j
    values = [laplace_bessel_transform(exponent, rho, order) for exponent in fit.exponents]
    shifted_coefficients = fit.coefficients * np.exp(fit.exponents * fit.q_origin)
    return np.dot(shifted_coefficients, np.asarray(values, dtype=complex))


def integrate_complex_images_range(
    fit: ComplexImageFit,
    rho: float,
    order: int,
    lower: float = 0.0,
    upper: float = np.inf,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
    limit: int = 100,
) -> complex:
    if fit.coefficients.size == 0:
        return 0.0 + 0.0j
    lower = float(lower)
    upper = float(upper)
    if lower < 0:
        raise ValueError("lower must be non-negative for complex-image integration.")
    if upper <= lower:
        return 0.0 + 0.0j
    if lower == 0 and np.isinf(upper):
        return integrate_complex_images(fit, rho, order)

    def integrand(q: float) -> complex:
        return evaluate_complex_images(fit, np.array([q]))[0] * jv(order, q * rho)

    value, _ = quad_vec(
        integrand,
        lower,
        upper,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    return value
