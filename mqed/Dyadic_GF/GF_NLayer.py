"""N-layer Sommerfeld dyadic Green's function for planar dielectric stacks.

The implementation follows the Tomas/Welsch convention used in
``local/derivation/multi_layer_GF.tex``.  Layers are ordered from bottom to
top.  The first and last layers are normally semi-infinite; finite internal
layers carry a physical thickness.  Source and observer are assumed to lie in
the same layer, with their z-coordinates measured upward from that layer's
lower interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
from loguru import logger
from scipy.integrate import quad_vec
from scipy.special import hankel1, jv

from mqed.Dyadic_GF.dcim import (
    fit_exponentials,
    integrate_complex_images,
    integrate_complex_images_range,
)
from mqed.Dyadic_GF.sommerfeld_singularities import (
    BranchCutConfig,
    PoleResidue,
    PoleSearchConfig,
    SommerfeldPole,
    branch_cut_integral_kz,
    branch_cut_samples_kz,
    find_poles_by_winding,
    pole_integral_contribution,
    residue_vector_by_contour,
    residue_vector_by_limit,
)
from mqed.utils.SI_unit import c, eV_to_J, hbar


@dataclass(frozen=True)
class LayerSpec:
    """One layer in a bottom-to-top planar stack.

    Args:
        epsilon: Complex relative permittivity at the active frequency.
        thickness_m: Layer thickness in meters. Use ``None`` for semi-infinite
            boundary layers.
        name: Human-readable layer label.
    """

    epsilon: complex
    thickness_m: Optional[float]
    name: str = "layer"


class NLayerGreenFunction:
    r"""Dyadic Green's function for source/observer in one layer of an N-layer stack.

    The scattered tensor is assembled from seven Sommerfeld integrals.  All
    multilayer physics enters through the generalized upward/downward Fresnel
    coefficients ``r_plus`` and ``r_minus`` computed by recursive Airy formulas.

    For a source and observer in layer ``j``, the total Green tensor is

    .. math::

       \mathbf G(\mathbf r,\mathbf r_0,\omega)
       = \mathbf G_0^{(j)}(\mathbf r,\mathbf r_0,\omega)
       + \mathbf G_{\mathrm{sc}}^{(j)}(\rho,\phi,z,z_0,\omega).

    The scattered term has the angular-spectrum form

    .. math::

       \mathbf G_{\mathrm{sc}}^{(j)} = \frac{i}{4\pi}
       \left(\mathbf M_s + \mathbf M_p\right),

    where ``M_s`` and ``M_p`` are assembled from seven scalar Sommerfeld
    integrals.  The layer stack enters those integrals only through recursive
    effective mirrors ``r_minus`` and ``r_plus`` and the source-layer Airy
    denominator

    .. math::

       D_\sigma(q) = 1-r_{j-}^{\sigma}(q)r_{j+}^{\sigma}(q)
       \exp(2i\beta_j d_j), \quad \sigma\in\{s,p\}.

    Args:
        layers: Bottom-to-top layer stack.
        source_layer: Zero-based index of the layer containing source and
            observer. The layer must have finite thickness unless it is used in
            a one-sided limit.
        omega: Angular frequency in rad/s.
        qmax: Optional finite upper limit for the Sommerfeld integral.
        epsabs: Absolute quadrature tolerance.
        epsrel: Relative quadrature tolerance.
        limit: Maximum number of ``quad_vec`` subintervals.
        split_propagating: If true, split the integration at the source-layer
            light line ``sqrt(eps_j) * k0`` when it is real.
        integration_method: Numerical integration method. ``singularity_aware``
            uses direct real-axis quadrature split at branch points and near-real
            pole projections; explicit pole/branch terms remain diagnostic.
            ``branch_cut_dcim`` is an experimental pole-plus-branch-cut
            exponential-fit method validated against ``singularity_aware``.
        dcim_q_start: Lower q bound for legacy DCIM sampling.
        dcim_q_stop: Upper q bound for legacy DCIM sampling.
        dcim_sample_count: Number of spectral samples used by DCIM diagnostics.
        dcim_image_count: Number of exponentials/images in DCIM fits.
        hybrid_direct_q_stop: Direct-quadrature cutoff for hybrid DCIM.
        hybrid_tail_q_stop: Tail-fit endpoint for hybrid DCIM validation.
        hybrid_sample_count: Tail sample count for hybrid DCIM.
        hybrid_image_count: Number of tail exponentials in hybrid DCIM.
        hybrid_validation_rtol: Maximum accepted relative finite-tail error.
        hybrid_validate_tail: If true, compare fitted tail against direct tail.
        hybrid_fallback_to_direct: If true, rejected hybrid fits return direct results.
        fixed_grid_points: Number of q nodes for fixed-grid integration.
        pole_search_real_min_factor: Lower ``Re(q)/abs(k0)`` pole-search bound.
        pole_search_real_max_factor: Upper ``Re(q)/abs(k0)`` pole-search bound.
        pole_search_imag_min_factor: Lower ``Im(q)/abs(k0)`` pole-search bound.
        pole_search_imag_max_factor: Upper ``Im(q)/abs(k0)`` pole-search bound.
        pole_search_max_depth: Maximum recursive pole-search subdivision depth.
        pole_search_contour_points: Denominator samples per search-box side.
        pole_search_residual_tol: Accepted residual ``abs(D_sigma(q_p))``.
        pole_search_branch_guard_factor: Branch-point exclusion radius in ``abs(k0)`` units.
        pole_residue_radius_factor: Initial residue-contour radius in ``abs(k0)`` units.
        pole_residue_points: Circular-contour samples for residue extraction.
        pole_contribution_prefactor: Diagnostic pole-term prefactor before Hankel factors.
        branch_cut_t_limit_factor: Branch-cut ``kz`` integration half-width in ``abs(k0)`` units.
        branch_cut_side_offset_factor: Two-side branch offset in ``abs(k0)`` units.
        branch_cut_layers: Layers whose light-line branch cuts are included in
            ``branch_cut_dcim``. Use ``"auto"`` for the two exterior layers.
        branch_cut_sample_count: Uniform ``kz`` samples per branch cut for GPOF fitting.
        branch_cut_image_count: Number of exponentials fitted per branch-cut kernel.
        branch_cut_validation_rtol: Maximum relative error versus ``singularity_aware``.
        branch_cut_validate: If true, validate the branch-cut approximation.
        branch_cut_fallback_to_singularity_aware: If true, rejected fits return
            the validated split-quadrature reference.
        branch_cut_include_poles: Add explicitly extracted Airy-denominator pole terms.
        branch_cut_prefactor: Explicit multiplier for fitted branch-cut contour terms.
        branch_cut_jump_sign: Sign multiplying the two-side branch-cut jump.
        branch_cut_use_hankel: Use outgoing Hankel functions on branch contours.
    """

    def __init__(
        self,
        layers: Sequence[LayerSpec],
        source_layer: int,
        omega: float,
        qmax: Optional[float] = None,
        epsabs: float = 1e-9,
        epsrel: float = 1e-9,
        limit: int = 400,
        split_propagating: bool = False,
        integration_method: str = "direct",
        dcim_q_start: float = 0.0,
        dcim_q_stop: Optional[float] = None,
        dcim_sample_count: int = 128,
        dcim_image_count: int = 16,
        hybrid_direct_q_stop: Optional[float] = None,
        hybrid_tail_q_stop: Optional[float] = None,
        hybrid_sample_count: Optional[int] = None,
        hybrid_image_count: Optional[int] = None,
        hybrid_validation_rtol: float = 5e-2,
        hybrid_validate_tail: bool = True,
        hybrid_fallback_to_direct: bool = True,
        fixed_grid_points: int = 2048,
        pole_search_real_min_factor: float = 0.0,
        pole_search_real_max_factor: float = 6.0,
        pole_search_imag_min_factor: float = -2.0,
        pole_search_imag_max_factor: float = 1e-3,
        pole_search_max_depth: int = 10,
        pole_search_contour_points: int = 24,
        pole_search_residual_tol: float = 1e-6,
        pole_search_branch_guard_factor: float = 1e-4,
        pole_residue_radius_factor: float = 1e-4,
        pole_residue_points: int = 96,
        pole_contribution_prefactor: complex = 1j * np.pi,
        branch_cut_t_limit_factor: float = 8.0,
        branch_cut_side_offset_factor: float = 1e-6,
        branch_cut_layers: Sequence[int] | str | None = "auto",
        branch_cut_sample_count: int = 128,
        branch_cut_image_count: int = 16,
        branch_cut_validation_rtol: float = 5e-2,
        branch_cut_validate: bool = True,
        branch_cut_fallback_to_singularity_aware: bool = True,
        branch_cut_include_poles: bool = True,
        branch_cut_prefactor: complex = 1.0 + 0.0j,
        branch_cut_jump_sign: float = 1.0,
        branch_cut_use_hankel: bool = True,
    ):
        if len(layers) < 2:
            raise ValueError("An N-layer Green's function requires at least two layers.")
        if not 0 <= source_layer < len(layers):
            raise ValueError("source_layer must be a valid zero-based layer index.")

        self.layers = tuple(layers)
        self.source_layer = int(source_layer)
        self.omega = float(omega)
        self.c = c
        self.k0 = self.omega / self.c
        self.qmax = qmax
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = int(limit)
        self.split_propagating = bool(split_propagating)
        self.integration_method = integration_method.strip().lower()
        valid_methods = {
            "direct",
            "dcim",
            "hybrid_dcim",
            "fixed_grid",
            "singularity_aware",
            "branch_cut_dcim",
        }
        if self.integration_method not in valid_methods:
            raise ValueError(
                "integration_method must be 'direct', 'dcim', 'hybrid_dcim', "
                "'fixed_grid', 'singularity_aware', or 'branch_cut_dcim'."
            )
        self.dcim_q_start = float(dcim_q_start)
        self.dcim_q_stop = dcim_q_stop
        self.dcim_sample_count = int(dcim_sample_count)
        self.dcim_image_count = int(dcim_image_count)
        self.hybrid_direct_q_stop = hybrid_direct_q_stop
        self.hybrid_tail_q_stop = hybrid_tail_q_stop
        self.hybrid_sample_count = int(
            self.dcim_sample_count if hybrid_sample_count is None else hybrid_sample_count
        )
        self.hybrid_image_count = int(
            self.dcim_image_count if hybrid_image_count is None else hybrid_image_count
        )
        self.hybrid_validation_rtol = float(hybrid_validation_rtol)
        self.hybrid_validate_tail = bool(hybrid_validate_tail)
        self.hybrid_fallback_to_direct = bool(hybrid_fallback_to_direct)
        self.fixed_grid_points = int(fixed_grid_points)
        self.pole_search_real_min_factor = float(pole_search_real_min_factor)
        self.pole_search_real_max_factor = float(pole_search_real_max_factor)
        self.pole_search_imag_min_factor = float(pole_search_imag_min_factor)
        self.pole_search_imag_max_factor = float(pole_search_imag_max_factor)
        self.pole_search_max_depth = int(pole_search_max_depth)
        self.pole_search_contour_points = int(pole_search_contour_points)
        self.pole_search_residual_tol = float(pole_search_residual_tol)
        self.pole_search_branch_guard_factor = float(pole_search_branch_guard_factor)
        self.pole_residue_radius_factor = float(pole_residue_radius_factor)
        self.pole_residue_points = int(pole_residue_points)
        self.pole_contribution_prefactor = complex(pole_contribution_prefactor)
        self.branch_cut_t_limit_factor = float(branch_cut_t_limit_factor)
        self.branch_cut_side_offset_factor = float(branch_cut_side_offset_factor)
        self.branch_cut_layers = branch_cut_layers
        self.branch_cut_sample_count = int(branch_cut_sample_count)
        self.branch_cut_image_count = int(branch_cut_image_count)
        self.branch_cut_validation_rtol = float(branch_cut_validation_rtol)
        self.branch_cut_validate = bool(branch_cut_validate)
        self.branch_cut_fallback_to_singularity_aware = bool(
            branch_cut_fallback_to_singularity_aware
        )
        self.branch_cut_include_poles = bool(branch_cut_include_poles)
        self.branch_cut_prefactor = complex(branch_cut_prefactor)
        self.branch_cut_jump_sign = float(branch_cut_jump_sign)
        self.branch_cut_use_hankel = bool(branch_cut_use_hankel)
        self.last_hybrid_dcim_report: dict[str, object] | None = None
        self.last_singularity_report: dict[str, object] | None = None
        self.last_branch_cut_dcim_report: dict[str, object] | None = None
        self._pole_cache: dict[tuple, list[SommerfeldPole]] = {}

        self.eps = np.array([layer.epsilon for layer in self.layers], dtype=complex)
        self.thickness_m = [layer.thickness_m for layer in self.layers]
        self.source_thickness_m = self.thickness_m[self.source_layer]

    @staticmethod
    def _beta_phys(eps: complex, k0: float, q):
        r"""Physical z-wavevector branch for one homogeneous layer.

        .. math::

           \beta(q) = \sqrt{\epsilon k_0^2-q^2+i0^+}.

        The branch is chosen with ``Im(beta) >= 0`` so evanescent waves decay
        away from interfaces.  For nearly real propagating waves, ``Re(beta)``
        is kept non-negative.
        """
        beta = np.lib.scimath.sqrt(eps * k0**2 - q**2 + 1j * 1e-12)
        if np.ndim(beta):
            flip = (np.imag(beta) < 0) | (
                (np.abs(np.imag(beta)) < 1e-18) & (np.real(beta) < 0)
            )
            beta[flip] = -beta[flip]
            return beta
        if np.imag(beta) < 0 or (abs(np.imag(beta)) < 1e-18 and np.real(beta) < 0):
            return -beta
        return beta

    def _kz(self, layer: int, q):
        r"""Layer-indexed vertical wavevector.

        .. math::

           \beta_l(q)=\sqrt{\epsilon_l k_0^2-q^2+i0^+}.

        Args:
            layer: Zero-based layer index in the bottom-to-top stack.
            q: In-plane Sommerfeld wave number.
        """
        return self._beta_phys(self.eps[layer], self.k0, q)

    def _fresnel(self, layer_i: int, layer_j: int, q, polarization: str):
        r"""Bare Fresnel reflection coefficient from layer ``i`` to layer ``j``.

        TE polarization:

        .. math::

           R_{ij}^{s}=\frac{\beta_i-\beta_j}{\beta_i+\beta_j}.

        TM polarization:

        .. math::

           R_{ij}^{p}=\frac{\epsilon_j\beta_i-\epsilon_i\beta_j}
           {\epsilon_j\beta_i+\epsilon_i\beta_j}.
        """
        beta_i = self._kz(layer_i, q)
        beta_j = self._kz(layer_j, q)

        if polarization == "s":
            return (beta_i - beta_j) / (beta_i + beta_j)
        if polarization == "p":
            eps_i = self.eps[layer_i]
            eps_j = self.eps[layer_j]
            return (eps_j * beta_i - eps_i * beta_j) / (eps_j * beta_i + eps_i * beta_j)
        raise ValueError("polarization must be 's' or 'p'.")

    def _finite_phase(self, layer: int, q):
        r"""Round-trip propagation factor through a finite layer.

        .. math::

           P_l(q)=\exp(2i\beta_l(q)d_l).

        This factor appears in the recursive effective-reflection formula.
        """
        thickness = self.thickness_m[layer]
        if thickness is None:
            raise ValueError(
                f"Layer {layer} is semi-infinite and cannot appear inside a recursive mirror."
            )
        return np.exp(2j * self._kz(layer, q) * thickness)

    def reflection_coefficient(self, q, direction: str, polarization: str):
        r"""Effective mirror seen from the source layer.

        ``direction='down'`` returns ``r_minus`` and ``direction='up'`` returns
        ``r_plus`` for either TE (``s``) or TM (``p``) polarization.
        """
        if direction == "down":
            return self._effective_reflection(self.source_layer, -1, q, polarization)
        if direction == "up":
            return self._effective_reflection(self.source_layer, +1, q, polarization)
        raise ValueError("direction must be 'down' or 'up'.")

    def _effective_reflection(self, layer: int, step: int, q, polarization: str):
        r"""Recursive Airy reflection coefficient for an arbitrary sub-stack.

        For a step toward the next layer ``n = layer + step``, the effective
        reflection is

        .. math::

           \widetilde R_{l,n}^{\sigma}=
           \frac{R_{l,n}^{\sigma}+\widetilde R_{n,n+step}^{\sigma}
           \exp(2i\beta_n d_n)}
           {1+R_{l,n}^{\sigma}\widetilde R_{n,n+step}^{\sigma}
           \exp(2i\beta_n d_n)}.

        The recursion terminates at the semi-infinite outer layer, where the
        effective reflection is the bare Fresnel coefficient.
        """
        next_layer = layer + step
        if next_layer < 0 or next_layer >= len(self.layers):
            return 0.0 + 0.0j

        direct = self._fresnel(layer, next_layer, q, polarization)
        beyond = next_layer + step
        if beyond < 0 or beyond >= len(self.layers):
            return direct

        sub_stack = self._effective_reflection(next_layer, step, q, polarization)
        phase = self._finite_phase(next_layer, q)
        return (direct + sub_stack * phase) / (1 + direct * sub_stack * phase)

    def _source_layer_thickness(self) -> float:
        if self.source_thickness_m is None:
            raise ValueError(
                "The full N-layer amplitude formula requires source_layer to have "
                "finite thickness. For a two-layer half-space, use GF_Sommerfeld."
            )
        return float(self.source_thickness_m)

    def amplitude_coefficients(self, q, z_observer: float, z_source: float):
        r"""Five scalar amplitudes entering the N-layer scattered tensor.

        For source and observer in layer ``j`` of thickness ``d_j``:

        .. math::

           C_s = \frac{e^{i\beta_j(z+z_0-d_j)}r_{j-}^s
           + e^{-i\beta_j(z+z_0-d_j)}r_{j+}^s
           + 2r_{j+}^sr_{j-}^s\cos[\beta_j(z-z_0)]e^{i\beta_j d_j}}
           {D_s}.

        TM amplitudes use the same direct reflection part and opposite signs
        for the round-trip term:

        .. math::

           C_p^{\pm} = \frac{B_p \pm 2r_{j+}^pr_{j-}^p
           \cos[\beta_j(z-z_0)]e^{i\beta_j d_j}}{D_p},

        .. math::

           S_p^{\pm} = \frac{A_p \pm 2i r_{j+}^pr_{j-}^p
           \sin[\beta_j(z-z_0)]e^{i\beta_j d_j}}{D_p},

        with ``B_p = exp(i beta_j(z+z0-d_j)) r_minus_p +
        exp(-i beta_j(z+z0-d_j)) r_plus_p`` and ``A_p`` using the same two
        terms with a minus sign between them.
        """
        d_j = self._source_layer_thickness()
        beta_j = self._kz(self.source_layer, q)

        r_minus_s = self.reflection_coefficient(q, "down", "s")
        r_plus_s = self.reflection_coefficient(q, "up", "s")
        r_minus_p = self.reflection_coefficient(q, "down", "p")
        r_plus_p = self.reflection_coefficient(q, "up", "p")

        upper_phase = np.exp(1j * beta_j * (z_observer + z_source - d_j))
        lower_phase = np.exp(-1j * beta_j * (z_observer + z_source - d_j))
        round_phase = np.exp(1j * beta_j * d_j)
        cos_term = np.cos(beta_j * (z_observer - z_source))
        sin_term = np.sin(beta_j * (z_observer - z_source))

        denom_s = 1 - r_minus_s * r_plus_s * np.exp(2j * beta_j * d_j)
        denom_p = 1 - r_minus_p * r_plus_p * np.exp(2j * beta_j * d_j)

        c_s = (
            upper_phase * r_minus_s
            + lower_phase * r_plus_s
            + 2 * r_plus_s * r_minus_s * cos_term * round_phase
        ) / denom_s

        base_p = upper_phase * r_minus_p + lower_phase * r_plus_p
        product_p = r_plus_p * r_minus_p * round_phase
        c_p_plus = (base_p + 2 * product_p * cos_term) / denom_p
        c_p_minus = (base_p - 2 * product_p * cos_term) / denom_p
        s_p_plus = (
            upper_phase * r_minus_p - lower_phase * r_plus_p + 2j * product_p * sin_term
        ) / denom_p
        s_p_minus = (
            upper_phase * r_minus_p - lower_phase * r_plus_p - 2j * product_p * sin_term
        ) / denom_p
        return c_s, c_p_plus, c_p_minus, s_p_plus, s_p_minus

    def airy_denominator(self, q, polarization: str):
        r"""Source-layer multiple-reflection denominator.

        .. math::

           D_\sigma(q)=1-r_{j-}^{\sigma}(q)r_{j+}^{\sigma}(q)
           e^{2i\beta_j(q)d_j}.

        Zeros of this denominator are guided-mode or plasmonic pole candidates.
        """
        d_j = self._source_layer_thickness()
        beta_j = self._kz(self.source_layer, q)
        r_minus = self.reflection_coefficient(q, "down", polarization)
        r_plus = self.reflection_coefficient(q, "up", polarization)
        return 1 - r_minus * r_plus * np.exp(2j * beta_j * d_j)

    def branch_points(self) -> np.ndarray:
        r"""Layer light-line branch points in the complex q plane.

        .. math::

           q_l = \sqrt{\epsilon_l}\,k_0.

        These branch points mark sheet changes of ``beta_l(q)`` and guide the
        low-q split used by the hybrid DCIM path.
        """
        return np.sqrt(self.eps + 0j) * self.k0


    def default_pole_search_config(self) -> PoleSearchConfig:
        r"""Build the default rectangular pole-search window in the complex q plane.

        The default window searches positive-real, negative-imaginary outgoing
        poles, i.e. guided/surface modes with ``q = q_r - i alpha`` under the
        ``exp(-i omega t)`` convention used by the outgoing Green function.
        The window is specified in dimensionless ``q/k0`` factors in the
        constructor and is converted to SI units here.
        """
        return PoleSearchConfig.from_k0(
            self.k0,
            real_min_factor=self.pole_search_real_min_factor,
            real_max_factor=self.pole_search_real_max_factor,
            imag_min_factor=self.pole_search_imag_min_factor,
            imag_max_factor=self.pole_search_imag_max_factor,
            contour_points_per_side=self.pole_search_contour_points,
            max_depth=self.pole_search_max_depth,
            residual_tol=self.pole_search_residual_tol,
            skip_branch_points=tuple(self.branch_points()),
            branch_point_guard_factor=self.pole_search_branch_guard_factor,
        )

    def find_poles(
        self,
        polarizations: Sequence[str] = ("s", "p"),
        config: PoleSearchConfig | None = None,
        use_cache: bool = True,
    ) -> list[SommerfeldPole]:
        r"""Locate source-layer guided/surface-wave pole candidates.

        Poles are zeros of the source-layer Airy denominator

        .. math::

           D_\sigma(q)=1-r_{j-}^{\sigma}(q)r_{j+}^{\sigma}(q)e^{2i\beta_j d_j}.

        The search uses the argument principle on recursively subdivided boxes,
        then refines each candidate with a local complex root solve.  This is a
        diagnostic/extraction routine; direct real-axis integration already
        includes off-axis poles implicitly, while DCIM fits may need this
        information to avoid missing far-field guided or plasmonic modes.
        """
        cfg = self.default_pole_search_config() if config is None else config
        all_poles: list[SommerfeldPole] = []
        for pol in polarizations:
            if pol not in {"s", "p"}:
                raise ValueError("polarizations must contain only 's' and/or 'p'.")
            key = (
                pol,
                cfg.real_min,
                cfg.real_max,
                cfg.imag_min,
                cfg.imag_max,
                cfg.contour_points_per_side,
                cfg.max_depth,
                cfg.min_box_size,
                cfg.residual_tol,
                cfg.dedup_tol,
            )
            if use_cache and key in self._pole_cache:
                all_poles.extend(self._pole_cache[key])
                continue
            def denominator(q, polarization=pol):
                return self.airy_denominator(q, polarization)

            poles = find_poles_by_winding(denominator, pol, cfg)
            if use_cache:
                self._pole_cache[key] = poles
            all_poles.extend(poles)
        all_poles.sort(key=lambda pole: (np.real(pole.q), np.imag(pole.q), pole.polarization))
        return all_poles

    def pole_residues(
        self,
        z_observer: float,
        z_source: float,
        poles: Sequence[SommerfeldPole] | None = None,
        config: PoleSearchConfig | None = None,
        method: str = "contour",
    ) -> list[PoleResidue]:
        r"""Estimate residues of the seven Bessel-free kernels at each pole.

        The residues are for ``bessel_free_kernels(q,z,z0)``, i.e. before
        multiplication by Bessel/Hankel functions and before assembling the
        tensor.  ``method='contour'`` is slower but safer; ``method='limit'`` is
        useful when poles are well isolated.
        """
        located = list(self.find_poles(config=config) if poles is None else poles)
        if not located:
            return []
        base_radius = max(self.pole_residue_radius_factor * abs(self.k0), 1e-12)
        residues: list[PoleResidue] = []
        for pole in located:
            distances = [abs(pole.q - other.q) for other in located if other is not pole]
            for branch_point in self.branch_points():
                distances.append(abs(pole.q - branch_point))
            radius = base_radius
            positive_distances = [distance for distance in distances if distance > 0]
            if positive_distances:
                radius = min(radius, 0.25 * min(positive_distances))
            def kernel(q):
                return self.bessel_free_kernels(q, z_observer, z_source)

            if method == "contour":
                residues.append(
                    residue_vector_by_contour(
                        kernel,
                        pole,
                        radius=radius,
                        points=self.pole_residue_points,
                    )
                )
            elif method == "limit":
                residues.append(residue_vector_by_limit(kernel, pole, step=radius))
            else:
                raise ValueError("method must be 'contour' or 'limit'.")
        return residues

    def pole_contribution_integrals(
        self,
        rho: float,
        z_observer: float,
        z_source: float,
        poles: Sequence[SommerfeldPole] | None = None,
        prefactor: complex | None = None,
        use_hankel: bool = True,
    ) -> np.ndarray:
        r"""Return the seven scalar pole terms implied by the extracted residues.

        Because contour orientation and Green-function normalization differ
        between Sommerfeld formulations, this routine is deliberately explicit:
        validate the ``prefactor`` against a known case before adding the result
        to production data.  The default is ``i*pi`` for an outgoing half-line
        Hankel convention.
        """
        orders = [0, 2, 0, 2, 1, 1, 0]
        scale = self.pole_contribution_prefactor if prefactor is None else complex(prefactor)
        values = np.zeros(7, dtype=complex)
        for residue in self.pole_residues(z_observer, z_source, poles=poles):
            values += pole_integral_contribution(
                residue,
                rho,
                orders,
                prefactor=scale,
                use_hankel=use_hankel,
            )
        return values

    def default_branch_cut_config(self, branch_layer: int | None = None) -> BranchCutConfig:
        r"""Build a default ``kz``-plane branch-cut diagnostic configuration.

        The branch layer defines

        .. math::

           k_l=\sqrt{\epsilon_l}\,k_0,\qquad
           q_\pm(t)=\sqrt{k_l^2-(t\pm i\eta)^2}.

        Args:
            branch_layer: Optional layer index. Defaults to ``source_layer``.

        Returns:
            A :class:`BranchCutConfig` using the solver tolerances and constructor
            factors for path length and side offset.
        """
        layer = self.source_layer if branch_layer is None else int(branch_layer)
        return BranchCutConfig.from_k0(
            self.k0,
            branch_layer=layer,
            t_limit_factor=self.branch_cut_t_limit_factor,
            side_offset_factor=self.branch_cut_side_offset_factor,
            epsabs=self.epsabs,
            epsrel=self.epsrel,
            limit=max(80, self.limit // 2),
        )

    def branch_cut_integrals(
        self,
        rho: float,
        z_observer: float,
        z_source: float,
        branch_layer: int | None = None,
        config: BranchCutConfig | None = None,
    ) -> np.ndarray:
        r"""Diagnostic branch-cut contribution for one selected light line.

        This evaluates a two-side contour around a Sommerfeld branch cut in the
        vertical-wavenumber variable of ``branch_layer``.  It is mainly for
        validating branch-cut DCIM ideas and for identifying lateral-wave
        sensitivity.  It is not enabled automatically by ``direct`` or
        ``hybrid_dcim`` because the exact contour sign must be validated for the
        user's chosen Green-function normalization.
        """
        cfg = self.default_branch_cut_config(branch_layer) if config is None else config
        branch_k = np.sqrt(self.eps[cfg.branch_layer] + 0j) * self.k0
        orders = [0, 2, 0, 2, 1, 1, 0]
        return branch_cut_integral_kz(
            lambda q: self.bessel_free_kernels(q, z_observer, z_source),
            branch_wavenumber=branch_k,
            rho=rho,
            orders=orders,
            config=cfg,
        )

    def branch_cut_samples(
        self,
        z_observer: float,
        z_source: float,
        branch_layer: int | None = None,
        t_values: np.ndarray | None = None,
        side: int = +1,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Sample the seven kernels along a kz-parameterized branch-cut side.

        These samples can be passed to a matrix-pencil/DCIM routine to test
        whether the radiation/lateral-wave branch contribution can be compressed
        by exponentials on the selected path.
        """
        cfg = self.default_branch_cut_config(branch_layer)
        if t_values is None:
            t_values = np.linspace(-cfg.t_limit, cfg.t_limit, self.dcim_sample_count)
        branch_k = np.sqrt(self.eps[cfg.branch_layer] + 0j) * self.k0
        return branch_cut_samples_kz(
            lambda q: self.bessel_free_kernels(q, z_observer, z_source),
            branch_wavenumber=branch_k,
            t_values=np.asarray(t_values, dtype=float),
            side_offset=cfg.side_offset,
            side=side,
        )

    def singularity_breakpoints(
        self,
        config: PoleSearchConfig | None = None,
        include_poles: bool = True,
    ) -> list[float]:
        r"""Return positive real-q breakpoints for safer direct quadrature.

        The returned values split the unchanged Sommerfeld integral

        .. math::

           I_m(\rho)=\sum_a\int_{q_a}^{q_{a+1}}F_m(q)J_{\nu_m}(q\rho)\,dq

        at real light-line branch points and, optionally, projections of poles
        close enough to the real axis to degrade adaptive quadrature.

        Args:
            config: Optional pole-search configuration used when ``include_poles`` is true.
            include_poles: Include near-real pole projections in addition to branch points.
        """
        q_stop = np.inf if self.qmax is None else float(self.qmax)
        k0_abs = max(abs(self.k0), 1.0)
        guard = max(self.pole_search_branch_guard_factor * k0_abs, 1e-12)
        points: list[float] = []
        for bp in self.branch_points():
            if abs(np.imag(bp)) <= guard and np.real(bp) > 0 and np.real(bp) < q_stop:
                points.append(float(np.real(bp)))
        if include_poles:
            imag_tol = max(abs(self.pole_search_imag_max_factor) * abs(self.k0) * 10.0, guard)
            for pole in self.find_poles(config=config):
                if abs(np.imag(pole.q)) <= imag_tol and np.real(pole.q) > 0 and np.real(pole.q) < q_stop:
                    points.append(float(np.real(pole.q)))
        if not points:
            return []
        points = sorted(points)
        deduped: list[float] = []
        for point in points:
            if point <= 0:
                continue
            if deduped and abs(point - deduped[-1]) <= guard:
                deduped[-1] = 0.5 * (deduped[-1] + point)
            else:
                deduped.append(point)
        return deduped

    def compute_integrals_singularity_aware(
        self,
        rho: float,
        z_observer: float,
        z_source: float,
    ) -> np.ndarray:
        r"""Direct quadrature split at branch points and near-real pole projections.

        This mode does not change the Sommerfeld integral mathematically.  It
        improves robustness of ``quad_vec`` by exposing non-smooth locations as
        integration boundaries and records the detected branch points/poles in
        ``last_singularity_report``.  Explicit branch-cut and pole terms are
        available through :meth:`branch_cut_integrals` and
        :meth:`pole_contribution_integrals` for validated decompositions.
        """
        q_stop = np.inf if self.qmax is None else float(self.qmax)
        breakpoints = self.singularity_breakpoints()
        finite_breaks = [point for point in breakpoints if point > 0 and point < q_stop]
        edges = [0.0] + finite_breaks
        values = np.zeros(7, dtype=complex)
        for lower, upper in zip(edges[:-1], edges[1:]):
            if upper > lower:
                values += self.direct_integrals_over_range(rho, z_observer, z_source, lower, upper)
        last_lower = edges[-1]
        values += self.direct_integrals_over_range(rho, z_observer, z_source, last_lower, q_stop)
        poles = self.find_poles()
        self.last_singularity_report = {
            "method": "singularity_aware",
            "branch_points": self.branch_points(),
            "breakpoints": finite_breaks,
            "pole_count": len(poles),
            "poles": poles,
            "note": (
                "Real-axis quadrature was split at branch points and near-real pole projections. "
                "Explicit pole/branch-cut contour terms are diagnostic and must be validated "
                "before being added to production Green tensors."
            ),
        }
        return values

    def _branch_cut_dcim_layer_indices(self) -> list[int]:
        """Return unique layer indices used by the branch-cut DCIM contour."""
        layers = self.branch_cut_layers
        if layers is None or layers == "auto":
            candidates = [0, len(self.layers) - 1]
        elif layers == "source":
            candidates = [self.source_layer]
        elif isinstance(layers, str):
            candidates = [int(item.strip()) for item in layers.split(",") if item.strip()]
        else:
            candidates = [int(layer) for layer in layers]

        unique: list[int] = []
        for layer in candidates:
            if layer < 0:
                layer += len(self.layers)
            if not 0 <= layer < len(self.layers):
                raise ValueError("branch_cut_layers contains an invalid layer index.")
            if layer not in unique:
                unique.append(layer)
        return unique

    def _branch_cut_mapped_q(self, branch_wavenumber: complex, t: float, sign: float):
        """Map a vertical wavenumber sample to the physical q sheet."""
        kz = complex(float(t), sign * self.branch_cut_side_offset_factor * abs(self.k0))
        q = np.lib.scimath.sqrt(branch_wavenumber**2 - kz**2)
        if np.imag(q) < 0 or (abs(np.imag(q)) < 1e-18 and np.real(q) < 0):
            q = -q
        dqdt = 0.0 + 0.0j if abs(q) < 1e-30 else -kz / q
        return complex(q), dqdt

    @staticmethod
    def _integrate_finite_exponential_fit(fit, lower: float, upper: float) -> complex:
        """Integrate a finite-interval exponential fit in its sampling variable."""
        if fit.coefficients.size == 0:
            return 0.0 + 0.0j
        length = float(upper - lower)
        if length <= 0:
            return 0.0 + 0.0j
        values = []
        for coefficient, exponent in zip(fit.coefficients, fit.exponents):
            if abs(exponent) < 1e-14:
                values.append(coefficient * length)
            else:
                values.append(coefficient * (1.0 - np.exp(-exponent * length)) / exponent)
        return complex(np.sum(values))

    def _branch_cut_dcim_contribution(
        self,
        rho: float,
        z_observer: float,
        z_source: float,
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        r"""Fit Sommerfeld branch-cut integrands by exponentials and integrate them.

        The Chen-Chew-Jiang branch-cut DCIM deforms the Sommerfeld path to
        branch cuts in a vertical-wavenumber variable.  For each selected layer
        this method samples the two-side contour

        .. math::

           q_\pm(t)=\sqrt{k_l^2-(t\pm i\eta)^2},\qquad
           \frac{dq_\pm}{dt}=-\frac{t\pm i\eta}{q_\pm},

        fits each already-Hankel-weighted jump integrand with GPOF exponentials
        in the real variable ``t``, and analytically integrates those fitted
        exponentials over the finite sampled path.  The result is experimental
        and is validated before use by :meth:`compute_integrals_branch_cut_dcim`.
        """
        if self.branch_cut_sample_count < 2 * self.branch_cut_image_count + 1:
            raise ValueError(
                "branch_cut_sample_count must be at least 2 * branch_cut_image_count + 1."
            )

        orders = [0, 2, 0, 2, 1, 1, 0]
        t_limit = self.branch_cut_t_limit_factor * abs(self.k0)
        t_values = np.linspace(-t_limit, t_limit, self.branch_cut_sample_count)
        fit_axis = t_values + t_limit
        total = np.zeros(7, dtype=complex)
        reports: list[dict[str, object]] = []

        for layer in self._branch_cut_dcim_layer_indices():
            branch_k = np.sqrt(self.eps[layer] + 0j) * self.k0
            samples = np.zeros((self.branch_cut_sample_count, 7), dtype=complex)
            for idx, t_value in enumerate(t_values):
                plus_q, plus_dqdt = self._branch_cut_mapped_q(branch_k, t_value, +1.0)
                minus_q, minus_dqdt = self._branch_cut_mapped_q(branch_k, t_value, -1.0)
                plus_kernels = self.bessel_free_kernels(plus_q, z_observer, z_source)
                minus_kernels = self.bessel_free_kernels(minus_q, z_observer, z_source)
                plus_special = np.array(
                    [
                        hankel1(order, plus_q * rho)
                        if self.branch_cut_use_hankel
                        else jv(order, plus_q * rho)
                        for order in orders
                    ],
                    dtype=complex,
                )
                minus_special = np.array(
                    [
                        hankel1(order, minus_q * rho)
                        if self.branch_cut_use_hankel
                        else jv(order, minus_q * rho)
                        for order in orders
                    ],
                    dtype=complex,
                )
                samples[idx] = self.branch_cut_jump_sign * (
                    plus_kernels * plus_special * plus_dqdt
                    - minus_kernels * minus_special * minus_dqdt
                )

            layer_values = np.zeros(7, dtype=complex)
            residuals: list[float] = []
            for kernel_index in range(7):
                fit = fit_exponentials(
                    fit_axis,
                    samples[:, kernel_index],
                    self.branch_cut_image_count,
                    q_origin=0.0,
                )
                residuals.append(np.inf if fit.relative_residual is None else fit.relative_residual)
                layer_values[kernel_index] = self._integrate_finite_exponential_fit(
                    fit,
                    lower=0.0,
                    upper=2.0 * t_limit,
                )

            layer_values *= self.branch_cut_prefactor
            total += layer_values
            reports.append(
                {
                    "layer": layer,
                    "branch_wavenumber": branch_k,
                    "t_limit": t_limit,
                    "fit_residuals": residuals,
                    "values": layer_values,
                }
            )
        return total, reports

    def compute_integrals_branch_cut_dcim(
        self,
        rho: float,
        z_observer: float,
        z_source: float,
    ) -> np.ndarray:
        r"""Experimental branch-cut DCIM with validation/fallback.

        This implements the Chen-Chew-Jiang idea of decomposing the far-field
        Sommerfeld integral into fitted radiation-mode branch cuts plus guided
        pole residues.  Because this codebase's production kernels are
        half-line ``J_nu`` integrals while the paper uses Hankel-contour scalar
        Green functions, the approximation is accepted only after comparison to
        the ``singularity_aware`` split-quadrature reference unless validation is
        explicitly disabled.
        """
        branch_values, branch_reports = self._branch_cut_dcim_contribution(
            rho,
            z_observer,
            z_source,
        )
        poles = self.find_poles() if self.branch_cut_include_poles else []
        pole_values = (
            self.pole_contribution_integrals(rho, z_observer, z_source, poles=poles)
            if self.branch_cut_include_poles
            else np.zeros(7, dtype=complex)
        )
        approximation = branch_values + pole_values

        reference = None
        relative_errors = None
        max_relative_error = None
        accepted = True
        reason = "validation disabled"
        result = approximation
        if self.branch_cut_validate:
            reference = self.compute_integrals_singularity_aware(rho, z_observer, z_source)
            scale = np.maximum(np.abs(reference), 1e-30)
            relative_errors = np.abs(approximation - reference) / scale
            finite_errors = relative_errors[np.isfinite(relative_errors)]
            max_relative_error = float(np.max(finite_errors)) if finite_errors.size else np.inf
            accepted = max_relative_error <= self.branch_cut_validation_rtol
            if accepted:
                reason = (
                    "branch-cut DCIM validation accepted: max relative error "
                    f"{max_relative_error:.3e}"
                )
            else:
                reason = (
                    "branch-cut DCIM validation failed: max relative error "
                    f"{max_relative_error:.3e} exceeds {self.branch_cut_validation_rtol:.3e}"
                )
                logger.warning(
                    "Branch-cut DCIM rejected; {}. {}",
                    reason,
                    "Falling back to singularity_aware."
                    if self.branch_cut_fallback_to_singularity_aware
                    else "Returning rejected approximation.",
                )
                if self.branch_cut_fallback_to_singularity_aware:
                    result = reference

        self.last_branch_cut_dcim_report = {
            "method": "branch_cut_dcim",
            "accepted": accepted,
            "reason": reason,
            "branch_layers": self._branch_cut_dcim_layer_indices(),
            "branch_values": branch_values,
            "branch_reports": branch_reports,
            "pole_count": len(poles),
            "poles": poles,
            "pole_values": pole_values,
            "approximation": approximation,
            "reference": reference,
            "relative_errors": relative_errors,
            "max_relative_error": max_relative_error,
            "fallback_to_singularity_aware": self.branch_cut_fallback_to_singularity_aware,
        }
        return result

    def complex_quad(
        self,
        func: Callable[[float], Union[complex, np.ndarray]],
        a: float,
        qmax: Optional[float] = None,
        epsabs: Optional[float] = None,
        epsrel: Optional[float] = None,
        limit: Optional[int] = None,
        split_propagating: Optional[bool] = None,
    ) -> Union[complex, np.ndarray]:
        r"""Vector-valued Sommerfeld quadrature helper.

        Computes

        .. math::

           \int_a^{q_{\max}} f(q)\,dq

        with ``scipy.integrate.quad_vec``.  When requested, the integral is split
        at the real source-layer light line to separate propagating and
        evanescent parts.
        """
        qmax_eff = self.qmax if qmax is None else qmax
        epsabs_eff = self.epsabs if epsabs is None else epsabs
        epsrel_eff = self.epsrel if epsrel is None else epsrel
        limit_eff = self.limit if limit is None else limit
        split_eff = self.split_propagating if split_propagating is None else split_propagating
        upper = np.inf if qmax_eff is None else qmax_eff

        def _segment(lower, upper_bound):
            result, _ = quad_vec(
                func,
                lower,
                upper_bound,
                epsabs=epsabs_eff,
                epsrel=epsrel_eff,
                limit=limit_eff,
            )
            return result

        light_line = np.real(np.sqrt(self.eps[self.source_layer] + 0j) * self.k0)
        if split_eff and np.isfinite(light_line) and upper > light_line > a:
            return _segment(a, light_line) + _segment(light_line, upper)
        return _segment(a, upper)

    def direct_integrals_over_range(
        self,
        rho: float,
        z_observer: float,
        z_source: float,
        lower: float,
        upper: float,
        epsabs: Optional[float] = None,
        epsrel: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> np.ndarray:
        r"""Directly integrate the seven scalar Sommerfeld integrals on a finite interval.

        For Bessel orders ``nu = [0,2,0,2,1,1,0]`` and Bessel-free kernels
        ``F_m(q)``, this returns

        .. math::

           I_m(\rho; q_a,q_b)=\int_{q_a}^{q_b}F_m(q)J_{\nu_m}(q\rho)\,dq.

        This finite-interval form is used by both the validated direct solver
        and the hybrid DCIM finite-tail validation.
        """
        if upper <= lower:
            return np.zeros(7, dtype=complex)

        def integrand(q: float) -> np.ndarray:
            kernels = self.bessel_free_kernels(q, z_observer, z_source)
            j0 = jv(0, q * rho)
            j1 = jv(1, q * rho)
            j2 = jv(2, q * rho)
            return kernels * np.array([j0, j2, j0, j2, j1, j1, j0], dtype=complex)

        result = self.complex_quad(
            integrand,
            a=lower,
            qmax=upper,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            split_propagating=False,
        )
        return np.asarray(result, dtype=complex)

    def bessel_free_kernels(self, q: float, z_observer: float, z_source: float) -> np.ndarray:
        r"""Seven Bessel-free scalar kernels before angular integration.

        The returned vector is

        .. math::

           \left[\frac{C_s q e^{i\beta_jd_j}}{2\beta_j},
           \frac{C_s q e^{i\beta_jd_j}}{2\beta_j},
           \frac{C_p^-q\beta_j e^{i\beta_jd_j}}{2k_j^2},
           \frac{C_p^-q\beta_j e^{i\beta_jd_j}}{2k_j^2},
           \frac{iS_p^+q^2 e^{i\beta_jd_j}}{k_j^2},
           \frac{iS_p^-q^2 e^{i\beta_jd_j}}{k_j^2},
           \frac{C_p^+q^3 e^{i\beta_jd_j}}{\beta_jk_j^2}\right].

        Multiplication by ``[J0,J2,J0,J2,J1,J1,J0]`` gives the seven
        Sommerfeld integrands.
        """
        layer_k_sq = self.eps[self.source_layer] * self.k0**2
        d_j = self._source_layer_thickness()
        beta_j = self._kz(self.source_layer, q)
        c_s, c_p_plus, c_p_minus, s_p_plus, s_p_minus = self.amplitude_coefficients(
            q,
            z_observer,
            z_source,
        )
        exp_d = np.exp(1j * beta_j * d_j)
        return np.array(
            [
                c_s * q * exp_d / (2 * beta_j),
                c_s * q * exp_d / (2 * beta_j),
                c_p_minus * q * beta_j * exp_d / (2 * layer_k_sq),
                c_p_minus * q * beta_j * exp_d / (2 * layer_k_sq),
                1j * s_p_plus * q**2 * exp_d / layer_k_sq,
                1j * s_p_minus * q**2 * exp_d / layer_k_sq,
                c_p_plus * q**3 * exp_d / (beta_j * layer_k_sq),
            ],
            dtype=complex,
        )

    def compute_integrals(self, rho: float, z_observer: float, z_source: float) -> np.ndarray:
        r"""Compute the seven Sommerfeld integrals for the selected method.

        Direct mode evaluates

        .. math::

           I_m(\rho)=\int_0^{q_{\max}}F_m(q)J_{\nu_m}(q\rho)\,dq.

        ``dcim`` performs the legacy whole-domain exponential fit.  ``hybrid_dcim``
        evaluates the low-q interval directly and fits only the evanescent tail.
        """
        if self.integration_method == "dcim":
            return self.compute_integrals_dcim(rho, z_observer, z_source)
        if self.integration_method == "hybrid_dcim":
            return self.compute_integrals_hybrid_dcim(rho, z_observer, z_source)
        if self.integration_method == "fixed_grid":
            return self.compute_integrals_fixed_grid(rho, z_observer, z_source)
        if self.integration_method == "singularity_aware":
            return self.compute_integrals_singularity_aware(rho, z_observer, z_source)
        if self.integration_method == "branch_cut_dcim":
            return self.compute_integrals_branch_cut_dcim(rho, z_observer, z_source)

        def integrand(q: float) -> np.ndarray:
            kernels = self.bessel_free_kernels(q, z_observer, z_source)
            j0 = jv(0, q * rho)
            j1 = jv(1, q * rho)
            j2 = jv(2, q * rho)
            return kernels * np.array([j0, j2, j0, j2, j1, j1, j0], dtype=complex)

        result = self.complex_quad(integrand, a=0)
        return np.asarray(result, dtype=complex)

    def _fixed_grid_q_stop(self) -> float:
        if self.qmax is None:
            raise ValueError("fixed_grid integration requires a finite simulation.integration.qmax.")
        return float(self.qmax)

    def _sample_fixed_grid_kernels(self, z_observer: float, z_source: float):
        if self.fixed_grid_points < 2:
            raise ValueError("fixed_grid_points must be at least 2.")
        q_values = np.linspace(0.0, self._fixed_grid_q_stop(), self.fixed_grid_points)
        kernel_samples = np.array(
            [self.bessel_free_kernels(q, z_observer, z_source) for q in q_values],
            dtype=complex,
        )
        return q_values, kernel_samples

    def compute_integrals_fixed_grid(
        self,
        rho: float,
        z_observer: float,
        z_source: float,
    ) -> np.ndarray:
        return self.compute_integrals_fixed_grid_for_rhos(
            np.array([rho], dtype=float),
            z_observer,
            z_source,
        )[0]

    def compute_integrals_fixed_grid_for_rhos(
        self,
        rhos: np.ndarray,
        z_observer: float,
        z_source: float,
    ) -> np.ndarray:
        q_values, kernel_samples = self._sample_fixed_grid_kernels(z_observer, z_source)
        q_rho = q_values[:, None] * np.asarray(rhos, dtype=float)[None, :]
        bessel_factors = np.stack(
            [
                jv(0, q_rho),
                jv(2, q_rho),
                jv(0, q_rho),
                jv(2, q_rho),
                jv(1, q_rho),
                jv(1, q_rho),
                jv(0, q_rho),
            ],
            axis=2,
        )
        integrand = kernel_samples[:, None, :] * bessel_factors
        return np.trapezoid(integrand, x=q_values, axis=0)

    def _hybrid_default_direct_q_stop(self) -> float:
        if self.hybrid_direct_q_stop is not None:
            return float(self.hybrid_direct_q_stop)
        source_light_line = abs(np.sqrt(self.eps[self.source_layer] + 0j) * self.k0)
        branch_scale = float(np.max(np.abs(self.branch_points())))
        return max(6.0 * branch_scale, 2.0 * source_light_line, 1e6)

    def _hybrid_default_tail_q_stop(self, direct_stop: float) -> float:
        if self.hybrid_tail_q_stop is not None:
            return float(self.hybrid_tail_q_stop)
        if self.dcim_q_stop is not None:
            return float(self.dcim_q_stop)
        if self.qmax is not None:
            return float(self.qmax)
        return 5.0 * direct_stop

    def compute_integrals_hybrid_dcim(
        self,
        rho: float,
        z_observer: float,
        z_source: float,
    ) -> np.ndarray:
        r"""Hybrid direct/GPOF-DCIM evaluation of the Sommerfeld integrals.

        The integral is split as

        .. math::

           I_m = \int_0^{q_c}F_m(q)J_{\nu_m}(q\rho)dq
           + \int_{q_c}^{\infty}F_m(q)J_{\nu_m}(q\rho)dq.

        The first term is direct quadrature.  On the tail, samples on
        ``[q_c,q_f]`` are fit to shifted images

        .. math::

           F_m(q)\approx\sum_{n=1}^{N_i}a_{mn}e^{-s_{mn}(q-q_c)}.

        The fitted tail is accepted only if its finite interval integral over
        ``[q_c,q_f]`` agrees with direct finite-tail quadrature within
        ``hybrid_validation_rtol``; otherwise the method records diagnostics and
        falls back to direct quadrature when configured to do so.
        """
        direct_stop = self._hybrid_default_direct_q_stop()
        tail_stop = self._hybrid_default_tail_q_stop(direct_stop)
        if tail_stop <= direct_stop:
            raise ValueError("hybrid_tail_q_stop/dcim_q_stop/qmax must exceed hybrid_direct_q_stop.")
        if self.hybrid_sample_count < 2 * self.hybrid_image_count + 1:
            raise ValueError("hybrid_sample_count must be at least 2 * hybrid_image_count + 1.")

        low_q = self.direct_integrals_over_range(
            rho,
            z_observer,
            z_source,
            lower=0.0,
            upper=direct_stop,
        )
        q_values = np.linspace(direct_stop, tail_stop, self.hybrid_sample_count)
        kernel_samples = np.array(
            [self.bessel_free_kernels(q, z_observer, z_source) for q in q_values],
            dtype=complex,
        )
        bessel_orders = [0, 2, 0, 2, 1, 1, 0]
        fits = []
        tail_values = np.zeros(7, dtype=complex)
        fit_residuals = []
        for kernel_index, order in enumerate(bessel_orders):
            fit = fit_exponentials(
                q_values,
                kernel_samples[:, kernel_index],
                self.hybrid_image_count,
                q_origin=float(q_values[0]),
            )
            fits.append(fit)
            fit_residuals.append(np.inf if fit.relative_residual is None else fit.relative_residual)
            tail_values[kernel_index] = integrate_complex_images_range(
                fit,
                rho,
                order,
                lower=direct_stop,
                upper=np.inf,
                epsabs=self.epsabs,
                epsrel=self.epsrel,
                limit=max(80, self.limit // 2),
            )

        report: dict[str, object] = {
            "method": "hybrid_dcim",
            "direct_q_stop": direct_stop,
            "tail_fit_q_stop": tail_stop,
            "fit_residuals": fit_residuals,
            "accepted": True,
            "reason": "tail fit accepted without finite-tail validation",
        }

        if self.hybrid_validate_tail:
            reference_tail = self.direct_integrals_over_range(
                rho,
                z_observer,
                z_source,
                lower=direct_stop,
                upper=tail_stop,
                epsabs=max(self.epsabs, 1e-7),
                epsrel=max(self.epsrel, 1e-7),
                limit=max(80, self.limit // 2),
            )
            fitted_tail = np.zeros(7, dtype=complex)
            for kernel_index, order in enumerate(bessel_orders):
                fitted_tail[kernel_index] = integrate_complex_images_range(
                    fits[kernel_index],
                    rho,
                    order,
                    lower=direct_stop,
                    upper=tail_stop,
                    epsabs=max(self.epsabs, 1e-7),
                    epsrel=max(self.epsrel, 1e-7),
                    limit=max(80, self.limit // 2),
                )
            scale = np.maximum(np.abs(reference_tail), 1e-30)
            tail_relative_errors = np.abs(fitted_tail - reference_tail) / scale
            finite_errors = tail_relative_errors[np.isfinite(tail_relative_errors)]
            max_tail_error = float(np.max(finite_errors)) if finite_errors.size else np.inf
            report.update(
                {
                    "finite_tail_reference": reference_tail,
                    "finite_tail_fit": fitted_tail,
                    "tail_relative_errors": tail_relative_errors,
                    "max_tail_relative_error": max_tail_error,
                }
            )
            if max_tail_error > self.hybrid_validation_rtol:
                report["accepted"] = False
                report["reason"] = (
                    "finite-tail validation failed: max relative error "
                    f"{max_tail_error:.3e} exceeds {self.hybrid_validation_rtol:.3e}"
                )
                logger.warning(
                    "Hybrid DCIM rejected; {}. {}",
                    report["reason"],
                    "Falling back to direct quadrature." if self.hybrid_fallback_to_direct else "",
                )
                self.last_hybrid_dcim_report = report
                if self.hybrid_fallback_to_direct:
                    return self.direct_integrals_over_range(
                        rho,
                        z_observer,
                        z_source,
                        lower=0.0,
                        upper=self.qmax if self.qmax is not None else tail_stop,
                    )
            else:
                report["reason"] = (
                    "finite-tail validation accepted: max relative error "
                    f"{max_tail_error:.3e}"
                )

        self.last_hybrid_dcim_report = report
        return low_q + tail_values

    def compute_integrals_dcim(self, rho: float, z_observer: float, z_source: float) -> np.ndarray:
        r"""Legacy whole-domain complex-image approximation.

        This path fits

        .. math::

           F_m(q)\approx\sum_n a_{mn}e^{-s_{mn}q}, \quad 0\le q\le q_f,

        then applies analytic Laplace-Bessel transforms on ``[0,infinity)``.
        It is retained as a diagnostic because metal multilayer kernels usually
        require the split-tail treatment implemented by ``hybrid_dcim``.
        """
        logger.warning(
            "The DCIM path is experimental and is not validated for metal multilayer "
            "Sommerfeld kernels. Use integration_method='direct' as the reference and "
            "accept DCIM results only after stack-specific validation."
        )
        q_stop = self.dcim_q_stop if self.dcim_q_stop is not None else self.qmax
        if q_stop is None:
            raise ValueError("DCIM requires a finite dcim_q_stop or qmax.")
        q_values = np.linspace(self.dcim_q_start, float(q_stop), self.dcim_sample_count)
        kernel_samples = np.array(
            [self.bessel_free_kernels(q, z_observer, z_source) for q in q_values],
            dtype=complex,
        )
        bessel_orders = [0, 2, 0, 2, 1, 1, 0]
        values = np.zeros(7, dtype=complex)
        for kernel_index, order in enumerate(bessel_orders):
            fit = fit_exponentials(
                q_values,
                kernel_samples[:, kernel_index],
                self.dcim_image_count,
            )
            values[kernel_index] = integrate_complex_images(fit, rho, order)
        return values

    def vacuum_component(self, x: float, y: float, z_observer: float, z_source: float):
        r"""Homogeneous Green tensor inside the source layer.

        .. math::

           \mathbf G_0^{(j)}(\mathbf R)=\frac{e^{ik_jR}}{4\pi R k_j^2}
           \left[(\mathbf I-\hat{R}\hat{R})k_j^2
           +\frac{3\hat{R}\hat{R}-\mathbf I}{R^2}
           +\frac{i k_j(\mathbf I-3\hat{R}\hat{R})}{R}\right].

        At coincident points the usual imaginary self term ``i k_j/(6 pi) I``
        is returned.
        """
        k_layer = np.sqrt(self.eps[self.source_layer] + 0j) * self.k0
        r_vec = np.array([x, y, z_observer - z_source], dtype=float)
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-12:
            return 1j * k_layer / (6 * np.pi) * np.eye(3, dtype=complex)

        unit_r = r_vec / r_mag
        identity = np.eye(3)
        outer = np.outer(unit_r, unit_r)
        term1 = (identity - outer) * k_layer**2
        term2 = (3 * outer - identity) / r_mag**2
        term3 = (identity - 3 * outer) * (1j * k_layer / r_mag)
        prefactor = np.exp(1j * k_layer * r_mag) / (4 * np.pi * r_mag * k_layer**2)
        return prefactor * (term1 + term2 + term3)

    def scatter_component(self, x: float, y: float, z_observer: float, z_source: float):
        r"""Scattered Green tensor assembled from TE and TM angular spectra.

        .. math::

           \mathbf G_{sc}^{(j)}=\frac{i}{4\pi}
           \left[\mathbf M_s(I_1,I_2)+\mathbf M_p(I_3,\ldots,I_6)\right].
        """
        rho = np.sqrt(x**2 + y**2)
        integrals = self.compute_integrals(rho, z_observer, z_source)
        return self.scatter_component_from_integrals(x, y, integrals)

    def scatter_component_from_integrals(self, x: float, y: float, integrals: np.ndarray):
        ms = self.scattering_s_component(x, y, integrals)
        mp = self.scattering_p_component(x, y, integrals)
        return 1j / (4 * np.pi) * (ms + mp)

    @staticmethod
    def _phi(x: float, y: float) -> float:
        return 0.0 if x == 0 and y == 0 else float(np.arctan2(y, x))

    def scattering_s_component(self, x: float, y: float, integrals: np.ndarray):
        r"""TE tensor block after angular integration.

        .. math::

           \mathbf M_s=\begin{pmatrix}
           I_1+\cos2\phi I_2 & \sin2\phi I_2 & 0\\
           \sin2\phi I_2 & I_1-\cos2\phi I_2 & 0\\
           0 & 0 & 0
           \end{pmatrix}.
        """
        phi = self._phi(x, y)
        i1, i2 = integrals[0], integrals[1]
        return np.array(
            [
                [i1 + np.cos(2 * phi) * i2, np.sin(2 * phi) * i2, 0],
                [np.sin(2 * phi) * i2, i1 - np.cos(2 * phi) * i2, 0],
                [0, 0, 0],
            ],
            dtype=complex,
        )

    def scattering_p_component(self, x: float, y: float, integrals: np.ndarray):
        r"""TM tensor block after angular integration.

        .. math::

           \mathbf M_p=\begin{pmatrix}
           -I_3+\cos2\phi I_4 & \sin2\phi I_4 & -\cos\phi I_5^+\\
           \sin2\phi I_4 & -I_3-\cos2\phi I_4 & -\sin\phi I_5^+\\
           \cos\phi I_5^- & \sin\phi I_5^- & I_6
           \end{pmatrix}.

        In an N-layer stack, ``I5+`` and ``I5-`` are generally different because
        the TM amplitudes ``S_p_plus`` and ``S_p_minus`` are different.
        """
        phi = self._phi(x, y)
        i3, i4, i5_plus, i5_minus, i6 = integrals[2:]
        return np.array(
            [
                [
                    -i3 + np.cos(2 * phi) * i4,
                    np.sin(2 * phi) * i4,
                    -np.cos(phi) * i5_plus,
                ],
                [
                    np.sin(2 * phi) * i4,
                    -i3 - np.cos(2 * phi) * i4,
                    -np.sin(phi) * i5_plus,
                ],
                [np.cos(phi) * i5_minus, np.sin(phi) * i5_minus, i6],
            ],
            dtype=complex,
        )

    def calculate_total_Green_function(
        self,
        x: float,
        y: float,
        z_observer: float,
        z_source: float,
    ):
        r"""Total source-layer dyadic Green tensor.

        .. math::

           \mathbf G = \mathbf G_0^{(j)} + \mathbf G_{sc}^{(j)}.
        """
        logger.debug(
            "N-layer Green tensor at omega={} eV, source_layer={}",
            self.omega * hbar / eV_to_J,
            self.source_layer,
        )
        return self.vacuum_component(x, y, z_observer, z_source) + self.scatter_component(
            x,
            y,
            z_observer,
            z_source,
        )

    def calculate_total_Green_functions_for_x_values(
        self,
        x_values: np.ndarray,
        y: float,
        z_observer: float,
        z_source: float,
    ):
        if self.integration_method != "fixed_grid":
            return np.array(
                [
                    self.calculate_total_Green_function(x, y, z_observer, z_source)
                    for x in x_values
                ],
                dtype=complex,
            )

        rhos = np.sqrt(np.asarray(x_values, dtype=float) ** 2 + y**2)
        integrals_by_rho = self.compute_integrals_fixed_grid_for_rhos(
            rhos,
            z_observer,
            z_source,
        )
        values = np.zeros((len(x_values), 3, 3), dtype=complex)
        for index, (x, integrals) in enumerate(zip(x_values, integrals_by_rho)):
            values[index] = self.vacuum_component(
                x,
                y,
                z_observer,
                z_source,
            ) + self.scatter_component_from_integrals(x, y, integrals)
        return values
