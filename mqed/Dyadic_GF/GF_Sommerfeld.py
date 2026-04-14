from typing import Callable, Optional, Union

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import jv  # Bessel function of the first kind
from loguru import logger
from mqed.utils.SI_unit import c, hbar, eV_to_J


class Greens_function_analytical:
    r"""
    Dyadic Green's function for a planar interface with cylindrical symmetry.

    Architecture Note (N-Layer Extensibility)
    ------------------------------------------
    The reflection coefficients and z-wavevectors are computed via dedicated
    methods rather than inline lambdas.  This design separates the *integral
    machinery* (Sommerfeld integrals, quadrature, tensor assembly — shared by
    all planar geometries) from the *layer physics* (Fresnel coefficients,
    phase factors — specific to the number of layers).

    To extend to N layers, subclass this and override:

    * :meth:`_kz` — return kz in any layer
    * :meth:`_rs` / :meth:`_rp` — return generalized reflection coefficients
      (e.g., recursive Tomas 1995 or transfer-matrix formalism)

    The Sommerfeld integrals in :meth:`compute_integrals` call these methods
    and remain unchanged.

    s- and p-polarized Fresnel coefficients (two-layer):

    .. math::

       r_s(q) = \frac{K_{z,0}(q) - K_{z,1}(q)}{K_{z,0}(q) + K_{z,1}(q)}

    .. math::

       r_p(q) = \frac{\epsilon_1 K_{z,0}(q) - \epsilon_0 K_{z,1}(q)}{\epsilon_1 K_{z,0}(q) + \epsilon_0 K_{z,1}(q)}

    where

    .. math::

       K_{z,i}(q) = \sqrt{\epsilon_i k_0^2 - q^2}.

    Total field is vacuum plus reflected part:

    .. math::

       \overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega) = \overline{\overline{\mathbf{G}}}_0(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega) + \overline{\overline{\mathbf{G}}}_{\text{refl}}^{(i)}(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega).
    """

    def __init__(
        self,
        metal_epsi: complex,
        omega: float,
        eps_0: float = 1.0,
        qmax: Optional[float] = None,
        epsabs: float = 1e-10,
        epsrel: float = 1e-10,
        limit: int = 400,
        split_propagating: bool = True,
    ):
        """
        Initializes the two-layer Green's function calculator.

        The two media are:
          * Layer 0 (upper half-space, z > 0): permittivity ``eps_0``
            — typically vacuum (1.0).
          * Layer 1 (lower half-space, z < 0): permittivity ``metal_epsi``
            — typically a dispersive metal like Ag or Au.

        Args:
            metal_epsi: Complex permittivity of the metal (layer 1).
            omega:      Angular frequency [rad/s].
            eps_0:      Permittivity of the upper half-space (layer 0).
                        Default 1.0 (vacuum).
            qmax:       Upper integration limit for Sommerfeld integrals.
                        ``None`` → integrate to infinity.
            epsabs:     Absolute error tolerance for ``quad_vec``.
            epsrel:     Relative error tolerance for ``quad_vec``.
            limit:      Maximum number of adaptive subintervals.
            split_propagating: If True, split the integral at q = k0
                        (propagating/evanescent boundary) for better
                        numerical accuracy.
        """
        self.metal_epsi = metal_epsi
        self.omega = omega
        self.eps_0 = eps_0
        self.c = c  # speed of light in SI units
        self.k0 = self.omega / self.c  # free-space wavenumber

        # Quadrature settings (passed through to complex_quad)
        self.qmax = qmax
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = limit
        self.split_propagating = split_propagating
    
    # ------------------------------------------------------------------
    #  Layer physics: kz, reflection coefficients
    #
    #  These are the ONLY methods that encode the number/arrangement of
    #  layers.  Everything below (Sommerfeld integrals, tensor assembly)
    #  is geometry-agnostic and calls these through the public API.
    #
    #  To add N-layer support, subclass and override _kz, _rs, _rp with
    #  recursive Fresnel / transfer-matrix implementations.  The integral
    #  machinery will "just work".
    # ------------------------------------------------------------------

    @staticmethod
    def _beta_phys(eps: complex, k0: float, q):
        r"""
        Physical z-wavevector in a medium with permittivity ``eps``.

        .. math::

           K_{z}(q) = \sqrt{\epsilon\, k_0^2 - q^2}

        The branch is chosen so that Im(K_z) >= 0 (evanescent waves decay
        away from the interface) and, when Im(K_z) ≈ 0, Re(K_z) >= 0
        (propagating waves travel upward).

        A tiny imaginary offset ``+i·10⁻¹²`` is added under the square root
        to lift the branch cut off the real axis — this is the standard
        ``i0⁺`` prescription.

        Args:
            eps: Complex permittivity of the medium.
            k0:  Free-space wavenumber ω/c [1/m].
            q:   In-plane (transverse) wavevector component.

        Returns:
            Complex Kz value(s) with correct branch.

        Note:
            Made a ``@staticmethod`` so it can be called without an instance
            (useful in parallel workers or standalone utilities).  The k0
            argument replaces the previous ``self.k0`` access.
        """
        b = np.lib.scimath.sqrt(eps * k0**2 - q**2 + 1j * 1e-12)
        # Enforce Im(b) >= 0; if Im ≈ 0, enforce Re(b) >= 0.
        if np.ndim(b):
            flip = (np.imag(b) < 0) | (
                (np.abs(np.imag(b)) < 1e-18) & (np.real(b) < 0)
            )
            b[flip] = -b[flip]
            return b
        return (
            -b
            if (np.imag(b) < 0 or (abs(np.imag(b)) < 1e-18 and np.real(b) < 0))
            else b
        )

    def _kz(self, layer: int, q):
        r"""
        z-wavevector component in a given layer.

        For the current two-layer geometry:
          * layer 0 → upper half-space (permittivity ``self.eps_0``)
          * layer 1 → lower half-space (permittivity ``self.metal_epsi``)

        Override in an N-layer subclass to index into an array of
        permittivities: ``self.eps[layer]``.

        Args:
            layer: Layer index (0 = upper, 1 = metal).
            q:     In-plane wavevector.

        Returns:
            Complex Kz in the requested layer.
        """
        eps = self.eps_0 if layer == 0 else self.metal_epsi
        return self._beta_phys(eps, self.k0, q)

    def _rs(self, q):
        r"""
        s-polarized (TE) Fresnel reflection coefficient at the interface.

        Two-layer formula:

        .. math::

           r_s(q) = \frac{K_{z,0} - K_{z,1}}{K_{z,0} + K_{z,1}}

        For N-layer extension, replace with the generalized reflection
        coefficient computed via recursive Tomas (1995) relations or
        transfer-matrix method.

        Args:
            q: In-plane wavevector (scalar or array).

        Returns:
            Complex reflection coefficient.
        """
        kz0 = self._kz(0, q)
        kz1 = self._kz(1, q)
        return (kz0 - kz1) / (kz0 + kz1)

    def _rp(self, q):
        r"""
        p-polarized (TM) Fresnel reflection coefficient at the interface.

        Two-layer formula:

        .. math::

           r_p(q) = \frac{\epsilon_1 K_{z,0} - \epsilon_0 K_{z,1}}
                         {\epsilon_1 K_{z,0} + \epsilon_0 K_{z,1}}

        For N-layer extension, replace with the generalized reflection
        coefficient.

        Args:
            q: In-plane wavevector (scalar or array).

        Returns:
            Complex reflection coefficient.
        """
        kz0 = self._kz(0, q)
        kz1 = self._kz(1, q)
        return (
            (self.metal_epsi * kz0 - self.eps_0 * kz1)
            / (self.metal_epsi * kz0 + self.eps_0 * kz1)
        )
    
    

    def complex_quad(self,
                    func: Callable[[float], Union[complex, np.ndarray]],
                    a: float,
                    qmax: Optional[float] = None,
                    epsabs: Optional[float] = None,
                    epsrel: Optional[float] = None,
                    limit: Optional[int] = None,
                    split_propagating: Optional[bool] = None) -> Union[complex, np.ndarray]:
        """
        Integrates a complex-valued function using scipy's quad_vec function.
        Args:
            func (callable): The complex-valued function to integrate.
            a (float): The lower limit of integration.
            qmax (float): Maximum q value for integration. Defaults to np.inf.
            epsabs (float, optional): Absolute error tolerance. Defaults to 1e-10.
            epsrel (float, optional): Relative error tolerance. Defaults to 1e-10
            limit (int, optional): Maximum number of subintervals. Defaults to 400.
            returns:
                complex: The result of the integration."""
        
        qmax_eff = self.qmax if qmax is None else qmax
        epsabs_eff = self.epsabs if epsabs is None else epsabs
        epsrel_eff = self.epsrel if epsrel is None else epsrel
        limit_eff = self.limit if limit is None else limit
        split_eff = self.split_propagating if split_propagating is None else split_propagating

        upper = np.inf if qmax_eff is None else qmax_eff

        def _segment(lower, upper_bound):
            res, _ = quad_vec(func, lower, upper_bound, epsabs=epsabs_eff, epsrel=epsrel_eff, limit=limit_eff)
            return res

        if split_eff and upper > self.k0 and self.k0 > a:
            parts = []
            if self.k0 > a:
                parts.append(_segment(a, self.k0))
            if upper > self.k0:
                parts.append(_segment(self.k0, upper))
            return sum(parts)

        return _segment(a, upper)

    
    def vacuum_component(self,
        x: float,
        y: float,
        z1: float,
        z2: float,):
        r"""
        Vacuum dyadic Green's function.

        .. math::

           \overline{\overline{\mathbf{G}}}_0(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega) = \frac{e^{ik_0 R_{\alpha\beta}}}{4\pi R_{\alpha\beta}}\Big[ \left(\overline{\overline{\mathbf{I}}}_3-\mathbf{e}_\mathrm{R}\mathbf{e}_\mathrm{R}\right) + \left(3\mathbf{e}_\mathrm{R}\mathbf{e}_\mathrm{R}-\overline{\overline{\mathbf{I}}}_3\right)\left(\frac{1}{(k_0 R_{\alpha\beta})^{2}}-\frac{i}{k_0 R_{\alpha\beta}}\right)\Big].

        Args:
            x: x-distance between the two points.
            y: y-distance between the two points.
            z1: z-coordinate of the first point.
            z2: z-coordinate of the second point.
        Returns:
            np.ndarray: 3x3 vacuum Green's tensor.
        """

        R_vec = np.array([x,y,z1-z2])
        R_mag = np.linalg.norm(R_vec)
        if R_mag < 1e-12: #near field limit
            return 1j * self.k0 / (6 * np.pi) * np.eye(3)

        unit_R = R_vec / R_mag
        I3 = np.eye(3) # 3x3 identity matrix
        R_outer_R = np.outer(unit_R, unit_R)

        term1 = (I3 - R_outer_R) * self.k0**2
        term2 = (3 * R_outer_R - I3) / R_mag**2
        term3 = (I3 - 3* R_outer_R) * (1j * self.k0 / R_mag)
        
        prefactor = (np.exp(1j * self.k0 * R_mag)) / (4* np.pi * R_mag *self.k0**2) #previous mistake: missing k0^2 in the denominator

        # Vaccum Green's function
        G0 = prefactor * (term1 + term2 + term3)
        return G0
    
    def scatter_component(self,
        x: float,
        y: float,
        z1: float,
        z2: float,):
        r"""
        Scattering part of the Green's function.

        .. math::

           \overline{\overline{\mathbf{G}}}_{\text{refl}}^{(i)}(\rho, \phi, z, z', \omega) = \int_{0}^{+\infty} \frac{i\,dk_{\rho}}{4\pi} \Big[ R_s\,\overline{\overline{\mathbf{M}}}_s + R_p\,\overline{\overline{\mathbf{M}}}_p \Big] e^{iK_{z,i}(k_{\rho}, \omega)(z+z')}.

        Args:
            x: x-distance between the two points.
            y: y-distance between the two points.
            z1: z-coordinate of the first point.
            z2: z-coordinate of the second point.
        """
        rho = np.sqrt(x**2 + y**2)
        integrals = self.compute_integrals(rho, z1, z2)

        Ms = self.scattering_s_component(x, y, z1, z2, integrals=integrals)
        Mp = self.scattering_p_component(x, y, z1, z2, integrals=integrals)
        prefactor = 1j / (4 * np.pi)
        G_scatter = prefactor * (Ms + Mp)
        return G_scatter

    
    def compute_integrals(self, rho: float, z1: float, z2: float):
        r"""
        Compute the six Sommerfeld integrals in one vector-valued quadrature.

        .. math::

           I_1 = \int_{0}^{\infty} dq\; R_s(q) \frac{q}{2K_{z,0}} J_0(q\rho) e^{iK_{z,0}(z_1+z_2)}

        .. math::

           I_2 = \int_{0}^{\infty} dq\; R_s(q) \frac{q}{2K_{z,0}} J_2(q\rho) e^{iK_{z,0}(z_1+z_2)}

        .. math::

           I_3 = \int_{0}^{\infty} dq\; R_p(q) \frac{qK_{z,0}}{2k_0^2} J_0(q\rho) e^{iK_{z,0}(z_1+z_2)}

        .. math::

           I_4 = \int_{0}^{\infty} dq\; R_p(q) \frac{qK_{z,0}}{2k_0^2} J_2(q\rho) e^{iK_{z,0}(z_1+z_2)}

        .. math::

           I_5 = \int_{0}^{\infty} dq\; R_p(q) \frac{i q^2}{k_0^2} J_1(q\rho) e^{iK_{z,0}(z_1+z_2)}

        .. math::

           I_6 = \int_{0}^{\infty} dq\; R_p(q) \frac{q^3}{K_{z,0}k_0^2} J_0(q\rho) e^{iK_{z,0}(z_1+z_2)}
        """

        def integrand(q: float) -> np.ndarray:
            kz0 = self._kz(0, q)           # ← uses method, not lambda
            expz = np.exp(1j * kz0 * (z1 + z2))
            rs = self._rs(q)                # ← method call (overridable)
            rp = self._rp(q)                # ← method call (overridable)
            j0 = jv(0, q * rho)
            j1 = jv(1, q * rho)
            j2 = jv(2, q * rho)

            i1 = rs * (q / (2 * kz0)) * j0 * expz
            i2 = rs * (q / (2 * kz0)) * j2 * expz
            i3 = rp * (q * kz0 / (2 * self.k0**2)) * j0 * expz
            i4 = rp * (q * kz0 / (2 * self.k0**2)) * j2 * expz
            i5 = rp * (1j * q**2 / self.k0**2) * j1 * expz
            i6 = rp * (q**3 / (kz0 * self.k0**2)) * j0 * expz
            return np.array([i1, i2, i3, i4, i5, i6], dtype=complex)

        result = self.complex_quad(integrand, a=0)
        return np.asarray(result, dtype=complex)

    def I1_integral(self, rho: float, z1: float, z2: float):
        return self.compute_integrals(rho, z1, z2)[0]

    def I2_integral(self, rho: float, z1: float, z2: float):
        return self.compute_integrals(rho, z1, z2)[1]

    def I3_integral(self, rho: float, z1: float, z2: float):
        return self.compute_integrals(rho, z1, z2)[2]

    def I4_integral(self, rho: float, z1: float, z2: float):
        return self.compute_integrals(rho, z1, z2)[3]

    def I5_integral(self, rho: float, z1: float, z2: float):
        return self.compute_integrals(rho, z1, z2)[4]

    def I6_integral(self, rho: float, z1: float, z2: float):
        return self.compute_integrals(rho, z1, z2)[5]
    
    def scattering_s_component(self,
        x: float,
        y: float,
        z1: float,
        z2: float,
        integrals: Optional[np.ndarray] = None,):
        r"""
        s-polarized scattering tensor.

        .. math::

           \overline{\overline{\mathbf{M}}}_s(k_{\rho}, \omega) = \frac{k_{\rho}}{2K_{z,i}} \begin{bmatrix} J_0 + \cos(2\phi)J_2 & \sin(2\phi)J_2 & 0 \\ \sin(2\phi)J_2 & J_0 - \cos(2\phi)J_2 & 0 \\ 0 & 0 & 0 \end{bmatrix}.

        Args:
            x: x-distance between the two points.
            y: y-distance between the two points.
            z1: z-coordinate of the first point.
            z2: z-coordinate of the second point.
        Returns:
            np.ndarray: 3x3 s-polarized scattering tensor.
        """
        rho = np.sqrt(x**2 + y**2)
        if rho == 0:
            phi = 0.0
        else:
            phi = np.arctan2(y, x)

        vals = self.compute_integrals(rho, z1, z2) if integrals is None else integrals
        I1, I2 = vals[0], vals[1]

        Ms = np.array([[I1 + np.cos(2*phi)*I2, np.sin(2*phi)*I2, 0],
                        [np.sin(2*phi)*I2, I1 - np.cos(2*phi)*I2, 0],
                        [0,                 0,              0]], dtype = complex)
        
        return Ms
    
    def scattering_p_component(self,
        x: float,
        y: float,
        z1: float,
        z2: float,
        integrals: Optional[np.ndarray] = None,):
        r"""
        p-polarized scattering tensor.

        .. math::

           \overline{\overline{\mathbf{M}}}_p(k_{\rho}, \omega) = \frac{-k_{\rho}K_{z,i}}{2k_i^2} \begin{bmatrix} J_0 - \cos(2\phi)J_2 & -\sin(2\phi)J_2 & \frac{2ik_{\rho}}{K_{z,i}}\cos\phi\,J_1 \\ -\sin(2\phi)J_2 & J_0 + \cos(2\phi)J_2 & \frac{2ik_{\rho}}{K_{z,i}}\sin\phi\,J_1 \\ -\frac{2ik_{\rho}}{K_{z,i}}\cos\phi\,J_1 & -\frac{2ik_{\rho}}{K_{z,i}}\sin\phi\,J_1 & -\frac{2k_{\rho}^2}{K_{z,i}^2}J_0 \end{bmatrix}.

        Args:
            x: x-distance between the two points.
            y: y-distance between the two points.
            z1: z-coordinate of the first point.
            z2: z-coordinate of the second point.
        Returns:
            np.ndarray: 3x3 p-polarized scattering tensor.
        """
        rho = np.sqrt(x**2 + y**2)
        if rho == 0:
            phi = 0.0
        else:
            phi = np.arctan2(y, x)
        vals = self.compute_integrals(rho, z1, z2) if integrals is None else integrals
        I3, I4, I5, I6 = vals[2], vals[3], vals[4], vals[5]

        Mp = np.array([[ -I3 + np.cos(2*phi)*I4, np.sin(2*phi)*I4,  -np.cos(phi) * I5],
                        [np.sin(2*phi)*I4, -I3 - np.cos(2*phi)*I4, -np.sin(phi) * I5],
                        [np.cos(phi) * I5,  np.sin(phi) * I5, I6]], dtype = complex)
        return Mp
    
    def calculate_total_Green_function(self,
        x: float,
        y: float,
        z1: float,
        z2: float,):
        r"""
        Total Green's function = vacuum + scattering.

        .. math::

           \overline{\overline{\mathbf{G}}} = \overline{\overline{\mathbf{G}}}_0 + \overline{\overline{\mathbf{G}}}_{\text{SC}}^{(i)}.

        Args:
            x: x-distance between the two points.
            y: y-distance between the two points.
            z1: z-coordinate of the first point.
            z2: z-coordinate of the second point.
        Returns:
            np.ndarray: 3x3 total Green's tensor.
        """
        logger.debug(f"Calculating Green's function for points ({x}, {y}, {z1}) and ({0}, {0}, {z2}) at omega={(self.omega*hbar/eV_to_J):.3e} eV")


        G0 = self.vacuum_component(x, y, z1, z2)
        Gsc = self.scatter_component(x, y, z1, z2)
        G_total = G0 + Gsc
        return G_total
