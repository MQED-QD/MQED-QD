import numpy as np
from scipy.integrate import quad_vec
from scipy.special import jv # Bessel function of the first kind
from loguru import logger
from mqed.utils.SI_unit import  c, hbar, eV_to_J

class Greens_function_analytical:
    """
    Calculates the Dyadic Green's function in the presence of a planar interfaces. 
    Since there is cylindrical sysmetry, we can calculate the Green's function by Besseel Functions.
    Define the s-and p-polarized reflection coefficients from Fresnel equations:
    ..math::
        r_s(q) = \frac{K_{z,0}(q) - K_{z,1}(q)}{K_{z,0}(q) + K_{z,1}(q)}
        r_p(q) = \frac{\epsilon_1 K_{z,0}(q) - \epsilon_0 K_{z,1}(q)}{\epsilon_1 K_{z,0}(q) + \epsilon_0 K_{z,1}(q)}
    where
    ..math::
        K_{z,i}(q) = \sqrt{\epsilon_i k_0^2 - q^2}
    The total Green's function is the sum of the vacuum component and the scattering component:
    ..math::
        \overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega) = \overline{\overline{\mathbf{G}}}_0(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega) + \overline{\overline{\mathbf{G}}}_{\text{refl}}^{(i)}(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega)
    """
    def __init__(self,
        metal_epsi: complex,
        omega: float,
        eps_0 = 1.0, 
    ):
        """
        Initializes the calculation with the system's physical parameters.
        Args:
            metal_epsi (complex): The permittivity of the metal.
            omega (float): The angular frequency of the light.
            eps_0 (float, optional): The permittivity of free space. 
        """
        self.metal_epsi = metal_epsi
        self.omega = omega
        self.eps_0 = eps_0
        self.c = c# speed of light in SI unit
        self.k0 = self.omega/self.c  # free space wavenumber

        # Pre-define the Fresnel equations for reflection coefficients
        self._kz0 = lambda q: self._beta_phys(self.eps_0, q)
        self._kz1 = lambda q: self._beta_phys(self.metal_epsi, q)

        # Reflection coefficients for s- and p-polarized waves
        self._rs = lambda q: (self._kz0(q) - self._kz1(q))/ (self._kz0(q) + self._kz1(q))
        self._rp = lambda q: (self.metal_epsi * self._kz0(q) - self.eps_0 * self._kz1(q)) / \
                            (self.metal_epsi * self._kz0(q) + self.eps_0 * self._kz1(q))
    
    def _beta_phys(self, eps, q):
        """
        Calculates the physical wavevector component in the z-direction.
        This function ensures that the imaginary part of the wavevector is non-negative,
        enforcing the correct physical decay behavior in lossy media.
        Args:
            eps (complex): The permittivity of the medium.(vacuum or material)
            q (float or np.ndarray): The transverse wavevector component.
        """
        # tiny i0+ ONLY here (do not add imag to k0 itself)
        b = np.lib.scimath.sqrt(eps * self.k0**2 - q**2 + 1j*1e-12)
        # enforce decay: Im(b) >= 0 (and if ~0, choose Re(b) >= 0)
        if np.ndim(b):
            flip = (np.imag(b) < 0) | ((np.abs(np.imag(b)) < 1e-18) & (np.real(b) < 0))
            b[flip] = -b[flip]
            return b
        return -b if (np.imag(b) < 0 or (abs(np.imag(b)) < 1e-18 and np.real(b) < 0)) else b
    
    

    def complex_quad(self,
                    func: callable,
                    a: float,
                    qmax=np.inf,
                    epsabs=1e-10,
                    epsrel=1e-10,
                    limit=400,):
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
        
        result, _ = quad_vec(func, a, qmax, epsabs=epsabs, epsrel=epsrel, limit=limit)
        return result

    
    def vacuum_component(self,
        x: float,
        y: float,
        z1: float,
        z2: float,):
        """
        Calculates the vacuum component of the Green's function.
        .. math::
            \overline{\overline{\mathbf{G}}}_0(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega_\mathrm{M})=&
            \frac{e^{ik_0 R_{\alpha\beta}}}{4\pi R_{\alpha\beta}}
            \left\{\vphantom{\frac{e^R}{R}}
            \left(\overline{\overline{\mathbf{I}}}_3-{\mathbf{e}}_\mathrm{R}  {\mathbf{e}}_\mathrm{R} \right)\right. +\left.\left(3 {\mathbf{e}}_\mathrm{R}   {\mathbf{e}}_\mathrm{R}  -\overline{\overline{\mathbf{I}}}_3\right)\left[\frac{1}{(k_0 R_{\alpha\beta})^{2}}-\frac{i}{k_0 R_{\alpha\beta}}\right]
            \right\}
        
        Args:
            x (float): The x-distance between the two points.
            y (float): The y-distance between the two points.
            z1 (float): The z-coordinate of the first point.
            z2 (float): The z-coordinate of the second point.
        Returns:
            np.ndarray: The vacuum component of the Green's function as a 3x3 matrix
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
        """
        Calculates the scattering component of the Green's function.
        .. math::
            \overline{\overline{\mathbf{G}}}_{\text{refl}}^{(i)}(\rho, \phi, z, z', \omega) = 
            \int_{0}^{+\infty} \frac{idk_{\rho}}{4\pi} \left[ R_s(k_{\rho}, \omega) \overline{\overline{\mathbf{M}}}_s(k_{\rho}, \omega) + R_p(k_{\rho}, \omega) \overline{\overline{\mathbf{M}}}_p(k_{\rho}, \omega) \right] e^{iK_{z,i}(k_{\rho}, \omega)(z+z')}
        Args:
            x (float): The x-distance between the two points.
            y (float): The y-distance between the two points.
            z1 (float): The z-coordinate of the first point.
            z2 (float): The z-coordinate of the second point.
            """
        Ms = self.scattering_s_component(x, y, z1, z2)
        Mp = self.scattering_p_component(x, y, z1, z2)
        prefactor = 1j / (4 * np.pi)
        G_scatter = prefactor * (Ms + Mp)
        return G_scatter

    
    def I1_integral(self,
        rho: float,
        z1: float,
        z2: float,):
        """
        Integrand for the I1 integral in the scattering component of the Green's function.
        .. math::
            I_1 = \int_{0}^{\infty} dq R_s(q, \omega) \frac{q}{2K_{z,0}} J_0(q\rho) e^{iK_{z,0}(z_1+z_2)}
        Args:
            rho (float): The radial distance between the two points in the xy-plane.
            z1 (float): The z-coordinate of the first point.
            z2 (float): The z-coordinate of the second point.
        Returns:
            complex: The value of the integrand at the given q.
        """
        integrand = lambda q: self._rs(q) * (q / (2*self._kz0(q))) * jv(0, q * rho) * np.exp(1j * self._kz0(q) * (z1 + z2))
        I1 = self.complex_quad(integrand, a=0)
        return I1
    
    def I2_integral(self,
        rho: float,
        z1: float,
        z2: float,):
        """
        Integrand for the I2 integral in the scattering component of the Green's function.
        .. math::
            I_2 = \int_{0}^{\infty} dq R_s(q, \omega) \frac{q}{2K_{z,0}} J_2(q\rho) e^{iK_{z,0}(z_1+z_2)}
        Args:
            rho (float): The radial distance between the two points in the xy-plane.
            z1 (float): The z-coordinate of the first point. (Doner)
            z2 (float): The z-coordinate of the second point. (Acceptor)
        Returns:
            complex: The value of the integrand at the given q.
        """
        integrand = lambda q: self._rs(q) * (q / (2 * self._kz0(q))) * jv(2, q * rho) * np.exp(1j * self._kz0(q) * (z1 + z2))
        I2 = self.complex_quad(integrand, a=0)
        return I2
    
    def I3_integral(self,
        rho: float,
        z1: float,
        z2: float,):
        """
        Integrand for the I3 integral in the scattering component of the Green's function.
        .. math::
            I_3 = \int_{0}^{\infty} dq R_p(q, \omega) \frac{qK_{z,0}}{2k_0^2} J_0(q\rho) e^{iK_{z,0}(z_1+z_2)}
        Args:
            rho (float): The radial distance between the two points in the xy-plane.
            z1 (float): The z-coordinate of the first point. (Doner)
            z2 (float): The z-coordinate of the second point. (Acceptor)
        Returns:
            complex: The value of the integrand at the given q.
        """
        integrand = lambda q: self._rp(q) * (q * self._kz0(q) / (2 * self.k0**2)) * jv(0, q * rho) * np.exp(1j * self._kz0(q) * (z1 + z2))
        I3 = self.complex_quad(integrand, a=0)
        return I3
    
    def I4_integral(self,
        rho: float,
        z1: float,
        z2: float,):
        """
        Integrand for the I4 integral in the scattering component of the Green's function.
        .. math::
            I_4 = \int_{0}^{\infty} dq R_p(q, \omega) \frac{qK_{z,0}}{2k_0^2} J_2(q\rho) e^{iK_{z,0}(z_1+z_2)}
        Args:
            rho (float): The radial distance between the two points in the xy-plane.
            z1 (float): The z-coordinate of the first point. (Doner)
            z2 (float): The z-coordinate of the second point. (Acceptor)
        Returns:
            complex: The value of the integrand at the given q.
        """
        integrand = lambda q: self._rp(q) * (q * self._kz0(q) / (2 * self.k0**2)) * jv(2, q * rho) * np.exp(1j * self._kz0(q) * (z1 + z2))
        I4 = self.complex_quad(integrand, a=0)
        return I4
    
    def I5_integral(self,
        rho: float,
        z1: float,
        z2: float,):
        """
        Integrand for the I5 integral in the scattering component of the Green's function.
        .. math::
            I_5 = \int_{0}^{\infty} dq R_p(q, \omega) \frac{iq^2}{k_0^2} J_1(q\rho) e^{iK_{z,0}(z_1+z_2)}
        Args:
            rho (float): The radial distance between the two points in the xy-plane.
            z1 (float): The z-coordinate of the first point. (Doner)
            z2 (float): The z-coordinate of the second point. (Acceptor)
        Returns:
            complex: The value of the integrand at the given q.
        """
        integrand = lambda q: self._rp(q) * (1j * q**2 / self.k0**2) * jv(1, q * rho) * np.exp(1j * self._kz0(q) * (z1 + z2))
        I5 = self.complex_quad(integrand, a=0)
        return I5
    
    def I6_integral(self,
        rho: float,
        z1: float,
        z2: float,):
        """
        Integrand for the I6 integral in the scattering component of the Green's function.
        .. math::
            I_6 = \int_{0}^{\infty} dq R_p(q, \omega) \frac{q^3}{K_{z,0}k_0^2} J_0(q\rho) e^{iK_{z,0}(z_1+z_2)}
        Args:
            rho (float): The radial distance between the two points in the xy-plane.
            z1 (float): The z-coordinate of the first point. (Doner)
            z2 (float): The z-coordinate of the second point. (Acceptor)
        Returns:
            complex: The value of the integrand at the given q.
        """
        integrand = lambda q: self._rp(q) * (q**3 / (self._kz0(q)*self.k0**2)) * jv(0, q * rho) * np.exp(1j * self._kz0(q) * (z1 + z2))
        I6 = self.complex_quad(integrand, a=0 )
        return I6
    
    def scattering_s_component(self,
        x: float,
        y: float,
        z1: float,
        z2: float,):
        """
        Calculates the s-polarized scattering component of the Green's function.
        .. math::
            \overline{\overline{\mathbf{M}}}_s(k_{\rho}, \omega) = \frac{k_{\rho}}{2K_{z,i}(k_{\rho}, \omega)} \begin{bmatrix} J_0(k_{\rho}\rho) + \cos(2\phi)J_2(k_{\rho}\rho) & \sin(2\phi)J_2(k_{\rho}\rho) & 0 \\ \sin(2\phi)J_2(k_{\rho}\rho) & J_0(k_{\rho}\rho) - \cos(2\phi)J_2(k_{\rho}\rho) & 0 \\ 0 & 0 & 0 \end{bmatrix}
        Args:
            x (float): The x-distance between the two points.
            y (float): The y-distance between the two points.
            z1 (float): The z-coordinate of the first point.
            z2 (float): The z-coordinate of the second point.
        Returns:
            np.ndarray: The s-polarized scattering component of the Green's function as a 3x3 matrix
        """
        rho = np.sqrt(x**2 + y**2)
        if rho == 0: phi = 0
        else: phi = np.arctan2(y, x)

        I1 = self.I1_integral(rho, z1, z2)
        I2 = self.I2_integral(rho, z1, z2)

        Ms = np.array([[I1 + np.cos(2*phi)*I2, np.sin(2*phi)*I2, 0],
                        [np.sin(2*phi)*I2, I1 - np.cos(2*phi)*I2, 0],
                        [0,                 0,              0]], dtype = complex)
        
        return Ms
    
    def scattering_p_component(self,
        x: float,
        y: float,
        z1: float,
        z2: float,):
        """
        Calculates the p-polarized scattering component of the Green's function.
        .. math::
            \overline{\overline{\mathbf{M}}}_p(k_{\rho}, \omega) = \frac{-k_{\rho}K_{z,i}(k_{\rho}, \omega)}{2k_i^2(\omega)} \begin{bmatrix} J_0(k_{\rho}\rho) - \cos(2\phi)J_2(k_{\rho}\rho) & -\sin(2\phi)J_2(k_{\rho}\rho) & \frac{2ik_{\rho}}{K_{z,i}(k_{\rho}, \omega)}\cos(\phi)J_1(k_{\rho}\rho) \\ -\sin(2\phi)J_2(k_{\rho}\rho) & J_0(k_{\rho}\rho) + \cos(2\phi)J_2(k_{\rho}\rho) & \frac{2ik_{\rho}}{K_{z,i}(k_{\rho}, \omega)}\sin(\phi)J_1(k_{\rho}\rho) \\ \frac{-2ik_{\rho}}{K_{z,i}(k_{\rho}, \omega)}\cos(\phi)J_1(k_{\rho}\rho)) & \frac{-2ik_{\rho}}{K_{z,i}(k_{\rho}, \omega)}\sin(\phi)J_1(k_{\rho}\rho) & \frac{-2k_{\rho}^2}{K_{z,i}^2(k_{\rho}, \omega)}J_0(k_{\rho}\rho) \end{bmatrix}
        Args:
            x (float): The x-distance between the two points.
            y (float): The y-distance between the two points.
            z1 (float): The z-coordinate of the first point.
            z2 (float): The z-coordinate of the second point.
        """
        rho = np.sqrt(x**2 + y**2)
        if rho == 0: phi = 0
        else: phi = np.arctan2(y, x)
        I3 = self.I3_integral(rho, z1, z2)
        I4 = self.I4_integral(rho, z1, z2)
        I5 = self.I5_integral(rho, z1, z2)
        I6 = self.I6_integral(rho, z1, z2)

        Mp = np.array([[ -I3 + np.cos(2*phi)*I4, np.sin(2*phi)*I4,  -np.cos(phi) * I5],
                        [np.sin(2*phi)*I4, -I3 - np.cos(2*phi)*I4, -np.sin(phi) * I5],
                        [np.cos(phi) * I5,  np.sin(phi) * I5, I6]], dtype = complex)
        return Mp
    
    def calculate_total_Green_function(self,
        x: float,
        y: float,
        z1: float,
        z2: float,):
        """
        Calculates the total Green's function as the sum of the vacuum and scattering components.
        .. math::
            \overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega) = \overline{\overline{\mathbf{G}}}_0(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega) + \overline{\overline{\mathbf{G}}}_{\text{SC}}^{(i)}(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega)
        Args:
            x (float): The x-distance between the two points.
            y (float): The y-distance between the two points.
            z1 (float): The z-coordinate of the first point.
            z2 (float): The z-coordinate of the second point.
        Returns:
            np.ndarray: The total Green's function as a 3x3 matrix
        """
        logger.debug(f"Calculating Green's function for points ({x}, {y}, {z1}) and ({0}, {0}, {z2}) at omega={(self.omega*hbar/eV_to_J):.3e} eV")


        G0 = self.vacuum_component(x, y, z1, z2)
        Gsc = self.scatter_component(x, y, z1, z2)
        G_total = G0 + Gsc
        return G_total