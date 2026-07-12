.. _theory-n-layer:

N-Layer Planar Geometry
=======================

This section summarizes the Green-function formulation used for finite planar
multilayers.  It extends the two-layer Sommerfeld-integral picture to an
arbitrary stack of homogeneous layers.  The goal here is to explain the objects
used by the implementation, not to reproduce every algebraic step.  Detailed
derivations of multilayer dyadic Green functions can be found in [Tomas1995]_,
[Dung1998]_, and [Novotny2012NLayer]_.  The GPOF/DCIM acceleration idea follows
the standard complex-image literature [Chow1991]_, [Aksun1996]_, [Hua1990]_.


Geometry
--------

The structure is translationally invariant in the :math:`x`--:math:`y` plane
and piecewise homogeneous along :math:`z`:

.. math::

   \epsilon(z,\omega) = \epsilon_\ell(\omega),
   \qquad z_{\ell} < z < z_{\ell+1}.

Top and bottom layers are usually semi-infinite, while internal layers have
finite thickness.  In the current workflow the source and observer are in a
chosen source layer, with vertical coordinates :math:`z'` and :math:`z` and
lateral separation :math:`\rho`.


Spectral-domain Green function
------------------------------

After Fourier transforming in the transverse directions, Maxwell's equation
reduces to a one-dimensional Sommerfeld representation over the in-plane wave
number :math:`k_\rho`.  For layer :math:`\ell`, define

.. math::

   \beta_\ell(k_\rho,\omega)
   = \sqrt{\epsilon_\ell(\omega)\frac{\omega^2}{c^2}-k_\rho^2},
   \qquad \mathrm{Im}\,\beta_\ell \ge 0.

The effect of all layers above and below the source layer is collected into
generalized reflection coefficients
:math:`\widetilde R_+^{s,p}(k_\rho)` and
:math:`\widetilde R_-^{s,p}(k_\rho)`.  These are obtained by the usual Fresnel
recursions through the stack.  Multiple reflections inside the source layer
produce Airy denominators of the form

.. math::

   D_q(k_\rho) = 1 - \widetilde R_-^q\widetilde R_+^q
   e^{2i\beta_j d_j},
   \qquad q=s,p,

where :math:`j` is the source layer and :math:`d_j` is its thickness.  Zeros of
:math:`D_q` correspond to guided modes, Fabry--Pérot modes, or plasmonic modes.

The scattered Green tensor is assembled from a small set of scalar Sommerfeld
integrals,

.. math::

   I_m(\rho,z,z',\omega)
   = \int_0^\infty K_m(k_\rho;z,z',\omega)
     J_{\nu_m}(k_\rho\rho)\,dk_\rho,

where :math:`J_{\nu_m}` is a Bessel function and :math:`K_m` contains the
generalized reflections, phase factors, Airy denominators, and polarization
prefactors.  The total Green tensor is

.. math::

   \mathbf{G} = \mathbf{G}_0^{(j)} + \mathbf{G}_\mathrm{Sc}.


Numerical strategies
--------------------

Direct quadrature is the reference method: it evaluates the Sommerfeld
integrals directly, splitting the integration domain near light lines and other
known features.  It is robust, but expensive when many separations, energies, or
layers are needed.

The singularity-aware idea is to avoid asking a generic quadrature or fit to
resolve sharp resonant structure blindly.  Candidate poles are located from
small values or roots of :math:`D_s` and :math:`D_p`; their residue contributions
are extracted explicitly.  The remaining smooth integrand is then much easier to
integrate.

Hybrid DCIM uses this same separation of difficulty.  A low-:math:`k_\rho`
interval containing branch points and possible mode structure is kept as direct
quadrature.  The smoother evanescent tail is fitted by the
generalized pencil-of-function (GPOF) method,

.. math::

   K_m(k_\rho) \approx \sum_{r=1}^{N_i}
   a_{mr}\,e^{-\alpha_{mr}(k_\rho-k_c)},
   \qquad k_\rho \ge k_c.

Each exponential term becomes a complex-image contribution that can be evaluated
much more cheaply for repeated spatial separations.  In practice the tail fit
must be validated against direct finite-tail quadrature; if the fit fails, the
safe fallback is direct integration.


Implementation
--------------

The N-layer implementation is in :mod:`mqed.Dyadic_GF.GF_NLayer` and is driven
by :mod:`mqed.Dyadic_GF.main_nlayer`.  The tutorial
:ref:`tutorial-gf-nlayer` shows how to run direct, singularity-aware, and
hybrid-DCIM examples.  The output uses the shared ``separation`` HDF5 layout,
indexed by energy and lateral separation :math:`\rho`.


References
----------

.. [Tomas1995] M. S. Tomaš, "Green function for multilayers: Light scattering
   in planar cavities," *Phys. Rev. A* **51**, 2545--2559 (1995).

.. [Dung1998] H. T. Dung, L. Knöll, and D.-G. Welsch, "Three-dimensional
   quantization of the electromagnetic field in dispersive and absorbing
   inhomogeneous dielectrics," *Phys. Rev. A* **57**, 3931--3942 (1998).

.. [Chow1991] Y. L. Chow, J. J. Yang, D. G. Fang, and G. E. Howard, "A closed-
   form spatial Green's function for the thick microstrip substrate," *IEEE
   Trans. Microw. Theory Tech.* **39**, 588--592 (1991).

.. [Aksun1996] M. I. Aksun, "A robust approach for the derivation of closed-
   form Green's functions," *IEEE Trans. Microw. Theory Tech.* **44**,
   651--658 (1996).

.. [Hua1990] Y. Hua and T. K. Sarkar, "Matrix pencil method for estimating
   parameters of exponentially damped/undamped sinusoids in noise," *IEEE
   Trans. Acoust. Speech Signal Process.* **38**, 814--824 (1990).

.. [Novotny2012NLayer] L. Novotny and B. Hecht, *Principles of Nano-Optics*,
   2nd ed. (Cambridge University Press, 2012).
