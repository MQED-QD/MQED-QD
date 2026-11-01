.. _theory-n-layer:

N-Layer Planar Geometry
=======================

A planar stack changes the field through repeated reflection and transmission
at its interfaces. This chapter extends the :ref:`two-layer theory
<theory-two-layer>` to a source and observer in the same layer of a finite
stack. The angular-spectrum construction follows [Tomas1995]_, [Dung1998]_,
and [Novotny2012NLayer]_. The source-layer scattering dyadic and stack
recursion are also derived in [Tomas2002]_, especially its Appendix. That
work uses a different Green-tensor delta-function normalization; the
prefactors below follow the unit-delta convention of
``mqed.Dyadic_GF.GF_NLayer`` throughout.


Geometry and wave equation
--------------------------

Number layers :math:`0,\ldots,N-1` from bottom to top, with relative
permittivities :math:`\epsilon_l(\omega)` and nonmagnetic permeability.
Layers :math:`0` and :math:`N-1` are normally semi-infinite; an internal
layer :math:`l` occupies :math:`Z_l<z_{\rm global}<Z_l+d_l`. Source and
observer lie in layer :math:`j`. For a finite source layer use local heights
:math:`0<z',z<d_j` measured upward from its lower interface, and
:math:`\mathbf r'=(0,0,z')`, :math:`\mathbf r=(\rho\cos\phi,\rho\sin\phi,z)`.
The supported semi-infinite source case is the **top** exterior: then
:math:`z,z'>0` are heights above the top-stack interface. Values at an
interface require a specified one-sided limit; a point source exactly on an
interface needs a separate prescription. A source in the
bottom exterior or observation across different layers is not covered by
these same-region formulas or by this solver.

With time dependence :math:`e^{-i\omega t}`, :math:`k_0=\omega/c`, and an
electric-current Green tensor normalized to a unit dyadic delta function,

.. math::

   [\boldsymbol\nabla\times\boldsymbol\nabla\times
    -k_0^2\epsilon(z,\omega)]\mathbf G(\mathbf r,\mathbf r',\omega)
   =\mathbf I\,\delta(\mathbf r-\mathbf r').

Write :math:`\mathbf G=\mathbf G_0^{(j)}+\mathbf G_{\rm Sc}^{(j)}`. Here
:math:`\mathbf G_0^{(j)}` is the outgoing homogeneous **source-medium** tensor
with :math:`k_j^2=\epsilon_j k_0^2`, not necessarily the literal vacuum
tensor even though the implementation calls it ``vacuum_component``. For
distinct points, with :math:`R=|\mathbf r-\mathbf r'|` and
:math:`\widehat{\mathbf R}=(\mathbf r-\mathbf r')/R`, it is

.. math::

   \mathbf G_0^{(j)}=
   \frac{e^{ik_jR}}{4\pi R}\left[
   (\mathbf I-\widehat{\mathbf R}\widehat{\mathbf R})
   +(3\widehat{\mathbf R}\widehat{\mathbf R}-\mathbf I)
   \left(\frac{1}{(k_jR)^2}-\frac{i}{k_jR}\right)\right].

At coincident points the code returns the conventional finite term
:math:`ik_j\mathbf I/(6\pi)` while omitting the singular local contribution.
In an absorbing host this alone is not a complete coincident-point prescription.

Transverse Fourier transformation separates each plane wave:

.. math::

   \mathbf G_{\rm Sc}^{(j)}(\boldsymbol\rho,z,z')=
   \int\frac{d^2\mathbf q}{(2\pi)^2}
   e^{i\mathbf q\cdot\boldsymbol\rho}
   \mathbf g_{\rm Sc}^{(j)}(\mathbf q;z,z'),\qquad
   \beta_l(q)=\sqrt{\epsilon_l k_0^2-q^2},\quad q=|\mathbf q|.

The outgoing branch has :math:`\operatorname{Im}\beta_l\ge0`, and positive
real :math:`\beta_l` for lossless propagating waves. Thus an upward wave
:math:`e^{i\beta_l z}` decays upward when evanescent; the downward counterpart
decays in the opposite direction. Continue this branch consistently when
considering complex :math:`q` or lossy layers.


Interface and stack reflections
-------------------------------

For a wave incident from layer :math:`a` on adjacent layer :math:`b`, the
electric TE and TM coefficients used here are

.. math::

   r_{ab}^s=\frac{\beta_a-\beta_b}{\beta_a+\beta_b},\qquad
   r_{ab}^p=\frac{\epsilon_b\beta_a-\epsilon_a\beta_b}
                   {\epsilon_b\beta_a+\epsilon_a\beta_b}.

In this TM convention the polarization vectors for upward and downward
waves may be chosen as :math:`\mathbf p_l^\pm=(\mp\beta_l\widehat{\mathbf q}
+q\widehat{\mathbf z})/k_l`; reversing the TM basis reverses some
intermediate reflection signs, but not the physical tensor. The displayed
:math:`r^p` is the coefficient used in the formulas below and in the
implementation. For two media, these reduce to the coefficients in
:ref:`two-layer theory <theory-two-layer>`.

Let :math:`\widetilde R_{l,\sigma}^{(h)}` be the effective reflection seen
from layer :math:`l` looking in direction :math:`h=+1` (up) or :math:`h=-1`
(down), where :math:`\sigma=s,p`. Set :math:`n=l+h`. If :math:`n` is the
outermost layer in that direction, terminate with
:math:`\widetilde R_{l,\sigma}^{(h)}=r_{ln}^\sigma`. Otherwise :math:`n`
is finite and the recursion is

.. math::

   \widetilde R_{l,\sigma}^{(h)}=
   \frac{r_{ln}^\sigma+
     \widetilde R_{n,\sigma}^{(h)}e^{2i\beta_n d_n}}
    {1+r_{ln}^\sigma\widetilde R_{n,\sigma}^{(h)}
       e^{2i\beta_n d_n}}.

If there is no interface in a given direction, its effective reflection is
zero. The phase traverses the **next** finite layer, not the current one.
Define :math:`r_-^\sigma=\widetilde R_{j,\sigma}^{(-1)}` and
:math:`r_+^\sigma=\widetilde R_{j,\sigma}^{(+1)}`. This recursion preserves
both finite-film interference and reflections from the exterior substrates.


Finite source layer: repeated paths
-----------------------------------

For :math:`0<z,z'<d_j`, the cavity round-trip factor and its Airy
denominator are

.. math::

   Q_\sigma=r_-^\sigma r_+^\sigma e^{2i\beta_jd_j},\qquad
   D_\sigma=1-Q_\sigma,\qquad \sigma=s,p.

When :math:`|Q_\sigma|<1`, :math:`1/D_\sigma=\sum_{m=0}^\infty Q_\sigma^m`:
each term adds another complete bounce between the effective mirrors. This
geometric expansion is conditional; the quotient defines the analytically
continued response elsewhere. Set

.. math::

   L=e^{i\beta_j(z+z'-d_j)},\qquad
   U=e^{-i\beta_j(z+z'-d_j)},\qquad
   P_\sigma=r_-^\sigma r_+^\sigma e^{i\beta_jd_j},\qquad
   \Delta=z-z'.

The five source-aware amplitudes in the solver are

.. math::

   \begin{aligned}
   C_s&=\frac{r_-^s L+r_+^s U+2P_s\cos(\beta_j\Delta)}{D_s},\\
   C_p^\pm&=\frac{r_-^p L+r_+^p U
                  \mathbin{\pm}2P_p\cos(\beta_j\Delta)}{D_p},\\
   S_p^\pm&=\frac{r_-^p L-r_+^p U
                  \mathbin{\pm}2iP_p\sin(\beta_j\Delta)}{D_p}.
   \end{aligned}

The two one-reflection paths have lengths :math:`z+z'` and
:math:`2d_j-z-z'` after multiplication by :math:`e^{i\beta_jd_j}`;
:math:`2P_\sigma\cos(\beta_j\Delta)` combines the two double-reflection
paths. Likewise, the signed sine terms distinguish source-to-observer and
observer-to-source TM cross components. The kernels use **scaled** amplitudes
:math:`\overline C_s=e^{i\beta_jd_j}C_s`,
:math:`\overline C_p^\pm=e^{i\beta_jd_j}C_p^\pm`, and
:math:`\overline S_p^\pm=e^{i\beta_jd_j}S_p^\pm`. The code evaluates these
directly with the nonnegative path lengths above and
:math:`2d_j\mathbin{\pm}(z-z')`, avoiding overflow from an unscaled
evanescent exponential that would later cancel.


Sommerfeld kernels and tensor components
----------------------------------------

The azimuthal Fourier integral yields Bessel functions
:math:`J_m=J_m(q\rho)`. To fix every factor and sign, define seven integrals
:math:`I_a=\int_0^\infty F_a(q)J_{\nu_a}(q\rho)\,dq` (or a configured
finite upper cutoff), with :math:`k_j^2=\epsilon_jk_0^2`:

.. math::

   \begin{aligned}
   I_1&=\int_0^\infty \frac{q\overline C_s}{2\beta_j}J_0\,dq,&
   I_2&=\int_0^\infty \frac{q\overline C_s}{2\beta_j}J_2\,dq,\\
   I_3&=\int_0^\infty \frac{q\beta_j\overline C_p^-}{2k_j^2}J_0\,dq,&
   I_4&=\int_0^\infty \frac{q\beta_j\overline C_p^-}{2k_j^2}J_2\,dq,\\
   I_5^+&=\int_0^\infty \frac{iq^2\overline S_p^+}{k_j^2}J_1\,dq,&
   I_5^-&=\int_0^\infty \frac{iq^2\overline S_p^-}{k_j^2}J_1\,dq,\\
   I_6&=\int_0^\infty \frac{q^3\overline C_p^+}{\beta_j k_j^2}J_0\,dq.
   \end{aligned}

Here each :math:`J_m` is evaluated at :math:`q\rho` inside its own
integral; :math:`\beta_j`, amplitudes, and effective reflections all depend
on :math:`q` and :math:`\omega`. The first two integrals are TE, the other
five TM. In Cartesian components the angular integration gives

.. math::

   \mathbf G_{\rm Sc}^{(j)}=\frac{i}{4\pi}
   \left[
   \begin{pmatrix}
   I_1+\cos(2\phi)I_2&\sin(2\phi)I_2&0\\
   \sin(2\phi)I_2&I_1-\cos(2\phi)I_2&0\\
   0&0&0
   \end{pmatrix}
   +
   \begin{pmatrix}
   -I_3+\cos(2\phi)I_4&\sin(2\phi)I_4&-\cos\phi\,I_5^+\\
   \sin(2\phi)I_4&-I_3-\cos(2\phi)I_4&-\sin\phi\,I_5^+\\
   \cos\phi\,I_5^-&\sin\phi\,I_5^-&I_6
   \end{pmatrix}\right].

In particular, :math:`I_5^+` and :math:`I_5^-` are generally unequal in a
finite cavity. Their differing signs in the :math:`xz` and :math:`zx`
entries are part of the chosen TM polarization convention, not an extra
factor of two. Adding :math:`\mathbf G_0^{(j)}` gives the total tensor.


Top exterior and mode interpretation
------------------------------------

For the supported top exterior source, :math:`r_+^\sigma=0`; there is no
finite source-layer thickness and the cavity denominator tends to one.
Writing :math:`r_-^\sigma` for the reflection of the **entire** stack below,
the scaled amplitudes reduce to

.. math::

   \overline C_s=r_-^s e^{i\beta_j(z+z')},\qquad
   \overline C_p^+=\overline C_p^-
   =\overline S_p^+=\overline S_p^-
   =r_-^p e^{i\beta_j(z+z')}.

The same seven kernels apply with these one-sided amplitudes. For an actual
single interface this is the two-layer limit; for a film stack it is **not**
equivalent to replacing the lower stack by one homogeneous half-space. For
finite :math:`j`, zeros of :math:`D_\sigma` can mark guided, cavity, or
surface-mode poles only when the outgoing sheet and numerator are checked;
branch points occur at :math:`q=\sqrt{\epsilon_l}k_0`. In the top exterior,
the relevant pole characteristic is instead the denominator of the recursive
downward reflection, since the source-cavity denominator is one. The solver
uses a ratio-free recursion for pole searches so that internal reflection
denominators aren't lost to cancellation. Neither a zero of a reflection
*numerator* nor every small value on the real axis is automatically a mode.


Numerical strategies
--------------------

``direct`` evaluates the seven real-:math:`q` Bessel integrals by quadrature;
``fixed_grid`` samples the kernels once for several separations but requires
a finite cutoff. ``componentwise`` instead integrates assembled tensor entries
with separate real and imaginary error control. ``singularity_aware`` keeps
the **same** real-axis integral, splitting it at real light lines and near-real
pole projections. It records pole diagnostics but does not automatically
replace the integral by explicit residues.

``pole_subtracted_direct`` subtracts detected simple-pole kernels and adds
them back numerically using the same Bessel convention. The optional
``pole_aware_hybrid_dcim`` applies this idea before fitting a tail. Such pole
searches and subtractions need validation for the particular stack; off-axis
poles are already implicit in direct real-axis quadrature. Diagnostic contour
residues and branch-cut terms should not be added to production tensors
without checking their contour normalization and agreement with direct results.

Complex-image methods draw on [Chow1991]_, [Aksun1996]_, and the matrix-pencil
method of [Hua1990]_. Legacy ``dcim`` fits sampled kernels over a configured
finite interval; ``hybrid_dcim`` integrates the low-:math:`q` region directly
and fits only a finite high-:math:`q` tail with exponentials. The implementation
can validate that fitted tail against direct finite-tail quadrature and fall
back to direct integration when it fails. ``branch_cut_dcim`` is an
experimental pole/branch-cut fit with validation and fallback, not a general
guarantee of exact pole extraction. At zero lateral separation, DCIM-family
methods use the singularity-aware direct path instead.


Implementation
--------------

The formulas above correspond to ``mqed.Dyadic_GF.GF_NLayer``, driven by
``mqed.Dyadic_GF.main_nlayer``. The :ref:`tutorial-gf-nlayer` gives stack
coordinates, integration choices, and runnable examples. Results use the
shared ``separation`` HDF5 layout indexed by energy and lateral separation
:math:`\rho`.


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

.. [Tomas2002] M. S. Tomaš, "Casimir force in absorbing multilayers," *Phys.
   Rev. A* **66**, 052103 (2002), doi:`10.1103/PhysRevA.66.052103
   <https://doi.org/10.1103/PhysRevA.66.052103>`_.
