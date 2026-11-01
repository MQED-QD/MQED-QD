.. _theory-mie:

Mie Theory for Spherical Cavities
=================================

Concentric dielectric interfaces separate Maxwell's equations into independent
electric (TM) and magnetic (TE) multipoles. This section sets out the wave
conventions, boundary problem, and same-region Green tensor used to interpret
the Mie workflow. Standard treatments of spherical scattering and dyadic
Green functions are given in [Bohren1983]_ and [Chew1995]_; [Lee2020]_
discusses resonance energy transfer near localized surface plasmon
polaritons. These references provide physical context, not a validation of
this implementation. An explicit exterior-sphere scattering Green tensor is
given in [Suttorp2011]_, Eqs. (3.2)--(3.8); its wave equation uses a different
overall sign, so its coefficient prefactors cannot be transferred here without
reconciling conventions.


Geometry and Maxwell equation
-----------------------------

Take time dependence :math:`e^{-i\omega t}`, nonmagnetic media
(:math:`\mu=\mu_0`), and locally homogeneous, isotropic relative permittivity
:math:`\varepsilon_j(\omega)` in each region. Write
:math:`k_0=\omega/c` and :math:`k_j=k_0\sqrt{\varepsilon_j}`; the outgoing
branch is selected for passive media. The electric Green tensor satisfies

.. math::

   \left[\boldsymbol\nabla\times\boldsymbol\nabla\times
   -k_0^2\varepsilon(\mathbf r,\omega)\mathbf I\right]
   \mathbf G(\mathbf r,\mathbf r',\omega)
   =\mathbf I\,\delta(\mathbf r-\mathbf r').

Here :math:`\mathbf r'` is the source and :math:`\mathbf r` is the observer;
the first and second Cartesian indices of :math:`\mathbf G` refer to those
points, respectively. Interfaces are concentric about the origin:

* ``sphere`` has an exterior region 0 and a homogeneous interior region 1,
  separated at :math:`r=a`.
* ``simplecavity`` has a cavity in region 1 and an exterior region 0,
  separated at :math:`r=a`.
* ``coreshell`` has exterior region 0 at :math:`r>b`, shell region 1 at
  :math:`a<r<b`, and core or cavity region 2 at :math:`r<a`, where
  :math:`0<a<b`.

For the formulas below, both points lie strictly inside the *same* homogeneous
region: outside the sphere or core-shell structure, or inside the simple
cavity or core-shell core. No shell-source or cross-region Green tensor is
asserted here. The core-shell cavity example is illustrated in
:ref:`tutorial-gf-mie`.


Radial and vector spherical waves
---------------------------------

Let :math:`Y_{nm}(\widehat{\mathbf r})` be orthonormal scalar spherical
harmonics with :math:`n\geq1` and :math:`-n\leq m\leq n`. Define

.. math::

   \mathbf\Psi_{nm}=\frac{\boldsymbol\nabla_\Omega Y_{nm}}
                           {\sqrt{n(n+1)}},\qquad
   \mathbf X_{nm}=\mathbf\Psi_{nm}\times\widehat{\mathbf r},\qquad
   \boldsymbol\nabla_\Omega
      =\widehat{\boldsymbol\theta}\,\partial_\theta
       +\widehat{\boldsymbol\phi}\,\frac{\partial_\phi}{\sin\theta}.

At a pole, the angular basis is understood through its regular limiting
combination. For a radial solution :math:`z_n(x)`, put

.. math::

   \psi_n(x)=xj_n(x),\qquad
   \xi_n(x)=xh_n^{(1)}(x),\qquad
   D_n^{(z)}(x)=\frac{d}{dx}[xz_n(x)].

The regular choice is :math:`z_n=j_n` and the outgoing choice is
:math:`z_n=h_n^{(1)}=j_n+iy_n`, appropriate to :math:`e^{-i\omega t}`.
Their Riccati functions are :math:`\psi_n` and :math:`\xi_n`;
:math:`D_n^{(j)}=\psi_n'` and :math:`D_n^{(h)}=\xi_n'`. In a homogeneous
region of wave number :math:`k`, define

.. math::

   \begin{aligned}
   \mathbf M_{nm}^{(z)}(\mathbf r;k)
      &=z_n(kr)\mathbf X_{nm}(\widehat{\mathbf r}),\\
   \mathbf N_{nm}^{(z)}(\mathbf r;k)
      &=\frac{1}{k}\boldsymbol\nabla\times
        \mathbf M_{nm}^{(z)}(\mathbf r;k)\\
      &=\frac{n(n+1)z_n(kr)}{kr\sqrt{n(n+1)}}
        Y_{nm}\widehat{\mathbf r}
        +\frac{D_n^{(z)}(kr)}{kr}
         \mathbf\Psi_{nm}(\widehat{\mathbf r}).
   \end{aligned}

Thus :math:`\mathbf M` is transverse to the radius (TE) and
:math:`\mathbf N` has a radial electric component (TM). With the chosen
:math:`\mathbf X`, :math:`\boldsymbol\nabla\times\mathbf M=k\mathbf N`
and :math:`\boldsymbol\nabla\times\mathbf N=k\mathbf M`. Other angular
phase conventions change the basis and its expansion coefficients together,
not the physical Green tensor.


Matching at spherical interfaces
--------------------------------

At every interface without free surface currents, tangential
:math:`\mathbf E` and :math:`\mathbf H` are continuous, where
:math:`\mathbf H=(i\omega\mu_0)^{-1}\boldsymbol\nabla\times\mathbf E`.
Orthogonality of the angular harmonics separates each :math:`(n,m)` and
polarization. For a unit-amplitude electric wave with radial type
:math:`z\in\{j,h\}` in region :math:`j`, define its two boundary entries at
:math:`r=R` by

.. math::

   \mathbf t_{j,n}^{\mathrm{TE},z}(R)
     =\begin{pmatrix}z_n(k_jR)\\D_n^{(z)}(k_jR)\end{pmatrix},
   \qquad
   \mathbf t_{j,n}^{\mathrm{TM},z}(R)
     =\begin{pmatrix}D_n^{(z)}(k_jR)/k_j\\
                      k_jz_n(k_jR)\end{pmatrix}.

In each polarization the first entries are proportional to tangential
:math:`\mathbf E`, and the second to tangential :math:`\mathbf H`.
The factors removed from each row are identical on both sides of a given
interface: for TE they are :math:`1` and :math:`1/(i\omega\mu_0R)`;
for TM they are :math:`1/R` and :math:`1/(i\omega\mu_0)`.
Consequently these two-component vectors can be matched directly. They are
*boundary data*, not Green-tensor coefficient formulas.

For an exterior source and observer, the wave incident on a sphere is regular
at the origin when continued to the boundary. For either polarization
:math:`P\in\{\mathrm{TE},\mathrm{TM}\}`, its scattered outgoing amplitude
:math:`R_n^P` and regular interior amplitude :math:`T_n^P` solve

.. math::

   \begin{pmatrix}
     \mathbf t_{0,n}^{P,h}(a)&-\mathbf t_{1,n}^{P,j}(a)
   \end{pmatrix}
   \begin{pmatrix}R_n^P\\T_n^P\end{pmatrix}
   =-\mathbf t_{0,n}^{P,j}(a).

Each displayed :math:`\mathbf t` is a two-entry column, so this is a
:math:`2\times2` system. For a source and observer inside a simple cavity,
the source-generated wave incident on the interface is outgoing *within the
cavity*. Its reflected wave is regular at the center; the exterior
transmitted wave is outgoing at infinity. The corresponding system is

.. math::

   \begin{pmatrix}
     \mathbf t_{1,n}^{P,j}(a)&-\mathbf t_{0,n}^{P,h}(a)
   \end{pmatrix}
   \begin{pmatrix}R_n^P\\T_n^P\end{pmatrix}
   =-\mathbf t_{1,n}^{P,h}(a).

For a core source inside a core-shell sphere, both regular and outgoing
waves are needed in the shell. Writing their amplitudes as :math:`B_n^P`
and :math:`C_n^P`, and the exterior outgoing amplitude as :math:`T_n^P`,
the core reflection amplitude is the first unknown in

.. math::

   \begin{pmatrix}
     \mathbf t_{2,n}^{P,j}(a)&-\mathbf t_{1,n}^{P,j}(a)&
       -\mathbf t_{1,n}^{P,h}(a)&\mathbf 0\\
     \mathbf 0&\mathbf t_{1,n}^{P,j}(b)&
       \mathbf t_{1,n}^{P,h}(b)&-\mathbf t_{0,n}^{P,h}(b)
   \end{pmatrix}
   \begin{pmatrix}R_n^P\\B_n^P\\C_n^P\\T_n^P\end{pmatrix}
   =\begin{pmatrix}-\mathbf t_{2,n}^{P,h}(a)\\\mathbf 0\end{pmatrix}.

The block rows contain two scalar conditions each, giving a
:math:`4\times4` system. For an exterior core-shell source, interchange the
roles of the incident and outgoing exterior waves: match a unit regular
exterior wave plus an unknown outgoing wave to regular and outgoing shell
waves at :math:`b`, and match those shell waves to a regular core wave at
:math:`a`. In all cases the response depends on :math:`n`, frequency and
polarization, but not :math:`m` for concentric isotropic layers. These
systems specify boundary matching without asserting a particular closed-form
coefficient convention.


Source-aware scattering dyadic
------------------------------

For source and observer in the same region :math:`s`, separate the direct
homogeneous-medium field from the interface response:

.. math::

   \mathbf G(\mathbf r,\mathbf r',\omega)
     =\mathbf G_{\mathrm{hom},s}(\mathbf r,\mathbf r',\omega)
      +\mathbf G_{\mathrm{Sc}}(\mathbf r,\mathbf r',\omega),
   \qquad
   \mathbf G_{\mathrm{hom},s}
     =\left(\mathbf I+\frac{\boldsymbol\nabla\boldsymbol\nabla}{k_s^2}\right)
       \frac{e^{ik_s|\mathbf r-\mathbf r'|}}
            {4\pi|\mathbf r-\mathbf r'|}.

The derivatives act on the observer coordinate, and this expression is
understood distributionally at the source. ``vacuum`` in the output names
denotes this direct term in the *source region*, even when
:math:`\varepsilon_s\ne1`; ``structure`` denotes the boundary-induced
term. In particular, the two terms should not be confused with vacuum
propagation plus a transmission tensor for points in different regions.

To state the scattering expansion without hiding its source dependence, use
:math:`\widetilde{\mathbf M}_{nm}` and
:math:`\widetilde{\mathbf N}_{nm}` for the **source angular duals** of the
observer functions :math:`\mathbf M_{nm}` and :math:`\mathbf N_{nm}`.
For the complex spherical-harmonic convention above, the dual replaces
:math:`Y_{nm}`, :math:`\mathbf\Psi_{nm}`, and :math:`\mathbf X_{nm}` at
:math:`\widehat{\mathbf r'}` by their complex conjugates; it does *not*
conjugate :math:`k_s` or the radial functions. The tensor product below
therefore maps a source dipole vector to an observer field, rather than
multiplying two observer waves:

.. math::

   \begin{aligned}
   \mathbf G_{\mathrm{Sc}}(\mathbf r,\mathbf r',\omega)
    =\sum_{n=1}^{\infty}\sum_{m=-n}^{n}\big[&
       C_n^{\mathrm{TE}}(\omega)
       \mathbf M_{nm}^{(f)}(\mathbf r;k_s)\otimes
       \widetilde{\mathbf M}_{nm}^{(g)}(\mathbf r';k_s)\\
      +&C_n^{\mathrm{TM}}(\omega)
       \mathbf N_{nm}^{(f)}(\mathbf r;k_s)\otimes
       \widetilde{\mathbf N}_{nm}^{(g)}(\mathbf r';k_s)
       \big].
   \end{aligned}

For an exterior pair :math:`(f,g)=(h,h)`: the interface response is outgoing
at the observer and its source projection uses an outgoing radial wave.
For a cavity or core pair :math:`(f,g)=(j,j)`: the reflected field is regular
at the observer and its source projection uses a regular radial wave. In
either case :math:`C_n^P` includes the boundary reflection amplitude, the
homogeneous Green-function source normalization, and any angular phase or
normalization associated with the chosen dual. No numerical prefactor is
implied by the displayed expansion. A source dipole :math:`\mathbf p`
enters through contraction of the *second* index with :math:`\mathbf p`;
the code assembles the Cartesian tensor from three independent source
orientations. The incident direct wave is accounted for only in
:math:`\mathbf G_{\mathrm{hom},s}`, not again in this sum.


Truncation, coincidence, and use
--------------------------------

Numerical evaluation replaces the infinite sum by :math:`1\leq n\leq
n_{\max}`. Increase :math:`n_{\max}` until the tensor components or the
projected observable of interest stabilize at each frequency and position.
When either point approaches an interface, or near narrow resonances, higher
multipoles may matter; a fixed order does not guarantee
convergence. The :math:`m` sum retains all :math:`-n\leq m\leq n`.

At coincidence, :math:`\mathbf G_{\mathrm{hom},s}` has a singular
point-source part. The scattering part is finite for a source strictly away
from an interface when the multipole series converges. An LDOS or Purcell
calculation uses a specified regularized coincident homogeneous contribution
together with :math:`\operatorname{Im}\mathbf G_{\mathrm{Sc}}(\mathbf r',
\mathbf r',\omega)`; it is not the literal finite value of the full
unregularized tensor. The implementation supplies the homogeneous
:math:`ik_s\mathbf I/(6\pi)` term at coincidence. For lossy hosts, or a
point on an interface, the coincident limit needs further physical care.

The imaginary part of the projected Green tensor governs local emission
properties, while its off-diagonal projection enters dipole-dipole
coupling. Resonant structure in these quantities can reflect spherical
interface modes, as discussed for energy transfer in [Lee2020]_. Spherical
geometry has no lateral translational symmetry: a ``scan`` fixes one source
and lists explicit observer positions, whereas ``pair`` evaluates emitter
position pairs. See :ref:`tutorial-gf-mie` for the core-shell cavity example
and HDF5 output.

The implementation is in :mod:`mqed.Dyadic_GF.GF_Mie` and is driven by
:mod:`mqed.Dyadic_GF.main_mie`. It is experimental and should be benchmarked
against literature or independent methods for each new physical regime; no
literature validation is claimed here. The current core-shell scan path is
intended for source and observer inside the core/cavity. Exterior pairs are
implemented for a sphere and core-shell sphere, and cavity pairs for a simple
cavity. Shell-region observation and core-shell cross-region transmission
are not supported production paths; a zero structure placeholder must not
be interpreted as a physical Green tensor.


References
----------

.. [Lee2020] M. W. Lee and L. Y. Hsu, "Controllable frequency dependence of
   resonance energy transfer coupled with localized surface plasmon polaritons,"
   *J. Phys. Chem. Lett.* **11**, 6796--6804 (2020).

.. [Bohren1983] C. F. Bohren and D. R. Huffman, *Absorption and Scattering of
   Light by Small Particles* (Wiley, 1983).

.. [Chew1995] W. C. Chew, *Waves and Fields in Inhomogeneous Media* (IEEE
    Press, 1995).

.. [Suttorp2011] L. G. Suttorp and A. J. van Wonderen, "Quantized media with
   absorptive scatterers and modified atomic emission rates," *Opt. Commun.*
   **284**, 2943 (2011), doi:`10.1016/j.optcom.2011.02.036
   <https://doi.org/10.1016/j.optcom.2011.02.036>`_.
