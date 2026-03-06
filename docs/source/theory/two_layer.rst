.. _theory-two-layer:

Two-Layer Planar Geometry
==========================

This section derives the scattering dyadic Green's function for a planar
two-layer geometry — the simplest nontrivial system in which a dielectric
interface modifies the electromagnetic field.  We state the Fresnel reflection
coefficients for *s*- and *p*-polarised waves, then give the resulting
scattering Green's function as Sommerfeld integrals in cylindrical coordinates.
The mathematical discussion follows [Wu2018]_; a full step-by-step derivation
(boundary conditions, coordinate rotation, Bessel-function identities) can be
found in [Novotny2012]_ Ch. 10 and [Sarabandi]_.  A detailed derivation is
also included in :download:`this PDF </_static/two_layer/two_layer_theory.pdf>`.


Geometry
--------

.. figure:: /_static/two_layer/Two_Layer.png
   :align: center
   :width: 80%

   Two-layer planar geometry.  The source (donor) is in medium *i* above the
   interface and the field point (acceptor) is at lateral separation
   :math:`\rho`.

The source (donor) is at :math:`\mathbf{r}' = (0, 0, z')` in medium *i*
(above the interface) and the field point (acceptor) is at
:math:`\mathbf{r} = (\rho, \varphi, z)`, also in medium *i*.  The interface
sits at :math:`z = 0`.


Fresnel Reflection Coefficients
---------------------------------

Applying the electromagnetic boundary conditions (tangential
:math:`\mathbf{E}` and :math:`\mathbf{H}` continuous across the interface)
to *s*- and *p*-polarised plane waves yields the Fresnel reflection
coefficients.

Define the perpendicular wave vector in medium :math:`l`:

.. math::
   :label: eq-Kz

   K_{z,l}(\omega, k_\rho)
   = \sqrt{\varepsilon_{r,l}(\omega)\,\frac{\omega^2}{c^2} - k_\rho^2},
   \qquad l = i, j,

where :math:`k_\rho` is the in-plane wave vector magnitude and the branch is
chosen so that :math:`\mathrm{Im}\,K_{z,l} \geq 0` (outgoing/evanescent
waves).

**s-polarisation** (TE — electric field parallel to the interface):

.. math::
   :label: eq-Rs

   R_s(k_\rho, \omega)
   = \frac{K_{z,i} - K_{z,j}}{K_{z,i} + K_{z,j}}.

**p-polarisation** (TM — magnetic field parallel to the interface):

.. math::
   :label: eq-Rp

   R_p(k_\rho, \omega)
   = \frac{\varepsilon_{r,j}\,K_{z,i} - \varepsilon_{r,i}\,K_{z,j}}
          {\varepsilon_{r,j}\,K_{z,i} + \varepsilon_{r,i}\,K_{z,j}}.


.. _theory-scattering-gf:

Scattering Green's Function
------------------------------

The total dyadic Green's function decomposes as

.. math::

   \overline{\overline{\mathbf{G}}}(\mathbf{r}, \mathbf{r}', \omega)
   = \overline{\overline{\mathbf{G}}}_0(\mathbf{r}, \mathbf{r}', \omega)
   + \overline{\overline{\mathbf{G}}}_\mathrm{Sc}(\mathbf{r}, \mathbf{r}', \omega),

where :math:`\overline{\overline{\mathbf{G}}}_0` is the closed-form vacuum
Green's function and
:math:`\overline{\overline{\mathbf{G}}}_\mathrm{Sc}` encodes the effect of
the interface.

Vacuum Green's Function
^^^^^^^^^^^^^^^^^^^^^^^^

The free-space dyadic Green's function has the closed-form analytical
expression

.. math::

   \overline{\overline{\mathbf{G}}}_0(\mathbf{r}_\alpha,\mathbf{r}_\beta,\omega)
   = \frac{e^{ik_0 R_{\alpha\beta}}}{4\pi R_{\alpha\beta}}
   \Big[
   \left(\overline{\overline{\mathbf{I}}}_3
         - \mathbf{e}_\mathrm{R}\mathbf{e}_\mathrm{R}\right)
   + \left(3\mathbf{e}_\mathrm{R}\mathbf{e}_\mathrm{R}
         - \overline{\overline{\mathbf{I}}}_3\right)
   \left(\frac{1}{(k_0 R_{\alpha\beta})^{2}}
         - \frac{i}{k_0 R_{\alpha\beta}}\right)
   \Big],

where :math:`R_{\alpha\beta} = |\mathbf{r}_\alpha - \mathbf{r}_\beta|` is the
distance between the source and field points,
:math:`\mathbf{e}_\mathrm{R} = (\mathbf{r}_\alpha - \mathbf{r}_\beta)/R_{\alpha\beta}`
is the unit vector along that direction,
:math:`\overline{\overline{\mathbf{I}}}_3` is the :math:`3\times 3` identity
tensor, and :math:`k_0 = \omega/c` is the free-space wave number.

Scattering Part
^^^^^^^^^^^^^^^^

In cylindrical coordinates the scattering part is given by the
Sommerfeld integral:

.. math::

   \overline{\overline{\mathbf{G}}}_\mathrm{Sc}(\rho, \varphi, z, z', \omega)
   = \int_{0}^{+\infty} \frac{i\,dk_{\rho}}{4\pi}
   \Big[ R_s\,\overline{\overline{\mathbf{M}}}^{(s)}
       + R_p\,\overline{\overline{\mathbf{M}}}^{(p)} \Big]\,
   e^{iK_{z,i}(k_{\rho},\,\omega)(z+z')}.


Sommerfeld Integrals
^^^^^^^^^^^^^^^^^^^^^

Both matrices are built from six Sommerfeld integrals over the in-plane
wave vector :math:`k_\rho`, each involving the Fresnel coefficient, Bessel
functions :math:`J_n`, and the phase factor
:math:`e^{i K_{z,i}(z + z')}`:

.. math::
   :label: eq-sommerfeld

   \begin{aligned}
   I_1 &= \int_0^\infty \frac{R_s\,k_\rho}{2\,K_{z,i}}\;
           J_0(k_\rho\rho)\;e^{iK_{z,i}(z+z')}\,dk_\rho, \\[4pt]
   I_2 &= \int_0^\infty \frac{R_s\,k_\rho}{2\,K_{z,i}}\;
           J_2(k_\rho\rho)\;e^{iK_{z,i}(z+z')}\,dk_\rho, \\[4pt]
   I_3 &= \int_0^\infty \frac{R_p\,k_\rho\,K_{z,i}}{2\,k_0^2}\;
           J_0(k_\rho\rho)\;e^{iK_{z,i}(z+z')}\,dk_\rho, \\[4pt]
   I_4 &= \int_0^\infty \frac{R_p\,k_\rho\,K_{z,i}}{2\,k_0^2}\;
           J_2(k_\rho\rho)\;e^{iK_{z,i}(z+z')}\,dk_\rho, \\[4pt]
   I_5 &= \int_0^\infty \frac{i\,R_p\,k_\rho^2}{k_0^2}\;
           J_1(k_\rho\rho)\;e^{iK_{z,i}(z+z')}\,dk_\rho, \\[4pt]
   I_6 &= \int_0^\infty \frac{R_p\,k_\rho^3}{K_{z,i}\,k_0^2}\;
           J_0(k_\rho\rho)\;e^{iK_{z,i}(z+z')}\,dk_\rho,
   \end{aligned}

where :math:`k_0 = \omega/c`.  In the implementation
(:mod:`mqed.Dyadic_GF.GF_Sommerfeld`) the integration is split at
:math:`k_\rho = k_0` (the propagating/evanescent boundary) for numerical
accuracy and evaluated with :func:`scipy.integrate.quad_vec`.


:math:`\mathbf{M}^{(s)}` — s-wave contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
   :label: eq-Ms

   \mathbf{M}^{(s)} =
   \begin{pmatrix}
   I_1 + \cos 2\varphi\;I_2  &  \sin 2\varphi\;I_2       &  0 \\
   \sin 2\varphi\;I_2        &  I_1 - \cos 2\varphi\;I_2  &  0 \\
   0                          &  0                          &  0
   \end{pmatrix}.

The *s*-wave has no :math:`z`-component because the electric field lies
entirely in the interface plane.


:math:`\mathbf{M}^{(p)}` — p-wave contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
   :label: eq-Mp

   \mathbf{M}^{(p)} =
   \begin{pmatrix}
   -I_3 + \cos 2\varphi\;I_4  &  \sin 2\varphi\;I_4        &  -\cos\varphi\;I_5 \\
   \sin 2\varphi\;I_4         &  -I_3 - \cos 2\varphi\;I_4  &  -\sin\varphi\;I_5 \\
   \cos\varphi\;I_5           &  \sin\varphi\;I_5            &  I_6
   \end{pmatrix}.

The *p*-wave carries all the :math:`z`-components of the scattered field.


Implementation
--------------

The class :class:`mqed.Dyadic_GF.GF_Sommerfeld.Greens_function_analytical`
evaluates these integrals for arrays of energies and lateral separations
:math:`\rho`.  See the :doc:`/tutorials/GF_Sommerfeld` tutorial for usage
and the :ref:`config-dyadic-gf` section for all available configuration
parameters.


References
----------

.. [Wu2018] J. S. Wu, Y. C. Lin, Y. L. Sheu, and L. Y. Hsu,
   "Characteristic distance of resonance energy transfer coupled with
   surface plasmon polaritons,"
   *J. Phys. Chem. Lett.* **9**, 7032–7039 (2018).

.. [Novotny2012] L. Novotny and B. Hecht, *Principles of Nano-Optics*,
   2nd ed. (Cambridge University Press, 2012), Ch. 10.

.. [Sarabandi] K. Sarabandi, "Dyadic Green's function," Lecture Notes,
   University of Michigan.
