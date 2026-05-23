.. _theory-macroscopic-qed:

===========================
Macroscopic QED Framework
===========================

This section summarizes the macroscopic quantum electrodynamics (MQED) framework
implemented in this package. For detailed derivations, see our manuscript
[Liu2026]_.

Overview
--------

We consider an ensemble of :math:`N_{\mathrm{M}}` quantum subsystems (e.g., molecules)
coupled to quantized electromagnetic fields in arbitrary dielectric environments.
The total Hamiltonian is

.. math::

   \hat{H} = \hat{H}_\mathrm{M} + \hat{H}_\mathrm{P} + \hat{V}_\mathrm{MP},

where :math:`\hat{H}_\mathrm{M}` describes the matter (two-level subsystems),
:math:`\hat{H}_\mathrm{P}` describes the quantized electromagnetic field dressed by
the dielectric medium, and :math:`\hat{V}_\mathrm{MP}` is the light--matter coupling
in the multipolar gauge. The key distinction from standard cavity QED is that the
bosonic field operators :math:`\hat{\mathbf{f}}(\mathbf{r},\omega)` create and
annihilate photons *dressed* by the dielectric medium, rather than vacuum photons.
We refer the reader to [Liu2025]_ for the full derivation of each term.


The Dyadic Green's Function
----------------------------

The central quantity in the MQED framework is the **dyadic Green's function**
:math:`\overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega)`,
which satisfies the macroscopic Maxwell's equation in the frequency domain:

.. math::

   \left( \frac{\omega^2}{c^2}\epsilon_\mathrm{r}(\mathbf{r}_\alpha,\omega)
   - \boldsymbol{\nabla} \times \boldsymbol{\nabla} \times \right)
   \overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega)
   = -\delta(\mathbf{r}_\alpha - \mathbf{r}_\beta)\,
   \overline{\overline{\mathbf{I}}}_3,

where :math:`\epsilon_\mathrm{r}(\mathbf{r},\omega)` is the relative permittivity and
:math:`\overline{\overline{\mathbf{I}}}_3` is the :math:`3\times 3` identity tensor.

The dyadic Green's function encodes the spatial propagation of dressed photons in
linear, inhomogeneous, dispersive, and absorbing dielectric environments.
**Once the dyadic Green's function is determined, all electromagnetic properties of
the dielectric environment that are relevant to the quantum subsystems—including
energy transfer rates, spontaneous emission modification, and spectral
shifts—can be computed.** It is the single object that bridges classical
electrodynamics to the quantum dynamics of molecules near dielectric structures.


Vacuum and Scattering Contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since the macroscopic Maxwell's equation is linear, the dyadic Green's function
decomposes into vacuum and scattering contributions:

.. math::

   \overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega)
   = \overline{\overline{\mathbf{G}}}_0(\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega)
   + \overline{\overline{\mathbf{G}}}_\mathrm{Sc}(\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega).

The vacuum Green's function :math:`\overline{\overline{\mathbf{G}}}_0` has a known
closed-form expression, while the **scattering Green's function**
:math:`\overline{\overline{\mathbf{G}}}_\mathrm{Sc}` encodes all effects of the
dielectric environment. Determining the scattering Green's function is almost always
the key computational challenge in the MQED framework. This package provides solvers
based on Sommerfeld integrals (for planar geometries) and boundary element methods
(BEM, for arbitrary nanostructures).


Quantum Master Equation
------------------------

By tracing out the photonic degrees of freedom from the Heisenberg equation of
motion, one obtains a quantum master equation for the reduced density matrix
:math:`\hat{\rho}_\mathrm{M}(t)` of the molecular subsystems. Under the Markov
approximation (valid in the weak light--matter coupling regime where the
generalized spectral density varies smoothly near the molecular transition
frequency :math:`\omega_\mathrm{M}`), the master equation takes the Lindblad form:

.. math::

   \frac{\partial}{\partial t} \hat{\rho}_\mathrm{M}(t)
   = -\frac{i}{\hbar}\left[\hat{H}_\mathrm{M}
     + \hat{H}_\mathrm{CP}
     + \hat{H}_\mathrm{DDI},\;
     \hat{\rho}_\mathrm{M}(t)\right]
   + \sum_{\alpha,\beta}^{N_\mathrm{M}} \Gamma_{\alpha\beta}
     \left(\hat{\sigma}_\beta^{-}\,\hat{\rho}_\mathrm{M}(t)\,\hat{\sigma}_\alpha^{+}
     - \frac{1}{2}\left\{\hat{\sigma}_\alpha^{+}\hat{\sigma}_\beta^{-},\;
     \hat{\rho}_\mathrm{M}(t)\right\}\right).

The coherent and dissipative parts of this equation are entirely determined by the
dyadic Green's function through two key quantities: the **dipole--dipole interaction**
:math:`V_{\alpha\beta}` and the **generalized dissipation rate**
:math:`\Gamma_{\alpha\beta}`, described below.


.. _theory-ddi:

Dipole--Dipole Interaction :math:`V_{\alpha\beta}`
----------------------------------------------------

The dipole--dipole interaction (DDI) governs the **coherent** energy exchange
between quantum subsystems :math:`\alpha` and :math:`\beta`. It enters the
master equation through the DDI Hamiltonian:

.. math::

   \hat{H}_\mathrm{DDI} = \sum_{\alpha \neq \beta}^{N_\mathrm{M}}
   V_{\alpha\beta}\,\hat{\sigma}_\alpha^{+}\hat{\sigma}_\beta^{-},

where the DDI coupling strength is given by the **real part** of the dyadic Green's
function:

.. math::
   :label: eq-ddi

   V_{\alpha\beta}
   = \frac{-\omega_\mathrm{M}^2}{\epsilon_0 c^2}\,
   \boldsymbol{\mu}_\alpha \cdot
   \mathrm{Re}\,\overline{\overline{\mathbf{G}}}
   (\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega_\mathrm{M})
   \cdot \boldsymbol{\mu}_\beta.

Here :math:`\boldsymbol{\mu}_\alpha = \mu_\alpha \mathbf{n}_\alpha` is the
transition dipole moment of subsystem :math:`\alpha`.
The DDI strength is sensitive to the separation, relative orientation, and
the dielectric environment. The ratio :math:`V_{\alpha\beta}/V_{\alpha\beta}^{(0)}`
(where the superscript :math:`(0)` denotes the vacuum value) quantifies the
**field enhancement** due to the dielectric structure—this is precisely what
is computed in the :doc:`/tutorials/field_enhancement` tutorial.


.. _theory-generalized-dissipation:

Generalized Dissipation Rate :math:`\Gamma_{\alpha\beta}`
-----------------------------------------------------------

The generalized dissipation rate governs the **incoherent** (dissipative) dynamics
and is given by the **imaginary part** of the dyadic Green's function:

.. math::
   :label: eq-gamma

   \Gamma_{\alpha\beta}
   = \frac{2\omega_\mathrm{M}^2}{\hbar\epsilon_0 c^2}\,
   \boldsymbol{\mu}_\alpha \cdot
   \mathrm{Im}\,\overline{\overline{\mathbf{G}}}
   (\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega_\mathrm{M})
   \cdot \boldsymbol{\mu}_\beta.

The **diagonal** elements :math:`\Gamma_{\alpha\alpha}` describe the spontaneous
emission rate of subsystem :math:`\alpha` modified by the dielectric environment.
The ratio to the free-space rate gives the **Purcell factor**:

.. math::

   \frac{\Gamma_{\alpha\alpha}}{\Gamma_0}
   = \frac{6\pi c}{\omega_\mathrm{M}}\,
   \mathbf{n}_\alpha \cdot
   \mathrm{Im}\,\overline{\overline{\mathbf{G}}}
   (\mathbf{r}_\alpha, \mathbf{r}_\alpha, \omega_\mathrm{M})
   \cdot \mathbf{n}_\alpha.

The **off-diagonal** elements :math:`\Gamma_{\alpha\beta}` (:math:`\alpha \neq \beta`)
describe **cooperative dissipation** (super- and sub-radiance) between subsystems.
The ratio :math:`\Gamma_{\alpha\beta}/\Gamma_{\alpha\beta}^{(0)}` quantifies
environment-induced modification of cooperative decay, which is also computed in the
:doc:`/tutorials/field_enhancement` tutorial.


Casimir--Polder Potential
--------------------------

The Casimir--Polder (CP) potential describes the transition energy shift of a single
subsystem induced by the dielectric environment:

.. math::

   \Lambda_\alpha^\mathrm{Sc}
   = \mathcal{P}\int_0^\infty d\omega\;
   \frac{\omega^2}{\pi\varepsilon_0 c^2}
   \left(\frac{1}{\omega + \omega_\mathrm{M}}
   - \frac{1}{\omega - \omega_\mathrm{M}}\right)
   \boldsymbol{\mu}_\alpha \cdot
   \mathrm{Im}\,\overline{\overline{\mathbf{G}}}_\mathrm{Sc}
   (\mathbf{r}_\alpha, \mathbf{r}_\alpha, \omega)
   \cdot \boldsymbol{\mu}_\alpha,

where :math:`\mathcal{P}` denotes the Cauchy principal value. This shift depends on
the scattering Green's function evaluated at the same spatial point, integrated over
all frequencies.


.. _theory-resonance-enhancement:

Resonance Energy Transfer Enhancement Factor
----------------------------------------------

The quantities :math:`V_{\alpha\beta}` and :math:`\Gamma_{\alpha\beta}` characterize
the coherent and incoherent channels separately. A single scalar that captures the
**overall enhancement of resonance energy transfer** (RET) by the dielectric
environment is the **enhancement factor** :math:`\gamma`:

.. math::
   :label: eq-enhancement-factor

   \gamma
   = \left|
   \frac{\boldsymbol{\mu}_\alpha \cdot
   \overline{\overline{\mathbf{G}}}
   (\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega_\mathrm{M})
   \cdot \boldsymbol{\mu}_\beta}
   {\boldsymbol{\mu}_\alpha \cdot
   \overline{\overline{\mathbf{G}}}_0
   (\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega_\mathrm{M})
   \cdot \boldsymbol{\mu}_\beta}
   \right|^2.

This is the squared modulus of the ratio between the **total** (dielectric-dressed)
and **vacuum** Green's functions projected onto the donor--acceptor dipole
orientations. Because the RET rate is proportional to
:math:`|V_{\alpha\beta}|^2 + |\Gamma_{\alpha\beta}|^2` (coherent plus incoherent
channels), :math:`\gamma` directly measures how much the dielectric environment
enhances or suppresses the energy transfer rate relative to free space.

In the implementation (:mod:`mqed.utils.enhancement`), this is computed as:

.. code-block:: python

   gamma = np.abs(g_da_total / g_da_vac) ** 2

where ``g_da_total`` and ``g_da_vac`` are the orientation-projected Green's functions
in the dielectric environment and in vacuum, respectively.

The enhancement factor is computed by the ``mqed_RET`` command; see the
:ref:`tutorial-ret` section of the :doc:`/tutorials/field_enhancement` tutorial.


Summary
--------

The table below summarizes the key physical quantities and their relation to the
dyadic Green's function:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Quantity
     - Physical role
     - Green's function component
   * - :math:`V_{\alpha\beta}` :eq:`eq-ddi`
     - Coherent dipole--dipole coupling
     - :math:`\mathrm{Re}\,\overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega_\mathrm{M})`
   * - :math:`\Gamma_{\alpha\beta}` :eq:`eq-gamma`
     - Generalized dissipation / cooperative decay
     - :math:`\mathrm{Im}\,\overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha, \mathbf{r}_\beta, \omega_\mathrm{M})`
   * - :math:`\Gamma_{\alpha\alpha}/\Gamma_0`
     - Purcell factor (emission enhancement)
     - :math:`\mathrm{Im}\,\overline{\overline{\mathbf{G}}}(\mathbf{r}_\alpha, \mathbf{r}_\alpha, \omega_\mathrm{M})`
   * - :math:`\Lambda_\alpha^\mathrm{Sc}`
     - Casimir--Polder energy shift
     - :math:`\mathrm{Im}\,\overline{\overline{\mathbf{G}}}_\mathrm{Sc}(\mathbf{r}_\alpha, \mathbf{r}_\alpha, \omega)` (frequency integral)
   * - :math:`\gamma` :eq:`eq-enhancement-factor`
     - RET enhancement factor
     - :math:`\left|\overline{\overline{\mathbf{G}}}/\overline{\overline{\mathbf{G}}}_0\right|^2` (projected, squared modulus)

In summary, **the dyadic Green's function is the single central object that fully
characterizes how a dielectric environment modifies the quantum dynamics of nearby
emitters.** Computing it accurately is the primary computational task of this package,
and all physical observables—DDI, dissipation, Purcell enhancement, and
Casimir--Polder shifts—follow directly from it.

.. seealso::

   - :doc:`/tutorials/field_enhancement` — compute :math:`V_{\alpha\beta}` and
     :math:`\Gamma_{\alpha\beta}` ratios for a planar dielectric interface.
   - :doc:`/tutorials/GF_Sommerfeld` — compute dyadic Green's functions via
     Sommerfeld integrals for planar geometries.


References
----------

.. [Liu2026] G. Liu *et al.*, "Liu, G., Wang, S. and Chen, H.T., 2026. MQED-QD: An Open-Source Package for 
Quantum Dynamics Simulation in Complex Dielectric Environments. Journal of Chemical Theory and Computation."