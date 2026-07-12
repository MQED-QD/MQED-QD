.. _theory-mie:

Mie Theory for Spherical Cavities
=================================

This section gives a compact overview of the spherical Green-function workflow
used by the Mie driver.  The implementation follows generalized Mie theory for
spherical particles and cavities; the full derivation is outside the scope of
this documentation.  For detailed formulas and physical discussion, see
[Lee2020]_ and standard texts on electromagnetic scattering [Bohren1983]_,
[Chew1995]_.


Geometry
--------

Mie theory is useful when the dielectric environment is radially layered rather
than planar.  The current workflow supports a homogeneous sphere, a simple
spherical cavity, and a core-shell sphere.  In the core-shell case, the regions
are ordered by radius:

* exterior medium,
* shell material,
* core or cavity material.

The dyadic Green tensor still decomposes into vacuum and structure-mediated
parts,

.. math::

   \mathbf{G}(\mathbf r,\mathbf r',\omega)
   = \mathbf{G}_0(\mathbf r,\mathbf r',\omega)
   + \mathbf{G}_\mathrm{Sc}(\mathbf r,\mathbf r',\omega),

but the scattering term is expanded in spherical rather than cylindrical waves.


Spherical-wave expansion
------------------------

The electromagnetic field of a point dipole can be expanded in vector spherical
wave functions.  Each multipole order :math:`n` and azimuthal index :math:`m`
has electric and magnetic components built from spherical Bessel or Hankel
functions and angular functions related to associated Legendre functions.  The
dielectric boundaries determine Mie coefficients by enforcing continuity of the
tangential electric and magnetic fields at each spherical interface.

Schematically,

.. math::

   \mathbf{G}_\mathrm{Sc}
   = \sum_{n=1}^{n_\mathrm{max}}\sum_{m=-n}^{n}
   \left(
     A_{nm}\,\mathbf{M}_{nm}\mathbf{M}_{nm}
     + B_{nm}\,\mathbf{N}_{nm}\mathbf{N}_{nm}
   \right),

where :math:`\mathbf{M}_{nm}` and :math:`\mathbf{N}_{nm}` are vector spherical
wave functions and :math:`A_{nm}`, :math:`B_{nm}` represent the appropriate Mie
response coefficients.  The truncation order :math:`n_\mathrm{max}` controls the
number of multipoles included.


Physical interpretation
-----------------------

The Mie coefficients encode localized surface-plasmon and cavity resonances of
the spherical structure.  Near resonance, the imaginary part of the Green tensor
can vary strongly with frequency and position.  In macroscopic QED this directly
modifies the spectral density, resonance energy transfer, and Purcell factor.
This is the mechanism discussed in [Lee2020]_ for frequency-dependent energy
transfer coupled to localized surface plasmon polaritons.

Unlike planar Sommerfeld solvers, spherical Mie calculations do not have
horizontal translational symmetry.  A scan output means one fixed source and a
list of explicit observer positions.  A pair output means all
:math:`\mathbf{G}(\mathbf r_i,\mathbf r_j)` tensors for an emitter list.


Implementation notes
--------------------

The implementation is in :mod:`mqed.Dyadic_GF.GF_Mie` and is driven by
:mod:`mqed.Dyadic_GF.main_mie`.  The tutorial :ref:`tutorial-gf-mie` explains
the core-shell spherical-cavity example, the ``scan`` HDF5 layout, and the
spectral-density workflow.

The code is intended as a practical generalized-Mie workflow for MQED studies,
but it should be benchmarked for each new physical regime.  In particular, the
current core-shell scan path is intended for source and observer points inside
the core/cavity; shell-region observation points are not yet a validated
production path.


References
----------

.. [Lee2020] M. W. Lee and L. Y. Hsu, "Controllable frequency dependence of
   resonance energy transfer coupled with localized surface plasmon polaritons,"
   *J. Phys. Chem. Lett.* **11**, 6796--6804 (2020).

.. [Bohren1983] C. F. Bohren and D. R. Huffman, *Absorption and Scattering of
   Light by Small Particles* (Wiley, 1983).

.. [Chew1995] W. C. Chew, *Waves and Fields in Inhomogeneous Media* (IEEE
   Press, 1995).
