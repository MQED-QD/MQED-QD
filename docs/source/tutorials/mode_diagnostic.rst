.. _tutorial-mode-diagnostic:

=======================================
Diagnosing modes in a planar DBR stack
=======================================

This tutorial uses two independent commands: first sample reflection and modal
characteristics into a compressed NumPy file, then plot that file. The maps
help locate *candidate* features for closer study. A bright stripe, a large
reflection amplitude, a small real-axis denominator, or a reported pole is not
proof of a Bloch surface wave (BSW).

Run the commands from the repository root in an environment with ``mqed``
installed. Neither command runs the N-layer Sommerfeld integral or produces a
Green tensor. The diagnostic reads a YAML file directly with
``OmegaConf.load``; it does **not** compose Hydra ``defaults`` or resolve
Hydra-only interpolations. The existing local config below is directly
loadable despite its unused ``defaults`` and ``hydra`` entries: the scan needs
its explicit ``materials``, ``stack.layers`` and
``simulation.energy_eV.min/max/points``. In particular, this command does not
use ``simulation.integration`` or ``parallel`` settings. A Hydra-composed
wrapper or a segmented-only energy grid is not a substitute for this file.

Use the existing silver-backed DBR stack
========================================

Use the already present file (do not create or edit a new YAML):

.. code-block:: text

   local/configs/Dyadic_GF/infinite_horizontal_singularity/DBR_silver_z_1nm_emission.yaml

Its active layers are ordered **bottom to top**, with zero-based indices:

.. list-table:: Stack in the existing config
   :header-rows: 1
   :widths: 18 48 34

   * - Layer
     - Material / region
     - Thickness
   * - 0
     - Silver substrate (``silver_BEM`` optical data)
     - Semi-infinite
   * - 1--8
     - Four ZnS/MgF2 pairs (ZnS first, MgF2 second)
     - 90 nm ZnS / 148 nm MgF2 per pair
   * - 9
     - ``DBP_emitter_spacer``, constant :math:`\epsilon=3+0j`
     - 20 nm
   * - 10
     - Top air, :math:`\epsilon=1+0j`
     - Semi-infinite

This is a **silver-backed dielectric DBR**, not an all-dielectric stack.
``stack.source_layer: 9`` selects the finite DBP incident/source region by
default. The config's ``simulation.position.zD_nm: 1.0`` and ``zA_nm: 1.0``
are distances from the **lower DBP interface** inside that 20 nm layer for a
later Green-function calculation. If placing an emitter instead in top air
(source layer 10), its height is nonnegative and measured **above the top
DBP/air interface**. The mode diagnostic reads neither emitter position nor
orientation; changing ``--source-layer`` does not move an emitter.

The existing YAML sets ``simulation.energy_eV`` to 1.0--3.0 eV with **201**
points. If omitted, ``--energy-min``, ``--energy-max`` and ``--energy-points``
each take their values from that config (201 points across this interval),
not from a Hydra-composed grid. The shorter 41-point grid below is an explicit
quick-look override; refine it before analyzing a narrow feature. The
silver data are loaded from the configured package spreadsheet, so that
material data must be available in the installed environment.

Stage 1: sample a modest real-q grid
====================================

.. code-block:: bash

   python -m mqed.Dyadic_GF.mode_diagnostic \
     --config local/configs/Dyadic_GF/infinite_horizontal_singularity/DBR_silver_z_1nm_emission.yaml \
     --source-layer 9 \
     --energy-min 1.0 --energy-max 3.0 --energy-points 41 \
     --q-min 0.0 --q-max 5.0 --q-points 101 \
     --output /tmp/dbr_dbp9_modes.npz

This samples 41 energies and 101 values of :math:`q/k_0`. It computes the
complex generalized ``down`` and ``up`` reflection amplitudes seen from
finite layer 9 and the finite-source Airy characteristic
:math:`D_\sigma=1-r_{\mathrm{down},\sigma}r_{\mathrm{up},\sigma}
\exp(2 i k_{z,9}d_9)` for TE and TM separately. A small value on the real
grid is only a screening signal, especially in a lossy or radiative stack.
The console reports how many samples are finite. ``--source-layer`` overrides
``stack.source_layer`` in this invocation without editing the YAML.

To view the same stack *from top air*, repeat with index 10 and a separate
output. This is a different incident region, not a change in stack geometry:

.. code-block:: bash

   python -m mqed.Dyadic_GF.mode_diagnostic \
     --config local/configs/Dyadic_GF/infinite_horizontal_singularity/DBR_silver_z_1nm_emission.yaml \
     --source-layer 10 \
     --energy-min 1.0 --energy-max 3.0 --energy-points 41 \
     --q-min 0.0 --q-max 5.0 --q-points 101 \
     --output /tmp/dbr_air10_modes.npz

Top air has no neighboring layer above it. Its ``up`` reflection is therefore
invalid, and its saved ``characteristic`` is a *downward recursive reflection
denominator*, **not** the finite-cavity Airy denominator used for layer 9.
Do not compare the two raw denominator magnitudes as if they had identical
normalization or physical meaning.

Stage 2: plot either NPZ file
=============================

.. code-block:: bash

   python -m mqed.plotting.plot_mode_diagnostic \
     /tmp/dbr_dbp9_modes.npz --direction down \
     --output /tmp/dbr_dbp9_down.png

   python -m mqed.plotting.plot_mode_diagnostic \
     /tmp/dbr_air10_modes.npz --output /tmp/dbr_air10_down.png

The plot command takes the NPZ as its positional ``input``. Its only optional
switch is ``--direction`` (``down`` by default, or ``up``); ``--output`` is
required. For layer 9, ``--direction up`` displays the DBP-to-air view. For
layer 10, ``up`` contains no valid reflection samples, so use ``down``. Each
image has a TE (``s``) panel and a TM (``p``) panel. Color is
:math:`\log_{10}|r|` for the **amplitude** reflection coefficient, not
power reflectance :math:`|r|^2`. Invalid or zero-amplitude values are masked.
The dashed air light line is :math:`q/k_0=1`. A dotted source-layer light
line appears only where its saved permittivity is real and positive, at
:math:`q/k_0=\sqrt{\epsilon_{\mathrm{source}}}`; in the actual DBP config
this is :math:`\sqrt{3}\simeq 1.732`, **not** 1.5. In top air the two lines
coincide. A lossy or metallic
source has no simple real source-layer light line, so the plot omits it.

Read the saved data
===================

The compressed ``.npz`` stores complex arrays, metadata and explicit masks.
Let ``M`` be the number of energies and ``N`` the number of momenta:

.. list-table:: NPZ fields
   :header-rows: 1
   :widths: 30 20 50

   * - Keys
     - Shape
     - Meaning
   * - ``energy_eV``, ``q_over_k0``
     - ``(M,)``, ``(N,)``
     - Strictly ascending energy in eV and nonnegative dimensionless momentum
       (a singleton is allowed; duplicate coordinates are not). For CLI grids,
       equal minimum and maximum therefore require exactly one point.
   * - ``k0_m_inv``, ``q_m_inv``
     - ``(M,)``, ``(M, N)``
     - Vacuum wave number and in-plane wave number in inverse meters.
   * - ``source_layer``, ``source_layer_name``, ``source_epsilon``
     - scalar, scalar, ``(M,)``
     - Source index, its name and its energy-dependent complex permittivity.
   * - ``directions``, ``polarizations``
     - ``(2,)``, ``(2,)``
     - Axis labels ``[down, up]`` and ``[s, p]`` (TE, TM).
   * - ``reflection``, ``reflection_valid``
     - ``(M, N, 2, 2)``
     - Complex amplitude and finite-value mask in energy, q, direction,
       polarization order. Missing boundary directions are NaN/false.
   * - ``characteristic``, ``characteristic_valid``
     - ``(M, N, 2)``
     - Complex modal characteristic and finite-value mask in energy, q,
       polarization order.
   * - ``characteristic_kind``
     - scalar
     - ``finite_source_airy_denominator`` for layer 9 or
       ``top_exterior_reflection_denominator`` for layer 10.

Here is a concrete check of a finite TM reflection sample and the nearest
real-grid denominator value. The mask check matters near singularities and
for absent boundary directions. The sample at :math:`q/k_0=1.5` is beyond
the air light line but **below** the DBP source light line
:math:`\sqrt{3}`; it is not an evanescent-incident DBP sample:

.. code-block:: python

   import numpy as np

   with np.load("/tmp/dbr_dbp9_modes.npz", allow_pickle=False) as data:
       print(data.files)
       print(data["source_layer"].item(), data["source_layer_name"].item())
       print(data["characteristic_kind"].item())
       print(data["reflection"].shape, data["characteristic"].shape)
       energy_index = int(np.argmin(abs(data["energy_eV"] - 2.0)))
       q_index = int(np.argmin(abs(data["q_over_k0"] - 1.5)))
       down = list(data["directions"]).index("down")
       tm = list(data["polarizations"]).index("p")
       if data["reflection_valid"][energy_index, q_index, down, tm]:
           r = data["reflection"][energy_index, q_index, down, tm]
           print("E (eV), q/k0:", data["energy_eV"][energy_index],
                 data["q_over_k0"][q_index])
           print("k0, q (1/m):", data["k0_m_inv"][energy_index],
                 data["q_m_inv"][energy_index, q_index])
           print("TM down r, |r|:", r, abs(r))
       if data["characteristic_valid"][energy_index, q_index, tm]:
           print("TM |D|:", abs(data["characteristic"][energy_index, q_index, tm]))

:math:`k_0=E/(\hbar c)` after converting eV to joules, and
:math:`q=(q/k_0)k_0`. Thus equal horizontal coordinates at different energies
are *different* physical :math:`q` values. Real :math:`q/k_0>1` is beyond
the air light line, not necessarily beyond every layer's light line. In an
evanescent incident channel, :math:`|r|` can exceed 1 without violating
energy conservation; do not interpret it as an ordinary propagating-wave
power reflectance.

From a feature to a mode hypothesis
===================================

1. Look for coherent TE or TM structure across several nearby energies and
   momenta rather than a single bright pixel. Inspect both incident regions
   and the corresponding ``characteristic_valid`` mask. Refine the energy and
   q grids around a promising feature: 41 points over 1 to 3 eV give 0.05 eV
   spacing and can miss narrow or strongly dispersing resonances.
2. Check the relevant light lines and expected stop band. A possible air-side
   bound surface feature should not be radiative into air; another substrate
   or layer may still offer a leakage channel. Determine the polarization and
   dispersion, not just the peak height. A BSW is tied to a dielectric DBR
   stop band and surface termination; an SPP needs a suitable metal/dielectric
   interface and is ordinarily TM. A Tamm-like interface state is tied to an
   interface/termination and its stop-band physics. Hybridization and lossy
   stacks can blur these categories. This silver-backed geometry admits
   metal-interface as well as dielectric-stack hypotheses, but the scan alone
   establishes none of them as a BSW, SPP or Tamm mode.
3. Optionally search for complex-q candidates at a selected energy by adding
   ``--pole-energy 2.0`` to the stage-1 command (keep the other options and
   use a new output filename). The search is bounded by the requested *real*
   ``--q-min``/``--q-max`` values in units of that energy's :math:`k_0`, with
   :math:`10^{-6} \leq \operatorname{Im}(q)/k_0 \leq 0.2`. For a passive
   lossy Drude/air interface at 2 eV, the TM interface expression
   :math:`q/k_0=\sqrt{\epsilon_m/(\epsilon_m+1)}` gives approximately
   :math:`1.0351+0.0020i`; the solver's decaying vertical-wavevector branch
   has a small TM characteristic there. With :math:`\exp(iqx-i\omega t)`,
   positive :math:`\operatorname{Im}q` means attenuation along positive x.
   This diagnostic's bounded search does not change the N-layer solver's
   separate default pole-search window or establish a universal sheet choice
   for radiative or multilayer modes. It saves
   ``pole_energy_eV`` (scalar), ``pole_q_m_inv`` and ``pole_q_over_k0``
   (complex arrays), ``pole_polarization`` (``s``/``p``), and
   ``pole_residual`` (real array). The residual is the magnitude of the
   solver's normalized, ratio-free pole-search characteristic, not the raw
   real-axis ``characteristic`` above. No returned poles can simply mean none
   were found *in that box* (or that a candidate was missed by the numerical
   winding search); it is not proof of absence. The plot marks only :math:`\operatorname{Re}q/k_0`
   at the selected energy, not the attenuation from :math:`\operatorname{Im}q`.
4. For any candidate, check the physical branch/sheet, outgoing or decaying
   boundary conditions, polarization, and field profile on both sides of the
   interface. Quantify localization and loss or propagation length from
   complex :math:`q`, and vary energy to establish a stable dispersion branch.
   Compare the same scan against unstacked or unterminated control structures
   to separate DBR-specific behavior from an ordinary interface resonance or
   a finite-film cavity feature. A reflection peak alone can also be an
   interference or radiative feature, not a bound mode.
5. If the goal is emitter coupling, place the emitter in the correct region
   and test its position and dipole-orientation overlap with the candidate
   field. Resolve the emitter linewidth and modal energy width with a finer
   energy grid. Finally compare with an N-layer Green-tensor or spectral-density
   calculation using appropriate converged Sommerfeld integration. This
   diagnostic scans real :math:`q` and optional pole candidates; it does
   **not** replace the Sommerfeld integral or prove a BSW, SPP or Tamm mode.
