"""Integration test: AGNI (real atmosphere) coupled to atmodeller (real outgas).

Per-iteration coupling: AGNI computes F_atm and surface gas vmrs;
atmodeller back-solves JAX-based volatile partitioning (Bower+2025,
ApJ 995:59) and writes per-gas partial pressures into hf_row. The
two solvers run in alternation; this file exercises the
integration-tier portions of that boundary:

- Pair-wise schema validators round-trip ``atmos_clim.module='agni'``
  with ``outgas.module='atmodeller'``.
- Atmodeller's solver_mode enum is exactly {'robust', 'basic'};
  the documented defaults round-trip.
- solver_max_steps and solver_multistart are strictly positive
  (the ``gt(0)`` contract; 0 / -1 raise).
- The optical-depth aggregator at the AGNI side returns a
  monotonic profile from TOA to surface; the matrix design lock
  requires every AGNI x X integration to assert
  ``tau_atm_TOA < 0.5 * tau_atm_surface``.
- The Path-C wrapper merge propagates both the AGNI 1.10.2
  diagnostic keys AND the atmodeller-side fO2 + O residual keys
  into hf_row through the registered helpfile columns.

atmodeller is an optional dependency (the docker-retired CI image
installs it; bare PR-CI conda envs need not). The module-top
``pytest.importorskip('atmodeller')`` follows the existing
``test_path_c_atmodeller.py`` pattern.

The full two-timestep AGNI + atmodeller coupled run with real Julia
+ real JAX sits well above the slow-tier per-step budget on Linux
GHA. The slow-tier aragog x atmodeller sibling at
``test_slow_aragog_atmodeller.py`` exercises the atmodeller leg
with a real interior solver; the AGNI leg is exercised by the
existing ``test_smoke_modules.py`` chain.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('atmodeller')

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


# ---------------------------------------------------------------------------
# Schema-validator round-trips for the (agni, atmodeller) production combo.
# ---------------------------------------------------------------------------


def test_outgas_module_atmodeller_round_trips_through_schema():
    """``outgas.module='atmodeller'`` is inside the documented enum.

    Discrimination: every member of the enum round-trips, and an
    obviously-invalid name is rejected.
    """
    from proteus.config._outgas import Outgas

    for known in ('calliope', 'atmodeller', 'dummy'):
        o = Outgas(module=known)
        assert o.module == known
    with pytest.raises(ValueError, match=r'(?i)module'):
        Outgas(module='not_a_real_outgas_module')


def test_atmodeller_solver_mode_enum_is_robust_or_basic():
    """``Atmodeller.solver_mode`` is restricted to {'robust', 'basic'}.

    Discrimination: both round-trip with identity check; any third
    value is rejected at validator time.
    """
    from proteus.config._outgas import Atmodeller

    for known in ('robust', 'basic'):
        a = Atmodeller(solver_mode=known)
        assert a.solver_mode == known
    with pytest.raises(ValueError, match=r'(?i)solver_mode'):
        Atmodeller(solver_mode='turbo')
    # Discrimination on the default: the production default is
    # documented as 'robust'; a regression that flipped the default
    # would surface here.
    default = Atmodeller()
    assert default.solver_mode == 'robust'


def test_atmodeller_solver_step_and_multistart_must_be_positive():
    """``solver_max_steps`` and ``solver_multistart`` are
    ``validators.gt(0)`` — strictly positive integers.

    Edge: limit-input case (0 and -1 raise). Documented defaults
    (256, 10) round-trip. Discrimination guard: a regression that
    swapped ``gt(0)`` for ``ge(0)`` would accept 0; the explicit
    zero-raise rejects that.
    """
    from proteus.config._outgas import Atmodeller

    with pytest.raises(ValueError, match=r'(?i)solver_max_steps'):
        Atmodeller(solver_max_steps=0)
    with pytest.raises(ValueError, match=r'(?i)solver_max_steps'):
        Atmodeller(solver_max_steps=-1)
    with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
        Atmodeller(solver_multistart=0)
    with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
        Atmodeller(solver_multistart=-1)
    default = Atmodeller()
    assert default.solver_max_steps > 0
    assert default.solver_multistart > 0
    # Defaults documented in the dataclass docstring.
    assert default.solver_max_steps == 256
    assert default.solver_multistart == 10


def test_atmodeller_none_sentinel_coerced_to_python_none_on_eos_fields():
    """The ``none_if_none`` converter on the eos_* and solubility_*
    fields turns the lowercase string 'none' into Python None at
    construction (the converter is case-sensitive).

    Discrimination: lowercase 'none' coerces; uppercase 'None' /
    'NONE' do NOT (those pass through as literal strings). A
    regression that dropped the converter would leave the
    lowercase sentinel as 'none' and break downstream atmodeller
    dispatch which checks against the Python None literal. A
    regression that broadened the converter to case-insensitive
    would coerce 'None' too, changing the contract.
    """
    from proteus.config._outgas import Atmodeller

    # Lowercase sentinel coerces.
    a_lower = Atmodeller(eos_H2O='none')
    assert a_lower.eos_H2O is None
    # Uppercase variants pass through unchanged (case-sensitive).
    for non_sentinel in ('None', 'NONE'):
        a_passthrough = Atmodeller(eos_H2O=non_sentinel)
        assert a_passthrough.eos_H2O == non_sentinel, (
            f'{non_sentinel!r} should pass through; got {a_passthrough.eos_H2O!r}'
        )
    # Non-sentinel string passes through.
    a_real = Atmodeller(eos_H2O='SHV_CORK')
    assert a_real.eos_H2O == 'SHV_CORK'


# ---------------------------------------------------------------------------
# Optical-depth monotonicity at the AGNI side of the AGNI x atmodeller pair.
# Matrix design lock: every AGNI x X integration test must assert this.
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_agni_atmodeller_optical_depth_monotonic_from_TOA_to_surface():
    """Drive ``_summarise_tau_band`` with a profile representative of
    an atmodeller-equilibrated reducing atmosphere: H2-CO-CH4 with
    low LW continuum opacity but non-trivial gas absorption. Confirm
    ``tau_atm_TOA < 0.5 * tau_atm_surface``.

    Physical scenario: reducing IW-2 atmosphere; tau is smaller than
    in the H2O wet-greenhouse case but still grows by 2-3 orders of
    magnitude from TOA to surface.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    # Four levels (TOA -> surface), four bands. Lower magnitudes
    # than the wet-greenhouse case but the same monotone-with-depth
    # property.
    tau_band = np.array(
        [
            [0.002, 0.001, 0.0005, 0.0003],  # TOA
            [0.04, 0.06, 0.02, 0.03],
            [0.4, 0.5, 0.25, 0.3],
            [2.5, 4.0, 1.5, 2.0],  # surface
        ]
    )
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=4, nbands=4)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)
    assert tau_TOA == pytest.approx(np.mean(tau_band[0, :]), rel=1e-12)
    assert tau_surface == pytest.approx(np.mean(tau_band[-1, :]), rel=1e-12)
    # Monotonicity (matrix design lock).
    assert tau_TOA < tau_surface
    # Scale guard: gap covers ~3 orders of magnitude, well below
    # 0.5x surface.
    assert tau_TOA < 0.5 * tau_surface


@pytest.mark.physics_invariant
def test_agni_atmodeller_optical_depth_bounded_below_by_zero():
    """Optical depth is non-negative everywhere by construction. A
    regression that admitted a negative tau_band value would corrupt
    the aggregator (mean over mixed-sign values can land at a near-
    zero TOA-surface gap and bypass the monotonicity check).

    Edge: pin the boundedness invariant.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    tau_band = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.3],
            [2.0, 1.5],
        ]
    )
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=3, nbands=2)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)
    assert tau_TOA >= 0.0
    assert tau_surface > 0.0
    # Strictly less than surface even when TOA is zero.
    assert tau_TOA < tau_surface


# ---------------------------------------------------------------------------
# Wrapper merge: the Path C atmodeller-derived fO2 + O residual flow.
# ---------------------------------------------------------------------------


def test_agni_atmodeller_helpfile_keys_register_path_c_columns():
    """Path C atmodeller writes ``fO2_shift_IW_derived``, ``O_res``,
    and the per-gas pressure columns into hf_row. The wrapper merge
    guard depends on every key being registered in
    ``GetHelpfileKeys()``.

    Discrimination: pin the AGNI diagnostics, the atmodeller-derived
    fO2 + O residual, AND a representative per-gas pressure column.
    Each is independently registered; a regression that dropped any
    one would fail the per-key assertion.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    agni_diagnostic_keys = (
        'tau_atm_TOA',
        'tau_atm_surface',
        'agni_Ra_max',
        'agni_t_conv_over_t_rad',
    )
    atmodeller_path_c_keys = (
        'fO2_shift_IW_derived',
        'O_res',
    )
    pressure_keys = (
        'H2O_bar',
        'CO2_bar',
        'H2_bar',
        'CO_bar',
        'N2_bar',
    )
    for key in agni_diagnostic_keys + atmodeller_path_c_keys + pressure_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in agni_diagnostic_keys + atmodeller_path_c_keys + pressure_keys:
        assert key in row
        assert row[key] == 0.0
        assert isinstance(row[key], float)
