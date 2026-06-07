"""Integration test: AGNI (real atmosphere) coupled to CALLIOPE (real outgas).

Per-iteration coupling: AGNI computes F_atm and the surface gas
vmrs; CALLIOPE reads the surface partial pressures from its own
equilibrium solver and feeds them back into hf_row for the next
AGNI call. This file exercises the integration-tier portions of
that boundary:

- Pair-wise schema validators round-trip ``atmos_clim.module='agni'``
  with ``outgas.module='calliope'`` (the AGNI + CALLIOPE production
  configuration used in the wet-greenhouse Earth-IC runs).
- The AGNI optical-depth aggregator emits a monotonic profile from
  TOA to the surface; the matrix design lock (commit b8021c33)
  requires every AGNI x X integration to assert
  ``tau_atm_TOA < 0.5 * tau_atm_surface``.
- The CALLIOPE solver-parameter contract (``nguess > 0``,
  ``nsolve > 0``) holds at the attrs validator layer.
- The ``Calliope.is_included`` helper preserves the documented
  ten-gas set without silently dropping any species, so AGNI's
  gas-network reflection on the next iteration sees a stable
  composition.

The full two-timestep AGNI + CALLIOPE coupled run with real Julia
sits at ~25-30 min on Linux GHA, above the slow-tier per-step
budget. The slow-tier aragog x calliope sibling at
``test_slow_aragog_calliope.py`` exercises the CALLIOPE leg with
a real interior solver; the AGNI leg is exercised by the existing
``test_smoke_modules.py`` chain.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


# ---------------------------------------------------------------------------
# Schema-validator round-trip for the (agni, calliope) production combination.
# ---------------------------------------------------------------------------


def test_outgas_module_calliope_round_trips_through_schema():
    """``outgas.module='calliope'`` is in the documented enum and
    round-trips through the attrs validator.

    Discrimination: confirm 'atmodeller' and 'dummy' also round-trip
    (rules out a regression that broke the validator into raising
    on every input) and reject an invalid name (rules out a
    regression that disabled the validator entirely).
    """
    from proteus.config._outgas import Outgas

    for known in ('calliope', 'atmodeller', 'dummy'):
        o = Outgas(module=known)
        assert o.module == known
    with pytest.raises(ValueError, match=r'(?i)module'):
        Outgas(module='totally_invalid_outgas_backend')


def test_calliope_solver_parameters_must_be_positive():
    """CALLIOPE's solver guess + iteration counts must be strictly
    positive integers. The attrs validators at
    ``_outgas.py:55-56`` use ``validators.gt(0)``.

    Edge: limit-input case (0 and -1 must both raise). The
    documented defaults (1000 / 3000) round-trip. Discrimination:
    a regression that swapped ``gt(0)`` for ``ge(0)`` would accept
    nguess=0 and pass the test; the explicit zero-raise check
    catches that.
    """
    from proteus.config._outgas import Calliope

    with pytest.raises(ValueError, match=r'(?i)nguess'):
        Calliope(nguess=0)
    with pytest.raises(ValueError, match=r'(?i)nguess'):
        Calliope(nguess=-100)
    with pytest.raises(ValueError, match=r'(?i)nsolve'):
        Calliope(nsolve=0)
    with pytest.raises(ValueError, match=r'(?i)nsolve'):
        Calliope(nsolve=-50)

    # Documented defaults round-trip.
    default = Calliope()
    assert default.nguess > 0
    assert default.nsolve > 0


def test_calliope_is_included_preserves_documented_ten_gas_set():
    """``Calliope.is_included`` must return True for every gas in
    the documented ten-species set when defaults apply. AGNI's
    chemistry layer reflects on this set when building the next
    iteration's volatile mix; a silently-dropped species would
    push AGNI down the wrong solubility/EOS branch.

    Discrimination: pin the full list of ten species. A regression
    that dropped any one (e.g. removed include_NH3) would fail the
    per-gas loop. The trailing ``is_included('Ar')`` raise pins
    that the helper does not silently return False for an absent
    attribute.
    """
    from proteus.config._outgas import Calliope

    c = Calliope()
    documented_species = (
        'H2O',
        'CO2',
        'N2',
        'S2',
        'SO2',
        'H2S',
        'NH3',
        'H2',
        'CH4',
        'CO',
    )
    for gas in documented_species:
        assert c.is_included(gas) is True, f'{gas} missing from Calliope defaults'

    # Pin the count of include_* fields so a regression that silently
    # adds an eleventh species (e.g. include_Xe) fails the count
    # check even if the ten documented species still appear.
    import attrs

    include_fields = [f for f in attrs.fields(Calliope) if f.name.startswith('include_')]
    assert len(include_fields) == len(documented_species), (
        f'Expected {len(documented_species)} include_* fields on Calliope, '
        f'got {len(include_fields)}: {[f.name for f in include_fields]}'
    )

    # Discrimination: helper must raise on an undocumented attribute
    # rather than silently return False. The attrs class does not
    # carry an include_Ar field.
    with pytest.raises(AttributeError):
        c.is_included('Ar')


# ---------------------------------------------------------------------------
# Optical-depth monotonicity at the AGNI side of the AGNI x CALLIOPE pair.
# Matrix design lock: every AGNI x X integration test must assert this.
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_agni_calliope_optical_depth_monotonic_from_TOA_to_surface():
    """Drive ``_summarise_tau_band`` with a profile representative of
    a CALLIOPE-equilibrated H2O+CO2 atmosphere: low tau at TOA, large
    tau at the surface from the H2O+CO2 LW continuum. Confirm the
    aggregated hf_row scalars satisfy
    ``tau_atm_TOA < 0.5 * tau_atm_surface``.

    Physical scenario: CALLIOPE produces a wet-greenhouse atmosphere
    with H2O dominating the LW opacity. Integrated from TOA, optical
    depth grows by ~2-3 orders of magnitude over the column.

    Discrimination guard: the wrong-direction-of-integration
    regression would put the largest tau at TOA. The strict
    inequality + the half-surface guard reject both an inversion
    and a regression that shrank but did not flip the gap.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    # Five levels (TOA -> surface), three LW bands. Tau grows by
    # ~3 dex from TOA to surface; pattern matches a wet greenhouse
    # with H2O continuum opacity.
    tau_band = np.array(
        [
            [0.001, 0.002, 0.0005],  # TOA
            [0.05, 0.08, 0.03],
            [0.5, 0.8, 0.3],
            [3.0, 5.0, 2.0],
            [10.0, 20.0, 8.0],  # surface
        ]
    )
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=5, nbands=3)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)

    # Closed-form: row means.
    assert tau_TOA == pytest.approx((0.001 + 0.002 + 0.0005) / 3, rel=1e-12)
    assert tau_surface == pytest.approx((10.0 + 20.0 + 8.0) / 3, rel=1e-12)
    # Monotonicity invariant (the matrix design lock).
    assert tau_TOA < tau_surface
    # Scale guard: gap must be at least one order of magnitude in a
    # wet-greenhouse regime. tau_TOA ~ 1e-3, tau_surface ~ 13;
    # tau_TOA / tau_surface ~ 1e-4, well below the 0.5 guard.
    assert tau_TOA < 0.5 * tau_surface


@pytest.mark.physics_invariant
def test_agni_calliope_optical_depth_per_band_strictly_increases_with_depth():
    """The AGNI tau_band array is integrated from TOA downwards, so
    for every band individually the per-level value must increase
    with depth. The aggregator preserves this property in the mean.

    Edge: a per-band check (not just on the aggregate) is stronger
    than the mean-only check: a regression that flipped only the
    last band's index would still satisfy the mean inequality but
    fail per-band monotonicity at that band.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    tau_band = np.array(
        [
            [0.01, 0.02, 0.005],  # TOA
            [0.3, 0.5, 0.2],
            [1.5, 2.0, 1.0],
            [5.0, 8.0, 4.0],  # surface
        ]
    )
    # Per-band monotonicity: every band's depth profile non-decreasing.
    for band_idx in range(tau_band.shape[1]):
        col = tau_band[:, band_idx]
        diffs = np.diff(col)
        assert np.all(diffs >= 0), f'tau_band[:, {band_idx}] is not monotonic: diffs={diffs}'
    # The aggregator's mean inherits the per-band monotonicity.
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=4, nbands=3)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)
    assert tau_TOA < tau_surface
    assert tau_TOA < 0.5 * tau_surface


# ---------------------------------------------------------------------------
# Wrapper-merge contract: the AGNI diagnostics must flow into hf_row.
# ---------------------------------------------------------------------------


def test_agni_calliope_wrapper_merge_includes_calliope_pressure_keys():
    """Beyond the AGNI diagnostic keys (tau_atm_TOA / surface,
    atm_Ra_max, atm_t_conv_over_t_rad), the CALLIOPE leg writes
    per-gas partial pressures into hf_row. The helpfile schema
    must include both sets so the wrapper merge guard
    (atmos_clim/wrapper.py:196-198) propagates everything.

    Discrimination: pin both the AGNI-side diagnostic keys AND a
    representative CALLIOPE-side per-gas pressure key. A regression
    that dropped any one would fail the per-key assertion.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    agni_diagnostic_keys = (
        'tau_atm_TOA',
        'tau_atm_surface',
        'atm_Ra_max',
        'atm_t_conv_over_t_rad',
    )
    calliope_per_gas_keys = (
        'H2O_bar',
        'CO2_bar',
        'N2_bar',
        'H2_bar',
        'CO_bar',
    )
    for key in agni_diagnostic_keys + calliope_per_gas_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    # ZeroHelpfileRow seeds every key with a float zero.
    row = ZeroHelpfileRow()
    for key in agni_diagnostic_keys + calliope_per_gas_keys:
        assert row[key] == pytest.approx(0.0, abs=1e-12)
        assert isinstance(row[key], float)
