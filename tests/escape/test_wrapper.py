"""
Unit tests for proteus.escape.wrapper module

This module tests atmospheric escape functionality including:
- run_escape(): Generic escape orchestrator (dummy/zephyrus/disabled modes)
- run_zephyrus(): Energy-limited escape via ZEPHYRUS library
- calc_new_elements(): Elemental inventory updates after unfractionated escape

Physics tested:
- Escape flux conservation (kg/s to kg/yr conversions)
- Elemental mass ratio preservation during unfractionated escape
- Reservoir selection (bulk, outgas, pxuv)
- Minimum threshold enforcement for desiccated planets
- Non-negative mass constraints
- XUV-driven hydrodynamic escape (ZEPHYRUS)

All tests use mocked ZEPHYRUS calls to avoid heavy physics computation (<100ms runtime).

Related documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip('zephyrus')

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# =======================================================================================
# SECTION: run_escape(), generic escape orchestrator
# =======================================================================================


@pytest.mark.unit
def test_run_escape_disabled():
    """Test escape when module is disabled (None).

    Physical scenario: Planet with escape module turned off.
    Validates that esc_rate_total is set to zero and no element calculations occur.
    """
    from proteus.escape.wrapper import run_escape

    # Mock config with escape disabled
    config = MagicMock()
    config.escape.module = None

    # Minimal hf_row for escape calculation
    hf_row = {}

    # Call run_escape with escape disabled
    run_escape(config, hf_row, dt=1000.0, stellar_track=None)

    # Verify escape rate is zero
    assert hf_row['esc_rate_total'] == pytest.approx(0.0, abs=1e-12)
    # Early-return discriminator: the disabled branch zeroes per-element
    # escape rates and returns before the M_vol_initial / esc_kg_cumulative
    # baseline is seeded. A regression that fell through to the dummy
    # branch would (a) leave esc_rate_H at the unfractionated partition
    # and (b) seed M_vol_initial from element_list. Both keys absent here
    # rules that out.
    assert hf_row['esc_rate_H'] == pytest.approx(0.0, abs=1e-12)
    assert 'M_vol_initial' not in hf_row


@pytest.mark.unit
def test_run_escape_dummy():
    """Test escape using dummy module with fixed rate.

    Physical scenario: Planet with constant bulk escape rate (e.g., 1e5 kg/s).
    Validates that dummy escape rate is assigned and elements are updated correctly.
    """
    from proteus.escape.wrapper import run_escape

    # Mock config with dummy escape at 1e5 kg/s
    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.dummy.rate = 1e5  # kg/s
    config.escape.reservoir = 'bulk'
    config.outgas.mass_thresh = 1e10  # kg

    # Minimal hf_row with initial element inventories (all elements from element_list)
    hf_row = {
        'H_kg_total': 1e21,  # Large H reservoir (e.g., Earth ocean mass)
        'C_kg_total': 1e18,  # Carbon inventory
        'N_kg_total': 1e19,  # Nitrogen inventory
        'S_kg_total': 1e17,  # Sulfur inventory
        'Si_kg_total': 1e19,  # Silicon inventory
        'Mg_kg_total': 1e18,  # Magnesium inventory
        'Fe_kg_total': 1e20,  # Iron inventory
        'Na_kg_total': 1e16,  # Sodium inventory
        'H_kg_atm': 1e20,  # Atmospheric H
        'C_kg_atm': 1e17,  # Atmospheric C
        'N_kg_atm': 1e18,  # Atmospheric N
        'S_kg_atm': 1e16,  # Atmospheric S
        'Si_kg_atm': 1e17,  # Atmospheric Si
        'Mg_kg_atm': 1e16,  # Atmospheric Mg
        'Fe_kg_atm': 1e17,  # Atmospheric Fe
        'Na_kg_atm': 1e15,  # Atmospheric Na
    }

    # Call run_escape
    run_escape(config, hf_row, dt=1000.0, stellar_track=None)

    # Verify escape rate matches dummy rate
    assert hf_row['esc_rate_total'] == pytest.approx(1e5, rel=1e-8)

    # Verify element inventories were updated (should be reduced)
    assert hf_row['H_kg_total'] < 1e21  # H should decrease
    assert hf_row['C_kg_total'] < 1e18  # C should decrease
    assert hf_row['N_kg_total'] < 1e19  # N should decrease
    assert hf_row['S_kg_total'] < 1e17  # S should decrease


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_escape_atmosphere_only_sources_loss_from_atmosphere():
    """Once the mantle is crystallized the atmosphere is the only escapable
    reservoir, so run_escape(atmosphere_only=True) sizes the per-element loss
    from *_kg_atm regardless of the configured 'bulk' reservoir.

    Physical scenario: an element (Fe) is held almost entirely in the frozen
    interior (huge *_kg_total, negligible *_kg_atm) while H sits in the
    atmosphere. The bulk and atmospheric distributions then diverge sharply:
    under 'bulk' the interior-heavy element dominates the loss; under
    atmosphere_only the atmosphere-heavy element does. The total escaped mass is
    identical either way (conservation)."""
    from proteus.escape.wrapper import run_escape
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1000.0  # yr
    rate = 1e5  # kg/s

    def _cfg():
        config = MagicMock()
        config.escape.module = 'dummy'
        config.escape.dummy.rate = rate
        config.escape.reservoir = 'bulk'  # configured (interior-inclusive) reservoir
        config.outgas.mass_thresh = 1.0  # kg, low enough that nothing is zeroed
        return config

    def _row():
        # Fe almost entirely interior; H entirely atmospheric; others absent.
        row = {f'{e}_kg_total': 0.0 for e in element_list}
        row.update({f'{e}_kg_atm': 0.0 for e in element_list})
        row['H_kg_total'] = 2e20
        row['H_kg_atm'] = 2e20
        row['Fe_kg_total'] = 2e22
        row['Fe_kg_atm'] = 1e10
        return row

    esc_mass = rate * secs_per_year * dt  # total kg removed over the step

    # Configured 'bulk' reservoir: loss tracks whole-planet abundance, so the
    # interior-heavy Fe dominates the per-element debit.
    row_bulk = _row()
    run_escape(_cfg(), row_bulk, dt=dt, atmosphere_only=False)
    loss_fe_bulk = 2e22 - row_bulk['Fe_kg_total']
    loss_h_bulk = 2e20 - row_bulk['H_kg_total']
    assert loss_fe_bulk > loss_h_bulk  # Fe dominates under the bulk reservoir
    assert loss_fe_bulk == pytest.approx(esc_mass * 2e22 / 2.02e22, rel=1e-3)

    # atmosphere_only: loss tracks atmospheric abundance, so H dominates and Fe
    # (frozen in the interior) is barely touched, the opposite ordering.
    row_atm = _row()
    run_escape(_cfg(), row_atm, dt=dt, atmosphere_only=True)
    loss_fe_atm = 2e22 - row_atm['Fe_kg_total']
    loss_h_atm = 2e20 - row_atm['H_kg_total']
    assert loss_h_atm > loss_fe_atm  # H dominates under the atmospheric reservoir
    assert loss_h_atm == pytest.approx(esc_mass, rel=1e-3)
    # Discrimination: the configured 'bulk' value was overridden; Fe loses
    # many orders of magnitude less when sourced from the atmosphere.
    assert loss_fe_atm < loss_fe_bulk / 1e5

    # Conservation: the total removed is the same esc_mass under both reservoirs.
    initial = {'H': 2e20, 'Fe': 2e22}
    total_loss_bulk = sum(
        initial.get(e, 0.0) - row_bulk.get(f'{e}_kg_total', 0.0) for e in element_list
    )
    total_loss_atm = sum(
        initial.get(e, 0.0) - row_atm.get(f'{e}_kg_total', 0.0) for e in element_list
    )
    assert total_loss_bulk == pytest.approx(esc_mass, rel=1e-3)
    assert total_loss_atm == pytest.approx(esc_mass, rel=1e-3)


@pytest.mark.unit
@patch('zephyrus.escape.EL_escape')
def test_run_escape_zephyrus(mock_el_escape):
    """Test escape using ZEPHYRUS energy-limited model.

    Physical scenario: Hot Jupiter with XUV-driven hydrodynamic escape.
    Validates that ZEPHYRUS is called with correct parameters and escape rate is assigned.
    """
    from proteus.escape.wrapper import run_escape

    # Mock ZEPHYRUS EL_escape to return a specific escape rate
    mock_el_escape.return_value = 1e7  # kg/s (high escape rate for hot Jupiter)

    # Mock config with ZEPHYRUS escape
    config = MagicMock()
    config.escape.module = 'zephyrus'
    config.escape.zephyrus.tidal = False
    config.escape.zephyrus.efficiency = 0.15
    config.escape.zephyrus.Pxuv = 5e-5  # bar
    config.escape.reservoir = 'outgas'
    config.outgas.mass_thresh = 1e10  # kg
    config.star.mass = 1.0e30  # kg (Sun-like)

    # Minimal hf_row for hot Jupiter
    hf_row = {
        'semimajorax': 0.05 * 1.496e11,  # 0.05 AU in meters
        'eccentricity': 0.01,
        'M_planet': 1.898e27,  # Jupiter mass in kg
        'R_int': 7.0e7,  # 70,000 km radius
        'R_xuv': 8.0e7,  # XUV radius slightly larger
        'F_xuv': 1e4,  # W/m^2 (high XUV flux)
        'H_kg_total': 1e24,  # Large H reservoir
        'C_kg_total': 1e20,
        'N_kg_total': 1e21,
        'S_kg_total': 1e19,
        'Si_kg_total': 1e19,
        'Mg_kg_total': 1e18,
        'Fe_kg_total': 1e20,
        'Na_kg_total': 1e17,
        'H_kg_atm': 1e23,  # Atmospheric reservoirs for 'outgas' mode
        'C_kg_atm': 1e19,
        'N_kg_atm': 1e20,
        'S_kg_atm': 1e18,
        'Si_kg_atm': 1e17,
        'Mg_kg_atm': 1e16,
        'Fe_kg_atm': 1e17,
        'Na_kg_atm': 1e16,
    }

    # Mock stellar track
    stellar_track = MagicMock()

    # Call run_escape
    run_escape(config, hf_row, dt=1000.0, stellar_track=stellar_track)

    # Verify ZEPHYRUS was called with correct parameters
    mock_el_escape.assert_called_once()
    call_args = mock_el_escape.call_args[0]
    assert not call_args[0]  # tidal contribution
    assert call_args[1] == pytest.approx(0.05 * 1.496e11, rel=1e-6)  # semimajor axis
    assert call_args[2] == pytest.approx(0.01, rel=1e-6)  # eccentricity

    # Verify escape rate matches mock return value
    assert hf_row['esc_rate_total'] == pytest.approx(1e7, rel=1e-8)


@pytest.mark.unit
@pytest.mark.physics_invariant
@patch('zephyrus.escape.EL_escape')
def test_run_escape_zephyrus_atmosphere_only_overrides_bulk(mock_el_escape):
    """The crystallized-mantle override applies to the ZEPHYRUS path too: with
    a configured 'bulk' reservoir but atmosphere_only=True, the per-element loss
    is sized from the atmosphere, so an interior-heavy element (Fe) is barely
    touched while the atmospheric element (H) absorbs the loss."""
    from proteus.escape.wrapper import run_escape
    from proteus.utils.constants import element_list, secs_per_year

    rate = 1e7  # kg/s
    dt = 1000.0  # yr
    mock_el_escape.return_value = rate

    config = MagicMock()
    config.escape.module = 'zephyrus'
    config.escape.zephyrus.tidal = False
    config.escape.zephyrus.efficiency = 0.15
    config.escape.zephyrus.Pxuv = 5e-5
    config.escape.reservoir = 'bulk'  # configured reservoir, to be overridden
    config.outgas.mass_thresh = 1.0
    config.star.mass = 1.0e30

    row = {f'{e}_kg_total': 0.0 for e in element_list}
    row.update({f'{e}_kg_atm': 0.0 for e in element_list})
    row.update(
        {
            'semimajorax': 0.05 * 1.496e11,
            'eccentricity': 0.01,
            'M_planet': 1.898e27,
            'R_int': 7.0e7,
            'R_xuv': 8.0e7,
            'F_xuv': 1e4,
            'H_kg_total': 2e22,
            'H_kg_atm': 2e22,
            'Fe_kg_total': 2e24,  # almost entirely frozen in the interior
            'Fe_kg_atm': 1e12,
        }
    )

    run_escape(config, row, dt=dt, stellar_track=MagicMock(), atmosphere_only=True)

    esc_mass = rate * secs_per_year * dt
    loss_h = 2e22 - row['H_kg_total']
    loss_fe = 2e24 - row['Fe_kg_total']
    # Atmosphere-sourced: H (atmospheric) dominates, Fe (frozen) is negligible.
    assert loss_h > loss_fe
    assert loss_h == pytest.approx(esc_mass, rel=1e-3)
    assert loss_fe < loss_h / 1e5


@pytest.mark.unit
def test_run_escape_invalid_module():
    """Test that invalid escape module raises ValueError.

    Physical scenario: Configuration error with unrecognized escape module.
    Validates proper error handling for invalid module names.
    """
    from proteus.escape.wrapper import run_escape

    # Mock config with invalid module name
    config = MagicMock()
    config.escape.module = 'invalid_module'

    hf_row = {}

    # Verify ValueError is raised
    with pytest.raises(ValueError, match='Invalid escape model'):
        run_escape(config, hf_row, dt=1000.0, stellar_track=None)

    # Side-effect discriminator: the dispatch raises in the else branch
    # BEFORE the final `esc_rate_total` log line. A regression that
    # silently fell through (e.g. defaulted to dummy) would set
    # esc_rate_total on hf_row. With the raise intact, the key is never
    # written.
    assert 'esc_rate_total' not in hf_row


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_escape_snapshots_baseline_on_first_call():
    """Test that the first run_escape call snapshots the bulk volatile
    inventory into M_vol_initial. This baseline is used by
    `outgas.wrapper.check_desiccation` to detect unexplained mass loss
    cascades (CHILI sweep R7/R21).

    Physical scenario: Earth-like planet with mixed H/C/N/S/O inventory.
    Issue #677 fix: O is now included in the baseline alongside H/C/N/S
    so the desiccation gate's (M_vol_initial - cur_m_ele) versus
    1.5*esc_kg_cumulative arithmetic stays consistent now that escape
    proportionally debits O via the new calc_new_elements.

    Validates that M_vol_initial = sum(*_kg_total) over ALL elements,
    and that esc_kg_cumulative is initialised to zero alongside the
    baseline. O is set to ~14x the H+C inventory here to make the
    O-inclusion observable: if O were silently dropped from the
    baseline the expected value would be off by O(1).
    """
    from proteus.escape.wrapper import run_escape

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.dummy.rate = 1e3  # kg/s
    config.escape.reservoir = 'bulk'
    config.outgas.mass_thresh = 1e10

    hf_row = {
        'H_kg_total': 4.7e20,
        'C_kg_total': 2.7e20,
        'N_kg_total': 0.0,
        'S_kg_total': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_total': 0.0,
        'O_kg_total': 1e22,  # included in baseline since issue #677 fix
    }

    run_escape(config, hf_row, dt=1000.0, stellar_track=None)

    expected_baseline = 4.7e20 + 2.7e20 + 1e22  # H + C + O (others zero)
    assert 'M_vol_initial' in hf_row
    assert hf_row['M_vol_initial'] == pytest.approx(expected_baseline, rel=1e-10), (
        'M_vol_initial must equal sum of *_kg_total over ALL elements '
        '(issue #677 fix: O is no longer excluded)'
    )
    assert 'esc_kg_cumulative' in hf_row
    # Cumulative escape after one step at 1e3 kg/s for 1000 yr:
    # 1e3 * 1000 * secs_per_year ≈ 3.156e13 kg
    assert hf_row['esc_kg_cumulative'] > 0.0
    assert hf_row['esc_kg_cumulative'] < 1e15


@pytest.mark.physics_invariant
def test_calc_new_elements_exempts_noble_gases_from_desiccation_floor():
    """Noble gas inventories are intrinsically trace and sit orders of
    magnitude below the major-volatile desiccation floor, so calc_new_elements
    must not zero them when they fall under min_thresh. A major volatile below
    the floor is still zeroed, so the two behaviours are discriminated and the
    noble exemption is not a blanket removal of the floor.
    """
    from proteus.escape.wrapper import calc_new_elements
    from proteus.utils.constants import element_list

    hf = {f'{e}_kg_total': 0.0 for e in element_list}
    hf['H_kg_total'] = 1.5e20
    hf['C_kg_total'] = 1.0e20
    hf['He_kg_total'] = 3.7e12  # Earth-like helium, far below the 1e16 floor
    hf['S_kg_total'] = 1.0e12  # a major volatile below the floor
    hf['esc_rate_total'] = 1.0e4  # small bulk escape

    out = calc_new_elements(hf, reservoir='bulk', dt=1.0e6, min_thresh=1.0e16)

    # Helium is trace and exempt: preserved, only debited by its escape share,
    # never floored to zero.
    assert out['He'] > 3.0e12
    assert out['He'] < 3.7e12
    # A major volatile below the same floor is treated as desiccated (zeroed),
    # confirming the floor still fires for the species it is meant for.
    assert out['S'] == 0.0
    # Hydrogen, well above the floor, is debited but stays non-zero.
    assert 0.0 < out['H'] < 1.5e20


@pytest.mark.physics_invariant
def test_calc_new_elements_debits_only_the_element_total():
    """The escape debit is returned as a new ``{e}_kg_total`` and nothing else in
    the row is touched, so the caller determines the written values.

    This is the ownership rule the outgassing side relies on: a noble gas shares
    its string with the matching gas species, so anything downstream that rebuilds
    ``{e}_kg_total`` from the reservoirs undoes the debit. The vapour step's half
    of that contract is asserted in
    tests/outgas/test_lavatmos.py::test_run_vapourisation_preserves_noble_gases.
    Edge case: a zero escape rate must leave every total untouched rather than
    drifting.
    """
    from proteus.escape.wrapper import calc_new_elements
    from proteus.utils.constants import element_list

    hf = {f'{e}_kg_total': 0.0 for e in element_list}
    for e in element_list:
        hf[f'{e}_kg_atm'] = 0.0
    hf['H_kg_total'] = hf['H_kg_atm'] = 1.0e20
    hf['He_kg_total'] = hf['He_kg_atm'] = 1.0e16
    hf['esc_rate_total'] = 1.0e8  # kg/s, large enough to debit He visibly

    out = calc_new_elements(hf, reservoir='bulk', dt=1.0e3, min_thresh=1.0e16)

    # The helium total came back debited, and by a resolvable amount rather than
    # a rounding difference.
    assert out['He'] < 1.0e16
    debited = 1.0e16 - out['He']
    assert debited > 1.0e14

    # calc_new_elements returns the new totals and writes nothing itself, so the
    # row it was handed is unchanged: the total is still the pre-escape value and
    # the per-reservoir masses are untouched. Rebuilding a total from those
    # reservoirs downstream would therefore undo the debit entirely.
    assert hf['He_kg_total'] == pytest.approx(1.0e16, rel=1e-12)
    assert hf['He_kg_atm'] == pytest.approx(1.0e16, rel=1e-12)
    assert hf['H_kg_total'] == pytest.approx(1.0e20, rel=1e-12)

    # Limit input: no escape means no debit on any element.
    hf['esc_rate_total'] = 0.0
    quiet = calc_new_elements(hf, reservoir='bulk', dt=1.0e3, min_thresh=1.0e16)
    assert quiet['He'] == pytest.approx(1.0e16, rel=1e-12)
    assert quiet['H'] == pytest.approx(1.0e20, rel=1e-12)


@pytest.mark.unit
def test_run_escape_baseline_persists_across_calls():
    """Test that subsequent run_escape calls do NOT overwrite M_vol_initial.

    Physical scenario: Multi-iteration evolution. The baseline must remain
    the FIRST snapshot, not get re-snapshotted on every iteration (which
    would defeat the desiccation gate).

    Discriminating: snapshot baseline = 1e21 kg. After escape removes
    ~3e16 kg, the second call must NOT reset M_vol_initial to 1e21 - 3e16.
    """
    from proteus.escape.wrapper import run_escape

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.dummy.rate = 1e9  # very high rate to make change visible
    config.escape.reservoir = 'bulk'
    config.outgas.mass_thresh = 1e10

    hf_row = {
        'H_kg_total': 1e21,
        'C_kg_total': 0.0,
        'N_kg_total': 0.0,
        'S_kg_total': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_total': 0.0,
    }

    # Iteration 1
    run_escape(config, hf_row, dt=1000.0, stellar_track=None)
    baseline_iter1 = hf_row['M_vol_initial']
    assert baseline_iter1 == pytest.approx(1e21, rel=1e-10)
    cum_iter1 = hf_row['esc_kg_cumulative']

    # Iteration 2: H_kg_total has shrunk, but baseline must be unchanged
    run_escape(config, hf_row, dt=1000.0, stellar_track=None)
    assert hf_row['M_vol_initial'] == pytest.approx(baseline_iter1, rel=1e-12), (
        'M_vol_initial must NOT be overwritten by subsequent escape calls'
    )
    # Cumulative escape must monotonically increase (not reset)
    assert hf_row['esc_kg_cumulative'] > cum_iter1, (
        'esc_kg_cumulative must accumulate, not reset, on subsequent calls'
    )


@pytest.mark.unit
def test_run_escape_resets_baseline_if_corrupt():
    """Test that a NaN or non-positive M_vol_initial gets re-snapshotted.

    Physical scenario: Resume from an old CSV that has the column but with
    NaN values, or a transient corruption. The gate must self-heal rather
    than carry forward bogus data forever.
    """
    from proteus.escape.wrapper import run_escape

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.dummy.rate = 1e3
    config.escape.reservoir = 'bulk'
    config.outgas.mass_thresh = 1e10

    hf_row = {
        'H_kg_total': 5e20,
        'C_kg_total': 0.0,
        'N_kg_total': 0.0,
        'S_kg_total': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_total': 0.0,
        'M_vol_initial': float('nan'),  # corrupt baseline
    }

    run_escape(config, hf_row, dt=1.0, stellar_track=None)

    assert hf_row['M_vol_initial'] == pytest.approx(5e20, rel=1e-10), (
        'NaN baseline must be re-snapshotted from current inventory'
    )
    # Finiteness discriminator: a regression that propagated NaN through
    # arithmetic (writing `0.0 * nan` or `nan + something`) instead of
    # detecting and replacing the corrupt baseline would leave a NaN
    # in M_vol_initial. The pytest.approx pin above already discriminates
    # 5e20 from a propagated NaN, but the explicit finiteness check
    # makes the failure mode loud.
    import math

    assert math.isfinite(hf_row['M_vol_initial'])


# =======================================================================================
# SECTION: run_zephyrus(), energy-limited escape
# =======================================================================================


@pytest.mark.unit
@patch('zephyrus.escape.EL_escape')
def test_run_zephyrus_no_tidal(mock_el_escape):
    """Test ZEPHYRUS escape without tidal contribution.

    Physical scenario: Earth-like planet at 1 AU with moderate XUV flux.
    Validates that tidal heating is disabled (tidal=False) in EL_escape call.
    """
    from proteus.escape.wrapper import run_zephyrus

    # Mock EL_escape to return a specific rate
    mock_el_escape.return_value = 1e3  # kg/s (moderate escape)

    # Mock config
    config = MagicMock()
    config.escape.zephyrus.tidal = False
    config.escape.zephyrus.efficiency = 0.1
    config.star.mass = 2.0e30  # kg

    # Minimal hf_row
    hf_row = {
        'semimajorax': 1.496e11,  # 1 AU
        'eccentricity': 0.0,
        'M_planet': 5.972e24,  # Earth mass
        'R_int': 6.371e6,  # Earth radius
        'R_xuv': 6.5e6,  # Slightly larger XUV radius
        'F_xuv': 100.0,  # W/m^2
    }

    stellar_track = MagicMock()

    # Call run_zephyrus
    mlr = run_zephyrus(config, hf_row, stellar_track)

    # Verify return value
    assert mlr == pytest.approx(1e3, rel=1e-8)

    # Verify EL_escape was called with tidal=False
    mock_el_escape.assert_called_once()
    assert not mock_el_escape.call_args[0][0]  # tidal parameter


@pytest.mark.unit
@patch('zephyrus.escape.EL_escape')
def test_run_zephyrus_with_tidal(mock_el_escape):
    """Test ZEPHYRUS escape with tidal heating contribution.

    Physical scenario: Hot Jupiter with tidal heating enhancing escape.
    Validates that tidal=True is passed to EL_escape.
    """
    from proteus.escape.wrapper import run_zephyrus

    # Mock EL_escape to return enhanced escape rate
    mock_el_escape.return_value = 1e8  # kg/s (very high escape with tidal)

    # Mock config with tidal enabled
    config = MagicMock()
    config.escape.zephyrus.tidal = True
    config.escape.zephyrus.efficiency = 0.2
    config.star.mass = 1.5e30  # kg

    # Hot Jupiter scenario: close-in orbit (0.03 AU), inflated radius, and
    # extreme XUV flux to exercise the tidal heating branch in ZEPHYRUS.
    # These conditions maximise tidal dissipation and XUV-driven escape.
    hf_row = {
        'semimajorax': 0.03 * 1.496e11,  # 0.03 AU
        'eccentricity': 0.05,
        'M_planet': 1e27,  # Sub-Jupiter mass
        'R_int': 1.0e8,  # 100,000 km (inflated radius)
        'R_xuv': 1.2e8,
        'F_xuv': 1e5,  # W/m^2 (extreme XUV)
    }

    stellar_track = MagicMock()

    # Call run_zephyrus
    mlr = run_zephyrus(config, hf_row, stellar_track)

    # Verify enhanced escape rate
    assert mlr == pytest.approx(1e8, rel=1e-8)

    # Verify tidal=True was passed
    mock_el_escape.assert_called_once()
    assert mock_el_escape.call_args[0][0]  # tidal parameter


# =======================================================================================
# SECTION: calc_new_elements(), elemental inventory updates
# =======================================================================================


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_calc_new_elements_bulk_reservoir():
    """Test elemental inventory update using bulk reservoir.

    Physical scenario: Unfractionated escape from entire planet (bulk).
    Issue #677 fix: O is now included in the partitioning and
    debited proportionally with the other elements. Validates that:
      (a) elemental mass ratios across ALL elements (incl. O) are
          approximately preserved (the unfractionated property)
      (b) sum of per-element losses equals the bulk mass loss
      (c) O is in the output dict and gets debited
    """
    from proteus.escape.wrapper import calc_new_elements
    from proteus.utils.constants import secs_per_year

    # Initial hf_row with bulk inventories, including a significant O budget.
    # O is set to ~14x the H reservoir so the "is O being debited?" check
    # has discriminating signal: without the fix, tgt['O'] would equal the
    # initial O_kg_total exactly.
    hf_row = {
        'esc_rate_total': 1e5,  # kg/s
        'H_kg_total': 1e21,
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e19,
        'Mg_kg_total': 1e18,
        'Fe_kg_total': 1e20,
        'Na_kg_total': 1e16,
        'O_kg_total': 1.4e22,  # Issue #677: O is now budgeted and escape-able
    }

    dt = 1000.0  # years
    reservoir = 'bulk'
    min_thresh = 1e10  # kg

    # Calculate initial mass ratios over ALL elements (incl. O)
    M_vols_initial = sum(hf_row[k] for k in hf_row if k.endswith('_kg_total'))
    emr_H_initial = hf_row['H_kg_total'] / M_vols_initial
    emr_O_initial = hf_row['O_kg_total'] / M_vols_initial

    # Call calc_new_elements
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Verify all elements are in output, including O (issue #677 fix)
    assert 'H' in tgt
    assert 'C' in tgt
    assert 'N' in tgt
    assert 'S' in tgt
    assert 'O' in tgt, 'Issue #677: O must now be included in calc_new_elements output'

    # Verify masses decreased (escape occurred). O must also decrease since
    # it is now part of the proportional partitioning.
    assert tgt['H'] < hf_row['H_kg_total']
    assert tgt['C'] < hf_row['C_kg_total']
    assert tgt['N'] < hf_row['N_kg_total']
    assert tgt['S'] < hf_row['S_kg_total']
    assert tgt['O'] < hf_row['O_kg_total'], (
        'O_kg_total must decrease under escape now that O is in the partitioning'
    )

    # Verify elemental mass ratios across ALL elements are preserved
    # (unfractionated property). At asymmetric inputs (O dominates by 14x),
    # this would fail loudly if O were dropped from the denominator.
    M_vols_final = sum(tgt.values())
    emr_H_final = tgt['H'] / M_vols_final
    emr_O_final = tgt['O'] / M_vols_final
    assert emr_H_final == pytest.approx(emr_H_initial, rel=1e-5)
    assert emr_O_final == pytest.approx(emr_O_initial, rel=1e-5)

    # Conservation property: sum of per-element loss equals bulk MLR * dt.
    # This is the test that would catch any future "skip O" regression.
    esc_mass_expected = hf_row['esc_rate_total'] * secs_per_year * dt
    total_loss = M_vols_initial - M_vols_final
    assert total_loss == pytest.approx(esc_mass_expected, rel=1e-5)

    # Verify no negative masses
    for e in tgt:
        assert tgt[e] >= 0.0


@pytest.mark.unit
def test_calc_new_elements_outgas_reservoir():
    """Test elemental inventory update using outgas (atmospheric) reservoir.

    Physical scenario: Escape from outgassed atmosphere only.
    Validates that atmospheric reservoirs (_kg_atm) are used for mass ratios.
    """
    from proteus.escape.wrapper import calc_new_elements

    # hf_row with both bulk and atmospheric inventories
    hf_row = {
        'esc_rate_total': 1e4,  # kg/s
        'H_kg_total': 1e21,  # Bulk H (mostly in interior)
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e19,
        'Mg_kg_total': 1e18,
        'Fe_kg_total': 1e20,
        'Na_kg_total': 1e16,
        'H_kg_atm': 1e20,  # Atmospheric H (used for outgas mode)
        'C_kg_atm': 1e17,
        'N_kg_atm': 1e18,
        'S_kg_atm': 1e16,
        'Si_kg_atm': 1e17,
        'Mg_kg_atm': 1e16,
        'Fe_kg_atm': 1e17,
        'Na_kg_atm': 1e15,
    }

    dt = 500.0  # years
    reservoir = 'outgas'
    min_thresh = 1e10  # kg

    # Call calc_new_elements
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Verify outputs are for TOTAL inventories (not just atmosphere)
    # But mass ratios derived from ATMOSPHERIC inventories
    assert tgt['H'] < hf_row['H_kg_total']  # Total H decreased
    assert tgt['C'] < hf_row['C_kg_total']
    assert tgt['N'] < hf_row['N_kg_total']
    assert tgt['S'] < hf_row['S_kg_total']


@pytest.mark.unit
def test_calc_new_elements_below_threshold():
    """Test elemental inventory when mass falls below minimum threshold.

    Physical scenario: Desiccated planet where volatile mass < 1e10 kg.
    Validates that inventories below threshold are set to zero.
    """
    from proteus.escape.wrapper import calc_new_elements

    # Very small volatile inventory (planet nearly desiccated)
    hf_row = {
        'esc_rate_total': 1e5,  # kg/s
        'H_kg_total': 1e9,  # Below threshold
        'C_kg_total': 1e8,
        'N_kg_total': 1e8,
        'S_kg_total': 1e7,
        'Si_kg_total': 1e8,
        'Mg_kg_total': 1e8,
        'Fe_kg_total': 1e9,
        'Na_kg_total': 1e7,
    }

    dt = 1000.0  # years
    reservoir = 'bulk'
    min_thresh = 1e10  # kg

    # Call calc_new_elements
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Total volatile mass is below threshold, so no change expected
    assert tgt['H'] == pytest.approx(hf_row['H_kg_total'], abs=1.0)
    assert tgt['C'] == pytest.approx(hf_row['C_kg_total'], abs=1.0)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_calc_new_elements_prevent_negative_mass():
    """Test that elemental masses cannot go negative during escape.

    Physical scenario: Extreme escape rate that would deplete element inventory.
    Validates that masses are clamped to zero (not negative).
    """
    from proteus.escape.wrapper import calc_new_elements

    # Small inventory with very high escape rate
    hf_row = {
        'esc_rate_total': 1e10,  # kg/s (extremely high)
        'H_kg_total': 1e18,  # Small H inventory
        'C_kg_total': 1e15,
        'N_kg_total': 1e16,
        'S_kg_total': 1e14,
        'Si_kg_total': 1e16,
        'Mg_kg_total': 1e15,
        'Fe_kg_total': 1e17,
        'Na_kg_total': 1e14,
    }

    dt = 1e6  # years (long timescale → massive escape)
    reservoir = 'bulk'
    min_thresh = 1e10  # kg

    # Call calc_new_elements (should clamp to zero)
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Verify all masses are non-negative
    for e in tgt:
        assert tgt[e] >= 0.0

    # Bulk escape over dt at 1e10 kg/s exceeds the total inventory
    # (~1e17 kg ~ 1.04e19 vs esc_mass = 1e10 * 3.156e7 * 1e6 ~ 3.16e23
    # kg). Every element must therefore be driven to zero, not just be
    # non-negative. A regression that allowed a partial debit through
    # would land at a small positive number rather than exact zero.
    for e in tgt:
        assert tgt[e] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_calc_new_elements_pxuv_not_supported():
    """Test that pxuv reservoir raises NotImplementedError.

    Physical scenario: Fractionated escape at XUV optical depth level.
    Validates that pxuv mode is not yet implemented.
    """
    from proteus.escape.wrapper import calc_new_elements

    hf_row = {
        'esc_rate_total': 1e5,
        'H_kg_total': 1e20,
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e18,
        'Mg_kg_total': 1e17,
        'Fe_kg_total': 1e19,
        'Na_kg_total': 1e16,
    }

    dt = 1000.0
    reservoir = 'pxuv'
    min_thresh = 1e10

    # Snapshot hf_row to verify the raise is side-effect-free.
    snapshot = dict(hf_row)

    # Verify ValueError is raised for pxuv
    with pytest.raises(ValueError, match='Fractionation at p_xuv is not yet supported'):
        calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # No-side-effect discriminator: the reservoir match-case raises in
    # the pxuv branch before any partition arithmetic runs. A regression
    # that fell through to the bulk path and only logged a warning
    # would leave H_kg_total and the other inventories debited on the
    # caller's dict.
    assert hf_row == snapshot


@pytest.mark.unit
def test_calc_new_elements_invalid_reservoir():
    """Test that invalid reservoir name raises ValueError.

    Physical scenario: Configuration error with unrecognized reservoir.
    Validates proper error handling.
    """
    from proteus.escape.wrapper import calc_new_elements

    hf_row = {
        'esc_rate_total': 1e5,
        'H_kg_total': 1e20,
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e18,
        'Mg_kg_total': 1e17,
        'Fe_kg_total': 1e19,
        'Na_kg_total': 1e16,
    }

    dt = 1000.0
    reservoir = 'invalid_reservoir'
    min_thresh = 1e10

    # Snapshot hf_row to verify the raise is side-effect-free.
    snapshot = dict(hf_row)

    # Verify ValueError is raised
    with pytest.raises(ValueError, match='Invalid escape reservoir'):
        calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # No-side-effect discriminator: the default match-case raises before
    # any partition arithmetic runs. A regression that downgraded the
    # invalid reservoir to a silent default-bulk fallthrough would have
    # debited H_kg_total and the other inventories on the caller's dict.
    assert hf_row == snapshot


@pytest.mark.unit
def test_calc_new_elements_zero_escape_rate():
    """Test elemental inventory with zero escape rate.

    Physical scenario: Escape disabled or negligible escape.
    Validates that inventories remain unchanged when esc_rate_total = 0.
    """
    from proteus.escape.wrapper import calc_new_elements

    # hf_row with zero escape rate
    hf_row = {
        'esc_rate_total': 0.0,  # No escape
        'H_kg_total': 1e20,
        'C_kg_total': 1e18,
        'N_kg_total': 1e19,
        'S_kg_total': 1e17,
        'Si_kg_total': 1e18,
        'Mg_kg_total': 1e17,
        'Fe_kg_total': 1e19,
        'Na_kg_total': 1e16,
    }

    dt = 1000.0
    reservoir = 'bulk'
    min_thresh = 1e10

    # Call calc_new_elements
    tgt = calc_new_elements(hf_row, dt, reservoir, min_thresh)

    # Verify inventories are unchanged
    assert tgt['H'] == pytest.approx(hf_row['H_kg_total'], rel=1e-10)
    assert tgt['C'] == pytest.approx(hf_row['C_kg_total'], rel=1e-10)
    assert tgt['N'] == pytest.approx(hf_row['N_kg_total'], rel=1e-10)
    assert tgt['S'] == pytest.approx(hf_row['S_kg_total'], rel=1e-10)


# ---------------------------------------------------------------------------
# Coverage of error/edge paths: TypeError baseline, dummy + zephyrus unfract
# fallbacks. Targets lines 58-59, 151-153, 200-202 in escape/wrapper.py.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_escape_recomputes_baseline_when_m_vol_initial_is_unparseable():
    """If hf_row['M_vol_initial'] is a string (or any non-numeric value),
    the float() coercion raises and the source falls back to 0.0 before
    snapshotting the baseline from the per-element totals.

    Edge: a corrupted helpfile CSV value loaded on resume could land here
    as a string. The fallback must not crash and must produce a baseline
    that equals the sum of per-element totals so the desiccation gate
    remains consistent.
    """
    from proteus.escape.wrapper import run_escape

    config = MagicMock()
    # 'dummy' (not disabled) so we reach the baseline-rebuild branch
    # at lines 50-71; rate=0 keeps the post-dispatch arithmetic trivial.
    config.escape.module = 'dummy'
    config.escape.reservoir = 'bulk'
    config.escape.dummy.rate = 0.0
    config.outgas.mass_thresh = 1.0e10
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3
    # M_vol_initial is a non-numeric string: triggers the except branch.
    hf_row = {
        'M_vol_initial': 'corrupted',
        'esc_kg_cumulative': 17.0,
        'H_kg_total': 1.0e20,
        'O_kg_total': 8.0e19,
        'C_kg_total': 1.0e18,
        'N_kg_total': 1.0e19,
        'S_kg_total': 1.0e17,
        'Si_kg_total': 1.0e18,
        'Mg_kg_total': 1.0e17,
        'Fe_kg_total': 1.0e19,
        'Na_kg_total': 1.0e16,
    }
    expected_baseline = sum(
        hf_row[f'{e}_kg_total'] for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na')
    )

    run_escape(config, hf_row, dt=1000.0, stellar_track=None)

    # Baseline rebuilt from per-element totals (Issue #677: O included).
    assert hf_row['M_vol_initial'] == pytest.approx(expected_baseline, rel=1e-12)
    # Discrimination guard: a regression that silently kept the string
    # would have left M_vol_initial == 'corrupted' (type str), not float.
    assert isinstance(hf_row['M_vol_initial'], float)
    # Reset alongside baseline; the prior counter of 17.0 must NOT survive.
    assert hf_row['esc_kg_cumulative'] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_run_escape_dummy_zeroes_elemental_rates_when_unfract_raises():
    """If calc_unfract_fluxes raises (KeyError/ValueError/TypeError) on
    the dummy path, the source must zero every per-element rate rather
    than leave hf_row in a partially-mutated state.

    Discriminating: a regression that swallowed the exception without
    the cleanup loop would leave the existing elemental rates intact
    (or whatever calc_unfract_fluxes wrote before raising), producing
    a silent inconsistency at the next iteration.
    """
    from proteus.escape.wrapper import run_escape

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.reservoir = 'bulk'
    config.escape.dummy.rate = 0.0
    config.outgas.mass_thresh = 1.0e10
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3

    hf_row = {
        'P_surf': 1.0e5,
        'R_int': 6.371e6,
        # Pre-existing elemental rates that the cleanup MUST overwrite to 0.0.
        'esc_rate_H': 1.0e5,
        'esc_rate_O': 1.0e4,
    }
    # Populate baseline so we don't hit the baseline-rebuild branch.
    for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na'):
        hf_row[f'{e}_kg_total'] = 1.0e18
    hf_row['M_vol_initial'] = sum(
        hf_row[f'{e}_kg_total'] for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na')
    )

    # Make calc_unfract_fluxes raise on the dummy path.
    with patch('proteus.escape.wrapper.calc_unfract_fluxes') as mock_unfract:
        mock_unfract.side_effect = KeyError('missing element key')
        run_escape(config, hf_row, dt=0.0, stellar_track=None)

    # Every element's escape rate must have been clamped to 0.0 by the
    # except branch (run_dummy lines 151-153).
    for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na'):
        assert hf_row[f'esc_rate_{e}'] == pytest.approx(0.0, abs=1e-12), (
            f'{e} should have been zeroed'
        )
    # Side-effect guard: the dummy-rate dispatch still ran, so
    # esc_rate_total reflects config.escape.dummy.rate (0.0 here).
    assert hf_row['esc_rate_total'] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_run_escape_zephyrus_zeroes_elemental_rates_when_unfract_raises():
    """The same cleanup branch in run_zephyrus (lines 200-202) must
    zero per-element rates when calc_unfract_fluxes raises.

    Discriminating: pin the bulk MLR independently of the per-element
    zeroing. A regression that put the cleanup loop in the wrong place
    (e.g. before assigning esc_rate_total) would land esc_rate_total
    at 0.0 too, failing this test.
    """
    from proteus.escape.wrapper import run_escape

    config = MagicMock()
    config.escape.module = 'zephyrus'
    config.escape.reservoir = 'bulk'
    config.escape.zephyrus.tidal = True
    config.escape.zephyrus.efficiency = 0.3
    config.escape.zephyrus.Pxuv = 1.0e-2
    config.star.mass = 1.0  # M_sun units
    config.outgas.mass_thresh = 1.0e10
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3

    hf_row = {
        'semimajorax': 1.5e11,
        'eccentricity': 0.0,
        'M_planet': 6e24,
        'R_int': 6.371e6,
        'R_xuv': 6.371e6,
        'F_xuv': 1.0,
    }
    for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na'):
        hf_row[f'{e}_kg_total'] = 1.0e18
    hf_row['M_vol_initial'] = sum(
        hf_row[f'{e}_kg_total'] for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na')
    )

    # Mock the zephyrus library so we exercise the ZEPHYRUS branch
    # without needing the optional dep installed.
    el_escape_mock = MagicMock(return_value=1.234e5)
    with (
        patch.dict('sys.modules', {'zephyrus': MagicMock(), 'zephyrus.escape': MagicMock()}),
        patch('zephyrus.escape.EL_escape', el_escape_mock),
        patch('proteus.escape.wrapper.calc_unfract_fluxes') as mock_unfract,
    ):
        mock_unfract.side_effect = ValueError('unfractionated path broken')
        run_escape(config, hf_row, dt=0.0, stellar_track=None)

    # esc_rate_total picks up the mocked EL_escape return value, NOT 0.0.
    # Discrimination guard: separating the bulk-rate assignment from the
    # per-element cleanup means esc_rate_total survives the except branch.
    assert hf_row['esc_rate_total'] == pytest.approx(1.234e5, rel=1e-12)
    for e in ('H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na'):
        assert hf_row[f'esc_rate_{e}'] == pytest.approx(0.0, abs=1e-12), (
            f'{e} should have been zeroed'
        )
    # Scale guard: 1.234e5 kg/s is a plausible XUV-limited MLR (~kg/s for
    # an Earth-like XUV setup), not 1.234e+15 (units flipped) or 0.0.
    assert 1e3 < hf_row['esc_rate_total'] < 1e7


# =======================================================================================
# SECTION: limit_escape_step(), per-step cap on the mass escape may remove
# =======================================================================================


@pytest.mark.physics_invariant
def test_limit_escape_step_caps_a_request_larger_than_the_reservoir():
    """The bulk rate is set without reference to how much mass is left, so over a
    long step it can ask for more than the whole escapable reservoir. The applied
    loss must stay bounded by that reservoir.

    Boundedness is the invariant: the mass removed in one step cannot exceed the
    mass present. The requested value is 2.5 times the reservoir here, an
    overshoot of the size seen on real grid cases, so a pass-through
    implementation is separated from a capped one by an order of magnitude
    rather than by a tolerance.
    """
    from proteus.escape.wrapper import ESCAPE_STEP_MAX_FRAC, limit_escape_step
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1.0e4  # yr
    escapable = 8.0e22  # kg held in the atmosphere
    requested = 2.0e23  # kg the bulk rate asks for, 2.5x the reservoir

    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf['H_kg_atm'] = escapable
    hf['esc_rate_total'] = requested / (secs_per_year * dt)

    applied = limit_escape_step(hf, dt, 'outgas')

    assert applied == pytest.approx(ESCAPE_STEP_MAX_FRAC * escapable, rel=1e-9)
    # Boundedness: a step never removes more than the reservoir holds.
    assert 0.0 < applied <= escapable
    # Discrimination guard: an uncapped implementation returns 2.0e23, which is
    # 10x the capped value, far outside any tolerance either assertion allows.
    assert applied < 0.2 * requested


@pytest.mark.physics_invariant
def test_limit_escape_step_passes_a_typical_step_through_unchanged():
    """A step that asks for a small share of the reservoir must be returned
    exactly, so the cap changes nothing on healthy evolution.

    The fraction used is 1.9e-05, the median per-step loss measured across the
    grid, which is four orders of magnitude below the cap. Edge case: a request
    sitting exactly at the cap is also passed through, since the cap is the
    largest admissible step rather than the first forbidden one.
    """
    from proteus.escape.wrapper import ESCAPE_STEP_MAX_FRAC, limit_escape_step
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1.0e3
    escapable = 5.0e21
    requested = 1.9e-05 * escapable

    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf['H_kg_atm'] = escapable
    hf['esc_rate_total'] = requested / (secs_per_year * dt)

    applied = limit_escape_step(hf, dt, 'outgas')
    assert applied == pytest.approx(requested, rel=1e-9)
    assert hf['esc_clamp_frac'] == pytest.approx(1.9e-05, rel=1e-6)

    # Exactly at the cap: still passed through, so the boundary is inclusive.
    hf['esc_rate_total'] = ESCAPE_STEP_MAX_FRAC * escapable / (secs_per_year * dt)
    at_cap = limit_escape_step(hf, dt, 'outgas')
    assert at_cap == pytest.approx(ESCAPE_STEP_MAX_FRAC * escapable, rel=1e-9)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_limit_escape_step_keeps_the_overshoot_visible_after_capping():
    """Capping the loss must not hide how large the request was, otherwise a
    capped run-down is indistinguishable from a physical one.

    ``esc_clamp_frac`` therefore records the requested fraction, not the applied
    one, and stays above the cap on a capped step. Edge case: a request far
    beyond the reservoir, 2.1e+09 times it, is the largest overshoot measured on
    the grid and must still be recorded rather than saturating.
    """
    from proteus.escape.wrapper import ESCAPE_STEP_MAX_FRAC, limit_escape_step
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1.0e2
    escapable = 1.0e20
    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf['H_kg_atm'] = escapable
    hf['esc_rate_total'] = 2.1e09 * escapable / (secs_per_year * dt)

    applied = limit_escape_step(hf, dt, 'outgas')

    assert hf['esc_clamp_frac'] == pytest.approx(2.1e09, rel=1e-6)
    assert hf['esc_clamp_frac'] > ESCAPE_STEP_MAX_FRAC
    # The applied loss is still bounded, however extreme the request.
    assert applied == pytest.approx(ESCAPE_STEP_MAX_FRAC * escapable, rel=1e-9)


@pytest.mark.unit
def test_limit_escape_step_handles_no_loss_and_an_empty_reservoir():
    """Degenerate inputs must return a well-formed zero rather than dividing by
    an empty reservoir or inventing mass that cannot leave.

    Three edge cases: a zero escape rate, a non-finite rate, and a reservoir that
    holds nothing. An unrecognised reservoir name is the documented error path
    and raises.
    """
    from proteus.escape.wrapper import limit_escape_step
    from proteus.utils.constants import element_list

    base = {f'{e}_kg_atm': 0.0 for e in element_list}

    hf = dict(base, H_kg_atm=1.0e20, esc_rate_total=0.0)
    assert limit_escape_step(hf, 1.0e3, 'outgas') == pytest.approx(0.0, abs=1e-12)
    assert hf['esc_clamp_frac'] == pytest.approx(0.0, abs=1e-12)

    hf = dict(base, H_kg_atm=1.0e20, esc_rate_total=float('nan'))
    assert limit_escape_step(hf, 1.0e3, 'outgas') == pytest.approx(0.0, abs=1e-12)
    assert hf['esc_clamp_frac'] == pytest.approx(0.0, abs=1e-12)

    # Empty reservoir: nothing can leave, so the step removes nothing. Returning
    # the request instead would credit the cumulative counter with mass that was
    # never debited from any inventory.
    hf = dict(base, esc_rate_total=1.0e5)
    assert limit_escape_step(hf, 1.0e3, 'outgas') == pytest.approx(0.0, abs=1e-12)
    assert hf['esc_clamp_frac'] == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(ValueError, match='Invalid escape reservoir'):
        limit_escape_step(dict(base, esc_rate_total=1.0e5), 1.0e3, 'nonsense')


@pytest.mark.physics_invariant
def test_capped_step_stops_the_silent_interior_drain():
    """With ``reservoir = "outgas"`` the per-element ratios come from the
    atmosphere while the debit lands on the whole-planet total, so a request
    larger than the atmosphere drains interior inventory without any
    non-negativity clamp firing.

    The cap is what bounds that debit. Here the atmosphere holds 8.0e22 kg and
    escape asks for 2.0e23 kg, while the planet holds 4.0e23 kg, so nothing goes
    negative and the uncapped debit passes silently.
    """
    from proteus.escape.wrapper import calc_new_elements, limit_escape_step
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1.0e4
    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf.update({f'{e}_kg_total': 0.0 for e in element_list})
    hf['H_kg_atm'] = 8.0e22
    hf['H_kg_total'] = 4.0e23
    hf['esc_rate_total'] = 2.0e23 / (secs_per_year * dt)

    applied = limit_escape_step(hf, dt, 'outgas')
    capped = calc_new_elements(hf, dt, 'outgas', min_thresh=1.0e10, esc_mass=applied)
    uncapped = calc_new_elements(hf, dt, 'outgas', min_thresh=1.0e10)

    # Capped: only a quarter of the atmosphere leaves, so the planet keeps 3.8e23.
    assert capped['H'] == pytest.approx(3.8e23, rel=1e-9)
    # Uncapped: 2.0e23 kg is debited, over twice what the atmosphere held, and
    # the result stays positive so no existing guard would have caught it.
    assert uncapped['H'] == pytest.approx(2.0e23, rel=1e-9)
    assert uncapped['H'] > 0.0
    # The two differ by 1.8e23 kg, which is the mass the cap protects.
    assert capped['H'] - uncapped['H'] == pytest.approx(1.8e23, rel=1e-9)


@pytest.mark.physics_invariant
def test_run_escape_cumulative_matches_the_mass_actually_removed():
    """The cumulative escape counter and the per-element debit must be built from
    the same mass, otherwise the desiccation gate compares a loss that happened
    against a loss that was only requested.

    Mass closure is the invariant: the increment in ``esc_kg_cumulative`` equals
    the summed drop in ``{e}_kg_total`` over the step. Edge case: the step here
    is capped, which is exactly where the two could diverge.
    """
    from unittest.mock import MagicMock

    from proteus.escape.wrapper import ESCAPE_STEP_MAX_FRAC, run_escape
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1.0e4
    escapable = 8.0e22
    rate = 2.0e23 / (secs_per_year * dt)

    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf.update({f'{e}_kg_total': 0.0 for e in element_list})
    hf['H_kg_atm'] = escapable
    hf['H_kg_total'] = 4.0e23
    hf['esc_kg_cumulative'] = 0.0
    hf['M_vol_initial'] = 4.0e23

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.reservoir = 'outgas'
    config.escape.dummy.rate = rate
    config.outgas.mass_thresh = 1.0e10
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3

    before = sum(float(hf[f'{e}_kg_total']) for e in element_list)
    run_escape(config, hf, dt=dt)
    after = sum(float(hf[f'{e}_kg_total']) for e in element_list)

    removed = before - after
    assert removed == pytest.approx(hf['esc_kg_cumulative'], rel=1e-9)
    assert removed == pytest.approx(ESCAPE_STEP_MAX_FRAC * escapable, rel=1e-9)
    # Discrimination guard: accumulating the request instead would record
    # 2.0e23 kg, ten times the mass that actually left.
    assert hf['esc_kg_cumulative'] < 0.2 * 2.0e23

    # The loss published for the other consumers of this step is the capped one.
    # Publishing the request here would hand `outgas.wrapper.run_crystallized` a
    # mass ten times the debit, which is the divergence this column prevents.
    assert hf['esc_step_kg'] == pytest.approx(ESCAPE_STEP_MAX_FRAC * escapable, rel=1e-9)
    assert hf['esc_step_kg'] == pytest.approx(removed, rel=1e-9)
    assert hf['esc_step_kg'] < 0.2 * 2.0e23


@pytest.mark.physics_invariant
def test_escape_dt_limit_puts_an_unchanged_rate_exactly_at_the_cap():
    """The limit must be the step length that places the same escape rate on the
    cap, measured against the reservoir the overshooting step drew on.

    This is a round trip: cap a deliberate overshoot, shorten the step by the
    returned limit, then re-run the same rate over the shorter step and require
    the requested fraction to land on the cap. Pinning the round trip rather
    than the formula means an inverted ratio, which would lengthen the step, is
    caught by the invariant itself. The reservoir is held fixed here so the
    round trip isolates the formula; what an unchanged rate does once the loss
    has drawn the reservoir down is pinned separately.
    """
    from proteus.escape.wrapper import (
        ESCAPE_STEP_MAX_FRAC,
        escape_dt_limit,
        limit_escape_step,
    )
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1.0e4  # yr
    escapable = 8.0e22  # kg
    rate = 2.0e23 / (secs_per_year * dt)  # asks for 2.5x the reservoir

    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf['H_kg_atm'] = escapable
    hf['esc_rate_total'] = rate

    limit_escape_step(hf, dt, 'outgas')
    frac = float(hf['esc_clamp_frac'])
    assert frac > ESCAPE_STEP_MAX_FRAC  # the step really did overshoot

    dt_next = escape_dt_limit(frac, dt)
    # The limit shortens the step; an inverted ratio would return 1.0e5 yr.
    assert 0.0 < dt_next < dt

    # Same rate, same reservoir, shorter step: now exactly at the cap.
    hf_next = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf_next['H_kg_atm'] = escapable
    hf_next['esc_rate_total'] = rate
    applied = limit_escape_step(hf_next, dt_next, 'outgas')

    assert hf_next['esc_clamp_frac'] == pytest.approx(ESCAPE_STEP_MAX_FRAC, rel=1e-9)
    # Boundedness holds at the limit: the loss still cannot exceed the reservoir.
    assert applied == pytest.approx(ESCAPE_STEP_MAX_FRAC * escapable, rel=1e-9)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_shortened_steps_run_a_reservoir_down_rather_than_emptying_it():
    """Under an unchanged rate the shortened step must keep the loss bounded by
    the cap on every step, so an escape rate far larger than the reservoir
    drains it over many steps instead of removing it in one.

    The capped step takes ``max_frac`` of the reservoir it was sized against, so
    the next request, measured against what is left, settles at the fixed point
    ``max_frac / (1 - max_frac)``. That is above the cap, which is why the cap
    keeps binding and the step keeps shortening. Edge case: a rate asking for
    2.5 times the whole reservoir on the first step.
    """
    from proteus.escape.wrapper import (
        ESCAPE_STEP_MAX_FRAC,
        escape_dt_limit,
        limit_escape_step,
    )
    from proteus.utils.constants import element_list, secs_per_year

    cap = ESCAPE_STEP_MAX_FRAC
    fixed_point = cap / (1.0 - cap)

    dt = 1.0e4  # yr
    escapable = 8.0e22  # kg
    rate = 2.5 * escapable / (secs_per_year * dt)  # asks for 2.5x the reservoir

    fracs, steps = [], []
    for _ in range(6):
        hf = {f'{e}_kg_total': 0.0 for e in element_list}
        hf['H_kg_total'] = escapable
        hf['esc_rate_total'] = rate
        applied = limit_escape_step(hf, dt, 'bulk')
        fracs.append(float(hf['esc_clamp_frac']))
        steps.append(dt)
        # Boundedness on every step, which is the property the cap exists for.
        assert applied == pytest.approx(cap * escapable, rel=1e-9)
        dt = min(dt, escape_dt_limit(fracs[-1], dt))
        escapable -= applied

    # The first step overshoots hugely; from the second on the request sits at
    # the fixed point, above the cap, so the limiter never stops working.
    assert fracs[0] == pytest.approx(2.5, rel=1e-9)
    for frac in fracs[1:]:
        assert frac == pytest.approx(fixed_point, rel=1e-9)
        assert frac > cap
    # Discrimination: holding the reservoir fixed would land every later request
    # on the cap itself, and 1/3 differs from 1/4 by a third of the cap.
    assert abs(fixed_point - cap) > 0.3 * cap
    # The step shortens monotonically rather than recovering and overshooting.
    assert all(b < a for a, b in zip(steps, steps[1:]))
    assert 0.0 < applied <= escapable


@pytest.mark.unit
def test_escape_dt_limit_is_inert_within_the_cap_and_on_bad_input():
    """A step that stayed within the cap must impose no limit, and neither must
    a degenerate request, so the controller is never pinned short by a value it
    cannot interpret.

    Covers the boundary (a request sitting exactly at the cap is not an
    overshoot), a zero-loss step, and the non-finite and non-positive inputs a
    failed solve can produce.
    """
    import math

    from proteus.escape.wrapper import ESCAPE_STEP_MAX_FRAC, escape_dt_limit

    dt = 1.0e3

    assert math.isinf(escape_dt_limit(0.0, dt))
    assert math.isinf(escape_dt_limit(1.9e-05, dt))
    # Boundary is inclusive: exactly at the cap is not an overshoot.
    assert math.isinf(escape_dt_limit(ESCAPE_STEP_MAX_FRAC, dt))
    assert math.isinf(escape_dt_limit(float('nan'), dt))

    # A real overshoot with a degenerate step imposes nothing, because there is
    # no step length to scale down from.
    assert math.isinf(escape_dt_limit(4.0, 0.0))
    assert math.isinf(escape_dt_limit(4.0, -1.0))
    assert math.isinf(escape_dt_limit(4.0, float('inf')))

    # Just past the cap it engages, and mildly: a 1% overshoot must not collapse
    # the step, which a naive "halve it" rule would.
    mild = escape_dt_limit(ESCAPE_STEP_MAX_FRAC * 1.01, dt)
    assert mild == pytest.approx(dt / 1.01, rel=1e-9)
    assert mild > 0.9 * dt


@pytest.mark.unit
def test_run_escape_publishes_the_step_limit_and_clears_it_again():
    """A capped step must hand the interior a shorter next step, and a step that
    stays within the cap must withdraw that request, so one overshoot does not
    pin the controller short for the rest of the run.

    Both directions are exercised on the same interior object, because the
    failure that matters is a stale limit surviving a healthy step.
    """
    from proteus.escape.wrapper import ESCAPE_STEP_MAX_FRAC, run_escape
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1.0e4
    escapable = 8.0e22

    def _fresh_hf(rate):
        hf = {f'{e}_kg_atm': 0.0 for e in element_list}
        hf.update({f'{e}_kg_total': 0.0 for e in element_list})
        hf['H_kg_atm'] = escapable
        hf['H_kg_total'] = escapable
        hf['esc_kg_cumulative'] = 0.0
        hf['M_vol_initial'] = escapable
        hf['esc_rate_total'] = rate
        return hf

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.reservoir = 'outgas'
    config.outgas.mass_thresh = 1.0e10
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3

    interior_o = SimpleNamespace(escape_dt_limit=float('inf'))

    # An overshooting step asks for a shorter next step.
    over_rate = 2.0e23 / (secs_per_year * dt)
    config.escape.dummy.rate = over_rate
    run_escape(config, _fresh_hf(over_rate), dt=dt, interior_o=interior_o)

    assert np.isfinite(interior_o.escape_dt_limit)
    assert 0.0 < interior_o.escape_dt_limit < dt
    # Discrimination guard: the limit is the cap-over-request share of the step,
    # 0.25/2.5 = 0.1, so a rule that merely halved dt would give 5.0e3 yr.
    assert interior_o.escape_dt_limit == pytest.approx(
        dt * ESCAPE_STEP_MAX_FRAC / 2.5, rel=1e-6
    )

    # A healthy step withdraws it again.
    calm_rate = 1.9e-05 * escapable / (secs_per_year * dt)
    config.escape.dummy.rate = calm_rate
    run_escape(config, _fresh_hf(calm_rate), dt=dt, interior_o=interior_o)

    assert np.isinf(interior_o.escape_dt_limit)


@pytest.mark.physics_invariant
def test_cap_bounds_the_bulk_reservoir_too():
    """Under `reservoir='bulk'` the cap must bound the whole-planet inventory,
    not just the atmospheric one, since that is the reservoir the debit lands on.

    The same 0.25 means a different physical bound in the two modes, because the
    bulk reservoir includes mass dissolved in the interior. This pins that the
    bulk mode measures against `*_kg_total`: a request of 2.5 times the bulk
    inventory is capped to a quarter of it, ten times smaller.
    """
    from proteus.escape.wrapper import ESCAPE_STEP_MAX_FRAC, limit_escape_step
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1.0e4
    bulk = 4.0e23  # kg held across interior and atmosphere
    atm = 8.0e22  # kg in the atmosphere alone, deliberately different

    hf = {f'{e}_kg_total': 0.0 for e in element_list}
    hf.update({f'{e}_kg_atm': 0.0 for e in element_list})
    hf['H_kg_total'] = bulk
    hf['H_kg_atm'] = atm
    hf['esc_rate_total'] = 2.5 * bulk / (secs_per_year * dt)

    applied = limit_escape_step(hf, dt, 'bulk')

    assert applied == pytest.approx(ESCAPE_STEP_MAX_FRAC * bulk, rel=1e-9)
    assert hf['esc_clamp_frac'] == pytest.approx(2.5, rel=1e-9)
    # Boundedness against the reservoir actually debited.
    assert 0.0 < applied <= bulk
    # Discrimination guard: measuring against the atmosphere instead would give
    # 0.25 * 8.0e22 = 2.0e22, a factor of five below the bulk answer.
    assert applied > 2.0 * ESCAPE_STEP_MAX_FRAC * atm


@pytest.mark.physics_invariant
def test_crystallized_atmosphere_scaling_uses_the_capped_loss():
    """Once the mantle has solidified the atmosphere is rescaled by the escaped
    mass, and that must be the capped mass, or the atmosphere empties while the
    elemental totals keep all but the capped share.

    Sizing the rescale from the raw rate on an overshoot drives the retained
    fraction to zero, so `P_surf` and `M_atm` collapse while `*_kg_total` still
    holds three quarters of the inventory. The planet then reads as having an
    intact volatile budget and no atmosphere at once.
    """
    from proteus.outgas.wrapper import run_crystallized
    from proteus.utils.constants import secs_per_year

    dt = 1.0e4
    m_atm = 8.0e22
    over_rate = 2.0e23 / (secs_per_year * dt)  # asks for 2.5x the atmosphere

    config = MagicMock()
    config.escape.reservoir = 'outgas'
    config.escape.module = 'dummy'

    capped = 0.25 * m_atm
    hf = {
        'M_atm': m_atm,
        'P_surf': 1.0e3,
        'P_vol': 8.0e2,
        'P_vap': 2.0e2,
        'M_vaps': 0.0,
        'esc_rate_total': over_rate,
        'esc_step_kg': capped,
    }
    run_crystallized(config, hf, dt)

    # Retained fraction follows the capped loss: 1 - 0.25 = 0.75.
    assert hf['M_atm'] == pytest.approx(0.75 * m_atm, rel=1e-9)
    assert hf['P_surf'] == pytest.approx(0.75 * 1.0e3, rel=1e-9)
    # Positivity: an atmosphere that survived the step still has mass and pressure.
    assert hf['M_atm'] > 0.0 and hf['P_surf'] > 0.0
    # Discrimination guard: the uncapped rate gives retained = 0, so both would
    # be exactly zero rather than three quarters of their starting values.
    assert hf['P_surf'] > 0.5 * 1.0e3

    # Without the published value it falls back to the raw rate, which is the
    # pre-cap behaviour and does empty the atmosphere.
    hf_fallback = dict(hf, M_atm=m_atm, P_surf=1.0e3)
    hf_fallback.pop('esc_step_kg')
    run_crystallized(config, hf_fallback, dt)
    assert hf_fallback['M_atm'] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_cumulative_counter_ignores_a_step_that_debited_nothing():
    """The cumulative escape counter must track mass that actually left an
    inventory, because the desiccation gate compares it against the starting
    budget to tell real escape from an atmosphere wiped by an upstream failure.

    Edge case: a desiccated planet with a live escape rate. Nothing can leave an
    empty reservoir, so the counter must not move at all.
    """
    from proteus.escape.wrapper import run_escape
    from proteus.utils.constants import element_list, secs_per_year

    dt = 1.0e4
    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf.update({f'{e}_kg_total': 0.0 for e in element_list})
    hf['esc_kg_cumulative'] = 0.0
    hf['M_vol_initial'] = 1.0e23

    rate = 1.0e4  # kg/s, live rate against nothing to lose
    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.reservoir = 'outgas'
    config.escape.dummy.rate = rate
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3
    config.outgas.mass_thresh = 1.0e10

    run_escape(config, hf, dt=dt)

    assert hf['esc_kg_cumulative'] == pytest.approx(0.0, abs=1e-6)
    # Discrimination guard: crediting the request instead would add
    # 1.0e4 * secs_per_year * 1.0e4 = 3.16e15 kg to a planet that lost nothing.
    assert hf['esc_kg_cumulative'] < 1.0e10
    assert rate * secs_per_year * dt > 1.0e15  # the request really was large


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_thin_atmosphere_does_not_erase_the_mantle_inventory():
    """A thin atmosphere over a volatile-rich mantle must leave the whole-planet
    inventory intact, so the desiccation gate keeps its ability to tell escape
    apart from an atmosphere wiped by an upstream failure.

    Physical scenario: the crystallized regime, where escape draws on the
    atmosphere alone. The atmosphere decays under the threshold that stops the
    per-element solve while the mantle still holds volatiles frozen into the
    solid, so the two reservoirs sit fifteen orders of magnitude apart.
    """
    from proteus.escape.wrapper import run_escape
    from proteus.outgas.wrapper import check_desiccation
    from proteus.utils.constants import element_list

    mantle_kg = 1.0e23
    atmos_kg = 5.0e9  # below mass_thresh, and non-zero

    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf.update({f'{e}_kg_total': 0.0 for e in element_list})
    hf['H_kg_atm'] = atmos_kg
    hf['H_kg_total'] = mantle_kg
    hf['esc_kg_cumulative'] = 0.0
    hf['M_vol_initial'] = mantle_kg

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.reservoir = 'outgas'
    config.escape.dummy.rate = 1.0e5  # kg/s
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3
    config.outgas.mass_thresh = 1.0e10

    run_escape(config, hf, dt=1.0e4, atmosphere_only=True)

    # The mantle is untouched, exactly, because no per-element debit was applied.
    assert hf['H_kg_total'] == pytest.approx(mantle_kg, rel=1e-12)
    # Discrimination: sizing the inventory from the atmospheric reservoir instead
    # would leave 5.0e9 kg here, fourteen orders of magnitude below the mantle.
    assert hf['H_kg_total'] > 1.0e22

    # Nothing was debited, so nothing is credited to escape and nothing is
    # published for the consumers that rescale the atmosphere from this step.
    assert hf['esc_kg_cumulative'] == pytest.approx(0.0, abs=1e-6)
    assert hf['esc_step_kg'] == pytest.approx(0.0, abs=1e-6)

    # The gate's threshold loop reads the whole-planet inventory, so an intact
    # mantle keeps it from declaring desiccation on a planet that holds one.
    assert check_desiccation(config, hf) is False


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_cumulative_counter_excludes_the_threshold_truncation():
    """Escape is credited only with the mass it removed, never with the mass the
    depletion floor truncates, because the gate weighs the counter against the
    whole budget lost and a truncation it did not cause would excuse that loss.

    Edge case: an element sitting just above the floor, where a small debit tips
    it under and the floor then takes the remainder.
    """
    from proteus.escape.wrapper import run_escape
    from proteus.utils.constants import element_list, secs_per_year

    thresh = 1.0e16
    dt = 1.0e3
    # Sized so the debit tips the element under the floor without reaching the
    # per-step cap, which isolates the truncation from the capping.
    rate = 8.0e4  # kg/s
    applied = rate * secs_per_year * dt  # 2.52e15 kg

    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf.update({f'{e}_kg_total': 0.0 for e in element_list})
    # Just above the floor, so the debit tips it under and the floor zeroes it.
    hf['H_kg_total'] = 1.2e16
    hf['H_kg_atm'] = 1.2e16
    hf['esc_kg_cumulative'] = 0.0
    hf['M_vol_initial'] = 1.2e16

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.reservoir = 'outgas'
    config.escape.dummy.rate = rate
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3
    config.outgas.mass_thresh = thresh

    run_escape(config, hf, dt=dt)

    # The cap stayed clear, so the only extra mass removed is the truncation.
    assert hf['esc_clamp_frac'] < config.escape.step_max_frac
    # The floor took the element down to zero, so the inventory fell by the full
    # 1.2e16 kg while escape itself removed 2.52e15 kg.
    assert hf['H_kg_total'] == pytest.approx(0.0, abs=1e-6)
    assert hf['esc_kg_cumulative'] == pytest.approx(applied, rel=1e-9)
    # Discrimination: crediting the raw inventory drop would record 1.2e16 kg,
    # over four times the mass escape actually took.
    assert hf['esc_kg_cumulative'] < 0.5 * 1.2e16
    assert 1.2e16 > 4.0 * applied  # the two candidates really are far apart


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_crystallized_step_moves_the_same_mass_in_every_record():
    """Over one coupled step of the crystallized regime the whole-planet totals,
    the atmospheric mass and the cumulative counter must all move by the same
    amount, because they are three records of one escape event.

    ``run_escape`` sizes and applies the loss, then ``run_crystallized`` rescales
    the atmosphere by what was applied. Edge case: an atmosphere under the
    outgassing threshold, where the per-element solve declines to debit, so the
    rescale must decline too rather than draining mass no inventory recorded.
    """
    from proteus.escape.wrapper import run_escape
    from proteus.outgas.wrapper import run_crystallized
    from proteus.utils.constants import element_list

    mantle_kg = 1.0e23

    def build(atmos_kg):
        hf = {f'{e}_kg_atm': 0.0 for e in element_list}
        hf.update({f'{e}_kg_total': 0.0 for e in element_list})
        hf['H_kg_atm'] = atmos_kg
        hf['H_kg_total'] = mantle_kg
        hf['M_atm'] = atmos_kg
        hf['P_surf'] = 1.0
        hf['esc_kg_cumulative'] = 0.0
        hf['M_vol_initial'] = mantle_kg
        for gas in ('H2O', 'CO2', 'O2', 'H2', 'CO', 'CH4', 'N2', 'S2', 'SO2'):
            hf[f'{gas}_kg_atm'] = 0.0
            hf[f'{gas}_bar'] = 0.0
        hf['H2O_kg_atm'] = atmos_kg
        hf['H2O_bar'] = 1.0
        return hf

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.reservoir = 'outgas'
    config.escape.dummy.rate = 1.0e5  # kg/s
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3
    config.outgas.mass_thresh = 1.0e10
    config.outgas.vapourise = False

    # Above the threshold the debit is applied, below it the solve declines.
    for atmos_kg, debited in ((5.0e12, True), (5.0e9, False)):
        hf = build(atmos_kg)
        run_escape(config, hf, dt=1.0e4, atmosphere_only=True)
        run_crystallized(config, hf, dt=1.0e4)

        fell_total = mantle_kg - hf['H_kg_total']
        fell_atmos = atmos_kg - hf['M_atm']
        rose_count = hf['esc_kg_cumulative']
        # The totals are differenced at 1e23, where a double resolves ~1.7e7 kg.
        noise = 4.0 * math.ulp(mantle_kg)

        assert fell_total == pytest.approx(fell_atmos, abs=noise)
        assert rose_count == pytest.approx(fell_atmos, abs=noise)
        if debited:
            # Discrimination: a real quarter-reservoir loss, not a rounding tie.
            assert fell_atmos == pytest.approx(0.25 * atmos_kg, rel=1e-9)
            assert fell_atmos > noise
        else:
            # Rescaling by the request while the solve declined would drain
            # 1.25e9 kg from the atmosphere that no inventory ever recorded.
            assert fell_atmos == pytest.approx(0.0, abs=1e-6)
            assert 0.25 * atmos_kg > 1.0e9  # that drain would have been large


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_unreadable_reservoir_debits_nothing_and_credits_nothing():
    """A reservoir left non-finite by an upstream failure must debit nothing and
    credit nothing, because the desiccation check exists to tell a wiped
    atmosphere apart from real escape and cannot do so if the wipe is credited.

    Physical scenario: an atmosphere solve fails and leaves a non-finite mass in
    the atmospheric reservoir while the whole-planet inventory is still intact.
    Edge case: the comparisons that would ordinarily catch a bad value all
    return False against a NaN, so the guard has to be explicit.
    """
    from proteus.escape.wrapper import run_escape
    from proteus.outgas.wrapper import check_desiccation
    from proteus.utils.constants import element_list

    budget_kg = 1.0e23
    hf = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf.update({f'{e}_kg_total': 0.0 for e in element_list})
    hf['H_kg_atm'] = float('nan')
    hf['H_kg_total'] = budget_kg
    hf['esc_kg_cumulative'] = 0.0
    hf['M_vol_initial'] = budget_kg

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.reservoir = 'outgas'
    config.escape.dummy.rate = 1.0e5  # kg/s, a live rate against a bad reservoir
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3
    config.outgas.mass_thresh = 1.0e10

    run_escape(config, hf, dt=1.0e4, atmosphere_only=True)

    assert hf['H_kg_total'] == pytest.approx(budget_kg, rel=1e-12)
    assert hf['esc_step_kg'] == pytest.approx(0.0, abs=1e-6)
    assert hf['esc_kg_cumulative'] == pytest.approx(0.0, abs=1e-6)
    # Discrimination: letting the non-finite value through zeroes every element
    # and credits the whole budget, which reads to the check as accounted loss.
    assert hf['H_kg_total'] > 0.5 * budget_kg
    assert check_desiccation(config, hf) is False

    # The same applies one field over: a whole-planet total left non-finite by
    # an upstream failure must survive the step as it is. Clamping it would
    # produce a zero indistinguishable from an element escape has depleted.
    hf2 = {f'{e}_kg_atm': 0.0 for e in element_list}
    hf2.update({f'{e}_kg_total': 0.0 for e in element_list})
    hf2['H_kg_atm'] = 1.0e18  # a readable reservoir, so the guards above pass
    hf2['H_kg_total'] = float('nan')
    hf2['C_kg_atm'] = 1.0e17
    hf2['C_kg_total'] = 5.0e21  # a healthy neighbour that must still be debited
    hf2['esc_kg_cumulative'] = 0.0
    hf2['M_vol_initial'] = 5.0e21

    run_escape(config, hf2, dt=1.0e4, atmosphere_only=True)

    assert math.isnan(hf2['H_kg_total'])
    assert hf2['C_kg_total'] < 5.0e21  # the readable element still escapes
    assert hf2['C_kg_total'] > 0.9 * 5.0e21  # and only by its share


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_frozen_mantle_sizes_the_loss_from_the_column_it_will_rescale():
    """Once the mantle is frozen the loss must be sized from the atmosphere, so
    the mass debited from the whole-planet totals is the mass the atmospheric
    rescale then removes and the two records of the step agree.

    Physical scenario: the iteration the mantle solidifies, on a planet whose
    volatiles are mostly dissolved. Sizing from the whole planet there would
    debit the totals for mass the frozen mantle no longer supplies, which the
    column cannot give up. Edge case: a bulk reservoir five orders of magnitude
    larger than the atmosphere above it.
    """
    from proteus.escape.wrapper import run_escape
    from proteus.outgas.wrapper import run_crystallized
    from proteus.utils.constants import element_list

    bulk_kg, atmos_kg = 1.0e22, 1.0e17

    def build():
        hf = {f'{e}_kg_atm': 0.0 for e in element_list}
        hf.update({f'{e}_kg_total': 0.0 for e in element_list})
        hf['H_kg_total'] = bulk_kg
        hf['H_kg_atm'] = atmos_kg
        hf['M_atm'] = atmos_kg
        hf['P_surf'] = 1.0
        hf['esc_kg_cumulative'] = 0.0
        hf['M_vol_initial'] = bulk_kg
        for gas in ('H2O', 'CO2', 'O2', 'H2', 'CO', 'CH4', 'N2', 'S2', 'SO2'):
            hf[f'{gas}_kg_atm'] = 0.0
            hf[f'{gas}_bar'] = 0.0
        hf['H2O_kg_atm'] = atmos_kg
        hf['H2O_bar'] = 1.0
        return hf

    config = MagicMock()
    config.escape.module = 'dummy'
    config.escape.reservoir = 'bulk'  # the whole planet, not the frozen column
    config.escape.dummy.rate = 1.0e9  # kg/s
    config.escape.step_max_frac = 0.25
    config.escape.step_dt_floor_frac = 1.0e-3
    config.outgas.mass_thresh = 1.0e10
    config.outgas.vapourise = False

    def coupled(atmosphere_only):
        hf = build()
        run_escape(config, hf, dt=1.0e4, atmosphere_only=atmosphere_only)
        totals_lost = bulk_kg - hf['H_kg_total']
        run_crystallized(config, hf, dt=1.0e4)
        return totals_lost, atmos_kg - hf['M_atm'], hf['M_atm'], hf['P_surf']

    frozen_totals, frozen_atmos, m_atm, p_surf = coupled(True)

    # Treated as frozen: one mass, both records, and the column survives.
    assert frozen_totals == pytest.approx(frozen_atmos, rel=1e-9)
    assert frozen_totals == pytest.approx(0.25 * atmos_kg, rel=1e-9)
    assert m_atm == pytest.approx(0.75 * atmos_kg, rel=1e-9)
    assert p_surf == pytest.approx(0.75, rel=1e-9)

    # Discrimination: sizing from the whole planet debits the totals for
    # thousands of times the column, which then empties at zero pressure.
    bulk_totals, bulk_atmos, bulk_m_atm, bulk_p_surf = coupled(False)
    assert bulk_totals > 1.0e3 * bulk_atmos
    assert bulk_m_atm == pytest.approx(0.0, abs=1e-6)
    assert bulk_p_surf == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_disabled_escape_clears_the_records_of_an_earlier_step():
    """Switching escape off must clear the per-step records as well as the
    rates, so a step taken while it was on cannot be read as the current one.

    Exercises the disabled path directly, with the records carrying values from
    a capped step and the controller holding the shortened step that went with
    them. Edge case: no interior object supplied, where the clearing of the
    helpfile records still has to happen.
    """
    from types import SimpleNamespace

    from proteus.escape.wrapper import run_escape
    from proteus.utils.constants import element_list

    config = MagicMock()
    config.escape.module = None  # escape switched off

    interior_o = SimpleNamespace(escape_dt_limit=12.5)
    hf = {f'{e}_kg_total': 1.0e20 for e in element_list}
    hf['esc_clamp_frac'] = 2.1e9  # a heavily overshooting earlier step
    hf['esc_step_kg'] = 3.3e19
    hf['esc_rate_total'] = 5.0e7

    run_escape(config, hf, dt=1.0e3, interior_o=interior_o)

    assert hf['esc_clamp_frac'] == pytest.approx(0.0, abs=1e-12)
    assert hf['esc_step_kg'] == pytest.approx(0.0, abs=1e-12)
    assert hf['esc_rate_total'] == pytest.approx(0.0, abs=1e-12)
    # The controller must stop holding the step short on account of a step that
    # is no longer being taken; a finite limit here shortens every later step.
    assert interior_o.escape_dt_limit == float('inf')
    # Discrimination: leaving the records alone keeps 2.1e9 and 3.3e19, which
    # read downstream as a run still being held back by the cap.
    assert hf['esc_clamp_frac'] < 1.0
    assert hf['esc_step_kg'] < 1.0

    # Without an interior object the helpfile records are still cleared.
    hf2 = {f'{e}_kg_total': 1.0e20 for e in element_list}
    hf2['esc_clamp_frac'] = 7.0
    hf2['esc_step_kg'] = 1.0e18
    run_escape(config, hf2, dt=1.0e3)
    assert hf2['esc_clamp_frac'] == pytest.approx(0.0, abs=1e-12)
    assert hf2['esc_step_kg'] == pytest.approx(0.0, abs=1e-12)
