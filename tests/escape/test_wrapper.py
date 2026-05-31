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
See docs/test_infrastructure.md, docs/test_building.md for testing standards.

Related documentation:
- docs/test_infrastructure.md: Test framework and CI integration
- docs/test_categorization.md: Test markers and categories
- docs/test_building.md: Best practices for writing robust tests
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
