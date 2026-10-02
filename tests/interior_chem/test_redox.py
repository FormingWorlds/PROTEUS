"""Melt Fe3+/Fe2+ redox tracking (``src/proteus/interior_chem/redox.py``).

Covers the initial ferric fraction ``f_0`` now that it is a user-settable
config field (``planet.ferric_fraction_initial``) rather than a module
constant. Contract clauses exercised here:

* Step 1-2 (Eq 1-3): Fe2+ is seeded from the ferrous FeO fraction,
  ``W_FET * M_melt / MU_FEO``, and Fe3+ is added on top so that the ferric
  fraction equals the value the caller asked for.
* The open-interval domain of ``f_0``: the tracker forms ``f/(1-f)``, so
  the endpoints are rejected rather than silently producing a degenerate
  or infinite redox ratio.
* End-to-end propagation: a non-default value in the config reaches
  ``hf_row['ferric_frac_mantle']`` on the first coupling step.
* The no-op contract: the tracker does nothing unless
  ``planet.fO2_source == 'from_mantle_redox'``.
* Persistence: the Fe-metal diagnostics are helpfile schema columns and
  survive the write to and read from ``runtime_helpfile.csv``.

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.interior_chem.disproportionation import activity_Fe_metal
from proteus.interior_chem.redox import (
    D_FE3_BRG,
    D_FE3_SHALLOW,
    MU_FEO,
    MU_FEO15,
    P_CUTOFF_GPA,
    W_FET,
    _init_state,
    _metal_saturation_step,
    update_melt_redox,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Asymmetric mesh: one fully molten cell, one half molten, one solid, with
# unequal cell masses. Symmetric or all-equal input would let a bug that
# drops the phi weighting still reproduce the right melt mass.
_PHI = np.array([1.0, 0.5, 0.0])
_MASS = np.array([2.0e21, 4.0e21, 6.0e21])
# Straddles P_CUTOFF_GPA so both the Cpx/Opx and bridgmanite branches of
# the partition-coefficient split are exercised in every test.
_PRES = np.array([10.0e9, 30.0e9, 50.0e9])
# Monotonically increasing with depth and spanning 1200 K, so the coolest and
# hottest cells give clearly different equilibrium constants and the binding
# cell the metal step selects is unambiguous.
_TEMP = np.array([2200.0, 2800.0, 3400.0])

# Melt inventory for the direct chemistry comparisons below: bulk-silicate
# iron on a single-cation oxide basis, in moles per 100 g of melt.
_N_FET_TEST = 7.82 / 71.844
_N_SIL_TEST = _N_FET_TEST + 1.85945

# M_melt = 1.0*2e21 + 0.5*4e21 + 0.0*6e21 = 4e21 kg
_M_MELT = 4.0e21


def _make_config(f_0: float, source: str = 'from_mantle_redox') -> MagicMock:
    """Minimal config exposing only the fields the tracker reads."""
    config = MagicMock()
    config.planet.fO2_source = source
    config.planet.ferric_fraction_initial = f_0
    config.outgas.T_floor = 700.0  # the outgassing temperature floor default
    return config


def _make_interior() -> MagicMock:
    """Interior_t stand-in carrying the three radial profiles and no state."""
    interior = MagicMock()
    interior.phi = _PHI
    interior.mass = _MASS
    interior.pres = _PRES
    interior.temp = _TEMP
    interior.redox_state = None
    return interior


@pytest.mark.physics_invariant
def test_init_state_splits_the_melt_iron_inventory_by_the_requested_ferric_fraction():
    """Step 1-2: the ferrous FeO fraction fixes Fe2+ at W_FET * M_melt / MU_FEO,
    and Fe3+ is added on top so that Fe3+ / (Fe2+ + Fe3+) equals f_0.

    Uses f_0 = 0.25 rather than 0.5 so a swapped Fe3+/Fe2+ assignment lands
    at 0.75 and cannot pass, and so the two readings of W_FET (ferrous only
    versus total iron) differ by a factor 1 / (1 - f_0) = 1.33 in total iron.
    """
    f_0 = 0.25
    state = _init_state(_PHI, _MASS, _PRES, f_0)

    n_fe2_expected = W_FET * _M_MELT / MU_FEO  # mol of Fe2+ in the melt
    n_total = state.n_fe3_melt + state.n_fe2_melt

    # Fe2+ is set by the ferrous FeO fraction alone.
    assert state.n_fe2_melt == pytest.approx(n_fe2_expected, rel=1e-12)
    # The split is the requested fraction, not the module default of 0.1.
    assert state.n_fe3_melt / n_total == pytest.approx(f_0, rel=1e-12)
    assert state.ferric_frac == pytest.approx(f_0, rel=1e-12)
    # Closure: total iron is Fe2+ / (1 - f_0), i.e. Fe3+ sits on top of the
    # ferrous inventory rather than being carved out of it.
    assert n_total == pytest.approx(n_fe2_expected / (1.0 - f_0), rel=1e-12)
    # Discrimination guard: reading W_FET as total iron would give a total of
    # n_fe2_expected, 25% below the correct value, far outside rel=1e-12.
    assert abs(n_total / n_fe2_expected - 1.0) > 0.3
    # The FeO1.5 mass fraction carried into the metal step matches Fe3+.
    assert state.w_feo15 == pytest.approx(
        state.n_fe3_melt * MU_FEO15 / _M_MELT, rel=1e-12
    )
    # Swapped-assignment guard: Fe2+ is the majority reservoir at f_0 = 0.25,
    # so a transposed split would put 0.75 here and miss by 0.5.
    assert abs(state.n_fe3_melt / n_total - (1.0 - f_0)) > 0.4
    # Sign guard: mole counts are physical amounts, never negative.
    assert state.n_fe3_melt > 0
    assert state.n_fe2_melt > 0
    # Scale guard: ~5.9e21 mol, not ~5.9e18 (a g-vs-kg slip in MU_FEO) or
    # ~5.9e24 (dropping the phi weighting and using the whole mantle mass).
    assert 1e21 < n_total < 1e22


@pytest.mark.physics_invariant
def test_redox_ratio_is_the_odds_form_of_the_configured_ferric_fraction():
    """The seeded redox ratio is f/(1-f), and it increases monotonically with
    f_0. Pins the mapping the Hirschmann fO2 expression consumes via log10.

    f_0 = 0.2 gives 0.25 and f_0 = 0.8 gives 4.0: a factor of 16 apart, so a
    regression that returned f_0 itself (0.2 vs 0.8, factor 4) cannot pass.
    """
    low = _init_state(_PHI, _MASS, _PRES, 0.2)
    high = _init_state(_PHI, _MASS, _PRES, 0.8)

    assert low.redox_ratio == pytest.approx(0.2 / 0.8, rel=1e-12)
    assert high.redox_ratio == pytest.approx(0.8 / 0.2, rel=1e-12)
    # Monotonicity: a more oxidised melt has the larger Fe3+/Fe2+ ratio.
    assert high.redox_ratio > low.redox_ratio
    # Discrimination: returning f_0 unchanged would give 0.2 and 0.8, a
    # ratio of 4; the odds form gives 0.25 and 4.0, a ratio of 16.
    assert high.redox_ratio / low.redox_ratio == pytest.approx(16.0, rel=1e-12)
    # Both ratios are strictly positive, so log10 in the fO2 term is defined.
    assert low.redox_ratio > 0


@pytest.mark.parametrize('bad_f0', [0.0, 1.0, -0.1, 1.5, 2.0])
def test_init_state_rejects_a_ferric_fraction_outside_the_open_unit_interval(bad_f0):
    """Error contract: the endpoints and any value outside (0, 1) raise rather
    than seeding a degenerate state.

    f_0 = 1 divides by zero forming f/(1-f); f_0 = 0 sends log10(ratio) to
    -inf inside the fO2 expression. Both would otherwise propagate as a nan
    or inf Delta-IW several steps downstream of the real mistake.
    """
    with pytest.raises(ValueError, match=r'\(0, 1\)'):
        _init_state(_PHI, _MASS, _PRES, bad_f0)


def test_init_state_accepts_values_just_inside_the_open_interval():
    """Edge case: the interval is open, not empty. Values adjacent to the
    rejected endpoints are valid and produce a finite redox ratio.

    Pairs with the rejection test above: together they pin the boundary at
    exactly 0 and 1 rather than at some wider interior margin.
    """
    near_zero = _init_state(_PHI, _MASS, _PRES, 1e-6)
    near_one = _init_state(_PHI, _MASS, _PRES, 1.0 - 1e-6)

    assert np.isfinite(near_zero.redox_ratio)
    assert np.isfinite(near_one.redox_ratio)
    # The odds form diverges at 1, so the near-one ratio is enormous but
    # still finite; near zero it is tiny but still strictly positive.
    assert near_zero.redox_ratio > 0
    assert near_one.redox_ratio > 1e5


@pytest.mark.physics_invariant
def test_partition_coefficients_switch_at_the_bridgmanite_pressure_cutoff():
    """Step 1-2: the per-cell Fe3+ partition coefficient takes the bridgmanite
    value at and below the cutoff depth and the Cpx/Opx mean above it.

    The mesh straddles P_CUTOFF_GPA (10, 30, 50 GPa against a 22 GPa cutoff),
    so a dropped or inverted comparison changes at least one cell.
    """
    state = _init_state(_PHI, _MASS, _PRES, 0.3)

    # 10 GPa is shallower than the 22 GPa cutoff: Cpx/Opx assemblage.
    assert state.D_fe3_cell[0] == pytest.approx(D_FE3_SHALLOW, rel=1e-12)
    # 30 and 50 GPa are deeper: bridgmanite.
    assert state.D_fe3_cell[1] == pytest.approx(D_FE3_BRG, rel=1e-12)
    assert state.D_fe3_cell[2] == pytest.approx(D_FE3_BRG, rel=1e-12)
    # Inverted-comparison guard: the two branches differ by ~0.4, far beyond
    # any tolerance, so a flipped >= would fail the assertions above.
    assert abs(D_FE3_BRG - D_FE3_SHALLOW) > 0.3
    # Partition coefficients are bounded fractions, not free parameters.
    assert 0.0 < D_FE3_SHALLOW < 1.0
    assert _PRES[1] > P_CUTOFF_GPA * 1e9


@pytest.mark.parametrize('f_0', [0.02, 0.1, 0.45])
def test_configured_ferric_fraction_reaches_the_helpfile_on_the_first_step(f_0):
    """End-to-end: planet.ferric_fraction_initial is what the tracker seeds and
    reports, so a parameter sweep over it actually varies the model.

    Sweeps a reducing, the default, and a strongly oxidised melt. Before this
    field existed every case would have reported 0.1.
    """
    interior = _make_interior()
    hf_row = {'T_magma': 2000.0}

    update_melt_redox(interior, hf_row, _make_config(f_0))

    assert hf_row['ferric_frac_mantle'] == pytest.approx(f_0, rel=1e-12)
    # The state really was seeded from it, not just echoed into hf_row.
    assert interior.redox_state.ferric_frac == pytest.approx(f_0, rel=1e-12)
    # Delta-IW is finite and physically sized: magma-ocean surface values sit
    # within a few log units of the buffer, not at nan or hundreds.
    assert np.isfinite(hf_row['fO2_shift_IW_mantle'])
    assert -20.0 < hf_row['fO2_shift_IW_mantle'] < 20.0


def test_a_more_oxidised_initial_melt_gives_a_higher_surface_delta_iw():
    """Monotonicity across the new parameter: raising f_0 at fixed T_magma
    raises the surface Delta-IW, because Delta-IW goes as log10(f/(1-f))/a
    with a > 0 in Hirschmann Eq 21.

    This is the property a parameter study over f_0 relies on; a wiring bug
    that ignored the config would make the two runs identical.
    """
    hf_reduced: dict = {'T_magma': 2000.0}
    hf_oxidised: dict = {'T_magma': 2000.0}

    update_melt_redox(_make_interior(), hf_reduced, _make_config(0.05))
    update_melt_redox(_make_interior(), hf_oxidised, _make_config(0.40))

    assert hf_oxidised['fO2_shift_IW_mantle'] > hf_reduced['fO2_shift_IW_mantle']
    # Discrimination: the separation is log10(odds ratio)/a, which for these
    # two fractions is over 4 log units. A config that failed to propagate
    # would give exactly 0.0 separation.
    separation = hf_oxidised['fO2_shift_IW_mantle'] - hf_reduced['fO2_shift_IW_mantle']
    assert separation > 1.0
    # Same temperature, so the IW buffer term cancels and the difference is
    # attributable to the redox ratio alone.
    assert hf_reduced['T_magma'] == pytest.approx(hf_oxidised['T_magma'], rel=1e-12)


@pytest.mark.parametrize('source', ['user_constant', 'from_O_budget'])
def test_tracker_is_a_no_op_and_writes_nothing_under_other_fo2_sources(source):
    """Error/guard contract: under any fO2_source other than from_mantle_redox
    the tracker returns without seeding state or writing helpfile columns.

    Verifies the guard has no side effect, not merely that it returns: a
    partially seeded state would make a later switch of source inconsistent.
    """
    interior = _make_interior()
    hf_row = {'T_magma': 2000.0}

    update_melt_redox(interior, hf_row, _make_config(0.25, source=source))

    assert interior.redox_state is None
    assert 'ferric_frac_mantle' not in hf_row
    assert 'fO2_shift_IW_mantle' not in hf_row


# ── Fe metal saturation (Steps 9a-9h) ──────────────────────────────────────


def _crystallise(f_0, n_steps=6):
    """Run the tracker through a solidifying mantle and return (state, rows).

    Melt fraction falls monotonically so Fe3+ is progressively partitioned
    into the solid, which is what drives the melt towards or away from the
    metal saturation boundary.
    """
    config = _make_config(f_0)
    interior = _make_interior()
    rows = []
    for step in range(n_steps):
        interior.phi = np.clip(_PHI - np.array([0.12, 0.07, 0.0]) * step, 0.0, None)
        hf_row = {'T_magma': 2200.0}
        update_melt_redox(interior, hf_row, config)
        rows.append(hf_row)
    return interior.redox_state, rows





@pytest.mark.physics_invariant
def test_reduced_melt_saturates_in_metal_and_becomes_more_oxidised():
    """A melt below the saturation threshold exsolves iron metal, and doing so
    raises its ferric fraction because the metal removes iron while leaving the
    oxygen behind."""
    f_0 = 0.005
    reduced, rows = _crystallise(f_0)

    assert rows[-1]['n_fe_metal_mantle'] > 0.0
    # The reaction brought the melt down onto the saturation boundary: it
    # started an order of magnitude supersaturated and must not be left so.
    assert rows[-1]['a_fe_max_mantle'] <= 1.0
    # Discriminating against partitioning alone. Fe3+ is incompatible in both
    # mineral assemblages, so the ferric fraction rises either way, but
    # partitioning alone lifts it by well under a factor of two over this many
    # steps, whereas the metal reaction drives it several times higher.
    assert reduced.ferric_frac > 2.0 * f_0


@pytest.mark.physics_invariant
def test_metal_diagnostics_reach_the_helpfile_csv(tmp_path, caplog):
    """The Fe-metal diagnostics the tracker writes into hf_row are
    helpfile schema columns, so they pass the row filter, are written to
    runtime_helpfile.csv and read back unchanged.

    Uses the reduced melt (f_0 = 0.005) so metal actually forms and the
    columns carry non-zero values; an all-zero column would also pass a
    presence check if the values were being dropped and re-seeded with 0.
    """
    from proteus.utils.coupler import (
        CreateHelpfileFromDict,
        ExtendHelpfile,
        ReadHelpfileFromCSV,
        WriteHelpfileToCSV,
        ZeroHelpfileRow,
    )

    metal_keys = (
        'a_fe_max_mantle',
        'a_fe_max_cell_mantle',
        'fe_metal_kg_mantle',
        'n_fe_metal_mantle',
        'n_fe_metal_step_mantle',
    )
    config = _make_config(0.005)
    interior = _make_interior()
    rows = []
    hf_all = None
    with caplog.at_level('WARNING'):
        for step in range(6):
            interior.phi = np.clip(_PHI - np.array([0.12, 0.07, 0.0]) * step, 0.0, None)
            hf_row = ZeroHelpfileRow()
            hf_row['Time'] = float(step)
            hf_row['T_magma'] = 2200.0
            update_melt_redox(interior, hf_row, config)
            rows.append(hf_row)
            hf_all = (CreateHelpfileFromDict(hf_row) if hf_all is None
                      else ExtendHelpfile(hf_all, hf_row))
    # The row filter no longer reports the metal keys as unknown.
    assert not any('not declared in GetHelpfileKeys' in r.message for r in caplog.records)

    WriteHelpfileToCSV(str(tmp_path), hf_all)
    csv = ReadHelpfileFromCSV(str(tmp_path))
    for key in metal_keys:
        assert key in csv.columns
        # Round trip: every step's value survives, not just the column.
        # rel=1e-9 because the CSV stores 11 significant digits (%.10e).
        np.testing.assert_allclose(
            csv[key].to_numpy(), [r[key] for r in rows], rtol=1e-9, atol=0.0
        )

    # Edge case: the first call runs no metal check, so all three are zero.
    assert csv['n_fe_metal_mantle'].iloc[0] == pytest.approx(0.0, abs=1e-30)
    assert csv['a_fe_max_mantle'].iloc[0] == pytest.approx(0.0, abs=1e-30)
    # No cell is tested on the first call, so the index is the -1 sentinel,
    # not 0 (which would name a real cell).
    assert csv['a_fe_max_cell_mantle'].iloc[0] == -1.0
    # Metal formed and was recorded, so a zero-filled column cannot pass.
    assert csv['n_fe_metal_mantle'].iloc[-1] > 0.0
    # Conservation: with no redissolution, the cumulative metal equals the
    # running sum of the per-step metal. A column holding a different
    # quantity (for example the metal activity) would break this closure.
    np.testing.assert_allclose(
        csv['n_fe_metal_mantle'].to_numpy(),
        np.cumsum(csv['n_fe_metal_step_mantle'].to_numpy()),
        rtol=1e-9, atol=0.0,
    )
    assert np.all(np.diff(csv['n_fe_metal_mantle'].to_numpy()) >= 0.0)

    # The mass column is the cumulative moles times the molar mass of Fe
    # metal, 0.055845 kg/mol.
    kg = csv['fe_metal_kg_mantle'].to_numpy()
    mol = csv['n_fe_metal_mantle'].to_numpy()
    np.testing.assert_allclose(kg, mol * 0.055845, rtol=1e-9, atol=0.0)
    # Discrimination guard: using the FeO molar mass (0.07184 kg/mol) instead
    # would overstate the mass by 29 %, far outside rtol=1e-9.
    assert abs(kg[-1] / (mol[-1] * 0.07184) - 1.0) > 0.2

    # Error contract: the schema is enforced, so a row that lacks one of the
    # metal columns is rejected rather than written with a gap.
    broken = dict(rows[-1])
    del broken['n_fe_metal_step_mantle']
    with pytest.raises(Exception, match='missing expected keys'):
        ExtendHelpfile(hf_all, broken)


@pytest.mark.physics_invariant
def test_surface_delta_iw_is_eq13_at_1_bar_and_the_outgassing_temperature(caplog):
    """The offset handed to the outgassing is Schaefer et al.'s surface fO2
    (fO2lowP_H22.m): Eq 13 at 1 bar, int(dV dP) = 0, with the melt ratio,
    minus IW, both at T_out = max(T_magma, outgas.T_floor) -- the
    temperature CALLIOPE and atmodeller solve at. 302 K is a surface
    temperature an interior step can return; at that temperature the
    relation gives an offset near -11 that the chemistry cannot solve.
    """
    from proteus.interior_chem.redox import _iw_buffer_bower2022, _log10_fO2_surface

    def diw_at(T, state):
        return _log10_fO2_surface(state.redox_ratio, T, state.X) - _iw_buffer_bower2022(T)

    config = _make_config(0.1344)
    results = {}
    for T in (302.342, 700.0, 2200.0, 3500.0):
        interior = _make_interior()
        hf_row = {'T_magma': T}
        caplog.clear()
        with caplog.at_level('WARNING'):
            update_melt_redox(interior, hf_row, config)
        warned = any('below outgas.T_floor' in r.message for r in caplog.records)
        results[T] = (hf_row['fO2_shift_IW_mantle'], interior.redox_state, warned)

    for T, (diw, state, warned) in results.items():
        T_out = max(T, 700.0)
        assert diw == pytest.approx(diw_at(T_out, state), rel=1e-12), T
        assert warned == (T < 700.0)
    # Discrimination guard: at the raw 302 K the offset is far lower, so a
    # value evaluated at T_magma itself could not pass the check above.
    cold, cold_state, _ = results[302.342]
    assert diw_at(302.342, cold_state) < cold - 10.0
    # The temperature genuinely enters: at a fixed ratio the offset differs
    # between 700, 2200 and 3500 K (it is not monotonic in T, so only
    # distinctness is asserted).
    vals = [results[T][0] for T in (700.0, 2200.0, 3500.0)]
    assert min(abs(x - y) for i, x in enumerate(vals) for y in vals[i + 1:]) > 0.5


@pytest.mark.physics_invariant
def test_freezing_the_shallow_melt_does_not_shift_the_surface_delta_iw():
    """Regression for the uppermost-melt-cell convention, which jumped by
    tens of log units once the shallow cells froze: the deepest remaining
    melt cell then carried a large int(dV dP) against a 1-bar buffer. The
    melt is homogeneous, so the surface offset may depend only on its
    Fe3+/Fe2+ and T_out, not on where melt remains. On the first call the
    ratio is exactly f_0 whatever phi is, so the two meshes below carry the
    same melt redox state.
    """
    config = _make_config(0.1)
    pres = np.array([0.3e9, 30.0e9, 80.0e9])
    temp = np.array([2000.0, 3200.0, 3900.0])
    mass = np.array([1.0e21, 3.0e21, 5.0e21])
    out = {}
    for name, phi in (('molten', np.array([1.0, 1.0, 1.0])),
                      ('shallow frozen', np.array([0.0, 0.0, 1.0]))):
        interior = _make_interior()
        interior.phi, interior.pres, interior.temp, interior.mass = phi, pres, temp, mass
        hf_row = {'T_magma': 2500.0}
        update_melt_redox(interior, hf_row, config)
        out[name] = (hf_row['fO2_shift_IW_mantle'], interior.redox_state)

    (d_m, s_m), (d_f, s_f) = out['molten'], out['shallow frozen']
    assert s_m.redox_ratio == pytest.approx(s_f.redox_ratio, rel=1e-15)
    assert d_f == pytest.approx(d_m, rel=1e-12)
    # Discrimination guard: the diagnostic profile does see the deep cell,
    # whose fO2 sits many log units above the 1-bar value at the same ratio,
    # so evaluating Delta-IW there would have failed the check above.
    assert s_f.fO2_cell == 2
    assert s_f.log10_fO2_cell[2] - s_m.log10_fO2_cell[0] > 5.0


@pytest.mark.physics_invariant
def test_eq13_pressure_term_matches_the_uncompressed_volume_analytic_limit():
    """Below the Deng splice pressure dV(FeO1.5 - FeO) is the constant
    uncompressed value, so int dV dP = dV0 (P - P0) exactly and Eq 13
    shifts log10 fO2 by dV0 (P - P0) / (a R T ln10) relative to 1 bar.
    The expectation is built by hand from that closed form, not from the
    interpolation table.
    """
    import math

    from proteus.interior_chem import eos_deng
    from proteus.interior_chem.redox import _A, _R, _log10_fO2_profile, _log10_fO2_surface

    T = 4000.0                       # p_splice(4000 K) ~ 1.29 GPa
    P = np.array([1.0e-4, 0.3, 0.9])  # all below the splice
    assert np.all(P < eos_deng.p_splice(T))
    ratio = 0.1 / 0.9
    state = _init_state(_PHI, _MASS, _PRES, 0.1)
    prof, valid = _log10_fO2_profile(ratio, np.full(3, T), P, state.X)
    assert np.all(valid)
    expected = _log10_fO2_surface(ratio, T, state.X) \
        + eos_deng._DV0 * (P - 1.0e-4) / (_A * _R * T * math.log(10))
    np.testing.assert_allclose(prof, expected, rtol=0, atol=1e-9)
    # P -> 1 bar recovers the surface relation (pressure term vanishes).
    assert prof[0] == pytest.approx(_log10_fO2_surface(ratio, T, state.X), abs=1e-12)
    # Order of magnitude guard: ~0.25 log units per 0.5 GPa here.
    assert 0.1 < prof[1] - prof[0] < 0.3


@pytest.mark.physics_invariant
def test_fo2_profile_is_resolved_per_melt_cell_and_nan_in_solid():
    """The Step 10 profile covers every melt cell at its own (T, P); solid
    cells carry NaN. At a uniform ratio and fixed T, fO2 rises with P
    because dV(FeO1.5 - FeO) > 0 over the mantle range."""
    from proteus.interior_chem.redox import _log10_fO2_profile

    config = _make_config(0.1)
    interior = _make_interior()
    update_melt_redox(interior, {'T_magma': 2200.0}, config)
    prof = interior.redox_state.log10_fO2_cell
    assert prof.shape == _PHI.shape
    assert np.all(np.isfinite(prof[_PHI > 0]))
    assert np.all(np.isnan(prof[_PHI == 0]))
    # Same (T, P) per cell as a direct call.
    direct, _ = _log10_fO2_profile(
        interior.redox_state.redox_ratio, _TEMP, _PRES / 1e9, interior.redox_state.X
    )
    np.testing.assert_allclose(prof[_PHI > 0], direct[_PHI > 0], rtol=1e-12)

    flat, _ = _log10_fO2_profile(0.1 / 0.9, np.full(4, 3000.0),
                                 np.array([0.1, 5.0, 25.0, 100.0]),
                                 interior.redox_state.X)
    assert np.all(np.diff(flat) > 0)


@pytest.mark.physics_invariant
def test_uppermost_melt_cell_index_is_found_whatever_the_grid_order():
    """fO2_top_index (diagnostic) is the lowest-pressure melt cell whether
    the backend orders cells surface-first (SPIDER) or CMB-first (Aragog),
    and a solid lid is skipped; the surface offset does not depend on the
    order either."""
    config = _make_config(0.1)
    phi = np.array([0.0, 1.0, 0.5])            # solid lid on top
    pres = np.array([0.5e9, 3.0e9, 30.0e9])
    temp = np.array([1600.0, 2500.0, 3300.0])
    mass = np.array([1.0e21, 3.0e21, 5.0e21])

    offsets = []
    for order in (slice(None), slice(None, None, -1)):
        interior = _make_interior()
        interior.phi, interior.pres = phi[order], pres[order]
        interior.temp, interior.mass = temp[order], mass[order]
        hf_row = {'T_magma': 2000.0}
        update_melt_redox(interior, hf_row, config)
        assert interior.pres[interior.redox_state.fO2_cell] == pytest.approx(3.0e9)
        offsets.append(hf_row['fO2_shift_IW_mantle'])
    assert offsets[0] == pytest.approx(offsets[1], rel=1e-12)


def test_oxidised_melt_stays_below_metal_saturation():
    """A melt at the published bulk-silicate ferric fraction sits far from the
    saturation boundary and forms no metal even with the check enabled.

    This is the boundary case on the other side of the threshold: the physics
    is on, and the answer is still zero.
    """
    state, rows = _crystallise(0.10)

    assert rows[-1]['n_fe_metal_mantle'] == pytest.approx(0.0, abs=1e-30)
    # The diagnostic activity is reported and is well under unity, so the
    # zero comes from an evaluated criterion, not a skipped branch.
    assert 0.0 < rows[-1]['a_fe_max_mantle'] < 1.0


@pytest.mark.physics_invariant
def test_metal_step_conserves_iron_and_oxygen():
    """Exsolving metal moves iron out of the melt without creating or
    destroying iron or oxygen.

    Conservation closure is the primary assertion, so it also serves as the
    exponent guard: any stoichiometric coefficient error breaks it.
    """
    state, _ = _crystallise(0.005, n_steps=3)
    fe_before = state.n_fe2_melt + state.n_fe3_melt + float(np.sum(state.n_fe_metal_cell))
    o_before = state.n_fe2_melt + 1.5 * state.n_fe3_melt

    xi = _metal_saturation_step(state, _TEMP, _PRES, _PHI, _MASS)

    fe_after = state.n_fe2_melt + state.n_fe3_melt + float(np.sum(state.n_fe_metal_cell))
    o_after = state.n_fe2_melt + 1.5 * state.n_fe3_melt

    assert fe_after == pytest.approx(fe_before, rel=1e-12)
    assert o_after == pytest.approx(o_before, rel=1e-12)
    assert np.isfinite(xi)


def test_metal_is_deposited_in_a_single_cell_not_smeared_across_the_mantle():
    """Metal accumulates where it formed rather than being distributed over
    every cell, because only the most supersaturated cell hosts equilibrium
    while the melt itself is homogeneous.

    The resulting radial field is what a later transport model consumes, so
    the locality matters as much as the total.
    """
    state, _ = _crystallise(0.005, n_steps=3)
    occupied = int(np.count_nonzero(state.n_fe_metal_cell))

    assert occupied == 1
    # The occupied cell is one that actually holds melt: a solid cell cannot
    # exsolve metal from a liquid.
    assert state.n_fe_metal_cell[np.argmax(state.n_fe_metal_cell)] > 0.0
    assert float(np.sum(state.n_fe_metal_cell)) > 0.0


@pytest.mark.reference_pinned
def test_the_pressure_term_moves_metal_formation_away_from_the_coolest_cell():
    """The reaction volume change relocates the most supersaturated cell off
    the shallow cool end of the mantle and down into it.

    Anchor: Schaefer et al. (2024) JGR Planets 129, e2023JE008262, Section
    3.1.1, where metal forms at the base of the whole-mantle magma ocean and
    not at all in the 500 km models. On standard-state energies alone the
    activity depends on temperature only and the trend runs the other way,
    which is why the pressure term is not optional. The comparison is driven
    through the chemistry directly, since the model always includes it.
    """
    n2, n3 = 0.99 * _N_FET_TEST, 0.01 * _N_FET_TEST
    without_P, _ = activity_Fe_metal(_TEMP, n2, n3, _N_SIL_TEST,
                                     P_gpa=_PRES / 1e9, use_P_term=False)
    with_P, _ = activity_Fe_metal(_TEMP, n2, n3, _N_SIL_TEST,
                                  P_gpa=_PRES / 1e9, use_P_term=True)

    assert int(np.argmax(without_P)) == 0
    assert int(np.argmax(with_P)) > 0
    # Both are live comparisons rather than an artefact of one being all-zero.
    assert float(np.max(without_P)) > 0.0
    assert float(np.max(with_P)) > 0.0


def test_metal_saturation_does_not_run_before_any_crystallisation():
    """The first coupling step seeds the reservoirs and forms no metal, since
    no solid has yet grown to drive the melt anywhere.

    This is the limit-input contract at the start of a run: Schaefer likewise
    begins checking only after the first solid layer appears.
    """
    config = _make_config(0.005)
    interior = _make_interior()
    hf_row = {'T_magma': 2200.0}
    update_melt_redox(interior, hf_row, config)

    assert hf_row['n_fe_metal_mantle'] == pytest.approx(0.0, abs=1e-30)
    # The reservoirs were still seeded, so the step was skipped rather than
    # the whole tracker having been short-circuited.
    assert interior.redox_state.n_fe3_melt > 0.0


@pytest.mark.physics_invariant
def test_a_fe_max_cell_is_the_cell_that_receives_the_metal():
    """The binding-cell index names the cell holding the largest a_Fe, and on
    every step that forms metal it is exactly the cell whose inventory grew.
    A column filled with argmax over the wrong array, or left stale from an
    earlier step, would fail one of the two checks."""
    config = _make_config(0.005)
    interior = _make_interior()
    formed_any = False
    for step in range(6):
        interior.phi = np.clip(_PHI - np.array([0.12, 0.07, 0.0]) * step, 0.0, None)
        before = (None if interior.redox_state is None
                  else interior.redox_state.n_fe_metal_cell.copy())
        hf_row = {'T_magma': 2200.0}
        update_melt_redox(interior, hf_row, config)
        idx = hf_row['a_fe_max_cell_mantle']
        state = interior.redox_state
        if before is None:
            assert idx == -1.0
            continue
        assert idx == float(np.argmax(state.a_fe_cell))
        assert state.a_fe_cell[int(idx)] == hf_row['a_fe_max_mantle']
        grown = np.flatnonzero(state.n_fe_metal_cell > before)
        if hf_row['n_fe_metal_step_mantle'] > 0.0:
            formed_any = True
            np.testing.assert_array_equal(grown, [int(idx)])
        else:
            assert grown.size == 0
    # Guard against a fixture that never saturates, which would make the
    # metal-location check above vacuous.
    assert formed_any


def _make_int_snapshot(path, n_stag):
    """Minimal Aragog-style _int.nc: a staggered dimension and one field."""
    import netCDF4 as nc

    with nc.Dataset(path, mode='w') as ds:
        ds.createDimension('staggered', n_stag)
        v = ds.createVariable('temp_s', np.float64, ('staggered',))
        v[:] = np.arange(n_stag, dtype=float)


def test_fo2_profile_round_trips_through_the_interior_snapshot(tmp_path):
    """The Step 10 profile appended to an _int.nc reads back cell for cell,
    NaN (solid) included, with the top-cell index; the backend's own fields
    are untouched, and a second append overwrites rather than failing."""
    import netCDF4 as nc

    from proteus.interior_chem.redox import write_fO2_profile_ncdf

    fpath = str(tmp_path / '1000_int.nc')
    _make_int_snapshot(fpath, _PHI.size)
    interior = _make_interior()
    update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
    state = interior.redox_state

    assert write_fO2_profile_ncdf(fpath, state)
    with nc.Dataset(fpath) as ds:
        got = np.asarray(ds['log10_fO2_s'][:], dtype=float)
        assert ds['log10_fO2_s'].units == 'log10(bar)'
        assert int(ds['fO2_top_index'][...]) == state.fO2_cell
        np.testing.assert_array_equal(ds['temp_s'][:], np.arange(_PHI.size))
    np.testing.assert_array_equal(np.isnan(got), _PHI == 0)
    np.testing.assert_allclose(got[_PHI > 0], state.log10_fO2_cell[_PHI > 0], rtol=0, atol=0)

    # Second step on the same file: values replaced, no duplicate-variable error.
    state.log10_fO2_cell = state.log10_fO2_cell + 1.0
    assert write_fO2_profile_ncdf(fpath, state)
    with nc.Dataset(fpath) as ds:
        again = np.asarray(ds['log10_fO2_s'][:], dtype=float)
    np.testing.assert_allclose(again[_PHI > 0], got[_PHI > 0] + 1.0, rtol=0, atol=1e-12)


def test_fo2_profile_is_not_written_to_a_missing_or_mismatched_snapshot(tmp_path):
    import netCDF4 as nc

    from proteus.interior_chem.redox import write_fO2_profile_ncdf

    interior = _make_interior()
    update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
    state = interior.redox_state

    assert not write_fO2_profile_ncdf(str(tmp_path / 'absent_int.nc'), state)
    assert not write_fO2_profile_ncdf(str(tmp_path / 'x_int.nc'), None)

    fpath = str(tmp_path / 'wrong_int.nc')
    _make_int_snapshot(fpath, _PHI.size + 2)
    assert not write_fO2_profile_ncdf(fpath, state)
    with nc.Dataset(fpath) as ds:
        assert 'log10_fO2_s' not in ds.variables


def test_standalone_redox_snapshot_is_self_contained(tmp_path):
    """SPIDER has no _int.nc, so its <time>_redox.nc must carry the profile
    together with the P, T, phi it was computed on and the step's scalars,
    so the file alone can reproduce the reported offset's input cell."""
    import netCDF4 as nc

    from proteus.interior_chem.redox import write_redox_ncdf

    interior = _make_interior()
    hf_row = {'T_magma': 2200.0}
    update_melt_redox(interior, hf_row, _make_config(0.1))
    state = interior.redox_state

    fpath = str(tmp_path / '884p700_redox.nc')
    assert write_redox_ncdf(fpath, state, 884.7, interior, hf_row)
    with nc.Dataset(fpath) as ds:
        assert len(ds.dimensions['staggered']) == _PHI.size
        np.testing.assert_allclose(ds['pres_s'][:], _PRES / 1e9, rtol=1e-15)
        assert ds['pres_s'].units == 'GPa'
        np.testing.assert_array_equal(ds['temp_s'][:], _TEMP)
        np.testing.assert_array_equal(ds['phi_s'][:], _PHI)
        got = np.asarray(ds['log10_fO2_s'][:], dtype=float)
        top = int(ds['fO2_top_index'][...])
        assert float(ds['time'][...]) == pytest.approx(884.7)
        assert float(ds['fO2_shift_IW_mantle'][...]) == hf_row['fO2_shift_IW_mantle']
        assert float(ds['ferric_frac_mantle'][...]) == hf_row['ferric_frac_mantle']
        P_top = float(ds['pres_s'][top])
    np.testing.assert_array_equal(np.isnan(got), _PHI == 0)
    np.testing.assert_allclose(got[_PHI > 0], state.log10_fO2_cell[_PHI > 0], rtol=0, atol=0)
    # The stored top index points at the lowest-pressure melt cell.
    assert P_top == pytest.approx(np.min(_PRES[_PHI > 0]) / 1e9)
    assert not write_redox_ncdf(str(tmp_path / 'none_redox.nc'), None, 0.0, interior, hf_row)


@pytest.mark.physics_invariant
def test_cells_below_the_solid_threshold_carry_no_melt_iron():
    """phi < PHI_SOLID counts as solid: such a cell is left out of the
    initial melt inventory exactly as a phi = 0 cell is."""
    from proteus.interior_chem.redox import PHI_SOLID

    phi_trace = np.array([1.0, 0.5, 0.5 * PHI_SOLID])
    phi_zero = np.array([1.0, 0.5, 0.0])
    out = []
    for phi in (phi_trace, phi_zero):
        interior = _make_interior()
        interior.phi = phi
        update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
        out.append(interior.redox_state)
    assert out[0].n_fe2_melt == pytest.approx(out[1].n_fe2_melt, rel=1e-15)
    assert out[0].n_fe2_melt == pytest.approx(W_FET * _M_MELT / MU_FEO, rel=1e-12)
    # The interior's own phi is not modified.
    assert interior.phi is phi_zero


@pytest.mark.physics_invariant
def test_a_trace_melt_cell_is_not_tested_for_metal_and_has_no_fo2():
    """A deep, hot cell with phi just below PHI_SOLID would otherwise be
    the binding cell (the pressure term favours depth); as solid it is
    skipped by the metal check and left NaN in the fO2 profile."""
    from proteus.interior_chem.redox import PHI_SOLID

    pres = np.array([10.0e9, 30.0e9, 60.0e9])
    temp = np.array([2200.0, 3000.0, 3800.0])
    mass = np.array([2.0e21, 4.0e21, 6.0e21])
    config = _make_config(0.01)

    def run(phi_deep):
        interior = _make_interior()
        interior.pres, interior.temp, interior.mass = pres, temp, mass
        interior.phi = np.array([1.0, 1.0, 1.0])
        update_melt_redox(interior, {'T_magma': 2200.0}, config)
        interior.phi = np.array([0.95, 0.95, phi_deep])
        update_melt_redox(interior, {'T_magma': 2200.0}, config)
        return interior.redox_state

    st_melt = run(2.0 * PHI_SOLID)
    st_trace = run(0.5 * PHI_SOLID)
    # Discrimination guard: as melt, the deep cell is the most saturated one.
    assert st_melt.a_fe_max_cell == 2
    assert st_trace.a_fe_cell[2] == 0.0
    assert st_trace.a_fe_max_cell != 2
    assert np.isnan(st_trace.log10_fO2_cell[2])
    assert np.isfinite(st_melt.log10_fO2_cell[2])


@pytest.mark.physics_invariant
def test_crossing_the_solid_threshold_crystallises_the_remaining_melt():
    """A cell dropping from phi = 0.5 to just below PHI_SOLID crystallises
    all of its remaining melt that step: the reservoirs end up the same as
    for a drop straight to phi = 0."""
    from proteus.interior_chem.redox import PHI_SOLID

    out = []
    for phi_last in (0.8 * PHI_SOLID, 0.0):
        interior = _make_interior()
        update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
        interior.phi = np.array([1.0, phi_last, 0.0])
        update_melt_redox(interior, {'T_magma': 2200.0}, _make_config(0.1))
        st = interior.redox_state
        out.append((st.n_fe2_melt + float(np.sum(st.n_fe_metal_cell)), st.n_fe3_melt,
                    st.phi_prev[1]))
    (fe2_a, fe3_a, prev_a), (fe2_b, fe3_b, prev_b) = out
    assert prev_a == 0.0 and prev_b == 0.0
    assert fe2_a == pytest.approx(fe2_b, rel=1e-12)
    assert fe3_a == pytest.approx(fe3_b, rel=1e-12)
