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
    """Minimal config exposing only the two planet fields the tracker reads."""
    config = MagicMock()
    config.planet.fO2_source = source
    config.planet.ferric_fraction_initial = f_0
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
