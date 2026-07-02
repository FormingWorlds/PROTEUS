"""Tests for the PROTEUS-side from_O_budget atmodeller wrapper.

Covers the dispatch logic added to ``proteus.outgas.atmodeller`` when
``planet.fO2_source == 'from_O_budget'``:

* The wrapper builds a 5-key mass_constraints dict (H/C/N/S/O sourced
  from ``hf_row``) and an empty fugacity_constraints dict under from_O_budget.
* The wrapper preserves the legacy contract under user_constant (4-key
  mass_constraints, IronWustiteBuffer fugacity constraint).
* After the solve, the wrapper writes ``fO2_shift_IW_derived`` from
  atmodeller's back-computed ``log10dIW_1_bar`` output and reports the
  O mass residual.
* The authoritative ``hf_row['O_kg_total']`` is preserved across the
  call so per-iteration drift cannot accumulate into the escape pipeline.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip the whole module if atmodeller isn't installed. The macOS CI
# runner doesn't pull atmodeller (it isn't a hard dependency in
# pyproject.toml); Docker-based runners have it pre-installed in the
# image. Skipping at module scope is cleaner than guarding every test,
# and avoids leaking config-load failures from check_module_dependencies
# when a test builds a Config with outgas.module = "atmodeller".
pytest.importorskip(
    'atmodeller',
    reason='atmodeller package not installed; from_O_budget atmodeller wrapper tests skipped',
)

from proteus.utils.constants import element_list, gas_list, noble_gases  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


logging.getLogger('atmodeller').setLevel(logging.WARNING)


def _make_from_o_budget_config():
    """Build a MagicMock Config consistent with the atmodeller backend
    under from_O_budget dispatch."""
    config = MagicMock()
    config.outgas.module = 'atmodeller'
    config.outgas.fO2_shift_IW = 4.0
    config.outgas.T_floor = 1200.0
    config.outgas.mass_thresh = 1e8
    config.outgas.solver_atol = 1e-8
    config.outgas.solver_rtol = 1e-5
    config.outgas.atmodeller.solver_max_steps = 100
    config.outgas.atmodeller.solver_multistart = 1
    config.outgas.atmodeller.solver_mode = 'robust'
    config.outgas.atmodeller.include_condensates = False
    # Disable every solubility law to keep the species network minimal
    for name in ('H2O', 'CO2', 'H2', 'N2', 'S2', 'CO', 'CH4'):
        setattr(config.outgas.atmodeller, f'solubility_{name}', None)
    for name in ('H2O', 'CO2', 'H2', 'CH4', 'CO'):
        setattr(config.outgas.atmodeller, f'eos_{name}', None)
    config.planet.fO2_source = 'from_O_budget'
    config.interior_struct.core_frac = 0.3
    return config


def _earth_hf_row(O_kg_total: float = 1.0e22) -> dict:
    """Helpfile row populated up to the point ``calc_surface_pressures_atmodeller``
    is called."""
    hf: dict = {
        'M_mantle': 4.03e24,
        'M_int': 5.97e24,
        'M_planet': 5.97e24,
        'gravity': 9.81,
        'R_int': 6.371e6,
        'Phi_global': 1.0,
        'T_magma': 1800.0,
        'Time': 0.0,
    }
    for e in element_list:
        hf[f'{e}_kg_total'] = 0.0
    hf['H_kg_total'] = 1.5e20
    hf['C_kg_total'] = 1.5e19
    hf['N_kg_total'] = 8.0e18
    hf['S_kg_total'] = 8.0e20
    hf['O_kg_total'] = O_kg_total
    for s in gas_list:
        hf[f'{s}_kg_atm'] = 0.0
        hf[f'{s}_kg_liquid'] = 0.0
        hf[f'{s}_kg_solid'] = 0.0
        hf[f'{s}_vmr'] = 0.0
        hf[f'{s}_bar'] = 0.0
    return hf


def _fake_atmodeller_output(log10dIW: float = -0.1):
    """Build a MagicMock that mimics ``EquilibriumModel.solve(...)`` ->
    ``model.output`` chain. The output's ``asdict()`` populates the keys
    the wrapper reads (``O2_g.log10dIW_1_bar`` and per-species
    ``dissolved_mass``); ``quick_look()`` provides partial pressures.
    """
    out = MagicMock()
    out.quick_look.return_value = {
        'H2O_g': np.array(50.0),
        'CO2_g': np.array(5.0),
        'N2_g': np.array(1.0),
        'S2_g': np.array(0.5),
        'O2_g': np.array(1e-6),
    }
    out.total_pressure.return_value = np.array(56.5)
    out.asdict.return_value = {
        'O2_g': {'log10dIW_1_bar': np.array(log10dIW)},
        'H2O_g': {'dissolved_mass': np.array(2.0e19)},
        'CO2_g': {'dissolved_mass': np.array(1.0e18)},
        'state': {'temperature': np.array(1800.0), 'pressure': np.array(56.5)},
    }
    return out


# ---------------------------------------------------------------------------
# Dispatch: from_O_budget builds the 5-key mass_constraints
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_o_budget_builds_5_element_mass_constraints():
    """Under fO2_source = 'from_O_budget' atmodeller must receive O as
    an element-mass constraint. The legacy path constrained only
    H/C/N/S; from_O_budget adds the user O budget.

    Discriminating: the call_args's mass_constraints dict carries the
    'O' key with the value from hf_row['O_kg_total']. A regression that
    forgot to add O would leave only H/C/N/S, and atmodeller would
    silently fall back to the IW fugacity buffer if one were still set.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output(log10dIW=-0.5)

    with patch(
        'atmodeller.EquilibriumModel',
        return_value=fake_model,
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert set(mass_kwargs.keys()) == {'H', 'C', 'N', 'S', 'O'}
    assert mass_kwargs['O'] == pytest.approx(1.0e22)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_from_o_budget_drops_O2_fugacity_constraint():
    """The IronWustiteBuffer fugacity constraint must not be passed
    under from_O_budget. atmodeller would otherwise over-constrain the system
    (5 mass equations + 1 fugacity = 6 constraints on 5 elements).

    Discriminating: the call_args's fugacity_constraints dict is empty.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    fug_kwargs = fake_model.solve.call_args.kwargs['fugacity_constraints']
    assert fug_kwargs == {}
    # Constraint-count invariant: from_O_budget uses 5 element-mass constraints
    # (H/C/N/S/O) and zero fugacity constraints. A regression that
    # added O to mass_constraints but forgot to drop the IW buffer
    # would over-constrain the system (5 mass + 1 fugacity > 5 unknowns).
    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert len(mass_kwargs) + len(fug_kwargs) == 5
    # Discrimination: 'O' must be in mass_constraints (the from_O_budget
    # invariant), not silently dropped from both.
    assert 'O' in mass_kwargs


@pytest.mark.unit
def test_legacy_path_keeps_4_element_mass_constraints_and_fO2_buffer():
    """Mirror of the from_O_budget tests: under fO2_source = 'user_constant'
    the wrapper passes H/C/N/S (no O) and a single IW fugacity
    constraint. A regression that swapped which path got the buffer
    would silently invert the legacy chemistry.
    """
    from atmodeller.thermodata import IronWustiteBuffer

    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row()

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    fug_kwargs = fake_model.solve.call_args.kwargs['fugacity_constraints']
    assert set(mass_kwargs.keys()) == {'H', 'C', 'N', 'S'}
    assert 'O' not in mass_kwargs
    assert 'O2_g' in fug_kwargs
    assert isinstance(fug_kwargs['O2_g'], IronWustiteBuffer)


# ---------------------------------------------------------------------------
# Helpfile plumbing: fO2_shift_IW_derived comes from atmodeller's output
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_from_o_budget_writes_solver_derived_fO2_to_helpfile():
    """Under from_O_budget the wrapper must read atmodeller's back-computed
    log10dIW_1_bar from ``output.asdict()`` and write it to
    ``hf_row['fO2_shift_IW_derived']``. This is the from_O_budget scientific
    output.

    Discriminating: the fake output reports an offset of -2.7. A
    regression that fell back to the wrapper-level default
    (config.outgas.fO2_shift_IW = 4.0) would record 4.0, the wrong
    value.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output(log10dIW=-2.7)

    # Pre-set the wrapper default (mirrors what run_outgassing does
    # before dispatch). The wrapper must overwrite it.
    hf_row['fO2_shift_IW_derived'] = config.outgas.fO2_shift_IW

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    assert hf_row['fO2_shift_IW_derived'] == pytest.approx(-2.7)
    # Discrimination: a regression that left the pre-seeded default
    # (config.outgas.fO2_shift_IW = 4.0) in place would land at 4.0,
    # which differs from the solver's -2.7 by 6.7 dex (well above any
    # tolerance). Pin the magnitude of the disagreement.
    assert abs(hf_row['fO2_shift_IW_derived'] - config.outgas.fO2_shift_IW) > 1.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_legacy_path_preserves_pre_dispatch_fO2_echo():
    """Under user_constant the wrapper must NOT overwrite
    fO2_shift_IW_derived with atmodeller's back-computed value, because
    the fugacity constraint already set the offset to exactly
    config.outgas.fO2_shift_IW. Letting atmodeller's Hirschmann-buffer
    reconstruction overwrite the echo would introduce a small drift
    (buffer-formula and 1-bar-vs-P-evaluation differences) that breaks
    bit-for-bit echo of the user input.

    Discriminating: the fake output reports a perturbed log10dIW of
    -2.7, but config.outgas.fO2_shift_IW is 4.0. A regression that
    unconditionally writes the solver value would record -2.7 instead
    of 4.0.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row()
    hf_row['fO2_shift_IW_derived'] = config.outgas.fO2_shift_IW  # pre-seed

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output(log10dIW=-2.7)

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    assert hf_row['fO2_shift_IW_derived'] == pytest.approx(config.outgas.fO2_shift_IW)
    # Discrimination: the solver's -2.7 must NOT have overwritten the
    # echoed 4.0. A regression that unconditionally writes the solver
    # value would land 6.7 dex away from the user input.
    assert abs(hf_row['fO2_shift_IW_derived'] - (-2.7)) > 1.0


# ---------------------------------------------------------------------------
# Per-species kg_total invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_atmodeller_maintains_per_species_kg_total_invariant():
    """The CALLIOPE wrapper provides ``{species}_kg_total = kg_atm +
    kg_liquid + kg_solid`` natively via its solver output. The
    atmodeller wrapper used to leave the column at its previous
    iteration's value, which produced systematically wrong totals when a
    downstream subtraction-fallback read a stale kg_total. The wrapper
    must now write the invariant explicitly per species.

    Discriminating: the fake output sets H2O atmospheric mass via the
    pressure-mass conversion and dissolved mass via output.asdict.
    A regression that skipped the kg_total write would leave the column
    at 0.0 (the fixture seeds zero) while kg_atm + kg_liquid is
    non-zero.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    for sp in ('H2O', 'CO2', 'N2', 'S2'):
        kg_atm = hf_row[f'{sp}_kg_atm']
        kg_liq = hf_row[f'{sp}_kg_liquid']
        kg_sol = hf_row[f'{sp}_kg_solid']
        kg_total = hf_row[f'{sp}_kg_total']
        assert kg_total == pytest.approx(kg_atm + kg_liq + kg_sol, rel=1e-12)
    # Positivity / non-stale discrimination: at least one species (H2O
    # is the largest in the fake output) must have a strictly positive
    # post-call kg_total. A regression that skipped the write entirely
    # would leave the fixture's seeded 0.0; that would still satisfy
    # the conservation closure above (since kg_atm + kg_liq + kg_sol
    # would also be 0 if the per-species writes were skipped), making
    # the closure alone insufficient.
    assert hf_row['H2O_kg_total'] > 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_atmodeller_kg_total_not_stale_after_call():
    """A regression that read stale ``_kg_total`` from a prior
    iteration would silently propagate a wrong value forward. Pre-load
    the fixture with absurd ``_kg_total`` values and verify the wrapper
    overwrites them with the correct ``kg_atm + kg_liquid + kg_solid``.

    Discriminating: H2O pre-loaded at 1e30 kg; if the wrapper left that
    untouched, the kg_total invariant would fail by ~10 orders of
    magnitude.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()
    for sp in ('H2O', 'CO2', 'N2', 'S2'):
        hf_row[f'{sp}_kg_total'] = 1.0e30  # impossible value

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    for sp in ('H2O', 'CO2', 'N2', 'S2'):
        kg_atm = hf_row[f'{sp}_kg_atm']
        kg_liq = hf_row[f'{sp}_kg_liquid']
        kg_sol = hf_row[f'{sp}_kg_solid']
        # The total must be the post-solve sum, not the pre-loaded 1e30.
        assert hf_row[f'{sp}_kg_total'] == pytest.approx(kg_atm + kg_liq + kg_sol)
        assert hf_row[f'{sp}_kg_total'] < 1.0e25  # nowhere near the pre-load


# ---------------------------------------------------------------------------
# Authoritative-O preservation across the call
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_from_o_budget_preserves_authoritative_O_kg_total():
    """Under from_O_budget the user O budget is authoritative. atmodeller's
    converged element_density at iteration N is mass_atm + mass_dissolved
    which differs from the target by the solver's residual. Letting
    that drift contaminate ``hf_row['O_kg_total']`` would cascade into
    the escape pipeline (which reads this column to compute the next
    iteration's debit), so the wrapper restores the authoritative input
    after the solve.

    Discriminating: hf_row['O_kg_total'] equals the input target to
    floating-point precision, regardless of what atmodeller computed
    for atm + dissolved O.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    assert hf_row['O_kg_total'] == pytest.approx(1.0e22, rel=1e-12)
    # Discrimination: the wrapper must have passed the SAME O value
    # into atmodeller's mass_constraints. A regression where the
    # solver wrote a residual-perturbed value into O_kg_total would
    # also break the mass-constraint input pin.
    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert mass_kwargs['O'] == pytest.approx(1.0e22, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_legacy_path_does_not_overwrite_O_kg_total():
    """Mirror: under user_constant the atmodeller wrapper does not
    touch ``O_kg_total`` (atmodeller's output doesn't include O as a
    per-element constraint, and the wrapper doesn't reconstruct it).
    The value carried by hf_row before the call (from
    ``_resolve_oxygen_budget`` at IC, or from a previous iteration's
    escape debit) survives unchanged.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row(O_kg_total=5.5e21)

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    assert hf_row['O_kg_total'] == pytest.approx(5.5e21, rel=1e-12)
    # Discrimination: the legacy path must NOT have added 'O' to the
    # mass_constraints dict (atmodeller solves O via the IW fugacity
    # buffer here). A regression that mistakenly added it would
    # silently over-constrain the system.
    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert 'O' not in mass_kwargs


# ---------------------------------------------------------------------------
# Edge: O below mass threshold under from_O_budget must fail loudly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_o_budget_with_O_below_threshold_raises():
    """from_O_budget cannot work if the user O budget falls below the mass
    threshold (no constraint to invert against). The wrapper must fail
    with a clear ValueError rather than silently producing chemistry
    that ignores the O input.

    Discriminating: error names the field the user should change.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    # mass_thresh = 1e8 by config; set O budget well below it.
    hf_row = _earth_hf_row(O_kg_total=1.0)

    with pytest.raises(ValueError, match='from_O_budget'):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    # Boundary discrimination: an O budget comfortably above the
    # mass_thresh of 1e8 kg on the same from_O_budget config must NOT raise.
    # The rejection is driven by the threshold, not by from_O_budget itself.
    # A regression that hard-raised on every from_O_budget call would fail
    # this.
    hf_row_ok = _earth_hf_row(O_kg_total=1.0e22)
    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()
    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row_ok)
    assert hf_row_ok['O_kg_total'] == pytest.approx(1.0e22, rel=1e-12)


# ---------------------------------------------------------------------------
# O residual is computed from the species O distribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_o_budget_writes_finite_O_residual():
    """Under from_O_budget the wrapper must compute and write ``O_res`` from
    target - (atmospheric_O + dissolved_O). A finite, bounded residual
    is the strongest assertion available without the real solver; the
    fake output's hand-built numbers give us a deterministic check.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    target_O = 1.0e22
    hf_row = _earth_hf_row(O_kg_total=target_O)

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    assert 'O_res' in hf_row
    assert np.isfinite(hf_row['O_res'])
    # The fake output produces a finite atmospheric + dissolved O, so
    # the residual must be strictly less than the full target.
    assert abs(hf_row['O_res']) < target_O


# ---------------------------------------------------------------------------
# Branch coverage: unknown solubility/EOS keys, T_floor early return,
# solver exception, zero P_total VMR fallback. These exercise paths the
# main from_O_budget tests above do not reach because their config uses
# valid keys, a high enough T_magma, and a successful fake solve.
# ---------------------------------------------------------------------------


def test_atmodeller_warns_when_solubility_key_unknown(caplog):
    """A configured solubility model name that does NOT appear in
    atmodeller's solubility registry must log a warning and continue
    with no solubility (the species is still added).

    Discriminating: pin the gas name in the warning message so a
    regression that silently dropped the warning, or that crashed on
    the unknown key instead of warning, would fail this test.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.outgas.atmodeller.solubility_H2O = 'totally_not_a_real_law'
    hf_row = _earth_hf_row()
    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()
    with (
        caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.atmodeller'),
        patch('atmodeller.EquilibriumModel', return_value=fake_model),
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    # Warning text must name the missing model.
    assert any('totally_not_a_real_law' in rec.message for rec in caplog.records), (
        f'expected solubility-name warning, got: {[r.message for r in caplog.records]}'
    )
    # Discrimination guard: the dispatch still completed (the solve
    # ran), so hf_row carries a P_surf entry.
    assert 'P_surf' in hf_row


def test_atmodeller_warns_when_eos_key_unknown(caplog):
    """A configured real-gas EOS name not in atmodeller's eos registry
    must log a warning ('using ideal gas') and continue.

    Edge: same shape as the solubility case but the warning text
    differs; pin both the gas name and the 'ideal gas' fallback
    string so future renames of the message do not silently lose
    discrimination.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.outgas.atmodeller.eos_H2O = 'phantom_eos_v999'
    hf_row = _earth_hf_row()
    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()
    with (
        caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.atmodeller'),
        patch('atmodeller.EquilibriumModel', return_value=fake_model),
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    eos_warnings = [r.message for r in caplog.records if 'phantom_eos_v999' in r.message]
    assert eos_warnings, 'expected EOS-name warning for phantom_eos_v999'
    assert any('ideal gas' in m for m in eos_warnings), (
        'EOS warning should mention the ideal-gas fallback'
    )


def test_atmodeller_clamps_t_magma_to_floor(caplog):
    """When ``T_magma < config.outgas.T_floor`` the solve runs at the floor.

    The outgassing temperature is clamped from below (the same
    semantics as the calliope entry point), not skipped: the Planet
    state must carry T_floor, the solver must be invoked, and the clip
    warning must fire. Discriminating: a regression back to the skip
    behavior would never call solve; a regression that dropped the
    clamp would build the Planet at the raw 1800 K.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.outgas.T_floor = 2000.0  # above the hf_row T_magma of 1800 K
    hf_row = _earth_hf_row()

    planet_kwargs = {}

    def fake_planet(**kwargs):
        planet_kwargs.update(kwargs)
        return MagicMock()

    class SolveAttempted(RuntimeError):
        pass

    fake_model = MagicMock()
    fake_model.solve.side_effect = SolveAttempted('solver invoked')
    with (
        caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.atmodeller'),
        patch('atmodeller.EquilibriumModel', return_value=fake_model),
        patch('atmodeller.Planet', side_effect=fake_planet),
        pytest.raises(SolveAttempted),
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    # The solve was attempted (not skipped) and the planet state was
    # built at the clamped floor temperature, not the raw T_magma.
    assert fake_model.solve.call_count == 1
    assert planet_kwargs['temperature'] == pytest.approx(2000.0, rel=1e-12)
    assert planet_kwargs['temperature'] != pytest.approx(1800.0, rel=1e-3)
    # The clip warning fired.
    assert any('clipped' in rec.message for rec in caplog.records)


def test_atmodeller_propagates_solver_exception(caplog):
    """If atmodeller's ``model.solve(...)`` raises, the wrapper must
    log the failure and re-raise (lines 277-279). The caller is
    responsible for the abort + status-file update.

    Discriminating: pin both that the original exception bubbles up
    (a regression that swallowed it would mask the failure mode) and
    that the log message names the failure.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()
    fake_model = MagicMock()
    fake_model.solve.side_effect = RuntimeError('contrived solver failure')
    with (
        caplog.at_level(logging.ERROR, logger='fwl.proteus.outgas.atmodeller'),
        patch('atmodeller.EquilibriumModel', return_value=fake_model),
        pytest.raises(RuntimeError, match='contrived solver failure'),
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    assert any('Atmodeller solve failed' in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Noble gas wiring (mocked solver)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_noble_gas_passed_as_mass_constraint_and_uses_total_mass_output():
    """A helium inventory reaches atmodeller as its own element mass
    constraint, and the wrapper reads the per-species reservoir masses from
    atmodeller's conserved total_mass output rather than the pressure formula.

    Discriminating: the mass constraint carries the exact He budget; the
    tracked He total equals atmodeller's total_mass and the atmospheric mass
    is total minus dissolved, so the pressure-derived value (which would
    misweight the light noble gas) is not used.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)
    hf_row['He_kg_total'] = 3.0e16  # a helium inventory switches He on
    for gas in noble_gases:
        hf_row[f'{gas}_bar'] = 0.0

    out = _fake_atmodeller_output(log10dIW=-0.5)
    # He partial pressure plus atmodeller's conserved masses for helium.
    ql = dict(out.quick_look.return_value)
    ql['He_g'] = np.array(0.02)
    out.quick_look.return_value = ql
    ad = dict(out.asdict.return_value)
    ad['He_g'] = {'total_mass': np.array(3.0e16), 'dissolved_mass': np.array(1.0e16)}
    out.asdict.return_value = ad

    fake_model = MagicMock()
    fake_model.output = out

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)

    # He is a mass constraint with the supplied budget.
    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert 'He' in mass_kwargs
    assert mass_kwargs['He'] == pytest.approx(3.0e16)

    # The tracked reservoirs come from atmodeller's total_mass (conserved) and
    # dissolved_mass, not the mole-fraction pressure formula.
    assert hf_row['He_kg_total'] == pytest.approx(3.0e16, rel=1e-9)
    assert hf_row['He_kg_liquid'] == pytest.approx(1.0e16, rel=1e-9)
    assert hf_row['He_kg_atm'] == pytest.approx(2.0e16, rel=1e-9)
    assert hf_row['He_kg_atm'] + hf_row['He_kg_liquid'] == pytest.approx(3.0e16, rel=1e-9)
    # Discrimination: the pressure formula p*1e5*area/g at p=0.02 bar would
    # give ~2.6e16 kg for the atmospheric part alone and a total above the
    # 3e16 budget, so the total-mass path is what keeps He conserved.
    assert hf_row['He_kg_total'] < 3.1e16


@pytest.mark.unit
def test_no_noble_budget_leaves_atmodeller_species_unchanged():
    """With no noble inventory, no noble gas is added to the species network
    or the mass constraints, so a run without a noble budget is unaffected by
    the noble gas wiring.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)  # no noble budget

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output(log10dIW=-0.5)

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)

    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    for gas in noble_gases:
        assert gas not in mass_kwargs
        assert hf_row.get(f'{gas}_kg_total', 0.0) == 0.0
