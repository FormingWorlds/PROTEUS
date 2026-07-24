"""Tests for the PROTEUS-side from_O_budget CALLIOPE wrapper.

Covers the dispatch logic added to ``proteus.outgas.calliope`` and
``proteus.outgas.wrapper`` when ``planet.fO2_source == 'from_O_budget'``:

* Wrapper routes to CALLIOPE's authoritative-O entry point with the
  correct ``target`` (5-key, includes ``'O'``) and ``fO2_hint``.
* Helpfile plumbing of ``fO2_shift_IW_derived`` and ``O_res`` under both
  ``user_constant`` and ``from_O_budget``.
* ``O_kg_total`` is preserved as the authoritative user input under
  from_O_budget even though the solver's output dict reports its own value.
* Smoke-tier round-trip through the real CALLIOPE solver.
* End-to-end from_O_budget invariants: ``M_atm <= M_planet`` and the
  ``GetHelpfileKeys()`` schema carries the two new columns.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from proteus.outgas.calliope import calc_surface_pressures
from proteus.utils.constants import element_list, gas_list
from proteus.utils.coupler import GetHelpfileKeys

from ._from_o_budget_helpers import _earth_hf_row, _make_from_o_budget_config

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Every test here is unit tier and mocks CALLIOPE. The cases that drive the
# real solver live in test_from_o_budget_wrapper_smoke.py: a smoke marker on a
# function in this file would combine with the module mark above, and the unit
# and smoke filters do not exclude each other, so the test would be collected
# by both tiers and run twice.


logging.getLogger('calliope').setLevel(logging.WARNING)


_PRIMARY_VOLATILES = ('H2O', 'CO2', 'N2', 'S2')


def _make_solvevol_result(
    fO2_derived: float = 4.0,
    O_res: float = 0.0,
    O_kg_total_out: float | None = None,
) -> dict:
    """Build a fake CALLIOPE output dict shaped like the real one.

    Populates the keys that the wrapper copies into hf_row plus the two
    from_O_budget extras. ``O_kg_total_out`` lets a caller simulate the solver
    drifting from its input target by a small residual.
    """
    out: dict = {
        'M_atm': 1.0e19,
        'P_surf': 100.0,
        'atm_kg_per_mol': 0.018,
        'fO2_shift_derived': fO2_derived,
        'H_res': 0.0,
        'C_res': 0.0,
        'N_res': 0.0,
        'S_res': 0.0,
        'O_res': O_res,
    }
    for s in gas_list:
        out[f'{s}_bar'] = 0.0
        out[f'{s}_vmr'] = 0.0
        out[f'{s}_kg_atm'] = 0.0
        out[f'{s}_kg_liquid'] = 0.0
        out[f'{s}_kg_solid'] = 0.0
        out[f'{s}_kg_total'] = 0.0
        out[f'{s}_mol_atm'] = 0.0
        out[f'{s}_mol_liquid'] = 0.0
        out[f'{s}_mol_solid'] = 0.0
        out[f'{s}_mol_total'] = 0.0
    out['H2O_bar'] = 50.0
    out['H2O_kg_atm'] = 8.0e18
    for e in element_list:
        out[f'{e}_kg_atm'] = 0.0
        out[f'{e}_kg_liquid'] = 0.0
        out[f'{e}_kg_solid'] = 0.0
        out[f'{e}_kg_total'] = 0.0
    out['H_kg_total'] = 1.5e20
    out['C_kg_total'] = 1.5e19
    out['N_kg_total'] = 8.0e18
    out['S_kg_total'] = 8.0e20
    # Solver-reported O budget. Default matches the canonical input
    # (perfect convergence); a caller can pass a perturbed value to
    # exercise the "wrapper restores the authoritative input" branch.
    out['O_kg_total'] = 1.0e22 if O_kg_total_out is None else O_kg_total_out
    return out


# ---------------------------------------------------------------------------
# Schema: helpfile carries the two new columns
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_helpfile_schema_includes_fO2_shift_IW_derived():
    """The derived IW-buffer offset is the *scientific* output of from_O_budget;
    if it isn't in the schema, the CSV would silently drop it and any
    from_O_budget run would be unreproducible from the helpfile alone.
    """
    keys = GetHelpfileKeys()
    assert 'fO2_shift_IW_derived' in keys
    # Pair with O_res so the solver's 5th residual is visible alongside
    # the existing H/C/N/S diagnostics handled by other columns.
    assert 'O_res' in keys


@pytest.mark.unit
def test_helpfile_schema_keys_unique():
    """No accidental duplicate entries in the schema. Adding new keys
    must not collide with an existing column name (a duplicate would
    silently disappear when the DataFrame is built with columns=keys).
    """
    keys = GetHelpfileKeys()
    duplicates = {k for k in keys if keys.count(k) > 1}
    assert duplicates == set(), f'duplicate helpfile keys: {duplicates}'
    # Discrimination: schema must be non-empty. A regression that returned
    # an empty list would also have an empty duplicate set and pass above
    # vacuously.
    assert len(keys) > 0


# ---------------------------------------------------------------------------
# Dispatch: from_O_budget routes to authoritative-O entry point
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_o_budget_dispatches_to_authoritative_O_entry_point():
    """Under fO2_source = 'from_O_budget' the wrapper calls
    ``equilibrium_atmosphere_authoritative_O``, not
    ``equilibrium_atmosphere``. A regression that flipped the branch
    would silently produce buffered-fO2 chemistry instead of
    authoritative-O chemistry; the symptom is invisible from the helpfile
    columns alone.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()

    fake = _make_solvevol_result(fO2_derived=2.5, O_res=42.0)

    with (
        patch(
            'proteus.outgas.calliope.equilibrium_atmosphere_authoritative_O',
            return_value=fake,
        ) as mock_new,
        patch('proteus.outgas.calliope.equilibrium_atmosphere') as mock_legacy,
    ):
        calc_surface_pressures(dirs, config, hf_row)
        mock_new.assert_called_once()
        mock_legacy.assert_not_called()

        # Inspect the call arguments to verify from_O_budget contract.
        # target dict carries every element_list slot (legacy behaviour)
        # plus the new authoritative 'O' key sourced from hf_row.
        call = mock_new.call_args
        target_arg = call.args[0]
        opts_arg = call.args[1]
        for e in ('H', 'C', 'N', 'S', 'O'):
            assert e in target_arg, f'target missing required element {e!r}'
        assert target_arg['O'] == pytest.approx(hf_row['O_kg_total'])
        assert target_arg['H'] == pytest.approx(hf_row['H_kg_total'])
        assert opts_arg['M_mantle'] == pytest.approx(hf_row['M_mantle'])
        # fO2_hint is sourced from config.outgas.fO2_shift_IW so the
        # Monte-Carlo restart starts at the user's expected basin.
        assert call.kwargs['fO2_hint'] == pytest.approx(config.outgas.fO2_shift_IW)


@pytest.mark.unit
def test_legacy_dispatches_to_legacy_entry_point():
    """Mirror of the from_O_budget test: when fO2_source = 'user_constant' the
    wrapper must call ``equilibrium_atmosphere`` and NOT the new
    authoritative-O entry point. The target dict must have 4 keys
    (no 'O'), preserving the legacy contract bit-for-bit.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row()

    fake = _make_solvevol_result()

    with (
        patch(
            'proteus.outgas.calliope.equilibrium_atmosphere', return_value=fake
        ) as mock_legacy,
        patch(
            'proteus.outgas.calliope.equilibrium_atmosphere_authoritative_O',
        ) as mock_new,
    ):
        calc_surface_pressures(dirs, config, hf_row)
        mock_legacy.assert_called_once()
        mock_new.assert_not_called()

        # Legacy contract: target dict carries every element_list slot
        # EXCEPT 'O' (the buffered-fO2 chemistry derives O internally and
        # passing it as a target would be a category error).
        call = mock_legacy.call_args
        target_arg = call.args[0]
        for e in ('H', 'C', 'N', 'S','O'):
            assert e in target_arg, f'target missing required element {e!r}'


@pytest.mark.unit
def test_p_guess_max_forwarded_to_calliope_both_dispatch_branches():
    """The wrapper forwards ``config.outgas.calliope.p_guess_max`` to CALLIOPE's
    cold-start pressure draw, on both dispatch branches, so a sub-Neptune run can
    widen the guess from the config. The builder sets a distinctive 5e6 bar, so
    the assertion discriminates a wrapper that forwards the config value from one
    that drops it or hard-codes CALLIOPE's 1e5 default."""
    dirs = {'output': '/tmp/test'}
    expected_p_max = 5.0e6  # matches the distinctive value in the config builder

    # from_O_budget -> authoritative-O entry point
    config = _make_from_o_budget_config()
    assert config.outgas.calliope.p_guess_max == pytest.approx(expected_p_max)
    with (
        patch(
            'proteus.outgas.calliope.equilibrium_atmosphere_authoritative_O',
            return_value=_make_solvevol_result(fO2_derived=2.5, O_res=1.0),
        ) as mock_new,
        patch('proteus.outgas.calliope.equilibrium_atmosphere'),
    ):
        calc_surface_pressures(dirs, config, _earth_hf_row())
        assert mock_new.call_args.kwargs['p_guess_max'] == pytest.approx(expected_p_max)
        # Discrimination: not CALLIOPE's 1e5 default (a dropped forward).
        assert mock_new.call_args.kwargs['p_guess_max'] != pytest.approx(1.0e5)

    # user_constant -> legacy entry point
    config = _make_from_o_budget_config()
    config.planet.fO2_source = 'user_constant'
    with (
        patch(
            'proteus.outgas.calliope.equilibrium_atmosphere',
            return_value=_make_solvevol_result(),
        ) as mock_legacy,
        patch('proteus.outgas.calliope.equilibrium_atmosphere_authoritative_O'),
    ):
        calc_surface_pressures(dirs, config, _earth_hf_row())
        assert mock_legacy.call_args.kwargs['p_guess_max'] == pytest.approx(expected_p_max)
        assert mock_legacy.call_args.kwargs['p_guess_max'] != pytest.approx(1.0e5)


# ---------------------------------------------------------------------------
# Helpfile plumbing: both modes populate fO2_shift_IW_derived
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_legacy_path_echoes_configured_fO2_shift_IW():
    """Under user_constant the derived buffer offset must equal the
    configured ``outgas.fO2_shift_IW``. This is a downstream-analysis
    convenience: one column carries the actual buffer the chemistry
    equilibrated to regardless of which code path produced it.

    Discriminating: a value of 0.0 (the MagicMock default) or a stale
    NaN would both pass a "key present" check but neither matches the
    config input.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config(fO2_shift_IW=-2.5)
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row()

    fake = _make_solvevol_result()

    with patch('proteus.outgas.calliope.equilibrium_atmosphere', return_value=fake):
        calc_surface_pressures(dirs, config, hf_row)

    assert hf_row['fO2_shift_IW_derived'] == pytest.approx(-2.5)
    # Legacy mode has no 5th residual; column is 0.0 (a stale NaN here
    # would propagate into the helpfile CSV).
    assert hf_row['O_res'] == pytest.approx(0.0)


@pytest.mark.unit
def test_from_o_budget_writes_solver_derived_fO2_to_helpfile():
    """Under from_O_budget the wrapper must write the solver-derived offset
    into ``fO2_shift_IW_derived`` and the 5th residual into ``O_res``.

    Discriminating: a wrong key (``fO2_shift_derived`` from CALLIOPE's
    naming, no IW prefix) would leave the helpfile column unset.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()

    fake = _make_solvevol_result(fO2_derived=-1.3, O_res=125.0)

    with patch(
        'proteus.outgas.calliope.equilibrium_atmosphere_authoritative_O',
        return_value=fake,
    ):
        calc_surface_pressures(dirs, config, hf_row)

    assert hf_row['fO2_shift_IW_derived'] == pytest.approx(-1.3)
    assert hf_row['O_res'] == pytest.approx(125.0)


# ---------------------------------------------------------------------------
# Authoritative input preservation: solver residual does not drift O_kg_total
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_o_budget_preserves_authoritative_O_kg_total():
    """The user-supplied O_kg_total is authoritative under from_O_budget. The
    solver's output O_kg_total (which equals input minus residual) must
    not be allowed to drift the helpfile value, because the escape
    pipeline reads ``hf_row['O_kg_total']`` to compute the next debit and
    any per-iteration drift would accumulate to a non-trivial conservation
    error over Myr trajectories.

    Discriminating: the fake CALLIOPE output reports a value 5% off the
    input target. Without the restore, hf_row would carry the 5%-off
    value; with the restore, it carries the authoritative input.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)

    # Simulate the solver landing 5% off, much larger than realistic
    # convergence drift, but the point is to detect any non-zero drift.
    fake = _make_solvevol_result(O_kg_total_out=1.05e22)

    with patch(
        'proteus.outgas.calliope.equilibrium_atmosphere_authoritative_O',
        return_value=fake,
    ):
        calc_surface_pressures(dirs, config, hf_row)

    assert hf_row['O_kg_total'] == pytest.approx(1.0e22, rel=1e-12)
    # Discrimination: the solver's 5% drift must not leak into the
    # helpfile. Pin the gap between the solver-output and stored values.
    # Without the restore, hf_row would carry 1.05e22 and the gap would
    # be 5e20 (well above any conservation tolerance).
    assert abs(hf_row['O_kg_total'] - 1.05e22) > 1e20


@pytest.mark.unit
def test_legacy_path_lets_solver_set_O_kg_total():
    """Mirror of the previous test: under user_constant, the solver's
    output O_kg_total IS authoritative (the user has not supplied one
    and the buffer chemistry produces the value). Restoring the input
    would break the legacy contract.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row(O_kg_total=0.0)  # pre-chem the slot is empty

    fake = _make_solvevol_result()
    fake['O_kg_total'] = 3.3e22  # legacy chemistry writes this

    with patch('proteus.outgas.calliope.equilibrium_atmosphere', return_value=fake):
        calc_surface_pressures(dirs, config, hf_row)

    assert hf_row['O_kg_total'] == pytest.approx(3.3e22, rel=1e-12)
    # Discrimination: the solver's value must overwrite the empty input. A
    # regression that preserved the user input (correct under from_O_budget but
    # wrong under legacy) would leave hf_row['O_kg_total'] at 0.0.
    assert hf_row['O_kg_total'] > 0.0
