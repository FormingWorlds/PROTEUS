"""Tests for the PROTEUS-side Path C CALLIOPE wrapper.

Covers the dispatch logic added to ``proteus.outgas.calliope`` and
``proteus.outgas.wrapper`` when ``planet.fO2_source == 'from_O_budget'``:

* Wrapper routes to CALLIOPE's authoritative-O entry point with the
  correct ``target`` (5-key, includes ``'O'``) and ``fO2_hint``.
* Helpfile plumbing of ``fO2_shift_IW_derived`` and ``O_res`` under both
  ``user_constant`` and ``from_O_budget``.
* ``O_kg_total`` is preserved as the authoritative user input under
  Path C even though the solver's output dict reports its own value.
* Smoke-tier round-trip through the real CALLIOPE solver.
* End-to-end Path C invariants: ``M_atm <= M_planet`` and the
  ``GetHelpfileKeys()`` schema carries the two new columns.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.outgas.calliope import calc_surface_pressures
from proteus.utils.constants import element_list, gas_list, vol_list
from proteus.utils.coupler import GetHelpfileKeys

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Mixed-tier file: 8 unit tests + 3 smoke tests. The smoke tests carry
# @pytest.mark.smoke per-function and run alongside other smoke tests in
# the smoke surface of PR CI; the unit filter excludes them. New tests
# in this file default to unit unless they exercise a real binary call.


logging.getLogger('calliope').setLevel(logging.WARNING)


_PRIMARY_VOLATILES = ('H2O', 'CO2', 'N2', 'S2')


def _make_solvevol_result(
    fO2_derived: float = 4.0,
    O_res: float = 0.0,
    O_kg_total_out: float | None = None,
) -> dict:
    """Build a fake CALLIOPE output dict shaped like the real one.

    Populates the keys that the wrapper copies into hf_row plus the two
    Path C extras. ``O_kg_total_out`` lets a caller simulate the solver
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


def _make_path_c_config(fO2_shift_IW: float = 4.0):
    """Build a MagicMock Config consistent with Path C dispatch.

    Only the fields the wrapper reaches into are populated; everything
    else inherits the MagicMock default (auto-attribute) and is irrelevant
    to the wrapper's control flow because solvevol_inp/target construction
    is the only consumer.
    """
    config = MagicMock()
    config.outgas.module = 'calliope'
    config.outgas.fO2_shift_IW = fO2_shift_IW
    config.outgas.T_floor = 1200.0
    config.outgas.mass_thresh = 1e8
    config.outgas.solver_atol = 1e-8
    config.outgas.solver_rtol = 1e-5
    config.outgas.calliope.nguess = 100
    config.outgas.calliope.nsolve = 500
    config.outgas.calliope.solubility = True
    for s in vol_list:
        setattr(config.outgas.calliope, f'include_{s}', True)
    config.outgas.calliope.is_included = lambda s: True
    config.planet.fO2_source = 'from_O_budget'
    config.planet.volatile_mode = 'elements'
    config.planet.volatile_reservoir = 'mantle'
    config.planet.elements.use_metallicity = False
    config.planet.elements.H_mode = 'kg'
    config.planet.elements.H_budget = 1.5e20
    config.planet.elements.C_mode = 'kg'
    config.planet.elements.C_budget = 1.5e19
    config.planet.elements.N_mode = 'kg'
    config.planet.elements.N_budget = 8.0e18
    config.planet.elements.S_mode = 'kg'
    config.planet.elements.S_budget = 8.0e20
    config.planet.elements.O_mode = 'kg'
    config.planet.elements.O_budget = 1.0e22
    return config


def _earth_hf_row(O_kg_total: float = 1.0e22) -> dict:
    """Helpfile row populated up to the point ``calc_surface_pressures``
    is called: structural quantities and per-element targets are set, but
    the gas-phase quantities are not."""
    hf: dict = {
        'M_mantle': 4.03e24,
        'M_int': 5.97e24,
        'gravity': 9.81,
        'R_int': 6.371e6,
        'Phi_global': 1.0,
        'T_magma': 1800.0,
        'Time': 0.0,
    }
    # element_list = [H, O, C, N, S, Si, Mg, Fe, Na]; CALLIOPE only
    # consumes H/C/N/S(/O). Pre-populate every element so the target
    # construction (loop over element_list) can read the slot uniformly.
    for e in element_list:
        hf[f'{e}_kg_total'] = 0.0
    hf['H_kg_total'] = 1.5e20
    hf['C_kg_total'] = 1.5e19
    hf['N_kg_total'] = 8.0e18
    hf['S_kg_total'] = 8.0e20
    hf['O_kg_total'] = O_kg_total
    return hf


# ---------------------------------------------------------------------------
# Schema: helpfile carries the two new columns
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_helpfile_schema_includes_fO2_shift_IW_derived():
    """The derived IW-buffer offset is the *scientific* output of Path C;
    if it isn't in the schema, the CSV would silently drop it and any
    Path C run would be unreproducible from the helpfile alone.
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
# Dispatch: Path C routes to authoritative-O entry point
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_path_c_dispatches_to_authoritative_O_entry_point():
    """Under fO2_source = 'from_O_budget' the wrapper calls
    ``equilibrium_atmosphere_authoritative_O``, not
    ``equilibrium_atmosphere``. A regression that flipped the branch
    would silently produce buffered-fO2 chemistry instead of
    authoritative-O chemistry; the symptom is invisible from the helpfile
    columns alone.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_path_c_config()
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

        # Inspect the call arguments to verify Path C contract.
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
    """Mirror of the Path C test: when fO2_source = 'user_constant' the
    wrapper must call ``equilibrium_atmosphere`` and NOT the new
    authoritative-O entry point. The target dict must have 4 keys
    (no 'O'), preserving the legacy contract bit-for-bit.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_path_c_config()
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
        for e in ('H', 'C', 'N', 'S'):
            assert e in target_arg, f'target missing required element {e!r}'
        assert 'O' not in target_arg, "legacy path must not pass 'O' as a target"


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
    config = _make_path_c_config(fO2_shift_IW=-2.5)
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
def test_path_c_writes_solver_derived_fO2_to_helpfile():
    """Under Path C the wrapper must write the solver-derived offset
    into ``fO2_shift_IW_derived`` and the 5th residual into ``O_res``.

    Discriminating: a wrong key (``fO2_shift_derived`` from CALLIOPE's
    naming, no IW prefix) would leave the helpfile column unset.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_path_c_config()
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
def test_path_c_preserves_authoritative_O_kg_total():
    """The user-supplied O_kg_total is authoritative under Path C. The
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
    config = _make_path_c_config()
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
    config = _make_path_c_config()
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row(O_kg_total=0.0)  # pre-chem the slot is empty

    fake = _make_solvevol_result()
    fake['O_kg_total'] = 3.3e22  # legacy chemistry writes this

    with patch('proteus.outgas.calliope.equilibrium_atmosphere', return_value=fake):
        calc_surface_pressures(dirs, config, hf_row)

    assert hf_row['O_kg_total'] == pytest.approx(3.3e22, rel=1e-12)
    # Discrimination: the solver's value must overwrite the empty input. A
    # regression that preserved the user input (correct under Path C but
    # wrong under legacy) would leave hf_row['O_kg_total'] at 0.0.
    assert hf_row['O_kg_total'] > 0.0


# ---------------------------------------------------------------------------
# Smoke: round-trip through the real CALLIOPE solver
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    'dIW',
    [
        pytest.param(
            -2.0,
            marks=pytest.mark.xfail(
                reason='CALLIOPE solver convergence is non-deterministic at extreme reducing conditions',
                strict=False,
                raises=RuntimeError,
            ),
        ),
        0.0,
        2.0,
        4.0,
    ],
)
def test_smoke_path_c_round_trip_through_wrapper(dIW):
    """Round-trip: run the wrapper in legacy mode at ``fO2_shift_IW = dIW``
    to derive the implied O budget, then re-run in Path C mode with that
    budget and verify ``fO2_shift_IW_derived ≈ dIW``.

    Mirrors the CALLIOPE Stage 2 ``TestRoundTrip`` pattern but exercises
    the *PROTEUS wrapper* (calc_surface_pressures + construct_options +
    target assembly), so a regression in the wrapper-level glue (wrong
    target dict, wrong fO2_hint plumbing, key-name mismatch in the
    output filter) is caught here, not just in CALLIOPE.
    """
    dirs = {'output': '/tmp/test'}

    # Legacy leg
    config_legacy = _make_path_c_config(fO2_shift_IW=dIW)
    config_legacy.planet.fO2_source = 'user_constant'
    hf_row_legacy = _earth_hf_row()
    calc_surface_pressures(dirs, config_legacy, hf_row_legacy)

    target_O = hf_row_legacy['O_kg_total']
    assert target_O > 0, 'legacy run must produce a positive O budget'
    assert hf_row_legacy['fO2_shift_IW_derived'] == pytest.approx(dIW)

    # Path C leg, feeding the implied O budget back in
    config_path_c = _make_path_c_config(fO2_shift_IW=dIW)
    hf_row_path_c = _earth_hf_row(O_kg_total=target_O)
    calc_surface_pressures(dirs, config_path_c, hf_row_path_c)

    derived = hf_row_path_c['fO2_shift_IW_derived']
    delta = abs(derived - dIW)

    # 0.05 dex is the same tolerance used by CALLIOPE Stage 2
    # TestRoundTrip; a wrapper-level glue bug (wrong fO2_hint, wrong
    # target['O'] source) would push delta well past 0.1 dex.
    assert delta < 0.05, f'wrapper round-trip failed at dIW={dIW}: derived={derived:.4f}'

    # Path C must preserve the authoritative O budget across the call.
    assert hf_row_path_c['O_kg_total'] == pytest.approx(target_O, rel=1e-12)


@pytest.mark.smoke
def test_smoke_path_c_residual_bounded_by_tolerance():
    """Under Path C the wrapper writes the 5th residual into ``O_res``.
    With a converged solve the absolute residual must be well below the
    target; otherwise the chemistry would silently mis-conserve O across
    iterations. Discriminating: a value above ``target_O * 1e-3`` (0.1%)
    would indicate the per-element tolerance gate is broken.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_path_c_config(fO2_shift_IW=2.0)

    # Bootstrap the O budget from a legacy run so it's by-construction reachable.
    cfg_seed = _make_path_c_config(fO2_shift_IW=2.0)
    cfg_seed.planet.fO2_source = 'user_constant'
    seed_hf = _earth_hf_row()
    calc_surface_pressures(dirs, cfg_seed, seed_hf)
    target_O = seed_hf['O_kg_total']

    hf_row = _earth_hf_row(O_kg_total=target_O)
    calc_surface_pressures(dirs, config, hf_row)

    assert np.isfinite(hf_row['O_res'])
    assert abs(hf_row['O_res']) <= max(target_O * 1e-3, 1e9)


@pytest.mark.smoke
def test_smoke_path_c_mass_conservation_invariant():
    """End-to-end check that issue #677's mass-conservation invariant
    holds under Path C: M_atm <= M_planet (which here is approximated
    by M_int + sum of element budgets, since the wrapper does not run
    update_planet_mass).

    The atmosphere mass aggregated from CALLIOPE's per-species
    ``s_kg_atm`` outputs must not exceed the planet's total tracked
    mass. A bug that re-introduced the O-skipping asymmetry would let
    M_atm exceed this bound at high H budgets.
    """
    dirs = {'output': '/tmp/test'}
    config = _make_path_c_config(fO2_shift_IW=4.0)

    cfg_seed = _make_path_c_config(fO2_shift_IW=4.0)
    cfg_seed.planet.fO2_source = 'user_constant'
    seed_hf = _earth_hf_row()
    calc_surface_pressures(dirs, cfg_seed, seed_hf)
    target_O = seed_hf['O_kg_total']

    hf_row = _earth_hf_row(O_kg_total=target_O)
    calc_surface_pressures(dirs, config, hf_row)

    M_atm = sum(float(hf_row.get(s + '_kg_atm', 0.0)) for s in gas_list)
    M_planet_lb = hf_row['M_int'] + (
        hf_row['H_kg_total']
        + hf_row['C_kg_total']
        + hf_row['N_kg_total']
        + hf_row['S_kg_total']
        + hf_row['O_kg_total']
    )

    assert M_atm > 0, 'sanity: outgassing produced a non-empty atmosphere'
    assert M_atm <= M_planet_lb, (
        f'Path C atmosphere ({M_atm:.3e} kg) exceeds tracked planet '
        f'mass lower bound ({M_planet_lb:.3e} kg), issue #677 regression?'
    )
