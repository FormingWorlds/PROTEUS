"""Coupling of the Aragog sub-grid lid: surface temperature, surface flux, conductance."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

from proteus.atmos_clim.common import lid_boundary, lid_flux_cap, lid_open_flux
from proteus.interior_energetics.aragog import (
    AragogRunner,
    _call_start,
    _lid_surface_of_row,
    _set_lid_surface,
    _surface_flux_of_step,
    lid_cap_secant,
    lid_resolve_skip,
)
from tests.atmos_clim.test_common import _row

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]
YEAR = 3.15576e7


def _out(weight, dE=-1.0e24, r=6.4e6, flux_top=7.0):
    return SimpleNamespace(
        lid_weight=weight,
        step_dE_F_int_J=dE,
        r_basic=np.array([3.0e6, r]),
        heat_flux=np.array([1.0, flux_top]),
    )


def test_flux_mode_without_a_lid_books_the_flux_of_the_atmosphere():
    """In flux mode a step with no lid at its start or end books the F_atm of the row."""
    hf = {'F_atm': 42.0, 'Time': 100.0}
    out = _out(float('nan'))
    assert _surface_flux_of_step(out, hf, 'flux', 110.0) == pytest.approx(42.0, rel=1e-15)
    assert _surface_flux_of_step(_out(0.0), hf, 'flux', 110.0) == pytest.approx(42.0, rel=1e-15)


def test_an_output_without_lid_fields_reads_as_no_lid():
    """An Aragog without the sub-grid lid has no lid fields: its step books ``F_atm``."""
    out = SimpleNamespace(step_dE_F_int_J=-1.0e24, r_basic=np.array([3.0e6, 6.4e6]))
    assert _surface_flux_of_step(
        out, {'F_atm': 42.0, 'Time': 100.0}, 'flux', 110.0
    ) == pytest.approx(42.0, rel=1e-15)
    assert (
        _surface_flux_of_step(_out(0.7), {'F_atm': 42.0, 'Time': 100.0}, 'flux', 110.0) != 42.0
    )


def test_grey_body_mode_books_the_surface_node():
    """In grey_body mode the step books the surface heat flux of the interior."""
    assert _surface_flux_of_step(
        _out(1.0), {'F_atm': 42.0, 'Time': 0.0}, 'grey_body', 5.0
    ) == pytest.approx(7.0, rel=1e-15)
    # without a lid the grey-body mode books the same surface node
    assert _surface_flux_of_step(
        _out(0.0), {'F_atm': 42.0, 'Time': 0.0}, 'grey_body', 5.0
    ) == pytest.approx(7.0, rel=1e-15)


@pytest.mark.parametrize('dt', [1.0, 10.0, 1000.0])
@pytest.mark.physics_invariant
def test_a_live_lid_books_the_integrated_flux_of_the_step(dt):
    """``F_int A dt = -step_dE_F_int_J``: the energy of the step and the booked flux agree."""
    out = _out(0.7)
    F = _surface_flux_of_step(out, {'F_atm': 42.0, 'Time': 100.0}, 'flux', 100.0 + dt)
    area = 4.0 * np.pi * 6.4e6**2
    assert F * area * dt * YEAR == pytest.approx(-out.step_dE_F_int_J, rel=1e-12)
    assert F != pytest.approx(42.0)  # the lid books its own integral, not the F_atm of the row


def test_a_state_with_no_step_books_the_flux_of_the_atmosphere():
    """A state with no step (no end time, or one at the row time) books the F_atm of the row."""
    assert _surface_flux_of_step(
        _out(1.0), {'F_atm': 42.0, 'Time': 100.0}, 'flux', 100.0
    ) == pytest.approx(42.0, rel=1e-15)
    assert _surface_flux_of_step(
        _out(1.0), {'F_atm': 42.0, 'Time': 100.0}, 'flux', None
    ) == pytest.approx(42.0, rel=1e-15)


@pytest.mark.parametrize(
    'row, mode, on, want',
    [
        ({'T_surf': 300.0}, 'flux', True, 300.0),
        ({'T_surf': 300.0}, 'flux', False, None),
        ({'T_surf': 300.0}, 'grey_body', True, None),
        ({'T_surf': 0.0}, 'flux', True, None),
        ({}, 'flux', True, None),
    ],
)
def test_the_lid_takes_the_surface_of_the_atmosphere_only_in_flux_mode(row, mode, on, want):
    """The lid takes the row surface only in flux mode with the lid on and a surface above 0."""
    assert _lid_surface_of_row(row, mode, on) == want
    assert _lid_surface_of_row(row, mode, False) is None  # no lid, no held surface


@pytest.mark.parametrize('w', [0.01, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.physics_invariant
def test_coupled_budget_of_a_step_closes_at_every_weight(w):
    """The balance of the atmosphere, the boundary value and the booked flux of one step.

    The interior loses ``w G_lid (T_b - T) + (1 - w) F_open``, the rule of the Aragog lid. At a
    state that does not change within the step, ``F_int`` equals ``F_atm``.
    """
    sigma, T_eq = 5.670374419e-8, 255.0
    row = _row(w, Time=100.0)
    d, k, T_melt = lid_boundary(row)
    T = brentq(lambda v: sigma * (v**4 - T_eq**4) - k / d * (T_melt - v), T_eq, T_melt)
    row.update(T_surf=T, F_atm=sigma * (T**4 - T_eq**4))
    loss = w * 4.0 * (1400.0 - T) + (1.0 - w) * lid_open_flux(row)
    r, dt = 6.4e6, 10.0
    out = _out(w, dE=-loss * 4.0 * np.pi * r**2 * dt * YEAR, r=r)
    F_int = _surface_flux_of_step(out, row, 'flux', 100.0 + dt)
    assert row['F_atm'] == pytest.approx(k / d * (T_melt - T), rel=1e-9)  # the call balance
    assert F_int - row['F_atm'] == pytest.approx(0.0, abs=1e-9 * row['F_atm'])


@pytest.mark.parametrize(
    'obc, mush, held', [(4, True, True), (4, False, False), (1, True, False)]
)
def test_the_solver_boundary_takes_the_held_surface_and_its_cap(obc, mush, held):
    """The solver boundary holds the row surface with its cap only in flux mode with the mush base."""
    bc = SimpleNamespace(outer_boundary_condition=obc, lid_mush_base=mush)
    _set_lid_surface(bc, _row(1.0, T_surf=900.0))
    assert bc.lid_surface_temperature == (900.0 if held else None)
    assert bc.lid_surface_flux_cap == (pytest.approx(2000.0) if held else None)
    _set_lid_surface(bc, _row(0.0, T_surf=900.0))
    assert bc.lid_surface_flux_cap is None


def _resumed_row(tmp_path, drop, **cols):
    """Last row of a stored helpfile without ``drop``, read as a resumed run reads it."""
    import pandas as pd

    from proteus.utils.coupler import ReadHelpfileFromCSV, ZeroHelpfileRow

    row = ZeroHelpfileRow()
    row.update(T_surf=900.0, T_magma=1450.0, F_atm=42.0, boundary_layer_thickness=0.01)
    row.update(boundary_layer_conductivity=2.0, **cols)
    for key in drop:
        del row[key]
    pd.DataFrame([row, row]).to_csv(tmp_path / 'runtime_helpfile.csv', sep='\t', index=False)
    return ReadHelpfileFromCSV(str(tmp_path)).iloc[-1].to_dict()


def test_a_run_stored_without_the_lid_columns_resumes_as_no_lid(tmp_path):
    """The resumed row of a helpfile without the lid columns acts as a surface without a lid."""
    from proteus.utils.coupler import GetHelpfileKeys

    lid_keys = [k for k in GetHelpfileKeys() if k.startswith('lid_')]
    assert len(lid_keys) == 9
    hf = _resumed_row(tmp_path, lid_keys)
    assert all(hf[k] == 0.0 for k in lid_keys)
    assert lid_boundary(hf) == (0.01, 2.0, 1450.0)
    assert lid_open_flux(hf) == pytest.approx(42.0, rel=1e-15)
    assert lid_flux_cap(hf) is None
    bc = SimpleNamespace(outer_boundary_condition=4, lid_mush_base=True)
    _set_lid_surface(bc, hf)
    assert (bc.lid_surface_temperature, bc.lid_surface_flux_cap) == (900.0, None)


def test_a_lid_resumed_without_the_skin_conductivity_is_refused_with_its_cause(tmp_path):
    """A stored lid row without boundary_layer_conductivity resumes with k = 0; the cap refuses it."""
    hf = _resumed_row(
        tmp_path,
        ['boundary_layer_conductivity'],
        lid_weight=1.0,
        lid_conductance=100.0,
        lid_T_base=1400.0,
    )
    assert hf['boundary_layer_conductivity'] == 0.0
    bc = SimpleNamespace(outer_boundary_condition=4, lid_mush_base=True)
    with pytest.raises(ValueError, match='boundary_layer_conductivity'):
        _set_lid_surface(bc, hf)


@pytest.mark.parametrize(('value', 'want'), [(0.1733, 0.1733), (0.0, 0.0), (float('nan'), 0.0)])
def test_helpfile_carries_the_lid_thickness(value, want):
    """``lid_delta`` [m] is a helpfile column: the solver thickness of the sub-grid lid, 0 where the
    solver has no lid (NaN); a helpfile written without the column resumes with 0."""
    from unittest.mock import MagicMock, patch

    from proteus.interior_energetics.aragog import AragogRunner
    from proteus.utils.coupler import RESUMABLE_ZERO_FILL_KEYS, GetHelpfileKeys

    assert 'lid_delta' in GetHelpfileKeys()
    assert 'lid_delta' in RESUMABLE_ZERO_FILL_KEYS
    out = MagicMock()
    out.T_core = 3800.0
    out.T_basic = np.array([4100.0, 3900.0, 3500.0, 3000.0])
    out.r_basic = np.array([3.5e6, 4.5e6, 5.5e6, 6.4e6])
    out.mass_stag = np.ones(3)
    out.lid_delta = value
    with patch('proteus.interior_energetics.aragog._estimate_T_pot', return_value=3000.0):
        res = AragogRunner._build_helpfile_output(out, {'F_atm': 1e5})
    assert res['lid_delta'] == want


def _skin_config(surface_d=0.01, surface_k=2.0, solid_cond=4.0, delta_on=None, lid=True):
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(surface_d=surface_d, surface_k=surface_k),
        interior_energetics=SimpleNamespace(
            solid_cond=solid_cond,
            aragog=SimpleNamespace(lid_delta_on=delta_on, lid_mush_base=lid),
        ),
    )


def test_lid_onset_thickness_of_the_default_skin_is_the_validated_value(caplog):
    """The schema defaults of the skin and the solid conductivity give k d / k_skin = 0.02 m
    exactly, the onset thickness of the lid validation runs."""
    import attrs

    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._interior import Interior
    from proteus.interior_energetics.aragog import lid_delta_on_of

    atm, ie = attrs.fields(AtmosClim), attrs.fields(Interior)
    cfg = _skin_config(atm.surface_d.default, atm.surface_k.default, ie.solid_cond.default)
    with caplog.at_level('WARNING'):
        assert lid_delta_on_of(cfg) == pytest.approx(0.02, rel=1e-15)
    assert not caplog.records
    assert lid_delta_on_of(_skin_config(0.02, 1.0, 3.0)) == pytest.approx(0.06, rel=1e-15)


def test_an_explicit_lid_onset_thickness_is_used_and_warned_beyond_10_percent(caplog):
    """An explicit value is used as given; it is logged as a warning when it differs from the
    skin value by more than 10 %, and used without a check on a saturation skin."""
    from proteus.interior_energetics.aragog import lid_delta_on_of

    with caplog.at_level('WARNING'):
        assert lid_delta_on_of(_skin_config(delta_on=0.021)) == pytest.approx(0.021, rel=1e-15)
    assert not caplog.records
    with caplog.at_level('WARNING'):
        assert lid_delta_on_of(_skin_config(delta_on=0.5)) == pytest.approx(0.5, rel=1e-15)
    assert any('lid_delta_on' in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level('WARNING'):
        assert lid_delta_on_of(
            _skin_config('saturation', 'saturation', delta_on=0.3)
        ) == pytest.approx(0.3, rel=1e-15)
    assert not caplog.records
    with pytest.raises(ValueError, match='lid_delta_on'):
        lid_delta_on_of(_skin_config('saturation', 'saturation'))


@pytest.mark.parametrize(
    ('delta_on', 'warns'),
    [
        (0.022, False),
        (0.018, False),
        (0.0219, False),
        (0.0181, False),
        (0.0221, True),
        (0.0179, True),
    ],
)
def test_the_lid_onset_warning_starts_at_10_percent_from_the_skin(caplog, delta_on, warns):
    """Values 9.5 % from the 0.02 m skin value pass quietly, values 10.5 % away warn."""
    from proteus.interior_energetics.aragog import lid_delta_on_of

    with caplog.at_level('WARNING'):
        assert lid_delta_on_of(_skin_config(delta_on=delta_on)) == delta_on
    assert any('lid_delta_on' in r.getMessage() for r in caplog.records) is warns


@pytest.mark.parametrize(('lid', 'want'), [(True, 0.02), (False, 0.0)])
def test_helpfile_carries_the_lid_onset_thickness(lid, want):
    """``lid_delta_on`` [m] is a helpfile column: the onset thickness the solver uses, 0 without
    the lid; a helpfile written without the column resumes with 0."""
    from proteus.interior_energetics.aragog import _lid_onset_out
    from proteus.utils.coupler import RESUMABLE_ZERO_FILL_KEYS, GetHelpfileKeys

    assert 'lid_delta_on' in GetHelpfileKeys()
    assert 'lid_delta_on' in RESUMABLE_ZERO_FILL_KEYS
    bc = SimpleNamespace(subgrid_lid=lid, lid_delta_on=0.02)
    solver = SimpleNamespace(parameters=SimpleNamespace(boundary_conditions=bc))
    assert _lid_onset_out(SimpleNamespace(aragog_solver=solver)) == want
    assert _lid_onset_out(None) == 0.0


@pytest.mark.parametrize(('w', 'lid'), [(0.0, True), (0.4, True), (1.0, True), (1.0, False)])
def test_update_solver_passes_the_lid_boundary_of_the_row(w, lid):
    """The interior takes the open-surface flux of the row under a live lid and the flux of the
    atmosphere without one, and holds the lid surface at the row surface with the row cap. A
    solver without the lid (a resume with the lid switched off) takes the flux of the atmosphere
    even where the row still has a lid weight."""
    bc = SimpleNamespace(
        outer_boundary_condition=4,
        subgrid_lid=lid,
        lid_mush_base=lid,
        outer_boundary_value=None,
        lid_surface_temperature=-1.0,
        lid_surface_flux_cap=-1.0,
        equilibrium_temperature=None,
    )
    params = SimpleNamespace(
        solver=SimpleNamespace(start_time=None, end_time=None),
        boundary_conditions=bc,
        energy=SimpleNamespace(tidal_array=None),
    )
    interior_o = SimpleNamespace(
        aragog_solver=SimpleNamespace(parameters=params, solution=None), tides=np.zeros(3)
    )
    row = _row(w, G=21.9, T_b=1440.0, T_surf=727.3, F_atm=7000.0, lid_F_open=165000.0)
    row.update(Time=30000.0, T_eqm=255.0)
    AragogRunner.update_solver(25.0, row, interior_o)
    assert params.solver.end_time == pytest.approx(30025.0, rel=1e-15)
    assert bc.outer_boundary_value == (165000.0 if w > 0.0 and lid else 7000.0)
    assert bc.lid_surface_temperature == (727.3 if lid else None)
    if w > 0.0 and lid:
        assert bc.lid_surface_flux_cap == pytest.approx(21.9 * (1440.0 - 727.3), rel=1e-14)
    else:
        assert bc.lid_surface_flux_cap is None
    assert bc.equilibrium_temperature == pytest.approx(255.0, rel=1e-15)


def _lid_out_row(lid_weight=0.37, F_step=15500.0, dt=25.0):
    """A solver output with distinct lid values and a step that loses ``F_step`` [W/m^2]."""
    from unittest.mock import MagicMock

    out = MagicMock()
    out.T_core = 3800.0
    out.T_basic = np.array([4100.0, 3900.0, 3500.0, 1589.0])
    out.r_basic = np.array([3.5e6, 4.5e6, 5.5e6, 6.4e6])
    out.mass_stag = np.ones(3)
    out.lid_weight, out.lid_conductance, out.lid_delta = lid_weight, 21.9, 0.18
    out.lid_T_surface, out.lid_T_base = 727.0, 1440.0
    area = 4.0 * np.pi * 6.4e6**2
    out.step_dE_F_int_J = -F_step * area * dt * 3.15576e7
    return out


def test_helpfile_carries_each_lid_column_from_its_field():
    """Each lid column of the helpfile comes from its own solver field, F_int is the flux of the
    step integral under a live lid, and lid_F_diff is F_int less the F_atm of the row."""
    from unittest.mock import patch

    out = _lid_out_row()
    row = {'F_atm': 15000.0, 'Time': 30000.0}
    with patch('proteus.interior_energetics.aragog._estimate_T_pot', return_value=3000.0):
        res = AragogRunner._build_helpfile_output(
            out, row, surface_bc_mode='flux', t_end=30025.0
        )
        bc = SimpleNamespace(subgrid_lid=True, lid_delta_on=0.0213)
        solver = SimpleNamespace(parameters=SimpleNamespace(boundary_conditions=bc))
        with_solver = AragogRunner._build_helpfile_output(
            out,
            row,
            SimpleNamespace(aragog_solver=solver),
            surface_bc_mode='flux',
            t_end=30025.0,
        )
    assert res['lid_delta_on'] == 0.0
    assert with_solver['lid_delta_on'] == pytest.approx(bc.lid_delta_on, rel=1e-15)
    want = dict(
        lid_weight=0.37,
        lid_conductance=21.9,
        lid_delta=0.18,
        lid_T_surface=727.0,
        lid_T_base=1440.0,
        lid_T_node=1589.0,
    )
    assert {k: res[k] for k in want} == want
    assert res['F_int'] == pytest.approx(15500.0, rel=1e-12)
    assert res['lid_F_diff'] == pytest.approx(500.0, rel=1e-9)


def test_lid_flux_difference_is_zero_without_a_lid_in_grey_body_mode():
    """In grey_body mode F_int is the surface node flux and lid_F_diff reads 0: the column
    records the lid booking of the flux mode only."""
    from unittest.mock import patch

    out = _lid_out_row(lid_weight=float('nan'))
    out.heat_flux = np.array([90.0, 80.0, 500.0])
    row = {'F_atm': 2000.0, 'Time': 30000.0}
    with patch('proteus.interior_energetics.aragog._estimate_T_pot', return_value=3000.0):
        res = AragogRunner._build_helpfile_output(
            out, row, surface_bc_mode='grey_body', t_end=30025.0
        )
    assert res['F_int'] == pytest.approx(500.0, rel=1e-15)
    assert res['lid_weight'] == pytest.approx(0.0, abs=1e-300)
    assert res['lid_F_diff'] == pytest.approx(0.0, abs=1e-300)


@pytest.mark.parametrize('end_weight', [float('nan'), 0.0])
@pytest.mark.physics_invariant
def test_a_step_that_starts_with_a_lid_books_its_integral_when_the_lid_ends(end_weight):
    """A step that starts under a live lid and ends without one (hand-over, weight below the
    floor) books the flux of its step integral, not the F_atm the lid replaced."""
    out = _lid_out_row(lid_weight=end_weight)
    row = {'F_atm': 15000.0, 'Time': 30000.0, 'lid_weight': 1.0}
    got = _surface_flux_of_step(out, row, 'flux', 30025.0)
    assert got == pytest.approx(15500.0, rel=1e-12)
    no_lid = {'F_atm': 15000.0, 'Time': 30000.0, 'lid_weight': 0.0}
    assert _surface_flux_of_step(out, no_lid, 'flux', 30025.0) == pytest.approx(
        15000.0, rel=1e-15
    )


def test_a_weight_without_a_conductance_is_no_lid_on_every_path():
    """A row with a lid weight but no lid conductance is no lid for the atmosphere call, the
    cap, the open-surface flux and the boundary value of the interior alike."""
    row = _row(0.5, G=0.0, T_surf=1300.0, F_atm=7.0, lid_F_open=120000.0)
    assert lid_boundary(row) == (0.01, 2.0, 1450.0)
    assert lid_flux_cap(row) is None
    assert lid_open_flux(row) == pytest.approx(7.0, rel=1e-15)
    bc = SimpleNamespace(
        outer_boundary_condition=4,
        subgrid_lid=True,
        lid_mush_base=True,
        outer_boundary_value=None,
        lid_surface_temperature=None,
        lid_surface_flux_cap=None,
        equilibrium_temperature=None,
    )
    params = SimpleNamespace(
        solver=SimpleNamespace(start_time=None, end_time=None),
        boundary_conditions=bc,
        energy=SimpleNamespace(tidal_array=None),
    )
    interior_o = SimpleNamespace(
        aragog_solver=SimpleNamespace(parameters=params, solution=None), tides=np.zeros(3)
    )
    row.update(Time=1.0, T_eqm=255.0)
    AragogRunner.update_solver(1.0, row, interior_o)
    assert bc.outer_boundary_value == pytest.approx(7.0, rel=1e-15)


def test_lid_surface_takes_the_slope_with_the_cap():
    """The slope of the row goes to the solver with the cap; without a cap it is None."""
    bc = SimpleNamespace(outer_boundary_condition=4, lid_mush_base=True)
    _set_lid_surface(bc, _row(1.0, T_surf=900.0, F_atm=1.0), slope=87.0)
    assert bc.lid_surface_flux_cap is not None
    assert bc.lid_surface_flux_slope == 87.0
    _set_lid_surface(bc, _row(0.0, T_surf=900.0, F_atm=1.0), slope=87.0)
    assert bc.lid_surface_flux_cap is None
    assert bc.lid_surface_flux_slope is None


def test_secant_of_the_cap():
    """The secant of the atmosphere flux between the held and the end surface, at least 0."""
    assert lid_cap_secant(700.0, 710.0, 15000.0, 15870.0) == pytest.approx(87.0, rel=1e-12)
    assert lid_cap_secant(700.0, 710.0, 15000.0, 14000.0) == 0.0


@pytest.mark.parametrize(
    ('tan', 'sec', 'skip'),
    [
        (0.0, 0.0, True),
        (86.98, 86.98, True),
        (86.98, 86.76, False),
        (0.0, 0.0766, False),
        (100.0, 100.05, True),
    ],
)
def test_resolve_is_skipped_where_the_secant_is_the_tangent(tan, sec, skip):
    """A re-solve with a secant within 1e-3 of the tangent changes nothing and is skipped."""
    assert lid_resolve_skip(tan, sec) is skip


def _corrector(T_end, F_end, tangent=0.0, flux_at=True):
    """Runner and solver stubs: the solve records the slope and the start state it sees."""
    bc = SimpleNamespace(
        lid_surface_temperature=700.0,
        lid_surface_flux_cap=1.0e4,
        lid_surface_flux_slope=tangent,
    )
    solver = SimpleNamespace(
        parameters=SimpleNamespace(
            boundary_conditions=bc, solver=SimpleNamespace(end_time=110.0)
        ),
        _solution='start',
        _lid_init='lid0',
        _last_compression_J=1.0,
    )
    seen = []

    def solve(hf_row, interior_o, S0_entry=None):
        seen.append(
            (
                bc.lid_surface_flux_slope,
                solver._solution,
                solver._lid_init,
                solver.parameters.solver.end_time,
                S0_entry,
            )
        )
        solver._solution, solver._lid_init = 'end', None
        solver.parameters.solver.end_time = 99.0
        return SimpleNamespace(lid_T_surface=T_end[len(seen) - 1])

    runner = SimpleNamespace(
        _solve_with_retry=solve, _lid_flux_at=(lambda T: F_end) if flux_at else None
    )
    interior_o = SimpleNamespace(aragog_solver=solver)
    return runner, solver, interior_o, seen


@pytest.mark.parametrize('T_end', [690.0, 700.0])
def test_no_resolve_where_the_lid_ends_at_or_below_the_held_surface(T_end):
    """The cap did not act at the end: the first solve stands."""
    runner, solver, io, seen = _corrector([T_end], 0.0)
    snap = _call_start(solver)
    out = runner._solve_with_retry({}, io, 'S0')
    assert AragogRunner._correct_lid_cap(runner, {'F_atm': 15000.0}, io, 'S0', out, snap) is out
    assert len(seen) == 1
    assert getattr(io, 'lid_cap_resolves', 0) == 0


@pytest.mark.parametrize(('dT', 'resolves'), [(0.5e-6, False), (2.0e-6, True)])
def test_the_cap_acts_only_above_its_activation_margin(dT, resolves):
    """The re-solve starts only where the lid surface ends more than LID_CAP_DT_ACT = 1e-6 K
    above the held one: half the margin keeps the first solve, twice the margin solves again."""
    runner, solver, io, seen = _corrector([700.0 + dT] * 2, 15000.0 + 2000.0 * dT)
    snap = _call_start(solver)
    out = runner._solve_with_retry({}, io, 'S0')
    out2 = AragogRunner._correct_lid_cap(runner, {'F_atm': 15000.0}, io, 'S0', out, snap)
    assert len(seen) == (2 if resolves else 1)
    assert (out2 is out) is not resolves
    if resolves:
        assert seen[1][0] == pytest.approx(2000.0, rel=1e-6)


def test_resolve_takes_the_secant_from_the_start_of_the_call():
    """Where the lid ends above the held surface the call runs again from its start state with
    the secant as slope, once, and the count and wall time of the re-solve are kept."""
    runner, solver, io, seen = _corrector([710.0, 705.0], 15870.0, tangent=0.0)
    snap = _call_start(solver)
    out = runner._solve_with_retry({}, io, 'S0')
    out2 = AragogRunner._correct_lid_cap(runner, {'F_atm': 15000.0}, io, 'S0', out, snap)
    assert out2.lid_T_surface == 705.0
    assert len(seen) == 2
    assert seen[1] == (pytest.approx(87.0, rel=1e-12), 'start', 'lid0', 110.0, 'S0')
    assert io.lid_cap_resolves == 1
    assert io.lid_cap_resolve_wall >= 0.0


def test_a_resolve_keeps_one_stall_window_entry_per_step():
    """The re-solve of the corrector runs the step again from its start: the stall window holds
    the entry of the re-solve only, not also the one of the discarded first solve."""
    runner, solver, io, seen = _corrector([710.0, 705.0], 15870.0, tangent=0.0)
    io.aragog_step_progress = [(1.0, 1.0)]
    solve = runner._solve_with_retry

    def solve_and_book(hf_row, interior_o, S0_entry=None):
        out = solve(hf_row, interior_o, S0_entry)
        interior_o.aragog_step_progress.append((float(len(seen)), 1.0))
        return out

    runner._solve_with_retry = solve_and_book
    snap = _call_start(solver, io)
    out = runner._solve_with_retry({}, io, 'S0')
    AragogRunner._correct_lid_cap(runner, {'F_atm': 15000.0}, io, 'S0', out, snap)
    assert len(seen) == 2
    assert io.aragog_step_progress == [(1.0, 1.0), (2.0, 1.0)]


def test_resolve_is_skipped_or_off_without_a_provider():
    """A secant equal to the tangent, or no flux provider, keeps the first solve."""
    for kw in (dict(tangent=87.0), dict(tangent=0.0, flux_at=False)):
        runner, solver, io, seen = _corrector([710.0], 15870.0, **kw)
        snap = _call_start(solver)
        out = runner._solve_with_retry({}, io, 'S0')
        assert (
            AragogRunner._correct_lid_cap(runner, {'F_atm': 15000.0}, io, 'S0', out, snap)
            is out
        )
        assert len(seen) == 1


def test_guard_and_corrector_on_a_saturating_flux():
    """In the first row under a flux ceiling the surface jumps up while F_atm falls: the row slope
    is below 0 and the guard sets F' = 0 (the cap alone). The lid then ends far above the held
    surface, and the corrector runs the call again with the small secant of the ceiling."""
    import pandas as pd

    from proteus.atmos_clim.common import lid_flux_slope

    hf_all = pd.DataFrame(
        {
            'Time': [29921.01, 29950.67],
            'T_surf': [728.996, 819.271],
            'F_atm': [15808.86, 13749.08],
        }
    )
    row = {'Time': 29950.67, 'T_surf': 819.271, 'F_atm': 13749.08}
    tangent = lid_flux_slope(hf_all, row)
    assert tangent == 0.0
    F_end = 13749.08 + 0.07658 * (1406.255 - 819.271)
    runner, solver, io, seen = _corrector([1406.255, 1406.144], F_end, tangent=tangent)
    solver.parameters.boundary_conditions.lid_surface_temperature = 819.271
    snap = _call_start(solver)
    out = runner._solve_with_retry({}, io, 'S0')
    out2 = AragogRunner._correct_lid_cap(runner, row, io, 'S0', out, snap)
    assert out2.lid_T_surface == 1406.144
    assert seen[1][0] == pytest.approx(0.07658, rel=1e-9)
    assert io.lid_cap_resolves == 1


def test_update_solver_passes_the_slope_of_the_last_two_rows():
    """With the run table, update_solver hands Aragog the slope of F_atm with T_surf over the last
    2 rows next to the cap of a live lid."""
    import pandas as pd

    from proteus.atmos_clim.common import lid_flux_slope

    bc = SimpleNamespace(
        outer_boundary_condition=4,
        subgrid_lid=True,
        lid_mush_base=True,
        outer_boundary_value=None,
        lid_surface_temperature=-1.0,
        lid_surface_flux_cap=-1.0,
        lid_surface_flux_slope=-1.0,
        equilibrium_temperature=None,
    )
    params = SimpleNamespace(
        solver=SimpleNamespace(start_time=None, end_time=None),
        boundary_conditions=bc,
        energy=SimpleNamespace(tidal_array=None),
    )
    interior_o = SimpleNamespace(
        aragog_solver=SimpleNamespace(parameters=params, solution=None), tides=np.zeros(3)
    )
    row = _row(0.4, G=21.9, T_b=1440.0, T_surf=727.3, F_atm=7000.0, lid_F_open=165000.0)
    row.update(Time=30000.0, T_eqm=255.0)
    hf_all = pd.DataFrame(
        {'Time': [29975.0, 30000.0], 'T_surf': [717.3, 727.3], 'F_atm': [6500.0, 7000.0]}
    )
    AragogRunner.update_solver(25.0, row, interior_o, hf_all=hf_all)
    assert lid_flux_slope(hf_all, row) == pytest.approx(50.0, rel=1e-12)
    assert bc.lid_surface_flux_slope == pytest.approx(50.0, rel=1e-12)


def test_the_structure_update_hands_the_run_table_to_update_solver():
    """The update to a new structure passes the run table on, so the row slope survives it."""
    from unittest.mock import MagicMock, patch

    rows, got = object(), []
    interior_o = SimpleNamespace(ic=2, aragog_solver=MagicMock(), _last_entropy=None)
    config = MagicMock()
    config.interior_energetics.aragog.core_bc = 'quasi_steady'
    with (
        patch.object(AragogRunner, 'update_structure'),
        patch.object(AragogRunner, '_refresh_entropy_eos'),
        patch.object(
            AragogRunner,
            'update_solver',
            side_effect=lambda *a, hf_all=None, **k: got.append(hf_all),
        ),
    ):
        AragogRunner._apply_structure_update(
            config, {'Time': 1.0}, interior_o, 1.0, hf_all=rows
        )
    assert got == [rows]


class _AfterSolve(Exception):
    pass


def test_run_solver_runs_the_corrector_on_the_first_solve():
    """run_solver hands the output of the first solve to the cap corrector of the call, before
    it builds the helpfile output."""
    seen = []
    solver = SimpleNamespace(
        _S0=np.zeros(3), parameters=SimpleNamespace(solver=SimpleNamespace(end_time=1.0))
    )

    second, built = SimpleNamespace(dt_actual=1.0), []

    def correct(hf_row, interior_o, S0_entry, out, snap):
        seen.append((out, S0_entry.copy(), snap))
        return second

    def build(out, *a, **k):
        built.append(out)
        raise _AfterSolve

    first = SimpleNamespace(dt_actual=1.0)
    runner = SimpleNamespace(
        _use_jax=False,
        _solve_with_retry=lambda hf_row, interior_o, S0_entry: first,
        _correct_lid_cap=correct,
        _build_helpfile_output=build,
        _config=SimpleNamespace(interior_energetics=SimpleNamespace(surface_bc_mode='flux')),
    )
    io = SimpleNamespace(aragog_solver=solver)
    with pytest.raises(_AfterSolve):
        AragogRunner.run_solver(
            runner, {'Time': 10.0, 'boundary_layer_thickness': 0.01}, io, {'output': '.'}
        )
    assert len(seen) == 1 and seen[0][0] is first
    assert len(built) == 1 and built[0] is second  # built from the output of the corrector
    assert np.array_equal(seen[0][1], np.zeros(3))
    assert seen[0][2]['end_time'] == 1.0


def _provider_config(module='dummy', fixed_flux=-1.0, prevent_warming=False):
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(module=module, dummy=SimpleNamespace(fixed_flux=fixed_flux)),
        planet=SimpleNamespace(prevent_warming=prevent_warming),
    )


def test_the_corrector_has_the_dummy_flux_only(monkeypatch):
    """The flux provider of the corrector is the dummy formula for the dummy atmosphere, with the
    clip of prevent_warming, and none for AGNI, JANUS or a fixed dummy flux."""
    import proteus.atmos_clim.dummy as dummy
    from proteus.interior_energetics.wrapper import lid_flux_provider

    monkeypatch.setattr(dummy, 'net_flux', lambda config, hf_row, T: 3.0 * T - 2000.0)
    f = lid_flux_provider(_provider_config(), {})
    assert f(1000.0) == pytest.approx(1000.0, rel=1e-15)
    assert f(500.0) == pytest.approx(-500.0, rel=1e-15)
    f = lid_flux_provider(_provider_config(prevent_warming=True), {})
    assert f(1000.0) == pytest.approx(1000.0, rel=1e-15)
    assert f(500.0) == 1.0e-8
    for cfg in (
        _provider_config('agni'),
        _provider_config('janus'),
        _provider_config(fixed_flux=1.0e4),
    ):
        assert lid_flux_provider(cfg, {}) is None


@pytest.mark.parametrize('F_end', [float('nan'), float('inf'), -float('inf')])
def test_a_secant_from_a_non_finite_flux_is_zero(F_end):
    """A non-finite end flux gives the secant 0 (the cap alone), never a non-finite slope."""
    assert lid_cap_secant(700.0, 710.0, 15000.0, F_end) == 0.0
    assert lid_cap_secant(700.0, 710.0, float('nan'), 15870.0) == 0.0
    assert (
        lid_cap_secant(700.0, 700.0, 15000.0, 15001.0) == 0.0
    )  # no step above the held surface
