"""Coupling of the Aragog sub-grid lid: surface temperature, surface flux, conductance."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

from proteus.atmos_clim.common import lid_boundary, lid_flux_cap, lid_open_flux
from proteus.interior_energetics.aragog import (
    AragogRunner,
    _lid_surface_of_row,
    _set_lid_surface,
    _surface_flux_of_step,
)

pytestmark = pytest.mark.unit
YEAR = 3.15576e7


def _out(weight, dE=-1.0e24, r=6.4e6, flux_top=7.0):
    return SimpleNamespace(
        lid_weight=weight,
        step_dE_F_int_J=dE,
        r_basic=np.array([3.0e6, r]),
        heat_flux=np.array([1.0, flux_top]),
    )


def test_flux_mode_without_a_lid_books_the_flux_of_the_atmosphere():
    hf = {'F_atm': 42.0, 'Time': 100.0}
    out = _out(float('nan'))
    assert _surface_flux_of_step(out, hf, 'flux', 110.0) == 42.0
    assert _surface_flux_of_step(_out(0.0), hf, 'flux', 110.0) == 42.0


def test_an_output_without_lid_fields_reads_as_no_lid():
    """An Aragog without the sub-grid lid has no lid fields: its step books ``F_atm``."""
    out = SimpleNamespace(step_dE_F_int_J=-1.0e24, r_basic=np.array([3.0e6, 6.4e6]))
    assert _surface_flux_of_step(out, {'F_atm': 42.0, 'Time': 100.0}, 'flux', 110.0) == 42.0
    assert (
        _surface_flux_of_step(_out(0.7), {'F_atm': 42.0, 'Time': 100.0}, 'flux', 110.0) != 42.0
    )


def test_grey_body_mode_books_the_surface_node():
    assert (
        _surface_flux_of_step(_out(1.0), {'F_atm': 42.0, 'Time': 0.0}, 'grey_body', 5.0) == 7.0
    )


@pytest.mark.parametrize('dt', [1.0, 10.0, 1000.0])
def test_a_live_lid_books_the_integrated_flux_of_the_step(dt):
    """``F_int A dt = -step_dE_F_int_J``: the energy of the step and the booked flux agree."""
    out = _out(0.7)
    F = _surface_flux_of_step(out, {'F_atm': 42.0, 'Time': 100.0}, 'flux', 100.0 + dt)
    area = 4.0 * np.pi * 6.4e6**2
    assert F * area * dt * YEAR == pytest.approx(-out.step_dE_F_int_J, rel=1e-12)


def test_a_state_with_no_step_books_the_flux_of_the_atmosphere():
    assert (
        _surface_flux_of_step(_out(1.0), {'F_atm': 42.0, 'Time': 100.0}, 'flux', 100.0) == 42.0
    )
    assert (
        _surface_flux_of_step(_out(1.0), {'F_atm': 42.0, 'Time': 100.0}, 'flux', None) == 42.0
    )


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
    assert _lid_surface_of_row(row, mode, on) == want


def _row(w, G=4.0, T_b=1400.0, T_node=1500.0, **kw):
    return dict(
        lid_weight=w,
        lid_conductance=G,
        lid_T_base=T_b,
        lid_T_node=T_node,
        T_magma=1450.0,
        boundary_layer_thickness=kw.pop('d', 0.01),
        boundary_layer_conductivity=kw.pop('k', 2.0),
        **kw,
    )


def test_boundary_is_the_skin_without_a_lid_and_a_parallel_sum_with_one():
    no_lid = {
        'T_magma': 1450.0,
        'boundary_layer_thickness': 0.01,
        'boundary_layer_conductivity': 2.0,
    }
    assert lid_boundary(no_lid) == (0.01, 2.0, 1450.0)
    assert lid_boundary(_row(0.0)) == (0.01, 2.0, 1450.0)
    assert lid_boundary(_row(1.0)) == (2.0 / 4.0, 2.0, 1400.0)
    d, k, T = lid_boundary(_row(0.5))
    G = k / d
    assert G == pytest.approx(0.5 * 4.0 + 0.5 * 200.0, rel=1e-14)
    assert G * T == pytest.approx(0.5 * 4.0 * 1400.0 + 0.5 * 200.0 * 1500.0, rel=1e-14)


def test_call_caps_the_lid_conductance_at_the_skin():
    """In the call a lid conducts at most like the skin, so a thin lid never takes more than
    the open surface would."""
    d, k, T = lid_boundary(_row(1.0, G=400.0))
    assert k / d == pytest.approx(200.0, rel=1e-14)
    assert T == pytest.approx(1400.0, rel=1e-14)
    d, k, T = lid_boundary(_row(0.5, G=400.0))
    assert k / d == pytest.approx(200.0, rel=1e-14)
    assert (k / d) * T == pytest.approx(0.5 * 200.0 * 1400.0 + 0.5 * 200.0 * 1500.0, rel=1e-14)
    # at the skin value and below it the lid conductance passes unchanged
    assert lid_boundary(_row(1.0, G=200.0))[0] == pytest.approx(0.01, rel=1e-14)
    assert lid_boundary(_row(1.0, G=199.0))[0] == pytest.approx(2.0 / 199.0, rel=1e-14)


def test_boundary_reads_the_skin_of_the_columns():
    """A skin from the columns, as from condensate layers, sets the open branch."""
    d, k, T = lid_boundary(_row(0.5, d=0.1, k=1.0))
    assert k / d == pytest.approx(0.5 * 4.0 + 0.5 * 10.0, rel=1e-14)
    assert (k / d) * T == pytest.approx(0.5 * 4.0 * 1400.0 + 0.5 * 10.0 * 1500.0, rel=1e-14)
    assert lid_open_flux(_row(0.5, d=0.1, k=1.0, T_surf=900.0)) == pytest.approx(10.0 * 600.0)


def test_a_zero_skin_passes_without_a_lid_and_is_refused_with_one():
    """Without a lid the zero columns of a fixed surface pass unchanged; a live lid refuses."""
    assert lid_boundary(_row(0.0, d=0.0, k=0.0)) == (0.0, 0.0, 1450.0)
    assert lid_open_flux(_row(0.0, d=0.0, k=0.0, F_atm=42.0)) == 42.0
    with pytest.raises(ValueError, match='needs a skin'):
        lid_boundary(_row(0.5, d=0.0, k=0.0))
    with pytest.raises(ValueError, match='needs a skin'):
        lid_open_flux(_row(0.5, d=0.0, k=0.0, T_surf=900.0))
    with pytest.raises(ValueError, match='needs a skin'):
        lid_boundary(_row(0.5, d=0.0, k=2.0))


def test_the_two_parts_sum_to_the_balance_of_the_atmosphere():
    row = _row(0.3, T_surf=900.0, F_atm=1.0)
    d, k, T_melt = lid_boundary(row)
    F_lid = 4.0 * (1400.0 - 900.0)
    F_open = lid_open_flux(row)
    assert F_open == pytest.approx(200.0 * (1500.0 - 900.0), rel=1e-14)
    assert 0.3 * F_lid + 0.7 * F_open == pytest.approx(k / d * (T_melt - 900.0), rel=1e-12)


def test_open_flux_without_a_lid_is_the_flux_of_the_atmosphere():
    assert lid_open_flux({'F_atm': 42.0}) == 42.0
    assert lid_open_flux(_row(0.0, F_atm=42.0)) == 42.0


@pytest.mark.parametrize('w', [0.01, 0.2, 0.5, 0.9, 1.0])
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
    assert F_int - row['F_atm'] == pytest.approx(0.0, abs=1e-9 * row['F_atm'])


def test_cap_is_the_flux_the_call_gave_the_lid():
    """The cap is the lid branch of the atmosphere balance: G_call (T_b - T_surf)."""
    assert lid_flux_cap(_row(1.0, T_surf=900.0)) == pytest.approx(4.0 * 500.0, rel=1e-14)
    assert lid_flux_cap(_row(0.4, T_surf=900.0)) == pytest.approx(4.0 * 500.0, rel=1e-14)
    # a lid above the skin conductance gets the capped conductance of the call
    assert lid_flux_cap(_row(1.0, G=400.0, T_surf=900.0)) == pytest.approx(200.0 * 500.0)
    # no lid, or a surface at or above the base: no cap
    assert lid_flux_cap(_row(0.0, T_surf=900.0)) is None
    assert lid_flux_cap(_row(1.0, G=0.0, T_surf=900.0)) is None
    assert lid_flux_cap(_row(1.0, T_surf=1400.0)) is None
    with pytest.raises(ValueError, match='needs a skin'):
        lid_flux_cap(_row(0.5, d=0.0, k=0.0, T_surf=900.0))


@pytest.mark.parametrize(
    'obc, mush, held', [(4, True, True), (4, False, False), (1, True, False)]
)
def test_the_solver_boundary_takes_the_held_surface_and_its_cap(obc, mush, held):
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
    assert lid_open_flux(hf) == 42.0
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
        assert lid_delta_on_of(cfg) == 0.02
    assert not caplog.records
    assert lid_delta_on_of(_skin_config(0.02, 1.0, 3.0)) == pytest.approx(0.06, rel=1e-15)


def test_an_explicit_lid_onset_thickness_is_used_and_warned_beyond_10_percent(caplog):
    """An explicit value is used as given; it is logged as a warning when it differs from the
    skin value by more than 10 %, and used without a check on a saturation skin."""
    from proteus.interior_energetics.aragog import lid_delta_on_of

    with caplog.at_level('WARNING'):
        assert lid_delta_on_of(_skin_config(delta_on=0.021)) == 0.021
    assert not caplog.records
    with caplog.at_level('WARNING'):
        assert lid_delta_on_of(_skin_config(delta_on=0.5)) == 0.5
    assert any('lid_delta_on' in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level('WARNING'):
        assert lid_delta_on_of(_skin_config('saturation', 'saturation', delta_on=0.3)) == 0.3
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
    assert bc.equilibrium_temperature == 255.0


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


@pytest.mark.parametrize('end_weight', [float('nan'), 0.0])
def test_a_step_that_starts_with_a_lid_books_its_integral_when_the_lid_ends(end_weight):
    """A step that starts under a live lid and ends without one (hand-over, weight below the
    floor) books the flux of its step integral, not the F_atm the lid replaced."""
    out = _lid_out_row(lid_weight=end_weight)
    row = {'F_atm': 15000.0, 'Time': 30000.0, 'lid_weight': 1.0}
    got = _surface_flux_of_step(out, row, 'flux', 30025.0)
    assert got == pytest.approx(15500.0, rel=1e-12)
    no_lid = {'F_atm': 15000.0, 'Time': 30000.0, 'lid_weight': 0.0}
    assert _surface_flux_of_step(out, no_lid, 'flux', 30025.0) == 15000.0
