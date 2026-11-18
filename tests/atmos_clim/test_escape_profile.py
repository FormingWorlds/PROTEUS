"""Unit tests for ``proteus.atmos_clim.escape_profile``, the column kept for escape.

Invariants checked: the stored column runs from the deepest level up, with
pressure strictly falling and radius strictly rising; a rejected climate solve
never replaces the column of an accepted one.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from proteus.atmos_clim.escape_profile import column_from_arrays, store_escape_profile

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _agni_atm(p_top_down, r_top_down, t_top_down, vmr_top_down, transparent=False):
    """Stand-in for an AGNI atmosphere: cell-centre arrays ordered top to bottom."""
    return SimpleNamespace(
        p=p_top_down,
        r=r_top_down,
        tmp=t_top_down,
        gas_names=list(vmr_top_down),
        gas_vmr=vmr_top_down,
        transparent=transparent,
    )


def _config(module, dispatcher=True):
    """Config stub selecting the climate backend and whether escape uses the dispatcher."""
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(module=module),
        escape=SimpleNamespace(uses_dispatcher=dispatcher),
    )


@pytest.mark.physics_invariant
def test_a_top_down_column_is_stored_base_to_top():
    """AGNI orders its levels from the top; escape reads them from the base.

    Physical scenario: five levels from 1e-6 to 1e2 Pa, given top first, with
    a water mixing ratio that falls with height and a CO2 one that is zero
    everywhere. The stored column starts at the deepest level and drops the
    absent gas.
    """
    p = np.array([1e-6, 1e-4, 1e-2, 1e0, 1e2])
    r = np.array([7.0e6, 6.9e6, 6.8e6, 6.7e6, 6.6e6])
    T = np.array([300.0, 310.0, 320.0, 330.0, 340.0])
    vmr = {'H2O': np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 'CO2': np.zeros(5)}
    col = column_from_arrays(p, r, T, vmr)
    np.testing.assert_allclose(col['p'], p[::-1], rtol=1e-15)
    np.testing.assert_allclose(col['r'], r[::-1], rtol=1e-15)
    np.testing.assert_allclose(col['T'], T[::-1], rtol=1e-15)
    np.testing.assert_allclose(col['vmr']['H2O'], vmr['H2O'][::-1], rtol=1e-15)
    assert 'CO2' not in col['vmr']
    assert np.all(np.diff(col['p']) < 0.0) and np.all(np.diff(col['r']) > 0.0)


@pytest.mark.physics_invariant
def test_levels_that_do_not_rise_are_dropped():
    """A level whose radius does not rise above the one below it is dropped.

    Physical scenario: the second level from the base repeats the radius of
    the base, as a degenerate cell would. Keeping it would hand the dispatcher
    a column that is not monotone in radius, which it refuses.
    """
    p = np.array([1e5, 1e4, 1e3, 1e2])
    r = np.array([6.4e6, 6.4e6, 6.5e6, 6.6e6])
    T = np.array([500.0, 400.0, 300.0, 250.0])
    col = column_from_arrays(p, r, T, {'N2': np.ones(4)})
    np.testing.assert_allclose(col['p'], [1e5, 1e3, 1e2], rtol=1e-15)
    np.testing.assert_allclose(col['T'], [500.0, 300.0, 250.0], rtol=1e-15)


def test_a_malformed_column_is_refused():
    """Arrays of different lengths, too few usable levels, or no gas are refused."""
    with pytest.raises(ValueError, match='differ in length'):
        column_from_arrays([1.0, 2.0], [1.0], [1.0, 1.0], {})
    with pytest.raises(ValueError, match='three usable levels'):
        column_from_arrays([1e3, 1e2], [1.0, 2.0], [300.0, 300.0], {'N2': [1.0, 1.0]})
    with pytest.raises(ValueError, match='no gas'):
        column_from_arrays([1e3, 1e2, 1e1], [1.0, 2.0, 3.0], [300.0] * 3, {'N2': [0.0] * 3})


def test_a_rejected_solve_keeps_the_last_accepted_column(caplog):
    """A solve the climate solver rejected leaves the stored column in place.

    The column is then marked as not current, so the escape step records that
    it read an older structure. A column that cannot be read keeps the old
    one in the same way and logs why.
    """
    p = np.array([1e-4, 1e0, 1e4])
    r = np.array([6.6e6, 6.5e6, 6.4e6])
    T = np.array([200.0, 250.0, 300.0])
    good = _agni_atm(p, r, T, {'N2': np.ones(3)})
    atmos_o = SimpleNamespace(
        _atm=good, converged=True, escape_profile=None, escape_profile_current=False
    )
    store_escape_profile(atmos_o, _config('agni'), {})
    assert atmos_o.escape_profile_current is True
    stored = atmos_o.escape_profile
    np.testing.assert_allclose(stored['p'], p[::-1], rtol=1e-15)

    atmos_o._atm = _agni_atm(p * 2.0, r, T, {'N2': np.ones(3)})
    atmos_o.converged = False
    store_escape_profile(atmos_o, _config('agni'), {})
    assert atmos_o.escape_profile is stored
    assert atmos_o.escape_profile_current is False

    atmos_o.converged = True
    atmos_o._atm = SimpleNamespace()
    with caplog.at_level(logging.WARNING):
        store_escape_profile(atmos_o, _config('agni'), {})
    assert atmos_o.escape_profile is stored
    assert atmos_o.escape_profile_current is False
    assert 'not stored for escape' in caplog.text


def test_janus_heights_are_measured_from_the_surface():
    """JANUS heights start at the surface, so the interior radius is added."""
    atm = SimpleNamespace(
        p=np.array([1e5, 1e3, 1e1]),
        z=np.array([0.0, 5.0e4, 1.0e5]),
        tmp=np.array([300.0, 250.0, 200.0]),
        x_gas={'H2O': np.array([0.5, 0.5, 0.5])},
    )
    atmos_o = SimpleNamespace(
        _atm_janus_last=atm,
        converged=True,
        escape_profile=None,
        escape_profile_current=False,
    )
    store_escape_profile(atmos_o, _config('janus'), {'R_int': 6.4e6})
    np.testing.assert_allclose(atmos_o.escape_profile['r'], [6.4e6, 6.45e6, 6.5e6], rtol=1e-15)
    np.testing.assert_allclose(atmos_o.escape_profile['p'], [1e5, 1e3, 1e1], rtol=1e-15)
    assert atmos_o.escape_profile_current is True


def test_the_dummy_atmosphere_stores_no_column():
    """The dummy atmosphere has no column; escape builds its own stand-in."""
    atmos_o = SimpleNamespace(
        converged=True, escape_profile={'p': 1}, escape_profile_current=True
    )
    store_escape_profile(atmos_o, _config('dummy'), {})
    assert atmos_o.escape_profile is None
    assert atmos_o.escape_profile_current is False


@pytest.mark.physics_invariant
def test_levels_with_no_pressure_or_temperature_are_dropped():
    """Levels at zero pressure or temperature, and NaN mixing ratios, never reach escape.

    Physical scenario: a five-level column whose second level from the top
    has T = 0 K and whose top level has p = 0 Pa, with a NaN water mixing
    ratio at the base. The two bad levels are dropped and the NaN reads as
    no water, so the stored column is three levels with positive p and T.
    """
    p = np.array([0.0, 1e-2, 1e0, 1e2, 1e4])
    r = np.array([7.0e6, 6.9e6, 6.8e6, 6.7e6, 6.6e6])
    T = np.array([300.0, 0.0, 320.0, 330.0, 340.0])
    vmr = {'H2O': np.array([0.1, 0.2, 0.3, 0.4, np.nan]), 'N2': np.full(5, 0.5)}
    col = column_from_arrays(p, r, T, vmr)
    np.testing.assert_allclose(col['p'], [1e4, 1e2, 1e0], rtol=1e-15)
    np.testing.assert_allclose(col['T'], [340.0, 330.0, 320.0], rtol=1e-15)
    np.testing.assert_allclose(col['vmr']['H2O'], [0.0, 0.4, 0.3], rtol=1e-15)
    assert np.all(col['p'] > 0.0) and np.all(col['T'] > 0.0)


def test_no_column_is_kept_without_the_dispatcher_or_for_a_transparent_agni():
    """Escape keeps no column when it does not use the dispatcher, or when AGNI is transparent.

    Below its surface-pressure threshold AGNI collapses the column onto the
    top level at its floor temperature, which escape must not read as an
    atmosphere: the stored column is cleared so escape builds its stand-in
    from the surface pressure. Without the dispatcher nothing is stored.
    """
    p = np.array([1e-4, 1e0, 1e4])
    r = np.array([6.6e6, 6.5e6, 6.4e6])
    T = np.array([200.0, 250.0, 300.0])
    stale = {'p': np.ones(3)}
    for atm, dispatcher in (
        (_agni_atm(p, r, T, {'N2': np.ones(3)}, transparent=True), True),
        (_agni_atm(p, r, T, {'N2': np.ones(3)}), False),
    ):
        atmos_o = SimpleNamespace(
            _atm=atm, converged=True, escape_profile=stale, escape_profile_current=True
        )
        store_escape_profile(atmos_o, _config('agni', dispatcher=dispatcher), {})
        assert atmos_o.escape_profile is None
        assert atmos_o.escape_profile_current is False
    # The same opaque column under the dispatcher is stored.
    atmos_o = SimpleNamespace(
        _atm=_agni_atm(p, r, T, {'N2': np.ones(3)}),
        converged=True,
        escape_profile=None,
        escape_profile_current=False,
    )
    store_escape_profile(atmos_o, _config('agni'), {})
    assert atmos_o.escape_profile_current is True
    np.testing.assert_allclose(atmos_o.escape_profile['p'], p[::-1], rtol=1e-15)


def _write_snapshot(data_dir, time, p, r_or_z, T, vmr, janus=False, transparent=False):
    """Write an atmosphere snapshot with the variables PROTEUS reads, cell centres top first."""
    import netCDF4 as nc

    from proteus.utils.helper import snapshot_path_for_time

    path = snapshot_path_for_time(str(data_dir), time, '_atm.nc')
    n, gases = len(p), list(vmr)
    with nc.Dataset(path, 'w') as ds:
        ds.createDimension('nlev_c', n)
        ds.createDimension('nlev_l', n + 1)
        ds.createDimension('ngases', len(gases))
        ds.createDimension('nchars', 16)
        ds.createDimension('one', 1)
        edges = np.concatenate([[p[0] / 2.0], np.sqrt(p[:-1] * p[1:]), [p[-1] * 2.0]])
        for name, dim, val in (
            ('p', 'nlev_c', p),
            ('pl', 'nlev_l', edges),
            ('tmp', 'nlev_c', T),
            ('tmpl', 'nlev_l', np.interp(np.arange(n + 1) - 0.5, np.arange(n), T)),
        ):
            ds.createVariable(name, 'f8', (dim,))[:] = val
        height = 'z' if janus else 'r'
        ds.createVariable(height, 'f8', ('nlev_c',))[:] = r_or_z
        ds.createVariable(height + 'l', 'f8', ('nlev_l',))[:] = np.interp(
            np.arange(n + 1) - 0.5, np.arange(n), r_or_z
        )
        ds.createVariable('planet_radius', 'f8', ('one',))[:] = 1.0
        ds.createVariable('gravity', 'f8', ('nlev_c',))[:] = np.full(n, 9.8)
        ds.createVariable('transparent', 'S1', ('one',))[:] = np.array(
            [b'y' if transparent else b'n']
        )
        names = ds.createVariable('gases', 'S1', ('ngases', 'nchars'))
        names[:] = np.array([list(g.ljust(16)) for g in gases], dtype='S1')
        ds.createVariable('x_gas', 'f8', ('nlev_c', 'ngases'))[:] = np.column_stack(
            [vmr[g] for g in gases]
        )
    return path


def _resume_case(tmp_path, stale, times=(10.0, 20.0, 30.0)):
    """Helpfile of a resumed run whose last row carries ``atm_levels_stale = stale``."""
    import pandas as pd

    (tmp_path / 'data').mkdir(exist_ok=True)
    rows = [{'Time': t, 'R_int': 6.4e6, 'atm_levels_stale': 0.0} for t in times]
    rows[-1]['atm_levels_stale'] = float(stale)
    return pd.DataFrame(rows)


def _fresh_atmos():
    return SimpleNamespace(escape_profile=None, escape_profile_current=False)


@pytest.mark.physics_invariant
def test_a_resume_restores_the_column_the_continuous_run_holds(tmp_path):
    """A resumed AGNI run hands escape the column of its last converged solve.

    Physical scenario: an N2-H2O column of five levels from 1e-6 to 1e2 Pa
    written at the resumed row. Stored from the live struct and restored from
    the snapshot, the two columns agree level by level, and the restored one
    is current, as after an accepted solve. Without the restore the first
    escape step of a resumed run read the isothermal stand-in instead.
    """
    from proteus.atmos_clim.escape_profile import restore_escape_profile

    p = np.array([1e-6, 1e-4, 1e-2, 1e0, 1e2])
    r = np.array([7.0e6, 6.9e6, 6.8e6, 6.7e6, 6.6e6])
    T = np.array([900.0, 700.0, 500.0, 400.0, 350.0])
    vmr = {
        'H2O': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'N2': np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
    }
    live = SimpleNamespace(
        _atm=_agni_atm(p, r, T, vmr),
        converged=True,
        escape_profile=None,
        escape_profile_current=False,
    )
    store_escape_profile(live, _config('agni'), {})

    hf_all = _resume_case(tmp_path, stale=0)
    _write_snapshot(tmp_path / 'data', 30.0, p, r, T, vmr)
    resumed = _fresh_atmos()
    restore_escape_profile(resumed, _config('agni'), hf_all, str(tmp_path))
    assert resumed.escape_profile_current is True
    for key in ('p', 'r', 'T'):
        np.testing.assert_allclose(
            resumed.escape_profile[key], live.escape_profile[key], rtol=1e-15
        )
    assert set(resumed.escape_profile['vmr']) == set(live.escape_profile['vmr'])
    for gas, x in live.escape_profile['vmr'].items():
        np.testing.assert_allclose(resumed.escape_profile['vmr'][gas], x, rtol=1e-15)


def test_a_resume_after_rejected_solves_restores_the_last_accepted_column(tmp_path):
    """After a streak of rejected solves the restored column is the last accepted one.

    The resumed row carries ``atm_levels_stale = 1``, so the continuous run
    held the column of the row before it, marked as not current. The rejected
    snapshot of the resumed row is not read.
    """
    from proteus.atmos_clim.escape_profile import restore_escape_profile

    p = np.array([1e-4, 1e0, 1e4])
    r = np.array([6.6e6, 6.5e6, 6.4e6])
    T = np.array([200.0, 250.0, 300.0])
    hf_all = _resume_case(tmp_path, stale=1)
    _write_snapshot(tmp_path / 'data', 20.0, p, r, T, {'N2': np.ones(3)})
    _write_snapshot(tmp_path / 'data', 30.0, p * 3.0, r, T * 2.0, {'N2': np.ones(3)})
    resumed = _fresh_atmos()
    restore_escape_profile(resumed, _config('agni'), hf_all, str(tmp_path))
    assert resumed.escape_profile_current is False
    np.testing.assert_allclose(resumed.escape_profile['p'], p[::-1], rtol=1e-15)
    np.testing.assert_allclose(resumed.escape_profile['T'], T[::-1], rtol=1e-15)


def test_a_resumed_janus_column_starts_at_the_interior_radius(tmp_path):
    """JANUS snapshots hold heights, so the restored radii add the row's interior radius."""
    from proteus.atmos_clim.escape_profile import restore_escape_profile

    hf_all = _resume_case(tmp_path, stale=0)
    _write_snapshot(
        tmp_path / 'data',
        30.0,
        np.array([1e1, 1e3, 1e5]),
        np.array([1.0e5, 5.0e4, 0.0]),
        np.array([200.0, 250.0, 300.0]),
        {'H2O': np.full(3, 0.5)},
        janus=True,
    )
    resumed = _fresh_atmos()
    restore_escape_profile(resumed, _config('janus'), hf_all, str(tmp_path))
    np.testing.assert_allclose(resumed.escape_profile['r'], [6.4e6, 6.45e6, 6.5e6], rtol=1e-15)
    assert resumed.escape_profile_current is True


def test_a_resume_without_a_usable_snapshot_leaves_the_stand_in(tmp_path, caplog):
    """No column is restored from a missing, corrupt, or transparent snapshot, or past the record.

    Escape then builds its isothermal stand-in, as it did before, and the resume
    goes on. Without the dispatcher, or with the dummy atmosphere, nothing is
    read at all.
    """
    from proteus.atmos_clim.escape_profile import restore_escape_profile

    p = np.array([1e-4, 1e0, 1e4])
    r = np.array([6.6e6, 6.5e6, 6.4e6])
    T = np.array([200.0, 250.0, 300.0])

    hf_all = _resume_case(tmp_path, stale=0)
    resumed = _fresh_atmos()
    with caplog.at_level(logging.WARNING):
        restore_escape_profile(resumed, _config('agni'), hf_all, str(tmp_path))
    assert resumed.escape_profile is None and resumed.escape_profile_current is False
    assert 'no atmosphere snapshot of the last converged solve' in caplog.text

    # A snapshot cut short by a crash is passed over, not raised.
    from proteus.utils.helper import snapshot_path_for_time

    cut = snapshot_path_for_time(str(tmp_path / 'data'), 30.0, '_atm.nc')
    _write_snapshot(tmp_path / 'data', 30.0, p, r, T, {'N2': np.ones(3)})
    whole = open(cut, 'rb').read()
    with open(cut, 'wb') as hdl:
        hdl.write(whole[: len(whole) // 3])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        restore_escape_profile(resumed, _config('agni'), hf_all, str(tmp_path))
    assert resumed.escape_profile is None
    assert 'escape column not restored' in caplog.text

    _write_snapshot(tmp_path / 'data', 30.0, p, r, T, {'N2': np.ones(3)}, transparent=True)
    restore_escape_profile(resumed, _config('agni'), hf_all, str(tmp_path))
    assert resumed.escape_profile is None and resumed.escape_profile_current is False

    restore_escape_profile(
        resumed, _config('agni'), _resume_case(tmp_path, stale=3), str(tmp_path)
    )
    assert resumed.escape_profile is None

    _write_snapshot(tmp_path / 'data', 30.0, p, r, T, {'N2': np.ones(3)})
    for config in (_config('agni', dispatcher=False), _config('dummy')):
        restore_escape_profile(resumed, config, hf_all, str(tmp_path))
        assert resumed.escape_profile is None
