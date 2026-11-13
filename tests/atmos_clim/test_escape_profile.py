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


def _agni_atm(p_top_down, r_top_down, t_top_down, vmr_top_down):
    """Stand-in for an AGNI atmosphere: cell-centre arrays ordered top to bottom."""
    return SimpleNamespace(
        p=p_top_down,
        r=r_top_down,
        tmp=t_top_down,
        gas_names=list(vmr_top_down),
        gas_vmr=vmr_top_down,
    )


def _config(module):
    """Config stub selecting the climate backend."""
    return SimpleNamespace(atmos_clim=SimpleNamespace(module=module))


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
