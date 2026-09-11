"""Unit tests for ``proteus.orbit.common``: the ``Tides_t`` container and
the adaptive substep controller shared by the star-planet and
planet-satellite tidal evolution models.

Exercises:

- ``Tides_t``: interaction lookup/registration contract.
- ``run_adaptive_orbit_substeps``: the shared accept/reject controller,
  including its ``get_C_planet`` refresh call (mocked here; see
  ``tests/interior_energetics/test_common.py`` for ``get_C_planet``'s
  own physics tests since it now lives in
  ``proteus.interior_energetics.common``).

Anti-happy-path coverage:

- ``Tides_t.get`` on an unregistered (primary, perturber) pair must
  raise ``KeyError`` rather than returning a default.

The Kepler solver and Hansen-coefficient machinery previously covered
here now live in ``proteus.orbit.hansen``; see ``test_hansen.py``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from proteus.config._orbit import OrbitSolver
from proteus.orbit.common import Tides_t, run_adaptive_orbit_substeps

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Tides_t
# ---------------------------------------------------------------------------


def test_tides_t_get_raises_keyerror_for_unregistered_interaction():
    """Error-contract case: querying an interaction that was never
    ``add``-ed must raise ``KeyError`` rather than returning ``None`` or
    a default, since callers (``sp1d``) index straight into ``.nmk``
    without a None-check.
    """
    tides = Tides_t()
    with pytest.raises(KeyError):
        tides.get(primary='planet', perturber='star')
    # No side effect: a failed lookup must not have registered anything.
    assert tides.interactions == []


def test_tides_t_add_from_file_populates_interaction_from_a_real_netcdf(tmp_path):
    """``add_from_file`` must register a (primary, perturber) interaction
    (same object ``add`` would return) and populate its ``nmk``/``sigma``/
    ``LNk`` arrays directly from a real netCDF lookup file, with the
    complex Love number correctly reassembled from its real/imaginary
    parts.
    """
    import netCDF4 as nc

    nmk_rows = np.array([[2, 0, -1], [2, 2, 3]], dtype=np.int64)
    sigma = np.array([0.1, 0.2], dtype=np.float64)
    lnk = np.array([0.3 - 0.05j, 0.4 - 0.06j], dtype=np.complex128)

    path = tmp_path / 'lookup.nc'
    with nc.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.createDimension('mode', len(sigma))
        ds.createVariable('n', 'i4', ('mode',))[:] = nmk_rows[:, 0]
        ds.createVariable('m', 'i4', ('mode',))[:] = nmk_rows[:, 1]
        ds.createVariable('k', 'i4', ('mode',))[:] = nmk_rows[:, 2]
        ds.createVariable('sigma', 'f8', ('mode',))[:] = sigma
        ds.createVariable('LNk_real', 'f8', ('mode',))[:] = np.real(lnk)
        ds.createVariable('LNk_imag', 'f8', ('mode',))[:] = np.imag(lnk)

    tides = Tides_t()
    interaction = tides.add_from_file('planet', 'satellite', str(path))

    # Same registered object `add`/`get` would return, not a detached copy.
    assert interaction is tides.get(primary='planet', perturber='satellite')
    np.testing.assert_array_equal(interaction.nmk, nmk_rows)
    np.testing.assert_allclose(interaction.sigma, sigma)
    np.testing.assert_allclose(interaction.LNk, lnk)


def test_tides_t_add_is_idempotent_for_the_same_pair():
    """Calling ``add`` twice for the same (primary, perturber) pair must
    return the SAME object both times, not create a duplicate
    interaction (``sp1d`` calls ``.get`` repeatedly assuming a single
    canonical entry per pair).
    """
    tides = Tides_t()
    first = tides.add(primary='planet', perturber='star')
    second = tides.add(primary='planet', perturber='star')
    assert first is second
    assert len(tides.interactions) == 1
    # A different perturber must create a genuinely distinct entry.
    third = tides.add(primary='planet', perturber='moon')
    assert third is not first
    assert len(tides.interactions) == 2


# ---------------------------------------------------------------------------
# run_adaptive_orbit_substeps
#
# Every production caller (sp1d, ps1d, ps1d_evec) currently passes
# needs_c_planet=True (sp0d/ps0d bypass this controller entirely -- see
# orbit.py/satellite.py's dispatch), so several of this function's own
# branches (needs_c_planet=False, a degenerate/raising C_planet refresh,
# a rejected substep's diagnostic log, the every-5000-steps progress log)
# are not reachable through any current indirect caller. Exercised here
# directly against the function's own documented contract instead.
# ---------------------------------------------------------------------------


def _make_solver_config(**overrides) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            orbit=SimpleNamespace(solver=OrbitSolver(**overrides)),
            interior_energetics=SimpleNamespace(module='aragog'),
        ),
    )


def _make_interior_o(dt: float) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            radius=np.array([0.0, 3.0e6, 6.371e6]), density=np.array([5500.0, 5000.0]), dt=dt
        ),
    )


def test_run_adaptive_orbit_substeps_skips_c_planet_refresh_when_not_needed():
    """``needs_c_planet=False`` must never call ``get_C_planet`` nor
    write ``hf_row['C_int']`` -- the branch every current production
    caller (sp1d/ps1d/ps1d_evec) skips by always passing True."""
    config = _make_solver_config(dt0_yr=1.0, dt_max_yr=10.0, max_rel_da=1.0)
    interior_o = _make_interior_o(dt=5.0)
    hf_row: dict = {'Time': 0.0, 'x': 1.0}

    def step_fn(hf_row, dt_yr, t_elapsed_yr):
        hf_row['x'] += dt_yr
        return None

    with patch('proteus.orbit.common.get_C_planet') as mock_get_c:
        run_adaptive_orbit_substeps(
            hf_row,
            config,
            Tides_t(),
            interior_o,
            'testmodel',
            step_fn,
            lambda hf_row: True,
            lambda hf_row, snapshot: {},
            {},
            needs_c_planet=False,
        )
    mock_get_c.assert_not_called()
    assert 'C_int' not in hf_row


def test_run_adaptive_orbit_substeps_logs_error_on_degenerate_c_planet_result(caplog):
    """A ``get_C_planet`` result that is zero or non-finite must be
    logged as an error and the structural update skipped, rather than
    propagating a degenerate value into ``hf_row['C_int']`` silently
    (which would divide-by-zero the very next AM-conserving rescale)."""
    config = _make_solver_config(dt0_yr=1.0, dt_max_yr=10.0)
    interior_o = _make_interior_o(dt=5.0)
    hf_row: dict = {'Time': 0.0, 'C_int': 1.0e37, 'axial_period': 86400.0}

    def fake_get_c_planet(hf_row, config, interior_o):
        hf_row['C_int'] = 0.0  # degenerate

    with (
        patch('proteus.orbit.common.get_C_planet', side_effect=fake_get_c_planet),
        caplog.at_level(logging.ERROR, logger='fwl.proteus.orbit.common'),
    ):
        run_adaptive_orbit_substeps(
            hf_row,
            config,
            Tides_t(),
            interior_o,
            'testmodel',
            lambda hf_row, dt_yr, t_elapsed_yr: None,
            lambda hf_row: True,
            lambda hf_row, snapshot: {},
            {},
            needs_c_planet=True,
        )
    assert any('structural update will be skipped' in rec.message for rec in caplog.records)
    # Discrimination: despite the degenerate target, axial_period must
    # NOT have been rescaled (the guard inside _rescale_c_planet_to
    # skips the rescale when target_c_p == 0) -- a broken guard would
    # divide by zero here instead of leaving spin untouched.
    assert hf_row['axial_period'] == pytest.approx(86400.0, rel=1e-12)


def test_run_adaptive_orbit_substeps_reraises_when_c_planet_refresh_raises(caplog):
    """An exception from ``get_C_planet`` itself must be logged and
    RE-raised, not swallowed -- a genuinely broken interior state should
    stop the run, not silently continue with a stale C_planet."""
    config = _make_solver_config(dt0_yr=1.0, dt_max_yr=10.0)
    interior_o = _make_interior_o(dt=5.0)
    hf_row: dict = {'Time': 0.0, 'C_int': 1.0e37, 'axial_period': 86400.0}

    with (
        patch('proteus.orbit.common.get_C_planet', side_effect=RuntimeError('boom')),
        caplog.at_level(logging.ERROR, logger='fwl.proteus.orbit.common'),
        pytest.raises(RuntimeError, match='boom'),
    ):
        run_adaptive_orbit_substeps(
            hf_row,
            config,
            Tides_t(),
            interior_o,
            'testmodel',
            lambda hf_row, dt_yr, t_elapsed_yr: None,
            lambda hf_row: True,
            lambda hf_row, snapshot: {},
            {},
            needs_c_planet=True,
        )
    assert any('C_planet update RAISED' in rec.message for rec in caplog.records)


def test_run_adaptive_orbit_substeps_logs_nonfinite_fields_on_rejected_substep(caplog):
    """A rejected substep (``state_is_valid_fn`` returns False) must log
    which fields were actually non-finite, not just that a rejection
    happened -- the diagnostic this controller relies on to distinguish
    a genuine solver blow-up from a spuriously tight tolerance."""
    config = _make_solver_config(dt0_yr=1.0, dt_max_yr=10.0, shrink=0.5, max_substeps=5)
    interior_o = _make_interior_o(dt=5.0)
    hf_row: dict = {'Time': 0.0, 'x': 1.0}

    def step_fn(hf_row, dt_yr, t_elapsed_yr):
        hf_row['x'] = float('nan')
        return None

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.orbit.common'):
        run_adaptive_orbit_substeps(
            hf_row,
            config,
            Tides_t(),
            interior_o,
            'testmodel',
            step_fn,
            lambda hf_row: np.isfinite(hf_row['x']),
            lambda hf_row, snapshot: {},
            {},
            needs_c_planet=False,
        )
    reject_records = [
        rec.message for rec in caplog.records if 'state_is_valid_fn rejected' in rec.message
    ]
    assert len(reject_records) > 0
    # Discrimination: the logged diagnostic must actually name the bad
    # field, not just report a generic rejection.
    assert any("'x'" in msg for msg in reject_records)


def test_run_adaptive_orbit_substeps_logs_progress_every_5000_accepted_steps(caplog):
    """Every 5000th ACCEPTED substep must emit a progress log line --
    otherwise a long-running call (millions of substeps for a gentle
    ODE forced through a small dt_max_yr) gives no feedback at all
    until it either completes or exhausts max_substeps.

    dt_yr is held exactly constant across every substep by having
    ``rel_change_fn`` return a value pinned at exactly the no-growth
    threshold (0.3 * limit): the growth condition is strict (`<`), so
    equality never grows dt_yr. With dt0_yr=dt_max_yr=1.0 yr and
    t_total_yr=5000 yr, this takes EXACTLY 5000 accepted substeps to
    complete, hitting the modulo check exactly once, cheaply (no real
    physics in step_fn).
    """
    config = _make_solver_config(
        dt0_yr=1.0, dt_max_yr=1.0, growth=1.5, shrink=0.5, max_rel_da=1.0, max_substeps=6000
    )
    interior_o = _make_interior_o(dt=5000.0)
    hf_row: dict = {'Time': 0.0, 'x': 0.0}

    def step_fn(hf_row, dt_yr, t_elapsed_yr):
        hf_row['x'] += dt_yr
        return None

    def rel_change_fn(hf_row, snapshot):
        return {'dx': 0.3}  # exactly the no-growth threshold at limit=1.0

    with caplog.at_level(logging.DEBUG, logger='fwl.proteus.orbit.common'):
        run_adaptive_orbit_substeps(
            hf_row,
            config,
            Tides_t(),
            interior_o,
            'testmodel',
            step_fn,
            lambda hf_row: True,
            rel_change_fn,
            {'dx': 1.0},
            needs_c_planet=False,
        )
    assert hf_row['x'] == pytest.approx(5000.0, rel=1e-9)
    progress_records = [
        rec.message for rec in caplog.records if 'progress t_elapsed=' in rec.message
    ]
    assert len(progress_records) > 0
    assert any('n_steps=5000' in msg for msg in progress_records)
