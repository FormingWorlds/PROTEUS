"""A Zalmoxis structure that breaks an invariant stops the run with status 33.

These tests check an error contract (the exception that reaches each caller and the status
file), not a physical bound, so they carry no physics_invariant marker.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from proteus.interior_energetics.wrapper import (
    _solve_structure_with_adiabat_or_rollback,
    update_structure_from_interior,
)
from proteus.interior_struct.zalmoxis import StructureInvariantError
from tests.interior_energetics.test_wrapper_adiabat_ic import (
    _monotonic_config,
    _monotonic_dirs,
    _monotonic_hf_row,
    _monotonic_interior_o,
)
from tests.interior_struct.test_zalmoxis import _run_gate_solver

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _invariant_error():
    return StructureInvariantError('breaks cmb_mass: deviation 1.0e+00')


@pytest.mark.parametrize(
    'error, status',
    [
        pytest.param(_invariant_error(), '33', id='invariant_error_writes_33'),
        pytest.param(RuntimeError('not converged'), None, id='non_converged_writes_no_status'),
    ],
)
def test_only_an_invariant_error_writes_status_33(tmp_path, monkeypatch, caplog, error, status):
    """Zalmoxis main raises inside the real zalmoxis_solver path."""
    with caplog.at_level(logging.ERROR), pytest.raises(type(error)) as raised:
        _run_gate_solver(tmp_path, monkeypatch, 'Seager2007:silicate', None, None, [error])
    assert raised.value is error
    if status is None:
        assert not (tmp_path / 'status').exists()
        assert 'breaks an invariant' not in caplog.text
    else:
        lines = (tmp_path / 'status').read_text().splitlines()
        assert lines == [status, 'Error (Interior structure invariant)']
        assert 'breaks an invariant' in caplog.text and 'breaks cmb_mass' in caplog.text


def _resolve(config, outdir, hf_row):
    return _solve_structure_with_adiabat_or_rollback(
        config, outdir, hf_row, 0, lambda r, P: 3000.0
    )


@pytest.mark.parametrize(
    'error',
    [
        pytest.param(RuntimeError('not converged'), id='runtime_error_rolls_back'),
        pytest.param(ValueError('bad table'), id='value_error_rolls_back'),
        pytest.param(FloatingPointError('overflow'), id='floating_point_error_rolls_back'),
    ],
)
def test_the_adiabat_resolve_rolls_back_a_solver_failure(tmp_path, monkeypatch, caplog, error):
    """The rollback returns (None, False), logs the injected error and writes no status."""
    with caplog.at_level(logging.WARNING):
        result = _run_gate_solver(
            tmp_path, monkeypatch, 'Seager2007:silicate', None, None, [error], solve=_resolve
        )
    assert result == (None, False)
    assert f're-solve failed ({error})' in caplog.text
    assert not (tmp_path / 'status').exists()


def test_an_invariant_error_passes_the_rollback_of_the_adiabat_resolve(tmp_path, monkeypatch):
    """The adiabat re-solve rolls back a RuntimeError, ValueError or FloatingPointError; an
    invariant error is none of these, so it reaches the caller and status 33 stays."""
    error = _invariant_error()
    with pytest.raises(StructureInvariantError) as raised:
        _run_gate_solver(
            tmp_path, monkeypatch, 'Seager2007:silicate', None, None, [error], solve=_resolve
        )
    assert raised.value is error
    lines = (tmp_path / 'status').read_text().splitlines()
    assert lines == ['33', 'Error (Interior structure invariant)']


@pytest.mark.parametrize(
    'error, fails',
    [
        pytest.param(_invariant_error(), 0, id='invariant_error_stops_the_evolution'),
        pytest.param(RuntimeError('not converged'), 1, id='runtime_error_keeps_the_structure'),
    ],
)
def test_the_evolution_resolve_lets_only_an_invariant_error_through(tmp_path, error, fails):
    """The time-step re-solve counts a RuntimeError and keeps the previous structure; an
    invariant error reaches the caller and does not count as a convergence failure."""
    config = _monotonic_config(module='aragog', temperature_mode='liquidus_super')
    interior_o = _monotonic_interior_o()
    dirs, _ = _monotonic_dirs(tmp_path, 'PREV GEOMETRY\n')
    hf_row = _monotonic_hf_row(1.2007e7, M_target=5.972e24)
    before = dict(hf_row)

    def solver(cfg, outdir, row, **kwargs):
        row.update(R_int=9.0e6, P_cmb=1.0e11, gravity=1.0)  # a partial write, then the error
        raise error

    with patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=solver):
        if fails:
            update_structure_from_interior(dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8)
            assert {k: hf_row[k] for k in ('R_int', 'P_cmb', 'gravity')} == {
                k: before[k] for k in ('R_int', 'P_cmb', 'gravity')
            }
            assert interior_o.structure_stale is True
        else:
            with pytest.raises(StructureInvariantError) as raised:
                update_structure_from_interior(
                    dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
                )
            assert raised.value is error
    assert interior_o.zalmoxis_fail_count == fails


def test_a_status_file_that_cannot_be_written_does_not_hide_the_invariant_error(
    tmp_path, monkeypatch, caplog
):
    """A failed status write is logged; the invariant error still reaches the caller."""
    error, seen = _invariant_error(), None
    with (
        patch(
            'proteus.interior_struct.zalmoxis.UpdateStatusfile',
            side_effect=OSError('read-only output'),
        ) as status,
        caplog.at_level(logging.ERROR),
    ):
        try:
            _run_gate_solver(tmp_path, monkeypatch, 'Seager2007:silicate', None, None, [error])
        except Exception as exc:  # noqa: BLE001 -- which error escapes is the assertion
            seen = exc
    assert seen is error, repr(seen)
    status.assert_called_once_with({'output': str(tmp_path)}, 33)
    assert 'read-only output' in caplog.text
