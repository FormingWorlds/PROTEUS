"""A Zalmoxis structure that breaks an invariant stops the run with status 33."""

from __future__ import annotations

import logging

import pytest
from zalmoxis.invariants import StructureInvariantError

from proteus.interior_energetics.wrapper import _solve_structure_with_adiabat_or_rollback
from tests.interior_struct.test_zalmoxis import _run_gate_solver

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.parametrize(
    'error, status',
    [
        (StructureInvariantError('breaks cmb_mass: deviation 1.0e+00'), '33'),
        (RuntimeError('not converged'), None),
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


def test_an_invariant_error_passes_the_rollback_of_the_adiabat_resolve(tmp_path, monkeypatch):
    """The adiabat re-solve rolls back a RuntimeError, ValueError or FloatingPointError; an
    invariant error is none of these, so it reaches the caller and status 33 stays."""
    error = StructureInvariantError('breaks cmb_mass: deviation 1.0e+00')

    def resolve(config, outdir, hf_row, tf):
        return _solve_structure_with_adiabat_or_rollback(
            config, outdir, hf_row, 0, lambda r, P: 3000.0
        )

    with pytest.raises(StructureInvariantError) as raised:
        _run_gate_solver(
            tmp_path, monkeypatch, 'Seager2007:silicate', None, None, [error], solve=resolve
        )
    assert raised.value is error
    lines = (tmp_path / 'status').read_text().splitlines()
    assert lines == ['33', 'Error (Interior structure invariant)']
