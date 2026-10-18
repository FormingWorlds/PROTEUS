"""A Zalmoxis structure that breaks an invariant stops the run with status 33."""

from __future__ import annotations

import logging

import pytest
from zalmoxis.invariants import StructureInvariantError

import proteus.interior_struct.zalmoxis as pz

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.parametrize(
    'error, status',
    [
        (StructureInvariantError('breaks cmb_mass: deviation 1.0e+00'), '33'),
        (RuntimeError('not converged'), None),
    ],
)
def test_only_an_invariant_error_writes_status_33(tmp_path, monkeypatch, caplog, error, status):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(pz, 'load_zalmoxis_configuration', fail)
    with caplog.at_level(logging.ERROR), pytest.raises(type(error)) as raised:
        pz.zalmoxis_solver(None, str(tmp_path), {})
    assert raised.value is error
    if status is None:
        assert not (tmp_path / 'status').exists()
        assert 'breaks an invariant' not in caplog.text
    else:
        lines = (tmp_path / 'status').read_text().splitlines()
        assert lines == [status, 'Error (Interior structure invariant)']
        assert 'breaks an invariant' in caplog.text and 'breaks cmb_mass' in caplog.text
