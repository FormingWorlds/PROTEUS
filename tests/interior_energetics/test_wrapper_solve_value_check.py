"""A Zalmoxis structure solve that leaves R_int or M_int not finite and positive is a failed solve.

Covers the start-up solves of ``determine_interior_radius_with_zalmoxis``, the adiabat re-solve
with rollback shared by the IC and the equilibration loop, the linear-guess solves of
``equilibrate_initial_state``, and the committed-mass warning on non-finite values.
"""

from __future__ import annotations

import logging
import math
from unittest.mock import MagicMock, patch

import pytest

from proteus.interior_energetics import wrapper as wmod
from proteus.interior_energetics.wrapper import _warn_committed_mass_miss
from tests.interior_energetics.test_wrapper_adiabat_ic import (
    _config,
    _equilibrate_config,
    _gate_config,
    _good_adiabat,
    _linear_hf_row,
    _patch_equilibrate_io,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.timeout(30),
    pytest.mark.usefixtures('_no_core_table'),
]

BAD_SOLVE_MSG = 'non-finite or non-positive R_int or M_int'
MISS_MSG = 'Committed structure misses its mass target'
R_OK, M_OK = 1.2007e7, _linear_hf_row()['M_int_target']

BAD_PAIRS = [
    pytest.param(R_OK, math.nan, id='NaN mass'),
    pytest.param(R_OK, math.inf, id='inf mass'),
    pytest.param(R_OK, 0.0, id='zero mass'),
    pytest.param(R_OK, -M_OK, id='negative mass'),
    pytest.param(math.nan, M_OK, id='NaN radius'),
    pytest.param(math.inf, M_OK, id='inf radius'),
    pytest.param(0.0, M_OK, id='zero radius'),
]


def _writing_solver(R_int, M_int):
    """Stub of zalmoxis_solver that writes the given R_int and M_int into the row."""

    def solve(cfg, outdir, row, **kwargs):
        row['R_int'], row['M_int'], row['P_surf'] = R_int, M_int, 1.0e5
        return 6.0e6, None

    return solve


def _run_first_solve(R_int, M_int, hf_row):
    """Run the first structure solve of the start-up with a stub solver that writes R_int, M_int."""
    with (
        patch.object(wmod, 'get_nlevb', return_value=80),
        patch.object(wmod, 'Interior_t', return_value=MagicMock(ic=1)),
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_writing_solver(R_int, M_int),
        ),
        patch('proteus.interior_struct.zalmoxis.generate_spider_tables', return_value=None),
        patch.object(wmod, '_provide_spider_eos_tables'),
        patch.object(wmod, 'run_interior'),
    ):
        wmod.determine_interior_radius_with_zalmoxis(
            {'output': '/tmp/out'},
            _gate_config('aragog', 'isothermal'),
            MagicMock(),
            hf_row,
            '/tmp/out',
        )


def _run_linear_equilibration(R_int, M_int, hf_row):
    """Run one linear-guess equilibration iteration with a stub solver that writes R_int, M_int."""
    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_writing_solver(R_int, M_int),
        ),
        _patch_equilibrate_io(),
    ):
        wmod.equilibrate_initial_state(
            {'output': '/tmp/out'},
            _equilibrate_config(temperature_mode='isothermal', max_iter=1),
            hf_row,
            '/tmp/out',
        )


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('R_int', 'M_int'), BAD_PAIRS)
def test_adiabat_re_solve_with_a_bad_value_rolls_back(tmp_path, R_int, M_int):
    """The adiabat re-solve refuses the structure and restores the row and the output file."""
    hf_row = _linear_hf_row()
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    out_path = data_dir / 'zalmoxis_output.dat'
    out_path.write_text('LINEAR\n')
    linear_mesh = str(tmp_path / 'linear_mesh.dat')
    solve = _writing_solver(R_int, M_int)

    def bad_solver(cfg, outdir, row, **kwargs):
        out_path.write_text('ADIABAT\n')
        return solve(cfg, outdir, row, **kwargs)

    adiabat = _good_adiabat()
    with (
        patch(
            'proteus.interior_energetics.wrapper._build_superliquidus_adiabat_tp',
            return_value=(lambda r, P: 5000.0, adiabat['P'], adiabat['T']),
        ),
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=bad_solver),
    ):
        mesh = wmod._resolve_adiabatic_ic_structure(
            _config(), str(tmp_path), hf_row, num_spider_nodes=0, linear_mesh_file=linear_mesh
        )

    assert mesh == linear_mesh
    assert hf_row == _linear_hf_row()
    assert out_path.read_text() == 'LINEAR\n'


@pytest.mark.physics_invariant
@pytest.mark.parametrize('target', [0.0, math.nan], ids=['zero target', 'NaN target'])
@pytest.mark.parametrize(
    'M_int',
    [math.nan, math.inf, 0.0, -M_OK],
    ids=['NaN mass', 'inf mass', 'zero mass', 'negative mass'],
)
def test_adiabat_re_solve_without_a_target_refuses_a_bad_mass(tmp_path, caplog, target, M_int):
    """With no usable mass target the anchor is skipped, so the finite check alone refuses the mass."""
    hf_row = {**_linear_hf_row(), 'M_int_target': target}
    saved = dict(hf_row)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'zalmoxis_output.dat').write_text('LINEAR\n')
    adiabat = _good_adiabat()
    with (
        patch.object(
            wmod,
            '_build_superliquidus_adiabat_tp',
            return_value=(lambda r, P: 5000.0, adiabat['P'], adiabat['T']),
        ),
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_writing_solver(R_OK, M_int),
        ),
        caplog.at_level(logging.WARNING),
    ):
        mesh, ok = wmod._solve_structure_with_adiabat_or_rollback(
            _config(), str(tmp_path), hf_row, 0, lambda r, P: 5000.0
        )

    assert (mesh, ok) == (None, False)
    assert hf_row.keys() == saved.keys()
    assert all(
        hf_row[k] == saved[k] or (hf_row[k] != hf_row[k] and saved[k] != saved[k])
        for k in saved
    )
    failed = [r.getMessage() for r in caplog.records if 're-solve failed' in r.getMessage()]
    assert len(failed) == 1 and BAD_SOLVE_MSG in failed[0] and 'mass-anchor' not in failed[0]


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('R_int', 'M_int'), BAD_PAIRS)
def test_first_structure_solve_with_a_bad_value_raises(R_int, M_int):
    """The first solve has no fall-back, so a bad structure stops the start-up."""
    with pytest.raises(RuntimeError, match=BAD_SOLVE_MSG) as err:
        _run_first_solve(R_int, M_int, {'P_cmb': 1.456e12})
    assert 'start-up or post-impact structure solve' in str(err.value)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('R_int', 'M_int'), BAD_PAIRS)
def test_equilibration_linear_solve_with_a_bad_value_raises(R_int, M_int):
    """The linear-guess solve of the equilibration loop has no fall-back either."""
    with pytest.raises(RuntimeError, match=BAD_SOLVE_MSG) as err:
        _run_linear_equilibration(R_int, M_int, _linear_hf_row())
    assert 'equilibration' in str(err.value)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('R_int', 'M_int'), BAD_PAIRS)
def test_equilibration_fall_back_solve_with_a_bad_value_raises(tmp_path, R_int, M_int):
    """After a refused adiabat re-solve, a bad linear fall-back solve stops the loop."""
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'zalmoxis_output.dat').write_text('LINEAR\n')
    calls = []
    bad = _writing_solver(R_int, M_int)

    def solver(cfg, outdir, row, temperature_function=None, **kwargs):
        calls.append(temperature_function)
        if temperature_function is not None:  # the adiabat attempt misses the mass anchor
            row['M_int'] = M_OK * 1.10
            return 6.0e6, None
        return bad(cfg, outdir, row)

    adiabat = _good_adiabat()
    with (
        patch.object(
            wmod,
            '_build_superliquidus_adiabat_tp',
            return_value=(lambda r, P: 5000.0, adiabat['P'], adiabat['T']),
        ),
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=solver),
        _patch_equilibrate_io(),
        pytest.raises(RuntimeError, match=BAD_SOLVE_MSG),
    ):
        wmod.equilibrate_initial_state(
            {'output': str(tmp_path)},
            _equilibrate_config(max_iter=1),
            _linear_hf_row(),
            str(tmp_path),
        )
    assert len(calls) == 2 and callable(calls[0]) and calls[1] is None


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'row',
    [
        pytest.param({}, id='empty row'),
        pytest.param({'R_int': R_OK}, id='no mass'),
        pytest.param({'M_int': M_OK}, id='no radius'),
    ],
)
def test_an_absent_radius_or_mass_is_a_failed_solve(row):
    """A solve that writes no R_int or no M_int is refused like a zero value."""
    with pytest.raises(RuntimeError, match=BAD_SOLVE_MSG) as err:
        wmod._require_finite_structure(row, 'test solve')
    assert 'R_int=' in str(err.value) and 'M_int=' in str(err.value)


@pytest.mark.physics_invariant
def test_good_values_pass_every_start_up_solve(tmp_path):
    """Finite, positive values pass the first solve and the equilibration solve unchanged."""
    hf_row = {'P_cmb': 1.456e12}
    _run_first_solve(R_OK, M_OK, hf_row)
    assert (hf_row['R_int'], hf_row['M_int']) == (R_OK, M_OK)

    row = _linear_hf_row()
    _run_linear_equilibration(R_OK, M_OK, row)
    assert (row['R_int'], row['M_int']) == (R_OK, M_OK)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('m_int', 'm_target', 'warns'),
    [
        pytest.param(math.nan, M_OK, True, id='NaN mass'),
        pytest.param(math.inf, M_OK, True, id='inf mass'),
        pytest.param(M_OK, math.nan, True, id='NaN target'),
        pytest.param(M_OK, math.inf, True, id='inf target'),
        pytest.param(0.0, M_OK, False, id='zero mass'),
        pytest.param(math.nan, 0.0, False, id='zero target'),
    ],
)
def test_committed_mass_warning_on_non_finite_values(caplog, m_int, m_target, warns):
    """A NaN or inf mass or target with both keys set warns once; a zero key returns silent."""
    with caplog.at_level(logging.WARNING):
        _warn_committed_mass_miss({'M_int': m_int, 'M_int_target': m_target})
    hits = [r for r in caplog.records if MISS_MSG in r.getMessage()]
    assert len(hits) == (1 if warns else 0)
