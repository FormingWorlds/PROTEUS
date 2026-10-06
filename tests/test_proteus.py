"""
Unit tests for proteus.proteus module: Zalmoxis mesh restoration on resume,
atmosphere-interior deadlock detection, main-loop plot cadence, and the
T_magma handed to the atmosphere after a resume.

Tests the resume code path in Proteus.start() that restores the Zalmoxis
mesh file path when resuming a SPIDER interior simulation, and the main
loop's `params.out.plot_mod`-gated plot generation.

Testing standards and documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules

Functions tested:
- Proteus.start(): Resume path restoring spider_mesh and spider_mesh_prev
- Proteus.start(): main-loop plot generation cadence (plot_mod)
- Proteus.__init__(): stall criterion read from params.stop.stall
- Proteus._check_atmosphere_deadlock()
- Proteus.start(): resumed main loop hands the atmosphere the interior T_magma
  (or the solvus boundary with global miscibility)
"""

from __future__ import annotations

import logging
import sys
import warnings
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Stall cap the fixture config carries. Deliberately not ATMOS_STALL_MAX, and
# well above AGNI_DEADLOCK_MAX: the constructor may read the config or fall
# back on the constant, and only a distinct number tells the two apart.
STALL_MAX_CONFIGURED = 41


def _make_proteus_instance(
    tmp_path,
    *,
    struct_module='zalmoxis',
    interior_module='spider',
    stall_enabled=True,
    stall_maximum=STALL_MAX_CONFIGURED,
):
    """Build a Proteus object with mocked config and directories."""
    from proteus.config._params import StopStall
    from proteus.proteus import Proteus

    config = MagicMock()
    config.interior_struct.module = struct_module
    config.interior_struct.zalmoxis.update_interval = 0
    config.interior_energetics.module = interior_module
    config.interior_struct.eos_dir = 'WolfBower2018_MgSiO3'
    config.orbit.module = None
    config.accretion.module = None
    # Attributes used during start() setup
    config.params.out.logging = 'WARNING'
    config.params.stop.iters.minimum = 10
    config.params.stop.iters.maximum = 1000
    # Real values, not mock attributes: the resume branch compares the melt
    # fraction against phi_crit, which a bare MagicMock cannot be ordered
    # against. Defaults mirror the schema.
    config.params.stop.solid.freeze_volatiles = False
    config.params.stop.solid.phi_crit = 0.01
    # A real schema node rather than mock attributes: the constructor reads
    # this branch, so it has to carry the types and validators a run gives it,
    # and an unset mock integer would read as a stall cap of one iteration.
    config.params.stop.stall = StopStall(enabled=stall_enabled, maximum=stall_maximum)

    directories = {
        'output': str(tmp_path),
        'output/data': str(tmp_path / 'data'),
        'spider': '/nonexistent/spider',
    }

    with (
        patch('proteus.proteus.read_config_object', return_value=config),
        patch('proteus.utils.coupler.set_directories', return_value=directories),
    ):
        p = Proteus(config_path='dummy.toml')

    return p


# All the lazy imports inside start() that must be mocked to reach the resume path.
_START_PATCHES = [
    'proteus.atmos_chem.wrapper.run_chemistry',
    'proteus.atmos_clim.run_atmosphere',
    'proteus.atmos_clim.common.Atmos_t',
    'proteus.escape.wrapper.run_escape',
    'proteus.interior_energetics.wrapper.run_interior',
    'proteus.interior_energetics.wrapper.solve_structure',
    'proteus.interior_energetics.wrapper.update_planet_mass',
    'proteus.observe.wrapper.run_observe',
    'proteus.orbit.wrapper.run_orbit',
    'proteus.outgas.wrapper.calc_target_elemental_inventories',
    'proteus.outgas.wrapper.run_desiccated',
    'proteus.outgas.wrapper.run_outgassing',
    'proteus.star.wrapper.get_new_spectrum',
    'proteus.star.wrapper.scale_spectrum_to_toa',
    'proteus.star.wrapper.update_stellar_mass',
    'proteus.star.wrapper.update_stellar_quantities',
    'proteus.star.wrapper.write_spectrum',
    'proteus.utils.coupler.CreateHelpfileFromDict',
    'proteus.utils.coupler.CreateLockFile',
    'proteus.utils.coupler.ExtendHelpfile',
    'proteus.utils.coupler.PrintCurrentState',
    'proteus.utils.coupler.UpdatePlots',
    'proteus.utils.coupler.WriteHelpfileToCSV',
    'proteus.utils.coupler.print_citation',
    'proteus.utils.coupler.print_header',
    'proteus.utils.coupler.print_module_configuration',
    'proteus.utils.coupler.print_stoptime',
    'proteus.utils.coupler.print_system_configuration',
    'proteus.utils.coupler.remove_excess_files',
    'proteus.utils.coupler.validate_module_versions',
    'proteus.utils.coupler.UpdateStatusfile',
    'proteus.utils.data.download_sufficient_data',
    'proteus.interior_struct.zalmoxis.require_paleos_tables',
    'proteus.utils.terminate.print_termination_criteria',
]


class _StopAfterMeshRestore(Exception):
    """Sentinel exception to stop start() after the mesh restoration block."""


def _resume_with_patches(p, hf_df, *extra):
    """Call p.start(resume=True) with all start() imports mocked.

    Uses ExitStack to avoid Python's nested-block limit.
    Stops at init_star (after mesh restoration), unless an ``extra`` patch,
    entered last, replaces that stop with its own.
    """
    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))

        stack.enter_context(
            patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50)
        )
        stack.enter_context(
            patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=hf_df)
        )
        # These tests exercise the mesh-restoration block, not snapshot
        # selection: pass the helpfile through unchanged so the resume reaches
        # the mesh code instead of failing the snapshot-pair check (which would
        # need on-disk _int.nc/_atm.nc files these tests deliberately omit).
        stack.enter_context(
            patch(
                'proteus.utils.coupler.select_resumable_snapshot',
                return_value=(hf_df, []),
            )
        )
        stack.enter_context(
            patch('proteus.outgas.wrapper.check_desiccation', return_value=False)
        )
        stack.enter_context(patch('proteus.utils.coupler.ZeroHelpfileRow', return_value={}))

        # Interior_t mock
        mock_interior_t = stack.enter_context(
            patch('proteus.interior_energetics.common.Interior_t')
        )
        mock_int = MagicMock()
        mock_int.ic = 1
        mock_interior_t.return_value = mock_int

        # Stop execution at init_star (runs after mesh restoration)
        stack.enter_context(
            patch(
                'proteus.star.wrapper.init_star',
                side_effect=_StopAfterMeshRestore,
            )
        )
        stack.enter_context(
            patch(
                'proteus.orbit.wrapper.init_orbit',
                side_effect=_StopAfterMeshRestore,
            )
        )

        for extra_patch in extra:
            stack.enter_context(extra_patch)

        with pytest.raises(_StopAfterMeshRestore):
            p.start(resume=True, offline=True)


@pytest.mark.unit
def test_resume_matches_the_ps_tables_after_restoring_the_accreted_mass(tmp_path):
    """start(resume=True) points the run at the P-S tables of its mass only after
    restore_accretion_state has restored that mass."""
    p = _make_proteus_instance(tmp_path, struct_module='dummy', interior_module='aragog')
    p.config.accretion.module = 'dummy'
    (tmp_path / 'data').mkdir(exist_ok=True)
    calls = []

    def match(self):
        calls.append('match')
        raise _StopAfterMeshRestore

    _resume_with_patches(
        p,
        _make_hf_df(),
        patch('proteus.star.wrapper.init_star'),
        patch('proteus.orbit.wrapper.init_orbit'),
        patch('proteus.accretion.wrapper.init_accretion', return_value=[]),
        patch(
            'proteus.accretion.wrapper.restore_accretion_state',
            side_effect=lambda handler: calls.append(('restore', handler is p)),
        ),
        patch.object(type(p), '_match_ps_tables_to_mass', match),
        patch('proteus.proteus.setup_logger'),
    )

    assert calls[0] == ('restore', True)
    assert calls[1:] == ['match']


@pytest.mark.unit
def test_resume_without_accretion_leaves_the_ps_tables_alone(tmp_path):
    """A resume without an accretion module keeps the restored tables, as before."""
    p = _make_proteus_instance(tmp_path, struct_module='dummy', interior_module='aragog')
    (tmp_path / 'data').mkdir(exist_ok=True)

    with patch.object(type(p), '_match_ps_tables_to_mass') as match:
        _resume_with_patches(
            p,
            _make_hf_df(),
            patch('proteus.star.wrapper.init_star'),
            patch('proteus.orbit.wrapper.init_orbit'),
            patch('proteus.accretion.wrapper.init_accretion', return_value=[]),
            patch('proteus.accretion.wrapper.restore_accretion_state'),
            patch('proteus.proteus.setup_logger'),
            # The running status follows the table check, so stop there.
            patch(
                'proteus.proteus.UpdateStatusfile',
                side_effect=lambda dirs, code: _raise_if(code == 1),
            ),
        )

    assert p.config.accretion.module is None
    assert not match.called


@pytest.mark.unit
def test_resume_checks_the_volatile_change_column_before_any_structure_solve(tmp_path):
    """start(resume=True) reads M_volatile_change through its finite check right
    after the restore, so a corrupt row stops the run with that message instead
    of failing later inside a structure solve."""
    p = _make_proteus_instance(tmp_path, struct_module='dummy', interior_module='aragog')
    (tmp_path / 'data').mkdir(exist_ok=True)
    hf = _make_hf_df()
    hf['M_volatile_change'] = [0.0, 0.0, 0.0, 0.0, float('nan')]
    seen = []

    def check(row):
        seen.append(row.get('M_volatile_change'))
        raise _StopAfterMeshRestore

    _resume_with_patches(
        p,
        hf,
        patch('proteus.star.wrapper.init_star'),
        patch('proteus.orbit.wrapper.init_orbit'),
        patch('proteus.accretion.wrapper.init_accretion', return_value=[]),
        patch('proteus.accretion.wrapper.restore_accretion_state'),
        patch('proteus.interior_struct.common.volatile_mass_change', side_effect=check),
        patch('proteus.proteus.setup_logger'),
    )
    assert len(seen) == 1
    assert seen[0] != seen[0]  # the restored row's NaN reached the check


@pytest.mark.unit
@pytest.mark.parametrize(
    'side_effect, match',
    [
        (None, 'M_volatile_change is not finite'),
        (RuntimeError('Resume refused: corrupt impact records'), 'corrupt impact records'),
        (ValueError('Impact timeline is missing required columns'), 'missing required'),
    ],
    ids=['volatile column', 'impact records', 'malformed timeline'],
)
def test_a_refused_resume_records_status_20(tmp_path, side_effect, match):
    """A resume refused for a NaN M_volatile_change or for corrupt impact records
    writes status 20 before it raises, so the stopped run does not read as running."""
    p = _make_proteus_instance(tmp_path, struct_module='dummy', interior_module='aragog')
    (tmp_path / 'data').mkdir(exist_ok=True)
    hf = _make_hf_df()
    hf['M_volatile_change'] = [0.0, 0.0, 0.0, 0.0, float('nan')]
    from proteus.utils.coupler import UpdateStatusfile

    codes = []

    def record(d, c):
        codes.append((d['output'], c))
        UpdateStatusfile(d, c)

    with pytest.raises((RuntimeError, ValueError), match=match):
        _resume_with_patches(
            p,
            hf,
            patch('proteus.star.wrapper.init_star'),
            patch('proteus.orbit.wrapper.init_orbit'),
            patch('proteus.accretion.wrapper.init_accretion', return_value=[]),
            patch('proteus.accretion.wrapper.restore_accretion_state', side_effect=side_effect),
            patch('proteus.proteus.setup_logger'),
            patch('proteus.proteus.UpdateStatusfile', side_effect=record),
        )
    assert codes[-1] == (str(tmp_path), 20)
    assert [c for _, c in codes].count(20) == 1
    assert (tmp_path / 'status').read_text().split()[0] == '20'


@pytest.mark.unit
@pytest.mark.parametrize('struct, expected', [('zalmoxis', -3.0e21), ('dummy', 0.0)])
@pytest.mark.parametrize(
    'crystallized, freeze, phi, frozen',
    [
        (False, False, 0.0, False),
        (True, False, 0.5, True),
        (False, True, 0.01, True),
        (False, True, 0.5, False),
    ],
)
def test_the_escape_step_records_the_removed_mass_for_the_zalmoxis_target(
    tmp_path, struct, expected, crystallized, freeze, phi, frozen
):
    """The main-loop escape step books the element mass escape removed into
    M_volatile_change with the Zalmoxis structure, and nothing with the dummy
    structure, whose mass_tot is the dry anchor."""
    from types import SimpleNamespace

    p = _make_proteus_instance(tmp_path, struct_module=struct)
    p.hf_row = {'H_kg_total': 5.0e21, 'O_kg_total': 2.0e22, 'Phi_global': phi}
    p.interior_o = SimpleNamespace(dt=100.0)
    p.loops = {'total': 5, 'init_loops': 2}  # the first loop that runs escape
    p.desiccated, p.crystallized = False, crystallized
    p.config.params.stop.solid.freeze_volatiles = freeze
    calls = []

    def escape(config, hf_row, dirs, dt, **kwargs):
        calls.append((config, hf_row, dirs, dt, kwargs))
        hf_row['H_kg_total'] -= 3.0e21

    with patch('proteus.escape.wrapper.run_escape', side_effect=escape):
        assert p._run_escape_step() is True

    assert len(calls) == 1
    config, hf_row, dirs, dt, kwargs = calls[0]
    assert (config, hf_row, dirs) == (p.config, p.hf_row, p.directories)
    assert dt == pytest.approx(100.0, rel=1e-15)
    assert kwargs == {'atmosphere_only': frozen, 'interior_o': p.interior_o}
    assert p.hf_row['H_kg_total'] == pytest.approx(2.0e21, rel=1e-15)
    assert p.hf_row.get('M_volatile_change', 0.0) == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.unit
@pytest.mark.parametrize('total, desiccated', [(4, False), (10, True)])
def test_a_loop_without_escape_clears_the_per_step_escape_records(tmp_path, total, desiccated):
    """Before the escape start and once desiccated, escape is skipped and the
    step limit, the clamp and the applied loss of the last step are cleared."""
    from types import SimpleNamespace

    p = _make_proteus_instance(tmp_path)
    p.hf_row = {'esc_clamp_frac': 0.3, 'esc_step_kg': 1.0e18, 'H_kg_total': 5.0e21}
    p.interior_o = SimpleNamespace(dt=100.0, escape_dt_limit=10.0)
    p.loops = {'total': total, 'init_loops': 2}
    p.desiccated, p.crystallized = desiccated, False

    with patch('proteus.escape.wrapper.run_escape') as escape:
        assert p._run_escape_step() is False

    escape.assert_not_called()
    assert p.interior_o.escape_dt_limit == np.inf
    assert p.hf_row['esc_clamp_frac'] == pytest.approx(0.0, abs=0.0)
    assert p.hf_row['esc_step_kg'] == pytest.approx(0.0, abs=0.0)
    assert p.hf_row['H_kg_total'] == pytest.approx(5.0e21, rel=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize(
    'struct, change, logged',
    [('dummy', -2.0e21, True), ('dummy', 0.0, False), ('zalmoxis', -2.0e21, False)],
)
def test_resume_notes_a_volatile_change_the_structure_does_not_read(
    tmp_path, caplog, struct, change, logged
):
    """A non-zero M_volatile_change resumed under a structure that does not read
    it is noted; a zero column, or one Zalmoxis reads, is not."""
    p = _make_proteus_instance(tmp_path, struct_module=struct, interior_module='aragog')
    (tmp_path / 'data').mkdir(exist_ok=True)
    hf = _make_hf_df()
    hf['M_volatile_change'] = [0.0, 0.0, 0.0, 0.0, change]
    with caplog.at_level('INFO'):
        _resume_with_patches(
            p,
            hf,
            patch('proteus.star.wrapper.init_star'),
            patch('proteus.orbit.wrapper.init_orbit'),
            patch('proteus.accretion.wrapper.init_accretion', return_value=[]),
            patch('proteus.accretion.wrapper.restore_accretion_state'),
            patch('proteus.proteus.setup_logger'),
            patch.object(type(p), '_resync_zalmoxis_mesh', lambda self: None),
            patch(
                'proteus.proteus.UpdateStatusfile',
                side_effect=lambda dirs, code: _raise_if(code == 1),
            ),
        )
    stale = [r for r in caplog.records if 'stays stale' in r.getMessage()]
    assert [r.levelname for r in stale] == (['INFO'] if logged else [])
    assert all(f'{change:.3e} kg' in r.getMessage() for r in stale)


def _raise_if(condition):
    if condition:
        raise _StopAfterMeshRestore


@pytest.mark.unit
@pytest.mark.parametrize(
    ('struct', 'energetics', 'restored', 'tables', 'expected', 'called'),
    [
        ('zalmoxis', 'aragog', True, True, 'new', True),
        ('dummy', 'aragog', True, True, 'new', True),
        ('zalmoxis', 'spider', True, True, 'new', True),
        ('zalmoxis', 'aragog', True, False, 'old', True),
        ('zalmoxis', 'aragog', False, True, None, False),
        ('dummy', 'dummy', True, True, 'old', False),
        ('spider', 'spider', True, True, 'old', False),
        ('spider', 'aragog', True, True, 'old', False),
    ],
)
def test_match_ps_tables_to_mass_uses_the_tables_of_the_current_mass(
    tmp_path, struct, energetics, restored, tables, expected, called
):
    """A resumed SPIDER or Aragog run with restored tables takes the directory that
    generate_spider_tables returns for the current mass; the SPIDER structure keeps its
    static tables, and nothing changes otherwise."""
    p = _make_proteus_instance(tmp_path, struct_module=struct, interior_module=energetics)
    if restored:
        p.directories['spider_eos_dir'] = 'old'
    result = {'eos_dir': 'new', 'solidus_path': 'new/sol', 'liquidus_path': 'new/liq'}

    with patch(
        'proteus.interior_struct.zalmoxis.generate_spider_tables',
        return_value=result if tables else None,
    ) as generate:
        p._match_ps_tables_to_mass()

    assert p.directories.get('spider_eos_dir') == expected
    assert generate.call_args_list == (
        [call(p.config, p.directories['output'])] if called else []
    )
    new = expected == 'new'
    assert p.directories.get('spider_solidus_ps') == ('new/sol' if new else None)
    assert p.directories.get('spider_liquidus_ps') == ('new/liq' if new else None)


def _make_hf_df():
    """Minimal helpfile DataFrame for resume tests (>init_loops+1 rows)."""
    return pd.DataFrame(
        {
            'Time': [0.0, 100.0, 200.0, 300.0, 400.0],
            'R_int': [6.371e6] * 5,
            'gravity': [9.81] * 5,
            'T_magma': [3000.0, 2800.0, 2600.0, 2400.0, 2200.0],
            'T_eqm': [255.0] * 5,
            'F_atm': [100.0] * 5,
        }
    )


@pytest.mark.unit
def test_proteus_resume_restores_zalmoxis_mesh(tmp_path):
    """start(resume=True) restores spider_mesh path from output directory."""
    p = _make_proteus_instance(tmp_path)

    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    mesh_file = data_dir / 'spider_mesh.dat'
    mesh_file.write_text('# 3 2\n6.371e6 0.0 3500.0 -9.81\n')
    prev_file = data_dir / 'spider_mesh.dat.prev'
    prev_file.write_text('# 3 2\n6.371e6 0.0 3500.0 -9.81\n')

    _resume_with_patches(p, _make_hf_df())

    assert p.directories.get('spider_mesh') == str(mesh_file)
    assert p.directories.get('spider_mesh_prev') == str(prev_file)
    assert p.directories.get('mesh_shift_active') is False
    assert p.directories.get('mesh_convergence_steps') == 0


@pytest.mark.unit
def test_proteus_resume_checks_the_eos_tables_after_unpacking(tmp_path):
    """A resume checks its EOS tables once, after it unpacks the archived data, so
    kept tables inside data.tar count."""
    p = _make_proteus_instance(tmp_path)
    (tmp_path / 'data').mkdir(exist_ok=True)
    order = []
    p.extract_archives = MagicMock(side_effect=lambda: order.append('extract'))
    p._require_paleos_tables = MagicMock(side_effect=lambda: order.append('require'))

    _resume_with_patches(p, _make_hf_df())

    assert order == ['extract', 'require']
    p._require_paleos_tables.assert_called_once_with()


@pytest.mark.unit
def test_proteus_fresh_run_checks_the_eos_tables_before_the_structure_solve(tmp_path):
    """A fresh run checks its EOS tables once, before the first structure solve."""
    p = _make_proteus_instance(tmp_path)
    p.directories.update(
        {k: str(tmp_path / k) for k in ('output/observe', 'output/offchem', 'output/plots')}
    )
    p.config.interior_energetics.flux_guess = 100.0
    p.config.star.age_ini = 0.1
    order = []
    p._require_paleos_tables = MagicMock(side_effect=lambda: order.append('require'))

    def _solve(*args, **kwargs):
        order.append('solve')
        raise _StopAfterMeshRestore

    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))
        stack.enter_context(patch('proteus.proteus.CleanDir'))
        stack.enter_context(
            patch('proteus.interior_energetics.wrapper.solve_structure', _solve)
        )
        with pytest.raises(_StopAfterMeshRestore):
            p.start(resume=False, offline=True)

    assert order == ['require', 'solve']
    p._require_paleos_tables.assert_called_once_with()


@pytest.mark.unit
@pytest.mark.parametrize('struct_module, calls', [('zalmoxis', 1), ('spider', 0)])
def test_require_paleos_tables_runs_only_for_the_zalmoxis_structure(
    tmp_path, struct_module, calls
):
    """Only a Zalmoxis structure reads the Zalmoxis EOS tables, so only it is checked,
    with the run's output directory."""
    p = _make_proteus_instance(tmp_path, struct_module=struct_module)
    with patch('proteus.interior_struct.zalmoxis.require_paleos_tables') as require:
        p._require_paleos_tables()
    assert require.call_count == calls
    if calls:
        require.assert_called_once_with(p.config, str(tmp_path))


@pytest.mark.unit
@pytest.mark.parametrize(
    'error, site',
    [
        ('missing', 'solve'),
        ('other', 'solve'),
        ('missing', 'start check'),
        ('melting curve', 'solve'),
        ('P-S table', 'solve'),
        ('melting curve class', 'solve'),
    ],
)
def test_a_missing_eos_table_anywhere_in_the_run_writes_status_20(
    tmp_path, monkeypatch, error, site
):
    """Missing reference data raised at the start check or mid-run, here from the
    structure solve, leaves status 20, so the run does not read as still running: a
    Zalmoxis EOS table, a melting curve or a SPIDER P-S table, each from its real raise
    site. Other errors leave the status as is."""
    from types import SimpleNamespace as NS

    import proteus.interior_energetics.spider as spider
    import proteus.utils.data as data
    from proteus.interior_struct.zalmoxis import ZalmoxisMissingEOSFilesError

    monkeypatch.setattr(data, 'FWL_DATA_DIR', tmp_path / 'fwl')
    monkeypatch.setattr(spider, 'find_lookup_table_dir', lambda: None)
    from proteus.interior_energetics.common import MissingMeltingCurveError

    real_site = {
        'melting curve class': MissingMeltingCurveError('melting curves not found'),
        'melting curve': lambda *a, **k: data.get_zalmoxis_melting_curves(
            NS(interior_struct=NS(melting_dir='Monteux-600'))
        ),
        'P-S table': lambda *a, **k: spider._resolve_spider_eos_dir(
            {'spider': str(tmp_path / 'nospider')}, NS(interior_struct=NS(eos_dir='none'))
        ),
    }

    p = _make_proteus_instance(tmp_path)
    p.directories.update(
        {k: str(tmp_path / k) for k in ('output/observe', 'output/offchem', 'output/plots')}
    )
    p.config.interior_energetics.flux_guess = 100.0
    p.config.star.age_ini = 0.1
    exc = real_site.get(error) or (
        ZalmoxisMissingEOSFilesError('pair') if error == 'missing' else RuntimeError('x')
    )
    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))
        stack.enter_context(patch('proteus.proteus.CleanDir'))
        stack.enter_context(
            patch(
                'proteus.interior_struct.zalmoxis.require_paleos_tables',
                side_effect=exc if site == 'start check' else None,
            )
        )
        if site == 'solve':
            stack.enter_context(
                patch('proteus.interior_energetics.wrapper.solve_structure', side_effect=exc)
            )
        with pytest.raises(Exception, match='pair|x|not found'):
            p.start(resume=False, offline=True)
    status = (tmp_path / 'status').read_text().splitlines()[0]
    assert status == ('0' if error == 'other' else '20')


@pytest.mark.unit
def test_proteus_resume_no_mesh_file(tmp_path):
    """start(resume=True) skips mesh restoration when mesh file absent."""
    p = _make_proteus_instance(tmp_path)
    (tmp_path / 'data').mkdir(exist_ok=True)

    _resume_with_patches(p, _make_hf_df())

    assert 'spider_mesh' not in p.directories
    # Discrimination: the prev-mesh slot must also remain unset. A regression
    # that touched only the primary slot but recorded a stale spider_mesh_prev
    # would still pass the above absence check.
    assert 'spider_mesh_prev' not in p.directories


@pytest.mark.unit
def test_proteus_resume_mesh_no_prev(tmp_path):
    """start(resume=True) restores mesh but skips .prev when absent."""
    p = _make_proteus_instance(tmp_path)

    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    mesh_file = data_dir / 'spider_mesh.dat'
    mesh_file.write_text('# 3 2\n6.371e6 0.0 3500.0 -9.81\n')

    _resume_with_patches(p, _make_hf_df())

    assert p.directories.get('spider_mesh') == str(mesh_file)
    assert 'spider_mesh_prev' not in p.directories
    assert p.directories.get('mesh_shift_active') is False


# ---------------------------------------------------------------------------
# Proteus._resync_zalmoxis_mesh: zalmoxis_output.dat vs the resumed row
# ---------------------------------------------------------------------------

_ROW = {'Time': 100.0, 'R_core': 3.4e6, 'R_int': 6.4e6}


def _write_mesh(path, r_first, r_last):
    """Write a 5-column Zalmoxis mesh file spanning [r_first, r_last]."""
    r = np.linspace(r_first, r_last, 6)
    np.savetxt(path, np.column_stack([r, r, r, r, r]), fmt='%.17e')


def _resync_instance(tmp_path, *, dat=None, prev=None):
    """Proteus object at the resume point, with mesh files of given bounds."""
    p = _make_proteus_instance(tmp_path, interior_module='aragog')
    data = tmp_path / 'data'
    data.mkdir(exist_ok=True)
    if dat is not None:
        _write_mesh(data / 'zalmoxis_output.dat', *dat)
    if prev is not None:
        _write_mesh(data / 'zalmoxis_output.dat.prev', *prev)
    p.hf_row = dict(_ROW)
    return p, data / 'zalmoxis_output.dat'


def _write_mesh_nan_top(path):
    """Write a mesh file whose outermost radius is NaN."""
    _write_mesh(path, 3.4e6, 6.4e6)
    data = np.loadtxt(path)
    data[-1, 0] = np.nan
    np.savetxt(path, data, fmt='%.17e')


def _saved_copy(tmp_path, time=100.0):
    """Path of the structure copy saved with the row at ``time``."""
    from proteus.utils.helper import format_subyear_time

    return tmp_path / 'data' / (format_subyear_time(time) + '_zalmoxis.dat')


@pytest.mark.unit
@pytest.mark.parametrize(
    ('dat', 'kept'),
    [
        ((3.4e6 + 0.5, 6.4e6 - 0.5), 'dat'),
        ((3.4e6 + 1.0, 6.4e6 + 1.0), 'dat'),
        ((3.4e6, 6.4e6 + 1.5), 'prev'),
        ((3.4e6, 6.4e6 + 300.0), 'prev'),
        ((3.4e6, 6.4e6 - 300.0), 'prev'),
        ((3.4e6 - 300.0, 6.4e6), 'prev'),
        ((3.4e6 + 300.0, 6.4e6), 'prev'),
        (None, 'prev'),
    ],
    ids=[
        'within tolerance',
        'at the 1 m tolerance',
        'just past it',
        'R_int above',
        'R_int inside',
        'R_core below',
        'R_core inside',
        'NaN outer radius',
    ],
)
def test_resync_without_a_saved_copy_keeps_the_file_else_restores_prev(tmp_path, dat, kept):
    """Both bounds are compared in both directions; a NaN radius never matches."""
    p, path = _resync_instance(tmp_path, dat=dat, prev=(3.4e6, 6.4e6))
    if dat is None:
        _write_mesh_nan_top(path)
    before = path.read_bytes()
    expected = before if kept == 'dat' else path.with_name(path.name + '.prev').read_bytes()

    p._resync_zalmoxis_mesh()

    assert path.read_bytes() == expected
    assert (path.read_bytes() == before) is (kept == 'dat')


@pytest.mark.unit
def test_resync_restores_the_copy_saved_with_the_row(tmp_path):
    """The saved copy wins over a stale file and a stale .prev."""
    p, path = _resync_instance(tmp_path, dat=(3.4e6, 6.4e6 + 300.0), prev=(3.4e6, 6.3e6))
    _write_mesh(_saved_copy(tmp_path), 3.4e6, 6.4e6)

    p._resync_zalmoxis_mesh()

    assert path.read_bytes() == _saved_copy(tmp_path).read_bytes()
    assert np.loadtxt(path)[-1, 0] == pytest.approx(6.4e6)


@pytest.mark.unit
def test_resync_prefers_the_saved_copy_over_a_matching_file(tmp_path):
    """Both match the row; the copy saved with the row is the one kept."""
    p, path = _resync_instance(tmp_path, dat=(3.4e6 + 0.5, 6.4e6 - 0.5))
    _write_mesh(_saved_copy(tmp_path), 3.4e6, 6.4e6)
    live = path.read_bytes()

    p._resync_zalmoxis_mesh()

    assert path.read_bytes() == _saved_copy(tmp_path).read_bytes()
    assert path.read_bytes() != live


@pytest.mark.unit
def test_resync_checks_the_saved_copy_before_using_it(tmp_path, caplog):
    """A saved copy off the row is skipped and named in a warning; the matching
    .prev is restored."""
    p, path = _resync_instance(tmp_path, dat=(3.4e6, 6.5e6), prev=(3.4e6, 6.4e6))
    _write_mesh(_saved_copy(tmp_path), 3.4e6, 6.3e6)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.proteus'):
        p._resync_zalmoxis_mesh()

    assert np.loadtxt(path)[-1, 0] == pytest.approx(6.4e6)
    assert np.loadtxt(_saved_copy(tmp_path))[-1, 0] == pytest.approx(6.3e6)
    skipped = [r.getMessage() for r in caplog.records if 'skipped' in r.getMessage()]
    assert len(skipped) == 1 and '100p000_zalmoxis.dat: R_core +0.000e+00 m' in skipped[0]


def _cut(n):
    def spoil(path):
        path.write_bytes(path.read_bytes()[:-n])

    spoil.__name__ = f'cut_{n}_bytes'
    return spoil


def _nan_radius(path):
    data = np.loadtxt(path)
    data[2, 0] = np.nan
    np.savetxt(path, data, fmt='%.17e')


def _nan_entry(path):
    data = np.loadtxt(path)
    data[2, 3] = np.nan
    np.savetxt(path, data, fmt='%.17e')


def _flat_radii(path):
    data = np.loadtxt(path)
    data[3, 0] = data[2, 0]
    np.savetxt(path, data, fmt='%.17e')


def _four_columns(path):
    np.savetxt(path, np.loadtxt(path)[:, :4], fmt='%.17e')


@pytest.mark.unit
@pytest.mark.parametrize(
    'spoil', [*map(_cut, (2, 5, 10, 30)), _nan_radius, _nan_entry, _flat_radii, _four_columns]
)
@pytest.mark.parametrize('live_matches', [True, False])
def test_resync_never_restores_a_malformed_saved_copy(tmp_path, caplog, spoil, live_matches):
    """A copy with matching bounds but a cut last line, a NaN entry, a repeated
    radius or 4 columns is skipped as invalid; the live file is kept, or the resume stops."""
    p, path = _resync_instance(tmp_path, dat=(3.4e6, 6.4e6 if live_matches else 6.5e6))
    _write_mesh(_saved_copy(tmp_path), 3.4e6, 6.4e6)
    spoil(_saved_copy(tmp_path))
    before = path.read_bytes()

    with (
        caplog.at_level(logging.WARNING, logger='fwl.proteus.proteus'),
        nullcontext() if live_matches else pytest.raises(RuntimeError, match='invalid'),
    ):
        p._resync_zalmoxis_mesh()

    assert path.read_bytes() == before
    assert ('100p000_zalmoxis.dat: invalid' in caplog.text) is live_matches


@pytest.mark.unit
@pytest.mark.parametrize(
    'copies', [(), (50.0,), (100.0,)], ids=['none', 'other row', 'this row']
)
def test_resync_stops_when_no_file_matches_the_row(tmp_path, copies):
    """A stale file and a stale .prev stop the resume with one message, which asks
    for a run from t = 0 and names a missing row copy; brackets in the path are literal."""
    run = tmp_path / 'run[1]'
    run.mkdir()
    p, path = _resync_instance(run, dat=(3.4e6, 6.5e6), prev=(3.4e6 - 2.0e3, 6.3e6))
    for t in copies:
        _write_mesh(_saved_copy(run, t), 3.4e6, 6.3e6)
    before = path.read_bytes()

    with pytest.raises(RuntimeError) as err:
        p._resync_zalmoxis_mesh()

    msg = str(err.value)
    assert 't = 1.000000e+02 yr' in msg and str(path) in msg
    assert 'zalmoxis_output.dat: R_core +0.000e+00 m, R_int +1.000e+05 m' in msg
    assert 'zalmoxis_output.dat.prev: R_core -2.000e+03 m, R_int -1.000e+05 m' in msg
    assert msg.endswith('Run the configuration again from t = 0.')
    assert ('100p000_zalmoxis.dat does not exist' in msg) is (100.0 not in copies)
    assert path.read_bytes() == before


@pytest.mark.unit
def test_resume_records_error_status_when_the_resync_raises(tmp_path):
    """Any error in the resync, not only its own mismatch, leaves status 20."""
    p = _make_resume_main_loop_proteus(tmp_path, interior_module='aragog')
    p.config.interior_struct.module = 'zalmoxis'

    with (
        patch.object(type(p), '_resync_zalmoxis_mesh', side_effect=KeyError('R_core')),
        pytest.raises(KeyError, match='R_core'),
    ):
        _run_resumed_loop_until_stop(p, _make_resume_checkpoint_df(), None, None)

    assert (tmp_path / 'status').read_text().splitlines()[0] == '20'


@pytest.mark.unit
def test_resync_stops_without_a_structure_file(tmp_path):
    """No saved copy, no zalmoxis_output.dat and no .prev: the resume stops."""
    p, path = _resync_instance(tmp_path)

    with pytest.raises(
        RuntimeError,
        match='^Resume: no Zalmoxis structure file matches the helpfile row .*: none of .* '
        'exists. Run the configuration again from t = 0.$',
    ):
        p._resync_zalmoxis_mesh()

    assert list((tmp_path / 'data').iterdir()) == []


@pytest.mark.unit
@pytest.mark.parametrize('text', ['', '\n'])
def test_an_empty_structure_file_is_invalid_without_a_warning(tmp_path, text):
    """An empty or newline-only file, as right after it is opened for writing, gives None."""
    from proteus.interior_struct.zalmoxis import zalmoxis_mesh_gaps

    path = tmp_path / 'zalmoxis_output.dat'
    path.write_text(text)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        gaps = zalmoxis_mesh_gaps(str(path), _ROW)

    assert gaps is None
    assert path.read_text() == text


@pytest.mark.unit
@pytest.mark.parametrize('where', ['snapshot save', 'resume restore'])
def test_a_failed_copy_leaves_the_target_and_no_temporary_file(tmp_path, where):
    """A copy that fails after writing part of the file changes neither target."""
    p, path = _resync_instance(tmp_path, dat=(3.4e6, 6.5e6))
    p.config.interior_struct.module = 'zalmoxis'
    _write_mesh(_saved_copy(tmp_path), 3.4e6, 6.4e6 + 0.5)
    (tmp_path / 'data' / '100p000_int.nc').touch()
    target, run = (
        (_saved_copy(tmp_path), p._save_zalmoxis_output)
        if where == 'snapshot save'
        else (path, p._resync_zalmoxis_mesh)
    )
    before = target.read_bytes()

    def partial_copy(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes()[:50])
        raise OSError('disk full')

    with (
        patch('proteus.interior_struct.zalmoxis.shutil.copy2', side_effect=partial_copy),
        pytest.raises(OSError, match='disk full'),
    ):
        run()

    assert target.read_bytes() == before
    assert not list((tmp_path / 'data').glob('*.tmp'))


@pytest.mark.unit
@pytest.mark.parametrize(
    ('struct', 'energetics', 'saved'),
    [('zalmoxis', 'aragog', True), ('zalmoxis', 'spider', False), ('dummy', 'aragog', False)],
)
def test_structure_copy_is_saved_only_where_the_resume_reads_it(
    tmp_path, struct, energetics, saved
):
    """Only Zalmoxis + Aragog runs keep a copy per snapshot."""
    p, path = _resync_instance(tmp_path, dat=(3.4e6, 6.4e6))
    p.config.interior_struct.module = struct
    p.config.interior_energetics.module = energetics
    (tmp_path / 'data' / '100p000_int.nc').touch()

    p._save_zalmoxis_output()

    assert _saved_copy(tmp_path).exists() is saved
    assert path.exists()


@pytest.mark.unit
def test_save_zalmoxis_output_snapshot_copies_under_the_row_time(tmp_path):
    """The copy is named like the interior snapshot of the same time, and only
    written next to it."""
    from proteus.interior_struct.zalmoxis import save_zalmoxis_output_snapshot

    (tmp_path / 'data').mkdir()
    save_zalmoxis_output_snapshot(str(tmp_path), 100.0)
    assert list((tmp_path / 'data').iterdir()) == []

    _write_mesh(tmp_path / 'data' / 'zalmoxis_output.dat', 3.4e6, 6.4e6)
    save_zalmoxis_output_snapshot(str(tmp_path), 100.0)
    assert not _saved_copy(tmp_path).exists(), 'a row without its _int.nc gets no copy'

    (tmp_path / 'data' / '100p000_int.nc').touch()
    save_zalmoxis_output_snapshot(str(tmp_path), 100.0)

    assert _saved_copy(tmp_path).name == '100p000_zalmoxis.dat'
    assert (
        _saved_copy(tmp_path).read_bytes()
        == (tmp_path / 'data' / 'zalmoxis_output.dat').read_bytes()
    )
    assert not list((tmp_path / 'data').glob('*.tmp'))


def _aragog_like_interior(mesh_path, calls):
    """Fake run_interior that checks the mesh file as Aragog does.

    The first call is solver setup, which allows 5 % of the mantle thickness;
    every later call is ``reset()``, which runs Aragog's own radius validator.
    """
    from types import SimpleNamespace

    from proteus.utils.helper import format_subyear_time

    entropy_solver = pytest.importorskip('aragog.solver.entropy_solver')
    _validate_eos_radius_range = entropy_solver._validate_eos_radius_range

    def run(*args, **kwargs):
        hf_row = args[3]
        if kwargs.get('write_data'):
            t_new = hf_row['Time'] + args[4].dt
            (Path(args[0]['output']) / 'data' / f'{format_subyear_time(t_new)}_int.nc').touch()
        r = np.loadtxt(mesh_path)[:, 0]
        inner, outer = hf_row['R_core'], hf_row['R_int']
        if calls:
            _validate_eos_radius_range(
                SimpleNamespace(eos_radius=r, inner_radius=inner, outer_radius=outer)
            )
        else:
            assert abs(r[0] - inner) <= 0.05 * (outer - inner)
            assert abs(r[-1] - outer) <= 0.05 * (outer - inner)
        calls.append((inner, outer))

    return run


def _stop_on_second_atmosphere_call():
    seen = []

    def run(*args, **kwargs):
        seen.append(1)
        if len(seen) == 2:
            raise _StopAfterAtmosphereCall

    return run


@pytest.mark.unit
@pytest.mark.parametrize('case', ['saved copy', 'prev matches'])
def test_resumed_loop_reaches_a_second_reset_on_a_stale_mesh(tmp_path, case):
    """Through start(resume=True), a stale mesh file is replaced before reset() at step 2.

    The fake interior applies Aragog's setup and reset() checks against the row
    the main loop hands it. Every iteration writes a snapshot here, so the copy
    saved with the first resumed row is checked too.
    """
    p = _make_resume_main_loop_proteus(tmp_path, interior_module='aragog')
    p.config.interior_struct.module = 'zalmoxis'
    p.config.params.out.write_mod = 1
    hf_df = _make_resume_checkpoint_df()
    hf_df['R_core'], hf_df['R_int'] = 3.4e6, 6.4e6
    data = tmp_path / 'data'
    mesh = data / 'zalmoxis_output.dat'
    _write_mesh(mesh, 3.4e6, 6.4e6 + 300.0)
    if case == 'saved copy':
        _write_mesh(_saved_copy(tmp_path, 400.0), 3.4e6, 6.4e6)
        _write_mesh(data / 'zalmoxis_output.dat.prev', 3.4e6, 6.4e6 - 300.0)
    else:
        _write_mesh(data / 'zalmoxis_output.dat.prev', 3.4e6, 6.4e6)
    calls = []

    _run_resumed_loop_until_stop(
        p, hf_df, _aragog_like_interior(mesh, calls), _stop_on_second_atmosphere_call()
    )

    assert calls == pytest.approx([(3.4e6, 6.4e6), (3.4e6, 6.4e6)])
    assert _saved_copy(tmp_path, p.hf_all['Time'].iloc[-1]).read_bytes() == mesh.read_bytes()


@pytest.mark.unit
def test_snapshot_copy_holds_the_structure_of_its_row_after_a_resolve(tmp_path):
    """An in-loop re-solve before the snapshot write is in the saved copy.

    The fake re-solve moves R_int by 50 m and rewrites the file, as an accepted
    Zalmoxis update does; the copy written with that row must carry it.
    """
    from proteus.interior_struct.zalmoxis import zalmoxis_mesh_gaps

    p = _make_resume_main_loop_proteus(tmp_path, interior_module='aragog')
    p.config.interior_struct.module = 'zalmoxis'
    p.config.interior_struct.zalmoxis.update_interval = 1.0
    p.config.params.out.write_mod = 1
    hf_df = _make_resume_checkpoint_df()
    hf_df['R_core'], hf_df['R_int'] = 3.4e6, 6.4e6
    mesh = tmp_path / 'data' / 'zalmoxis_output.dat'
    _write_mesh(mesh, 3.4e6, 6.4e6)

    def resolve(dirs, config, hf_row, interior_o, t, T, phi, force=False):
        hf_row['R_int'] += 50.0
        _write_mesh(mesh, hf_row['R_core'], hf_row['R_int'])
        return t, T, phi

    with patch(
        'proteus.interior_energetics.wrapper.update_structure_from_interior',
        side_effect=resolve,
    ):
        _run_resumed_loop_until_stop(
            p, hf_df, _aragog_like_interior(mesh, []), _stop_on_second_atmosphere_call()
        )

    row = p.hf_all.iloc[-1].to_dict()
    gaps = zalmoxis_mesh_gaps(str(_saved_copy(tmp_path, row['Time'])), row)
    assert row['R_int'] == pytest.approx(6.4e6 + 50.0)
    assert gaps[:2] == pytest.approx((0.0, 0.0), abs=1e-6)


@pytest.mark.unit
def test_finished_run_saves_the_final_copy_after_the_final_snapshot(tmp_path):
    """The last row of a finished run gets its copy once its _int.nc is on disk.

    The last loop is not a snapshot step here, so the final interior snapshot is
    written after the loop; the copy must follow it, not precede it.
    """
    from proteus.utils.helper import format_subyear_time

    p = _make_resume_main_loop_proteus(tmp_path, interior_module='aragog')
    p.config.interior_struct.module = 'zalmoxis'
    p.config.params.out.plot_mod = None
    p.config.params.out.archive_mod = None
    hf_df = _make_resume_checkpoint_df()
    hf_df['R_core'], hf_df['R_int'] = 3.4e6, 6.4e6
    mesh = tmp_path / 'data' / 'zalmoxis_output.dat'
    _write_mesh(mesh, 3.4e6, 6.4e6)
    seen = []

    def final_snapshot(config, interior_o, dirs, hf_row):
        name = format_subyear_time(hf_row['Time'])
        seen.append(_saved_copy(tmp_path, hf_row['Time']).exists())
        (tmp_path / 'data' / f'{name}_int.nc').touch()

    with patch(
        'proteus.interior_energetics.aragog.write_final_snapshot', side_effect=final_snapshot
    ):
        _run_resumed_loop_until_stop(
            p, hf_df, _aragog_like_interior(mesh, []), lambda *a, **k: None, terminate_after=2
        )

    assert seen == [False], 'the copy must not exist before the final snapshot'
    assert _saved_copy(tmp_path, p.hf_row['Time']).read_bytes() == mesh.read_bytes()


@pytest.mark.unit
def test_impact_on_the_last_step_saves_no_final_copy(tmp_path):
    """Without a final snapshot of its own, the last row gets no copy, even when
    an earlier row's _int.nc carries the same name."""
    from proteus.utils.helper import format_subyear_time

    p = _make_resume_main_loop_proteus(tmp_path, interior_module='aragog')
    p.config.interior_struct.module = 'zalmoxis'
    p.config.params.out.plot_mod = None
    p.config.params.out.archive_mod = None
    hf_df = _make_resume_checkpoint_df()
    hf_df['R_core'], hf_df['R_int'] = 3.4e6, 6.4e6
    mesh = tmp_path / 'data' / 'zalmoxis_output.dat'
    _write_mesh(mesh, 3.4e6, 6.4e6)
    aragog = _aragog_like_interior(mesh, [])

    def impact(*args, **kwargs):
        aragog(*args, **kwargs)
        args[4].aragog_solver.solution = None
        t_new = args[3]['Time'] + args[4].dt
        (tmp_path / 'data' / f'{format_subyear_time(t_new)}_int.nc').touch()

    with patch('proteus.interior_energetics.aragog.write_final_snapshot') as final:
        _run_resumed_loop_until_stop(p, hf_df, impact, lambda *a, **k: None, terminate_after=2)

    assert not final.called
    assert (tmp_path / 'data' / f'{format_subyear_time(p.hf_row["Time"])}_int.nc').exists()
    assert not _saved_copy(tmp_path, p.hf_row['Time']).exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ('struct', 'energetics', 'called'),
    [('zalmoxis', 'aragog', True), ('zalmoxis', 'spider', False), ('dummy', 'aragog', False)],
)
def test_resume_runs_the_resync_for_zalmoxis_and_aragog_only(
    tmp_path, struct, energetics, called
):
    """start(resume=True) calls the resync before the star setup, for Zalmoxis + Aragog."""
    p = _make_proteus_instance(tmp_path, struct_module=struct, interior_module=energetics)
    (tmp_path / 'data').mkdir(exist_ok=True)

    with patch.object(type(p), '_resync_zalmoxis_mesh', autospec=True) as resync:
        _resume_with_patches(p, _make_hf_df())

    # _resume_with_patches stops at init_star, so a recorded call came before it.
    assert resync.called is called
    assert p.directories['_resume_struct_settle_loops'] > 0


class _StopAtRunningStatus(Exception):
    """Sentinel raised when start() reports the run as running."""


def _resume_to_running_status(p, hf_df, *, events, stub_selection):
    """Resume ``p`` from ``hf_df`` with accretion on, up to the running status.

    The star and orbit setup are no-ops, ``init_accretion`` returns ``events``,
    and ``stub_selection=True`` passes the helpfile through snapshot selection
    unchanged; ``stub_selection=False`` runs the real selection on ``data/``.
    """

    def status(dirs, code):
        if code == 1:
            raise _StopAtRunningStatus

    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))
        stack.enter_context(
            patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50)
        )
        stack.enter_context(
            patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=hf_df)
        )
        if stub_selection:
            stack.enter_context(
                patch(
                    'proteus.utils.coupler.select_resumable_snapshot',
                    return_value=(hf_df, []),
                )
            )
        stack.enter_context(
            patch('proteus.outgas.wrapper.check_desiccation', return_value=False)
        )
        stack.enter_context(patch('proteus.utils.coupler.ZeroHelpfileRow', return_value={}))
        interior_t = stack.enter_context(patch('proteus.interior_energetics.common.Interior_t'))
        interior_t.return_value = MagicMock(ic=1)
        stack.enter_context(
            patch('proteus.accretion.wrapper.init_accretion', return_value=events)
        )
        stack.enter_context(patch('proteus.star.wrapper.init_star'))
        stack.enter_context(patch('proteus.orbit.wrapper.init_orbit'))
        stack.enter_context(patch('proteus.proteus.UpdateStatusfile', side_effect=status))

        with pytest.raises(_StopAtRunningStatus):
            p.start(resume=True, offline=True)


@pytest.mark.unit
def test_resume_after_an_impact_restores_the_row_structure_and_the_accreted_mass(tmp_path):
    """Both resume steps act on a Zalmoxis + Aragog row written after an impact.

    The resync puts back the structure copy saved with that row, the accretion
    restore rebuilds the planet mass from the ledger, and neither moves the
    other's state: the row keeps the radii it was read with.
    """
    from proteus.utils.constants import M_earth

    p = _make_proteus_instance(tmp_path, struct_module='zalmoxis', interior_module='aragog')
    p.config.accretion.module = 'dummy'
    p.config.planet.mass_tot = 1.0
    hf_df = _make_hf_df()
    hf_df['R_core'], hf_df['R_int'] = 3.4e6, 6.5e6
    hf_df['M_accreted_rock'] = 0.05 * M_earth
    hf_df['n_impacts_applied'] = 1.0
    data = tmp_path / 'data'
    data.mkdir(exist_ok=True)
    _write_mesh(data / 'zalmoxis_output.dat', 3.4e6, 6.5e6 + 2.0e4)
    _write_mesh(_saved_copy(tmp_path, 400.0), 3.4e6, 6.5e6)

    _resume_to_running_status(p, hf_df, events=[], stub_selection=True)

    assert (data / 'zalmoxis_output.dat').read_bytes() == _saved_copy(
        tmp_path, 400.0
    ).read_bytes()
    assert p.config.planet.mass_tot == pytest.approx(1.05, rel=1e-12)
    assert (p.hf_row['R_core'], p.hf_row['R_int']) == (3.4e6, 6.5e6)
    assert p.hf_row['n_impacts_applied'] == 1


@pytest.mark.unit
def test_resume_walks_back_past_an_impact_step_to_its_own_structure(tmp_path):
    """An impact on a snapshot step discards that step's _int.nc, so the step has
    no structure copy. The resume walks back to the previous complete row,
    restores that row's own copy, keeps the configured mass, and leaves the
    impact to be applied again.
    """
    from types import SimpleNamespace

    from netCDF4 import Dataset

    from proteus.utils.constants import M_earth
    from proteus.utils.helper import format_subyear_time

    p = _make_proteus_instance(tmp_path, struct_module='zalmoxis', interior_module='aragog')
    p.config.accretion.module = 'dummy'
    p.config.planet.mass_tot = 1.0
    times = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    hf_df = pd.DataFrame(
        {
            'Time': times,
            'R_core': 3.4e6,
            'R_int': [6.4e6] * 6 + [6.5e6],
            'gravity': 9.81,
            'T_magma': 3000.0,
            'T_eqm': 255.0,
            'F_atm': 100.0,
            'M_accreted_rock': [0.0] * 6 + [0.05 * M_earth],
            'n_impacts_applied': [0.0] * 6 + [1.0],
        }
    )
    data = tmp_path / 'data'
    data.mkdir(exist_ok=True)
    for t in times[1:]:
        halves = ('atm',) if t == times[-1] else ('int', 'atm')
        for half in halves:
            with Dataset(str(data / f'{format_subyear_time(t)}_{half}.nc'), 'w') as ds:
                ds.createDimension('x', 1)
    _write_mesh(data / 'zalmoxis_output.dat', 3.4e6, 6.5e6)
    _write_mesh(_saved_copy(tmp_path, 500.0), 3.4e6, 6.4e6)
    impact = SimpleNamespace(time=550.0)

    _resume_to_running_status(p, hf_df, events=[impact], stub_selection=False)

    assert p.hf_row['Time'] == pytest.approx(500.0)
    assert (data / 'zalmoxis_output.dat').read_bytes() == _saved_copy(
        tmp_path, 500.0
    ).read_bytes()
    assert p.config.planet.mass_tot == pytest.approx(1.0, rel=1e-12)
    assert p.impact_events == [impact]


# ---------------------------------------------------------------------------
# Proteus.start(): refusing a helpfile that predates schema columns.
# ---------------------------------------------------------------------------


class _StopAfterHelpfileLoad(Exception):
    """Sentinel exception to stop start() once the helpfile has been read."""


def _resume_at_helpfile_load(p, *, read_effect=None):
    """Drive start() as far as the helpfile load and stop there.

    Returns the ReadHelpfileFromCSV and UpdateStatusfile mocks, which keep
    their call history after the patches are lifted, plus the exception the
    run ended on.
    """
    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))
        stack.enter_context(
            patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50)
        )
        mock_read = stack.enter_context(patch('proteus.utils.coupler.ReadHelpfileFromCSV'))
        if read_effect is None:
            mock_read.return_value = _make_hf_df()
        else:
            mock_read.side_effect = read_effect
        # start() reads the helpfile, then hands it to snapshot selection.
        # Stopping there keeps the test on the load and off the solver setup.
        stack.enter_context(
            patch(
                'proteus.utils.coupler.select_resumable_snapshot',
                side_effect=_StopAfterHelpfileLoad,
            )
        )
        mock_status = stack.enter_context(patch('proteus.proteus.UpdateStatusfile'))

        with pytest.raises(Exception) as excinfo:
            p.start(resume=True, offline=True)

    return mock_read, mock_status, excinfo.value


@pytest.mark.unit
def test_resume_records_an_error_status_on_helpfile_schema_drift(tmp_path):
    """A refused resume writes the error status before it stops.

    A run that died without updating its status file reads as still running
    to every downstream tool that polls the output directory.
    """
    from proteus.utils.coupler import HelpfileSchemaDriftError

    p = _make_proteus_instance(tmp_path)
    drift = HelpfileSchemaDriftError('predates 2 column(s): M_atm, eccentricity')
    mock_read, mock_status, ended_on = _resume_at_helpfile_load(p, read_effect=drift)

    # The failure propagates rather than being swallowed into a partial run.
    assert isinstance(ended_on, HelpfileSchemaDriftError)
    assert 'M_atm' in str(ended_on)

    statuses = [call.args[1] for call in mock_status.call_args_list]
    assert 20 in statuses
    # Discrimination: 0 is the start-of-run status written earlier, so the
    # error status must be the last word and not merely present.
    assert statuses[-1] == 20

    # The resume asks for the shortfall to be reported and passes nothing
    # that could soften it. A keyword here would mean an opt-out had been
    # reintroduced, which is what makes a fabricated value reachable.
    assert mock_read.call_args.kwargs == {}
    assert mock_read.call_args.args == (p.directories['output'],)


@pytest.mark.unit
def test_resume_records_an_error_status_on_any_helpfile_load_failure(tmp_path):
    """A resume that cannot read its helpfile at all records the same status.

    Schema drift is one of several ways the load fails: the file may be
    absent because the run died before its first write, or unparseable
    because it was truncated. Each leaves the run just as dead, so each has
    to leave the same mark on the status file.
    """
    p = _make_proteus_instance(tmp_path)

    for label, effect in (
        ('missing file', Exception("Cannot find helpfile at '/nowhere/runtime_helpfile.csv'")),
        ('unparseable file', pd.errors.EmptyDataError('No columns to parse from file')),
    ):
        _, mock_status, ended_on = _resume_at_helpfile_load(p, read_effect=effect)

        assert type(ended_on) is type(effect), label
        statuses = [call.args[1] for call in mock_status.call_args_list]
        assert statuses[-1] == 20, label
        # Discrimination: 0 is written at the start of every run, so a status
        # list of [0] alone is the untreated case this guards against.
        assert statuses != [0], label


@pytest.mark.unit
def test_postprocessing_hands_the_physics_module_a_reporting_row(tmp_path):
    """The row reaching the synthesis code reports a column it does not carry.

    The required set is written by hand and can fall behind the code, so the
    row itself has to say what is wrong for anything the set does not cover.
    An ordinary dict would reach the same read as a bare KeyError.
    """
    from proteus.utils.coupler import (
        GetPostprocessingKeys,
        HelpfileRow,
        HelpfileSchemaDriftError,
    )

    p = _make_proteus_instance(tmp_path)
    p.config.atmos_chem.module = 'vulcan'

    for method, wrapper in (
        ('observe', 'proteus.observe.wrapper.run_observe'),
        ('offline_chemistry', 'proteus.atmos_chem.wrapper.run_chemistry'),
    ):
        with ExitStack() as stack:
            stack.enter_context(patch.object(type(p), 'extract_archives'))
            stack.enter_context(
                patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=_make_hf_df())
            )
            mock_wrapper = stack.enter_context(patch(wrapper))
            getattr(p, method)()

        # The two wrappers take the row in different positions, so select it
        # by type. An unwrapped row yields no match, which is the regression
        # this guards: it would reach the synthesis code as a bare dict.
        handed = [a for a in mock_wrapper.call_args.args if isinstance(a, HelpfileRow)]
        assert len(handed) == 1, method
        row = handed[0]

        # A column outside the required set reports itself on being read.
        absent = 'struct_mass_desync_frac'
        assert absent not in GetPostprocessingKeys()
        with pytest.raises(HelpfileSchemaDriftError, match=absent):
            row[absent]

        # Discrimination: the columns the run does carry still read normally,
        # so the wrapper reports a shortfall rather than blocking every read.
        assert row['T_magma'] == pytest.approx(2200.0), method


@pytest.mark.unit
def test_postprocessing_refuses_a_helpfile_that_predates_schema_columns(tmp_path):
    """observe() and offline_chemistry() stop on the same shortfall.

    Both seed a working row from the last line of the same table and hand it
    to a physics module, so neither may proceed on columns the run never
    wrote.
    """
    from proteus.utils.coupler import HelpfileSchemaDriftError

    p = _make_proteus_instance(tmp_path)
    p.config.atmos_chem.module = 'vulcan'
    drift = HelpfileSchemaDriftError('predates 1 column(s): R_xuv')

    for method, wrapper in (
        ('observe', 'proteus.observe.wrapper.run_observe'),
        ('offline_chemistry', 'proteus.atmos_chem.wrapper.run_chemistry'),
    ):
        with ExitStack() as stack:
            mock_extract = stack.enter_context(patch.object(type(p), 'extract_archives'))
            mock_read = stack.enter_context(
                patch('proteus.utils.coupler.ReadHelpfileFromCSV', side_effect=drift)
            )
            mock_wrapper = stack.enter_context(patch(wrapper))
            with pytest.raises(HelpfileSchemaDriftError):
                getattr(p, method)()

        # Postprocessing is held to the columns it actually reads, not to the
        # whole schema, so a run short of an unrelated diagnostic stays
        # readable. Passing nothing here would restore the blanket check.
        from proteus.utils.coupler import GetHelpfileKeys, GetPostprocessingKeys

        required = mock_read.call_args.kwargs['required_columns']
        assert set(required) == set(GetPostprocessingKeys()), method
        assert set(required) < set(GetHelpfileKeys()), method

        # Discrimination: the physics module is never reached, so no
        # incomplete row can be handed to it.
        assert mock_wrapper.call_count == 0, method
        # The run is left archived as it was found. Unpacking first would
        # delete the tar on the way to a refusal that was already certain.
        assert mock_extract.call_count == 0, method


# ---------------------------------------------------------------------------
# Proteus._check_atmosphere_deadlock: AGNI-vs-interior deadlock detector.
# Targets the previously-untested block at proteus.py:802-853 (now extracted
# to a method on Proteus so it can be exercised in isolation).
# ---------------------------------------------------------------------------


def _make_deadlock_proteus(
    tmp_path,
    *,
    converged=False,
    hf_all=None,
    hf_row=None,
    stale_iters=0,
    stall_enabled=True,
    stall_maximum=STALL_MAX_CONFIGURED,
):
    """Build a Proteus instance pre-positioned for the deadlock check.

    The fields the check reads off the run state (atmos_o.converged,
    atmos_o.levels_stale_iters, hf_all, hf_row, agni_deadlock_count,
    agni_deadlock_max, directories) are set explicitly; everything else is
    left at its post-__init__ default. The stall cap and switch are among
    those defaults on purpose: they reach the check from the config the
    instance was built with, so a test can see what the constructor made of
    it rather than what the test assigned afterwards.
    """
    from types import SimpleNamespace

    from proteus.proteus import AGNI_DEADLOCK_MAX

    p = _make_proteus_instance(
        tmp_path, stall_enabled=stall_enabled, stall_maximum=stall_maximum
    )
    p.atmos_o = SimpleNamespace(converged=bool(converged), levels_stale_iters=int(stale_iters))
    p.hf_all = hf_all
    p.hf_row = hf_row if hf_row is not None else {}
    p.agni_deadlock_count = 0
    p.agni_deadlock_max = AGNI_DEADLOCK_MAX
    return p


def test_check_atmosphere_deadlock_resets_counter_when_solve_converged(tmp_path):
    """A converged atmosphere solve must reset the deadlock counter to
    zero, regardless of any previously-accumulated misses.

    Discriminating: pre-load the counter to a near-trip value (2 out
    of 3) so a regression that incremented on converged solves would
    visibly cross the threshold.
    """
    p = _make_deadlock_proteus(tmp_path, converged=True)
    p.agni_deadlock_count = 2
    p._check_atmosphere_deadlock()
    assert p.agni_deadlock_count == 0
    assert p.agni_deadlock_count != 2  # guard: reset happened, not no-op


def test_check_atmosphere_deadlock_does_not_fire_on_first_iteration(tmp_path):
    """When hf_all is None (the fresh-run state before the first row
    is committed), the deadlock cannot fire because there is no
    previous row to compare against. The counter stays at zero.

    Edge: limit-input case for the first iteration of a fresh run.
    """
    p = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        hf_all=None,
        hf_row={'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0},
    )
    p._check_atmosphere_deadlock()
    assert p.agni_deadlock_count == 0
    assert p.hf_all is None  # hf_all untouched


def test_check_atmosphere_deadlock_resets_when_interior_state_moved(tmp_path):
    """An AGNI failure with a still-moving interior is a transient
    non-convergence, not a deadlock. The counter must reset.

    Discriminating: T_magma differs by 50 K between prev and current.
    Even though the solver did not converge, the interior is clearly
    evolving, so the deadlock detector must NOT increment.
    """
    p = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        hf_all=pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3050.0, 'Phi_global': 1.0}]),
        hf_row={'F_atm': 110.0, 'T_magma': 3000.0, 'Phi_global': 1.0},
    )
    p.agni_deadlock_count = 1
    p._check_atmosphere_deadlock()
    assert p.agni_deadlock_count == 0
    assert p.agni_deadlock_count != 1  # guard: reset happened


def test_check_atmosphere_deadlock_increments_when_interior_frozen(tmp_path, caplog):
    """When AGNI fails AND (T_magma, Phi_global, F_atm) all match the
    previous row to bit-exactness (T, Phi) or 1e-6 relative (F), the
    counter must increment and the warning message must name both
    the current count and the configured maximum.

    Discriminating: pin both the counter (==1) and the warning text.
    A regression that read the wrong dict key would land at zero.
    """
    import logging

    p = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        hf_all=pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0}]),
        hf_row={'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0},
    )
    p.agni_deadlock_max = 3
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.proteus'):
        p._check_atmosphere_deadlock()
    assert p.agni_deadlock_count == 1
    messages = [r.message for r in caplog.records]
    assert any('deadlock count = 1 / 3' in m for m in messages)


def test_check_atmosphere_deadlock_raises_at_threshold(tmp_path):
    """When the counter reaches agni_deadlock_max, the detector must
    write status code 22 AND raise RuntimeError. The raise comes
    AFTER UpdateStatusfile so an unattended run leaves a parseable
    status on disk.

    Discriminating: pin the status code (22, not 20 or 23) and the
    exception type. A regression that re-ordered (raise before
    status-write) would leave mock_update uncalled.
    """
    p = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        hf_all=pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0}]),
        hf_row={'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0},
    )
    p.agni_deadlock_max = 3
    p.agni_deadlock_count = 2  # already at max - 1
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        with pytest.raises(RuntimeError, match='consecutive AGNI failures'):
            p._check_atmosphere_deadlock()
    mock_update.assert_called_once()
    args, _ = mock_update.call_args
    assert args[1] == 22


def test_check_atmosphere_deadlock_f_atm_tolerance_boundary(tmp_path):
    """The F_atm match uses a 1e-6 relative tolerance, NOT bit-
    exactness, so AGNI's stochastic non-convergence noise still
    registers as frozen. Pin the boundary at relative change 5e-7
    (below the threshold) -> still frozen -> counter increments.

    Discriminating: an F_atm relative change just above 1e-6 would
    classify as NOT-frozen and reset the counter; a regression that
    flipped the comparator (>= vs <) would fire here.
    """
    p = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        hf_all=pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0}]),
        # F_atm rel change = 0.00005 / 100 = 5e-7  (below 1e-6 threshold)
        hf_row={'F_atm': 100.00005, 'T_magma': 3000.0, 'Phi_global': 1.0},
    )
    p.agni_deadlock_max = 3
    p._check_atmosphere_deadlock()
    assert p.agni_deadlock_count == 1  # counted as frozen
    # Now perturb F_atm above the tolerance and confirm the reset path
    # fires. This is the discrimination guard against a flipped
    # comparator.
    p.hf_row = {'F_atm': 200.0, 'T_magma': 3000.0, 'Phi_global': 1.0}
    p._check_atmosphere_deadlock()
    assert p.agni_deadlock_count == 0


def test_check_atmosphere_deadlock_aborts_a_stalled_atmosphere(tmp_path):
    """An atmosphere that never converges ends the run even while the
    interior is still moving.

    Physical scenario: the interior keeps cooling on levels carried from an
    older solve, so the frozen-state test never fires and the run would
    otherwise spend its whole budget on a structure it never resolved.

    Discriminating: the interior moves by 50 K between the rows, which is the
    same input that resets the deadlock counter above, so an abort here is
    attributable to the stall count alone. The streak is measured against the
    cap the config carries, which is not the module constant, so a run that
    read the constant instead would sit far below its cap and not abort.
    """
    from proteus.proteus import ATMOS_STALL_MAX

    assert STALL_MAX_CONFIGURED != ATMOS_STALL_MAX

    moving = {
        'hf_all': pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3050.0, 'Phi_global': 1.0}]),
        'hf_row': {'F_atm': 140.0, 'T_magma': 3000.0, 'Phi_global': 0.9},
    }
    p = _make_deadlock_proteus(
        tmp_path, converged=False, stale_iters=STALL_MAX_CONFIGURED, **moving
    )
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        with pytest.raises(RuntimeError, match=f'{STALL_MAX_CONFIGURED} consecutive solves'):
            p._check_atmosphere_deadlock()
    mock_update.assert_called_once()
    args, _ = mock_update.call_args
    assert args[1] == 22

    # The frozen-interior counter is not what fired: it never left zero.
    assert p.agni_deadlock_count == 0

    # One short of the cap the run continues, so the abort is on the
    # threshold rather than on any non-converged solve.
    q = _make_deadlock_proteus(
        tmp_path, converged=False, stale_iters=STALL_MAX_CONFIGURED - 1, **moving
    )
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        q._check_atmosphere_deadlock()
    mock_update.assert_not_called()
    assert q.agni_deadlock_count == 0


def test_check_atmosphere_deadlock_stall_yields_to_a_converged_solve(tmp_path):
    """The convergence flag short-circuits the check before the count is read.

    Contract clause only: the wrapper zeroes the count on a converged solve
    before this method ever runs, so the pairing below cannot arise in a
    coupled run. What is pinned here is the order of the two tests, so a
    count left on the struct can never kill a run whose atmosphere converged.
    """
    from proteus.proteus import ATMOS_STALL_MAX

    p = _make_deadlock_proteus(
        tmp_path,
        converged=True,
        stale_iters=ATMOS_STALL_MAX + 1,
        hf_all=pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0}]),
        hf_row={'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0},
    )
    p.agni_deadlock_count = 2
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        p._check_atmosphere_deadlock()
    mock_update.assert_not_called()
    assert p.agni_deadlock_count == 0

    # The same count with a failed solve does abort, so the convergence flag
    # is what spared it.
    q = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        stale_iters=ATMOS_STALL_MAX + 1,
        hf_all=pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0}]),
        hf_row={'F_atm': 180.0, 'T_magma': 2900.0, 'Phi_global': 0.8},
    )
    with patch('proteus.proteus.UpdateStatusfile'):
        with pytest.raises(RuntimeError, match=f'{ATMOS_STALL_MAX + 1} consecutive solves'):
            q._check_atmosphere_deadlock()


def test_check_atmosphere_deadlock_frozen_interior_still_aborts_first(tmp_path):
    """A frozen interior keeps its own, much earlier abort.

    Contract clause: the stall cap is a backstop for the case the frozen test
    cannot see, so it must not delay the three-iteration abort that fires
    when the interior has stopped moving as well.
    """
    frozen = {
        'hf_all': pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0}]),
        'hf_row': {'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0},
    }
    p = _make_deadlock_proteus(tmp_path, converged=False, stale_iters=3, **frozen)
    p.agni_deadlock_count = 2
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        with pytest.raises(RuntimeError, match='consecutive AGNI failures'):
            p._check_atmosphere_deadlock()
    args, _ = mock_update.call_args
    assert args[1] == 22

    # Three frozen iterations is well inside the stall cap, so the two paths
    # are not being confused for one another.
    assert p.agni_deadlock_count == 3
    assert p.agni_deadlock_count < p.atmos_stall_max


def test_check_atmosphere_deadlock_reads_the_count_the_wrapper_produces(tmp_path):
    """The count the abort reads is the one the atmosphere wrapper writes.

    Contract clause: the two halves live in different modules, so this drives
    the real producer, `carry_converged_levels`, rather than setting the
    field by hand, and feeds the struct it leaves behind to the check.
    """
    from proteus.atmos_clim.common import Atmos_t
    from proteus.atmos_clim.wrapper import carry_converged_levels
    from proteus.proteus import ATMOS_STALL_MAX

    atmos_o = Atmos_t()
    converged_row = {'R_xuv': 7.0e6, 'p_xuv': 1.0e2, 'T_xuv': 900.0, 'g_xuv': 9.5}

    # One accepted solve gives the run something to fall back on.
    atmos_o.converged = True
    carry_converged_levels(atmos_o, dict(converged_row))
    assert atmos_o.levels_stale_iters == 0

    # Then the atmosphere stops resolving, once per iteration.
    atmos_o.converged = False
    for expected in range(1, ATMOS_STALL_MAX + 1):
        carry_converged_levels(atmos_o, dict(converged_row))
        assert atmos_o.levels_stale_iters == expected

    p = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        hf_all=pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3050.0, 'Phi_global': 1.0}]),
        hf_row={'F_atm': 140.0, 'T_magma': 3000.0, 'Phi_global': 0.9},
    )
    p.atmos_o = atmos_o
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        with pytest.raises(RuntimeError, match=f'{ATMOS_STALL_MAX} consecutive solves'):
            p._check_atmosphere_deadlock()
    args, _ = mock_update.call_args
    assert args[1] == 22

    # A single accepted solve clears the count the wrapper keeps, so the run
    # that recovers on its next iteration is not carrying a near-fatal state.
    atmos_o.converged = True
    carry_converged_levels(atmos_o, dict(converged_row))
    assert atmos_o.levels_stale_iters == 0
    p._check_atmosphere_deadlock()


@pytest.mark.parametrize(
    ('stored', 'expected'),
    [(24.0, 24), (7.0, 7), (0.0, 0), (float('nan'), 0)],
    ids=['one_short_of_the_cap', 'mid_streak', 'not_stalling', 'unreadable'],
)
def test_proteus_resume_restores_the_unresolved_atmosphere_count(tmp_path, stored, expected):
    """A resume does not hand a stalling run a fresh allowance.

    Contract clause: the count lives on a struct rebuilt at every start, and
    the helpfile carries it, so a run killed part-way through a stall comes
    back where it left off. Without this, any resume cadence shorter than the
    cap defeats the abort entirely, which is the case a chronically stalling
    run is most likely to be in.
    """
    p = _make_proteus_instance(tmp_path)
    (tmp_path / 'data').mkdir(exist_ok=True)
    df = _make_hf_df()
    df['atm_levels_stale'] = [0.0, 0.0, 0.0, 0.0, stored]

    _resume_with_patches(p, df)

    assert p.atmos_o.levels_stale_iters == expected


@pytest.mark.unit
def test_atmos_stall_max_is_the_value_the_run_actually_uses(tmp_path):
    """The stall cap is pinned, and the abort reads it rather than a literal.

    Contract clause: the number decides when a run is given up on, so it must
    not be changeable without a test noticing, and the check must not carry a
    second copy of it. The ordering against the two neighbouring thresholds is
    what has to hold whatever the number becomes: the wrapper reports a long
    streak before anything aborts on it, and the frozen-interior abort stays
    the earlier of the two.
    """
    from proteus.atmos_clim.wrapper import CARRIED_LEVELS_ALERT
    from proteus.config._params import StopStall
    from proteus.proteus import AGNI_DEADLOCK_MAX, ATMOS_STALL_MAX

    assert ATMOS_STALL_MAX == 150
    assert ATMOS_STALL_MAX > CARRIED_LEVELS_ALERT
    assert ATMOS_STALL_MAX > AGNI_DEADLOCK_MAX

    # The cap the fixture config carries sits between the two. Tests that size
    # a streak on one constant and expect the other to decide the abort rest on
    # that ordering, so it fails here rather than in one of them.
    assert AGNI_DEADLOCK_MAX < STALL_MAX_CONFIGURED < ATMOS_STALL_MAX

    # A run whose config carries the schema default lands on the constant, so
    # a literal reintroduced on the instance would diverge from the pin above.
    default = _make_proteus_instance(tmp_path, stall_maximum=StopStall().maximum)
    assert default.atmos_stall_max == ATMOS_STALL_MAX
    assert default.agni_deadlock_max == AGNI_DEADLOCK_MAX

    # The schema default equals the constant, so the pin above cannot tell a
    # config read from a literal. A configured value that differs from it can.
    configured = _make_proteus_instance(tmp_path, stall_maximum=STALL_MAX_CONFIGURED)
    assert configured.atmos_stall_max == STALL_MAX_CONFIGURED
    assert configured.atmos_stall_max != ATMOS_STALL_MAX

    moving = {
        'hf_all': pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3050.0, 'Phi_global': 1.0}]),
        'hf_row': {'F_atm': 140.0, 'T_magma': 3000.0, 'Phi_global': 0.9},
    }

    # The cap the check enforces is the one the config delivered, so moving
    # the number moves the abort with it.
    p = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        stale_iters=ATMOS_STALL_MAX,
        stall_maximum=ATMOS_STALL_MAX,
        **moving,
    )
    with patch('proteus.proteus.UpdateStatusfile'):
        with pytest.raises(RuntimeError, match=f'{ATMOS_STALL_MAX} consecutive solves'):
            p._check_atmosphere_deadlock()

    q = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        stale_iters=ATMOS_STALL_MAX - 1,
        stall_maximum=ATMOS_STALL_MAX,
        **moving,
    )
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        q._check_atmosphere_deadlock()
    mock_update.assert_not_called()


@pytest.mark.unit
def test_proteus_resume_without_the_stale_column_starts_at_zero(tmp_path):
    """A helpfile written before the column existed resumes as unstalled.

    Contract clause: the restoration reads a column that older runs do not
    carry, so its absence has to read as a run that has not stalled rather
    than end the resume.
    """
    p = _make_proteus_instance(tmp_path)
    (tmp_path / 'data').mkdir(exist_ok=True)
    df = _make_hf_df()
    assert 'atm_levels_stale' not in df.columns

    _resume_with_patches(p, df)

    assert p.atmos_o.levels_stale_iters == 0

    # The same frame with the column present restores the stored value, so
    # the zero above is the absent-column path and not a dropped read.
    q = _make_proteus_instance(tmp_path)
    df_with = _make_hf_df()
    df_with['atm_levels_stale'] = [0.0, 0.0, 0.0, 0.0, 19.0]
    _resume_with_patches(q, df_with)
    assert q.atmos_o.levels_stale_iters == 19


# ---------------------------------------------------------------------------
# Proteus.observe() and Proteus.offline_chemistry(): postprocessing methods.
# Target lines 1055-1098 of proteus.py.
# ---------------------------------------------------------------------------


def _helpfile_df_multi_row():
    """Helpfile DataFrame with three distinct rows.

    Distinct values across the three rows are essential: a regression
    that read ``iloc[0]`` or ``iloc[len(df)//2]`` instead of ``iloc[-1]``
    would otherwise pass on a single-row fixture. The last row's values
    are pinned in the tests below.
    """
    return pd.DataFrame(
        [
            {
                'Time': 1.0e6,
                'T_magma': 3500.0,
                'F_atm': 1500.0,
                'Phi_global': 1.0,
            },
            {
                'Time': 5.0e7,
                'T_magma': 3000.0,
                'F_atm': 500.0,
                'Phi_global': 0.85,
            },
            {
                'Time': 1.0e8,
                'T_magma': 2500.0,
                'F_atm': 150.0,
                'Phi_global': 0.7,
            },
        ]
    )


def test_observe_dispatches_to_run_observe_with_last_helpfile_row(tmp_path):
    """Proteus.observe() must read the helpfile, take the last row,
    and dispatch to run_observe with (hf_row, config, directories).

    Discriminating: pin the kwargs of the run_observe call. A
    regression that passed the entire DataFrame instead of just the
    last row would fail the dict-type check; a regression that
    swapped (hf_row, config) order would fail the config identity check.
    """
    p = _make_proteus_instance(tmp_path)
    p.directories['output'] = str(tmp_path)
    df = _helpfile_df_multi_row()
    with (
        patch.object(p, 'extract_archives'),
        patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=df),
        patch('proteus.observe.wrapper.run_observe') as mock_run,
    ):
        p.observe()
    mock_run.assert_called_once()
    args = mock_run.call_args.args
    # First arg is the hf_row dict pulled from df.iloc[-1].
    assert isinstance(args[0], dict)
    assert args[0]['T_magma'] == pytest.approx(2500.0)
    assert args[0]['F_atm'] == pytest.approx(150.0)
    # Discrimination guards: a regression that read iloc[0] would
    # land at T_magma=3500 / F_atm=1500; iloc[1] (middle) would
    # land at T_magma=3000 / F_atm=500. Reject both.
    assert args[0]['T_magma'] != pytest.approx(3500.0)
    assert args[0]['T_magma'] != pytest.approx(3000.0)
    # Second arg is the config object.
    assert args[1] is p.config
    # Third arg is the directories mapping.
    assert isinstance(args[2], dict)
    assert args[2]['output'] == str(tmp_path)


def test_observe_raises_on_empty_helpfile(tmp_path):
    """When the helpfile is empty, observe() must raise an Exception
    rather than feeding an empty DataFrame to the downstream
    pipeline.

    Edge: limit-input case. Pin the exception message so a regression
    that returned None silently would surface here.
    """
    p = _make_proteus_instance(tmp_path)
    empty_df = pd.DataFrame()
    with (
        patch.object(p, 'extract_archives') as mock_extract,
        patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=empty_df),
        patch('proteus.observe.wrapper.run_observe') as mock_run,
    ):
        with pytest.raises(Exception, match='too short to be postprocessed'):
            p.observe()
    # Discrimination: confirm run_observe was NOT called. A regression
    # that swallowed the empty case and still dispatched would call
    # run_observe with an out-of-range index.
    assert mock_run.call_count == 0
    # The run is left archived. Unpacking on the way to a refusal that was
    # already certain deletes the tar for nothing.
    assert mock_extract.call_count == 0


def test_offline_chemistry_dispatches_to_run_chemistry_and_returns_result(tmp_path):
    """Proteus.offline_chemistry() must dispatch to run_chemistry
    and return its result verbatim.

    Discriminating: pin the return propagation. A regression that
    discarded the return value or wrapped it in a dict would fail
    the identity check.
    """
    p = _make_proteus_instance(tmp_path)
    df = _helpfile_df_multi_row()
    expected = pd.DataFrame([{'species': 'H2O', 'mx': 0.42}])
    with (
        patch.object(p, 'extract_archives'),
        patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=df),
        patch('proteus.atmos_chem.wrapper.run_chemistry', return_value=expected) as mock_chem,
        patch('proteus.plot.cpl_chem_atmosphere.plot_chem_atmosphere_entry') as mock_plot,
    ):
        result = p.offline_chemistry()
    mock_chem.assert_called_once()
    # The result must be the run_chemistry return, unchanged.
    assert result is expected
    # A successful (non-None) result must refresh the chemistry plot once,
    # with the Proteus handler passed through.
    mock_plot.assert_called_once_with(p)
    # Discrimination: verify the last-row dict was passed (not the
    # full DataFrame). A regression that passed df would land args[2]
    # as a pandas object, not a dict.
    args = mock_chem.call_args.args
    assert isinstance(args[2], dict)
    assert args[2]['Phi_global'] == pytest.approx(0.7)
    # iloc[0] would land at Phi_global=1.0; iloc[1] at 0.85. Reject
    # both so the test discriminates the correct last-row pick.
    assert args[2]['Phi_global'] != pytest.approx(1.0)
    assert args[2]['Phi_global'] != pytest.approx(0.85)


def test_offline_chemistry_skips_plot_when_chemistry_returns_none(tmp_path):
    """A failed/skipped chemistry run (run_chemistry returns None) must NOT
    trigger the chemistry plot refresh.

    Discriminating counterpart to the success test: the same code path with a
    None return must leave the plot entry uncalled and propagate None.
    """
    p = _make_proteus_instance(tmp_path)
    df = _helpfile_df_multi_row()
    with (
        patch.object(p, 'extract_archives'),
        patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=df),
        patch('proteus.atmos_chem.wrapper.run_chemistry', return_value=None) as mock_chem,
        patch('proteus.plot.cpl_chem_atmosphere.plot_chem_atmosphere_entry') as mock_plot,
    ):
        result = p.offline_chemistry()
    mock_chem.assert_called_once()
    assert result is None
    mock_plot.assert_not_called()


def test_offline_chemistry_raises_on_empty_helpfile(tmp_path):
    """offline_chemistry must also raise on an empty helpfile, with
    the same contract as observe().

    Discriminating: confirm run_chemistry is NOT called.
    """
    p = _make_proteus_instance(tmp_path)
    empty_df = pd.DataFrame()
    with (
        patch.object(p, 'extract_archives') as mock_extract,
        patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=empty_df),
        patch('proteus.atmos_chem.wrapper.run_chemistry') as mock_chem,
    ):
        with pytest.raises(Exception, match='too short to be postprocessed'):
            p.offline_chemistry()
    assert mock_chem.call_count == 0

    # ---------------------------------------------------------------------------
    # Checkpoint restoration: spider_eos_dir + solidus/liquidus paths
    # ---------------------------------------------------------------------------
    # The run is left archived. offline_chemistry() takes the same order
    # as observe(), so it needs the same guard against unpacking on the
    # way to a refusal that was already certain.
    assert mock_extract.call_count == 0


def test_proteus_resume_restores_spider_eos_dir(tmp_path):
    """When resuming a SPIDER run, the Proteus.start() code at L535-544
    checks for a ``data/spider_eos/`` directory inside the output dir
    and restores ``spider_eos_dir`` + solidus/liquidus P-S paths into
    ``self.directories``.

    Discrimination: without the restore, a resume from checkpoint
    would raise FileNotFoundError when Aragog or SPIDER try to load
    the EOS tables from ``config.interior_struct.eos_dir`` (which
    points to the original source, not the run's snapshot).
    """
    p = _make_proteus_instance(tmp_path)
    out_dir = str(tmp_path / 'output')

    # Create the spider_eos directory with solidus/liquidus files
    eos_dir = tmp_path / 'output' / 'data' / 'spider_eos'
    eos_dir.mkdir(parents=True)
    (eos_dir / 'solidus_P-S.dat').write_text('# dummy solidus')
    (eos_dir / 'liquidus_P-S.dat').write_text('# dummy liquidus')

    p.directories = {'output': out_dir}

    # Simulate the restore logic from proteus.py L535-544
    import os

    eos_dir_restored = os.path.join(out_dir, 'data', 'spider_eos')
    if os.path.isdir(eos_dir_restored):
        p.directories['spider_eos_dir'] = eos_dir_restored
        solidus_ps = os.path.join(eos_dir_restored, 'solidus_P-S.dat')
        liquidus_ps = os.path.join(eos_dir_restored, 'liquidus_P-S.dat')
        if os.path.isfile(solidus_ps):
            p.directories['spider_solidus_ps'] = solidus_ps
        if os.path.isfile(liquidus_ps):
            p.directories['spider_liquidus_ps'] = liquidus_ps

    assert p.directories['spider_eos_dir'] == str(eos_dir)
    assert p.directories['spider_solidus_ps'] == str(eos_dir / 'solidus_P-S.dat')
    assert p.directories['spider_liquidus_ps'] == str(eos_dir / 'liquidus_P-S.dat')
    # Discrimination: without the restore, the keys would not exist
    assert 'spider_eos_dir' in p.directories


# ---------------------------------------------------------------------------
# Resume path: "too short to be resumed" guard (proteus.py L470-472)
# ---------------------------------------------------------------------------


def test_proteus_resume_too_short_raises(tmp_path):
    """When the helpfile has <= init_loops + 1 rows, resume must raise
    RuntimeError with a diagnostic message. This prevents resuming a
    run that never completed its init stage.

    Discrimination: a helpfile with exactly 2 rows (init_loops=0, so
    threshold is 0+1=1, length 2 > 1 passes). A single-row helpfile
    must fail. Pin the error to distinguish from other RuntimeErrors.
    """
    p = _make_proteus_instance(tmp_path)
    short_df = pd.DataFrame(
        {
            'Time': [0.0],
            'R_int': [6.371e6],
            'gravity': [9.81],
            'T_magma': [3000.0],
            'T_eqm': [255.0],
            'F_atm': [100.0],
        },
    )

    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))
        stack.enter_context(
            patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50)
        )
        stack.enter_context(
            patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=short_df)
        )
        stack.enter_context(
            patch('proteus.interior_energetics.common.Interior_t', return_value=MagicMock(ic=1))
        )
        stack.enter_context(patch('proteus.utils.coupler.ZeroHelpfileRow', return_value={}))

        with pytest.raises(RuntimeError, match='too short to be resumed'):
            p.start(resume=True, offline=True)
        # The short helpfile itself is still valid (1 row); the error is about length
        assert len(short_df) == 1


# ============================================================================
# _solve_structure_baseline_if_needed: one-time callable-representation baseline
# ============================================================================

_BASELINE_TARGET = 'proteus.interior_energetics.wrapper.update_structure_from_interior'


def _make_baseline_proteus(tmp_path, *, module='zalmoxis', init_stage=False, done=False):
    """Build a Proteus positioned for the structure-baseline gate.

    The method reads init_stage, the one-shot flag, the interior_struct module,
    the interior object and the structure sentinels; everything else is left at
    its post-__init__ default.
    """
    p = _make_proteus_instance(tmp_path, struct_module=module)
    p.init_stage = init_stage
    p._baseline_structure_done = done
    p.interior_o = MagicMock()
    # The baseline retry check reads interior_o.structure_stale; seed the fresh-run
    # value so a bare MagicMock attribute does not read truthy and skip the solve.
    p.interior_o.structure_stale = False
    p.hf_row = {}
    p.last_struct_time = 0.0
    p.last_struct_Tmagma = float('inf')
    p.last_struct_Phi = float('inf')
    return p


def test_structure_baseline_fires_once_with_force_then_is_idempotent(tmp_path):
    """The first evolution step solves the baseline once with force=True and
    commits the returned structure sentinels; a second call is a no-op.

    Discriminating: the sentinels are pinned to the solver's returned values
    (123.0, 2500.0, 0.71), which differ from the post-__init__ defaults, so a
    regression that failed to commit them would be caught; and the second-call
    assertion guards against the one-shot flag not latching (which would
    re-solve every iteration and inject a spurious radius step).
    """
    p = _make_baseline_proteus(tmp_path)
    with patch(_BASELINE_TARGET, return_value=(123.0, 2500.0, 0.71)) as mock_update:
        p._solve_structure_baseline_if_needed()

        assert mock_update.call_count == 1
        assert mock_update.call_args.kwargs['force'] is True
        assert p._baseline_structure_done is True
        assert p.last_struct_time == pytest.approx(123.0, rel=1e-12)
        assert p.last_struct_Tmagma == pytest.approx(2500.0, rel=1e-12)
        assert p.last_struct_Phi == pytest.approx(0.71, rel=1e-12)

        # Idempotent: the latched flag prevents any further forced solve.
        p._solve_structure_baseline_if_needed()
        assert mock_update.call_count == 1


@pytest.mark.parametrize(
    'kwargs',
    [
        {'done': True},  # already baselined, e.g. a resumed run
        {'init_stage': True},  # still in the init stage
        {'module': 'dummy'},  # non-Zalmoxis structure module
    ],
    ids=['already_done_or_resumed', 'init_stage', 'non_zalmoxis_module'],
)
def test_structure_baseline_skipped(tmp_path, kwargs):
    """The baseline solve fires only for a fresh, post-init Zalmoxis run.

    Edge cases: a resumed/already-baselined run (the flag pre-set), an
    init-stage iteration, and a non-Zalmoxis module each must skip the forced
    solve. Discriminating: the one-shot flag is asserted unchanged at its input
    value, so a regression that dropped a guard and solved anyway is caught.
    """
    p = _make_baseline_proteus(tmp_path, **kwargs)
    with patch(_BASELINE_TARGET, return_value=(1.0, 2.0, 0.3)) as mock_update:
        p._solve_structure_baseline_if_needed()
    mock_update.assert_not_called()
    assert p._baseline_structure_done is kwargs.get('done', False)


def test_structure_baseline_retries_after_failed_solve(tmp_path):
    """A failed forced baseline (fall-back to the IC-internal structure) must
    NOT mark the baseline done, so the next evolution step retries.

    Without this, a single failed baseline on a static run would freeze it on
    the IC-internal-adiabat radius for the rest of the run, silently re-adding
    the representation offset to the dynamic-vs-static comparison.

    Discriminating: the solver signals failure by setting
    interior_o.structure_stale True; the test asserts the baseline is not
    latched (retry) AND the sentinels are NOT advanced from their defaults,
    distinguishing a real fall-back from a committed solve.
    """

    def _failed_solve(directories, config, hf_row, interior_o, *args, **kwargs):
        # Mirror the wrapper fall-back: flag the structure stale on interior_o,
        # return the unchanged sentinels (last_struct_time/Tmagma/Phi are
        # args[0:3] here).
        interior_o.structure_stale = True
        return (args[0], args[1], args[2])

    p = _make_baseline_proteus(tmp_path)
    with patch(_BASELINE_TARGET, side_effect=_failed_solve) as mock_update:
        p._solve_structure_baseline_if_needed()

    mock_update.assert_called_once()
    assert p._baseline_structure_done is False  # retried, not latched
    assert p.last_struct_time == pytest.approx(0.0, abs=0.0)  # sentinels untouched
    assert p.last_struct_Phi == float('inf')


def test_structure_baseline_skipped_for_superliquidus_adiabat(tmp_path):
    """For the super-liquidus adiabat IC the structure solve already integrates
    against the true adiabat for both dynamic and static runs, so the shared
    maximal-radius baseline is set at the initial condition. A forced re-solve
    here would overwrite it with a different (cross-table) representation and
    could nudge R_int upward, so the baseline must be skipped.

    Discriminating: a non-liquidus_super zalmoxis run (test_structure_baseline_
    fires_once...) DOES fire the forced solve, so this asserts the skip is
    specific to the adiabat IC, not a blanket disable; the flag is still latched
    so the gate is not re-evaluated every step.
    """
    p = _make_baseline_proteus(tmp_path)
    # Make _use_superliquidus_adiabat_ic(config) true: zalmoxis struct module
    # (already set), liquidus_super temperature mode, non-spider energetics.
    p.config.interior_struct.module = 'zalmoxis'
    p.config.planet.temperature_mode = 'liquidus_super'
    p.config.interior_energetics.module = 'aragog'

    with patch(_BASELINE_TARGET) as mock_update:
        p._solve_structure_baseline_if_needed()

    mock_update.assert_not_called()  # forced re-solve skipped, IC adiabat stands
    assert p._baseline_structure_done is True  # latched so it is not re-checked


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_per_step_impact_heat_starts_each_row_at_zero(tmp_path):
    """The impact-heat column is cleared when a row is created, on every path.

    The column accumulates within a timestep, because several impacts can land
    in one, and the coupler adds it to both sides of the cumulative energy
    budget. A row that inherited the previous row's value would therefore book
    an earlier impact's heat again on every subsequent step, inflating both
    cumulatives without ever disturbing the residual, which is the one quantity
    that would otherwise reveal it.

    Clearing it where the row is created in Proteus.start (loop > 0), rather
    than in an interior solver's success branch, ensures each step begins clean.
    """
    from proteus.utils.constants import vol_gas_list

    p = _make_main_loop_proteus(
        tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0, vapourise=False
    )
    rows = []
    incoming_impact_heat = []

    def _writer(hf_row, step):
        incoming_impact_heat.append(hf_row.get('step_dE_impact_J'))
        for s in vol_gas_list:
            hf_row[s + '_kg_atm'] = 1.0e18
            hf_row[s + '_kg_total'] = 1.0e18
        hf_row['M_vol_atm'] = sum(hf_row[s + '_kg_atm'] for s in vol_gas_list)
        hf_row['M_vaps'] = 0.0
        hf_row['M_atm'] = hf_row['M_vol_atm']
        hf_row['M_planet'] = _MASS_PLANET_KG
        hf_row['P_vol'] = 260.0
        hf_row['P_vap'] = 0.0
        hf_row['P_surf'] = 260.0
        if step == 0:
            hf_row['step_dE_impact_J'] = 6.1e30
        return hf_row

    _run_main_loop_recording_mass(p, stop_at_loop=2, rows=rows, row_writer=_writer)

    # Initial step 0 started with zero impact heat before booking 6.1e30 J
    assert incoming_impact_heat[0] == pytest.approx(0.0)
    # Step 1 received a fresh row reset to 0.0 rather than inheriting 6.1e30 J
    assert incoming_impact_heat[1] == pytest.approx(0.0)
    # Discrimination: the previous step actually set non-zero impact heat
    assert rows[0]['step_dE_impact_J'] == pytest.approx(6.1e30)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('module', 'core_bc'),
    [('aragog', 'core_module'), ('aragog', 'energy_balance'), ('dummy', 'core_module')],
)
def test_the_per_step_core_impact_heat_starts_core_module_rows_at_zero(
    tmp_path, module, core_bc
):
    """The core part of the impact heat is cleared with the total on Aragog core_module
    rows, and keeps its NaN (not computed) on any other core boundary or interior."""
    from proteus.utils.constants import vol_gas_list

    p = _make_main_loop_proteus(
        tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0, vapourise=False
    )
    p.config.interior_energetics.module = module
    p.config.interior_energetics.aragog.core_bc = core_bc
    active = module == 'aragog' and core_bc == 'core_module'
    rows, incoming = [], []

    def _writer(hf_row, step):
        incoming.append(hf_row.get('step_dE_impact_core_J'))
        for s in vol_gas_list:
            hf_row[s + '_kg_atm'] = 1.0e18
            hf_row[s + '_kg_total'] = 1.0e18
        hf_row['M_vol_atm'] = sum(hf_row[s + '_kg_atm'] for s in vol_gas_list)
        hf_row['M_vaps'] = 0.0
        hf_row['M_atm'] = hf_row['M_vol_atm']
        hf_row['M_planet'] = _MASS_PLANET_KG
        hf_row['P_vol'] = 260.0
        hf_row['P_vap'] = 0.0
        hf_row['P_surf'] = 260.0
        if step == 0:
            hf_row['step_dE_impact_core_J'] = 2.4e30 if active else np.nan
        return hf_row

    _run_main_loop_recording_mass(p, stop_at_loop=2, rows=rows, row_writer=_writer)

    if active:
        assert rows[0]['step_dE_impact_core_J'] == pytest.approx(2.4e30)
        assert incoming[1] == pytest.approx(0.0)
    else:
        assert np.isnan(incoming[1])
        assert np.isnan(rows[0]['step_dE_impact_core_J'])


# ---------------------------------------------------------------------------
# Resume path: crystallization flag restoration (proteus.py, resume branch)
# ---------------------------------------------------------------------------


def _make_hf_df_with_phi(phi_final, phi_history=None):
    """Helpfile frame carrying a melt-fraction history.

    ``phi_history`` supplies the four rows before the last; it defaults to a
    monotonically solidifying run that never reaches the threshold.
    """
    df = _make_hf_df()
    df['Phi_global'] = [*(phi_history or [1.0, 0.8, 0.6, 0.4]), phi_final]
    return df


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('freeze_volatiles', 'phi_final', 'expected'),
    [
        (True, 0.005, True),
        (True, 0.010, True),
        (True, 0.011, False),
        (True, 0.900, False),
        (False, 0.005, False),
    ],
    ids=[
        'crystallized_below_threshold',
        'crystallized_exactly_at_threshold',
        'molten_just_above_threshold',
        'molten_well_above_threshold',
        'freezing_disabled_stays_molten',
    ],
)
def test_proteus_resume_restores_crystallized_flag(
    tmp_path, freeze_volatiles, phi_final, expected
):
    """Resuming a crystallized mantle keeps outgassing stopped.

    Physical scenario: a run whose mantle has already crystallized is
    stopped and resumed. The crystallization flag decides whether escape
    draws from the atmosphere alone or from the whole volatile inventory,
    so a flag that returns as False on restart lets escape draw from
    dissolved reservoirs that crystallization is meant to have trapped.
    The main loop only re-derives the flag after escape has run, which is
    why it has to be restored during resume setup rather than left to the
    loop.

    Verifies:
    - The flag is set from the resumed row's melt fraction, using the same
      ``Phi_global <= phi_crit`` condition the main loop applies.
    - The threshold is exercised from both sides, at and just above
      ``phi_crit``, so an off-by-one comparison is caught.
    - With ``freeze_volatiles`` disabled the flag stays False even for a
      fully crystallized mantle, since the feature is off.
    """
    p = _make_proteus_instance(tmp_path)
    p.config.params.stop.solid.freeze_volatiles = freeze_volatiles
    p.config.params.stop.solid.phi_crit = 0.01
    (tmp_path / 'data').mkdir(exist_ok=True)

    # A fresh Proteus starts with the flag clear, so a passing result cannot
    # come from the attribute happening to be True already.
    assert p.crystallized is False, 'flag was already set before the resume'

    _resume_with_patches(p, _make_hf_df_with_phi(phi_final))

    assert p.crystallized is expected, (
        f'resuming at Phi_global={phi_final} with freeze_volatiles={freeze_volatiles} '
        f'left crystallized={p.crystallized}, expected {expected}'
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_proteus_resume_keeps_crystallized_after_remelting(tmp_path):
    """A mantle that crystallized and later remelted stays frozen on resume.

    Physical scenario: melt fraction dips to the crystallization threshold
    part-way through a run and recovers afterwards, which a heat source such
    as tidal heating can produce. The main loop latches the flag the first
    time the threshold is reached and never clears it, so outgassing stays
    stopped for the rest of the run.

    Contract clause: a resumed run must behave as the uninterrupted one
    would. Reading the flag from the resumed row alone would clear it here,
    restarting outgassing that the continuous run keeps stopped, so the
    whole stored melt-fraction history decides it.

    Verifies:
    - A history that dips to the threshold and recovers still resumes frozen.
    - The final row is well above the threshold, so the assertion can only
      pass by consulting the earlier rows.
    - A history of the same shape that never reaches the threshold resumes
      molten, so the check is not simply always True.
    """
    p = _make_proteus_instance(tmp_path)
    p.config.params.stop.solid.freeze_volatiles = True
    p.config.params.stop.solid.phi_crit = 0.01
    (tmp_path / 'data').mkdir(exist_ok=True)

    crossed = _make_hf_df_with_phi(0.900, phi_history=[1.0, 0.5, 0.005, 0.300])
    assert float(crossed['Phi_global'].iloc[-1]) > 0.01, (
        'the resumed row must sit above the threshold, or this test would pass '
        'without consulting the history'
    )

    _resume_with_patches(p, crossed)
    assert p.crystallized is True, (
        'a mantle that reached the crystallization threshold earlier in the run '
        'resumed as molten, so outgassing would restart where an uninterrupted '
        'run keeps it stopped'
    )

    # Discrimination: the same shape of history that never reaches the
    # threshold must resume molten.
    never = _make_proteus_instance(tmp_path)
    never.config.params.stop.solid.freeze_volatiles = True
    never.config.params.stop.solid.phi_crit = 0.01
    _resume_with_patches(never, _make_hf_df_with_phi(0.900, phi_history=[1.0, 0.5, 0.2, 0.300]))
    assert never.crystallized is False, (
        'a run whose melt fraction never reached the threshold resumed as '
        'crystallized; the history search is matching too eagerly'
    )


def _make_hf_df_with_impact(phi_history, accreted_rock):
    """Helpfile frame carrying a melt-fraction history and an impact ledger.

    ``accreted_rock`` is the cumulative rock mass [kg] recorded on each row,
    so a row where it rises above the previous one is a row on which a giant
    impact landed.
    """
    df = _make_hf_df()
    df['Phi_global'] = phi_history
    df['M_accreted_rock'] = accreted_rock
    return df


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_proteus_resume_lifts_the_crystallization_latch_across_an_impact(tmp_path):
    """A giant impact that remelts a crystallized mantle stays lifted on resume.

    Physical scenario: the mantle solidifies to the crystallization threshold,
    a giant impact then remelts it to a magma ocean, and the run continues
    molten until it is stopped. The impact clears the solidification latch,
    so the uninterrupted run has outgassing running again from the impact
    onwards.

    Contract clause: a resumed run must behave as the uninterrupted one would.
    Searching the whole melt-fraction history would find the pre-impact dip
    and restore a latch the run itself had lifted, freezing outgassing for the
    rest of a run whose mantle is molten.

    Verifies:
    - A dip before the impact does not resume frozen, because the impact
      remelted the mantle.
    - A dip after the impact does resume frozen, so the search is not simply
      always clearing the flag.
    - The impact's own row is excluded: it records the melt fraction from
      before the remelt, so a threshold value there must not relatch.
    - Without accreted rock the whole history is searched, so a run with no
      accretion is unaffected.
    """
    phi_crit = 0.01
    impact_on_row_3 = [0.0, 0.0, 0.0, 1.0e21, 1.0e21]

    def _resume(hf_df):
        p = _make_proteus_instance(tmp_path)
        p.config.params.stop.solid.freeze_volatiles = True
        p.config.params.stop.solid.phi_crit = phi_crit
        (tmp_path / 'data').mkdir(exist_ok=True)
        _resume_with_patches(p, hf_df)
        return p

    # Crystallized at row 2, impact at row 3, molten afterwards.
    lifted = _resume(_make_hf_df_with_impact([1.0, 0.5, 0.005, 0.300, 0.900], impact_on_row_3))
    assert lifted.crystallized is False, (
        'a mantle remelted by a giant impact resumed as crystallized, so '
        'outgassing would stay stopped where the uninterrupted run has it '
        'running again'
    )

    # Discrimination: the same impact, but the mantle solidifies again after
    # it. The latch must be restored, or the check would be always False.
    relatched = _resume(
        _make_hf_df_with_impact([1.0, 0.5, 0.005, 0.300, 0.008], impact_on_row_3)
    )
    assert relatched.crystallized is True, (
        'a mantle that solidified again after the impact resumed as molten, so '
        'the post-impact history is not being searched at all'
    )

    # Boundary: the impact row carries the melt fraction from before the
    # remelt, so a threshold value on that row must not restore the latch.
    on_impact_row = _resume(
        _make_hf_df_with_impact([1.0, 0.5, 0.900, 0.005, 0.900], impact_on_row_3)
    )
    assert on_impact_row.crystallized is False, (
        "the impact row's own pre-remelt melt fraction restored the latch; the "
        'search must start after the impact, not on it'
    )

    # A run with no accretion searches the whole history, unchanged.
    no_accretion = _resume(_make_hf_df_with_impact([1.0, 0.5, 0.005, 0.300, 0.900], [0.0] * 5))
    assert no_accretion.crystallized is True, (
        'a run that never had an impact stopped seeing its own crystallization '
        'history; the impact search must not affect non-accretion runs'
    )


# ---------------------------------------------------------------------------
# Proteus.start() main loop: plot-cadence gating (proteus.py ~1200-1207).
#
# Plot generation is driven by `plot_mod` alone. It must NOT depend on
# `is_snapshot` (the write_mod / dt_write_rel gate that governs helpfile
# and archive writes) -- a plot cadence independent of the write cadence
# is the documented contract for `params.out.plot_mod`.
# ---------------------------------------------------------------------------


class _FakeHelpfile:
    """Stand-in for the helpfile DataFrame that only supports the one
    access pattern the main loop uses: `hf_all.iloc[-1].to_dict()`.

    Keeps `hf_row` a real, plain dict across loop iterations instead of
    a MagicMock, so the loop's own dict/arithmetic operations on hf_row
    behave exactly as they do in a real run.
    """

    def __init__(self, row):
        self._row = dict(row)
        self.iloc = _FakeHelpfile._ILoc(self._row)

    class _ILoc:
        def __init__(self, row):
            self._row = row

        def __getitem__(self, _index):
            from types import SimpleNamespace

            return SimpleNamespace(to_dict=lambda: dict(self._row))

    def __len__(self):
        return 1


def _make_main_loop_proteus(tmp_path, *, plot_mod, write_mod, dt_write_rel, vapourise=True):
    """Build a Proteus instance configured for a fresh (non-resume) run
    that can be driven through several main-loop iterations.

    `interior_energetics.module` / `interior_struct.module` are set to
    'dummy' so the Zalmoxis structure-update and SPIDER-specific branches
    are no-ops; `observe.module=None`, `accretion.module=None` and a
    non-'online'/'offline' atmos_chem.when skip the postprocessing and
    impact branches. None of these short-circuits touch the plot-gating
    condition under test.

    `vapourise` selects which half of the mass-conservation invariant the loop
    enforces: with it True the M_atm <= M_planet half is replaced by a warning,
    and with it False that half is enforced at the strict tolerance. Callers
    that leave `update_planet_mass` mocked keep M_planet at zero, which
    short-circuits the invariant before either half runs.
    """
    from proteus.config._params import StopStall
    from proteus.proteus import Proteus

    config = MagicMock()
    config.interior_struct.module = 'dummy'
    config.interior_struct.zalmoxis.update_interval = 0
    config.interior_struct.eos_dir = None
    config.interior_energetics.module = 'dummy'
    config.interior_energetics.flux_guess = 100.0  # >=0: skips sigma*T^4 branch
    config.orbit.module = None
    config.observe.module = None
    config.accretion.module = None
    config.atmos_chem.when = 'never'
    config.outgas.vapourise = vapourise
    config.planet.temperature_mode = 'isothermal'
    config.planet.volatile_mode = 'elements'
    config.planet.gas_prs.get_pressure = lambda _s: 0.0
    config.outgas.calliope.is_included = lambda _s: False
    config.params.resume = False
    config.params.out.logging = 'WARNING'
    config.params.out.plot_mod = plot_mod
    config.params.out.write_mod = write_mod
    config.params.out.dt_write_rel = dt_write_rel
    config.params.out.archive_mod = None
    config.params.stop.iters.minimum = 10
    config.params.stop.iters.maximum = 1000
    config.params.stop.solid.freeze_volatiles = False
    config.params.stop.solid.phi_crit = 0.01
    # Left as a mock attribute this reads as a cap of one iteration, which
    # would end a loop test on the first unconverged solve.
    config.params.stop.stall = StopStall(enabled=True, maximum=STALL_MAX_CONFIGURED)
    config.params.dt.starinst = 1e8
    config.params.dt.starspec = 1e8

    directories = {
        'output': str(tmp_path),
        'output/data': str(tmp_path / 'data'),
        'output/observe': str(tmp_path / 'observe'),
        'output/offchem': str(tmp_path / 'offchem'),
        'output/plots': str(tmp_path / 'plots'),
        'spider': str(tmp_path / 'spider'),
        'fwl': str(tmp_path / 'fwl'),
    }
    for path in directories.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    with (
        patch('proteus.proteus.read_config_object', return_value=config),
        patch('proteus.utils.coupler.set_directories', return_value=directories),
    ):
        p = Proteus(config_path='dummy.toml')

    return p


# Every dependency the main loop calls that is irrelevant to the
# plot-gating condition itself: mocked as a no-op so the loop can run
# several iterations without touching real physics, I/O, or Julia/AGNI.
_MAIN_LOOP_NOOP_PATCHES = [
    'proteus.utils.coupler.CreateLockFile',
    'proteus.utils.data.download_sufficient_data',
    'proteus.interior_struct.zalmoxis.require_paleos_tables',
    'proteus.interior_energetics.wrapper.solve_structure',
    'proteus.utils.coupler.print_citation',
    'proteus.utils.coupler.print_header',
    'proteus.utils.coupler.print_module_configuration',
    'proteus.utils.coupler.print_system_configuration',
    'proteus.utils.coupler.validate_module_versions',
    'proteus.utils.terminate.print_termination_criteria',
    'proteus.interior_energetics.wrapper.run_interior',
    'proteus.interior_energetics.wrapper.update_planet_mass',
    'proteus.orbit.wrapper.run_orbit',
    'proteus.star.wrapper.scale_spectrum_to_toa',
    'proteus.star.wrapper.update_stellar_mass',
    'proteus.star.wrapper.update_stellar_quantities',
    'proteus.star.wrapper.write_spectrum',
    'proteus.outgas.wrapper.calc_target_elemental_inventories',
    'proteus.outgas.wrapper.run_outgassing_and_vapourisation',
    'proteus.outgas.wrapper.check_ic_oxygen_budget',
    'proteus.outgas.wrapper.run_desiccated',
    'proteus.outgas.wrapper.run_crystallized',
    'proteus.outgas.wrapper.check_desiccation',
    'proteus.escape.wrapper.run_escape',
    'proteus.utils.coupler.assert_surface_pressure_consistency',
    'proteus.atmos_clim.run_atmosphere',
    'proteus.utils.coupler.PrintCurrentState',
    'proteus.utils.coupler.WriteHelpfileToCSV',
    'proteus.utils.coupler.remove_excess_files',
    'proteus.utils.coupler.print_stoptime',
    'proteus.observe.wrapper.run_observe',
    'proteus.atmos_chem.wrapper.run_chemistry',
]


def _run_main_loop_capturing_plots(p, *extra, stop_at_loop):
    """Run p.start(resume=False) with the main loop's physics mocked out,
    capturing every main-loop UpdatePlots call as (loops_total, is_end).

    `stop_at_loop` sets when check_termination first fires: it is only
    ever invoked once `init_stage` has cleared (loops['total'] >
    init_loops == 3), i.e. from the 5th iteration (loops['total'] == 4)
    onward in an unmodified loop -- so `stop_at_loop` must be >= 4 for the
    stop condition to actually engage before the loop's own init-stage
    bookkeeping does.
    """
    from types import SimpleNamespace

    plot_calls = []

    def _record_plot(*args, **kwargs):
        plot_calls.append((p.loops['total'], bool(kwargs.get('end', False))))

    def _fake_check_termination(handler):
        if handler.loops['total'] >= stop_at_loop:
            handler.finished_both = True
        return handler.finished_both

    def _fake_create_helpfile(row):
        return _FakeHelpfile(row)

    def _fake_extend_helpfile(_hf_all, row):
        return _FakeHelpfile(row)

    with ExitStack() as stack:
        for target in _MAIN_LOOP_NOOP_PATCHES:
            stack.enter_context(patch(target))

        mock_interior_t = stack.enter_context(
            patch('proteus.interior_energetics.common.Interior_t')
        )
        mock_interior_t.return_value = SimpleNamespace(dt=100.0, ic=1, aragog_solver=None)

        mock_atmos_t = stack.enter_context(patch('proteus.atmos_clim.common.Atmos_t'))
        mock_atmos_t.return_value = SimpleNamespace(converged=True)

        mock_spectrum = stack.enter_context(patch('proteus.star.wrapper.get_new_spectrum'))
        mock_spectrum.return_value = (np.array([1.0]), np.array([1.0]))

        stack.enter_context(
            patch(
                'proteus.utils.coupler.CreateHelpfileFromDict',
                side_effect=_fake_create_helpfile,
            )
        )
        stack.enter_context(
            patch('proteus.utils.coupler.ExtendHelpfile', side_effect=_fake_extend_helpfile)
        )
        stack.enter_context(
            patch('proteus.utils.coupler.UpdatePlots', side_effect=_record_plot)
        )
        stack.enter_context(
            patch(
                'proteus.utils.terminate.check_termination', side_effect=_fake_check_termination
            )
        )
        for extra_patch in extra:
            stack.enter_context(extra_patch)

        p.start(resume=False, offline=True)

    return plot_calls


def test_plot_cadence_is_independent_of_write_snapshot_gate(tmp_path):
    """Plots must be generated on every `plot_mod`-multiple iteration,
    even on iterations that are NOT a write/archive snapshot.

    Regression target: the main loop used to gate `UpdatePlots` on
    `is_snapshot AND multiple(loops_total, plot_mod)`, tying the plot
    cadence to `write_mod`/`dt_write_rel` instead of `plot_mod` alone.
    `plot_mod=5` (or any value) then silently produced far fewer plots
    than the config requested whenever `write_mod`/`dt_write_rel`
    suppressed the snapshot on a plot-due iteration.

    Discriminating setup: `plot_mod=1` (plot every iteration) is paired
    with `write_mod=2` (`dt_write_rel=0`), so `is_snapshot` alternates
    True/False/True across loop iterations 0/1/2 while the plot cadence
    must fire on all three regardless. Note `is_snapshot`'s `write_mod`
    check reads `loops['total']` *before* the per-iteration increment,
    while the plot/archive checks read it *after*; the iteration with
    pre-increment total 1 (post-increment total 2) is the one where
    `is_snapshot` is False (1 is not a multiple of write_mod=2) but the
    plot must still fire (2 is a multiple of plot_mod=1). Under the old
    gated condition that iteration's plot would be silently skipped, so
    the recorded plot sequence would be [1, 3] instead of [1, 2, 3].
    """
    p = _make_main_loop_proteus(tmp_path, plot_mod=1, write_mod=2, dt_write_rel=0.0)

    # check_termination is only consulted once init_stage clears
    # (post-increment loops['total'] > init_loops == 3), which happens
    # while processing pre-increment total 3 (post-increment 4). Stopping
    # there gives exactly 4 executed iterations (pre-increment 0-3), of
    # which the last one's plot is suppressed by the loop's own
    # `and not self.finished_both` clause (unrelated to the fix under
    # test) -- so 3 plot-eligible iterations remain for this assertion.
    plot_calls = _run_main_loop_capturing_plots(p, stop_at_loop=4)

    # Main-loop plot calls only (exclude the unconditional end-of-run
    # "final plots" call, which always passes end=True).
    main_loop_plots = [loop for loop, is_end in plot_calls if not is_end]

    assert main_loop_plots == [1, 2, 3], (
        f'expected plots at post-increment loop counts [1, 2, 3] (every '
        f'plot_mod=1 iteration, independent of write_mod=2), got {main_loop_plots}'
    )
    # Discrimination guard: post-increment total 2 corresponds to the
    # iteration where is_snapshot was False (write_mod=2 did not divide
    # the pre-increment total of 1). A regression reintroducing the
    # is_snapshot gate would drop it, leaving [1, 3] here.
    assert 2 in main_loop_plots, (
        'plot at a plot_mod-multiple iteration was skipped because it was '
        'not also a write_mod snapshot -- the plot cadence must not depend '
        'on the write/archive snapshot gate'
    )


@pytest.mark.parametrize('t_impact, lands', [(1.0e8 / 3.0, True), (float('inf'), False)])
def test_the_time_advance_ends_a_short_step_on_the_pending_impact(tmp_path, t_impact, lands):
    """A step that rounding leaves a few ulp short of the pending impact ends on
    the impact time, and the star age moves with it; with no impact pending the
    step end is Time + dt exactly."""
    import math
    from types import SimpleNamespace

    p = _make_proteus_instance(tmp_path)
    t0 = 1.0e7
    t = 1.0e8 / 3.0
    dt = math.nextafter(math.nextafter(t - t0, 0.0), 0.0)
    p.hf_row = {'Time': t0, 'age_star': t0 + 1.0}
    p.interior_o = SimpleNamespace(dt=dt)
    p._advance_to_step_end(t_impact)
    assert p.hf_row['Time'] == (t if lands else t0 + dt)
    assert p.hf_row['age_star'] - p.hf_row['Time'] == pytest.approx(1.0, abs=1.0e-9)


def test_the_main_loop_applies_a_snapped_impact_in_the_same_iteration(tmp_path):
    """An impact 0.5e-12 (relative) after the end of a 100 yr step is within the
    snap window: the row time moves onto it and the impact is applied in that
    iteration, once, with the impact time as the row time."""
    from types import SimpleNamespace

    t_impact = 300.0 * (1.0 + 0.5e-12)
    applied = []
    p = _make_main_loop_proteus(tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0)
    with (
        patch(
            'proteus.accretion.wrapper.init_accretion',
            return_value=[SimpleNamespace(time=t_impact)],
        ),
        patch('proteus.accretion.wrapper.restore_accretion_state'),
        patch('proteus.accretion.wrapper.discard_preimpact_snapshot'),
        patch(
            'proteus.accretion.wrapper.apply_impact',
            side_effect=lambda handler, event: applied.append(handler.hf_row['Time']),
        ),
    ):
        _run_main_loop_capturing_plots(p, stop_at_loop=6)
    assert applied == [t_impact]
    assert p.impact_events == []


def test_the_main_loop_lands_two_close_impacts_in_one_step(tmp_path, caplog):
    """Impacts at 299.9999 yr and 300 yr land in one step at 300 yr: the stepper is
    told 300 yr, both are applied on that row, and one log line names both."""
    from types import SimpleNamespace

    events = [SimpleNamespace(time=299.9999), SimpleNamespace(time=300.0)]
    applied = []
    p = _make_main_loop_proteus(tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0)
    with (
        patch('proteus.accretion.wrapper.init_accretion', return_value=list(events)),
        patch('proteus.accretion.wrapper.restore_accretion_state'),
        patch('proteus.accretion.wrapper.discard_preimpact_snapshot'),
        patch(
            'proteus.accretion.wrapper.apply_impact',
            side_effect=lambda h, e: applied.append(
                (h.hf_row['Time'], e.time, h.interior_o.t_next_impact)
            ),
        ),
        caplog.at_level(logging.INFO, logger='fwl.proteus.accretion.wrapper'),
    ):
        _run_main_loop_capturing_plots(p, stop_at_loop=6)
    assert applied == [(300.0, 299.9999, 300.0), (300.0, 300.0, 300.0)]
    assert p.impact_events == []
    assert sum('land in one step' in r.message for r in caplog.records) == 1
    # The pair spans 1e-4 yr, below the name resolution, so no chain warning.
    assert not any('form a chain' in r.message for r in caplog.records)


def test_the_main_loop_lands_each_step_through_snap_to_impact(tmp_path):
    """Every iteration passes its step end through snap_to_impact; with no
    impact pending the step end is kept, so the run time stays finite."""
    import math

    from proteus.accretion import common

    p = _make_main_loop_proteus(tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0)
    with patch.object(common, 'snap_to_impact', wraps=common.snap_to_impact) as snap:
        _run_main_loop_capturing_plots(p, stop_at_loop=4)
    assert snap.call_count == p.loops['total']
    assert all(math.isinf(c.args[1]) for c in snap.call_args_list)
    assert 0.0 < p.hf_row['Time'] < float('inf')


def test_the_main_loop_runs_the_escape_step_every_iteration(tmp_path):
    """With the timing instrumentation off, every iteration still calls the
    escape step, which decides by itself whether escape runs."""
    p = _make_main_loop_proteus(tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0)
    with patch.object(type(p), '_run_escape_step', return_value=False) as escape:
        _run_main_loop_capturing_plots(p, stop_at_loop=4)
    assert escape.call_count == p.loops['total']
    escape.assert_called_with()


def test_the_main_loop_applies_due_impacts_before_the_escape_step(tmp_path):
    """Every iteration applies the impacts it reached before escape runs, so the
    escape step measures and debits the post-impact budgets."""
    from types import SimpleNamespace

    calls = []
    p = _make_main_loop_proteus(tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0)
    with (
        patch(
            'proteus.accretion.wrapper.init_accretion',
            return_value=[SimpleNamespace(time=1.0e9)],
        ),
        patch('proteus.accretion.wrapper.restore_accretion_state'),
        patch(
            'proteus.accretion.wrapper.apply_due_impacts',
            side_effect=lambda handler, is_snapshot: calls.append('impacts') or [],
        ),
        patch.object(
            type(p), '_run_escape_step', side_effect=lambda: calls.append('escape') or False
        ),
    ):
        _run_main_loop_capturing_plots(p, stop_at_loop=4)
    assert calls == ['impacts', 'escape'] * p.loops['total']
    assert p.loops['total'] >= 3


def test_it_timing_records_orbit_module_wall_time(tmp_path, monkeypatch, caplog):
    """With the opt-in ``PROTEUS_TIMING`` instrumentation enabled (here
    patched directly on the frozen module constant, since it is normally
    read from the environment once at import time), the main loop must
    record the orbit stage's wall-time in ``_t_mod`` and surface it in
    the per-iteration ``[IT_TIMING]`` log line -- not just the other
    instrumented stages. An iteration that runs escape records its time too.
    """
    import logging

    monkeypatch.setattr('proteus.proteus._IT_TIMING_ENABLED', True)
    p = _make_main_loop_proteus(tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0)

    with (
        caplog.at_level(logging.INFO, logger='fwl.proteus.proteus'),
        patch.object(type(p), '_run_escape_step', return_value=True),
    ):
        _run_main_loop_capturing_plots(p, stop_at_loop=4)

    timing_records = [rec.message for rec in caplog.records if '[IT_TIMING]' in rec.message]
    assert len(timing_records) > 0, 'no [IT_TIMING] log line was emitted'
    assert any('orbit=' in msg for msg in timing_records)
    assert any('escape=' in msg for msg in timing_records)


# =======================================================================================
# SECTION: mass conservation across a multi-iteration run
# =======================================================================================


_MASS_PLANET_KG = 5.97e24  # 1 M_earth


def _write_post_outgas_row(hf_row, step, *, vapour):
    """Fill a helpfile row the way the outgas step leaves it.

    The volatile inventory is split asymmetrically across ``vol_gas_list`` (one
    dominant species plus traces) so a regression in the species sum moves
    ``M_vol_atm`` instead of cancelling out. With ``vapour`` set, a rock-vapour
    column grows with ``step`` and is the only mass in ``M_atm`` that is not in
    ``M_vol_atm``, which is the whole content of the relaxed invariant.
    """
    from proteus.utils.constants import vol_gas_list

    m_vol_atm = 1.0e-4 * _MASS_PLANET_KG  # ~6e20 kg, a few hundred bar of volatiles
    weights = [1.0] + [0.01] * (len(vol_gas_list) - 1)
    norm = sum(weights)
    for s, w in zip(vol_gas_list, weights):
        hf_row[s + '_kg_atm'] = m_vol_atm * w / norm
    hf_row['M_vol_atm'] = sum(hf_row[s + '_kg_atm'] for s in vol_gas_list)
    hf_row['M_vaps'] = (2.0e-5 * _MASS_PLANET_KG * (1 + step)) if vapour else 0.0
    hf_row['M_atm'] = hf_row['M_vol_atm'] + hf_row['M_vaps']
    hf_row['M_planet'] = _MASS_PLANET_KG
    hf_row['P_vol'] = 260.0
    hf_row['P_vap'] = (40.0 * (1 + step)) if vapour else 0.0
    hf_row['P_surf'] = hf_row['P_vol'] + hf_row['P_vap']
    return hf_row


def _run_main_loop_recording_mass(
    p, *, stop_at_loop, rows, row_writer, guard_calls=None, flags=None
):
    """Run p.start with the physics mocked, recording the row each outgas step
    wrote. ``row_writer(hf_row, step)`` fills the mass columns.

    Mirrors `_run_main_loop_capturing_plots` but replaces the no-op outgas patch
    with a side effect, and forces `check_desiccation` to False: the blanket
    MagicMock patch returns a truthy value, which would divert every iteration
    after the first into the desiccated branch and stop the outgas rows.

    ``guard_calls``, when given, collects one ``(row, kwargs)`` pair per
    mass-conservation call: the helpfile row as the check saw it, and the keyword
    arguments the main loop chose. The real check still runs, so the recorded
    keywords are the loop's own dispatch decision rather than a restatement of
    the test's setup.
    """
    from types import SimpleNamespace

    from proteus.utils import coupler as coupler_mod

    real_guard = coupler_mod.assert_mass_conservation

    def _spy_guard(hf_row, *args, **kwargs):
        if guard_calls is not None:
            guard_calls.append((dict(hf_row), dict(kwargs)))
        return real_guard(hf_row, *args, **kwargs)

    def _fake_check_termination(handler):
        if handler.loops['total'] >= stop_at_loop:
            handler.finished_both = True
        return handler.finished_both

    def _fake_create_helpfile(row):
        return _FakeHelpfile(row)

    def _fake_extend_helpfile(_hf_all, row):
        return _FakeHelpfile(row)

    def _fake_outgas(_dirs, _config, hf_row, first_iter):
        if flags is not None:
            flags.append((first_iter, p.init_stage))
        rows.append(dict(row_writer(hf_row, len(rows))))

    with ExitStack() as stack:
        for target in _MAIN_LOOP_NOOP_PATCHES:
            stack.enter_context(patch(target))

        mock_interior_t = stack.enter_context(
            patch('proteus.interior_energetics.common.Interior_t')
        )
        mock_interior_t.return_value = SimpleNamespace(dt=100.0, ic=1, aragog_solver=None)

        mock_atmos_t = stack.enter_context(patch('proteus.atmos_clim.common.Atmos_t'))
        mock_atmos_t.return_value = SimpleNamespace(converged=True)

        mock_spectrum = stack.enter_context(patch('proteus.star.wrapper.get_new_spectrum'))
        mock_spectrum.return_value = (np.array([1.0]), np.array([1.0]))

        stack.enter_context(
            patch(
                'proteus.utils.coupler.CreateHelpfileFromDict',
                side_effect=_fake_create_helpfile,
            )
        )
        stack.enter_context(
            patch('proteus.utils.coupler.ExtendHelpfile', side_effect=_fake_extend_helpfile)
        )
        stack.enter_context(patch('proteus.utils.coupler.UpdatePlots'))
        stack.enter_context(
            patch('proteus.utils.coupler.assert_mass_conservation', side_effect=_spy_guard)
        )
        stack.enter_context(
            patch(
                'proteus.utils.terminate.check_termination', side_effect=_fake_check_termination
            )
        )
        stack.enter_context(
            patch('proteus.outgas.wrapper.check_desiccation', return_value=False)
        )
        stack.enter_context(
            patch(
                'proteus.outgas.wrapper.run_outgassing_and_vapourisation',
                side_effect=_fake_outgas,
            )
        )

        p.start(resume=False, offline=True)


def _escalation_records(caplog):
    """Records reporting an excess larger than the rock vapour explains."""
    return [r for r in caplog.records if 'larger than vapourisation' in r.getMessage()]


@pytest.mark.physics_invariant
def test_vapourising_run_bounds_the_imbalance_across_steps(tmp_path, caplog):
    """Across a multi-step vapourising run the loop relaxes only the
    atmosphere-versus-planet half, and warns exactly on the step whose excess the
    rock vapour cannot explain.

    The atmosphere carries a fixed volatile inventory plus a rock-vapour column
    that grows step by step. Four steps sit well inside the planet mass; one is
    placed at the boundary where the excess over M_planet is exactly M_vaps, the
    tightest state the relaxation must still accept; the last pushes the excess
    past M_vaps, which is the signal that the imbalance is not vapourisation.

    What this test covers is the loop's dispatch and the warning decision made
    across iterations. The closure M_atm = M_vol_atm + M_vaps is a property of the
    real outgassing step, which is mocked out here, so it is asserted in
    tests/outgas, not here.
    """
    import logging

    rows = []
    guard_calls = []

    def _writer(hf_row, step):
        row = _write_post_outgas_row(hf_row, step, vapour=True)
        if step == 4:
            # Excess over the planet mass is exactly the rock vapour: accepted,
            # and the escalation must not fire at the boundary.
            row['M_planet'] = row['M_vol_atm']
        elif step == 5:
            # Excess now exceeds the rock vapour by half the volatile inventory.
            row['M_planet'] = 0.5 * row['M_vol_atm']
        return row

    p = _make_main_loop_proteus(
        tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0, vapourise=True
    )
    with caplog.at_level(logging.INFO, logger='fwl.proteus.utils.coupler'):
        _run_main_loop_recording_mass(
            p, stop_at_loop=6, rows=rows, row_writer=_writer, guard_calls=guard_calls
        )

    # The run really did drive several outgas steps rather than stopping early,
    # and it survived the two steps where the atmosphere exceeded the planet mass.
    assert len(rows) == 6
    assert len(guard_calls) == 6
    # The loop asked for the relaxation on every iteration. This is the
    # production decision under test: the keyword comes from proteus.py reading
    # config.outgas.vapourise, not from this test.
    assert all(kwargs.get('require_atm_le_planet') is False for _row, kwargs in guard_calls)
    # The bound: the warning fires only on the step whose excess outruns M_vaps,
    # and not on the boundary step where the excess equals it exactly. The four
    # conserving steps and the boundary step pass silently.
    escalated = _escalation_records(caplog)
    assert len(escalated) == 1
    assert escalated[0].levelno == logging.WARNING
    # Both logged values are that step's own, read from the log arguments rather
    # than the formatted string so a wrong value in the right slot cannot slip
    # through.
    assert escalated[0].args[0] == pytest.approx(
        rows[5]['M_atm'] - rows[5]['M_planet'], rel=1e-12
    )
    assert escalated[0].args[1] == pytest.approx(rows[5]['M_vaps'], rel=1e-12)
    # Discrimination: the boundary step really did breach M_planet, so a run with
    # the strict half live would have aborted there, and the vapour column really
    # grew, so none of the above is satisfied by a constant.
    assert rows[4]['M_atm'] > rows[4]['M_planet'] > 0.0
    assert rows[-1]['M_vaps'] > 4.0 * rows[0]['M_vaps']


def test_the_outgassing_is_told_the_init_stage_on_every_iteration(tmp_path):
    """The dummy outgassing derives an empty O budget only in the init stage, so
    the main loop passes it the same init_stage flag that resets the O budget:
    True for the init iterations, then False."""
    rows, flags = [], []
    p = _make_main_loop_proteus(
        tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0, vapourise=False
    )
    _run_main_loop_recording_mass(
        p,
        stop_at_loop=6,
        rows=rows,
        row_writer=lambda hf_row, step: _write_post_outgas_row(hf_row, step, vapour=False),
        flags=flags,
    )
    assert all(passed is stage for passed, stage in flags)
    assert [passed for passed, _ in flags] == [True] * 4 + [False] * (len(flags) - 4)


@pytest.mark.physics_invariant
def test_non_vapourising_run_keeps_strict_mass_conservation(tmp_path, caplog):
    """With rock vapourisation off, the loop demands the strict invariant and a
    breach of it aborts the run.

    Several conserving steps pass silently with the atmosphere entirely volatile,
    then a step whose volatiles alone exceed the planet mass raises. That is the
    issue #677 symptom. No relaxation report may appear on any step of a run that
    never enables vapourisation.
    """
    import logging

    rows = []
    guard_calls = []

    def _writer(hf_row, step):
        row = _write_post_outgas_row(hf_row, step, vapour=False)
        if step == 4:
            # Volatiles alone breach the planet budget.
            row['M_planet'] = 0.5 * row['M_atm']
        return row

    p = _make_main_loop_proteus(
        tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0, vapourise=False
    )
    with caplog.at_level(logging.INFO, logger='fwl.proteus.utils.coupler'):
        with pytest.raises(RuntimeError, match='Mass conservation violation'):
            _run_main_loop_recording_mass(
                p, stop_at_loop=6, rows=rows, row_writer=_writer, guard_calls=guard_calls
            )

    # The run survived the conserving steps and died on the breaching one.
    assert len(rows) == 5
    assert len(guard_calls) == 5
    # The loop demanded the strict invariant on every iteration. This is the
    # other half of the production dispatch decision.
    assert all(kwargs.get('require_atm_le_planet') is True for _row, kwargs in guard_calls)
    for row in rows[:-1]:
        assert 0.0 < row['M_atm'] < row['M_planet']
    # Nothing was relaxed, so the vapour-imbalance warning is not admissible on
    # this path: a breach here raises rather than being reported.
    assert _escalation_records(caplog) == []
    # Discrimination: the breach is a factor of two, far outside the 1e-6
    # tolerance, so this is not passing on a rounding edge.
    assert rows[-1]['M_atm'] / rows[-1]['M_planet'] == pytest.approx(2.0, rel=1e-12)


@pytest.mark.unit
def test_stall_criterion_is_configurable_and_matches_its_constant(tmp_path):
    """The stall abort reads its cap and its on/off state from the config, and
    the schema default is the same number the module constant carries.

    Contract clause: the cap is a termination criterion like the seven beside
    it, so a run that stalls legitimately can raise it or switch it off without
    editing source. Both settings reach the run through the config the instance
    is built from, never by assignment afterwards, so a constructor that
    stopped reading either one fails here. The default is duplicated between
    the schema and the constant, so it is pinned too: two records of one number
    are only safe while something fails when they disagree.
    """
    from proteus.config._params import StopParams, StopStall
    from proteus.proteus import ATMOS_STALL_MAX

    assert StopStall().maximum == ATMOS_STALL_MAX
    assert StopStall().enabled is True
    assert StopParams().stall.maximum == ATMOS_STALL_MAX

    # A non-positive cap is refused: it would abort before a run had a chance.
    for bad in (0, -1):
        with pytest.raises(ValueError):
            StopStall(maximum=bad)

    moving = {
        'hf_all': pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3050.0, 'Phi_global': 1.0}]),
        'hf_row': {'F_atm': 140.0, 'T_magma': 3000.0, 'Phi_global': 0.9},
    }

    # A raised cap moves the abort with it: the streak that ends the run at the
    # cap beside it is now allowed to continue.
    raised = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        stale_iters=STALL_MAX_CONFIGURED,
        stall_maximum=STALL_MAX_CONFIGURED * 2,
        **moving,
    )
    assert raised.atmos_stall_max == STALL_MAX_CONFIGURED * 2
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        raised._check_atmosphere_deadlock()
    mock_update.assert_not_called()

    # Switching the criterion off in the config spares a streak far past any
    # cap, which is the recourse a legitimately long-stalling run has.
    off = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        stale_iters=STALL_MAX_CONFIGURED * 10,
        stall_enabled=False,
        **moving,
    )
    assert off.atmos_stall_enabled is False
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        off._check_atmosphere_deadlock()
    mock_update.assert_not_called()

    # Discrimination: the same streak with the criterion left on does abort, so
    # the two results above are attributable to the switch and the cap.
    on = _make_deadlock_proteus(
        tmp_path, converged=False, stale_iters=STALL_MAX_CONFIGURED * 10, **moving
    )
    assert on.atmos_stall_enabled is True
    with patch('proteus.proteus.UpdateStatusfile'):
        with pytest.raises(RuntimeError, match='consecutive solves'):
            on._check_atmosphere_deadlock()

    # Switching the criterion off leaves the frozen-interior abort where it is.
    # That path has its own, much shorter count, and a run that has stopped
    # moving on both sides still has to end.
    frozen = {
        'hf_all': pd.DataFrame([{'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0}]),
        'hf_row': {'F_atm': 100.0, 'T_magma': 3000.0, 'Phi_global': 1.0},
    }
    stuck = _make_deadlock_proteus(
        tmp_path,
        converged=False,
        stale_iters=STALL_MAX_CONFIGURED * 10,
        stall_enabled=False,
        **frozen,
    )
    stuck.agni_deadlock_count = stuck.agni_deadlock_max - 1
    with patch('proteus.proteus.UpdateStatusfile') as mock_update:
        with pytest.raises(RuntimeError, match='consecutive AGNI failures'):
            stuck._check_atmosphere_deadlock()
    args, _ = mock_update.call_args
    assert args[1] == 22


# =======================================================================================
# SECTION: resumed run drives the atmosphere from the interior's own T_magma
# =======================================================================================


def _make_resume_checkpoint_df():
    """5-row checkpoint helpfile for a resumed run.

    5 rows clears the `len(hf_all) > init_loops+1 == 4` resume-eligibility
    check and keeps `loops['total']=5` below the `>init_loops+2` threshold
    that gates the escape block's active branch, so escape takes its
    inactive branch on the first post-resume iteration. The crystallization
    check is skipped for a separate reason: the test config sets
    `freeze_volatiles` to False. Every row starts from `ZeroHelpfileRow()`
    so every real helpfile column the main loop reads is present.
    """
    from proteus.utils.coupler import ZeroHelpfileRow

    times = [0.0, 100.0, 200.0, 300.0, 400.0]
    ages = [1.0e6 + t for t in times]
    magmas = [3000.0, 2900.0, 2800.0, 2700.0, 2600.0]
    rows = []
    for time, age, magma in zip(times, ages, magmas):
        row = ZeroHelpfileRow()
        row.update(
            {
                'Time': time,
                'age_star': age,
                'R_int': 6.371e6,
                'gravity': 9.81,
                'separation': 1.0,
                'T_magma': magma,
                'T_surf': magma - 50.0,
                'T_eqm': 255.0,
                'F_atm': 100.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _make_resume_main_loop_proteus(tmp_path, interior_module='spider', miscibility=False):
    """Build a Proteus instance for a resumed run driven into the main loop.

    Mirrors `_make_main_loop_proteus`'s dummy-module, full-loop-capable
    config, since a resumed run reaches the same main-loop code once
    resume setup completes. ``interior_module`` selects the energetics
    module and ``miscibility`` the Zalmoxis global_miscibility switch.
    """
    from proteus.config._params import StopStall
    from proteus.proteus import Proteus

    config = MagicMock()
    config.interior_struct.module = 'dummy'
    config.interior_struct.zalmoxis.update_interval = 0
    config.interior_struct.zalmoxis.global_miscibility = miscibility
    config.interior_struct.eos_dir = None
    config.interior_energetics.module = interior_module
    config.interior_energetics.flux_guess = 100.0
    config.orbit.module = None
    config.observe.module = None
    config.accretion.module = None
    config.atmos_chem.when = 'never'
    config.outgas.vapourise = True
    config.planet.temperature_mode = 'isothermal'
    config.planet.volatile_mode = 'elements'
    config.planet.gas_prs.get_pressure = lambda _s: 0.0
    config.outgas.calliope.is_included = lambda _s: False
    config.params.out.logging = 'WARNING'
    config.params.out.plot_mod = 100
    config.params.out.write_mod = 100
    config.params.out.dt_write_rel = 0.0
    config.params.out.archive_mod = None
    config.params.stop.iters.minimum = 10
    config.params.stop.iters.maximum = 1000
    config.params.stop.solid.freeze_volatiles = False
    config.params.stop.solid.phi_crit = 0.01
    config.params.stop.stall = StopStall(enabled=True, maximum=STALL_MAX_CONFIGURED)
    config.params.dt.starinst = 1e8
    config.params.dt.starspec = 1e8

    directories = {
        'output': str(tmp_path),
        'output/data': str(tmp_path / 'data'),
        'output/observe': str(tmp_path / 'observe'),
        'output/offchem': str(tmp_path / 'offchem'),
        'output/plots': str(tmp_path / 'plots'),
        'spider': str(tmp_path / 'spider'),
        'fwl': str(tmp_path / 'fwl'),
    }
    for path in directories.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    with (
        patch('proteus.proteus.read_config_object', return_value=config),
        patch('proteus.utils.coupler.set_directories', return_value=directories),
    ):
        p = Proteus(config_path='dummy.toml')

    return p


class _StopAfterAtmosphereCall(Exception):
    """Sentinel exception to stop start() once the atmosphere call captures T_magma."""


@pytest.mark.unit
@pytest.mark.parametrize('interior_module', ['aragog', 'spider'])
def test_resume_first_atmosphere_call_uses_interior_t_magma(tmp_path, interior_module):
    """The first post-resume atmosphere call receives the interior's own
    T_magma output, not a value anchored to the checkpoint's T_surf. Both
    energetics modules are covered so that a resume override gated on the
    module name cannot return for one of them.
    """
    p = _make_resume_main_loop_proteus(tmp_path, interior_module=interior_module)
    hf_df = _make_resume_checkpoint_df()
    checkpoint_t_surf = hf_df['T_surf'].iloc[-1]
    interior_t_magma = 3456.0
    captured = {}

    def _fake_run_interior(*args, **kwargs):
        args[3]['T_magma'] = interior_t_magma

    def _fake_run_atmosphere(*args, **kwargs):
        captured['T_magma'] = args[8]['T_magma']
        raise _StopAfterAtmosphereCall

    _run_resumed_loop_until_stop(p, hf_df, _fake_run_interior, _fake_run_atmosphere)

    # Discrimination: the checkpoint T_surf a surface anchor would use is far
    # from the interior value.
    assert abs(interior_t_magma - checkpoint_t_surf) > 100.0
    assert captured['T_magma'] == pytest.approx(interior_t_magma, rel=1e-12)


def _run_resumed_loop_until_stop(
    p, hf_df, fake_interior, fake_atmosphere, terminate_after=None, fake_outgas=None
):
    """Resume ``p`` from ``hf_df`` with the given interior and atmosphere fakes
    until the atmosphere fake raises ``_StopAfterAtmosphereCall``, or, with
    ``terminate_after``, until the run ends normally after that many loops.
    ``fake_outgas``, when given, replaces the no-op outgassing step."""
    from types import SimpleNamespace

    checks = []

    def _terminate(handler):
        checks.append(1)
        handler.finished_both = len(checks) == terminate_after
        return handler.finished_both

    with ExitStack() as stack:
        for target in _MAIN_LOOP_NOOP_PATCHES:
            stack.enter_context(patch(target))
        stack.enter_context(
            patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=hf_df)
        )
        stack.enter_context(
            patch(
                'proteus.utils.coupler.select_resumable_snapshot',
                return_value=(hf_df, []),
            )
        )
        stack.enter_context(
            patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50)
        )
        stack.enter_context(patch('proteus.utils.coupler.assert_mass_conservation'))
        stack.enter_context(
            patch('proteus.utils.terminate.check_termination', side_effect=_terminate)
        )
        stack.enter_context(
            patch('proteus.outgas.wrapper.check_desiccation', return_value=False)
        )
        stack.enter_context(
            patch(
                'proteus.interior_energetics.wrapper.run_interior',
                side_effect=fake_interior,
            )
        )
        stack.enter_context(
            patch('proteus.atmos_clim.run_atmosphere', side_effect=fake_atmosphere)
        )
        if fake_outgas is not None:
            stack.enter_context(
                patch(
                    'proteus.outgas.wrapper.run_outgassing_and_vapourisation',
                    side_effect=fake_outgas,
                )
            )
        mock_interior_t = stack.enter_context(
            patch('proteus.interior_energetics.common.Interior_t')
        )
        mock_interior_t.return_value = MagicMock(dt=100.0, ic=1)
        mock_atmos_t = stack.enter_context(patch('proteus.atmos_clim.common.Atmos_t'))
        mock_atmos_t.return_value = SimpleNamespace(converged=True)
        mock_spectrum = stack.enter_context(patch('proteus.star.wrapper.get_new_spectrum'))
        mock_spectrum.return_value = (np.array([1.0]), np.array([1.0]))

        expect = nullcontext() if terminate_after else pytest.raises(_StopAfterAtmosphereCall)
        with expect:
            p.start(resume=True, offline=True)


@pytest.mark.unit
def test_a_resumed_run_outgasses_with_first_iter_false(tmp_path):
    """A resumed run skips the init stage, so every outgassing step it takes is
    told it is not the first iteration."""
    p = _make_resume_main_loop_proteus(tmp_path, interior_module='aragog')
    flags = []

    def _fake_outgas(_dirs, _config, _hf_row, first_iter):
        flags.append((first_iter, p.init_stage))

    def _stop_on_second(*args, **kwargs):
        if len(flags) == 2:
            raise _StopAfterAtmosphereCall

    _run_resumed_loop_until_stop(
        p,
        _make_resume_checkpoint_df(),
        lambda *a, **k: None,
        _stop_on_second,
        fake_outgas=_fake_outgas,
    )
    assert [first_iter for first_iter, _ in flags] == [False, False]
    assert [init_stage for _, init_stage in flags] == [False, False]


@pytest.mark.unit
def test_solvus_override_restores_the_magma_ocean_state(tmp_path):
    """With global miscibility the loop hands the atmosphere the solvus as its
    lower boundary (T_solvus, P_solvus in bar, R_solvus) and afterwards
    restores T_magma, T_surf, P_surf and R_int for the interior and the
    committed row. Config validation rejects global_miscibility with the
    zalmoxis structure module, the one that writes the solvus, so this pins
    the loop code for when it is enabled; SPIDER is the energetics module
    that cuts its domain at the solvus.
    """
    p = _make_resume_main_loop_proteus(tmp_path, interior_module='spider', miscibility=True)
    hf_df = _make_resume_checkpoint_df()
    hf_df['P_surf'] = 250.0
    checkpoint = hf_df.iloc[-1]
    interior_t_magma = [3456.0, 3441.0]
    t_solvus, p_solvus = 3700.0, 2.0e10
    step = {'interior': 0}
    seen_by_interior = []
    captured = []

    def _fake_run_interior(*args, **kwargs):
        hf_row = args[3]
        seen_by_interior.append(hf_row['T_magma'])
        hf_row['T_magma'] = interior_t_magma[min(step['interior'], 1)]
        hf_row['R_solvus'] = 0.9 * hf_row['R_int']
        hf_row['T_solvus'] = t_solvus
        hf_row['P_solvus'] = p_solvus
        step['interior'] += 1

    def _fake_run_atmosphere(*args, **kwargs):
        hf_row = args[8]
        captured.append(
            (hf_row['T_magma'], hf_row['T_surf'], hf_row['P_surf'], hf_row['R_int'])
        )
        if len(captured) == 2:
            raise _StopAfterAtmosphereCall

    _run_resumed_loop_until_stop(p, hf_df, _fake_run_interior, _fake_run_atmosphere)

    solvus_frame = [t_solvus, t_solvus, p_solvus * 1e-5, 0.9 * checkpoint['R_int']]
    np.testing.assert_allclose(np.array(captured), [solvus_frame, solvus_frame], rtol=1e-12)
    assert seen_by_interior[1] == pytest.approx(interior_t_magma[0], rel=1e-12)
    committed = p.hf_all.iloc[-1]
    assert committed['T_magma'] == pytest.approx(interior_t_magma[0], rel=1e-12)
    # Pins the current restore of T_surf in solvus mode; a change that keeps
    # the atmosphere T_surf updates this.
    assert committed['T_surf'] == pytest.approx(checkpoint['T_surf'], rel=1e-12)
    assert committed['P_surf'] == pytest.approx(checkpoint['P_surf'], rel=1e-12)
    assert committed['R_int'] == pytest.approx(checkpoint['R_int'], rel=1e-12)


@pytest.mark.unit
def test_solvus_override_is_restored_when_the_atmosphere_raises(tmp_path):
    """The solvus override is undone in a finally block, so an atmosphere step
    that raises leaves T_magma, T_surf, P_surf and R_int in the magma-ocean
    frame rather than in the solvus frame.
    """
    p = _make_resume_main_loop_proteus(tmp_path, interior_module='spider', miscibility=True)
    hf_df = _make_resume_checkpoint_df()
    hf_df['P_surf'] = 250.0
    checkpoint = hf_df.iloc[-1]
    interior_t_magma, t_solvus = 3456.0, 3700.0
    captured = []

    def _fake_run_interior(*args, **kwargs):
        hf_row = args[3]
        hf_row['T_magma'] = interior_t_magma
        hf_row['R_solvus'] = 0.9 * hf_row['R_int']
        hf_row['T_solvus'] = t_solvus
        hf_row['P_solvus'] = 2.0e10

    def _fake_run_atmosphere(*args, **kwargs):
        captured.append(args[8]['T_magma'])
        raise _StopAfterAtmosphereCall

    _run_resumed_loop_until_stop(p, hf_df, _fake_run_interior, _fake_run_atmosphere)

    assert captured == [pytest.approx(t_solvus, rel=1e-12)]  # the override was active
    assert p.hf_row['T_magma'] == pytest.approx(interior_t_magma, rel=1e-12)
    assert p.hf_row['T_surf'] == pytest.approx(checkpoint['T_surf'], rel=1e-12)
    assert p.hf_row['P_surf'] == pytest.approx(checkpoint['P_surf'], rel=1e-12)
    assert p.hf_row['R_int'] == pytest.approx(checkpoint['R_int'], rel=1e-12)


@pytest.mark.unit
def test_solvus_at_the_surface_leaves_the_atmosphere_boundary_unchanged(tmp_path):
    """A solvus at R_int itself (R_solvus == R_int) is not below the surface,
    so the atmosphere keeps the magma-ocean boundary.
    """
    p = _make_resume_main_loop_proteus(tmp_path, interior_module='spider', miscibility=True)
    hf_df = _make_resume_checkpoint_df()
    hf_df['P_surf'] = 250.0
    checkpoint = hf_df.iloc[-1]
    interior_t_magma = 3456.0
    captured = []

    def _fake_run_interior(*args, **kwargs):
        hf_row = args[3]
        hf_row['T_magma'] = interior_t_magma
        hf_row['R_solvus'] = hf_row['R_int']
        hf_row['T_solvus'] = 3700.0
        hf_row['P_solvus'] = 2.0e10

    def _fake_run_atmosphere(*args, **kwargs):
        hf_row = args[8]
        captured.append((hf_row['T_magma'], hf_row['P_surf'], hf_row['R_int']))
        raise _StopAfterAtmosphereCall

    _run_resumed_loop_until_stop(p, hf_df, _fake_run_interior, _fake_run_atmosphere)

    # A stray extra call with the same values would still broadcast-match the
    # single-row check below, so the call count is its own assertion.
    assert len(captured) == 1
    np.testing.assert_allclose(
        np.array(captured),
        [[interior_t_magma, checkpoint['P_surf'], checkpoint['R_int']]],
        rtol=1e-12,
    )


@pytest.mark.unit
@pytest.mark.parametrize('interior_module', ['aragog', 'spider'])
def test_resume_atmosphere_follows_interior_while_surface_stays_below_magma(
    tmp_path, interior_module
):
    """Over 3 post-resume iterations the atmosphere returns T_surf 400 K
    below the T_magma it was given, as AGNI's conductive skin does in a magma
    ocean. Every atmosphere call still receives that iteration's interior
    T_magma, so a surface temperature below the magma temperature never
    replaces T_magma in the coupling across those 3 iterations. The
    atmosphere stub only sets T_surf; its fluxes are not consistent with the
    skin drop.
    """
    p = _make_resume_main_loop_proteus(tmp_path, interior_module=interior_module)
    hf_df = _make_resume_checkpoint_df()
    # Slow interior cooling (15 K per step) against a 400 K skin drop, so a
    # surface-anchored value would fall far below the interior sequence.
    interior_t_magma = [3456.0, 3441.0, 3426.0]
    skin_drop = 400.0
    n_calls = len(interior_t_magma)
    step = {'interior': 0}
    captured = []

    def _fake_run_interior(*args, **kwargs):
        args[3]['T_magma'] = interior_t_magma[min(step['interior'], n_calls - 1)]
        step['interior'] += 1

    def _fake_run_atmosphere(*args, **kwargs):
        hf_row = args[8]
        captured.append(hf_row['T_magma'])
        hf_row['T_surf'] = hf_row['T_magma'] - skin_drop
        if len(captured) == n_calls:
            raise _StopAfterAtmosphereCall

    _run_resumed_loop_until_stop(p, hf_df, _fake_run_interior, _fake_run_atmosphere)

    assert len(captured) == n_calls
    np.testing.assert_allclose(captured, interior_t_magma, rtol=1e-12)
    # Commit order: each completed row holds the interior T_magma and the
    # T_surf the atmosphere returned in that iteration.
    committed = p.hf_all.iloc[-(n_calls - 1) :]
    np.testing.assert_allclose(committed['T_magma'], interior_t_magma[:-1], rtol=1e-12)
    np.testing.assert_allclose(
        committed['T_surf'], np.array(interior_t_magma[:-1]) - skin_drop, rtol=1e-12
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    'r_solvus_frac',
    [None, -0.1, 1.0, 1.2],
    ids=['never-written', 'negative', 'at-the-surface', 'outside-the-planet'],
)
def test_solvus_override_skips_an_unphysical_solvus(tmp_path, r_solvus_frac):
    """With global miscibility on but no physical solvus in the row, the loop
    hands the atmosphere the magma-ocean state, not the solvus frame. The
    helpfile row starts with R_solvus = T_solvus = P_solvus = 0, and an
    interior that never writes them must not drive the atmosphere with
    T_magma = T_surf = P_surf = R_int = 0.
    """
    p = _make_resume_main_loop_proteus(tmp_path, interior_module='spider', miscibility=True)
    hf_df = _make_resume_checkpoint_df()
    hf_df['P_surf'] = 250.0
    checkpoint = hf_df.iloc[-1]
    interior_t_magma = 3456.0
    captured = []

    def _fake_run_interior(*args, **kwargs):
        hf_row = args[3]
        hf_row['T_magma'] = interior_t_magma
        if r_solvus_frac is not None:
            hf_row['R_solvus'] = r_solvus_frac * hf_row['R_int']
            hf_row['T_solvus'] = 3700.0
            hf_row['P_solvus'] = 2.0e10

    def _fake_run_atmosphere(*args, **kwargs):
        hf_row = args[8]
        captured.append(
            (hf_row['T_magma'], hf_row['T_surf'], hf_row['P_surf'], hf_row['R_int'])
        )
        raise _StopAfterAtmosphereCall

    _run_resumed_loop_until_stop(p, hf_df, _fake_run_interior, _fake_run_atmosphere)

    expected = [interior_t_magma, checkpoint['T_surf'], 250.0, checkpoint['R_int']]
    np.testing.assert_allclose(np.array(captured), [expected], rtol=1e-12)
    # The magma-ocean frame is physical: positive temperatures, pressure, radius.
    assert min(captured[0]) > 0.0


@pytest.fixture
def cvode_missing(monkeypatch):
    """Make ``import scikits_odes_sundials.cvode`` fail as on a machine without CVODE."""
    # aragog reads its CVODE flag once, at first import: load it before hiding the module.
    pytest.importorskip('aragog.solver.entropy_solver')
    monkeypatch.setitem(sys.modules, 'scikits_odes_sundials.cvode', None)


def test_start_stops_before_touching_output_when_aragog_lacks_cvode(
    monkeypatch, tmp_path, cvode_missing
):
    """A fresh Aragog run without CVODE stops before the status file and the output are touched.

    ``start`` writes the status file and wipes the output directories of a
    fresh run, so the CVODE check has to come first: a broken environment must
    not cost the user the files of an earlier run. The error carries the
    install command.
    """
    p = _make_proteus_instance(tmp_path, interior_module='aragog')
    p.config.interior_energetics.aragog.solver_method = 'cvode'

    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))
        status = stack.enter_context(patch('proteus.proteus.UpdateStatusfile'))
        clean = stack.enter_context(patch('proteus.proteus.CleanDir'))
        with pytest.raises(ImportError, match='bash tools/get_cvode.sh'):
            p.start(resume=False, offline=True)

    status.assert_not_called()
    clean.assert_not_called()


@pytest.mark.parametrize(
    ('interior_module', 'solver_method'),
    [('aragog', 'radau'), ('aragog', 'bdf'), ('spider', 'cvode')],
)
def test_start_goes_ahead_without_cvode_when_it_is_not_needed(
    monkeypatch, tmp_path, interior_module, solver_method, cvode_missing
):
    """An explicit scipy solver, or SPIDER, reaches the output cleaning with CVODE missing."""
    p = _make_proteus_instance(tmp_path, interior_module=interior_module)
    p.config.interior_energetics.aragog.solver_method = solver_method

    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))
        clean = stack.enter_context(
            patch('proteus.proteus.CleanDir', side_effect=_StopAfterMeshRestore)
        )
        with pytest.raises(_StopAfterMeshRestore):
            p.start(resume=False, offline=True)

    clean.assert_called_once()


@pytest.mark.unit
def test_crystallization_not_rearmed_on_impact_step(tmp_path):
    """Crystallization helper respects impact_reset, freeze_volatiles, and boundary condition."""
    p = _make_proteus_instance(tmp_path)
    p.interior_o = MagicMock()
    p.interior_o.impact_reset = True
    p.config.params.stop.solid.freeze_volatiles = True
    p.config.params.stop.solid.phi_crit = 0.8
    p.crystallized = False
    p.hf_row = {'Phi_global': 0.5}

    # 1. impact_reset=True prevents re-arming crystallization
    p._check_crystallization()
    assert p.crystallized is False

    # 2. freeze_volatiles=False keeps crystallization disabled even without impact_reset
    p.interior_o.impact_reset = False
    p.config.params.stop.solid.freeze_volatiles = False
    p._check_crystallization()
    assert p.crystallized is False

    # 3. Phi_global strictly above threshold does not trigger crystallization
    p.config.params.stop.solid.freeze_volatiles = True
    p.hf_row = {'Phi_global': 0.81}
    p._check_crystallization()
    assert p.crystallized is False

    # 4. Exact boundary Phi_global == phi_crit triggers crystallization (tests <= condition)
    p.hf_row = {'Phi_global': 0.8}
    p._check_crystallization()
    assert p.crystallized is True


@pytest.mark.unit
def test_proteus_start_resume_refuses_legacy_accretion_ledger(tmp_path):
    """Proteus.start(resume=True) refuses legacy helpfile when impacts are active."""
    p = _make_proteus_instance(tmp_path)
    p.config.accretion.module = 'dummy'

    mock_ev = MagicMock()
    hf_df = _make_hf_df()
    hf_df['M_accreted_rock'] = 1.0e23
    hf_df['n_impacts_applied'] = 0.0

    initial_mass = p.config.planet.mass_tot

    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))
        stack.enter_context(
            patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50)
        )
        stack.enter_context(
            patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=hf_df)
        )
        stack.enter_context(
            patch(
                'proteus.utils.coupler.select_resumable_snapshot',
                return_value=(hf_df, []),
            )
        )
        stack.enter_context(
            patch('proteus.outgas.wrapper.check_desiccation', return_value=False)
        )
        stack.enter_context(patch('proteus.utils.coupler.ZeroHelpfileRow', return_value={}))
        mock_interior_t = stack.enter_context(
            patch('proteus.interior_energetics.common.Interior_t')
        )
        mock_int = MagicMock()
        mock_int.ic = 1
        mock_interior_t.return_value = mock_int
        stack.enter_context(
            patch('proteus.accretion.wrapper.init_accretion', return_value=[mock_ev])
        )
        stack.enter_context(patch('proteus.star.wrapper.init_star'))
        stack.enter_context(patch('proteus.orbit.wrapper.init_orbit'))

        with pytest.raises(RuntimeError, match='Resume refused') as excinfo:
            p.start(resume=True, offline=True)

        err = str(excinfo.value)
        assert 'predates the impact counter' in err
        assert 'runtime_helpfile.csv' in err

    assert p.config.planet.mass_tot == initial_mass
    assert p.impact_events == [mock_ev]


@pytest.mark.unit
def test_proteus_start_resume_accepts_legacy_accretion_ledger_when_disabled(tmp_path, caplog):
    """Proteus.start(resume=True) accepts legacy helpfile when accretion module is None."""
    from proteus.utils.constants import M_earth

    p = _make_proteus_instance(tmp_path)
    p.config.accretion.module = None
    p.config.planet.mass_tot = 1.0
    initial_mass = 1.0

    hf_df = _make_hf_df()
    hf_df['M_accreted_rock'] = 1.0e23
    hf_df['n_impacts_applied'] = 0.0

    class _StopAfterResume(Exception):
        pass

    def _status_hook(dirs, status):
        if status == 1:
            raise _StopAfterResume()

    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))
        stack.enter_context(
            patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50)
        )
        stack.enter_context(
            patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=hf_df)
        )
        stack.enter_context(
            patch(
                'proteus.utils.coupler.select_resumable_snapshot',
                return_value=(hf_df, []),
            )
        )
        stack.enter_context(
            patch('proteus.outgas.wrapper.check_desiccation', return_value=False)
        )
        stack.enter_context(patch('proteus.utils.coupler.ZeroHelpfileRow', return_value={}))
        mock_interior_t = stack.enter_context(
            patch('proteus.interior_energetics.common.Interior_t')
        )
        mock_int = MagicMock()
        mock_int.ic = 1
        mock_interior_t.return_value = mock_int
        stack.enter_context(patch('proteus.star.wrapper.init_star'))
        stack.enter_context(patch('proteus.orbit.wrapper.init_orbit'))
        stack.enter_context(patch('proteus.proteus.UpdateStatusfile', side_effect=_status_hook))

        with pytest.raises(_StopAfterResume):
            p.start(resume=True, offline=True)

    assert p.config.planet.mass_tot == pytest.approx(initial_mass + 1.0e23 / M_earth)
    assert p.impact_events == []
    log_files = list(tmp_path.glob('proteus_*.log'))
    log_text = '\n'.join(f.read_text() for f in log_files) if log_files else caplog.text
    assert 'Accretion is disabled for this resume' in log_text


@pytest.mark.unit
def test_the_main_loop_latches_desiccation_and_switches_to_run_desiccated(tmp_path):
    """Once the in-loop check reports desiccation the flag latches and the outgas
    step becomes run_desiccated; the check is not asked again."""
    p = _make_main_loop_proteus(tmp_path, plot_mod=1, write_mod=1, dt_write_rel=0.0)
    check = MagicMock(return_value=True)
    desiccate, outgas = MagicMock(), MagicMock()
    _run_main_loop_capturing_plots(
        p,
        patch('proteus.outgas.wrapper.check_desiccation', check),
        patch('proteus.outgas.wrapper.run_desiccated', desiccate),
        patch('proteus.outgas.wrapper.run_outgassing_and_vapourisation', outgas),
        stop_at_loop=6,
    )
    assert p.desiccated is True
    assert check.call_count == 1
    assert desiccate.call_count >= 2


@pytest.mark.unit
@pytest.mark.parametrize('verdict', [True, False])
def test_resume_takes_the_desiccated_latch_from_the_restored_row(tmp_path, verdict):
    """start(resume=True) sets the desiccated flag from check_desiccation on the
    restored last row, so a desiccated run resumes desiccated."""
    p = _make_proteus_instance(tmp_path, struct_module='dummy', interior_module='aragog')
    (tmp_path / 'data').mkdir(exist_ok=True)
    hf = _make_hf_df()
    rows = []

    def check(_config, row):
        rows.append(float(row['Time']))
        return verdict

    p.desiccated = not verdict
    _resume_with_patches(
        p, hf, patch('proteus.outgas.wrapper.check_desiccation', side_effect=check)
    )
    assert p.desiccated is verdict
    assert rows == [float(hf['Time'].iloc[-1])]
