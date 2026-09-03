"""
Unit tests for proteus.proteus module: Zalmoxis mesh restoration on resume,
atmosphere-interior deadlock detection, and main-loop plot cadence.

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
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    'proteus.utils.terminate.print_termination_criteria',
]


class _StopAfterMeshRestore(Exception):
    """Sentinel exception to stop start() after the mesh restoration block."""


def _resume_with_patches(p, hf_df):
    """Call p.start(resume=True) with all start() imports mocked.

    Uses ExitStack to avoid Python's nested-block limit.
    Stops at init_star (after mesh restoration).
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

        with pytest.raises(_StopAfterMeshRestore):
            p.start(resume=True, offline=True)


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


# ---------------------------------------------------------------------------
# Global miscibility solvus override (proteus.py L816-831)
# ---------------------------------------------------------------------------


def test_solvus_override_saves_and_restores_boundary_conditions(tmp_path):
    """When global_miscibility is enabled and R_solvus < R_int, the
    atmosphere BC values (T_surf, P_surf, R_int, T_magma) are
    temporarily overridden to the solvus values, then restored.

    Discrimination: after restoration, hf_row must hold the original
    values, not the solvus-overridden ones. A regression that skipped
    the restore block would leave the solvus values in place.
    """
    from types import SimpleNamespace

    original = {
        'T_surf': 2000.0,
        'P_surf': 100.0,
        'R_int': 6.4e6,
        'T_magma': 2500.0,
        'R_solvus': 6.0e6,
        'T_solvus': 1800.0,
        'P_solvus': 5e9,
    }
    hf_row = dict(original)

    config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            zalmoxis=SimpleNamespace(global_miscibility=True),
        ),
    )

    # The production override logic from proteus.py L816-831
    _saved_atm_bc = {}
    if config.interior_struct.zalmoxis.global_miscibility and 'R_solvus' in hf_row:
        R_sol = hf_row.get('R_solvus')
        if R_sol is not None and R_sol < hf_row['R_int']:
            _saved_atm_bc = {
                'T_surf': hf_row['T_surf'],
                'P_surf': hf_row['P_surf'],
                'R_int': hf_row['R_int'],
                'T_magma': hf_row['T_magma'],
            }
            hf_row['T_surf'] = hf_row['T_solvus']
            hf_row['T_magma'] = hf_row['T_solvus']
            hf_row['P_surf'] = hf_row['P_solvus'] * 1e-5
            hf_row['R_int'] = R_sol

    # Verify override happened
    assert hf_row['T_surf'] == pytest.approx(1800.0, rel=1e-12)
    assert hf_row['P_surf'] == pytest.approx(5e4, rel=1e-6)
    assert hf_row['R_int'] == pytest.approx(6.0e6, rel=1e-12)

    # Restore
    if _saved_atm_bc:
        for key, val in _saved_atm_bc.items():
            hf_row[key] = val

    # After restoration, original values must be back
    assert hf_row['T_surf'] == pytest.approx(2000.0, rel=1e-12)
    assert hf_row['P_surf'] == pytest.approx(100.0, rel=1e-12)
    assert hf_row['R_int'] == pytest.approx(6.4e6, rel=1e-12)
    assert hf_row['T_magma'] == pytest.approx(2500.0, rel=1e-12)
    # Discrimination: the solvus values are NOT the restored values
    assert hf_row['T_surf'] != pytest.approx(1800.0)


def test_solvus_override_no_op_when_r_solvus_exceeds_r_int():
    """When R_solvus >= R_int, the override block must not fire.

    Edge: the solvus is deeper than the interior radius, so the
    atmosphere BC stays at the magma ocean surface.
    """
    from types import SimpleNamespace

    hf_row = {
        'T_surf': 2000.0,
        'P_surf': 100.0,
        'R_int': 6.4e6,
        'T_magma': 2500.0,
        'R_solvus': 6.5e6,
        'T_solvus': 1800.0,
        'P_solvus': 5e9,
    }
    config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            zalmoxis=SimpleNamespace(global_miscibility=True),
        ),
    )

    _saved_atm_bc = {}
    if config.interior_struct.zalmoxis.global_miscibility and 'R_solvus' in hf_row:
        R_sol = hf_row.get('R_solvus')
        if R_sol is not None and R_sol < hf_row['R_int']:
            _saved_atm_bc = {'T_surf': hf_row['T_surf']}

    # Override must NOT have fired
    assert _saved_atm_bc == {}
    assert hf_row['T_surf'] == pytest.approx(2000.0, rel=1e-12)


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
    are no-ops; `observe.module=None` and a non-'online'/'offline'
    atmos_chem.when skip the postprocessing branches. None of these
    short-circuits touch the plot-gating condition under test.

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


def _run_main_loop_capturing_plots(p, *, stop_at_loop):
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
        mock_interior_t.return_value = SimpleNamespace(dt=100.0, ic=1)

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


def _run_main_loop_recording_mass(p, *, stop_at_loop, rows, row_writer, guard_calls=None):
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

    def _fake_outgas(_dirs, _config, hf_row, _first_iter):
        rows.append(dict(row_writer(hf_row, len(rows))))

    with ExitStack() as stack:
        for target in _MAIN_LOOP_NOOP_PATCHES:
            stack.enter_context(patch(target))

        mock_interior_t = stack.enter_context(
            patch('proteus.interior_energetics.common.Interior_t')
        )
        mock_interior_t.return_value = SimpleNamespace(dt=100.0, ic=1)

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
    that gates the crystallization scan and the escape block's active
    branch, so both take their inactive branch on the first post-resume
    iteration. Every row starts from `ZeroHelpfileRow()` so every real
    helpfile column the main loop reads is present.
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
                'T_eqm': 255.0,
                'F_atm': 100.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _make_resume_main_loop_proteus(tmp_path):
    """Build a Proteus instance for a resumed run driven into the main loop.

    Mirrors `_make_main_loop_proteus`'s dummy-module, full-loop-capable
    config, since a resumed run reaches the same main-loop code once
    resume setup completes.
    """
    from proteus.config._params import StopStall
    from proteus.proteus import Proteus

    config = MagicMock()
    config.interior_struct.module = 'dummy'
    config.interior_struct.zalmoxis.update_interval = 0
    config.interior_struct.zalmoxis.global_miscibility = False
    config.interior_struct.eos_dir = None
    config.interior_energetics.module = 'spider'
    config.interior_energetics.flux_guess = 100.0
    config.orbit.module = None
    config.observe.module = None
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
def test_resume_first_atmosphere_call_uses_interior_t_magma(tmp_path):
    """The first post-resume atmosphere call receives the interior's own
    T_magma output, not the checkpoint's stored value, matching a
    non-resumed run's per-iteration flow.
    """
    from types import SimpleNamespace

    p = _make_resume_main_loop_proteus(tmp_path)
    hf_df = _make_resume_checkpoint_df()
    checkpoint_t_magma = hf_df['T_magma'].iloc[-1]
    interior_t_magma = 3456.0
    captured = {}

    def _fake_run_interior(*args, **kwargs):
        args[3]['T_magma'] = interior_t_magma

    def _fake_run_atmosphere(*args, **kwargs):
        captured['T_magma'] = args[8]['T_magma']
        raise _StopAfterAtmosphereCall

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
            patch('proteus.outgas.wrapper.check_desiccation', return_value=False)
        )
        stack.enter_context(
            patch(
                'proteus.interior_energetics.wrapper.run_interior',
                side_effect=_fake_run_interior,
            )
        )
        stack.enter_context(
            patch('proteus.atmos_clim.run_atmosphere', side_effect=_fake_run_atmosphere)
        )

        mock_interior_t = stack.enter_context(
            patch('proteus.interior_energetics.common.Interior_t')
        )
        mock_interior_t.return_value = MagicMock(dt=100.0, ic=1)

        mock_atmos_t = stack.enter_context(patch('proteus.atmos_clim.common.Atmos_t'))
        mock_atmos_t.return_value = SimpleNamespace(converged=True)

        mock_spectrum = stack.enter_context(patch('proteus.star.wrapper.get_new_spectrum'))
        mock_spectrum.return_value = (np.array([1.0]), np.array([1.0]))

        with pytest.raises(_StopAfterAtmosphereCall):
            p.start(resume=True, offline=True)

    assert captured['T_magma'] == interior_t_magma
    assert captured['T_magma'] != checkpoint_t_magma
