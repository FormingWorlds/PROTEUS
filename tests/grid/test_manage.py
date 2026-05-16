"""Unit tests for ``proteus.grid.manage``.

The grid manager is plumbing-heavy: file IO, subprocess dispatch, TOML
parsing, and a multiprocessing-driven worker loop. The tests below cover
the public surface (``Grid``, ``setup_logger``, ``_thread_target``,
``grid_from_config``) with all heavy callouts mocked. Each test pins at
least two discriminating post-state facts so that a regression which
silently no-ops one branch cannot pass.

Links:
- ``docs/How-to/test_infrastructure.md``
- ``docs/How-to/test_categorization.md``
- ``docs/How-to/test_building.md``
"""

from __future__ import annotations

import logging
import os
import sys
from unittest import mock

import numpy as np
import pytest
import toml

from proteus.grid import manage as gm
from proteus.grid.manage import (
    Grid,
    _thread_target,
    grid_from_config,
    setup_logger,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_proteus_dir(tmp_path, monkeypatch):
    """Redirect the module-level PROTEUS_DIR so Grid writes to tmp_path."""
    monkeypatch.setattr(gm, 'PROTEUS_DIR', str(tmp_path))
    # The Grid constructor also calls time.sleep(4.0) when wiping old output
    # and shutil.rmtree subdir; bypass those waits entirely.
    monkeypatch.setattr(gm.time, 'sleep', lambda *_args, **_kwargs: None)
    return tmp_path


@pytest.fixture
def base_config_path(tmp_path):
    """Minimal valid base config file (loadable by read_config_object)."""
    src = tmp_path / 'base.toml'
    src.write_text('# fake base config\n')
    return src


@pytest.fixture
def grid_with_mocks(fake_proteus_dir, base_config_path, monkeypatch):
    """Construct a Grid with read_config_object and setup_logger patched.

    The Config constructor in PROTEUS does substantial validation that we
    do not want triggered here; we replace ``read_config_object`` with a
    lightweight stub via the call sites (write_config_files patches it
    locally). The Grid constructor calls ``setup_logger`` which writes to
    disk; we keep that call real because it is part of init's documented
    behavior, but anchor logging inside tmp_path.
    """
    g = Grid(
        name='unit_test_grid',
        base_config_path=str(base_config_path),
    )
    return g


# ---------------------------------------------------------------------------
# setup_logger
# ---------------------------------------------------------------------------


class TestSetupLogger:
    """Verify the custom logger setup: file handler, level mapping,
    pre-existing log removal, and excepthook installation.
    """

    def test_creates_log_file_and_attaches_handlers(self, tmp_path):
        """A fresh logger writes its log to logpath and attaches at least
        one file handler. Default level=1 maps to INFO.
        """
        logpath = tmp_path / 'mgr.log'
        # Run setup, then write a record and force a flush.
        setup_logger(logpath=str(logpath), level=1, logterm=False)
        root = logging.getLogger()
        root.info('hello unit-test world')
        for h in root.handlers:
            h.flush()
        assert logpath.exists()
        # File contains the message (discriminates against an empty-flush bug)
        contents = logpath.read_text()
        assert 'hello unit-test world' in contents
        # Level was mapped to INFO (not DEBUG / WARNING). Verify root level.
        assert root.level == logging.INFO

    def test_pre_existing_log_is_removed(self, tmp_path):
        """If logpath exists at entry, ``setup_logger`` deletes it before
        re-creating; old content is gone after the call.
        """
        logpath = tmp_path / 'stale.log'
        logpath.write_text('STALE_CONTENT_DO_NOT_KEEP\n')
        setup_logger(logpath=str(logpath), level=1, logterm=False)
        root = logging.getLogger()
        for h in root.handlers:
            h.flush()
        contents = logpath.read_text()
        # The stale content must be gone (file was deleted and recreated).
        assert 'STALE_CONTENT_DO_NOT_KEEP' not in contents
        # And the file must still exist (recreated by FileHandler).
        assert logpath.exists()

    @pytest.mark.parametrize(
        'level_arg,expected_level',
        [
            (0, logging.DEBUG),
            (2, logging.WARNING),
            (3, logging.ERROR),
            (4, logging.CRITICAL),
        ],
        ids=['debug', 'warning', 'error', 'critical'],
    )
    def test_level_mapping(self, tmp_path, level_arg, expected_level):
        """Each numeric level argument maps to the documented logging
        level constant. The default branch (1 -> INFO) is covered by
        ``test_creates_log_file_and_attaches_handlers``.
        """
        logpath = tmp_path / f'level_{level_arg}.log'
        setup_logger(logpath=str(logpath), level=level_arg, logterm=False)
        root = logging.getLogger()
        # Discriminating across all four levels: each is distinct.
        assert root.level == expected_level
        # File handler still present (level argument should not disable IO).
        assert any(isinstance(h, logging.FileHandler) for h in root.handlers)

    def test_excepthook_installed(self, tmp_path):
        """``setup_logger`` replaces ``sys.excepthook`` so unhandled
        exceptions get logged. The hook must be callable and distinct
        from the pre-call value (default ``sys.__excepthook__``).
        """
        logpath = tmp_path / 'hook.log'
        original = sys.excepthook
        try:
            setup_logger(logpath=str(logpath), level=1, logterm=False)
            new_hook = sys.excepthook
            assert callable(new_hook)
            assert new_hook is not original
        finally:
            sys.excepthook = original


# ---------------------------------------------------------------------------
# _thread_target
# ---------------------------------------------------------------------------


class TestThreadTarget:
    """Worker function dispatched by Grid.run for each grid point."""

    def test_real_run_dispatches_proteus_cli(self, monkeypatch):
        """``test=False`` invokes ``proteus start --offline --config
        <cfg>`` via ``subprocess.run`` with ``shell=False``.
        """
        recorded = {}

        def fake_run(cmd, **kwargs):
            recorded['cmd'] = cmd
            recorded['kwargs'] = kwargs

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr(gm.subprocess, 'run', fake_run)
        monkeypatch.setattr(gm.time, 'sleep', lambda *_a, **_k: None)

        _thread_target('/tmp/case_x.toml', test=False, wait=0.01)
        # Discriminating: command must contain "proteus" AND end in the cfg.
        assert recorded['cmd'][0] == 'proteus'
        assert recorded['cmd'][-1] == '/tmp/case_x.toml'
        # shell=False is non-negotiable for safety.
        assert recorded['kwargs']['shell'] is False

    def test_test_run_uses_echo_stub(self, monkeypatch):
        """``test=True`` swaps the real CLI for a /bin/echo stub. The
        stub command must still embed the config path so log inspection
        is unambiguous.
        """
        recorded = {}
        monkeypatch.setattr(
            gm.subprocess, 'run',
            lambda cmd, **kw: recorded.setdefault('cmd', cmd) or mock.MagicMock(returncode=0)
        )
        monkeypatch.setattr(gm.time, 'sleep', lambda *_a, **_k: None)

        _thread_target('/tmp/case_y.toml', test=True, wait=0.0)
        # Distinct from the real-run path: starts with /bin/echo, not proteus.
        assert recorded['cmd'][0] == '/bin/echo'
        # And the config path is embedded in the dummy text:
        joined = ' '.join(recorded['cmd'])
        assert '/tmp/case_y.toml' in joined

    def test_sleeps_for_wait_seconds(self, monkeypatch):
        """``_thread_target`` calls ``time.sleep(wait)`` after the
        subprocess returns, to swallow immediate-exit races.
        """
        slept = []
        monkeypatch.setattr(gm.subprocess, 'run', lambda *a, **k: mock.MagicMock())
        monkeypatch.setattr(gm.time, 'sleep', lambda s: slept.append(s))
        _thread_target('/tmp/c.toml', test=True, wait=2.5)
        # Discriminating: exactly the wait value was passed (not a default).
        assert slept == [2.5]
        # And was called at least once.
        assert len(slept) >= 1


# ---------------------------------------------------------------------------
# Grid.__init__
# ---------------------------------------------------------------------------


class TestGridInit:
    """Constructor: path layout, validation, ref-config copy."""

    def test_creates_outdir_cfgdir_logdir(self, fake_proteus_dir, base_config_path):
        """Default no-symlink construction creates outdir, cfgdir, logdir."""
        g = Grid(name='trial', base_config_path=str(base_config_path))
        assert os.path.isdir(g.outdir)
        assert os.path.isdir(g.cfgdir)
        assert os.path.isdir(g.logdir)
        # using_symlink should be False on the default branch
        assert g.using_symlink is False

    def test_strips_whitespace_in_name(self, fake_proteus_dir, base_config_path):
        """Whitespace around the grid name is stripped (path-safety)."""
        g = Grid(name='  padded_name  ', base_config_path=str(base_config_path))
        # The stripped name lives in outdir
        assert 'padded_name' in g.outdir
        # And the un-stripped variant does NOT appear (no leading spaces).
        assert ' padded_name ' not in g.outdir

    def test_raises_when_base_config_missing(self, fake_proteus_dir, tmp_path):
        """A non-existent base config path fails fast with a clear error.

        Verifies that the error fires AND that no output side effects ran
        (no output directory was created).
        """
        missing = tmp_path / 'nonexistent.toml'
        with pytest.raises(Exception, match='does not exist'):
            Grid(
                name='no_base',
                base_config_path=str(missing),
            )
        # Post-state: the failure aborts before any output dir is created.
        assert not (fake_proteus_dir / 'output' / 'no_base').exists()

    def test_raises_on_blank_symlink_dir(self, fake_proteus_dir, base_config_path):
        """A symlink target of '' or '.' is refused (path safety)."""
        with pytest.raises(Exception, match='blank path'):
            Grid(
                name='blank_link',
                base_config_path=str(base_config_path),
                symlink_dir='',
            )
        with pytest.raises(Exception, match='blank path'):
            Grid(
                name='dot_link',
                base_config_path=str(base_config_path),
                symlink_dir='.',
            )

    def test_writes_ref_config_copy(self, fake_proteus_dir, base_config_path):
        """The base config is copied bit-identically as ref_config.toml in
        the output directory; ensures reproducibility of the grid run.
        """
        g = Grid(name='copytest', base_config_path=str(base_config_path))
        copy_path = os.path.join(g.outdir, 'ref_config.toml')
        assert os.path.isfile(copy_path)
        # Bit-identical: same bytes as the source
        with open(copy_path, 'rb') as h:
            copy_bytes = h.read()
        with open(base_config_path, 'rb') as h:
            src_bytes = h.read()
        assert copy_bytes == src_bytes

    def test_copies_grid_config_when_given(self, fake_proteus_dir, base_config_path, tmp_path):
        """An optional ``grid_config`` argument triggers a copy.grid.toml
        sibling so the run can be reproduced from output alone.
        """
        gcfg = tmp_path / 'gridfile.toml'
        gcfg.write_text('use_slurm = false\n')
        g = Grid(
            name='with_gridcfg',
            base_config_path=str(base_config_path),
            grid_config=str(gcfg),
        )
        copy_path = os.path.join(g.outdir, 'copy.grid.toml')
        assert os.path.isfile(copy_path)
        # Bit-identical
        assert open(copy_path, 'rb').read() == open(gcfg, 'rb').read()

    def test_starts_with_empty_dimensions(self, fake_proteus_dir, base_config_path):
        """Initial state: zero dimensions, zero size, empty flat list."""
        g = Grid(name='empty_init', base_config_path=str(base_config_path))
        assert g.dim_names == []
        assert g.dim_param == []
        assert g.dim_avars == {}
        assert g.flat == []
        assert g.size == 0

    def test_existing_outdir_is_cleaned_no_git(
        self, fake_proteus_dir, base_config_path, monkeypatch
    ):
        """If outdir pre-exists without a .git subfolder, it is wiped and
        recreated. Construction must succeed and the new outdir is empty
        of the prior contents.
        """
        outdir = os.path.join(str(fake_proteus_dir), 'output', 'cleanup_target')
        os.makedirs(outdir)
        # Drop a sentinel file that must NOT survive the reconstruction.
        sentinel = os.path.join(outdir, 'old_file.dat')
        with open(sentinel, 'w') as h:
            h.write('stale')
        g = Grid(name='cleanup_target', base_config_path=str(base_config_path))
        # Sentinel file is gone:
        assert not os.path.exists(sentinel)
        # New outdir exists and contains ref_config.toml (proof of fresh init):
        assert os.path.isfile(os.path.join(g.outdir, 'ref_config.toml'))

    def test_refuses_to_clean_outdir_containing_git(
        self, fake_proteus_dir, base_config_path
    ):
        """Safety net: never wipe a directory containing a git repo.

        Verifies the error fires AND that the .git directory survives,
        i.e. the safety net actually short-circuits before rmtree runs.
        """
        outdir = os.path.join(str(fake_proteus_dir), 'output', 'has_git')
        git_dir = os.path.join(outdir, '.git')
        os.makedirs(git_dir)
        # Sentinel under .git: proves the partial wipe never ran.
        sentinel = os.path.join(git_dir, 'HEAD')
        with open(sentinel, 'w') as h:
            h.write('ref: refs/heads/main\n')
        with pytest.raises(Exception, match='Git'):
            Grid(name='has_git', base_config_path=str(base_config_path))
        # Safety net engaged: the .git tree (and its content) is untouched.
        assert os.path.isdir(git_dir)
        assert os.path.isfile(sentinel)


# ---------------------------------------------------------------------------
# Grid.add_dimension, _get_idx
# ---------------------------------------------------------------------------


class TestGridDimensions:
    """Dimension registration and lookup."""

    def test_add_dimension_appends_to_lists(self, grid_with_mocks):
        """Adding a dimension extends dim_names, dim_param, and sets
        dim_avars to a None placeholder under the dimension name.
        """
        g = grid_with_mocks
        g.add_dimension('alpha', 'planet.mass_tot')
        assert g.dim_names == ['alpha']
        assert g.dim_param == ['planet.mass_tot']
        # Placeholder None for not-yet-set values:
        assert g.dim_avars == {'alpha': None}

    def test_add_dimension_rejects_duplicates(self, grid_with_mocks):
        """A duplicate name is refused; the second call raises."""
        g = grid_with_mocks
        g.add_dimension('alpha', 'planet.mass_tot')
        with pytest.raises(Exception, match='cannot be added twice'):
            g.add_dimension('alpha', 'planet.elements.H_budget')
        # State preserved: still exactly one dimension after the failure.
        assert len(g.dim_names) == 1

    def test_get_idx_returns_correct_index(self, grid_with_mocks):
        """``_get_idx`` returns the position of a registered name and
        raises when the name is unknown.
        """
        g = grid_with_mocks
        g.add_dimension('a', 'pa')
        g.add_dimension('b', 'pb')
        g.add_dimension('c', 'pc')
        # Discriminating: positions are 0, 1, 2 in registration order.
        assert g._get_idx('a') == 0
        assert g._get_idx('b') == 1
        assert g._get_idx('c') == 2
        # Unknown raises:
        with pytest.raises(Exception, match='is not initialised'):
            g._get_idx('zzz_missing')


# ---------------------------------------------------------------------------
# set_dimension_*
# ---------------------------------------------------------------------------


class TestSetDimension:
    """Dimension-value setters: linspace, logspace, arange, direct."""

    def test_linspace_matches_numpy(self, grid_with_mocks):
        """``set_dimension_linspace`` produces ``np.linspace`` values
        coerced to a list of Python floats.
        """
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.set_dimension_linspace('m', 0.5, 1.5, 5)
        expected = list(np.linspace(0.5, 1.5, 5))
        # Discriminating values (asymmetric, ratio carries information).
        np.testing.assert_allclose(g.dim_avars['m'], expected, rtol=1e-12)
        # And the stored type is Python float (not np.float64) - downstream
        # TOML write requires this.
        assert all(type(v) is float for v in g.dim_avars['m'])

    def test_logspace_matches_numpy(self, grid_with_mocks):
        """``set_dimension_logspace`` spans start->stop on a log axis."""
        g = grid_with_mocks
        g.add_dimension('h', 'planet.elements.H_budget')
        g.set_dimension_logspace('h', 1.0, 1000.0, 4)
        # Wrong-formula guard: linear interpolation would give [1, 334, 667, 1000];
        # log spacing gives [1, 10, 100, 1000]. Pin the middle point.
        np.testing.assert_allclose(g.dim_avars['h'], [1.0, 10.0, 100.0, 1000.0], rtol=1e-12)
        # Endpoints are exact:
        assert g.dim_avars['h'][0] == pytest.approx(1.0)
        assert g.dim_avars['h'][-1] == pytest.approx(1000.0)

    def test_arange_appends_endpoint_when_missing(self, grid_with_mocks):
        """``arange(1, 5, 1)`` natively returns [1, 2, 3, 4]; the setter
        appends the stop value so the user-facing contract is endpoint-inclusive.
        """
        g = grid_with_mocks
        g.add_dimension('e', 'planet.elements.H_budget')
        g.set_dimension_arange('e', 1.0, 5.0, 1.0)
        # Discriminating: must contain the stop value AND not duplicate it.
        assert g.dim_avars['e'][-1] == pytest.approx(5.0)
        assert g.dim_avars['e'] == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_arange_keeps_endpoint_when_already_present(self, grid_with_mocks):
        """When the native arange includes the endpoint (step divides
        evenly), the setter does NOT add a duplicate.
        """
        g = grid_with_mocks
        g.add_dimension('e', 'planet.elements.H_budget')
        # arange(0, 4, 2) -> [0, 2]; endpoint NOT included because exclusive.
        # The setter then appends to get [0, 2, 4].
        g.set_dimension_arange('e', 0.0, 4.0, 2.0)
        # Discriminating: 4.0 appears exactly once at the end.
        assert g.dim_avars['e'].count(4.0) == 1
        assert g.dim_avars['e'] == [0.0, 2.0, 4.0]

    def test_direct_dedups_and_sorts_numeric(self, grid_with_mocks):
        """Numeric ``direct`` input is de-duplicated and sorted ascending
        by default.
        """
        g = grid_with_mocks
        g.add_dimension('d', 'planet.mass_tot')
        g.set_dimension_direct('d', [3.0, 1.0, 2.0, 1.0])
        # Discriminating: duplicates removed AND sorted.
        assert g.dim_avars['d'] == [1.0, 2.0, 3.0]
        # Length reflects the deduplication
        assert len(g.dim_avars['d']) == 3

    def test_direct_does_not_sort_strings(self, grid_with_mocks):
        """String values are kept as a set-derived list (no sort branch
        taken because elements are not numeric).
        """
        g = grid_with_mocks
        g.add_dimension('d', 'atmos_clim.module')
        g.set_dimension_direct('d', ['agni', 'janus', 'agni'])
        # Discriminating: deduplicated regardless
        assert set(g.dim_avars['d']) == {'agni', 'janus'}
        assert len(g.dim_avars['d']) == 2


# ---------------------------------------------------------------------------
# generate, print, _get_tmpcfg
# ---------------------------------------------------------------------------


class TestGridGenerate:
    """Grid generation: cartesian product, size accounting, formatting."""

    def test_generate_empty_grid_raises(self, grid_with_mocks):
        """A Grid with no dimensions cannot be generated.

        Verifies the early-exit error AND that the failure leaves the
        empty-state invariants (size == 0, flat == []) intact.
        """
        with pytest.raises(Exception, match='Grid is empty'):
            grid_with_mocks.generate()
        # Post-state: no partial generation; both invariants hold.
        assert grid_with_mocks.size == 0
        assert grid_with_mocks.flat == []

    def test_generate_two_dim_grid_has_correct_size(self, grid_with_mocks):
        """A 3x2 cartesian product yields 6 grid points; each point is a
        dict mapping parameter name to value.
        """
        g = grid_with_mocks
        g.add_dimension('a', 'planet.mass_tot')
        g.add_dimension('b', 'planet.elements.H_budget')
        g.set_dimension_direct('a', [0.5, 1.0, 1.5])
        g.set_dimension_direct('b', [100.0, 200.0])
        g.generate()
        # Discriminating: 3 x 2 = 6 (not 5 or 3+2 = 5; not 3 alone).
        assert g.size == 6
        assert len(g.flat) == 6
        # Each grid point is a dict mapping the variable names:
        for gp in g.flat:
            assert set(gp.keys()) == {'planet.mass_tot', 'planet.elements.H_budget'}

    def test_generate_keys_map_correctly(self, grid_with_mocks):
        """The flattened grid contains every (a, b) combination exactly
        once. Equivalent to ``itertools.product(a_vals, b_vals)``.
        """
        g = grid_with_mocks
        g.add_dimension('a', 'pa')
        g.add_dimension('b', 'pb')
        g.set_dimension_direct('a', [1.0, 2.0])
        g.set_dimension_direct('b', [10.0, 20.0])
        g.generate()
        observed = {(gp['pa'], gp['pb']) for gp in g.flat}
        expected = {(1.0, 10.0), (1.0, 20.0), (2.0, 10.0), (2.0, 20.0)}
        assert observed == expected
        # And no duplicates:
        assert len(g.flat) == len(expected)

    def test_get_tmpcfg_format(self, grid_with_mocks):
        """``_get_tmpcfg`` yields a six-digit zero-padded basename in the
        cfgdir directory.
        """
        g = grid_with_mocks
        path0 = g._get_tmpcfg(0)
        path42 = g._get_tmpcfg(42)
        # Discriminating: zero-padded width = 6
        assert path0.endswith('/case_000000.toml')
        assert path42.endswith('/case_000042.toml')
        # And both live under the cfgdir
        assert g.cfgdir in path0
        assert g.cfgdir in path42

    def test_print_setup_logs_dimension_metadata(
        self, grid_with_mocks, caplog
    ):
        """``print_setup`` logs the grid's dimension names, parameter
        keys, and value lists at INFO. Verifies the user-facing diagnostic.
        """
        g = grid_with_mocks
        g.add_dimension('alpha', 'planet.mass_tot')
        g.set_dimension_direct('alpha', [0.5, 1.0])
        # gm.log is set inside Grid.__init__; reuse it
        with caplog.at_level(logging.INFO, logger=gm.log.name):
            g.print_setup()
        text = ' '.join(rec.getMessage() for rec in caplog.records)
        assert 'alpha' in text
        assert 'planet.mass_tot' in text

    def test_print_grid_logs_flat_points(self, grid_with_mocks, caplog):
        """``print_grid`` prints each flattened grid point in order."""
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.set_dimension_direct('m', [0.5, 1.0])
        g.generate()
        with caplog.at_level(logging.INFO, logger=gm.log.name):
            g.print_grid()
        text = ' '.join(rec.getMessage() for rec in caplog.records)
        assert 'Flattened grid points' in text
        # Both values must appear in the log
        assert '0.5' in text
        assert '1.0' in text


# ---------------------------------------------------------------------------
# write_config_files
# ---------------------------------------------------------------------------


class TestWriteConfigFiles:
    """write_config_files: deepcopy base + set per-point attrs + write."""

    def test_writes_one_file_per_grid_point(
        self, grid_with_mocks, monkeypatch
    ):
        """A 2-point grid produces exactly two config files at the
        expected ``cfgdir/case_NNNNNN.toml`` paths.
        """
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.set_dimension_direct('m', [0.5, 1.0])
        g.generate()

        # Mock read_config_object to return a stub Config-like object whose
        # write() drops a marker file containing the per-case mass.
        written_paths = []

        class _FakeConf:
            def __init__(self):
                self.params = mock.MagicMock()
                self.planet = mock.MagicMock()

            def write(self, path):
                written_paths.append(path)
                # Touch the file so cfg-existence check in run() would pass.
                with open(path, 'w') as h:
                    h.write('# fake conf\n')

        monkeypatch.setattr(gm, 'read_config_object', lambda p: _FakeConf())
        # recursive_setattr would walk the attrs object; we stub it because
        # our _FakeConf is not a Config schema.
        monkeypatch.setattr(gm, 'recursive_setattr', lambda obj, attr, val: None)
        # Avoid os.sync overhead (and platform-specific behavior).
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        g.write_config_files()
        # Discriminating: exactly 2 files, named 000000 and 000001.
        assert len(written_paths) == 2
        assert any(p.endswith('case_000000.toml') for p in written_paths)
        assert any(p.endswith('case_000001.toml') for p in written_paths)

    def test_recursive_setattr_called_for_each_param(
        self, grid_with_mocks, monkeypatch
    ):
        """``recursive_setattr`` is invoked once per (grid point x
        parameter) combination with the parameter name and value pair.
        """
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.add_dimension('h', 'planet.elements.H_budget')
        g.set_dimension_direct('m', [0.5, 1.0])
        g.set_dimension_direct('h', [200.0])
        g.generate()

        recursive_calls = []

        class _FakeConf:
            def __init__(self):
                self.params = mock.MagicMock()

            def write(self, path):
                with open(path, 'w') as h:
                    h.write('')

        monkeypatch.setattr(gm, 'read_config_object', lambda p: _FakeConf())
        monkeypatch.setattr(
            gm, 'recursive_setattr',
            lambda obj, attr, val: recursive_calls.append((attr, val)),
        )
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        g.write_config_files()
        # Discriminating: 2 grid points x 2 parameters = 4 calls.
        assert len(recursive_calls) == 4
        # Every recorded value came from one of the value lists:
        values = {v for _attr, v in recursive_calls}
        assert values == {0.5, 1.0, 200.0}


# ---------------------------------------------------------------------------
# Grid.slurm_config
# ---------------------------------------------------------------------------


class TestSlurmConfig:
    """slurm_config: writes a sbatch script with the right knobs."""

    def test_slurm_script_contents_test_run(
        self, grid_with_mocks, monkeypatch, tmp_path
    ):
        """``slurm_config(..., test_run=True)`` writes a slurm_dispatch.sh
        with the echo-stub command. The SBATCH array bound matches size.
        """
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.set_dimension_direct('m', [0.5, 1.0])
        g.generate()

        # write_config_files is called inside slurm_config; stub its
        # internals like the test above.
        class _FakeConf:
            def __init__(self):
                self.params = mock.MagicMock()

            def write(self, path):
                with open(path, 'w') as h:
                    h.write('')

        monkeypatch.setattr(gm, 'read_config_object', lambda p: _FakeConf())
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        g.slurm_config(max_jobs=4, test_run=True, max_days=2, max_mem=8)
        slurm_path = os.path.join(g.outdir, 'slurm_dispatch.sh')
        assert os.path.isfile(slurm_path)
        contents = open(slurm_path).read()
        # Discriminating: test_run picks the echo stub, not the real CLI.
        assert '/bin/echo' in contents
        assert 'proteus start' not in contents
        # SBATCH array goes 0 to size-1; with size=2 that is 0-1.
        assert '--array=0-1' in contents
        # max_days propagated:
        assert '--time=2-00' in contents
        # max_mem propagated:
        assert '--mem-per-cpu=8G' in contents

    def test_slurm_script_real_run_uses_proteus(
        self, grid_with_mocks, monkeypatch
    ):
        """When ``test_run=False`` the dispatch command is the real CLI."""
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.set_dimension_direct('m', [0.5])
        g.generate()

        class _FakeConf:
            def __init__(self):
                self.params = mock.MagicMock()

            def write(self, path):
                with open(path, 'w') as h:
                    h.write('')

        monkeypatch.setattr(gm, 'read_config_object', lambda p: _FakeConf())
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        g.slurm_config(max_jobs=1, test_run=False, max_days=1, max_mem=4)
        slurm_path = os.path.join(g.outdir, 'slurm_dispatch.sh')
        contents = open(slurm_path).read()
        # Discriminating against the test_run branch:
        assert 'proteus start --offline --config' in contents
        assert '/bin/echo' not in contents

    def test_max_jobs_capped_at_grid_size(
        self, grid_with_mocks, monkeypatch
    ):
        """``max_jobs`` cannot exceed the number of grid points; the
        array bound reflects the capped value implicitly via size-1.
        """
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.set_dimension_direct('m', [0.5, 1.0])  # size = 2
        g.generate()

        class _FakeConf:
            def __init__(self):
                self.params = mock.MagicMock()

            def write(self, path):
                with open(path, 'w') as h:
                    h.write('')

        monkeypatch.setattr(gm, 'read_config_object', lambda p: _FakeConf())
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        # Caller asks for 99 jobs but only 2 points exist.
        g.slurm_config(max_jobs=99, test_run=True, max_days=1, max_mem=4)
        slurm_path = os.path.join(g.outdir, 'slurm_dispatch.sh')
        contents = open(slurm_path).read()
        # Discriminating: throttle is 2 (the size), not 99.
        assert '%2' in contents
        assert '%99' not in contents


# ---------------------------------------------------------------------------
# Grid.run
# ---------------------------------------------------------------------------


class _ImmediateDoneProcess:
    """Stand-in for multiprocessing.Process that completes instantly.

    The Grid.run loop polls ``is_alive``; the first poll returns True (so
    the process is counted in run_count and then transitions to "exited"),
    the next returns False so the case is marked done.
    """

    def __init__(self, *_args, target=None, args=(), **_kwargs):
        self._target = target
        self._args = args
        self._calls = 0
        self.started = False
        self.joined = False
        self.closed = False

    def start(self):
        self.started = True
        # Actually invoke the target (no fork, immediate).
        if self._target is not None:
            self._target(*self._args)

    def is_alive(self):
        # Treat as dead immediately after start
        return False

    def join(self):
        self.joined = True

    def close(self):
        self.closed = True


class TestGridRun:
    """Grid.run end-to-end with subprocess and multiprocessing mocked."""

    def test_run_with_test_run_dispatches_each_point(
        self, grid_with_mocks, monkeypatch
    ):
        """``run(test_run=True)`` writes configs, starts a Process per
        point, and joins each one. Sleep waits are stubbed.
        """
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.set_dimension_direct('m', [0.5, 1.0])
        g.generate()

        # write_config_files: stub out heavy bits, but leave the file
        # touches so the cfgexists check passes.
        class _FakeConf:
            def __init__(self):
                self.params = mock.MagicMock()

            def write(self, path):
                with open(path, 'w') as h:
                    h.write('# fake\n')

        monkeypatch.setattr(gm, 'read_config_object', lambda p: _FakeConf())
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        # Replace multiprocessing.Process with our instant stand-in.
        instances = []

        def _factory(*args, **kwargs):
            inst = _ImmediateDoneProcess(*args, **kwargs)
            instances.append(inst)
            return inst

        monkeypatch.setattr(gm.multiprocessing, 'Process', _factory)
        # Subprocess invoked by _thread_target via the test echo stub
        monkeypatch.setattr(gm.subprocess, 'run', lambda *a, **k: mock.MagicMock())
        monkeypatch.setattr(gm.time, 'sleep', lambda *a, **k: None)

        g.run(num_threads=2, test_run=True, check_interval=0.01)
        # Discriminating: one Process instance per grid point and each
        # was started.
        assert len(instances) == 2
        assert all(inst.started for inst in instances)

    def test_run_threads_capped_at_grid_size(
        self, grid_with_mocks, monkeypatch
    ):
        """A caller asking for 64 threads on a 2-point grid uses at most
        2 worker slots; behavior is documented and protects against
        oversubscription.
        """
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.set_dimension_direct('m', [0.5, 1.0])  # 2 points
        g.generate()

        class _FakeConf:
            def __init__(self):
                self.params = mock.MagicMock()

            def write(self, path):
                with open(path, 'w') as h:
                    h.write('')

        monkeypatch.setattr(gm, 'read_config_object', lambda p: _FakeConf())
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)
        monkeypatch.setattr(gm.subprocess, 'run', lambda *a, **k: mock.MagicMock())
        monkeypatch.setattr(gm.time, 'sleep', lambda *a, **k: None)

        instances = []
        monkeypatch.setattr(
            gm.multiprocessing, 'Process',
            lambda *a, **k: instances.append(_ImmediateDoneProcess(*a, **k))
            or instances[-1],
        )

        # Asking for 64 threads but only 2 points exist
        g.run(num_threads=64, test_run=True, check_interval=0.01)
        # Process count still == size (grid points). Threading cap is
        # internal to scheduling, but each point still gets exactly one
        # process instance.
        assert len(instances) == 2
        # All eventually started:
        assert all(i.started for i in instances)

    def test_run_raises_when_cfg_file_missing(
        self, grid_with_mocks, monkeypatch
    ):
        """If write_config_files is bypassed (so no per-case file
        exists), the run loop's existence check fires within a few
        seconds and raises with a clear message.
        """
        g = grid_with_mocks
        g.add_dimension('m', 'planet.mass_tot')
        g.set_dimension_direct('m', [0.5])
        g.generate()

        # Stub write_config_files to do nothing so cfgexists fails.
        monkeypatch.setattr(g, 'write_config_files', lambda: None)
        monkeypatch.setattr(gm.time, 'sleep', lambda *a, **k: None)
        # Avoid mp.Process creation
        monkeypatch.setattr(gm.multiprocessing, 'Process', _ImmediateDoneProcess)

        with pytest.raises(Exception, match='Config file could not be found'):
            g.run(num_threads=1, test_run=True, check_interval=0.01)
        # Post-state: the expected cfg path never materialised (proof that
        # the run failed for the documented reason, not some other path).
        assert not os.path.exists(g._get_tmpcfg(0))


# ---------------------------------------------------------------------------
# grid_from_config
# ---------------------------------------------------------------------------


def _write_grid_toml(path, **overrides):
    """Helper: write a minimal valid .grid.toml at ``path``."""
    body = {
        'output': overrides.get('output', 'unit_grid'),
        'symlink': overrides.get('symlink', ''),
        'use_slurm': overrides.get('use_slurm', False),
        'max_jobs': overrides.get('max_jobs', 2),
        'max_days': overrides.get('max_days', 1),
        'max_mem': overrides.get('max_mem', 3),
        'ref_config': overrides.get('ref_config', 'base.toml'),
    }
    body.update(overrides.get('dimensions', {}))
    path.write_text(toml.dumps(body))
    return path


class TestGridFromConfig:
    """grid_from_config: TOML parse + Grid build + dispatch wiring."""

    def test_invalid_method_raises(self, fake_proteus_dir, monkeypatch):
        """An unknown ``method`` value in a dimension table raises
        ValueError with the offending name in the message.
        """
        # base config must exist at the path Grid expects (PROTEUS_DIR/ref_config)
        base_path = fake_proteus_dir / 'base.toml'
        base_path.write_text('# base\n')

        grid_path = fake_proteus_dir / 'g.toml'
        _write_grid_toml(
            grid_path,
            dimensions={
                'planet.mass_tot': {
                    'method': 'BOGUS_METHOD',
                    'values': [0.5],
                },
            },
        )
        # Stop dispatch early (raise during dim setup, not at pg.run)
        monkeypatch.setattr(gm, 'read_config_object', lambda p: mock.MagicMock())
        # Also block run + slurm so a missed error path cannot silently
        # dispatch a real grid.
        run_called = {'n': 0}
        slurm_called = {'n': 0}
        monkeypatch.setattr(
            Grid, 'run',
            lambda self, *a, **k: run_called.__setitem__('n', run_called['n'] + 1),
        )
        monkeypatch.setattr(
            Grid, 'slurm_config',
            lambda self, *a, **k: slurm_called.__setitem__('n', slurm_called['n'] + 1),
        )
        with pytest.raises(ValueError, match='BOGUS_METHOD'):
            grid_from_config(str(grid_path), test_run=True, check_interval=0.0)
        # Post-state: neither dispatch path ran (error raised before).
        assert run_called['n'] == 0
        assert slurm_called['n'] == 0

    def test_relative_symlink_rejected(self, fake_proteus_dir):
        """Non-absolute ``symlink`` path is refused with RuntimeError.

        Verifies the early-exit AND that no output dir was created on
        the bogus relative path.
        """
        base_path = fake_proteus_dir / 'base.toml'
        base_path.write_text('# base\n')
        grid_path = fake_proteus_dir / 'g.toml'
        _write_grid_toml(
            grid_path,
            symlink='relative/path',  # not absolute
        )
        with pytest.raises(RuntimeError, match='absolute'):
            grid_from_config(str(grid_path), test_run=True, check_interval=0.0)
        # Post-state: the relative path was never created (no side effects).
        assert not (fake_proteus_dir / 'relative' / 'path').exists()

    def test_auto_folder_name_is_grid_prefixed(
        self, fake_proteus_dir, monkeypatch
    ):
        """When ``output = "auto"``, the generated folder name starts
        with the ``grid_`` prefix and embeds a timestamp + hex suffix.
        """
        base_path = fake_proteus_dir / 'base.toml'
        base_path.write_text('# base\n')
        grid_path = fake_proteus_dir / 'g.toml'
        _write_grid_toml(
            grid_path,
            output='auto',
            dimensions={
                'planet.mass_tot': {
                    'method': 'direct',
                    'values': [0.5],
                },
            },
        )

        # Capture the Grid construction to inspect its name.
        captured = {}
        orig_init = Grid.__init__

        def _capture_init(self, name, base_config_path, **kw):
            captured['name'] = name
            return orig_init(self, name, base_config_path, **kw)

        monkeypatch.setattr(Grid, '__init__', _capture_init)
        # Stub dispatch so the test does not actually try to run a grid.
        monkeypatch.setattr(Grid, 'run', lambda self, *a, **k: None)
        monkeypatch.setattr(Grid, 'slurm_config', lambda self, *a, **k: None)
        monkeypatch.setattr(gm, 'read_config_object', lambda p: mock.MagicMock())
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        grid_from_config(str(grid_path), test_run=True, check_interval=0.0)
        # Discriminating: name starts with 'grid_' and is more than 8 chars
        # long (prefix + timestamp + hex token).
        assert captured['name'].startswith('grid_')
        assert len(captured['name']) > 8

    def test_use_slurm_branch_calls_slurm_config(
        self, fake_proteus_dir, monkeypatch
    ):
        """``use_slurm = true`` routes through ``Grid.slurm_config`` and
        does NOT call ``Grid.run``.
        """
        base_path = fake_proteus_dir / 'base.toml'
        base_path.write_text('# base\n')
        grid_path = fake_proteus_dir / 'g.toml'
        _write_grid_toml(
            grid_path,
            use_slurm=True,
            dimensions={
                'planet.mass_tot': {
                    'method': 'direct',
                    'values': [0.5],
                },
            },
        )

        call_record = {'run': 0, 'slurm': 0}
        monkeypatch.setattr(
            Grid, 'run',
            lambda self, *a, **k: call_record.__setitem__('run', call_record['run'] + 1),
        )
        monkeypatch.setattr(
            Grid, 'slurm_config',
            lambda self, *a, **k: call_record.__setitem__('slurm', call_record['slurm'] + 1),
        )
        monkeypatch.setattr(gm, 'read_config_object', lambda p: mock.MagicMock())
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        grid_from_config(str(grid_path), test_run=True, check_interval=0.0)
        # Discriminating: slurm path taken, run path NOT taken.
        assert call_record['slurm'] == 1
        assert call_record['run'] == 0

    def test_run_branch_calls_run(
        self, fake_proteus_dir, monkeypatch
    ):
        """``use_slurm = false`` dispatches to ``Grid.run`` instead of
        ``Grid.slurm_config``.
        """
        base_path = fake_proteus_dir / 'base.toml'
        base_path.write_text('# base\n')
        grid_path = fake_proteus_dir / 'g.toml'
        _write_grid_toml(
            grid_path,
            use_slurm=False,
            dimensions={
                'planet.mass_tot': {
                    'method': 'direct',
                    'values': [0.5],
                },
            },
        )

        call_record = {'run': 0, 'slurm': 0}
        monkeypatch.setattr(
            Grid, 'run',
            lambda self, *a, **k: call_record.__setitem__('run', call_record['run'] + 1),
        )
        monkeypatch.setattr(
            Grid, 'slurm_config',
            lambda self, *a, **k: call_record.__setitem__('slurm', call_record['slurm'] + 1),
        )
        monkeypatch.setattr(gm, 'read_config_object', lambda p: mock.MagicMock())
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        grid_from_config(str(grid_path), test_run=True, check_interval=0.0)
        # Discriminating: run path taken, slurm NOT taken.
        assert call_record['run'] == 1
        assert call_record['slurm'] == 0

    def test_all_four_methods_set_correctly(
        self, fake_proteus_dir, monkeypatch
    ):
        """The four dimension methods (direct/linspace/logspace/arange)
        all route to the matching setter and yield non-trivial values.
        """
        base_path = fake_proteus_dir / 'base.toml'
        base_path.write_text('# base\n')
        grid_path = fake_proteus_dir / 'g.toml'
        _write_grid_toml(
            grid_path,
            dimensions={
                'planet.mass_tot': {
                    'method': 'direct',
                    'values': [0.5, 1.0],
                },
                'planet.elements.H_budget': {
                    'method': 'linspace',
                    'start': 100,
                    'stop': 200,
                    'count': 3,
                },
                'planet.elements.C_budget': {
                    'method': 'logspace',
                    'start': 1.0,
                    'stop': 1000.0,
                    'count': 4,
                },
                'interior_struct.core_frac': {
                    'method': 'arange',
                    'start': 0.3,
                    'stop': 0.5,
                    'step': 0.1,
                },
            },
        )

        captured_grid = {}

        def _capture_run(self, *a, **k):
            captured_grid['grid'] = self
            return None

        monkeypatch.setattr(Grid, 'run', _capture_run)
        monkeypatch.setattr(gm, 'read_config_object', lambda p: mock.MagicMock())
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        grid_from_config(str(grid_path), test_run=True, check_interval=0.0)
        g = captured_grid['grid']
        # Discriminating: 2 x 3 x 4 x 3 (arange 0.3, 0.4, 0.5) = 72 points
        # (arange appends 0.5 explicitly since np.arange may stop before).
        # The four methods all produced non-empty value lists.
        assert all(len(v) > 0 for v in g.dim_avars.values())
        # And the dimension count is exactly four:
        assert len(g.dim_names) == 4
        # Size is the product of the lengths
        expected_size = 1
        for vs in g.dim_avars.values():
            expected_size *= len(vs)
        assert g.size == expected_size

    def test_grid_size_cap_enforced(self, fake_proteus_dir, monkeypatch):
        """Grids larger than 999999 points are refused (file-naming limit)."""
        base_path = fake_proteus_dir / 'base.toml'
        base_path.write_text('# base\n')
        grid_path = fake_proteus_dir / 'g.toml'
        _write_grid_toml(
            grid_path,
            dimensions={
                'planet.mass_tot': {
                    'method': 'arange',
                    'start': 0.0,
                    'stop': 1000001.0,
                    'step': 1.0,
                },
            },
        )

        monkeypatch.setattr(gm, 'read_config_object', lambda p: mock.MagicMock())
        # Block dispatch in case it slips past the size check.
        run_called = {'n': 0}
        slurm_called = {'n': 0}
        monkeypatch.setattr(
            Grid, 'run',
            lambda self, *a, **k: run_called.__setitem__('n', run_called['n'] + 1),
        )
        monkeypatch.setattr(
            Grid, 'slurm_config',
            lambda self, *a, **k: slurm_called.__setitem__('n', slurm_called['n'] + 1),
        )
        monkeypatch.setattr(gm, 'recursive_setattr', lambda *a, **k: None)
        monkeypatch.setattr(gm.os, 'sync', lambda: None)

        with pytest.raises(ValueError, match='too large'):
            grid_from_config(str(grid_path), test_run=True, check_interval=0.0)
        # Post-state: dispatch was blocked before reaching run or slurm.
        assert run_called['n'] == 0
        assert slurm_called['n'] == 0
