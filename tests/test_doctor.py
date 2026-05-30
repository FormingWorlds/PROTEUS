"""Unit tests for the PROTEUS doctor diagnostics (doctor.py).

Tests the structured check system: CheckResult, individual check
functions, pin-aware version comparison, and the doctor/update entry
points. Does not make real network calls or modify the installation.

Testing standards:
  - docs/How-to/testing.md
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import Mock, patch

import pytest

from proteus.doctor import (
    FAIL,
    PASS,
    WARN,
    CheckResult,
    _editable_checkout_path,
    _git_dirty,
    _git_head,
    check_env_var,
    check_fwl_data,
    check_julia,
    check_python_package,
    run_all_checks,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


class TestCheckResult:
    """CheckResult data class and serialisation."""

    def test_to_dict_includes_all_fields(self):
        """Serialisation round-trips all five fields."""
        r = CheckResult(
            name='FWL_DATA',
            category='environment',
            status=PASS,
            message='/data',
            fix_cmd=None,
        )
        d = r.to_dict()
        assert d['name'] == 'FWL_DATA'
        assert d['category'] == 'environment'
        assert d['status'] == 'pass'
        assert d['fix_cmd'] is None

    def test_to_dict_includes_fix_cmd_when_present(self):
        """Fix commands survive serialisation for JSON output."""
        r = CheckResult(
            name='julia',
            category='environment',
            status=FAIL,
            message='not found',
            fix_cmd='curl -fsSL https://install.julialang.org | sh',
        )
        d = r.to_dict()
        assert 'julialang' in d['fix_cmd']


class TestCheckEnvVar:
    """Environment variable checks."""

    def test_pass_when_set_and_path_exists(self, tmp_path):
        """A set variable pointing to an existing path passes."""
        with patch.dict(os.environ, {'TEST_VAR': str(tmp_path)}):
            r = check_env_var('TEST_VAR')
        assert r.status == PASS
        assert str(tmp_path) in r.message

    def test_fail_when_not_set(self):
        """An unset variable fails with a fix suggestion."""
        env = os.environ.copy()
        env.pop('NONEXISTENT_VAR', None)
        with patch.dict(os.environ, env, clear=True):
            r = check_env_var('NONEXISTENT_VAR')
        assert r.status == FAIL
        assert r.fix_cmd is not None

    def test_warn_when_path_missing(self):
        """A set variable pointing to a nonexistent path warns."""
        with patch.dict(os.environ, {'BAD_PATH': '/no/such/path'}):
            r = check_env_var('BAD_PATH', validate_path=True)
        assert r.status == WARN

    def test_warn_when_required_file_missing(self, tmp_path):
        """A valid path that lacks a required file warns."""
        with patch.dict(os.environ, {'RDIR': str(tmp_path)}):
            r = check_env_var('RDIR', required_file='bin/radlib.a')
        assert r.status == WARN
        assert 'radlib.a' in r.message

    def test_pass_when_required_file_present(self, tmp_path):
        """A valid path with the required file passes."""
        (tmp_path / 'bin').mkdir()
        (tmp_path / 'bin' / 'radlib.a').touch()
        with patch.dict(os.environ, {'RDIR': str(tmp_path)}):
            r = check_env_var('RDIR', required_file='bin/radlib.a')
        assert r.status == PASS


class TestCheckFwlData:
    """FWL_DATA subdirectory checks."""

    def test_reports_present_subdirs(self, tmp_path):
        """Populated subdirectories report as present."""
        (tmp_path / 'spectral_files').mkdir()
        (tmp_path / 'spectral_files' / 'data.bin').touch()
        (tmp_path / 'stellar_spectra').mkdir()
        (tmp_path / 'stellar_spectra' / 'sun.txt').touch()
        with patch.dict(os.environ, {'FWL_DATA': str(tmp_path)}):
            results = check_fwl_data()
        statuses = {r.name: r.status for r in results}
        assert statuses['FWL_DATA/spectral_files'] == PASS
        assert statuses['FWL_DATA/stellar_spectra'] == PASS

    def test_reports_missing_subdirs(self, tmp_path):
        """Missing subdirectories warn with a fix command."""
        with patch.dict(os.environ, {'FWL_DATA': str(tmp_path)}):
            results = check_fwl_data()
        for r in results:
            assert r.status == WARN
            assert r.fix_cmd is not None

    def test_skips_when_fwl_data_unset(self):
        """No checks when FWL_DATA is not set."""
        env = os.environ.copy()
        env.pop('FWL_DATA', None)
        with patch.dict(os.environ, env, clear=True):
            results = check_fwl_data()
        assert results == []


class TestCheckJulia:
    """Julia version checks."""

    def test_pass_for_correct_version(self):
        """Julia 1.11.x passes."""
        with patch('proteus.doctor._julia_version', return_value='1.11.8'):
            r = check_julia()
        assert r.status == PASS

    def test_warn_for_wrong_version(self):
        """Julia 1.12.x warns with a juliaup fix."""
        with patch('proteus.doctor._julia_version', return_value='1.12.1'):
            r = check_julia()
        assert r.status == WARN
        assert 'juliaup' in r.fix_cmd

    def test_fail_when_missing(self):
        """Missing Julia fails with install command."""
        with patch('proteus.doctor._julia_version', return_value=None):
            r = check_julia()
        assert r.status == FAIL
        assert 'julialang' in r.fix_cmd


class TestCheckPythonPackage:
    """Python package version checks against pyproject.toml specs."""

    def test_pass_when_installed_satisfies_spec(self):
        """An installed version within the spec range passes."""
        from packaging.requirements import Requirement

        spec = Requirement('fwl-aragog>=26.05.13')
        with patch('proteus.doctor.importlib.metadata.version', return_value='26.5.13'):
            with patch('proteus.doctor._editable_checkout_path', return_value=None):
                r = check_python_package('fwl-aragog', spec)
        assert r.status == PASS

    def test_fail_when_below_spec(self):
        """A version below the minimum bound fails."""
        from packaging.requirements import Requirement

        spec = Requirement('fwl-aragog>=26.05.13')
        with patch('proteus.doctor.importlib.metadata.version', return_value='25.1.1'):
            with patch('proteus.doctor._editable_checkout_path', return_value=None):
                r = check_python_package('fwl-aragog', spec)
        assert r.status == FAIL
        assert 'requires' in r.message

    def test_fail_when_not_installed(self):
        """A missing package fails with a pip install command."""
        from importlib.metadata import PackageNotFoundError

        with patch(
            'proteus.doctor.importlib.metadata.version',
            side_effect=PackageNotFoundError('fwl-vulcan'),
        ):
            r = check_python_package('fwl-vulcan', None)
        assert r.status == FAIL
        assert 'pip install' in r.fix_cmd

    def test_editable_annotation_included(self):
        """Editable installs show the checkout path and git hash."""
        from packaging.requirements import Requirement

        spec = Requirement('fwl-aragog>=26.05.13')
        with (
            patch('proteus.doctor.importlib.metadata.version', return_value='26.5.13'),
            patch(
                'proteus.doctor._editable_checkout_path',
                return_value='/tmp/aragog',
            ),
            patch('proteus.doctor._git_short_head', return_value='d051902'),
            patch('proteus.doctor._git_dirty', return_value=False),
        ):
            r = check_python_package('fwl-aragog', spec)
        assert r.status == PASS
        assert 'editable' in r.message
        assert 'd051902' in r.message


def _init_test_repo(path):
    """Create a git repo with one commit at the given path.

    Sets local user.name/user.email so commits work on CI runners
    that have no global git config.
    """
    p = str(path)
    subprocess.run(['git', 'init', p], check=True, capture_output=True)
    subprocess.run(
        ['git', '-C', p, 'config', 'user.name', 'Test'], check=True, capture_output=True
    )
    subprocess.run(
        ['git', '-C', p, 'config', 'user.email', 'test@test.com'],
        check=True,
        capture_output=True,
    )
    (path / 'f.txt').write_text('x')
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', 'init'], check=True, capture_output=True)


class TestGitHelpers:
    """Git helper functions."""

    def test_git_head_returns_hash(self, tmp_path):
        """Returns full hash for a real git repo."""
        _init_test_repo(tmp_path)
        h = _git_head(str(tmp_path))
        assert h is not None
        assert len(h) == 40

    def test_git_head_returns_none_for_non_repo(self, tmp_path):
        """Non-repo path returns None."""
        assert _git_head(str(tmp_path)) is None

    def test_git_dirty_false_for_clean_repo(self, tmp_path):
        """Clean repo returns False."""
        _init_test_repo(tmp_path)
        assert _git_dirty(str(tmp_path)) is False

    def test_git_dirty_true_for_modified_file(self, tmp_path):
        """Modified tracked file makes repo dirty."""
        _init_test_repo(tmp_path)
        (tmp_path / 'f.txt').write_text('changed')
        assert _git_dirty(str(tmp_path)) is True

    def test_git_dirty_none_for_non_repo(self, tmp_path):
        """Non-repo path returns None so the caller can mark the dirty state
        as unknown rather than silently reporting a clean tree."""
        assert _git_dirty(str(tmp_path)) is None


class TestEditableCheckoutPath:
    """PEP 610 editable install detection."""

    def test_returns_none_for_missing_package(self):
        """Missing packages return None."""
        from importlib.metadata import PackageNotFoundError

        with patch(
            'proteus.doctor.importlib.metadata.distribution',
            side_effect=PackageNotFoundError('x'),
        ):
            assert _editable_checkout_path('x') is None

    def test_returns_none_for_wheel_install(self):
        """Wheel installs have no direct_url.json."""
        dist = Mock()
        dist.read_text.return_value = None
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            assert _editable_checkout_path('x') is None

    def test_returns_path_for_editable_install(self):
        """Editable installs return the decoded file path."""
        dist = Mock()
        dist.read_text.return_value = (
            '{"url": "file:///Users/Tim%20L/git/aragog", "dir_info": {"editable": true}}'
        )
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            path = _editable_checkout_path('fwl-aragog')
        assert path == '/Users/Tim L/git/aragog'

    def test_returns_none_for_malformed_json(self):
        """Corrupted direct_url.json returns None."""
        dist = Mock()
        dist.read_text.return_value = '{not json'
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            assert _editable_checkout_path('x') is None


class TestRunAllChecks:
    """Integration: run_all_checks returns a flat list of CheckResults."""

    def test_returns_list_of_check_results(self):
        """All items are CheckResult instances with valid status values."""
        with (
            patch('proteus.doctor._dependency_specs', return_value={}),
            patch('proteus.doctor._module_pins', return_value={}),
        ):
            results = run_all_checks()
        assert len(results) > 0
        for r in results:
            assert isinstance(r, CheckResult)
            assert r.status in (PASS, WARN, FAIL)
