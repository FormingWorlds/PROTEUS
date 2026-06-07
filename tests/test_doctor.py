"""Unit tests for the PROTEUS doctor diagnostics (doctor.py).

Tests the structured check system: CheckResult, individual check
functions, pin-aware version comparison, and the doctor/update entry
points. Does not make real network calls or modify the installation.

Testing standards:
  - docs/How-to/testing.md
"""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import Mock, patch

import pytest

from proteus.doctor import (
    FAIL,
    PASS,
    PYTHON_PACKAGES,
    WARN,
    CheckResult,
    _editable_checkout_path,
    _git_dirty,
    _git_head,
    _git_short_head,
    _imported_package_dir,
    _julia_version,
    check_env_var,
    check_fwl_data,
    check_julia,
    check_python_package,
    doctor_entry,
    run_all_checks,
    update_entry,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_python_packages_excludes_optional_backends():
    """`proteus doctor` must check only mandatory packages.

    The optional GPL backends (fwl-vulcan, atmodeller) are not pulled by a
    default install, so listing them in PYTHON_PACKAGES would make doctor
    report a missing package and update-all try to install software a standard
    run does not need. The mandatory framework packages must stay so their
    versions are still verified.
    """
    optional = {'fwl-vulcan', 'atmodeller'}
    assert optional.isdisjoint(PYTHON_PACKAGES), (
        f'doctor must not check optional backends as mandatory: '
        f'{optional & set(PYTHON_PACKAGES)}'
    )
    # Discrimination: an empty list would also pass the disjointness check
    # while silently disabling all version verification; pin the mandatory
    # framework packages that must remain.
    mandatory = {'fwl-proteus', 'fwl-aragog', 'fwl-calliope', 'fwl-zalmoxis'}
    assert mandatory <= set(PYTHON_PACKAGES), (
        f'doctor lost mandatory package checks: {mandatory - set(PYTHON_PACKAGES)}'
    )


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
        # The failing status and identity serialise alongside the fix, so
        # JSON consumers can pair the command with the failed check.
        assert d['status'] == 'fail'
        assert d['name'] == 'julia'


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
        # The message names the offending path so the user can fix it.
        assert '/no/such/path' in r.message

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
        # A passing check carries no fix command.
        assert r.fix_cmd is None


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
        # Discrimination: with the variable set, the same call reports
        # per-subdirectory results, so the empty list above comes from
        # the unset branch and not from a function that always returns
        # nothing.
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {'FWL_DATA': td}):
                assert len(check_fwl_data()) > 0


class TestCheckJulia:
    """Julia version checks."""

    def test_pass_for_supported_versions(self):
        """Julia 1.11.x and 1.12.x both pass."""
        for ver in ('1.11.8', '1.12.6'):
            with patch('proteus.doctor._julia_version', return_value=ver):
                r = check_julia()
            assert r.status == PASS
            # The detected version is reported and no fix is suggested.
            assert ver in r.message
            assert r.fix_cmd is None

    def test_warn_for_wrong_version(self):
        """Julia versions outside 1.11/1.12 warn with a juliaup fix.

        1.10 (too old) and 1.13 (untested release) both warn; the
        contrast against the passing 1.12 above pins the boundary.
        """
        for ver in ('1.10.4', '1.13.0'):
            with patch('proteus.doctor._julia_version', return_value=ver):
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
        # The installed version is reported for the human reading the
        # doctor output.
        assert '26.5.13' in r.message

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
            # Imported package lives under the editable checkout, so the
            # annotation keeps the checkout/hash form.
            patch(
                'proteus.doctor._imported_package_dir',
                return_value='/tmp/aragog/aragog',
            ),
        ):
            r = check_python_package('fwl-aragog', spec)
        assert r.status == PASS
        assert 'editable' in r.message
        assert 'd051902' in r.message
        # The import matches the checkout, so no shadow warning is shown.
        assert 'importing from' not in r.message

    def test_editable_annotation_flags_shadowed_import(self):
        """When a non-editable install shadows the editable sibling on the
        import path, the annotation reports the imported location instead of
        silently trusting the editable checkout metadata."""
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
            # Imported package is in site-packages, NOT under the checkout.
            patch(
                'proteus.doctor._imported_package_dir',
                return_value='/usr/lib/python3.12/site-packages/aragog',
            ),
        ):
            r = check_python_package('fwl-aragog', spec)
        assert r.status == PASS
        assert 'importing from' in r.message
        assert 'site-packages/aragog' in r.message
        # The misleading checkout->hash form is replaced, not appended.
        assert 'editable @' not in r.message


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
        """A non-repo path returns None (unknown), distinct from the
        40-char hash a real repo yields."""
        assert _git_head(str(tmp_path)) is None
        # Contrast: the same helper on an initialised repo returns a hash,
        # so None is a genuine "not a repo" signal, not a constant return.
        _init_test_repo(tmp_path)
        head = _git_head(str(tmp_path))
        assert head is not None and len(head) == 40

    def test_git_dirty_false_for_clean_repo(self, tmp_path):
        """Clean repo returns False."""
        _init_test_repo(tmp_path)
        assert _git_dirty(str(tmp_path)) is False

    def test_git_dirty_true_for_modified_file(self, tmp_path):
        """Modified tracked file makes repo dirty."""
        _init_test_repo(tmp_path)
        (tmp_path / 'f.txt').write_text('changed')
        assert _git_dirty(str(tmp_path)) is True
        # Restoring the tracked content returns the repo to clean, so the
        # True above reflects the modification and not a stuck reading.
        (tmp_path / 'f.txt').write_text('x')
        assert _git_dirty(str(tmp_path)) is False

    def test_git_dirty_none_for_non_repo(self, tmp_path):
        """Non-repo path returns None so the caller can mark the dirty state
        as unknown rather than silently reporting a clean tree."""
        assert _git_dirty(str(tmp_path)) is None
        # Contrast: an initialised clean repo returns False, not None, so
        # None genuinely distinguishes "not a repo" from "clean tree".
        _init_test_repo(tmp_path)
        assert _git_dirty(str(tmp_path)) is False


class TestEditableCheckoutPath:
    """PEP 610 editable install detection."""

    def test_returns_none_for_missing_package(self):
        """Missing packages return None."""
        from importlib.metadata import PackageNotFoundError

        with patch(
            'proteus.doctor.importlib.metadata.distribution',
            side_effect=PackageNotFoundError('x'),
        ) as mock_dist:
            result = _editable_checkout_path('x')
        assert result is None
        # Discrimination: the helper attempted the metadata lookup and
        # swallowed PackageNotFoundError, rather than returning None blind.
        assert mock_dist.call_count == 1

    def test_returns_none_for_wheel_install(self):
        """Wheel installs have no direct_url.json."""
        dist = Mock()
        dist.read_text.return_value = None
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            result = _editable_checkout_path('x')
        assert result is None
        # Discrimination: the helper consulted direct_url.json (absent for
        # a wheel) rather than returning None without inspecting it.
        assert dist.read_text.call_count == 1

    def test_returns_path_for_editable_install(self):
        """Editable installs return the decoded file path."""
        dist = Mock()
        dist.read_text.return_value = (
            '{"url": "file:///Users/Tim%20L/git/aragog", "dir_info": {"editable": true}}'
        )
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            path = _editable_checkout_path('fwl-aragog')
        assert path == '/Users/Tim L/git/aragog'
        # The %20 escape was decoded rather than passed through, and the
        # file:// scheme was stripped.
        assert '%20' not in path
        assert not path.startswith('file:')

    def test_returns_none_for_malformed_json(self):
        """Corrupted direct_url.json returns None."""
        dist = Mock()
        dist.read_text.return_value = '{not json'
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            result = _editable_checkout_path('x')
        assert result is None
        # Discrimination: malformed JSON is swallowed (returns None) only
        # after the helper read and tried to parse direct_url.json.
        assert dist.read_text.call_count == 1


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


def _mixed_results() -> list[CheckResult]:
    """A pass + a fixable fail + a fixable warn, mirroring a real diagnose run."""
    return [
        CheckResult('FWL_DATA', 'environment', PASS, '/data', None),
        CheckResult('Julia', 'environment', FAIL, 'not found', 'curl -fsSL https://x | sh'),
        CheckResult('fwl-mors', 'versions', WARN, 'outdated', 'pip install -U fwl-mors'),
    ]


class TestDoctorEntry:
    """`proteus doctor` reporting (doctor_entry)."""

    def test_json_output_serialises_every_result(self, capsys):
        """`--json` prints a parseable array carrying each check's status and
        fix command, not a human-readable summary."""
        with patch('proteus.doctor.run_all_checks', return_value=_mixed_results()):
            doctor_entry(output_json=True)
        parsed = json.loads(capsys.readouterr().out)
        assert len(parsed) == 3
        assert {r['name'] for r in parsed} == {'FWL_DATA', 'Julia', 'fwl-mors'}
        # Discrimination: the failing check carries its fix command through to
        # JSON; a regression that dropped fix_cmd would surface here.
        julia = next(r for r in parsed if r['name'] == 'Julia')
        assert julia['status'] == FAIL and julia['fix_cmd']

    def test_human_output_counts_failures_and_points_to_update(self, capsys):
        """Human output tallies failures and warnings and suggests `proteus
        update` when fixable issues remain."""
        with patch('proteus.doctor.run_all_checks', return_value=_mixed_results()):
            doctor_entry()
        out = capsys.readouterr().out
        assert '1 failed' in out and '1 warnings' in out
        assert 'proteus update' in out
        # Discrimination: the all-clear line must be absent when checks fail.
        assert 'All checks passed.' not in out

    def test_all_pass_reports_clear_and_no_update_hint(self, capsys):
        """With every check passing, the summary is the all-clear line and the
        update hint is suppressed."""
        with patch(
            'proteus.doctor.run_all_checks',
            return_value=[CheckResult('FWL_DATA', 'environment', PASS, '/data', None)],
        ):
            doctor_entry()
        out = capsys.readouterr().out
        assert 'All checks passed.' in out
        # Discrimination: no failure tally and no fix suggestion on a clean run.
        assert 'failed' not in out and 'proteus update' not in out


class TestUpdateEntry:
    """`proteus update` fix execution (update_entry)."""

    def test_nothing_to_fix_returns_early(self, capsys):
        """When no check is both failing and fixable, update reports nothing to
        do and does not enter the dry-run/fix path."""
        with patch(
            'proteus.doctor.run_all_checks',
            return_value=[CheckResult('FWL_DATA', 'environment', PASS, '/data', None)],
        ):
            update_entry(dry_run=True)
        out = capsys.readouterr().out
        assert 'Nothing to update' in out
        # Discrimination: early return means the dry-run banner never prints.
        assert 'Dry run' not in out

    def test_dry_run_lists_fixes_without_executing(self, capsys):
        """Dry run lists the fixable commands but executes none of them."""
        with (
            patch('proteus.doctor.run_all_checks', return_value=_mixed_results()),
            patch('proteus.doctor.subprocess.run') as mock_run,
        ):
            update_entry(dry_run=True)
        out = capsys.readouterr().out
        assert 'Dry run' in out
        assert 'pip install -U fwl-mors' in out
        # Error contract: dry run must not execute any fix command.
        assert mock_run.call_count == 0

    def test_executes_each_fix_from_repo_root_then_rechecks(self, tmp_path):
        """A real update runs one command per fixable check from the repo root
        and re-runs the diagnostics afterwards."""
        (tmp_path / 'tools').mkdir()
        with (
            patch('proteus.doctor.run_all_checks', return_value=_mixed_results()),
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor.subprocess.run') as mock_run,
            patch('proteus.doctor.doctor_entry') as mock_recheck,
        ):
            update_entry(dry_run=False)
        # Two fixable checks (Julia + mors) -> two commands.
        assert mock_run.call_count == 2
        # Commands run from the repo root, not the process cwd.
        assert all(c.kwargs.get('cwd') == str(tmp_path) for c in mock_run.call_args_list)
        # Diagnostics re-run exactly once after the fixes.
        assert mock_recheck.call_count == 1

    def test_refuses_to_fix_without_tools_directory(self, capsys, tmp_path):
        """A wheel install (no tools/ directory) cannot run fix commands; update
        refuses rather than executing anything."""
        # tmp_path deliberately has no tools/ subdirectory.
        with (
            patch('proteus.doctor.run_all_checks', return_value=_mixed_results()),
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor.subprocess.run') as mock_run,
        ):
            update_entry(dry_run=False)
        out = capsys.readouterr().out
        # The refusal is reported to the user with the reason.
        assert 'Cannot run fix commands' in out
        # Error contract: nothing executed when the source tree is absent.
        assert mock_run.call_count == 0

    def test_reports_command_failure_and_continues_batch(self, capsys, tmp_path):
        """A fix command that exits non-zero is reported with its return code,
        and the remaining fixes still run (one failure does not abort the run)."""
        (tmp_path / 'tools').mkdir()
        boom = subprocess.CalledProcessError(returncode=2, cmd='boom')
        with (
            patch('proteus.doctor.run_all_checks', return_value=_mixed_results()),
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor.subprocess.run', side_effect=[boom, Mock()]) as mock_run,
            patch('proteus.doctor.doctor_entry'),
        ):
            update_entry(dry_run=False)
        out = capsys.readouterr().out
        assert 'command failed' in out
        # Discrimination: the exit code is surfaced, and the second fix still ran.
        assert 'exit 2' in out
        assert mock_run.call_count == 2


class TestGitAndJuliaHelpers:
    """Subprocess-backed version/SCM helpers (mocked subprocess)."""

    def test_git_short_head_returns_hash(self):
        """A successful `git rev-parse --short HEAD` is stripped and returned."""
        with patch('proteus.doctor.subprocess.check_output', return_value='a1b2c3d\n'):
            head = _git_short_head('/some/repo')
        assert head == 'a1b2c3d'
        # Discrimination: the trailing newline is stripped, not returned raw.
        assert '\n' not in head

    def test_git_short_head_none_on_error(self):
        """A non-git path (CalledProcessError) yields None, not an exception."""
        err = subprocess.CalledProcessError(returncode=128, cmd='git')
        with patch('proteus.doctor.subprocess.check_output', side_effect=err):
            assert _git_short_head('/not/a/repo') is None
        # Error contract: a missing git binary is also swallowed to None.
        with patch('proteus.doctor.subprocess.check_output', side_effect=FileNotFoundError):
            assert _git_short_head('/not/a/repo') is None

    def test_julia_version_parses_last_token(self):
        """`julia --version` output is parsed to its trailing version token."""
        with patch(
            'proteus.doctor.subprocess.check_output',
            return_value='julia version 1.11.2\n',
        ):
            assert _julia_version() == '1.11.2'
        # Discrimination: the word 'version' is not returned; only the number.
        with patch(
            'proteus.doctor.subprocess.check_output',
            return_value='julia version 1.10.0\n',
        ):
            assert _julia_version() == '1.10.0'

    def test_julia_version_none_when_missing(self):
        """A missing or erroring julia binary yields None rather than raising;
        a working julia returns its parsed version."""
        with patch('proteus.doctor.subprocess.check_output', side_effect=FileNotFoundError):
            assert _julia_version() is None
        # A non-zero exit (CalledProcessError) is also swallowed to None.
        err = subprocess.CalledProcessError(returncode=1, cmd='julia')
        with patch('proteus.doctor.subprocess.check_output', side_effect=err):
            assert _julia_version() is None
        # Discrimination: a working julia returns the version string, so None
        # is the error signal rather than a constant return.
        with patch(
            'proteus.doctor.subprocess.check_output',
            return_value='julia version 1.11.2\n',
        ):
            assert _julia_version() == '1.11.2'


class TestEditableAndImportedPaths:
    """Editable-install resolution and the import-path cross-check."""

    def test_editable_checkout_returns_file_path(self):
        """An editable install with a file:// direct_url returns its on-disk
        path, with URL-encoding decoded."""
        dist = Mock()
        dist.read_text.return_value = json.dumps(
            {'url': 'file:///home/me/my%20pkg', 'dir_info': {'editable': True}}
        )
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            path = _editable_checkout_path('fwl-aragog')
        assert path == '/home/me/my pkg'  # %20 decoded to a space
        # Discrimination: a non-file URL on an otherwise-editable dist is not
        # treated as a local checkout.
        dist.read_text.return_value = json.dumps(
            {'url': 'https://x/y', 'dir_info': {'editable': True}}
        )
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            assert _editable_checkout_path('fwl-aragog') is None

    def test_editable_checkout_none_for_non_editable(self):
        """A regular (non-editable) install returns None even though it has a
        direct_url, so the editable annotation is not falsely shown."""
        dist = Mock()
        dist.read_text.return_value = json.dumps(
            {'url': 'file:///opt/pkg', 'dir_info': {'editable': False}}
        )
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            assert _editable_checkout_path('fwl-aragog') is None
        # Error contract: malformed direct_url.json is swallowed to None.
        dist.read_text.return_value = '{not valid json'
        with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
            assert _editable_checkout_path('fwl-aragog') is None

    def test_imported_package_dir_from_top_level(self):
        """The import path is resolved via top_level.txt and find_spec, returning
        the directory of the module that ``import`` would actually load."""
        dist = Mock()
        dist.read_text.return_value = 'aragog\n'
        spec = Mock()
        spec.origin = '/env/site-packages/aragog/__init__.py'
        with (
            patch('proteus.doctor.importlib.metadata.distribution', return_value=dist),
            patch('proteus.doctor.importlib.util.find_spec', return_value=spec),
        ):
            path = _imported_package_dir('fwl-aragog')
        assert path == '/env/site-packages/aragog'
        # Discrimination: the file name is stripped to its directory.
        assert not path.endswith('__init__.py')

    def test_imported_package_dir_none_when_unresolvable(self):
        """When find_spec cannot locate the module (or the spec has no origin),
        the helper returns None instead of raising; a resolvable spec returns
        its directory."""
        dist = Mock()
        dist.read_text.return_value = 'aragog\n'
        with (
            patch('proteus.doctor.importlib.metadata.distribution', return_value=dist),
            patch('proteus.doctor.importlib.util.find_spec', return_value=None),
        ):
            assert _imported_package_dir('fwl-aragog') is None
        # A namespace-style spec with no origin is also unresolvable.
        spec_no_origin = Mock()
        spec_no_origin.origin = None
        with (
            patch('proteus.doctor.importlib.metadata.distribution', return_value=dist),
            patch('proteus.doctor.importlib.util.find_spec', return_value=spec_no_origin),
        ):
            assert _imported_package_dir('fwl-aragog') is None
        # Discrimination: a spec with an origin returns its directory, so None
        # is the unresolvable signal rather than a constant.
        spec_ok = Mock()
        spec_ok.origin = '/x/aragog/__init__.py'
        with (
            patch('proteus.doctor.importlib.metadata.distribution', return_value=dist),
            patch('proteus.doctor.importlib.util.find_spec', return_value=spec_ok),
        ):
            assert _imported_package_dir('fwl-aragog') == '/x/aragog'
