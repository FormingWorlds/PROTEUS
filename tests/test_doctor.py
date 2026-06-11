"""Unit tests for the PROTEUS doctor diagnostics (doctor.py).

Tests the structured check system: CheckResult, individual check
functions, pin-aware version comparison, and the doctor/update entry
points. Does not make real network calls or modify the installation.

Testing standards:
  - docs/How-to/testing.md
"""

from __future__ import annotations

import importlib.metadata
import io
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
    _collect_environment_info,
    _conda_build_lines,
    _editable_checkout_path,
    _git_dirty,
    _git_head,
    _git_short_head,
    _imported_package_dir,
    _julia_version,
    _julia_version_at,
    _print_support_prompt,
    _run_fix_command,
    _Tee,
    _write_failure_log,
    check_env_var,
    check_fwl_data,
    check_git_module,
    check_julia,
    check_python_package,
    doctor_entry,
    run_all_checks,
    run_doctor,
    run_update,
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
        """An unset variable fails with human advice that update must not run."""
        env = os.environ.copy()
        env.pop('NONEXISTENT_VAR', None)
        with patch.dict(os.environ, env, clear=True):
            r = check_env_var('NONEXISTENT_VAR')
        assert r.status == FAIL
        # The fix is a placeholder export line, not a runnable command, so it
        # must be flagged non-auto-fixable; otherwise `proteus update` feeds
        # 'export VAR=<path>  # ...' to the shell and it errors on '<path>'.
        assert r.auto_fixable is False
        assert '<path>' in r.fix_cmd

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

    def test_pass_when_editable_dev_version_above_bound(self):
        """A setuptools-scm dev version a few commits past the tagged release
        satisfies a >= bound, even though it is a pre-release in PEP 440 terms."""
        from packaging.requirements import Requirement

        # 26.6.1.post1.dev8 sorts ABOVE 26.6.1, so it satisfies >=26.06.01;
        # the default SpecifierSet membership would wrongly exclude it as a
        # pre-release and report a failure.
        spec = Requirement('fwl-calliope>=26.06.01')
        with patch(
            'proteus.doctor.importlib.metadata.version',
            return_value='26.6.1.post1.dev8+g1e3ad73dd',
        ):
            with patch('proteus.doctor._editable_checkout_path', return_value=None):
                r = check_python_package('fwl-calliope', spec)
        assert r.status == PASS
        assert '26.6.1.post1.dev8' in r.message

    def test_fail_when_dev_version_below_bound(self):
        """Allowing pre-releases must not blanket-pass them: a dev version that
        sorts BELOW the bound still fails, so the check respects ordering."""
        from packaging.requirements import Requirement

        # 26.6.1.dev1 is a dev release of 26.6.1 and sorts BELOW it, so it does
        # not satisfy >=26.6.1; a naive prereleases=True must not let it pass.
        spec = Requirement('fwl-calliope>=26.06.01')
        with patch('proteus.doctor.importlib.metadata.version', return_value='26.6.1.dev1'):
            with patch('proteus.doctor._editable_checkout_path', return_value=None):
                r = check_python_package('fwl-calliope', spec)
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


class TestCheckGitModulePinMismatch:
    """check_git_module's fix suggestion when a git submodule is off its pin."""

    @staticmethod
    def _pins() -> dict:
        """Pinned refs distinct from the head returned by the patched helper."""
        return {'socrates': {'ref': 'a' * 40}, 'agni': {'ref': 'b' * 40}}

    def test_socrates_mismatch_fix_also_rebuilds_agni(self, tmp_path, monkeypatch):
        """A SOCRATES checkout off its pin must suggest a fix that rebuilds AGNI.

        Rebuilding SOCRATES re-clones its tree and deletes the Julia wrapper
        sources AGNI generates inside it, so a SOCRATES-only refresh leaves AGNI
        unable to precompile. The suggested command therefore chains the AGNI
        rebuild after the SOCRATES one, in that order.
        """
        monkeypatch.setenv('RAD_DIR', str(tmp_path))
        with (
            patch('proteus.doctor._module_pins', return_value=self._pins()),
            patch('proteus.doctor._git_head', return_value='c' * 40),
            patch('proteus.doctor._get_socrates_version', return_value='soc-test'),
        ):
            r = check_git_module('SOCRATES', {})
        assert r.status == WARN
        assert 'tools/get_socrates.sh' in r.fix_cmd
        assert 'tools/get_agni.sh' in r.fix_cmd
        # The SOCRATES rebuild must run before the AGNI one, else AGNI would
        # regenerate against the stale tree and be wiped again.
        assert r.fix_cmd.index('get_socrates.sh') < r.fix_cmd.index('get_agni.sh')
        # Discrimination: the broken behaviour was a bare SOCRATES refresh with
        # no AGNI step; that command has no shell chain.
        assert '&&' in r.fix_cmd

    def test_agni_mismatch_fix_does_not_chain_socrates(self, tmp_path):
        """An AGNI checkout off its pin rebuilds only AGNI.

        AGNI's own rebuild regenerates the wrappers from the current SOCRATES
        tree, so it needs no SOCRATES step; chaining one would re-clone SOCRATES
        needlessly. This guards the asymmetry: only a SOCRATES refresh drags
        AGNI along, never the reverse.
        """
        with (
            patch('proteus.doctor._module_pins', return_value=self._pins()),
            patch('proteus.doctor._git_head', return_value='c' * 40),
            patch('proteus.doctor._get_agni_version', return_value='agni-test'),
        ):
            r = check_git_module('AGNI', {'agni': str(tmp_path)})
        assert r.status == WARN
        assert 'tools/get_agni.sh' in r.fix_cmd
        # Discrimination: the AGNI fix must not pull in a SOCRATES rebuild.
        assert 'get_socrates.sh' not in r.fix_cmd
        assert '&&' not in r.fix_cmd

    def test_update_runs_the_chained_socrates_fix(self, tmp_path, monkeypatch):
        """`proteus update` actually executes the chained SOCRATES fix.

        Constructing the right string is useless unless `proteus update` runs
        it. This drives check_git_module's real off-pin result through
        update_entry with the runner mocked, and asserts the chained command,
        AGNI rebuild and all, reaches the shell unattended exactly once.
        """
        monkeypatch.setenv('RAD_DIR', str(tmp_path))
        (tmp_path / 'tools').mkdir()
        with (
            patch('proteus.doctor._module_pins', return_value=self._pins()),
            patch('proteus.doctor._git_head', return_value='c' * 40),
            patch('proteus.doctor._get_socrates_version', return_value='soc-test'),
        ):
            warn = check_git_module('SOCRATES', {})
        # An off-pin module must be auto-fixable, else update would only print
        # the advice and the chain would never run.
        assert warn.status == WARN and warn.auto_fixable
        with (
            patch('proteus.doctor.run_all_checks', return_value=[warn]),
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor._run_fix_command', return_value=0) as mock_fix,
            patch('proteus.doctor.doctor_entry', return_value=True),
        ):
            update_entry(dry_run=False)
        assert mock_fix.call_count == 1
        ran = mock_fix.call_args_list[0].args[0]
        assert 'get_socrates.sh' in ran and 'get_agni.sh' in ran
        # Discrimination: a regression dropping the AGNI step would still run
        # the SOCRATES half, so assert the chain survived end to end.
        assert '&&' in ran


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
            patch('proteus.doctor._run_fix_command') as mock_fix,
        ):
            update_entry(dry_run=True)
        out = capsys.readouterr().out
        assert 'Dry run' in out
        assert 'pip install -U fwl-mors' in out
        # Error contract: dry run must not execute any fix command.
        assert mock_fix.call_count == 0

    def test_executes_each_fix_from_repo_root_then_rechecks(self, tmp_path):
        """A real update runs one command per fixable check from the repo root
        and re-runs the diagnostics afterwards."""
        (tmp_path / 'tools').mkdir()
        with (
            patch('proteus.doctor.run_all_checks', return_value=_mixed_results()),
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor._run_fix_command', return_value=0) as mock_fix,
            patch('proteus.doctor.doctor_entry') as mock_recheck,
        ):
            update_entry(dry_run=False)
        # Two fixable checks (Julia + mors) -> two commands.
        assert mock_fix.call_count == 2
        # Commands run from the repo root, not the process cwd (cwd is arg 2).
        assert all(c.args[1] == tmp_path for c in mock_fix.call_args_list)
        # Diagnostics re-run exactly once after the fixes.
        assert mock_recheck.call_count == 1

    def test_manual_advice_is_shown_but_never_executed(self, capsys, tmp_path):
        """An unset env var carries human advice ('export VAR=<path>'), not a
        runnable command. update lists it under manual action and never passes
        it to the shell, while still running the genuinely fixable check."""
        (tmp_path / 'tools').mkdir()
        results = [
            CheckResult(
                'FC_DIR',
                'environment',
                FAIL,
                'not set',
                'export FC_DIR=<path>  # add to your shell rc file',
                auto_fixable=False,
            ),
            CheckResult('AGNI', 'versions', WARN, 'differs from pin', 'bash tools/get_agni.sh'),
        ]
        with (
            patch('proteus.doctor.run_all_checks', return_value=results),
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor._run_fix_command', return_value=0) as mock_fix,
            patch('proteus.doctor.doctor_entry', return_value=True),
        ):
            update_entry(dry_run=False)
        # Only the runnable AGNI fix is executed; the export advice is not.
        assert mock_fix.call_count == 1
        ran = mock_fix.call_args_list[0].args[0]
        assert ran == 'bash tools/get_agni.sh'
        assert 'export FC_DIR' not in ran
        # The manual item is still surfaced so the user knows to act on it.
        out = capsys.readouterr().out
        assert 'manual action' in out
        assert 'FC_DIR' in out

    def test_refuses_to_fix_without_tools_directory(self, capsys, tmp_path):
        """A wheel install (no tools/ directory) cannot run fix commands; update
        refuses rather than executing anything."""
        # tmp_path deliberately has no tools/ subdirectory.
        with (
            patch('proteus.doctor.run_all_checks', return_value=_mixed_results()),
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor._run_fix_command') as mock_fix,
        ):
            result = update_entry(dry_run=False)
        out = capsys.readouterr().out
        # The refusal is reported to the user with the reason.
        assert 'Cannot run fix commands' in out
        # Error contract: nothing executed, and the install is reported unhealthy.
        assert mock_fix.call_count == 0
        assert result is False

    def test_reports_command_failure_and_continues_batch(self, capsys, tmp_path):
        """A fix command that exits non-zero is reported with its return code,
        and the remaining fixes still run (one failure does not abort the run)."""
        (tmp_path / 'tools').mkdir()
        with (
            patch('proteus.doctor.run_all_checks', return_value=_mixed_results()),
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor._run_fix_command', side_effect=[2, 0]) as mock_fix,
            patch('proteus.doctor.doctor_entry', return_value=True),
        ):
            result = update_entry(dry_run=False)
        out = capsys.readouterr().out
        assert 'command failed' in out
        # Discrimination: the exit code is surfaced, and the second fix still ran.
        assert 'exit 2' in out
        assert mock_fix.call_count == 2
        # A failed fix makes the whole run unhealthy even if the re-check passes.
        assert result is False

    def test_unfixable_failure_is_reported_not_called_all_clear(self, capsys):
        """A failing check with no automatic fix is reported as such, not as
        'All checks passed', and the run is flagged unhealthy."""
        results = [
            CheckResult('FWL_DATA', 'environment', PASS, '/data', None),
            CheckResult('SOCRATES', 'environment', FAIL, 'build missing', None),
        ]
        with patch('proteus.doctor.run_all_checks', return_value=results):
            result = update_entry(dry_run=False)
        out = capsys.readouterr().out
        # The false all-clear must not appear when an unfixable failure exists.
        assert 'All checks passed' not in out
        assert 'none of them have an automatic fix' in out
        assert result is False

    def test_unfixable_warning_only_is_healthy(self, capsys):
        """A warning with no fix is surfaced but does not mark the install
        unhealthy: warnings are not failures."""
        results = [
            CheckResult('FWL_DATA/spectral_files', 'data', WARN, 'empty', None),
        ]
        with patch('proteus.doctor.run_all_checks', return_value=results):
            result = update_entry(dry_run=False)
        out = capsys.readouterr().out
        assert 'none of them have an automatic fix' in out
        # Discrimination: a warning-only run is still healthy (True), unlike a FAIL.
        assert result is True


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


class TestEnvironmentInfo:
    """The auto-collected environment block written into the failure log."""

    def test_block_reports_resolved_versions_not_just_package_names(self, monkeypatch):
        """The block reports each package's resolved version (or '(not
        installed)'), the platform, and the interpreter, so the log alone is
        enough to diagnose an install problem."""
        monkeypatch.setattr('proteus.doctor._julia_version', lambda: '1.12.6')
        monkeypatch.setattr('proteus.doctor._conda_build_lines', list)

        # Return a sentinel version for one package and mark the rest absent, so
        # the assertions fail if the version lookup itself were broken (which a
        # bare `'fwl-proteus:' in info` substring check would not catch).
        def fake_version(pkg):
            if pkg == 'fwl-proteus':
                return '9.9.9-sentinel'
            raise importlib.metadata.PackageNotFoundError(pkg)

        monkeypatch.setattr('proteus.doctor.importlib.metadata.version', fake_version)
        info = _collect_environment_info()
        assert 'platform:' in info and 'python:' in info
        assert 'julia (on PATH): 1.12.6' in info
        # The resolved version is reported, not merely the package name.
        assert 'fwl-proteus: 9.9.9-sentinel' in info
        # Discrimination: a genuinely-absent package renders as not installed,
        # so a lookup that silently returned nothing would fail this.
        assert 'fwl-mors: (not installed)' in info
        # This is the environment block, not the doctor check report.
        assert info.startswith('=== Environment')

    def test_path_vars_are_flagged_missing_and_bound_julia_is_reported(self, monkeypatch):
        """A path variable pointing at a non-existent location is flagged, an
        unset variable is shown as unset, and the juliacall-bound Julia is
        reported separately from the one on PATH."""
        monkeypatch.setattr('proteus.doctor._julia_version', lambda: '1.12.6')
        monkeypatch.setattr('proteus.doctor._julia_version_at', lambda exe: '1.11.5')
        monkeypatch.setattr('proteus.doctor._conda_build_lines', list)
        monkeypatch.setenv('FWL_DATA', '/no/such/path/fwl')
        monkeypatch.setenv('PYTHON_JULIAPKG_EXE', '/opt/julia/bin/julia')
        monkeypatch.delenv('RAD_DIR', raising=False)
        info = _collect_environment_info()
        # A path that does not exist is flagged, an unset var is shown as unset.
        assert 'FWL_DATA=/no/such/path/fwl  (MISSING)' in info
        assert 'RAD_DIR=(unset)' in info
        # The bound Julia is reported and is distinguished from the PATH one.
        assert 'julia (juliacall-bound): 1.11.5' in info
        assert 'julia (on PATH): 1.12.6' in info

    def test_collection_survives_a_broken_package_distribution(self, monkeypatch):
        """A corrupt distribution (version() raising something other than
        PackageNotFoundError) does not abort collection: the block still reports
        the other fields and flags the bad package rather than crashing."""
        monkeypatch.setattr('proteus.doctor._conda_build_lines', list)

        def broken_version(pkg):
            if pkg == 'fwl-proteus':
                raise ValueError('corrupt METADATA')
            raise importlib.metadata.PackageNotFoundError(pkg)

        monkeypatch.setattr('proteus.doctor.importlib.metadata.version', broken_version)
        info = _collect_environment_info()
        # Collection continued past the broken package (platform line present).
        assert 'platform:' in info
        # The broken package is flagged, not silently dropped.
        assert 'fwl-proteus: (version unavailable' in info
        assert 'fwl-mors: (not installed)' in info

    def test_failure_log_prepends_env_and_strips_colour(self, tmp_path, monkeypatch):
        """The log file leads with the environment block and the run transcript
        with ANSI colour codes removed, and is named for the command."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            'proteus.doctor._collect_environment_info', lambda: 'ENV_BLOCK_SENTINEL\n'
        )
        path = _write_failure_log('doctor', 'before \x1b[31mRED\x1b[0m after')
        assert path.parent == tmp_path
        assert path.name.startswith('proteus_doctor_') and path.name.endswith('.log')
        text = path.read_text()
        assert 'ENV_BLOCK_SENTINEL' in text
        # ANSI escape codes are stripped, but the words survive
        assert '\x1b[' not in text
        assert 'before RED after' in text

    def test_failure_log_leaves_plain_text_unchanged(self, tmp_path, monkeypatch):
        """A transcript with no colour codes is written through verbatim, so the
        ANSI stripping never mangles ordinary output."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr('proteus.doctor._collect_environment_info', lambda: 'E\n')
        path = _write_failure_log('doctor', 'plain line one\nplain line two\n')
        text = path.read_text()
        assert 'plain line one\nplain line two' in text
        # Discrimination: the transcript section header precedes the transcript.
        assert text.index('=== proteus doctor output ===') < text.index('plain line one')

    def test_failure_log_falls_back_to_tempdir_when_cwd_unwritable(self, tmp_path, monkeypatch):
        """If the working directory cannot be written, the log still lands in the
        system temp directory rather than crashing the command."""
        cwd = tmp_path / 'cwd'
        cwd.mkdir()
        fallback = tmp_path / 'fallback'
        fallback.mkdir()
        monkeypatch.setattr('proteus.doctor.Path.cwd', classmethod(lambda cls: cwd))
        monkeypatch.setattr('proteus.doctor.tempfile.gettempdir', lambda: str(fallback))
        monkeypatch.setattr('proteus.doctor._collect_environment_info', lambda: 'E\n')

        real_write = type(cwd).write_text

        def picky_write(self, data, *args, **kwargs):
            if self.parent == cwd:
                raise OSError('cannot write here')
            return real_write(self, data, *args, **kwargs)

        monkeypatch.setattr('proteus.doctor.Path.write_text', picky_write, raising=False)
        path = _write_failure_log('doctor', 'body-text')
        # The write fell through to the temp directory, not the unwritable cwd.
        assert path is not None and path.parent == fallback
        assert 'body-text' in path.read_text()

    def test_failure_log_returns_none_when_no_directory_writable(self, tmp_path, monkeypatch):
        """When neither the working directory nor temp is writable, the writer
        reports failure with None rather than raising; with a writable directory
        it returns a real path, so None is the unwritable signal, not a constant."""
        monkeypatch.setattr('proteus.doctor._collect_environment_info', lambda: 'E\n')
        monkeypatch.chdir(tmp_path)
        # Control: a writable working directory yields an actual log file.
        ok_path = _write_failure_log('doctor', 'control')
        assert ok_path is not None and ok_path.exists()

        def always_fail(self, *args, **kwargs):
            raise OSError('no space left')

        monkeypatch.setattr('proteus.doctor.Path.write_text', always_fail, raising=False)
        assert _write_failure_log('doctor', 'x') is None


class TestRunLogging:
    """run_doctor / run_update write a log and a help prompt on failure only."""

    def test_doctor_failure_writes_log_and_points_user_to_it(
        self, tmp_path, monkeypatch, capsys
    ):
        """A failing check writes a named log file in the working directory and
        the prompt tells the user that exact path plus the support channels."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr('proteus.doctor._collect_environment_info', lambda: 'ENVINFO\n')
        with patch('proteus.doctor.run_all_checks', return_value=_mixed_results()):
            result = run_doctor()
        assert result is False
        logs = list(tmp_path.glob('proteus_doctor_*.log'))
        assert len(logs) == 1
        log_text = logs[0].read_text()
        # the log is self-contained: env block + the failing check transcript
        assert 'ENVINFO' in log_text and 'Julia' in log_text
        out = capsys.readouterr().out
        # the user is told the exact file to send and where to send it
        assert str(logs[0]) in out
        assert 'proteus_dev@formingworlds.space' in out
        assert 'github.com/FormingWorlds/PROTEUS/issues' in out
        assert 'discussions' in out

    def test_doctor_all_pass_writes_no_log_and_no_prompt(self, tmp_path, monkeypatch, capsys):
        """A clean run leaves no log file behind and prints no support prompt."""
        monkeypatch.chdir(tmp_path)
        passing = [CheckResult('FWL_DATA', 'environment', PASS, '/data', None)]
        with patch('proteus.doctor.run_all_checks', return_value=passing):
            result = run_doctor()
        assert result is True
        assert list(tmp_path.glob('proteus_*.log')) == []
        out = capsys.readouterr().out
        assert 'proteus_dev@formingworlds.space' not in out

    def test_doctor_json_mode_is_not_logged(self, tmp_path, monkeypatch, capsys):
        """JSON output is for scripts: no log file and no prompt, just JSON."""
        monkeypatch.chdir(tmp_path)
        with patch('proteus.doctor.run_all_checks', return_value=_mixed_results()):
            result = run_doctor(output_json=True)
        assert result is False  # a failing check still reports False
        assert list(tmp_path.glob('proteus_*.log')) == []
        parsed = json.loads(capsys.readouterr().out)
        assert len(parsed) == 3

    def test_update_failed_fix_captures_command_output_in_log(
        self, tmp_path, monkeypatch, capsys
    ):
        """When a fix command fails, its own output (stdout and stderr), not just
        its exit code, is captured in the log, so the log alone explains why the
        fix failed; the run is reported as not healthy and the prompt fires."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'tools').mkdir()
        monkeypatch.setattr('proteus.doctor._repo_root', lambda: tmp_path)
        monkeypatch.setattr('proteus.doctor._collect_environment_info', lambda: 'ENVINFO\n')
        # A real fix command that writes to both streams then fails. stderr is
        # merged into stdout by the runner, so both markers must reach the log.
        fix = 'echo OUT_MARKER_A; echo ERR_MARKER_B 1>&2; exit 7'
        failing = [CheckResult('Julia', 'environment', FAIL, 'broken', fix)]
        with patch('proteus.doctor.run_all_checks', return_value=failing):
            result = run_update()
        assert result is False
        logs = list(tmp_path.glob('proteus_update_*.log'))
        assert len(logs) == 1
        log_text = logs[0].read_text()
        # The actual reason (the command's own output) is in the log, not lost.
        assert 'OUT_MARKER_A' in log_text
        assert 'ERR_MARKER_B' in log_text
        assert 'exit 7' in log_text
        out = capsys.readouterr().out
        assert str(logs[0]) in out
        assert 'proteus_dev@formingworlds.space' in out

    def test_doctor_crash_is_logged_with_traceback_and_prompt(
        self, tmp_path, monkeypatch, capsys
    ):
        """An unexpected exception is captured, not propagated as a bare
        traceback: a log carrying the traceback is written, the support prompt
        is shown, and the command reports failure."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr('proteus.doctor._collect_environment_info', lambda: 'ENVINFO\n')
        with patch('proteus.doctor.run_all_checks', side_effect=RuntimeError('boom_xyz_123')):
            result = run_doctor()
        assert result is False
        logs = list(tmp_path.glob('proteus_doctor_*.log'))
        assert len(logs) == 1
        log_text = logs[0].read_text()
        # The traceback and the exception message are in the log.
        assert 'Traceback' in log_text and 'boom_xyz_123' in log_text
        captured = capsys.readouterr()
        assert 'proteus_dev@formingworlds.space' in captured.out
        # The crash is surfaced to the user too, not buried only in the log.
        assert 'boom_xyz_123' in (captured.out + captured.err)

    def test_doctor_prompts_even_when_log_cannot_be_written(self, monkeypatch, capsys):
        """If the log cannot be saved anywhere, the user is still told how to get
        help (with the output to copy) rather than getting a silent failure."""
        monkeypatch.setattr('proteus.doctor._collect_environment_info', lambda: 'E\n')

        def always_fail(self, *args, **kwargs):
            raise OSError('disk full')

        monkeypatch.setattr('proteus.doctor.Path.write_text', always_fail, raising=False)
        with patch('proteus.doctor.run_all_checks', return_value=_mixed_results()):
            result = run_doctor()
        assert result is False
        out = capsys.readouterr().out
        # The fallback message and the support channel are both shown.
        assert 'could not be written' in out
        assert 'proteus_dev@formingworlds.space' in out


class TestTee:
    """The dual-stream writer that mirrors console output into the log buffer."""

    def test_mirrors_writes_to_both_streams_and_reports_length(self):
        """Every write reaches both the primary and the mirror, and write()
        returns the character count its callers (io machinery) expect."""
        primary, mirror = io.StringIO(), io.StringIO()
        tee = _Tee(primary, mirror)
        n = tee.write('hello')
        assert n == len('hello')
        assert primary.getvalue() == 'hello'
        assert mirror.getvalue() == 'hello'

    def test_isatty_follows_the_primary_stream(self):
        """isatty is delegated to the primary so colour is kept for the terminal
        and suppressed for a non-terminal primary; a regression that hard-coded
        either answer would fail one of these cases."""
        tty_primary = io.StringIO()
        tty_primary.isatty = lambda: True
        assert _Tee(tty_primary, io.StringIO()).isatty() is True
        # A plain StringIO primary is not a terminal.
        assert _Tee(io.StringIO(), io.StringIO()).isatty() is False


class TestRunFixCommand:
    """The fix-command runner that streams subprocess output through stdout."""

    def test_streams_combined_output_and_returns_exit_code(self, tmp_path, capsys):
        """Both stdout and stderr of the child are echoed through the parent's
        stdout (so the log can capture them) and the real exit code is returned;
        a non-zero exit is reported faithfully, not swallowed."""
        cmd = 'echo to_out; echo to_err 1>&2; exit 5'
        code = _run_fix_command(cmd, tmp_path)
        assert code == 5
        out = capsys.readouterr().out
        # stderr was merged into stdout, so both markers appear on stdout.
        assert 'to_out' in out and 'to_err' in out

    def test_successful_command_returns_zero(self, tmp_path, capsys):
        """A command that succeeds returns 0 and its output is still streamed."""
        code = _run_fix_command('echo done_marker', tmp_path)
        assert code == 0
        assert 'done_marker' in capsys.readouterr().out


class TestVersionAndCondaHelpers:
    """Environment probes that shell out (mocked subprocess)."""

    def test_julia_version_at_parses_a_specific_binary(self):
        """The bound-Julia probe returns the trailing version token from a
        specific binary, and yields None when that binary is absent."""
        with patch(
            'proteus.doctor.subprocess.check_output', return_value='julia version 1.11.5\n'
        ):
            assert _julia_version_at('/opt/julia/bin/julia') == '1.11.5'
        # A missing binary is swallowed to None, not raised.
        with patch('proteus.doctor.subprocess.check_output', side_effect=FileNotFoundError):
            assert _julia_version_at('/nope/julia') is None

    def test_conda_build_lines_filters_to_the_clash_packages(self):
        """Only the HDF5/netCDF/MPI packages are returned, each indented; an
        unrelated package is excluded so the block stays focused."""
        listing = (
            '# packages in environment\n'
            'hdf5      1.14.3  mpi_mpich_h1\n'
            'libnetcdf 4.9.2   nompi_h2\n'
            'numpy     1.26.4  py312_0\n'
        )
        with patch('proteus.doctor.subprocess.check_output', return_value=listing):
            lines = _conda_build_lines()
        assert any('hdf5' in line and 'mpi_mpich' in line for line in lines)
        assert any('libnetcdf' in line for line in lines)
        # Discrimination: an unrelated package is not included.
        assert not any('numpy' in line for line in lines)

    def test_conda_build_lines_empty_when_conda_absent_or_errors(self):
        """A missing conda and a conda that exits non-zero both yield an empty
        list, not an error, so the environment block simply omits the section."""
        with patch('proteus.doctor.subprocess.check_output', side_effect=FileNotFoundError):
            assert _conda_build_lines() == []
        # A conda that errors (non-zero exit) is also swallowed to empty.
        err = subprocess.CalledProcessError(returncode=1, cmd='conda')
        with patch('proteus.doctor.subprocess.check_output', side_effect=err):
            assert _conda_build_lines() == []


class TestSupportPromptAndCliExit:
    """The support prompt rendering and the CLI exit-code contract."""

    def test_prompt_names_the_log_and_all_support_channels(self, tmp_path, capsys):
        """With a saved log the prompt gives its exact path, the support email,
        and both GitHub channels, so the user knows what to send and where."""
        log = tmp_path / 'proteus_doctor_x.log'
        _print_support_prompt('doctor', log)
        out = capsys.readouterr().out
        assert str(log) in out
        assert 'proteus_dev@formingworlds.space' in out
        assert 'github.com/FormingWorlds/PROTEUS/issues' in out
        assert 'discussions' in out

    def test_prompt_handles_a_missing_log_path(self, capsys):
        """With no log written the prompt says so and tells the user to copy the
        output instead, rather than printing a bogus path."""
        _print_support_prompt('update', None)
        out = capsys.readouterr().out
        assert 'could not be written' in out
        assert 'copy the output above' in out
        # Discrimination: there is no 'saved to' line when there is no file.
        assert 'saved to' not in out

    def test_doctor_cli_exits_nonzero_on_failure(self, tmp_path, monkeypatch):
        """`proteus doctor` returns a non-zero process exit code when a check
        fails, so scripts and CI can gate on it."""
        from click.testing import CliRunner

        from proteus.cli import doctor as doctor_cmd

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr('proteus.doctor._collect_environment_info', lambda: 'E\n')
        with patch('proteus.doctor.run_all_checks', return_value=_mixed_results()):
            result = CliRunner().invoke(doctor_cmd, [])
        # Exit code 1 specifically (a click usage error would be 2), and the
        # failure path actually ran, so the support prompt is in the output.
        assert result.exit_code == 1
        assert 'proteus_dev@formingworlds.space' in result.output

    def test_doctor_cli_exits_zero_when_clean(self, tmp_path, monkeypatch):
        """A clean diagnose exits zero, so a passing check does not break a
        `proteus doctor && ...` chain."""
        from click.testing import CliRunner

        from proteus.cli import doctor as doctor_cmd

        monkeypatch.chdir(tmp_path)
        passing = [CheckResult('FWL_DATA', 'environment', PASS, '/data', None)]
        with patch('proteus.doctor.run_all_checks', return_value=passing):
            result = CliRunner().invoke(doctor_cmd, [])
        assert result.exit_code == 0
        # Discrimination: no failure exit means no support prompt was printed.
        assert 'proteus_dev@formingworlds.space' not in result.output
