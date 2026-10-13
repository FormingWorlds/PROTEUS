"""Unit tests for the PROTEUS doctor diagnostics (doctor.py).

Tests the structured check system: CheckResult, individual check
functions, pin-aware version comparison, and the doctor/update entry
points. Does not make real network calls or modify the installation.

Testing standards:
  - docs/How-to/testing.md
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from helpers import (
    PROTEUS_ROOT,
    git,
    git_call_is_read_only,
    install_git_guard,
    scratch_proteus_root,
)

from proteus.doctor import (
    _GIT_NOTE_TIMEOUT_S,
    _MODULE_FOLDERS,
    _PAIRED_MODULES,
    _PYTHON_MODULE_REPOS,
    _STANDARD_DISTRIBUTIONS,
    _SUPPORT_EMAIL,
    ENVIRONMENT_VARS,
    FAIL,
    GIT_MODULES,
    PASS,
    PYTHON_PACKAGES,
    STANDARD_GIT_MODULES,
    STANDARD_MODULE_PACKAGES,
    WARN,
    CheckResult,
    _below_floor_advice,
    _collect_environment_info,
    _conda_build_lines,
    _dependency_specs,
    _editable_checkout_path,
    _floor_tag_name,
    _get_script_for_package,
    _git_dirty,
    _git_head,
    _git_note_output,
    _git_short_head,
    _git_stdout,
    _imported_package_dir,
    _julia_version,
    _julia_version_at,
    _names_obliqua,
    _print_support_prompt,
    _release_tag_note,
    _run_fix_command,
    _setup_command,
    _Tee,
    _write_failure_log,
    check_cvode,
    check_env_var,
    check_fwl_data,
    check_git_module,
    check_julia,
    check_python_package,
    doctor_entry,
    is_standard_module_installed,
    run_all_checks,
    run_doctor,
    run_update,
    update_entry,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.fixture
def mock_own_clone(tmp_path):
    """Populate synthetic checkout and mock _is_own_clone to True."""
    (tmp_path / '.keep').touch()
    with patch('proteus.doctor._is_own_clone', return_value=True):
        yield


@pytest.mark.unit
def test_python_packages_excludes_standard_module_packages():
    """`proteus doctor` must list standard package extras in STANDARD_MODULE_PACKAGES."""
    assert set(STANDARD_MODULE_PACKAGES) == {'fwl-vulcan', 'atmodeller', 'fwl-morrigan'}
    assert set(STANDARD_MODULE_PACKAGES).isdisjoint(PYTHON_PACKAGES), (
        f'doctor must not check standard module extras as mandatory: '
        f'{set(STANDARD_MODULE_PACKAGES) & set(PYTHON_PACKAGES)}'
    )
    mandatory = {'fwl-proteus', 'fwl-aragog', 'fwl-calliope', 'fwl-zalmoxis'}
    assert mandatory <= set(PYTHON_PACKAGES), (
        f'doctor lost mandatory package checks: {mandatory - set(PYTHON_PACKAGES)}'
    )


@pytest.mark.unit
def test_git_modules_excludes_standard_git_modules():
    """`proteus doctor` must not check Obliqua as a mandatory git module."""
    assert 'Obliqua' not in GIT_MODULES
    assert 'Obliqua' in STANDARD_GIT_MODULES
    assert {'AGNI', 'SOCRATES'} <= set(GIT_MODULES)
    assert {'AGNI', 'SOCRATES'}.isdisjoint(STANDARD_GIT_MODULES)


@pytest.mark.unit
def test_standard_distributions_matches_standard_module_packages(tmp_path):
    """_STANDARD_DISTRIBUTIONS maps the 3 standard modules to their distribution packages."""
    assert set(_STANDARD_DISTRIBUTIONS.values()) == {'fwl-morrigan', 'fwl-vulcan', 'atmodeller'}
    assert set(_STANDARD_DISTRIBUTIONS.values()) == set(STANDARD_MODULE_PACKAGES)
    assert is_standard_module_installed('NoSuchModule', tmp_path) is False


@pytest.mark.unit
def test_python_packages_covers_every_pinned_fwl_dependency():
    """Every FWL package pinned in pyproject must be one doctor checks.

    A package carrying a version floor but absent from the list is a silent
    hole: doctor reports the whole stack green while the installed version sits
    below the bound the framework relies on. fwl-io is the costliest case to
    miss, because its floor exists to keep out releases with neither a timeout
    nor a retry on the data fetch, where an unreachable archive hangs a run
    rather than failing it.

    Both directions matter, and they fail differently. A pinned package missing
    from the list is never checked at all. A listed package whose bound the
    reader does not see is checked but always passes, since the version
    comparison is skipped when no specifier is found; that is the same silent
    green, reached from the other side. The spec reader matches dependency
    names containing ``fwl-``, so a pin written with an underscore, or a
    dependency dropped from pyproject, produces exactly that.

    Verifies:
    - Every dependency the spec reader discovers is checked.
    - Every checked package carries a version bound the reader actually reads,
      the project's own distribution aside.
    - The spec reader found something, so neither comparison is vacuously
      satisfied by an empty set.
    """
    specs = _dependency_specs()
    pinned = set(specs)
    assert pinned, 'no FWL dependencies were discovered; the comparisons below are vacuous'
    assert 'fwl-io' in pinned, (
        'fwl-io is no longer a pinned dependency, so this test is guarding nothing'
    )

    # `_dependency_specs` reads [project] dependencies and STANDARD_MODULE_PACKAGES
    # from pyproject.toml.
    checked = set(PYTHON_PACKAGES) | set(STANDARD_MODULE_PACKAGES)
    missing = pinned - checked
    assert not missing, (
        f'pyproject pins these FWL packages but doctor does not check them: {sorted(missing)}'
    )

    # fwl-proteus is the distribution being diagnosed, not one of its own
    # dependencies, so it carries no bound to read.
    enforced = {name for name, spec in specs.items() if spec and spec.specifier}
    unbounded = checked - enforced - {'fwl-proteus'}
    assert not unbounded, (
        f'doctor checks these but reads no version bound for them, so they pass at any '
        f'installed version: {sorted(unbounded)}'
    )


def test_dependency_specs_reads_standard_module_packages_and_omits_other_extras():
    """Dependency specs reader loads standard modules from pyproject extras and omits other extras."""
    specs = _dependency_specs()
    assert 'atmodeller' in specs
    assert 'fwl-vulcan' in specs
    assert 'fwl-morrigan' in specs
    assert 'torch' not in specs
    assert 'botorch' not in specs


def test_dependency_specs_ignores_malformed_optional_dependency():
    """Dependency specs reader ignores unparseable strings in pyproject extras."""
    malformed = {'project': {'optional-dependencies': {'test': ['invalid:::specifier']}}}
    with patch('proteus.doctor._read_pyproject', return_value=malformed):
        specs = _dependency_specs()
    assert specs == {}
    assert 'invalid:::specifier' not in specs


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
        # The fix is advice with a placeholder, not a runnable command, so it must
        # be non-auto-fixable; `proteus update` would fail on '<path>' in a shell.
        assert r.auto_fixable is False
        assert '<path>' in r.fix_cmd

    @pytest.mark.parametrize('name', ['FWL_DATA', 'RAD_DIR', 'FC_DIR', 'PYTHON_JULIAPKG_EXE'])
    def test_unset_install_var_advice_uses_the_conda_env(self, monkeypatch, name):
        """The advice for an unset install variable sets it in the conda env and reactivates it.

        Activating the already active env does not load the variable, so the
        advice deactivates first; without conda it names the exports file. It
        names no shell start file.
        """
        monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv('CONDA_DEFAULT_ENV', 'proteus')
        r = check_env_var(name)
        assert r.fix_cmd == (
            f'conda env config vars set {name}=<path>, then conda deactivate && '
            'conda activate proteus (without conda: source .proteus-env.sh in the PROTEUS folder)'
        )
        assert r.auto_fixable is False
        assert 'rc file' not in r.fix_cmd and 'export ' not in r.fix_cmd

    @pytest.mark.parametrize('active', ['base', None], ids=['base-active', 'no-env-active'])
    def test_advice_without_a_proteus_env_active_does_not_name_base(self, monkeypatch, active):
        """With base or no env active, the advice names the PROTEUS env as a placeholder.

        `conda env config vars set` without `-n` would set the variable in base;
        the advice passes `-n <your PROTEUS environment>` and activates that env.
        """
        if active:
            monkeypatch.setenv('CONDA_DEFAULT_ENV', active)
        else:
            monkeypatch.delenv('CONDA_DEFAULT_ENV', raising=False)
        monkeypatch.delenv('FWL_DATA', raising=False)
        r = check_env_var('FWL_DATA')
        assert r.fix_cmd == (
            'conda env config vars set -n <your PROTEUS environment> FWL_DATA=<path>, then '
            'conda activate <your PROTEUS environment> '
            '(without conda: source .proteus-env.sh in the PROTEUS folder)'
        )
        assert 'base' not in r.fix_cmd
        assert 'conda deactivate' not in r.fix_cmd

    def test_pythonjuliapkg_exe_path_is_validated(self, monkeypatch, tmp_path):
        """The env-var check of PYTHON_JULIAPKG_EXE warns for a path that does not exist."""
        assert ('PYTHON_JULIAPKG_EXE', True, None) in ENVIRONMENT_VARS
        monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(tmp_path / 'no-such' / 'julia'))
        r = check_env_var('PYTHON_JULIAPKG_EXE', validate_path=True)
        assert r.status == WARN
        assert str(tmp_path / 'no-such' / 'julia') in r.message

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
        (tmp_path / 'atmos_clim' / 'spectral_files').mkdir(parents=True)
        (tmp_path / 'atmos_clim' / 'spectral_files' / 'data.bin').touch()
        (tmp_path / 'star' / 'spectra' / 'solar').mkdir(parents=True)
        (tmp_path / 'star' / 'spectra' / 'solar' / 'sun.txt').touch()
        with patch.dict(os.environ, {'FWL_DATA': str(tmp_path)}):
            results = check_fwl_data()
        statuses = {r.name: r.status for r in results}
        assert statuses['FWL_DATA/atmos_clim/spectral_files'] == PASS
        assert statuses['FWL_DATA/star/spectra'] == PASS

    def test_reports_missing_subdirs(self, tmp_path):
        """Missing subdirectories warn with a fix command."""
        with patch.dict(os.environ, {'FWL_DATA': str(tmp_path)}):
            results = check_fwl_data()
        for r in results:
            assert r.status == WARN
            assert r.fix_cmd is not None

    @pytest.mark.parametrize('movable', [True, False])
    def test_an_older_layout_is_fixed_by_relocate(self, tmp_path, movable):
        """A data set in the older layout is fixed by moving it only when a relocate dry
        run would move it; otherwise, e.g. with wrong checksums, by its download."""
        from types import SimpleNamespace

        legacy = tmp_path / 'spectral_files' / 'Dayspring' / '48'
        legacy.mkdir(parents=True)
        (legacy / 'Dayspring.sf').touch()
        plan = SimpleNamespace(ready=[SimpleNamespace(legacy_dir=legacy)] if movable else [])
        with (
            patch.dict(os.environ, {'FWL_DATA': str(tmp_path)}),
            patch('fwl_io.relocate.plan_relocations', return_value=plan) as planned,
        ):
            fixes = {r.name: r.fix_cmd for r in check_fwl_data()}
        planned.assert_called_once_with(str(tmp_path))
        want = 'fwl-io relocate' if movable else 'proteus get spectral'
        assert fixes['FWL_DATA/atmos_clim/spectral_files'] == want
        assert fixes['FWL_DATA/star/spectra'] == 'proteus get stellar'

    def test_the_relocate_dry_run_runs_only_for_a_missing_set_with_older_data(self, tmp_path):
        """The dry run hashes the whole tree, so it runs only when a data set is missing
        and its older-layout folder holds files."""
        (tmp_path / 'atmos_clim' / 'spectral_files' / 'Dayspring').mkdir(parents=True)
        (tmp_path / 'spectral_files' / 'Dayspring').mkdir(parents=True)
        (tmp_path / 'spectral_files' / 'Dayspring' / 'old.sf').touch()
        with (
            patch.dict(os.environ, {'FWL_DATA': str(tmp_path)}),
            patch('fwl_io.relocate.plan_relocations') as planned,
        ):
            check_fwl_data()
        planned.assert_not_called()

    def test_relocate_is_proposed_through_a_symlinked_data_root(self, tmp_path):
        """FWL_DATA reached through a symlink still matches the resolved legacy folders a
        relocate dry run reports."""
        from types import SimpleNamespace

        real = tmp_path / 'real'
        (real / 'spectral_files' / 'Dayspring' / '48').mkdir(parents=True)
        (tmp_path / 'link').symlink_to(real)
        plan = SimpleNamespace(
            ready=[SimpleNamespace(legacy_dir=real / 'spectral_files' / 'Dayspring' / '48')]
        )
        with (
            patch.dict(os.environ, {'FWL_DATA': str(tmp_path / 'link')}),
            patch('fwl_io.relocate.plan_relocations', return_value=plan),
        ):
            fixes = {r.name: r.fix_cmd for r in check_fwl_data()}
        assert fixes['FWL_DATA/atmos_clim/spectral_files'] == 'fwl-io relocate'

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

    @pytest.fixture(autouse=True)
    def _no_bound_julia(self, monkeypatch):
        """Check the Julia on PATH unless a test sets PYTHON_JULIAPKG_EXE itself."""
        monkeypatch.delenv('PYTHON_JULIAPKG_EXE', raising=False)

    def test_pass_for_julia_113(self):
        """Julia 1.13.x passes, with the detected version reported and no fix."""
        for ver in ('1.13.1', '1.13.0'):
            with patch('proteus.doctor._julia_version', return_value=ver):
                r = check_julia()
            assert r.status == PASS
            assert r.message == ver
            assert r.fix_cmd is None

    @pytest.mark.parametrize(
        'ver',
        ['1.12.6', '1.11.5', '1.14.0', '1.10.4'],
        ids=['julia-1.12', 'julia-1.11', 'julia-1.14', 'julia-1.10'],
    )
    def test_julia_other_than_113_warns_and_names_obliqua(self, monkeypatch, ver):
        """Any Julia other than 1.13 is a warning, not a failure, that names Obliqua and 1.13.

        Obliqua does not resolve on 1.12, and 1.11, 1.10 and 1.14 are untested
        with PROTEUS. The advice adds
        1.13 and points PYTHON_JULIAPKG_EXE at it in the conda env, which
        ``proteus update`` does not do, and never changes the default Julia.
        """
        monkeypatch.setenv('CONDA_DEFAULT_ENV', 'proteus')
        with patch('proteus.doctor._julia_version', return_value=ver):
            r = check_julia()
        assert r.status == WARN
        assert r.status != FAIL
        assert r.message.startswith(f'{ver} (')
        assert 'Obliqua' in r.message
        assert 'tested on Julia 1.13' in r.message
        assert 'deprecated' not in r.message
        assert r.fix_cmd.startswith(
            'juliaup add 1.13, then conda env config vars set PYTHON_JULIAPKG_EXE=<the path that '
            'julia +1.13 -e \'print(joinpath(Sys.BINDIR, "julia"))\' prints>, then conda deactivate'
        )
        assert 'source .proteus-env.sh' in r.fix_cmd
        assert 'juliaup default' not in r.fix_cmd
        assert r.auto_fixable is False

    def test_fail_when_missing(self):
        """Missing Julia fails with install command."""
        with patch('proteus.doctor._julia_version', return_value=None):
            r = check_julia()
        assert r.status == FAIL
        assert 'julialang' in r.fix_cmd

    @staticmethod
    def _julia_stub(directory: Path, version: str) -> Path:
        """An executable ``julia`` in ``directory`` that reports ``version``."""
        directory.mkdir(parents=True)
        exe = directory / 'julia'
        exe.write_text(f'#!/bin/sh\necho "julia version {version}"\n')
        exe.chmod(0o755)
        return exe

    def test_checks_the_julia_in_pythonjuliapkg_exe_first(self, tmp_path, monkeypatch):
        """PYTHON_JULIAPKG_EXE at a 1.13.1 binary passes while PATH holds a 1.10 julia.

        Unsetting the variable makes the same check report the PATH 1.10 as a
        warning, so the pass comes from the variable and not from the PATH julia.
        """
        bound = self._julia_stub(tmp_path / 'juliaup' / 'bin', '1.13.1')
        self._julia_stub(tmp_path / 'path', '1.10.4')
        monkeypatch.setenv('PATH', str(tmp_path / 'path'))
        monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(bound))
        r = check_julia()
        assert r.status == PASS
        assert r.message == f'1.13.1 at {bound} (PYTHON_JULIAPKG_EXE)'
        assert r.fix_cmd is None
        monkeypatch.delenv('PYTHON_JULIAPKG_EXE')
        fallback = check_julia()
        assert fallback.status == WARN
        assert fallback.message.startswith('1.10.4 (')
        assert 'juliaup add 1.13' in fallback.fix_cmd
        assert 'juliaup default' not in fallback.fix_cmd

    @pytest.mark.parametrize('kind', ['missing', 'not-executable', 'directory'])
    def test_fails_when_pythonjuliapkg_exe_is_set_but_unusable(
        self, tmp_path, monkeypatch, kind
    ):
        """A PYTHON_JULIAPKG_EXE that is no executable file fails, even with a 1.13 julia on PATH.

        juliacall does not fall back to the PATH julia from a set variable; it
        fails at import. The message names the variable and its value, and the
        fix adds 1.13 and points the variable at it.
        """
        bound = tmp_path / 'bound' / 'julia'
        if kind == 'not-executable':
            self._julia_stub(bound.parent, '1.13.1').chmod(0o644)
        elif kind == 'directory':
            bound.mkdir(parents=True)
        self._julia_stub(tmp_path / 'path', '1.13.1')
        monkeypatch.setenv('PATH', str(tmp_path / 'path'))
        monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(bound))
        monkeypatch.setenv('CONDA_DEFAULT_ENV', 'proteus')
        r = check_julia()
        assert r.status == FAIL
        assert r.message == (
            f'PYTHON_JULIAPKG_EXE={bound} is not an executable file; juliacall does not '
            'fall back to the julia on PATH and fails at import'
        )
        assert r.fix_cmd.startswith(
            'juliaup add 1.13, then conda env config vars set PYTHON_JULIAPKG_EXE=<the path'
        )
        assert r.auto_fixable is False
        monkeypatch.delenv('PYTHON_JULIAPKG_EXE')
        assert check_julia().status == PASS

    @pytest.mark.parametrize(
        ('bound', 'sentence'),
        [(True, True), (False, False)],
        ids=['bound-1.11', 'path-1.11'],
    )
    def test_advice_for_a_bound_111_names_the_openssl_exception(
        self, tmp_path, monkeypatch, bound, sentence
    ):
        """A Julia 1.11 in PYTHON_JULIAPKG_EXE gets one more sentence: why the installer bound it.

        The installer binds 1.11 only after an OpenSSL_jll import failure with a
        Python that links OpenSSL < 3.5; a 1.11 on PATH was not bound by it.
        """
        exe = self._julia_stub(tmp_path / 'jl', '1.11.9')
        if bound:
            monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(exe))
        else:
            monkeypatch.setenv('PATH', str(exe.parent))
        r = check_julia()
        assert r.status == WARN
        extra = (
            '. The installer binds Julia 1.11 only when the import of PROTEUS fails on '
            'OpenSSL_jll because Python links an OpenSSL older than 3.5; after '
            "conda install -c conda-forge 'openssl>=3.5', Julia 1.13 can be used"
        )
        assert r.fix_cmd.endswith(extra) is sentence
        assert r.fix_cmd.startswith('juliaup add 1.13, then ')

    @pytest.mark.parametrize(
        'ver', ['1.130.2', '11.3.0', '1.1.3'], ids=['1.130', '11.3', '1.1']
    )
    def test_only_minor_13_of_major_1_passes(self, ver):
        """1.130, 11.3 and 1.1.3 hold the digits 1 and 13 or 3 but are not Julia 1.13."""
        with patch('proteus.doctor._julia_version', return_value=ver):
            r = check_julia()
        assert r.status == WARN
        assert r.message.startswith(f'{ver} (')


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

    def test_below_spec_editable_module_with_a_setup_script_gets_advice(self):
        """An editable Zalmoxis clone below its floor gets advice that proteus update does not
        run: the fetch, the detached checkout of the full ref of the tag that the pin reader
        picks after the fetch (the clone does not have the tag yet) and a reinstall. The clone
        is not the Zalmoxis folder of this PROTEUS root, so the setup script with --force,
        which replaces that folder, is not offered; nor are `git pull` or a re-run."""
        from packaging.requirements import Requirement

        spec = Requirement('fwl-zalmoxis>=26.07.17')
        with (
            patch('proteus.doctor.importlib.metadata.version', return_value='26.5.13'),
            patch(
                'proteus.doctor._editable_checkout_path',
                return_value='/home/u/PROTEUS/Zalmoxis',
            ),
            patch('proteus.doctor._is_own_clone', return_value=True),
            patch('proteus.doctor._tag_commits', return_value={'26.05.13': 'c0b8412' * 5}),
            patch('proteus.doctor._git_short_head', return_value='c0b8412'),
            patch('proteus.doctor._git_dirty', return_value=False),
            patch('proteus.doctor._git_note_output', return_value=(1, '')),
            patch('proteus.doctor._imported_package_dir', return_value=None),
        ):
            r = check_python_package('fwl-zalmoxis', spec)
        pins = PROTEUS_ROOT / 'tools' / '_module_pins.py'
        assert r.status == FAIL
        assert r.auto_fixable is False
        assert r.fix_cmd == (
            'git -C /home/u/PROTEUS/Zalmoxis fetch --tags && git -C /home/u/PROTEUS/Zalmoxis'
            ' checkout --detach "refs/tags/$(git -C /home/u/PROTEUS/Zalmoxis tag -l |'
            f' python {pins} fwl-zalmoxis floor-tag)"'
            ' && pip install -e /home/u/PROTEUS/Zalmoxis'
        )
        assert 'git pull' not in r.fix_cmd
        assert '--force' not in r.fix_cmd

    def test_below_spec_editable_branch_module_fix_keeps_git_pull(self, tmp_path):
        """A package without a setup script (fwl-io) in a developer's clone on a branch keeps
        `git pull && pip install -e .`, a command proteus update runs; the advice of the
        modules with a setup script must not swallow this case."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'io'
        checkout.mkdir()
        _init_test_repo(checkout)
        spec = Requirement('fwl-io>=99.0.0')
        with (
            patch('proteus.doctor.importlib.metadata.version', return_value='26.6.1'),
            patch('proteus.doctor._editable_checkout_path', return_value=str(checkout)),
            patch('proteus.doctor._imported_package_dir', return_value=None),
        ):
            r = check_python_package('fwl-io', spec)
        assert r.status == FAIL
        assert r.fix_cmd == f'cd {shlex.quote(str(checkout))} && git pull && pip install -e .'
        assert r.auto_fixable is True
        assert '--force' not in r.fix_cmd

    def test_below_spec_detached_clone_with_the_tag_gets_the_short_advice(self, tmp_path):
        """A MORS clone that has the floor tag gets the checkout of ``tags/26.10.06``, with the
        path shell-quoted (it holds a space); it is not the MORS folder of this PROTEUS root,
        so get_python_module.sh MORS --force is not offered."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'my clones' / 'MORS'
        _tagged_module(checkout)
        git(checkout, 'checkout', '-q', 'tags/26.10.05')
        spec = Requirement('fwl-mors>=26.10.6')
        with (
            patch('proteus.doctor.importlib.metadata.version', return_value='26.10.5'),
            patch('proteus.doctor._editable_checkout_path', return_value=str(checkout)),
            patch('proteus.doctor._imported_package_dir', return_value=None),
        ):
            r = check_python_package('fwl-mors', spec)
        quoted = shlex.quote(str(checkout))
        assert quoted.startswith("'")
        assert r.status == FAIL
        assert r.auto_fixable is False
        assert r.fix_cmd == (
            f'git -C {quoted} fetch --tags && git -C {quoted} checkout --detach'
            f' refs/tags/26.10.06 && pip install -e {quoted}'
        )

    def test_below_spec_editable_non_git_checkout_gets_the_pip_advice(self, tmp_path):
        """When the checkout of a package without a setup script is no git repository, the
        annotation says that it is not a git clone of its own, and the fix is the pip install
        of the release, run by hand, with no git command and no tag checkout."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'io'
        checkout.mkdir()
        spec = Requirement('fwl-io>=26.10.8')
        with (
            patch('proteus.doctor.importlib.metadata.version', return_value='26.6.1'),
            patch('proteus.doctor._editable_checkout_path', return_value=str(checkout)),
            patch('proteus.doctor._imported_package_dir', return_value=None),
        ):
            r = check_python_package('fwl-io', spec)
        assert r.status == FAIL
        assert r.fix_cmd == (
            f'pip install -U "fwl-io>=26.10.8"  # {shlex.quote(str(checkout))} is not a git'
            ' clone of its own, so no git command updates it; this installs the release in'
            ' place of the editable folder'
        )
        assert r.auto_fixable is False
        assert r.message == (
            '26.6.1 (requires >=26.10.8) [editable @ io; not a git clone of its own]'
        )

    def test_below_spec_wheel_install_fix_upgrades_via_pip(self):
        """With no editable checkout (a wheel install), neither branch applies:
        the fix upgrades from PyPI against the version bound."""
        from packaging.requirements import Requirement

        spec = Requirement('fwl-aragog>=26.07.04')
        with (
            patch('proteus.doctor.importlib.metadata.version', return_value='26.5.13'),
            patch('proteus.doctor._editable_checkout_path', return_value=None),
        ):
            r = check_python_package('fwl-aragog', spec)
        assert r.status == FAIL
        assert r.fix_cmd.startswith('pip install -U')
        # No git operation is offered when there is no checkout to operate on.
        assert 'git pull' not in r.fix_cmd
        assert 'get_aragog.sh' not in r.fix_cmd

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
            side_effect=PackageNotFoundError('fwl-aragog'),
        ):
            r = check_python_package('fwl-aragog', None)
        assert r.status == FAIL
        assert 'pip install' in r.fix_cmd

    @pytest.mark.parametrize(
        ('pkg', 'script'),
        [
            ('fwl-vulcan', 'tools/get_vulcan.sh'),
            ('fwl-morrigan', 'tools/get_morrigan.sh'),
            ('atmodeller', 'tools/get_atmodeller.sh'),
        ],
    )
    def test_warn_when_standard_module_not_installed(self, pkg, script, tmp_path):
        """A missing standard module package warns with its setup command."""
        from importlib.metadata import PackageNotFoundError

        root = tmp_path / 'repo'
        (root / script).parent.mkdir(parents=True, exist_ok=True)
        (root / script).touch()
        with (
            patch('proteus.doctor._repo_root', return_value=root),
            patch(
                'proteus.doctor.importlib.metadata.version',
                side_effect=PackageNotFoundError(pkg),
            ),
        ):
            r = check_python_package(pkg, None)
        assert r.status == WARN
        assert r.message == 'not installed'
        assert r.fix_cmd == f'cd {shlex.quote(str(root))} && bash {script}'

    @pytest.mark.parametrize(
        ('pkg', 'extra'),
        [
            ('fwl-vulcan', 'vulcan'),
            ('fwl-morrigan', 'morrigan'),
            ('atmodeller', 'atmodeller'),
        ],
    )
    def test_absent_standard_module_without_setup_script_suggests_pip_extra(self, pkg, extra):
        """When setup script is absent, the warning suggests installing the extra."""
        from importlib.metadata import PackageNotFoundError

        with (
            patch(
                'proteus.doctor.importlib.metadata.version',
                side_effect=PackageNotFoundError(pkg),
            ),
            patch('proteus.doctor._setup_command', return_value=None),
        ):
            r = check_python_package(pkg, None)
        assert r.status == WARN
        assert r.message == 'not installed'
        assert r.fix_cmd == f'pip install "fwl-proteus[{extra}]"'

    def test_absent_mandatory_package_still_fails(self):
        """A missing mandatory Python package still returns FAIL."""
        from importlib.metadata import PackageNotFoundError

        with patch(
            'proteus.doctor.importlib.metadata.version',
            side_effect=PackageNotFoundError('fwl-mors'),
        ):
            r = check_python_package('fwl-mors', None)
        assert r.status == FAIL
        assert r.fix_cmd == 'pip install fwl-mors'

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
            patch('proteus.doctor._is_own_clone', return_value=True),
            patch('proteus.doctor._git_short_head', return_value='d051902'),
            patch('proteus.doctor._git_dirty', return_value=False),
            patch('proteus.doctor._release_tag_note', return_value=''),
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


def _git_commit_all(path, message):
    """Stage and commit everything in the repo at the given path."""
    p = str(path)
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', message], check=True, capture_output=True)


def _detached_clone(tmp_path, checkout, late_tag: str) -> list[str]:
    """Clone a 3-commit remote to ``checkout`` at tag 26.10.05 on its first commit.

    After the clone, ``late_tag`` goes on the second commit in the remote
    only. Returns the 3 commit hashes in order.
    """
    source = tmp_path / 'source'
    source.mkdir()
    git(source, 'init', '-q', '-b', 'main')
    commits = []
    for i in range(3):
        (source / 'version.txt').write_text(f'{i}\n')
        git(source, 'add', 'version.txt')
        git(source, 'commit', '-q', '-m', f'commit {i}')
        commits.append(git(source, 'rev-parse', 'HEAD'))
    git(source, 'tag', '26.10.05', commits[0])
    remote = tmp_path / 'MORS.git'
    git(tmp_path, 'clone', '-q', '--bare', str(source), str(remote))
    git(tmp_path, 'clone', '-q', str(remote), str(checkout))
    git(checkout, 'checkout', '-q', 'tags/26.10.05')
    git(source, 'tag', late_tag, commits[1])
    git(source, 'push', '-q', str(remote), f'refs/tags/{late_tag}')
    return commits


def _stub_pip(tmp_path, monkeypatch):
    """Put a recording pip, and this test's python, first on PATH; return pip's log."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'pip.log'
    (bindir / 'pip').write_text(f'#!/bin/sh\necho "$@" >> {shlex.quote(str(log))}\n')
    (bindir / 'pip').chmod(0o755)
    (bindir / 'python').symlink_to(sys.executable)
    monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}{os.environ["PATH"]}')
    return log


def _tagged_module(path, annotated: bool = False) -> list[str]:
    """Build a repository with 3 commits on main, the tag 26.10.05 on the first and 26.10.06
    on the second; return the 3 commit hashes. ``annotated`` makes 26.10.06 an annotated tag,
    whose tag object differs from its commit."""
    path.mkdir(parents=True)
    git(path, 'init', '-q', '-b', 'main')
    commits = []
    for i in range(3):
        (path / 'version.txt').write_text(f'{i}\n')
        git(path, 'add', 'version.txt')
        git(path, 'commit', '-q', '-m', f'commit {i}')
        commits.append(git(path, 'rev-parse', 'HEAD'))
    git(path, 'tag', '26.10.05', commits[0])
    if annotated:
        git(path, 'tag', '-a', '-m', 'release', '26.10.06', commits[1])
    else:
        git(path, 'tag', '26.10.06', commits[1])
    return commits


def _check_editable_mors(checkout) -> CheckResult:
    """Run check_python_package on fwl-mors 26.10.6, editable at ``checkout``, floor 26.10.6."""
    from packaging.requirements import Requirement

    with (
        patch('proteus.doctor.importlib.metadata.version', return_value='26.10.6'),
        patch('proteus.doctor._editable_checkout_path', return_value=str(checkout)),
        patch('proteus.doctor._imported_package_dir', return_value=None),
    ):
        return check_python_package('fwl-mors', Requirement('fwl-mors>=26.10.6'))


class TestReleaseTagNote:
    """The version check of an editable clone names a branch or a commit that is not the floor tag."""

    def test_branch_clone_names_the_branch_and_the_tag(self, tmp_path):
        """A clone on the branch main, moved back to the commit of 26.10.05, names the branch
        and the tag 26.10.06 that the floor 26.10.6 selects; the check still passes, as the
        version satisfies the floor."""
        checkout = tmp_path / 'MORS'
        commits = _tagged_module(checkout)
        git(checkout, 'reset', '-q', '--hard', commits[0])
        r = _check_editable_mors(checkout)
        assert r.status == PASS
        assert r.message == (
            f'26.10.6 [editable @ MORS -> {commits[0][:7]};'
            ' on branch main; PROTEUS expects the release tag 26.10.06]'
        )

    @pytest.mark.parametrize('where', ['detached', 'branch'])
    def test_clone_newer_than_the_tag_gets_no_note(self, tmp_path, where):
        """A clone one commit past the floor tag, detached or on main, satisfies the >= floor:
        no note, as in the installer, which prints the neutral line of a newer clone. The same
        clone moved to a commit on a line of history without the tag gets the note, so the
        note is decided by the ancestry of HEAD."""
        checkout = tmp_path / 'MORS'
        commits = _tagged_module(checkout)
        if where == 'detached':
            git(checkout, 'checkout', '-q', '--detach', commits[2])
        r = _check_editable_mors(checkout)
        assert r.message == f'26.10.6 [editable @ MORS -> {commits[2][:7]}]'
        assert r.status == PASS
        git(checkout, 'checkout', '-q', '--detach', commits[0])
        (checkout / 'version.txt').write_text('side\n')
        git(checkout, 'commit', '-q', '-am', 'side')
        side = git(checkout, 'rev-parse', 'HEAD')
        assert _check_editable_mors(checkout).message.endswith(
            f'; at commit {side[:7]}, not the release tag; PROTEUS expects the release tag'
            ' 26.10.06]'
        )

    @pytest.mark.parametrize('source', ['CI variable', 'PROTEUS checkout'])
    def test_aragog_clone_on_the_paired_branch_gets_no_note(
        self, tmp_path, monkeypatch, source
    ):
        """An Aragog clone on the branch PROTEUS is on, here on a line of history without the
        tag, is the paired branch of the installer: no note. The PROTEUS branch comes from
        GITHUB_HEAD_REF or from the PROTEUS checkout. The same clone gets the note when PROTEUS
        is on another branch, and as fwl-mors, whose setup script pairs no branch."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'aragog'
        commits = _tagged_module(checkout)
        git(checkout, 'checkout', '-q', '-b', 'tl/paired-feature', commits[0])
        (checkout / 'version.txt').write_text('paired\n')
        git(checkout, 'commit', '-q', '-am', 'paired')
        root = tmp_path / 'PROTEUS'
        _init_test_repo(root)
        monkeypatch.delenv('GITHUB_HEAD_REF', raising=False)
        monkeypatch.delenv('GITHUB_REF_NAME', raising=False)
        if source == 'CI variable':
            monkeypatch.setenv('GITHUB_HEAD_REF', 'tl/paired-feature')
        else:
            git(root, 'checkout', '-q', '-b', 'tl/paired-feature')

        def note(dist: str) -> str:
            with patch('proteus.doctor._repo_root', return_value=root):
                return _release_tag_note(str(checkout), None, Requirement(f'{dist}>=26.10.6'))

        expected = '; on branch tl/paired-feature; PROTEUS expects the release tag 26.10.06'
        assert note('fwl-aragog') == ''
        assert note('fwl-mors') == expected
        monkeypatch.setenv('GITHUB_REF_NAME', 'other-feature')
        monkeypatch.delenv('GITHUB_HEAD_REF', raising=False)
        assert note('fwl-aragog') == expected

    def test_older_tag_is_named_as_found(self, tmp_path):
        """A clone at the older tag 26.10.05 is named by that tag and its commit, and the check
        only reads it: HEAD stays at that tag."""
        checkout = tmp_path / 'MORS'
        commits = _tagged_module(checkout)
        git(checkout, 'checkout', '-q', 'tags/26.10.05')
        r = _check_editable_mors(checkout)
        assert (
            f'; at tag 26.10.05 ({commits[0][:7]}), not the release tag;'
            ' PROTEUS expects the release tag 26.10.06]'
        ) in r.message
        assert git(checkout, 'rev-parse', 'HEAD') == commits[0]

    @pytest.mark.parametrize('annotated', [False, True], ids=['lightweight', 'annotated'])
    def test_clone_at_the_floor_tag_gets_no_note(self, tmp_path, annotated):
        """A clone detached at the floor tag gets the plain annotation; for an annotated tag
        the commit, not the tag object, is compared with HEAD."""
        checkout = tmp_path / 'MORS'
        commits = _tagged_module(checkout, annotated=annotated)
        git(checkout, 'checkout', '-q', 'tags/26.10.06')
        r = _check_editable_mors(checkout)
        assert r.message == f'26.10.6 [editable @ MORS -> {commits[1][:7]}]'
        assert r.status == PASS

    def test_branch_without_the_floor_tag_names_the_floor_release(self, tmp_path):
        """A branch clone whose tags hold no 26.10.6 names the floor release that PROTEUS
        expects, in the words of the installer, in place of a tag."""
        checkout = tmp_path / 'MORS'
        _tagged_module(checkout)
        git(checkout, 'tag', '-d', '26.10.06')
        r = _check_editable_mors(checkout)
        assert r.message.endswith('; on branch main; PROTEUS expects release 26.10.6]')
        assert 'release tag' not in r.message

    @pytest.mark.parametrize(
        'requirement',
        [None, 'fwl-mors==26.10.6', 'fwl-mors>=26.10.6,>=26.10.5'],
        ids=['no requirement', 'exact pin', 'two floors'],
    )
    def test_no_note_without_a_single_floor(self, tmp_path, requirement):
        """Without a requirement, or without one >= floor, there is nothing to expect: a branch
        clone gets no note, where a note would say that the tag could not be determined."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'MORS'
        commits = _tagged_module(checkout)
        git(checkout, 'reset', '-q', '--hard', commits[0])
        spec = Requirement(requirement) if requirement else None
        assert _release_tag_note(str(checkout), commits[0][:7], spec) == ''
        floor = _release_tag_note(
            str(checkout), commits[0][:7], Requirement('fwl-mors>=26.10.6')
        )
        assert floor == '; on branch main; PROTEUS expects the release tag 26.10.06'

    def test_tag_that_shares_its_name_with_a_branch_is_found(self, tmp_path):
        """A branch named 26.10.06 next to the tag 26.10.06 does not hide the tag: the short
        ref name of the tag would read tags/26.10.06, which names no release."""
        checkout = tmp_path / 'MORS'
        commits = _tagged_module(checkout)
        git(checkout, 'reset', '-q', '--hard', commits[0])
        git(checkout, 'branch', '26.10.06', commits[0])
        assert git(checkout, 'for-each-ref', '--format=%(refname:short)', 'refs/tags') == (
            '26.10.05\ntags/26.10.06'
        )
        r = _check_editable_mors(checkout)
        assert r.message.endswith('; on branch main; PROTEUS expects the release tag 26.10.06]')

    def test_floor_tag_name_reads_ascii_digits_only(self):
        """A tag of other Unicode digits is skipped, not passed to Version, which raises on
        it; the ASCII tag of the floor is still found."""
        from packaging.requirements import Requirement

        spec = Requirement('fwl-mors>=26.10.6')
        fullwidth = '\uff12\uff16.\uff11\uff10.\uff10\uff16'
        arabic = '\u0662\u0666.\u0661\u0660.\u0660\u0666'
        assert _floor_tag_name(spec, [fullwidth, arabic, '26.10.06']) == '26.10.06'
        assert _floor_tag_name(spec, [fullwidth]) is None

    def test_no_note_for_a_folder_that_is_not_its_own_clone(self, tmp_path, monkeypatch):
        """An empty .git inside another clone makes git answer for that clone; the annotation
        says that the folder is not a git clone of its own, with no note, where reading the
        outer clone would name its branch main and its tag."""
        outer = tmp_path / 'PROTEUS'
        _tagged_module(outer)
        inner = outer / 'MORS'
        (inner / '.git').mkdir(parents=True)
        monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
        assert _git_note_output(str(inner), 'symbolic-ref', '-q', '--short', 'HEAD') == (
            0,
            'main\n',
        )
        r = _check_editable_mors(inner)
        assert r.message == '26.10.6 [editable @ MORS; not a git clone of its own]'
        assert 'PROTEUS expects' not in r.message
        assert 'on branch' not in r.message

    @staticmethod
    def _recorded_check(checkout, requirement: str | None) -> tuple[CheckResult, list]:
        """Run the fwl-mors check of ``checkout`` and record the argv and timeout of each call."""
        from packaging.requirements import Requirement

        real_run = subprocess.run
        calls = []

        def recording_run(argv, **kwargs):
            calls.append((argv, kwargs.get('timeout')))
            return real_run(argv, **kwargs)

        with (
            patch('proteus.doctor.subprocess.run', side_effect=recording_run),
            patch('proteus.doctor.importlib.metadata.version', return_value='26.10.6'),
            patch('proteus.doctor._editable_checkout_path', return_value=str(checkout)),
            patch('proteus.doctor._imported_package_dir', return_value=None),
        ):
            spec = Requirement(requirement) if requirement else None
            return check_python_package('fwl-mors', spec), calls

    def test_git_calls_of_the_check_are_read_only_with_a_timeout(self, tmp_path):
        """Every git call of the check of an editable clone behind its tag, the note's
        included, has the timeout of the note and is on the read-only list of the test git
        guard, and the note costs at most 4 calls more than the check without a floor, which
        gives no note: the top level, the branch, the tags and the ancestry of HEAD."""
        checkout = tmp_path / 'MORS'
        commits = _tagged_module(checkout)
        git(checkout, 'reset', '-q', '--hard', commits[0])
        r, calls = self._recorded_check(checkout, 'fwl-mors>=26.10.6')
        _, plain = self._recorded_check(checkout, None)
        assert 'on branch main' in r.message
        assert all(argv[0] == 'git' for argv, _ in calls)
        assert [timeout for _, timeout in calls] == [_GIT_NOTE_TIMEOUT_S] * len(calls)
        assert [argv for argv, _ in calls if not git_call_is_read_only(argv[1:])] == []
        assert 0 < len(calls) - len(plain) <= 4
        assert 0 < _GIT_NOTE_TIMEOUT_S <= 30

    def test_check_keeps_the_index_of_a_clone_with_stale_stat_data(self, tmp_path):
        """A tracked file whose mtime changed after the commit leaves stale stat data in the
        index, which a plain git status rewrites; the check runs status with
        --no-optional-locks, and the bytes of .git/index stay the same."""
        checkout = tmp_path / 'MORS'
        _tagged_module(checkout)
        os.utime(checkout / 'version.txt', (1_000_000_000, 1_000_000_000))
        index = (checkout / '.git' / 'index').read_bytes()
        r, calls = self._recorded_check(checkout, 'fwl-mors>=26.10.6')
        assert (checkout / '.git' / 'index').read_bytes() == index
        assert ['git', '--no-optional-locks', '-C', str(checkout), 'status', '--porcelain'] in [
            argv for argv, _ in calls
        ]
        assert ' (dirty)' not in r.message

    def test_no_note_for_a_package_without_a_setup_script(self, tmp_path):
        """An editable fwl-io clone on a branch, with one >= floor and a tag of it, gets no
        note: no installer clones fwl-io at its floor. The same clone checked as fwl-mors,
        which a setup script clones, gets the note."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'fwl-io'
        commits = _tagged_module(checkout)
        git(checkout, 'reset', '-q', '--hard', commits[0])
        with (
            patch('proteus.doctor.importlib.metadata.version', return_value='26.10.6'),
            patch('proteus.doctor._editable_checkout_path', return_value=str(checkout)),
            patch('proteus.doctor._imported_package_dir', return_value=None),
        ):
            r = check_python_package('fwl-io', Requirement('fwl-io>=26.10.6'))
        assert _setup_command('fwl-io') is None
        assert r.status == PASS
        assert 'PROTEUS expects' not in r.message
        assert 'on branch' not in r.message
        assert (
            'PROTEUS expects the release tag 26.10.06' in _check_editable_mors(checkout).message
        )

    def test_branch_at_the_tag_commit_gets_no_note_as_in_the_installer(self, tmp_path):
        """The branch main moved to the commit of the floor tag gets no note, and the
        installer's state line for the same clone says that it is at the release tag; a note
        that names the branch before HEAD is compared with the tag says on branch main."""
        checkout = tmp_path / 'MORS'
        commits = _tagged_module(checkout)
        git(checkout, 'reset', '-q', '--hard', commits[1])
        r = _check_editable_mors(checkout)
        assert r.message == f'26.10.6 [editable @ MORS -> {commits[1][:7]}]'
        root = scratch_proteus_root(tmp_path / 'PROTEUS', ['fwl-mors>=26.10.6'])
        snippet = (
            f'source {shlex.quote(str(root / "tools" / "_get_common.sh"))}\n'
            f'report_clone_state fwl-mors {shlex.quote(str(checkout))} "get_python_module.sh MORS"\n'
        )
        done = subprocess.run(
            ['bash', '-c', 'set -euo pipefail\n' + snippet],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout == f'fwl-mors: {checkout} is at the release tag 26.10.06\n'
        assert git(checkout, 'symbolic-ref', '--short', 'HEAD') == 'main'

    def test_git_dir_in_the_environment_does_not_redirect_the_note(self, tmp_path, monkeypatch):
        """With GIT_DIR naming another repository on the branch elsewhere, the annotation of a
        MORS clone on main names its own commit and branch: git -C answers for GIT_DIR
        otherwise."""
        checkout = tmp_path / 'MORS'
        commits = _tagged_module(checkout)
        git(checkout, 'reset', '-q', '--hard', commits[0])
        other = tmp_path / 'other'
        other.mkdir()
        git(other, 'init', '-q', '-b', 'elsewhere')
        (other / 'other.txt').write_text('other\n')
        git(other, 'add', 'other.txt')
        git(other, 'commit', '-q', '-m', 'other')
        monkeypatch.setenv('GIT_DIR', str(other / '.git'))
        r = _check_editable_mors(checkout)
        assert r.message == (
            f'26.10.6 [editable @ MORS -> {commits[0][:7]};'
            ' on branch main; PROTEUS expects the release tag 26.10.06]'
        )
        assert 'elsewhere' not in r.message

    def test_git_that_does_not_finish_gives_no_note(self, tmp_path):
        """A git call that times out, or a missing git, gives no note and no exception."""
        err = subprocess.TimeoutExpired(cmd='git', timeout=_GIT_NOTE_TIMEOUT_S)
        with patch('proteus.doctor.subprocess.run', side_effect=err):
            assert _git_note_output(str(tmp_path), 'symbolic-ref', 'HEAD') is None
            assert _release_tag_note(str(tmp_path), 'abc1234', None) == ''
        with patch('proteus.doctor.subprocess.run', side_effect=FileNotFoundError):
            assert _git_note_output(str(tmp_path), 'symbolic-ref', 'HEAD') is None

    def test_clone_git_cannot_read_gets_no_note(self, tmp_path, monkeypatch):
        """A folder that is no git repository gives no note, also with a floor: rev-parse
        exits 128."""
        from packaging.requirements import Requirement

        monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
        assert _git_note_output(str(tmp_path), 'rev-parse', '--show-toplevel')[0] == 128
        spec = Requirement('fwl-mors>=26.10.6')
        assert _release_tag_note(str(tmp_path), 'abc1234', spec) == ''

    @pytest.mark.parametrize(
        ('requirement', 'tags', 'expected'),
        [
            ('fwl-mors>=26.10.6', ['26.10.05', '26.10.06', '26.11.01'], '26.10.06'),
            ('fwl-mors>=26.10.6', ['v26.10.6', 'release-26.10.6'], 'v26.10.6'),
            ('fwl-mors>=0.2', ['0.2.0', '0.1.0'], '0.2.0'),
            ('fwl-mors>=26.10.6', ['26.10.06', '26.10.6'], None),
            ('fwl-mors>=26.10.6', ['26.10.60', '26.1.6'], None),
            ('fwl-mors==26.10.6', ['26.10.6'], None),
            (
                'fwl-mors>=26.10.6',
                ['\uff12\uff16.\uff11\uff10.\uff10\uff16', '26.10.06'],
                '26.10.06',
            ),
        ],
        ids=[
            'zero padded tag',
            'v prefix, other names skipped',
            'trailing zero',
            'two tags of the floor',
            'near misses',
            'no floor',
            'tag of other Unicode digits skipped',
        ],
    )
    def test_floor_tag_name_matches_the_pin_reader(self, requirement, tags, expected):
        """The tag of the floor compares release numbers, as tools/_module_pins.py does: zero
        padding, a v prefix and trailing zeros do not matter, 26.10.60 is not 26.10.6, a tag
        of fullwidth digits is no release number, and two matches or no >= floor give no
        tag. The pin reader the installer runs picks the same tag, or refuses, for every
        case with a floor."""
        from packaging.requirements import Requirement

        spec = Requirement(requirement)
        assert _floor_tag_name(spec, tags) == expected
        loader = importlib.util.spec_from_file_location(
            'module_pins', PROTEUS_ROOT / 'tools' / '_module_pins.py'
        )
        pins = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(pins)
        floors = [s.version for s in spec.specifier if s.operator == '>=']
        try:
            picked = pins.match_floor_tag('fwl-mors', floors[0], tags) if floors else None
        except LookupError:
            picked = None
        assert picked == expected

    def test_editable_atmodeller_clone_gets_version_and_no_tag_note(self, tmp_path):
        """An editable atmodeller clone gets its version and no release tag note."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'Atmodeller'
        _init_test_repo(checkout)
        spec = Requirement('atmodeller>=1.0.2')
        with (
            patch('proteus.doctor.importlib.metadata.version', return_value='1.0.2'),
            patch('proteus.doctor._editable_checkout_path', return_value=str(checkout)),
            patch('proteus.doctor._imported_package_dir', return_value=None),
            patch(
                'proteus.doctor._get_script_for_package', return_value='tools/get_atmodeller.sh'
            ),
            patch(
                'proteus.doctor._module_pins', return_value={'atmodeller': {'ref': 'abc1234'}}
            ),
        ):
            r = check_python_package('atmodeller', spec)
        assert r.status == PASS
        assert 'atmodeller' in r.name
        assert '1.0.2' in r.message
        assert 'PROTEUS expects' not in r.message
        assert 'release tag' not in r.message

    def test_below_floor_advice_for_pinned_commit_module(self, tmp_path):
        """A module with a commit pin suggests checkout of the ref without tag lookup."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'Atmodeller'
        _init_test_repo(checkout)
        spec = Requirement('atmodeller>=1.0.2')
        pins = {'atmodeller': {'ref': '77e81aac5147bad7eec727fba5b299e055db2d4d'}}
        with (
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor._module_pins', return_value=pins),
        ):
            adv = _below_floor_advice(
                'atmodeller', str(checkout), spec, 'tools/get_atmodeller.sh'
            )
        path_q = shlex.quote(os.path.abspath(str(checkout)))
        ref_q = shlex.quote('77e81aac5147bad7eec727fba5b299e055db2d4d')
        expected = (
            f'git -C {path_q} fetch origin && git -C {path_q} checkout --detach '
            f'{ref_q} && pip install -e {path_q}'
        )
        assert adv.startswith(expected)
        assert 'tools/get_atmodeller.sh --force' in adv
        assert '--tags' not in adv
        assert 'refs/tags' not in adv

    @pytest.mark.parametrize(
        ('dist', 'floor', 'pins', 'script', 'named', 'absent'),
        [
            (
                'atmodeller',
                '1.0.2',
                {'atmodeller': {'ref': '77e81aac5147bad7eec727fba5b299e055db2d4d'}},
                'tools/get_atmodeller.sh',
                'its pinned commit',
                'release tag',
            ),
            (
                'fwl-morrigan',
                '26.10.0',
                {},
                'tools/get_morrigan.sh',
                'the release tag',
                'pinned',
            ),
        ],
        ids=['commit pin', 'release tag'],
    )
    def test_below_floor_advice_for_non_own_clone_outside_module_folder(
        self, tmp_path, dist, floor, pins, script, named, absent
    ):
        """Advice for a non-own clone outside the module folder names its pinned commit or tag."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'custom'
        checkout.mkdir()
        with (
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._is_own_clone', return_value=False),
        ):
            adv = _below_floor_advice(
                dist, str(checkout), Requirement(f'{dist}>={floor}'), script
            )
        assert named in adv
        assert absent not in adv

    def test_below_floor_advice_for_non_own_clone_in_pinned_module_folder(self, tmp_path):
        """A non-own clone in a pinned module folder advises manual removal without --force."""
        from packaging.requirements import Requirement

        checkout = tmp_path / 'Atmodeller'
        checkout.mkdir()
        (checkout / 'empty.txt').write_text('content\n')
        pins = {'atmodeller': {'ref': '77e81aac5147bad7eec727fba5b299e055db2d4d'}}
        with (
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._is_own_clone', return_value=False),
        ):
            adv = _below_floor_advice(
                'atmodeller',
                str(checkout),
                Requirement('atmodeller>=1.0.2'),
                'tools/get_atmodeller.sh',
            )
        path_q = shlex.quote(os.path.abspath(str(checkout)))
        root_q = shlex.quote(str(tmp_path))
        expected = f'move {path_q} away or delete it by hand, then run: cd {root_q} && bash tools/get_atmodeller.sh'
        assert adv == expected
        assert '--force' not in adv


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
        """A repo whose tracked content matches HEAD is clean, and an ignored
        file leaves it clean.

        The dirty state is read from ``git status --porcelain``, which honours
        .gitignore. That matters here because a PROTEUS checkout carries
        gitignored run output: if ignored paths counted, every environment
        report after a simulation would call the tree dirty.
        """
        _init_test_repo(tmp_path)
        assert _git_dirty(str(tmp_path)) is False

        # Edge: an ignored path holding files does not disturb the clean state.
        # The .gitignore is committed so it is not itself an untracked file.
        (tmp_path / '.gitignore').write_text('output/\n')
        _git_commit_all(tmp_path, 'add gitignore')
        (tmp_path / 'output').mkdir()
        (tmp_path / 'output' / 'run.log').write_text('junk')
        assert _git_dirty(str(tmp_path)) is False

    def test_git_dirty_true_for_untracked_file(self, tmp_path):
        """A file git has never seen makes the tree dirty.

        ``git status --porcelain`` lists untracked files alongside modified
        ones, so an untracked file is the case that separates the current
        behaviour from a tracked-only check such as ``git diff --quiet``, which
        would keep reporting clean and let a stray file pass unreported.
        """
        _init_test_repo(tmp_path)
        assert _git_dirty(str(tmp_path)) is False
        (tmp_path / 'untracked.txt').write_text('new')
        assert _git_dirty(str(tmp_path)) is True

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


class TestGetScriptForPackage:
    """Mapping from an FWL package to its tools/get_<name>.sh setup script.

    The mapping decides whether `proteus doctor`/`update` offers a `git pull`
    (tracking-branch checkout) or a re-run of the setup script (tag-pinned,
    detached-HEAD checkout). See issue #779.
    """

    def test_returns_script_for_tag_pinned_packages(self):
        """aragog and zalmoxis are installed by a setup script that pins to a
        version tag; the helper resolves each to its real script path."""
        # These scripts exist in the repo, so the helper returns the path.
        assert _get_script_for_package('fwl-aragog') == 'tools/get_aragog.sh'
        assert _get_script_for_package('fwl-zalmoxis') == 'tools/get_zalmoxis.sh'
        # The fwl- prefix is stripped: the script is get_aragog.sh, not
        # get_fwl-aragog.sh.
        assert 'fwl-' not in _get_script_for_package('fwl-aragog')

    def test_case_insensitive_prefix_strip(self):
        """The distribution name is lowercased before the script lookup, so a
        mixed-case name still resolves (dist names are normally lowercase, but
        the helper must not depend on that)."""
        assert _get_script_for_package('FWL-Zalmoxis') == 'tools/get_zalmoxis.sh'
        # A fully upper-case spelling of a second package folds too, so the
        # line above is not one spelling happening to work.
        assert _get_script_for_package('FWL-ARAGOG') == 'tools/get_aragog.sh'
        # Folding must not invent a script either: a mixed-case package with
        # none still takes the git-pull path.
        assert _get_script_for_package('FWL-Calliope') is None

    def test_name_without_fwl_prefix_is_used_verbatim(self):
        """The fwl- strip is conditional: a name lacking the prefix is looked up
        as-is, not mangled. 'aragog' resolves to the same script as
        'fwl-aragog', while a prefixless name with no script is still None."""
        # Prefix absent, so no strip happens; the bare stem is used directly.
        assert _get_script_for_package('aragog') == 'tools/get_aragog.sh'
        # Discrimination: the with- and without-prefix forms agree, confirming
        # the strip only removes a leading 'fwl-' and never truncates the stem.
        assert _get_script_for_package('aragog') == _get_script_for_package('fwl-aragog')
        # A prefixless name with no matching script returns None, not a path.
        assert _get_script_for_package('calliope') is None

    def test_returns_none_for_branch_tracking_package(self):
        """calliope has no setup script (it is cloned on a tracking branch), so
        the helper returns None and the caller falls back to git pull."""
        # No tools/get_calliope.sh exists, so None signals the git-pull path.
        assert _get_script_for_package('fwl-calliope') is None
        # A package name that maps to no script at all is also None, not a
        # path to a nonexistent file.
        assert _get_script_for_package('fwl-not-a-real-package') is None


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

    def test_run_all_checks_handles_standard_module_check_exception(self):
        """run_all_checks records a check error when check_python_package raises."""
        with (
            patch('proteus.doctor._dependency_specs', return_value={}),
            patch('proteus.doctor._module_pins', return_value={}),
            patch('proteus.doctor.check_python_package', side_effect=RuntimeError('boom')),
        ):
            results = run_all_checks()
        errors = [r for r in results if r.message == 'check error: boom']
        assert len(errors) >= len(STANDARD_MODULE_PACKAGES)
        assert any(r.name == 'atmodeller' for r in errors)


class TestCheckGitModulePinMismatch:
    """check_git_module's fix suggestion when a git submodule is off its pin."""

    @staticmethod
    def _pins() -> dict:
        """Pinned refs distinct from the head returned by the patched helper."""
        return {'socrates': {'ref': 'a' * 40}, 'agni': {'ref': 'b' * 40}}

    def test_socrates_mismatch_fix_also_rebuilds_agni(
        self, tmp_path, monkeypatch, mock_own_clone
    ):
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

    def test_agni_mismatch_fix_does_not_chain_socrates(self, tmp_path, mock_own_clone):
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

    def test_update_runs_the_chained_socrates_fix(self, tmp_path, monkeypatch, mock_own_clone):
        """`proteus update` actually executes the chained SOCRATES fix.

        Constructing the right string is useless unless `proteus update` runs
        it. This drives check_git_module's real off-pin result through
        update_entry with the runner mocked, and asserts the chained command,
        AGNI rebuild and all, reaches the fix runner exactly once.
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
        # the SOCRATES half, so assert the chain survived in full.
        assert '&&' in ran


class TestCheckGitModuleInstallState:
    """check_git_module's not-installed (FAIL) and on-pin (PASS) results."""

    @staticmethod
    def _pins() -> dict:
        """Pinned refs used to drive the on-pin branch via a matching head."""
        return {'socrates': {'ref': 'a' * 40}, 'agni': {'ref': 'b' * 40}}

    def test_socrates_not_installed_fix_is_unchained(self, monkeypatch):
        """A missing SOCRATES suggests a bare rebuild, with no AGNI step chained.

        With RAD_DIR unset the SOCRATES tree is absent, so RAD_DIR cannot point a
        chained AGNI rebuild at anything; the fix is deliberately the lone
        SOCRATES install. This is the inverse of the off-pin case, where the tree
        is present and the AGNI rebuild is chained on.
        """
        monkeypatch.delenv('RAD_DIR', raising=False)
        # The not-installed branch returns before any pin is read; patch
        # _module_pins only to keep the test off the real pyproject.
        with patch('proteus.doctor._module_pins', return_value={}):
            r = check_git_module('SOCRATES', {})
        assert r.status == FAIL
        assert r.message == 'not installed'
        assert r.fix_cmd == 'bash tools/get_socrates.sh'
        # Discrimination: the off-pin SOCRATES fix chains AGNI with `&&`; the
        # not-installed fix must not, because RAD_DIR has nowhere to point.
        assert '&&' not in r.fix_cmd
        assert 'get_agni.sh' not in r.fix_cmd

    def test_agni_not_installed_reports_its_own_setup(self):
        """A missing AGNI checkout suggests the AGNI install and nothing else.

        AGNI's path comes from the dirs mapping, not RAD_DIR, so an empty mapping
        is the not-installed edge case. The fix names AGNI's own script and does
        not drag in a SOCRATES rebuild.
        """
        # The not-installed branch returns before any pin is read; patch
        # _module_pins only to keep the test off the real pyproject.
        with patch('proteus.doctor._module_pins', return_value={}):
            r = check_git_module('AGNI', {})
        assert r.status == FAIL
        assert r.message == 'not installed'
        assert r.fix_cmd == 'bash tools/get_agni.sh'
        # Discrimination: never chain a SOCRATES rebuild onto an AGNI fix.
        assert 'get_socrates.sh' not in r.fix_cmd
        assert '&&' not in r.fix_cmd

    def test_socrates_on_pin_passes_with_no_fix(self, tmp_path, monkeypatch, mock_own_clone):
        """A SOCRATES checkout exactly on its pin passes and suggests no fix.

        When HEAD matches the pinned ref the tree is already correct, so the
        result is PASS carrying the resolved version and short hash, and crucially
        no rebuild command: surfacing a fix here would send the user to re-clone a
        tree that needs nothing.
        """
        monkeypatch.setenv('RAD_DIR', str(tmp_path))
        on_pin = 'a' * 40  # matches the SOCRATES ref returned by _pins()
        with (
            patch('proteus.doctor._module_pins', return_value=self._pins()),
            patch('proteus.doctor._git_head', return_value=on_pin),
            patch('proteus.doctor._get_socrates_version', return_value='soc-test'),
        ):
            r = check_git_module('SOCRATES', {})
        assert r.status == PASS
        assert r.fix_cmd is None
        # The resolved version and matched short hash are reported so the user
        # can confirm the pin, rather than a bare status.
        assert 'soc-test' in r.message
        assert on_pin[:8] in r.message
        # Discrimination: an off-pin head would read 'differs from pin' and warn;
        # the on-pin path must not.
        assert 'differs' not in r.message

    def test_obliqua_on_pin_reports_resolved_version(self, tmp_path, mock_own_clone):
        """An on-pin Obliqua checkout reports its real version, not a placeholder.

        check_git_module dispatches to a per-module version getter; Obliqua's
        getter (_get_obliqua_version) must actually be wired into that dispatch,
        not silently fall through to the generic '?' placeholder used for
        modules with no getter at all.
        """
        pins = {'obliqua': {'ref': 'a' * 40}}
        (tmp_path / 'Project.toml').write_text('name = "Obliqua"\n')
        (tmp_path / 'Manifest.toml').touch()
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._git_head', return_value='a' * 40),
            patch('proteus.doctor._get_obliqua_version', return_value='0.1.0'),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(tmp_path)}, required=False)
        assert r.status == PASS
        assert '0.1.0' in r.message
        # Discrimination: the placeholder used when a module has no version
        # getter wired up must not appear once the getter is actually called.
        assert r.message != '?'

    def test_unrecognized_module_name_falls_back_to_placeholder_version(
        self, tmp_path, mock_own_clone
    ):
        """A module name with no dedicated version getter degrades to '?'.

        The name dispatch (AGNI/SOCRATES/Obliqua) is exhaustive over today's
        git-pinned modules, but the trailing ``else`` exists so a future
        module added to GIT_MODULES/STANDARD_GIT_MODULES without a matching
        version-getter branch fails soft with a placeholder instead of
        raising or reporting a fabricated version.
        """
        pins = {'futuremodule': {'ref': 'a' * 40}}
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._git_head', return_value='a' * 40),
        ):
            r = check_git_module('FutureModule', {'futuremodule': str(tmp_path)})
        assert r.status == PASS
        assert r.message == '? (aaaaaaaa)'
        # Discrimination: Obliqua's dedicated getter must still resolve a
        # real-looking version rather than also degrading to the fallback,
        # proving the placeholder is specific to unrecognized names.
        with (
            patch('proteus.doctor._module_pins', return_value={'obliqua': {'ref': 'a' * 40}}),
            patch('proteus.doctor._git_head', return_value='a' * 40),
            patch('proteus.doctor._get_obliqua_version', return_value='0.1.0'),
        ):
            obliqua_r = check_git_module('Obliqua', {'obliqua': str(tmp_path)}, required=False)
        assert obliqua_r.message != '?'

    def test_socrates_version_unreadable_still_classifies(
        self, tmp_path, monkeypatch, mock_own_clone
    ):
        """An unreadable SOCRATES version degrades to '?' instead of crashing.

        Reading the version can fail on a half-built tree, so the lookup is
        guarded: the version label falls back to '?' and the check still compares
        HEAD to the pin. The user gets an actionable off-pin warning carrying the
        '?' placeholder, not an exception that aborts the whole diagnose run.
        """
        monkeypatch.setenv('RAD_DIR', str(tmp_path))
        with (
            patch('proteus.doctor._module_pins', return_value=self._pins()),
            patch('proteus.doctor._git_head', return_value='c' * 40),
            patch(
                'proteus.doctor._get_socrates_version',
                side_effect=OSError('version file missing'),
            ),
        ):
            r = check_git_module('SOCRATES', {})
        # Head 'c'*40 differs from the pin 'a'*40, so the guarded version still
        # lands on the off-pin warning rather than swallowing the mismatch.
        assert r.status == WARN
        assert '?' in r.message
        # Discrimination: the failure degraded the version slot to '?' but kept
        # the actionable fix, instead of crashing or dropping the suggestion.
        assert r.fix_cmd is not None
        assert 'get_socrates.sh' in r.fix_cmd


class TestCheckGitModuleStandard:
    """check_git_module(required=False): Obliqua's not-required contract."""

    def test_missing_standard_module_warns_when_not_required(self):
        """A missing checkout with required=False returns a WARN CheckResult."""
        with patch('proteus.doctor._module_pins', return_value={}):
            res = check_git_module('Obliqua', {}, required=False)
            required_result = check_git_module('Obliqua', {}, required=True)
        assert res is not None
        assert res.status == WARN
        assert res.message == 'not installed'
        assert 'get_obliqua.sh' in res.fix_cmd
        assert required_result is not None
        assert required_result.status == FAIL

    def test_missing_required_module_still_fails(self):
        """Default (``required=True``) behaviour is unchanged: a missing
        checkout is still a FAIL, not silently skipped -- the new
        parameter must not weaken AGNI/SOCRATES's existing contract.
        """
        with patch('proteus.doctor._module_pins', return_value={}):
            r = check_git_module('AGNI', {}, required=True)
        assert r is not None
        assert r.status == FAIL
        assert r.message == 'not installed'

    def test_present_standard_module_on_pin_still_reports_pass(self, tmp_path, mock_own_clone):
        """Once actually installed, a standard module IS checked and
        pin-compared normally (PASS on a matching HEAD) -- ``required``
        only controls the missing-checkout case, not whether an existing
        checkout gets verified.
        """
        pins = {'obliqua': {'ref': 'a' * 40}}
        (tmp_path / 'Project.toml').write_text('name = "Obliqua"\n')
        (tmp_path / 'Manifest.toml').touch()
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._git_head', return_value='a' * 40),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(tmp_path)}, required=False)
        assert r is not None
        assert r.status == PASS
        assert r.fix_cmd is None

    def test_present_standard_module_off_pin_still_warns(self, tmp_path, mock_own_clone):
        """An installed-but-off-pin standard module gets advice and auto_fixable=False."""
        pins = {'obliqua': {'ref': 'a' * 40}}
        root = tmp_path / 'repo'
        root.mkdir()
        obliqua_dir = tmp_path / 'Obliqua'
        obliqua_dir.mkdir()
        (obliqua_dir / 'Project.toml').write_text('name = "Obliqua"\n')
        with (
            patch('proteus.doctor._repo_root', return_value=root),
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._git_head', return_value='b' * 40),
            patch('proteus.doctor._get_obliqua_version', return_value='0.1.0'),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(obliqua_dir)}, required=False)
        assert r is not None
        assert r.status == WARN
        assert r.auto_fixable is False
        expected_fix = (
            f'cd {shlex.quote(str(root))} && bash tools/get_obliqua.sh --force '
            f'{shlex.quote(os.path.abspath(str(obliqua_dir)))}  # replaces the folder with a '
            'new clone at the pin, which deletes everything in it'
        )
        assert r.fix_cmd == expected_fix
        with (
            patch('proteus.doctor._module_pins', return_value={'agni': {'ref': 'a' * 40}}),
            patch('proteus.doctor._git_head', return_value='b' * 40),
        ):
            agni_r = check_git_module('AGNI', {'agni': str(tmp_path)}, required=True)
        assert agni_r.auto_fixable is True

    @pytest.mark.parametrize(
        ('pins', 'head', 'message'),
        [
            (
                {'obliqua': {'ref': 'a' * 40}},
                'a' * 40,
                '0.1.0 (aaaaaaaa): Julia environment not instantiated',
            ),
            (
                {},
                None,
                '0.1.0: Julia environment not instantiated',
            ),
        ],
        ids=['with head', 'without head'],
    )
    def test_obliqua_missing_manifest_warns_julia_not_instantiated(
        self, tmp_path, mock_own_clone, pins, head, message
    ):
        """An on-pin Obliqua folder without Manifest.toml warns, naming the head when git has one."""
        (tmp_path / 'Project.toml').write_text('name = "Obliqua"\n')
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._git_head', return_value=head),
            patch('proteus.doctor._get_obliqua_version', return_value='0.1.0'),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(tmp_path)}, required=False)
        assert r.status == WARN
        assert r.auto_fixable is True
        assert r.message == message
        assert r.fix_cmd == 'bash tools/get_obliqua.sh'

    def test_obliqua_off_pin_takes_precedence_over_missing_manifest(
        self, tmp_path, mock_own_clone
    ):
        """Off-pin warning takes precedence over missing Manifest.toml."""
        pins = {'obliqua': {'ref': 'a' * 40}}
        (tmp_path / 'Project.toml').write_text('name = "Obliqua"\n')
        with (
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._git_head', return_value='b' * 40),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(tmp_path)}, required=False)
        assert r.status == WARN
        assert r.auto_fixable is False
        assert 'differs from pin' in r.message
        assert '--force' in r.fix_cmd

    def test_empty_standard_module_folder_in_git_repo_counts_as_not_installed(self, tmp_path):
        """An empty standard module folder in a git repo counts as not installed."""
        _init_test_repo(tmp_path)
        obliqua_dir = tmp_path / 'Obliqua'
        obliqua_dir.mkdir()
        pins = {'obliqua': {'ref': 'a' * 40}}
        with patch('proteus.doctor._module_pins', return_value=pins):
            r = check_git_module('Obliqua', {'obliqua': str(obliqua_dir)}, required=False)
        assert r.status == WARN
        assert r.message == 'not installed'
        assert r.fix_cmd == 'bash tools/get_obliqua.sh'
        assert r.auto_fixable is True
        assert 'differs from pin' not in r.message

    def test_obliqua_in_outer_repo_without_own_clone_reports_mark(self, tmp_path):
        """A non-clone Obliqua directory in a git repo does not read the outer repo commit."""
        _init_test_repo(tmp_path)
        obliqua_dir = tmp_path / 'Obliqua Copy'
        obliqua_dir.mkdir()
        (obliqua_dir / 'Project.toml').write_text('version = "0.1.0"\n')
        (obliqua_dir / 'Manifest.toml').touch()
        root = tmp_path / 'repo'
        outer_head = _git_head(str(tmp_path))
        pins = {'obliqua': {'ref': '0' * 40}}
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._repo_root', return_value=root),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(obliqua_dir)}, required=False)
        assert r.status == WARN
        assert (
            r.message
            == '0.1.0 [not a git clone of its own]: the commit cannot be compared with the pin'
        )
        assert outer_head[:8] not in r.message
        assert 'differs from pin' not in r.message
        assert r.fix_cmd == (
            f'move {shlex.quote(str(obliqua_dir))} away or delete it, then run: '
            f'cd {shlex.quote(str(root))} && bash tools/get_obliqua.sh'
        )
        assert r.auto_fixable is False

    def test_obliqua_in_outer_repo_missing_manifest_warns_not_own_clone(self, tmp_path):
        """A non-clone Obliqua directory missing Manifest.toml warns that it is not its own clone."""
        _init_test_repo(tmp_path)
        obliqua_dir = tmp_path / 'Obliqua'
        obliqua_dir.mkdir()
        (obliqua_dir / 'Project.toml').write_text('name = "Obliqua"\nversion = "0.1.0"\n')
        root = tmp_path / 'repo'
        root.mkdir()
        pins = {'obliqua': {'ref': '0' * 40}}
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._repo_root', return_value=root),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(obliqua_dir)}, required=False)
        assert r.status == WARN
        assert (
            r.message
            == '0.1.0 [not a git clone of its own]: the commit cannot be compared with the pin'
        )
        assert r.fix_cmd == (
            f'move {shlex.quote(str(obliqua_dir))} away or delete it, then run: '
            f'cd {shlex.quote(str(root))} && bash tools/get_obliqua.sh'
        )
        assert r.auto_fixable is False

    def test_obliqua_foreign_project_clone_warns_and_advises_move_away(self, tmp_path):
        """A clone of another project at Obliqua path warns and gives move-away advice."""
        root = tmp_path / 'repo'
        root.mkdir()
        obliqua_dir = tmp_path / 'Obliqua'
        _init_test_repo(obliqua_dir)
        (obliqua_dir / 'Project.toml').write_text('name = "AGNI"\nversion = "1.13.0"\n')
        pins = {'obliqua': {'ref': '0' * 40}}
        with (
            patch('proteus.doctor._repo_root', return_value=root),
            patch('proteus.doctor._module_pins', return_value=pins),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(obliqua_dir)}, required=False)
        assert r.status == WARN
        assert (
            r.message
            == '1.13.0: a git clone of another project (its Project.toml does not name Obliqua)'
        )
        assert r.fix_cmd == (
            f'move {shlex.quote(str(obliqua_dir))} away or delete it, then run: '
            f'cd {shlex.quote(str(root))} && bash tools/get_obliqua.sh'
        )
        assert '--force' not in r.fix_cmd
        assert r.auto_fixable is False

    def test_obliqua_own_clone_off_pin_gives_forced_advice(self, tmp_path):
        """An Obliqua directory that is its own clone warns and advises --force when off pin."""
        _init_test_repo(tmp_path)
        obliqua_dir = tmp_path / 'Obliqua'
        _init_test_repo(obliqua_dir)
        (obliqua_dir / 'Project.toml').write_text('name = "Obliqua"\nversion = "0.1.0"\n')
        (obliqua_dir / 'Manifest.toml').touch()
        obliqua_head = _git_head(str(obliqua_dir))
        pins = {'obliqua': {'ref': '0' * 40}}
        with patch('proteus.doctor._module_pins', return_value=pins):
            r = check_git_module('Obliqua', {'obliqua': str(obliqua_dir)}, required=False)
        assert r.status == WARN
        assert 'differs from pin' in r.message
        assert obliqua_head[:8] in r.message
        assert r.fix_cmd is not None and '--force' in r.fix_cmd
        assert r.auto_fixable is False

    def test_names_obliqua_direct(self, tmp_path):
        """_names_obliqua tests Project.toml for name = "Obliqua" at start of line."""
        folder = tmp_path / 'obliqua_project'
        folder.mkdir()
        assert _names_obliqua(str(folder)) is False

        (folder / 'Project.toml').write_text('version = "0.1.0"\nname="Obliqua"\n')
        assert _names_obliqua(str(folder)) is False

        (folder / 'Project.toml').write_text('# comment\nversion = "0.1.0"\nname = "Obliqua"\n')
        assert _names_obliqua(str(folder)) is True

    def test_broken_link_or_regular_file_at_module_path(self, tmp_path):
        """A broken link or regular file at module path reports not installed with manual delete fix."""
        root = tmp_path / 'repo'
        root.mkdir()
        broken_link = tmp_path / 'broken_obliqua'
        broken_link.symlink_to(tmp_path / 'nonexistent')
        assert os.path.lexists(str(broken_link)) and not os.path.isdir(str(broken_link))

        with patch('proteus.doctor._repo_root', return_value=root):
            r_link = check_git_module('Obliqua', {'obliqua': str(broken_link)}, required=False)
        assert r_link.status == WARN
        assert r_link.message == 'not installed: a file or a broken link is at its path'
        assert r_link.fix_cmd == (
            f'delete {shlex.quote(str(broken_link))}, then run: '
            f'cd {shlex.quote(str(root))} && bash tools/get_obliqua.sh'
        )
        assert r_link.auto_fixable is False

        regular_file = tmp_path / 'regular_agni'
        regular_file.write_text('not a directory\n')
        with patch('proteus.doctor._repo_root', return_value=root):
            r_file = check_git_module('AGNI', {'agni': str(regular_file)}, required=True)
        assert r_file.status == FAIL
        assert r_file.message == 'not installed: a file or a broken link is at its path'
        assert r_file.fix_cmd == (
            f'delete {shlex.quote(str(regular_file))}, then run: '
            f'cd {shlex.quote(str(root))} && bash tools/get_agni.sh'
        )
        assert r_file.auto_fixable is False

    def test_symlink_to_empty_directory_at_module_path(self, tmp_path):
        """A link to an empty directory reports not installed with automatic fix."""
        empty_dir = tmp_path / 'empty_dir'
        empty_dir.mkdir()
        link_dir = tmp_path / 'link_to_empty'
        link_dir.symlink_to(empty_dir)
        assert os.path.islink(str(link_dir)) and os.path.isdir(str(link_dir))

        r = check_git_module('Obliqua', {'obliqua': str(link_dir)}, required=False)
        assert r.status == WARN
        assert r.message == 'not installed'
        assert r.fix_cmd == 'bash tools/get_obliqua.sh'
        assert r.auto_fixable is True

    def test_socrates_not_installed_rad_dir_nonexistent_names_rad_dir(
        self, tmp_path, monkeypatch
    ):
        """RAD_DIR pointing to nonexistent folder reports not installed naming RAD_DIR."""
        monkeypatch.setenv('RAD_DIR', str(tmp_path / 'nowhere'))
        r = check_git_module('SOCRATES', {}, required=True)
        assert r.status == FAIL
        assert r.message == 'not installed'
        assert r.fix_cmd == 'bash tools/get_socrates.sh "$RAD_DIR"'

        monkeypatch.delenv('RAD_DIR', raising=False)
        r_unset = check_git_module('SOCRATES', {}, required=True)
        assert r_unset.status == FAIL
        assert r_unset.message == 'not installed'
        assert r_unset.fix_cmd == 'bash tools/get_socrates.sh'

    def test_empty_required_module_folder_in_git_repo_counts_as_not_installed(self, tmp_path):
        """An empty required module folder in a git repo counts as not installed."""
        _init_test_repo(tmp_path)
        agni_dir = tmp_path / 'AGNI'
        agni_dir.mkdir()
        pins = {'agni': {'ref': 'a' * 40}}
        with patch('proteus.doctor._module_pins', return_value=pins):
            r = check_git_module('AGNI', {'agni': str(agni_dir)}, required=True)
        assert r.status == FAIL
        assert r.message == 'not installed'
        assert r.fix_cmd == 'bash tools/get_agni.sh'

    def test_agni_in_outer_repo_without_own_clone_warns_cannot_compare_pin(self, tmp_path):
        """A populated non-clone AGNI folder in a git repo warns that commit cannot be compared."""
        _init_test_repo(tmp_path)
        agni_dir = tmp_path / 'AGNI Copy'
        agni_dir.mkdir()
        (agni_dir / 'Project.toml').write_text('version = "1.13.0"\n')
        root = tmp_path / 'repo'
        outer_head = _git_head(str(tmp_path))
        pins = {'agni': {'ref': 'a' * 40}}
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._get_agni_version', return_value='1.13.0'),
            patch('proteus.doctor._repo_root', return_value=root),
        ):
            r = check_git_module('AGNI', {'agni': str(agni_dir)}, required=True)
        assert r.status == WARN
        assert (
            r.message
            == '1.13.0 [not a git clone of its own]: the commit cannot be compared with the pin'
        )
        assert outer_head[:8] not in r.message
        assert r.fix_cmd == (
            f'move {shlex.quote(str(agni_dir))} away or delete it, then run: '
            f'cd {shlex.quote(str(root))} && bash tools/get_agni.sh'
        )
        assert r.auto_fixable is False

    def test_socrates_without_own_clone_warns_and_names_rad_dir(self, tmp_path, monkeypatch):
        """A populated non-clone SOCRATES folder warns and includes RAD_DIR in the fix command."""
        socrates_dir = tmp_path / 'socrates copy'
        socrates_dir.mkdir()
        (socrates_dir / 'version').write_text('24.07\n')
        root = tmp_path / 'repo'
        monkeypatch.setenv('RAD_DIR', str(socrates_dir))
        with (
            patch('proteus.doctor._get_socrates_version', return_value='24.07'),
            patch('proteus.doctor._repo_root', return_value=root),
        ):
            r = check_git_module('SOCRATES', {}, required=True)
        assert r.status == WARN
        assert r.message.endswith(': the commit cannot be compared with the pin')
        assert r.fix_cmd == (
            f'move {shlex.quote(str(socrates_dir))} away or delete it, then run: '
            f'cd {shlex.quote(str(root))} && '
            'bash tools/get_socrates.sh "$RAD_DIR" && RAD_DIR="$RAD_DIR" bash tools/get_agni.sh 0'
        )
        assert r.auto_fixable is False

    def test_empty_socrates_rad_dir_not_installed_names_rad_dir(self, tmp_path, monkeypatch):
        """An empty RAD_DIR folder reports SOCRATES not installed with RAD_DIR in the fix command."""
        socrates_dir = tmp_path / 'socrates'
        socrates_dir.mkdir()
        monkeypatch.setenv('RAD_DIR', str(socrates_dir))
        r = check_git_module('SOCRATES', {}, required=True)
        assert r.status == FAIL
        assert r.message == 'not installed'
        assert r.fix_cmd == 'bash tools/get_socrates.sh "$RAD_DIR"'

    def test_agni_clone_without_commit_warns_cannot_compare_pin(self, tmp_path):
        """A git repository without commits at a pinned module path warns."""
        root = tmp_path / 'repo'
        root.mkdir()
        agni_dir = tmp_path / 'AGNI'
        agni_dir.mkdir()
        subprocess.run(['git', 'init'], cwd=agni_dir, check=True, capture_output=True)
        (agni_dir / 'Project.toml').write_text('[project]\n')
        pins = {'agni': {'ref': 'a' * 40}}
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._get_agni_version', return_value='1.13.0'),
            patch('proteus.doctor._repo_root', return_value=root),
        ):
            r = check_git_module('AGNI', {'agni': str(agni_dir)}, required=True)
        assert r.status == WARN
        assert r.message == '1.13.0: the commit cannot be compared with the pin'
        assert r.fix_cmd == (
            f'move {shlex.quote(str(agni_dir))} away or delete it, then run: '
            f'cd {shlex.quote(str(root))} && bash tools/get_agni.sh'
        )
        assert r.auto_fixable is False

    def test_obliqua_clone_without_commit_warns_cannot_compare_pin(self, tmp_path):
        """An Obliqua clone without commits warns even with Manifest present."""
        root = tmp_path / 'repo'
        root.mkdir()
        obliqua_dir = tmp_path / 'Obliqua'
        obliqua_dir.mkdir()
        subprocess.run(['git', 'init'], cwd=obliqua_dir, check=True, capture_output=True)
        (obliqua_dir / 'Project.toml').write_text('name = "Obliqua"\nversion = "0.1.0"\n')
        (obliqua_dir / 'Manifest.toml').write_text('manifest_format = "2.0"\n')
        pins = {'obliqua': {'ref': 'b' * 40}}
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._repo_root', return_value=root),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(obliqua_dir)}, required=False)
        assert r.status == WARN
        assert r.message == '0.1.0: the commit cannot be compared with the pin'
        assert r.fix_cmd == (
            f'move {shlex.quote(str(obliqua_dir))} away or delete it, then run: '
            f'cd {shlex.quote(str(root))} && bash tools/get_obliqua.sh'
        )
        assert '--force' not in r.fix_cmd
        assert r.auto_fixable is False

    def test_obliqua_clone_without_commit_without_manifest_warns_cannot_compare_pin(
        self, tmp_path
    ):
        """An Obliqua clone without commits warns before the Manifest check."""
        root = tmp_path / 'repo'
        root.mkdir()
        obliqua_dir = tmp_path / 'Obliqua'
        obliqua_dir.mkdir()
        subprocess.run(['git', 'init'], cwd=obliqua_dir, check=True, capture_output=True)
        (obliqua_dir / 'Project.toml').write_text('name = "Obliqua"\nversion = "0.1.0"\n')
        pins = {'obliqua': {'ref': 'b' * 40}}
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._repo_root', return_value=root),
        ):
            r = check_git_module('Obliqua', {'obliqua': str(obliqua_dir)}, required=False)
        assert r.status == WARN
        assert r.message == '0.1.0: the commit cannot be compared with the pin'
        assert r.fix_cmd == (
            f'move {shlex.quote(str(obliqua_dir))} away or delete it, then run: '
            f'cd {shlex.quote(str(root))} && bash tools/get_obliqua.sh'
        )
        assert 'Julia environment not instantiated' not in r.message
        assert r.auto_fixable is False

    def test_unpinned_module_clone_without_commit_passes(self, tmp_path):
        """A module clone without commit passes when the module has no pin."""
        root = tmp_path / 'repo'
        root.mkdir()
        agni_dir = tmp_path / 'AGNI'
        agni_dir.mkdir()
        subprocess.run(['git', 'init'], cwd=agni_dir, check=True, capture_output=True)
        (agni_dir / 'Project.toml').write_text('[project]\n')
        with (
            patch('proteus.doctor._module_pins', return_value={}),
            patch('proteus.doctor._get_agni_version', return_value='1.13.0'),
            patch('proteus.doctor._repo_root', return_value=root),
        ):
            r = check_git_module('AGNI', {'agni': str(agni_dir)}, required=True)
        assert r.status == PASS
        assert r.message == '1.13.0'

    def test_run_all_checks_warns_obliqua_when_not_installed(self, tmp_path):
        """With no Obliqua checkout, run_all_checks returns WARN."""
        with (
            patch('proteus.doctor._dependency_specs', return_value={}),
            patch('proteus.doctor._module_pins', return_value={}),
            patch(
                'proteus.doctor.get_proteus_directories',
                return_value={'proteus': str(tmp_path)},
            ),
        ):
            results = run_all_checks()
        obliqua_results = [r for r in results if r.name == 'Obliqua']
        assert len(obliqua_results) == 1
        assert obliqua_results[0].status == WARN
        assert obliqua_results[0].message == 'not installed'
        assert 'get_obliqua.sh' in obliqua_results[0].fix_cmd
        # Discrimination: AGNI (mandatory, also missing here) must still be
        # reported as a real FAIL -- proving the standard git module checks
        # did not also silently swallow the mandatory git-module checks.
        agni_results = [r for r in results if r.name == 'AGNI']
        assert len(agni_results) == 1
        assert agni_results[0].status == FAIL

    def test_run_all_checks_includes_obliqua_when_installed(self, tmp_path, mock_own_clone):
        """With a real Obliqua checkout present, run_all_checks
        must include its CheckResult in the top-level report -- the
        counterpart to the omitted-when-absent case above, confirming
        ``required=False`` only changes the missing-checkout behaviour,
        not whether an installed standard module is reported at all.
        """
        pins = {'obliqua': {'ref': 'a' * 40}}
        (tmp_path / 'Project.toml').write_text('name = "Obliqua"\n')
        (tmp_path / 'Manifest.toml').touch()
        with (
            patch('proteus.doctor._dependency_specs', return_value={}),
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._git_head', return_value='a' * 40),
            patch(
                'proteus.doctor.get_proteus_directories',
                return_value={'proteus': str(tmp_path), 'obliqua': str(tmp_path)},
            ),
        ):
            results = run_all_checks()
        obliqua_results = [r for r in results if r.name == 'Obliqua']
        assert len(obliqua_results) == 1
        assert obliqua_results[0].status == PASS

    def test_run_all_checks_reports_check_error_for_standard_module_that_raises(self):
        """A crash inside the standard-module loop must degrade to a FAIL
        CheckResult (the same recovery pattern the mandatory-module loop
        already has), not propagate and abort every other check.
        """
        with (
            patch('proteus.doctor._dependency_specs', return_value={}),
            patch('proteus.doctor._module_pins', return_value={}),
            patch(
                'proteus.doctor.check_git_module',
                side_effect=RuntimeError('boom_standard_xyz'),
            ),
        ):
            results = run_all_checks()
        obliqua_results = [r for r in results if r.name == 'Obliqua']
        assert len(obliqua_results) == 1
        assert obliqua_results[0].status == FAIL
        assert 'boom_standard_xyz' in obliqua_results[0].message


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
        """An unset env var carries human advice ('conda env config vars set VAR=<path>'), not a
        runnable command. update lists it under manual action and never passes
        it to the shell, while still running the genuinely fixable check."""
        (tmp_path / 'tools').mkdir()
        results = [
            CheckResult(
                'FC_DIR',
                'environment',
                FAIL,
                'not set',
                'conda env config vars set FC_DIR=<path>, then conda deactivate && '
                'conda activate proteus (without conda: source .proteus-env.sh in the PROTEUS folder)',
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
        assert 'FC_DIR' not in ran
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
            CheckResult('FWL_DATA/atmos_clim/spectral_files', 'data', WARN, 'empty', None),
        ]
        with patch('proteus.doctor.run_all_checks', return_value=results):
            result = update_entry(dry_run=False)
        out = capsys.readouterr().out
        assert 'none of them have an automatic fix' in out
        # Discrimination: a warning-only run is still healthy (True), unlike a FAIL.
        assert result is True

    def test_update_entry_runs_no_command_for_non_clone_git_module_warning(
        self, capsys, tmp_path
    ):
        """update_entry runs no fix command for a non-clone git module warning."""
        _init_test_repo(tmp_path)
        agni_dir = tmp_path / 'AGNI'
        agni_dir.mkdir()
        (agni_dir / 'Project.toml').write_text('version = "1.13.0"\n')
        pins = {'agni': {'ref': 'a' * 40}}
        with (
            patch('proteus.doctor._module_pins', return_value=pins),
            patch('proteus.doctor._get_agni_version', return_value='1.13.0'),
        ):
            r = check_git_module('AGNI', {'agni': str(agni_dir)}, required=True)
        runner = Mock(return_value=0)
        with (
            patch('proteus.doctor.run_all_checks', return_value=[r]),
            patch('proteus.doctor._run_fix_command', runner),
            patch('proteus.doctor.doctor_entry', return_value=True),
        ):
            result = update_entry(dry_run=False)
        out = capsys.readouterr().out
        assert result is True
        runner.assert_not_called()
        assert 'none of them have an automatic fix' in out
        assert 'tools/get_agni.sh' in out

    def test_update_entry_with_absent_vulcan_dry_run_and_real(self, capsys, tmp_path):
        """With fwl-vulcan absent, dry run prints setup command and real run executes it."""
        from importlib.metadata import PackageNotFoundError

        (tmp_path / 'tools').mkdir()
        (tmp_path / 'tools' / 'get_vulcan.sh').touch()

        with (
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch(
                'proteus.doctor.importlib.metadata.version',
                side_effect=PackageNotFoundError('fwl-vulcan'),
            ),
            patch('proteus.doctor._editable_checkout_path', return_value=None),
        ):
            r = check_python_package('fwl-vulcan', None)

        assert r.status == WARN
        assert r.auto_fixable is True
        assert 'bash tools/get_vulcan.sh' in r.fix_cmd

        recorder = Mock(return_value=0)
        with (
            patch('proteus.doctor._repo_root', return_value=tmp_path),
            patch('proteus.doctor.run_all_checks', return_value=[r]),
            patch('proteus.doctor._run_fix_command', recorder),
            patch('proteus.doctor.doctor_entry', return_value=True),
        ):
            update_entry(dry_run=True)
            out = capsys.readouterr().out
            assert 'Dry run' in out
            assert 'bash tools/get_vulcan.sh' in out
            recorder.assert_not_called()

            update_entry(dry_run=False)
            recorder.assert_called_once_with(r.fix_cmd, tmp_path)


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

    def test_git_stdout_gives_none_for_a_git_the_system_refuses_to_run(self):
        """A git that the system refuses to run gives None, not an exception.

        PermissionError is an OSError other than FileNotFoundError, as a git
        without its execute bit raises; a handler that named only
        FileNotFoundError would let it through. A working git gives its output.
        """
        err = PermissionError(13, 'Permission denied', 'git')
        with patch('proteus.doctor.subprocess.check_output', side_effect=err) as run:
            assert _git_stdout('/some/repo', 'rev-parse', 'HEAD') is None
        assert run.call_count == 1
        assert run.call_args.kwargs['timeout'] == _GIT_NOTE_TIMEOUT_S
        with patch('proteus.doctor.subprocess.check_output', return_value='abc\n'):
            assert _git_stdout('/some/repo', 'rev-parse', 'HEAD') == 'abc\n'

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
        assert 'dev@proteus-framework.org' in out
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
        assert 'dev@proteus-framework.org' not in out

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
        assert 'dev@proteus-framework.org' in out

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
        assert 'dev@proteus-framework.org' in captured.out
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
        assert 'dev@proteus-framework.org' in out


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


class TestBelowFloorAdvice:
    """The fix of an editable module clone below its floor is advice that changes nothing."""

    @staticmethod
    def _site(tmp_path) -> tuple:
        """A PROTEUS root with a MORS clone at 26.10.05; the remote gets 26.10.06 after the clone.

        Returns the root, the clone and its 3 commits.
        """
        root = scratch_proteus_root(tmp_path / 'PROTEUS', ['fwl-mors>=26.10.6'])
        checkout = root / 'MORS'
        commits = _detached_clone(tmp_path, checkout, late_tag='26.10.06')
        return root, checkout, commits

    @staticmethod
    def _check(root, checkout) -> CheckResult:
        """Run check_python_package on fwl-mors 26.10.5, editable at ``checkout``, floor 26.10.6."""
        from packaging.requirements import Requirement

        with (
            patch('proteus.doctor._repo_root', return_value=root),
            patch('proteus.doctor.importlib.metadata.version', return_value='26.10.5'),
            patch('proteus.doctor._editable_checkout_path', return_value=str(checkout)),
            patch('proteus.doctor._imported_package_dir', return_value=None),
        ):
            return check_python_package('fwl-mors', Requirement('fwl-mors>=26.10.6'))

    def test_check_runs_no_git_write_and_gives_the_advice(self, tmp_path, monkeypatch):
        """Behind a git wrapper that lets only read-only calls reach the clone or the PROTEUS
        root, the check runs and keeps HEAD, its status call carries --no-optional-locks; the
        fix holds the fetch, the checkout of the tag the fetch brings, and the --force
        command, and proteus update may not run it."""
        root, checkout, commits = self._site(tmp_path)
        guard = tmp_path / 'guard'
        guard.mkdir()
        log, marker = tmp_path / 'git-calls.log', tmp_path / 'git-write.marker'
        install_git_guard(guard, log, marker, roots=[root], trees=[checkout])
        monkeypatch.setenv('PATH', f'{guard}{os.pathsep}{os.environ["PATH"]}')
        r = self._check(root, checkout)
        calls = log.read_text().splitlines()
        assert not marker.exists(), calls
        args = [c.split('\t')[1].split() for c in calls]
        assert [a for a in args if not git_call_is_read_only(a)] == []
        assert ['--no-optional-locks', '-C', str(checkout), 'status', '--porcelain'] in args
        assert git(checkout, 'rev-parse', 'HEAD') == commits[0]
        assert r.status == FAIL
        assert r.auto_fixable is False
        pins = root / 'tools' / '_module_pins.py'
        assert f'git -C {checkout} fetch --tags && ' in r.fix_cmd
        assert (
            f'git -C {checkout} checkout --detach "refs/tags/$(git -C {checkout} tag -l |'
            f' python {pins} fwl-mors floor-tag)"'
        ) in r.fix_cmd
        assert r.fix_cmd.endswith(
            f'  # or replace the clone: cd {root} && bash tools/get_python_module.sh MORS --force'
        )

    @pytest.mark.parametrize(
        'has_tag', [False, True], ids=['tag after fetch', 'tag in the clone']
    )
    def test_hand_commands_move_the_clone_to_the_tag_the_fetch_brings(
        self, tmp_path, monkeypatch, has_tag
    ):
        """Run from another folder, the commands before the comment fetch the tag 26.10.06,
        check it out detached and reinstall the clone, although branches named 26.10.06 and
        tags/26.10.06 point at the tip; its untracked file stays. Without the tag in the clone
        the checkout picks it after the fetch, with it the checkout names
        ``refs/tags/26.10.06``."""
        root, checkout, commits = self._site(tmp_path)
        (checkout / 'notes.txt').write_text('kept\n')
        git(checkout, 'branch', '26.10.06', commits[2])
        git(checkout, 'branch', 'tags/26.10.06', commits[2])
        if has_tag:
            git(checkout, 'fetch', '-q', '--tags')
        r = self._check(root, checkout)
        assert ('checkout --detach refs/tags/26.10.06 ' in r.fix_cmd) is has_tag
        pip_log = _stub_pip(tmp_path, monkeypatch)
        hand = r.fix_cmd.split('  # ')[0]
        done = subprocess.run(
            ['bash', '-c', hand], cwd=tmp_path, capture_output=True, text=True, timeout=30
        )
        assert done.returncode == 0, done.stderr
        assert git(checkout, 'rev-parse', 'HEAD') == commits[1]
        assert git(checkout, 'rev-parse', 'HEAD') not in (commits[0], commits[2])
        detached = subprocess.run(
            ['git', '-C', str(checkout), 'symbolic-ref', '-q', 'HEAD'], capture_output=True
        )
        assert detached.returncode == 1
        assert (checkout / 'notes.txt').read_text() == 'kept\n'
        assert pip_log.read_text() == f'install -e {checkout}\n'

    @pytest.mark.parametrize(
        ('where', 'force'),
        [('module folder', True), ('symlink to it', True), ('elsewhere', False)],
    )
    def test_force_part_only_for_the_clone_in_the_module_folder(self, tmp_path, where, force):
        """The --force command, which replaces <root>/MORS, follows the hand commands only when
        the editable checkout resolves to that folder, also through a symlink; a clone moved
        to another folder gets the hand commands alone."""
        root, checkout, _ = self._site(tmp_path)
        if where == 'symlink to it':
            editable = tmp_path / 'link-to-MORS'
            editable.symlink_to(checkout)
        elif where == 'elsewhere':
            editable = tmp_path / 'elsewhere' / 'MORS'
            editable.parent.mkdir()
            checkout.rename(editable)
        else:
            editable = checkout
        r = self._check(root, editable)
        hand = f'git -C {editable} fetch --tags && git -C {editable} checkout --detach "refs/tags/$('
        tail = f'  # or replace the clone: cd {root} && bash tools/get_python_module.sh MORS --force'
        assert r.fix_cmd.startswith(hand)
        assert r.fix_cmd.endswith(f' && pip install -e {editable}{tail if force else ""}')
        assert r.fix_cmd.count('--force') == int(force)
        assert r.auto_fixable is False

    def test_update_shows_the_advice_and_does_not_run_it(self, tmp_path, capsys):
        """proteus update lists the advice among the issues without an automatic fix, runs no
        fix command and reports the install as unhealthy."""
        root, checkout, _ = self._site(tmp_path)
        r = self._check(root, checkout)
        runner = Mock(return_value=0)
        with (
            patch('proteus.doctor.run_all_checks', return_value=[r]),
            patch('proteus.doctor._run_fix_command', runner),
            patch('proteus.doctor.doctor_entry', return_value=True),
        ):
            healthy = update_entry()
        out = capsys.readouterr().out
        assert healthy is False
        runner.assert_not_called()
        assert 'none of them have an automatic fix' in out
        assert 'get_python_module.sh MORS --force' in out

    def test_setup_command_names_the_script_of_each_module(self):
        """Aragog, Zalmoxis, VULCAN, Morrigan and get_python_module.sh modules get their command."""
        assert _setup_command('fwl-aragog') == 'tools/get_aragog.sh'
        assert _setup_command('fwl-zalmoxis') == 'tools/get_zalmoxis.sh'
        assert _setup_command('fwl-mors') == 'tools/get_python_module.sh MORS'
        assert _setup_command('FWL-Calliope') == 'tools/get_python_module.sh CALLIOPE'
        assert _setup_command('fwl-vulcan') == 'tools/get_vulcan.sh'
        assert _setup_command('fwl-morrigan') == 'tools/get_morrigan.sh'
        assert _setup_command('fwl-io') is None
        assert {'fwl-vulcan', 'fwl-morrigan'}.isdisjoint(PYTHON_PACKAGES)
        expected_folders = {*PYTHON_PACKAGES, *STANDARD_MODULE_PACKAGES} - {
            'fwl-proteus',
            'fwl-io',
        }
        assert set(_MODULE_FOLDERS) == expected_folders
        paired = {
            f'fwl-{s.stem.removeprefix("get_")}'
            for s in (PROTEUS_ROOT / 'tools').glob('get_*.sh')
            if 'checkout_paired_or_floor ' in s.read_text()
        }
        assert paired == set(_PAIRED_MODULES) == {'fwl-aragog', 'fwl-zalmoxis'}
        text = (PROTEUS_ROOT / 'tools' / 'get_python_module.sh').read_text()
        accepted = re.search(r'^modules="([^"]*)"$', text, re.MULTILINE).group(1).split()
        assert sorted(_PYTHON_MODULE_REPOS.values()) == sorted(accepted)
        assert all(dist == f'fwl-{repo.lower()}' for dist, repo in _PYTHON_MODULE_REPOS.items())

        # The module folder of each script with a fixed destination is the one it clones into.
        for dist, folder in _MODULE_FOLDERS.items():
            cmd = _setup_command(dist)
            if cmd is None:
                continue
            script = PROTEUS_ROOT / cmd.split()[0]
            if dist not in _PYTHON_MODULE_REPOS:
                assert f'workpath="$root/{folder}/"' in script.read_text(), dist
        assert len(_MODULE_FOLDERS) == 9

    def test_setup_command_and_script_for_atmodeller_in_tree(self, tmp_path):
        """When tools/get_atmodeller.sh exists, setup command and script helper resolve it."""
        tools = tmp_path / 'tools'
        tools.mkdir(parents=True)
        (tools / 'get_atmodeller.sh').touch()
        with patch('proteus.doctor._repo_root', return_value=tmp_path):
            assert _get_script_for_package('atmodeller') == 'tools/get_atmodeller.sh'
            assert _setup_command('atmodeller') == 'tools/get_atmodeller.sh'

    def test_without_the_source_tree_a_clone_gets_git_pull(self, tmp_path):
        """A wheel install has no tools folder, so no setup script: the fix of an editable
        clone below its floor is the git pull of a package without a script."""
        _, checkout, _ = self._site(tmp_path)
        r = self._check(tmp_path / 'wheel', checkout)
        assert r.fix_cmd == f'cd {shlex.quote(str(checkout))} && git pull && pip install -e .'
        assert r.auto_fixable is True

    @staticmethod
    def _folder_in_a_proteus_repository(tmp_path, kind: str, where: str) -> tuple:
        """A PROTEUS root that is a git repository on main with a modified tracked file, and a
        MORS folder with setup.py inside it, without .git or with an empty .git, at the module
        folder or below another folder of the root. Returns the root and the folder."""
        root = scratch_proteus_root(tmp_path / 'PROTEUS', ['fwl-mors>=26.10.6'])
        (root / 'notes.txt').write_text('PROTEUS\n')
        git(root, 'init', '-q', '-b', 'main')
        git(root, 'add', '-A')
        git(root, 'commit', '-q', '-m', 'PROTEUS')
        (root / 'notes.txt').write_text('changed\n')
        folder = root / 'MORS' if where == 'module folder' else root / 'mine' / 'MORS'
        folder.mkdir(parents=True)
        (folder / 'setup.py').write_text('# mine\n')
        if kind == 'empty .git':
            (folder / '.git').mkdir()
        return root, folder

    @pytest.mark.parametrize('kind', ['no .git', 'empty .git'])
    @pytest.mark.parametrize('where', ['module folder', 'elsewhere in the root'])
    def test_folder_that_is_not_its_own_clone_gets_no_git_command(
        self, tmp_path, monkeypatch, kind, where
    ):
        """An editable MORS folder below its floor that is not a git clone of its own, inside a
        PROTEUS root that is a git repository, gets advice with no git command, whose fetch and
        checkout would act on the PROTEUS repository: the --force replacement for the module
        folder, else a comment that says to move it away. Behind the git guard only read-only
        calls run; proteus update runs nothing, and running the comment changes nothing in the
        PROTEUS repository (HEAD, branch, tags, status)."""
        root, folder = self._folder_in_a_proteus_repository(tmp_path, kind, where)
        monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
        guard = tmp_path / 'guard'
        guard.mkdir()
        log, marker = tmp_path / 'git-calls.log', tmp_path / 'git-write.marker'
        install_git_guard(guard, log, marker, roots=[root], trees=[folder])
        monkeypatch.setenv('PATH', f'{guard}{os.pathsep}{os.environ["PATH"]}')
        state = [
            git(root, 'rev-parse', 'HEAD'),
            git(root, 'symbolic-ref', '--short', 'HEAD'),
            git(root, 'tag', '-l'),
            git(root, '--no-optional-locks', 'status', '--porcelain'),
        ]
        r = self._check(root, folder)
        calls = log.read_text().splitlines()
        assert not marker.exists(), calls
        assert [c for c in calls if not git_call_is_read_only(c.split('\t')[1].split())] == []
        assert r.status == FAIL
        assert r.auto_fixable is False
        assert 'git -C' not in r.fix_cmd
        assert 'fetch' not in r.fix_cmd and 'checkout' not in r.fix_cmd
        setup = f'cd {root} && bash tools/get_python_module.sh MORS'
        if where == 'module folder':
            assert r.fix_cmd == (
                f'{setup} --force  # {folder} is not a git clone of its own; this replaces it'
                ' with a new clone, which deletes everything in it'
            )
        else:
            assert r.fix_cmd == (
                f'# {folder} is not a git clone of its own, so no git command moves it to the'
                f' release tag; move it away, then run: {setup}'
            )
            done = subprocess.run(
                ['bash', '-c', r.fix_cmd], cwd=tmp_path, capture_output=True, timeout=10
            )
            assert done.returncode == 0
        runner = Mock(return_value=0)
        with (
            patch('proteus.doctor.run_all_checks', return_value=[r]),
            patch('proteus.doctor._run_fix_command', runner),
        ):
            assert update_entry() is False
        runner.assert_not_called()
        assert state == [
            git(root, 'rev-parse', 'HEAD'),
            git(root, 'symbolic-ref', '--short', 'HEAD'),
            git(root, 'tag', '-l'),
            git(root, '--no-optional-locks', 'status', '--porcelain'),
        ]
        assert sorted(q.name for q in folder.iterdir()) == sorted(
            ['setup.py'] + (['.git'] if kind == 'empty .git' else [])
        )

    @staticmethod
    def _check_io(root, checkout) -> CheckResult:
        """Run check_python_package on fwl-io 26.10.5, editable at ``checkout``, floor 26.10.6;
        fwl-io has no setup script."""
        from packaging.requirements import Requirement

        with (
            patch('proteus.doctor._repo_root', return_value=root),
            patch('proteus.doctor.importlib.metadata.version', return_value='26.10.5'),
            patch('proteus.doctor._editable_checkout_path', return_value=str(checkout)),
            patch('proteus.doctor._imported_package_dir', return_value=None),
        ):
            return check_python_package('fwl-io', Requirement('fwl-io>=26.10.6'))

    @pytest.mark.parametrize('kind', ['no .git', 'empty .git'])
    def test_package_without_a_script_in_a_folder_that_is_not_its_own_clone_gets_no_git_pull(
        self, tmp_path, monkeypatch, kind
    ):
        """An editable fwl-io below its floor, in a folder that is not a git clone of its own
        inside a PROTEUS root that is a git repository, gets the pip advice with no git
        command, which git pull would run on the PROTEUS repository; it is not auto-fixable,
        so proteus update runs nothing, and the root keeps its HEAD, branch and status."""
        root, folder = self._folder_in_a_proteus_repository(tmp_path, kind, 'elsewhere')
        monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
        state = [
            git(root, 'rev-parse', 'HEAD'),
            git(root, 'symbolic-ref', '--short', 'HEAD'),
            git(root, '--no-optional-locks', 'status', '--porcelain'),
        ]
        r = self._check_io(root, folder)
        assert r.status == FAIL
        assert r.auto_fixable is False
        assert r.fix_cmd == (
            f'pip install -U "fwl-io>=26.10.6"  # {folder} is not a git clone of its own, so'
            ' no git command updates it; this installs the release in place of the editable'
            ' folder'
        )
        assert 'git pull' not in r.fix_cmd and 'cd ' not in r.fix_cmd
        runner = Mock(return_value=0)
        with (
            patch('proteus.doctor.run_all_checks', return_value=[r]),
            patch('proteus.doctor._run_fix_command', runner),
        ):
            assert update_entry() is False
        runner.assert_not_called()
        assert state == [
            git(root, 'rev-parse', 'HEAD'),
            git(root, 'symbolic-ref', '--short', 'HEAD'),
            git(root, '--no-optional-locks', 'status', '--porcelain'),
        ]
        assert 'M notes.txt' in state[2]

    def test_package_without_a_script_in_its_own_clone_keeps_git_pull(self, tmp_path):
        """An editable fwl-io below its floor in a git clone of its own keeps the automatic
        git pull fix, and proteus update runs exactly that command."""
        root, checkout, _ = self._site(tmp_path)
        r = self._check_io(root, checkout)
        assert r.status == FAIL
        assert r.auto_fixable is True
        assert r.fix_cmd == f'cd {shlex.quote(str(checkout))} && git pull && pip install -e .'
        runner = Mock(return_value=0)
        with (
            patch('proteus.doctor.run_all_checks', return_value=[r]),
            patch('proteus.doctor._run_fix_command', runner),
            patch('proteus.doctor._repo_root', return_value=root),
            patch('proteus.doctor.doctor_entry', return_value=True),
        ):
            assert update_entry() is True
        runner.assert_called_once_with(r.fix_cmd, root)

    @pytest.mark.parametrize('kind', ['no .git', 'empty .git'])
    def test_folder_that_is_not_its_own_clone_shows_no_commit_of_the_root(
        self, tmp_path, monkeypatch, kind
    ):
        """The annotation of an editable MORS folder that is not a git clone of its own says so,
        without a commit or a dirty marker: the short head and the status of the folder would
        be those of the PROTEUS repository, which has a modified file. Neither call runs."""
        from packaging.requirements import Requirement

        root, folder = self._folder_in_a_proteus_repository(tmp_path, kind, 'module folder')
        monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
        assert _git_short_head(str(folder)) == git(root, 'rev-parse', '--short', 'HEAD')
        assert _git_dirty(str(folder)) is True
        with (
            patch('proteus.doctor._repo_root', return_value=root),
            patch('proteus.doctor.importlib.metadata.version', return_value='26.10.6'),
            patch('proteus.doctor._editable_checkout_path', return_value=str(folder)),
            patch('proteus.doctor._imported_package_dir', return_value=None),
            patch('proteus.doctor._git_short_head', wraps=_git_short_head) as short,
            patch('proteus.doctor._git_dirty', wraps=_git_dirty) as dirty,
        ):
            r = check_python_package('fwl-mors', Requirement('fwl-mors>=26.10.6'))
        assert r.status == PASS
        assert r.message == '26.10.6 [editable @ MORS; not a git clone of its own]'
        assert git(root, 'rev-parse', '--short', 'HEAD') not in r.message
        short.assert_not_called()
        dirty.assert_not_called()


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
        assert 'dev@proteus-framework.org' in out
        assert 'github.com/FormingWorlds/PROTEUS/issues' in out
        assert 'discussions' in out

    def test_support_email_constant_is_exact_and_rendered_verbatim(self, tmp_path, capsys):
        """The support address is pinned to its exact value, and the failure
        prompt renders that exact constant exactly once. A stray leading or
        trailing character, whitespace, or a duplicate copy in the constant
        fails the equality check. The exact count also catches the address
        being rendered more than once, which the substring checks elsewhere
        cannot tell apart from a single occurrence."""
        assert _SUPPORT_EMAIL == 'dev@proteus-framework.org'
        _print_support_prompt('doctor', tmp_path / 'proteus_doctor.log')
        out = capsys.readouterr().out
        assert out.count(_SUPPORT_EMAIL) == 1

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
        assert 'dev@proteus-framework.org' in result.output

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
        assert 'dev@proteus-framework.org' not in result.output


class TestCheckCvode:
    """check_cvode reports whether Aragog can use its production integrator."""

    def test_pass_when_the_wrapper_imports(self):
        """An importable wrapper passes and suggests nothing.

        Contract clause: the check exists to surface a silent fallback, so on
        a healthy install it must be quiet. A fix command on a passing check
        would put an unnecessary conda build in front of `proteus update`.
        """
        with (
            patch('proteus.doctor.importlib.util.find_spec', return_value=object()),
            patch('proteus.doctor.importlib.import_module', return_value=object()),
        ):
            r = check_cvode()
        assert r.status == PASS
        assert r.fix_cmd is None
        assert r.category == 'environment'

    def test_warn_when_the_wrapper_is_absent(self):
        """A missing wrapper warns and names the script that installs it.

        Contract clause: without the wrapper Aragog integrates with scipy
        Radau, which is a different solver, so the operator has to be told
        before a long coupled run rather than only in the per-solve log line.
        """
        with patch('proteus.doctor.importlib.util.find_spec', return_value=None):
            r = check_cvode()
        assert r.status == WARN
        assert r.fix_cmd == 'bash tools/get_cvode.sh'
        # Runnable from the repo root, so `proteus update` can apply it
        # rather than only printing it.
        assert r.auto_fixable is True
        # The message says what the run does instead, not just that something
        # is missing.
        assert 'Radau' in r.message

    def test_warn_when_the_wrapper_is_installed_but_does_not_load(self):
        """A wrapper that imports and then fails warns, naming the failure.

        Physical scenario for the operator: the wrapper is compiled against
        the SUNDIALS C library, so an ABI or version mismatch leaves the
        package importable by name while the extension fails to load. That
        state passes a presence check and still falls back to Radau, so it
        has to be caught here.
        """
        with (
            patch('proteus.doctor.importlib.util.find_spec', return_value=object()),
            patch(
                'proteus.doctor.importlib.import_module',
                side_effect=ImportError('libsundials_cvode.so: cannot open'),
            ),
        ):
            r = check_cvode()
        assert r.status == WARN
        assert 'ImportError' in r.message
        assert r.fix_cmd == 'bash tools/get_cvode.sh'

    def test_the_check_is_wired_into_the_diagnose_run(self):
        """`proteus doctor` runs the check rather than only defining it.

        A check that is never called is the failure mode this guards: the
        function can be correct and the operator still never sees it.
        """
        with (
            patch('proteus.doctor._dependency_specs', return_value={}),
            patch('proteus.doctor._module_pins', return_value={}),
        ):
            results = run_all_checks()
        cvode = [r for r in results if r.name == 'cvode']
        assert len(cvode) == 1, f'expected one cvode check, found {len(cvode)}'
        assert cvode[0].category == 'environment'
