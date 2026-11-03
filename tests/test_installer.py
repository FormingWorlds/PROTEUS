"""Unit tests for the PROTEUS unified installer (install.sh).

Tests the pre-flight check logic, argument parsing, helper functions,
and error messages by running install.sh in controlled subprocess
environments. Does not execute actual installations (SOCRATES, AGNI,
pip install); those are tested by the CI pipeline and smoke tests.

The Python modules the installer clones are pinned to the release tag of
their version floor: ``checkout_floor_tag`` (``tools/_get_common.sh``) and
``clone_and_install`` (lifted from install.sh at run time) run against
local scratch repositories and a recording ``pip`` stub, without network.
The module lines of the Python phase run whole, with the setup scripts of
Aragog and Zalmoxis and a git wrapper that lets only read-only calls reach a
module clone, to pin that an existing clone is installed as it is and the one
line that names each clone that is not at its release tag.

``setup_conda_env`` (lifted from install.sh) runs against a fake conda base:
a ``conda`` executable and a ``conda.sh`` shell function that record their
calls, so no env is made or activated on the host.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import json
import os
import pty
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import termios
import tomllib
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT, git, install_git_guard, scratch_proteus_root

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

INSTALL_SH = PROTEUS_ROOT / 'install.sh'
TOOLS_DIR = PROTEUS_ROOT / 'tools'


@pytest.fixture(autouse=True)
def stub_conda(tmp_path_factory, monkeypatch):
    """Put a no-op conda first on PATH, so the failure diagnostics do not wait on a cold conda."""
    bindir = tmp_path_factory.mktemp('stub_conda')
    stub = bindir / 'conda'
    stub.write_text('#!/bin/sh\nexit 0\n')
    stub.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}{os.environ["PATH"]}')


class TestInstallerArgParsing:
    """Verify argument parsing and help output."""

    def test_help_flag_exits_zero(self):
        """--help prints usage and exits with code 0."""
        result = subprocess.run(
            ['bash', str(INSTALL_SH), '--help'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert 'Usage' in result.stdout
        assert '--all-data' in result.stdout
        assert '--interactive' in result.stdout

    def test_h_flag_exits_zero(self):
        """Short -h flag also works."""
        result = subprocess.run(
            ['bash', str(INSTALL_SH), '-h'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert 'Usage' in result.stdout

    def test_unknown_arg_exits_nonzero(self):
        """Unknown arguments cause a non-zero exit with error message."""
        result = subprocess.run(
            ['bash', str(INSTALL_SH), '--bogus-flag'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert 'Unknown argument' in output

    def test_interactive_flag_accepted(self):
        """--interactive and its -i short form are accepted by the argument loop.

        The flag is checked ahead of --help, which exits 0 on its own. The
        bogus-flag run is what gives the exit code meaning: the loop visits
        every argument in order and rejects an unrecognised one even when
        --help follows, so a clean exit here is --interactive being accepted
        rather than --help short-circuiting the parse.
        """
        for flag in ('--interactive', '-i'):
            result = subprocess.run(
                ['bash', str(INSTALL_SH), flag, '--help'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f'{flag} rejected: {result.stdout + result.stderr}'
            assert 'Unknown argument' not in result.stdout + result.stderr

        # Control: an unrecognised flag in the same position does fail, so the
        # clean exits above are not an artefact of the trailing --help.
        bogus = subprocess.run(
            ['bash', str(INSTALL_SH), '--not-a-flag', '--help'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert bogus.returncode != 0
        assert 'Unknown argument' in bogus.stdout + bogus.stderr

    def test_verbose_flag_accepted_set_and_documented(self):
        """--verbose is accepted ahead of --help, sets VERBOSE and has a line in --help.

        The parser arm is pinned as text, because the variable it sets is what
        run_logged reads; a misspelt variable would pass the exit-code check alone.
        """
        result = subprocess.run(
            ['bash', str(INSTALL_SH), '--verbose', '--help'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'Unknown argument' not in result.stdout + result.stderr
        help_lines = [ln for ln in result.stdout.splitlines() if ln.startswith('  --verbose ')]
        assert len(help_lines) == 1
        assert 'log file' in help_lines[0]
        arms = [
            ln.strip()
            for ln in INSTALL_SH.read_text().splitlines()
            if ln.strip().startswith('--verbose)')
        ]
        assert arms == ['--verbose)  VERBOSE="true" ;;']


def _bin_dir_without(tmp_path, omitted: set[str], path: str | None = None) -> Path:
    """Mirror every executable on `path` (default PATH) into a fresh directory, minus `omitted`.

    Lets a test run the installer as if the host lacked one specific tool,
    without disturbing anything else the pre-flight checks need.
    """
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    for entry in (path or os.environ['PATH']).split(os.pathsep):
        source = Path(entry)
        if not source.is_dir():
            continue
        for exe in source.iterdir():
            if exe.name in omitted or (bindir / exe.name).exists():
                continue
            try:
                if exe.is_file() and os.access(exe, os.X_OK):
                    (bindir / exe.name).symlink_to(exe)
            except OSError:
                continue
    return bindir


@pytest.fixture(scope='session')
def system_path(tmp_path_factory) -> str:
    """One PATH entry with the tools of /usr/bin and /bin, minus conda, julia and juliaup.

    A host can hold any of the three in /usr/bin, where a test of the missing
    tool would find it behind its own stubs.
    """
    hidden = {'conda', 'julia', 'juliaup'}
    return str(_bin_dir_without(tmp_path_factory.mktemp('system'), hidden, '/usr/bin:/bin'))


def _run_preflight(bindir: Path) -> str:
    """Run install.sh with PATH restricted to `bindir`; return its combined output."""
    env = os.environ.copy()
    env['PATH'] = str(bindir)
    # An env other than base is active, so the installer makes no env of its own.
    if env.get('CONDA_DEFAULT_ENV', 'base') in ('', 'base'):
        env['CONDA_DEFAULT_ENV'] = 'proteus'
    result = subprocess.run(
        ['bash', str(INSTALL_SH)],
        capture_output=True,
        text=True,
        timeout=25,
        env=env,
        cwd=str(PROTEUS_ROOT),
    )
    return result.stdout + result.stderr


class TestPreflightChecks:
    """Verify pre-flight check failure modes."""

    def test_fails_without_conda_env(self):
        """Without an env and with a conda that cannot activate one, the installer stops with conda instructions.

        The conda stub of this file answers every call with nothing, so it names no
        envs directory in which the installer could look for the env. Discrimination:
        the error must name the manual commands and the step that failed.
        """
        env = os.environ.copy()
        for var in ('CONDA_DEFAULT_ENV', 'CONDA_PREFIX', 'PIXI_ENVIRONMENT_NAME'):
            env.pop(var, None)
        result = subprocess.run(
            ['bash', str(INSTALL_SH)],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
            cwd=str(PROTEUS_ROOT),
        )
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert 'conda' in output.lower()
        assert 'conda create -n proteus python=3.12' in output
        assert "'conda config --show envs_dirs' named no envs directory." in output

    def test_fails_outside_proteus_repo(self, tmp_path):
        """Running install.sh from outside the PROTEUS repo fails because
        pyproject.toml is not found.

        Discrimination: the error must mention pyproject.toml or
        'repository root' so the user understands the problem.
        """
        install_copy = tmp_path / 'install.sh'
        install_copy.write_text(INSTALL_SH.read_text())
        result = subprocess.run(
            ['bash', str(install_copy)],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(tmp_path),
            env=os.environ.copy(),
        )
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert 'pyproject.toml' in output or 'repository root' in output.lower()

    def test_netcdf_fortran_required_even_when_c_interface_present(self, tmp_path):
        """A host with nc-config but no nf-config is rejected at pre-flight.

        SOCRATES needs the Fortran interface, so the C interface alone is not
        enough. Discrimination: a check that accepts either tool reports
        nothing missing on such a host, and the build fails many phases later
        with a far less obvious message. Requiring the Fortran package to be
        named, and the C package not to be, fails against that check rather
        than merely restating it.
        """
        output = _run_preflight(_bin_dir_without(tmp_path, {'nf-config'}))
        assert 'Missing system packages' in output
        assert 'netcdf-fortran-dev' in output
        assert 'netcdf-dev' not in output

    def test_both_netcdf_interfaces_named_when_both_absent(self, tmp_path):
        """With neither tool present, both packages are named.

        The opposite boundary from the test above. Reporting only one would
        send a user who installs it round the same failure a second time.
        """
        output = _run_preflight(_bin_dir_without(tmp_path, {'nc-config', 'nf-config'}))
        assert 'Missing system packages' in output
        assert 'netcdf-dev' in output
        assert 'netcdf-fortran-dev' in output


def _echo_e_lines(content: str) -> list[int]:
    """Return the 1-based line numbers carrying a live (uncommented) echo -e."""
    return [
        i + 1
        for i, line in enumerate(content.split('\n'))
        if 'echo -e' in line and not line.strip().startswith('#')
    ]


class TestBashSyntax:
    """Verify the script is valid bash."""

    def test_installer_is_valid_bash(self, tmp_path):
        """install.sh parses without syntax errors, and bash -n would say so.

        A clean parse prints nothing at all, so an empty stderr is part of the
        contract. The broken copy is the control: it proves bash -n in this
        harness really does reject malformed input, so the clean result above
        is a property of install.sh and not of a check that always passes.
        """
        result = subprocess.run(
            ['bash', '-n', str(INSTALL_SH)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f'Bash syntax error: {result.stderr}'
        assert result.stderr == ''

        broken = tmp_path / 'broken.sh'
        broken.write_text(INSTALL_SH.read_text() + '\nif [ -z "$X" ; then\n')
        bad = subprocess.run(
            ['bash', '-n', str(broken)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert bad.returncode != 0

    def test_no_echo_e_usage(self):
        """Script uses printf instead of echo -e for portability.

        echo -e is not POSIX; printf is. The scanner ignores commented lines,
        so the synthetic pair pins both of its branches: a live echo -e is
        reported and a commented one is not. Without that, a scanner that
        silently found nothing would make the empty result above look clean.
        """
        content = INSTALL_SH.read_text()
        assert _echo_e_lines(content) == [], f'echo -e found on lines: {_echo_e_lines(content)}'

        # The scanner reports a live echo -e and skips a commented one.
        synthetic = 'echo hello\n# echo -e "commented"\necho -e "live"\n'
        assert _echo_e_lines(synthetic) == [3]


def _install_sh_blocks(*markers: str) -> str:
    """The shipped lines of install.sh from each line holding a marker to the next blank line."""
    lines = INSTALL_SH.read_text().splitlines()
    blocks = []
    for marker in markers:
        start = next(i for i, ln in enumerate(lines) if ln.startswith(marker))
        end = next(i for i, ln in enumerate(lines) if not ln.strip() and i > start)
        blocks += lines[start:end]
    return '\n'.join(blocks) + '\n'


def _run_with_helpers(script: str, **env: str) -> subprocess.CompletedProcess:
    """Run a bash snippet after install.sh's colour codes and log helpers, under its shell options."""
    defs = 'set -euo pipefail\n' + _install_sh_blocks("RED='", 'info()')
    return subprocess.run(
        ['bash', '-c', defs + script],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=10,
    )


class TestLogHelpers:
    """Verify where the installer's log helpers write."""

    def test_diagnostics_go_to_stderr_and_info_to_stdout(self):
        """warn and fail write only to stderr, info only to stdout.

        Discrimination: a captured helper that calls warn or fail keeps its
        stdout clean only while both stay off stdout; info must not move.
        """
        out = _run_with_helpers('info one\nwarn two\nfail three\n')
        assert out.returncode == 0, out.stderr
        assert '[INFO]' in out.stdout and 'one' in out.stdout
        assert '[WARN]' not in out.stdout and '[FAIL]' not in out.stdout
        assert '[WARN]' in out.stderr and 'two' in out.stderr
        assert '[FAIL]' in out.stderr and 'three' in out.stderr
        assert '[INFO]' not in out.stderr


class TestBelowFloorStop:
    """The Verification phase stops only when a module is below its version floor."""

    @staticmethod
    def _run(tmp_path, stub_body: str, failed: str = '') -> subprocess.CompletedProcess:
        """Run the floor block of Phase 9 and die with a stub python3 that records its call."""
        bindir = tmp_path / 'bin'
        bindir.mkdir()
        stub = bindir / 'python3'
        stub.write_text(f'#!/bin/bash\nprintf "%s" "$*" > {tmp_path}/argv\n{stub_body}\n')
        stub.chmod(0o755)
        script = (
            f'CURRENT_PHASE=9\nFAILED_MODULES={shlex.quote(failed)}\ncollect_env_info() {{ :; }}\n'
            + _install_sh_blocks('die()', '# Only a module below')
            + 'echo after-floor-check\n'
        )
        return _run_with_helpers(script, PATH=f'{bindir}:{os.environ["PATH"]}')

    def test_below_floor_dies_with_each_fix_command(self, tmp_path):
        """Two modules below their floor: one FAIL line each, then die with the failed modules.

        Discrimination: a stray stdout line of the import carries no prefix, so it never
        becomes a FAIL line; the Phase 7 summary of failed modules reaches the stop.
        """
        out = self._run(
            tmp_path,
            "echo 'juliapkg: stray line'; echo 'below-floor: fwl-janus: fix-janus'; "
            "echo 'below-floor: fwl-vulcan: bash tools/get_vulcan.sh'",
            failed='AGNI (bash tools/get_agni.sh)',
        )
        assert out.returncode == 1, out.stdout
        assert 'below_floor_fixes' in (tmp_path / 'argv').read_text()
        assert out.stderr.count('[FAIL]') == 3
        assert 'Below its minimum version: fwl-janus: fix-janus' in out.stderr
        assert 'Below its minimum version: fwl-vulcan: bash tools/get_vulcan.sh' in out.stderr
        assert 'stray line' not in out.stderr
        assert 'Not installed: AGNI (bash tools/get_agni.sh).' in out.stderr
        assert 'Installation failed at Phase 9.' in out.stdout
        assert 'after-floor-check' not in out.stdout

    @pytest.mark.parametrize(
        ('stub_body', 'warned'),
        [('exit 0', False), ("echo 'stray line'", False), ('echo boom >&2; exit 1', True)],
    )
    def test_no_module_below_floor_keeps_exit_0(self, tmp_path, stub_body, warned):
        """No module below its floor, a stray line, or a failed check keep exit 0 (ci-pixi)."""
        out = self._run(tmp_path, stub_body)
        assert out.returncode == 0, out.stderr
        assert 'after-floor-check' in out.stdout
        assert '[FAIL]' not in out.stderr
        assert ('Could not check the module versions' in out.stderr) == warned


_VAR_FUNCTIONS = (
    'command_exists',
    'set_install_var',
    'bind_install_vars',
    'note_old_shell_exports',
    'conda_activate_command',
    'print_next_steps',
    'print_exit_hint',
)

# A space and a single quote are the two values that a naive quoting breaks.
_VALUES = {
    'FWL_DATA': '/data/FWL DATA',
    'PYTHON_JULIAPKG_EXE': '/opt/julia-1.13.1/bin/julia',
    'RAD_DIR': "/home/o'brien/socrates",
    'FC_DIR': '/home/u/AGNI/fastchem',
}

# The same 4 variables without a single quote, all of which conda can hold.
_PLAIN = {**_VALUES, 'RAD_DIR': '/home/obrien/socrates'}

_SYSTEM_PATH = '/usr/bin:/bin'


def _set_script(values: dict[str, str]) -> str:
    """Bash lines that call set_install_var once per variable, in the order of ``values``."""
    return ''.join(f'set_install_var {name} {shlex.quote(v)}\n' for name, v in values.items())


def _sourced(env_file: Path) -> dict[str, str]:
    """The 4 variables as a fresh bash sees them after sourcing ``env_file``."""
    out = subprocess.run(
        [
            'bash',
            '--noprofile',
            '--norc',
            '-c',
            'source "$1"; printf "%s\\0" "$FWL_DATA" "$PYTHON_JULIAPKG_EXE" "$RAD_DIR" "$FC_DIR"',
            '_',
            str(env_file),
        ],
        capture_output=True,
        text=True,
        env={'PATH': _SYSTEM_PATH},
        timeout=10,
        check=True,
    )
    return dict(zip(_VALUES, out.stdout.split('\0')[:4]))


def _conda_calls(log: Path) -> list[list[str]]:
    """The argument lists the conda stub recorded, one list per call."""
    if not log.exists():
        return []
    calls, current = [], []
    for line in log.read_text().splitlines():
        if line == '---':
            calls.append(current)
            current = []
        else:
            current.append(line)
    return calls


def _start_file_state(home: Path) -> dict[str, tuple]:
    """Bytes, mode and symlink state of every shell start file under ``home``."""
    state = {}
    for name in ('.zshrc', '.bashrc', '.bash_profile', '.profile', '.zprofile'):
        path = home / name
        if not path.exists() and not path.is_symlink():
            continue
        link = os.readlink(path) if path.is_symlink() else None
        state[name] = (path.read_bytes(), path.stat().st_mode, os.lstat(path).st_mode, link)
    return state


def _write_start_file(path: Path, text: str) -> None:
    """Write a shell start file, or skip on a host that refuses to create one."""
    try:
        path.write_text(text)
    except PermissionError:
        pytest.skip(
            'this host refuses to create shell start files, even under a temporary HOME'
        )


class TestInstallVars:
    """set_install_var records the 4 install variables without editing a shell start file."""

    @pytest.fixture
    def site(self, tmp_path):
        """A PROTEUS folder, an empty HOME and a conda stub that records its arguments.

        The stub exits with ``STUB_CONDA_RC`` (default 0), so a test can make the
        conda call fail.
        """
        root = tmp_path / 'proteus'
        root.mkdir()
        home = tmp_path / 'home'
        home.mkdir()
        stubs = tmp_path / 'stubs'
        _stub(
            stubs / 'conda',
            '#!/bin/sh\nprintf "%s\\n" "$@" >> "$STUB_CONDA_LOG"\n'
            'echo --- >> "$STUB_CONDA_LOG"\nexit "${STUB_CONDA_RC:-0}"\n',
        )
        env = {
            'PATH': f'{stubs}{os.pathsep}{_SYSTEM_PATH}',
            'HOME': str(home),
            'STUB_CONDA_LOG': str(tmp_path / 'conda.log'),
        }
        return {
            'root': root,
            'home': home,
            'env_file': root / '.proteus-env.sh',
            'log': tmp_path / 'conda.log',
            'prefix': str(tmp_path / 'envs' / 'proteus'),
            'env': env,
        }

    @staticmethod
    def _call(site, script: str, **env: str) -> subprocess.CompletedProcess:
        """Run ``script`` after the shipped variable functions, under install.sh's options."""
        defs = (
            'set -euo pipefail\n'
            + f'SCRIPT_DIR={shlex.quote(str(site["root"]))}\n'
            + _install_sh_blocks("RED='", 'info()', 'PROTEUS_ENV_FILE=')
            + ''.join(_install_sh_function(f) for f in _VAR_FUNCTIONS)
        )
        return subprocess.run(
            ['bash', '-c', defs + script],
            capture_output=True,
            text=True,
            env={**site['env'], **env},
            timeout=20,
        )

    def test_four_variables_reach_the_exports_file_and_one_conda_call(self, site):
        """4 variables give 4 export lines that source back unchanged, and one conda call.

        The conda stub gets `env config vars set -p <prefix>` with the 4 pairs in
        one call. A second run with new values replaces the lines: still 4, with
        the new values. The closing message tells the env `proteus` to reactivate.
        """
        conda_env = {'CONDA_PREFIX': site['prefix'], 'CONDA_DEFAULT_ENV': 'proteus'}
        script = _set_script(_PLAIN) + 'bind_install_vars\nprint_next_steps\n'
        result = self._call(site, script, **conda_env)
        assert result.returncode == 0, result.stdout + result.stderr
        lines = site['env_file'].read_text().splitlines()
        assert len(lines) == 4
        assert [ln.split('=', 1)[0] for ln in lines] == [f'export {n}' for n in _PLAIN]
        assert all(ln.split('=', 1)[1].startswith("'") for ln in lines)
        assert _sourced(site['env_file']) == _PLAIN
        pairs = [f'{n}={v}' for n, v in _PLAIN.items()]
        assert _conda_calls(site['log']) == [
            ['env', 'config', 'vars', 'set', '-p', site['prefix'], *pairs]
        ]
        assert 'conda deactivate && conda activate proteus' in result.stdout
        assert 'source .proteus-env.sh' not in result.stdout

        newer = {n: f'{v}-2' for n, v in _PLAIN.items()}
        again = self._call(site, _set_script(newer) + 'bind_install_vars\n', **conda_env)
        assert again.returncode == 0, again.stdout + again.stderr
        assert len(site['env_file'].read_text().splitlines()) == 4
        assert _sourced(site['env_file']) == newer
        assert len(_conda_calls(site['log'])) == 2

    def test_value_with_a_single_quote_stays_out_of_conda(self, site):
        """RAD_DIR=/home/o'brien/socrates is in the exports file only, named in one warning.

        conda writes `export RAD_DIR='<value>'` unescaped, which breaks each later
        activation. The other 3 pairs go to conda in one call; the closing line
        sources the exports file, which reads the value back unchanged.
        """
        conda_env = {'CONDA_PREFIX': site['prefix'], 'CONDA_DEFAULT_ENV': 'proteus'}
        script = _set_script(_VALUES) + 'bind_install_vars\nprint_next_steps\n'
        result = self._call(site, script, **conda_env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _sourced(site['env_file']) == _VALUES
        pairs = [f'{n}={v}' for n, v in _VALUES.items() if n != 'RAD_DIR']
        assert _conda_calls(site['log']) == [
            ['env', 'config', 'vars', 'set', '-p', site['prefix'], *pairs]
        ]
        assert not [a for call in _conda_calls(site['log']) for a in call if "'" in a]
        warnings = [ln for ln in result.stderr.splitlines() if 'Not bound' in ln]
        assert warnings == [
            f'{_WARN_PREFIX}Not bound to the conda env: RAD_DIR, whose value holds a single '
            "quote that conda's activation script cannot read; load it with: "
            f'source {site["env_file"]}'
        ]
        assert 'source .proteus-env.sh' in result.stdout
        assert 'conda deactivate' not in result.stdout

    def test_only_quoted_values_make_no_conda_call(self, site):
        """With every value holding a single quote conda is not called, and the warning names them."""
        quoted = {'FWL_DATA': "/d/o'b", 'RAD_DIR': "/r/o'b"}
        script = _set_script(quoted) + 'bind_install_vars\n'
        result = self._call(site, script, CONDA_PREFIX=site['prefix'])
        assert result.returncode == 0, result.stdout + result.stderr
        assert _conda_calls(site['log']) == []
        assert 'Not bound to the conda env: FWL_DATA RAD_DIR,' in result.stderr
        assert 'Bound ' not in result.stdout

    def test_variable_set_twice_keeps_one_line_and_one_pair(self, site):
        """A variable set twice in one run keeps its last value, once in the file and in conda.

        This is the Julia 1.11 fall-back, which sets PYTHON_JULIAPKG_EXE again in phase 7.
        """
        script = (
            _set_script(_PLAIN)
            + "set_install_var PYTHON_JULIAPKG_EXE '/opt/julia-1.11.9/bin/julia'\n"
            + 'bind_install_vars\n'
        )
        result = self._call(site, script, CONDA_PREFIX=site['prefix'])
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(site['env_file'].read_text().splitlines()) == 4
        assert (
            _sourced(site['env_file'])['PYTHON_JULIAPKG_EXE'] == '/opt/julia-1.11.9/bin/julia'
        )
        (call,) = _conda_calls(site['log'])
        exe_pairs = [a for a in call if a.startswith('PYTHON_JULIAPKG_EXE=')]
        assert exe_pairs == ['PYTHON_JULIAPKG_EXE=/opt/julia-1.11.9/bin/julia']
        assert len(call) == 6 + 4

    @pytest.mark.parametrize(
        'env',
        [{}, {'CONDA_PREFIX': '/envs/default', 'PIXI_ENVIRONMENT_NAME': 'default'}],
        ids=['no-conda-env', 'pixi-env'],
    )
    def test_without_conda_env_only_the_exports_file(self, site, env):
        """Without a conda env, or in a pixi env, conda is not called and the message sources the file."""
        script = _set_script(_VALUES) + 'bind_install_vars\nprint_next_steps\n'
        result = self._call(site, script, **env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _conda_calls(site['log']) == []
        assert not site['log'].exists()
        assert _sourced(site['env_file']) == _VALUES
        assert 'source .proteus-env.sh' in result.stdout
        assert 'conda activate' not in result.stdout

    def test_failed_conda_call_warns_once_and_names_the_exports_file(self, site):
        """A failing conda call warns once, names the exports file, and the run goes on."""
        script = _set_script(_VALUES) + 'bind_install_vars\nprint_next_steps\n'
        result = self._call(site, script, CONDA_PREFIX=site['prefix'], STUB_CONDA_RC='1')
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(_conda_calls(site['log'])) == 1
        assert result.stderr.count('Could not bind') == 1
        assert str(site['env_file']) in result.stderr
        assert 'source .proteus-env.sh' in result.stdout
        assert 'conda deactivate' not in result.stdout

    def test_failed_run_names_the_exports_file_on_exit(self, site):
        """A run that stops after a variable was set names the exports file; a clean exit does not."""
        failed = self._call(site, 'set_install_var FWL_DATA /d\nexit 3\n')
        assert failed.returncode == 3
        assert f'source {site["env_file"]}' in failed.stdout
        clean = self._call(site, 'set_install_var FWL_DATA /d\n')
        assert clean.returncode == 0
        assert 'NOTE' not in clean.stdout

    def test_shell_start_files_keep_bytes_mode_and_symlink(self, site):
        """The whole variable flow leaves ~/.zshrc and a symlinked ~/.bashrc as they were.

        Both hold an old `export FWL_DATA=/old` line, which a writer of start
        files would delete, and mode 600, which a rewrite through a temporary
        file turns into 644.
        """
        home = site['home']
        _write_start_file(home / '.zshrc', '# zsh\nexport FWL_DATA=/old\nalias ll="ls -l"\n')
        (home / '.zshrc').chmod(0o600)
        real = home / 'dotfiles' / 'bashrc'
        real.parent.mkdir()
        real.write_text('# bash\nexport FWL_DATA=/old\nexport PATH="$HOME/bin:$PATH"\n')
        real.chmod(0o600)
        try:
            (home / '.bashrc').symlink_to(real)
        except PermissionError:
            pytest.skip(
                'this host refuses to create shell start files, even under a temporary HOME'
            )
        before = _start_file_state(home)
        real_before = real.read_bytes()
        script = (
            _set_script(_VALUES)
            + 'bind_install_vars\nnote_old_shell_exports\nprint_next_steps\n'
        )
        result = self._call(site, script, CONDA_PREFIX=site['prefix'])
        assert result.returncode == 0, result.stdout + result.stderr
        assert _start_file_state(home) == before
        assert set(before) == {'.zshrc', '.bashrc'}
        assert before['.bashrc'][3] == str(real)
        assert real.read_bytes() == real_before
        assert sorted(p.name for p in home.iterdir()) == ['.bashrc', '.zshrc', 'dotfiles']

    def test_note_names_file_and_line_of_an_old_export(self, site):
        """The note names the file and line number of an old export and leaves the file alone.

        `export FWL_DATAX=` and an indented export are not install lines, so
        only line 3 is named; a note that counted from 0 would say line 2.
        """
        zshrc = site['home'] / '.zshrc'
        text = '# zsh\nexport FWL_DATAX=/x\nexport FWL_DATA=/old\n  export RAD_DIR=/r\n'
        _write_start_file(zshrc, text)
        result = self._call(site, 'note_old_shell_exports\n')
        assert result.returncode == 0, result.stdout + result.stderr
        assert f'{zshrc}, line 3: export FWL_DATA=/old' in result.stderr
        assert 'FWL_DATAX' not in result.stderr
        assert 'RAD_DIR=/r' not in result.stderr
        assert 'delete those lines by hand' in result.stderr
        assert zshrc.read_text() == text

    def test_note_names_a_depot_export(self, site):
        """The way back of a kept value starts at this line of the start file."""
        zshrc = site['home'] / '.zshrc'
        _write_start_file(zshrc, '# zsh\nexport JULIA_DEPOT_PATH=/data/depot\n')
        result = self._call(site, 'note_old_shell_exports\n')
        assert f'{zshrc}, line 2: export JULIA_DEPOT_PATH=/data/depot' in result.stderr

    def test_note_reads_the_fish_ksh_bash_login_and_zshenv_files(self, site):
        """An old export in fish's config.fish, ~/.kshrc, ~/.bash_login or ~/.zshenv is named.

        Each file holds its export on another line (1 to 4), so a note that names a
        line of the wrong file fails, and the files keep their bytes.
        """
        files = {
            '.config/fish/config.fish': 'export FC_DIR=/old/fastchem',
            '.kshrc': 'export RAD_DIR=/old/socrates',
            '.bash_login': 'export PYTHON_JULIAPKG_EXE=/old/julia',
            '.zshenv': 'export FWL_DATA=/old/data',
        }
        texts = {}
        for n, (name, line) in enumerate(files.items()):
            path = site['home'] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            texts[path] = '# comment\n' * n + line + '\n'
            _write_start_file(path, texts[path])
        result = self._call(site, 'note_old_shell_exports\n')
        assert result.returncode == 0, result.stdout + result.stderr
        named = [ln.strip() for ln in result.stderr.splitlines() if ', line ' in ln]
        expected = [
            f'{site["home"] / name}, line {n + 1}: {line}'
            for n, (name, line) in enumerate(files.items())
        ]
        assert sorted(named) == sorted(expected)
        assert all(path.read_text() == text for path, text in texts.items())

    def test_no_note_without_old_exports(self, site):
        """A HOME without shell start files gets no note."""
        result = self._call(site, 'note_old_shell_exports\n')
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stderr == ''
        assert result.stdout == ''

    @pytest.mark.parametrize(
        ('program', 'recorded'),
        [('built', True), ('missing', False), ('not-executable', False)],
        ids=['fastchem-built', 'folder-without-program', 'program-not-executable'],
    )
    def test_fc_dir_is_recorded_only_for_a_built_fastchem(self, site, program, recorded):
        """Phase 5 records FC_DIR when AGNI/fastchem/fastchem is executable, not for the folder alone.

        A failed build leaves the folder (with a clone and no program, or a
        program without its mode); FC_DIR at it would point AGNI at no solver.
        """
        fastchem = site['root'] / 'AGNI' / 'fastchem'
        fastchem.mkdir(parents=True)
        if program != 'missing':
            _stub(fastchem / 'fastchem', '#!/bin/sh\n')
        if program == 'not-executable':
            (fastchem / 'fastchem').chmod(0o644)
        block = _install_sh_blocks('# FC_DIR only')
        assert block.count('set_install_var FC_DIR') == 1
        script = f'AGNI_REAL={shlex.quote(str(site["root"] / "AGNI"))}\n' + block
        result = self._call(site, script)
        assert result.returncode == 0, result.stdout + result.stderr
        lines = site['env_file'].read_text().splitlines() if site['env_file'].exists() else []
        assert lines == ([f"export FC_DIR='{fastchem}'"] if recorded else [])
        assert ('Set FC_DIR=' in result.stdout) is recorded

    def test_variables_are_bound_once_after_the_import_check_and_before_the_data(self):
        """bind_install_vars runs once, right after the import check of Phase 7, before Phase 8.

        The Julia 1.11 fall-back in verify_proteus_import sets the last variable;
        the data downloads of Phase 8 take long and can be interrupted, which
        would leave a complete install whose new shell has no variable. Each
        anchor is asserted to be in install.sh once.
        """
        lines = INSTALL_SH.read_text().splitlines()

        def at(text: str) -> int:
            hits = [i for i, ln in enumerate(lines) if ln == text]
            assert len(hits) == 1, (text, hits)
            return hits[0]

        bind = at('bind_install_vars')
        assert at('verify_proteus_import') < at('info "Python packages installed"') < bind
        code = [ln for ln in lines[at('verify_proteus_import') : bind] if ln.strip()]
        assert [ln for ln in code if not ln.startswith('#')] == [
            'verify_proteus_import',
            'info "Python packages installed"',
        ]
        assert bind < at('CURRENT_PHASE=8') < at('case "$DATA_MODE" in')
        assert at('CURRENT_PHASE=9') < at('proteus doctor 2>&1 || true')
        assert at('print_next_steps') > at('proteus doctor 2>&1 || true')


_START_FILE = re.compile(
    r'zshrc|bashrc|bash_profile|\.profile|zprofile|config/fish|RC_FILE|rc_file'
)
_FILE_WRITE = re.compile(r'(?<![0-9&])>(?!&)|\bmv\b|\bsed\s+-i|\btee\b|\btouch\b')


def _start_file_writes(text: str) -> list[str]:
    """Uncommented lines that name a shell start file and write, move or touch a file."""
    return [
        ln
        for ln in text.splitlines()
        if not ln.lstrip().startswith('#') and _START_FILE.search(ln) and _FILE_WRITE.search(ln)
    ]


class TestNoShellFileEdit:
    """install.sh edits no shell start file and keeps AGNI's FastChem script away from one."""

    def test_no_write_to_a_shell_start_file(self):
        """No line naming a start file writes it, and the reader of those files writes nothing.

        The control is a writer of start files: its 3 file operations are
        flagged and its stderr warning is not, so the empty result for
        install.sh is not a scanner that finds nothing.
        """
        text = INSTALL_SH.read_text()
        assert _start_file_writes(text) == []
        assert 'append_export_to_rc' not in text
        assert 'detect_shell_rc' not in text
        reader = _install_sh_function('note_old_shell_exports')
        assert '.zshrc' in reader
        assert [ln for ln in reader.splitlines() if _FILE_WRITE.search(ln)] == []
        writer = (
            '    grep -v "^export ${var_name}=" "$rc_file" > "${rc_file}.tmp" 2>/dev/null || true\n'
            '    mv "${rc_file}.tmp" "$rc_file"\n'
            '    echo "$line" >> "$rc_file"\n'
            '    warn "Shell config: $RC_FILE" >&2\n'
        )
        assert len(_start_file_writes(writer)) == 3

    def test_fastchem_runs_without_yes_and_without_input(self):
        """get_fastchem.sh runs once, without -y, without FC_DIR and with stdin from /dev/null.

        With -y it answers its own question to add FC_DIR to the shell start file.
        """
        calls = [
            ln.strip()
            for ln in INSTALL_SH.read_text().splitlines()
            if 'get_fastchem.sh' in ln and not ln.lstrip().startswith(('#', 'warn', 'echo'))
        ]
        assert calls == ['env -u FC_DIR bash src/get_fastchem.sh </dev/null || true']
        assert '-y' not in calls[0].split()
        assert calls[0] in _install_sh_function('build_fastchem')
        assert 'run_logged "Building FastChem" build_fastchem || true' in INSTALL_SH.read_text()

    @pytest.mark.parametrize(
        ('builds', 'status', 'expected'),
        [(True, 1, 0), (True, 0, 0), (False, 0, 1), (False, 1, 1)],
        ids=[
            'built, script exits 1',
            'built, script exits 0',
            'not built',
            'not built, exits 1',
        ],
    )
    def test_fastchem_result_is_the_built_program(self, tmp_path, builds, status, expected):
        """build_fastchem succeeds exactly when the program exists, whatever the script returns.

        AGNI's script exits 1 after a good build when it wrote no shell start file. The stub
        records FC_DIR and what it reads from standard input: unset, and nothing.
        """
        agni = tmp_path / 'AGNI'
        (agni / 'src').mkdir(parents=True)
        made = 'mkdir -p fastchem && printf "#!/bin/sh\\n" > fastchem/fastchem && chmod +x fastchem/fastchem'
        (agni / 'src' / 'get_fastchem.sh').write_text(
            '#!/bin/bash\n'
            'echo "FC_DIR=${FC_DIR-unset}" > seen.txt\n'
            'read -r answer; echo "stdin=[$answer]" >> seen.txt\n'
            + (made + '\n' if builds else '')
            + f'exit {status}\n'
        )
        script = (
            _install_sh_function('build_fastchem')
            + f'cd {shlex.quote(str(agni))}\nbuild_fastchem\n'
        )
        result = subprocess.run(
            ['bash', '-euo', 'pipefail', '-c', script],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, 'FC_DIR': '/an/old/value'},
            input='y\n',
        )
        assert result.returncode == expected, result.stderr
        assert (agni / 'seen.txt').read_text().splitlines() == ['FC_DIR=unset', 'stdin=[]']


class TestDiskSpaceCheck:
    """Verify disk space detection."""

    def test_available_disk_gb_returns_positive_integer(self):
        """POSIX df -k based disk check returns a positive integer.

        Discrimination: the result must be > 0; a broken command would
        return 0 or fail. We test the actual command used by install.sh.
        """
        result = subprocess.run(
            ['bash', '-c', 'df -k . | awk \'NR==2 {printf "%d\\n", $4/1024/1024}\''],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        gb = int(result.stdout.strip())
        assert gb > 0, f'Disk space check returned {gb} GB'


class TestNonInteractive:
    """Verify non-interactive mode behavior."""

    def test_default_non_interactive_does_not_block_on_stdin(self):
        """Default mode (non-interactive) does not read from stdin.

        Passes /dev/null as stdin, with no conda or pixi env inherited from the
        test runner: the installer reaches the conda step and stops there, where
        the conda stub of this file names no envs directory. An inherited pixi
        env would let it pass that step and go on to the Julia step with the
        real HOME, so the exact stop line is asserted, not any line with conda.
        """
        if not _process_substitution_writable():
            pytest.skip(
                'bash cannot write to a process substitution on this host (/dev/fd refused)'
            )
        env = os.environ.copy()
        for var in ('CONDA_DEFAULT_ENV', 'CONDA_PREFIX', 'PIXI_ENVIRONMENT_NAME'):
            env.pop(var, None)
        result = subprocess.run(
            ['bash', str(INSTALL_SH)],
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=str(PROTEUS_ROOT),
        )
        assert result.returncode == 1
        output = result.stdout + result.stderr
        assert "'conda config --show envs_dirs' named no envs directory." in output
        assert 'Installation failed at Phase 1.' in output
        assert 'Phase 2' not in output


def _module_repo(path: Path, tag: str | None = '26.10.06') -> tuple[str, str]:
    """Build a repository with 3 commits and ``tag`` on the second one.

    Returns the tagged commit and the tip, which differ, so a checkout that
    stays on the branch is told apart from one at the tag.
    """
    path.mkdir(parents=True)
    git(path, 'init', '-q', '-b', 'main')
    for i in range(3):
        (path / 'version.txt').write_text(f'{i}\n')
        git(path, 'add', 'version.txt')
        git(path, 'commit', '-q', '-m', f'commit {i}')
        if i == 1 and tag:
            git(path, 'tag', tag)
    tagged = git(path, 'rev-parse', 'HEAD~1')
    return tagged, git(path, 'rev-parse', 'HEAD')


def _run_checkout_floor_tag(
    root: Path, repo: Path, prelude: str, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Source the copied helper library and run checkout_floor_tag on ``repo`` for fwl-mors."""
    return subprocess.run(
        [
            '/bin/bash',
            '-c',
            f'{prelude}source "{root}/tools/_get_common.sh"\n'
            f'checkout_floor_tag fwl-mors "{repo}"\necho AFTER\n',
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )


def _path_without_python(tmp_path: Path) -> str:
    """A PATH directory holding git and realpath but no python, as a shell outside conda sees."""
    bindir = tmp_path / 'no-python-bin'
    bindir.mkdir()
    for tool in ('git', 'realpath'):
        found = shutil.which(tool)
        if found:
            (bindir / tool).symlink_to(found)
    return str(bindir)


class TestCheckoutFloorTag:
    """checkout_floor_tag puts a clone on the release tag of its floor, or stops."""

    @pytest.mark.parametrize('prelude', ['', 'set -euo pipefail\n'], ids=['plain', 'errexit'])
    def test_checks_out_the_tag_matching_the_floor(self, tmp_path, prelude):
        """The floor 26.10.6 selects the tag 26.10.06 on the second commit, detached.

        The tip is a later commit, so a helper that leaves the clone on its
        branch fails the equality; the floor text is not a tag, so an exact
        text lookup fails the checkout. Branches named 26.10.06 and
        tags/26.10.06 point at the tip, so a checkout of ``tags/26.10.06``
        switches to the second and lands there.
        """
        repo = tmp_path / 'MORS'
        tagged, tip = _module_repo(repo)
        git(repo, 'branch', '26.10.06', tip)
        git(repo, 'branch', 'tags/26.10.06', tip)
        root = scratch_proteus_root(tmp_path / 'proteus', ['fwl-mors>=26.10.6'])
        result = _run_checkout_floor_tag(root, repo, prelude)
        assert result.returncode == 0, result.stderr
        head = git(repo, 'rev-parse', 'HEAD')
        assert head == tagged
        assert head != tip
        detached = subprocess.run(
            ['git', '-C', str(repo), 'symbolic-ref', '-q', 'HEAD'],
            capture_output=True,
            timeout=10,
        )
        assert detached.returncode == 1
        assert f'fwl-mors: checked out release tag 26.10.06 at {tagged[:7]}' in result.stdout
        assert result.stdout.rstrip().endswith('AFTER')

    @pytest.mark.parametrize('prelude', ['', 'set -euo pipefail\n'], ids=['plain', 'errexit'])
    def test_no_matching_tag_exits_1_and_keeps_head(self, tmp_path, prelude):
        """A floor with no tag stops the caller with the cause, and HEAD stays put.

        ``AFTER`` must not print: the helper exits rather than returns, in a
        shell with or without errexit, so no caller carries on at the tip.
        """
        repo = tmp_path / 'MORS'
        _, tip = _module_repo(repo)
        root = scratch_proteus_root(tmp_path / 'proteus', ['fwl-mors>=26.11.1'])
        result = _run_checkout_floor_tag(root, repo, prelude)
        assert result.returncode == 1
        assert 'AFTER' not in result.stdout
        assert git(repo, 'rev-parse', 'HEAD') == tip
        assert "'fwl-mors' floor 26.11.1: no tag matches it" in result.stderr
        assert '26.10.06' in result.stderr
        assert f'cannot choose the fwl-mors release tag in {repo}; see the message above' in (
            result.stderr
        )

    @pytest.mark.parametrize(
        ('requirement', 'extra_tag', 'cause'),
        [
            ('fwl-mors>=26.10.6', '26.10.6', '2 tags match it: 26.10.06, 26.10.6'),
            ('fwl-mors==26.10.6', None, 'has no single >= version floor'),
        ],
        ids=['two tags of the floor', 'requirement without a floor'],
    )
    def test_pin_failure_keeps_the_pin_reader_cause(
        self, tmp_path, requirement, extra_tag, cause
    ):
        """When the pin reader fails, its cause stands alone and is not contradicted.

        Two tags of one version, or an exact pin, are not cases of a missing
        tag, so the line that follows only points at the cause; HEAD stays
        at the tip and the caller stops.
        """
        repo = tmp_path / 'MORS'
        tagged, tip = _module_repo(repo)
        if extra_tag:
            git(repo, 'tag', extra_tag, tagged)
        root = scratch_proteus_root(tmp_path / 'proteus', [requirement])
        result = _run_checkout_floor_tag(root, repo, 'set -euo pipefail\n')
        assert result.returncode == 1
        assert 'AFTER' not in result.stdout
        assert cause in result.stderr
        assert f'cannot choose the fwl-mors release tag in {repo}; see the message above' in (
            result.stderr
        )
        assert 'no release tag' not in result.stderr
        assert git(repo, 'rev-parse', 'HEAD') == tip

    def test_missing_python_is_named_before_git_runs(self, tmp_path):
        """Without python on PATH the helper names python, not the tags.

        HEAD stays at the tip; a helper that pipes the tags into a missing
        python reports a tag that does not match.
        """
        repo = tmp_path / 'MORS'
        _, tip = _module_repo(repo)
        root = scratch_proteus_root(tmp_path / 'proteus', ['fwl-mors>=26.10.6'])
        env = {**os.environ, 'PATH': _path_without_python(tmp_path)}
        result = _run_checkout_floor_tag(root, repo, '', env=env)
        assert result.returncode == 1
        assert 'AFTER' not in result.stdout
        assert 'python is not on PATH, so the fwl-mors floor in pyproject.toml' in result.stderr
        assert 'tag' not in result.stderr
        assert git(repo, 'rev-parse', 'HEAD') == tip

    def test_directory_that_is_no_repository_is_reported_as_such(self, tmp_path):
        """A directory git does not read as a repository fails at the tag listing.

        git's own message and the listing step are named; a helper that
        pipes the failed listing on reports that no release tag was seen.
        """
        plain = tmp_path / 'MORS'
        plain.mkdir()
        root = scratch_proteus_root(tmp_path / 'proteus', ['fwl-mors>=26.10.6'])
        env = {**os.environ, 'GIT_CEILING_DIRECTORIES': str(tmp_path)}
        result = _run_checkout_floor_tag(root, plain, 'set -euo pipefail\n', env=env)
        assert result.returncode == 1
        assert 'AFTER' not in result.stdout
        assert 'not a git repository' in result.stderr
        assert (
            f'git cannot list the tags of {plain}; see the git message above' in result.stderr
        )
        assert 'saw no release tags' not in result.stderr
        assert list(plain.iterdir()) == []

    def test_moving_a_detached_local_commit_shows_the_git_warning(self, tmp_path):
        """A detached clone with a local commit lands on the tag, and git's warning
        that the move leaves 1 commit behind reaches the user.

        ``--quiet`` hides that warning, so the commit would be left without
        a word; the tag checkout itself still succeeds.
        """
        repo = tmp_path / 'MORS'
        tagged, _ = _module_repo(repo)
        git(repo, 'checkout', '-q', '--detach', 'HEAD~2')
        (repo / 'version.txt').write_text('local work\n')
        git(repo, 'commit', '-q', '-am', 'local work')
        local = git(repo, 'rev-parse', 'HEAD')
        root = scratch_proteus_root(tmp_path / 'proteus', ['fwl-mors>=26.10.6'])
        result = _run_checkout_floor_tag(root, repo, 'set -euo pipefail\n')
        assert result.returncode == 0, result.stderr
        assert git(repo, 'rev-parse', 'HEAD') == tagged != local
        assert 'leaving 1 commit behind' in result.stderr
        assert local[:7] in result.stderr
        assert result.stdout.rstrip().endswith('AFTER')


def _install_sh_function(name: str) -> str:
    """The text of a shell function in install.sh, a one-line one or a block to its ``}``."""
    lines = INSTALL_SH.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f'{name}()'))
    if lines[start].rstrip().endswith('}'):
        return lines[start] + '\n'
    end = next(i for i, ln in enumerate(lines) if ln == '}' and i > start)
    return '\n'.join(lines[start : end + 1]) + '\n'


# run_logged and the helpers it calls; the colour codes and info/warn/fail come with "info()".
_RUN_LOGGED_FUNCTIONS = ('log_file_sentence', 'log_catch_up', 'to_terminal', 'run_logged')

# The exit trap that removes the temp file of run_logged where print_exit_hint is not lifted.
_RUN_LOGGED_CLEANUP = 'trap \'rm -f "${RUN_LOGGED_TMP:-}"\' EXIT\n'

# Shell variables of the installer's log state, kept out of the test environment.
_LOG_STATE = ('LOGFILE', 'VERBOSE', 'LOG_TEE', 'RUN_LOGGED_TOLD', 'LOG_SENTENCE_SHOWN')


def _env_without_log_state() -> dict[str, str]:
    """The test process environment without the installer's log variables."""
    return {k: v for k, v in os.environ.items() if k not in _LOG_STATE}


def _clone_and_install_calls() -> list[list[str]]:
    """The arguments of every clone_and_install call in install.sh."""
    return [
        shlex.split(line)[1:]
        for line in INSTALL_SH.read_text().splitlines()
        if line.startswith('clone_and_install ')
    ]


# The prefix install.sh's warn puts before a message.
_WARN_PREFIX = '\x1b[1;33m[WARN]\x1b[0m  '


def _run_module_phase(
    env: dict, script: str, cwd: Path | None = None, timeout: int = 25
) -> subprocess.CompletedProcess:
    """Run ``script`` after install.sh's log helpers and its module-phase functions, under its options."""
    defs = (
        'set -euo pipefail\nCURRENT_PHASE=6\n'
        + _install_sh_blocks("RED='", 'info()')
        + ''.join(
            _install_sh_function(f)
            for f in (
                'command_exists',
                'collect_env_info',
                'die',
                *_RUN_LOGGED_FUNCTIONS,
                'clone_and_install',
                'report_stayed_clones',
            )
        )
        + _RUN_LOGGED_CLEANUP
    )
    return subprocess.run(
        ['bash', '-c', defs + script],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=timeout,
    )


class TestCloneAndInstall:
    """clone_and_install pins a new clone to the floor tag and installs an existing one as it is."""

    @pytest.fixture
    def site(self, tmp_path):
        """A PROTEUS copy, local bare remotes for MORS and JANUS, and a pip stub.

        JANUS carries only the tag 26.07.10, below its floor 26.10.08.
        """
        root = scratch_proteus_root(
            tmp_path / 'proteus', ['fwl-mors>=26.10.6', 'fwl-janus>=26.10.08']
        )
        remotes = tmp_path / 'remotes' / 'FormingWorlds'
        shas = {}
        for repo, tag in (('MORS', '26.10.06'), ('JANUS', '26.07.10')):
            work = tmp_path / 'work' / repo
            shas[repo] = _module_repo(work, tag)
            git(tmp_path, 'clone', '-q', '--bare', str(work), str(remotes / f'{repo}.git'))
        stubs = tmp_path / 'stubs'
        stubs.mkdir()
        pip = stubs / 'pip'
        pip.write_text('#!/bin/sh\necho "$@" >> "$STUB_PIP_LOG"\n')
        pip.chmod(0o755)
        log = tmp_path / 'pip.log'
        log.write_text('')
        env = {
            **_env_without_log_state(),
            'PATH': f'{stubs}{os.pathsep}{os.environ["PATH"]}',
            'GH_PREFIX': f'{tmp_path / "remotes"}/',
            'SCRIPT_DIR': str(root),
            'STUB_PIP_LOG': str(log),
            # A github URL, from any version of the scripts, goes to the local remotes.
            'GIT_CONFIG_COUNT': '1',
            'GIT_CONFIG_KEY_0': f'url.{tmp_path / "remotes"}/.insteadOf',
            'GIT_CONFIG_VALUE_0': 'https://github.com/',
        }
        env.pop('PROTEUS_CLONE_STAYS', None)
        # The script names the clone below the resolved PROTEUS root.
        clone = Path(os.path.realpath(root)) / 'MORS'
        return {
            'root': root,
            'shas': shas,
            'log': log,
            'env': env,
            'clone': clone,
            'tmp': tmp_path,
        }

    @staticmethod
    def _call(site, *args: str, **env: str) -> subprocess.CompletedProcess:
        """Run the shipped clone_and_install with ``args``, under install.sh's options."""
        return _run_module_phase(
            {**site['env'], **env}, shlex.join(['clone_and_install', *args])
        )

    def test_new_clone_is_installed_at_the_floor_tag(self, site):
        """A new clone sits at the tagged commit, detached, and is installed editable.

        The tip is one commit past the tag, so a clone left on the default
        branch fails the first equality.
        """
        tagged, tip = site['shas']['MORS']
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        clone = site['root'] / 'MORS'
        assert git(clone, 'rev-parse', 'HEAD') == tagged != tip
        detached = subprocess.run(
            ['git', '-C', str(clone), 'symbolic-ref', '-q', 'HEAD'],
            capture_output=True,
            timeout=10,
        )
        assert detached.returncode == 1
        assert site['log'].read_text().splitlines() == [
            f'install -U -e {os.path.realpath(clone)}'
        ]

    @staticmethod
    def _detached_at_old_tag(site) -> tuple[Path, str]:
        """Clone MORS into the root, detached at the tag 26.10.05 on the first commit.

        The tag goes into the remote first, so the clone carries it as the
        installer's clone of an older floor does. Returns the clone and the
        first commit.
        """
        tagged, _ = site['shas']['MORS']
        remote = site['root'].parent / 'remotes' / 'FormingWorlds' / 'MORS.git'
        first = git(remote, 'rev-parse', f'{tagged}~1')
        git(remote, 'tag', '26.10.05', first)
        clone = site['root'] / 'MORS'
        git(site['root'], 'clone', '-q', str(remote), str(clone))
        git(clone, 'checkout', '-q', 'tags/26.10.05')
        return clone, first

    def test_branch_clone_is_left_on_its_commit(self, site):
        """A second run reports a clone on a branch behind the tag, with the commands that move
        it by hand, and installs it unchanged.

        The clone is put on a new branch at the commit below the tag first, as a
        developer would; a function that checks out the floor tag again moves it.
        """
        tagged, _ = site['shas']['MORS']
        clone, real = site['root'] / 'MORS', site['clone']
        assert self._call(site, 'MORS', 'fwl-mors').returncode == 0
        git(clone, 'checkout', '-q', '-b', 'old', 'HEAD~1')
        below = git(clone, 'rev-parse', 'HEAD')
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == below != tagged
        lines = result.stdout.splitlines()
        line = (
            f'fwl-mors: {real} stays as it is: on branch old, behind the tag;'
            ' PROTEUS expects the release tag 26.10.06'
        )
        assert lines.count(line) == 1
        assert lines[lines.index(line) + 3] == (
            f'        git -C {real} checkout --detach refs/tags/26.10.06'
        )
        assert 'checked out release tag' not in result.stdout
        assert 'is at the release tag' not in result.stdout
        assert site['log'].read_text().splitlines()[1:] == [f'install -U -e {real}']

    def test_logged_stay_line_and_its_commands_reach_the_terminal(self, site):
        """With a log file, the stay line of a clone behind its tag and the 4 command lines
        below it stay on the terminal, each once, and the line of the setup script does not."""
        real = site['clone']
        self._detached_at_old_tag(site)
        logfile = site['tmp'] / 'install.log'
        result = self._call(site, 'MORS', 'fwl-mors', LOGFILE=str(logfile))
        assert result.returncode == 0, result.stdout + result.stderr
        shown = result.stdout.splitlines()
        stay = [ln for ln in shown if 'stays as it is' in ln]
        assert len(stay) == 1 and 'behind the tag' in stay[0], shown
        assert stay[0].endswith('PROTEUS expects the release tag 26.10.06')
        for line in (
            f'        git -C {real} fetch --tags',
            f'        git -C {real} checkout --detach refs/tags/26.10.06',
            f'        pip install -e {real}',
            '        bash tools/get_python_module.sh MORS --force',
        ):
            assert shown.count(line) == 1, line
        assert 'Set up MORS...' not in result.stdout
        assert 'Set up MORS...' in logfile.read_text()

    def test_logged_newer_clone_line_reaches_the_terminal(self, site):
        """With a log file, the one line of a clone that is newer than its tag stays on the
        terminal, and no command line follows it."""
        clone, real = site['root'] / 'MORS', site['clone']
        assert self._call(site, 'MORS', 'fwl-mors').returncode == 0
        git(clone, 'checkout', '-q', 'main')
        result = self._call(site, 'MORS', 'fwl-mors', LOGFILE=str(site['tmp'] / 'install.log'))
        assert result.returncode == 0, result.stdout + result.stderr
        shown = result.stdout.splitlines()
        line = f'fwl-mors: {real} is newer than the release tag 26.10.06 (branch main)'
        assert shown.count(line) == 1, shown
        assert not any('git -C' in ln or '--force' in ln for ln in shown)

    def test_detached_clone_at_an_older_tag_stays_where_it_is(self, site):
        """A clone the installer left at the tag of an older floor (26.10.05) stays there, with
        its untracked file, and is named as behind the tag 26.10.06; nothing is cloned.

        A function that moves the clone lands on the tag, and one that pulls
        lands on the tip.
        """
        tagged, tip = site['shas']['MORS']
        clone, first = self._detached_at_old_tag(site)
        (clone / 'notes.txt').write_text('kept\n')
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == first
        assert first not in (tagged, tip)
        assert (clone / 'notes.txt').read_text() == 'kept\n'
        assert (
            f'fwl-mors: {site["clone"]} stays as it is: commit {first[:7]} (26.10.05), behind the'
            ' tag; PROTEUS expects the release tag 26.10.06'
        ) in result.stdout.splitlines()
        assert 'checked out release tag' not in result.stdout
        assert 'Cloning' not in result.stdout
        assert site['log'].read_text().splitlines() == [f'install -U -e {site["clone"]}']

    def test_detached_clone_at_the_floor_tag_is_not_cloned_again(self, site):
        """A clone already at the floor tag is installed in place: an untracked marker
        file survives, so no new clone replaced it."""
        tagged, _ = site['shas']['MORS']
        clone, real = site['root'] / 'MORS', site['clone']
        assert self._call(site, 'MORS', 'fwl-mors').returncode == 0
        (clone / 'marker.txt').write_text('kept\n')
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        assert (clone / 'marker.txt').read_text() == 'kept\n'
        assert git(clone, 'rev-parse', 'HEAD') == tagged
        assert f'fwl-mors: {real} is at the release tag 26.10.06' in result.stdout
        assert 'stays as it is' not in result.stdout
        assert 'Cloning' not in result.stdout
        assert site['log'].read_text().splitlines()[1:] == [f'install -U -e {real}']

    def test_detached_clone_ahead_of_the_floor_tag_is_in_order(self, site):
        """A detached clone that the user put on the tip, past the floor tag, satisfies the >=
        floor: it stays there with one line that says it is newer, without hand commands, and
        is installed; the installer does not move it back."""
        tagged, tip = site['shas']['MORS']
        clone, _ = self._detached_at_old_tag(site)
        git(clone, 'checkout', '-q', '--detach', tip)
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == tip != tagged
        line = f'fwl-mors: {site["clone"]} is newer than the release tag 26.10.06 (commit {tip[:7]})'
        assert result.stdout.splitlines().count(line) == 1
        assert 'stays as it is' not in result.stdout
        assert 'fetch --tags' not in result.stdout
        assert 'checked out release tag' not in result.stdout
        assert 'is at the release tag' not in result.stdout
        assert site['log'].read_text().splitlines() == [f'install -U -e {site["clone"]}']

    def test_clone_with_an_unreachable_remote_is_kept_without_a_fetch(self, site):
        """With the remote gone, a clone behind the tag is still kept and installed, and no
        fetch is tried: there is no fetch warning, and HEAD stays on its commit."""
        clone, first = self._detached_at_old_tag(site)
        remote = site['root'].parent / 'remotes' / 'FormingWorlds' / 'MORS.git'
        remote.rename(remote.with_name('MORS.moved'))
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'fetch' not in (result.stdout + result.stderr).replace('fetch --tags', '')
        assert 'behind the tag' in result.stdout
        assert git(clone, 'rev-parse', 'HEAD') == first
        assert site['log'].read_text().splitlines() == [f'install -U -e {site["clone"]}']

    def test_folder_that_is_not_a_git_clone_is_installed_as_it_is(self, site):
        """A MORS folder without .git is installed as it is and keeps its file; the installer
        never replaces an existing folder. The stay line names the release the floor 26.10.6
        of pyproject.toml asks for, as there is no clone to ask for tags."""
        plain = site['root'] / 'MORS'
        plain.mkdir()
        (plain / 'setup.py').write_text('# mine\n')
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        line = (
            f'fwl-mors: {site["clone"]} stays as it is: not a git clone;'
            ' PROTEUS expects release 26.10.6'
        )
        assert [ln for ln in result.stdout.splitlines() if 'stays as it is' in ln] == [line]
        assert 'release tag 26.10' not in result.stdout
        assert sorted(p.name for p in plain.iterdir()) == ['setup.py']
        assert site['log'].read_text().splitlines() == [f'install -U -e {site["clone"]}']

    def test_clone_git_cannot_read_is_installed_as_it_is(self, site):
        """A MORS folder whose .git git cannot read (rev-parse exits 128) is installed as it
        is with the stay line of an unreadable clone, and nothing is cloned over it."""
        broken = site['root'] / 'MORS'
        (broken / '.git').mkdir(parents=True)
        site['env']['GIT_CEILING_DIRECTORIES'] = str(site['root'].parent)
        rc = subprocess.run(
            ['git', '-C', str(broken), 'rev-parse', '--show-toplevel'],
            capture_output=True,
            env=site['env'],
            timeout=10,
        ).returncode
        assert rc == 128
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        line = (
            f'fwl-mors: {site["clone"]} stays as it is: not a readable git clone;'
            ' PROTEUS expects release 26.10.6'
        )
        assert [ln for ln in result.stdout.splitlines() if 'stays as it is' in ln] == [line]
        assert sorted(p.name for p in broken.iterdir()) == ['.git']
        assert site['log'].read_text().splitlines() == [f'install -U -e {site["clone"]}']

    def test_detached_clone_with_a_local_commit_is_kept(self, site):
        """A detached clone holding a commit that is on no remote is kept and installed: the
        commit stays checked out and is named on a line of history without the tag."""
        clone, _ = self._detached_at_old_tag(site)
        (clone / 'version.txt').write_text('local work\n')
        git(clone, 'commit', '-q', '-am', 'local work')
        local = git(clone, 'rev-parse', 'HEAD')
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        line = (
            f'fwl-mors: {site["clone"]} stays as it is: commit {local[:7]}'
            f' (26.10.05-1-g{local[:7]}), on a line of history without the tag;'
            ' PROTEUS expects the release tag 26.10.06'
        )
        assert result.stdout.splitlines().count(line) == 1
        assert git(clone, 'rev-parse', 'HEAD') == local
        assert (clone / 'version.txt').read_text() == 'local work\n'
        assert site['log'].read_text().splitlines() == [f'install -U -e {site["clone"]}']

    def test_logged_clone_shows_the_tag_line_and_logs_the_rest(self, site):
        """With a log file, the release tag line stays on the terminal and git's output does not.

        get_python_module.sh prints `Cloning <uri> -> <dir>` before the tag line;
        a helper that shows all output, or none, fails one of the two checks.
        """
        tagged, _ = site['shas']['MORS']
        logfile = site['tmp'] / 'install.log'
        result = self._call(site, 'MORS', 'fwl-mors', LOGFILE=str(logfile))
        assert result.returncode == 0, result.stdout + result.stderr
        terminal = result.stdout + result.stderr
        tag_line = f'fwl-mors: checked out release tag 26.10.06 at {tagged[:7]}'
        assert tag_line in result.stdout
        assert 'FormingWorlds/MORS.git ->' not in terminal
        assert re.search(r'Setting up MORS: done in \d+ s', result.stdout)
        logged = logfile.read_text().splitlines()
        assert logged.count(tag_line) == 1
        assert any('FormingWorlds/MORS.git ->' in ln for ln in logged)

    @pytest.mark.parametrize('logged', [False, True], ids=['no-log-file', 'log-file'])
    def test_floor_without_tag_stops_the_installer(self, site, logged):
        """A floor with no release tag fails, names the module, and installs nothing.

        The new clone is removed, so a re-run clones again rather than
        installing the branch tip as an existing clone. With a log file the
        cause still reaches the terminal, in the last lines of the failed step.
        """
        env = {'LOGFILE': str(site['tmp'] / 'install.log')} if logged else {}
        result = self._call(site, 'JANUS', 'fwl-janus', **env)
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert 'Could not install JANUS (fwl-janus); see the error above' in output
        assert "'fwl-janus' floor 26.10.08: no tag matches it" in output
        assert site['log'].read_text() == ''
        assert not (site['root'] / 'JANUS').exists()


_PAIRED_BRANCH = 'tl/paired-feature'


class TestModulePhaseSummary:
    """The Python module phase of install.sh writes nothing to an existing clone and names, in
    one line, every clone that is not at its release tag."""

    REPOS = ('MORS', 'JANUS', 'CALLIOPE', 'ZEPHYRUS', 'aragog', 'Zalmoxis')

    @pytest.fixture
    def site(self, tmp_path):
        """A PROTEUS copy with the six module scripts, a local remote per module with the tag
        26.10.06 on the second of 3 commits, stubs for pip and ssh, git URLs rewritten to the
        local remotes, and a git wrapper first on PATH that lets only read-only calls reach a
        module folder or the PROTEUS root."""
        repos = {r: f'fwl-{r.lower()}' for r in self.REPOS}
        root = scratch_proteus_root(
            tmp_path / 'proteus', [f'{d}>=26.10.6' for d in repos.values()]
        )
        for script in ('get_aragog.sh', 'get_zalmoxis.sh'):
            shutil.copy2(TOOLS_DIR / script, root / 'tools' / script)
        base = tmp_path / 'remotes'
        shas = {}
        for repo in self.REPOS:
            shas[repo] = _module_repo(tmp_path / 'work' / repo)
            git(
                tmp_path,
                'clone',
                '-q',
                '--bare',
                str(tmp_path / 'work' / repo),
                str(base / 'FormingWorlds' / f'{repo}.git'),
            )
        stubs = tmp_path / 'stubs'
        stubs.mkdir()
        for name, body in (('pip', 'echo "$@" >> "$STUB_PIP_LOG"'), ('ssh', 'exit 255')):
            (stubs / name).write_text(f'#!/bin/sh\n{body}\n')
            (stubs / name).chmod(0o755)
        guard = tmp_path / 'guard'
        guard.mkdir()
        calls, marker = tmp_path / 'git-calls.log', tmp_path / 'git-write.marker'
        install_git_guard(
            guard, calls, marker, roots=[root], trees=[root / r for r in self.REPOS]
        )
        (tmp_path / 'tmp').mkdir()
        env = {
            **os.environ,
            'PATH': f'{guard}{os.pathsep}{stubs}{os.pathsep}{os.environ["PATH"]}',
            'GH_PREFIX': f'{base}/',
            'SCRIPT_DIR': str(root),
            'STUB_PIP_LOG': str(tmp_path / 'pip.log'),
            'TMPDIR': str(tmp_path / 'tmp'),
            'GIT_CONFIG_COUNT': '1',
            'GIT_CONFIG_KEY_0': f'url.{base}/.insteadOf',
            'GIT_CONFIG_VALUE_0': 'https://github.com/',
            'GIT_CEILING_DIRECTORIES': str(tmp_path),
        }
        for name in (
            'GIT_SSH_COMMAND',
            'GITHUB_HEAD_REF',
            'GITHUB_REF_NAME',
            'PROTEUS_CLONE_STAYS',
        ):
            env.pop(name, None)
        return {
            'root': root,
            'real': Path(os.path.realpath(root)),
            'shas': shas,
            'env': env,
            'remotes': base / 'FormingWorlds',
            'calls': calls,
            'marker': marker,
        }

    @staticmethod
    def _clone(site, repo: str, *checkout: str) -> Path:
        """Clone ``repo`` from its remote into the root, then check out ``checkout`` if given."""
        clone = site['root'] / repo
        git(site['root'], 'clone', '-q', str(site['remotes'] / f'{repo}.git'), str(clone))
        if checkout:
            git(clone, 'checkout', '-q', *checkout)
        return clone

    @staticmethod
    def _phase_block() -> str:
        """The module lines of install.sh, from the summary file to the shipped summary call.

        Each anchor is found by its text and must be in install.sh once.
        """
        lines = INSTALL_SH.read_text().splitlines()
        starts = [i for i, ln in enumerate(lines) if ln.startswith('PROTEUS_CLONE_STAYS=$(')]
        ends = [i for i, ln in enumerate(lines) if ln == 'report_stayed_clones']
        assert len(starts) == 1 and len(ends) == 1, (starts, ends)
        assert starts[0] < ends[0]
        return '\n'.join(lines[starts[0] : ends[0] + 1]) + '\n'

    def _run_phase(self, site) -> subprocess.CompletedProcess:
        """Run the module lines of install.sh with the git wrapper; no write may reach a clone."""
        result = _run_module_phase(
            site['env'], self._phase_block(), cwd=site['root'], timeout=60
        )
        assert not site['marker'].exists(), site['calls'].read_text()
        return result

    def test_summary_names_exactly_the_clones_that_stay(self, site):
        """MORS on a branch below the tag, CALLIOPE as a plain folder and aragog one commit
        below the tag are named in the order the installer reached them, in one warning that
        says what the install does with them; JANUS, cloned new at the tag, Zalmoxis, already
        at it, and ZEPHYRUS, one commit past it, are not.

        Each clone that stays gets its stay line once and keeps its HEAD, ZEPHYRUS gets the
        line of a newer clone, and no line says that a clone was checked out at the release
        tag. The summary file is removed.
        """
        real = site['real']
        mors = self._clone(site, 'MORS', '-b', 'old', '26.10.06~1')
        (site['root'] / 'CALLIOPE').mkdir()
        (site['root'] / 'CALLIOPE' / 'setup.py').write_text('# mine\n')
        tip = site['shas']['ZEPHYRUS'][1]
        zephyrus = self._clone(site, 'ZEPHYRUS', '--detach', tip)
        below = git(site['remotes'] / 'aragog.git', 'rev-parse', '26.10.06~1')
        aragog = self._clone(site, 'aragog', '--detach', below)
        self._clone(site, 'Zalmoxis', 'tags/26.10.06')
        heads = {c: git(c, 'rev-parse', 'HEAD') for c in (mors, zephyrus, aragog)}

        result = self._run_phase(site)
        assert result.returncode == 0, result.stdout + result.stderr
        out = (result.stdout + result.stderr).splitlines()
        summary = [ln for ln in out if 'Not at their release tag' in ln]
        assert summary == [
            f'{_WARN_PREFIX}Not at their release tag: MORS (branch old, behind),'
            f' CALLIOPE (not a git clone), aragog (commit {below[:7]}, behind).'
            ' A clone that is older than the minimum version is not used: the PROTEUS install'
            ' takes that release from PyPI. The commands printed above move a clone to its tag'
            ' or replace it; a paired branch without the release has no command, its release'
            ' comes from PyPI.'
        ]
        expected = 'PROTEUS expects the release tag 26.10.06'
        for dist, repo, found in (
            ('fwl-mors', 'MORS', f'on branch old, behind the tag; {expected}'),
            ('fwl-calliope', 'CALLIOPE', 'not a git clone; PROTEUS expects release 26.10.6'),
            ('fwl-aragog', 'aragog', f'commit {below[:7]}, behind the tag; {expected}'),
        ):
            assert out.count(f'{dist}: {real / repo} stays as it is: {found}') == 1
            assert not [ln for ln in out if ln.startswith(f'{dist}: checked out release tag')]
        zephyrus_line = f'fwl-zephyrus: {real / "ZEPHYRUS"} is newer than the release tag'
        assert out.count(f'{zephyrus_line} 26.10.06 (commit {tip[:7]})') == 1
        assert not [ln for ln in out if ln.startswith('fwl-zephyrus: checked out release tag')]
        assert {c: git(c, 'rev-parse', 'HEAD') for c in heads} == heads
        assert any(ln.startswith('fwl-janus: checked out release tag 26.10.06') for ln in out)
        assert f'fwl-zalmoxis: {real / "Zalmoxis"} is at the release tag 26.10.06' in out
        assert list(Path(site['env']['TMPDIR']).iterdir()) == []

    def test_no_summary_when_every_clone_is_at_its_tag(self, site):
        """With every module already cloned at its tag, the phase prints the at-tag line 6
        times, no summary line and leaves no summary file; a summary printed without a check
        names an empty list."""
        for repo in self.REPOS:
            self._clone(site, repo, 'tags/26.10.06')
        result = self._run_phase(site)
        assert result.returncode == 0, result.stdout + result.stderr
        output = result.stdout + result.stderr
        assert 'Not at their release tag' not in output
        assert output.count('is at the release tag 26.10.06') == 6
        assert 'stays as it is' not in output
        assert list(Path(site['env']['TMPDIR']).iterdir()) == []

    def test_no_summary_when_every_clone_is_new(self, site):
        """With no module cloned yet, each is cloned detached at the tag of its floor and no
        summary line follows. Every remote also has the newer tag 26.11.01 on its tip, so a
        choice of the newest tag lands on the tip."""
        for repo in self.REPOS:
            git(site['remotes'] / f'{repo}.git', 'tag', '26.11.01', 'main')
        result = self._run_phase(site)
        assert result.returncode == 0, result.stdout + result.stderr
        output = result.stdout + result.stderr
        assert 'Not at their release tag' not in output
        assert output.count('checked out release tag 26.10.06') == 6
        heads = {repo: git(site['root'] / repo, 'rev-parse', 'HEAD') for repo in self.REPOS}
        assert heads == {repo: site['shas'][repo][0] for repo in self.REPOS}
        assert all(heads[repo] != site['shas'][repo][1] for repo in self.REPOS)
        assert list(Path(site['env']['TMPDIR']).iterdir()) == []

    def test_aragog_clone_on_a_paired_branch_is_not_named(self, site):
        """PROTEUS on a feature branch that the aragog remote has gives a new aragog clone on
        that branch, which is in order: no summary line, neither after the run that clones it
        nor after a second run, which prints the paired-branch line for it."""
        work = site['root'].parent / 'work' / 'aragog'
        git(work, 'checkout', '-q', '-b', _PAIRED_BRANCH)
        (work / 'version.txt').write_text('paired\n')
        git(work, 'commit', '-q', '-am', 'paired commit')
        git(work, 'push', '-q', str(site['remotes'] / 'aragog.git'), 'HEAD')
        for repo in ('MORS', 'JANUS', 'CALLIOPE', 'ZEPHYRUS', 'Zalmoxis'):
            self._clone(site, repo, 'tags/26.10.06')
        site['env']['GITHUB_HEAD_REF'] = _PAIRED_BRANCH
        result = self._run_phase(site)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'Not at their' not in result.stdout + result.stderr
        assert git(site['root'] / 'aragog', 'symbolic-ref', '--short', 'HEAD') == _PAIRED_BRANCH
        again = self._run_phase(site)
        assert again.returncode == 0, again.stdout + again.stderr
        assert 'Not at their' not in again.stdout + again.stderr
        assert (
            f'fwl-aragog: {site["real"] / "aragog"} is on the paired branch {_PAIRED_BRANCH},'
            ' as PROTEUS on that branch expects'
        ) in again.stdout.splitlines()
        assert 'stays as it is' not in again.stdout

    def test_aragog_clone_on_a_paired_branch_without_the_tag_is_named(self, site):
        """An aragog paired branch that starts below the tag commit, with a commit of its own,
        does not hold the release: on the run after the one that clones it, aragog gets the
        neutral line, the line that the PROTEUS install takes the release from PyPI and an
        entry in the summary, which the other 5 modules at their tag do not get. No hand
        command is printed for it and its HEAD stays on the branch."""
        work = site['root'].parent / 'work' / 'aragog'
        git(work, 'checkout', '-q', '-b', _PAIRED_BRANCH, '26.10.06~1')
        (work / 'version.txt').write_text('paired\n')
        git(work, 'commit', '-q', '-am', 'paired commit')
        git(work, 'push', '-q', str(site['remotes'] / 'aragog.git'), 'HEAD')
        paired = git(work, 'rev-parse', 'HEAD')
        for repo in ('MORS', 'JANUS', 'CALLIOPE', 'ZEPHYRUS', 'Zalmoxis'):
            self._clone(site, repo, 'tags/26.10.06')
        site['env']['GITHUB_HEAD_REF'] = _PAIRED_BRANCH
        assert self._run_phase(site).returncode == 0
        aragog = site['root'] / 'aragog'
        assert git(aragog, 'rev-parse', 'HEAD') == paired

        result = self._run_phase(site)
        assert result.returncode == 0, result.stdout + result.stderr
        out = (result.stdout + result.stderr).splitlines()
        neutral = (
            f'fwl-aragog: {site["real"] / "aragog"} is on the paired branch {_PAIRED_BRANCH},'
            ' as PROTEUS on that branch expects'
        )
        assert out[out.index(neutral) + 1] == (
            '    This branch does not hold the release 26.10.06: the PROTEUS install takes that'
            ' release from PyPI, not this folder.'
        )
        assert [ln for ln in out if 'Not at their release tag' in ln] == [
            f'{_WARN_PREFIX}Not at their release tag: aragog (paired branch {_PAIRED_BRANCH},'
            ' without the release 26.10.06). A clone that is older than the minimum version is'
            ' not used: the PROTEUS install takes that release from PyPI. The commands printed'
            ' above move a clone to its tag or replace it; a paired branch without the release'
            ' has no command, its release comes from PyPI.'
        ]
        assert not [ln for ln in out if 'git -C' in ln or 'stays as it is' in ln]
        assert git(aragog, 'symbolic-ref', '--short', 'HEAD') == _PAIRED_BRANCH
        assert git(aragog, 'rev-parse', 'HEAD') == paired
        assert list(Path(site['env']['TMPDIR']).iterdir()) == []

    @pytest.mark.parametrize('value', [None, ''], ids=['unset', 'empty'])
    def test_summary_tolerates_an_unset_or_empty_file_name(self, value):
        """report_stayed_clones returns 0 and prints nothing under set -u when the file name
        is unset or empty, as after a failed mktemp."""
        env = {k: v for k, v in os.environ.items() if k != 'PROTEUS_CLONE_STAYS'}
        if value is not None:
            env['PROTEUS_CLONE_STAYS'] = value
        result = _run_module_phase(env, 'report_stayed_clones\necho AFTER\n')
        assert result.returncode == 0, result.stderr
        assert result.stdout == 'AFTER\n'
        assert result.stderr == ''

    def test_summary_follows_the_last_module_clone(self):
        """The summary call is the line after the Zalmoxis setup and comes before the
        PROTEUS install and the import check, so a later failure does not lose it."""
        lines = INSTALL_SH.read_text().splitlines()
        call = lines.index('report_stayed_clones')
        assert lines[call - 2 : call] == [
            'run_logged "Setting up Zalmoxis" bash tools/get_zalmoxis.sh \\',
            '    || die "Could not set up Zalmoxis (bash tools/get_zalmoxis.sh); see the error above"',
        ]
        assert call < lines.index('verify_proteus_import')
        assert call < next(
            i for i, ln in enumerate(lines) if 'pip install -e ".[develop]"' in ln
        )


class TestCloneAndInstallCalls:
    """Every module install.sh clones has a floor to pin it to."""

    def test_each_call_names_a_distribution_with_a_floor(self):
        """Each call passes 2 arguments, and the 2nd has a >= floor in pyproject.toml.

        The set of distributions is pinned too, so an empty scan cannot pass.
        """
        calls = _clone_and_install_calls()
        assert [len(args) for args in calls] == [2] * len(calls)
        # get_python_module.sh installs <RepoName> as fwl-<lowercase name>.
        assert [args[1] for args in calls] == [f'fwl-{args[0].lower()}' for args in calls]
        assert {args[1] for args in calls} == {
            'fwl-mors',
            'fwl-janus',
            'fwl-calliope',
            'fwl-zephyrus',
        }
        for args in calls:
            floor = subprocess.run(
                [sys.executable, str(TOOLS_DIR / '_module_pins.py'), args[1], 'floor'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert floor.returncode == 0, floor.stderr
            assert floor.stdout.strip()[0].isdigit()


def _essential_data_lines() -> list[str]:
    """The lines of the ``essential)`` branch of install.sh's data-mode case, to its ``;;``."""
    lines = INSTALL_SH.read_text().splitlines()
    start = lines.index('    essential)')
    end = lines.index('        ;;', start)
    return [line.strip() for line in lines[start + 1 : end]]


def _default_star():
    """The star of a config that sets no ``[star]`` section."""
    import attrs

    from proteus.config._config import Config

    return attrs.fields(Config).star.default.factory()


class TestEssentialData:
    """The essential data set holds what the default config reads."""

    def test_essential_set_fetches_the_phoenix_grid_of_the_default_star(self):
        """The essential branch downloads the PHOENIX grid at the default star's
        [Fe/H] and [alpha/M], read from the config defaults, so a changed default
        star fails here rather than at an offline first run."""
        from proteus.utils.phoenix_helper import phoenix_to_grid

        star = _default_star()
        assert star.module == 'mors'
        assert star.mors.spectrum_source == 'phoenix'
        phoenix = [line for line in _essential_data_lines() if 'proteus get phoenix' in line]
        assert len(phoenix) == 1
        command, fallback = phoenix[0].split(' || ', 1)
        assert fallback == 'warn "PHOENIX download had issues"'
        words = shlex.split(command)
        assert words[:3] == ['proteus', 'get', 'phoenix']
        assert words[-1] == '2>&1'
        feh = float(words[words.index('--feh') + 1])
        alpha = float(words[words.index('--alpha') + 1])
        assert (feh, alpha) == (star.mors.phoenix_FeH, star.mors.phoenix_alpha)
        # The grid the CLI fetches is the one the default star snaps to.
        assert phoenix_to_grid(FeH=feh, alpha=alpha) == phoenix_to_grid(
            FeH=star.mors.phoenix_FeH, alpha=star.mors.phoenix_alpha
        )

    def test_essential_set_keeps_its_other_downloads(self):
        """The PHOENIX line joins the 4 other downloads, each with a warn fall-back,
        and the --all-data branch keeps its own PHOENIX line."""
        gets = [line for line in _essential_data_lines() if line.startswith('proteus get ')]
        assert [shlex.split(line.split(' || ')[0])[2] for line in gets] == [
            'spectral',
            'spectral',
            'stellar',
            'phoenix',
            'interiordata',
        ]
        assert all(' 2>&1 || warn "' in line for line in gets)
        commands = [
            line
            for line in INSTALL_SH.read_text().splitlines()
            if line.strip().startswith('proteus get phoenix ')
        ]
        assert len(commands) == 2


_JULIA_FUNCTIONS = (
    'command_exists',
    'collect_env_info',
    'die',
    'juliaup_config_matches',
    'juliaup_has_channel',
    'julia_answers_without_download',
    'resolve_real_julia_exe',
    'use_julia_channel',
    'julia_version_of',
    'report_proteus_julia',
    'stop_without_julia',
    'setup_julia',
)

# A stub julia: the default channel answers with its own directory, "+1.13" with
# STUB_BINDIR_113 (empty: no such channel), and --version with STUB_JULIA_VERSION.
_STUB_JULIA = """#!/bin/sh
echo "julia $*" >> "$STUB_LOG"
case "$1" in
    --version) echo "julia version $STUB_JULIA_VERSION" ;;
    +1.13) [ -n "$STUB_BINDIR_113" ] || exit 1; printf '%s' "$STUB_BINDIR_113" ;;
    +*) exit 1 ;;
    *) case "$*" in *Sys.BINDIR*) printf '%s' "$(dirname "$0")" ;; esac ;;
esac
"""


# juliaup: log the call, print STUB_JULIAUP_CONFIG for `api`, exit with STUB_JULIAUP_RC.
_STUB_JULIAUP = """#!/bin/sh
echo "juliaup $*" >> "$STUB_LOG"
[ "$1" != api ] || printf '%s' "${STUB_JULIAUP_CONFIG:-}"
[ -z "${STUB_JULIAUP_ERROR:-}" ] || echo "$STUB_JULIAUP_ERROR" >&2
exit "${STUB_JULIAUP_RC:-0}"
"""

_JULIAUP_READ = 'juliaup api getconfig1'

# Output of the real Juliaup 1.23.0 after `juliaup add 1.13`, the depot path replaced by /depot.
_RECORDED_GETCONFIG = PROTEUS_ROOT / 'tests' / 'data' / 'juliaup_getconfig1_1.23.0.json'


def _juliaup_config(default: str | None, *others: str) -> str:
    """The output of `juliaup api getconfig1` for a default channel and other installed channels."""

    def channel(name: str) -> dict:
        return {'Name': name, 'File': f'/jup/{name}/julia', 'Args': [], 'Version': '1.0.0'}

    return json.dumps(
        {
            'DefaultChannel': channel(default) if default else None,
            'OtherChannels': [channel(name) for name in others],
        },
        separators=(',', ':'),
    )


def _stub(path: Path, text: str) -> Path:
    """Write an executable shell script at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


class TestSetupJulia:
    """setup_julia selects an installed Julia 1.13 and never installs one; the default stays."""

    @pytest.fixture
    def site(self, tmp_path, system_path):
        """Stub julia, juliaup and python3 on a PATH without any other Julia.

        The 1.13 channel resolves to a scratch directory with its own julia, which
        reports 1.13.1, so a bare julia after the call shows which binary PATH finds.
        """
        stubs = tmp_path / 'stubs'
        _stub(stubs / 'julia', _STUB_JULIA)
        _stub(stubs / 'juliaup', _STUB_JULIAUP)
        _stub(stubs / 'python3', '#!/bin/sh\nexit 0\n')
        jl113 = tmp_path / 'juliaup' / 'julia-1.13.1' / 'bin'
        _stub(jl113 / 'julia', '#!/bin/sh\necho "julia version 1.13.1"\n')
        assert shutil.which('juliaup', path=system_path) is None
        assert shutil.which('julia', path=system_path) is None
        log = tmp_path / 'calls.log'
        log.write_text('')
        env = {
            'PATH': f'{stubs}{os.pathsep}{system_path}',
            'HOME': str(tmp_path / 'home'),
            'STUB_LOG': str(log),
            'STUB_BINDIR_113': str(jl113),
            'STUB_JULIAUP_CONFIG': _juliaup_config('1.12', '1.13'),
        }
        return {'stubs': stubs, 'jl113': jl113, 'log': log, 'env': env}

    @staticmethod
    def _call(site, version: str, **env: str) -> subprocess.CompletedProcess:
        """Run the shipped setup_julia with a default Julia of ``version``.

        After the call the script prints PATH, PYTHON_JULIAPKG_EXE, JULIA_CHANNEL
        and the version of the bare julia, one per line, prefixed with ``REPORT``.
        On every path, the only juliaup call that the run may make reads its channels.
        """
        defs = (
            'set -euo pipefail\nCURRENT_PHASE=2\n'
            + _install_sh_blocks("RED='", 'info()', 'REQUIRED_PYTHON_MAJOR=', 'JULIA_CHANNEL=')
            + ''.join(_install_sh_function(f) for f in _JULIA_FUNCTIONS)
        )
        report = (
            'setup_julia\n'
            'printf "REPORT %s\\n" "$PATH" "$PYTHON_JULIAPKG_EXE" "$JULIA_CHANNEL" "$(julia --version)"\n'
            'printf "JULIAUP_FOUND %s\\n" "$(command -v juliaup || true)"\n'
        )
        result = subprocess.run(
            ['bash', '-c', defs + report],
            capture_output=True,
            text=True,
            env={**site['env'], 'STUB_JULIA_VERSION': version, **env},
            timeout=25,
        )
        calls = site['log'].read_text().splitlines()
        assert [c for c in calls if c.startswith('juliaup') and c != _JULIAUP_READ] == []
        return result

    @staticmethod
    def _report(result: subprocess.CompletedProcess) -> list[str]:
        """The four values the script prints after setup_julia returns."""
        return [ln[7:] for ln in result.stdout.splitlines() if ln.startswith('REPORT ')]

    @staticmethod
    def _used_lines(result: subprocess.CompletedProcess) -> list[str]:
        """The lines of stdout that name the Julia PROTEUS uses."""
        return [ln for ln in result.stdout.splitlines() if 'PROTEUS uses Julia' in ln]

    @pytest.mark.parametrize(
        'version',
        ['1.10.4', '1.11.9', '1.12.6', '1.14.0'],
        ids=['default-1.10', 'default-1.11', 'default-1.12', 'default-1.14'],
    )
    def test_default_other_than_113_uses_the_installed_113_channel(self, site, version):
        """A default Julia other than 1.13 stays, and an installed 1.13 channel is used.

        Obliqua does not resolve on 1.12, and 1.11 and 1.14 are untested with
        PROTEUS. The only juliaup call reads its channels. The 1.13 binary leads
        PATH, so a bare julia reports 1.13.1, not the stub default;
        PYTHON_JULIAPKG_EXE is that binary, not the launcher stub, and one line
        names it and the default that stays.
        """
        result = self._call(site, version)
        assert result.returncode == 0, result.stdout + result.stderr
        calls = site['log'].read_text().splitlines()
        assert {c for c in calls if c.startswith('juliaup')} == {_JULIAUP_READ}
        path, exe, channel, bare = self._report(result)
        assert path.split(os.pathsep)[0] == str(site['jl113'])
        assert path.split(os.pathsep)[1] == str(site['stubs'])
        assert exe == str(site['jl113'] / 'julia')
        assert channel == '+1.13'
        assert bare == 'julia version 1.13.1'
        assert self._used_lines(result) == [
            f'\x1b[0;32m[INFO]\x1b[0m  PROTEUS uses Julia 1.13.1 at {exe}; '
            f'your default Julia {version} stays as it is.'
        ]
        assert 'deprecated' not in result.stdout + result.stderr

    def test_default_113_needs_no_channel_of_that_name(self, site):
        """A default channel `release` that runs 1.13 is enough: no channel 1.13, no stop, one read."""
        result = self._call(
            site, '1.13.1', STUB_JULIAUP_CONFIG=_juliaup_config('release', '1.11')
        )
        assert result.returncode == 0, result.stdout + result.stderr
        calls = site['log'].read_text().splitlines()
        assert {c for c in calls if c.startswith('juliaup')} == {_JULIAUP_READ}
        assert not [c for c in calls if c.startswith('julia +')]
        assert self._report(result)[1:3] == [str(site['stubs'] / 'julia'), '']

    @pytest.mark.parametrize('version', ['1.13.1', '1.13.0'], ids=['patch-1', 'patch-0'])
    def test_default_113_is_used_and_only_reads_juliaup(self, site, version):
        """A default Julia 1.13 is the Julia PROTEUS uses; the only juliaup call reads its channels.

        PATH is left as it was, PYTHON_JULIAPKG_EXE is the default's binary,
        and the one line names its version and path and no other default.
        """
        result = self._call(site, version)
        assert result.returncode == 0, result.stdout + result.stderr
        calls = site['log'].read_text().splitlines()
        assert {c for c in calls if c.startswith('juliaup')} == {_JULIAUP_READ}
        assert not [c for c in calls if c.startswith('julia +')]
        path, exe, channel, bare = self._report(result)
        assert path == site['env']['PATH']
        assert exe == str(site['stubs'] / 'julia')
        assert channel == ''
        assert bare == f'julia version {version}'
        assert self._used_lines(result) == [
            f'\x1b[0;32m[INFO]\x1b[0m  PROTEUS uses Julia {version} at {exe}.'
        ]
        assert 'stays as it is' not in result.stdout + result.stderr

    @pytest.mark.parametrize('version', ['1.10.4', '1.12.6'], ids=['julia-1.10', 'julia-1.12'])
    def test_other_julia_without_juliaup_stops(self, site, version):
        """A package-manager Julia other than 1.13 stops; the text gives the commands and installs nothing."""
        (site['stubs'] / 'juliaup').unlink()
        result = self._call(site, version)
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert f'PROTEUS runs on Julia 1.13, found {version}.' in output
        assert 'Julia 1.13 not found.' in output
        assert '  curl -fsSL https://install.julialang.org | sh\n' in result.stdout
        assert '  juliaup add 1.13\n' in result.stdout
        assert 'set PYTHON_JULIAPKG_EXE to its binary' in result.stdout
        assert 'juliaup default' not in output
        assert self._report(result) == []
        assert not [c for c in site['log'].read_text().splitlines() if c.startswith('julia +')]

    @pytest.mark.parametrize('version', ['1.10.4', '1.12.6'], ids=['julia-1.10', 'julia-1.12'])
    def test_other_julia_without_the_113_channel_stops_with_one_command(self, site, version):
        """juliaup without channel 1.13 is read and not written; `julia +1.13` never runs.

        The text names the one command, and the installer URL is not on it since
        juliaup is there.
        """
        result = self._call(site, version, STUB_JULIAUP_CONFIG=_juliaup_config('1.12', '1.11'))
        assert result.returncode == 1
        assert f'PROTEUS runs on Julia 1.13, found {version}.' in result.stderr
        assert '  juliaup add 1.13\n' in result.stdout
        assert 'install.julialang.org' not in result.stdout
        calls = site['log'].read_text().splitlines()
        assert {c for c in calls if c.startswith('juliaup')} == {_JULIAUP_READ}
        assert not [c for c in calls if c.startswith('julia +')]
        assert self._report(result) == []

    def test_listed_channel_without_a_binary_stops_instead_of_using_the_default(self, site):
        """A 1.13 channel that gives no binary stops; the 1.10 default is never used.

        A probe that falls back to the julia on PATH would set PYTHON_JULIAPKG_EXE
        to the 1.10 launcher stub and return 0.
        """
        result = self._call(site, '1.10.4', STUB_BINDIR_113='')
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert 'Could not locate the Julia 1.13 binary of its juliaup channel.' in output
        assert self._report(result) == []

    def test_unparsable_version_with_juliaup_uses_the_113_channel(self, site):
        """A julia whose --version holds no version counts as another version; no stop, no add.

        The line says the version is unknown.
        """
        result = self._call(site, 'unknown-build')
        assert result.returncode == 0, result.stdout + result.stderr
        calls = site['log'].read_text().splitlines()
        assert {c for c in calls if c.startswith('juliaup')} == {_JULIAUP_READ}
        path, exe, channel, bare = self._report(result)
        assert exe == str(site['jl113'] / 'julia')
        assert channel == '+1.13'
        assert self._used_lines(result) == [
            f'\x1b[0;32m[INFO]\x1b[0m  PROTEUS uses Julia 1.13.1 at {exe}; '
            'your default Julia (version unknown) stays as it is.'
        ]
        assert 'Could not parse' not in result.stdout + result.stderr

    @pytest.mark.parametrize(
        'env',
        [
            {'STUB_JULIAUP_CONFIG': _juliaup_config(None)},
            {'STUB_JULIAUP_RC': '1'},
            {'STUB_JULIAUP_CONFIG': 'not json'},
        ],
        ids=['no-default-channel', 'api-fails', 'api-prints-junk'],
    )
    def test_launcher_without_a_default_is_never_run(self, site, env):
        """A julia beside juliaup with no default channel gets no call: it can set itself up and download.

        The run stops with the commands, and the environment report of the
        failure block does not ask that julia for its version either.
        """
        result = self._call(site, '1.13.1', **env)
        assert result.returncode == 1
        assert 'Julia 1.13 not found.' in result.stderr
        calls = site['log'].read_text().splitlines()
        assert not [c for c in calls if c.startswith('julia ')]
        assert {c for c in calls if c.startswith('juliaup')} == {_JULIAUP_READ}

    def test_link_to_a_launcher_without_a_default_is_never_run(self, site):
        """A link in another folder to the launcher still counts as that launcher, relative or absolute."""
        for name, target in (
            ('abs', site['stubs'] / 'julia'),
            ('rel', Path('../../stubs/julia')),
        ):
            link = site['stubs'].parent / name / 'bin' / 'julia'
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            result = self._call(
                site,
                '1.13.1',
                PATH=f'{link.parent}{os.pathsep}{site["env"]["PATH"]}',
                STUB_JULIAUP_CONFIG=_juliaup_config(None),
            )
            assert result.returncode == 1, name
            assert not [
                c for c in site['log'].read_text().splitlines() if c.startswith('julia ')
            ]

    def test_link_to_a_launcher_with_a_default_is_used(self, site):
        """A link to a launcher that has a default 1.13 runs: absolute, relative and through 2 links."""
        first = site['stubs'].parent / 'abs' / 'bin' / 'julia'
        for name, target in (
            ('abs', site['stubs'] / 'julia'),
            ('rel', Path('../../stubs/julia')),
            ('two', first),
        ):
            link = site['stubs'].parent / name / 'bin' / 'julia'
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            result = self._call(
                site,
                '1.13.1',
                PATH=f'{link.parent}{os.pathsep}{site["env"]["PATH"]}',
                STUB_JULIAUP_CONFIG=_juliaup_config('release'),
            )
            assert result.returncode == 0, name + result.stdout + result.stderr
            assert self._report(result)[1:3] == [str(link), '']

    def test_link_to_a_julia_with_no_juliaup_beside_it_is_used(self, site):
        """A link to a plain Julia 1.13 needs no answer from a juliaup that does not answer."""
        link = site['stubs'].parent / 'own' / 'bin' / 'julia'
        link.parent.mkdir(parents=True)
        link.symlink_to(site['jl113'] / 'julia')
        result = self._call(
            site,
            '1.12.6',
            PATH=f'{link.parent}{os.pathsep}{site["env"]["PATH"]}',
            STUB_JULIAUP_RC='1',
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert self._report(result)[3] == 'julia version 1.13.1'
        assert [
            c for c in site['log'].read_text().splitlines() if c.startswith('juliaup')
        ] == []

    def test_recorded_juliaup_output_gives_the_same_answers(self, site):
        """The script reads the real output as it reads the test form, whose keys the real output holds."""
        real = _RECORDED_GETCONFIG.read_text().rstrip('\n')
        parsed = json.loads(real)
        test_form = json.loads(_juliaup_config('release', '1.13'))
        assert set(test_form['DefaultChannel']) <= set(parsed['DefaultChannel'])
        assert set(test_form['OtherChannels'][0]) <= set(parsed['OtherChannels'][0])
        assert [parsed['DefaultChannel']['Name'], parsed['OtherChannels'][0]['Name']] == [
            'release',
            '1.13',
        ]
        result = self._call(site, '1.12.6', STUB_JULIAUP_CONFIG=real)
        assert result.returncode == 0, result.stdout + result.stderr
        assert self._report(result)[1:3] == [str(site['jl113'] / 'julia'), '+1.13']
        assert real.count('"1.13"') == 1
        other = self._call(site, '1.12.6', STUB_JULIAUP_CONFIG=real.replace('"1.13"', '"1.12"'))
        assert other.returncode == 1
        assert 'Julia 1.13 not found.' in other.stderr

    def test_juliaup_with_a_depot_it_cannot_write_names_the_cause_and_the_route(self, site):
        """The stop quotes juliaup, names the folder of its lock file, and gives the route that works there.

        `juliaup add` cannot cure this, so the stop does not offer it.
        """
        error = 'Could not create lockfile: Permission denied (os error 13).'
        result = self._call(
            site,
            '1.13.1',
            STUB_JULIAUP_RC='1',
            STUB_JULIAUP_ERROR=error,
            JULIAUP_DEPOT_PATH='/shared/depot',
        )
        assert result.returncode == 1
        assert error in result.stderr
        assert '/shared/depot/juliaup' in result.stdout
        assert 'set PYTHON_JULIAPKG_EXE to the binary of a Julia 1.13' in result.stdout
        assert 'juliaup add' not in result.stdout
        assert 'update juliaup first' not in result.stderr

    def test_juliaup_too_old_for_the_api_gets_the_update_route(self, site):
        """An error that is no lock problem still has a route: the update command of that juliaup."""
        error = "error: unrecognized subcommand 'api'"
        result = self._call(site, '1.13.1', STUB_JULIAUP_RC='1', STUB_JULIAUP_ERROR=error)
        assert result.returncode == 1
        assert error in result.stderr
        assert 'official installer updates with: juliaup self update' in result.stdout

    def test_juliaup_that_does_not_answer_gets_its_own_line(self, site):
        """A failed API call names itself, since `juliaup add 1.13` alone cannot cure it."""
        result = self._call(site, '1.13.1', STUB_JULIAUP_RC='1')
        assert "'juliaup api getconfig1' failed" in result.stderr
        ok = self._call(site, '1.10.4', STUB_JULIAUP_CONFIG=_juliaup_config('1.12'))
        assert ok.returncode == 1
        assert 'getconfig1' not in ok.stderr

    def test_env_report_names_a_launcher_it_did_not_run(self, site):
        """The failure report does not call a launcher on PATH absent without saying why."""
        result = self._call(site, '1.13.1', STUB_JULIAUP_CONFIG=_juliaup_config(None))
        line = next(ln for ln in result.stdout.splitlines() if ln.startswith('julia:'))
        assert '(not on PATH, or a juliaup launcher with no default channel)' in line
        assert str(site['stubs'] / 'julia') in line

    def test_julia_apart_from_juliaup_is_asked_whatever_juliaup_says(self, site):
        """A julia in another folder than juliaup is no launcher, so a broken juliaup does not stop it."""
        own = site['jl113'].parents[2] / 'own' / 'bin'
        _stub(own / 'julia', _STUB_JULIA)
        (site['stubs'] / 'julia').unlink()
        path = f'{own}{os.pathsep}{site["env"]["PATH"]}'
        result = self._call(site, '1.13.1', STUB_JULIAUP_RC='1', PATH=path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert not [c for c in site['log'].read_text().splitlines() if c.startswith('juliaup')]
        assert self._report(result)[1] == str(own / 'julia')

    def test_bound_launcher_without_a_default_is_never_run(self, site):
        """A PYTHON_JULIAPKG_EXE that is a launcher with no default channel gets no call either."""
        result = self._call(
            site,
            '1.13.1',
            PYTHON_JULIAPKG_EXE=str(site['stubs'] / 'julia'),
            STUB_JULIAUP_CONFIG=_juliaup_config(None),
        )
        assert result.returncode == 1
        assert not [c for c in site['log'].read_text().splitlines() if c.startswith('julia ')]

    @pytest.mark.parametrize('juliaup', [True, False], ids=['with-juliaup', 'without-juliaup'])
    def test_bound_113_is_used_beside_a_path_julia_of_another_version(self, site, juliaup):
        """A PYTHON_JULIAPKG_EXE that reports 1.13 is the Julia PROTEUS uses, and nothing is added.

        The julia on PATH is a 1.12, as a cluster module or a second run in an env
        that binds 1.13 has it. Read from PATH alone, the version would stop the
        run without juliaup and add a second 1.13 with it. PATH stays as it was.
        """
        if not juliaup:
            (site['stubs'] / 'juliaup').unlink()
        bound = _stub(
            site['jl113'].parents[2] / 'own' / 'bin' / 'julia-1.13',
            '#!/bin/sh\necho "julia version 1.13.2"\n',
        )
        result = self._call(site, '1.12.6', PYTHON_JULIAPKG_EXE=str(bound))
        assert result.returncode == 0, result.stdout + result.stderr
        calls = site['log'].read_text().splitlines()
        assert not [c for c in calls if c.startswith(('juliaup', 'julia +'))]
        path, exe, channel, bare = self._report(result)
        assert (path, exe, channel) == (site['env']['PATH'], str(bound), '')
        assert bare == 'julia version 1.12.6'
        assert self._used_lines(result) == [
            f'\x1b[0;32m[INFO]\x1b[0m  PROTEUS uses Julia 1.13.2 at {bound}.'
        ]

    def test_bound_launcher_of_113_is_replaced_by_the_binary_it_runs(self, site):
        """A PYTHON_JULIAPKG_EXE that is a launcher reporting 1.13 becomes the binary in its Sys.BINDIR.

        A user who bound `$(which julia)` holds the juliaup launcher there; with
        the launcher bound, a later `juliaup default` changes the Julia PROTEUS uses.
        The stand-in launcher answers Sys.BINDIR with another folder, whose
        julia is what the variable holds after the call; juliaup is not called.
        """
        real = site['jl113'].parents[2] / 'real-1.13.4' / 'bin'
        _stub(real / 'julia', '#!/bin/sh\necho "julia version 1.13.4"\n')
        launcher = _stub(
            site['jl113'].parents[2] / 'launcher' / 'bin' / 'julia',
            '#!/bin/sh\ncase "$*" in\n'
            '    --version) echo "julia version 1.13.1" ;;\n'
            f'    *Sys.BINDIR*) printf "%s" "{real}" ;;\n'
            'esac\n',
        )
        result = self._call(site, '1.12.6', PYTHON_JULIAPKG_EXE=str(launcher))
        assert result.returncode == 0, result.stdout + result.stderr
        assert not [c for c in site['log'].read_text().splitlines() if c.startswith('juliaup')]
        path, exe, channel, _ = self._report(result)
        assert (path, exe, channel) == (site['env']['PATH'], str(real / 'julia'), '')
        assert exe != str(launcher)
        assert self._used_lines(result) == [
            f'\x1b[0;32m[INFO]\x1b[0m  PROTEUS uses Julia 1.13.4 at {real / "julia"}.'
        ]

    def test_bound_julia_of_another_version_gives_way_to_the_113_channel(self, site):
        """A PYTHON_JULIAPKG_EXE that reports 1.12 is not kept: PROTEUS uses the installed 1.13 channel."""
        bound = _stub(
            site['jl113'].parents[2] / 'own' / 'bin' / 'julia',
            '#!/bin/sh\necho "julia version 1.12.6"\n',
        )
        result = self._call(site, '1.10.4', PYTHON_JULIAPKG_EXE=str(bound))
        assert result.returncode == 0, result.stdout + result.stderr
        _, exe, channel, _ = self._report(result)
        assert (exe, channel) == (str(site['jl113'] / 'julia'), '+1.13')
        assert exe != str(bound)

    @pytest.mark.parametrize(
        ('bound', 'path', 'answer'),
        [('1.13.1', '1.10.4', 'YES'), ('1.11.9', '1.13.1', 'NO'), (None, '1.12.6', 'YES')],
        ids=['bound-1.13-path-1.10', 'bound-1.11-path-1.13', 'unset-path-1.12'],
    )
    def test_openssl_check_reads_the_julia_proteus_uses(self, site, bound, path, answer):
        """julia_minor_at_least_12 reads PYTHON_JULIAPKG_EXE when it is set, else the PATH julia.

        Each case has the 2 Julias on either side of 1.12, so reading the wrong one
        flips the answer.
        """
        env = {'STUB_JULIA_VERSION': path}
        if bound is not None:
            exe = _stub(
                site['jl113'].parents[2] / 'own' / 'julia',
                f'#!/bin/sh\necho "julia version {bound}"\n',
            )
            env['PYTHON_JULIAPKG_EXE'] = str(exe)
        defs = (
            'set -euo pipefail\n'
            + _install_sh_function('julia_version_of')
            + _install_sh_function('julia_minor_at_least_12')
        )
        result = subprocess.run(
            [
                'bash',
                '-c',
                defs + 'if julia_minor_at_least_12; then echo YES; else echo NO; fi\n',
            ],
            capture_output=True,
            text=True,
            env={**site['env'], **env},
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [answer]


class TestSetupJuliaWithoutJulia:
    """setup_julia without a julia on PATH: an installed 1.13 channel, or a stop with the commands."""

    @pytest.fixture
    def site(self, tmp_path, system_path):
        """A HOME of its own, and curl and python3 stubs on a PATH without julia.

        The 1.13 channel resolves to a folder with a julia that reports 1.13.1.
        """
        stubs = tmp_path / 'stubs'
        _stub(stubs / 'curl', '#!/bin/sh\necho "curl $*" >> "$STUB_LOG"\n')
        _stub(stubs / 'python3', '#!/bin/sh\nexit 0\n')
        jl113 = tmp_path / 'juliaup' / 'julia-1.13.1' / 'bin'
        _stub(jl113 / 'julia', '#!/bin/sh\necho "julia version 1.13.1"\n')
        home = tmp_path / 'home'
        home.mkdir()
        log = tmp_path / 'calls.log'
        log.write_text('')
        env = {
            'PATH': f'{stubs}{os.pathsep}{system_path}',
            'HOME': str(home),
            'STUB_LOG': str(log),
            'STUB_BINDIR_113': str(jl113),
            'STUB_JULIAUP_CONFIG': _juliaup_config('1.12', '1.13'),
        }
        return {'home': home, 'log': log, 'jl113': jl113, 'tmp': tmp_path, 'env': env}

    def test_neither_julia_nor_juliaup_stops_and_installs_nothing(self, site):
        """With no julia and no juliaup, the run stops with the 2 steps and calls nothing.

        No download, no file in HOME, no question: a run without a terminal
        behaves as one with a terminal, since nothing is asked.
        """
        result = TestSetupJulia._call(site, '1.13.1')
        assert result.returncode == 1
        assert site['log'].read_text() == ''
        assert list(site['home'].iterdir()) == []
        assert (
            'PROTEUS needs Julia 1.13. install.sh checks for Julia and does not install it.'
            in (result.stdout)
        )
        assert (
            '  curl -fsSL https://install.julialang.org | sh\n'
            '  (open a new terminal)\n'
            '  juliaup add 1.13\n'
        ) in result.stdout
        assert 'Your default Julia stays as it is.' in result.stdout
        assert 'https://proteus-framework.org/PROTEUS/How-to/installation.html' in result.stdout
        fails = [ln for ln in result.stderr.splitlines() if '[FAIL]' in ln]
        assert fails == ['\x1b[0;31m[FAIL]\x1b[0m  Julia 1.13 not found.']
        assert 'Installation failed at Phase 2.' in result.stdout
        assert TestSetupJulia._report(result) == []

    def test_juliaup_without_a_default_uses_the_113_channel(self, site):
        """juliaup with no default channel and an installed 1.13: `julia +1.13` runs, bare julia never.

        The launcher sits beside juliaup, as juliaup ships it. The line names the
        juliaup default that stays.
        """
        jup = site['tmp'] / 'jup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        _stub(jup / 'julia', _STUB_JULIA)
        result = TestSetupJulia._call(
            site,
            '1.10.4',
            PATH=f'{jup}{os.pathsep}{site["env"]["PATH"]}',
            STUB_JULIAUP_CONFIG=_juliaup_config(None, '1.13'),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        calls = site['log'].read_text().splitlines()
        assert {c for c in calls if c.startswith(('juliaup', 'curl'))} == {_JULIAUP_READ}
        assert [c for c in calls if c.startswith('julia ')] == [
            'julia +1.13 --startup-file=no -e print(Sys.BINDIR)'
        ]
        path, exe, channel, bare = TestSetupJulia._report(result)
        assert (exe, channel, bare) == (
            str(site['jl113'] / 'julia'),
            '+1.13',
            'julia version 1.13.1',
        )
        assert TestSetupJulia._used_lines(result) == [
            f'\x1b[0;32m[INFO]\x1b[0m  PROTEUS uses Julia 1.13.1 at {exe}; '
            'the default Julia of juliaup stays as it is.'
        ]

    def test_juliaup_without_julia_or_the_113_channel_stops_with_one_command(self, site):
        """A juliaup on PATH with neither julia nor channel 1.13 gets the one command, and no write."""
        jup = site['tmp'] / 'jup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        result = TestSetupJulia._call(
            site,
            '1.10.4',
            PATH=f'{jup}{os.pathsep}{site["env"]["PATH"]}',
            STUB_JULIAUP_CONFIG=_juliaup_config(None),
        )
        assert result.returncode == 1
        assert '  juliaup add 1.13\n' in result.stdout
        assert 'install.julialang.org' not in result.stdout
        assert set(site['log'].read_text().splitlines()) == {_JULIAUP_READ}

    def test_juliaup_config_without_a_juliaup_on_path_stops(self, site):
        """A ~/.julia/juliaup/juliaup.json with no juliaup found stops and points at that juliaup.

        The official installer would make a second juliaup on that shared configuration.
        """
        config = site['home'] / '.julia' / 'juliaup' / 'juliaup.json'
        config.parent.mkdir(parents=True)
        config.write_text('{"Default": "release"}\n')
        result = TestSetupJulia._call(site, '1.10.4')
        assert result.returncode == 1
        assert site['log'].read_text() == ''
        fails = [ln for ln in result.stderr.splitlines() if '[FAIL]' in ln]
        assert fails == [
            f'\x1b[0;31m[FAIL]\x1b[0m  A juliaup configuration exists at {config}, but no '
            'juliaup is on PATH or in ~/.juliaup/bin; put your juliaup on PATH and run '
            'bash install.sh again.'
        ]
        assert not (site['home'] / '.juliaup').exists()
        assert config.read_text() == '{"Default": "release"}\n'

    def test_juliaup_config_in_the_depot_path_stops(self, site):
        """A juliaup.json under JULIAUP_DEPOT_PATH, with none under ~/.julia, stops the same way.

        juliaup reads its configuration from ``$JULIAUP_DEPOT_PATH/juliaup`` when
        that is set, as the cluster guides set it; a check of ~/.julia alone misses it.
        """
        depot = site['home'].parent / 'data' / 'juliaup-depot'
        config = depot / 'juliaup' / 'juliaup.json'
        config.parent.mkdir(parents=True)
        config.write_text('{"Default": "1.10"}\n')
        result = TestSetupJulia._call(site, '1.10.4', JULIAUP_DEPOT_PATH=str(depot))
        assert result.returncode == 1
        assert site['log'].read_text() == ''
        fails = [ln for ln in result.stderr.splitlines() if '[FAIL]' in ln]
        assert fails == [
            f'\x1b[0;31m[FAIL]\x1b[0m  A juliaup configuration exists at {config}, but no '
            'juliaup is on PATH or in ~/.juliaup/bin; put your juliaup on PATH and run '
            'bash install.sh again.'
        ]
        assert sorted(p.name for p in site['home'].iterdir()) == []
        assert config.read_text() == '{"Default": "1.10"}\n'

    def test_juliaup_in_the_default_place_off_path_is_used(self, site):
        """A juliaup in ~/.juliaup/bin, with no julia and no juliaup on PATH, gives its 1.13 channel.

        setup_julia puts ~/.juliaup/bin on its own PATH, and only reads the channels.
        """
        jup = site['home'] / '.juliaup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        _stub(jup / 'julia', _STUB_JULIA)
        result = TestSetupJulia._call(site, '1.10.4')
        assert result.returncode == 0, result.stdout + result.stderr
        calls = site['log'].read_text().splitlines()
        assert {c for c in calls if c.startswith(('juliaup', 'curl'))} == {_JULIAUP_READ}
        path, exe, channel, _ = TestSetupJulia._report(result)
        assert str(jup) in path.split(os.pathsep)
        assert (exe, channel) == (str(site['jl113'] / 'julia'), '+1.13')
        assert TestSetupJulia._used_lines(result) == [
            f'\x1b[0;32m[INFO]\x1b[0m  PROTEUS uses Julia 1.13.1 at {exe}; '
            'your default Julia 1.10.4 stays as it is.'
        ]

    def test_off_path_juliaup_with_113_is_used_beside_a_julia_of_another_version(self, site):
        """Another julia on PATH does not hide the 1.13 channel of a juliaup in ~/.juliaup/bin."""
        jup = site['home'] / '.juliaup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        _stub(jup / 'julia', _STUB_JULIA)
        own = site['tmp'] / 'own'
        _stub(own / 'julia', '#!/bin/sh\necho "julia version 1.12.6"\n')
        result = TestSetupJulia._call(
            site, '1.13.1', PATH=f'{own}{os.pathsep}{site["env"]["PATH"]}'
        )
        assert result.returncode == 0, result.stdout + result.stderr
        path, exe, channel, bare = TestSetupJulia._report(result)
        assert [exe, channel] == [str(site['jl113'] / 'julia'), '+1.13']
        assert bare == 'julia version 1.13.1'
        folders = path.split(os.pathsep)
        assert folders.index(str(own)) < folders.index(str(jup))
        assert f'JULIAUP_FOUND {jup}/juliaup' in result.stdout.splitlines()

    def test_own_julia_113_keeps_an_off_path_juliaup_off_path(self, site):
        """With the user's 1.13 chosen, no later step can find the juliaup in ~/.juliaup/bin."""
        jup = site['home'] / '.juliaup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        _stub(jup / 'julia', _STUB_JULIA)
        own = site['tmp'] / 'own'
        _stub(own / 'julia', '#!/bin/sh\necho "julia version 1.13.2"\n')
        result = TestSetupJulia._call(
            site, '1.12.6', PATH=f'{own}{os.pathsep}{site["env"]["PATH"]}'
        )
        assert result.returncode == 0, result.stdout + result.stderr
        path, exe, channel, bare = TestSetupJulia._report(result)
        assert [exe, channel, bare] == [str(own / 'julia'), '', 'julia version 1.13.2']
        assert str(jup) not in path.split(os.pathsep)
        assert 'JULIAUP_FOUND ' in result.stdout.splitlines()

    def test_off_path_launcher_that_answers_with_no_julia_on_path_stays_on_path(self, site):
        """With no julia on PATH, the launcher in ~/.juliaup/bin is the Julia in use, so it stays."""
        jup = site['home'] / '.juliaup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        _stub(jup / 'julia', _STUB_JULIA)
        result = TestSetupJulia._call(
            site, '1.13.1', STUB_JULIAUP_CONFIG=_juliaup_config('1.13')
        )
        assert result.returncode == 0, result.stdout + result.stderr
        path, _, channel, bare = TestSetupJulia._report(result)
        assert [channel, bare] == ['', 'julia version 1.13.1']
        assert str(jup) in path.split(os.pathsep)
        assert f'JULIAUP_FOUND {jup}/juliaup' in result.stdout.splitlines()

    def test_juliaup_config_with_another_julia_names_the_file_and_no_second_install(self, site):
        """Another julia on PATH does not turn an existing juliaup configuration into a curl line."""
        config = site['home'] / '.julia' / 'juliaup' / 'juliaup.json'
        config.parent.mkdir(parents=True)
        config.write_text('{"Default": "release"}\n')
        own = site['tmp'] / 'own'
        _stub(own / 'julia', '#!/bin/sh\necho "julia version 1.12.6"\n')
        result = TestSetupJulia._call(
            site, '1.12.6', PATH=f'{own}{os.pathsep}{site["env"]["PATH"]}'
        )
        assert result.returncode == 1
        assert 'install.julialang.org' not in result.stdout
        assert 'juliaup add' not in result.stdout
        assert f'A juliaup configuration exists at {config}' in result.stderr
        assert 'put your juliaup on PATH' in result.stderr

    def test_juliaup_config_with_its_juliaup_found_gets_the_add_command(self, site):
        """The configuration matters only when no juliaup was found."""
        config = site['home'] / '.julia' / 'juliaup' / 'juliaup.json'
        config.parent.mkdir(parents=True)
        config.write_text('{"Default": "release"}\n')
        jup = site['home'] / '.juliaup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        _stub(jup / 'julia', _STUB_JULIA)
        result = TestSetupJulia._call(
            site, '1.12.6', STUB_JULIAUP_CONFIG=_juliaup_config('1.12')
        )
        assert result.returncode == 1
        assert f'  {jup}/juliaup add 1.13' in result.stdout.splitlines()
        assert 'A juliaup configuration exists' not in result.stderr

    def test_juliaup_config_with_a_julia_113_on_path_passes(self, site):
        """A plain julia 1.13 on PATH passes before the configuration matters."""
        config = site['home'] / '.julia' / 'juliaup' / 'juliaup.json'
        config.parent.mkdir(parents=True)
        config.write_text('{"Default": "release"}\n')
        own = site['tmp'] / 'own'
        _stub(own / 'julia', '#!/bin/sh\necho "julia version 1.13.2"\n')
        result = TestSetupJulia._call(
            site, '1.12.6', PATH=f'{own}{os.pathsep}{site["env"]["PATH"]}'
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert TestSetupJulia._report(result)[1] == str(own / 'julia')

    def test_stop_prints_a_command_that_survives_a_space_in_home(self, site):
        """The printed line, split as a shell splits it, is the juliaup and its 2 arguments."""
        home = site['tmp'] / 'my home'
        jup = home / '.juliaup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        _stub(jup / 'julia', _STUB_JULIA)
        result = TestSetupJulia._call(
            site, '1.12.6', HOME=str(home), STUB_JULIAUP_CONFIG=_juliaup_config('1.12')
        )
        assert result.returncode == 1
        lines = [ln for ln in result.stdout.splitlines() if ln.endswith(' add 1.13')]
        assert len(lines) == 1
        assert shlex.split(lines[0]) == [f'{jup}/juliaup', 'add', '1.13']
        failed = TestSetupJulia._call(site, '1.12.6', HOME=str(home), STUB_JULIAUP_RC='1')
        lines = [ln for ln in failed.stdout.splitlines() if ln.endswith(' self update')]
        assert len(lines) == 1
        assert shlex.split(lines[0].split(': ')[1]) == [f'{jup}/juliaup', 'self', 'update']

    def test_off_path_juliaup_too_old_for_the_api_gets_the_command_with_its_path(self, site):
        """The update route names the juliaup that failed, also when it is not on the user's PATH."""
        jup = site['home'] / '.juliaup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        _stub(jup / 'julia', _STUB_JULIA)
        result = TestSetupJulia._call(site, '1.12.6', STUB_JULIAUP_RC='1')
        assert result.returncode == 1
        assert f'official installer updates with: {jup}/juliaup self update' in result.stdout

    @pytest.mark.parametrize('other_julia', [True, False], ids=['other-julia', 'no-julia'])
    def test_off_path_juliaup_without_113_gets_the_command_with_its_path(
        self, site, other_julia
    ):
        """The stop prints a command that runs in the user's shell, and no second install of juliaup."""
        jup = site['home'] / '.juliaup' / 'bin'
        _stub(jup / 'juliaup', _STUB_JULIAUP)
        _stub(jup / 'julia', _STUB_JULIA)
        own = site['tmp'] / 'own'
        _stub(own / 'julia', '#!/bin/sh\necho "julia version 1.12.6"\n')
        path = f'{own}{os.pathsep}{site["env"]["PATH"]}' if other_julia else site['env']['PATH']
        result = TestSetupJulia._call(
            site, '1.12.6', PATH=path, STUB_JULIAUP_CONFIG=_juliaup_config('1.12')
        )
        assert result.returncode == 1
        assert f'  {jup}/juliaup add 1.13' in result.stdout.splitlines()
        assert 'install.julialang.org' not in result.stdout

    def test_stop_links_the_julia_step_of_the_page(self, site):
        """The stop names the page section that exists, for a reader on any cluster."""
        result = TestSetupJulia._call(site, '1.13.1')
        assert 'How-to/installation.html#2-install-julia' in result.stdout
        page = PROTEUS_ROOT / 'docs' / 'How-to' / 'installation.md'
        assert '\n## 2. Install Julia\n' in page.read_text()


class TestJuliaupHasChannel:
    """juliaup_has_channel is true only when the channel of that exact name is listed."""

    @staticmethod
    def _ask(tmp: Path, channel: str, printed: str, rc: int = 0) -> subprocess.CompletedProcess:
        """Run the shipped juliaup_has_channel with a juliaup stub that prints ``printed``.

        The answer is on stdout, Y or N; stderr stays empty, also for text that is no config.
        """
        (tmp / 'printed.txt').write_text(printed)
        stubs = tmp / 'stubs'
        _stub(stubs / 'juliaup', f'#!/bin/sh\ncat "{tmp}/printed.txt"\nexit {rc}\n')
        defs = 'set -euo pipefail\n' + ''.join(
            _install_sh_function(f) for f in ('juliaup_config_matches', 'juliaup_has_channel')
        )
        ask = f'if juliaup_has_channel {shlex.quote(channel)}; then echo Y; else echo N; fi\n'
        return subprocess.run(
            ['bash', '-c', defs + ask],
            capture_output=True,
            text=True,
            env={'PATH': f'{stubs}{os.pathsep}/usr/bin:/bin'},
            timeout=10,
        )

    @pytest.mark.parametrize(
        ('printed', 'rc', 'expected'),
        [
            (_juliaup_config('1.13'), 0, 'Y'),
            (_juliaup_config('1.12', 'release', '1.13', '1'), 0, 'Y'),
            (json.dumps(json.loads(_juliaup_config('1.13')), indent=2), 0, 'Y'),
            (_juliaup_config('1.13'), 1, 'N'),
            (_juliaup_config('1.12', '1.13.1', '1.130', '1x13', '11.13'), 0, 'N'),
            (_juliaup_config(None), 0, 'N'),
            ('', 0, 'N'),
            ('null', 0, 'N'),
            ('{"OtherChannels":[{"Name":13}]}', 0, 'N'),
            ('{"Note":"\\"Name\\":\\"1.13\\""}', 0, 'N'),
        ],
        ids=[
            'default',
            'other',
            'spaced-json',
            'api-fails',
            'near-names',
            'empty-state',
            'no-output',
            'null',
            'wrong-type',
            'name-inside-a-string',
        ],
    )
    def test_answer(self, tmp_path, printed, rc, expected):
        """Near names, a failed call and text that is no config answer no."""
        result = self._ask(tmp_path, '1.13', printed, rc)
        assert result.stdout.strip() == expected
        assert result.stderr == ''

    def test_answer_equals_a_json_read_of_the_names(self):
        """Over lists of near names, the shell answer equals a JSON read of the same output."""
        hypothesis = pytest.importorskip('hypothesis')
        st = hypothesis.strategies

        @hypothesis.settings(max_examples=25, deadline=None)
        @hypothesis.given(
            names=st.lists(
                st.sampled_from(
                    ['1.13', '1.13.1', '1.130', '1x13', '11.13', 'release', '1', '']
                ),
                max_size=4,
            )
        )
        def check(names):
            with tempfile.TemporaryDirectory() as td:
                result = self._ask(Path(td), '1.13', _juliaup_config(None, *names))
            assert result.stdout.strip() == ('Y' if '1.13' in names else 'N')
            assert result.stderr == ''

        check()

    def test_text_that_names_no_channel_answers_no(self):
        """Arbitrary text with no quoted channel name answers no and never stops the shell."""
        hypothesis = pytest.importorskip('hypothesis')
        st = hypothesis.strategies

        @hypothesis.settings(max_examples=25, deadline=None)
        @hypothesis.given(
            printed=st.text(
                alphabet=st.characters(exclude_categories=('Cs',), exclude_characters='\x00'),
                max_size=200,
            )
        )
        def check(printed):
            hypothesis.assume('"1.13"' not in printed)
            with tempfile.TemporaryDirectory() as td:
                result = self._ask(Path(td), '1.13', printed)
            assert result.stdout.strip() == 'N'
            assert result.stderr == ''

        check()


class TestAgniJuliaVersion:
    """Phase 5 warns once when an existing AGNI was built with another minor Julia version."""

    @pytest.mark.parametrize(
        ('manifest', 'warned'),
        [
            ('julia_version = "1.11.5"\n', True),
            ('julia_version = "1.12.0-rc1"\n', True),
            ('julia_version = "1.13.0"\n', False),
            ('manifest_format = "2.0"\n', False),
        ],
        ids=['built-1.11', 'built-1.12-rc', 'built-1.13-other-patch', 'no-version-line'],
    )
    def test_warns_only_for_another_minor_version(self, tmp_path, manifest, warned):
        """A Manifest.toml of Julia 1.11 or 1.12 warns with both versions; 1.13.0 and none do not.

        PROTEUS uses a stub Julia 1.13.1, so a patch release differs but the minor
        version does not; the warning names `proteus update-all`.
        """
        agni = tmp_path / 'AGNI'
        agni.mkdir()
        (agni / 'Manifest.toml').write_text('# This file is machine-generated\n\n' + manifest)
        exe = _stub(tmp_path / 'jl' / 'julia', '#!/bin/sh\necho "julia version 1.13.1"\n')
        defs = (
            'set -euo pipefail\n'
            + _install_sh_blocks("RED='", 'info()')
            + _install_sh_function('julia_version_of')
            + _install_sh_function('note_agni_julia_version')
        )
        result = subprocess.run(
            ['bash', '-c', defs + f'note_agni_julia_version {shlex.quote(str(agni))}\n'],
            capture_output=True,
            text=True,
            env={'PATH': '/usr/bin:/bin', 'PYTHON_JULIAPKG_EXE': str(exe)},
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == ''
        built = manifest.split('"')[1]
        expected = [
            f'{_WARN_PREFIX}AGNI at {agni} was built with Julia {built} (its Manifest.toml), '
            "and PROTEUS uses Julia 1.13.1; after the installer, run 'proteus update-all', "
            'which brings AGNI to its pin and instantiates it again.'
        ]
        assert result.stderr.splitlines() == (expected if warned else [])

    @pytest.mark.parametrize(
        ('built', 'warned'),
        [('1.13.1', True), ('1.11.5', False)],
        ids=['agni-built-with-1.13', 'agni-built-with-1.11'],
    )
    def test_julia_111_fall_back_names_an_agni_built_with_another_version(
        self, tmp_path, built, warned
    ):
        """After the switch of PROTEUS to Julia 1.11, an AGNI built with 1.13 gets the warning.

        Phase 5 built AGNI with 1.13 before the import check of Phase 7 moved
        PROTEUS to 1.11; the switch itself records the 1.11 binary.
        """
        root = tmp_path / 'proteus'
        (root / 'AGNI').mkdir(parents=True)
        (root / 'AGNI' / 'Manifest.toml').write_text(f'julia_version = "{built}"\n')
        bin111 = tmp_path / 'juliaup' / 'julia-1.11.9' / 'bin'
        exe = _stub(bin111 / 'julia', '#!/bin/sh\necho "julia version 1.11.9"\n')
        jup = tmp_path / 'jup'
        _stub(jup / 'juliaup', f"#!/bin/sh\nprintf '%s' '{_juliaup_config('1.13', '1.11')}'\n")
        _stub(jup / 'julia', f'#!/bin/sh\n[ "$1" = +1.11 ] && printf "%s" "{bin111}"\n')
        defs = (
            'set -euo pipefail\n'
            + f'SCRIPT_DIR={shlex.quote(str(root))}\n'
            + _install_sh_blocks("RED='", 'info()', 'PROTEUS_ENV_FILE=')
            + ''.join(
                _install_sh_function(f)
                for f in (
                    'command_exists',
                    'juliaup_config_matches',
                    'juliaup_has_channel',
                    'julia_version_of',
                    'resolve_real_julia_exe',
                    'use_julia_channel',
                    'set_install_var',
                    'reset_julia_env',
                    'note_agni_julia_version',
                    'fix_julia_111_for_openssl',
                    'print_exit_hint',
                )
            )
        )
        result = subprocess.run(
            [
                'bash',
                '-c',
                defs + 'fix_julia_111_for_openssl\necho "EXE=$PYTHON_JULIAPKG_EXE"\n',
            ],
            capture_output=True,
            text=True,
            env={'PATH': f'{jup}:/usr/bin:/bin', 'HOME': str(tmp_path)},
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert f'EXE={exe}' in result.stdout.splitlines()
        warning = (
            f'{_WARN_PREFIX}AGNI at {root / "AGNI"} was built with Julia {built} (its '
            "Manifest.toml), and PROTEUS uses Julia 1.11.9; after the installer, run 'proteus "
            "update-all', which brings AGNI to its pin and instantiates it again."
        )
        assert result.stderr.splitlines().count(warning) == (1 if warned else 0)
        assert [ln for ln in result.stderr.splitlines() if 'AGNI at' in ln] == (
            [warning] if warned else []
        )

    def test_julia_111_fall_back_adds_no_channel(self, tmp_path):
        """Without an installed 1.11 channel the fall-back returns 1 and only reads juliaup.

        PYTHON_JULIAPKG_EXE stays as it was, so the OpenSSL route of the import check is next.
        """
        log = tmp_path / 'calls.log'
        log.write_text('')
        jup = tmp_path / 'jup'
        _stub(
            jup / 'juliaup',
            f'#!/bin/sh\necho "juliaup $*" >> "{log}"\n'
            f"printf '%s' '{_juliaup_config('1.13')}'\n",
        )
        _stub(jup / 'julia', f'#!/bin/sh\necho "julia $*" >> "{log}"\n')
        defs = 'set -euo pipefail\nPYTHON_JULIAPKG_EXE=kept\n' + ''.join(
            _install_sh_function(f)
            for f in ('command_exists', 'juliaup_config_matches', 'juliaup_has_channel')
        )
        defs += _install_sh_function('fix_julia_111_for_openssl')
        result = subprocess.run(
            [
                'bash',
                '-c',
                defs
                + 'if fix_julia_111_for_openssl; then echo Y; else echo N; fi\n'
                + 'echo "EXE=$PYTHON_JULIAPKG_EXE"\n',
            ],
            capture_output=True,
            text=True,
            env={'PATH': f'{jup}:/usr/bin:/bin', 'HOME': str(tmp_path)},
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ['N', 'EXE=kept']
        assert log.read_text().splitlines() == [_JULIAUP_READ]

    def test_phase_5_checks_an_existing_agni_and_does_not_rebuild_it(self):
        """The check runs in the branch for an existing AGNI, which runs no build step."""
        lines = INSTALL_SH.read_text().splitlines()
        start = lines.index('    info "AGNI already installed at $SCRIPT_DIR/AGNI"')
        assert lines[start + 1] == '    note_agni_julia_version "$SCRIPT_DIR/AGNI"'
        assert lines[start + 2] == '    instantiate_julia_project "$SCRIPT_DIR/AGNI"'
        assert lines[start + 3] == 'else'


class TestNoJuliaInstall:
    """install.sh checks for Julia and never installs it or changes a default."""

    def test_install_commands_only_appear_as_text_for_the_user(self):
        """`juliaup add` and the installer URL occur only in lines that print or say them.

        `juliaup default` does not occur at all, here or in doctor.py, and prompt_yn is
        gone with its one question.
        """
        lines = INSTALL_SH.read_text().splitlines()
        prints = ('echo ', 'printf ', 'warn ', 'info ', 'die ', 'fail ', '#')
        for word in ('juliaup add', '%q add', 'install.julialang.org'):
            found = [ln.strip() for ln in lines if word in ln]
            assert found, word
            assert [ln for ln in found if not ln.startswith(prints)] == []
        text = INSTALL_SH.read_text()
        assert 'juliaup default' not in text
        assert 'prompt_yn' not in text
        doctor = PROTEUS_ROOT / 'src' / 'proteus' / 'doctor.py'
        assert 'juliaup default' not in doctor.read_text()

    @pytest.mark.parametrize('guide', ['kapteyn_cluster_guide.md', 'ioa_cluster_guide.md'])
    def test_cluster_guide_moves_the_depot_before_any_download(self, guide):
        """The guide sets the variable juliaup reads, active in the shell of its installer, which downloads a Julia."""
        text = (PROTEUS_ROOT / 'docs' / 'How-to' / guide).read_text()
        assert 'JULIAUP_HOME' not in text
        steps = [
            'export JULIAUP_DEPOT_PATH=',
            'source ~/.bashrc',
            'julialang.org |',
            'juliaup add 1.13',
        ]
        assert [step for step in steps if step not in text] == []
        places = [text.index(step) for step in steps]
        assert places == sorted(places)
        assert 'the juliaup program' in text
        assert 'first Julia into the folder of `JULIAUP_DEPOT_PATH`' in text

    def test_kapteyn_steps_set_up_julia_before_the_installer(self):
        """Item 7 of the numbered steps sends the reader to the Julia section, and then to the installer."""
        text = (PROTEUS_ROOT / 'docs' / 'How-to' / 'kapteyn_cluster_guide.md').read_text()
        item = text[text.index('\n7. ') :]
        item = item[: item.index('\n\n')]
        assert '(#julia-considerations)' in item
        assert item.index('(#julia-considerations)') < item.index('run the installer')
        assert '\n### Julia considerations\n' in text
        assert 'clone PROTEUS' in item
        assert item.index('(#julia-considerations)') < item.index('clone PROTEUS')

    def test_kapteyn_conda_item_stops_before_the_installer_and_makes_the_env(self):
        """Item 5 ends before the page's installer, and makes the env that item 6 installs into."""
        text = (PROTEUS_ROOT / 'docs' / 'How-to' / 'kapteyn_cluster_guide.md').read_text()
        item = text[text.index('\n5. ') : text.index('\n6. ')]
        assert 'up to and including' not in item
        assert 'stop before you clone PROTEUS' in item
        script = INSTALL_SH.read_text()
        python = '.'.join(
            re.search(rf'^REQUIRED_PYTHON_{part}=(\d+)', script, re.M).group(1)
            for part in ('MAJOR', 'MINOR')
        )
        make = f'conda create -n proteus python={python}\n'
        assert make in re.sub(r'^ +', '', item, flags=re.M)
        page = (PROTEUS_ROOT / 'docs' / 'How-to' / 'installation.md').read_text()
        assert (
            page.index('Miniforge3-')
            < page.index('Then clone PROTEUS:')
            < page.index('bash install.sh')
        )

    def test_ioa_guide_says_which_folder_each_sentence_means(self):
        """The prompt of the official installer sets the program folder; the Julia versions go to the depot."""
        text = (PROTEUS_ROOT / 'docs' / 'How-to' / 'ioa_cluster_guide.md').read_text()
        prompt = '`Enter the folder where you want to install Juliaup`'
        depot = 'first Julia into the folder of `JULIAUP_DEPOT_PATH`'
        assert [
            part for part in ('Customize installation', prompt, depot) if part not in text
        ] == []
        assert text.index(prompt) < text.index('the juliaup program') < text.index(depot)

    def test_install_page_says_when_the_installer_asks_for_juliaup_on_path(self):
        """The request for a juliaup on PATH comes only after no Julia 1.13 was found."""
        page = (PROTEUS_ROOT / 'docs' / 'How-to' / 'installation.md').read_text()
        start = 'When it finds no Julia 1.13 and no juliaup there'
        assert start in page
        sentence = page[page.index(start) :]
        sentence = sentence[: sentence.index('\n')]
        assert 'it names that file' in sentence
        assert 'asks you to put your juliaup on your `PATH`' in sentence

    def test_check_runs_before_the_env_is_made_and_once(self):
        """setup_conda_env checks Julia just before `conda create`; the Julia step then skips the check."""
        body = _install_sh_function('setup_conda_env').splitlines()
        create = next(i for i, ln in enumerate(body) if 'if ! conda create' in ln)
        assert '        setup_julia' in body[:create]
        check = body.index('        setup_julia')
        assert body[check + 1] == '        JULIA_SELECTED=true'
        lines = INSTALL_SH.read_text().splitlines()
        assert lines.count('[ "$JULIA_SELECTED" = true ] || setup_julia') == 1


_CONDA_FUNCTIONS = (
    'command_exists',
    'collect_env_info',
    'die',
    'stop_without_conda_env',
    'stop_on_env_folder',
    'setup_conda_env',
    'ask_yn',
    'conda_activate_command',
    'print_next_steps',
    'print_exit_hint',
)

# The conda executable: `info --base`, `config --show envs_dirs` (a missing folder, then the envs
# folder of the base), `env list` (base, a nameless env, then STUB_CONDA_ENVS with the mark
# STUB_CONDA_MARK) and `create -y -n <name> ...`, which makes the env unless STUB_CREATE_RC is set.
_STUB_CONDA_EXE = """#!/bin/sh
printf '%s\\n' "$@" >> "$STUB_CONDA_LOG"
echo --- >> "$STUB_CONDA_LOG"
case "$1 $2" in
    'info --base') printf '%s\\n' "$STUB_CONDA_BASE" ;;
    'config --show') printf 'envs_dirs:\\n  - %s\\n  - %s\\n' "$HOME/.conda/envs" "$STUB_CONDA_BASE/envs" ;;
    'env list')
        echo '# conda environments:'
        echo '#'
        echo "base                  *  $STUB_CONDA_BASE"
        echo "                         /elsewhere/proteus"
        for name in $STUB_CONDA_ENVS; do
            echo "$name  ${STUB_CONDA_MARK:-}  $STUB_CONDA_BASE/envs/$name"
        done
        ;;
    'create -y')
        [ "${STUB_CREATE_RC:-0}" = 0 ] || exit "$STUB_CREATE_RC"
        mkdir -p "$STUB_CONDA_BASE/envs/$4/bin"
        printf '#!/bin/sh\\n' > "$STUB_CONDA_BASE/envs/$4/bin/python"
        chmod +x "$STUB_CONDA_BASE/envs/$4/bin/python"
        ;;
esac
"""

# The shell function of conda.sh: `activate` reads an unset variable, as conda's activation
# scripts do, and sets the env (named STUB_ACTIVATE_AS, when set); the rest go to the executable.
_STUB_CONDA_SH = """conda() {
    [ "$1" = activate ] || { command conda "$@"; return; }
    printf '%s\\n' "$@" >> "$STUB_CONDA_LOG"
    echo --- >> "$STUB_CONDA_LOG"
    : "$SOME_UNSET_NAME"
    [ "${STUB_ACTIVATE_RC:-0}" = 0 ] || return "$STUB_ACTIVATE_RC"
    [ -z "${STUB_BOUND_DEPOT:-}" ] || export JULIA_DEPOT_PATH="$STUB_BOUND_DEPOT"
    CONDA_DEFAULT_ENV="${STUB_ACTIVATE_AS:-$2}"
    CONDA_PREFIX="$STUB_CONDA_BASE/envs/$2"
    PATH="$CONDA_PREFIX/bin:$PATH"
    export CONDA_DEFAULT_ENV CONDA_PREFIX PATH
}
"""

_MANUAL_COMMANDS = ('  conda create -n proteus python=3.12', '  conda activate proteus')


def _drain(master: int) -> str:
    """The output waiting on the master side of a pseudo-terminal, decoded, without a wait."""
    chunks = []
    while select.select([master], [], [], 0)[0]:
        try:
            data = os.read(master, 4096)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b''.join(chunks).decode(errors='replace')


# The line after a failed `conda create`.
_TOS_LINE = (
    "If conda printed a terms-of-service error above, run the 'conda tos accept' commands it "
    'printed, then bash install.sh again.'
)


class TestSetupCondaEnv:
    """setup_conda_env uses the active env, or uses or makes `proteus` and activates it in its own process."""

    @pytest.fixture
    def site(self, tmp_path, system_path):
        """A fake conda base with conda.sh, a conda executable and a python outside every env.

        The env list holds `proteus2` and a nameless env at a path ending in
        `/proteus`, so only an exact match on the name column finds no `proteus`.
        """
        base = tmp_path / 'conda'
        (base / 'etc' / 'profile.d').mkdir(parents=True)
        (base / 'etc' / 'profile.d' / 'conda.sh').write_text(_STUB_CONDA_SH)
        stubs = tmp_path / 'stubs'
        _stub(stubs / 'conda', _STUB_CONDA_EXE)
        _stub(stubs / 'python', '#!/bin/sh\nexit 0\n')
        _stub(stubs / 'python3', '#!/bin/sh\nexit 0\n')
        assert shutil.which('conda', path=system_path) is None
        env = {
            'PATH': f'{stubs}{os.pathsep}{system_path}',
            'HOME': str(tmp_path / 'home'),
            'STUB_CONDA_LOG': str(tmp_path / 'conda.log'),
            'STUB_CONDA_BASE': str(base),
            'STUB_CONDA_ENVS': 'proteus2',
        }
        return {'base': base, 'stubs': stubs, 'log': tmp_path / 'conda.log', 'env': env}

    @staticmethod
    def _existing_env(site, name: str, python: bool = True) -> Path:
        """Make the env ``name`` in the fake base and list it; return its prefix."""
        prefix = site['base'] / 'envs' / name
        (prefix / 'bin').mkdir(parents=True)
        if python:
            _stub(prefix / 'bin' / 'python', '#!/bin/sh\nexit 0\n')
        site['env']['STUB_CONDA_ENVS'] += f' {name}'
        return prefix

    @staticmethod
    def _call(
        site,
        requested: str = '',
        after: str = '',
        terminal: bytes | None = None,
        saved_stdout: Path | None = None,
        icrnl: bool = True,
        **env: str,
    ) -> subprocess.CompletedProcess:
        """Run the shipped setup_conda_env under install.sh's options, then ``after``.

        After the call the script prints CONDA_DEFAULT_ENV, CONDA_PREFIX, the python
        on PATH, CONDA_ACTIVATED_HERE, the kind of ``conda`` and the shell flags,
        one per line, prefixed with ``REPORT``. stdin is /dev/null, or with
        ``terminal`` a pseudo-terminal that holds those bytes as typed input.
        fd 3, the installer's saved stdout, is that terminal, or the file
        ``saved_stdout`` as after ``bash install.sh > file``; what the script
        wrote to the terminal is in the attribute ``terminal`` of the result.
        ``icrnl`` False keeps a typed carriage return in the line read.
        """
        defs = (
            'set -euo pipefail\nCURRENT_PHASE=1\nCONDA_ACTIVATED_HERE=false\n'
            + f'CONDA_ENV_REQUESTED={shlex.quote(requested)}\n'
            + _install_sh_blocks("RED='", 'info()', 'REQUIRED_PYTHON_MAJOR=')
            + ''.join(_install_sh_function(f) for f in _CONDA_FUNCTIONS)
            + 'setup_julia() { printf "setup_julia\\n---\\n" >> "${STUB_JULIA_LOG:-/dev/null}"; '
            '[ -z "${STUB_NO_JULIA:-}" ] || exit 1; }\n'
        )
        report = (
            'setup_conda_env\n'
            'printf "REPORT %s\\n" "${CONDA_DEFAULT_ENV:-}" "${CONDA_PREFIX:-}" '
            '"$(command -v python)" "$CONDA_ACTIVATED_HERE" "$(type -t conda)" "$-"\n'
        )
        if terminal is None:
            stdin, master = subprocess.DEVNULL, None
        else:
            try:
                master, stdin = pty.openpty()
            except OSError:
                pytest.skip('this host refuses a pseudo-terminal (pty)')
            if not icrnl:
                attrs = termios.tcgetattr(stdin)
                attrs[0] &= ~termios.ICRNL
                termios.tcsetattr(stdin, termios.TCSANOW, attrs)
            os.write(master, terminal)
            fd3 = '&0' if saved_stdout is None else shlex.quote(str(saved_stdout))
            defs = f'exec 3>{fd3}\n' + defs
        try:
            result = subprocess.run(
                ['bash', '-c', defs + report + after],
                capture_output=True,
                text=True,
                env={**site['env'], **env},
                stdin=stdin,
                timeout=25,
                preexec_fn=_default_signals,
            )
            result.terminal = '' if master is None else _drain(master)
            return result
        finally:
            if master is not None:
                os.close(master)
                os.close(stdin)

    @staticmethod
    def _report(result: subprocess.CompletedProcess) -> list[str]:
        """The six values the script prints after setup_conda_env returns."""
        return [ln[7:] for ln in result.stdout.splitlines() if ln.startswith('REPORT ')]

    @staticmethod
    def _env_lines(result: subprocess.CompletedProcess) -> list[str]:
        """The info lines that name the environment the installer uses."""
        return [ln for ln in result.stdout.splitlines() if 'environment: ' in ln]

    @pytest.mark.parametrize(
        'active',
        [{}, {'CONDA_DEFAULT_ENV': 'base', 'CONDA_PREFIX': '<base>'}],
        ids=['no-env-active', 'base-active'],
    )
    def test_no_env_or_base_makes_proteus_and_activates_it(self, site, active):
        """With no env or base active, no env `proteus` and no folder of it, the env is made once.

        The calls are pinned in order: list, the envs folders, one create with the
        required Python, the base folder, then activate. A substring or path match
        would find `proteus2` or `/elsewhere/proteus` and skip the create. With base
        active the function ends in `proteus`, so nothing goes into base.
        """
        active = {k: v.replace('<base>', str(site['base'])) for k, v in active.items()}
        result = self._call(site, **active)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _conda_calls(site['log']) == [
            ['env', 'list'],
            ['config', '--show', 'envs_dirs'],
            ['create', '-y', '-n', 'proteus', 'python=3.12'],
            ['info', '--base'],
            ['activate', 'proteus'],
        ]
        prefix = site['base'] / 'envs' / 'proteus'
        name, conda_prefix, python, here, kind, flags = self._report(result)
        assert (name, conda_prefix, python) == (
            'proteus',
            str(prefix),
            str(prefix / 'bin/python'),
        )
        assert here == 'true'
        assert kind == 'file'
        assert 'u' in flags
        assert self._env_lines(result) == [
            '\x1b[0;32m[INFO]\x1b[0m  Conda environment: proteus '
            '(made and activated by the installer for this run)'
        ]

    def test_made_env_checks_julia_first(self, site):
        """The Julia check comes before `conda create`; the log is shared with the setup_julia stand-in."""
        result = self._call(site, STUB_JULIA_LOG=str(site['log']))
        assert result.returncode == 0, result.stdout + result.stderr
        calls = _conda_calls(site['log'])
        assert calls.index(['setup_julia']) < calls.index(
            ['create', '-y', '-n', 'proteus', 'python=3.12']
        )

    def test_missing_julia_makes_no_env(self, site):
        """A setup_julia that stops ends the run with no `conda create` and no activation."""
        result = self._call(site, STUB_JULIA_LOG=str(site['log']), STUB_NO_JULIA='1')
        assert result.returncode == 1
        calls = _conda_calls(site['log'])
        assert ['setup_julia'] in calls
        assert [c for c in calls if c[0] in ('create', 'activate')] == []

    @pytest.mark.parametrize(
        ('shell', 'expected'),
        [('/big/x:', '/big/x:'), ('', ''), (None, '/bound:')],
        ids=['value', 'empty', 'unset'],
    )
    def test_activation_keeps_the_depot_of_the_shell(self, site, shell, expected):
        """Conda writes the bound value over the shell value, and the stay command needs that one."""
        self._existing_env(site, 'proteus')
        env = {} if shell is None else {'JULIA_DEPOT_PATH': shell}
        site['env'].pop('JULIA_DEPOT_PATH', None)
        result = self._call(
            site,
            after='printf "DEPOT=[%s]\\n" "${JULIA_DEPOT_PATH-unset}"\n',
            STUB_CONDA_ENVS='proteus',
            STUB_BOUND_DEPOT='/bound:',
            **env,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert f'DEPOT=[{expected}]' in result.stdout

    def test_existing_env_keeps_the_check_for_later(self, site):
        """An existing env can hold PYTHON_JULIAPKG_EXE, so no check runs before it is active."""
        self._existing_env(site, 'proteus')
        result = self._call(
            site, STUB_JULIA_LOG=str(site['log']), STUB_CONDA_ENVS='proteus', STUB_NO_JULIA='1'
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert ['setup_julia'] not in _conda_calls(site['log'])

    def test_whole_script_with_no_julia_makes_no_env(self, site, tmp_path):
        """The shipped script, not the stand-in: no Julia on PATH stops it before `conda create`."""
        root = scratch_proteus_root(tmp_path / 'proteus', [])
        shutil.copy(INSTALL_SH, root / 'install.sh')
        home = tmp_path / 'home'
        home.mkdir()
        result = subprocess.run(
            ['bash', 'install.sh', '--no-data'],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env={**site['env'], 'HOME': str(home)},
            cwd=root,
            timeout=25,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert 'PROTEUS needs Julia 1.13.' in result.stdout
        assert 'Installation failed at Phase 1.' in result.stdout
        assert 'create' not in [c[0] for c in _conda_calls(site['log'])]

    def test_whole_script_with_an_existing_env_stops_in_the_julia_step(self, site, tmp_path):
        """The shipped script, an env that exists and a launcher with no default: Phase 2 stops, nothing else moves.

        juliaup is read and julia gets no call; the env keeps its files, and no conda call changes it.
        """
        prefix = self._existing_env(site, 'proteus')
        before = sorted(prefix.rglob('*'))
        _stub(site['stubs'] / 'julia', _STUB_JULIA)
        _stub(site['stubs'] / 'juliaup', _STUB_JULIAUP)
        _stub(site['stubs'] / 'python3', '#!/bin/sh\necho 3.12\n')
        _stub(site['stubs'] / 'curl', '#!/bin/sh\necho "curl $*" >> "$STUB_LOG"\n')
        for tool in ('gfortran', 'git', 'make', 'cmake', 'unzip', 'nc-config', 'nf-config'):
            _stub(site['stubs'] / tool, '#!/bin/sh\nexit 0\n')
        log = tmp_path / 'calls.log'
        log.write_text('')
        root = scratch_proteus_root(tmp_path / 'proteus', [])
        shutil.copy(INSTALL_SH, root / 'install.sh')
        home = tmp_path / 'home'
        home.mkdir()
        result = subprocess.run(
            ['bash', 'install.sh', '--no-data'],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env={
                **site['env'],
                'HOME': str(home),
                'STUB_LOG': str(log),
                'STUB_JULIA_VERSION': '1.13.1',
                'STUB_BINDIR_113': '',
                'STUB_JULIAUP_CONFIG': _juliaup_config(None),
            },
            cwd=root,
            timeout=25,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert 'PROTEUS needs Julia 1.13.' in result.stdout
        assert 'Installation failed at Phase 2.' in result.stdout
        assert set(log.read_text().splitlines()) == {_JULIAUP_READ}
        assert sorted(prefix.rglob('*')) == before
        assert {'create', 'remove', 'install'}.isdisjoint(
            c[0] for c in _conda_calls(site['log'])
        )

    def test_active_env_keeps_the_check_for_later(self, site):
        """An active env is read in the Julia step, with its variables already set."""
        result = self._call(
            site,
            STUB_JULIA_LOG=str(site['log']),
            STUB_NO_JULIA='1',
            CONDA_DEFAULT_ENV='mine',
            CONDA_PREFIX='/x',
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert not site['log'].exists()

    def test_existing_proteus_is_used_with_a_warning_that_names_the_switch(self, site):
        """Without a terminal an env `proteus` that exists is activated, not made, and named in one warning.

        stdin is /dev/null, as in a script or a CI job: nothing asks, and the
        warning names the env, its prefix and `--conda-env` on stderr, with no
        "stop now", which such a run cannot do; no info line claims the env was
        made. A question without a terminal would read the end of input as no
        and stop.
        """
        prefix = self._existing_env(site, 'proteus')
        result = self._call(site)
        assert result.returncode == 0, result.stdout + result.stderr
        calls = _conda_calls(site['log'])
        assert calls == [['env', 'list'], ['info', '--base'], ['activate', 'proteus']]
        assert not [c for c in calls if c[0] == 'create']
        assert self._report(result)[:4] == [
            'proteus',
            str(prefix),
            str(prefix / 'bin/python'),
            'true',
        ]
        assert self._env_lines(result) == []
        notices = [ln for ln in result.stderr.splitlines() if 'Conda environment: ' in ln]
        assert notices == [
            f'{_WARN_PREFIX}Conda environment: installing INTO the existing environment '
            f'proteus ({prefix}); to use another one run: '
            'bash install.sh --conda-env <other name>'
        ]
        assert 'stop now' not in result.stdout + result.stderr
        assert '[y/N]' not in result.stdout + result.stderr

    @staticmethod
    def _question(prefix: Path) -> str:
        """The question for the existing env `proteus` at ``prefix``, as read prints it."""
        return f'Install PROTEUS INTO the existing conda environment proteus ({prefix})? [y/N] '

    @pytest.mark.parametrize(
        ('answer', 'icrnl'),
        [(b'y\n', True), (b'YES\n', True), (b' \tyes \n', True), (b'y\r\n', False)],
        ids=['y', 'YES', 'yes-in-white-space', 'y-carriage-return'],
    )
    def test_existing_proteus_on_a_terminal_is_used_after_yes(self, site, answer, icrnl):
        """On a terminal the installer asks before it uses an existing `proteus`; y or yes uses it.

        The question goes to the terminal (fd 3), not to stdout or stderr, which
        the installer copies into its log, and comes before `conda info` and the
        activation; after the yes one info line names the env and no warning
        follows. Case, spaces, tabs and a carriage return around the answer do
        not count.
        """
        prefix = self._existing_env(site, 'proteus')
        result = self._call(site, terminal=answer, icrnl=icrnl)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.terminal.count(self._question(prefix)) == 1
        assert '[y/N]' not in result.stdout + result.stderr
        assert _conda_calls(site['log']) == [
            ['env', 'list'],
            ['info', '--base'],
            ['activate', 'proteus'],
        ]
        assert self._report(result)[:2] == ['proteus', str(prefix)]
        assert self._env_lines(result) == [
            '\x1b[0;32m[INFO]\x1b[0m  Conda environment: installing into the existing '
            f'environment proteus ({prefix})'
        ]
        assert '[WARN]' not in result.stderr

    @pytest.mark.parametrize(
        'answer',
        [b'n\n', b'\n', b'\x04', b'nope, another one\n', b'yikes\n', b'y n\n', b'maybe\n'],
        ids=['no', 'empty-line', 'end-of-input', 'other-text', 'yikes', 'y-then-n', 'maybe'],
    )
    def test_existing_proteus_on_a_terminal_stops_without_yes(self, site, answer):
        """n, an empty line, Ctrl-D and every answer other than y or yes stop with status 1.

        Nothing is activated or made: the one conda call is the env list. One
        INFO line names `--conda-env`, and the failure block of a failed step is
        not printed. A default of yes would use the env for the empty line and
        Ctrl-D; a match on the first letter would take `yikes` and `y n`; a
        question after the activation would leave an activate call.
        """
        prefix = self._existing_env(site, 'proteus')
        result = self._call(site, terminal=answer)
        assert result.returncode == 1
        assert result.terminal.count(self._question(prefix)) == 1
        assert _conda_calls(site['log']) == [['env', 'list']]
        assert self._env_lines(result) == []
        assert (
            '\x1b[0;32m[INFO]\x1b[0m  The existing conda environment proteus stays as it is; '
            'to use another one run: bash install.sh --conda-env <other name>'
        ) in result.stdout.splitlines()
        assert '[FAIL]' not in result.stdout + result.stderr
        assert self._report(result) == []
        assert 'Installation failed' not in result.stdout

    def test_conda_env_option_naming_the_existing_env_asks_nothing_on_a_terminal(self, site):
        """`--conda-env proteus` names the env, so a terminal gets no question and the env is used.

        The typed `n` stays unread; a question that ignores the option would read it and stop.
        """
        prefix = self._existing_env(site, 'proteus')
        result = self._call(site, 'proteus', terminal=b'n\n')
        assert result.returncode == 0, result.stdout + result.stderr
        assert '[y/N]' not in result.stdout + result.stderr
        assert ['activate', 'proteus'] in _conda_calls(site['log'])
        assert self._report(result)[:2] == ['proteus', str(prefix)]

    def test_terminal_stdin_with_stdout_in_a_file_takes_the_warning_path(self, site, tmp_path):
        """stdin a terminal and the saved stdout a file, as `bash install.sh > out.log`, asks nothing.

        The prompt would go into the file while the installer waits unseen, so
        the existing env is used with the warning. The typed `n` stays unread:
        the line after the call reads it, and a question would have stopped on it.
        """
        prefix = self._existing_env(site, 'proteus')
        out = tmp_path / 'out.log'
        after = 'read -r unread\necho "UNREAD=$unread"\n'
        result = self._call(site, after=after, terminal=b'n\n', saved_stdout=out)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'UNREAD=n' in result.stdout.splitlines()
        assert ['activate', 'proteus'] in _conda_calls(site['log'])
        assert '[y/N]' not in result.stdout + result.stderr + result.terminal + out.read_text()
        notices = [ln for ln in result.stderr.splitlines() if 'Conda environment: ' in ln]
        assert notices == [
            f'{_WARN_PREFIX}Conda environment: installing INTO the existing environment '
            f'proteus ({prefix}); to use another one run: '
            'bash install.sh --conda-env <other name>'
        ]

    @pytest.mark.parametrize('mark', ['*', '+'], ids=['active-mark', 'frozen-mark'])
    def test_marked_env_line_gives_the_bare_prefix(self, site, mark):
        """A listed `proteus  <mark>  <prefix>` line gives the prefix without the mark.

        conda marks the active env with * and, in newer releases, a frozen env
        with +; a name match that kept the mark would name `<mark>  <prefix>`.
        The warning without a terminal holds the same prefix as the question.
        """
        prefix = self._existing_env(site, 'proteus')
        result = self._call(site, STUB_CONDA_MARK=mark)
        assert result.returncode == 0, result.stdout + result.stderr
        notices = [ln for ln in result.stderr.splitlines() if 'Conda environment: ' in ln]
        assert notices == [
            f'{_WARN_PREFIX}Conda environment: installing INTO the existing environment '
            f'proteus ({prefix}); to use another one run: '
            'bash install.sh --conda-env <other name>'
        ]
        assert f'{mark}  {prefix}' not in result.stderr
        assert self._report(result)[:2] == ['proteus', str(prefix)]

    def test_made_env_on_a_terminal_asks_nothing(self, site):
        """A `proteus` that the installer makes needs no question, also on a terminal."""
        result = self._call(site, terminal=b'n\n')
        assert result.returncode == 0, result.stdout + result.stderr
        assert '[y/N]' not in result.stdout + result.stderr
        assert ['create', '-y', '-n', 'proteus', 'python=3.12'] in _conda_calls(site['log'])
        assert self._report(result)[0] == 'proteus'

    @pytest.mark.parametrize(
        ('listed', 'line'),
        [
            (
                'PROTEUS',
                'The folder <envs>/proteus exists and conda lists it as the environment '
                'PROTEUS; the installer does not make an environment over it. Run bash '
                'install.sh --conda-env PROTEUS to install into it, or --conda-env <name> '
                'for another one.',
            ),
            (
                None,
                'The folder <envs>/proteus exists and conda lists no environment name for '
                'it; the installer does not make an environment over it. Run bash '
                'install.sh --conda-env <name> with another name.',
            ),
        ],
        ids=['env-PROTEUS-listed', 'folder-not-listed'],
    )
    def test_folder_of_the_name_under_another_case_stops_before_create(
        self, site, listed, line
    ):
        """No `conda create -y` runs on a folder `<envs>/proteus` that the name match missed.

        The file system: on macOS `envs/proteus` and `envs/PROTEUS` are one folder, so
        `[ -e envs/proteus ]` is true for the env `PROTEUS`. The test stands in for that
        on any file system by making the folder under the wanted spelling `proteus` as
        well (one folder on macOS, a second one on Linux). conda lists `PROTEUS` or no
        name at all; the stop line names the folder, the listed name and `--conda-env`.
        """
        envs = site['base'] / 'envs'
        if listed:
            self._existing_env(site, listed)
        (envs / 'proteus' / 'bin').mkdir(parents=True, exist_ok=True)
        _stub(envs / 'proteus' / 'bin' / 'python', '#!/bin/sh\nexit 0\n')
        result = self._call(site)
        assert result.returncode == 1
        calls = _conda_calls(site['log'])
        assert calls[:2] == [['env', 'list'], ['config', '--show', 'envs_dirs']]
        assert not [c for c in calls if c[0] in ('create', 'activate')]
        fails = [ln for ln in result.stderr.splitlines() if '[FAIL]' in ln]
        assert fails == [f'\x1b[0;31m[FAIL]\x1b[0m  {line.replace("<envs>", str(envs))}']
        assert (envs / 'proteus' / 'bin' / 'python').exists()
        assert self._report(result) == []

    @pytest.mark.parametrize('requested', ['', 'myenv'], ids=['no-option', 'option-agrees'])
    def test_active_env_is_used_without_any_conda_call(self, site, requested):
        """An active env other than base is used as it is; conda is not called at all.

        A `--conda-env` that names the active env agrees with it. The one line says
        that the installer installs into the active env, and no warning is printed.
        """
        prefix = self._existing_env(site, 'myenv')
        result = self._call(
            site,
            requested,
            CONDA_DEFAULT_ENV='myenv',
            CONDA_PREFIX=str(prefix),
            PATH=f'{prefix / "bin"}{os.pathsep}{site["env"]["PATH"]}',
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert _conda_calls(site['log']) == []
        assert self._report(result)[:4] == [
            'myenv',
            str(prefix),
            str(prefix / 'bin/python'),
            'false',
        ]
        assert self._env_lines(result) == [
            '\x1b[0;32m[INFO]\x1b[0m  Conda environment: myenv '
            '(active; the installer installs into it)'
        ]
        assert '[WARN]' not in result.stderr

    def test_conda_env_option_makes_and_activates_that_env(self, site):
        """`--conda-env sci` with no env active makes and activates `sci`, and no `proteus`."""
        result = self._call(site, 'sci')
        assert result.returncode == 0, result.stdout + result.stderr
        assert _conda_calls(site['log']) == [
            ['env', 'list'],
            ['config', '--show', 'envs_dirs'],
            ['create', '-y', '-n', 'sci', 'python=3.12'],
            ['info', '--base'],
            ['activate', 'sci'],
        ]
        assert self._report(result)[0] == 'sci'
        assert not (site['base'] / 'envs' / 'proteus').exists()

    def test_conda_env_option_disagreeing_with_active_env_stops(self, site):
        """`--conda-env sci` while `myenv` is active stops with one line naming both; nothing is made."""
        result = self._call(site, 'sci', CONDA_DEFAULT_ENV='myenv', CONDA_PREFIX='/envs/myenv')
        assert result.returncode == 1
        fails = [ln for ln in result.stderr.splitlines() if '[FAIL]' in ln]
        assert len(fails) == 1
        assert 'sci' in fails[0] and 'myenv' in fails[0]
        assert self._report(result) == []
        assert not [c for c in _conda_calls(site['log']) if c[0] in ('create', 'activate')]
        assert not (site['base'] / 'envs' / 'sci').exists()

    def test_base_named_by_the_option_stops(self, site):
        """`--conda-env base` stops before conda lists, makes or activates an env.

        The one call left is the `conda list` of die's diagnostics.
        """
        result = self._call(site, 'base')
        assert result.returncode == 1
        assert 'does not install into the conda base environment' in result.stderr
        assert _conda_calls(site['log']) == [['list']]
        assert self._report(result) == []

    def test_missing_conda_stops_with_the_page_that_installs_it(self, site):
        """Without conda on PATH the installer stops with the failed step and step 3 of the page.

        The `conda create` and `conda activate` commands cannot run there, so they
        are not printed; the link names the section that installs Miniforge.
        """
        (site['stubs'] / 'conda').unlink()
        result = self._call(site)
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert not [cmd for cmd in _MANUAL_COMMANDS if cmd in result.stdout.splitlines()]
        assert 'conda create' not in output
        steps = result.stdout.split('Install conda first', 1)[1].splitlines()
        assert steps[0] == ' (Miniforge), as step 3 of the installation page describes:'
        assert steps[1] == (
            '  https://proteus-framework.org/PROTEUS/How-to/installation.html'
            '#3-clone-proteus-and-set-up-python-environment'
        )
        assert 'conda is not on PATH.' in result.stderr
        assert 'Conda environment required.' in result.stderr
        assert self._report(result) == []
        assert not site['log'].exists()

    def test_failed_create_stops_before_activate(self, site):
        """A failing `conda create` stops with the manual commands and the step; activate is not tried.

        One line names the terms of service of the default channels, which conda
        can refuse to use until they are accepted; the manual `conda create`
        alone would meet the same refusal.
        """
        result = self._call(site, STUB_CREATE_RC='1')
        assert result.returncode == 1
        assert all(cmd in result.stdout.splitlines() for cmd in _MANUAL_COMMANDS)
        assert result.stdout.splitlines().count(_TOS_LINE) == 1
        assert "'conda create -y -n proteus python=3.12' failed." in result.stderr
        calls = _conda_calls(site['log'])
        assert ['create', '-y', '-n', 'proteus', 'python=3.12'] in calls
        assert not [c for c in calls if c[0] in ('activate', 'info')]
        assert self._report(result) == []

    @pytest.mark.parametrize(
        ('env', 'python', 'step'),
        [
            ({'STUB_ACTIVATE_RC': '1'}, True, "'conda activate proteus' failed."),
            ({'STUB_ACTIVATE_AS': 'other'}, True, 'left the env other active'),
            ({}, False, 'and python at <stubs>/python.'),
            ({'STUB_ACTIVATE_RC': '1'}, None, "'conda activate proteus' failed."),
        ],
        ids=[
            'activate-fails',
            'activate-leaves-other-env',
            'python-outside-the-env',
            'activate-of-the-made-env-fails',
        ],
    )
    def test_activation_that_misses_the_env_stops(self, site, env, python, step):
        """An activation that fails, leaves another env, or leaves a python outside the env stops.

        In the third case CONDA_DEFAULT_ENV is right, so only the python check
        catches it. The env exists in each case, listed or made by the
        installer, so only `conda activate` is printed: a `conda create -n
        proteus` there would ask to remove that env.
        """
        if python is not None:
            self._existing_env(site, 'proteus', python=python)
        result = self._call(site, **env)
        assert result.returncode == 1
        assert step.replace('<stubs>', str(site['stubs'])) in result.stderr
        steps = result.stdout.split('Activate the conda environment first:\n', 1)[1]
        assert steps.splitlines()[0] == '  conda activate proteus'
        assert 'conda create' not in result.stdout
        assert (site['base'] / 'envs' / 'proteus' / 'bin' / 'python').exists() is (
            python is not False
        )
        assert _TOS_LINE not in result.stdout
        assert ['activate', 'proteus'] in _conda_calls(site['log'])
        assert self._report(result) == []

    def test_pixi_env_makes_no_conda_call(self, site):
        """In a pixi env the function calls no conda, even without CONDA_DEFAULT_ENV."""
        result = self._call(site, PIXI_ENVIRONMENT_NAME='default')
        assert result.returncode == 0, result.stdout + result.stderr
        assert not site['log'].exists()
        assert self._report(result)[:4] == ['', '', str(site['stubs'] / 'python'), 'false']
        assert self._env_lines(result) == ['\x1b[0;32m[INFO]\x1b[0m  Pixi environment: default']

    def test_unset_variable_in_activation_passes_and_nounset_returns(self, site):
        """The stub's activate reads an unset variable; the function passes, and set -u is on after it.

        The control runs the same activate under set -u without the window, and it fails.
        """
        result = self._call(site)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'SOME_UNSET_NAME' not in site['env']
        assert 'u' in self._report(result)[5]
        assert 'e' in self._report(result)[5]
        conda_sh = site['base'] / 'etc' / 'profile.d' / 'conda.sh'
        control = subprocess.run(
            [
                'bash',
                '-c',
                f'set -euo pipefail\nsource {shlex.quote(str(conda_sh))}\nconda activate proteus\necho DONE\n',
            ],
            capture_output=True,
            text=True,
            env=site['env'],
            timeout=10,
        )
        assert control.returncode != 0
        assert 'SOME_UNSET_NAME' in control.stderr
        assert 'DONE' not in control.stdout

    @pytest.mark.parametrize(
        ('own_env', 'bind', 'line'),
        [
            (False, 'bound', '     conda activate proteus'),
            (True, 'bound', '     conda deactivate && conda activate myenv'),
            (False, 'none', '     conda activate proteus && source .proteus-env.sh'),
            (True, 'none', '     source .proteus-env.sh'),
        ],
        ids=['made-env-bound', 'own-env-bound', 'made-env-not-bound', 'own-env-not-bound'],
    )
    def test_closing_instruction_follows_who_activated_the_env(self, site, own_env, bind, line):
        """The env the installer activated needs only `conda activate`; the user's own needs both.

        The user's shell has no env active after the installer activated one in its
        own process, so a `conda deactivate` there would leave base, not reload the env.
        """
        env = {}
        if own_env:
            prefix = self._existing_env(site, 'myenv')
            env = {'CONDA_DEFAULT_ENV': 'myenv', 'CONDA_PREFIX': str(prefix)}
        after = f'SCRIPT_DIR=/p\nCONDA_BIND={bind}\nprint_next_steps\n'
        result = self._call(site, after=after, **env)
        assert result.returncode == 0, result.stdout + result.stderr
        steps = result.stdout.split('Next steps:\n', 1)[1].splitlines()
        assert steps[1] == line
        assert ('conda deactivate' in result.stdout) == (own_env and bind == 'bound')
        # The env the installer activated needs conda's shell functions in a shell without conda init.
        conda_sh = f'source "{site["base"]}/etc/profile.d/conda.sh"'
        assert (
            steps[2] == f"     (In a shell without 'conda init', run first: {conda_sh})"
        ) is (not own_env)
        assert result.stdout.count('profile.d/conda.sh') == (0 if own_env else 1)

    @pytest.mark.parametrize('own_env', [False, True], ids=['made-env', 'own-env'])
    def test_failed_run_names_the_env_to_activate(self, site, own_env):
        """A run that stops after the installer activated its env names `conda activate proteus`.

        After a failure with the user's own env active the hint has no conda line,
        because no variable was bound to the env yet.
        """
        env = {}
        if own_env:
            prefix = self._existing_env(site, 'myenv')
            env = {'CONDA_DEFAULT_ENV': 'myenv', 'CONDA_PREFIX': str(prefix)}
        after = 'INSTALL_VAR_NAMES=""\ntrap print_exit_hint EXIT\nexit 3\n'
        result = self._call(site, after=after, **env)
        assert result.returncode == 3
        hint = "To use it in your shell, run 'conda activate proteus'."
        assert (hint in result.stdout) == (not own_env)
        assert 'conda deactivate' not in result.stdout
        assert len(self._report(result)) == 6


class TestAskYn:
    """ask_yn writes its question to fd 3, the installer's saved terminal, and takes only y or yes."""

    @pytest.mark.parametrize(
        ('answer', 'default', 'yes'),
        [
            ('y\n', 'n', True),
            ('YES\n', 'n', True),
            (' \tYes \r\n', 'n', True),
            ('y\r\n', 'n', True),
            ('yikes\n', 'n', False),
            ('y n\n', 'n', False),
            ('maybe\n', 'n', False),
            ('\n', 'n', False),
            ('', 'n', False),
            ('\n', 'y', True),
            ('', 'y', True),
            ('yikes\n', 'y', False),
        ],
        ids=[
            'y',
            'YES',
            'yes-in-white-space',
            'y-carriage-return',
            'yikes',
            'y-then-n',
            'maybe',
            'empty-default-no',
            'end-of-input-default-no',
            'empty-default-yes',
            'end-of-input-default-yes',
            'yikes-default-yes',
        ],
    )
    def test_only_y_or_yes_is_yes(self, tmp_path, answer, default, yes):
        """y or yes in any case, after a trim of spaces, tabs and a carriage return, is yes.

        Every other text is no, also with the default yes; an empty answer and
        the end of input take the default. stdin is a pipe, so the answer keeps
        its carriage return as a terminal without ICRNL passes it. The question
        and its hint are in the fd 3 file only: stdout and stderr go to the log.
        """
        out = tmp_path / 'fd3.txt'
        script = (
            'set -euo pipefail\n'
            + _install_sh_function('ask_yn')
            + f'exec 3>{shlex.quote(str(out))}\n'
            + f'if ask_yn "Go on?" {default}; then echo YES; else echo NO; fi\n'
        )
        result = subprocess.run(
            ['bash', '-c', script],
            input=answer,
            capture_output=True,
            text=True,
            env={'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path)},
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ['YES' if yes else 'NO']
        assert out.read_text() == f'Go on? {"[Y/n]" if default == "y" else "[y/N]"} '
        assert result.stderr == ''


class TestCondaEnvOption:
    """install.sh takes `--conda-env <name>`, lists it in --help, and passes the name on."""

    def test_help_lists_the_option(self):
        """--help exits 0 and has one line for --conda-env with its argument."""
        result = subprocess.run(
            ['bash', str(INSTALL_SH), '--help'], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        lines = [ln for ln in result.stdout.splitlines() if ln.startswith('  --conda-env ')]
        assert len(lines) == 1
        assert '<name>' in lines[0] and 'proteus' in result.stdout

    @pytest.mark.parametrize(
        'args',
        [['--conda-env'], ['--conda-env', '--help'], ['--conda-env', '']],
        ids=['no-value', 'option-as-value', 'empty-value'],
    )
    def test_option_without_a_name_stops(self, args):
        """`--conda-env` without a name stops at parsing; a following option is not taken as the name."""
        result = subprocess.run(
            ['bash', str(INSTALL_SH), *args], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 1
        assert '--conda-env needs the name of a conda environment.' in result.stderr
        assert 'Usage' not in result.stdout

    def test_option_value_is_consumed(self):
        """`--conda-env sci --help` exits 0, so `sci` is the option's value and not an unknown argument."""
        result = subprocess.run(
            ['bash', str(INSTALL_SH), '--conda-env', 'sci', '--help'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'Unknown argument' not in result.stdout + result.stderr
        assert 'Usage' in result.stdout

    def test_option_reaches_setup_conda_env(self):
        """The whole installer with `--conda-env sci` and `myenv` active stops at the disagreement.

        This runs pre-flight up to the env step, so the name the parser stores is
        the one setup_conda_env reads.
        """
        if not _process_substitution_writable():
            pytest.skip(
                'bash cannot write to a process substitution on this host (/dev/fd refused)'
            )
        env = {k: v for k, v in os.environ.items() if k != 'PIXI_ENVIRONMENT_NAME'}
        env.update(CONDA_DEFAULT_ENV='myenv', CONDA_PREFIX='/envs/myenv')
        result = subprocess.run(
            ['bash', str(INSTALL_SH), '--conda-env', 'sci'],
            capture_output=True,
            text=True,
            timeout=25,
            env=env,
            cwd=str(PROTEUS_ROOT),
            stdin=subprocess.DEVNULL,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert '--conda-env sci and the active conda environment myenv disagree' in output
        assert 'Installation failed at Phase 1.' in output


# 3 lines on stdout and 2 on stderr, interleaved, then the exit status given as $1.
_FIVE_LINES = (
    'echo out-1; echo err-1 >&2; echo out-2; echo err-2 >&2; echo out-3; exit "${1:-0}"\n'
)
_FIVE = ['out-1', 'err-1', 'out-2', 'err-2', 'out-3']

# 100 numbered lines and exit status 7: the last 40 are lines 61 to 100.
_HUNDRED_LINES = 'i=1\nwhile [ "$i" -le 100 ]; do echo "line $i"; i=$((i + 1)); done\nexit 7\n'


def _process_substitution_writable() -> bool:
    """Whether bash here can write to a process substitution, as the installer's log tee does.

    The probe runs no installer code, so a skip on it hides no failure of run_logged.
    """
    probe = subprocess.run(
        ['bash', '-c', 'exec 5> >(cat >/dev/null) && echo ok'],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return probe.returncode == 0 and probe.stdout.strip() == 'ok'


def _run_logged(
    tmp_path: Path,
    script: str,
    *,
    logfile: Path | None,
    tee: bool = False,
    verbose: bool = False,
    stdin: str | None = None,
) -> subprocess.CompletedProcess:
    """Run ``script`` after the shipped run_logged, in ``tmp_path`` with five.sh and hundred.sh.

    ``tee`` adds the installer's own block that starts the log tee; ``logfile`` None
    leaves LOGFILE unset.
    """
    if tee and not _process_substitution_writable():
        pytest.skip(
            'bash cannot write to a process substitution on this host (/dev/fd refused)'
        )
    (tmp_path / 'five.sh').write_text(_FIVE_LINES)
    (tmp_path / 'hundred.sh').write_text(_HUNDRED_LINES)
    defs = 'set -euo pipefail\n'
    if logfile is not None:
        defs += f'LOGFILE={shlex.quote(str(logfile))}\n'
    if verbose:
        defs += 'VERBOSE=true\n'
    defs += (
        _install_sh_blocks("RED='", 'info()')
        + ''.join(_install_sh_function(f) for f in _RUN_LOGGED_FUNCTIONS)
        + _RUN_LOGGED_CLEANUP
    )
    if tee:
        defs += _install_sh_blocks('exec 3>&1')
    return subprocess.run(
        ['bash', '-c', defs + script],
        capture_output=True,
        text=True,
        input=stdin,
        cwd=tmp_path,
        env=_env_without_log_state(),
        timeout=25,
    )


def _lines(result: subprocess.CompletedProcess) -> list[str]:
    """The terminal of a run: its stdout lines, then its stderr lines."""
    return result.stdout.splitlines() + result.stderr.splitlines()


class TestRunLogged:
    """run_logged shows one line per step and keeps the step's output in the log file."""

    def test_output_goes_to_the_log_only(self, tmp_path):
        """A step that succeeds shows a start and an end line; its 5 lines are in the log only.

        Two steps run: the first start line names the log file and the second
        does not, and the log holds both outputs in order, each once.
        """
        log = tmp_path / 'install.log'
        script = (
            'run_logged "Building widget" bash five.sh\n'
            'run_logged --takes "a minute" "Building gadget" bash five.sh\n'
        )
        result = _run_logged(tmp_path, script, logfile=log)
        assert result.returncode == 0, result.stdout + result.stderr
        terminal = _lines(result)
        assert [ln for ln in terminal if ln in _FIVE] == []
        starts = [ln for ln in terminal if 'its output goes to' in ln]
        assert len(starts) == 2
        assert starts[0].endswith(f'Building widget; its output goes to {log}')
        assert starts[1].endswith(
            'Building gadget (can take a minute); its output goes to the log file'
        )
        assert re.search(r'Building widget: done in \d+ s$', result.stdout, re.M)
        assert re.search(r'Building gadget: done in \d+ s$', result.stdout, re.M)
        assert log.read_text().splitlines() == _FIVE + _FIVE

    def test_failure_shows_the_last_40_lines_and_returns_the_status(self, tmp_path):
        """A step that exits 7 returns 7 and shows lines 61 to 100 and the log file path.

        An off-by-one tail (39 or 41 lines) or the head of the output fails the
        list equality. A bare call under errexit stops the script with that status.
        """
        log = tmp_path / 'install.log'
        script = (
            'if run_logged "Building widget" bash hundred.sh; then echo RC=0; '
            'else echo "RC=$?"; fi\n'
        )
        result = _run_logged(tmp_path, script, logfile=log)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'RC=7' in result.stdout.splitlines()
        shown = [ln for ln in _lines(result) if ln.startswith('line ')]
        assert shown == [f'line {i}' for i in range(61, 101)]
        assert re.search(
            r'Building widget: failed with exit status 7 after \d+ s', result.stderr
        )
        assert f'The full output of the installer is in {log}' in result.stderr.splitlines()
        assert log.read_text().splitlines() == [f'line {i}' for i in range(1, 101)]

        bare = _run_logged(
            tmp_path, 'run_logged "Building widget" bash hundred.sh\necho AFTER\n', logfile=log
        )
        assert bare.returncode == 7
        assert 'AFTER' not in bare.stdout

    @pytest.mark.parametrize('status', [0, 3], ids=['step-succeeds', 'step-fails'])
    def test_log_tee_keeps_each_line_once_and_in_order(self, tmp_path, status):
        """Under the installer's log tee, each log line is there once, in the order written.

        The start line precedes the step's output and the end line follows it.
        A failed step shows its 5 lines on the terminal without writing them
        to the log a second time.
        """
        log = tmp_path / 'install.log'
        script = (
            f'if run_logged "Building widget" bash five.sh {status}; then echo STEP-OK; '
            'else echo "STEP-RC=$?"; fi\n'
        )
        result = _run_logged(tmp_path, script, logfile=log, tee=True)
        assert result.returncode == 0, result.stdout + result.stderr
        logged = log.read_text().splitlines()
        assert len(logged) == len(set(logged)), logged
        assert [ln for ln in logged if ln in _FIVE] == _FIVE
        start = next(i for i, ln in enumerate(logged) if 'Building widget; its output' in ln)
        end = next(i for i, ln in enumerate(logged) if 'Building widget: ' in ln)
        assert start < logged.index('out-1') < logged.index('out-3') < end
        shown = [ln for ln in _lines(result) if ln in _FIVE]
        assert shown == ([] if status == 0 else _FIVE)
        assert ('STEP-OK' if status == 0 else f'STEP-RC={status}') in logged

    @pytest.mark.parametrize('tee', [False, True], ids=['no-log-tee', 'log-tee'])
    def test_verbose_shows_and_logs_every_line(self, tmp_path, tee):
        """With --verbose the 5 lines reach the terminal and the log, each once, in order."""
        log = tmp_path / 'install.log'
        result = _run_logged(
            tmp_path,
            'run_logged "Building widget" bash five.sh\n',
            logfile=log,
            tee=tee,
            verbose=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert [ln for ln in _lines(result) if ln in _FIVE] == _FIVE
        assert [ln for ln in log.read_text().splitlines() if ln in _FIVE] == _FIVE
        assert re.search(r'Building widget: done in \d+ s', result.stdout)

    @pytest.mark.parametrize('where', ['unset', 'unwritable'])
    def test_without_a_writable_log_the_command_runs_as_is(self, tmp_path, where):
        """With LOGFILE unset, or in a missing directory, the output stays on the terminal.

        The exit status 3 comes back to the caller and no start line is printed.
        """
        logfile = None if where == 'unset' else tmp_path / 'missing' / 'install.log'
        script = (
            'if run_logged "Building widget" bash five.sh 3; then echo RC=0; '
            'else echo "RC=$?"; fi\n'
        )
        result = _run_logged(tmp_path, script, logfile=logfile)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.splitlines() == ['out-1', 'out-2', 'out-3', 'RC=3']
        assert result.stderr.splitlines() == ['err-1', 'err-2']
        assert not (tmp_path / 'missing').exists()

    def test_hidden_step_reads_no_answer_from_stdin(self, tmp_path):
        """A logged step gets end of input, so a question in it cannot wait unseen.

        The test feeds `yes` on stdin; a step that inherited it would log `got:yes`.
        """
        log = tmp_path / 'install.log'
        script = """run_logged "Asking" bash -c 'read -r x || x=EOF; echo "got:$x"'\n"""
        result = _run_logged(tmp_path, script, logfile=log, stdin='yes\n')
        assert result.returncode == 0, result.stdout + result.stderr
        assert log.read_text().splitlines() == ['got:EOF']
        assert 'got:' not in result.stdout + result.stderr


def _default_signals() -> None:
    """Give the child the default action for HUP, INT and TERM, and unblock them: a test
    runner started with ``nohup``, in the background or by a CI step can ignore or block them."""
    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, [sig])


def _run_logged_with_exit_hint(tmp_path: Path, script: str) -> subprocess.CompletedProcess:
    """Run ``script`` after run_logged and print_exit_hint, trapped on EXIT as in install.sh.

    TMPDIR is the empty folder ``tmp_path / 'tmpdir'``, where run_logged makes its temp file.
    """
    log = tmp_path / 'install.log'
    log.write_text('')
    (tmp_path / 'tmpdir').mkdir()
    defs = (
        'set -euo pipefail\nINSTALL_VAR_NAMES=""\n'
        + f'LOGFILE={shlex.quote(str(log))}\n'
        + _install_sh_blocks("RED='", 'info()')
        + ''.join(_install_sh_function(f) for f in (*_RUN_LOGGED_FUNCTIONS, 'print_exit_hint'))
        + 'trap print_exit_hint EXIT\n'
    )
    return subprocess.run(
        ['bash', '-c', defs + script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**_env_without_log_state(), 'TMPDIR': str(tmp_path / 'tmpdir')},
        timeout=25,
        start_new_session=True,
        preexec_fn=_default_signals,
    )


class TestRunLoggedTempFile:
    """run_logged keeps one temp file per installer run, which the exit trap removes."""

    def test_interrupted_step_leaves_no_temp_file(self, tmp_path):
        """A step that sends INT to the installer, as Ctrl-C does, leaves no proteus-install.* file.

        The step records the TMPDIR listing first, so the file was there while it
        ran; the run stops at the interrupt, before the line after the step.
        """
        (tmp_path / 'interrupt.sh').write_text(
            'ls "$TMPDIR" > seen.txt\nkill -INT "$PPID"\nkill -INT $$\nsleep 5\n'
        )
        result = _run_logged_with_exit_hint(
            tmp_path, 'run_logged "Building widget" bash interrupt.sh\necho AFTER\n'
        )
        assert result.returncode != 0
        assert 'AFTER' not in result.stdout
        seen = (tmp_path / 'seen.txt').read_text().split()
        assert len(seen) == 1 and seen[0].startswith('proteus-install.')
        assert list((tmp_path / 'tmpdir').iterdir()) == []

    def test_steps_share_one_temp_file_and_a_clean_exit_removes_it(self, tmp_path):
        """2 steps use 1 temp file, emptied between them; a clean exit removes it.

        The second step shows only its own stay line: a file that kept the first
        step's output would show `stays as it is: first` again.
        """
        script = (
            'run_logged "Step one" bash -c "echo \'stays as it is: first\'"\n'
            'run_logged "Step two" bash -c "echo \'stays as it is: second\'"\n'
            'ls "$TMPDIR" > listing.txt\n'
        )
        result = _run_logged_with_exit_hint(tmp_path, script)
        assert result.returncode == 0, result.stdout + result.stderr
        listing = (tmp_path / 'listing.txt').read_text().split()
        assert len(listing) == 1 and listing[0].startswith('proteus-install.')
        shown = [ln for ln in _lines(result) if ln.startswith('stays as it is')]
        assert shown == ['stays as it is: first', 'stays as it is: second']
        assert list((tmp_path / 'tmpdir').iterdir()) == []


# The build steps whose output goes to the log only, and the commands that stay on the terminal.
_BUILD_TARGETS = {
    'tools/get_socrates.sh',
    'tools/get_agni.sh',
    'src/get_fastchem.sh',
    'tools/get_python_module.sh',
    'tools/get_aragog.sh',
    'tools/get_zalmoxis.sh',
    'tools/get_cvode.sh',
    'pip install -e',
}
_TERMINAL_WORDS = {'proteus', 'read', 'read_with_default', 'juliaup', 'curl'}
_SEPARATORS = {'||', '&&', ';', ';;', '|', '&', '(', ')'}
_KEYWORDS = {'if', 'elif', 'then', 'else', 'do', 'while', 'until', '!'}


def _shell_commands(text: str) -> list[list[str]]:
    """The simple commands of a shell text, as word lists, without comment lines.

    Continued lines are joined first; a line shlex cannot split is returned whole.
    """
    joined = re.sub(r'\\\n', ' ', text)
    commands = []
    for line in joined.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            commands.append([line])
            continue
        current: list[str] = []
        for tok in [*tokens, ';']:
            if tok in _SEPARATORS:
                while current and current[0] in _KEYWORDS:
                    current.pop(0)
                if current:
                    commands.append(current)
                current = []
            else:
                current.append(tok)
    return commands


def _build_target(words: list[str]) -> str | None:
    """The build step a command runs: a _BUILD_TARGETS script after `bash`, or `pip install -e`."""
    for i, word in enumerate(words):
        if word == 'build_fastchem':
            return 'src/get_fastchem.sh'
        if word == 'bash' and i + 1 < len(words):
            hit = [t for t in _BUILD_TARGETS if words[i + 1].endswith(t)]
            if hit:
                return hit[0]
        if words[i : i + 3] == ['pip', 'install', '-e']:
            return 'pip install -e'
    return None


def _logged_and_terminal_calls(text: str) -> tuple[list[tuple[str, str]], list[list[str]]]:
    """The build commands as (first word, step), and the commands meant for the terminal.

    The second list holds each command that starts with a word of _TERMINAL_WORDS,
    or that run_logged wraps around one of them.
    """
    builds, terminal = [], []
    for words in _shell_commands(text):
        target = _build_target(words)
        if target:
            builds.append((words[0], target))
        if words[0] in _TERMINAL_WORDS or (
            words[0] == 'run_logged' and _TERMINAL_WORDS & set(words)
        ):
            terminal.append(words)
    return builds, terminal


class TestRunLoggedCalls:
    """install.sh logs each build step and leaves downloads and questions on the terminal."""

    def test_build_steps_go_through_run_logged_and_downloads_do_not(self):
        """Every build command of install.sh is run by run_logged; no download or question is.

        The 8 build steps are pinned as a set, and the 11 data downloads as a
        count, so a scan that finds nothing fails. The synthetic text is the
        control: a bare build and a logged download are both reported.
        """
        # build_fastchem wraps AGNI's script and is itself the logged step
        text = INSTALL_SH.read_text().replace(_install_sh_function('build_fastchem'), '')
        builds, terminal = _logged_and_terminal_calls(text)
        assert {target for _, target in builds} == _BUILD_TARGETS
        assert [b for b in builds if b[0] != 'run_logged'] == []
        # the install of PROTEUS itself; the module scripts install their own clones
        assert sum(1 for _, t in builds if t == 'pip install -e') == 1
        assert [w for w in terminal if w[0] == 'run_logged'] == []
        assert len([w for w in terminal if w[:2] == ['proteus', 'get']]) == 11
        assert [w[:2] for w in terminal if w[:2] == ['proteus', 'doctor']] == [
            ['proteus', 'doctor']
        ]

        control = (
            'bash tools/get_socrates.sh\n'
            'run_logged "Spectra" proteus get spectral 2>&1 || warn "x"\n'
            'warn "re-run bash tools/get_cvode.sh"\n'
        )
        c_builds, c_terminal = _logged_and_terminal_calls(control)
        assert c_builds == [('bash', 'tools/get_socrates.sh')]
        assert [w[0] for w in c_terminal] == ['run_logged']


def _run_logged_statements() -> list[str]:
    """Each statement of install.sh that calls run_logged, with its continued lines joined."""
    text = re.sub(r'\\\n\s*', ' ', INSTALL_SH.read_text())
    return [ln.strip() for ln in text.splitlines() if re.match(r'\s*(if ! )?run_logged ', ln)]


# The build steps that stop the installer, and the sentence that names each.
_STOPPING_STEPS = {
    'Building SOCRATES': 'Could not build SOCRATES (bash tools/get_socrates.sh)',
    'Installing AGNI': 'Could not install AGNI and its Julia packages (bash tools/get_agni.sh 0)',
    'Setting up Aragog': 'Could not set up Aragog (bash tools/get_aragog.sh)',
    'Setting up Zalmoxis': 'Could not set up Zalmoxis (bash tools/get_zalmoxis.sh)',
    'Installing PROTEUS': (
        'Could not install PROTEUS and its remaining dependencies (pip install -e ".[develop]")'
    ),
}


class TestFailedStepsEndInDie:
    """A failed build step ends in die, with the phase, the diagnostics and the log text."""

    def test_every_logged_step_ends_in_die_warn_or_true(self):
        """Each run_logged statement is in an `if !`, or ends in `|| die`, `|| warn` or `|| true`.

        Under set -e a bare failing step ends the run with no failure block. The 11
        statements are counted, so a scan that finds none fails.
        """
        statements = _run_logged_statements()
        assert len(statements) == 11
        bare = [
            s
            for s in statements
            if not s.startswith('if ! run_logged ')
            and not re.search(r'\|\| (die "|warn "|true$)', s)
        ]
        assert bare == []
        for step, sentence in _STOPPING_STEPS.items():
            (hit,) = [s for s in statements if f'"{step}' in s]
            assert shlex.split(hit.split('|| die ', 1)[1]) == [
                f'{sentence}; see the error above'
            ]

    @pytest.mark.parametrize('step', sorted(_STOPPING_STEPS))
    def test_failed_step_prints_the_failure_block(self, tmp_path, step):
        """The shipped statement of each step, with its script or pip failing, ends in die.

        The output names the phase, the step and the log file to send, and the
        exit status is die's 1, not the step's 3.
        """
        (statement,) = [s for s in _run_logged_statements() if f'"{step}' in s]
        tools = tmp_path / 'tools'
        for name in ('get_socrates.sh', 'get_agni.sh', 'get_aragog.sh', 'get_zalmoxis.sh'):
            _stub(tools / name, '#!/bin/sh\necho "step output"\nexit 3\n')
        stubs = tmp_path / 'stubs'
        _stub(stubs / 'pip', '#!/bin/sh\necho "pip output"\nexit 3\n')
        log = tmp_path / 'install.log'
        log.write_text('')
        defs = (
            'set -euo pipefail\nCURRENT_PHASE=6\nINSTALL_VAR_NAMES=""\n'
            + f'LOGFILE={shlex.quote(str(log))}\n'
            + _install_sh_blocks("RED='", 'info()')
            + ''.join(
                _install_sh_function(f)
                for f in ('command_exists', 'collect_env_info', 'die', *_RUN_LOGGED_FUNCTIONS)
            )
            + _RUN_LOGGED_CLEANUP
        )
        env = _env_without_log_state()
        env['PATH'] = f'{stubs}{os.pathsep}{env["PATH"]}'
        result = subprocess.run(
            ['bash', '-c', defs + statement + '\necho AFTER\n'],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            timeout=25,
        )
        assert result.returncode == 1
        assert 'AFTER' not in result.stdout
        fails = [ln for ln in result.stderr.splitlines() if '[FAIL]' in ln]
        assert fails[-1] == (
            f'\x1b[0;31m[FAIL]\x1b[0m  {_STOPPING_STEPS[step]}; see the error above'
        )
        assert 'Installation failed at Phase 6.' in result.stdout
        assert 'send that log file to dev@proteus-framework.org' in result.stdout


class TestFastChemPhase:
    """Phase 5 counts a FastChem as installed only when its program is executable."""

    @pytest.mark.parametrize(
        ('mode', 'built'), [(0o755, False), (0o644, True)], ids=['executable', 'not-executable']
    )
    def test_fastchem_without_an_executable_program_is_built_again(self, tmp_path, mode, built):
        """A fastchem/fastchem without its mode is not taken as installed; the build runs.

        The same test (-x) decides FC_DIR, so a program that FC_DIR skips is
        not reported as installed either.
        """
        fastchem = tmp_path / 'AGNI' / 'fastchem'
        _stub(fastchem / 'fastchem', '#!/bin/sh\n').chmod(mode)
        block = _install_sh_blocks('AGNI_REAL=$(cd')
        assert block.count('FastChem already installed') == 1
        defs = (
            'set -euo pipefail\n'
            + f'SCRIPT_DIR={shlex.quote(str(tmp_path))}\n'
            + _install_sh_blocks("RED='", 'info()')
            + 'run_logged() { echo "BUILD $*"; }\n'
        )
        result = subprocess.run(
            ['bash', '-c', defs + block],
            capture_output=True,
            text=True,
            env={'PATH': '/usr/bin:/bin'},
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert ('BUILD Building FastChem build_fastchem' in result.stdout) is built
        assert ('FastChem already installed' in result.stdout) is (not built)


class TestLogFileSentence:
    """die, run_logged and the exit hint name the log file in one sentence, once per failure."""

    @pytest.mark.parametrize(
        ('script', 'status', 'count'),
        [
            ('die "boom"\n', 1, 1),
            ('exit 3\n', 3, 1),
            ('run_logged "Building widget" bash five.sh 4\n', 4, 1),
            ('run_logged "Building widget" bash five.sh 4 || die "boom"\n', 1, 1),
            ('true\n', 0, 0),
        ],
        ids=['die', 'plain-exit', 'failed-step', 'failed-step-then-die', 'clean-exit'],
    )
    def test_failed_run_names_the_log_file_once(self, tmp_path, script, status, count):
        """A failed run prints the sentence once, from whichever of the three ends it.

        die and run_logged print it and the exit hint then holds it back; die
        after a failed run_logged step, as each stopping build step ends, holds
        it back too. A clean exit prints nothing.
        """
        log = tmp_path / 'install.log'
        log.write_text('')
        (tmp_path / 'five.sh').write_text(_FIVE_LINES)
        defs = (
            'set -euo pipefail\nCURRENT_PHASE=6\nINSTALL_VAR_NAMES=""\n'
            + f'LOGFILE={shlex.quote(str(log))}\n'
            + _install_sh_blocks("RED='", 'info()')
            + ''.join(
                _install_sh_function(f)
                for f in (
                    'command_exists',
                    'collect_env_info',
                    'die',
                    'print_exit_hint',
                    *_RUN_LOGGED_FUNCTIONS,
                )
            )
            + 'trap print_exit_hint EXIT\n'
        )
        result = subprocess.run(
            ['bash', '-c', defs + script],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=_env_without_log_state(),
            timeout=25,
        )
        assert result.returncode == status
        sentence = f'The full output of the installer is in {log}'
        assert _lines(result).count(sentence) == count


class TestFurtherModulesPhase:
    """The Further modules phase installs Morrigan, VULCAN, atmodeller, and Obliqua."""

    MODULES = ('Morrigan', 'VULCAN', 'atmodeller', 'Obliqua')
    SCRIPTS = (
        'get_morrigan.sh',
        'get_vulcan.sh',
        'get_atmodeller.sh',
        'get_obliqua.sh',
    )

    @pytest.fixture
    def site(self, tmp_path):
        """Prepare a scratch directory with stub setup scripts and log setup."""
        scratch = tmp_path / 'proteus'
        tools = scratch / 'tools'
        tools.mkdir(parents=True)
        log = scratch / 'calls.log'
        log.write_text('')
        for name, script in zip(self.MODULES, self.SCRIPTS, strict=True):
            body = (
                '#!/bin/bash\n'
                f'echo "{name} $0 PROTEUS_CLONE_STAYS=${{PROTEUS_CLONE_STAYS:-}}" >> '
                f'{shlex.quote(str(log))}\n'
                'exit 0\n'
            )
            stub = tools / script
            stub.write_text(body)
            stub.chmod(0o755)
        tmp_dir = scratch / 'tmp'
        tmp_dir.mkdir()
        install_log = scratch / 'install.log'
        install_log.write_text('')
        stubs = scratch / 'bin_stubs'
        stubs.mkdir()
        py_log = scratch / 'python.log'
        py_log.write_text('')
        stub_py = stubs / 'python3'
        stub_py.write_text(f'#!/bin/sh\necho "$@" >> {shlex.quote(str(py_log))}\nexit 0\n')
        stub_py.chmod(0o755)
        env = {
            **_env_without_log_state(),
            'PATH': f'{stubs}{os.pathsep}{os.environ["PATH"]}',
            'SCRIPT_DIR': str(scratch),
            'TMPDIR': str(tmp_dir),
            'LOGFILE': str(install_log),
        }
        for var in ('FAILED_MODULES', 'PROTEUS_CLONE_STAYS', 'FURTHER_STAYS'):
            env.pop(var, None)
        return {
            'scratch': scratch,
            'tools': tools,
            'calls_log': log,
            'install_log': install_log,
            'tmp_dir': tmp_dir,
            'python_log': py_log,
            'stubs': stubs,
            'env': env,
        }

    @staticmethod
    def _phase_block() -> str:
        """The shipped lines of install.sh for the further modules phase."""
        lines = INSTALL_SH.read_text().splitlines()
        start = next(
            i for i, ln in enumerate(lines) if ln == 'FAILED_MODULES="${FAILED_MODULES:-}"'
        )
        end = (
            next(
                i
                for i, ln in enumerate(lines)
                if i > start and ln.startswith('# Phase 8: Reference data')
            )
            - 1
        )
        return '\n'.join(lines[start:end]) + '\n'

    @staticmethod
    def _end_block() -> str:
        """The shipped lines of install.sh for the failed modules end warning."""
        lines = INSTALL_SH.read_text().splitlines()
        start = next(
            i for i, ln in enumerate(lines) if ln == 'if [ -n "$FAILED_MODULES" ]; then'
        )
        end = next(i for i, ln in enumerate(lines) if i > start and ln == 'fi')
        return '\n'.join(lines[start : end + 1]) + '\n'

    def _run_phase(
        self,
        site: dict,
        suffix: str = '',
        env_extra: dict | None = None,
    ) -> subprocess.CompletedProcess:
        """Run the shipped lines of the phase under the installer functions and environment."""
        scratch = site['scratch']
        defs = (
            'set -euo pipefail\nCURRENT_PHASE=7\n'
            + f'SCRIPT_DIR={shlex.quote(str(scratch))}\n'
            + f'TMPDIR={shlex.quote(str(site["tmp_dir"]))}\n'
            + f'LOGFILE={shlex.quote(str(site["install_log"]))}\n'
            + _install_sh_blocks("RED='", 'info()')
            + ''.join(
                _install_sh_function(f)
                for f in ('command_exists', 'collect_env_info', 'die', *_RUN_LOGGED_FUNCTIONS)
            )
            + _RUN_LOGGED_CLEANUP
        )
        env = {**site['env'], **(env_extra or {})}
        return subprocess.run(
            ['/bin/bash', '-c', defs + self._phase_block() + suffix],
            cwd=scratch,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_commands_run_in_order_with_run_logged_and_stays_file(self, site):
        """The four setup commands run in specified order through run_logged with stays file."""
        res = self._run_phase(site)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = site['calls_log'].read_text().splitlines()
        assert len(calls) == 4
        assert calls[0].startswith('Morrigan tools/get_morrigan.sh ')
        assert calls[1].startswith('VULCAN tools/get_vulcan.sh ')
        assert calls[2].startswith('atmodeller tools/get_atmodeller.sh ')
        assert calls[3].startswith('Obliqua tools/get_obliqua.sh ')
        stays_targets = [c.partition('PROTEUS_CLONE_STAYS=')[2] for c in calls]
        assert len(set(stays_targets)) == 1
        assert stays_targets[0] != ''
        assert 'proteus-clone-stays.' in stays_targets[0]
        assert 'Setting up Morrigan' in res.stdout
        assert 'Setting up VULCAN' in res.stdout
        assert 'Setting up atmodeller' in res.stdout
        assert 'Setting up Obliqua' in res.stdout
        assert 'unbound variable' not in res.stderr.lower()
        assert 'command not found' not in res.stderr.lower()

    @pytest.mark.parametrize(
        ('failed_name', 'failed_script'),
        [
            ('Morrigan', 'get_morrigan.sh'),
            ('VULCAN', 'get_vulcan.sh'),
            ('atmodeller', 'get_atmodeller.sh'),
            ('Obliqua', 'get_obliqua.sh'),
        ],
    )
    def test_each_module_fails_alone_warns_and_records(self, site, failed_name, failed_script):
        """Each module failure alone warns, lets others run, records command, and exits 0."""
        (site['tools'] / failed_script).write_text('#!/bin/bash\nexit 1\n')
        res = self._run_phase(site, suffix='\necho "RESULT_FAILED=$FAILED_MODULES"\n')
        assert res.returncode == 0, res.stdout + res.stderr
        warn_msg = f'{failed_name} is not installed: its setup step failed'
        cmd_msg = f'bash tools/{failed_script}'
        assert warn_msg in res.stdout + res.stderr
        assert cmd_msg in res.stdout + res.stderr
        calls = site['calls_log'].read_text().splitlines()
        assert len(calls) == 3
        assert not any(failed_name in c for c in calls)
        assert f'RESULT_FAILED={failed_name} ({cmd_msg})' in res.stdout

    def test_two_modules_fail_joined_by_semicolon_in_order(self, site):
        """Two failing modules are recorded in FAILED_MODULES joined by semicolon in order."""
        (site['tools'] / 'get_morrigan.sh').write_text('#!/bin/bash\nexit 1\n')
        (site['tools'] / 'get_atmodeller.sh').write_text('#!/bin/bash\nexit 1\n')
        res = self._run_phase(site, suffix='\necho "RESULT_FAILED=$FAILED_MODULES"\n')
        assert res.returncode == 0, res.stdout + res.stderr
        expected = (
            'Morrigan (bash tools/get_morrigan.sh); atmodeller (bash tools/get_atmodeller.sh)'
        )
        assert f'RESULT_FAILED={expected}' in res.stdout
        calls = site['calls_log'].read_text().splitlines()
        assert len(calls) == 2
        assert any('VULCAN' in c for c in calls)
        assert any('Obliqua' in c for c in calls)

    def test_none_fails_failed_modules_is_empty_and_end_block_prints_nothing(self, site):
        """When no module fails FAILED_MODULES is empty and end block prints no warning."""
        suffix = '\n' + self._end_block() + '\necho "RESULT_FAILED=$FAILED_MODULES"\n'
        res = self._run_phase(site, suffix=suffix)
        assert res.returncode == 0, res.stdout + res.stderr
        assert 'RESULT_FAILED=' in res.stdout
        assert res.stdout.partition('RESULT_FAILED=')[2].strip() == ''
        assert 'Not installed:' not in res.stdout + res.stderr

    def test_end_block_with_failed_module_prints_single_warning(self, site):
        """The end block prints exactly one warning naming each failed module and its command."""
        scratch = site['scratch']
        script = (
            'set -euo pipefail\n'
            + f'SCRIPT_DIR={shlex.quote(str(scratch))}\n'
            + _install_sh_blocks("RED='", 'info()')
            + 'FAILED_MODULES="Morrigan (bash tools/get_morrigan.sh)"\n'
            + self._end_block()
        )
        res = subprocess.run(
            ['/bin/bash', '-c', script],
            cwd=scratch,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert res.returncode == 0, res.stdout + res.stderr
        assert (
            'Not installed: Morrigan (bash tools/get_morrigan.sh).' in res.stdout + res.stderr
        )
        assert f'run its command from {scratch}' in res.stdout + res.stderr
        assert (res.stdout + res.stderr).count('Not installed:') == 1

    def test_clone_stays_warning_lists_recorded_clones_and_removes_file(self, site):
        """Stays warning prints all recorded clones and removes temporary stays file."""
        morrigan_stub = (
            '#!/bin/bash\n'
            'printf "Morrigan (commit 1234567)\\n" >> "$PROTEUS_CLONE_STAYS"\n'
            'exit 0\n'
        )
        obliqua_stub = (
            '#!/bin/bash\n'
            'printf "Obliqua (not a git clone)\\n" >> "$PROTEUS_CLONE_STAYS"\n'
            'exit 0\n'
        )
        (site['tools'] / 'get_morrigan.sh').write_text(morrigan_stub)
        (site['tools'] / 'get_obliqua.sh').write_text(obliqua_stub)
        suffix = '\necho "STAYS_EXISTS=$([ -e "$FURTHER_STAYS" ] && echo yes || echo no)"\n'
        res = self._run_phase(site, suffix=suffix)
        assert res.returncode == 0, res.stdout + res.stderr
        expected_warning = (
            'Not at their release tag or pinned commit: '
            'Morrigan (commit 1234567), Obliqua (not a git clone).'
        )
        assert expected_warning in res.stdout + res.stderr
        assert 'Each stays as it is;' in res.stdout + res.stderr
        assert 'STAYS_EXISTS=no' in res.stdout

        # Verify no warning printed when stays file is empty
        (site['tools'] / 'get_morrigan.sh').write_text('#!/bin/bash\nexit 0\n')
        (site['tools'] / 'get_obliqua.sh').write_text('#!/bin/bash\nexit 0\n')
        clean_res = self._run_phase(site)
        assert clean_res.returncode == 0
        assert 'Not at their release tag or pinned commit:' not in (
            clean_res.stdout + clean_res.stderr
        )

    def test_mktemp_failure_runs_commands_and_skips_stays_warning(self, site, tmp_path):
        """When mktemp fails the phase runs all setup commands and prints no stays warning."""
        stubs = tmp_path / 'bin_stubs'
        stubs.mkdir()
        (stubs / 'mktemp').write_text('#!/bin/sh\nexit 1\n')
        (stubs / 'mktemp').chmod(0o755)
        env_extra = {'PATH': f'{stubs}{os.pathsep}{site["env"]["PATH"]}'}
        res = self._run_phase(site, env_extra=env_extra)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = site['calls_log'].read_text().splitlines()
        assert len(calls) == 4
        assert 'Not at their release tag or pinned commit:' not in (res.stdout + res.stderr)
        assert '-c import proteus' in site['python_log'].read_text()

    def test_second_import_check_fails_when_proteus_unimportable(self, site, tmp_path):
        """When PROTEUS does not import after further modules, the phase dies with error output."""
        stubs = tmp_path / 'bin_stubs'
        stubs.mkdir()
        stub_py = stubs / 'python3'
        stub_py.write_text(
            '#!/bin/sh\n'
            'case "$*" in\n'
            '  *"import proteus"*) echo "ImportError: DLL load failed" >&2; exit 1 ;;\n'
            '  *) echo "Python 3.12.0"; exit 0 ;;\n'
            'esac\n'
        )
        stub_py.chmod(0o755)
        env_extra = {'PATH': f'{stubs}{os.pathsep}{site["env"]["PATH"]}'}
        suffix = '\necho "PHASE_8_RAN"\n'
        res = self._run_phase(site, suffix=suffix, env_extra=env_extra)
        assert res.returncode != 0, res.stdout + res.stderr
        assert (
            'PROTEUS does not import after the setup steps of Morrigan, VULCAN, atmodeller and Obliqua. Error output:'
            in res.stderr
        )
        assert 'ImportError: DLL load failed' in (res.stdout + res.stderr)
        assert (
            'PROTEUS imported before these steps and does not import now (see the error above).'
            in res.stderr
        )
        assert 'Steps that failed:' not in res.stderr
        assert 'PHASE_8_RAN' not in (res.stdout + res.stderr)

    def test_second_import_check_fails_with_failed_module_reported(self, site, tmp_path):
        """When import fails and a module step failed, the failed module is named in the die message."""
        stubs = tmp_path / 'bin_stubs'
        stubs.mkdir()
        stub_py = stubs / 'python3'
        stub_py.write_text(
            '#!/bin/sh\n'
            'case "$*" in\n'
            '  *"import proteus"*) echo "ImportError: DLL load failed" >&2; exit 1 ;;\n'
            '  *) echo "Python 3.12.0"; exit 0 ;;\n'
            'esac\n'
        )
        stub_py.chmod(0o755)
        (site['tools'] / 'get_vulcan.sh').write_text('#!/bin/bash\nexit 1\n')
        env_extra = {'PATH': f'{stubs}{os.pathsep}{site["env"]["PATH"]}'}
        suffix = '\necho "PHASE_8_RAN"\n'
        res = self._run_phase(site, suffix=suffix, env_extra=env_extra)
        assert res.returncode != 0, res.stdout + res.stderr
        assert (
            'PROTEUS does not import after the setup steps of Morrigan, VULCAN, atmodeller and Obliqua. Error output:'
            in res.stderr
        )
        assert (
            'PROTEUS imported before these steps and does not import now (see the error above).'
            in res.stderr
        )
        assert 'Steps that failed: VULCAN (bash tools/get_vulcan.sh).' in res.stderr
        assert 'PHASE_8_RAN' not in (res.stdout + res.stderr)

    def test_second_import_check_passes_when_proteus_import_succeeds(self, site):
        """When PROTEUS imports successfully after further modules, the phase continues."""
        suffix = '\necho "PHASE_8_RAN"\n'
        res = self._run_phase(site, suffix=suffix)
        assert res.returncode == 0, res.stdout + res.stderr
        assert 'PROTEUS does not import after the setup steps' not in (res.stdout + res.stderr)
        assert 'PHASE_8_RAN' in res.stdout
        assert site['python_log'].read_text().splitlines() == ['-c import proteus']

    def test_installer_phase_order_and_binding_positions(self):
        """Order of verify, bind_install_vars, further modules and second import check in install.sh."""
        lines = INSTALL_SH.read_text().splitlines()
        idx_phase6 = next(i for i, ln in enumerate(lines) if ln == 'CURRENT_PHASE=6')
        idx_verify = next(i for i, ln in enumerate(lines) if ln == 'verify_proteus_import')
        idx_bind = next(i for i, ln in enumerate(lines) if ln == 'bind_install_vars')
        idx_phase7 = next(i for i, ln in enumerate(lines) if ln == 'CURRENT_PHASE=7')
        idx_first_module = next(
            i for i, ln in enumerate(lines) if ln.startswith('install_further_module "')
        )
        assert idx_verify < idx_bind < idx_phase7 < idx_first_module
        assert not any(
            ln.startswith('CURRENT_PHASE=') for ln in lines[idx_phase6 + 1 : idx_verify]
        )
        idx_last_module = max(
            i for i, ln in enumerate(lines) if ln.startswith('install_further_module "')
        )
        idx_rm_stays = next(i for i, ln in enumerate(lines) if ln == 'rm -f "$FURTHER_STAYS"')
        idx_second_import = next(
            i
            for i, ln in enumerate(lines)
            if 'PROTEUS does not import after the setup steps' in ln
        )
        idx_phase8 = next(i for i, ln in enumerate(lines) if ln == 'CURRENT_PHASE=8')
        assert idx_last_module < idx_rm_stays < idx_second_import < idx_phase8

    def test_phase_seven_further_module_scripts_list_and_files(self):
        """Phase 7 further module invocations match the four expected scripts in order and files exist."""
        lines = INSTALL_SH.read_text().splitlines()
        invocations = [
            ln.strip() for ln in lines if ln.strip().startswith('install_further_module ')
        ]
        assert invocations == [
            'install_further_module "Morrigan" bash tools/get_morrigan.sh',
            'install_further_module "VULCAN" bash tools/get_vulcan.sh',
            'install_further_module "atmodeller" bash tools/get_atmodeller.sh',
            'install_further_module "Obliqua" bash tools/get_obliqua.sh',
        ]
        repo_root = INSTALL_SH.parent
        expected_scripts = [
            'tools/get_morrigan.sh',
            'tools/get_vulcan.sh',
            'tools/get_atmodeller.sh',
            'tools/get_obliqua.sh',
        ]
        for script_path in expected_scripts:
            assert (repo_root / script_path).is_file()


_DEPOT_FUNCTIONS = (
    'command_exists',
    'die',
    'collect_env_info',
    'julia_answers_without_download',
    'set_install_var',
    'is_proteus_install_folder',
    'julia_depot_holds_packages',
    'instantiate_julia_project',
    'setup_julia_depot',
    'reinstall_moved_lovepy',
)

_LOVEPY_STEP = 'run_logged() { printf "FURTHER=%s\\n" "$*"; }\nreinstall_moved_lovepy\n'

_STUB_DEPOT_JULIA = '#!/bin/sh\necho JULIA_RAN\necho "$PWD|$*|$JULIA_DEPOT_PATH|ld=[$LD_LIBRARY_PATH]" >> "$STUB_LOG"\nexit "${STUB_RC:-0}"\n'


class TestJuliaDepot:
    """setup_julia_depot gives each install its own Julia depot and keeps a choice of the user."""

    UNSET = object()

    @pytest.fixture
    def site(self, tmp_path):
        """A scratch HOME, conda prefix, PROTEUS root and a logging Julia stub."""
        for name in ('home', 'prefix', 'root'):
            (tmp_path / name).mkdir()
        julia = _stub(tmp_path / 'stub' / 'julia', _STUB_DEPOT_JULIA)
        log = tmp_path / 'julia.log'
        log.write_text('')
        return {'tmp': tmp_path, 'julia': julia, 'log': log}

    @staticmethod
    def _agni_manifest(site):
        """Give the PROTEUS root an AGNI folder with a Manifest."""
        (site['tmp'] / 'root' / 'AGNI').mkdir()
        (site['tmp'] / 'root' / 'AGNI' / 'Manifest.toml').write_text('m')

    def _run(self, site, value=UNSET, prefix='prefix', after='', **env):
        """Run the shipped setup_julia_depot and print the variable and the old depot it leaves."""
        tmp = site['tmp']
        defs = (
            'set -euo pipefail\n'
            + _install_sh_blocks("RED='", 'info()')
            + f'SCRIPT_DIR={shlex.quote(str(tmp / "root"))}\n'
            + 'PROTEUS_ENV_FILE="$SCRIPT_DIR/.proteus-env.sh"\nINSTALL_VAR_NAMES=""\n'
            + 'FAILED_MODULES=""\n'
            + 'LOGFILE=""\n'
            + ''.join(
                _install_sh_function(f) for f in (*_RUN_LOGGED_FUNCTIONS, *_DEPOT_FUNCTIONS)
            )
        )
        report = 'setup_julia_depot\nprintf "DEPOT=%s\\n" "$JULIA_DEPOT_PATH"\n' + after
        full_env = {
            'PATH': _SYSTEM_PATH,
            'HOME': str(tmp / 'home'),
            'STUB_LOG': str(site['log']),
            'PYTHON_JULIAPKG_EXE': str(site['julia']),
            **env,
        }
        full_env['CONDA_PREFIX'] = str(tmp / prefix)
        if value is not self.UNSET:
            full_env['JULIA_DEPOT_PATH'] = value
        result = subprocess.run(
            ['bash', '-c', defs + report],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=full_env,
            timeout=25,
        )
        return result

    def _call(self, site, value=UNSET, prefix='prefix', after='', **env):
        """``_run`` of a call that must end with status 0."""
        result = self._run(site, value, prefix, after, **env)
        assert result.returncode == 0, result.stdout
        return result

    @staticmethod
    def _marker(site):
        """The old depot that the file in the depot names, or None with no such file."""
        path = site['tmp'] / 'prefix' / 'julia_depot' / '.proteus-moved-from'
        return path.read_text().strip() if path.exists() else None

    @staticmethod
    def _value(result, key='DEPOT'):
        """The value of a report line of ``_call``."""
        return next(ln for ln in result.stdout.splitlines() if ln.startswith(f'{key}='))[
            len(key) + 1 :
        ]

    @pytest.mark.parametrize('value', [UNSET, ''], ids=['unset', 'empty'])
    def test_no_value_gets_the_depot_of_the_prefix_with_the_trailing_colon(self, site, value):
        """Unset or empty gives <prefix>/julia_depot: in the process and the exports file.

        An empty value is no choice: Julia then runs with no depot at all. The trailing
        colon keeps the bundled depots, without which each depot compiles the stdlibs again.
        """
        result = self._call(site, value)
        own = f'{site["tmp"] / "prefix"}/julia_depot:'
        assert self._value(result) == own
        assert (site['tmp'] / 'root' / '.proteus-env.sh').is_file()
        exports = (site['tmp'] / 'root' / '.proteus-env.sh').read_text()
        assert f"export JULIA_DEPOT_PATH='{own}'" in exports
        assert 'shares' not in result.stdout

    def test_two_prefixes_get_two_depots(self, site):
        """Install B does not get the depot of install A."""
        (site['tmp'] / 'prefix2').mkdir()
        first = self._value(self._call(site))
        second = self._value(self._call(site, prefix='prefix2'))
        assert first != second
        assert second == f'{site["tmp"] / "prefix2"}/julia_depot:'

    def test_own_value_stays_silent(self, site):
        """A re-run with the value of this install changes nothing and warns of nothing."""
        own = f'{site["tmp"] / "prefix"}/julia_depot:'
        result = self._call(site, own)
        assert self._value(result) == own
        assert 'shares' not in result.stdout
        assert 'WARN' not in result.stdout

    @pytest.mark.parametrize(
        'value',
        ['/big/x', '/big/x:', '/big/x:/y:', ':', ':/big/x', '/usr/julia_depot:'],
        ids=['plain', 'colon', 'list', 'only-colon', 'leading-colon', 'same-name-no-install'],
    )
    def test_value_of_the_user_stays_verbatim_and_says_it_is_shared(self, site, value):
        """A value set by the user stays, and one line names the limit and the way out."""
        result = self._call(site, value)
        assert self._value(result) == value
        assert (
            "export JULIA_DEPOT_PATH='" + value + "'"
            in (site['tmp'] / 'root' / '.proteus-env.sh').read_text()
        )
        assert 'shares' in result.stdout
        assert 'conda env config vars unset' in result.stdout
        assert 'new shell' in result.stdout

    def test_value_kept_on_pixi_names_the_manifest(self, site):
        """A shell value and conda play no part on pixi, so the line names pixi.toml."""
        result = self._call(site, '/big/x:', PIXI_ENVIRONMENT_NAME='default')
        assert 'pixi.toml' in result.stdout
        assert 'conda env config vars' not in result.stdout

    def test_tilde_path_to_another_install_is_replaced(self, site):
        """Julia expands the tilde, so the rule reads the same folder that Julia reads."""
        (site['tmp'] / 'home' / 'other' / 'conda-meta').mkdir(parents=True)
        result = self._call(site, '~/other/julia_depot:')
        assert self._value(result) == f'{site["tmp"] / "prefix"}/julia_depot:'
        assert 'another PROTEUS install' in result.stdout

    def test_depot_in_the_own_tree_stays(self, site):
        """The tree of this install is no other install, whatever env is active."""
        root = site['tmp'] / 'root'
        (root / 'src' / 'proteus').mkdir(parents=True)
        (root / 'install.sh').write_text('')
        result = self._call(site, f'{root}/julia_depot:')
        assert self._value(result) == f'{root}/julia_depot:'
        assert 'another PROTEUS install' not in result.stdout

    @pytest.mark.parametrize(
        'value_form', ['{new}:{full}', '{new}:{full}:'], ids=['list', 'list-colon']
    )
    def test_packages_in_a_later_entry_are_no_move(self, site, value_form):
        """Julia reads every entry, so a shared later depot means nothing moves."""
        full = site['tmp'] / 'shared'
        (full / 'packages').mkdir(parents=True)
        self._agni_manifest(site)
        value = value_form.format(new=site['tmp'] / 'new', full=full)
        result = self._call(site, value)
        assert 'MiB' not in result.stdout
        assert self._marker(site) is None

    def test_leading_colon_reads_the_home_depot(self, site):
        """A leading colon is ~/.julia, so its packages count and no move is announced."""
        (site['tmp'] / 'home' / '.julia' / 'packages').mkdir(parents=True)
        self._agni_manifest(site)
        assert 'MiB' not in self._call(site, ':').stdout
        assert self._marker(site) is None

    @pytest.mark.parametrize('form', ['abs', 'tilde', 'slash', 'dotdot', 'symlink'])
    @pytest.mark.parametrize(
        'parent', ['other-env', 'other-tree', 'own-env', 'own-root', 'plain', 'other-gone']
    )
    def test_install_script_and_doctor_agree_on_another_install(
        self, site, monkeypatch, parent, form
    ):
        """One verdict from both tools, over every parent and spelling of the same folder."""
        home = site['tmp']
        folders = {
            'other-env': home / 'other-env',
            'other-tree': home / 'other-tree',
            'own-env': home / 'prefix',
            'own-root': home / 'root',
            'plain': home / 'plain',
            'other-gone': home / 'other-gone',
        }
        folder = folders[parent]
        if parent != 'other-gone':
            folder.mkdir(exist_ok=True)
        (home / 'prefix' / 'conda-meta').mkdir(exist_ok=True)
        (folders['own-root'] / 'src' / 'proteus').mkdir(parents=True)
        (folders['own-root'] / 'install.sh').write_text('')
        if parent == 'other-env':
            (folder / 'conda-meta').mkdir()
        if parent == 'other-tree':
            (folder / 'src' / 'proteus').mkdir(parents=True)
            (folder / 'install.sh').write_text('')
        path = {
            'abs': str(folder),
            'slash': str(folder),
            'tilde': f'~/{folder.name}',
            'dotdot': f'{home}/stub/../{folder.name}',
            'symlink': f'{home}/link-{folder.name}',
        }[form]
        if form == 'symlink':
            (home / f'link-{folder.name}').symlink_to(folder)
        value = f'{path}/julia_depot{"/" if form == "slash" else ""}:'
        replaced = 'another PROTEUS install' in self._call(site, value, HOME=str(home)).stdout
        monkeypatch.setenv('HOME', str(home))
        monkeypatch.setenv('CONDA_PREFIX', str(site['tmp'] / 'prefix'))
        from proteus.utils.install_env import depot_of_other_install

        verdict = depot_of_other_install(value, site['tmp'] / 'root')
        assert replaced == verdict
        assert verdict == parent.startswith('other')

    @pytest.mark.parametrize(
        'marker', ['conda-meta', 'tree'], ids=['conda-or-pixi-env', 'proteus-tree']
    )
    def test_depot_of_another_install_is_replaced_with_a_warning(self, site, marker):
        """A depot inherited from another env or tree is the one value that does not stay.

        Conda keeps the value of the previous env when the next env has none.
        """
        other = site['tmp'] / 'other'
        if marker == 'conda-meta':
            (other / 'conda-meta').mkdir(parents=True)
        else:
            (other / 'src' / 'proteus').mkdir(parents=True)
            (other / 'install.sh').write_text('')
        result = self._call(site, f'{other}/julia_depot:')
        assert self._value(result) == f'{site["tmp"] / "prefix"}/julia_depot:'
        assert str(other) in result.stdout
        assert 'another PROTEUS install' in result.stdout

    def test_tree_marker_needs_both_files(self, site):
        """A folder with install.sh alone is no PROTEUS tree, so its depot stays."""
        other = site['tmp'] / 'other'
        other.mkdir()
        (other / 'install.sh').write_text('')
        result = self._call(site, f'{other}/julia_depot:')
        assert self._value(result) == f'{other}/julia_depot:'
        assert 'another PROTEUS install' not in result.stdout

    def test_own_depot_without_the_colon_stays(self, site):
        """The parent is an env, but it is this env: not another install, and not shared."""
        (site['tmp'] / 'prefix' / 'conda-meta').mkdir()
        value = f'{site["tmp"] / "prefix"}/julia_depot'
        result = self._call(site, value)
        assert self._value(result) == value
        assert 'shares' not in result.stdout
        assert 'no trailing colon' in result.stdout
        assert 'macOS arm64' in result.stdout

    @pytest.mark.parametrize('value', ['/big/x:', '/big/x::/y', ':/big/x', ':'])
    def test_value_with_the_bundled_depots_has_no_colon_line(self, site, value):
        """Each of these forms keeps the compile files of Julia."""
        assert 'no trailing colon' not in self._call(site, value).stdout

    @pytest.mark.parametrize(
        'form', ['{p}/julia_depot/:', '{p}/../prefix/julia_depot:', '{l}/julia_depot:']
    )
    def test_own_depot_in_another_spelling_is_not_shared(self, site, form):
        """A slash, a .. and a link name the own depot, and it need not exist yet."""
        prefix = site['tmp'] / 'prefix'
        (site['tmp'] / 'link').symlink_to(prefix)
        value = form.format(p=prefix, l=site['tmp'] / 'link')
        result = self._call(site, value)
        assert self._value(result) == value
        assert 'shares' not in result.stdout
        assert 'another PROTEUS install' not in result.stdout

    def test_value_of_a_removed_env_is_replaced_with_a_warning(self, site):
        """A depot that names a removed env would grow again in the old prefix."""
        gone = site['tmp'] / 'envs' / 'gone' / 'julia_depot'
        result = self._call(site, f'{gone}:')
        own = f'{site["tmp"] / "prefix"}/julia_depot:'
        assert self._value(result) == own
        warned = next(
            ln for ln in result.stdout.splitlines() if 'another PROTEUS install' in ln
        )
        assert str(gone) in warned
        assert own in warned
        assert not gone.parent.exists()

    @pytest.mark.parametrize(
        'value',
        [
            'rel:',
            'julia_depot:',
            './x:',
            '../x:',
            '-x/julia_depot:',
            '~nobody/x:',
            '~nobody',
            '$CONDA_PREFIX/julia_depot:',
            '/a/$b:',
            'rel',
        ],
    )
    def test_value_that_depends_on_the_folder_or_the_shell_stops(self, site, value):
        """The depot of such a value depends on the folder of each process, or holds no path."""
        result = self._run(site, value)
        assert result.returncode == 1
        assert f'JULIA_DEPOT_PATH={value}' in result.stdout
        assert 'absolute' in result.stdout
        assert not (site['tmp'] / 'root' / '.proteus-env.sh').exists()
        assert site['log'].read_text() == ''

    @pytest.mark.parametrize(
        'value',
        [
            '',
            ':',
            ':/x',
            '/x',
            '/x:',
            '~',
            '~:',
            '~/d:',
            '/x:rel',
            'rel:',
            '~nobody/x:',
            '$X/y:',
            '/a$b:',
            './x',
            '-x:',
        ],
    )
    def test_install_script_and_doctor_agree_on_a_value_that_stops(self, site, value):
        """One verdict from both tools over the forms of the first entry."""
        from proteus.utils.install_env import depot_value_is_unusable

        result = self._run(site, value)
        assert (result.returncode != 0) == depot_value_is_unusable(value)

    def test_move_names_size_and_stay_command_before_any_julia_runs(self, site):
        """An old install with an empty new depot hears the cost before the download."""
        env_project = site['tmp'] / 'prefix' / 'julia_env'
        env_project.mkdir()
        (env_project / 'Manifest.toml').write_text('m')
        (site['tmp'] / 'home' / '.julia' / 'packages').mkdir(parents=True)
        result = self._call(site)
        out = result.stdout
        assert 'MiB' in out
        assert 'JULIA_RAN' in out
        assert out.index('MiB') < out.index('JULIA_RAN')
        home = site['tmp'] / 'home'
        assert f'JULIA_DEPOT_PATH={home}/.julia: bash install.sh' in out
        assert 'macOS arm64' in out
        assert self._marker(site) == f'{home}/.julia'
        assert 'pixi.toml' not in out

    def test_move_on_pixi_names_the_manifest_line(self, site):
        """Pixi activation overrides a shell value, so the stay route is pixi.toml."""
        self._agni_manifest(site)
        (site['tmp'] / 'home' / '.julia' / 'packages').mkdir(parents=True)
        out = self._call(site, PIXI_ENVIRONMENT_NAME='default').stdout
        assert 'pixi.toml' in out
        assert 'To stay on the old depot:' not in out

    def test_inherited_depot_is_the_old_depot_of_the_move(self, site):
        """LovePy and the text follow the depot left behind, and its value cannot be the stay."""
        other = site['tmp'] / 'other'
        (other / 'conda-meta').mkdir(parents=True)
        (other / 'julia_depot' / 'packages').mkdir(parents=True)
        self._agni_manifest(site)
        out = self._call(site, f'{other}/julia_depot:').stdout
        assert self._marker(site) == f'{other}/julia_depot'
        assert f'{other}/julia_depot stays as it is' in out
        assert 'To stay on the old depot' not in out

    def test_retry_of_a_fresh_install_writes_no_note(self, site):
        """A clone with a Manifest and an old depot with no packages leaves nothing to carry."""
        self._agni_manifest(site)
        result = self._call(site)
        assert self._marker(site) is None
        assert 'To stay on the old depot' not in result.stdout
        assert 'MiB' not in result.stdout
        assert 'move to' not in result.stdout
        assert 'stays as it is' not in result.stdout
        assert 'install now' in result.stdout

    def test_depot_that_cannot_hold_the_note_warns(self, site):
        """A note that cannot be written loses the LovePy step, so the user hears it."""
        (site['tmp'] / 'home' / '.julia' / 'packages').mkdir(parents=True)
        self._agni_manifest(site)
        (site['tmp'] / 'prefix' / 'julia_depot').write_text('a file')
        out = self._call(site).stdout
        assert 'get_lovepy.sh' in out
        assert self._marker(site) is None

    def test_no_prefix_stops_with_a_message(self, site):
        """An env with no CONDA_PREFIX has no place for the depot."""
        tmp = site['tmp']
        defs = (
            'set -euo pipefail\nCURRENT_PHASE=3\nLOGFILE=""\n'
            + _install_sh_blocks("RED='", 'info()')
            + f'SCRIPT_DIR={shlex.quote(str(tmp / "root"))}\n'
            + ''.join(_install_sh_function(f) for f in _DEPOT_FUNCTIONS)
            + 'collect_env_info() { :; }\n'
        )
        done = subprocess.run(
            ['bash', '-c', defs + 'setup_julia_depot\n'],
            capture_output=True,
            text=True,
            env={'PATH': _SYSTEM_PATH, 'HOME': str(tmp / 'home')},
            timeout=25,
        )
        assert done.returncode == 1
        assert 'CONDA_PREFIX is not set' in done.stdout + done.stderr

    def test_fresh_install_has_no_move_line(self, site):
        """With no Manifest there is nothing to move and no Julia call."""
        result = self._call(site)
        assert 'MiB' not in result.stdout
        assert self._marker(site) is None
        assert site['log'].read_text() == ''

    def test_depot_with_packages_has_no_move_line(self, site):
        """A re-run on a filled depot prints nothing about a move and still instantiates."""
        env_project = site['tmp'] / 'prefix' / 'julia_env'
        env_project.mkdir()
        (env_project / 'Manifest.toml').write_text('m')
        (site['tmp'] / 'prefix' / 'julia_depot' / 'packages').mkdir(parents=True)
        result = self._call(site, LD_LIBRARY_PATH='/conda/lib')
        assert 'MiB' not in result.stdout
        assert self._marker(site) is None
        calls = site['log'].read_text().splitlines()
        assert len(calls) == 1
        assert f'--project={env_project}' in calls[0]
        assert '--startup-file=no' in calls[0]
        assert 'Pkg.instantiate()' in calls[0]
        assert calls[0].endswith(f'|{site["tmp"] / "prefix"}/julia_depot:|ld=[]')

    def test_tilde_value_is_read_as_the_home_folder(self, site):
        """Julia expands a leading tilde, so the packages check must too."""
        (site['tmp'] / 'home' / 'd' / 'packages').mkdir(parents=True)
        self._agni_manifest(site)
        result = self._call(site, '~/d:')
        assert self._value(result) == '~/d:'
        assert 'MiB' not in result.stdout

    def test_failed_instantiate_warns_with_its_command_and_goes_on(self, site):
        """No network with an empty depot: a warning that names the repeat, not a stop."""
        env_project = site['tmp'] / 'prefix' / 'julia_env'
        env_project.mkdir()
        (env_project / 'Manifest.toml').write_text('m')
        (site['tmp'] / 'home' / '.julia' / 'packages').mkdir(parents=True)
        after = 'printf "FAILED=%s\\n" "$FAILED_MODULES"\n'
        result = self._call(site, after=after, STUB_RC='1')
        both = result.stdout
        assert 'Could not install the Julia packages' in both
        assert 'Pkg.instantiate()' in both
        assert 'LD_LIBRARY_PATH=""' in both
        assert self._marker(site) is not None
        assert str(env_project) in self._value(result, 'FAILED')

    def test_failed_instantiate_of_agni_joins_the_list_of_the_end_of_the_run(self, site):
        """The list survives to the end of the script, so the run does not end as a plain success."""
        agni = site['tmp'] / 'root' / 'AGNI'
        agni.mkdir()
        (agni / 'Manifest.toml').write_text('m')
        (site['tmp'] / 'prefix' / 'julia_depot' / 'packages').mkdir(parents=True)
        after = f'instantiate_julia_project {shlex.quote(str(agni))}\nprintf "FAILED=%s\\n" "$FAILED_MODULES"\n'
        result = self._call(site, after=after, STUB_RC='1')
        assert str(agni) in self._value(result, 'FAILED')
        lines = INSTALL_SH.read_text().splitlines()
        assert 'FAILED_MODULES=""' not in lines
        assert 'FAILED_MODULES="${FAILED_MODULES:-}"' in lines

    def test_import_that_misses_julia_packages_names_the_repeat_and_the_page(self, tmp_path):
        """The import stop after a move with no network says how the next run repairs it."""
        stub = tmp_path / 'stub'
        _stub(
            stub / 'python3',
            '#!/bin/sh\necho "Package PythonCall is required but does not seem to be installed" >&2\nexit 1\n',
        )
        defs = (
            'set -euo pipefail\nCURRENT_PHASE=6\nLOGFILE=""\n'
            + _install_sh_blocks("RED='", 'info()')
            + ''.join(
                _install_sh_function(f)
                for f in ('command_exists', 'die', 'verify_proteus_import')
            )
            + 'collect_env_info() { :; }\nconda_has_mpi_netcdf() { echo MPI_STEP; return 1; }\n'
        )
        done = subprocess.run(
            ['bash', '-c', defs + 'verify_proteus_import\n'],
            capture_output=True,
            text=True,
            env={
                'PATH': f'{stub}:{_SYSTEM_PATH}',
                'HOME': str(tmp_path),
                'JULIA_DEPOT_PATH': '/d:',
            },
            timeout=25,
        )
        out = done.stdout + done.stderr
        assert done.returncode == 1
        assert 'bash install.sh' in out
        assert '#julia-depot-with-no-packages' in out
        assert 'MPI_STEP' not in out
        anchor = (INSTALL_SH.parent / 'docs' / 'How-to' / 'troubleshooting.md').read_text()
        assert '{#julia-depot-with-no-packages}' in anchor

    def test_julia_calls_before_the_depot_read_no_startup_file(self):
        """A start file of the user can write to ~/.julia before the depot is set."""
        text = INSTALL_SH.read_text()
        before = text[text.index('\nsetup_conda_env\n') : text.index('\nsetup_julia_depot\n')]
        calls = [ln for ln in before.splitlines() if '"$PYTHON_JULIAPKG_EXE" ' in ln]
        assert len(calls) == 2
        assert all('--startup-file=no' in ln for ln in calls)

    @pytest.mark.parametrize(
        ('text', 'called'),
        [('LovePy = "abc"\n', True), ('Other = "abc"\n', False), (None, False)],
        ids=['has-lovepy', 'other-package', 'no-environment'],
    )
    def test_lovepy_of_the_old_default_environment_comes_along(self, site, text, called):
        """LovePy lives in the default environment of the depot, so a move leaves it behind.

        The step names the script by its full path, and success removes the note of the move.
        """
        self._agni_manifest(site)
        (site['tmp'] / 'home' / '.julia' / 'packages').mkdir(parents=True)
        if text:
            env_dir = site['tmp'] / 'home' / '.julia' / 'environments' / 'v1.13'
            env_dir.mkdir(parents=True)
            (env_dir / 'Project.toml').write_text(text)
        after = _LOVEPY_STEP
        out = self._call(site, after=after).stdout
        script = f'{site["tmp"] / "root"}/tools/get_lovepy.sh'
        assert (f'bash {script}\n' in out and 'FURTHER=' in out) is called
        assert self._marker(site) is None

    def test_failed_lovepy_step_warns_and_goes_on(self, site):
        """A failed LovePy step names its repeat and does not stop the install."""
        self._agni_manifest(site)
        (site['tmp'] / 'home' / '.julia' / 'packages').mkdir(parents=True)
        env_dir = site['tmp'] / 'home' / '.julia' / 'environments' / 'v1.13'
        env_dir.mkdir(parents=True)
        (env_dir / 'Project.toml').write_text('LovePy = "abc"\n')
        after = 'run_logged() { return 3; }\nreinstall_moved_lovepy\necho WENT_ON\n'
        out = self._call(site, after=after).stdout
        assert 'To repeat the step: bash' in out
        assert 'tools/get_lovepy.sh' in out
        assert 'WENT_ON' in out
        assert self._marker(site) is not None

    def test_move_stopped_before_lovepy_sets_it_up_on_the_next_run(self, site):
        """The first run fills packages and stops; the note in the depot still names the move."""
        depot = site['tmp'] / 'prefix' / 'julia_depot'
        (depot / 'packages').mkdir(parents=True)
        home = site['tmp'] / 'home'
        (depot / '.proteus-moved-from').write_text(f'{home}/.julia\n')
        (home / '.julia' / 'packages').mkdir(parents=True)
        env_dir = home / '.julia' / 'environments' / 'v1.13'
        env_dir.mkdir(parents=True)
        (env_dir / 'Project.toml').write_text('LovePy = "abc"\n')
        after = _LOVEPY_STEP
        assert 'tools/get_lovepy.sh' in self._call(site, after=after).stdout
        assert self._marker(site) is None

    def test_no_move_keeps_lovepy_alone(self, site):
        """With no move the old depot is not read, and LovePy needs no step."""
        env_dir = site['tmp'] / 'home' / '.julia' / 'environments' / 'v1.13'
        env_dir.mkdir(parents=True)
        (env_dir / 'Project.toml').write_text('LovePy = "abc"\n')
        after = _LOVEPY_STEP
        assert 'FURTHER=' not in self._call(site, after=after).stdout
        assert self._marker(site) is None

    def test_order_in_the_script(self):
        """The depot is set before AGNI and the first import, and AGNI keeps its Manifest."""
        text = INSTALL_SH.read_text()
        main = text[text.index('phase 3 "Environment variables"') :]
        assert main.count('\nsetup_julia_depot\n') == 1
        assert main.count('\nreinstall_moved_lovepy\n') == 1
        depot = main.index('\nsetup_julia_depot\n')
        assert depot < main.index('tools/get_agni.sh')
        assert depot < main.index('\nverify_proteus_import\n')
        assert main.index('set_install_var PYTHON_JULIAPKG_EXE') < depot
        branch = main[main.index('AGNI already installed') : main.index('tools/get_agni.sh')]
        assert 'instantiate_julia_project "$SCRIPT_DIR/AGNI"' in branch
        lovepy = main.index('\nreinstall_moved_lovepy\n')
        assert main.index('phase 5 "AGNI') < lovepy < main.index('phase 6 "Python submodules"')

    def test_pixi_manifest_sets_the_depot_with_the_colon(self):
        """pixi run does not source the exports file, so its activation holds the value."""
        text = (PROTEUS_ROOT / 'pixi.toml').read_text()
        depot = tomllib.loads(text)['activation']['env']['JULIA_DEPOT_PATH']
        assert depot == '$CONDA_PREFIX/julia_depot:'
        assert text.count('[activation.env]') == 1

    def test_failure_report_names_both_depot_variables(self):
        """The environment block of a failed install prints the depot variables."""
        block = _install_sh_function('collect_env_info')
        assert 'JULIA_DEPOT_PATH' in block
        assert 'JULIAUP_DEPOT_PATH' in block
