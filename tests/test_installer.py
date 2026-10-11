"""Unit tests for the PROTEUS unified installer (install.sh).

Tests the pre-flight check logic, argument parsing, helper functions,
and error messages by running install.sh in controlled subprocess
environments. Does not execute actual installations (SOCRATES, AGNI,
pip install); those are tested by the CI pipeline and smoke tests.

The Python modules the installer clones are pinned to the release tag of
their version floor: ``checkout_floor_tag`` (``tools/_get_common.sh``) and
``clone_and_install`` (lifted from install.sh at run time) run against
local scratch repositories and a recording ``pip`` stub, without network.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT

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


def _bin_dir_without(tmp_path, omitted: set[str]) -> Path:
    """Mirror every executable on PATH into a fresh directory, minus `omitted`.

    Lets a test run the installer as if the host lacked one specific tool,
    without disturbing anything else the pre-flight checks need.
    """
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    for entry in os.environ['PATH'].split(os.pathsep):
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


def _run_preflight(bindir: Path) -> str:
    """Run install.sh with PATH restricted to `bindir`; return its combined output."""
    env = os.environ.copy()
    env['PATH'] = str(bindir)
    env.setdefault('CONDA_DEFAULT_ENV', 'proteus')
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
        """Without CONDA_DEFAULT_ENV, the installer fails with conda instructions.

        Discrimination: the error must mention 'conda' so users know what
        to do. A generic error would not be actionable.
        """
        env = os.environ.copy()
        env.pop('CONDA_DEFAULT_ENV', None)
        env.pop('CONDA_PREFIX', None)
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
        assert 'conda create' in output

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
        enough. Discrimination: the check previously accepted either tool, so
        on such a host it reported nothing missing and the build failed many
        phases later with a far less obvious message. Requiring the Fortran
        package to be named, and the C package not to be, fails against that
        earlier logic rather than merely restating it.
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


_VAR_FUNCTIONS = (
    'command_exists',
    'set_install_var',
    'bind_install_vars',
    'note_old_shell_exports',
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
        script = _set_script(_VALUES) + 'bind_install_vars\nprint_next_steps\n'
        result = self._call(site, script, **conda_env)
        assert result.returncode == 0, result.stdout + result.stderr
        lines = site['env_file'].read_text().splitlines()
        assert len(lines) == 4
        assert [ln.split('=', 1)[0] for ln in lines] == [f'export {n}' for n in _VALUES]
        assert all(ln.split('=', 1)[1].startswith("'") for ln in lines)
        assert _sourced(site['env_file']) == _VALUES
        pairs = [f'{n}={v}' for n, v in _VALUES.items()]
        assert _conda_calls(site['log']) == [
            ['env', 'config', 'vars', 'set', '-p', site['prefix'], *pairs]
        ]
        assert 'conda deactivate && conda activate proteus' in result.stdout
        assert 'source .proteus-env.sh' not in result.stdout

        newer = {n: f'{v}-2' for n, v in _VALUES.items()}
        again = self._call(site, _set_script(newer) + 'bind_install_vars\n', **conda_env)
        assert again.returncode == 0, again.stdout + again.stderr
        assert len(site['env_file'].read_text().splitlines()) == 4
        assert _sourced(site['env_file']) == newer
        assert len(_conda_calls(site['log'])) == 2

    def test_variable_set_twice_keeps_one_line_and_one_pair(self, site):
        """A variable set twice in one run keeps its last value, once in the file and in conda.

        This is the Julia 1.11 fall-back, which sets PYTHON_JULIAPKG_EXE again in phase 6.
        """
        script = (
            _set_script(_VALUES)
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

        Both hold an old `export FWL_DATA=/old` line, which the removed rc
        writer deleted, and mode 600, which its temp-file rename turned into 644.
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

    def test_no_note_without_old_exports(self, site):
        """A HOME without shell start files gets no note."""
        result = self._call(site, 'note_old_shell_exports\n')
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stderr == ''
        assert result.stdout == ''


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

        The control is the removed rc writer: its 3 file operations are flagged
        and its stderr warning is not, so the empty result for install.sh is not
        a scanner that finds nothing.
        """
        text = INSTALL_SH.read_text()
        assert _start_file_writes(text) == []
        assert 'append_export_to_rc' not in text
        assert 'detect_shell_rc' not in text
        reader = _install_sh_function('note_old_shell_exports')
        assert '.zshrc' in reader
        assert [ln for ln in reader.splitlines() if _FILE_WRITE.search(ln)] == []
        removed = (
            '    grep -v "^export ${var_name}=" "$rc_file" > "${rc_file}.tmp" 2>/dev/null || true\n'
            '    mv "${rc_file}.tmp" "$rc_file"\n'
            '    echo "$line" >> "$rc_file"\n'
            '    warn "Shell config: $RC_FILE" >&2\n'
        )
        assert len(_start_file_writes(removed)) == 3

    def test_fastchem_runs_without_yes_and_without_input(self):
        """get_fastchem.sh runs once, without -y, without FC_DIR and with stdin from /dev/null.

        With -y it answers its own question to add FC_DIR to the shell start file.
        """
        calls = [
            ln.strip()
            for ln in INSTALL_SH.read_text().splitlines()
            if 'get_fastchem.sh' in ln and not ln.lstrip().startswith(('#', 'warn', 'echo'))
        ]
        assert calls == [
            'run_logged "Building FastChem" env -u FC_DIR bash src/get_fastchem.sh </dev/null'
        ]
        assert '-y' not in calls[0].split()


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

        Passes /dev/null as stdin. The script should reach the conda
        pre-flight check and fail there, not block waiting for input.
        """
        env = os.environ.copy()
        env.pop('CONDA_DEFAULT_ENV', None)
        result = subprocess.run(
            ['bash', str(INSTALL_SH)],
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=str(PROTEUS_ROOT),
        )
        # Should fail at conda check, not block on stdin
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert 'conda' in output.lower()


def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` with a fixed identity and no signing; return its stdout."""
    result = subprocess.run(
        ['git', '-c', 'user.email=t@e.st', '-c', 'user.name=t', '-c', 'commit.gpgsign=false']
        + list(args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _module_repo(path: Path, tag: str | None = '26.10.06') -> tuple[str, str]:
    """Build a repository with 3 commits and ``tag`` on the second one.

    Returns the tagged commit and the tip, which differ, so a checkout that
    stays on the branch is told apart from one at the tag.
    """
    path.mkdir(parents=True)
    _git(path, 'init', '-q', '-b', 'main')
    for i in range(3):
        (path / 'version.txt').write_text(f'{i}\n')
        _git(path, 'add', 'version.txt')
        _git(path, 'commit', '-q', '-m', f'commit {i}')
        if i == 1 and tag:
            _git(path, 'tag', tag)
    tagged = _git(path, 'rev-parse', 'HEAD~1')
    return tagged, _git(path, 'rev-parse', 'HEAD')


def _proteus_copy(root: Path, dependencies: list[str]) -> Path:
    """Copy the helper library, the pin reader and get_python_module.sh below a pyproject.toml."""
    (root / 'tools').mkdir(parents=True)
    for name in ('_get_common.sh', '_module_pins.py', 'get_python_module.sh'):
        shutil.copy2(TOOLS_DIR / name, root / 'tools' / name)
    deps = ''.join(f'    "{d}",\n' for d in dependencies)
    (root / 'pyproject.toml').write_text(
        f'[project]\nname = "fixture"\ndependencies = [\n{deps}]\n'
    )
    return root


def _run_checkout_floor_tag(
    root: Path, repo: Path, prelude: str
) -> subprocess.CompletedProcess:
    """Source the copied helper library and run checkout_floor_tag on ``repo`` for fwl-mors."""
    return subprocess.run(
        [
            'bash',
            '-c',
            f'{prelude}source "{root}/tools/_get_common.sh"\n'
            f'checkout_floor_tag fwl-mors "{repo}"\necho AFTER\n',
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )


class TestCheckoutFloorTag:
    """checkout_floor_tag puts a clone on the release tag of its floor, or stops."""

    @pytest.mark.parametrize('prelude', ['', 'set -euo pipefail\n'], ids=['plain', 'errexit'])
    def test_checks_out_the_tag_matching_the_floor(self, tmp_path, prelude):
        """The floor 26.10.6 selects the tag 26.10.06 on the second commit.

        The tip is a later commit, so a helper that leaves the clone on its
        branch fails the equality; the floor text is not a tag, so an exact
        text lookup fails the checkout.
        """
        repo = tmp_path / 'MORS'
        tagged, tip = _module_repo(repo)
        root = _proteus_copy(tmp_path / 'proteus', ['fwl-mors>=26.10.6'])
        result = _run_checkout_floor_tag(root, repo, prelude)
        assert result.returncode == 0, result.stderr
        head = _git(repo, 'rev-parse', 'HEAD')
        assert head == tagged
        assert head != tip
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
        root = _proteus_copy(tmp_path / 'proteus', ['fwl-mors>=26.11.1'])
        result = _run_checkout_floor_tag(root, repo, prelude)
        assert result.returncode == 1
        assert 'AFTER' not in result.stdout
        assert _git(repo, 'rev-parse', 'HEAD') == tip
        assert "'fwl-mors' floor 26.11.1: no tag matches it" in result.stderr
        assert '26.10.06' in result.stderr


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


class TestCloneAndInstall:
    """clone_and_install pins a new clone to the floor tag and leaves an old one alone."""

    @pytest.fixture
    def site(self, tmp_path):
        """A PROTEUS copy, local bare remotes for MORS and JANUS, and a pip stub.

        JANUS carries only the tag 26.07.10, below its floor 26.10.08.
        """
        root = _proteus_copy(tmp_path / 'proteus', ['fwl-mors>=26.10.6', 'fwl-janus>=26.10.08'])
        remotes = tmp_path / 'remotes' / 'FormingWorlds'
        shas = {}
        for repo, tag in (('MORS', '26.10.06'), ('JANUS', '26.07.10')):
            work = tmp_path / 'work' / repo
            shas[repo] = _module_repo(work, tag)
            _git(tmp_path, 'clone', '-q', '--bare', str(work), str(remotes / f'{repo}.git'))
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
        }
        return {'root': root, 'shas': shas, 'log': log, 'env': env, 'tmp': tmp_path}

    @staticmethod
    def _call(site, *args: str, **env: str) -> subprocess.CompletedProcess:
        """Run the shipped clone_and_install with ``args``, under install.sh's options."""
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
                )
            )
        )
        return subprocess.run(
            ['bash', '-c', defs + shlex.join(['clone_and_install', *args])],
            capture_output=True,
            text=True,
            env={**site['env'], **env},
            timeout=25,
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
        assert _git(clone, 'rev-parse', 'HEAD') == tagged != tip
        detached = subprocess.run(
            ['git', '-C', str(clone), 'symbolic-ref', '-q', 'HEAD'],
            capture_output=True,
            timeout=10,
        )
        assert detached.returncode == 1
        assert site['log'].read_text().splitlines() == [
            f'install -U -e {os.path.realpath(clone)}'
        ]

    def test_existing_clone_is_left_on_its_commit(self, site):
        """A second run reports an existing clone and installs it unchanged.

        The clone is moved to the branch tip first, as a developer would; a
        function that checks out the floor tag again moves it back.
        """
        _, tip = site['shas']['MORS']
        clone = site['root'] / 'MORS'
        assert self._call(site, 'MORS', 'fwl-mors').returncode == 0
        _git(clone, 'checkout', '-q', 'main')
        result = self._call(site, 'MORS', 'fwl-mors')
        assert result.returncode == 0, result.stdout + result.stderr
        assert _git(clone, 'rev-parse', 'HEAD') == tip
        assert f'MORS already cloned at {clone} (26.10.06-1-g' in result.stdout
        assert 'not changed' in result.stdout
        assert len(site['log'].read_text().splitlines()) == 2

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
        assert re.search(r'Cloning and installing MORS: done in \d+ s', result.stdout)
        logged = logfile.read_text().splitlines()
        assert logged.count(tag_line) == 1
        assert any('FormingWorlds/MORS.git ->' in ln for ln in logged)

    @pytest.mark.parametrize('logged', [False, True], ids=['no-log-file', 'log-file'])
    def test_floor_without_tag_stops_the_installer(self, site, logged):
        """A floor with no release tag fails, names the distribution, and installs nothing.

        The new clone is removed, so a re-run clones again rather than
        installing the branch tip as an existing clone. With a log file the
        cause still reaches the terminal, in the last lines of the failed step.
        """
        env = {'LOGFILE': str(site['tmp'] / 'install.log')} if logged else {}
        result = self._call(site, 'JANUS', 'fwl-janus', **env)
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert 'Could not check out the fwl-janus release tag' in output
        assert "'fwl-janus' floor 26.10.08: no tag matches it" in output
        assert site['log'].read_text() == ''
        assert not (site['root'] / 'JANUS').exists()


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
        content = INSTALL_SH.read_text()
        assert content.count('proteus get phoenix ') == 2


_JULIA_FUNCTIONS = (
    'command_exists',
    'collect_env_info',
    'die',
    'prompt_yn',
    'resolve_real_julia_exe',
    'use_julia_channel',
    'warn_julia_111_deprecated',
    'add_julia_beside_default',
    'install_fresh_juliaup',
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


def _stub(path: Path, text: str) -> Path:
    """Write an executable shell script at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


class TestSetupJulia:
    """setup_julia adds the required Julia beside an unaccepted default and never moves it."""

    @pytest.fixture
    def site(self, tmp_path):
        """Stub julia, juliaup and python3 on a PATH without any other Julia.

        The 1.13 channel resolves to a scratch directory with its own julia, which
        reports 1.13.1, so a bare julia after the call shows which binary PATH finds.
        """
        stubs = tmp_path / 'stubs'
        _stub(stubs / 'julia', _STUB_JULIA)
        _stub(
            stubs / 'juliaup',
            '#!/bin/sh\necho "juliaup $*" >> "$STUB_LOG"\nexit "$STUB_JULIAUP_RC"\n',
        )
        _stub(stubs / 'python3', '#!/bin/sh\nexit 0\n')
        jl113 = tmp_path / 'juliaup' / 'julia-1.13.1' / 'bin'
        _stub(jl113 / 'julia', '#!/bin/sh\necho "julia version 1.13.1"\n')
        system = '/usr/bin:/bin'
        assert shutil.which('juliaup', path=system) is None
        assert shutil.which('julia', path=system) is None
        log = tmp_path / 'calls.log'
        log.write_text('')
        env = {
            'PATH': f'{stubs}{os.pathsep}{system}',
            'HOME': str(tmp_path / 'home'),
            'STUB_LOG': str(log),
            'STUB_BINDIR_113': str(jl113),
            'STUB_JULIAUP_RC': '0',
        }
        return {'stubs': stubs, 'jl113': jl113, 'log': log, 'env': env}

    @staticmethod
    def _call(site, version: str, **env: str) -> subprocess.CompletedProcess:
        """Run the shipped setup_julia with a default Julia of ``version``.

        After the call the script prints PATH, PYTHON_JULIAPKG_EXE, JULIA_CHANNEL
        and the version of the bare julia, one per line, prefixed with ``REPORT``.
        """
        defs = (
            'set -euo pipefail\nCURRENT_PHASE=2\nINTERACTIVE=false\n'
            + _install_sh_blocks("RED='", 'info()', 'REQUIRED_PYTHON_MAJOR=', 'JULIA_CHANNEL=')
            + ''.join(_install_sh_function(f) for f in _JULIA_FUNCTIONS)
        )
        report = (
            'setup_julia\n'
            'printf "REPORT %s\\n" "$PATH" "$PYTHON_JULIAPKG_EXE" "$JULIA_CHANNEL" "$(julia --version)"\n'
        )
        return subprocess.run(
            ['bash', '-c', defs + report],
            capture_output=True,
            text=True,
            env={**site['env'], 'STUB_JULIA_VERSION': version, **env},
            timeout=25,
        )

    @staticmethod
    def _report(result: subprocess.CompletedProcess) -> list[str]:
        """The four values the script prints after setup_julia returns."""
        return [ln[7:] for ln in result.stdout.splitlines() if ln.startswith('REPORT ')]

    @pytest.mark.parametrize(
        'version', ['1.10.4', '1.14.0'], ids=['default-below-set', 'default-above-set']
    )
    def test_unaccepted_default_gets_required_julia_beside_it(self, site, version):
        """An unaccepted default with juliaup gets `juliaup add 1.13` and no `juliaup default`.

        The 1.13 binary leads PATH, so a bare julia reports 1.13.1, not the stub
        default; PYTHON_JULIAPKG_EXE is that binary, not the launcher stub.
        """
        result = self._call(site, version)
        assert result.returncode == 0, result.stdout + result.stderr
        calls = site['log'].read_text().splitlines()
        assert 'juliaup add 1.13' in calls
        assert not [c for c in calls if c.startswith('juliaup default')]
        path, exe, channel, bare = self._report(result)
        assert path.split(os.pathsep)[0] == str(site['jl113'])
        assert path.split(os.pathsep)[1] == str(site['stubs'])
        assert exe == str(site['jl113'] / 'julia')
        assert channel == '+1.13'
        assert bare == 'julia version 1.13.1'
        assert (
            f'Julia 1.13 was added beside the default {version}, PROTEUS uses {exe}, '
            'the default is unchanged.'
        ) in result.stdout

    @pytest.mark.parametrize(
        'version', ['1.12.6', '1.11.9'], ids=['default-accepted', 'default-deprecated']
    )
    def test_accepted_default_calls_no_juliaup(self, site, version):
        """An accepted default (1.12, or the deprecated 1.11) calls juliaup not at all.

        PATH is left as it was and PYTHON_JULIAPKG_EXE is the default's binary;
        the 1.11 deprecation advice names no `juliaup default`.
        """
        result = self._call(site, version)
        assert result.returncode == 0, result.stdout + result.stderr
        calls = site['log'].read_text().splitlines()
        assert not [c for c in calls if c.startswith('juliaup ')]
        assert not [c for c in calls if c.startswith('julia +')]
        path, exe, channel, bare = self._report(result)
        assert path == site['env']['PATH']
        assert exe == str(site['stubs'] / 'julia')
        assert channel == ''
        assert bare == f'julia version {version}'
        assert ('deprecated' in result.stderr) is version.startswith('1.11')
        assert 'juliaup default' not in result.stdout + result.stderr

    def test_unaccepted_default_without_juliaup_stops(self, site):
        """Julia 1.10 without juliaup stops with advice that adds 1.13 and sets no default."""
        (site['stubs'] / 'juliaup').unlink()
        result = self._call(site, '1.10.4')
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert 'Julia version mismatch.' in output
        assert 'Install juliaup: curl -fsSL https://install.julialang.org | sh' in output
        assert 'Then: juliaup add 1.13' in output
        assert 'juliaup default' not in output
        assert self._report(result) == []
        assert not [c for c in site['log'].read_text().splitlines() if c.startswith('julia +')]

    def test_missing_channel_stops_instead_of_using_the_default(self, site):
        """A failed `juliaup add` with no 1.13 channel stops; the 1.10 default is never used.

        A probe that falls back to the julia on PATH would set PYTHON_JULIAPKG_EXE
        to the 1.10 launcher stub and return 0.
        """
        result = self._call(site, '1.10.4', STUB_JULIAUP_RC='1', STUB_BINDIR_113='')
        output = result.stdout + result.stderr
        assert result.returncode == 1
        assert "Could not locate the Julia 1.13 binary after 'juliaup add 1.13'." in output
        assert self._report(result) == []
        assert site['log'].read_text().splitlines().count('juliaup add 1.13') == 1


class TestJuliaupDefaultUse:
    """Only a juliaup installed by this run gets a new default Julia."""

    def test_juliaup_default_only_in_the_fresh_juliaup_install(self):
        """Every `juliaup default` line of install.sh lies in install_fresh_juliaup.

        The rule: the lines of install.sh holding `juliaup default` are exactly
        those of install_fresh_juliaup, which also downloads the juliaup installer,
        and there is one of them. doctor.py holds none.
        """
        everywhere = [
            ln for ln in INSTALL_SH.read_text().splitlines() if 'juliaup default' in ln
        ]
        fresh = _install_sh_function('install_fresh_juliaup')
        assert 'curl -fsSL https://install.julialang.org' in fresh
        assert everywhere == [ln for ln in fresh.splitlines() if 'juliaup default' in ln]
        assert [ln.strip() for ln in everywhere] == ['juliaup default "$req"']
        doctor = PROTEUS_ROOT / 'src' / 'proteus' / 'doctor.py'
        assert 'juliaup default' not in doctor.read_text()


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
    defs += _install_sh_blocks("RED='", 'info()') + ''.join(
        _install_sh_function(f) for f in _RUN_LOGGED_FUNCTIONS
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
_TERMINAL_WORDS = {'proteus', 'prompt_yn', 'read', 'read_with_default', 'juliaup', 'curl'}
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
        builds, terminal = _logged_and_terminal_calls(INSTALL_SH.read_text())
        assert {target for _, target in builds} == _BUILD_TARGETS
        assert [b for b in builds if b[0] != 'run_logged'] == []
        assert sum(1 for _, t in builds if t == 'pip install -e') == 2
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


class TestLogFileSentence:
    """die, run_logged and the exit hint name the log file in one sentence, once per failure."""

    @pytest.mark.parametrize(
        ('script', 'status', 'count'),
        [
            ('die "boom"\n', 1, 1),
            ('exit 3\n', 3, 1),
            ('run_logged "Building widget" bash five.sh 4\n', 4, 1),
            ('true\n', 0, 0),
        ],
        ids=['die', 'plain-exit', 'failed-step', 'clean-exit'],
    )
    def test_failed_run_names_the_log_file_once(self, tmp_path, script, status, count):
        """A failed run prints the sentence once, from whichever of the three ends it.

        die and run_logged print it and the exit hint then holds it back; a
        clean exit prints nothing.
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
