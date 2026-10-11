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


class TestIdempotency:
    """Verify the installer is safe to re-run."""

    def test_append_export_is_idempotent(self, tmp_path):
        """append_export_to_rc replaces existing lines rather than duplicating.

        Discrimination: running the function 3 times must result in
        exactly one copy. A naive append would produce 3 copies.
        """
        rc_file = tmp_path / '.testrc'
        rc_file.write_text('# existing content\nexport OTHER_VAR=hello\n')

        for _ in range(3):
            subprocess.run(
                [
                    'bash',
                    '-c',
                    f"""
                    append_export_to_rc() {{
                        local var_name="$1" var_value="$2" rc_file="$3"
                        local safe_value
                        safe_value=$(printf '%q' "$var_value")
                        local line="export ${{var_name}}=${{safe_value}}"
                        if [ -f "$rc_file" ]; then
                            grep -v "^export ${{var_name}}=" "$rc_file" > "${{rc_file}}.tmp" 2>/dev/null || true
                            mv "${{rc_file}}.tmp" "$rc_file"
                        fi
                        echo "$line" >> "$rc_file"
                    }}
                    append_export_to_rc 'TEST_VAR' '/some/path' '{rc_file}'
                    """,
                ],
                check=True,
                timeout=10,
            )

        content = rc_file.read_text()
        count = content.count('export TEST_VAR=')
        assert count == 1, f'Line appears {count} times (expected 1)'
        assert 'export OTHER_VAR=hello' in content

    def test_append_export_handles_spaces_in_path(self, tmp_path):
        """Paths with spaces are safely quoted via printf %q.

        Discrimination: the written RC line must be valid shell that
        sets the variable to the exact path including spaces.
        """
        rc_file = tmp_path / '.testrc'
        rc_file.write_text('')
        path_with_spaces = '/home/user name/FWL DATA'

        subprocess.run(
            [
                'bash',
                '-c',
                f"""
                append_export_to_rc() {{
                    local var_name="$1" var_value="$2" rc_file="$3"
                    local safe_value
                    safe_value=$(printf '%q' "$var_value")
                    local line="export ${{var_name}}=${{safe_value}}"
                    if [ -f "$rc_file" ]; then
                        grep -v "^export ${{var_name}}=" "$rc_file" > "${{rc_file}}.tmp" 2>/dev/null || true
                        mv "${{rc_file}}.tmp" "$rc_file"
                    fi
                    echo "$line" >> "$rc_file"
                }}
                append_export_to_rc 'FWL_DATA' '{path_with_spaces}' '{rc_file}'
                """,
            ],
            check=True,
            timeout=10,
        )

        content = rc_file.read_text()
        assert 'export FWL_DATA=' in content
        # Verify the written line is valid shell that produces the correct value
        result = subprocess.run(
            ['bash', '-c', f'source {rc_file} && echo "$FWL_DATA"'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.stdout.strip() == path_with_spaces


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
    defs = 'set -euo pipefail\n' + _install_sh_blocks("RED='", 'info()', 'detect_shell_rc() {')
    return subprocess.run(
        ['bash', '-c', defs + script],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=10,
    )


class TestShellRcDetection:
    """Verify the rc file path the installer captures from detect_shell_rc."""

    @pytest.mark.parametrize(
        ('shell', 'rc', 'warns'),
        [
            ('/usr/local/bin/fish', '.bashrc', True),
            ('/bin/zsh', '.zshrc', False),
            ('/bin/bash', '.bashrc', False),
        ],
    )
    def test_captured_rc_file_is_only_the_path(self, tmp_path, shell, rc, warns):
        """RC_FILE holds exactly the path; the unsupported-shell warning goes to stderr.

        Discrimination: with warn on stdout, the fish case captures the
        warning line in front of the path, so the equality fails.
        """
        out = _run_with_helpers(
            'RC_FILE=$(detect_shell_rc)\nprintf "%s" "$RC_FILE"\n',
            SHELL=shell,
            HOME=str(tmp_path),
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout == str(tmp_path / rc)
        assert ("Unsupported shell 'fish'" in out.stderr) is warns

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
            **os.environ,
            'PATH': f'{stubs}{os.pathsep}{os.environ["PATH"]}',
            'GH_PREFIX': f'{tmp_path / "remotes"}/',
            'SCRIPT_DIR': str(root),
            'STUB_PIP_LOG': str(log),
        }
        return {'root': root, 'shas': shas, 'log': log, 'env': env}

    @staticmethod
    def _call(site, *args: str) -> subprocess.CompletedProcess:
        """Run the shipped clone_and_install with ``args``, under install.sh's options."""
        defs = (
            'set -euo pipefail\nCURRENT_PHASE=6\n'
            + _install_sh_blocks("RED='", 'info()')
            + ''.join(
                _install_sh_function(f)
                for f in ('command_exists', 'collect_env_info', 'die', 'clone_and_install')
            )
        )
        return subprocess.run(
            ['bash', '-c', defs + shlex.join(['clone_and_install', *args])],
            capture_output=True,
            text=True,
            env=site['env'],
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

    def test_floor_without_tag_stops_the_installer(self, site):
        """A floor with no release tag fails, names the distribution, and installs nothing.

        The new clone is removed, so a re-run clones again rather than
        installing the branch tip as an existing clone.
        """
        result = self._call(site, 'JANUS', 'fwl-janus')
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
