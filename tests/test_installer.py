"""Unit tests for the PROTEUS unified installer (install.sh).

Tests the pre-flight check logic, argument parsing, helper functions,
and error messages by running install.sh in controlled subprocess
environments. Does not execute actual installations (SOCRATES, AGNI,
pip install); those are tested by the CI pipeline and smoke tests.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os
import subprocess

import pytest
from helpers import PROTEUS_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

INSTALL_SH = PROTEUS_ROOT / 'install.sh'


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
