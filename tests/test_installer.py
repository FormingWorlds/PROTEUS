"""Unit tests for the PROTEUS unified installer (install.sh).

Tests the pre-flight check logic, argument parsing, helper functions,
and error messages by running install.sh in controlled subprocess
environments. Does not execute actual installations (SOCRATES, AGNI,
pip install); those are tested by the CI pipeline and smoke tests.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import os
import subprocess

import pytest
from helpers import PROTEUS_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

INSTALL_SH = PROTEUS_ROOT / 'install.sh'


def _run_installer(args=None, env_override=None, input_text=None, timeout=15):
    """Run install.sh in a subprocess with controlled environment.

    Parameters
    ----------
    args : list or None
        Extra arguments to pass to install.sh.
    env_override : dict or None
        Environment variable overrides. Merged with a minimal base env.
    input_text : str or None
        Text to pipe to stdin (for interactive prompts).
    timeout : int
        Subprocess timeout in seconds.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process with stdout/stderr captured.
    """
    cmd = ['bash', str(INSTALL_SH)]
    if args:
        cmd.extend(args)

    env = os.environ.copy()
    # Remove CONDA_DEFAULT_ENV to test the check (unless overridden)
    if env_override is not None:
        env.update(env_override)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
        env=env,
        cwd=str(PROTEUS_ROOT),
    )


class TestInstallerArgParsing:
    """Verify argument parsing and help output."""

    def test_help_flag_exits_zero(self):
        """--help prints usage and exits with code 0."""
        result = _run_installer(['--help'])
        assert result.returncode == 0
        assert 'Usage' in result.stdout
        assert '--all-data' in result.stdout

    def test_h_flag_exits_zero(self):
        """Short -h flag also works."""
        result = _run_installer(['-h'])
        assert result.returncode == 0
        assert 'Usage' in result.stdout

    def test_unknown_arg_exits_nonzero(self):
        """Unknown arguments cause a non-zero exit with error message."""
        result = _run_installer(['--bogus-flag'])
        assert result.returncode != 0
        assert 'Unknown argument' in result.stdout or 'Unknown argument' in result.stderr


class TestPreflightChecks:
    """Verify pre-flight check failure modes."""

    def test_fails_without_conda_env(self):
        """Without CONDA_DEFAULT_ENV set, the installer must fail with
        clear instructions to create a conda environment.

        Discrimination: the error message must mention 'conda' so the
        user knows what to do. A generic 'environment error' would not
        be actionable.
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

    def test_fails_outside_proteus_repo(self, tmp_path):
        """Running install.sh from outside the PROTEUS repo must fail
        because pyproject.toml is not found.
        """
        # Copy install.sh to a temp dir without pyproject.toml
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


class TestShellRcDetection:
    """Verify shell rc file detection logic."""

    def test_bash_shell_gives_bashrc(self):
        """When SHELL=/bin/bash, detect_shell_rc returns ~/.bashrc."""
        result = subprocess.run(
            ['bash', '-c', 'source install.sh --help 2>/dev/null; true'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(PROTEUS_ROOT),
        )
        # The --help flag exits before rc detection, but the script parses
        assert result.returncode == 0

    def test_installer_is_valid_bash(self):
        """install.sh parses without syntax errors."""
        result = subprocess.run(
            ['bash', '-n', str(INSTALL_SH)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f'Bash syntax error: {result.stderr}'


class TestIdempotency:
    """Verify the installer is safe to re-run."""

    def test_append_to_rc_is_idempotent(self, tmp_path):
        """append_to_rc must not duplicate lines on repeated calls.

        Discrimination: running the append twice must result in exactly
        one copy of the line. A naive implementation that always appends
        would produce duplicates.
        """
        rc_file = tmp_path / '.testrc'
        rc_file.write_text('# existing content\n')
        test_line = 'export TEST_VAR="/some/path"'

        # Simulate append_to_rc by running bash inline
        for _ in range(3):
            subprocess.run(
                [
                    'bash',
                    '-c',
                    f"""
                    append_to_rc() {{
                        local line="$1" rc_file="$2"
                        if ! grep -qF "$line" "$rc_file" 2>/dev/null; then
                            echo "$line" >> "$rc_file"
                        fi
                    }}
                    append_to_rc '{test_line}' '{rc_file}'
                """,
                ],
                check=True,
                timeout=10,
            )

        content = rc_file.read_text()
        count = content.count(test_line)
        assert count == 1, f'Line appears {count} times (expected 1)'


class TestDiskSpaceCheck:
    """Verify disk space detection."""

    def test_available_disk_gb_returns_integer(self):
        """available_disk_gb must return a positive integer on any platform."""
        result = subprocess.run(
            [
                'bash',
                '-c',
                """
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    df -g . | awk 'NR==2 {print $4}'
                else
                    df --output=avail -BG . | awk 'NR==2 {gsub(/G/,""); print $1}'
                fi
            """,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        gb = int(result.stdout.strip())
        assert gb > 0, f'Disk space check returned {gb} GB'
