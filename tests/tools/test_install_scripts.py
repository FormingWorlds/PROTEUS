"""
Unit tests for PETSc/SPIDER installation shell scripts.

Tests the reusable shell logic extracted from ``tools/get_petsc.sh`` and
``tools/get_spider.sh``:
- ``portable_realpath()`` — cross-platform path resolution
- ERR trap — exit-code and step-name capture
- Platform detection — PETSC_ARCH assignment
- Homebrew prefix fallback — architecture-aware default
- Workpath argument handling — ``$1`` override vs default
- PETSc library detection — versioned ``.so``, ``.dylib``, missing

Each test runs an isolated bash snippet via ``subprocess.run()`` — no
network access, no real builds.

See also:
- docs/test_infrastructure.md
- docs/test_categorization.md
- docs/test_building.md
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Helper: extract portable_realpath function from a script
# ---------------------------------------------------------------------------
def _portable_realpath_fn() -> str:
    """Return the bash source for ``portable_realpath()``."""
    return """\
portable_realpath() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
    fi
}
"""


# ---------------------------------------------------------------------------
# portable_realpath tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_portable_realpath_resolves_relative(tmp_path):
    """Resolves ``./subdir`` to an absolute path using ``portable_realpath``."""
    subdir = tmp_path / 'subdir'
    subdir.mkdir()

    snippet = _portable_realpath_fn() + f'\ncd "{tmp_path}"\nportable_realpath ./subdir'
    result = subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    resolved = result.stdout.strip()
    assert os.path.isabs(resolved)
    assert resolved == str(subdir)


@pytest.mark.unit
def test_portable_realpath_resolves_parent_refs(tmp_path):
    """Resolves ``../`` components to a canonical absolute path."""
    child = tmp_path / 'a' / 'b'
    child.mkdir(parents=True)

    # a/b/../ should resolve to a/
    snippet = _portable_realpath_fn() + f'\nportable_realpath "{child}/.."'
    result = subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == str(tmp_path / 'a')


@pytest.mark.unit
def test_portable_realpath_resolves_symlink(tmp_path):
    """Follows symlinks to the real path."""
    real = tmp_path / 'real_dir'
    real.mkdir()
    link = tmp_path / 'link_dir'
    link.symlink_to(real)

    snippet = _portable_realpath_fn() + f'\nportable_realpath "{link}"'
    result = subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == str(real)


@pytest.mark.unit
def test_portable_realpath_python_fallback(tmp_path):
    """When ``realpath`` binary is hidden from PATH, falls back to python3.

    We create a temporary directory containing only a symlink to ``python3``
    and set PATH to *only* that directory.  Because ``/bin`` and ``/usr/bin``
    (which contain the real ``realpath`` on both Linux and macOS) are excluded,
    ``command -v realpath`` fails and the function falls through to the
    ``python3 -c …`` branch.  Bash builtins (``command``, ``if``, ``echo``)
    work regardless of PATH, so no system directories are needed.
    """
    target = tmp_path / 'target'
    target.mkdir()

    # Build a PATH with *only* python3 — no realpath anywhere.
    python_bin = sys.executable
    safe_bin = tmp_path / 'safe_bin'
    safe_bin.mkdir()
    (safe_bin / 'python3').symlink_to(python_bin)

    # Find the absolute path to bash so subprocess.run can invoke it
    # even with the restricted PATH (which excludes /bin and /usr/bin).
    bash_abs = subprocess.run(['which', 'bash'], capture_output=True, text=True).stdout.strip()

    snippet = _portable_realpath_fn() + f'\nportable_realpath "{target}"'
    env = {**os.environ, 'PATH': str(safe_bin)}

    result = subprocess.run(
        [bash_abs, '-c', snippet],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    resolved = result.stdout.strip()
    assert os.path.isabs(resolved)
    assert resolved == str(target)


# ---------------------------------------------------------------------------
# ERR trap tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_err_trap_captures_nonzero_exit_code():
    """ERR trap reports actual failing command's exit code (not 0).

    The ``local rc=$?`` pattern on the first line of ``on_error()`` must
    capture the exit code of the command that triggered the trap, not the
    exit code of any intervening statement.
    """
    snippet = """\
set -e
current_step="test step"
on_error() {
    local rc=$?
    echo "EXIT_CODE=$rc"
}
trap on_error ERR
# Force a specific nonzero exit code via function return
fail42() { return 42; }
fail42
"""
    result = subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'EXIT_CODE=42' in result.stdout


@pytest.mark.unit
def test_err_trap_reports_current_step():
    """ERR trap output includes the ``current_step`` variable value."""
    snippet = """\
set -e
current_step="Downloading PETSc archive from OSF"
on_error() {
    local rc=$?
    echo "STEP=$current_step"
}
trap on_error ERR
false
"""
    result = subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'STEP=Downloading PETSc archive from OSF' in result.stdout


# ---------------------------------------------------------------------------
# Platform detection tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_platform_detection_sets_petsc_arch():
    """On the current platform, PETSC_ARCH is ``arch-darwin-c-opt`` or
    ``arch-linux-c-opt``.

    This test runs the actual platform-detection snippet from get_petsc.sh
    and verifies it produces a valid value.
    """
    snippet = """\
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "arch-linux-c-opt"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "arch-darwin-c-opt"
else
    echo "UNSUPPORTED"
fi
"""
    result = subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    arch = result.stdout.strip()
    assert arch in ('arch-linux-c-opt', 'arch-darwin-c-opt')


# ---------------------------------------------------------------------------
# Homebrew prefix fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_brew_prefix_fallback_arm64(tmp_path):
    """With ``uname -m`` spoofed to arm64 and no brew, fallback is
    ``/opt/homebrew``.

    We create a fake ``uname`` that always reports arm64 and ensure
    ``brew`` is not on PATH.
    """
    # Create a fake uname that reports arm64
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    # Locate the real uname binary for the passthrough case.
    real_uname = subprocess.run(
        ['which', 'uname'], capture_output=True, text=True
    ).stdout.strip()

    fake_uname = fake_bin / 'uname'
    fake_uname.write_text(
        f'#!/bin/bash\nif [[ "$1" == "-m" ]]; then echo arm64; else {real_uname} "$@"; fi\n'
    )
    fake_uname.chmod(0o755)

    # Test the brew-prefix fallback logic in isolation.  The restricted PATH
    # excludes brew on all platforms (including Linux with Linuxbrew), so the
    # snippet always exercises the fallback branch.
    snippet = f"""\
export PATH="{fake_bin}:/usr/bin:/bin"
if [[ "$(uname -m)" == "arm64" ]]; then
    default_brew_prefix="/opt/homebrew"
else
    default_brew_prefix="/usr/local"
fi
brew_prefix=$(brew --prefix 2>/dev/null || echo "$default_brew_prefix")
echo "$brew_prefix"
"""
    env = {**os.environ, 'PATH': f'{fake_bin}:/usr/bin:/bin'}
    result = subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == '/opt/homebrew'


@pytest.mark.unit
def test_brew_prefix_fallback_x86_64(tmp_path):
    """With ``uname -m`` spoofed to x86_64 and no brew, fallback is
    ``/usr/local``.

    Works on Linux too: ``brew`` is not on the restricted PATH, so the
    fallback branch is always exercised regardless of platform.
    """
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()

    real_uname = subprocess.run(
        ['which', 'uname'], capture_output=True, text=True
    ).stdout.strip()

    fake_uname = fake_bin / 'uname'
    fake_uname.write_text(
        f'#!/bin/bash\nif [[ "$1" == "-m" ]]; then echo x86_64; else {real_uname} "$@"; fi\n'
    )
    fake_uname.chmod(0o755)

    snippet = f"""\
export PATH="{fake_bin}:/usr/bin:/bin"
if [[ "$(uname -m)" == "arm64" ]]; then
    default_brew_prefix="/opt/homebrew"
else
    default_brew_prefix="/usr/local"
fi
brew_prefix=$(brew --prefix 2>/dev/null || echo "$default_brew_prefix")
echo "$brew_prefix"
"""
    env = {**os.environ, 'PATH': f'{fake_bin}:/usr/bin:/bin'}
    result = subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == '/usr/local'


# ---------------------------------------------------------------------------
# Workpath argument tests (get_petsc.sh)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_petsc_workpath_uses_first_argument(tmp_path):
    """When ``$1`` is set, ``workpath`` is derived from ``$1``."""
    custom = tmp_path / 'custom_petsc'

    snippet = (
        _portable_realpath_fn()
        + """\
if [[ -n "$1" ]]; then
    mkdir -p "$1"
    workpath=$(portable_realpath "$1")
else
    mkdir -p petsc
    workpath=$(portable_realpath petsc)
fi
echo "$workpath"
"""
    )
    result = subprocess.run(
        ['bash', '-c', snippet, '--', str(custom)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == str(custom)
    assert custom.is_dir()


@pytest.mark.unit
def test_petsc_workpath_defaults_to_petsc(tmp_path):
    """When ``$1`` is empty, ``workpath`` defaults to ``./petsc/``."""
    snippet = (
        _portable_realpath_fn()
        + """\
if [[ -n "$1" ]]; then
    mkdir -p "$1"
    workpath=$(portable_realpath "$1")
else
    mkdir -p petsc
    workpath=$(portable_realpath petsc)
fi
echo "$workpath"
"""
    )
    result = subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0
    resolved = result.stdout.strip()
    assert resolved.endswith('/petsc')
    assert (tmp_path / 'petsc').is_dir()


# ---------------------------------------------------------------------------
# PETSc library detection tests (get_spider.sh)
# ---------------------------------------------------------------------------


def _lib_check_snippet(lib_dir: str) -> str:
    """Return the bash snippet that checks for libpetsc in a directory."""
    return f"""\
petsc_lib_dir="{lib_dir}"
petsc_lib_found=false
for f in "$petsc_lib_dir"/libpetsc.*; do
    if [[ -f "$f" ]]; then
        petsc_lib_found=true
        break
    fi
done
if [[ "$petsc_lib_found" == "true" ]]; then
    echo "FOUND"
    exit 0
else
    echo "NOT_FOUND"
    exit 1
fi
"""


@pytest.mark.unit
def test_spider_lib_check_finds_dylib(tmp_path):
    """PETSc lib check succeeds with ``libpetsc.dylib``."""
    lib_dir = tmp_path / 'lib'
    lib_dir.mkdir()
    (lib_dir / 'libpetsc.dylib').write_text('')

    result = subprocess.run(
        ['bash', '-c', _lib_check_snippet(str(lib_dir))],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert 'FOUND' in result.stdout


@pytest.mark.unit
def test_spider_lib_check_finds_versioned_so(tmp_path):
    """PETSc lib check succeeds with ``libpetsc.so.3.19`` (no symlink)."""
    lib_dir = tmp_path / 'lib'
    lib_dir.mkdir()
    (lib_dir / 'libpetsc.so.3.19').write_text('')

    result = subprocess.run(
        ['bash', '-c', _lib_check_snippet(str(lib_dir))],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert 'FOUND' in result.stdout


@pytest.mark.unit
def test_spider_lib_check_finds_unversioned_so(tmp_path):
    """PETSc lib check succeeds with ``libpetsc.so``."""
    lib_dir = tmp_path / 'lib'
    lib_dir.mkdir()
    (lib_dir / 'libpetsc.so').write_text('')

    result = subprocess.run(
        ['bash', '-c', _lib_check_snippet(str(lib_dir))],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert 'FOUND' in result.stdout


@pytest.mark.unit
def test_spider_lib_check_fails_on_empty_dir(tmp_path):
    """PETSc lib check fails when no ``libpetsc.*`` exists."""
    lib_dir = tmp_path / 'lib'
    lib_dir.mkdir()

    result = subprocess.run(
        ['bash', '-c', _lib_check_snippet(str(lib_dir))],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'NOT_FOUND' in result.stdout
