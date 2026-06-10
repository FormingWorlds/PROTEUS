"""
Unit tests for PETSc/SPIDER installation shell scripts.

Tests the reusable shell logic extracted from ``tools/get_petsc.sh`` and
``tools/get_spider.sh``:
- ``portable_realpath()``: cross-platform path resolution
- ERR trap: exit-code and step-name capture
- Platform detection: PETSC_ARCH assignment
- Homebrew prefix fallback: architecture-aware default
- Workpath argument handling: ``$1`` override vs default
- PETSc library detection: versioned ``.so``, ``.dylib``, missing

Each test runs an isolated bash snippet via ``subprocess.run()``, with no
network access and no real builds.

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

    # Build a PATH with *only* python3, no realpath anywhere.
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


# ============================================================================
# Regression: installation.md does not promote editable installs of PyPI deps
# ============================================================================


import re  # noqa: E402
import tomllib  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_installation_md_does_not_clone_aragog_or_zalmoxis():
    """Regression for PR #673 follow-up: installation.md must not tell
    users to ``git clone`` and ``pip install -e`` Aragog or Zalmoxis.
    These are PyPI deps (``fwl-aragog``, ``fwl-zalmoxis``) declared in
    pyproject.toml and installed automatically by
    ``pip install -e ".[develop]"``. Re-introducing editable-install
    instructions silently shadows the PyPI versions and breaks the
    documented dependency pinning.
    """
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / 'docs' / 'How-to' / 'installation.md').read_text(encoding='utf-8')
    # Match `git clone <url>/<aragog|Zalmoxis>` or `pip install -e <aragog|Zalmoxis>`.
    forbidden = re.compile(
        r'(git\s+clone[^\n]*?(aragog|Zalmoxis))'
        r'|(pip\s+install\s+-e\s+(aragog|Zalmoxis))',
        re.IGNORECASE,
    )
    matches = forbidden.findall(text)
    assert not matches, (
        f'installation.md re-introduced editable-install of Aragog/Zalmoxis: {matches!r}'
    )
    # Discrimination: installation.md must still cover the PyPI install
    # path. An empty file would also have zero matches above but would
    # silently delete the install instructions; pin the canonical PyPI
    # package name as evidence the file still documents the supported
    # path.
    assert 'fwl-aragog' in text or 'pip install -e ".[develop]"' in text


@pytest.mark.unit
def test_pyproject_pins_aragog_and_zalmoxis_pypi_packages():
    """Companion guarantee: pyproject.toml must continue pinning the
    PyPI distributions ``fwl-aragog`` and ``fwl-zalmoxis``. If either
    pin is removed, the rationale for not editable-installing them
    breaks and installation.md must be rewritten."""
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / 'pyproject.toml').read_text(encoding='utf-8')
    assert 'fwl-aragog' in text, 'pyproject.toml must pin fwl-aragog'
    assert 'fwl-zalmoxis' in text, 'pyproject.toml must pin fwl-zalmoxis'


@pytest.mark.unit
def test_pyproject_keeps_boreas_out_of_mandatory_dependencies():
    """BOREAS is installed only explicitly, via ``bash tools/get_boreas.sh``.

    Two clauses:
    1. ``[project] dependencies`` must not list boreas. Re-adding the
       direct git URL there would make BOREAS mandatory again and would
       also block PyPI uploads of fwl-proteus, since PyPI rejects
       packages whose dependency metadata contains direct references.
       The PyPI name ``boreas`` belongs to an unrelated project, so a
       plain ``boreas`` version pin would resolve to the wrong package.
    2. The pin lives in ``[tool.proteus.modules.boreas]`` with the
       ExoInteriors GitHub URL and a full 40-character commit SHA, which
       tools/get_boreas.sh and the CI setup action resolve through
       tools/_module_pins.py.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / 'pyproject.toml').read_text(encoding='utf-8'))

    deps = data['project']['dependencies']
    boreas_deps = [d for d in deps if 'boreas' in d.lower()]
    assert boreas_deps == [], (
        f'boreas must not be a mandatory dependency of fwl-proteus: {boreas_deps!r}'
    )
    # Discrimination: an empty dependencies list would also pass the
    # check above; pin a known-mandatory package as evidence the list
    # is intact.
    assert any('fwl-calliope' in d for d in deps), 'mandatory dependency list is intact'

    spec = data['tool']['proteus']['modules']['boreas']
    assert spec['url'].startswith('https://github.com/ExoInteriors/BOREAS'), (
        f'boreas pin must point at the ExoInteriors repo, got {spec["url"]!r}'
    )
    # Full-SHA pin: reproducible clone, short refs are ambiguous and
    # mutable upstream.
    assert re.fullmatch(r'[0-9a-f]{40}', spec['ref']), (
        f'boreas ref must be a full commit SHA, got {spec["ref"]!r}'
    )


@pytest.mark.unit
def test_optional_backends_vulcan_atmodeller_are_extras_not_mandatory():
    """VULCAN and atmodeller are optional backends, installed on demand.

    A standard PROTEUS run uses CALLIOPE for outgassing and no atmospheric
    chemistry, so neither package should be a mandatory dependency. Each
    must instead live in ``[project.optional-dependencies]`` under its own
    extra, keeping the published version pins so the optional install is
    reproducible. Re-adding either to ``[project] dependencies`` would force
    every PROTEUS user to pull a GPL-3.0 package the core does not need.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / 'pyproject.toml').read_text(encoding='utf-8'))

    deps = data['project']['dependencies']
    mandatory = [d for d in deps if 'vulcan' in d.lower() or 'atmodeller' in d.lower()]
    assert mandatory == [], (
        f'vulcan/atmodeller must not be mandatory dependencies: {mandatory!r}'
    )
    # Discrimination: an empty or truncated dependency list would also pass
    # the check above; confirm a known-mandatory backend is still present.
    assert any('fwl-calliope' in d for d in deps), 'mandatory dependency list is intact'

    extras = data['project']['optional-dependencies']
    # Membership, not exact-list: an extra may gain a second requirement
    # later (e.g. a transitive pin) without this guard going stale.
    assert any(r.startswith('atmodeller>=1.0.0') for r in extras.get('atmodeller', [])), (
        f'atmodeller extra must keep its pin, got {extras.get("atmodeller")!r}'
    )
    assert any(r.startswith('fwl-vulcan>=26.04.22') for r in extras.get('vulcan', [])), (
        f'vulcan extra must keep its pin, got {extras.get("vulcan")!r}'
    )
    # The default Aragog interior solver runs on JAX and its modules are
    # equinox Modules, so the jax/equinox stack must stay MANDATORY, not be
    # gated behind the optional atmodeller extra. equinox==0.13.2 calls
    # jax.core.mapped_aval (removed in jax 0.10), which is why jax/jaxlib are
    # pinned <0.10. Lifting any of these would break a standard run.
    assert 'jax<0.10' in deps and 'jaxlib<0.10' in deps, (
        'jax/jaxlib must stay pinned <0.10 for the default Aragog jax solver'
    )
    assert 'equinox==0.13.2' in deps, (
        'equinox must be a mandatory dependency for the default Aragog jax solver; '
        'it was previously pulled only transitively via atmodeller'
    )

    # VULCAN is a single-source PyPI package like fwl-aragog/fwl-zalmoxis: its
    # only pin is the extra above, and tools/get_vulcan.sh checks out the git
    # tag matching that floor. It must NOT also carry a [tool.proteus.modules]
    # SHA pin, which could drift from the PyPI release (the dual-pin trap).
    git_modules = data['tool']['proteus']['modules']
    assert 'vulcan' not in git_modules, (
        'vulcan must not have a [tool.proteus.modules] git pin; it is pinned '
        'once via the fwl-vulcan extra and the matching git tag, like '
        f'fwl-aragog/fwl-zalmoxis. Found: {sorted(git_modules)}'
    )


@pytest.mark.unit
def test_ci_setup_installs_every_declared_extra():
    """The CI setup action must install extras whose keys exist in pyproject.

    The CI composite action installs PROTEUS with a literal extras list, e.g.
    ``pip install -e ".[develop,vulcan,atmodeller]"``. pip treats an unknown
    extra as a warning and still exits 0, so a typo or a renamed extra would
    silently stop installing the optional backends, and their tests would skip
    instead of failing. This guard ties the CI string to the pyproject extra
    keys: every extra named in the action must be a real optional-dependency
    group.
    """
    repo_root = Path(__file__).resolve().parents[2]
    action = (repo_root / '.github/actions/setup-proteus/action.yml').read_text(
        encoding='utf-8'
    )
    data = tomllib.loads((repo_root / 'pyproject.toml').read_text(encoding='utf-8'))
    extra_keys = set(data['project']['optional-dependencies'])

    # Extract the bracketed extras from the `pip install -e ".[...]"` line(s).
    matches = re.findall(r'pip install -e "\.\[([^\]]+)\]"', action)
    assert matches, (
        'no `pip install -e ".[...]"` line found in setup-proteus action; '
        'the extras-install guard cannot verify CI'
    )
    ci_extras = {e.strip() for group in matches for e in group.split(',')}
    # The two optional physics backends must be installed by CI so their tests
    # run rather than skip.
    assert {'vulcan', 'atmodeller'} <= ci_extras, (
        f'CI must install the vulcan + atmodeller extras; found {sorted(ci_extras)}'
    )
    # Every extra named in CI must be a real pyproject extra (catches typos /
    # renames that pip would otherwise swallow).
    unknown = ci_extras - extra_keys
    assert not unknown, (
        f'CI references extras not declared in pyproject [project.optional-dependencies]: '
        f'{sorted(unknown)}; known extras are {sorted(extra_keys)}'
    )


# ---------------------------------------------------------------------------
# Dirty-checkout guard (shared shape across tools/get_*.sh)
# ---------------------------------------------------------------------------


def _extract_guard_block() -> str:
    """Extract the shipped dirty-checkout guard from tools/get_aragog.sh.

    Reading the block from the script under test (rather than copying it
    into the test) pins the exact shipped lines: any rewording or logic
    change in the guard re-runs through these cases.
    """
    from pathlib import Path

    tools_dir = Path(__file__).resolve().parents[2] / 'tools'
    script = (tools_dir / 'get_aragog.sh').read_text().splitlines()
    start = next(i for i, ln in enumerate(script) if 'Refuse to delete a checkout' in ln)
    end = next(i for i, ln in enumerate(script) if ln.startswith('rm -rf'))
    return '\n'.join(script[start:end])


def _run_guard(tmp_path, *args: str) -> subprocess.CompletedProcess:
    """Run the extracted guard with ``root`` pointing at ``tmp_path``."""
    snippet = 'root="$GUARD_ROOT"\n' + _extract_guard_block() + '\necho GUARD_PASSED\n'
    return subprocess.run(
        ['bash', '-c', snippet, 'guard', *args],
        capture_output=True,
        text=True,
        env={**os.environ, 'GUARD_ROOT': str(tmp_path)},
    )


def _git(cwd, *args: str) -> None:
    subprocess.run(
        ['git', '-c', 'user.email=t@e.st', '-c', 'user.name=t', *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_guard_blocks_dirty_and_unpushed_checkouts(tmp_path):
    """Tracked modifications and local-only commits block the refresh.

    A modified tracked file must exit 1 with the recovery command in the
    message; a repo whose commits exist on no remote (covers both the
    remote-less and the never-pushed case) must also block. Untracked
    files alone must NOT block: build artifacts and egg-info dirs are
    routine in refreshed checkouts.
    """
    workdir = tmp_path / 'aragog'
    workdir.mkdir()
    _git(workdir, 'init', '-q')
    (workdir / 'tracked.py').write_text('x = 1\n')
    _git(workdir, 'add', 'tracked.py')
    _git(workdir, 'commit', '-q', '-m', 'c1')

    # Local-only commit (no remotes at all): blocked.
    res = _run_guard(tmp_path)
    assert res.returncode == 1
    assert '--force' in res.stderr  # recovery command is named
    assert 'GUARD_PASSED' not in res.stdout

    # Same state plus a dirty tracked file: still blocked.
    (workdir / 'tracked.py').write_text('x = 2\n')
    res = _run_guard(tmp_path)
    assert res.returncode == 1
    assert 'uncommitted changes' in res.stderr


def test_guard_passes_clean_remote_backed_checkout(tmp_path):
    """A clean checkout whose commits are on a remote is refreshed.

    Mimics the normal installed state: a clone (origin exists), detached
    HEAD at a pinned ref, untracked build artifacts present. The guard
    must stay silent. A local commit on the detached HEAD then blocks:
    the commit exists on no remote and would be destroyed.
    """
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    _git(upstream, 'init', '-q')
    (upstream / 'f.py').write_text('a = 1\n')
    _git(upstream, 'add', 'f.py')
    _git(upstream, 'commit', '-q', '-m', 'c1')

    workdir = tmp_path / 'aragog'
    _git(tmp_path, 'clone', '-q', str(upstream), str(workdir))
    _git(workdir, 'checkout', '-q', '--detach', 'HEAD')
    (workdir / 'build_artifact.o').write_text('')  # untracked: must not block

    res = _run_guard(tmp_path)
    assert res.returncode == 0
    assert 'GUARD_PASSED' in res.stdout

    # Local commit on the detached HEAD: reachable from HEAD, on no
    # remote. This is the state a tag-pinned checkout enters when a
    # developer commits without branching; it must block.
    (workdir / 'f.py').write_text('a = 2\n')
    _git(workdir, 'add', 'f.py')
    _git(workdir, 'commit', '-q', '-m', 'local work')
    res = _run_guard(tmp_path)
    assert res.returncode == 1
    assert 'not on a remote' in res.stderr

    # --force bypasses deliberately.
    res = _run_guard(tmp_path, '--force')
    assert res.returncode == 0
    assert 'GUARD_PASSED' in res.stdout


# ---------------------------------------------------------------------------
# Portable-flag rewrite and guards (tools/get_socrates.sh)
# ---------------------------------------------------------------------------


def _extract_socrates_block(start_marker: str, end_marker: str) -> str:
    """Extract shipped lines of tools/get_socrates.sh between two markers.

    Reading the block from the script under test (rather than copying it
    into the test) pins the exact shipped lines: any rewording or logic
    change in the flag handling re-runs through these cases.
    """
    tools_dir = Path(__file__).resolve().parents[2] / 'tools'
    script = (tools_dir / 'get_socrates.sh').read_text().splitlines()
    start = next(i for i, ln in enumerate(script) if start_marker in ln)
    end = next(i for i, ln in enumerate(script) if end_marker in ln and i > start)
    return '\n'.join(script[start:end])


def _run_flag_rewrite(workdir) -> subprocess.CompletedProcess:
    """Run the shipped flag rewrite and its guards against a fixture tree."""
    block = _extract_socrates_block(
        'Compile against a portable instruction set', './build_code'
    )
    snippet = 'set -euo pipefail\ncd "$WORKDIR"\n' + block + '\necho REWRITE_OK\n'
    return subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
        env={**os.environ, 'WORKDIR': str(workdir)},
    )


def _run_post_build_guard(workdir) -> subprocess.CompletedProcess:
    """Run the shipped post-build flag check against a fixture bin/Mk_cmd."""
    pattern_line = _extract_socrates_block('nonportable_flags=', 'Belt-and-braces')
    block = _extract_socrates_block('Verify the flags that reached', '# Environment')
    snippet = (
        'set -euo pipefail\ncd "$WORKDIR"\n' + pattern_line + '\n' + block + '\necho GUARD_OK\n'
    )
    return subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
        env={**os.environ, 'WORKDIR': str(workdir)},
    )


# Shaped like real configure output, including the trailing spaces its
# echo lines leave behind.
_CONFIGURE_STYLE_MK_CMD = (
    '# Generated automatically\n'
    'FORTCOMP        = gfortran -Ofast -march=native -fallow-argument-mismatch -c \n'
    'LINK            = gfortran -Ofast -march=native -fallow-argument-mismatch \n'
    'LIBLINK         = ar rvu \n'
    'OMPARG          = -fopenmp \n'
)


def test_flag_rewrite_makes_configure_output_portable(tmp_path):
    """The shipped rewrite turns configure's default flags portable.

    Runs the rewrite block against a fixture make/Mk_cmd shaped like real
    configure output, with the non-portable flags on both the compile and
    link lines. Both occurrences must become '-O2 -fno-fast-math', no
    CPU-specific flag may remain anywhere, and OMPARG must pass through
    untouched (OpenMP is deliberately kept by the install path).
    """
    (tmp_path / 'make').mkdir()
    mk = tmp_path / 'make' / 'Mk_cmd'
    mk.write_text(_CONFIGURE_STYLE_MK_CMD)

    res = _run_flag_rewrite(tmp_path)

    assert res.returncode == 0, res.stderr
    assert 'REWRITE_OK' in res.stdout
    rewritten = mk.read_text()
    # Both FORTCOMP and LINK must be rewritten: a count of 1 would mean
    # the link line kept the host-specific flags.
    assert rewritten.count('-O2 -fno-fast-math') == 2
    assert '-march=native' not in rewritten
    assert '-Ofast' not in rewritten
    assert 'OMPARG          = -fopenmp' in rewritten


def test_flag_rewrite_stops_on_changed_configure_defaults(tmp_path):
    """A changed configure flag string stops the build with a clear error.

    If a SOCRATES release ships different optimisation defaults (here the
    aarch64 spelling '-mcpu=native', which the rewrite pattern does not
    match), the block must exit nonzero before ./build_code runs, name
    the file to update in the error, and leave the fixture unmodified
    rather than letting a host-specific binary compile.
    """
    (tmp_path / 'make').mkdir()
    mk = tmp_path / 'make' / 'Mk_cmd'
    changed = _CONFIGURE_STYLE_MK_CMD.replace('-Ofast -march=native', '-O3 -mcpu=native')
    mk.write_text(changed)

    res = _run_flag_rewrite(tmp_path)

    assert res.returncode == 1
    assert 'REWRITE_OK' not in res.stdout
    assert 'get_socrates.sh' in res.stderr  # error names the file to update
    assert mk.read_text() == changed  # fixture left untouched


def test_flag_rewrite_reports_missing_mk_cmd_as_configure_failure(tmp_path):
    """A missing make/Mk_cmd is diagnosed as a configure failure.

    When configure exits zero without writing make/Mk_cmd (or the file
    moves in a future SOCRATES release), the block must exit nonzero with
    an error pointing at the configure step, and must not emit the
    changed-defaults message, which would send the reader to the wrong
    fix (the rewrite pattern instead of the configure output).
    """
    # Deliberately no make/ directory: the fixture models a configure run
    # that produced no output file.
    res = _run_flag_rewrite(tmp_path)

    assert res.returncode == 1
    assert 'REWRITE_OK' not in res.stdout
    assert 'was not generated' in res.stderr
    # Discrimination: the changed-defaults diagnosis must not fire for a
    # missing file; the two failure modes need different fixes.
    assert 'defaults have' not in res.stderr


def test_post_build_guard_rejects_cpu_specific_template_flags(tmp_path):
    """A per-host template carrying CPU-specific flags fails the build.

    build_code can replace bin/Mk_cmd with a committed per-host template
    on recognised cluster hostnames. The shipped post-build check must
    accept the portable rewrite output and reject the known CPU-specific
    spellings of the compilers the committed templates use (gfortran
    '-march=native', ifx '-xHost' and '-ax<arch>').
    """
    (tmp_path / 'bin').mkdir()
    binmk = tmp_path / 'bin' / 'Mk_cmd'

    # Portable flags pass through.
    binmk.write_text('FORTCOMP = gfortran -O2 -fno-fast-math -c \n')
    res = _run_post_build_guard(tmp_path)
    assert res.returncode == 0, res.stderr
    assert 'GUARD_OK' in res.stdout

    # Host-specific template flags fail, across compiler vocabularies.
    for flags in ('-Ofast -march=native', '-O3 -xHost', '-O2 -axCORE-AVX512'):
        binmk.write_text(f'FORTCOMP = ifx {flags} -c \n')
        res = _run_post_build_guard(tmp_path)
        assert res.returncode == 1, f'{flags} not rejected'
        assert 'non-portable' in res.stderr
        assert 'GUARD_OK' not in res.stdout
