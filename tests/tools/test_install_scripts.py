"""
Unit tests for the installation shell scripts and the repository and CI
configuration invariants they depend on.

Reusable shell logic replicated inline from ``tools/get_petsc.sh`` and
``tools/get_spider.sh``:
- ERR trap: exit-code and step-name capture
- Platform detection: PETSC_ARCH assignment
- Homebrew prefix fallback: architecture-aware default
- Workpath argument handling: ``$1`` override vs default
- PETSc library detection: versioned ``.so``, ``.dylib``, missing

``tools/_get_common.sh``, the helper library every ``get_*.sh`` sources, is
sourced by the cases below rather than copied into them, so they run against
the shipped text:
- ``portable_realpath()``: cross-platform path resolution, including a
  destination that does not exist yet
- the checkout root and tools directory the library derives from its own
  location
- ``get_parse_args``: the ``--force`` switch and the optional install path
- ``guard_dirty_checkout``: the refusal to delete local work, and the
  pathspec exclusion ``get_socrates.sh`` passes it
- ``github_use_ssh`` and ``github_ssh_url``: the SSH probe and URL rewrite
- ``resolve_module_pin``: a missing pin stops the install
- the bootstrap in each script, which stops when the library is absent, and
  the invariant that no script carries a private copy of a helper

Blocks lifted out of the shipped scripts at run time, so that rewording a
script re-runs its cases against the new text:
- ``tools/get_socrates.sh``: the install-path resolution, the portable-flag
  rewrite, its post-build flag check, and the conditional AGNI-wrapper
  rebuild note

Whole scripts run against stubbed ``git`` and ``ssh``, to pin the clone
destination and transport each one resolves.

Also pins invariants that live in checked-in configuration and documentation
rather than in shell, each of which fails silently when its counterpart moves:
- ``pyproject.toml`` module pins and optional-dependency extras, which the
  scripts resolve through ``tools/_module_pins.py``
- the extras the ``setup-proteus`` composite action installs, against the
  extra keys pyproject declares
- the installation docs' guidance on editable installs, against those pins
- CI config leaving USER at the runner default, which the action's macOS
  ``brew install`` step requires

Each shell test runs an isolated bash snippet via ``subprocess.run()``, with
no network access and no real builds; the configuration tests read the
checked-in files directly.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

TOOLS_DIR = Path(__file__).resolve().parents[2] / 'tools'
COMMON_LIB = TOOLS_DIR / '_get_common.sh'


# ---------------------------------------------------------------------------
# Helpers: run bash against the shipped helper library
# ---------------------------------------------------------------------------
def _with_common(body: str, strict: bool = False) -> str:
    """Return a bash snippet that sources the shipped helper library.

    The library is sourced, not copied, so a change to it re-runs through
    every case below. ``strict`` mirrors the ``get_*`` scripts that enable
    ``set -euo pipefail``: the helpers must behave the same either way.
    """
    prelude = 'set -euo pipefail\n' if strict else ''
    return f'{prelude}source "{COMMON_LIB}"\n{body}'


def _run_bash(snippet: str, *argv: str, **kwargs) -> subprocess.CompletedProcess:
    """Run ``snippet`` with ``argv`` as its positional parameters."""
    return subprocess.run(
        ['bash', '-c', snippet, 'get_test.sh', *argv],
        capture_output=True,
        text=True,
        **kwargs,
    )


def _extract_script_block(script: str, start_marker: str, end_marker: str) -> str:
    """Return the shipped lines of ``tools/<script>`` between two markers.

    Reading the block from the script under test, rather than copying it
    here, keeps the cases below running against the text that ships.
    """
    lines = (TOOLS_DIR / script).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if start_marker in ln)
    end = next(i for i, ln in enumerate(lines) if end_marker in ln and i > start)
    return '\n'.join(lines[start:end])


def _write_stub(directory: Path, name: str, body: str) -> Path:
    """Write an executable stub command into ``directory``."""
    stub = directory / name
    stub.write_text(body)
    stub.chmod(0o755)
    return stub


# ---------------------------------------------------------------------------
# portable_realpath tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_portable_realpath_resolves_relative(tmp_path):
    """Resolves ``./subdir`` to an absolute path using ``portable_realpath``."""
    subdir = tmp_path / 'subdir'
    subdir.mkdir()

    snippet = _with_common(f'cd "{tmp_path}"\nportable_realpath ./subdir')
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
def test_portable_realpath_resolves_missing_path(tmp_path):
    """Resolves a destination that does not exist yet, as an install path.

    BSD realpath (macOS) rejects a missing leaf and GNU realpath a missing
    parent, so both a missing leaf and a missing nested path are covered.
    An empty result here is the regression: the caller feeds the value
    straight to ``git clone``, which then fails on an empty work-tree name.
    """
    leaf = tmp_path / 'not-created-yet'
    nested = tmp_path / 'no' / 'such' / 'tree'
    snippet = _with_common(f'portable_realpath "{leaf}"\nportable_realpath "{nested}"')
    result = subprocess.run(['bash', '-c', snippet], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    resolved = result.stdout.split()
    assert len(resolved) == 2, result.stdout
    # tmp_path is under a symlinked /var on macOS, so compare against the
    # resolved parent rather than against the literal input path.
    real_root = os.path.realpath(tmp_path)
    assert resolved[0] == os.path.join(real_root, 'not-created-yet')
    assert resolved[1] == os.path.join(real_root, 'no', 'such', 'tree')


@pytest.mark.unit
def test_portable_realpath_resolves_parent_refs(tmp_path):
    """Resolves ``../`` components to a canonical absolute path."""
    child = tmp_path / 'a' / 'b'
    child.mkdir(parents=True)

    # a/b/../ should resolve to a/
    snippet = _with_common(f'portable_realpath "{child}/.."')
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

    snippet = _with_common(f'portable_realpath "{link}"')
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

    snippet = _with_common(f'portable_realpath "{target}"')
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
# One shipped copy of the shared helpers (tools/_get_common.sh)
# ---------------------------------------------------------------------------

# A private copy of a shared helper is recognised by the text that used to
# be duplicated: the function header, the dirty test, and the SSH probe.
PRIVATE_HELPER_SIGNATURES = (
    ('portable_realpath() {', 'portable_realpath'),
    ('status --porcelain --untracked-files=no', 'the dirty-checkout guard'),
    ('-T git@github.com', 'the GitHub SSH probe'),
)

# Names a script can only use because the library defines or sets them.
LIBRARY_NAMES = (
    'portable_realpath',
    'get_parse_args',
    'guard_dirty_checkout',
    'github_use_ssh',
    'github_ssh_url',
    'resolve_module_pin',
    'proteus_root',
    'proteus_tools_dir',
)


def _get_scripts() -> list[Path]:
    """Return the shipped ``tools/get_*.sh`` scripts."""
    return sorted(TOOLS_DIR.glob('get_*.sh'))


@pytest.mark.unit
def test_no_get_script_carries_a_private_helper_copy():
    """Each shared helper is defined once, in the library.

    Nine scripts once carried their own ``portable_realpath``, and a fix to
    one copy left the other eight broken. A script that grows a private
    copy again is also outside the reach of the cases in this file, which
    read the library rather than the scripts.
    """
    scripts = _get_scripts()
    # Guard the guard: a mis-rooted glob would make the scan vacuous.
    assert len(scripts) >= 10, [s.name for s in scripts]

    offenders: dict[str, list[str]] = {}
    for script in scripts:
        for line in script.read_text().splitlines():
            stripped = line.strip()
            # Prose and user-facing messages may still name the commands.
            if stripped.startswith('#') or 'echo ' in stripped:
                continue
            for needle, label in PRIVATE_HELPER_SIGNATURES:
                if needle in stripped:
                    offenders.setdefault(script.name, []).append(label)
    assert offenders == {}, offenders


@pytest.mark.unit
def test_every_helper_user_sources_the_library():
    """A script calling a shared helper also sources the file defining it.

    Without the source line the helper name is unset, and under a shell
    without ``set -u`` the call is a silent no-op rather than an error.
    """
    users, missing = [], []
    for script in _get_scripts():
        text = script.read_text()
        if not any(name in text for name in LIBRARY_NAMES):
            continue
        users.append(script.name)
        if 'source "$_get_common"' not in text:
            missing.append(script.name)

    assert missing == [], missing
    # Guard the guard: a renamed helper would leave nothing to check.
    assert len(users) >= 10, users


@pytest.mark.unit
def test_a_missing_helper_library_stops_the_script(tmp_path):
    """A script whose library is absent stops before touching a checkout.

    Continuing would leave every helper call undefined and reach git with
    an empty destination, so the bootstrap names the missing file, exits
    non-zero, and runs no git command.
    """
    lone_tools = tmp_path / 'tools'
    lone_tools.mkdir()
    shutil.copy2(TOOLS_DIR / 'get_aragog.sh', lone_tools / 'get_aragog.sh')
    log = tmp_path / 'calls.log'
    log.write_text('')
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    _write_stub(stubs, 'git', '#!/bin/bash\necho "git $*" >> "$STUB_LOG"\nexit 0\n')

    res = subprocess.run(
        ['bash', str(lone_tools / 'get_aragog.sh')],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            'PATH': f'{stubs}:{os.environ["PATH"]}',
            'STUB_LOG': str(log),
        },
    )

    assert res.returncode == 1, res.stdout
    assert '_get_common.sh' in res.stderr
    assert log.read_text() == ''


@pytest.mark.unit
def test_library_derives_the_root_from_its_own_location(tmp_path):
    """The checkout root follows the library, not the caller's directory.

    ``proteus install-all`` runs the scripts through data.py without
    setting a working directory, so a root taken from the CWD would
    install into whichever tree the caller happened to be sitting in.
    """
    fake_tools = tmp_path / 'FakeProteus' / 'tools'
    fake_tools.mkdir(parents=True)
    shutil.copy2(COMMON_LIB, fake_tools / '_get_common.sh')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()

    snippet = (
        f'source "{fake_tools}/_get_common.sh"\necho "$proteus_root"\necho "$proteus_tools_dir"'
    )
    res = subprocess.run(['bash', '-c', snippet], cwd=elsewhere, capture_output=True, text=True)

    assert res.returncode == 0, res.stderr
    root, tools = res.stdout.split()
    assert root == os.path.realpath(tmp_path / 'FakeProteus')
    assert tools == os.path.realpath(fake_tools)
    # The caller's directory must not leak into either value.
    assert os.path.realpath(elsewhere) not in (root, tools)


@pytest.mark.unit
@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
@pytest.mark.parametrize(
    ('argv', 'expected'),
    [
        ([], 'false|'),
        (['--force'], 'true|'),
        (['some/path'], 'false|some/path'),
        (['--force', 'some/path'], 'true|some/path'),
        (['some/path', '--force'], 'true|some/path'),
        (['first/path', 'second/path'], 'false|first/path'),
    ],
    ids=[
        'no arguments',
        'force only',
        'path only',
        'force before path',
        'force after path',
        'second path ignored',
    ],
)
def test_get_parse_args_splits_force_from_the_install_path(argv, expected, strict):
    """The switch and the optional path are read in either order.

    Six scripts pass only ``--force`` and two also accept a destination, so
    a path must not be mistaken for the switch or the other way round. The
    empty argument list is the edge case: under ``set -u`` an unguarded
    ``"$@"`` would abort the script before the first helper ran.
    """
    body = 'get_parse_args "$@"\nprintf \'%s|%s\\n\' "$get_force" "$get_install_path"\n'
    res = _run_bash(_with_common(body, strict=strict), *argv)

    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ('probe_rc', 'expected'),
    [(1, 'true'), (255, 'false'), (0, 'false')],
    ids=['key accepted', 'key refused', 'shell opened'],
)
@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_github_use_ssh_reads_the_probe_exit_code(tmp_path, probe_rc, expected, strict):
    """Only exit code 1 means GitHub accepted the key.

    GitHub refuses the interactive shell that ``ssh -T`` asks for, so a
    working key exits 1 and a rejected one exits 255. Exit code 0 means
    something other than GitHub answered, which must not select SSH. The
    stub also prints a banner on stdout, as the real probe may: that text
    must not reach the answer, which the caller reads by command
    substitution.
    """
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    _write_stub(
        stubs,
        'ssh',
        f'#!/bin/bash\necho "Hi there! You have successfully authenticated."\nexit {probe_rc}\n',
    )

    body = 'answer=$(github_use_ssh)\necho "[$answer]"\n'
    env = {**os.environ, 'PATH': f'{stubs}:{os.environ["PATH"]}'}
    env.pop('GIT_SSH_COMMAND', None)
    res = _run_bash(_with_common(body, strict=strict), env=env)

    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == f'[{expected}]'
    # The banner is still shown to the user, on stderr.
    assert 'successfully authenticated' in res.stderr


@pytest.mark.unit
def test_github_use_ssh_honours_git_ssh_command(tmp_path):
    """GIT_SSH_COMMAND replaces the probe command, options included.

    CI sets it to a non-interactive, fast-failing ssh invocation; a probe
    that ignored it would hang on a host-key prompt instead.
    """
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    log = tmp_path / 'calls.log'
    _write_stub(
        stubs,
        'stub-ssh',
        '#!/bin/bash\necho "stub-ssh $*" > "$STUB_LOG"\nexit 1\n',
    )

    body = 'answer=$(github_use_ssh)\necho "[$answer]"\n'
    res = _run_bash(
        _with_common(body),
        env={
            **os.environ,
            'PATH': f'{stubs}:{os.environ["PATH"]}',
            'STUB_LOG': str(log),
            'GIT_SSH_COMMAND': 'stub-ssh -o BatchMode=yes',
        },
    )

    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == '[true]'
    assert log.read_text().strip() == 'stub-ssh -o BatchMode=yes -T git@github.com'


@pytest.mark.unit
@pytest.mark.parametrize(
    ('url', 'expected'),
    [
        (
            'https://github.com/FormingWorlds/PROTEUS.git',
            'git@github.com:FormingWorlds/PROTEUS.git',
        ),
        (
            'git@github.com:FormingWorlds/PROTEUS.git',
            'git@github.com:FormingWorlds/PROTEUS.git',
        ),
        ('https://gitlab.com/group/repo.git', 'https://gitlab.com/group/repo.git'),
    ],
    ids=['https github', 'already ssh', 'other host'],
)
def test_github_ssh_url_rewrites_only_github_https(url, expected):
    """Only an https github.com URL is rewritten for SSH transport.

    A pin that already names SSH, or a repository hosted elsewhere, must
    pass through unchanged; rewriting either one produces a URL git cannot
    resolve.
    """
    res = _run_bash(_with_common(f'github_ssh_url "{url}"'))

    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == expected


@pytest.mark.unit
def test_resolve_module_pin_reads_the_pin_and_stops_when_it_is_missing(tmp_path):
    """A pinned module resolves; an unpinned one stops the install.

    The url and ref go straight into ``git clone`` and ``git checkout``, so
    an empty pin would otherwise clone the default branch of nothing.
    """
    fake_tools = tmp_path / 'tools'
    fake_tools.mkdir()
    shutil.copy2(COMMON_LIB, fake_tools / '_get_common.sh')
    (fake_tools / '_module_pins.py').write_text('')
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    _write_stub(
        stubs,
        'python',
        '#!/bin/bash\n'
        'if [ "$2" = "pinned" ]; then\n'
        '    if [ "$3" = "url" ]; then\n'
        '        echo "https://github.com/FormingWorlds/Thing.git"\n'
        '    else\n'
        '        echo "v1.2.3"\n'
        '    fi\n'
        'fi\n'
        'exit 0\n',
    )
    env = {**os.environ, 'PATH': f'{stubs}:{os.environ["PATH"]}'}
    prelude = f'source "{fake_tools}/_get_common.sh"\n'

    good = subprocess.run(
        [
            'bash',
            '-c',
            prelude + 'resolve_module_pin pinned\necho "$module_url @ $module_ref"\n',
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert good.returncode == 0, good.stderr
    assert good.stdout.strip() == 'https://github.com/FormingWorlds/Thing.git @ v1.2.3'

    missing = subprocess.run(
        ['bash', '-c', prelude + 'resolve_module_pin absent\necho REACHED_CLONE\n'],
        capture_output=True,
        text=True,
        env=env,
    )
    assert missing.returncode == 1, missing.stdout
    assert 'could not resolve absent url/ref' in missing.stderr
    assert 'REACHED_CLONE' not in missing.stdout


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


def _petsc_workpath_block() -> str:
    """Return the shipped install-path resolution of tools/get_petsc.sh."""
    block = _extract_script_block(
        'get_petsc.sh', '# Default: ./petsc/ relative', 'export PETSC_DIR'
    )
    return _with_common(block + '\necho "$workpath"\n')


@pytest.mark.unit
def test_petsc_workpath_uses_first_argument(tmp_path):
    """A destination passed as ``$1`` becomes the resolved install path.

    data.py:get_petsc() passes the full path, so the argument must win over
    the ``./petsc/`` default and must come back absolute: PETSC_DIR is
    exported from it and read by SPIDER's Makefile from another directory.
    """
    custom = tmp_path / 'custom_petsc'

    result = _run_bash(_petsc_workpath_block(), str(custom))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(custom)
    assert custom.is_dir()


@pytest.mark.unit
def test_petsc_workpath_defaults_to_petsc(tmp_path):
    """With no argument the install path is ``./petsc/`` under the CWD.

    This is the documented no-argument install, and the empty argument list
    is the edge case the ``$1`` test must not mistake for a path.
    """
    result = _run_bash(_petsc_workpath_block(), cwd=str(tmp_path))

    assert result.returncode == 0, result.stderr
    resolved = result.stdout.strip()
    assert resolved.endswith('/petsc')
    assert resolved == os.path.join(os.path.realpath(tmp_path), 'petsc')
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

    installation_text = (repo_root / 'docs' / 'How-to' / 'installation.md').read_text(
        encoding='utf-8'
    )
    manual_text = (repo_root / 'docs' / 'How-to' / 'manual_installation.md').read_text(
        encoding='utf-8'
    )

    combined = installation_text + '\n' + manual_text

    # Match `git clone <url>/<aragog|Zalmoxis>` or `pip install -e <aragog|Zalmoxis>`.
    forbidden = re.compile(
        r'(git\s+clone[^\n]*?(aragog|Zalmoxis))'
        r'|(pip\s+install\s+-e\s+(aragog|Zalmoxis))',
        re.IGNORECASE,
    )
    matches = forbidden.findall(combined)
    assert not matches, (
        f'installation.md re-introduced editable-install of Aragog/Zalmoxis: {matches!r}'
    )
    # Discrimination: installation.md must still cover the PyPI install
    # path. An empty file would also have zero matches above but would
    # silently delete the install instructions; pin the canonical PyPI
    # package name as evidence the file still documents the supported
    # path.
    assert 'fwl-aragog' in combined or 'pip install -e ".[develop]"' in combined


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
    assert any(r.startswith('atmodeller>=1.0.2') for r in extras.get('atmodeller', [])), (
        f'atmodeller extra must keep its pin, got {extras.get("atmodeller")!r}'
    )
    assert any(r.startswith('fwl-vulcan>=26.04.22') for r in extras.get('vulcan', [])), (
        f'vulcan extra must keep its pin, got {extras.get("vulcan")!r}'
    )
    # The default Aragog interior solver runs on JAX and its modules are
    # equinox Modules, so the jax/equinox stack must stay MANDATORY, not be
    # gated behind the optional atmodeller extra. The pinned equinox build
    # targets a jax API that changed in 0.10, which is why jax/jaxlib are held
    # <0.10; the pin is the combination the Aragog numerics are validated
    # against. Lifting any of these would break a standard run.
    assert 'jax<0.10' in deps and 'jaxlib<0.10' in deps, (
        'jax/jaxlib must stay pinned <0.10 for the default Aragog jax solver'
    )
    assert 'equinox==0.13.8' in deps, (
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
def test_socrates_cache_key_covers_every_file_the_build_reads():
    """The SOCRATES cache key hashes the install script and its library.

    The key busts on a change to the build so a reworded install does not
    restore a tree the old one produced. get_socrates.sh sources the
    shared helpers, so a key naming only the script would restore a stale
    tree after a change to the SSH probe, the guard, or the pin lookup,
    and the staleness would be invisible: the restored build simply wins
    and no step reports a mismatch.
    """
    action = (TOOLS_DIR.parent / '.github/actions/setup-proteus/action.yml').read_text(
        encoding='utf-8'
    )
    key_lines = [ln for ln in action.splitlines() if 'key: socrates-' in ln]
    assert len(key_lines) == 1, key_lines

    hashed = re.search(r'hashFiles\(([^)]*)\)', key_lines[0])
    assert hashed, key_lines[0]
    assert 'tools/get_socrates.sh' in hashed.group(1)
    assert 'tools/_get_common.sh' in hashed.group(1)


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
# Dirty-checkout guard (tools/_get_common.sh, called by every get_*.sh)
# ---------------------------------------------------------------------------


def _run_guard(tmp_path, *args: str, pathspec: str = '') -> subprocess.CompletedProcess:
    """Run the shipped guard against the ``aragog`` checkout in ``tmp_path``.

    ``pathspec`` appends a git pathspec, as ``get_socrates.sh`` does for its
    regenerable build config.
    """
    body = (
        'get_parse_args "$@"\n'
        f'guard_dirty_checkout "$GUARD_ROOT/aragog" get_aragog.sh {pathspec}\n'
        'echo GUARD_PASSED\n'
    )
    return subprocess.run(
        ['bash', '-c', _with_common(body), 'guard', *args],
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


def test_guard_excludes_only_the_pathspec_the_caller_names(tmp_path):
    """An excluded path does not block the refresh, but its neighbours do.

    get_socrates.sh excludes make/Mk_cmd because configure rewrites it on
    every build, so treating it as user work would block every refresh.
    Without the exclusion the same modification must still block, which is
    what makes the exclusion the reason the refresh proceeds rather than a
    guard that stopped looking.
    """
    upstream = tmp_path / 'upstream'
    (upstream / 'make').mkdir(parents=True)
    _git(upstream, 'init', '-q')
    (upstream / 'make' / 'Mk_cmd').write_text('FORTCOMP = gfortran\n')
    (upstream / 'src.f90').write_text('end\n')
    _git(upstream, 'add', '.')
    _git(upstream, 'commit', '-q', '-m', 'c1')

    workdir = tmp_path / 'aragog'
    _git(tmp_path, 'clone', '-q', str(upstream), str(workdir))
    exclude = "-- ':(exclude)make/Mk_cmd'"

    # Regenerated build config only: the exclusion lets the refresh run.
    (workdir / 'make' / 'Mk_cmd').write_text('FORTCOMP = gfortran -Ofast\n')
    res = _run_guard(tmp_path, pathspec=exclude)
    assert res.returncode == 0, res.stderr
    assert 'GUARD_PASSED' in res.stdout

    # The same state without the exclusion blocks: the guard still looks.
    res = _run_guard(tmp_path)
    assert res.returncode == 1
    assert 'uncommitted changes' in res.stderr

    # A modification outside the excluded path blocks either way.
    (workdir / 'src.f90').write_text('stop\n')
    res = _run_guard(tmp_path, pathspec=exclude)
    assert res.returncode == 1
    assert 'GUARD_PASSED' not in res.stdout


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


def _extract_pattern_line() -> str:
    """Return the shipped nonportable_flags definition line."""
    tools_dir = Path(__file__).resolve().parents[2] / 'tools'
    script = (tools_dir / 'get_socrates.sh').read_text().splitlines()
    return next(ln for ln in script if ln.startswith('nonportable_flags='))


def _portable_env(workdir, portable: bool) -> dict:
    """Build the env for a snippet run, toggling the portable-flags switch."""
    env = {**os.environ, 'WORKDIR': str(workdir)}
    env.pop('SOCRATES_PORTABLE_FLAGS', None)
    if portable:
        env['SOCRATES_PORTABLE_FLAGS'] = '1'
    return env


def _run_flag_rewrite(workdir, portable: bool = True) -> subprocess.CompletedProcess:
    """Run the shipped flag rewrite and its guards against a fixture tree."""
    block = _extract_socrates_block('A missing make/Mk_cmd', './build_code')
    snippet = 'set -euo pipefail\ncd "$WORKDIR"\n' + block + '\necho REWRITE_OK\n'
    return subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
        env=_portable_env(workdir, portable),
    )


def _run_post_build_guard(workdir, portable: bool = True) -> subprocess.CompletedProcess:
    """Run the shipped post-build flag check against a fixture bin/Mk_cmd."""
    pattern_line = _extract_pattern_line()
    block = _extract_socrates_block('Verify the flags that reached', '# Environment')
    snippet = (
        'set -euo pipefail\ncd "$WORKDIR"\n' + pattern_line + '\n' + block + '\necho GUARD_OK\n'
    )
    return subprocess.run(
        ['bash', '-c', snippet],
        capture_output=True,
        text=True,
        env=_portable_env(workdir, portable),
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
    """A missing make/Mk_cmd is diagnosed as a configure failure in any mode.

    When configure exits zero without writing make/Mk_cmd (or the file
    moves in a future SOCRATES release), the block must exit nonzero with
    an error pointing at the configure step, and must not emit the
    changed-defaults message, which would send the reader to the wrong
    fix (the rewrite pattern instead of the configure output). The check
    guards the default (non-portable) build path too, so it runs with the
    portable switch off.
    """
    # Deliberately no make/ directory: the fixture models a configure run
    # that produced no output file.
    res = _run_flag_rewrite(tmp_path, portable=False)

    assert res.returncode == 1
    assert 'REWRITE_OK' not in res.stdout
    assert 'was not generated' in res.stderr
    # Discrimination: the changed-defaults diagnosis must not fire for a
    # missing file; the two failure modes need different fixes.
    assert 'defaults have' not in res.stderr


def test_flag_rewrite_skipped_without_portable_switch(tmp_path):
    """The default build keeps the upstream performance flags untouched.

    Without SOCRATES_PORTABLE_FLAGS=1 the rewrite must not run: the
    configure-style fixture passes through byte-identical, keeping
    '-Ofast -march=native' and never introducing the portable spelling.
    This pins the gate itself: local installs keep the upstream flags
    and only opted-in builds (CI) are rewritten.
    """
    (tmp_path / 'make').mkdir()
    mk = tmp_path / 'make' / 'Mk_cmd'
    mk.write_text(_CONFIGURE_STYLE_MK_CMD)

    res = _run_flag_rewrite(tmp_path, portable=False)

    assert res.returncode == 0, res.stderr
    assert 'REWRITE_OK' in res.stdout
    # Discrimination: the file is byte-identical, the native flags are
    # still present, and the portable spelling was never written.
    assert mk.read_text() == _CONFIGURE_STYLE_MK_CMD
    assert '-Ofast -march=native' in mk.read_text()
    assert '-O2 -fno-fast-math' not in mk.read_text()


def test_post_build_guard_inactive_without_portable_switch(tmp_path):
    """The post-build flag check only applies to opted-in portable builds.

    Without SOCRATES_PORTABLE_FLAGS=1 a bin/Mk_cmd carrying CPU-specific
    flags is the expected outcome of a default build and must pass; the
    same fixture fails when the switch is on (covered by the rejection
    test above), so this pins the gate rather than the pattern.
    """
    (tmp_path / 'bin').mkdir()
    binmk = tmp_path / 'bin' / 'Mk_cmd'
    binmk.write_text('FORTCOMP = gfortran -Ofast -march=native -c \n')

    res = _run_post_build_guard(tmp_path, portable=False)

    assert res.returncode == 0, res.stderr
    assert 'GUARD_OK' in res.stdout
    # Discrimination: with the switch on, this exact fixture is rejected.
    res_on = _run_post_build_guard(tmp_path, portable=True)
    assert res_on.returncode == 1
    assert 'GUARD_OK' not in res_on.stdout


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


# ---------------------------------------------------------------------------
# Install-path resolution (tools/get_socrates.sh)
# ---------------------------------------------------------------------------


def _run_install_path_block(root, argv, stub_resolver: str = '') -> subprocess.CompletedProcess:
    """Run the shipped argument split and install-path resolution.

    ``stub_resolver`` redefines ``portable_realpath`` after the library is
    sourced, so the empty-resolution branch can be reached without an
    unusable host.
    """
    block = _extract_script_block(
        'get_socrates.sh', '# Separate the --force flag', '# make/Mk_cmd is excluded'
    )
    body = f'set -u\nroot="{root}"\n' + stub_resolver + block + '\necho "SOCPATH=$socpath"\n'
    return _run_bash(_with_common(body), *argv)


@pytest.mark.unit
def test_install_path_resolves_before_the_checkout_exists(tmp_path):
    """A custom destination that does not exist yet resolves to a real path.

    git clone creates the destination, including its parents, so the script
    must not require the directory up front. An empty resolution is the
    failure this pins: git clone then reports an empty work-tree name.
    """
    dest = tmp_path / 'no' / 'socrates-here'
    res = _run_install_path_block(tmp_path / 'root', [str(dest)])

    assert res.returncode == 0, res.stderr
    socpath = res.stdout.strip().removeprefix('SOCPATH=')
    assert socpath, res.stderr
    assert socpath == os.path.join(os.path.realpath(tmp_path), 'no', 'socrates-here')


@pytest.mark.unit
def test_install_path_default_and_force_flag(tmp_path):
    """--force alone keeps the default destination; a path with it is honoured.

    The flag and the optional positional share one argument list, so the
    ordering of the two must not shift which value lands in socpath.
    """
    root = tmp_path / 'root'
    dest = tmp_path / 'elsewhere'

    default = _run_install_path_block(root, ['--force'])
    assert default.returncode == 0, default.stderr
    assert default.stdout.strip() == f'SOCPATH={root}/socrates'

    override = _run_install_path_block(root, ['--force', str(dest)])
    assert override.returncode == 0, override.stderr
    assert override.stdout.strip() == f'SOCPATH={os.path.realpath(tmp_path)}/elsewhere'


@pytest.mark.unit
def test_install_path_rejects_unresolvable_path(tmp_path):
    """An empty resolution stops the script instead of reaching git clone.

    set -euo pipefail is enabled further down the script, so a resolver that
    fails here would otherwise leave socpath empty and continue.
    """
    stub = 'portable_realpath() {\n    return 1\n}\n'
    res = _run_install_path_block(tmp_path / 'root', ['some/path'], stub_resolver=stub)

    assert res.returncode == 1, res.stdout
    assert 'could not resolve install path' in res.stderr
    assert 'some/path' in res.stderr
    assert 'SOCPATH=' not in res.stdout


# ---------------------------------------------------------------------------
# Clone destination and transport, whole scripts against stubbed git and ssh
# ---------------------------------------------------------------------------

# The commands the scripts probe for before they clone. Stubbing them keeps
# the run independent of what the host has installed.
_INSTALL_STUB_COMMANDS = (
    'pip',
    'make',
    'julia',
    'clang',
    'gfortran',
    'nc-config',
    'nf-config',
    'mpicc',
)

# script, arguments, expected destination, expected clone URL, ssh probe exit.
# The pinned URLs are the ones tools/_module_pins.py resolves today for the
# scripts that read a pin; moving a pin updates this table with it.
CLONE_CASES = (
    ('get_aragog.sh', (), '{root}/aragog/', 'git@github.com:FormingWorlds/aragog.git', 1),
    ('get_zalmoxis.sh', (), '{root}/Zalmoxis/', 'git@github.com:FormingWorlds/Zalmoxis.git', 1),
    ('get_boreas.sh', (), '{root}/BOREAS/', 'git@github.com:ExoInteriors/BOREAS.git', 1),
    ('get_lavatmos.sh', (), '{root}/LavAtmos/', 'git@github.com:FormingWorlds/LavAtmos', 1),
    (
        'get_thermoenginelite.sh',
        (),
        '{root}/ThermoEngineLite/',
        'git@github.com:FormingWorlds/ThermoEngineLite',
        1,
    ),
    ('get_vulcan.sh', (), '{root}/VULCAN/', 'git@github.com:FormingWorlds/VULCAN.git', 1),
    ('get_socrates.sh', (), '{root}/socrates', 'git@github.com:FormingWorlds/SOCRATES.git', 1),
    (
        'get_socrates.sh',
        ('{root}/custom/socrates',),
        '{root}/custom/socrates',
        'git@github.com:FormingWorlds/SOCRATES.git',
        1,
    ),
    ('get_agni.sh', (), '{root}/AGNI', 'https://github.com/nichollsh/AGNI.git', 1),
    (
        'get_spider.sh',
        ('{root}/custom/SPIDER',),
        '{root}/custom/SPIDER',
        'https://github.com/FormingWorlds/SPIDER.git',
        1,
    ),
    # SSH refused: the same destination, over https.
    ('get_aragog.sh', (), '{root}/aragog/', 'https://github.com/FormingWorlds/aragog.git', 255),
    ('get_boreas.sh', (), '{root}/BOREAS/', 'https://github.com/ExoInteriors/BOREAS.git', 255),
)

CLONE_IDS = (
    'aragog over ssh',
    'zalmoxis over ssh',
    'boreas over ssh',
    'lavatmos over ssh',
    'thermoenginelite over ssh',
    'vulcan over ssh',
    'socrates default path',
    'socrates custom path',
    'agni pinned url',
    'spider custom path',
    'aragog without ssh',
    'boreas without ssh',
)


def _fake_checkout(tmp_path) -> Path:
    """Build a throwaway PROTEUS root holding the shipped install scripts.

    Only the scripts, the helper library, the pin reader and pyproject.toml
    are copied, so a run cannot reach into the real checkout.
    """
    root = tmp_path / 'PROTEUS'
    tools = root / 'tools'
    tools.mkdir(parents=True)
    for src in [*TOOLS_DIR.glob('get_*.sh'), COMMON_LIB, TOOLS_DIR / '_module_pins.py']:
        shutil.copy2(src, tools / src.name)
    shutil.copy2(TOOLS_DIR.parent / 'pyproject.toml', root / 'pyproject.toml')

    # SPIDER validates a built PETSc before it clones.
    conf = root / 'petsc' / 'lib' / 'petsc' / 'conf'
    conf.mkdir(parents=True)
    (conf / 'variables').write_text('')
    (conf / 'rules').write_text('')
    for arch in ('arch-linux-c-opt', 'arch-darwin-c-opt'):
        lib = root / 'petsc' / arch / 'lib'
        lib.mkdir(parents=True)
        (lib / 'libpetsc.dylib').write_text('')
        (lib / 'libpetsc.so').write_text('')
    return root


@pytest.mark.unit
@pytest.mark.parametrize(
    ('script', 'argv', 'dest', 'url', 'probe_rc'), CLONE_CASES, ids=CLONE_IDS
)
def test_script_clones_the_expected_destination(tmp_path, script, argv, dest, url, probe_rc):
    """Each script resolves its destination and clones the pinned URL.

    Run whole against a stubbed git and ssh, in a throwaway checkout: the
    destination the script computes and the transport it selects are the
    two values a refactor of the shared helpers can silently change, and an
    empty or CWD-relative destination is what a broken path resolution
    produces. The steps after the clone need real trees, so the exit status
    is not the contract here; the recorded git call is.
    """
    root = _fake_checkout(tmp_path)
    real_root = os.path.realpath(root)
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    log = tmp_path / 'calls.log'
    log.write_text('')

    _write_stub(
        stubs,
        'git',
        '#!/bin/bash\n'
        'echo "git $*" >> "$STUB_LOG"\n'
        'if [ "$1" = "clone" ]; then\n'
        '    mkdir -p "${@: -1}/.git"\n'
        'fi\n'
        'exit 0\n',
    )
    _write_stub(stubs, 'ssh', f'#!/bin/bash\nexit {probe_rc}\n')
    for command in _INSTALL_STUB_COMMANDS:
        _write_stub(stubs, command, '#!/bin/bash\nexit 0\n')

    env = {
        **os.environ,
        'PATH': f'{stubs}:{os.environ["PATH"]}',
        'STUB_LOG': str(log),
        # Both are read before the scripts would sleep on a warning.
        'RAD_DIR': '',
        'LAVA_DIR': '',
    }
    env.pop('GIT_SSH_COMMAND', None)

    subprocess.run(
        ['bash', str(root / 'tools' / script), *(a.format(root=root) for a in argv)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    clones = [ln for ln in log.read_text().splitlines() if ln.startswith('git clone ')]
    assert len(clones) == 1, log.read_text()
    assert clones[0] == f'git clone {url} {dest.format(root=real_root)}'
    # Nothing may resolve outside the throwaway checkout.
    assert clones[0].split()[-1].startswith(real_root)


# ---------------------------------------------------------------------------
# Post-rebuild AGNI-wrapper note (tools/get_socrates.sh)
# ---------------------------------------------------------------------------


def _run_agni_rebuild_note(workdir, has_agni: bool) -> subprocess.CompletedProcess:
    """Run the shipped AGNI-rebuild note block against a fixture root.

    Extracts the conditional note tools/get_socrates.sh prints after a rebuild
    and runs it with ``root`` pointed at a fixture tree that does or does not
    contain an AGNI checkout.
    """
    if has_agni:
        (workdir / 'AGNI').mkdir()
    block = _extract_socrates_block('if [ -d "$root/AGNI" ]', 'exit 0')
    # The note references both the resolved root and the SOCRATES path it set
    # earlier in the script; supply both so the printed command is complete.
    snippet = f'root="{workdir}"\nsocpath="{workdir}/socrates"\n{block}'
    return subprocess.run(['bash', '-c', snippet], capture_output=True, text=True)


def test_socrates_rebuild_warns_to_regenerate_agni_wrappers(tmp_path):
    """After a rebuild, the script tells the user to regenerate AGNI's wrappers.

    Rebuilding SOCRATES re-clones its tree and deletes the Julia wrappers AGNI
    generates under socrates/julia, so the script must point the user at the
    AGNI rebuild that restores them. The note is conditional on an AGNI checkout
    being present.
    """
    # AGNI present: the note fires and names the exact rebuild command, anchored
    # at the resolved root so it is copy-pasteable from any directory.
    with_agni = tmp_path / 'with_agni'
    with_agni.mkdir()
    res = _run_agni_rebuild_note(with_agni, has_agni=True)
    assert res.returncode == 0, res.stderr
    assert 'get_agni.sh" 0' in res.stdout
    assert str(with_agni / 'tools' / 'get_agni.sh') in res.stdout
    # The command is self-contained: it sets RAD_DIR to the SOCRATES path the
    # script resolved, so a user who has not yet exported RAD_DIR can paste it
    # as-is. Discrimination: a bare `bash .../get_agni.sh 0` would omit this and
    # fail when RAD_DIR is unset.
    assert f'RAD_DIR="{with_agni}/socrates"' in res.stdout
    # The script path is quoted so a root containing spaces survives the paste.
    assert f'bash "{with_agni}/tools/get_agni.sh"' in res.stdout

    # Guard against the referenced rebuild script being renamed out from under
    # the note: the command it points at must name a script that exists.
    tools_dir = Path(__file__).resolve().parents[2] / 'tools'
    assert (tools_dir / 'get_agni.sh').is_file()

    # Discrimination: with no AGNI checkout there is nothing to rebuild, so the
    # note stays silent rather than point the user at a missing tree.
    without_agni = tmp_path / 'without_agni'
    without_agni.mkdir()
    res_absent = _run_agni_rebuild_note(without_agni, has_agni=False)
    assert res_absent.returncode == 0, res_absent.stderr
    assert 'get_agni.sh' not in res_absent.stdout


# ---------------------------------------------------------------------------
# CI must not pin USER (breaks the macOS Homebrew install)
# ---------------------------------------------------------------------------
# A USER pin reaches the job environment as a block-mapping key, as a flow-
# mapping entry, or as a dynamic export out of a run step. All three are
# forbidden, so the scan matches all three rather than only the block form.
_USER_PINS = (
    re.compile(r'^\s*["\']?USER["\']?\s*:'),
    re.compile(r'[{,]\s*["\']?USER["\']?\s*:'),
    re.compile(r'\bUSER\s*=.*GITHUB_ENV'),
)


def _pins_user(line: str) -> bool:
    """Return True when a CI config line pins USER into the job environment."""
    # Drop trailing comments so prose naming USER cannot trip the scan.
    text = line.split('#', 1)[0]
    return any(pattern.search(text) for pattern in _USER_PINS)


@pytest.mark.unit
def test_ci_config_never_pins_user():
    """CI config must leave USER at the runner default.

    Several jobs share the macOS leg of the ``setup-proteus`` composite action,
    which runs ``brew install``. Its preinstall check resolves the active
    account by name, so a USER naming an account that does not exist on the
    runner aborts the setup step before a single package is installed and every
    macOS job fails before reaching its tests. Pinning even a real account name
    is fragile, because it breaks whenever the runner account is renamed or the
    jobs move back into a container. The invariant is pinned here, for every
    workflow at once, rather than by a comment in whichever one was edited last.
    """
    repo_root = Path(__file__).resolve().parents[2]
    workflow_dir = repo_root / '.github/workflows'
    workflows = sorted(workflow_dir.glob('*.yml')) + sorted(workflow_dir.glob('*.yaml'))
    # Every composite action is scanned, not just the one that installs the
    # macOS packages today: any action calling brew is a place where a USER
    # pin would break the setup.
    actions = sorted(repo_root.glob('.github/actions/*/action.yml')) + sorted(
        repo_root.glob('.github/actions/*/action.yaml')
    )

    # Guard the guard: an empty or mis-rooted glob would make the scan below
    # pass by finding nothing to check.
    assert workflows, f'no workflow files resolved under {workflow_dir}'
    assert actions, f'no composite actions resolved under {repo_root}/.github/actions'
    targets = workflows + actions

    # Discrimination: the matcher must fire on every shape that reaches the job
    # environment and stay quiet on unrelated keys, prose, and same-suffix
    # variables, so a broken pattern cannot make this test pass vacuously.
    assert _pins_user('      USER: "ci-runner"')
    assert _pins_user("      'USER': ci-runner")
    assert _pins_user('    env: {USER: ci-runner}')
    assert _pins_user('        echo "USER=ci-runner" >> "$GITHUB_ENV"')
    assert not _pins_user('  COVERAGE_FILE: .coverage.unit.macos-latest')
    assert not _pins_user('  # Leave USER at the runner default')
    assert not _pins_user('        echo "FWL_USER=someone" >> "$GITHUB_ENV"')

    offenders = []
    for path in targets:
        for lineno, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if _pins_user(raw):
                offenders.append(f'{path.relative_to(repo_root)}:{lineno}: {raw.strip()}')

    assert not offenders, (
        'CI config must leave USER at the runner default. A USER naming an '
        'account that does not exist aborts the macOS brew install shared by '
        'several jobs, and pinning a real account name breaks on the next '
        f'runner rename. Remove these: {offenders}'
    )
