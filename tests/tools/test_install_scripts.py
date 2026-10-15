"""
Unit tests for the installation shell scripts and the repository and CI
configuration invariants they depend on.

Reusable shell logic replicated inline from ``tools/get_petsc.sh`` and
``tools/get_spider.sh``:
- ERR trap: exit-code and step-name capture
- Platform detection: PETSC_ARCH assignment
- macOS linker flags: no library path that could hide the downloaded SUNDIALS
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
- ``swap_in_checkout``: the old checkout leaves its path only once the new
  one is there, and comes back when the new one cannot move in or when a
  signal arrives between the two moves; without ``--force`` a folder that
  appeared during the clone stays
- ``report_clone_state``: the one line for a module folder at its release
  tag, or the stay line and the advice lines for any other folder
- ``github_use_ssh`` and ``github_ssh_url``: the SSH probe and URL rewrite
- ``resolve_module_pin``: a missing pin stops the install
- the bootstrap in each script, which stops when the library is absent, and
  the invariant that no script carries a private copy of a helper

Blocks lifted out of the shipped scripts at run time, so that rewording a
script re-runs its cases against the new text:
- ``tools/get_petsc.sh``: the install-path resolution, ``$1`` against the
  ``./petsc/`` default
- ``tools/get_socrates.sh``: the install-path resolution, the portable-flag
  rewrite, its post-build flag check, and the conditional AGNI-wrapper
  rebuild note

Whole scripts run against stubbed ``git`` and ``ssh``, to pin the clone
destination and transport each one resolves, and ``get_python_module.sh``,
``get_vulcan.sh``, ``get_morrigan.sh``, ``get_aragog.sh`` and
``get_zalmoxis.sh`` against local bare remotes, to pin the release tag each
new clone lands on and that a failed replacement keeps a clean clone; that
without ``--force`` only read-only git calls reach an existing module folder
of any kind, which, with pip and make stubbed, stays byte-equal and is
installed as it is, while an empty folder counts as absent; that the printed
hand commands work; and that the last two pair a branch only for a new clone
or ``--force``.

Also pins invariants that live in checked-in configuration and documentation
rather than in shell, each of which fails silently when its counterpart moves:
- ``pyproject.toml`` module pins and optional-dependency extras, which the
  scripts resolve through ``tools/_module_pins.py``
- the extras the ``setup-proteus`` composite action installs, against the
  extra keys pyproject declares
- the SOCRATES cache key, against every file its build step reads
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
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest
from helpers import (
    git,
    git_call_is_read_only,
    install_git_guard,
    tree_snapshot,
)

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
        preexec_fn=_default_signals,
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

# The shell the library exports, read from the library rather than listed
# here: a helper renamed in one place and not the other would otherwise
# drop out of both scans below without either failing.
LIBRARY_VARIABLES = ('proteus_root', 'proteus_tools_dir')


def _library_function_names() -> list[str]:
    """Return the function names ``tools/_get_common.sh`` defines."""
    names = re.findall(
        r'^(?:function\s+)?([a-z_][a-z0-9_]*)\s*\(\s*\)\s*\{',
        COMMON_LIB.read_text(),
        re.MULTILINE,
    )
    return [name for name in names if not name.startswith('_')]


# A private copy is a function definition of a library name, or the shell
# the library replaced: the dirty test and the SSH probe as commands.
# Matching the command form, not the text, keeps a mention inside a
# user-facing message (get_spider.sh names the probe in its
# troubleshooting output) from reading as a copy.
COPIED_SHELL_PATTERNS = (
    (r'git\s+(?:-C\s+\S+\s+)?status\s+--porcelain', 'the dirty-checkout guard'),
    (r'^(?:if\s+)?(?:\$\{GIT_SSH_COMMAND[^}]*\}|ssh)\s+-T\s+git@github\.com', 'the SSH probe'),
)

# The library derives the checkout root, so the only `dirname` a script needs
# is the one that finds the library. Anything else walking up from $0 or
# BASH_SOURCE is a second derivation, which is how a newly added script
# reintroduces the duplication (get_obliqua.sh arrived that way).
BOOTSTRAP_LINE = r'^source\s+"\$\(dirname\s+"\$\{BASH_SOURCE\[0\]\}"\)/_get_common\.sh"'


def _get_scripts() -> list[Path]:
    """Return the shipped ``tools/get_*.sh`` scripts."""
    return sorted(TOOLS_DIR.glob('get_*.sh'))


def _code_lines(script: Path) -> list[tuple[int, str]]:
    """Return ``(index, code)`` for each line, with comments stripped.

    Comments are dropped because a script's own prose names the helpers it
    calls, and a quoted ``#`` is not a comment.
    """
    stripped = []
    for index, line in enumerate(script.read_text().splitlines()):
        code = re.sub(r'(^|\s)#.*$', '', line) if line.count('"') % 2 == 0 else line
        stripped.append((index, code.strip()))
    return stripped


@pytest.mark.unit
def test_no_get_script_carries_a_private_helper_copy():
    """Each shared helper is defined once, in the library.

    Nine scripts once carried their own ``portable_realpath``, and a fix to
    one copy left the other eight broken. A private copy also shadows the
    library even in a script that sources it, so the definition, not the
    absence of the source line, is what has to be caught: the shadowed
    script would silently keep the old behaviour while every case in this
    file kept passing against the library.

    A second derivation of the checkout root counts too. It carries no
    helper name, so nothing else here would notice it, and it is the form
    in which a newly added script brings the duplication back.
    """
    scripts = _get_scripts()
    functions = _library_function_names()
    # Guard the guard: a mis-rooted glob or a failed name extraction would
    # make the scan vacuous.
    assert len(scripts) >= 10, [s.name for s in scripts]
    assert 'portable_realpath' in functions, functions
    assert len(functions) >= 6, functions

    # Accept the spellings bash accepts: `name() {`, `name ()  {`, and
    # `function name {`.
    definitions = [
        (
            rf'^(?:function\s+)?{re.escape(name)}\s*\(\s*\)\s*\{{|^function\s+{re.escape(name)}\s*\{{',
            f'a private {name}',
        )
        for name in functions
    ]

    offenders: dict[str, list[str]] = {}
    for script in scripts:
        for _, code in _code_lines(script):
            for pattern, label in (*definitions, *COPIED_SHELL_PATTERNS):
                if re.search(pattern, code):
                    offenders.setdefault(script.name, []).append(label)
            if (
                'dirname' in code
                and not re.match(BOOTSTRAP_LINE, code)
                and code != 'up=$(dirname "$up")'
            ):
                offenders.setdefault(script.name, []).append('a private root derivation')
    assert offenders == {}, offenders


@pytest.mark.unit
def test_every_helper_user_sources_the_library_before_calling_it():
    """A script calls a shared helper only after sourcing its definition.

    An unsourced or late-sourced call is worse than a missing one: without
    ``set -u`` the name is empty, so the call is a silent no-op. For
    ``guard_dirty_checkout`` that means the refusal to delete a checkout
    holding local work is simply absent, and the script deletes it.
    """
    names = (*_library_function_names(), *LIBRARY_VARIABLES)
    users, unsourced, late = [], [], []
    for script in _get_scripts():
        lines = _code_lines(script)
        uses = [i for i, code in lines if any(name in code for name in names)]
        if not uses:
            continue
        users.append(script.name)
        sourced = [i for i, code in lines if re.match(r'^source\s+.*_get_common\.sh', code)]
        if not sourced:
            unsourced.append(script.name)
        elif min(uses) < sourced[0]:
            late.append(script.name)

    assert unsourced == [], unsourced
    assert late == [], late
    # Guard the guard: a renamed helper would leave nothing to check.
    assert len(users) >= 10, users


@pytest.mark.unit
def test_a_missing_helper_library_stops_the_script(tmp_path):
    """A script whose library is absent stops before touching a checkout.

    bash names the missing file on its own, but half the scripts set no
    ``-e``, so without the ``|| exit 1`` the run carries on with every
    helper undefined and ``proteus_root`` empty: the work path becomes
    ``/aragog/``, which the script then deletes and clones into. Nothing
    may reach git.
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
def test_get_parse_args_leaves_the_callers_variables_alone():
    """Parsing the arguments does not clobber a caller's own variables.

    The scripts share a namespace with every helper they source, and
    ``arg`` is an ordinary name for a script to use. A helper that leaked
    its loop variable would overwrite the caller's value with the last
    argument, silently and only when arguments are passed.
    """
    body = (
        'arg=keepme\n'
        'get_parse_args --force some/path\n'
        'printf \'%s|%s|%s\\n\' "$arg" "$get_force" "$get_install_path"\n'
    )
    res = _run_bash(_with_common(body, strict=True))

    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == 'keepme|true|some/path'


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
            'GIT_SSH_COMMAND': 'stub-ssh -o CustomOption=foo',
        },
    )

    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == '[true]'
    assert (
        log.read_text().strip()
        == 'stub-ssh -o CustomOption=foo -o BatchMode=yes -o ConnectTimeout=10 -T git@github.com'
    )


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


def _sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    import hashlib

    return hashlib.sha256(data).hexdigest()


@pytest.mark.unit
@pytest.mark.parametrize('strict', [False, True])
def test_fetch_verified_falls_back_to_the_next_source_with_the_pinned_hash(tmp_path, strict):
    """A failed download, an empty url and a wrong file each give way to the next source.

    The archive goes straight into ``unzip`` and a PETSc build, so only a file
    with the pinned SHA-256 may end the search.
    """
    good, bad = tmp_path / 'good.zip', tmp_path / 'bad.zip'
    good.write_bytes(b'petsc archive')
    bad.write_bytes(b'an error page')
    dest = tmp_path / 'petsc.zip'
    body = (
        f'fetch_verified {_sha256(b"petsc archive")} "{dest}" '
        f'"file://{tmp_path}/missing.zip" "" "file://{bad}" "file://{good}"\n'
        'echo REACHED_UNZIP\n'
    )

    result = _run_bash(_with_common(body, strict))

    assert result.returncode == 0, result.stderr
    assert dest.read_bytes() == b'petsc archive'
    assert 'REACHED_UNZIP' in result.stdout
    assert f'download from file://{tmp_path}/missing.zip failed' in result.stderr
    assert (
        f'file://{bad} served a file with SHA-256 {_sha256(b"an error page")}' in result.stderr
    )


@pytest.mark.unit
@pytest.mark.parametrize('strict', [False, True])
def test_fetch_verified_stops_when_no_source_serves_the_pinned_file(tmp_path, strict):
    """With no matching source it returns 1 and leaves no archive, so a get_* script's
    ERR trap prints its troubleshooting text and the install stops before unzip."""
    bad = tmp_path / 'bad.zip'
    bad.write_bytes(b'an error page')
    dest = tmp_path / 'petsc.zip'
    call = f'fetch_verified {"0" * 64} "{dest}" "file://{bad}" ""'

    returned = _run_bash(_with_common(f'{call} || echo "RETURNED $?"\n', strict))
    trapped = _run_bash(
        _with_common(f'set -e\ntrap "echo TRAPPED" ERR\n{call}\necho REACHED_UNZIP\n', strict)
    )

    assert 'RETURNED 1' in returned.stdout
    assert not dest.exists()
    assert f'no source served petsc.zip with SHA-256 {"0" * 64}' in returned.stderr
    assert 'download from  failed' not in returned.stderr
    assert trapped.returncode == 1
    assert 'TRAPPED' in trapped.stdout and 'REACHED_UNZIP' not in trapped.stdout


@pytest.mark.unit
def test_fetch_verified_refuses_an_empty_hash_or_no_hash_tool(tmp_path):
    """A pin that could not be read, or no shasum and no sha256sum, stops before any download."""
    good = tmp_path / 'good.zip'
    good.write_bytes(b'petsc archive')
    call = 'fetch_verified "{}" "' + f'{tmp_path}/p.zip" "file://{good}"\n'
    blank = _run_bash(_with_common(call.format('')))
    no_tool = subprocess.run(
        ['/bin/bash', '-c', _with_common(call.format(_sha256(b'petsc archive')))],
        capture_output=True,
        text=True,
        env={'PATH': str(tmp_path / 'empty')},
    )
    assert blank.returncode == 1 and 'no SHA-256 pin for p.zip' in blank.stderr
    assert no_tool.returncode == 1
    assert 'neither shasum nor sha256sum is installed' in no_tool.stderr
    for result in (blank, no_tool):
        assert 'Downloading' not in result.stdout


@pytest.mark.unit
def test_get_petsc_tries_zenodo_then_the_mirror_from_its_pins(tmp_path):
    """The shipped download step passes the pinned hash and sources, overridable by
    PETSC_URL and PETSC_MIRROR_URL, to fetch_verified in that order."""
    fake_tools = tmp_path / 'tools'
    fake_tools.mkdir()
    shutil.copy2(COMMON_LIB, fake_tools / '_get_common.sh')
    (fake_tools / '_module_pins.py').write_text('')
    archive = tmp_path / 'mirror.zip'
    archive.write_bytes(b'petsc archive')
    wrong = tmp_path / 'wrong.zip'
    wrong.write_bytes(b'an error page')
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    _write_stub(
        stubs,
        'python',
        f'#!/bin/bash\ncase "$3" in sha256) echo {_sha256(b"petsc archive")} ;; '
        f'url) echo "file://{wrong}" ;; mirror) echo "$PIN_MIRROR" ;; esac\n',
    )
    block = _extract_script_block(
        'get_petsc.sh', '# Read the archive pins', 'current_step="Decompressing'
    )
    work = tmp_path / 'work'
    snippet = (
        f'set -e\nsource "{fake_tools}/_get_common.sh"\nworkpath="{work}"\n{block}\necho DONE\n'
    )
    env = {**os.environ, 'PATH': f'{stubs}:{os.environ["PATH"]}', 'PIN_MIRROR': ''}

    def run(**extra):
        return subprocess.run(
            ['bash', '-c', snippet], capture_output=True, text=True, env={**env, **extra}
        )

    # Both pinned sources serve a wrong file: the install stops.
    pinned = run(PIN_MIRROR=f'file://{wrong}')
    assert pinned.returncode == 1
    assert pinned.stderr.count(f'file://{wrong} served a file') == 2

    # An empty pin stops with the cause instead of skipping that source.
    empty_mirror = run()
    assert empty_mirror.returncode == 1
    assert 'cannot read the PETSc url, mirror and sha256' in empty_mirror.stderr

    # Zenodo first, then the mirror.
    mirrored = run(PIN_MIRROR=f'file://{archive}')
    assert mirrored.returncode == 0, mirrored.stderr
    out = mirrored.stdout
    assert out.index(f'Downloading file://{wrong}') < out.index(f'Downloading file://{archive}')
    assert (work / 'petsc.zip').read_bytes() == b'petsc archive'
    assert 'DONE' in out

    # PETSC_URL replaces the pinned url, and PETSC_MIRROR_URL the pinned mirror.
    overridden = run(PETSC_URL=f'file://{archive}', PIN_MIRROR=f'file://{wrong}')
    assert overridden.returncode == 0, overridden.stderr
    assert f'Downloading file://{wrong}' not in overridden.stdout
    mirror_set = run(PETSC_MIRROR_URL=f'file://{archive}', PIN_MIRROR=f'file://{wrong}')
    assert mirror_set.returncode == 0, mirror_set.stderr

    # A pin that cannot be read stops before the previous installation is removed.
    (work / 'old_build').write_text('kept')
    _write_stub(stubs, 'python', '#!/bin/bash\nexit 1\n')
    unreadable = run()
    assert unreadable.returncode == 1
    assert 'needs python 3.11 or newer (tomllib)' in unreadable.stderr
    assert (work / 'old_build').read_text() == 'kept'
    assert 'Downloading' not in unreadable.stdout


@pytest.mark.unit
def test_pyproject_pins_the_petsc_archive_by_sha256():
    """PETSc is pinned to its Zenodo archive by a SHA-256, with the DataverseNL file as mirror."""
    pin = tomllib.loads((TOOLS_DIR.parent / 'pyproject.toml').read_text())['tool']['proteus'][
        'modules'
    ]['petsc']
    assert pin['url'] == 'https://zenodo.org/records/15805756/files/petsc.zip?download=1'
    assert pin['sha256'] == 'c5bdb75048b609627bac7fdc83042078a629f5de0c6508b50166a351d2aa045d'
    assert pin['mirror'] == 'https://dataverse.nl/api/access/datafile/683669'


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
current_step="Downloading PETSc archive"
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
    assert 'STEP=Downloading PETSc archive' in result.stdout


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
# macOS linker flags (get_petsc.sh)


# A stub configure: records its arguments, its environment and four named variables.
CONFIGURE_STUB = """#!/bin/bash
printf "%s\\n" "$@" > args
export -p > env
for v in LDFLAGS LIBRARY_PATH LIBS CPATH; do printf "%s=%s\\n" "$v" "${!v-unset}"; done > named
"""


def _run_petsc_macos(tmp_path, tools, path_extra=':/usr/bin:/bin'):
    """Run the shipped platform and configure steps of get_petsc.sh as macOS with stub ``tools``."""
    for name in tools:
        _write_stub(tmp_path, name, '#!/bin/bash\necho /sdk\n')
    _write_stub(tmp_path, 'configure', CONFIGURE_STUB)
    block = _extract_script_block(
        'get_petsc.sh', 'current_step="Determining platform-specific flags"', '# 6. Build PETSc'
    )
    return subprocess.run(
        ['/bin/bash', '-c', f'set -euo pipefail\nOSTYPE=darwin24\nworkpath=.\n{block}\n'],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={'PATH': f'{tmp_path}{path_extra}'},
    )


def _petsc_configure_args(tmp_path) -> list[str]:
    """Return the configure arguments, after checking no build variable names a library path."""
    named = (tmp_path / 'named').read_text().split()
    assert named == [f'{v}=unset' for v in ('LDFLAGS', 'LIBRARY_PATH', 'LIBS', 'CPATH')]
    assert 'homebrew/lib' not in (tmp_path / 'env').read_text()
    args = (tmp_path / 'args').read_text().splitlines()
    assert [a for a in args if '-L' in a] == []
    return args


@pytest.mark.unit
def test_petsc_configure_on_macos_gets_no_library_path(tmp_path):
    """On macOS with a system MPI, PETSc configure gets LDFLAGS=-Wl,-w and no -L in any
    argument or build variable, so no library directory comes ahead of the SUNDIALS 2.5 it
    downloads."""
    result = _run_petsc_macos(tmp_path, ('xcrun', 'mpicc', 'mpirun'))
    assert result.returncode == 0, result.stderr
    args = _petsc_configure_args(tmp_path)
    assert {'LDFLAGS=-Wl,-w', '--download-sundials2'} <= set(args)
    assert '--download-mpich' not in args


@pytest.mark.unit
def test_petsc_configure_on_macos_without_mpicc_downloads_mpich(tmp_path):
    """Without mpicc on PATH, PETSc downloads MPICH and still gets only LDFLAGS=-Wl,-w."""
    result = _run_petsc_macos(tmp_path, ('xcrun',), path_extra='')
    assert result.returncode == 0, result.stderr
    args = _petsc_configure_args(tmp_path)
    assert {'LDFLAGS=-Wl,-w', '--download-sundials2', '--download-mpich'} <= set(args)


@pytest.mark.unit
def test_petsc_on_macos_without_xcrun_stops_before_configure(tmp_path):
    """Without xcrun the script stops with its install hint and never runs configure."""
    result = _run_petsc_macos(tmp_path, ('mpicc', 'mpirun'), path_extra='')
    assert result.returncode == 1
    assert 'xcrun not found' in result.stdout and 'xcode-select --install' in result.stdout
    assert not (tmp_path / 'args').exists()


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
    """VULCAN and atmodeller are package extras that the installer sets up.

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
    # the check above; confirm a known-mandatory dependency is still present.
    assert any('fwl-calliope' in d for d in deps), 'mandatory dependency list is intact'

    extras = data['project']['optional-dependencies']
    # Membership, not exact-list: an extra may gain a second requirement
    # later (e.g. a transitive pin) without this guard going stale.
    assert any(r.startswith('atmodeller>=1.0.2') for r in extras.get('atmodeller', [])), (
        f'atmodeller extra must keep its pin, got {extras.get("atmodeller")!r}'
    )
    assert any(r.startswith('fwl-vulcan>=26.10.14') for r in extras.get('vulcan', [])), (
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
@pytest.mark.parametrize(
    'step, build_cmd',
    [
        ('Build SOCRATES (cache miss)', 'bash tools/get_socrates.sh --force'),
        ('Build AGNI (Pkg.instantiate)', 'julia --project=. deps/build.jl nodata'),
    ],
    ids=['socrates step', 'agni step'],
)
@pytest.mark.parametrize(
    'ld_mode', [0o755, 0o644, None], ids=['clt ld', 'clt ld not executable', 'no clt ld']
)
@pytest.mark.parametrize(
    'preset',
    [None, '/Applications/Xcode.app/Contents/Developer'],
    ids=['no preset', 'xcode preset'],
)
def test_socrates_links_use_ld_from_the_clt_folder(tmp_path, step, build_cmd, ld_mode, preset):
    """The steps that link SOCRATES export DEVELOPER_DIR=<CLT> when the CLT has an ld.

    SOCRATES links with -L to the Command Line Tools SDK, and an Xcode ld
    older than that SDK cannot read its libSystem.tbd. The test runs the
    step script up to its build command, with the CLT path moved to a
    temporary folder and julia stubbed, and reads the variable in a child
    process at the point where the build would start.
    """
    clt = '/Library/Developer/CommandLineTools'
    action = (TOOLS_DIR.parent / '.github/actions/setup-proteus/action.yml').read_text(
        encoding='utf-8'
    )
    body = action.split(f'- name: {step}\n', 1)[1].split('\n    - name: ', 1)[0]
    lines = [ln.strip() for ln in body.splitlines()]
    assert lines.count(build_cmd) == 1, f'{step}: expected one {build_cmd!r} line'
    run = lines[lines.index('run: |') + 1 : lines.index(build_cmd)]

    fake = tmp_path / 'CLT'
    (fake / 'usr/bin').mkdir(parents=True)
    if ld_mode is not None:
        (fake / 'usr/bin/ld').write_text('#!/bin/sh\n')
        (fake / 'usr/bin/ld').chmod(ld_mode)
    (tmp_path / 'stub').mkdir()
    (tmp_path / 'stub/julia').write_text('#!/bin/sh\n')
    (tmp_path / 'stub/julia').chmod(0o755)
    (tmp_path / 'AGNI').mkdir()
    script = '\n'.join(run).replace(clt, str(fake)) + '\nprintenv DEVELOPER_DIR || echo unset\n'
    env = {'PATH': f'{tmp_path}/stub:{os.environ["PATH"]}', 'GITHUB_WORKSPACE': str(tmp_path)}
    if preset:
        env['DEVELOPER_DIR'] = preset
    out = subprocess.check_output(['bash', '-c', script], cwd=tmp_path, env=env, text=True)
    assert out.strip() == (str(fake) if ld_mode == 0o755 else preset or 'unset')


@pytest.mark.unit
def test_developer_dir_is_set_only_in_the_socrates_link_steps():
    """DEVELOPER_DIR occurs only as the export inside the two SOCRATES link steps.

    A GITHUB_ENV write, a step or job env entry, or a third step would move
    later steps (pip, Julia, the test tiers) to the Command Line Tools.
    """
    action = (TOOLS_DIR.parent / '.github/actions/setup-proteus/action.yml').read_text(
        encoding='utf-8'
    )
    head, *steps = action.split('\n    - name: ')
    found = [
        (step.split('\n', 1)[0], line.strip())
        for step in steps
        for line in step.splitlines()
        if 'DEVELOPER_DIR' in line
    ]
    export = 'export DEVELOPER_DIR=/Library/Developer/CommandLineTools'
    assert 'DEVELOPER_DIR' not in head
    assert found == [
        ('Build SOCRATES (cache miss)', export),
        ('Build AGNI (Pkg.instantiate)', export),
    ]


@pytest.mark.unit
def test_ci_setup_installs_every_declared_extra():
    """The CI setup action must install extras whose keys exist in pyproject.

    The CI composite action installs PROTEUS with a literal extras list, e.g.
    ``pip install -e ".[develop,vulcan,atmodeller]"``. pip treats an unknown
    extra as a warning and still exits 0, so a typo or a renamed extra would
    silently stop installing the package extras, and their tests would skip
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
    # The two optional physics modules must be installed by CI so their tests
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


def _run_guard(
    tmp_path,
    *args: str,
    pathspec: str = '',
    strict: bool = False,
    checkout: str = 'aragog',
    script_name: str = 'get_aragog.sh',
) -> subprocess.CompletedProcess:
    """Run the shipped guard against the ``checkout`` directory in ``tmp_path``.

    ``pathspec`` appends a git pathspec, as ``get_socrates.sh`` does for its
    regenerable build config. ``strict`` selects the shell the callers
    split over: get_boreas.sh and get_vulcan.sh enable
    ``set -euo pipefail``, the other six do not.
    """
    body = (
        'get_parse_args "$@"\n'
        f'guard_dirty_checkout "$GUARD_ROOT/{checkout}" {script_name} {pathspec}\n'
        'echo GUARD_PASSED\n'
    )
    return subprocess.run(
        ['bash', '-c', _with_common(body, strict=strict), 'guard', *args],
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


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_blocks_dirty_and_unpushed_checkouts(tmp_path, strict):
    """Tracked modifications and local-only commits block the refresh.

    A modified tracked file must exit 1 with the recovery command in the
    message; a repo whose commits exist on no remote (covers both the
    remote-less and the never-pushed case) must also block. Untracked
    files alone must NOT block: build artifacts and egg-info dirs are
    routine in refreshed checkouts. Both shells are exercised: the guard
    is shared by scripts that enable ``set -euo pipefail`` and scripts
    that do not.
    """
    workdir = tmp_path / 'aragog'
    workdir.mkdir()
    _git(workdir, 'init', '-q')
    (workdir / 'tracked.py').write_text('x = 1\n')
    _git(workdir, 'add', 'tracked.py')
    _git(workdir, 'commit', '-q', '-m', 'c1')

    # Local-only commit (no remotes at all): blocked.
    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 1
    assert '--force' in res.stderr  # recovery command is named
    assert 'GUARD_PASSED' not in res.stdout

    # Same state plus a dirty tracked file: still blocked.
    (workdir / 'tracked.py').write_text('x = 2\n')
    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 1
    assert 'uncommitted changes' in res.stderr


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_passes_clean_remote_backed_checkout(tmp_path, strict):
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

    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 0
    assert 'GUARD_PASSED' in res.stdout

    # Local commit on the detached HEAD: reachable from HEAD, on no
    # remote. This is the state a tag-pinned checkout enters when a
    # developer commits without branching; it must block.
    (workdir / 'f.py').write_text('a = 2\n')
    _git(workdir, 'add', 'f.py')
    _git(workdir, 'commit', '-q', '-m', 'local work')
    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 1
    assert 'not on a remote' in res.stderr

    # --force bypasses deliberately.
    res = _run_guard(tmp_path, '--force', strict=strict)
    assert res.returncode == 0
    assert 'GUARD_PASSED' in res.stdout


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_refreshes_a_checkout_killed_before_its_first_commit(tmp_path, strict):
    """A clone killed before any commit is refreshed rather than kept.

    ``git log HEAD`` exits 128 while HEAD is unborn, which is the state a
    ctrl-C during the first clone leaves behind. HEAD not resolving means
    there are no commits, so nothing can be unpushed and there is nothing
    to protect. install.sh and ``proteus install-all`` pass no ``--force``,
    so a guard that stopped here would leave the install wedged with no way
    through it short of running the script by hand.
    """
    workdir = tmp_path / 'aragog'
    workdir.mkdir()
    _git(workdir, 'init', '-q')

    res = _run_guard(tmp_path, strict=strict)

    assert res.returncode == 0, res.stderr
    assert 'GUARD_PASSED' in res.stdout
    assert 'Refusing to delete' not in res.stderr


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_keeps_staged_work_on_an_unborn_head(tmp_path, strict):
    """Staged work blocks the refresh even with no commit to hang it on.

    This is the same unborn HEAD as the case above, so the two together fix
    which half of the guard decides: the status probe, not the log probe. A
    regression that read an unresolvable HEAD as "nothing here" would delete
    these files.
    """
    workdir = tmp_path / 'aragog'
    workdir.mkdir()
    _git(workdir, 'init', '-q')
    # Staged and never committed: work a refresh would destroy.
    (workdir / 'work.py').write_text('x = 1\n')
    _git(workdir, 'add', 'work.py')

    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 1, res.stdout
    assert 'uncommitted changes' in res.stderr
    assert '--force' in res.stderr  # the recovery command is named
    assert 'GUARD_PASSED' not in res.stdout

    # --force still discards deliberately, in either shell.
    forced = _run_guard(tmp_path, '--force', strict=strict)
    assert forced.returncode == 0, forced.stderr
    assert 'GUARD_PASSED' in forced.stdout


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_keeps_unpushed_work_on_a_branch_that_is_not_checked_out(tmp_path, strict):
    """Commits on any branch block, not only those reachable from HEAD.

    A checkout sitting on an orphan branch has an unborn HEAD and an empty
    index, so both the dirty test and a HEAD-reachable commit test see
    nothing, while another branch still holds commits that exist on no
    remote. Reading HEAD alone deleted that work; the probe reads every ref.
    """
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    _git(upstream, 'init', '-q')
    (upstream / 'f.py').write_text('a = 1\n')
    _git(upstream, 'add', 'f.py')
    _git(upstream, 'commit', '-q', '-m', 'c1')

    workdir = tmp_path / 'aragog'
    _git(tmp_path, 'clone', '-q', str(upstream), str(workdir))
    (workdir / 'f.py').write_text('a = 2\n')
    _git(workdir, 'add', 'f.py')
    _git(workdir, 'commit', '-q', '-m', 'unpushed work')
    # Orphan branch: HEAD no longer resolves and the index is emptied, so
    # the commit above is reachable only through the branch it sits on.
    _git(workdir, 'checkout', '-q', '--orphan', 'fresh')
    _git(workdir, 'rm', '-rq', '--cached', '.')

    res = _run_guard(tmp_path, strict=strict)

    assert res.returncode == 1, res.stdout
    assert 'not on a remote' in res.stderr
    assert 'GUARD_PASSED' not in res.stdout


@pytest.mark.unit
def test_every_refreshing_installer_guards_before_it_deletes():
    """An installer that deletes and re-clones a checkout guards it first.

    What protects a developer's unpushed work is not that the helper exists
    but that every installer wiping a checkout calls it, and calls it before
    the delete. An installer that forgets is how the protection goes missing,
    and the per-script cases above cannot see it: they exercise the helper,
    not its callers.
    """
    # A module script deletes only its temporary clone, which holds no work of the user.
    deletes = re.compile(r'^\s*rm -rf "\$(?!tmp_clone")[a-z_]+"')
    sources = {p.name: p.read_text().splitlines() for p in _get_scripts()}
    refreshing = {
        name: lines
        for name, lines in sources.items()
        if any(deletes.search(ln) for ln in lines) and any('git clone' in ln for ln in lines)
    }

    # Guard the guard: discovery has to find the installers that replace a
    # sibling checkout, and leave out the one that unpacks an archive and so
    # has no git history to lose.
    assert set(refreshing) == {
        'get_boreas.sh',
        'get_lavatmos.sh',
        'get_socrates.sh',
        'get_spider.sh',
        'get_thermoenginelite.sh',
    }, sorted(refreshing)
    assert 'get_petsc.sh' not in refreshing, 'the archive installer holds no git history'
    for name in (
        'get_python_module.sh',
        'get_aragog.sh',
        'get_zalmoxis.sh',
        'get_vulcan.sh',
        'get_morrigan.sh',
    ):
        lines = sources[name]
        assert any(ln.startswith('install_existing_clone ') for ln in lines), name
        assert not any(ln.startswith('guard_dirty_checkout') for ln in lines), name

    unguarded, late = [], []
    for name, lines in refreshing.items():
        guard = next(
            (i for i, ln in enumerate(lines) if ln.startswith('guard_dirty_checkout')), None
        )
        first_delete = next(i for i, ln in enumerate(lines) if deletes.search(ln))
        if guard is None:
            unguarded.append(name)
        elif guard > first_delete:
            late.append(name)

    assert unguarded == [], f'installers replace a checkout with no guard: {unguarded}'
    assert late == [], f'installers delete before they guard: {late}'


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_refreshes_a_clone_carrying_an_upstream_tag(tmp_path, strict):
    """An upstream tag outside every branch is not local work.

    A clone fetches the upstream's tags, and a repository that has rewritten
    history keeps the old commits alive under an archive tag: aragog carries
    ten such commits. Counting them as unpushed stops `get_aragog.sh` on a
    clean checkout, and install.sh runs it under `set -e` with no `--force`,
    so every re-install would stop there telling the user to push work that
    is not theirs.
    """
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    # -b: the initial branch is named by the host's init.defaultBranch, and
    # this fixture returns to it by name.
    _git(upstream, 'init', '-q', '-b', 'main')
    (upstream / 'f.py').write_text('a = 1\n')
    _git(upstream, 'add', 'f.py')
    _git(upstream, 'commit', '-q', '-m', 'c1')
    # Archived work: committed on a branch, tagged, and the branch deleted,
    # so the tag is the only ref holding it.
    _git(upstream, 'checkout', '-q', '-b', 'archived')
    (upstream / 'old.py').write_text('legacy = 1\n')
    _git(upstream, 'add', 'old.py')
    _git(upstream, 'commit', '-q', '-m', 'archived work')
    _git(upstream, 'tag', '-a', 'archive/pre-rewrite', '-m', 'archived')
    _git(upstream, 'checkout', '-q', 'main')
    _git(upstream, 'branch', '-qD', 'archived')

    _git(tmp_path, 'clone', '-q', str(upstream), str(tmp_path / 'aragog'))

    res = _run_guard(tmp_path, strict=strict)

    assert res.returncode == 0, res.stderr
    assert 'GUARD_PASSED' in res.stdout
    assert 'not on a remote' not in res.stderr


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_keeps_local_work_that_carries_a_tag(tmp_path, strict):
    """A tag on the user's own unpushed commit does not excuse deleting it.

    This is why the tags are not simply excluded from the comparison: doing
    so clears the upstream archive tag above, but it also clears a commit the
    user made and tagged, which is exactly the work the guard exists to keep.
    """
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    _git(upstream, 'init', '-q')
    (upstream / 'f.py').write_text('a = 1\n')
    _git(upstream, 'add', 'f.py')
    _git(upstream, 'commit', '-q', '-m', 'c1')

    workdir = tmp_path / 'aragog'
    _git(tmp_path, 'clone', '-q', str(upstream), str(workdir))
    (workdir / 'f.py').write_text('a = 2\n')
    _git(workdir, 'add', 'f.py')
    _git(workdir, 'commit', '-q', '-m', 'my work')
    _git(workdir, 'tag', '-a', 'my-wip', '-m', 'wip')

    res = _run_guard(tmp_path, strict=strict)

    assert res.returncode == 1, res.stdout
    assert 'not on a remote' in res.stderr
    assert 'GUARD_PASSED' not in res.stdout


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_keeps_a_checkout_git_cannot_report_on(tmp_path, strict):
    """A checkout git cannot inspect at all is kept, in either shell.

    An incomplete ``.git`` (a clone killed once it had started writing
    refs, or a truncated copy) makes even ``git status`` exit 128. Reading
    that through a pipe made the outcome depend on the shell: under
    ``set -o pipefail`` the script stopped with no message at all, and
    without it the guard read an empty result and deleted the checkout.
    Whether it holds local work is unknown, so it is kept and the reason
    named, identically in both shells.
    """
    workdir = tmp_path / 'aragog'
    (workdir / '.git').mkdir(parents=True)  # present but not a repository

    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 1, res.stdout
    assert 'could not report the state' in res.stderr
    assert '--force' in res.stderr
    assert 'GUARD_PASSED' not in res.stdout

    # --force still discards deliberately, in either shell.
    forced = _run_guard(tmp_path, '--force', strict=strict)
    assert forced.returncode == 0, forced.stderr
    assert 'GUARD_PASSED' in forced.stdout


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_excludes_only_the_pathspec_the_caller_names(tmp_path, strict):
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
    res = _run_guard(tmp_path, pathspec=exclude, strict=strict)
    assert res.returncode == 0, res.stderr
    assert 'GUARD_PASSED' in res.stdout

    # The same state without the exclusion blocks: the guard still looks.
    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 1
    assert 'uncommitted changes' in res.stderr

    # A modification outside the excluded path blocks either way.
    (workdir / 'src.f90').write_text('stop\n')
    res = _run_guard(tmp_path, pathspec=exclude, strict=strict)
    assert res.returncode == 1
    assert 'GUARD_PASSED' not in res.stdout


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_passes_a_missing_or_empty_folder(tmp_path, strict):
    """A missing path and an empty folder hold nothing to lose, so both pass without --force."""
    missing = _run_guard(tmp_path, strict=strict)
    assert missing.returncode == 0, missing.stderr
    assert 'GUARD_PASSED' in missing.stdout
    (tmp_path / 'aragog').mkdir()
    empty = _run_guard(tmp_path, strict=strict)
    assert empty.returncode == 0, empty.stderr
    assert 'GUARD_PASSED' in empty.stdout
    assert empty.stderr == ''


# ---------------------------------------------------------------------------
# swap_in_checkout (tools/_get_common.sh)
# ---------------------------------------------------------------------------


def _run_swap(new: Path, dest: Path, strict: bool) -> subprocess.CompletedProcess:
    """Run the shipped swap_in_checkout under --force and print its status as ``SWAP_STATUS <n>``."""
    body = (
        f'get_force=true\nrc=0\nswap_in_checkout "{new}" "{dest}" || rc=$?\n'
        'echo "SWAP_STATUS $rc"\n'
    )
    return _run_bash(_with_common(body, strict=strict), timeout=20)


def _checkout_dir(path: Path, marker: str) -> Path:
    """Create ``path`` holding ``<marker>.txt`` and return it."""
    path.mkdir(parents=True)
    (path / f'{marker}.txt').write_text(f'{marker}\n')
    return path


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_swap_puts_the_new_checkout_in_place_of_the_old(tmp_path, strict):
    """The new checkout takes the path, the old copy is gone and no sibling is left.

    A plain ``mv new dest`` with the old checkout still there nests the new
    one inside it as ``dest/new``, which the first assertion tells apart.
    """
    dest = _checkout_dir(tmp_path / 'MORS', 'old')
    new = _checkout_dir(tmp_path / 'MORS.tmp.1', 'new')
    res = _run_swap(new, dest, strict)
    assert 'SWAP_STATUS 0' in res.stdout, res.stderr
    assert sorted(p.name for p in dest.iterdir()) == ['new.txt']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['MORS']
    assert res.stderr == ''


@pytest.mark.skipif(os.geteuid() == 0, reason='root removes a read-only folder')
@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_swap_with_an_old_copy_that_will_not_go_warns_and_names_it(tmp_path, strict):
    """An old checkout with a read-only subfolder cannot be removed: the new checkout is still
    in place and complete, status 0, and the warning names the leftover copy."""
    dest = _checkout_dir(tmp_path / 'MORS', 'old')
    locked = dest / 'locked'
    locked.mkdir()
    (locked / 'build.o').write_text('')
    locked.chmod(0o555)
    new = _checkout_dir(tmp_path / 'MORS.tmp.1', 'new')
    (new / 'src').mkdir()
    (new / 'src' / 'mod.py').write_text('x = 1\n')
    try:
        res = _run_swap(new, dest, strict)
        leftovers = [p for p in tmp_path.iterdir() if p.name != 'MORS']
        assert 'SWAP_STATUS 0' in res.stdout, res.stderr
        assert sorted(p.name for p in dest.iterdir()) == ['new.txt', 'src']
        assert (dest / 'src' / 'mod.py').read_text() == 'x = 1\n'
        assert len(leftovers) == 1
        assert leftovers[0].name.startswith('MORS.old.')
        assert (leftovers[0] / 'locked' / 'build.o').exists()
        assert f'delete {leftovers[0]} by hand' in res.stderr
        assert res.stderr.startswith('WARNING: ')
    finally:
        for path in tmp_path.rglob('locked'):
            path.chmod(0o755)


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_swap_that_cannot_move_the_new_checkout_keeps_the_old_one(tmp_path, strict):
    """A new checkout that does not exist fails the second move: status 1, the old
    checkout is back at its path with its file, and no aside copy is left."""
    dest = _checkout_dir(tmp_path / 'MORS', 'old')
    res = _run_swap(tmp_path / 'MORS.tmp.missing', dest, strict)
    assert 'SWAP_STATUS 1' in res.stdout
    assert sorted(p.name for p in dest.iterdir()) == ['old.txt']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['MORS']
    assert f'{dest} is unchanged' in res.stderr


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
    (
        'get_aragog.sh',
        (),
        '{root}/aragog.tmp.<pid>',
        'git@github.com:FormingWorlds/aragog.git',
        1,
    ),
    (
        'get_zalmoxis.sh',
        (),
        '{root}/Zalmoxis.tmp.<pid>',
        'git@github.com:FormingWorlds/Zalmoxis.git',
        1,
    ),
    (
        'get_python_module.sh',
        ('CALLIOPE',),
        '{root}/CALLIOPE.tmp.<pid>',
        'git@github.com:FormingWorlds/CALLIOPE.git',
        1,
    ),
    ('get_boreas.sh', (), '{root}/BOREAS/', 'git@github.com:ExoInteriors/BOREAS.git', 1),
    ('get_lavatmos.sh', (), '{root}/LavAtmos/', 'git@github.com:FormingWorlds/LavAtmos', 1),
    (
        'get_thermoenginelite.sh',
        (),
        '{root}/ThermoEngineLite/',
        'git@github.com:FormingWorlds/ThermoEngineLite',
        1,
    ),
    (
        'get_vulcan.sh',
        (),
        '{root}/VULCAN.tmp.<pid>',
        'git@github.com:FormingWorlds/VULCAN.git',
        1,
    ),
    (
        'get_atmodeller.sh',
        (),
        '{root}/Atmodeller.tmp.<pid>',
        'git@github.com:FormingWorlds/atmodeller.git',
        1,
    ),
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
        'get_obliqua.sh',
        (),
        '{root}/Obliqua.tmp.<pid>',
        'https://github.com/FormingWorlds/Obliqua.git',
        1,
    ),
    (
        'get_spider.sh',
        ('{root}/custom/SPIDER',),
        '{root}/custom/SPIDER',
        'https://github.com/FormingWorlds/SPIDER.git',
        1,
    ),
    # SSH refused: the same destination, over https.
    (
        'get_aragog.sh',
        (),
        '{root}/aragog.tmp.<pid>',
        'https://github.com/FormingWorlds/aragog.git',
        255,
    ),
    ('get_boreas.sh', (), '{root}/BOREAS/', 'https://github.com/ExoInteriors/BOREAS.git', 255),
    (
        'get_python_module.sh',
        ('ZEPHYRUS',),
        '{root}/ZEPHYRUS.tmp.<pid>',
        'https://github.com/FormingWorlds/ZEPHYRUS.git',
        255,
    ),
)

CLONE_IDS = (
    'aragog over ssh',
    'zalmoxis over ssh',
    'calliope over ssh',
    'boreas over ssh',
    'lavatmos over ssh',
    'thermoenginelite over ssh',
    'vulcan over ssh',
    'atmodeller over ssh',
    'socrates default path',
    'socrates custom path',
    'agni pinned url',
    'obliqua pinned url',
    'spider custom path',
    'aragog without ssh',
    'boreas without ssh',
    'zephyrus without ssh',
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
    # The stub julia stands in for every Julia, so a bound Julia of the host must not lead PATH.
    env.pop('PYTHON_JULIAPKG_EXE', None)

    subprocess.run(
        ['bash', str(root / 'tools' / script), *(a.format(root=root) for a in argv)],
        # Deliberately not the checkout: a destination taken from the
        # working directory would otherwise land on the expected path by
        # coincidence, and the CWD-relative regression would be invisible.
        # get_spider.sh is the one script whose default destination is
        # CWD-relative by design, so its case passes a path.
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    clones = [ln for ln in log.read_text().splitlines() if ln.startswith('git clone ')]
    assert len(clones) == 1, log.read_text()
    # A script that clones into a temporary sibling names it after its process id.
    clone = re.sub(r'\.tmp\.\d+$', '.tmp.<pid>', clones[0])
    assert clone == f'git clone {url} {dest.format(root=real_root)}'
    # Nothing may resolve outside the throwaway checkout, and in particular
    # not into the directory the script was run from.
    assert clones[0].split()[-1].startswith(real_root)
    assert not (tmp_path / dest.format(root='').lstrip('/')).exists()


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


# CVODE install: tools/get_cvode.sh run against stub python, pip and conda, and
# the two install.sh blocks that turn a CVODE failure into an abort.
_CVODE_INSTALL = 'pip install scikits-odes-sundials>=3.0,<4'
_CVODE_REBUILD = (
    'pip install --force-reinstall --no-deps --no-cache-dir --no-binary '
    'scikits-odes-sundials scikits-odes-sundials>=3.0,<4'
)


def _run_get_cvode(tmp_path, *, importable: str, present: bool, after: str):
    """Run the shipped ``tools/get_cvode.sh`` with stubs; return (result, pip/conda log).

    ``importable`` is whether the CVODE import works before the script runs,
    ``present`` whether pip lists the package, ``after`` whether the import
    works once a ``pip install`` has run. No network and no build happen.
    """
    stubs = tmp_path / 'bin'
    state = tmp_path / 'state'
    stubs.mkdir()
    state.mkdir()
    (state / 'importable').write_text(importable)
    if present:
        (state / 'pkg').write_text('')
    scripts = {
        'python': '[ "$(cat "$STATE/importable")" = yes ]\n',
        'pip': (
            'echo "pip $*" >> "$STATE/log"\n'
            'case "$1" in\n'
            '  show) [ -e "$STATE/pkg" ] ;;\n'
            '  install) : > "$STATE/pkg"; echo "$AFTER" > "$STATE/importable" ;;\n'
            'esac\n'
        ),
        'conda': 'echo "conda $*" >> "$STATE/log"\n',
    }
    for name, body in scripts.items():
        path = stubs / name
        path.write_text('#!/usr/bin/env bash\n' + body)
        path.chmod(0o755)
    script = Path(__file__).resolve().parents[2] / 'tools' / 'get_cvode.sh'
    env = {
        **os.environ,
        'PATH': f'{stubs}{os.pathsep}{os.environ["PATH"]}',
        'CONDA_PREFIX': str(tmp_path / 'env'),
        'CONDA_EXE': str(stubs / 'conda'),
        'STATE': str(state),
        'AFTER': after,
    }
    res = subprocess.run(['bash', str(script)], capture_output=True, text=True, env=env)
    log = (state / 'log').read_text() if (state / 'log').exists() else ''
    return res, log


@pytest.mark.unit
def test_get_cvode_does_nothing_when_the_three_names_import(tmp_path):
    """A working CVODE is left alone: no conda call, no pip call."""
    res, log = _run_get_cvode(tmp_path, importable='yes', present=True, after='yes')

    assert res.returncode == 0, res.stderr
    assert 'nothing to do' in res.stdout
    assert log == ''


@pytest.mark.unit
def test_get_cvode_installs_a_missing_package_without_forcing(tmp_path):
    """No wrapper installed: a plain pip install, not a forced rebuild."""
    res, log = _run_get_cvode(tmp_path, importable='no', present=False, after='yes')

    assert res.returncode == 0, res.stderr
    assert _CVODE_INSTALL in log
    assert '--force-reinstall' not in log


@pytest.mark.unit
def test_get_cvode_rebuilds_an_installed_wrapper_that_does_not_import(tmp_path):
    """pip calls an installed wrapper satisfied, so a broken one needs a forced source rebuild."""
    res, log = _run_get_cvode(tmp_path, importable='no', present=True, after='yes')

    assert res.returncode == 0, res.stderr
    assert _CVODE_REBUILD in log
    assert 'rebuilding' in res.stdout


@pytest.mark.unit
def test_get_cvode_fails_when_the_wrapper_still_does_not_import(tmp_path):
    """The hard gate after the install exits 1 instead of reporting success."""
    res, _ = _run_get_cvode(tmp_path, importable='no', present=True, after='no')

    assert res.returncode == 1
    assert 'does not import' in res.stderr
    assert 'importable' not in res.stdout


def _install_sh_block(marker: str) -> str:
    """The shipped lines of ``install.sh`` from the one holding ``marker`` to the next blank line."""
    lines = (Path(__file__).resolve().parents[2] / 'install.sh').read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if marker in ln)
    end = next(i for i, ln in enumerate(lines) if not ln.strip() and i > start)
    return '\n'.join(lines[start:end]) + '\n'


def _run_install_block(tmp_path, marker: str, *, exit_code: int):
    """Run one ``install.sh`` block with ``warn`` and ``die`` printing WARN and DIE, and stubs."""
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'tools' / 'get_cvode.sh').write_text(f'exit {exit_code}\n')
    stubs = tmp_path / 'bin'
    stubs.mkdir()
    (stubs / 'python').write_text(f'#!/usr/bin/env bash\nexit {exit_code}\n')
    (stubs / 'python').chmod(0o755)
    snippet = (
        'info() { :; }\nwarn() { echo "WARN: $1" >&2; }\ndie() { echo "DIE: $1" >&2; exit 1; }\n'
        + _install_sh_block('run_logged() {')
        + _install_sh_block(marker)
        + 'echo REACHED\n'
    )
    env = {**os.environ, 'PATH': f'{stubs}{os.pathsep}{os.environ["PATH"]}'}
    return subprocess.run(
        ['bash', '-c', snippet], capture_output=True, text=True, cwd=tmp_path, env=env
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    'marker',
    ['Setting up the SUNDIALS CVODE solver', 'python -c "from scikits_odes_sundials.cvode'],
    ids=['install-step', 'after-proteus-install'],
)
def test_install_sh_warns_and_goes_on_when_cvode_fails(tmp_path, marker):
    """A failed CVODE install, or a CVODE that stops importing, warns and the installer goes on.

    Only Aragog on ``solver_method = "cvode"`` needs it, and that run stops at
    setup, so SPIDER, radau and bdf users can finish the installation.
    """
    res = _run_install_block(tmp_path, marker, exit_code=1)

    assert res.returncode == 0, res.stderr
    assert 'WARN: CVODE' in res.stderr
    assert 'DIE' not in res.stderr
    assert 'bash tools/get_cvode.sh' in res.stderr
    assert 'REACHED' in res.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    'marker',
    ['Setting up the SUNDIALS CVODE solver', 'python -c "from scikits_odes_sundials.cvode'],
    ids=['install-step', 'after-proteus-install'],
)
def test_install_sh_goes_on_when_cvode_works(tmp_path, marker):
    """The same blocks stay silent when CVODE installs and imports."""
    res = _run_install_block(tmp_path, marker, exit_code=0)

    assert res.returncode == 0, res.stderr
    assert 'WARN' not in res.stderr
    assert 'REACHED' in res.stdout


@pytest.mark.unit
def test_pyproject_keeps_morrigan_out_of_mandatory_dependencies():
    """Morrigan is an optional extra, pinned once by version.

    The giant-impact model is needed only by ``accretion.module =
    "morrigan"`` runs, so it must not be a mandatory dependency of
    fwl-proteus. It lives in ``[project.optional-dependencies]`` under its
    own extra, carrying a published version floor, and must NOT also carry
    a ``[tool.proteus.modules]`` SHA pin: a second pin can drift from the
    PyPI release, which is the dual-pin trap fwl-vulcan, fwl-aragog and
    fwl-zalmoxis are all kept out of.

    The floor is written zero-padded like the release tag that
    tools/get_morrigan.sh checks out through ``checkout_floor_tag``. PEP 440
    treats the padded and normalised forms as the same version, so one
    string serves the resolver and names the tag.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / 'pyproject.toml').read_text(encoding='utf-8'))

    deps = data['project']['dependencies']
    morrigan_deps = [d for d in deps if 'morrigan' in d.lower()]
    assert morrigan_deps == [], (
        f'morrigan must not be a mandatory dependency of fwl-proteus: {morrigan_deps!r}'
    )
    # Discrimination: an empty dependencies list would also pass the check
    # above; pin a known-mandatory package as evidence the list is intact.
    assert any('fwl-calliope' in d for d in deps), 'mandatory dependency list is intact'

    extras = data['project']['optional-dependencies']
    morrigan_extra = extras.get('morrigan', [])
    assert any(r.startswith('fwl-morrigan>=') for r in morrigan_extra), (
        f'morrigan extra must keep its version floor, got {morrigan_extra!r}'
    )

    # Single pin: a git SHA alongside the version floor could drift from the
    # published release, so the module table must not carry morrigan.
    git_modules = data['tool']['proteus']['modules']
    assert 'morrigan' not in git_modules, (
        'morrigan must not have a [tool.proteus.modules] git pin; it is pinned '
        'once via the fwl-morrigan extra and the matching git tag, like '
        f'fwl-vulcan/fwl-aragog/fwl-zalmoxis. Found: {sorted(git_modules)}'
    )

    # The floor stays tag-shaped (zero-padded CalVer), so the floor text and
    # the tag name it selects read the same in pyproject.toml and git.
    floor = next(r for r in morrigan_extra if r.startswith('fwl-morrigan>=')).split('>=')[1]
    assert re.fullmatch(r'\d{2}\.\d{2}\.\d{2}', floor), (
        f'morrigan floor must be zero-padded CalVer to match the release tag, got {floor!r}'
    )

    # The script reads the floor through checkout_floor_tag, without a text search of its own.
    script = (repo_root / 'tools' / 'get_morrigan.sh').read_text(encoding='utf-8')
    assert 'checkout_floor_tag fwl-morrigan "$tmp_clone"' in script, (
        'tools/get_morrigan.sh must pin the checkout to the fwl-morrigan floor tag'
    )

    # A version in a comment of pyproject.toml is no floor: the pin reader parses TOML.
    poisoned = (
        (repo_root / 'pyproject.toml')
        .read_text(encoding='utf-8')
        .replace(
            f'morrigan = ["fwl-morrigan>={floor}"]',
            f'# later: needs fwl-morrigan>=99.99.99\nmorrigan = ["fwl-morrigan>={floor}"]',
        )
    )
    assert 'fwl-morrigan>=99.99.99' in poisoned
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'tools').mkdir()
        shutil.copy2(TOOLS_DIR / '_module_pins.py', Path(tmp) / 'tools' / '_module_pins.py')
        (Path(tmp) / 'pyproject.toml').write_text(poisoned, encoding='utf-8')
        extracted = subprocess.run(
            [
                sys.executable,
                str(Path(tmp) / 'tools' / '_module_pins.py'),
                'fwl-morrigan',
                'floor',
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    assert extracted == floor, (
        f'floor extraction picked {extracted!r} from a commented version instead of '
        f'the pin {floor!r}; get_morrigan.sh would check out a tag that does not exist'
    )


# Floor-tag clones against local remotes: get_python_module.sh and the module setup scripts

PYTHON_MODULE_SCRIPT = TOOLS_DIR / 'get_python_module.sh'
MANUAL_INSTALL_MD = TOOLS_DIR.parent / 'docs' / 'How-to' / 'manual_installation.md'


def _bare_remote(
    remotes: Path, repo: str, tag: str, files: dict[str, str] | None = None
) -> tuple[str, str]:
    """Publish ``remotes/FormingWorlds/<repo>.git`` with 3 commits and ``tag`` on the second.

    Every commit carries ``version.txt`` and ``fastchem_vulcan/Makefile``, so
    the VULCAN build step finds its directory; ``files`` maps further file names
    to their text. Returns the tagged commit and the tip, which differ, so a
    clone left on the branch is told apart.
    """
    work = remotes.parent / 'work' / repo
    (work / 'fastchem_vulcan').mkdir(parents=True)
    (work / 'fastchem_vulcan' / 'Makefile').write_text('all:\n')
    for name, text in (files or {}).items():
        (work / name).write_text(text)
    git(work, 'init', '-q', '-b', 'main')
    for i in range(3):
        (work / 'version.txt').write_text(f'{i}\n')
        git(work, 'add', '.')
        git(work, 'commit', '-q', '-m', f'commit {i}')
        if i == 1:
            git(work, 'tag', tag)
    shas = (git(work, 'rev-parse', 'HEAD~1'), git(work, 'rev-parse', 'HEAD'))
    git(remotes, 'clone', '-q', '--bare', str(work), str(remotes / f'{repo}.git'))
    return shas


def _tag_the_tip(site: dict, repo: str, tag: str = '26.11.01') -> None:
    """Put ``tag``, a release above every floor of the fixtures, on the tip of the remote of ``repo``.

    A ``_floor_tag_of`` that takes the newest tag then lands on the tip.
    """
    git(site['root'].parent / 'remotes' / 'FormingWorlds' / f'{repo}.git', 'tag', tag, 'main')


def _floor_tag_site(
    tmp_path: Path, pyproject: str, remotes: dict[str, str], files: dict[str, str] | None = None
) -> dict:
    """A PROTEUS copy, local bare remotes, and stubs for pip, make and ssh.

    The copy holds the shipped scripts, the helper library and the pin
    reader below ``pyproject``. ``remotes`` maps each repository name to
    the one tag it carries. The ssh stub refuses the key, so a script that
    probes clones over https, which git rewrites to the local remotes. The
    branch variables of CI are removed and git stops its repository search
    at ``tmp_path``, so get_aragog.sh and get_zalmoxis.sh pair no branch.
    """
    root = tmp_path / 'PROTEUS'
    (root / 'tools').mkdir(parents=True)
    for src in [*TOOLS_DIR.glob('get_*.sh'), COMMON_LIB, TOOLS_DIR / '_module_pins.py']:
        shutil.copy2(src, root / 'tools' / src.name)
    (root / 'pyproject.toml').write_text(pyproject)
    base = tmp_path / 'remotes'
    (base / 'FormingWorlds').mkdir(parents=True)
    shas = {
        repo: _bare_remote(base / 'FormingWorlds', repo, tag, files)
        for repo, tag in remotes.items()
    }
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    log = tmp_path / 'pip.log'
    log.write_text('')
    _write_stub(stubs, 'pip', '#!/bin/sh\necho "$@" >> "$STUB_PIP_LOG"\n')
    _write_stub(stubs, 'make', '#!/bin/sh\nexit 0\n')
    _write_stub(stubs, 'ssh', '#!/bin/sh\nexit 255\n')
    env = {
        **os.environ,
        'PATH': f'{stubs}{os.pathsep}{os.environ["PATH"]}',
        'STUB_PIP_LOG': str(log),
        'PROTEUS_MODULE_GIT_BASE': f'{base}/',
        'GIT_CONFIG_COUNT': '1',
        'GIT_CONFIG_KEY_0': f'url.{base}/.insteadOf',
        'GIT_CONFIG_VALUE_0': 'https://github.com/',
        'GIT_CEILING_DIRECTORIES': str(tmp_path),
    }
    for name in ('GIT_SSH_COMMAND', 'GITHUB_HEAD_REF', 'GITHUB_REF_NAME'):
        env.pop(name, None)
    return {'root': Path(os.path.realpath(root)), 'shas': shas, 'log': log, 'env': env}


def _default_signals() -> None:
    """Give the child the default action for HUP, INT and TERM, and unblock them.

    A shell cannot trap a signal that it found ignored, and a blocked signal stays pending;
    a test runner started with ``nohup``, in the background or by a CI step passes either on.
    """
    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, [sig])


def _run_tool(
    site: dict, script: str, *argv: str, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """Run ``tools/<script>`` of the site copy with ``argv``, from outside the copy."""
    return subprocess.run(
        ['bash', str(site['root'] / 'tools' / script), *argv],
        cwd=site['root'].parent if cwd is None else cwd,
        env=site['env'],
        capture_output=True,
        text=True,
        timeout=60,
        preexec_fn=_default_signals,
    )


def _is_detached(checkout: Path) -> bool:
    """Return whether ``checkout`` sits on a detached HEAD."""
    return (
        subprocess.run(
            ['git', '-C', str(checkout), 'symbolic-ref', '-q', 'HEAD'],
            capture_output=True,
            timeout=10,
        ).returncode
        == 1
    )


class TestGetPythonModule:
    """tools/get_python_module.sh clones a module at its floor tag, or stops cleanly."""

    @pytest.fixture
    def site(self, tmp_path):
        """MORS carries the tag of its floor 26.10.6; JANUS only 26.07.10, below 26.10.08."""
        pyproject = (
            '[project]\nname = "fixture"\n'
            'dependencies = ["fwl-mors>=26.10.6", "fwl-janus>=26.10.08"]\n'
        )
        return _floor_tag_site(tmp_path, pyproject, {'MORS': '26.10.06', 'JANUS': '26.07.10'})

    def test_new_clone_lands_on_the_floor_tag(self, site):
        """A new MORS clone sits detached at the tag 26.10.06 and is installed editable once.

        The tip is one commit past the tag and carries the newer tag
        26.11.01, so a clone left on the default branch, or one at the
        newest tag, fails the equality; the floor text 26.10.6 is not the tag
        name, so an exact-text checkout fails the run.
        """
        tagged, tip = site['shas']['MORS']
        _tag_the_tip(site, 'MORS')
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stdout + result.stderr
        clone = site['root'] / 'MORS'
        assert git(clone, 'rev-parse', 'HEAD') == tagged != tip
        assert _is_detached(clone)
        assert f'fwl-mors: checked out release tag 26.10.06 at {tagged[:7]}' in result.stdout
        assert site['log'].read_text().splitlines() == [f'install -U -e {clone}']

    @pytest.mark.parametrize(
        'argv',
        [
            (),
            ('mors',),
            ('Zalmoxis',),
            ('MORS', 'JANUS'),
            ('MORS', '--force', 'JANUS'),
            ('MORS JANUS',),
            ('--force',),
            ('--list',),
        ],
        ids=[
            'no name',
            'lower case',
            'module with its own script',
            'two names',
            'two names around force',
            'two names in one string',
            'force only',
            'option the script does not take',
        ],
    )
    def test_unknown_name_exits_2_before_any_clone(self, site, argv):
        """A missing, unknown or second module name prints the accepted names and exits 2.

        Nothing is cloned or installed: the copy holds only what the
        fixture put there, and the pip log stays empty. With 2 accepted
        names, a script that takes the first installs MORS and exits 0.
        """
        result = _run_tool(site, 'get_python_module.sh', *argv)
        assert result.returncode == 2
        assert 'usage: bash tools/get_python_module.sh <RepoName> [--force]' in result.stderr
        assert 'MORS JANUS CALLIOPE ZEPHYRUS' in result.stderr
        assert sorted(p.name for p in site['root'].iterdir()) == ['pyproject.toml', 'tools']
        assert site['log'].read_text() == ''

    @pytest.mark.parametrize(
        ('argv', 'hint'),
        [
            (('mors',), 'write MORS, not mors'),
            (('Calliope', '--force'), 'write CALLIOPE, not Calliope'),
            (('Zalmoxis',), None),
            (('MORSE',), None),
        ],
        ids=['lower case', 'mixed case with force', 'module with its own script', 'near miss'],
    )
    def test_name_that_differs_only_in_case_names_the_spelling(self, site, argv, hint):
        """A module name that matches an accepted name apart from case exits 2 with the usage
        and a line that names the right spelling; any other unknown name gets no such line,
        as no spelling would fix it. Nothing is cloned or installed."""
        result = _run_tool(site, 'get_python_module.sh', *argv)
        assert result.returncode == 2
        assert 'usage: bash tools/get_python_module.sh <RepoName> [--force]' in result.stderr
        hints = [ln for ln in result.stderr.splitlines() if 'case-sensitive' in ln]
        expected = [f'       The name is case-sensitive: {hint}'] if hint else []
        assert hints == expected
        assert sorted(p.name for p in site['root'].iterdir()) == ['pyproject.toml', 'tools']
        assert site['log'].read_text() == ''

    def test_failed_new_clone_does_not_call_a_missing_checkout_unchanged(self, site):
        """A first clone from a missing remote exits 1 with the clone failure alone: there was
        no checkout, so the message does not call it unchanged, and none is left."""
        site['env']['PROTEUS_MODULE_GIT_BASE'] = str(site['root'].parent / 'missing') + '/'
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 1
        assert 'ERROR: git clone failed' in result.stderr
        assert 'unchanged' not in result.stderr
        assert sorted(p.name for p in site['root'].iterdir()) == ['pyproject.toml', 'tools']
        assert site['log'].read_text() == ''

    def test_plain_folder_is_installed_as_it_is(self, site):
        """A MORS folder that is not a git clone is kept with its file, named in a stay line with
        the floor release and the --force command, and installed editable; nothing is cloned."""
        plain = site['root'] / 'MORS'
        plain.mkdir()
        (plain / 'notes.txt').write_text('mine\n')
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stdout + result.stderr
        lines = result.stdout.splitlines()
        stay = f'fwl-mors: {plain} stays as it is: not a git clone; PROTEUS expects release 26.10.6'
        assert lines[lines.index(stay) + 1 :][:2] == [
            (
                '    From the PROTEUS folder, replace it with a new clone, which deletes everything'
                ' in it, with:'
            ),
            '        bash tools/get_python_module.sh MORS --force',
        ]
        assert 'fetch --tags' not in result.stdout
        assert 'Cloning' not in result.stdout
        assert sorted(p.name for p in plain.iterdir()) == ['notes.txt']
        assert site['log'].read_text().splitlines() == [f'install -U -e {plain}']

    def test_worktree_with_a_modified_file_is_kept(self, site, tmp_path):
        """A MORS checkout that is a git worktree (.git is a file) with a modified tracked
        file is read as a clone of its own: it stays on its commit with the change, gets the
        stay line and is installed. A test for a .git directory calls it not a git clone."""
        source = tmp_path / 'mors-source'
        source.mkdir()
        git(source, 'init', '-q', '-b', 'main')
        (source / 'version.txt').write_text('0\n')
        git(source, 'add', 'version.txt')
        git(source, 'commit', '-q', '-m', 'commit 0')
        head = git(source, 'rev-parse', 'HEAD')
        clone = site['root'] / 'MORS'
        git(source, 'worktree', 'add', '-q', '--detach', str(clone))
        assert (clone / '.git').is_file()
        (clone / 'version.txt').write_text('local work\n')

        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stdout + result.stderr
        # The source has no tag, so the line names the floor release.
        assert (
            f'fwl-mors: {clone} stays as it is: commit {head[:7]};'
            ' PROTEUS expects release 26.10.6'
        ) in result.stdout.splitlines()
        assert (clone / 'version.txt').read_text() == 'local work\n'
        assert git(clone, 'rev-parse', 'HEAD') == head
        assert site['log'].read_text().splitlines() == [f'install -U -e {clone}']

    def test_modified_checkout_is_kept_then_replaced_with_force(self, site):
        """A clone at its tag with a modified tracked file is installed as it is without
        --force, with the at-tag line, and replaced by a new clone with it."""
        tagged, _ = site['shas']['MORS']
        clone = site['root'] / 'MORS'
        assert _run_tool(site, 'get_python_module.sh', 'MORS').returncode == 0
        (clone / 'version.txt').write_text('local work\n')

        kept = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert kept.returncode == 0, kept.stdout + kept.stderr
        assert f'fwl-mors: {clone} is at the release tag 26.10.06' in kept.stdout
        assert 'stays as it is' not in kept.stdout
        assert (clone / 'version.txt').read_text() == 'local work\n'
        assert len(site['log'].read_text().splitlines()) == 2

        forced = _run_tool(site, 'get_python_module.sh', 'MORS', '--force')
        assert forced.returncode == 0, forced.stdout + forced.stderr
        assert (clone / 'version.txt').read_text() == '1\n'
        assert git(clone, 'rev-parse', 'HEAD') == tagged
        assert len(site['log'].read_text().splitlines()) == 3

    def test_floor_without_tag_exits_1_and_leaves_no_clone(self, site):
        """JANUS has no tag at its floor: exit 1 with the cause, no clone, no install."""
        result = _run_tool(site, 'get_python_module.sh', 'JANUS')
        assert result.returncode == 1
        assert "'fwl-janus' floor 26.10.08: no tag matches it" in result.stderr
        assert '26.07.10' in result.stderr
        assert not (site['root'] / 'JANUS').exists()
        assert site['log'].read_text() == ''


@pytest.mark.parametrize('op', ['>=', ' >= '], ids=['operator alone', 'spaces around it'])
@pytest.mark.parametrize(
    ('script', 'repo', 'dest'),
    [('get_vulcan.sh', 'VULCAN', 'VULCAN'), ('get_morrigan.sh', 'Morrigan', 'Morrigan')],
    ids=['vulcan', 'morrigan'],
)
def test_optional_module_floor_finds_a_tag_with_other_zero_padding(
    tmp_path, script, repo, dest, op
):
    """The floor 26.4.22 selects the tag 26.04.22 on the second of 3 commits.

    An exact-text checkout of ``tags/26.4.22`` names no tag and fails; a
    clone left on the default branch lands on the tip. With spaces around
    ``>=``, a text search for ``fwl-vulcan>=`` finds no floor.
    """
    pyproject = (
        '[project]\nname = "fixture"\ndependencies = []\n'
        '[project.optional-dependencies]\n'
        f'vulcan = ["fwl-vulcan{op}26.4.22"]\nmorrigan = ["fwl-morrigan{op}26.4.22"]\n'
    )
    site = _floor_tag_site(tmp_path, pyproject, {repo: '26.04.22'})
    tagged, tip = site['shas'][repo]
    result = _run_tool(site, script)
    assert result.returncode == 0, result.stdout + result.stderr
    clone = site['root'] / dest
    assert git(clone, 'rev-parse', 'HEAD') == tagged != tip
    assert 'release tag 26.04.22' in result.stdout
    assert len(site['log'].read_text().splitlines()) == 1


_PAIRED_MODULES = [
    ('get_aragog.sh', 'aragog', 'fwl-aragog'),
    ('get_zalmoxis.sh', 'Zalmoxis', 'fwl-zalmoxis'),
]


@pytest.mark.parametrize(
    ('script', 'repo', 'dist'), _PAIRED_MODULES, ids=['aragog', 'zalmoxis']
)
def test_paired_module_floor_finds_a_tag_with_other_zero_padding(tmp_path, script, repo, dist):
    """With no paired branch, the floor 26.4.22 selects the tag 26.04.22 on the second
    of 3 commits, detached, and the clone is installed once.

    An exact-text checkout of ``tags/26.4.22`` names no tag and fails; a
    clone left on the default branch, or one at the newest tag 26.11.01,
    lands on the tip.
    """
    pyproject = f'[project]\nname = "fixture"\ndependencies = ["{dist}>=26.4.22"]\n'
    site = _floor_tag_site(tmp_path, pyproject, {repo: '26.04.22'})
    tagged, tip = site['shas'][repo]
    _tag_the_tip(site, repo)
    result = _run_tool(site, script)
    assert result.returncode == 0, result.stdout + result.stderr
    clone = site['root'] / repo
    assert git(clone, 'rev-parse', 'HEAD') == tagged != tip
    assert _is_detached(clone)
    assert f'{dist}: checked out release tag 26.04.22 at {tagged[:7]}' in result.stdout
    assert site['log'].read_text().splitlines() == [f'install -U -e {clone}']


@pytest.mark.parametrize(
    ('script', 'repo', 'dist'), _PAIRED_MODULES, ids=['aragog', 'zalmoxis']
)
def test_paired_module_floor_without_tag_exits_1_before_pip(tmp_path, script, repo, dist):
    """A floor of 26.11.1, above the only tag 26.04.22, exits 1 with the cause and
    does not install; a script that skips the tag checkout installs the tip."""
    pyproject = f'[project]\nname = "fixture"\ndependencies = ["{dist}>=26.11.1"]\n'
    site = _floor_tag_site(tmp_path, pyproject, {repo: '26.04.22'})
    result = _run_tool(site, script)
    assert result.returncode == 1
    assert f"'{dist}' floor 26.11.1: no tag matches it" in result.stderr
    assert '26.04.22' in result.stderr
    assert site['log'].read_text() == ''


_PAIRED_BRANCH = 'tl/paired-feature'
_WITHOUT_TAG_LINE = (
    '    This branch does not hold the release 26.04.22: the PROTEUS install takes that'
    ' release from PyPI, not this folder.'
)


def _paired_site(tmp_path: Path, repo: str, dist: str) -> dict:
    """A site whose remote of ``repo`` has the tag 26.04.22 and a branch of the PROTEUS feature.

    The branch ``_PAIRED_BRANCH`` carries one commit of its own off the
    commit below the tag, so a checkout of it is told apart from the tag and
    the tip, and it lies on a line of history without the tag. A ``git``
    wrapper first on PATH logs every call to ``site['git_log']``, and git
    rewrites the github URLs to the local remotes, so no call leaves the host.
    """
    pyproject = f'[project]\nname = "fixture"\ndependencies = ["{dist}>=26.4.22"]\n'
    site = _floor_tag_site(tmp_path, pyproject, {repo: '26.04.22'})
    work = tmp_path / 'remotes' / 'work' / repo
    git(work, 'checkout', '-q', '-b', _PAIRED_BRANCH, '26.04.22~1')
    (work / 'version.txt').write_text('paired\n')
    git(work, 'commit', '-q', '-am', 'paired commit')
    site['paired'] = git(work, 'rev-parse', 'HEAD')
    git(work, 'push', '-q', str(tmp_path / 'remotes' / 'FormingWorlds' / f'{repo}.git'), 'HEAD')
    log = tmp_path / 'git.log'
    log.write_text('')
    _write_stub(
        tmp_path / 'stubs',
        'git',
        f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> "{log}"\nexec {shutil.which("git")} "$@"\n',
    )
    site['git_log'] = log
    return site


def _git_calls(site: dict, subcommand: str) -> list[str]:
    """The logged git calls of the site whose subcommand is ``subcommand``."""
    return [
        ln for ln in site['git_log'].read_text().splitlines() if _subcommand(ln) == subcommand
    ]


def _subcommand(line: str) -> str:
    """The git subcommand of a logged argument line, after the ``-C`` and ``-c`` options."""
    words = line.split()
    i = 0
    while i < len(words) and words[i].startswith('-'):
        i += 2 if words[i] in ('-C', '-c') else 1
    return words[i] if i < len(words) else ''


@pytest.mark.parametrize(
    ('script', 'repo', 'dist'), _PAIRED_MODULES, ids=['aragog', 'zalmoxis']
)
class TestPairedModuleRerun:
    """get_aragog.sh and get_zalmoxis.sh keep an existing clone and pair a branch only for a new one."""

    def test_rerun_keeps_a_clone_behind_the_tag(self, tmp_path, script, repo, dist):
        """A clone detached one commit below the tag, with an untracked file, a stash entry
        and a marker in its .git, stays on its commit with all three; it gets the stay line
        and is installed again, and no clone, fetch or checkout runs.

        PROTEUS names a feature branch the remote has, so a script that clones
        anew lands on that branch, and one that moves the clone lands on the tag.
        """
        site = _paired_site(tmp_path, repo, dist)
        tagged, _ = site['shas'][repo]
        assert _run_tool(site, script).returncode == 0
        clone = site['root'] / repo
        git(clone, 'checkout', '-q', '--detach', 'HEAD~1')
        first = git(clone, 'rev-parse', 'HEAD')
        (clone / 'version.txt').write_text('stashed work\n')
        git(clone, 'stash', '-q')
        (clone / 'notes.txt').write_text('kept\n')
        (clone / '.git' / 'marker').write_text('old clone\n')
        site['git_log'].write_text('')
        site['env']['GITHUB_HEAD_REF'] = _PAIRED_BRANCH

        result = _run_tool(site, script)
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == first != tagged
        assert (clone / 'notes.txt').read_text() == 'kept\n'
        assert (clone / '.git' / 'marker').read_text() == 'old clone\n'
        assert 'stashed work' in git(clone, 'stash', 'show', '-p')
        line = (
            f'{dist}: {clone} stays as it is: commit {first[:7]}, behind the tag;'
            ' PROTEUS expects the release tag 26.04.22'
        )
        assert result.stdout.splitlines().count(line) == 1
        assert [_git_calls(site, sub) for sub in ('clone', 'fetch', 'checkout')] == [[]] * 3
        assert site['log'].read_text().splitlines() == [f'install -U -e {clone}'] * 2

    def test_rerun_keeps_a_clone_on_a_branch(self, tmp_path, script, repo, dist):
        """A clone a developer put on a branch old below the tag stays on old, with the stay
        line, and is installed again; git clone is not run and no tag line follows."""
        site = _paired_site(tmp_path, repo, dist)
        tagged, _ = site['shas'][repo]
        assert _run_tool(site, script).returncode == 0
        clone = site['root'] / repo
        git(clone, 'checkout', '-q', '-b', 'old', 'HEAD~1')
        below = git(clone, 'rev-parse', 'HEAD')
        site['git_log'].write_text('')

        result = _run_tool(site, script)
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(clone, 'symbolic-ref', '--short', 'HEAD') == 'old'
        assert git(clone, 'rev-parse', 'HEAD') == below != tagged
        line = (
            f'{dist}: {clone} stays as it is: on branch old, behind the tag;'
            ' PROTEUS expects the release tag 26.04.22'
        )
        assert result.stdout.splitlines().count(line) == 1
        assert 'checked out release tag' not in result.stdout
        assert 'is at the release tag' not in result.stdout
        assert _git_calls(site, 'clone') == []
        assert site['log'].read_text().splitlines()[1:] == [f'install -U -e {clone}']

    def test_force_replaces_the_clone_with_the_paired_branch(
        self, tmp_path, script, repo, dist
    ):
        """With --force and a PROTEUS feature branch that the remote has, a new clone on that
        branch replaces the old one: the marker in the old .git is gone."""
        site = _paired_site(tmp_path, repo, dist)
        assert _run_tool(site, script).returncode == 0
        clone = site['root'] / repo
        (clone / '.git' / 'marker').write_text('old clone\n')
        site['git_log'].write_text('')
        site['env']['GITHUB_HEAD_REF'] = _PAIRED_BRANCH

        result = _run_tool(site, script, '--force')
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(clone, 'symbolic-ref', '--short', 'HEAD') == _PAIRED_BRANCH
        assert git(clone, 'rev-parse', 'HEAD') == site['paired']
        assert not (clone / '.git' / 'marker').exists()
        assert len(_git_calls(site, 'clone')) == 1
        assert 'stays as it is' not in result.stdout

    def test_new_clone_takes_the_paired_branch(self, tmp_path, script, repo, dist):
        """Without a clone, a PROTEUS feature branch that the remote has gives a new clone on
        that branch, at its own commit, neither the tag nor the tip. The paired branch of the
        fixture starts below the tag, so the first install prints the line about the release
        from PyPI and records the summary entry, as a second run does."""
        site = _paired_site(tmp_path, repo, dist)
        tagged, tip = site['shas'][repo]
        stays = tmp_path / 'stays.txt'
        site['env']['PROTEUS_CLONE_STAYS'] = str(stays)
        site['env']['GITHUB_HEAD_REF'] = _PAIRED_BRANCH
        result = _run_tool(site, script)
        assert result.returncode == 0, result.stdout + result.stderr
        clone = site['root'] / repo
        assert git(clone, 'symbolic-ref', '--short', 'HEAD') == _PAIRED_BRANCH
        assert git(clone, 'rev-parse', 'HEAD') == site['paired']
        assert site['paired'] not in (tagged, tip)
        assert len(_git_calls(site, 'clone')) == 1
        paired_line = f"PROTEUS is on branch '{_PAIRED_BRANCH}'; installed paired {repo} branch"
        assert f'{paired_line} at {site["paired"][:7]}' in result.stdout
        assert result.stdout.splitlines().count(_WITHOUT_TAG_LINE) == 1
        assert stays.read_text() == (
            f'{repo} (paired branch {_PAIRED_BRANCH}, without the release 26.04.22)\n'
        )

    @pytest.mark.parametrize('source', ['CI variable', 'PROTEUS checkout'])
    def test_rerun_on_a_paired_branch_without_the_tag_names_it(
        self, tmp_path, script, repo, dist, source
    ):
        """A second run, with PROTEUS still on the feature branch (named by GITHUB_HEAD_REF, or
        the branch of the PROTEUS checkout), prints the neutral line for the clone on the paired
        branch and, as the branch lies on a line of history without the tag, the line that the
        PROTEUS install takes the release from PyPI, and the summary entry: no stay line, no
        hand commands, no clone, fetch or checkout. Once PROTEUS leaves the branch, a reader
        that does not know the paired branch prints the stay line."""
        site = _paired_site(tmp_path, repo, dist)
        if source == 'CI variable':
            site['env']['GITHUB_HEAD_REF'] = _PAIRED_BRANCH
        else:
            git(site['root'], 'init', '-q', '-b', _PAIRED_BRANCH)
            git(site['root'], 'add', '-A')
            git(site['root'], 'commit', '-q', '-m', 'PROTEUS')
        assert _run_tool(site, script).returncode == 0
        clone = site['root'] / repo
        assert git(clone, 'rev-parse', 'HEAD') == site['paired']
        stays = tmp_path / 'stays.txt'
        site['env']['PROTEUS_CLONE_STAYS'] = str(stays)
        site['git_log'].write_text('')

        result = _run_tool(site, script)
        assert result.returncode == 0, result.stdout + result.stderr
        line = f'{dist}: {clone} is on the paired branch {_PAIRED_BRANCH}, as PROTEUS on that'
        out = result.stdout.splitlines()
        assert [ln for ln in out if ln.startswith(f'{dist}: ')] == [f'{line} branch expects']
        assert out[out.index(f'{line} branch expects') + 1] == _WITHOUT_TAG_LINE
        assert 'stays as it is' not in result.stdout
        assert 'git -C' not in result.stdout and '--force' not in result.stdout
        assert stays.read_text() == (
            f'{repo} (paired branch {_PAIRED_BRANCH}, without the release 26.04.22)\n'
        )
        assert [_git_calls(site, sub) for sub in ('clone', 'fetch', 'checkout')] == [[]] * 3
        assert git(clone, 'rev-parse', 'HEAD') == site['paired']

        stays.unlink()
        site['env'].pop('GITHUB_HEAD_REF', None)
        site['env']['GITHUB_REF_NAME'] = 'other-feature'
        moved = _run_tool(site, script)
        assert moved.returncode == 0, moved.stdout + moved.stderr
        assert (
            f'{dist}: {clone} stays as it is: on branch {_PAIRED_BRANCH}, on a line of history'
            ' without the tag; PROTEUS expects the release tag 26.04.22'
        ) in moved.stdout.splitlines()
        assert stays.read_text() == f'{repo} (branch {_PAIRED_BRANCH}, other history)\n'

    @pytest.mark.parametrize(
        ('start', 'holds'),
        [('commit below the tag', False), ('commit past the tag', True)],
    )
    def test_paired_branch_line_names_the_release_only_when_head_lacks_it(
        self, tmp_path, script, repo, dist, start, holds
    ):
        """A clone on the paired branch reset to the commit below the tag (behind it) gets the
        neutral line, the line that the release comes from PyPI and the summary entry; reset to
        the commit past the tag, which holds the tag commit, it gets the neutral line alone and
        no entry. Neither gets a hand command, and HEAD stays where it was put."""
        site = _paired_site(tmp_path, repo, dist)
        tagged, tip = site['shas'][repo]
        site['env']['GITHUB_HEAD_REF'] = _PAIRED_BRANCH
        assert _run_tool(site, script).returncode == 0
        clone = site['root'] / repo
        target = tip if holds else git(clone, 'rev-parse', f'{tagged}~1')
        git(clone, 'reset', '-q', '--hard', target)
        stays = tmp_path / 'stays.txt'
        site['env']['PROTEUS_CLONE_STAYS'] = str(stays)

        result = _run_tool(site, script)
        assert result.returncode == 0, result.stdout + result.stderr
        out = result.stdout.splitlines()
        neutral = (
            f'{dist}: {clone} is on the paired branch {_PAIRED_BRANCH}, as PROTEUS on that'
            ' branch expects'
        )
        assert [ln for ln in out if ln.startswith(f'{dist}: ')] == [neutral]
        assert (_WITHOUT_TAG_LINE in out) is not holds
        assert 'git -C' not in result.stdout and '--force' not in result.stdout
        if holds:
            assert not stays.exists()
        else:
            assert out[out.index(neutral) + 1] == _WITHOUT_TAG_LINE
            assert stays.read_text() == (
                f'{repo} (paired branch {_PAIRED_BRANCH}, without the release 26.04.22)\n'
            )
        assert git(clone, 'rev-parse', 'HEAD') == target != tagged
        assert git(clone, 'symbolic-ref', '--short', 'HEAD') == _PAIRED_BRANCH

    def test_plain_folder_is_installed_as_it_is(self, tmp_path, script, repo, dist):
        """A folder that is not a git clone keeps its file, gets the stay line with the floor
        release and the --force command, and is installed; git clone is not run."""
        site = _paired_site(tmp_path, repo, dist)
        plain = site['root'] / repo
        plain.mkdir()
        (plain / 'notes.txt').write_text('mine\n')
        result = _run_tool(site, script)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (
            f'{dist}: {plain} stays as it is: not a git clone; PROTEUS expects release 26.4.22'
        ) in result.stdout.splitlines()
        assert f'        bash tools/{script} --force' in result.stdout.splitlines()
        assert sorted(q.name for q in plain.iterdir()) == ['notes.txt']
        assert _git_calls(site, 'clone') == []
        assert site['log'].read_text().splitlines() == [f'install -U -e {plain}']


def test_module_without_a_paired_branch_rule_gets_the_stay_line(tmp_path):
    """MORS pairs no branch: its clone on a branch named like the PROTEUS feature branch, which
    its origin has, gets the stay line with the hand commands and the summary entry of any
    other branch, and no paired-branch line; the clone stays on the branch."""
    site = _paired_site(tmp_path, 'MORS', 'fwl-mors')
    site['env']['GITHUB_HEAD_REF'] = _PAIRED_BRANCH
    assert _run_tool(site, 'get_python_module.sh', 'MORS').returncode == 0
    clone = site['root'] / 'MORS'
    git(clone, 'checkout', '-q', '-b', _PAIRED_BRANCH, f'origin/{_PAIRED_BRANCH}')
    stays = tmp_path / 'stays.txt'
    site['env']['PROTEUS_CLONE_STAYS'] = str(stays)

    result = _run_tool(site, 'get_python_module.sh', 'MORS')
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout.splitlines()
    assert (
        f'fwl-mors: {clone} stays as it is: on branch {_PAIRED_BRANCH}, on a line of history'
        ' without the tag; PROTEUS expects the release tag 26.04.22'
    ) in out
    assert f'        git -C {clone} checkout --detach refs/tags/26.04.22' in out
    assert 'is on the paired branch' not in result.stdout
    assert stays.read_text() == f'MORS (branch {_PAIRED_BRANCH}, other history)\n'
    assert git(clone, 'rev-parse', 'HEAD') == site['paired']


# Each module with a setup script: the script, its arguments, the clone folder and the distribution.
_MODULES = [
    ('get_python_module.sh', ('MORS',), 'MORS', 'fwl-mors'),
    ('get_python_module.sh', ('JANUS',), 'JANUS', 'fwl-janus'),
    ('get_python_module.sh', ('CALLIOPE',), 'CALLIOPE', 'fwl-calliope'),
    ('get_python_module.sh', ('ZEPHYRUS',), 'ZEPHYRUS', 'fwl-zephyrus'),
    ('get_aragog.sh', (), 'aragog', 'fwl-aragog'),
    ('get_zalmoxis.sh', (), 'Zalmoxis', 'fwl-zalmoxis'),
    ('get_vulcan.sh', (), 'VULCAN', 'fwl-vulcan'),
    ('get_morrigan.sh', (), 'Morrigan', 'fwl-morrigan'),
]
_MODULE_IDS = [m[2].lower() for m in _MODULES]

_ALL_FLOORS_PYPROJECT = (
    '[project]\nname = "fixture"\ndependencies = ['
    + ', '.join(f'"{m[3]}>=26.4.22"' for m in _MODULES[:6])
    + ']\n[project.optional-dependencies]\n'
    'vulcan = ["fwl-vulcan>=26.4.22"]\nmorrigan = ["fwl-morrigan>=26.4.22"]\n'
)


@pytest.mark.parametrize('cause', ['failing remote', 'floor without a tag'])
@pytest.mark.parametrize(('script', 'argv', 'repo', 'dist'), _MODULES, ids=_MODULE_IDS)
def test_clean_clone_survives_a_failed_replacement(tmp_path, script, argv, repo, dist, cause):
    """With --force, a clone at its tag stays byte-equal when the new clone fails.

    The remote is moved away, or the floor is raised to 26.11.1, which no
    tag carries. Every file of the clone, .git included, keeps its bytes,
    pip runs only for the first install and no temporary sibling is left; a
    script that deletes the checkout before it clones leaves none at all.
    """
    site = _floor_tag_site(tmp_path, _ALL_FLOORS_PYPROJECT, {repo: '26.04.22'})
    tagged, _ = site['shas'][repo]
    first = _run_tool(site, script, *argv)
    assert first.returncode == 0, first.stdout + first.stderr
    clone = site['root'] / repo
    (clone / 'notes.txt').write_text('kept\n')
    before = tree_snapshot(clone)
    if cause == 'failing remote':
        remote = tmp_path / 'remotes' / 'FormingWorlds' / f'{repo}.git'
        remote.rename(remote.with_name(f'{repo}.moved'))
        expected = f'git clone failed; {clone}'
    else:
        pyproject = site['root'] / 'pyproject.toml'
        pyproject.write_text(pyproject.read_text().replace('26.4.22', '26.11.1'))
        expected = 'floor 26.11.1: no tag matches it'

    result = _run_tool(site, script, *argv, '--force')
    assert result.returncode == 1
    assert expected in result.stderr
    assert tree_snapshot(clone) == before
    assert git(clone, 'rev-parse', 'HEAD') == tagged
    assert len(site['log'].read_text().splitlines()) == 1
    assert sorted(p.name for p in site['root'].iterdir()) == [repo, 'pyproject.toml', 'tools']


def _logging_mv(stubs: Path, log: Path) -> None:
    """Put an ``mv`` first on PATH that logs its arguments and makes the real move."""
    _write_stub(
        stubs, 'mv', f'#!/bin/sh\necho "mv $*" >> {shlex.quote(str(log))}\nexec /bin/mv "$@"\n'
    )


@pytest.mark.parametrize(('script', 'argv', 'repo', 'dist'), _MODULES, ids=_MODULE_IDS)
def test_force_replaces_any_clone_through_the_swap(tmp_path, script, argv, repo, dist):
    """--force puts a new clone at the tag in place of a clone on a branch with local files:
    the old one moves aside, the new one moves in, and the old copy is removed.

    The 2 logged moves name the clone, its ``.old.<pid>`` copy and the
    ``.tmp.<pid>`` clone; the marker in the old .git and the untracked file
    are gone, and no sibling is left.
    """
    site = _floor_tag_site(tmp_path, _ALL_FLOORS_PYPROJECT, {repo: '26.04.22'})
    tagged, tip = site['shas'][repo]
    assert _run_tool(site, script, *argv).returncode == 0
    clone = site['root'] / repo
    git(clone, 'checkout', '-q', 'main')
    (clone / 'notes.txt').write_text('mine\n')
    (clone / '.git' / 'marker').write_text('old clone\n')
    moves = tmp_path / 'mv.log'
    _logging_mv(tmp_path / 'stubs', moves)

    result = _run_tool(site, script, *argv, '--force')
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(clone, 'rev-parse', 'HEAD') == tagged != tip
    assert _is_detached(clone)
    assert not (clone / 'notes.txt').exists()
    assert not (clone / '.git' / 'marker').exists()
    lines = moves.read_text().splitlines()
    assert len(lines) == 2, lines
    assert re.fullmatch(
        rf'mv {re.escape(str(clone))} {re.escape(str(clone))}\.old\.\d+', lines[0]
    )
    assert re.fullmatch(
        rf'mv {re.escape(str(clone))}\.tmp\.\d+ {re.escape(str(clone))}', lines[1]
    )
    assert 'stays as it is' not in result.stdout
    assert sorted(p.name for p in site['root'].iterdir()) == [repo, 'pyproject.toml', 'tools']


@pytest.mark.parametrize(
    ('signal', 'status'),
    [('TERM', 143), ('INT', 130), ('HUP', 129)],
    ids=['TERM', 'INT', 'HUP'],
)
@pytest.mark.parametrize(
    ('stopped', 'message', 'calls'),
    [
        (2, 'interrupted while replacing {clone}; the old checkout is back in place', 3),
        (1, 'interrupted before replacing {clone}; it is unchanged', 1),
    ],
    ids=['between the two moves', 'at the first move'],
)
def test_signal_during_the_swap_leaves_the_old_clone_in_place(
    tmp_path, stopped, message, calls, signal, status
):
    """A TERM, INT or HUP during swap_in_checkout under --force leaves the old MORS clone at
    MORS.

    A stub ``mv`` first on PATH makes the real move, except that its call
    number ``stopped`` moves nothing and sends the signal to its parent
    process, the bash that runs the script. At the 2nd call the handler moves
    the old clone back (its 3rd mv); at the 1st nothing was set aside and it
    makes no move. The untracked file and HEAD are kept, the EXIT trap removed
    the temporary clone, no ``MORS.old.*`` is left, the status is 128 plus the
    signal number and stderr names what happened. Without the handler of the
    signal, MORS is missing after a TERM or HUP at the 2nd call, and bash goes
    on after an INT that its child survived.
    """
    pyproject = '[project]\nname = "fixture"\ndependencies = ["fwl-mors>=26.10.6"]\n'
    site = _floor_tag_site(tmp_path, pyproject, {'MORS': '26.10.06'})
    assert _run_tool(site, 'get_python_module.sh', 'MORS').returncode == 0
    clone = site['root'] / 'MORS'
    (clone / 'notes.txt').write_text('kept\n')
    head = git(clone, 'rev-parse', 'HEAD')
    count = tmp_path / 'mv.count'
    count.write_text('0\n')
    _write_stub(
        tmp_path / 'stubs',
        'mv',
        '#!/bin/sh\n'
        f'n=$(( $(cat {shlex.quote(str(count))}) + 1 ))\n'
        f'echo "$n" > {shlex.quote(str(count))}\n'
        f'if [ "$n" -eq {stopped} ]; then kill -{signal} "$PPID"; exit 1; fi\n'
        'exec /bin/mv "$@"\n',
    )

    result = _run_tool(site, 'get_python_module.sh', 'MORS', '--force')
    assert result.returncode == status, result.stdout + result.stderr
    assert (clone / 'notes.txt').read_text() == 'kept\n'
    assert git(clone, 'rev-parse', 'HEAD') == head
    assert sorted(p.name for p in site['root'].iterdir()) == ['MORS', 'pyproject.toml', 'tools']
    assert f'ERROR: {message.format(clone=clone)}' in result.stderr
    assert count.read_text() == f'{calls}\n'
    assert 'Done!' not in result.stdout


@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
@pytest.mark.parametrize(
    ('outcome', 'status'),
    [('replaced', 0), ('second move fails', 1), ('first move fails', 1)],
)
def test_swap_leaves_no_signal_trap_behind(tmp_path, strict, outcome, status):
    """After swap_in_checkout returns, after a swap, after a new checkout that cannot move in,
    and after an old one that cannot move aside, ``trap -p HUP INT TERM`` prints nothing: a
    handler left set would move a checkout of the caller's later steps on a signal. The old
    checkout ends at its path in the two failed cases.
    """
    dest = _checkout_dir(tmp_path / 'MORS', 'old')
    new = tmp_path / 'MORS.tmp.1'
    if outcome != 'second move fails':
        _checkout_dir(new, 'new')
    path = os.environ['PATH']
    if outcome == 'first move fails':
        stubs = tmp_path / 'stubs'
        stubs.mkdir()
        _write_stub(stubs, 'mv', '#!/bin/sh\nexit 1\n')
        path = f'{stubs}{os.pathsep}{path}'
    traps = tmp_path / 'traps.txt'
    # trap -p writes to a file: in a command substitution it would list the traps of a subshell.
    body = (
        f'get_force=true\nrc=0\nswap_in_checkout "{new}" "{dest}" || rc=$?\n'
        f'echo "SWAP_STATUS $rc"\ntrap -p HUP INT TERM >"{traps}"\n'
        f'trap "echo held" TERM\ntrap -p TERM >>"{traps}.check"\n'
    )
    env = {**os.environ, 'PATH': path}
    res = _run_bash(_with_common(body, strict=strict), env=env, timeout=20)
    assert f'SWAP_STATUS {status}' in res.stdout, res.stderr
    assert traps.read_text() == ''
    assert 'echo held' in Path(f'{traps}.check').read_text()
    marker = 'new.txt' if outcome == 'replaced' else 'old.txt'
    assert sorted(p.name for p in dest.iterdir()) == [marker]


# The states of an existing module folder that the script must leave as they are.
_STATES = [
    'at the tag',
    'at the tag with a modified file',
    'at the tag with a staged file',
    'on a branch newer than the tag',
    'on a branch behind the tag',
    'detached one tag behind',
    'detached ahead',
    'modified tracked file',
    'untracked file and stash entry',
    'tracked file with a changed mtime',
    'plain folder',
    'empty .git in the PROTEUS repository',
]


def _put_folder_in_state(site: dict, repo: str, dist: str, state: str) -> tuple[Path, str, str]:
    """Make the module folder of ``repo`` in the site root in ``state``.

    The remote carries 26.04.21 on its first commit and the floor tag
    26.04.22 on its second. Returns the folder, the line the script must
    print for it and the entry of the installer summary (empty for a clone at
    the tag or newer than it).
    """
    tagged, tip = site['shas'][repo]
    remote = site['root'].parent / 'remotes' / 'FormingWorlds' / f'{repo}.git'
    first = git(remote, 'rev-parse', f'{tagged}~1')
    git(remote, 'tag', '26.04.21', first)
    folder = site['root'] / repo
    at_tag = f'{dist}: {folder} is at the release tag 26.04.22'
    newer = f'{dist}: {folder} is newer than the release tag 26.04.22'
    expects = 'PROTEUS expects the release tag 26.04.22'
    if state in ('plain folder', 'empty .git in the PROTEUS repository'):
        (folder / 'fastchem_vulcan').mkdir(parents=True)
        (folder / 'fastchem_vulcan' / 'Makefile').write_text('all:\n')
        (folder / 'notes.txt').write_text('mine\n')
        if state == 'plain folder':
            found = 'not a git clone'
        else:
            (folder / '.git').mkdir()
            found = 'not a readable git clone'
        return (
            folder,
            f'{dist}: {folder} stays as it is: {found}; PROTEUS expects release 26.4.22',
            found,
        )
    git(site['root'], 'clone', '-q', str(remote), str(folder))
    if state.startswith('at the tag'):
        git(folder, 'checkout', '-q', 'tags/26.04.22')
        if state == 'at the tag':
            return folder, at_tag, ''
        (folder / 'version.txt').write_text('local work\n')
        if state == 'at the tag with a staged file':
            git(folder, 'add', 'version.txt')
        return folder, f'{at_tag}, with local changes', ''
    if state in ('on a branch newer than the tag', 'modified tracked file'):
        if state == 'modified tracked file':
            (folder / 'version.txt').write_text('local work\n')
        return folder, f'{newer} (branch main)', ''
    if state == 'on a branch behind the tag':
        git(folder, 'checkout', '-q', '-b', 'old', first)
        found = 'on branch old, behind the tag'
        summary = 'branch old, behind'
    elif state == 'detached one tag behind':
        git(folder, 'checkout', '-q', 'tags/26.04.21')
        found = f'commit {first[:7]} (26.04.21), behind the tag'
        summary = f'commit {first[:7]}, behind'
    elif state == 'detached ahead':
        git(folder, 'checkout', '-q', '--detach', tip)
        return folder, f'{newer} (commit {tip[:7]})', ''
    elif state == 'tracked file with a changed mtime':
        # Stale stat data in the index: git status rewrites .git/index unless its lock is optional.
        git(folder, 'checkout', '-q', 'tags/26.04.22')
        os.utime(folder / 'version.txt', (1_000_000_000, 1_000_000_000))
        return folder, at_tag, ''
    else:
        git(folder, 'checkout', '-q', 'tags/26.04.22')
        (folder / 'version.txt').write_text('stashed work\n')
        git(folder, 'stash', '-q')
        (folder / 'notes.txt').write_text('mine\n')
        return folder, at_tag, ''
    return folder, f'{dist}: {folder} stays as it is: {found}; {expects}', summary


def _repo_state(path: Path) -> tuple[str, str, str]:
    """HEAD, branch and ``git status --porcelain`` of the repository at ``path``, read without
    a write to its index."""
    return (
        git(path, 'rev-parse', 'HEAD'),
        git(path, 'symbolic-ref', '--short', 'HEAD'),
        git(path, '--no-optional-locks', 'status', '--porcelain'),
    )


@pytest.mark.parametrize('state', _STATES)
@pytest.mark.parametrize(('script', 'argv', 'repo', 'dist'), _MODULES, ids=_MODULE_IDS)
def test_no_git_write_and_no_change_in_an_existing_folder(
    tmp_path, script, argv, repo, dist, state
):
    """Without --force, a module script changes nothing in what is at the module path.

    A git wrapper first on PATH logs every call and refuses, with status 97
    and a marker file, every call on the module folder or the PROTEUS root,
    which is a git repository here, that is not on its read-only list (a
    plain ``git status`` is not: it can rewrite the index);
    GIT_CEILING_DIRECTORIES is unset, so an empty .git makes git answer for
    that root. After the run: status 0, no marker, every logged call read-only,
    every file of the folder (.git included, its index too, which a tracked
    file of changed mtime makes stale and which the status call for a clone
    at the tag with a modified or staged file reads) has its size and SHA-256
    of before (the pip and make stubs write their logs outside it, so no
    change is allowed), the state line is the expected one, the summary file
    has its entry (none for a clone at the tag or newer than it), and the root
    keeps its HEAD, branch and status. VULCAN builds fastchem in the kept
    folder before pip.
    """
    site = _floor_tag_site(tmp_path, _ALL_FLOORS_PYPROJECT, {repo: '26.04.22'})
    root = site['root']
    git(root, 'init', '-q', '-b', 'main')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', 'PROTEUS')
    folder, line, summary = _put_folder_in_state(site, repo, dist, state)
    index = folder / '.git' / 'index'
    index_before = index.read_bytes() if index.is_file() else None
    before, root_before = tree_snapshot(folder), _repo_state(root)

    guard = tmp_path / 'guard'
    guard.mkdir()
    log, marker = tmp_path / 'git-calls.log', tmp_path / 'git-write.marker'
    install_git_guard(guard, log, marker, roots=[root], trees=[folder])
    make_log = tmp_path / 'make.log'
    _write_stub(
        tmp_path / 'stubs', 'make', f'#!/bin/sh\npwd -P >> {shlex.quote(str(make_log))}\n'
    )
    stays = tmp_path / 'stays.txt'
    env = {**site['env'], 'PATH': f'{guard}{os.pathsep}{site["env"]["PATH"]}'}
    env['PROTEUS_CLONE_STAYS'] = str(stays)
    env.pop('GIT_CEILING_DIRECTORIES')
    result = subprocess.run(
        ['bash', str(root / 'tools' / script), *argv],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    calls = log.read_text().splitlines() if log.exists() else []
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists(), calls
    guarded = [c for c in calls if f'-C {folder}' in c or f'-C {root} ' in c]
    assert [c for c in guarded if not git_call_is_read_only(c.split('\t')[1].split())] == []
    assert (index.read_bytes() if index.is_file() else None) == index_before
    assert tree_snapshot(folder) == before
    assert result.stdout.splitlines().count(line) == 1, result.stdout
    assert result.stdout.splitlines()[-1] == 'Done!'
    assert (stays.read_text() if stays.exists() else '') == (
        f'{repo} ({summary})\n' if summary else ''
    )
    assert _repo_state(root) == root_before
    assert site['log'].read_text().splitlines() == [f'install -U -e {folder}']
    built = make_log.read_text().splitlines() if make_log.exists() else []
    assert built == ([f'{folder}/fastchem_vulcan'] if repo == 'VULCAN' else [])
    # The wrapper saw the calls, and the first one on a folder with .git proves its top level.
    on_folder = [c.split('\t')[1] for c in calls if f'-C {folder} ' in c]
    if state == 'plain folder':
        assert on_folder == []
    else:
        assert on_folder[0] == f'-C {folder} rev-parse --show-toplevel'


@pytest.mark.parametrize(
    ('argv', 'read_only'),
    [
        (['-C', '{repo}', 'rev-parse', '--show-toplevel'], True),
        (['-C', '{repo}', 'symbolic-ref', '-q', '--short', 'HEAD'], True),
        (['-C', '{repo}', 'tag', '-l'], True),
        (['--no-optional-locks', '-C', '{repo}', 'status', '--porcelain'], True),
        (['-C', '{repo}', 'status', '--porcelain'], False),
        (['-C', '{repo}', 'update-index', '-q', '--refresh'], False),
        (['-C', '{repo}', 'symbolic-ref', 'HEAD', 'refs/heads/other'], False),
        (['-C', '{repo}', 'symbolic-ref', '-d', 'HEAD'], False),
        (['-C', '{repo}', 'tag', '-l', '-d', '26.10.06'], False),
        (['-C', '{repo}', 'tag', '26.11.01'], False),
        (['-C', '{repo}', 'gc', '--auto'], False),
        (['clone', '-q', '{other}', '{repo}/sub'], False),
    ],
    ids=[
        'rev-parse',
        'symbolic-ref read',
        'tag list',
        'status without optional locks',
        'status',
        'update-index refresh',
        'symbolic-ref write',
        'symbolic-ref delete',
        'tag list with delete',
        'tag create',
        'gc',
        'clone into the folder',
    ],
)
def test_git_guard_runs_only_read_only_calls_on_a_guarded_folder(tmp_path, argv, read_only):
    """The guard of the clone tests runs a call on a guarded folder only when it is on the
    read-only list, which the Python check of the guard agrees with; any other call exits 97,
    writes the marker and leaves HEAD, the tags and .git/index byte-equal. The same call on
    a folder that is not guarded runs."""
    repo, other = tmp_path / 'MORS', tmp_path / 'other'
    for path in (repo, other):
        path.mkdir()
        git(path, 'init', '-q', '-b', 'main')
        (path / 'version.txt').write_text('0\n')
        git(path, 'add', 'version.txt')
        git(path, 'commit', '-q', '-m', 'commit 0')
        git(path, 'tag', '26.10.06')
        os.utime(path / 'version.txt', (1_000_000_000, 1_000_000_000))
    guard = tmp_path / 'guard'
    guard.mkdir()
    log, marker = tmp_path / 'git-calls.log', tmp_path / 'git-write.marker'
    install_git_guard(guard, log, marker, roots=[], trees=[repo])
    env = {**os.environ, 'PATH': f'{guard}{os.pathsep}{os.environ["PATH"]}'}
    env.pop('GIT_OPTIONAL_LOCKS', None)
    words = [w.format(repo=repo, other=other) for w in argv]
    index = (repo / '.git' / 'index').read_bytes()
    head = (repo / '.git' / 'HEAD').read_bytes()
    refs = git(repo, 'for-each-ref')

    result = subprocess.run(
        ['git', *words], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20
    )
    assert git_call_is_read_only(words) is read_only
    assert (result.returncode != 97) is read_only, result.stderr
    assert marker.exists() is not read_only
    if not read_only:
        assert (repo / '.git' / 'index').read_bytes() == index
        assert (repo / '.git' / 'HEAD').read_bytes() == head
        assert git(repo, 'for-each-ref') == refs
        assert not (repo / 'sub').exists()
        on_other = [w.replace(str(repo), str(other)) for w in words]
        marker.unlink()
        rerun = subprocess.run(
            ['git', *on_other], cwd=tmp_path, env=env, capture_output=True, timeout=20
        )
        assert rerun.returncode != 97
        assert not marker.exists()


def _other_repository(path: Path) -> None:
    """Make a repository at ``path`` on the branch ``elsewhere`` with one commit."""
    path.mkdir()
    git(path, 'init', '-q', '-b', 'elsewhere')
    (path / 'other.txt').write_text('other\n')
    git(path, 'add', 'other.txt')
    git(path, 'commit', '-q', '-m', 'other')


def _other_state(path: Path) -> tuple[str, str, bytes, str]:
    """HEAD, branch, .git/index bytes and status of the repository at ``path``, read in this
    order so that the status call cannot change the index bytes it is compared with."""
    index = (path / '.git' / 'index').read_bytes()
    return (
        git(path, 'rev-parse', 'HEAD'),
        git(path, 'symbolic-ref', '--short', 'HEAD'),
        index,
        git(path, '--no-optional-locks', 'status', '--porcelain'),
    )


def test_git_dir_in_the_environment_does_not_redirect_the_state_line(tmp_path):
    """With GIT_DIR naming another repository, a MORS folder with an empty .git in a PROTEUS
    root that is a git repository is not a readable clone, and the other repository keeps its
    HEAD, branch, index bytes and status.

    With GIT_DIR set, ``git -C MORS rev-parse --show-toplevel`` names MORS and every later
    call reads the other repository, so the line would say ``on branch elsewhere``.
    """
    site = _floor_tag_site(tmp_path, _ALL_FLOORS_PYPROJECT, {'MORS': '26.04.22'})
    root = site['root']
    git(root, 'init', '-q', '-b', 'main')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', 'PROTEUS')
    (root / 'MORS' / '.git').mkdir(parents=True)
    other = tmp_path / 'other'
    _other_repository(other)
    before = _other_state(other)
    site['env']['GIT_DIR'] = str(other / '.git')
    site['env'].pop('GIT_CEILING_DIRECTORIES')

    result = _run_tool(site, 'get_python_module.sh', 'MORS')
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f'fwl-mors: {root / "MORS"} stays as it is: not a readable git clone;'
        ' PROTEUS expects release 26.4.22'
    ) in result.stdout.splitlines()
    assert 'elsewhere' not in result.stdout
    assert _other_state(other) == before
    assert sorted(p.name for p in (root / 'MORS').iterdir()) == ['.git']


def test_new_clone_with_git_dir_in_the_environment_lands_at_its_tag(tmp_path):
    """With GIT_DIR naming another repository, a new MORS clone still lands detached at the
    tag 26.04.22 in MORS, and the other repository keeps its HEAD, branch, index bytes and
    status; the checkout of the tag would act on that repository otherwise."""
    site = _floor_tag_site(tmp_path, _ALL_FLOORS_PYPROJECT, {'MORS': '26.04.22'})
    tagged, tip = site['shas']['MORS']
    other = tmp_path / 'other'
    _other_repository(other)
    before = _other_state(other)
    site['env']['GIT_DIR'] = str(other / '.git')

    result = _run_tool(site, 'get_python_module.sh', 'MORS')
    assert result.returncode == 0, result.stdout + result.stderr
    clone = site['root'] / 'MORS'
    assert git(clone, 'rev-parse', 'HEAD') == tagged != tip
    assert _is_detached(clone)
    assert _other_state(other) == before


# Two scripts that share install_existing_clone, one under set -euo pipefail and one without.
_EXISTING_SCRIPTS = [
    ('get_python_module.sh', ('MORS',), 'MORS', 'fwl-mors'),
    ('get_aragog.sh', (), 'aragog', 'fwl-aragog'),
]

# A pip that logs its arguments and fails when STUB_PIP_FAIL is set or its last one is no folder.
_PICKY_PIP = (
    '#!/bin/sh\necho "$@" >> "$STUB_PIP_LOG"\nfor a; do last="$a"; done\n'
    '[ -z "${STUB_PIP_FAIL:-}" ] && [ -d "$last" ]\n'
)


@pytest.mark.parametrize(
    ('script', 'argv', 'repo', 'dist'), _EXISTING_SCRIPTS, ids=['mors', 'aragog']
)
class TestWhatIsAtTheModulePath:
    """install_existing_clone and swap_in_checkout for an empty folder, a symlink, a folder pip
    rejects, and a folder that appears while the new clone is made."""

    @staticmethod
    def _site(tmp_path, repo: str) -> dict:
        """A site with the floor tag 26.04.22 for every module and the picky pip stub."""
        site = _floor_tag_site(tmp_path, _ALL_FLOORS_PYPROJECT, {repo: '26.04.22'})
        _write_stub(tmp_path / 'stubs', 'pip', _PICKY_PIP)
        return site

    def test_empty_folder_gets_a_new_clone_at_the_tag(self, tmp_path, script, argv, repo, dist):
        """An empty folder at the module path holds nothing to lose: it is replaced by a new
        clone detached at the tag, which is installed; no stay line and no sibling are left."""
        site = self._site(tmp_path, repo)
        tagged, tip = site['shas'][repo]
        folder = site['root'] / repo
        folder.mkdir()
        result = _run_tool(site, script, *argv)
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(folder, 'rev-parse', 'HEAD') == tagged != tip
        assert _is_detached(folder)
        assert 'stays as it is' not in result.stdout
        assert f'{dist}: checked out release tag 26.04.22 at {tagged[:7]}' in result.stdout
        assert site['log'].read_text().splitlines() == [f'install -U -e {folder}']
        assert sorted(p.name for p in site['root'].iterdir()) == [
            repo,
            'pyproject.toml',
            'tools',
        ]

    @pytest.mark.parametrize(
        ('kind', 'project_file'),
        [
            ('dangling symlink', None),
            ('folder pip rejects', None),
            ('package pip cannot install', 'pyproject.toml'),
            ('package pip cannot install', 'setup.py'),
        ],
        ids=['dangling symlink', 'folder pip rejects', 'pyproject.toml', 'setup.py'],
    )
    def test_path_pip_cannot_install_exits_1_and_names_force(
        self, tmp_path, script, argv, repo, dist, kind, project_file
    ):
        """A dangling symlink, a folder with a file that pip rejects, or a package folder whose
        install fails (as on a network error) stays as it is: the stay line and the --force
        line are printed, pip fails, the exit status is 1 and the error names pip. Only a path
        without pyproject.toml and setup.py, which no pip install can use, gets the --force
        command after the error; a package folder does not, as its failure need not be the
        folder's. Nothing is cloned."""
        site = self._site(tmp_path, repo)
        path = site['root'] / repo
        if kind == 'dangling symlink':
            path.symlink_to(tmp_path / 'gone')
        else:
            path.mkdir()
            (path / (project_file or 'notes.txt')).write_text('mine\n')
            site['env']['STUB_PIP_FAIL'] = '1'
        result = _run_tool(site, script, *argv)
        force = f'bash tools/{script} {" ".join((*argv, "--force"))}'
        assert result.returncode == 1
        assert (
            f'{dist}: {path} stays as it is: not a git clone; PROTEUS expects release 26.4.22'
        ) in result.stdout.splitlines()
        assert f'        {force}' in result.stdout.splitlines()
        assert result.stderr.splitlines()[0] == (
            f'ERROR: pip could not install {path} (its message is above)'
        )
        assert (force in result.stderr) is (project_file is None)
        assert ('no pyproject.toml or setup.py' in result.stderr) is (project_file is None)
        assert site['log'].read_text().splitlines() == [f'install -U -e {path}']
        assert 'Done!' not in result.stdout
        assert 'Cloning' not in result.stdout
        if kind == 'dangling symlink':
            assert path.is_symlink() and os.readlink(path) == str(tmp_path / 'gone')
        else:
            assert sorted(p.name for p in path.iterdir()) == [project_file or 'notes.txt']

    def test_symlink_to_an_empty_folder_is_kept_and_pip_runs_on_it(
        self, tmp_path, script, argv, repo, dist
    ):
        """A symlink at the module path to an empty folder is a symlink, not an empty folder:
        it is kept, gets the stay line of a folder that is not a git clone, and pip runs on the
        link path. pip fails there (no project file), so the exit status is 1 with the --force
        command; the link and its empty target stay, and nothing is cloned."""
        site = self._site(tmp_path, repo)
        target = tmp_path / 'empty-target'
        target.mkdir()
        link = site['root'] / repo
        link.symlink_to(target)
        site['env']['STUB_PIP_FAIL'] = '1'
        result = _run_tool(site, script, *argv)
        assert result.returncode == 1
        assert (
            f'{dist}: {link} stays as it is: not a git clone; PROTEUS expects release 26.4.22'
        ) in result.stdout.splitlines()
        assert f'bash tools/{script} {" ".join((*argv, "--force"))}' in result.stderr
        assert site['log'].read_text().splitlines() == [f'install -U -e {link}']
        assert link.is_symlink() and os.readlink(link) == str(target)
        assert list(target.iterdir()) == []
        assert 'Cloning' not in result.stdout

    def test_symlink_to_a_clone_is_read_through_and_kept(
        self, tmp_path, script, argv, repo, dist
    ):
        """A symlink to a clone on a branch below the tag is kept: the stay line reads the
        branch through the link, the target is byte-equal and pip installs the link path. With
        --force only the link is replaced by a new clone at the tag; the target stays
        byte-equal."""
        site = self._site(tmp_path, repo)
        tagged, _ = site['shas'][repo]
        target = tmp_path / 'my-clones' / repo
        git(
            tmp_path,
            'clone',
            '-q',
            str(tmp_path / 'remotes' / 'FormingWorlds' / f'{repo}.git'),
            str(target),
        )
        git(target, 'checkout', '-q', '-b', 'old', '26.04.22~1')
        link = site['root'] / repo
        link.symlink_to(target)
        before = tree_snapshot(target)

        kept = _run_tool(site, script, *argv)
        assert kept.returncode == 0, kept.stdout + kept.stderr
        assert (
            f'{dist}: {link} stays as it is: on branch old, behind the tag;'
            ' PROTEUS expects the release tag 26.04.22'
        ) in kept.stdout.splitlines()
        assert link.is_symlink()
        assert tree_snapshot(target) == before
        assert site['log'].read_text().splitlines() == [f'install -U -e {link}']

        forced = _run_tool(site, script, *argv, '--force')
        assert forced.returncode == 0, forced.stdout + forced.stderr
        assert not link.is_symlink() and link.is_dir()
        assert git(link, 'rev-parse', 'HEAD') == tagged
        assert tree_snapshot(target) == before
        assert sorted(p.name for p in site['root'].iterdir()) == [
            repo,
            'pyproject.toml',
            'tools',
        ]

    def test_folder_that_appears_during_the_clone_stays(
        self, tmp_path, script, argv, repo, dist
    ):
        """A folder that appears at the module path while the new clone is made, here created
        by a git wrapper after its clone returns, stays with its file: exit 1 with the cause,
        no pip, and no temporary or old sibling is left."""
        site = self._site(tmp_path, repo)
        folder = site['root'] / repo
        _write_stub(
            tmp_path / 'stubs',
            'git',
            f'#!/bin/sh\n{shlex.quote(shutil.which("git"))} "$@"\nrc=$?\n'
            f'if [ "$1" = clone ]; then mkdir -p {shlex.quote(str(folder))}\n'
            f'    echo mine > {shlex.quote(str(folder / "mywork.txt"))}; fi\nexit $rc\n',
        )
        result = _run_tool(site, script, *argv)
        assert result.returncode == 1
        assert (
            f'ERROR: {folder} appeared while the new clone was made; it is unchanged'
            in result.stderr
        )
        assert sorted(p.name for p in folder.iterdir()) == ['mywork.txt']
        assert sorted(p.name for p in site['root'].iterdir()) == [
            repo,
            'pyproject.toml',
            'tools',
        ]
        assert site['log'].read_text() == ''


class TestReportCloneState:
    """The state line and the advice lines a module script prints for an existing folder."""

    @pytest.fixture
    def site(self, tmp_path):
        """A MORS remote with 26.10.06 on the second of 3 commits and a MORS clone on main."""
        pyproject = '[project]\nname = "fixture"\ndependencies = ["fwl-mors>=26.10.6"]\n'
        site = _floor_tag_site(tmp_path, pyproject, {'MORS': '26.10.06'})
        site['clone'] = site['root'] / 'MORS'
        git(
            site['root'],
            'clone',
            '-q',
            str(tmp_path / 'remotes' / 'FormingWorlds' / 'MORS.git'),
            str(site['clone']),
        )
        return site

    @staticmethod
    def _after(result: subprocess.CompletedProcess, prefix: str) -> list[str]:
        """The stdout lines from the one line that starts with ``prefix`` to ``Done!``."""
        lines = result.stdout.splitlines()
        start = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
        assert len(start) == 1, result.stdout
        return lines[start[0] : lines.index('Done!')]

    def test_branch_clone_behind_the_tag_gets_the_hand_commands_with_the_tag(self, site):
        """A clone on the branch old at the commit below the tag gets the stay line, then the
        fetch, the detached checkout of the full tag ref and the reinstall, in this order, and
        the --force command, each on its own indented line; the clone stays on old."""
        clone = site['clone']
        git(clone, 'checkout', '-q', '-b', 'old', '26.10.06~1')
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stderr
        assert self._after(result, 'fwl-mors: ') == [
            (
                f'fwl-mors: {clone} stays as it is: on branch old, behind the tag;'
                ' PROTEUS expects the release tag 26.10.06'
            ),
            '    From the PROTEUS folder, move it to the release tag by hand with:',
            f'        git -C {clone} fetch --tags',
            f'        git -C {clone} checkout --detach refs/tags/26.10.06',
            f'        pip install -e {clone}',
            '    Or replace it with a new clone, which deletes everything in it, with:',
            '        bash tools/get_python_module.sh MORS --force',
        ]
        assert git(clone, 'symbolic-ref', '--short', 'HEAD') == 'old'

    @pytest.mark.parametrize('where', ['branch', 'detached'])
    def test_clone_newer_than_the_tag_gets_one_line_and_no_summary(self, site, where):
        """The tip, one commit past the tag, on the branch main or detached, is newer than the
        release, which the >= floor allows: one line that names the branch or the commit, no
        hand commands, no --force line and no summary entry; the clone keeps its HEAD."""
        clone = site['clone']
        _, tip = site['shas']['MORS']
        if where == 'detached':
            git(clone, 'checkout', '-q', '--detach', tip)
        stays = site['root'].parent / 'stays.txt'
        site['env']['PROTEUS_CLONE_STAYS'] = str(stays)
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stderr
        found = 'branch main' if where == 'branch' else f'commit {tip[:7]}'
        assert self._after(result, 'fwl-mors: ') == [
            f'fwl-mors: {clone} is newer than the release tag 26.10.06 ({found})'
        ]
        assert 'fetch --tags' not in result.stdout
        assert '--force' not in result.stdout
        assert not stays.exists()
        assert git(clone, 'rev-parse', 'HEAD') == tip

    def test_commit_without_a_tag_before_it_is_named_once(self, site):
        """A clone detached at the first commit, which no tag precedes, is named by its short
        hash once: git describe gives the hash itself, which the line does not repeat."""
        clone = site['clone']
        git(clone, 'checkout', '-q', '--detach', '26.10.06~1')
        first = git(clone, 'rev-parse', 'HEAD')
        assert git(clone, 'describe', '--tags', '--always', 'HEAD') == first[:7]
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stderr
        assert self._after(result, 'fwl-mors: ')[0] == (
            f'fwl-mors: {clone} stays as it is: commit {first[:7]}, behind the tag;'
            ' PROTEUS expects the release tag 26.10.06'
        )
        assert f'({first[:7]})' not in result.stdout

    @pytest.mark.parametrize(
        'has_tag', [True, False], ids=['tag in the clone', 'tag after fetch']
    )
    def test_printed_commands_reach_the_tag_past_a_branch_of_its_name(self, site, has_tag):
        """The 3 printed commands, run from the PROTEUS folder in order, put HEAD detached on the
        tagged commit and reinstall the clone, although a branch named 26.10.06 and a branch
        named tags/26.10.06 point at the tip; ``checkout 26.10.06`` and ``checkout
        tags/26.10.06`` would switch to those branches.

        The clone is detached at the first commit, behind the tag. A clone that lacks the
        tag names the floor release and gets the checkout that picks the tag from the
        clone's tags, which the printed fetch brings.
        """
        clone = site['clone']
        tagged, tip = site['shas']['MORS']
        git(clone, 'checkout', '-q', '--detach', 'HEAD~2')
        first = git(clone, 'rev-parse', 'HEAD')
        git(clone, 'branch', '26.10.06', tip)
        git(clone, 'branch', 'tags/26.10.06', tip)
        if not has_tag:
            git(clone, 'tag', '-d', '26.10.06')
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stderr
        lines = self._after(result, 'fwl-mors: ')
        if has_tag:
            checkout = f'git -C {clone} checkout --detach refs/tags/26.10.06'
        else:
            assert lines[0] == (
                f'fwl-mors: {clone} stays as it is: commit {first[:7]};'
                ' PROTEUS expects release 26.10.6'
            )
            checkout = (
                f'git -C {clone} checkout --detach "refs/tags/$(git -C {clone} tag -l |'
                ' python tools/_module_pins.py fwl-mors floor-tag)"'
            )
        commands = [ln.strip() for ln in lines[2:5]]
        assert commands == [f'git -C {clone} fetch --tags', checkout, f'pip install -e {clone}']
        assert git(clone, 'rev-parse', 'HEAD') == first

        site['log'].write_text('')
        for command in commands:
            done = subprocess.run(
                ['bash', '-c', command],
                cwd=site['root'],
                env=site['env'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert done.returncode == 0, done.stderr
        assert git(clone, 'rev-parse', 'HEAD') == tagged
        assert tagged not in (tip, first)
        assert _is_detached(clone)
        assert site['log'].read_text().splitlines() == [f'install -e {clone}']

    def test_clone_git_cannot_read_gets_only_the_force_line(self, site):
        """A folder with an empty .git, in a PROTEUS root that is no git repository, is not a
        readable clone: the stay line is followed by the --force line alone, with no fetch,
        checkout or reinstall."""
        clone = site['clone']
        shutil.rmtree(clone)
        (clone / '.git').mkdir(parents=True)
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stderr
        assert self._after(result, 'fwl-mors: ') == [
            (
                f'fwl-mors: {clone} stays as it is: not a readable git clone;'
                ' PROTEUS expects release 26.10.6'
            ),
            (
                '    From the PROTEUS folder, replace it with a new clone, which deletes everything'
                ' in it, with:'
            ),
            '        bash tools/get_python_module.sh MORS --force',
        ]
        assert 'pip install -e' not in result.stdout

    def test_clone_on_another_line_of_history_says_so(self, site):
        """A commit off the first commit, which the tag does not contain and which does not
        contain the tag, is on a line of history without the tag; the summary says other
        history."""
        clone = site['clone']
        git(clone, 'checkout', '-q', '--detach', 'HEAD~2')
        (clone / 'version.txt').write_text('side\n')
        git(clone, 'commit', '-q', '-am', 'side')
        side = git(clone, 'rev-parse', 'HEAD')
        stays = site['root'].parent / 'stays.txt'
        site['env']['PROTEUS_CLONE_STAYS'] = str(stays)
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stderr
        assert (
            f'fwl-mors: {clone} stays as it is: commit {side[:7]},'
            ' on a line of history without the tag; PROTEUS expects the release tag 26.10.06'
        ) in result.stdout.splitlines()
        assert stays.read_text() == f'MORS (commit {side[:7]}, other history)\n'
        assert git(clone, 'rev-parse', 'HEAD') == side

    def test_folder_of_a_distribution_without_a_floor_names_no_release(self, site):
        """With no floor in pyproject.toml, the expected half says that the release tag could
        not be determined, and only the --force line follows for a plain folder."""
        plain = site['root'] / 'Other'
        plain.mkdir()
        snippet = (
            f'source {shlex.quote(str(site["root"] / "tools" / "_get_common.sh"))}\n'
            f'report_clone_state fwl-unpinned {shlex.quote(str(plain))} "get_other.sh"\n'
        )
        result = subprocess.run(
            ['bash', '-c', 'set -euo pipefail\n' + snippet],
            env=site['env'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            (
                f'fwl-unpinned: {plain} stays as it is: not a git clone;'
                ' the release tag could not be determined'
            ),
            (
                '    From the PROTEUS folder, replace it with a new clone, which deletes everything'
                ' in it, with:'
            ),
            '        bash tools/get_other.sh --force',
        ]

    def test_vulcan_builds_fastchem_in_the_kept_clone_before_pip(self, tmp_path):
        """A VULCAN clone on a branch is kept, make runs in its fastchem_vulcan folder, and pip
        installs it after the build; a failing build stops before pip with exit 1."""
        pyproject = (
            '[project]\nname = "fixture"\ndependencies = []\n'
            '[project.optional-dependencies]\nvulcan = ["fwl-vulcan>=26.4.22"]\n'
        )
        site = _floor_tag_site(tmp_path, pyproject, {'VULCAN': '26.04.22'})
        clone = site['root'] / 'VULCAN'
        git(
            site['root'],
            'clone',
            '-q',
            str(tmp_path / 'remotes' / 'FormingWorlds' / 'VULCAN.git'),
            str(clone),
        )
        order = site['log']
        _write_stub(
            tmp_path / 'stubs', 'make', f'#!/bin/sh\necho "make in $(pwd -P)" >> {order}\n'
        )
        result = _run_tool(site, 'get_vulcan.sh')
        assert result.returncode == 0, result.stdout + result.stderr
        assert order.read_text().splitlines() == [
            f'make in {clone}/fastchem_vulcan',
            f'install -U -e {clone}',
        ]
        assert f'fwl-vulcan: {clone} is newer than the release tag 26.04.22 (branch main)' in (
            result.stdout.splitlines()
        )
        assert f'fwl-vulcan: the build step ran in the existing folder {clone}' in result.stdout

        order.write_text('')
        _write_stub(tmp_path / 'stubs', 'make', '#!/bin/sh\nexit 2\n')
        failed = _run_tool(site, 'get_vulcan.sh')
        assert failed.returncode == 1
        assert 'ERROR: fastchem build failed' in failed.stderr
        assert order.read_text() == ''


def _manual_tool_commands() -> list[list[str]]:
    """The words of every ``bash tools/<script>`` command line in the manual install page."""
    return [
        line.split()[1:]
        for line in MANUAL_INSTALL_MD.read_text(encoding='utf-8').splitlines()
        if re.match(r'^bash tools/\S+\.sh\b', line.strip())
    ]


def _accepted_module_names() -> set[str]:
    """The module names tools/get_python_module.sh accepts, read from its ``modules=`` line."""
    match = re.search(r'^modules="([^"]*)"$', PYTHON_MODULE_SCRIPT.read_text(), re.MULTILINE)
    assert match, 'no modules= line in tools/get_python_module.sh'
    return set(match.group(1).split())


def test_manual_install_page_names_existing_scripts_and_modules():
    """Each ``bash tools/...`` command in the manual page names a shipped script.

    A ``get_python_module.sh`` command names a module the script accepts,
    and the page sets up each of those modules, so a renamed script or
    module, or one left out, fails here.
    """
    commands = _manual_tool_commands()
    scripts = {words[0] for words in commands}
    # Guard the guard: the page holds the per-module commands and the setup scripts.
    assert {
        'tools/get_python_module.sh',
        'tools/get_zalmoxis.sh',
        'tools/get_cvode.sh',
    } <= scripts
    missing = sorted(s for s in scripts if not (TOOLS_DIR.parent / s).is_file())
    assert missing == []

    accepted = _accepted_module_names()
    named = [words[1] for words in commands if words[0] == 'tools/get_python_module.sh']
    assert all(
        len(words) == 2 for words in commands if words[0] == 'tools/get_python_module.sh'
    )
    assert set(named) == accepted
    assert len(named) == len(accepted)


# ---------------------------------------------------------------------------
# Portable-flag rewrite and guards (tools/get_socrates.sh)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The Julia PROTEUS uses: use_proteus_julia and the setup scripts that run julia
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize('strict', [True, False], ids=['strict', 'lenient'])
@pytest.mark.parametrize('kind', ['unset', 'empty'], ids=str)
def test_use_proteus_julia_changes_nothing_without_the_variable(tmp_path, kind, strict):
    """With PYTHON_JULIAPKG_EXE unset or empty, PATH and stdout stay as they are, status 0.

    The unset case runs under ``set -u`` as well, where a bare expansion of
    the variable would stop the script.
    """
    env = {k: v for k, v in os.environ.items() if k != 'PYTHON_JULIAPKG_EXE'}
    env['PATH'] = '/usr/bin:/bin'
    if kind == 'empty':
        env['PYTHON_JULIAPKG_EXE'] = ''
    res = _run_bash(
        _with_common('use_proteus_julia\nprintf "PATH=%s\\n" "$PATH"\n', strict=strict),
        env=env,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == ['PATH=/usr/bin:/bin']
    assert res.stderr == ''


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['missing', 'not-executable', 'directory'], ids=str)
def test_use_proteus_julia_stops_on_a_variable_that_names_no_executable_file(tmp_path, kind):
    """A set PYTHON_JULIAPKG_EXE that names no executable file returns 1 with one error line.

    juliacall fails at import with such a value, so the default Julia on PATH
    must not stand in for it; PATH stays as it is and nothing is linked.
    """
    target = {
        'missing': str(tmp_path / 'none' / 'julia'),
        'not-executable': str(_write_stub(tmp_path, 'julia', '#!/bin/sh\n')),
        'directory': str(tmp_path),
    }[kind]
    if kind == 'not-executable':
        Path(target).chmod(0o644)
    env = {**os.environ, 'PATH': '/usr/bin:/bin', 'PYTHON_JULIAPKG_EXE': target}
    res = _run_bash(
        _with_common('use_proteus_julia || echo "STATUS=$?"\nprintf "PATH=%s\\n" "$PATH"\n'),
        env=env,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == ['STATUS=1', 'PATH=/usr/bin:/bin']
    assert res.stderr.splitlines() == [
        f'ERROR: PYTHON_JULIAPKG_EXE={target} is not an executable file; point it at the '
        'Julia PROTEUS uses, or unset it to use the Julia on PATH.'
    ]


def _common_copy(tmp_path: Path) -> Path:
    """Copy ``tools/_get_common.sh`` into ``tmp_path/proteus/tools``; return the copy.

    Sourced there, its ``proteus_root`` is ``tmp_path/proteus``, so the
    ``.proteus-julia`` folder that use_proteus_julia makes lands under ``tmp_path``.
    """
    tools = tmp_path / 'proteus' / 'tools'
    tools.mkdir(parents=True)
    shutil.copy(COMMON_LIB, tools / '_get_common.sh')
    return tools / '_get_common.sh'


@pytest.mark.unit
@pytest.mark.parametrize('strict', [True, False], ids=['strict', 'lenient'])
def test_use_proteus_julia_puts_the_bound_julia_first_on_path(tmp_path, strict):
    """An executable PYTHON_JULIAPKG_EXE is a bare julia, through the folder .proteus-julia.

    PATH holds a second julia, which a bare julia would find without the
    helper; the helper prints one line that names the bound binary, and the
    folder first on PATH holds the one link julia to it.
    """
    common = _common_copy(tmp_path)
    other = tmp_path / 'default' / 'bin'
    other.mkdir(parents=True)
    _write_stub(other, 'julia', '#!/bin/sh\necho default\n')
    bound_dir = tmp_path / 'julia-1.13.1' / 'bin'
    bound_dir.mkdir(parents=True)
    bound = _write_stub(bound_dir, 'julia', '#!/bin/sh\necho bound\n')
    env = {**os.environ, 'PATH': f'{other}:/usr/bin:/bin', 'PYTHON_JULIAPKG_EXE': str(bound)}
    prelude = 'set -euo pipefail\n' if strict else ''
    res = _run_bash(
        f'{prelude}source "{common}"\nuse_proteus_julia\nprintf "PATH=%s\\n" "$PATH"\njulia\n',
        env=env,
    )
    assert res.returncode == 0, res.stderr
    link_dir = os.path.realpath(tmp_path / 'proteus') + '/.proteus-julia'
    assert res.stdout.splitlines() == [
        f'Julia: {bound} (PYTHON_JULIAPKG_EXE)',
        f'PATH={link_dir}:{other}:/usr/bin:/bin',
        'bound',
    ]
    assert os.listdir(link_dir) == ['julia']
    assert os.path.realpath(Path(link_dir) / 'julia') == os.path.realpath(bound)


@pytest.mark.unit
def test_use_proteus_julia_shadows_no_other_program_of_the_binary_folder(tmp_path):
    """A bound julia-1.13 in a folder with a python: bare julia is it, python stays the env's.

    The folder of the binary (a shared one such as /opt/homebrew/bin) also holds
    a python, and the binary is not named julia; a bare julia on PATH is another
    version. After the helper, julia runs the bound binary and python the env's.
    A second call with another binary moves the link.
    """
    common = _common_copy(tmp_path)
    shared = tmp_path / 'shared' / 'bin'
    shared.mkdir(parents=True)
    bound = _write_stub(shared, 'julia-1.13', '#!/bin/sh\necho bound-1.13\n')
    _write_stub(shared, 'python', '#!/bin/sh\necho shared-python\n')
    env_bin = tmp_path / 'env' / 'bin'
    env_bin.mkdir(parents=True)
    _write_stub(env_bin, 'python', '#!/bin/sh\necho env-python\n')
    _write_stub(env_bin, 'julia', '#!/bin/sh\necho path-julia-1.10\n')
    newer = _write_stub(tmp_path / 'shared', 'julia-1.13.2', '#!/bin/sh\necho bound-1.13.2\n')
    env = {**os.environ, 'PATH': f'{env_bin}:/usr/bin:/bin', 'PYTHON_JULIAPKG_EXE': str(bound)}
    script = (
        f'set -euo pipefail\nsource "{common}"\n'
        'use_proteus_julia >/dev/null\njulia\npython\n'
        f'PYTHON_JULIAPKG_EXE={shlex.quote(str(newer))}\n'
        'use_proteus_julia >/dev/null\njulia\n'
    )
    res = _run_bash(script, env=env)
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == ['bound-1.13', 'env-python', 'bound-1.13.2']
    assert os.listdir(tmp_path / 'proteus' / '.proteus-julia') == ['julia']


def _julia_site(tmp_path: Path, bound: bool) -> dict:
    """A PROTEUS copy, local AGNI and Obliqua remotes and two logging julia stubs.

    The julia on PATH logs ``path-julia <args>``, writes a marker file and
    exits ``PATH_JULIA_RC`` (1 when ``bound``, else 0). With ``bound``,
    PYTHON_JULIAPKG_EXE names a second stub in another folder that logs
    ``bound-julia <args>`` and exits 0; without it the variable is unset.
    The AGNI remote carries a ``src/get_agni.sh`` that calls julia once, as
    AGNI's own script does, so the run shows which julia a child script finds.
    """
    root = _fake_checkout(tmp_path)
    remotes = tmp_path / 'remotes'
    remotes.mkdir()
    for repo, files in (
        ('AGNI', {'src/get_agni.sh': 'julia --project=. -e "instantiate $1"\n'}),
        ('Obliqua', {'Project.toml': 'name = "Obliqua"\n'}),
    ):
        work = tmp_path / 'work' / repo
        for name, text in files.items():
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            (work / name).write_text(text)
        git(work, 'init', '-q', '-b', 'main')
        git(work, 'add', '.')
        git(work, 'commit', '-q', '-m', 'initial')
        git(work, 'tag', 'v1.0.0')
        git(remotes, 'clone', '-q', '--bare', str(work), str(remotes / f'{repo}.git'))
    log = tmp_path / 'julia.log'
    log.write_text('')
    marker = tmp_path / 'path-julia-ran'
    stubs = tmp_path / 'stubs'
    stubs.mkdir()
    _write_stub(
        stubs,
        'julia',
        '#!/bin/sh\necho "path-julia $*" >> "$STUB_LOG"\n'
        ': > "$STUB_MARKER"\nexit "$PATH_JULIA_RC"\n',
    )
    env = {k: v for k, v in os.environ.items() if k != 'PYTHON_JULIAPKG_EXE'}
    env.update(
        {
            'PATH': f'{stubs}:{os.environ["PATH"]}',
            'STUB_LOG': str(log),
            'STUB_MARKER': str(marker),
            'PATH_JULIA_RC': '1' if bound else '0',
            'AGNI_GIT_URL': str(remotes / 'AGNI.git'),
            'AGNI_GIT_REF': 'v1.0.0',
            'OBLIQUA_GIT_URL': str(remotes / 'Obliqua.git'),
            'OBLIQUA_GIT_REF': 'v1.0.0',
            'GIT_CEILING_DIRECTORIES': str(tmp_path),
        }
    )
    exe = None
    if bound:
        bound_dir = tmp_path / 'juliaup' / 'julia-1.13.1' / 'bin'
        bound_dir.mkdir(parents=True)
        exe = _write_stub(
            bound_dir, 'julia', '#!/bin/sh\necho "bound-julia $*" >> "$STUB_LOG"\nexit 0\n'
        )
        env['PYTHON_JULIAPKG_EXE'] = str(exe)
    return {'root': root, 'env': env, 'log': log, 'marker': marker, 'exe': exe}


# The julia calls each script makes with the argument 0, in order; LovePy's
# Pkg.add names the URL and ref of its pin, so only its prefix is pinned.
_JULIA_CALLS = {
    'get_agni.sh': ['--project=. -e instantiate 0'],
    'get_obliqua.sh': [
        '--project=. -e using Pkg; Pkg.resolve(); Pkg.instantiate()',
    ],
    'get_lovepy.sh': [
        '-e using Pkg; Pkg.add(url="https://github.com/nichollsh/LovePy", rev="',
        '-e using LovePy; println("Installed to: "*pathof(LovePy))',
    ],
}


def _run_julia_script(site: dict, script: str) -> subprocess.CompletedProcess:
    """Run ``tools/<script> 0`` of the site copy, from outside the copy."""
    return subprocess.run(
        ['bash', str(site['root'] / 'tools' / script), '0'],
        cwd=site['root'].parent,
        env=site['env'],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _julia_calls(site: dict, who: str) -> list[str]:
    """The arguments of each logged call of the ``who`` julia, in order."""
    lines = site['log'].read_text().splitlines()
    return [ln.partition(' ')[2] for ln in lines if ln.partition(' ')[0] == who]


def _matches(calls: list[str], expected: list[str]) -> bool:
    """Whether each call equals its expected text, or starts with it for the Pkg.add of LovePy."""
    return len(calls) == len(expected) and all(
        c == e or (e.endswith('rev="') and c.startswith(e))
        for c, e in zip(calls, expected, strict=True)
    )


@pytest.mark.unit
@pytest.mark.parametrize('script', sorted(_JULIA_CALLS))
def test_setup_script_runs_the_bound_julia_and_never_the_path_one(tmp_path, script):
    """With PYTHON_JULIAPKG_EXE set, every julia call of the script reaches that binary.

    The julia on PATH would exit 1 and leave a marker file; neither happens,
    the script finishes, and the bound stub got each call in order. For
    AGNI the call comes from the cloned ``src/get_agni.sh``, a child script,
    so the PATH the helper sets reaches it as well.
    """
    site = _julia_site(tmp_path, bound=True)
    res = _run_julia_script(site, script)
    assert res.returncode == 0, res.stdout + res.stderr
    assert not site['marker'].exists(), site['log'].read_text()
    assert _julia_calls(site, 'path-julia') == []
    calls = _julia_calls(site, 'bound-julia')
    assert _matches(calls, _JULIA_CALLS[script]), calls
    assert res.stdout.splitlines().count(f'Julia: {site["exe"]} (PYTHON_JULIAPKG_EXE)') == 1


@pytest.mark.unit
@pytest.mark.parametrize('script', sorted(_JULIA_CALLS))
def test_setup_script_with_an_unusable_bound_julia_stops_before_any_julia(tmp_path, script):
    """A PYTHON_JULIAPKG_EXE that names a missing file stops the script with status 1.

    The julia on PATH, another version, would install into the wrong Julia; it
    runs no call, and the error names the variable. get_lovepy.sh has no
    ``set -e``, so it stops through ``use_proteus_julia || exit 1``.
    """
    site = _julia_site(tmp_path, bound=False)
    missing = tmp_path / 'juliaup' / 'gone' / 'julia'
    site['env']['PYTHON_JULIAPKG_EXE'] = str(missing)
    res = _run_julia_script(site, script)
    assert res.returncode == 1, res.stdout + res.stderr
    assert not site['marker'].exists()
    assert site['log'].read_text() == ''
    assert f'ERROR: PYTHON_JULIAPKG_EXE={missing} is not an executable file' in res.stderr


@pytest.mark.unit
@pytest.mark.parametrize('script', sorted(_JULIA_CALLS))
def test_setup_script_without_a_bound_julia_runs_the_path_one(tmp_path, script):
    """Without PYTHON_JULIAPKG_EXE the script runs the julia on PATH and names no bound Julia."""
    site = _julia_site(tmp_path, bound=False)
    res = _run_julia_script(site, script)
    assert res.returncode == 0, res.stdout + res.stderr
    assert site['marker'].exists()
    calls = _julia_calls(site, 'path-julia')
    assert _matches(calls, _JULIA_CALLS[script]), calls
    assert 'PYTHON_JULIAPKG_EXE' not in res.stdout + res.stderr


@pytest.mark.unit
def test_every_setup_script_that_runs_julia_calls_use_proteus_julia_first():
    """Each tools script whose code runs julia calls use_proteus_julia before its first julia.

    get_agni.sh counts through the ``src/get_agni.sh`` of AGNI that it runs;
    test_docker_image.sh runs julia inside the container, not on the host.
    """
    # Guard the guard: a renamed helper would make every script fail the same way.
    assert 'use_proteus_julia' in _library_function_names()
    runs_julia = {}
    for script in sorted(TOOLS_DIR.glob('*.sh')):
        lines = _code_lines(script)
        julia = [i for i, code in lines if re.search(r'(^|[\s(=])julia(\s|$)', code)]
        julia += [i for i, code in lines if 'src/get_agni.sh' in code]
        if script.name == 'test_docker_image.sh' or not julia:
            continue
        helper = [
            i
            for i, code in lines
            if code in ('use_proteus_julia', 'use_proteus_julia || exit 1')
        ]
        runs_julia[script.name] = bool(helper) and helper[0] < min(julia)
    assert runs_julia == {'get_agni.sh': True, 'get_lovepy.sh': True, 'get_obliqua.sh': True}


# The pages whose commands set a variable in the env `proteus`, which manual_installation.md makes.
_CONDA_VAR_PAGES = (
    'manual_installation.md',
    'optionalmodules_installation.md',
    'troubleshooting.md',
)


@pytest.mark.unit
def test_printed_and_documented_conda_var_commands_name_their_env(tmp_path):
    """Each `conda env config vars set` that a script prints or a page gives names its env.

    Without `-n` or `-p`, conda writes to the active env, which is base when the
    user types the command in a new shell. With an env active, a script prints
    the literal `-p "$CONDA_PREFIX"`, not the path of the env it ran in; with no
    env active it prints `-n <your environment>`, because an empty prefix is an
    error. The pages name `-n proteus`. A scan that finds no line fails.
    """
    printed = {}
    for active in (True, False):
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path)}
        if active:
            env['CONDA_PREFIX'] = '/envs/where-the-script-ran'
        for script in sorted(TOOLS_DIR.glob('get_*.sh')):
            lines = script.read_text().splitlines()
            for i, line in enumerate(lines):
                if 'conda env config vars set' in line and line.lstrip().startswith('echo '):
                    assert lines[i - 1].startswith("env_opt='-p "), script.name
                    out = subprocess.run(
                        ['bash', '-c', lines[i - 1] + '\n' + line],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=10,
                    )
                    printed[(script.name, active)] = out.stdout.strip()
    assert sorted(name for name, _ in printed) == sorted(
        ['get_lavatmos.sh', 'get_petsc.sh', 'get_socrates.sh'] * 2
    )
    for (name, active), text in printed.items():
        option = '-p "$CONDA_PREFIX" ' if active else '-n <your environment> '
        assert text.startswith('conda env config vars set ' + option), (name, active, text)
        assert 'where-the-script-ran' not in text
    docs = TOOLS_DIR.parent / 'docs' / 'How-to'
    commands = [
        (page, line)
        for page in _CONDA_VAR_PAGES
        for line in (docs / page).read_text().splitlines()
        if line.startswith('conda env config vars set')
    ]
    assert len(commands) == 6
    named = 'conda env config vars set -n proteus '
    assert [c for c in commands if not c[1].startswith(named)] == []


def _pinned_module_site(
    tmp_path: Path, repo: str, files: dict[str, str], pyproject: str
) -> dict:
    """A ``_floor_tag_site`` for a module that ``pyproject`` pins by commit.

    ``files`` are the files of the remote; ``@REF@`` in ``pyproject`` stands for the
    pinned commit, the second of the 3 commits of the remote, and the tip is the third.
    The remote has no tag, so ``git describe`` names a commit by its hash.
    """
    site = _floor_tag_site(tmp_path, '', {repo: 'unused'}, files)
    target, tip = site['shas'][repo]
    remotes = tmp_path / 'remotes' / 'FormingWorlds'
    git(remotes / f'{repo}.git', 'tag', '-d', 'unused')
    (site['root'] / 'pyproject.toml').write_text(pyproject.replace('@REF@', target))
    return {
        **site,
        'target_sha': target,
        'tip_sha': tip,
        'stubs': tmp_path / 'stubs',
        'remotes': remotes,
    }


def _atmodeller_site(tmp_path: Path) -> dict:
    """A pinned-commit site for atmodeller, with the pip stub of ``_floor_tag_site``."""
    pyproject = (
        '[project]\nname = "fixture"\ndependencies = []\n'
        '[project.optional-dependencies]\natmodeller = ["atmodeller>=1.0.2"]\n'
        '[tool.proteus.modules.atmodeller]\n'
        'url = "https://github.com/FormingWorlds/atmodeller.git"\n'
        'ref = "@REF@"\n'
    )
    files = {'pyproject.toml': '[project]\nname = "atmodeller"\nversion = "1.0.2"\n'}
    return _pinned_module_site(tmp_path, 'atmodeller', files, pyproject)


def _obliqua_site(tmp_path: Path) -> dict:
    """A pinned-commit site for Obliqua; its julia stub fails on ``FAIL_JULIA_PATTERN``."""
    pyproject = (
        '[project]\nname = "fixture"\ndependencies = []\n'
        '[tool.proteus.modules.obliqua]\n'
        'url = "https://github.com/FormingWorlds/Obliqua.git"\n'
        'ref = "@REF@"\n'
    )
    files = {
        'Project.toml': 'name = "Obliqua"\nuuid = "12345678-1234-1234-1234-123456789abc"\n',
    }
    site = _pinned_module_site(tmp_path, 'Obliqua', files, pyproject)
    log = tmp_path / 'julia.log'
    log.write_text('')
    _write_stub(
        site['stubs'],
        'julia',
        '#!/bin/sh\n'
        'echo "$@" >> "$STUB_JULIA_LOG"\n'
        'if [ ! -f "Project.toml" ]; then\n'
        '    echo "ERROR: no Project.toml in $PWD" >&2\n'
        '    exit 1\n'
        'fi\n'
        'case "$*" in\n'
        '    *"Pkg.instantiate"*) [ -f "Manifest.toml" ] || echo "# Manifest generated" > Manifest.toml ;;\n'
        'esac\n'
        'if [ -n "${FAIL_JULIA_PATTERN:-}" ]; then\n'
        '    case "$*" in\n'
        '        *"$FAIL_JULIA_PATTERN"*) exit 1 ;;\n'
        '    esac\n'
        'fi\n'
        'exit 0\n',
    )
    site['env'].pop('PYTHON_JULIAPKG_EXE', None)
    site['env']['STUB_JULIA_LOG'] = str(log)
    return {**site, 'log': log, 'pip_log': tmp_path / 'pip.log'}


@pytest.mark.unit
def test_atmodeller_single_pin_in_pyproject():
    """Verify the atmodeller extra floor and the pinned url and ref in pyproject.toml."""
    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / 'pyproject.toml').read_text(encoding='utf-8'))
    deps = data['project']['dependencies']
    assert not any('atmodeller' in d.lower() for d in deps), (
        'atmodeller is not a mandatory dependency'
    )
    extras = data['project']['optional-dependencies']
    assert any(r.startswith('atmodeller>=') for r in extras.get('atmodeller', [])), (
        'atmodeller extra keeps floor'
    )
    table = data['tool']['proteus']['modules']['atmodeller']
    assert table['url'] == 'https://github.com/FormingWorlds/atmodeller.git'
    assert re.fullmatch(r'[0-9a-f]{40}', table['ref'])


@pytest.mark.unit
def test_atmodeller_new_clone_and_editable_install(tmp_path):
    """Verify new clone lands at pinned commit and installs editable in Atmodeller."""
    site = _atmodeller_site(tmp_path)
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    clone = site['root'] / 'Atmodeller'
    assert clone.is_dir()
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']
    pip_log = site['log'].read_text()
    assert f'install -U -e {clone}' in pip_log


@pytest.mark.unit
def test_atmodeller_existing_folder_at_pin_stays_and_installs(tmp_path):
    """Verify an existing folder at the pinned commit stays and is reinstalled."""
    site = _atmodeller_site(tmp_path)
    assert _run_tool(site, 'get_atmodeller.sh').returncode == 0
    clone = site['root'] / 'Atmodeller'
    (clone / 'local_note.txt').write_text('keep me\n')
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert f'atmodeller: {clone} is at the pinned commit {site["target_sha"][:8]}' in res.stdout
    assert (clone / 'local_note.txt').read_text() == 'keep me\n'
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']


@pytest.mark.unit
def test_atmodeller_existing_folder_off_pin_stays_and_advises(tmp_path):
    """Verify an existing folder off-pin stays and advice names both commands."""
    site = _atmodeller_site(tmp_path)
    assert _run_tool(site, 'get_atmodeller.sh').returncode == 0
    clone = site['root'] / 'Atmodeller'
    git(clone, 'checkout', '-q', 'main')
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    target_short = site['target_sha'][:8]
    assert (
        f'atmodeller: {clone} stays as it is: commit {site["tip_sha"][:7]}; not at the pinned commit {target_short}'
        in res.stdout
    )
    lines = [ln.strip() for ln in res.stdout.splitlines()]
    idx_fetch = lines.index(f'git -C {clone} fetch origin')
    idx_checkout = lines.index(f'git -C {clone} checkout --detach {site["target_sha"]}')
    idx_pip = lines.index(f'pip install -e {clone}')
    idx_force = lines.index('bash tools/get_atmodeller.sh --force')
    assert idx_fetch < idx_checkout < idx_pip < idx_force
    assert not any(ln == 'bash tools/get_atmodeller.sh' for ln in lines)
    assert git(clone, 'rev-parse', 'HEAD') == site['tip_sha']


@pytest.mark.unit
def test_atmodeller_worktree_off_pin_stays_and_advises_force(tmp_path):
    """Verify an off-pin atmodeller git worktree stays and advice names the forced command."""
    site = _atmodeller_site(tmp_path)
    assert _run_tool(site, 'get_atmodeller.sh').returncode == 0
    folder = site['root'] / 'Atmodeller'
    base = tmp_path / 'base_clone'
    shutil.move(str(folder), str(base))
    git(base, 'worktree', 'add', '-q', '--detach', str(folder), site['tip_sha'])
    assert (folder / '.git').is_file()
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    lines = [ln.strip() for ln in res.stdout.splitlines()]
    idx_fetch = lines.index(f'git -C {folder} fetch origin')
    idx_checkout = lines.index(f'git -C {folder} checkout --detach {site["target_sha"]}')
    idx_pip = lines.index(f'pip install -e {folder}')
    idx_force = lines.index('bash tools/get_atmodeller.sh --force')
    assert idx_fetch < idx_checkout < idx_pip < idx_force
    assert 'Or move it away or delete it by hand, then run:' not in res.stdout
    assert git(folder, 'rev-parse', 'HEAD') == site['tip_sha']


_MODULE_SITES = (
    ('get_atmodeller.sh', _atmodeller_site, 'Atmodeller', 'atmodeller.git'),
    ('get_obliqua.sh', _obliqua_site, 'Obliqua', 'Obliqua.git'),
)


def _assert_read_only_git_calls(log_path: Path, folder: Path) -> None:
    """Assert all git calls recorded in log_path on folder are strictly read-only."""
    allowed = {
        'rev-parse',
        'status',
        'symbolic-ref',
        'describe',
        'merge-base',
        'log',
        'show',
        'for-each-ref',
        'tag',
    }
    forbidden = {'fetch', 'pull', 'checkout', 'reset', 'merge', 'clean'}
    calls = log_path.read_text().splitlines() if log_path.exists() else []
    assert len(calls) > 0, 'expected at least one git call'
    folder_real = str(os.path.realpath(folder))
    folder_calls = 0
    for line in calls:
        if line.startswith('BLOCKED\t'):
            parts = line.split('\t')
            target = parts[1] if len(parts) > 1 else ''
            args = parts[2].split() if len(parts) > 2 else []
        else:
            parts = line.split('\t')
            target = parts[0]
            args = parts[1].split() if len(parts) > 1 else []
        i = 0
        while i < len(args):
            if args[i] == '-C' and i + 1 < len(args):
                target = args[i + 1]
                i += 2
            elif args[i] in ('-c', '--git-dir', '--work-tree', '--namespace') and i + 1 < len(
                args
            ):
                i += 2
            elif args[i].startswith('-'):
                i += 1
            else:
                break
        sub = args[i] if i < len(args) else ''
        rest = args[i + 1 :]
        target_real = str(os.path.realpath(target))
        if target_real == folder_real or target_real.startswith(folder_real + '/'):
            folder_calls += 1
            assert sub not in forbidden, f'Forbidden write git command on {folder}: {line}'
            assert sub in allowed, f'Unexpected git command on {folder}: {line}'
            if sub == 'tag':
                assert any(opt in rest for opt in ('-l', '--list')), (
                    f'git tag without list option on {folder}: {line}'
                )
    assert folder_calls > 0, f'no git calls found targeting {folder}'


@pytest.mark.unit
@pytest.mark.parametrize(
    ('script', 'make_site', 'folder', 'remote'), _MODULE_SITES, ids=('atmodeller', 'Obliqua')
)
def test_module_existing_folder_gets_no_writing_git_commands(
    tmp_path, script, make_site, folder, remote
):
    """Verify an existing module folder receives no modifying git commands."""
    site = make_site(tmp_path)
    assert _run_tool(site, script).returncode == 0
    clone = site['root'] / folder
    guard = tmp_path / 'guard'
    guard.mkdir()
    log = tmp_path / 'git-calls.log'
    marker = tmp_path / 'git-write.marker'
    install_git_guard(guard, log, marker, roots=[site['root']], trees=[clone])
    site['env']['PATH'] = f'{guard}{os.pathsep}{site["env"]["PATH"]}'
    res = _run_tool(site, script)
    assert res.returncode == 0, res.stdout + res.stderr
    assert not marker.exists(), f'Write git command called: {log.read_text()}'
    assert clone.is_dir()
    if folder == 'Obliqua':
        assert (clone / 'Manifest.toml').exists()
    _assert_read_only_git_calls(log, clone)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('script', 'make_site', 'folder', 'remote'), _MODULE_SITES, ids=('atmodeller', 'Obliqua')
)
def test_module_failed_clone_leaves_existing_folder_unchanged(
    tmp_path, script, make_site, folder, remote
):
    """Verify failing clone exits 1 and leaves existing folder unchanged."""
    site = make_site(tmp_path)
    assert _run_tool(site, script).returncode == 0
    clone = site['root'] / folder
    (clone / 'canary.txt').write_text('original\n')
    before = tree_snapshot(clone)
    (site['remotes'] / remote).rename(site['remotes'] / (remote + '.moved'))
    res = _run_tool(site, script, '--force')
    assert res.returncode == 1
    assert 'ERROR: git clone failed' in res.stderr
    assert 'git checkout failed' not in res.stderr
    assert tree_snapshot(clone) == before
    assert not any(p.name.startswith(f'{folder}.tmp.') for p in site['root'].iterdir())


@pytest.mark.unit
@pytest.mark.parametrize(
    ('script', 'make_site', 'folder', 'remote'), _MODULE_SITES, ids=('atmodeller', 'Obliqua')
)
def test_module_failed_checkout_leaves_existing_folder_unchanged(
    tmp_path, script, make_site, folder, remote
):
    """Verify failing checkout exits 1 and leaves existing folder unchanged."""
    site = make_site(tmp_path)
    assert _run_tool(site, script).returncode == 0
    clone = site['root'] / folder
    before = tree_snapshot(clone)
    pyproject = site['root'] / 'pyproject.toml'
    pyproject.write_text(pyproject.read_text().replace(site['target_sha'], '0' * 40))
    res = _run_tool(site, script, '--force')
    assert res.returncode == 1
    assert 'ERROR: git checkout failed' in res.stderr
    assert tree_snapshot(clone) == before
    assert not any(p.name.startswith(f'{folder}.tmp.') for p in site['root'].iterdir())


@pytest.mark.unit
def test_atmodeller_failed_pip_exits_1(tmp_path):
    """Verify failing pip exits 1 and reports error."""
    site = _atmodeller_site(tmp_path)
    _write_stub(site['stubs'], 'pip', '#!/bin/sh\nexit 1\n')
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 1
    assert 'ERROR: editable install failed' in res.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    ('script', 'make_site', 'folder', 'remote'), _MODULE_SITES, ids=('atmodeller', 'Obliqua')
)
def test_module_force_replaces_existing_folder(tmp_path, script, make_site, folder, remote):
    """Verify --force replaces off-pin folder with a fresh clone at the pin."""
    site = make_site(tmp_path)
    assert _run_tool(site, script).returncode == 0
    clone = site['root'] / folder
    git(clone, 'checkout', '-q', 'main')
    (clone / 'untracked.txt').write_text('delete me\n')
    res = _run_tool(site, script, '--force')
    assert res.returncode == 0, res.stdout + res.stderr
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']
    assert not (clone / 'untracked.txt').exists()


@pytest.mark.unit
def test_atmodeller_script_holds_no_hardcoded_url_or_commit():
    """Verify get_atmodeller.sh contains no hardcoded URL or commit hash."""
    repo_root = Path(__file__).resolve().parents[2]
    script = (repo_root / 'tools' / 'get_atmodeller.sh').read_text(encoding='utf-8')
    assert 'https://' not in script
    assert 'git@github.com' not in script
    assert not re.search(r'[0-9a-f]{40}', script)
    pyproject = tomllib.loads((repo_root / 'pyproject.toml').read_text(encoding='utf-8'))
    ref = pyproject['tool']['proteus']['modules']['atmodeller']['ref']
    assert 'resolve_module_pin atmodeller' in script
    assert ref[:7] not in script


@pytest.mark.unit
def test_atmodeller_existing_folder_missing_pin_commit_advises(tmp_path):
    """Verify an existing folder missing the pin commit stays and advice is printed."""
    site = _atmodeller_site(tmp_path)
    assert _run_tool(site, 'get_atmodeller.sh').returncode == 0
    clone = site['root'] / 'Atmodeller'
    pyproject = site['root'] / 'pyproject.toml'
    fake_sha = 'a' * 40
    pyproject.write_text(pyproject.read_text().replace(site['target_sha'], fake_sha))
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert f'not at the pinned commit {fake_sha[:8]}' in res.stdout
    assert f'git -C {clone} checkout --detach {fake_sha}' in res.stdout
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']


@pytest.mark.unit
def test_atmodeller_existing_folder_abbreviated_pin_resolves_and_installs(tmp_path):
    """Verify an abbreviated pin hash matching HEAD resolves and installs."""
    site = _atmodeller_site(tmp_path)
    assert _run_tool(site, 'get_atmodeller.sh').returncode == 0
    clone = site['root'] / 'Atmodeller'
    short_pin = site['target_sha'][:12]
    pyproject = site['root'] / 'pyproject.toml'
    pyproject.write_text(pyproject.read_text().replace(site['target_sha'], short_pin))
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert f'atmodeller: {clone} is at the pinned commit {short_pin[:8]}' in res.stdout
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']


@pytest.mark.unit
def test_atmodeller_folder_is_ignored_by_git():
    """Verify .gitignore ignores all case forms of the atmodeller directory."""
    repo_root = Path(__file__).resolve().parents[2]
    for name in ('atmodeller', 'Atmodeller', 'ATMODELLER'):
        res = subprocess.run(
            ['git', 'check-ignore', name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert res.returncode == 0
        assert res.stdout.strip() == name


@pytest.mark.unit
def test_obliqua_new_clone_and_instantiate(tmp_path):
    """Verify new clone resolves, instantiates, and skips default environment registration."""
    site = _obliqua_site(tmp_path)
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    clone = site['root'] / 'Obliqua'
    assert clone.is_dir()
    assert (clone / 'Manifest.toml').is_file()
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']
    calls = site['log'].read_text().splitlines()
    assert '--project=. -e using Pkg; Pkg.resolve(); Pkg.instantiate()' in calls
    assert not any('Pkg.add' in c for c in calls)
    assert not any('using Obliqua' in c for c in calls)
    assert not any('Pkg.test' in c for c in calls)
    assert not site['pip_log'].exists() or site['pip_log'].read_text().strip() == ''


@pytest.mark.unit
def test_obliqua_register_option_runs_registration(tmp_path):
    """Verify --register adds Obliqua to default environment and checks import."""
    site = _obliqua_site(tmp_path)
    res = _run_tool(site, 'get_obliqua.sh', '--register')
    assert res.returncode == 0, res.stdout + res.stderr
    calls = site['log'].read_text().splitlines()
    assert '-e using Pkg; Pkg.add(path=".")' in calls
    assert '-e using Obliqua; println("Installed to: "*pathof(Obliqua))' in calls


@pytest.mark.unit
def test_obliqua_test_option_runs_test_suite(tmp_path):
    """Verify --test executes Obliqua test suite."""
    site = _obliqua_site(tmp_path)
    res = _run_tool(site, 'get_obliqua.sh', '--test')
    assert res.returncode == 0, res.stdout + res.stderr
    calls = site['log'].read_text().splitlines()
    assert '--project=. -e using Pkg; Pkg.test()' in calls


@pytest.mark.unit
def test_obliqua_existing_folder_at_pin_stays_and_instantiates(tmp_path):
    """Verify existing folder at pin stays, manifest is instantiated, and only instantiate runs."""
    site = _obliqua_site(tmp_path)
    assert _run_tool(site, 'get_obliqua.sh').returncode == 0
    clone = site['root'] / 'Obliqua'
    (clone / 'local_note.txt').write_text('keep me\n')
    manifest_bytes = b'# custom manifest bytes\nversion = 99\n'
    (clone / 'Manifest.toml').write_bytes(manifest_bytes)
    site['log'].write_text('')
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert f'Obliqua: {clone} is at the pinned commit {site["target_sha"][:8]}' in res.stdout
    assert (clone / 'local_note.txt').read_text() == 'keep me\n'
    assert (clone / 'Manifest.toml').read_bytes() == manifest_bytes
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']
    calls = site['log'].read_text().splitlines()
    assert calls == ['--project=. -e using Pkg; Pkg.instantiate()']
    assert not site['pip_log'].exists() or site['pip_log'].read_text().strip() == ''


@pytest.mark.unit
def test_obliqua_existing_folder_off_pin_stays_and_advises(tmp_path):
    """Verify existing folder off-pin stays and advice names git commands without pip."""
    site = _obliqua_site(tmp_path)
    assert _run_tool(site, 'get_obliqua.sh').returncode == 0
    clone = site['root'] / 'Obliqua'
    git(clone, 'checkout', '-q', 'main')
    manifest_bytes = b'# custom manifest bytes off pin\nversion = 101\n'
    (clone / 'Manifest.toml').write_bytes(manifest_bytes)
    site['log'].write_text('')
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    target_short = site['target_sha'][:8]
    assert (
        f'Obliqua: {clone} stays as it is: commit {site["tip_sha"][:7]}; '
        f'not at the pinned commit {target_short}'
    ) in res.stdout
    lines = [ln.strip() for ln in res.stdout.splitlines()]
    idx_fetch = lines.index(f'git -C {clone} fetch origin')
    idx_checkout = lines.index(f'git -C {clone} checkout --detach {site["target_sha"]}')
    idx_step = lines.index('bash tools/get_obliqua.sh')
    idx_force = lines.index('bash tools/get_obliqua.sh --force')
    assert idx_fetch < idx_checkout < idx_step < idx_force
    assert 'pip install -e' not in res.stdout
    assert git(clone, 'rev-parse', 'HEAD') == site['tip_sha']
    assert (clone / 'Manifest.toml').read_bytes() == manifest_bytes


@pytest.mark.unit
def test_obliqua_positional_zero_and_path_arguments(tmp_path):
    """Verify positional 0 and destination path arguments are honoured."""
    site = _obliqua_site(tmp_path)
    custom_dest = site['root'] / 'custom' / 'Obliqua'
    res = _run_tool(site, 'get_obliqua.sh', '0', str(custom_dest))
    assert res.returncode == 0, res.stdout + res.stderr
    assert custom_dest.is_dir()
    assert git(custom_dest, 'rev-parse', 'HEAD') == site['target_sha']
    calls = site['log'].read_text().splitlines()
    assert '--project=. -e using Pkg; Pkg.resolve(); Pkg.instantiate()' in calls
    assert not any('Pkg.test' in c for c in calls)


@pytest.mark.unit
def test_obliqua_unknown_option_exits_1_before_clone(tmp_path):
    """Verify an argument starting with -- that is unknown causes exit 1 before cloning."""
    site = _obliqua_site(tmp_path)
    _write_stub(site['stubs'], 'git', '#!/bin/sh\necho "GIT CALLED" >&2\nexit 99\n')
    res = _run_tool(site, 'get_obliqua.sh', '--regster')
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: unknown option --regster' in res.stderr
    assert 'GIT CALLED' not in res.stderr
    assert not (site['root'] / 'Obliqua').exists()
    assert not (site['root'] / '--regster').exists()
    assert site['log'].read_text() == ''


def _report_clone_state(folder: Path, pin: str) -> subprocess.CompletedProcess:
    """Run ``report_clone_state`` of the helper library on ``folder`` against the commit ``pin``."""
    script = (
        'source tools/_get_common.sh\n'
        f'report_clone_state TestModule {shlex.quote(str(folder))} get_test.sh {pin}\n'
    )
    res = subprocess.run(
        ['bash', '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    return res


@pytest.mark.unit
def test_pinned_clone_without_git_folder_reports_not_a_git_clone(tmp_path):
    """Verify a folder without .git reports not a git clone and advice omits git commands."""
    folder = tmp_path / 'custom_clone'
    folder.mkdir()
    (folder / 'file.txt').write_text('content\n')
    res = _report_clone_state(folder, 'abcdef1234567890abcdef1234567890abcdef12')
    expected_stay = (
        f'TestModule: {folder} stays as it is: not a git clone; '
        'not at the pinned commit abcdef12'
    )
    assert expected_stay in res.stdout
    expected_advice = (
        '    From the PROTEUS folder, move it away or delete it by hand, then run:\n'
        '        bash tools/get_test.sh'
    )
    assert expected_advice in res.stdout
    assert '--force' not in res.stdout
    assert 'fetch origin' not in res.stdout
    assert 'checkout --detach' not in res.stdout


@pytest.mark.unit
def test_pinned_clone_at_pin_reports_at_the_pinned_commit(tmp_path):
    """Verify a folder at the pinned commit reports it is at the pinned commit."""
    folder = tmp_path / 'pinned_clone'
    folder.mkdir()
    git(folder, 'init', '-q', '-b', 'main')
    (folder / 'file.txt').write_text('content\n')
    git(folder, 'add', '.')
    git(folder, 'commit', '-q', '-m', 'initial')
    sha = git(folder, 'rev-parse', 'HEAD')
    res = _report_clone_state(folder, sha)
    assert f'TestModule: {folder} is at the pinned commit {sha[:8]}' in res.stdout
    assert 'stays as it is' not in res.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    ('repo', 'dist', 'extra', 'built'),
    [
        ('Morrigan', 'fwl-morrigan', 'morrigan', False),
        ('VULCAN', 'fwl-vulcan', 'vulcan', True),
    ],
    ids=['morrigan', 'vulcan'],
)
def test_second_call_after_failed_pip_exits_1(tmp_path, repo, dist, extra, built):
    """Verify the second call after a failed pip reports the release tag and exits 1."""
    pyproject = (
        '[project]\nname = "fixture"\ndependencies = []\n'
        f'[project.optional-dependencies]\n{extra} = ["{dist}>=26.4.22"]\n'
    )
    site = _floor_tag_site(tmp_path, pyproject, {repo: '26.04.22'})
    _write_stub(tmp_path / 'stubs', 'pip', '#!/bin/sh\nexit 1\n')
    script = f'get_{extra}.sh'
    res1 = _run_tool(site, script)
    assert res1.returncode == 1
    assert 'ERROR: editable install failed' in res1.stderr
    clone = site['root'] / repo
    assert clone.is_dir()
    assert (clone / 'version.txt').exists()
    if repo == 'VULCAN':
        assert (clone / 'fastchem_vulcan').is_dir()
    res2 = _run_tool(site, script)
    assert res2.returncode == 1
    assert f'{dist}: {clone} is at the release tag 26.04.22' in res2.stdout
    assert (
        f'{dist}: the build step ran in the existing folder {clone}' in res2.stdout
    ) is built
    assert f'ERROR: pip could not install {clone} (its message is above)' in res2.stderr
    assert clone.is_dir()


@pytest.mark.unit
def test_vulcan_failed_fastchem_make_in_new_clone_leaves_no_folder(tmp_path):
    """Verify failing fastchem make in a new VULCAN clone exits 1 and leaves no folder."""
    pyproject = (
        '[project]\nname = "fixture"\ndependencies = []\n'
        '[project.optional-dependencies]\nvulcan = ["fwl-vulcan>=26.4.22"]\n'
    )
    site = _floor_tag_site(tmp_path, pyproject, {'VULCAN': '26.04.22'})
    _write_stub(tmp_path / 'stubs', 'make', '#!/bin/sh\nexit 2\n')
    res = _run_tool(site, 'get_vulcan.sh')
    assert res.returncode == 1
    assert 'ERROR: fastchem build failed' in res.stderr
    assert not (site['root'] / 'VULCAN').exists()
    assert not any(p.name.startswith('VULCAN.tmp') for p in site['root'].iterdir())
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_atmodeller_second_call_after_failed_pip_exits_1(tmp_path):
    """Verify second atmodeller call after failed pip reports pinned commit and exits 1."""
    site = _atmodeller_site(tmp_path)
    _write_stub(site['stubs'], 'pip', '#!/bin/sh\nexit 1\n')
    res1 = _run_tool(site, 'get_atmodeller.sh')
    assert res1.returncode == 1
    clone = site['root'] / 'Atmodeller'
    assert clone.is_dir()
    res2 = _run_tool(site, 'get_atmodeller.sh')
    assert res2.returncode == 1
    assert (
        f'atmodeller: {clone} is at the pinned commit {site["target_sha"][:8]}' in res2.stdout
    )
    assert f'ERROR: pip could not install {clone} (its message is above)' in res2.stderr
    assert clone.is_dir()


@pytest.mark.unit
def test_obliqua_positional_zero_alone_clones_to_default_folder(tmp_path):
    """Verify positional argument 0 alone clones to default Obliqua folder and leaves no folder 0."""
    site = _obliqua_site(tmp_path)
    res = _run_tool(site, 'get_obliqua.sh', '0')
    assert res.returncode == 0, res.stdout + res.stderr
    assert (site['root'] / 'Obliqua').is_dir()
    assert not (site['root'] / '0').exists()


@pytest.mark.unit
def test_obliqua_relative_destination_path(tmp_path):
    """Verify relative destination path creates clone without git directory errors."""
    site = _obliqua_site(tmp_path)
    run_dir = tmp_path / 'workdir'
    run_dir.mkdir()
    res = subprocess.run(
        ['bash', str(site['root'] / 'tools' / 'get_obliqua.sh'), 'rel/Obliqua'],
        cwd=run_dir,
        env=site['env'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    dest = run_dir / 'rel' / 'Obliqua'
    assert dest.is_dir()
    assert re.search(r'Obliqua at [0-9a-f]{7,}', res.stdout)
    assert 'fatal:' not in res.stderr


@pytest.mark.unit
def test_pinned_clone_stays_file_and_local_changes(tmp_path):
    """Verify pinned clone off pin records to stays file and tracked modification notes local changes."""
    site = _atmodeller_site(tmp_path)
    assert _run_tool(site, 'get_atmodeller.sh').returncode == 0
    clone = site['root'] / 'Atmodeller'
    git(clone, 'checkout', '-q', 'main')
    stays = tmp_path / 'stays.txt'
    site['env']['PROTEUS_CLONE_STAYS'] = str(stays)
    res_off = _run_tool(site, 'get_atmodeller.sh')
    assert res_off.returncode == 0, res_off.stdout + res_off.stderr
    assert stays.read_text().splitlines() == [f'Atmodeller (commit {site["tip_sha"][:7]})']

    git(clone, 'checkout', '-q', site['target_sha'])
    tracked = next(p for p in clone.iterdir() if p.is_file() and not p.name.startswith('.'))
    tracked.write_text(tracked.read_text() + 'modified\n')
    res_at = _run_tool(site, 'get_atmodeller.sh')
    assert res_at.returncode == 0, res_at.stdout + res_at.stderr
    assert (
        f'atmodeller: {clone} is at the pinned commit {site["target_sha"][:8]}, with local changes'
        in res_at.stdout
    )


@pytest.mark.unit
def test_pinned_clone_unresolvable_head(tmp_path):
    """Verify pinned clone without commits reports unresolvable HEAD without empty commit string."""
    site = _atmodeller_site(tmp_path)
    clone = site['root'] / 'Atmodeller'
    clone.mkdir(parents=True)
    git(clone, 'init', '-q')
    stays = tmp_path / 'stays.txt'
    site['env']['PROTEUS_CLONE_STAYS'] = str(stays)
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert 'a HEAD that git cannot resolve' in res.stdout
    assert 'commit ;' not in res.stdout
    assert stays.read_text().splitlines() == ['Atmodeller (HEAD unknown)']
    assert (
        '    From the PROTEUS folder, move it away or delete it by hand, then run:\n'
        '        bash tools/get_atmodeller.sh\n'
    ) in res.stdout
    assert '--force' not in res.stdout


@pytest.mark.unit
def test_obliqua_empty_destination_with_force_refused(tmp_path):
    """Verify empty destination with --force exits 1 without deleting current directory."""
    site = _obliqua_site(tmp_path)
    before = sorted(p.name for p in site['root'].iterdir())
    assert 'pyproject.toml' in before
    assert 'tools' in before
    res = subprocess.run(
        ['bash', str(site['root'] / 'tools' / 'get_obliqua.sh'), '--force', ''],
        cwd=site['root'],
        env=site['env'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: the destination is empty' in res.stderr
    assert sorted(p.name for p in site['root'].iterdir()) == before
    assert not any('.tmp.' in p.name for p in site['root'].parent.iterdir())
    assert not (site['root'] / '.git').exists()


@pytest.mark.unit
def test_obliqua_empty_destination_without_force_refused(tmp_path):
    """Verify empty destination without --force exits 1 before invoking julia."""
    site = _obliqua_site(tmp_path)
    res = _run_tool(site, 'get_obliqua.sh', '')
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: the destination is empty' in res.stderr
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_failed_julia_in_new_clone_exits_1(tmp_path):
    """Verify failed julia resolve or instantiate in new clone exits 1 with error message."""
    site = _obliqua_site(tmp_path)
    site['env']['FAIL_JULIA_PATTERN'] = 'Pkg.resolve'
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 1, res.stdout + res.stderr
    dest = site['root'] / 'Obliqua'
    assert f'ERROR: julia resolve or instantiate failed in {dest}' in res.stderr


@pytest.mark.unit
def test_obliqua_failed_julia_in_existing_folder_exits_1(tmp_path):
    """Verify failed julia instantiate in existing folder exits 1 with error message."""
    site = _obliqua_site(tmp_path)
    assert _run_tool(site, 'get_obliqua.sh').returncode == 0
    site['env']['FAIL_JULIA_PATTERN'] = 'Pkg.instantiate()'
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 1, res.stdout + res.stderr
    dest = site['root'] / 'Obliqua'
    assert f'ERROR: julia instantiate failed in {dest}' in res.stderr


@pytest.mark.unit
def test_obliqua_register_failed_pkg_add_exits_1(tmp_path):
    """Verify failed Pkg.add during --register exits 1 with error message."""
    site = _obliqua_site(tmp_path)
    site['env']['FAIL_JULIA_PATTERN'] = 'Pkg.add'
    res = _run_tool(site, 'get_obliqua.sh', '--register')
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: julia Pkg.add failed' in res.stderr


@pytest.mark.unit
def test_obliqua_register_failed_using_check_exits_1(tmp_path):
    """Verify failed using Obliqua check during --register exits 1 with error message."""
    site = _obliqua_site(tmp_path)
    site['env']['FAIL_JULIA_PATTERN'] = 'using Obliqua'
    res = _run_tool(site, 'get_obliqua.sh', '--register')
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: using Obliqua check failed' in res.stderr


@pytest.mark.unit
def test_obliqua_test_failed_pkg_test_exits_1(tmp_path):
    """Verify failed Pkg.test during --test exits 1 with error message."""
    site = _obliqua_site(tmp_path)
    site['env']['FAIL_JULIA_PATTERN'] = 'Pkg.test'
    res = _run_tool(site, 'get_obliqua.sh', '--test')
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: julia Pkg.test failed' in res.stderr


@pytest.mark.unit
@pytest.mark.parametrize('folder', ['custom', 'custom path'], ids=['plain', 'with space'])
def test_obliqua_custom_destination_off_pin_advises_escaped_path(tmp_path, folder):
    """Verify the advice names the custom folder, with spaces escaped by backslashes."""
    site = _obliqua_site(tmp_path)
    custom = site['root'] / folder / 'Obliqua'
    assert _run_tool(site, 'get_obliqua.sh', str(custom)).returncode == 0
    git(custom, 'checkout', '-q', 'main')
    res = _run_tool(site, 'get_obliqua.sh', str(custom))
    assert res.returncode == 0, res.stdout + res.stderr
    lines = [ln.strip() for ln in res.stdout.splitlines()]
    escaped = str(custom).replace(' ', r'\ ')
    idx_fetch = lines.index(f'git -C {escaped} fetch origin')
    idx_checkout = lines.index(f'git -C {escaped} checkout --detach {site["target_sha"]}')
    idx_step = lines.index(f'bash tools/get_obliqua.sh {escaped}')
    idx_force = lines.index(f'bash tools/get_obliqua.sh {escaped} --force')
    assert idx_fetch < idx_checkout < idx_step < idx_force
    assert 'bash tools/get_obliqua.sh --force' not in [ln.strip() for ln in lines]
    if ' ' in folder:
        assert f'bash tools/get_obliqua.sh {custom} --force' not in res.stdout


# ===================================================================
# Forced replace safety, ssh batch mode, and unreadable clones
# ===================================================================


@pytest.mark.unit
def test_obliqua_force_refuses_site_root(tmp_path):
    """Verify get_obliqua.sh --force refuses to delete the PROTEUS root directory."""
    site = _obliqua_site(tmp_path)
    marker = site['root'] / 'marker.txt'
    marker.write_text('keep this file\n')
    before_names = sorted(p.name for p in site['root'].iterdir())
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(site['root']))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {site["root"]} is {site["root"]} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res.stderr
    )
    assert sorted(p.name for p in site['root'].iterdir()) == before_names
    assert marker.read_text() == 'keep this file\n'
    assert not any(
        '.tmp.' in p.name or '.old.' in p.name for p in site['root'].parent.iterdir()
    )
    assert site['log'].read_text() == ''


@pytest.mark.unit
@pytest.mark.parametrize(
    'alias_mode',
    ['dot', 'tools-parent', 'trailing-slash', 'symlink'],
)
def test_obliqua_force_refuses_site_root_aliases(tmp_path, alias_mode):
    """Verify get_obliqua.sh --force refuses all alias paths targeting the PROTEUS root."""
    site = _obliqua_site(tmp_path)
    marker = site['root'] / 'marker.txt'
    marker.write_text('keep alias\n')
    before_names = sorted(p.name for p in site['root'].iterdir())
    if alias_mode == 'symlink':
        link = tmp_path / 'link_to_root'
        link.symlink_to(site['root'])
        target_path = str(link)
    else:
        path_forms = {
            'dot': str(site['root'] / '.'),
            'tools-parent': str(site['root'] / 'tools' / '..'),
            'trailing-slash': f'{site["root"]}/',
        }
        target_path = path_forms[alias_mode]
    res = _run_tool(site, 'get_obliqua.sh', '--force', target_path)
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {site["root"]} is {site["root"]} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res.stderr
    )
    assert sorted(p.name for p in site['root'].iterdir()) == before_names
    assert marker.read_text() == 'keep alias\n'
    assert not any(
        '.tmp.' in p.name or '.old.' in p.name for p in site['root'].parent.iterdir()
    )
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_refuses_parent_of_site_root(tmp_path):
    """Verify get_obliqua.sh --force refuses to delete the parent directory of PROTEUS root."""
    site = _obliqua_site(tmp_path)
    parent = site['root'].parent
    marker = parent / 'parent_marker.txt'
    marker.write_text('preserve parent\n')
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(parent))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {parent} is {site["root"]} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res.stderr
    )
    assert marker.read_text() == 'preserve parent\n'
    assert site['root'].is_dir()
    assert not any('.tmp.' in p.name or '.old.' in p.name for p in parent.iterdir())
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_refuses_home_and_home_parent(tmp_path):
    """Verify get_obliqua.sh --force refuses HOME and its parent directory."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    (fake_home / 'home_marker.txt').write_text('home content\n')
    site['env']['HOME'] = str(fake_home)
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(fake_home))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {fake_home} is {fake_home} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res.stderr
    )
    assert (fake_home / 'home_marker.txt').read_text() == 'home content\n'
    assert not any('.tmp.' in p.name for p in tmp_path.iterdir())

    home_parent = tmp_path / 'homeparent'
    home_nested = home_parent / 'home'
    home_nested.mkdir(parents=True)
    (home_parent / 'parent_marker.txt').write_text('parent content\n')
    site['env']['HOME'] = str(home_nested)
    res2 = _run_tool(site, 'get_obliqua.sh', '--force', str(home_parent))
    assert res2.returncode == 1, res2.stdout + res2.stderr
    assert (
        f'ERROR: {home_parent} is {home_nested} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res2.stderr
    )
    assert (home_parent / 'parent_marker.txt').read_text() == 'parent content\n'
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_refuses_non_clone_folder_with_files(tmp_path):
    """Verify get_obliqua.sh --force refuses a non-empty folder without an Obliqua Project.toml."""
    site = _obliqua_site(tmp_path)
    target = tmp_path / 'non_clone_folder'
    target.mkdir()
    marker = target / 'user_file.txt'
    marker.write_text('important data\n')
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(target))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} is not an Obliqua project (no Project.toml that names Obliqua); move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert marker.read_text() == 'important data\n'
    assert not (target / '.git').exists()
    assert not any('.tmp.' in p.name for p in tmp_path.iterdir())
    assert site['log'].read_text() == ''


@pytest.mark.unit
@pytest.mark.parametrize('has_project_toml', [False, True], ids=['readme-only', 'other-name'])
def test_obliqua_force_refuses_foreign_git_clone(tmp_path, has_project_toml):
    """Verify get_obliqua.sh --force refuses git clones that are not Obliqua."""
    site = _obliqua_site(tmp_path)
    target = tmp_path / 'foreign_clone'
    target.mkdir()
    git(target, 'init', '-q', '-b', 'main')
    (target / 'README.md').write_text('Foreign project\n')
    if has_project_toml:
        (target / 'Project.toml').write_text('name = "Other"\nuuid = "abcd"\n')
    git(target, 'add', '.')
    git(target, 'commit', '-q', '-m', 'init')
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(target))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} is not an Obliqua project (no Project.toml that names Obliqua); move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert (target / 'README.md').read_text() == 'Foreign project\n'
    assert not any('.tmp.' in p.name for p in tmp_path.iterdir())
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_refuses_regular_file(tmp_path):
    """Verify get_obliqua.sh --force refuses to replace a regular file."""
    site = _obliqua_site(tmp_path)
    target = tmp_path / 'single_file.txt'
    target.write_text('plain file content\n')
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(target))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} is a file, a broken link or a folder that cannot be entered; move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert target.is_file()
    assert target.read_text() == 'plain file content\n'
    assert not any('.tmp.' in p.name for p in tmp_path.iterdir())
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_foreign_git_clone_at_default_dest_without_force_refused(tmp_path):
    """Verify foreign git clone at default destination without --force is refused."""
    site = _obliqua_site(tmp_path)
    target = site['root'] / 'Obliqua'
    target.mkdir()
    git(target, 'init', '-q', '-b', 'main')
    (target / 'README.md').write_text('Foreign project\n')
    git(target, 'add', '.')
    git(target, 'commit', '-q', '-m', 'init')
    head = git(target, 'rev-parse', 'HEAD')
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} is not an Obliqua project (no Project.toml that names Obliqua); move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert (target / 'README.md').read_text() == 'Foreign project\n'
    assert git(target, 'rev-parse', 'HEAD') == head
    assert site['log'].read_text() == ''
    assert '--force' not in res.stdout


@pytest.mark.unit
def test_obliqua_foreign_git_clone_at_default_dest_with_force_refused(tmp_path):
    """Verify foreign git clone at default destination with --force is refused."""
    site = _obliqua_site(tmp_path)
    target = site['root'] / 'Obliqua'
    target.mkdir()
    git(target, 'init', '-q', '-b', 'main')
    (target / 'README.md').write_text('Foreign project\n')
    git(target, 'add', '.')
    git(target, 'commit', '-q', '-m', 'init')
    head = git(target, 'rev-parse', 'HEAD')
    res = _run_tool(site, 'get_obliqua.sh', '--force')
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} is not an Obliqua project (no Project.toml that names Obliqua); move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert (target / 'README.md').read_text() == 'Foreign project\n'
    assert git(target, 'rev-parse', 'HEAD') == head
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_git_init_only_folder_without_force_refused(tmp_path):
    """Verify folder with git init only without --force is refused."""
    site = _obliqua_site(tmp_path)
    target = site['root'] / 'Obliqua'
    target.mkdir()
    git(target, 'init', '-q', '-b', 'main')
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} is not an Obliqua project (no Project.toml that names Obliqua); move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert (target / '.git').is_dir()
    assert site['log'].read_text() == ''
    assert '--force' not in res.stdout


@pytest.mark.unit
def test_obliqua_uncommitted_clone_with_project_toml_advises_move_and_force_refused(tmp_path):
    """Verify uncommitted clone with Obliqua Project.toml advises manual removal and refuses --force."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    target = site['root'] / 'Obliqua'
    target.mkdir()
    git(target, 'init', '-q', '-b', 'main')
    (target / 'Project.toml').write_text('name = "Obliqua"\nuuid = "1234"\n')
    user_file = target / 'user.txt'
    user_file.write_text('keep user file\n')
    res_no_force = _run_tool(site, 'get_obliqua.sh')
    assert res_no_force.returncode == 0, res_no_force.stdout + res_no_force.stderr
    assert (
        '    From the PROTEUS folder, move it away or delete it by hand, then run:\n'
        '        bash tools/get_obliqua.sh\n'
    ) in res_no_force.stdout
    assert '--force' not in res_no_force.stdout
    site['log'].write_text('')
    before_names = sorted(p.name for p in target.iterdir())
    res_force = _run_tool(site, 'get_obliqua.sh', '--force')
    assert res_force.returncode == 1, res_force.stdout + res_force.stderr
    assert (
        f'ERROR: {target} holds files and is not a git clone of its own with a commit; move it away or delete it by hand; it is unchanged'
        in res_force.stderr
    )
    assert sorted(p.name for p in target.iterdir()) == before_names
    assert user_file.read_text() == 'keep user file\n'
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_refuses_site_root_with_different_letter_case(tmp_path):
    """Verify get_obliqua.sh --force refuses site root with swapped letter case on case-insensitive filesystems."""
    site = _obliqua_site(tmp_path)
    root = site['root']
    marker = root / 'site_marker.txt'
    marker.write_text('preserve site\n')
    other = root.parent / root.name.swapcase()
    assert str(other) != str(root)
    if not other.exists():
        pytest.skip('the file system tells letter case apart')
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(other))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {other} is {site["root"]} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res.stderr
    )
    assert root.is_dir()
    assert marker.read_text() == 'preserve site\n'


@pytest.mark.unit
def test_obliqua_force_refuses_unreadable_folder(tmp_path):
    """Verify get_obliqua.sh --force refuses a folder without permissions."""
    if os.geteuid() == 0:
        pytest.skip('root bypasses filesystem permissions')
    site = _obliqua_site(tmp_path)
    target = tmp_path / 'unreadable_dir'
    target.mkdir()
    target.chmod(0)
    try:
        res = _run_tool(site, 'get_obliqua.sh', '--force', str(target))
        assert res.returncode == 1, res.stdout + res.stderr
        assert (
            f'ERROR: {target} is a file, a broken link or a folder that cannot be entered; move it away or delete it by hand; it is unchanged'
            in res.stderr
        )
        assert target.exists()
    finally:
        target.chmod(0o755)


@pytest.mark.unit
@pytest.mark.parametrize('empty_git_dir', [False, True], ids=['no-git', 'empty-git-dir'])
def test_obliqua_force_refuses_plain_copy_inside_git_repo(tmp_path, empty_git_dir):
    """Verify get_obliqua.sh --force refuses plain copy of Obliqua inside an outer git repo."""
    site = _obliqua_site(tmp_path)
    git(site['root'], 'init', '-q', '-b', 'main')
    (site['root'] / 'outer.txt').write_text('outer repo\n')
    git(site['root'], 'add', '.')
    git(site['root'], 'commit', '-q', '-m', 'init-outer')
    target = site['root'] / 'Obliqua'
    target.mkdir()
    (target / 'Project.toml').write_text('name = "Obliqua"\nuuid = "1234"\n')
    (target / 'copy.txt').write_text('copied file\n')
    if empty_git_dir:
        (target / '.git').mkdir()
    res = _run_tool(site, 'get_obliqua.sh', '--force')
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} holds files and is not a git clone of its own with a commit; move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert (target / 'copy.txt').read_text() == 'copied file\n'
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_refuses_site_root_when_git_repo(tmp_path):
    """Verify get_obliqua.sh --force refuses site root even when it is a git repository."""
    site = _obliqua_site(tmp_path)
    git(site['root'], 'init', '-q', '-b', 'main')
    marker = site['root'] / 'site_marker.txt'
    marker.write_text('site marker\n')
    git(site['root'], 'add', 'site_marker.txt')
    git(site['root'], 'commit', '-q', '-m', 'init-site')
    before_names = sorted(p.name for p in site['root'].iterdir())
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(site['root']))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {site["root"]} is {site["root"]} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res.stderr
    )
    assert 'delete it by hand' not in res.stderr
    assert sorted(p.name for p in site['root'].iterdir()) == before_names
    assert marker.read_text() == 'site marker\n'


@pytest.mark.unit
def test_obliqua_no_force_refuses_site_root_when_git_repo(tmp_path):
    """Verify get_obliqua.sh without --force refuses site root when it is a git repository."""
    site = _obliqua_site(tmp_path)
    git(site['root'], 'init', '-q', '-b', 'main')
    marker = site['root'] / 'site_marker.txt'
    marker.write_text('site marker\n')
    git(site['root'], 'add', 'site_marker.txt')
    git(site['root'], 'commit', '-q', '-m', 'init-site')
    before_names = sorted(p.name for p in site['root'].iterdir())
    res = _run_tool(site, 'get_obliqua.sh', str(site['root']))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {site["root"]} is {site["root"]} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res.stderr
    )
    assert 'delete it by hand' not in res.stderr
    assert sorted(p.name for p in site['root'].iterdir()) == before_names
    assert marker.read_text() == 'site marker\n'
    assert site['log'].read_text() == ''


@pytest.mark.unit
@pytest.mark.parametrize('with_force', [False, True], ids=['no-force', 'force'])
def test_obliqua_refuses_home_when_git_repo(tmp_path, with_force):
    """Verify get_obliqua.sh refuses HOME directory when HOME is a git repository."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'fake_home'
    fake_home.mkdir()
    git(fake_home, 'init', '-q', '-b', 'main')
    marker = fake_home / 'home_marker.txt'
    marker.write_text('home marker\n')
    git(fake_home, 'add', 'home_marker.txt')
    git(fake_home, 'commit', '-q', '-m', 'init-home')
    site['env']['HOME'] = str(fake_home)
    args = ['--force', str(fake_home)] if with_force else [str(fake_home)]
    res = _run_tool(site, 'get_obliqua.sh', *args)
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {fake_home} is {fake_home} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res.stderr
    )
    assert 'delete it by hand' not in res.stderr
    assert marker.read_text() == 'home marker\n'
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_replaces_symlink_to_pinned_clone(tmp_path):
    """Verify get_obliqua.sh --force keeps a folder symlink and replaces its target clone."""
    site = _obliqua_site(tmp_path)
    first_clone = tmp_path / 'original_obliqua'
    res_orig = _run_tool(site, 'get_obliqua.sh', str(first_clone))
    assert res_orig.returncode == 0, res_orig.stdout + res_orig.stderr
    assert git(first_clone, 'rev-parse', 'HEAD') == site['target_sha']
    marker = first_clone / 'old_marker.txt'
    marker.write_text('old content\n')

    link_path = site['root'] / 'Obliqua'
    link_path.symlink_to(first_clone)
    assert link_path.is_symlink()

    res = _run_tool(site, 'get_obliqua.sh', '--force', str(link_path))
    assert res.returncode == 0, res.stdout + res.stderr
    assert link_path.is_symlink()
    assert first_clone.is_dir()
    assert (first_clone / '.git').is_dir()
    assert (first_clone / 'Project.toml').is_file()
    assert not marker.exists()
    assert git(first_clone, 'rev-parse', 'HEAD') == site['target_sha']


@pytest.mark.unit
def test_obliqua_force_replaces_symlink_to_empty_directory(tmp_path):
    """Verify get_obliqua.sh --force keeps a symlink to an empty directory and clones into its target."""
    site = _obliqua_site(tmp_path)
    empty_dir = tmp_path / 'empty_target'
    empty_dir.mkdir()
    link_path = site['root'] / 'Obliqua'
    link_path.symlink_to(empty_dir)
    res = _run_tool(site, 'get_obliqua.sh', '--force')
    assert res.returncode == 0, res.stdout + res.stderr
    assert link_path.is_symlink()
    assert empty_dir.is_dir()
    assert (empty_dir / '.git').is_dir()
    assert (empty_dir / 'Project.toml').is_file()
    assert git(empty_dir, 'rev-parse', 'HEAD') == site['target_sha']


@pytest.mark.unit
def test_obliqua_force_allowed_paths(tmp_path):
    """Verify get_obliqua.sh --force allows missing path and empty directory."""
    site = _obliqua_site(tmp_path)
    missing = site['root'] / 'missing_target'
    res_missing = _run_tool(site, 'get_obliqua.sh', '--force', str(missing))
    assert res_missing.returncode == 0, res_missing.stdout + res_missing.stderr
    assert (missing / 'Project.toml').is_file()
    assert git(missing, 'rev-parse', 'HEAD') == site['target_sha']

    empty = site['root'] / 'empty_target'
    empty.mkdir()
    res_empty = _run_tool(site, 'get_obliqua.sh', '--force', str(empty))
    assert res_empty.returncode == 0, res_empty.stdout + res_empty.stderr
    assert (empty / 'Project.toml').is_file()
    assert git(empty, 'rev-parse', 'HEAD') == site['target_sha']


@pytest.mark.unit
def test_obliqua_non_clone_folder_stays_and_advises_manual_removal(tmp_path):
    """Verify plain copy of Obliqua without --force stays, runs julia instantiate, and advises removal."""
    site = _obliqua_site(tmp_path)
    target = site['root'] / 'Obliqua'
    target.mkdir()
    (target / 'Project.toml').write_text('name = "Obliqua"\nuuid = "1234"\n')
    (target / 'custom.txt').write_text('data\n')
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert f'Obliqua: {target} stays as it is: not a git clone' in res.stdout
    assert 'From the PROTEUS folder, move it away or delete it by hand, then run:' in res.stdout
    assert 'bash tools/get_obliqua.sh' in res.stdout
    assert '--force' not in res.stdout
    assert (target / 'custom.txt').read_text() == 'data\n'
    calls = site['log'].read_text().splitlines()
    assert len(calls) == 1
    assert 'Pkg.instantiate' in calls[0]


@pytest.mark.unit
def test_atmodeller_non_clone_folder_stays_and_advises_manual_removal(tmp_path):
    """Verify non-clone atmodeller folder without --force stays and advises manual removal."""
    site = _atmodeller_site(tmp_path)
    target = site['root'] / 'Atmodeller'
    target.mkdir()
    (target / 'custom.txt').write_text('data\n')
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert f'atmodeller: {target} stays as it is: not a git clone' in res.stdout
    assert 'From the PROTEUS folder, move it away or delete it by hand, then run:' in res.stdout
    assert 'bash tools/get_atmodeller.sh' in res.stdout
    assert '--force' not in res.stdout
    assert (target / 'custom.txt').read_text() == 'data\n'


@pytest.mark.unit
@pytest.mark.parametrize(
    ('script', 'fixture_fn'),
    [
        ('get_atmodeller.sh', _atmodeller_site),
        (
            'get_vulcan.sh',
            lambda p: _floor_tag_site(
                p,
                '[project]\nname="fixture"\ndependencies=[]\n[project.optional-dependencies]\nvulcan=["fwl-vulcan>=26.4.22"]\n',
                {'VULCAN': '26.04.22'},
            ),
        ),
        (
            'get_morrigan.sh',
            lambda p: _floor_tag_site(
                p,
                '[project]\nname="fixture"\ndependencies=[]\n[project.optional-dependencies]\nmorrigan=["fwl-morrigan>=26.4.22"]\n',
                {'Morrigan': '26.04.22'},
            ),
        ),
    ],
    ids=['atmodeller', 'vulcan', 'morrigan'],
)
def test_modules_ssh_probe_batch_mode_arguments(tmp_path, script, fixture_fn):
    """Verify module setup scripts execute SSH probes with batch mode and timeout options."""
    site = fixture_fn(tmp_path)
    ssh_log = tmp_path / 'ssh_probe.log'
    _write_stub(
        site['root'].parent / 'stubs',
        'ssh',
        f'#!/bin/sh\necho "$@" >> {shlex.quote(str(ssh_log))}\nexit 255\n',
    )
    res = _run_tool(site, script)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = ssh_log.read_text().splitlines()
    assert len(calls) >= 1
    assert '-o BatchMode=yes' in calls[0]
    assert '-o ConnectTimeout=10' in calls[0]
    assert calls[0].endswith('-T git@github.com')
    assert 'git@github.com:' not in res.stdout


@pytest.mark.unit
def test_ssh_invocations_use_batch_mode():
    """Verify every non-comment shell command executing ssh includes BatchMode=yes and ConnectTimeout=10."""
    repo_root = Path(__file__).resolve().parents[2]
    sh_files = [repo_root / 'install.sh', *(repo_root / 'tools').glob('*.sh')]
    checked_count = 0
    matched_files = set()
    for path in sh_files:
        lines = path.read_text(encoding='utf-8').splitlines()
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(
                ('#', 'echo ', 'printf ', 'info ', 'warn ', 'fail ')
            ):
                continue
            if (
                re.search(r'(^|\$|\(|\s)ssh\s', stripped)
                or '${GIT_SSH_COMMAND:-ssh}' in stripped
            ):
                assert 'BatchMode=yes' in stripped, (
                    f'{path.name}:{idx} lacks BatchMode=yes: {line}'
                )
                assert 'ConnectTimeout=10' in stripped, (
                    f'{path.name}:{idx} lacks ConnectTimeout=10: {line}'
                )
                matched_files.add(path.name)
                checked_count += 1
    assert checked_count >= 2
    assert {'install.sh', '_get_common.sh'} <= matched_files


@pytest.mark.unit
@pytest.mark.parametrize(
    ('script', 'fixture_fn', 'folder_name'),
    [
        ('get_atmodeller.sh', _atmodeller_site, 'Atmodeller'),
        ('get_obliqua.sh', _obliqua_site, 'Obliqua'),
    ],
    ids=['atmodeller', 'obliqua'],
)
def test_pinned_clone_in_outer_git_repo_reports_not_a_readable_git_clone(
    tmp_path, script, fixture_fn, folder_name
):
    """Verify empty .git inside outer repo reports not a readable git clone without git -C advice."""
    site = fixture_fn(tmp_path)
    git(site['root'], 'init', '-q', '-b', 'main')
    (site['root'] / 'outer.txt').write_text('outer commit\n')
    git(site['root'], 'add', '.')
    git(site['root'], 'commit', '-q', '-m', 'outer')
    outer_hash = git(site['root'], 'rev-parse', 'HEAD')[:7]

    target = site['root'] / folder_name
    (target / '.git').mkdir(parents=True)
    (target / 'content.txt').write_text('data\n')
    if folder_name == 'Obliqua':
        (target / 'Project.toml').write_text('name = "Obliqua"\nuuid = "1234"\n')

    res = _run_tool(site, script)
    output = res.stdout + res.stderr
    assert f'{target} stays as it is: not a readable git clone' in output
    assert 'not a readable git clone' in output
    assert 'git -C' not in output
    assert outer_hash not in output
    assert 'From the PROTEUS folder, move it away or delete it by hand, then run:' in output
    assert f'bash tools/{script}' in output


@pytest.mark.unit
def test_obliqua_empty_folder_without_force_clones_and_instantiates(tmp_path):
    """Verify empty Obliqua folder without --force clones and runs julia instantiate."""
    site = _obliqua_site(tmp_path)
    clone = site['root'] / 'Obliqua'
    clone.mkdir()
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert (clone / '.git').is_dir()
    assert (clone / 'Project.toml').is_file()
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']
    calls = site['log'].read_text().splitlines()
    assert '--project=. -e using Pkg; Pkg.resolve(); Pkg.instantiate()' in calls


@pytest.mark.unit
def test_atmodeller_empty_folder_without_force_clones_and_installs(tmp_path):
    """Verify empty Atmodeller folder without --force clones and runs pip install."""
    site = _atmodeller_site(tmp_path)
    clone = site['root'] / 'Atmodeller'
    clone.mkdir()
    res = _run_tool(site, 'get_atmodeller.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert (clone / '.git').is_dir()
    assert (clone / 'pyproject.toml').is_file()
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']
    assert f'install -U -e {clone}' in site['log'].read_text()


def _path_without_tool(path: str, tool: str, scratch: Path) -> str:
    """Return ``path`` with each folder that holds ``tool`` replaced by links to its other entries."""
    parts = []
    for index, folder in enumerate(path.split(os.pathsep)):
        if not folder or not os.path.lexists(os.path.join(folder, tool)):
            parts.append(folder)
            continue
        mirror = scratch / f'path_{index}'
        mirror.mkdir(parents=True)
        for entry in Path(folder).iterdir():
            if entry.name != tool:
                (mirror / entry.name).symlink_to(entry)
        parts.append(str(mirror))
    return os.pathsep.join(parts)


@pytest.mark.unit
def test_obliqua_missing_julia_exits_1(tmp_path):
    """Verify get_obliqua.sh exits 1 when julia is missing from PATH."""
    site = _obliqua_site(tmp_path)
    site['env']['PATH'] = _path_without_tool(
        site['env']['PATH'], 'julia', tmp_path / 'no_julia'
    )
    assert shutil.which('julia', path=site['env']['PATH']) is None
    assert shutil.which('bash', path=site['env']['PATH']) is not None
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: julia is not on PATH' in res.stderr
    assert not (site['root'] / 'Obliqua').exists()


@pytest.mark.unit
def test_obliqua_missing_pin_table_and_env_exits_1(tmp_path):
    """Verify get_obliqua.sh exits 1 when pin table and env vars are missing."""
    site = _obliqua_site(tmp_path)
    (site['root'] / 'pyproject.toml').write_text(
        '[project]\nname = "fixture"\ndependencies = []\n'
    )
    site['env'].pop('OBLIQUA_GIT_URL', None)
    site['env'].pop('OBLIQUA_GIT_REF', None)
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: could not resolve obliqua url/ref from pyproject.toml' in res.stderr
    assert not (site['root'] / 'Obliqua').exists()


@pytest.mark.unit
def test_obliqua_env_overrides_missing_pin_table(tmp_path):
    """Verify environment variables allow cloning Obliqua without pin table in pyproject.toml."""
    site = _obliqua_site(tmp_path)
    (site['root'] / 'pyproject.toml').write_text(
        '[project]\nname = "fixture"\ndependencies = []\n'
    )
    site['env']['OBLIQUA_GIT_URL'] = 'https://github.com/FormingWorlds/Obliqua.git'
    site['env']['OBLIQUA_GIT_REF'] = site['target_sha']
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    clone = site['root'] / 'Obliqua'
    assert (clone / 'Project.toml').is_file()
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']


@pytest.mark.unit
def test_obliqua_non_clone_other_project_without_force_refused(tmp_path):
    """Verify non-clone folder with another Project.toml name without --force is refused."""
    site = _obliqua_site(tmp_path)
    target = site['root'] / 'Obliqua'
    target.mkdir()
    (target / 'Project.toml').write_text('name = "Other"\nuuid = "1234"\n')
    (target / 'other.txt').write_text('data\n')
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} is not an Obliqua project (no Project.toml that names Obliqua); move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert not (target / 'Manifest.toml').exists()
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_site_tools_folder_without_force_refused(tmp_path):
    """Verify passing the site tools folder without --force is refused without running julia."""
    site = _obliqua_site(tmp_path)
    tools_dir = site['root'] / 'tools'
    before_names = sorted(p.name for p in tools_dir.iterdir())
    res = _run_tool(site, 'get_obliqua.sh', str(tools_dir))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {tools_dir} is not an Obliqua project (no Project.toml that names Obliqua); move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert site['log'].read_text() == ''
    assert sorted(p.name for p in tools_dir.iterdir()) == before_names


@pytest.mark.unit
@pytest.mark.parametrize('with_force', [False, True], ids=['no-force', 'force'])
def test_obliqua_refuses_broken_symlink(tmp_path, with_force):
    """Verify broken symlink at default Obliqua destination is refused with and without --force."""
    site = _obliqua_site(tmp_path)
    link = site['root'] / 'Obliqua'
    link.symlink_to(tmp_path / 'does_not_exist')
    args = ['--force'] if with_force else []
    res = _run_tool(site, 'get_obliqua.sh', *args)
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {link} is a file, a broken link or a folder that cannot be entered; move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert link.is_symlink()
    assert site['log'].read_text() == ''


@pytest.mark.unit
@pytest.mark.parametrize('with_force', [False, True], ids=['no-force', 'force'])
def test_obliqua_refuses_regular_file_at_default_dest(tmp_path, with_force):
    """Verify regular file at default Obliqua destination is refused with and without --force."""
    site = _obliqua_site(tmp_path)
    target = site['root'] / 'Obliqua'
    target.write_text('regular file bytes\n')
    args = ['--force'] if with_force else []
    res = _run_tool(site, 'get_obliqua.sh', *args)
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} is a file, a broken link or a folder that cannot be entered; move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert target.is_file()
    assert target.read_text() == 'regular file bytes\n'
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_symlink_to_empty_folder_without_force_clones_into_target(tmp_path):
    """Verify symlink to empty folder without --force keeps symlink and clones into target."""
    site = _obliqua_site(tmp_path)
    empty_dir = tmp_path / 'empty_target'
    empty_dir.mkdir()
    link = site['root'] / 'Obliqua'
    link.symlink_to(empty_dir)
    res = _run_tool(site, 'get_obliqua.sh')
    assert res.returncode == 0, res.stdout + res.stderr
    assert link.is_symlink()
    assert (empty_dir / '.git').is_dir()
    assert git(empty_dir, 'rev-parse', 'HEAD') == site['target_sha']
    calls = site['log'].read_text().splitlines()
    assert '--project=. -e using Pkg; Pkg.resolve(); Pkg.instantiate()' in calls


@pytest.mark.unit
@pytest.mark.parametrize('suffix', ['//', '/.'], ids=['double-slash', 'slash-dot'])
def test_obliqua_missing_destination_with_trailing_slash_or_dot(tmp_path, suffix):
    """Verify missing destination ending in // or /. clones cleanly without leaving .tmp. folders."""
    site = _obliqua_site(tmp_path)
    target = tmp_path / 'newdir'
    res = _run_tool(site, 'get_obliqua.sh', f'{target}{suffix}')
    assert res.returncode == 0, res.stdout + res.stderr
    assert (target / '.git').is_dir()
    assert git(target, 'rev-parse', 'HEAD') == site['target_sha']
    assert not any('.tmp.' in p.name for p in target.iterdir())
    assert not any('.tmp.' in p.name for p in tmp_path.iterdir())


@pytest.mark.unit
def test_obliqua_existing_empty_destination_with_double_slash(tmp_path):
    """Verify existing empty destination ending in // clones cleanly without leaving .tmp. folders."""
    site = _obliqua_site(tmp_path)
    target = tmp_path / 'emptydir'
    target.mkdir()
    res = _run_tool(site, 'get_obliqua.sh', f'{target}//')
    assert res.returncode == 0, res.stdout + res.stderr
    assert (target / '.git').is_dir()
    assert git(target, 'rev-parse', 'HEAD') == site['target_sha']
    assert not any('.tmp.' in p.name for p in target.iterdir())
    assert not any('.tmp.' in p.name for p in tmp_path.iterdir())


@pytest.mark.unit
def test_obliqua_refuses_filesystem_root_without_force(tmp_path):
    """Verify passing / without --force is refused before invoking git or julia."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    git_log = tmp_path / 'git_root.log'
    git_log.write_text('')
    julia_log = site['log']
    _write_stub(
        site['stubs'],
        'git',
        f'#!/bin/sh\necho "$@" >> {shlex.quote(str(git_log))}\nexit 1\n',
    )
    _write_stub(
        site['stubs'],
        'julia',
        f'#!/bin/sh\necho "$@" >> {shlex.quote(str(julia_log))}\nexit 1\n',
    )
    res = _run_tool(site, 'get_obliqua.sh', '/')
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: / is {site["root"]} or a folder that holds it, and cannot be the Obliqua folder; it is unchanged'
        in res.stderr
    )
    assert git_log.read_text() == ''
    assert julia_log.read_text() == ''


def _temp_or_old_entries(root: Path) -> list[Path]:
    """Return paths up to 3 levels under ``root`` whose name holds ``.tmp.`` or ``.old.``."""
    dirs = [root, *root.glob('*'), *root.glob('*/*')]
    return [
        p
        for d in dirs
        if d.is_dir()
        for p in d.iterdir()
        if '.tmp.' in p.name or '.old.' in p.name
    ]


def _setup_missing_middle_case(
    tmp_path: Path, site: dict, case: str
) -> tuple[str, Path, Path, Path]:
    """Set up one of the three missing-middle-folder dotdot targets under ``tmp_path``."""
    fake_home = tmp_path / 'home'
    fake_home.mkdir(exist_ok=True)
    site['env']['HOME'] = str(fake_home)
    if case == 'victim':
        pointed = tmp_path / 'victim'
        pointed.mkdir()
        marker = pointed / 'precious.txt'
        missing = tmp_path / 'nx'
        dest_arg = f'{tmp_path}/nx/../victim'
    elif case == 'home':
        pointed = fake_home
        marker = pointed / 'precious.txt'
        missing = tmp_path / 'zz'
        dest_arg = f'{tmp_path}/zz/../home'
    else:
        pointed = site['root'] / 'tools'
        marker = pointed / 'precious.txt'
        missing = site['root'] / 'nonexist'
        dest_arg = f'{site["root"]}/nonexist/../tools'
    marker.write_text('precious bytes\n')
    return dest_arg, pointed, marker, missing


@pytest.mark.unit
@pytest.mark.parametrize(
    'case', ['victim', 'home', 'root-tools'], ids=['victim', 'home', 'root-tools']
)
def test_obliqua_force_refuses_missing_middle_folder_with_dotdot(tmp_path, case):
    """Verify --force refuses paths traversing a non-existent folder and .. without replacing targets."""
    site = _obliqua_site(tmp_path)
    dest_arg, pointed, marker, missing = _setup_missing_middle_case(tmp_path, site, case)
    before_names = sorted(p.name for p in pointed.iterdir())
    res = _run_tool(site, 'get_obliqua.sh', '--force', dest_arg)
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f"ERROR: {dest_arg} goes through a folder that does not exist and then through '..'; give the path of the folder itself; nothing is changed"
        in res.stderr
    )
    assert sorted(p.name for p in pointed.iterdir()) == before_names
    assert marker.read_text() == 'precious bytes\n'
    assert not missing.exists()
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
@pytest.mark.parametrize(
    'case', ['victim', 'home', 'root-tools'], ids=['victim', 'home', 'root-tools']
)
def test_obliqua_without_force_refuses_missing_middle_folder_with_dotdot(tmp_path, case):
    """Verify get_obliqua.sh without --force refuses paths traversing a non-existent folder and .."""
    site = _obliqua_site(tmp_path)
    dest_arg, pointed, marker, missing = _setup_missing_middle_case(tmp_path, site, case)
    before_names = sorted(p.name for p in pointed.iterdir())
    res = _run_tool(site, 'get_obliqua.sh', dest_arg)
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f"ERROR: {dest_arg} goes through a folder that does not exist and then through '..'; give the path of the folder itself; nothing is changed"
        in res.stderr
    )
    assert sorted(p.name for p in pointed.iterdir()) == before_names
    assert marker.read_text() == 'precious bytes\n'
    assert not missing.exists()
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_refuses_relative_missing_middle_folder_with_dotdot(tmp_path):
    """Verify --force refuses a relative destination traversing a missing folder and .."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    sub = tmp_path / 'sub'
    sub.mkdir()
    victim = tmp_path / 'victim'
    victim.mkdir()
    marker = victim / 'precious.txt'
    marker.write_text('precious bytes\n')
    missing = tmp_path / 'x'
    before_names = sorted(p.name for p in victim.iterdir())
    res = _run_tool(site, 'get_obliqua.sh', '--force', '../x/../victim', cwd=sub)
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f"ERROR: {sub}/../x/../victim goes through a folder that does not exist and then through '..'; give the path of the folder itself; nothing is changed"
        in res.stderr
    )
    assert sorted(p.name for p in victim.iterdir()) == before_names
    assert marker.read_text() == 'precious bytes\n'
    assert not missing.exists()
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_replaces_clone_via_existing_dotdot_path(tmp_path):
    """Verify --force replaces an off-pin Obliqua clone when all folders in a .. path exist."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    there = tmp_path / 'there'
    there.mkdir()
    clone = tmp_path / 'clone'
    assert _run_tool(site, 'get_obliqua.sh', str(clone)).returncode == 0
    git(clone, 'checkout', '-q', site['tip_sha'])
    marker = clone / 'local_marker.txt'
    marker.write_text('remove me\n')
    res = _run_tool(site, 'get_obliqua.sh', '--force', f'{there}/../clone')
    assert res.returncode == 0, res.stdout + res.stderr
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']
    assert not marker.exists()
    assert _temp_or_old_entries(tmp_path) == []


@pytest.mark.unit
def test_obliqua_force_on_new_destination_refuses_folder_appearing_during_clone(tmp_path):
    """Verify --force on an absent destination does not overwrite a folder created during git clone."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    real_git = shutil.which('git')
    assert real_git is not None
    target = tmp_path / 'new_obliqua'
    late_file = target / 'late.txt'
    _write_stub(
        site['stubs'],
        'git',
        '#!/bin/sh\n'
        'if [ "${1:-}" = "clone" ]; then\n'
        f'    mkdir -p {shlex.quote(str(target))}\n'
        f'    printf "late bytes\\n" > {shlex.quote(str(late_file))}\n'
        'fi\n'
        f'exec {shlex.quote(real_git)} "$@"\n',
    )
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(target))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} appeared while the new clone was made; it is unchanged' in res.stderr
    )
    assert sorted(p.name for p in target.iterdir()) == ['late.txt']
    assert late_file.read_text() == 'late bytes\n'
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
@pytest.mark.parametrize(
    'mode', ['worktree', 'separate-git-dir'], ids=['worktree', 'separate-git-dir']
)
def test_obliqua_force_refuses_git_worktree(tmp_path, mode):
    """Verify --force refuses a destination whose .git is a file such as a worktree or separate-git-dir clone."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    clone = site['root'] / 'Obliqua'
    if mode == 'worktree':
        assert _run_tool(site, 'get_obliqua.sh').returncode == 0
        target = tmp_path / 'wt'
        git(clone, 'worktree', 'add', '-q', '--detach', str(target))
    else:
        gd = tmp_path / 'gd'
        target = tmp_path / 'sep'
        git(
            tmp_path,
            'clone',
            '-q',
            f'--separate-git-dir={gd}',
            str(site['remotes'] / 'Obliqua.git'),
            str(target),
        )
    assert (target / '.git').is_file()
    marker = target / 'wt_marker.txt'
    marker.write_text('worktree bytes\n')
    before_names = sorted(p.name for p in target.iterdir())
    site['log'].write_text('')
    res = _run_tool(site, 'get_obliqua.sh', '--force', str(target))
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {target} is a folder whose .git is a file (a git worktree or a submodule); move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert sorted(p.name for p in target.iterdir()) == before_names
    assert marker.read_text() == 'worktree bytes\n'
    if mode == 'worktree':
        assert str(target) in git(clone, 'worktree', 'list')
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
@pytest.mark.parametrize('force_args', [(), ('--force',)], ids=['no-force', 'force'])
def test_obliqua_refuses_file_in_middle_of_path(tmp_path, force_args):
    """Verify get_obliqua.sh refuses a destination path that has a regular file as an ancestor."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    afile = tmp_path / 'afile'
    afile.write_text('middle file bytes\n')
    before_names = sorted(p.name for p in tmp_path.iterdir())
    res = _run_tool(site, 'get_obliqua.sh', *force_args, f'{afile}/x/Obliqua')
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {afile} is a file, a broken link or a folder that cannot be entered; move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == before_names
    assert afile.is_file()
    assert afile.read_text() == 'middle file bytes\n'
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
@pytest.mark.parametrize('force_args', [(), ('--force',)], ids=['no-force', 'force'])
def test_obliqua_clones_into_nested_missing_destination(tmp_path, force_args):
    """Verify get_obliqua.sh clones into a missing destination two levels below an existing folder."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    target = tmp_path / 'a' / 'b' / 'Obliqua'
    res = _run_tool(site, 'get_obliqua.sh', *force_args, str(target))
    assert res.returncode == 0, res.stdout + res.stderr
    assert target.is_dir()
    assert git(target, 'rev-parse', 'HEAD') == site['target_sha']
    assert _temp_or_old_entries(tmp_path) == []


@pytest.mark.unit
@pytest.mark.parametrize('force_args', [(), ('--force',)], ids=['no-force', 'force'])
def test_obliqua_refuses_destination_with_trailing_line_break(tmp_path, force_args):
    """Verify get_obliqua.sh refuses a destination with a trailing line break without touching sibling folders."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    d = tmp_path / 'd'
    d.mkdir()
    nl_clone = d / 'nl'
    assert _run_tool(site, 'get_obliqua.sh', str(nl_clone)).returncode == 0
    clone_marker = nl_clone / 'clone_marker.txt'
    clone_marker.write_text('clone marker bytes\n')
    clone_head = git(nl_clone, 'rev-parse', 'HEAD')
    nl_newline_str = str(d) + '/nl\n'
    os.mkdir(nl_newline_str)
    nl_newline_marker = os.path.join(nl_newline_str, 'newline_marker.txt')
    with open(nl_newline_marker, 'w', encoding='utf-8') as f:
        f.write('newline marker bytes\n')
    site['log'].write_text('')
    res = _run_tool(site, 'get_obliqua.sh', *force_args, nl_newline_str)
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: the destination holds a line break; nothing is changed' in res.stderr
    assert clone_marker.read_text() == 'clone marker bytes\n'
    assert git(nl_clone, 'rev-parse', 'HEAD') == clone_head
    with open(nl_newline_marker, encoding='utf-8') as f:
        assert f.read() == 'newline marker bytes\n'
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_refuses_destination_with_line_break_in_middle(tmp_path):
    """Verify get_obliqua.sh refuses a destination with a line break in the middle of the path."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    d = tmp_path / 'd'
    d.mkdir()
    dest_arg = str(d) + '/a\nb/Obliqua'
    res = _run_tool(site, 'get_obliqua.sh', dest_arg)
    assert res.returncode == 1, res.stdout + res.stderr
    assert 'ERROR: the destination holds a line break; nothing is changed' in res.stderr
    assert list(d.iterdir()) == []
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_force_resolves_symlink_before_dotdot_to_clone(tmp_path):
    """Verify --force resolves a folder symlink physically before .. and replaces the physical clone."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    t = tmp_path / 't'
    sub = t / 'real' / 'sub'
    sub.mkdir(parents=True)
    link = t / 'link'
    link.symlink_to(sub)
    clone = t / 'real' / 'clone'
    assert _run_tool(site, 'get_obliqua.sh', str(clone)).returncode == 0
    git(clone, 'checkout', '-q', site['tip_sha'])
    marker = clone / 'marker.txt'
    marker.write_text('off-pin marker\n')
    res = _run_tool(site, 'get_obliqua.sh', '--force', f'{t}/link/../clone')
    assert res.returncode == 0, res.stdout + res.stderr
    assert git(clone, 'rev-parse', 'HEAD') == site['target_sha']
    assert not marker.exists()
    assert not (t / 'clone').exists()
    assert _temp_or_old_entries(tmp_path) == []


@pytest.mark.unit
@pytest.mark.parametrize('force_args', [(), ('--force',)], ids=['no-force', 'force'])
def test_obliqua_resolves_symlink_before_dotdot_and_refuses_non_obliqua_folder(
    tmp_path, force_args
):
    """Verify symlink followed by .. resolves to the physical target and refuses a non-Obliqua folder."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    t = tmp_path / 't'
    sub = t / 'real' / 'sub'
    sub.mkdir(parents=True)
    link = t / 'link'
    link.symlink_to(sub)
    other = t / 'real' / 'other'
    other.mkdir(parents=True)
    user_file = other / 'user.txt'
    user_file.write_text('keep other file\n')
    res = _run_tool(site, 'get_obliqua.sh', *force_args, f'{t}/link/../other')
    assert res.returncode == 1, res.stdout + res.stderr
    assert (
        f'ERROR: {other} is not an Obliqua project (no Project.toml that names Obliqua); move it away or delete it by hand; it is unchanged'
        in res.stderr
    )
    assert user_file.read_text() == 'keep other file\n'
    assert not (t / 'other').exists()
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''


@pytest.mark.unit
def test_obliqua_worktree_off_pin_advises_move_without_force_and_refuses_force(tmp_path):
    """Verify an off-pin Obliqua worktree advises moving or deleting without --force, and refuses --force."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    clone = tmp_path / 'base_clone'
    assert _run_tool(site, 'get_obliqua.sh', str(clone)).returncode == 0
    wt = tmp_path / 'wt'
    git(clone, 'worktree', 'add', '-q', '--detach', str(wt), site['tip_sha'])
    site['log'].write_text('')
    res = _run_tool(site, 'get_obliqua.sh', str(wt))
    assert res.returncode == 0, res.stdout + res.stderr
    lines = [ln.strip() for ln in res.stdout.splitlines()]
    idx_fetch = lines.index(f'git -C {wt} fetch origin')
    idx_checkout = lines.index(f'git -C {wt} checkout --detach {site["target_sha"]}')
    idx_or = lines.index('Or move it away or delete it by hand, then run:')
    assert idx_fetch < idx_checkout < idx_or
    assert lines.count(f'bash tools/get_obliqua.sh {wt}') == 2
    assert '--force' not in res.stdout
    assert '--project=. -e using Pkg; Pkg.instantiate()' in site['log'].read_text().splitlines()

    site['log'].write_text('')
    res_force = _run_tool(site, 'get_obliqua.sh', '--force', str(wt))
    assert res_force.returncode == 1, res_force.stdout + res_force.stderr
    assert (
        f'ERROR: {wt} is a folder whose .git is a file (a git worktree or a submodule); move it away or delete it by hand; it is unchanged'
        in res_force.stderr
    )
    assert site['log'].read_text() == ''


@pytest.mark.unit
@pytest.mark.parametrize('force_args', [(), ('--force',)], ids=['no-force', 'force'])
def test_obliqua_resolves_symlink_before_dotdot_when_name_exists_beside_link(
    tmp_path, force_args
):
    """Verify symlink followed by .. clones into the physical target when the name exists beside the link."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    t = tmp_path / 't'
    sub = t / 'real' / 'sub'
    sub.mkdir(parents=True)
    link = t / 'link'
    link.symlink_to(sub)
    shell_name = t / 'name'
    shell_name.mkdir(parents=True)
    marker = shell_name / 'marker.txt'
    marker.write_text('beside link bytes\n')
    before_names = sorted(p.name for p in shell_name.iterdir())
    kernel_name = t / 'real' / 'name'
    res = _run_tool(site, 'get_obliqua.sh', *force_args, f'{t}/link/../name')
    assert res.returncode == 0, res.stdout + res.stderr
    assert kernel_name.is_dir()
    assert git(kernel_name, 'rev-parse', 'HEAD') == site['target_sha']
    assert sorted(p.name for p in shell_name.iterdir()) == before_names
    assert marker.read_text() == 'beside link bytes\n'
    assert _temp_or_old_entries(tmp_path) == []


@pytest.mark.unit
@pytest.mark.parametrize('force_args', [(), ('--force',)], ids=['no-force', 'force'])
def test_obliqua_refuses_symlink_to_folder_ending_with_line_break(tmp_path, force_args):
    """Verify get_obliqua.sh refuses a symlink to a folder whose name ends with a line break."""
    site = _obliqua_site(tmp_path)
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    site['env']['HOME'] = str(fake_home)
    d = tmp_path / 'd'
    d.mkdir()
    nl_clone = d / 'nl'
    assert _run_tool(site, 'get_obliqua.sh', str(nl_clone)).returncode == 0
    clone_marker = nl_clone / 'clone_marker.txt'
    clone_marker.write_text('clone marker bytes\n')
    clone_head = git(nl_clone, 'rev-parse', 'HEAD')
    nl_newline_str = str(d) + '/nl\n'
    os.mkdir(nl_newline_str)
    nl_newline_marker = os.path.join(nl_newline_str, 'newline_marker.txt')
    with open(nl_newline_marker, 'w', encoding='utf-8') as f:
        f.write('newline marker bytes\n')
    link = d / 'link'
    os.symlink(nl_newline_str, str(link))
    site['log'].write_text('')
    res = _run_tool(site, 'get_obliqua.sh', *force_args, str(link))
    assert res.returncode == 1, res.stdout + res.stderr
    assert f'ERROR: {link} resolves to another folder; nothing is changed' in res.stderr
    assert clone_marker.read_text() == 'clone marker bytes\n'
    assert git(nl_clone, 'rev-parse', 'HEAD') == clone_head
    with open(nl_newline_marker, encoding='utf-8') as f:
        assert f.read() == 'newline marker bytes\n'
    assert link.is_symlink()
    assert _temp_or_old_entries(tmp_path) == []
    assert site['log'].read_text() == ''
