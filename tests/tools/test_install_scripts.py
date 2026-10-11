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
- ``guard_dirty_checkout``: the refusal to delete local work or a folder
  that is not a git clone, a commit at a tag of the remote counted as
  known, and the pathspec exclusion ``get_socrates.sh`` passes it
- ``swap_in_checkout``: the old checkout leaves its path only once the new
  one is there, and comes back when the new one cannot move in
- ``move_clone_to_floor_tag``: a clone behind the floor tag moves forward
  in place with its untracked files and stash, any other clone stays
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
new clone lands on, that a failed replacement keeps a clean clone, and that
``get_python_module.sh`` moves an existing clone in place.

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
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest
from helpers import git, scratch_proteus_root

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
            if 'dirname' in code and not re.match(BOOTSTRAP_LINE, code):
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
    deletes = re.compile(r'^\s*rm -rf "\$[a-z_]+"')
    sources = {p.name: p.read_text().splitlines() for p in _get_scripts()}
    refreshing = {
        name: lines
        for name, lines in sources.items()
        if any(deletes.search(ln) for ln in lines) and any('git clone' in ln for ln in lines)
    }

    # Guard the guard: discovery has to find the installers that replace a
    # sibling checkout, and leave out the one that unpacks an archive and so
    # has no git history to lose.
    assert set(refreshing) >= {'get_morrigan.sh', 'get_boreas.sh', 'get_socrates.sh'}, sorted(
        refreshing
    )
    assert 'get_petsc.sh' not in refreshing, 'the archive installer holds no git history'
    assert len(refreshing) >= 8, sorted(refreshing)

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


@pytest.mark.unit
def test_morrigan_guard_protects_its_own_checkout(tmp_path):
    """The accretion installer guards the ``Morrigan/`` checkout it deletes.

    ``tools/get_morrigan.sh`` refreshes a sibling clone a developer may also
    be working in. It calls the shared guard, so what has to be right here is
    what it passes: the Morrigan path and its own name for the recovery hint.
    A call copied from another installer would still guard, but would inspect
    the wrong tree and pass silently on a dirty Morrigan checkout.
    """
    call = next(
        ln
        for ln in (TOOLS_DIR / 'get_morrigan.sh').read_text().splitlines()
        if ln.startswith('guard_dirty_checkout')
    )
    assert '"$workpath"' in call and 'get_morrigan.sh' in call, call
    # Discrimination: a call lifted from another installer names its tree.
    assert 'BOREAS' not in call and 'aragog' not in call
    assert 'workpath="$root/Morrigan/"' in (TOOLS_DIR / 'get_morrigan.sh').read_text()

    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    _git(upstream, 'init', '-q')
    (upstream / 'f.py').write_text('a = 1\n')
    _git(upstream, 'add', 'f.py')
    _git(upstream, 'commit', '-q', '-m', 'c1')

    workdir = tmp_path / 'Morrigan'
    _git(tmp_path, 'clone', '-q', str(upstream), str(workdir))
    _git(workdir, 'checkout', '-q', '--detach', 'HEAD')
    (workdir / 'morrigan.egg-info').write_text('')  # untracked: must not block

    res = _run_guard(tmp_path, checkout='Morrigan', script_name='get_morrigan.sh')
    assert res.returncode == 0, res.stderr
    assert 'GUARD_PASSED' in res.stdout

    # A local-only commit is exactly the state of a developer branch that
    # has not been pushed; refreshing would destroy it.
    (workdir / 'f.py').write_text('a = 2\n')
    _git(workdir, 'add', 'f.py')
    _git(workdir, 'commit', '-q', '-m', 'local work')
    res = _run_guard(tmp_path, checkout='Morrigan', script_name='get_morrigan.sh')
    assert res.returncode == 1
    assert 'not on a remote' in res.stderr
    assert 'get_morrigan.sh --force' in res.stderr
    assert 'GUARD_PASSED' not in res.stdout


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
def test_guard_refuses_a_folder_that_is_not_a_git_clone(tmp_path, strict):
    """A non-empty folder without .git, and a plain file at the path, are refused unless --force.

    The folder holds a file of the user's own; a guard that passes a path
    without .git lets the script delete it. --force still passes.
    """
    plain = tmp_path / 'aragog'
    plain.mkdir()
    (plain / 'notes.txt').write_text('mine\n')
    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 1, res.stdout
    assert f'{plain} is not a git clone, so it is not replaced.' in res.stderr
    assert 'Move it away, or run  bash tools/get_aragog.sh --force' in res.stderr
    assert 'GUARD_PASSED' not in res.stdout
    assert (plain / 'notes.txt').read_text() == 'mine\n'

    forced = _run_guard(tmp_path, '--force', strict=strict)
    assert forced.returncode == 0, forced.stderr
    assert 'GUARD_PASSED' in forced.stdout

    (tmp_path / 'Morrigan').write_text('a file, not a folder\n')
    as_file = _run_guard(tmp_path, checkout='Morrigan', script_name='get_morrigan.sh')
    assert as_file.returncode == 1
    assert 'is not a git clone' in as_file.stderr


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


def _remote_with_tag_on_a_deleted_branch(tmp_path: Path, annotated: bool) -> tuple[Path, str]:
    """Publish a bare remote whose tag 26.04.22 sits on a commit of a deleted branch.

    Returns the remote and the tagged commit, which no branch of the remote holds.
    """
    work = tmp_path / 'upstream'
    work.mkdir()
    git(work, 'init', '-q', '-b', 'main')
    (work / 'f.py').write_text('a = 1\n')
    git(work, 'add', 'f.py')
    git(work, 'commit', '-q', '-m', 'c1')
    git(work, 'checkout', '-q', '-b', 'release')
    (work / 'f.py').write_text('a = 2\n')
    git(work, 'commit', '-q', '-am', 'release commit')
    if annotated:
        git(work, 'tag', '-a', '26.04.22', '-m', 'release')
    else:
        git(work, 'tag', '26.04.22')
    tagged = git(work, 'rev-parse', 'HEAD')
    git(work, 'checkout', '-q', 'main')
    git(work, 'branch', '-qD', 'release')
    remote = tmp_path / 'remote.git'
    git(tmp_path, 'clone', '-q', '--bare', str(work), str(remote))
    return remote, tagged


@pytest.mark.parametrize('annotated', [False, True], ids=['lightweight tag', 'annotated tag'])
@pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
def test_guard_passes_a_clone_detached_at_a_tag_on_no_remote_branch(
    tmp_path, strict, annotated
):
    """A clean clone detached at a release tag of its remote is clean, though no remote branch
    holds the tagged commit; a commit made on top of it is still local work."""
    remote, tagged = _remote_with_tag_on_a_deleted_branch(tmp_path, annotated)
    clone = tmp_path / 'aragog'
    git(tmp_path, 'clone', '-q', str(remote), str(clone))
    git(clone, 'checkout', '-q', 'tags/26.04.22')
    assert git(clone, 'rev-parse', 'HEAD') == tagged
    assert git(clone, 'branch', '-r', '--contains', tagged) == ''

    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 0, res.stderr
    assert 'GUARD_PASSED' in res.stdout

    (clone / 'f.py').write_text('a = 3\n')
    git(clone, 'commit', '-q', '-am', 'local work')
    res = _run_guard(tmp_path, strict=strict)
    assert res.returncode == 1
    assert 'not on a remote' in res.stderr


# ---------------------------------------------------------------------------
# swap_in_checkout (tools/_get_common.sh)
# ---------------------------------------------------------------------------


def _run_swap(new: Path, dest: Path, strict: bool) -> subprocess.CompletedProcess:
    """Run the shipped swap_in_checkout and print its status as ``SWAP_STATUS <n>``."""
    body = f'rc=0\nswap_in_checkout "{new}" "{dest}" || rc=$?\necho "SWAP_STATUS $rc"\n'
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
        '{root}/Obliqua',
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


def _bare_remote(remotes: Path, repo: str, tag: str) -> tuple[str, str]:
    """Publish ``remotes/FormingWorlds/<repo>.git`` with 3 commits and ``tag`` on the second.

    Every commit carries ``version.txt`` and ``fastchem_vulcan/Makefile``, so
    the VULCAN build step finds its directory. Returns the tagged commit and
    the tip, which differ, so a clone left on the branch is told apart.
    """
    work = remotes.parent / 'work' / repo
    (work / 'fastchem_vulcan').mkdir(parents=True)
    (work / 'fastchem_vulcan' / 'Makefile').write_text('all:\n')
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


def _floor_tag_site(tmp_path: Path, pyproject: str, remotes: dict[str, str]) -> dict:
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
        repo: _bare_remote(base / 'FormingWorlds', repo, tag) for repo, tag in remotes.items()
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


def _run_tool(site: dict, script: str, *argv: str) -> subprocess.CompletedProcess:
    """Run ``tools/<script>`` of the site copy with ``argv``, from outside the copy."""
    return subprocess.run(
        ['bash', str(site['root'] / 'tools' / script), *argv],
        cwd=site['root'].parent,
        env=site['env'],
        capture_output=True,
        text=True,
        timeout=60,
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

        The tip is one commit past the tag, so a clone left on the default
        branch fails the equality; the floor text 26.10.6 is not the tag
        name, so an exact-text checkout fails the run.
        """
        tagged, tip = site['shas']['MORS']
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

    def test_second_run_moves_the_clone_in_place_and_keeps_local_files(self, site):
        """A second run without --force moves a clone behind the tag forward in place.

        The clone sits one commit below the tag with an untracked file and a
        stash entry; it ends at the tag with both kept and is installed
        again. A script that clones anew deletes both, and one that pulls
        lands on the tip.
        """
        tagged, tip = site['shas']['MORS']
        clone = site['root'] / 'MORS'
        assert _run_tool(site, 'get_python_module.sh', 'MORS').returncode == 0
        git(clone, 'checkout', '-q', '--detach', 'HEAD~1')
        (clone / 'version.txt').write_text('stashed work\n')
        git(clone, 'stash', '-q')
        (clone / 'notes.txt').write_text('kept\n')

        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == tagged != tip
        assert (clone / 'notes.txt').read_text() == 'kept\n'
        assert len(git(clone, 'stash', 'list').splitlines()) == 1
        assert 'stashed work' in git(clone, 'stash', 'show', '-p')
        assert f'fwl-mors: checked out release tag 26.10.06 at {tagged[:7]}' in result.stdout
        assert 'Cloning' not in result.stdout
        assert site['log'].read_text().splitlines() == [f'install -U -e {clone}'] * 2

    def test_plain_folder_is_refused_without_force(self, site):
        """A MORS folder that is not a git clone is refused with exit 1 and the --force command;
        its file stays, nothing is cloned and pip is not called."""
        plain = site['root'] / 'MORS'
        plain.mkdir()
        (plain / 'notes.txt').write_text('mine\n')
        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 1
        assert f'{plain} is not a git clone' in result.stderr
        assert 'bash tools/get_python_module.sh MORS --force' in result.stderr
        assert sorted(p.name for p in plain.iterdir()) == ['notes.txt']
        assert site['log'].read_text() == ''

    def test_worktree_with_a_modified_file_is_kept(self, site, tmp_path):
        """A MORS checkout that is a git worktree (.git is a file) with a modified
        tracked file is refused with exit 1; the file keeps its change and pip
        is not called. A guard that looks for a .git directory deletes it."""
        source = tmp_path / 'mors-source'
        source.mkdir()
        git(source, 'init', '-q', '-b', 'main')
        (source / 'version.txt').write_text('0\n')
        git(source, 'add', 'version.txt')
        git(source, 'commit', '-q', '-m', 'commit 0')
        clone = site['root'] / 'MORS'
        git(source, 'worktree', 'add', '-q', '--detach', str(clone))
        assert (clone / '.git').is_file()
        (clone / 'version.txt').write_text('local work\n')

        result = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert result.returncode == 1
        assert 'uncommitted changes' in result.stderr
        assert 'bash tools/get_python_module.sh MORS --force' in result.stderr
        assert (clone / 'version.txt').read_text() == 'local work\n'
        assert site['log'].read_text() == ''

    def test_modified_checkout_is_refused_then_replaced_with_force(self, site):
        """A clone with a modified tracked file is kept without --force and replaced with it.

        The refusal names the recovery command with the module name; with
        --force the clone returns to the tagged content.
        """
        tagged, _ = site['shas']['MORS']
        clone = site['root'] / 'MORS'
        assert _run_tool(site, 'get_python_module.sh', 'MORS').returncode == 0
        (clone / 'version.txt').write_text('local work\n')

        refused = _run_tool(site, 'get_python_module.sh', 'MORS')
        assert refused.returncode == 1
        assert 'bash tools/get_python_module.sh MORS --force' in refused.stderr
        assert (clone / 'version.txt').read_text() == 'local work\n'
        assert len(site['log'].read_text().splitlines()) == 1

        forced = _run_tool(site, 'get_python_module.sh', 'MORS', '--force')
        assert forced.returncode == 0, forced.stdout + forced.stderr
        assert (clone / 'version.txt').read_text() == '1\n'
        assert git(clone, 'rev-parse', 'HEAD') == tagged
        assert len(site['log'].read_text().splitlines()) == 2

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
    clone left on the default branch lands on the tip.
    """
    pyproject = f'[project]\nname = "fixture"\ndependencies = ["{dist}>=26.4.22"]\n'
    site = _floor_tag_site(tmp_path, pyproject, {repo: '26.04.22'})
    tagged, tip = site['shas'][repo]
    result = _run_tool(site, script)
    assert result.returncode == 0, result.stdout + result.stderr
    clone = site['root'] / repo
    assert git(clone, 'rev-parse', 'HEAD') == tagged != tip
    assert _is_detached(clone)
    assert f'{dist}: checked out release tag 26.04.22 at {tagged[:7]}' in result.stdout
    assert site['log'].read_text().splitlines() == ['install -U -e .']


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


_ALL_FLOORS_PYPROJECT = (
    '[project]\nname = "fixture"\n'
    'dependencies = ["fwl-mors>=26.4.22", "fwl-aragog>=26.4.22", "fwl-zalmoxis>=26.4.22"]\n'
    '[project.optional-dependencies]\n'
    'vulcan = ["fwl-vulcan>=26.4.22"]\nmorrigan = ["fwl-morrigan>=26.4.22"]\n'
)


@pytest.mark.parametrize('cause', ['failing remote', 'floor without a tag'])
@pytest.mark.parametrize(
    ('script', 'argv', 'repo'),
    [
        ('get_python_module.sh', ('MORS', '--force'), 'MORS'),
        ('get_aragog.sh', (), 'aragog'),
        ('get_zalmoxis.sh', (), 'Zalmoxis'),
        ('get_vulcan.sh', (), 'VULCAN'),
        ('get_morrigan.sh', (), 'Morrigan'),
    ],
    ids=['mors replaced with force', 'aragog', 'zalmoxis', 'vulcan', 'morrigan'],
)
def test_clean_clone_survives_a_failed_replacement(tmp_path, script, argv, repo, cause):
    """A clean clone at its tag stays as it was when the new clone fails.

    The remote is moved away, or the floor is raised to 26.11.1, which no
    tag carries. HEAD stays byte-identical, an untracked file stays, pip
    runs only for the first install and no temporary sibling is left; a
    script that deletes the checkout before it clones leaves none at all.
    """
    site = _floor_tag_site(tmp_path, _ALL_FLOORS_PYPROJECT, {repo: '26.04.22'})
    tagged, _ = site['shas'][repo]
    first = _run_tool(site, script, *argv)
    assert first.returncode == 0, first.stdout + first.stderr
    clone = site['root'] / repo
    (clone / 'notes.txt').write_text('kept\n')
    head_file = (clone / '.git' / 'HEAD').read_bytes()
    if cause == 'failing remote':
        remote = tmp_path / 'remotes' / 'FormingWorlds' / f'{repo}.git'
        remote.rename(remote.with_name(f'{repo}.moved'))
        expected = 'git clone failed; '
    else:
        pyproject = site['root'] / 'pyproject.toml'
        pyproject.write_text(pyproject.read_text().replace('26.4.22', '26.11.1'))
        expected = 'floor 26.11.1: no tag matches it'

    result = _run_tool(site, script, *argv)
    assert result.returncode == 1
    assert expected in result.stderr
    assert (clone / '.git' / 'HEAD').read_bytes() == head_file
    assert git(clone, 'rev-parse', 'HEAD') == tagged
    assert (clone / 'notes.txt').read_text() == 'kept\n'
    assert len(site['log'].read_text().splitlines()) == 1
    assert sorted(p.name for p in site['root'].iterdir()) == [repo, 'pyproject.toml', 'tools']


class TestMoveCloneToFloorTag:
    """move_clone_to_floor_tag moves a clone behind the floor tag forward in place, and only that."""

    @pytest.fixture
    def mover(self, tmp_path):
        """A MORS clone of a bare remote: 3 commits on main, the tag 26.10.06 on the second,
        and a branch ``side`` with one commit off the first; the floor is 26.10.6."""
        root = scratch_proteus_root(tmp_path / 'proteus', ['fwl-mors>=26.10.6'])
        work = tmp_path / 'work'
        work.mkdir()
        git(work, 'init', '-q', '-b', 'main')
        commits = []
        for i in range(3):
            (work / 'version.txt').write_text(f'{i}\n')
            git(work, 'add', 'version.txt')
            git(work, 'commit', '-q', '-m', f'commit {i}')
            commits.append(git(work, 'rev-parse', 'HEAD'))
        git(work, 'tag', '26.10.06', commits[1])
        git(work, 'checkout', '-q', '-b', 'side', commits[0])
        (work / 'version.txt').write_text('side\n')
        git(work, 'commit', '-q', '-am', 'side commit')
        side = git(work, 'rev-parse', 'HEAD')
        git(work, 'checkout', '-q', 'main')
        remote = tmp_path / 'MORS.git'
        git(tmp_path, 'clone', '-q', '--bare', str(work), str(remote))
        clone = tmp_path / 'MORS'
        git(tmp_path, 'clone', '-q', str(remote), str(clone))
        shas = {'first': commits[0], 'tagged': commits[1], 'tip': commits[2], 'side': side}
        return {'root': root, 'clone': clone, 'remote': remote, 'shas': shas}

    @staticmethod
    def _move(mover, strict: bool = True) -> subprocess.CompletedProcess:
        """Source the copied library and run move_clone_to_floor_tag fwl-mors on the clone."""
        prelude = 'set -euo pipefail\n' if strict else ''
        snippet = (
            f'{prelude}source "{mover["root"]}/tools/_get_common.sh"\n'
            f'move_clone_to_floor_tag fwl-mors "{mover["clone"]}"\necho AFTER\n'
        )
        return subprocess.run(
            ['bash', '-c', snippet], capture_output=True, text=True, timeout=20
        )

    @staticmethod
    def _reflog_length(clone: Path) -> int:
        """Return the number of entries in the HEAD reflog of ``clone``."""
        return len(git(clone, 'reflog', 'show', 'HEAD').splitlines())

    @pytest.mark.parametrize('strict', [False, True], ids=['plain shell', 'errexit shell'])
    def test_clone_behind_the_tag_moves_forward_and_keeps_local_files(self, mover, strict):
        """A clone detached at the first commit, with an untracked file and a stash entry,
        ends at the tag with both kept; the tip is a commit further, so a pull fails."""
        clone, shas = mover['clone'], mover['shas']
        git(clone, 'checkout', '-q', '--detach', shas['first'])
        (clone / 'version.txt').write_text('stashed work\n')
        git(clone, 'stash', '-q')
        (clone / 'notes.txt').write_text('kept\n')
        result = self._move(mover, strict)
        assert result.returncode == 0, result.stderr
        assert result.stdout.rstrip().endswith('AFTER')
        assert git(clone, 'rev-parse', 'HEAD') == shas['tagged'] != shas['tip']
        assert (clone / 'notes.txt').read_text() == 'kept\n'
        assert len(git(clone, 'stash', 'list').splitlines()) == 1
        assert f'fwl-mors: checked out release tag 26.10.06 at {shas["tagged"][:7]}' in (
            result.stdout
        )

    def test_clone_at_the_tag_is_left_untouched(self, mover):
        """A clone at the tag keeps its HEAD and its reflog length, and no checkout runs.

        git writes no reflog entry for a checkout of the commit HEAD is on, so
        the missing checkout message is what tells a redundant checkout apart.
        """
        clone, shas = mover['clone'], mover['shas']
        git(clone, 'checkout', '-q', 'tags/26.10.06')
        before = self._reflog_length(clone)
        result = self._move(mover)
        assert result.returncode == 0, result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == shas['tagged']
        assert self._reflog_length(clone) == before
        assert f'fwl-mors: {clone} is at the release tag 26.10.06' in result.stdout
        assert 'checked out release tag' not in result.stdout

    @pytest.mark.parametrize('state', ['tip', 'side'], ids=['ahead of the tag', 'diverged'])
    def test_clone_not_behind_the_tag_stays(self, mover, state):
        """A clone past the tag, or on another line of history, stays where it is with one
        line that says so; a helper without the ancestor test moves it back to the tag."""
        clone, shas = mover['clone'], mover['shas']
        git(clone, 'checkout', '-q', '--detach', shas[state])
        before = self._reflog_length(clone)
        result = self._move(mover)
        assert result.returncode == 0, result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == shas[state] != shas['tagged']
        assert self._reflog_length(clone) == before
        assert (
            f'fwl-mors: {clone} is not behind the release tag 26.10.06; it stays as it is'
            f' at {shas[state][:7]}'
        ) in result.stdout

    def test_clone_on_a_branch_stays(self, mover):
        """A clone on a branch behind the tag is a developer's and stays on that branch."""
        clone, shas = mover['clone'], mover['shas']
        git(clone, 'checkout', '-q', '-b', 'old', shas['first'])
        result = self._move(mover)
        assert result.returncode == 0, result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == shas['first']
        assert git(clone, 'symbolic-ref', '--short', 'HEAD') == 'old'
        assert f'fwl-mors: {clone} is on branch old; it stays as it is' in result.stdout

    @pytest.mark.parametrize('work', ['local commit', 'modified tracked file'])
    def test_local_work_is_refused(self, mover, work):
        """A local commit, or a modified tracked file, stops the helper with exit 1 before any
        checkout; HEAD and the file keep their state and the caller does not go on."""
        clone, shas = mover['clone'], mover['shas']
        git(clone, 'checkout', '-q', '--detach', shas['first'])
        (clone / 'version.txt').write_text('local work\n')
        if work == 'local commit':
            git(clone, 'commit', '-q', '-am', 'local work')
        head = git(clone, 'rev-parse', 'HEAD')
        result = self._move(mover)
        assert result.returncode == 1
        assert 'AFTER' not in result.stdout
        assert 'Refusing to move it to another commit' in result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == head
        assert (clone / 'version.txt').read_text() == 'local work\n'

    def test_failed_fetch_is_a_warning_and_the_local_tags_serve(self, mover):
        """With the remote gone the fetch fails with a warning, and the tag the clone already
        has still moves it forward."""
        clone, shas = mover['clone'], mover['shas']
        git(clone, 'checkout', '-q', '--detach', shas['first'])
        mover['remote'].rename(mover['remote'].with_name('MORS.moved'))
        result = self._move(mover)
        assert result.returncode == 0, result.stderr
        assert f'WARNING: cannot fetch the tags of {clone}' in result.stderr
        assert git(clone, 'rev-parse', 'HEAD') == shas['tagged']


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
