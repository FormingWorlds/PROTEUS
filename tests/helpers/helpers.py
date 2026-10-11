# Helper functions for running the tests
from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from PIL import Image

PROTEUS_ROOT = Path(__file__).parents[2]
NEGLECT = [
    'N2_mol_atm',
    'N2_mol_total',
    'N2_kg_atm',
    'N2_kg_total',
    'CH4_mol_atm',
    'CH4_mol_total',
    'CH4_kg_atm',
    'CH4_kg_total',
    'CH4_bar',
    'F_net',
    'runtime',
]


def resize_to_match(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Resize img2 to match img1's size if they don't match
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    return img1, img2


def df_intersect(df1, df2):
    """
    Filter two dataframes to only the columns which they have in common.
    """

    # Intersection of columns
    inter = set(df1.columns).intersection(set(df2.columns))
    inter = list(inter)

    # Filter
    out1 = df1.loc[:, df1.columns.isin(inter)]
    out2 = df2.loc[:, df2.columns.isin(inter)]

    # Return
    return out1, out2


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` with a fixed identity and no signing; return its stdout.

    Parameters
    ----------
    cwd : Path
        Working directory of the git call.
    *args : str
        The git arguments, for example ``'rev-parse', 'HEAD'``.

    Returns
    -------
    str
        The standard output, stripped.
    """
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


def scratch_proteus_root(root: Path, dependencies: list[str]) -> Path:
    """Build a PROTEUS root with the module clone tools below a ``pyproject.toml``.

    Parameters
    ----------
    root : Path
        Directory to create; it must not exist yet.
    dependencies : list of str
        The ``[project] dependencies`` of the fixture ``pyproject.toml``.

    Returns
    -------
    Path
        ``root``, holding ``tools/_get_common.sh``, ``tools/_module_pins.py``
        and ``tools/get_python_module.sh`` copied from this tree.
    """
    (root / 'tools').mkdir(parents=True)
    for name in ('_get_common.sh', '_module_pins.py', 'get_python_module.sh'):
        shutil.copy2(PROTEUS_ROOT / 'tools' / name, root / 'tools' / name)
    deps = ''.join(f'    "{d}",\n' for d in dependencies)
    (root / 'pyproject.toml').write_text(
        f'[project]\nname = "fixture"\ndependencies = [\n{deps}]\n'
    )
    return root


# git subcommands that change a repository or its working tree; symbolic-ref writes with 2 operands.
GIT_WRITE_SUBCOMMANDS = (
    'fetch checkout pull reset clean stash merge rebase switch restore commit tag branch gc'
    ' remote config submodule add rm mv apply cherry-pick revert worktree update-ref'
).split()

_GIT_GUARD = """#!/bin/bash
# Logs every git call; refuses with status 97 a write to a guarded repository.
log={log}
printf '%s\\t%s\\n' "$PWD" "$*" >>"$log"
orig=("$@")
repo="$PWD"
sub=""
rest=()
while [ $# -gt 0 ]; do
    case "$1" in
        -C) shift; case "$1" in /*) repo="$1" ;; *) repo="$repo/$1" ;; esac ;;
        -c | --git-dir | --work-tree | --namespace) shift ;;
        -*) ;;
        *) sub="$1"; shift; rest=("$@"); break ;;
    esac
    shift
done
write=false
case " {writes} " in *" $sub "*) write=true ;; esac
if [ "$sub" = symbolic-ref ]; then
    n=0
    for a in "${{rest[@]}}"; do case "$a" in -*) ;; *) n=$((n + 1)) ;; esac; done
    [ "$n" -lt 2 ] || write=true
fi
if [ "$write" = true ]; then
    real=$(cd "$repo" 2>/dev/null && pwd -P) || real="$repo"
    for dir in {exact}; do
        [ "$real" != "$dir" ] || blocked=1
    done
    for dir in {trees}; do
        case "$real/" in "$dir"/*) blocked=1 ;; esac
    done
    if [ -n "${{blocked:-}}" ]; then
        printf 'BLOCKED\\t%s\\t%s\\n' "$real" "${{orig[*]}}" >>"$log"
        : >{marker}
        exit 97
    fi
fi
exec {git} "${{orig[@]}}"
"""


def install_git_guard(
    bindir: Path, log: Path, marker: Path, roots: list[Path], trees: list[Path]
) -> Path:
    """Write a ``git`` wrapper into ``bindir`` that logs every call and refuses guarded writes.

    A call whose subcommand is in ``GIT_WRITE_SUBCOMMANDS``, or ``symbolic-ref``
    with 2 operands, and whose repository (the ``-C`` directories, else the
    working directory) is one of ``roots`` or lies in one of ``trees``, exits
    97 and creates ``marker``; every other call runs the real git.

    Parameters
    ----------
    bindir : Path
        Directory to put first on PATH.
    log : Path
        File that gets one line per call: the working directory and the arguments.
    marker : Path
        File created by a refused call.
    roots : list of Path
        Repositories guarded by exact path, such as the PROTEUS root.
    trees : list of Path
        Folders guarded with everything below them, such as a module clone.

    Returns
    -------
    Path
        The wrapper.
    """
    real_git = shutil.which('git')
    assert real_git and Path(real_git).parent != bindir, 'the real git must be on PATH'

    def words(paths: list[Path]) -> str:
        return ' '.join(shlex.quote(os.path.realpath(p)) for p in paths) or "''"

    wrapper = bindir / 'git'
    wrapper.write_text(
        _GIT_GUARD.format(
            log=shlex.quote(str(log)),
            marker=shlex.quote(str(marker)),
            writes=' '.join(GIT_WRITE_SUBCOMMANDS),
            exact=words(roots),
            trees=words(trees),
            git=shlex.quote(real_git),
        )
    )
    wrapper.chmod(0o755)
    return wrapper


def tree_snapshot(root: Path) -> dict[str, tuple]:
    """Return every entry below ``root``, ``.git`` included, with its kind, size and SHA-256.

    Parameters
    ----------
    root : Path
        Folder to list; symlinks are recorded with their target, not followed.

    Returns
    -------
    dict of str to tuple
        Relative path to ``('dir',)``, ``('link', target)`` or
        ``('file', size, sha256)``.
    """
    snap: dict[str, tuple] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            rel = str(path.relative_to(root))
            if path.is_symlink():
                snap[rel] = ('link', os.readlink(path))
            elif path.is_dir():
                snap[rel] = ('dir',)
            else:
                data = path.read_bytes()
                snap[rel] = ('file', len(data), hashlib.sha256(data).hexdigest())
    return snap
