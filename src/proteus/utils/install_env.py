"""Record the install variables of PROTEUS without editing a shell start file.

``install.sh`` (``set_install_var``) and ``proteus install-all --export-env``
use the same two places: the active conda env other than ``base``, through
``conda env config vars set``, and the exports file ``<PROTEUS root>/.proteus-env.sh``
of ``export NAME='value'`` lines, which a shell without a conda env sources.
A value with a single quote goes to the exports file only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

ENV_FILE_NAME = '.proteus-env.sh'
REQUIRED_JULIA = '1.13'


def julia_depot_entries(value: str | None) -> list[str]:
    """Return the depot folders that Julia reads from a ``JULIA_DEPOT_PATH`` value.

    Julia 1.13 reads an empty first entry as ``~/.julia`` plus the bundled
    depots, and a later empty entry as the bundled depots alone, so that
    ``/x:`` does not read ``~/.julia``. The bundled depots are not listed.

    Parameters
    ----------
    value : str or None
        The value; None when the variable is not set.

    Returns
    -------
    list of str
        The folders in order, without repeats, empty for an empty value.
    """
    if value is None:
        value = ':'
    if not value:
        return []
    parts = value.split(':')
    if not parts[0]:
        parts[0] = '~/.julia'
    return list(dict.fromkeys(os.path.expanduser(part) for part in parts if part))


def depot_of_other_install(value: str | None, root: Path) -> bool:
    """Return whether a depot value is the ``julia_depot`` of another PROTEUS install.

    That is a first entry named ``julia_depot`` beside ``conda-meta``, in a
    PROTEUS tree or in a folder that is gone from an ``envs`` folder that exists
    (a removed conda or pixi env), other than the active env and the tree
    ``root``. ``install.sh`` applies the same rule. Any other missing folder can
    still come, for example on a disk that is not mounted, and stays.
    Conda keeps the value of the previous env when the next env has none, so
    such a value is inherited and not chosen.

    Parameters
    ----------
    value : str or None
        Value of ``JULIA_DEPOT_PATH``.
    root : Path
        PROTEUS source tree of this install.

    Returns
    -------
    bool
        True for a depot that belongs to another install or to one that is gone.
    """
    entries = julia_depot_entries(value)
    if not entries or Path(entries[0]).name != 'julia_depot':
        return False
    parent = Path(entries[0]).parent
    own = {root.resolve(), Path(os.environ.get('CONDA_PREFIX') or root).resolve()}
    if parent.resolve() in own:
        return False
    return (
        (not parent.exists() and parent.parent.name == 'envs' and parent.parent.is_dir())
        or (parent / 'conda-meta').is_dir()
        or ((parent / 'install.sh').is_file() and (parent / 'src' / 'proteus').is_dir())
    )


def other_depot_words(value: str | None) -> str:
    """Name what the depot of ``depot_of_other_install`` is, for a message.

    Parameters
    ----------
    value : str or None
        Value of ``JULIA_DEPOT_PATH``, with at least one entry.

    Returns
    -------
    str
        The first entry and what it belongs to.
    """
    first = Path(julia_depot_entries(value)[0])
    if first.parent.exists():
        return f'{first} is the depot of another PROTEUS install'
    return f'{first} is the depot of an environment that is gone'


def own_julia_depot() -> str | None:
    """Return the depot value of the active env, the one ``install.sh`` gives it.

    Returns
    -------
    str or None
        ``<CONDA_PREFIX>/julia_depot:``, or None with no env or with ``base``,
        which PROTEUS does not install into.
    """
    prefix = os.environ.get('CONDA_PREFIX')
    if not prefix or os.environ.get('CONDA_DEFAULT_ENV') == 'base':
        return None
    return f'{prefix}/julia_depot:'


def depot_value_is_unusable(value: str | None) -> bool:
    """Return whether the first entry of a depot value names no fixed folder.

    A relative entry depends on the folder of each process, ``~user`` is not
    expanded by Julia and a literal ``$`` was never expanded. ``install.sh``
    stops on the same forms. An empty first entry is ``~/.julia``.

    Parameters
    ----------
    value : str or None
        Value of ``JULIA_DEPOT_PATH``.

    Returns
    -------
    bool
        True for a first entry that names no fixed folder.
    """
    first = (value or '').split(':')[0]
    fixed = not first or first == '~' or first.startswith(('/', '~/'))
    return '$' in first or not fixed


def missing_julia_packages(project: Path, entries: list[str]) -> list[str]:
    """Return the direct dependencies of a Julia project that no depot entry holds.

    The check is a folder ``packages/<name>`` in any entry, so another version
    of a package passes. A stdlib and a package added with ``Pkg.develop`` have
    no ``git-tree-sha1`` in the Manifest and no such folder, so they do not count.
    As in Julia, ``Manifest-v<major>.<minor>.toml`` of the required Julia wins
    over ``Manifest.toml``.

    Parameters
    ----------
    project : Path
        Folder of the Julia project.
    entries : list of str
        Depot folders, from ``julia_depot_entries``.

    Returns
    -------
    list of str
        Package names, empty with no Manifest.

    Raises
    ------
    ValueError
        When the ``Project.toml`` or the Manifest cannot be read.
    """
    path = project / f'Manifest-v{REQUIRED_JULIA}.toml'
    if not path.is_file():
        path = project / 'Manifest.toml'
    if not path.is_file():
        return []
    try:
        manifest = tomllib.loads(path.read_bytes().decode())
        direct = tomllib.loads((project / 'Project.toml').read_bytes().decode()).get('deps', {})
    except (OSError, ValueError) as exc:
        raise ValueError(
            f'{project}: Project.toml or Manifest file cannot be read: {exc}'
        ) from exc
    table = manifest.get('deps', manifest)
    return [
        name
        for name in direct
        if any('git-tree-sha1' in item for item in table.get(name, []))
        and not any((Path(entry) / 'packages' / name).is_dir() for entry in entries)
    ]


def _sh_quote(value: str) -> str:
    """Quote ``value`` in single quotes for a POSIX shell, as ``install.sh`` does.

    Parameters
    ----------
    value : str
        Value to quote.

    Returns
    -------
    str
        The value in single quotes, each ``'`` written as ``'\\''``.
    """
    return "'" + value.replace("'", "'\\''") + "'"


def write_env_file(root: Path, variables: dict[str, str]) -> Path:
    """Write one ``export NAME='value'`` line per variable to the exports file.

    A line of the same variable is replaced and the other lines stay. The file
    is rewritten in place, so its inode, mode and any symlink to it remain.

    Parameters
    ----------
    root : Path
        PROTEUS source tree that holds the exports file.
    variables : dict[str, str]
        Variable names and values.

    Returns
    -------
    Path
        Path of the exports file.
    """
    path = root / ENV_FILE_NAME
    kept: list[str] = []
    if path.is_file():
        prefixes = tuple(f'export {name}=' for name in variables)
        kept = [ln for ln in path.read_text().splitlines() if not ln.startswith(prefixes)]
    lines = kept + [f'export {name}={_sh_quote(value)}' for name, value in variables.items()]
    path.write_text('\n'.join(lines) + '\n')
    return path


def active_conda_prefix() -> str | None:
    """Return the prefix of the active conda env that ``conda`` can configure.

    Returns
    -------
    str or None
        ``CONDA_PREFIX``, or None when it is unset, when ``conda`` is not on
        ``PATH``, when the env is ``base`` (``CONDA_DEFAULT_ENV``), which
        PROTEUS does not install into, or when the env is a pixi env
        (``PIXI_ENVIRONMENT_NAME`` set), whose activation does not read conda
        env variables.
    """
    prefix = os.environ.get('CONDA_PREFIX')
    if not prefix or os.environ.get('PIXI_ENVIRONMENT_NAME'):
        return None
    if os.environ.get('CONDA_DEFAULT_ENV') == 'base' or shutil.which('conda') is None:
        return None
    return prefix


def quoted_names(variables: dict[str, str]) -> list[str]:
    """Return the names of the variables whose value holds a single quote.

    conda writes a variable into its activation script as ``export NAME='value'``
    without escaping, so such a value breaks each later activation of the env.

    Parameters
    ----------
    variables : dict[str, str]
        Variable names and values.

    Returns
    -------
    list[str]
        The names, in the order of ``variables``.
    """
    return [name for name, value in variables.items() if "'" in value]


def bind_to_conda_env(prefix: str, variables: dict[str, str]) -> bool:
    """Set the ``variables`` in the conda env at ``prefix`` in one conda call.

    A variable of ``quoted_names`` is left out; it stays in the exports file.

    Parameters
    ----------
    prefix : str
        Prefix of the conda env.
    variables : dict[str, str]
        Variable names and values.

    Returns
    -------
    bool
        Whether ``conda env config vars set`` succeeded; False without a call
        when no variable is left.
    """
    skipped = quoted_names(variables)
    pairs = [f'{name}={value}' for name, value in variables.items() if name not in skipped]
    if not pairs:
        return False
    cmd = ['conda', 'env', 'config', 'vars', 'set', '-p', prefix, *pairs]
    try:
        subprocess.run(
            cmd, check=True, stdin=subprocess.DEVNULL, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def reload_instruction(bound: bool) -> str:
    """Return the command that loads the recorded variables into a shell.

    Conda sets env variables at activation, and activating the env that is
    already active does not reload them, so the env is deactivated first.

    Parameters
    ----------
    bound : bool
        Whether the variables are bound to the active conda env.

    Returns
    -------
    str
        The reactivation command, or the ``source`` command of the exports file.
    """
    if bound:
        env = os.environ.get('CONDA_DEFAULT_ENV') or os.environ.get('CONDA_PREFIX', '<env>')
        return f'conda deactivate && conda activate {env}'
    return f'source {ENV_FILE_NAME}'


def export_install_vars(root: Path, variables: dict[str, str]) -> tuple[Path, str | None, bool]:
    """Record ``variables`` in the exports file and in the active conda env.

    Parameters
    ----------
    root : Path
        PROTEUS source tree that holds the exports file.
    variables : dict[str, str]
        Variable names and values.

    Returns
    -------
    tuple[Path, str or None, bool]
        The exports file, the prefix of the active conda env (None without
        one, and for ``base``), and whether the conda call succeeded.
    """
    path = write_env_file(root, variables)
    prefix = active_conda_prefix()
    bound = prefix is not None and bind_to_conda_env(prefix, variables)
    return path, prefix, bound
