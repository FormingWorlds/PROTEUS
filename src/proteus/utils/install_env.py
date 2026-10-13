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
from pathlib import Path

ENV_FILE_NAME = '.proteus-env.sh'


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
