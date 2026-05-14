from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from functools import partial
from importlib.metadata import PackageNotFoundError
from typing import Callable
from urllib.parse import unquote, urlparse

import click
import requests
from attr import dataclass
from packaging.version import Version

from proteus.utils.coupler import (
    _get_agni_version,
    _get_socrates_version,
    get_proteus_directories,
)

DIRS = get_proteus_directories()

HEADER_STYLE = {'fg': 'yellow', 'underline': True, 'bold': True}
OK_STYLE = {'fg': 'green'}
ERROR_STYLE = {'fg': 'red'}
DEFAULT_STYLE = {'fg': 'yellow'}


@dataclass
class BasePackage:
    name: str

    def current_version(self) -> Version: ...

    def latest_version(self) -> Version: ...

    def get_status_message(self) -> str:
        try:
            current_version = self.current_version()
            latest_version = self.latest_version()

        except BaseException as exc:
            message = click.style(f'{exc.__class__.__name__} - {exc}', **ERROR_STYLE)

        else:
            if latest_version > current_version:
                message = click.style(
                    f'Update available {current_version} -> {latest_version}', fg='yellow'
                )
            elif latest_version < current_version:
                message = click.style(
                    (
                        f'Local version {current_version} is newer than latest release '
                        f'{latest_version}'
                    ),
                    **DEFAULT_STYLE,
                )
            else:
                message = click.style('ok', **OK_STYLE)

        name = click.style(self.name, **OK_STYLE)
        return f'{name}: {message}'


def _editable_checkout_path(dist_name: str) -> str | None:
    """Return the local path of an editable install, or None.

    pip records editable installs in ``direct_url.json`` (PEP 610) inside
    the distribution's ``.dist-info/`` directory. When the package is
    installed via ``pip install -e <path>``, ``dir_info.editable`` is
    True and ``url`` is a ``file://`` URL pointing at the checkout.

    Parameters
    ----------
    dist_name : str
        The distribution name (e.g. ``"fwl-aragog"``).

    Returns
    -------
    str or None
        Absolute path to the editable checkout, or ``None`` if the
        package is installed from a wheel or not installed at all.
    """
    try:
        dist = importlib.metadata.distribution(dist_name)
    except PackageNotFoundError:
        return None
    raw = dist.read_text('direct_url.json')
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not info.get('dir_info', {}).get('editable'):
        return None
    url = info.get('url', '')
    if not url.startswith('file://'):
        return None
    return unquote(urlparse(url).path)


def _git_checkout_state(path: str) -> tuple[str, bool] | None:
    """Return ``(short_hash, dirty)`` for a git checkout, or None.

    Reads ``git rev-parse --short HEAD`` and ``git status --porcelain``.
    Returns ``None`` when ``path`` is not a git checkout or git is
    unavailable.

    Parameters
    ----------
    path : str
        Absolute path to a directory expected to contain a git working
        tree.

    Returns
    -------
    tuple[str, bool] or None
        ``(short_commit_hash, is_dirty)``. ``is_dirty`` is True when the
        working tree has uncommitted modifications.
    """
    try:
        short_hash = subprocess.check_output(
            ['git', '-C', path, 'rev-parse', '--short', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ['git', '-C', path, 'status', '--porcelain'],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return short_hash, bool(porcelain.strip())


class PythonPackage(BasePackage):
    def current_version(self) -> Version:
        return Version(importlib.metadata.version(self.name))

    def latest_version(self) -> Version:
        response = requests.get(f'https://pypi.org/pypi/{self.name}/json')
        response.raise_for_status()
        return Version(response.json()['info']['version'])

    def editable_annotation(self) -> str | None:
        """Return a human-readable annotation for an editable install.

        Format: ``"editable @ <basename> -> <hash>"`` or
        ``"editable @ <basename> -> <hash> (dirty)"``. Returns ``None``
        when the package is installed from a wheel.
        """
        checkout = _editable_checkout_path(self.name)
        if checkout is None:
            return None
        state = _git_checkout_state(checkout)
        basename = os.path.basename(checkout.rstrip('/'))
        if state is None:
            return f'editable @ {basename}'
        short_hash, dirty = state
        marker = ' (dirty)' if dirty else ''
        return f'editable @ {basename} -> {short_hash}{marker}'

    def get_status_message(self) -> str:
        base_message = super().get_status_message()
        annotation = self.editable_annotation()
        if annotation is None:
            return base_message
        return f'{base_message} [{click.style(annotation, fg="cyan")}]'


@dataclass
class GitPackage(BasePackage):
    owner: str
    version_getter: Callable

    def current_version(self) -> Version:
        try:
            return Version(self.version_getter())
        except FileNotFoundError as exc:
            raise PackageNotFoundError(f'{self.name} is not installed.') from exc

    def latest_version(self) -> Version:
        # Prefer GitHub's "latest formal release" endpoint, but some
        # FWL repos (e.g. SOCRATES) only ship tags. Fall back to the
        # tags list when releases/latest returns 404.
        base = f'https://api.github.com/repos/{self.owner}/{self.name}'
        response = requests.get(f'{base}/releases/latest')
        if response.status_code == 404:
            tags_response = requests.get(f'{base}/tags')
            tags_response.raise_for_status()
            tags = tags_response.json()
            if not tags:
                raise RuntimeError(f'{self.name}: no releases or tags found on GitHub')
            # /tags returns most-recent-first; first entry's name is the
            # latest tag string.
            return Version(tags[0]['name'])
        response.raise_for_status()
        return Version(response.json()['tag_name'])


PACKAGES = (
    PythonPackage(name='fwl-aragog'),
    PythonPackage(name='fwl-calliope'),
    PythonPackage(name='fwl-janus'),
    PythonPackage(name='fwl-proteus'),
    PythonPackage(name='fwl-mors'),
    PythonPackage(name='fwl-vulcan'),
    PythonPackage(name='fwl-zephyrus'),
    PythonPackage(name='fwl-zalmoxis'),
    GitPackage(name='AGNI', owner='nichollsh', version_getter=partial(_get_agni_version, DIRS)),
    GitPackage(
        name='SOCRATES', owner='FormingWorlds', version_getter=partial(_get_socrates_version)
    ),
)


def get_env_var_status_message(var: str) -> str:
    if os.environ.get(var):
        message = click.style('ok', **OK_STYLE)
    else:
        message = click.style('Variable not set.', **ERROR_STYLE)

    name = click.style(var, **OK_STYLE)
    return f'{name}: {message}'


VARIABLES = (
    'FWL_DATA',
    'RAD_DIR',
    'ZALMOXIS_ROOT',
    'FC_DIR',
    'PYTHON_JULIAPKG_EXE',
    'LA_DIR',
)


def doctor_entry():
    click.secho('Environment variables', **HEADER_STYLE)
    for var in VARIABLES:
        message = get_env_var_status_message(var)
        click.echo(message)

    click.secho('\nPackages', **HEADER_STYLE)
    for package in PACKAGES:
        message = package.get_status_message()
        click.echo(message)
