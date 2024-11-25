from __future__ import annotations

import importlib.metadata
import os
from functools import partial
from importlib.metadata import PackageNotFoundError
from typing import Callable

import click
import requests
from attr import dataclass

from proteus.utils.coupler import _get_agni_version, get_proteus_directories

DIRS = get_proteus_directories()

HEADER_STYLE = {'fg': 'yellow', 'underline': True, 'bold': True}
OK_STYLE = {'fg': 'green'}
ERROR_STYLE = {'fg': 'red'}
DEFAULT_STYLE = {'fg': 'yellow'}


@dataclass
class BasePackage:
    name: str

    def current_version(self) -> str: ...

    def latest_version(self) -> str: ...

    def get_status_message(self) -> str:
        try:
            current_version = self.current_version()
            latest_version = self.latest_version()
        except BaseException as exc:
            message = click.style(str(exc), **ERROR_STYLE)
        else:
            if current_version != latest_version:
                message = click.style(
                    f'Update available {current_version} -> {latest_version}', fg='yellow'
                )
            else:
                message = click.style('ok', **OK_STYLE)

        name = click.style(self.name, **OK_STYLE)
        return f'{name}: {message}'


class PythonPackage(BasePackage):
    def current_version(self) -> str:
        return importlib.metadata.version(self.name)

    def latest_version(self) -> str:
        response = requests.get(f'https://pypi.org/pypi/{self.name}/json')
        if not response.ok:
            response.raise_for_status()

        return response.json()['info']['version']


@dataclass
class GitPackage(BasePackage):
    owner: str
    version_getter: Callable

    def current_version(self) -> str:
        try:
            return self.version_getter()
        except FileNotFoundError as exc:
            raise PackageNotFoundError(f'{self.name} is not installed.') from exc

    def latest_version(self) -> str:
        response = requests.get(
            f'https://api.github.com/repos/{self.owner}/{self.name}/releases/latest'
        )
        return response.json()['name']


PACKAGES = (
    PythonPackage(name='aragog'),
    PythonPackage(name='fwl-calliope'),
    PythonPackage(name='fwl-janus'),
    PythonPackage(name='fwl-proteus'),
    PythonPackage(name='fwl-mors'),
    PythonPackage(name='fwl-zephyrus'),
    GitPackage(name='AGNI', owner='nichollsh', version_getter=partial(_get_agni_version, DIRS)),
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
)


def doctor_entry():
    click.secho('Packages', **HEADER_STYLE)
    for package in PACKAGES:
        message = package.get_status_message()
        click.echo(message)

    click.secho('\nEnvironment variables', **HEADER_STYLE)
    for var in VARIABLES:
        message = get_env_var_status_message(var)
        click.echo(message)
