from __future__ import annotations

import importlib.metadata
from functools import partial
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Callable

import click
import requests
from attr import dataclass

from proteus.utils.coupler import _get_agni_version, get_proteus_dir

PROTEUS_DIR = Path(get_proteus_dir())

DIRS = {'agni': PROTEUS_DIR / 'agni'}


@dataclass
class PythonPackage:
    name: str

    def current_version(self) -> str:
        return importlib.metadata.version(self.name)

    def latest_version(self) -> str:
        response = requests.get(f'https://pypi.org/pypi/{self.name}/json')
        if not response.ok:
            response.raise_for_status()

        return response.json()['info']['version']


@dataclass
class GitPackage:
    name: str
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


DEPENDENCIES = (
    PythonPackage(name='fwl-proteus'),
    PythonPackage(name='fwl-mors'),
    PythonPackage(name='fwl-calliope'),
    PythonPackage(name='fwl-zephyrus'),
    PythonPackage(name='aragog'),
    GitPackage(name='AGNI', owner='nichollsh', version_getter=partial(_get_agni_version, DIRS)),
)


def doctor_entry():
    for package in DEPENDENCIES:
        try:
            current_version = package.current_version()
            latest_version = package.latest_version()
        except BaseException as exc:
            message = click.style(str(exc), fg='red')
        else:
            if current_version != latest_version:
                message = click.style(f'Update available {current_version} -> {latest_version}', fg='yellow')
            else:
                message = click.style('ok', fg='green')

        name = click.style(package.name, fg='green', bold=True)
        click.echo(f'{name}: {message}')
