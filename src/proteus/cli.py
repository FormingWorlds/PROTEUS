from __future__ import annotations

from pathlib import Path

import click


@click.group()
def cli():
    pass


@click.command()
def version():
    """Print version and exit"""
    from . import __version__

    print(__version__)


@click.command()
@click.option(
    '-c',
    '--config',
    'config_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help='Path to config file',
)
@click.option(
    '-r',
    '--resume',
    is_flag=True,
    default=False,
    help='Resume simulation from disk',
)
def start(config_path: Path, resume: bool):
    """Start proteus run"""
    from .proteus import Proteus
    runner = Proteus(config_path=config_path)
    runner.start(resume=resume)


cli.add_command(start)
cli.add_command(version)


if __name__ == '__main__':
    cli()
