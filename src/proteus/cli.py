from __future__ import annotations

import sys
from pathlib import Path

import click

from .proteus import Proteus

config_option = click.option(
    '-c',
    '--config',
    'config_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help='Path to config file',
)


@click.group()
def cli():
    pass


@click.command()
def version():
    """Print version and exit"""
    from . import __version__

    print(__version__)


@click.command()
@click.argument('plots', nargs=-1)
@config_option
@click.option(
    '-l',
    '--list',
    'list_plots',
    is_flag=True,
    default=False,
    help='List available plots and exit',
)
def plot(plots: str, config_path: Path, list_plots: bool):
    from .plot import plot_dispatch

    if list_plots:
        click.echo('Available plots:')
        click.echo(' '.join(plot_dispatch))
        sys.exit(0)

    click.echo(f'Config: {config_path}')

    handler = Proteus(config_path=config_path)

    for plot in plots:
        click.echo(f'Plotting: {plot}')
        plot_func = plot_dispatch[plot]
        plot_func(handler=handler)


@click.command()
@config_option
@click.option(
    '-r',
    '--resume',
    is_flag=True,
    default=False,
    help='Resume simulation from disk',
)
def start(config_path: Path, resume: bool):
    """Start proteus run"""
    runner = Proteus(config_path=config_path)
    runner.start(resume=resume)


cli.add_command(plot)
cli.add_command(start)
cli.add_command(version)


if __name__ == '__main__':
    cli()
