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
    required=True,
)


@click.group()
@click.version_option(package_name='fwl-proteus')
def cli():
    pass


def list_plots(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    from .plot import plot_dispatch

    click.echo(' '.join(plot_dispatch))
    sys.exit()


@click.command()
@click.argument('plots', nargs=-1)
@config_option
@click.option(
    '-l',
    '--list',
    is_flag=True,
    default=False,
    help='List available plots and exit',
    is_eager=True,
    expose_value=False,
    callback=list_plots,
)
def plot(plots: str, config_path: Path):
    """(Re-)generate plots from completed run"""
    from .plot import plot_dispatch

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


if __name__ == '__main__':
    cli()
