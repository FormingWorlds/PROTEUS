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


@click.group()
def get():
    """Get data and modules"""
    pass


@click.command()
@click.option('-n', '--name', 'fname', type=str, help='Name of the spectra')
@click.option('-b', '--band', 'nband', type=int, help='Number of the band', default=256)
def spectral(**kwargs):
    """Get spectral files

    By default, download all files.
    """
    from .utils.data import download_spectral_files
    download_spectral_files(**kwargs)


@click.command()
def stellar():
    """Get stellar spectra"""
    from .utils.data import download_evolution_tracks, download_stellar_spectra

    for track in ["Spada","Baraffe"]:
        download_evolution_tracks(track)
    download_stellar_spectra()

@click.command()
def surfaces():
    """Get surface albedos"""
    from .utils.data import download_surface_albedos
    download_surface_albedos()

@click.command()
def reference():
    """Get reference data (exoplanet populations, mass-radius curves, etc.)"""
    from .utils.data import download_exoplanet_data, download_massradius_data
    download_exoplanet_data()
    download_massradius_data()


@click.command()
def socrates():
    """Set up SOCRATES"""
    from .utils.data import get_socrates
    get_socrates()

@click.command()
def petsc():
    """Set up PETSc"""
    from .utils.data import get_petsc
    get_petsc()

@click.command()
def spider():
    """Set up SPIDER"""
    from .utils.data import get_spider
    get_spider()

cli.add_command(get)
get.add_command(spectral)
get.add_command(surfaces)
get.add_command(reference)
get.add_command(stellar)
get.add_command(socrates)
get.add_command(petsc)
get.add_command(spider)

if __name__ == '__main__':
    cli()
