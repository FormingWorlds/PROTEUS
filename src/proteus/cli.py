from __future__ import annotations

import sys
from pathlib import Path

import click

from proteus import Proteus
from proteus import __version__ as proteus_version

config_option = click.option(
    '-c',
    '--config',
    'config_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help='Path to config file',
    required=True,
)


@click.group()
@click.version_option(version=proteus_version)
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
def plot(plots, config_path: Path):
    """(Re-)generate plots from completed run"""
    from .plot import plot_dispatch

    click.echo(f'Config: {config_path}')

    handler = Proteus(config_path=config_path)

    if "all" in plots:
        plots = list(plot_dispatch.keys())

    for plot in plots:
        if plot not in plot_dispatch.keys():
            click.echo(f"Invalid plot: {plot}")
        else:
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
@click.option(
    '-o',
    '--offline',
    is_flag=True,
    default=False,
    help='Run in offline mode; do not connect to the internet',
)
def start(config_path: Path, resume: bool, offline: bool):
    """Start proteus run"""
    runner = Proteus(config_path=config_path)
    runner.start(resume=resume, offline=offline)


cli.add_command(plot)
cli.add_command(start)


@click.group()
def get():
    """Get data and modules"""
    pass


@click.command()
@click.option('-n', '--name',  'name',  type=str, help='Name of spectral file group')
@click.option('-b', '--bands', 'bands', type=str, help='Number of bands')
def spectral(**kwargs):
    """Get spectral files

    By default, download all files.
    """
    from .utils.data import download_spectral_file
    download_spectral_file(kwargs["name"],kwargs["bands"])


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

@click.command()
def doctor():
    """Diagnose your PROTEUS installation"""
    from .doctor import doctor_entry
    doctor_entry()

@click.command()
@config_option
def offchem(config_path: Path):
    """Run offline chemistry on PROTEUS output files"""
    runner = Proteus(config_path=config_path)
    runner.run_offchem()


cli.add_command(get)
get.add_command(spectral)
get.add_command(surfaces)
get.add_command(reference)
get.add_command(stellar)
get.add_command(socrates)
get.add_command(petsc)
get.add_command(spider)
cli.add_command(doctor)
cli.add_command(offchem)

if __name__ == '__main__':
    cli()
