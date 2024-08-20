from __future__ import annotations

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
def start():
    """Start proteus run"""
    pass


cli.add_command(start)
cli.add_command(version)


if __name__ == '__main__':
    cli()
