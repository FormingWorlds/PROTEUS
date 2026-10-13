from __future__ import annotations

import importlib.util
import logging
import os
import sys
from difflib import get_close_matches
from pathlib import Path

# Prevent workers from using each other's CPUs to avoid
#     oversubscription and improve performance
os.environ['OMP_NUM_THREADS'] = '1'  # noqa
os.environ['MKL_NUM_THREADS'] = '1'  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # noqa
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # noqa

# Optional --deterministic mode: extra-strict numerical reproducibility for
# coupled runs that fail on noise-floor floating-point divergence. Activated
# by passing --deterministic on any subcommand. We intercept it here in raw
# sys.argv (before click parses it) because JAX/XLA env vars must be set
# BEFORE any module that imports JAX is imported, and `from proteus import
# Proteus` below transitively imports JAX via Aragog.
_PROTEUS_DETERMINISTIC_SENTINEL = 'PROTEUS_DETERMINISTIC_APPLIED'
_DETERMINISTIC_XLA_FLAG = '--xla_cpu_enable_fast_math=false'


def _apply_deterministic_env(environ) -> None:
    """Set the deterministic env vars in `environ` (in-place).

    Idempotent: appends the XLA flag only if not already present, sets
    JAX_ENABLE_X64=1, and marks the sentinel. Caller decides whether to
    re-exec; this function does NOT re-exec on its own (so it is safe to
    call from tests).
    """
    environ[_PROTEUS_DETERMINISTIC_SENTINEL] = '1'
    environ['JAX_ENABLE_X64'] = '1'
    xla_existing = environ.get('XLA_FLAGS', '').strip()
    if _DETERMINISTIC_XLA_FLAG not in xla_existing:
        environ['XLA_FLAGS'] = (xla_existing + ' ' + _DETERMINISTIC_XLA_FLAG).strip()


def _should_apply_deterministic(argv, environ) -> bool:
    """Return True iff --deterministic is in argv and sentinel is not set."""
    return '--deterministic' in argv and environ.get(_PROTEUS_DETERMINISTIC_SENTINEL) != '1'


if _should_apply_deterministic(sys.argv, os.environ):
    _apply_deterministic_env(os.environ)
    # Re-exec so the env vars take effect before JAX is imported. The form
    # depends on how PROTEUS was launched: the `proteus` console script puts
    # an executable launcher in argv[0] that can be re-run directly, whereas
    # `python -m proteus.cli` runs this file as __main__ with the module path
    # in argv[0], which is not directly executable and must be re-run via -m.
    if __name__ == '__main__':
        os.execv(sys.executable, [sys.executable, '-m', 'proteus.cli', *sys.argv[1:]])
    else:
        os.execvp(sys.argv[0], sys.argv)

import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import tomllib  # noqa: E402

import click  # noqa: E402

from proteus import Proteus  # noqa: E402
from proteus import __version__ as proteus_version  # noqa: E402
from proteus.config import UnknownConfigKeyError, read_config_object  # noqa: E402
from proteus.utils.data import download_sufficient_data  # noqa: E402
from proteus.utils.helper import (  # noqa: E402
    get_proteus_dir,
    link_proteus_julia,
    proteus_julia_exe,
    resolve_fwl_data_dir,
)
from proteus.utils.install_env import (  # noqa: E402
    export_install_vars,
    quoted_names,
    reload_instruction,
)
from proteus.utils.logs import bootstrap_logger, setup_logger  # noqa: E402

log = logging.getLogger('fwl.' + __name__)

config_option = click.option(
    '-c',
    '--config',
    'config_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help='Path to config file',
    required=True,
)

output_option = click.option(
    '-o',
    '--output',
    'output_path',
    type=click.Path(exists=True, dir_okay=True, path_type=Path, resolve_path=True),
    help='Path to output folder',
    required=True,
)


class ConfigRejectedError(click.ClickException):
    """A refused configuration, reported at error level on the 'fwl' logger.

    click prints a ClickException as a bare ``Error: ...`` line that carries no
    level, so a refusal arrived untagged among the level-tagged lines around
    it. Overriding how it is shown keeps everything click gives the caller (no
    traceback, exit code 1) while routing the text through the same logger and
    formatter as the rest of the run, where it is marked ERROR.
    """

    def show(self, file=None) -> None:
        log.error(self.format_message())


class ConfigAwareGroup(click.Group):
    """Command group that presents a refused configuration as a CLI error.

    Every command that reads a configuration can refuse it over unrecognised
    keys. Catching that here rather than in each command keeps the message in
    the same style as the rest of the CLI, and reaches subcommands too, since
    they are invoked through this group. Only the configuration-key error is
    caught, so an unrelated failure still surfaces with its traceback.
    """

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except UnknownConfigKeyError as exc:
            raise ConfigRejectedError(str(exc)) from exc


@click.group(cls=ConfigAwareGroup)
@click.version_option(version=proteus_version)
def cli():
    # Ensure the 'fwl' logger has a handler as early as possible, before any
    # subcommand constructs Proteus() or calls into modules that log. Without
    # this, log records emitted before a command's own setup_logger call (e.g.
    # from Proteus.__init__ -> set_directories, or from commands like
    # grid-summarise / grid-pack that never call setup_logger) fall through to
    # logging.lastResort and INFO/DEBUG messages are dropped. See issue #708.
    bootstrap_logger()


# ----------------
# 'plot' command
# ----------------


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
    setup_logger(
        logpath=os.path.join(handler.directories['output'], 'plot.log'),
        logterm=True,
        level=handler.config.params.out.logging,
    )

    if plots[0] == 'list':
        click.echo('Available plots: ' + ', '.join(plot_dispatch.keys()))
        return

    if 'all' in plots:
        plots = list(plot_dispatch.keys())

    for plot in plots:
        if plot not in plot_dispatch.keys():
            click.echo(f'Invalid plot: {plot}')
        else:
            click.echo(f'Plotting: {plot}')
            plot_func = plot_dispatch[plot]
            plot_func(handler=handler)


cli.add_command(plot)
# ----------------
# 'start' command
# ----------------


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
@click.option(
    '--deterministic',
    is_flag=True,
    default=False,
    help=(
        'Force extra-strict numerical reproducibility (sets JAX_ENABLE_X64=1 '
        'and XLA_FLAGS=--xla_cpu_enable_fast_math=false on top of the always-on '
        'BLAS thread pins). Use when a coupled run fails on noise-floor '
        'floating-point divergence (e.g. Aragog T_core-jump-guard exhaustion '
        'on numerically fragile anchors). The flag is intercepted before JAX '
        'imports and triggers a one-shot self re-exec; the sentinel env var '
        'PROTEUS_DETERMINISTIC_APPLIED=1 is set in the child process.'
    ),
)
def start(config_path: Path, resume: bool, offline: bool, deterministic: bool):
    """Start proteus run"""
    if deterministic and os.environ.get(_PROTEUS_DETERMINISTIC_SENTINEL) != '1':
        click.secho(
            '[!] --deterministic was requested but the env-var re-exec did not '
            'take effect. Numerical determinism is NOT pinned. Investigate.',
            fg='yellow',
        )
    runner = Proteus(config_path=config_path)
    runner.start(resume=resume, offline=offline)


cli.add_command(start)

# --------------
# 'get' command, with subcommands
# --------------


@click.group()
def get():
    """Get data and modules"""
    # Scope the name to the calling user: a shared sticky temp directory
    # refuses removal of another user's logfile.
    log_path = Path(tempfile.gettempdir()) / f'proteus_get_{os.getuid()}.log'
    try:
        setup_logger(
            logpath=str(log_path),
            logterm=True,
            level='INFO',
        )
    except OSError as exc:
        # A logfile that cannot be written must not stop a download.
        logging.getLogger('fwl').handlers.clear()
        bootstrap_logger(level='INFO').warning(f'Cannot write {log_path}: {exc}')


@click.command()
@click.option(
    '-n',
    '--name',
    'name',
    type=str,
    default=None,
    help='Name of spectral file group (default: all groups)',
)
@click.option(
    '-b',
    '--bands',
    'bands',
    type=str,
    default=None,
    help='Number of bands (default: all band counts for the group)',
)
def spectral(**kwargs):
    """Get spectral files

    By default, download all spectral files. Use -n GROUP to download
    every band count for one group, or -n GROUP -b BANDS for one file.
    """
    from .utils.data import download_spectral_files

    download_spectral_files(kwargs['name'], kwargs['bands'])


@click.command()
def stellar():
    """Get stellar spectra"""
    from .utils.data import download_stellar_spectra, download_stellar_tracks

    for track in ['Spada', 'Baraffe']:
        download_stellar_tracks(track)
    download_stellar_spectra()


# MUSCLES and solar star names available online (on Zenodo)
STARS_ONLINE: dict[str, list[str]] = {
    'muscles': [
        'gj1132',
        'gj1214',
        'gj15a',
        'gj163',
        'gj176',
        'gj436',
        'gj551',
        'gj581',
        'gj649',
        'gj667c',
        'gj674',
        'gj676a',
        'gj699',
        'gj729',
        'gj832',
        'gj832_synth',
        'gj849',
        'gj876',
        'hd40307',
        'hd85512',
        'hd97658',
        'l-980-5',
        'lhs-2686',
        'trappist-1',
        'v-eps-eri',
        'l-98-59',
        'hat-p-12',
        'hat-p-26',
        'hd-149026',
        'l-678-39',
        'lp-791-18',
        'toi-193',
        'wasp-127',
        'wasp-17',
        'wasp-43',
        'wasp-77a',
    ],
    'solar': [
        'sun',
        'sun0.6ga',
        'sun1.8ga',
        'sun2.4ga',
        'sun2.7ga',
        'sun3.8ga',
        'sun4.4ga',
        'sun5.6gyr',
        'sunmodern',
    ],
}


def normalize_star_name(star_name: str | None) -> str | None:
    """Normalize user input to match our canonical star IDs."""
    if not star_name:
        return None
    # Keep it conservative: lowercase + strip. (Add more rules later if desired.)
    return star_name.lower().strip()


def validate_star_name(star_name: str | None, *, catalog: str = 'muscles') -> str | None:
    """Validate and return a normalized star name.

    Raises a ClickException with "did you mean" suggestions if the star is unknown.
    """
    norm = normalize_star_name(star_name)
    if not norm:
        return None

    if catalog not in STARS_ONLINE:
        raise click.ClickException(
            f"Unknown catalog '{catalog}'. Available: {', '.join(sorted(STARS_ONLINE))}."
        )

    choices = sorted(set(STARS_ONLINE[catalog]))
    if norm not in choices:
        suggestions = get_close_matches(norm, choices, n=8, cutoff=0.6)
        hint = f' Did you mean: {", ".join(suggestions)}?' if suggestions else ''
        raise click.ClickException(
            f"Unknown {catalog.upper()} star '{star_name}'.{hint} "
            f'Use --list to see all available names.'
        )
    return norm


@click.command()
@click.option(
    '--star',
    'star_name',
    required=False,
    type=str,
    default=None,
    help="Star name for MUSCLES spectrum (e.g. 'trappist-1', 'gj876').",
)
@click.option(
    '--all',
    'download_all',
    is_flag=True,
    default=False,
    help='Download all available MUSCLES spectra.',
)
@click.option(
    '--list',
    'list_stars',
    is_flag=True,
    default=False,
    help='List available MUSCLES star names and exit.',
)
def muscles(star_name: str | None, download_all: bool, list_stars: bool):
    """Download MUSCLES stellar spectrum(s). See https://proteus-framework.org/PROTEUS/data.html#using-a-mega-muscles-observed-spectrum for more information."""
    from .utils.data import download_muscles

    if list_stars:
        for s in STARS_ONLINE['muscles']:
            click.echo(s)
        return

    if download_all:
        if star_name:
            raise click.ClickException('Use either --all or --star NAME (not both).')
        targets = STARS_ONLINE['muscles']
    else:
        if not star_name:
            raise click.ClickException('Provide --star NAME or use --all.')
        star = validate_star_name(star_name, catalog='muscles')
        targets = [star]

    ok, failed = 0, []

    for s in targets:
        click.echo(f'Downloading MUSCLES: {s} ...')
        if download_muscles(s):
            ok += 1
        else:
            failed.append(s)

    click.secho(f'Done. OK: {ok}/{len(targets)}', fg='green' if ok else 'red')

    if failed:
        click.secho(f'Failed ({len(failed)}): {", ".join(failed)}', fg='yellow')
        if ok == 0:
            raise click.ClickException('All MUSCLES downloads failed.')


@click.command()
@click.option(
    '--feh',
    'FeH',
    required=True,
    type=float,
    help='Stellar metallicity [Fe/H]. Solar is 0.0, subsolar is negative, supersolar is positive. Input will automatically be rounded to the nearest allowed grid value.',
)
@click.option(
    '--alpha',
    required=True,
    type=float,
    help='Stellar alpha enhancement [alpha/M]. Solar is 0.0, subsolar is negative, supersolar is positive. Input will automatically be rounded to the nearest allowed grid value for the given [Fe/H].',
)
@click.option(
    '--teff',
    required=False,
    type=float,
    default=None,
    help='Optional Teff [K] to choose the correct PHOENIX alpha availability. ',
)
def phoenix(FeH: float, alpha: float, teff: float | None):
    """Download PHOENIX grid ZIP for the nearest allowed (FeH, alpha) point. See https://proteus-framework.org/PROTEUS/data.html#using-a-phoenix-synthetic-spectrum for more information."""
    from .utils.data import download_phoenix
    from .utils.phoenix_helper import phoenix_to_grid

    grid = phoenix_to_grid(FeH=FeH, alpha=alpha, Teff=teff)
    ok = download_phoenix(alpha=grid['alpha'], FeH=grid['FeH'])
    if not ok:
        raise click.ClickException(
            f'Failed to download PHOENIX grid for [Fe/H]={grid["FeH"]:+.1f}, '
            f'[alpha/M]={grid["alpha"]:+.1f}.'
        )
    else:
        click.secho(
            f'Downloaded PHOENIX grid for [Fe/H]={grid["FeH"]:+.1f}, '
            f'[alpha/M]={grid["alpha"]:+.1f}.',
            fg='green',
        )


@click.command()
def solar():
    """Download the available solar spectra."""
    from .data import STELLAR_SPECTRA_SOLAR, dataset_dir
    from .utils.data import GetFWLData, download_stellar_spectra

    # Where the data should end up
    solar_dir = dataset_dir(STELLAR_SPECTRA_SOLAR, data_root=GetFWLData())

    try:
        download_stellar_spectra(folders=('solar',))
    except Exception as e:
        raise click.ClickException(f'Failed to download solar spectra: {e}') from e

    # Did we actually get files?
    if solar_dir.exists():
        files = sorted(p for p in solar_dir.rglob('*') if p.is_file())
        if files:
            click.secho('Solar spectra downloaded successfully.', fg='green')
            click.echo(f'Location: {solar_dir}')
            return

    # If we get here, the downloader didn't raise but we can't find files.
    raise click.ClickException(
        'Solar download finished without an exception, but no files were found where expected.\n'
        f'Expected: {solar_dir}\n'
        'Run `proteus get solar` again; if this repeats, remove that folder first.'
    )


@click.command()
def surfaces():
    """Get surface albedos"""
    from .utils.data import download_surface_albedos

    download_surface_albedos()


@click.command()
def scattering():
    """Get scattering radiative properties"""
    from .utils.data import download_scattering

    download_scattering()


@click.command()
def refractive():
    """Get aerosol refractive indices for Mie scattering"""
    from .utils.data import download_refractive

    download_refractive()


@click.command()
def reference():
    """Get reference data (exoplanet populations, mass-radius curves, etc.)"""
    from .utils.data import download_exoplanet_data, download_massradius_data

    download_exoplanet_data()
    download_massradius_data()


@click.command()
@click.option(
    '--config-path',
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path('input/all_options.toml'),
    help='Path to the TOML config file',
)
def interiordata(config_path: Path):
    """Get interior lookup tables, melting curves, and structure EOS tables"""
    from .utils.data import (
        download_eos_dynamic,
        download_interior_lookuptables,
        download_melting_curves,
        download_zalmoxis_eos_for_config,
        needs_spider_ps_tables,
    )

    download_interior_lookuptables(clean=True)

    configuration = read_config_object(config_path)
    download_melting_curves(configuration, clean=True)

    # Structure-solver EOS tables (e.g. the PALEOS set) for the config's
    # Zalmoxis setup. Without these on disk, an offline run fails inside
    # the structure solver.
    download_zalmoxis_eos_for_config(configuration)

    # The P-S lookup set SPIDER and Aragog read without a PALEOS table set.
    if needs_spider_ps_tables(configuration):
        download_eos_dynamic()


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
get.add_command(scattering)
get.add_command(refractive)
get.add_command(muscles)
get.add_command(phoenix)
get.add_command(solar)
get.add_command(reference)
get.add_command(stellar)
get.add_command(interiordata)
get.add_command(socrates)
get.add_command(petsc)
get.add_command(spider)

# ----------------
# doctor utility
# ----------------


@click.command()
@click.option('--json', 'output_json', is_flag=True, help='Output results as JSON.')
def doctor(output_json: bool):
    """Diagnose your PROTEUS installation.

    Checks environment variables, reference data, Python package versions,
    and git module pins. Each check reports pass/warn/fail with a fix
    command where applicable.
    """
    from .doctor import run_doctor

    if not run_doctor(output_json=output_json):
        raise SystemExit(1)


@click.command()
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing.')
def update(dry_run: bool):
    """Update PROTEUS and its submodules.

    Runs the same checks as ``proteus doctor``, then executes the
    suggested fix commands for any failing or warning checks. Use
    ``--dry-run`` to preview without making changes.
    """
    from .doctor import run_update

    if not run_update(dry_run=dry_run):
        raise SystemExit(1)


cli.add_command(doctor)
cli.add_command(update)

# ----------------
# 'archive' commands
# ----------------


@click.command()
@config_option
def create_archives(config_path: Path):
    """Pack the output files in tar archives"""
    runner = Proteus(config_path=config_path)
    runner.create_archives()


@click.command()
@config_option
def extract_archives(config_path: Path):
    """Unpack the output files from existing tar archives"""
    runner = Proteus(config_path=config_path)
    runner.extract_archives()


cli.add_command(create_archives)
cli.add_command(extract_archives)

# ----------------
# 'offchem' and 'observe' postprocessing commands
# ----------------


@click.command()
@config_option
def offchem(config_path: Path):
    """Run offline chemistry on PROTEUS output files"""
    runner = Proteus(config_path=config_path)
    setup_logger(
        logpath=os.path.join(runner.directories['output'], 'offchem.log'),
        logterm=True,
        level=runner.config.params.out.logging,
    )
    runner.offline_chemistry()


@click.command()
@config_option
def observe(config_path: Path):
    """Run synthetic observations pipeline"""
    runner = Proteus(config_path=config_path)
    setup_logger(
        logpath=os.path.join(runner.directories['output'], 'observe.log'),
        logterm=True,
        level=runner.config.params.out.logging,
    )
    runner.observe()


cli.add_command(offchem)
cli.add_command(observe)

# ----------------
# GridPROTEUS and BO inference scheme, runners
# ----------------


@click.command()
@config_option
@click.option(
    '--dry-run',
    is_flag=True,
    default=False,
    help='Generate the grid and write per-case config files without launching '
    'PROTEUS simulations. Useful for validating a grid before spending compute.',
)
def grid(config_path: Path, dry_run: bool):
    """Run GridPROTEUS to generate a grid of forward models"""
    from proteus.grid.manage import grid_from_config

    grid_from_config(config_path, test_run=dry_run)


# Distributions that only the inference scheme needs, shipped as the
# `inference` extra. `[project.optional-dependencies].inference` in
# pyproject.toml is the authority; a test pins the two together.
INFERENCE_DISTRIBUTIONS = ('torch', 'botorch', 'gpytorch')


def _absent_inference_distribution(exc: ImportError) -> str | None:
    """Name the inference distribution whose absence raised ``exc``.

    Parameters
    ----------
    exc : ImportError
        Error raised while importing the inference entry point.

    Returns
    -------
    str or None
        The missing distribution, or None when the error came from anything
        else: a failure inside a package that is present, an unrelated
        module, or one of the three sitting installed but failing to import
        one of its own submodules.

    Notes
    -----
    Absence is confirmed against the import system instead of being inferred
    from the error alone, so the advice to install the extra is never given
    for a package that is already in the environment.
    """
    if not isinstance(exc, ModuleNotFoundError):
        # The module was found and something inside it failed: a version
        # mismatch or a broken install, not a missing extra.
        return None
    root = (exc.name or '').split('.')[0]
    if root not in INFERENCE_DISTRIBUTIONS:
        return None
    try:
        if importlib.util.find_spec(root) is not None:
            return None
    except (ImportError, ValueError):
        # Present but unusable: a broken install, not a missing extra.
        return None
    return root


@click.command()
@config_option
def infer(config_path: Path):
    """Use Bayesian optimisation to infer parameters from observables"""
    try:
        from proteus.inference.inference import infer_from_config
    except ImportError as exc:
        missing = _absent_inference_distribution(exc)
        if missing is None:
            raise
        raise click.ClickException(
            f"Parameter inference needs '{missing}', which is not installed. "
            'The optimisation stack ships as an optional extra: '
            'pip install "fwl-proteus[inference]"'
        ) from exc

    infer_from_config(config_path)


cli.add_command(grid)
cli.add_command(infer)

# ----------------
# Grid status check and grid packaging
# ----------------


@cli.command()
@output_option
@click.option(
    '-s',
    '--status',
    default=None,
    type=str,
    help='List cases with this status. '
    "For example: 'completed', 'running', 'error', or 'code=x' for some error code x.",
)
def grid_summarise(output_path: Path, status: str):
    """
    Summarise the current status of a grid, showing which fraction of cases end with
    each status code.

    Provide `status` to have the command explicitly list cases which end with a particular
    status code.
    """
    from proteus.grid.summarise import summarise as gsummarise

    gsummarise(output_path, status)


@cli.command()
@output_option
def grid_pack(output_path: Path):
    """Packagage grid points into single ZIP file, for easy backup and sharing"""
    from proteus.grid.pack import pack as gpack

    gpack(output_path)


# ----------------
# installer
# ----------------


def _built_socrates(root: Path) -> Path | None:
    """Return the SOCRATES folder for ``RAD_DIR``: the first one that holds ``bin/radlib.a``.

    Parameters
    ----------
    root : Path
        PROTEUS source tree.

    Returns
    -------
    Path or None
        ``<root>/socrates`` (resolved) when it is built, else the ``RAD_DIR``
        of the environment when that is built, else None.
    """
    if (root / 'socrates' / 'bin' / 'radlib.a').is_file():
        return (root / 'socrates').resolve()
    rad_dir = os.environ.get('RAD_DIR')
    if rad_dir and (Path(rad_dir) / 'bin' / 'radlib.a').is_file():
        return Path(rad_dir)
    return None


def _install_vars(root: Path, fwl_data: Path, rad_dir: Path | None) -> dict[str, str]:
    """Return the install variables that ``--export-env`` records.

    ``RAD_DIR`` appears only for a built SOCRATES (``_built_socrates``), so
    that a folder without one does not override a valid ``RAD_DIR`` at each
    activation. ``PYTHON_JULIAPKG_EXE`` appears only when it is set and names an
    executable file (``proteus_julia_exe``), because ``install.sh`` points it
    at the real binary of Julia 1.13; a ``julia`` on ``PATH`` can be another
    version or the juliaup launcher, so it is never taken. ``FC_DIR``
    appears only when the FastChem program is built.

    Parameters
    ----------
    root : Path
        PROTEUS source tree.
    fwl_data : Path
        Reference data folder.
    rad_dir : Path or None
        Built SOCRATES folder, or None when there is none.

    Returns
    -------
    dict[str, str]
        Variable names and values.
    """
    variables = {'FWL_DATA': str(fwl_data)}
    if rad_dir is not None:
        variables['RAD_DIR'] = str(rad_dir)
    else:
        click.secho(
            f'[!] No SOCRATES build (bin/radlib.a) in {root / "socrates"} or in RAD_DIR, so '
            'RAD_DIR is not recorded; bash tools/get_socrates.sh builds it.',
            fg='yellow',
        )
    julia_exe = proteus_julia_exe()
    if julia_exe:
        variables['PYTHON_JULIAPKG_EXE'] = julia_exe
    else:
        click.secho(
            '[!] PYTHON_JULIAPKG_EXE is not set to an executable Julia, so it is not '
            'recorded; bash install.sh selects Julia 1.13 and records it.',
            fg='yellow',
        )
    fastchem = root / 'AGNI' / 'fastchem'
    program = fastchem / 'fastchem'
    if program.is_file() and os.access(program, os.X_OK):
        variables['FC_DIR'] = str(fastchem.resolve())
    return variables


def _export_env(root: Path, variables: dict[str, str]) -> None:
    """Record ``variables`` in the conda env and the exports file, and say how to load them.

    Parameters
    ----------
    root : Path
        PROTEUS source tree that holds the exports file.
    variables : dict[str, str]
        Variable names and values.
    """
    quoted = quoted_names(variables)
    names = ', '.join(variables)
    bindable = ', '.join(name for name in variables if name not in quoted)
    path, prefix, bound = export_install_vars(root, variables)
    click.secho(f'[+] Wrote {names} to {path}', fg='green')
    if bound:
        click.secho(f'[+] Set {bindable} in the conda env at {prefix}', fg='green')
    elif prefix is not None and bindable:
        click.secho(
            f'[!] Could not set {bindable} in the conda env at {prefix}; they are in {path}',
            fg='yellow',
        )
    elif prefix is None and os.environ.get('CONDA_DEFAULT_ENV') == 'base':
        click.secho(
            '[!] The conda base environment is active, and PROTEUS sets nothing in base; '
            f'the variables are in {path} only',
            fg='yellow',
        )
    if quoted and prefix is not None:
        click.secho(
            f'[!] Not set in the conda env: {", ".join(quoted)}, whose value holds a single '
            f"quote that conda's activation script cannot read; load it with: source {path}",
            fg='yellow',
        )
    loaded = bound and not quoted
    where = '' if loaded else f' (in {root})'
    click.secho(f'[i] To load them, run{where}: {reload_instruction(loaded)}', fg='yellow')


def is_julia_installed() -> bool:
    """Return whether a Julia is available to PROTEUS.

    Returns
    -------
    bool
        True when ``PYTHON_JULIAPKG_EXE`` names an executable file or a
        ``julia`` is on ``PATH``.
    """
    return proteus_julia_exe() is not None or shutil.which('julia') is not None


def _report_julia() -> None:
    """Print the version of the Julia PROTEUS uses, with a warning when it is not 1.13.

    The check is ``doctor.check_julia``, which reads ``PYTHON_JULIAPKG_EXE``
    and else the ``julia`` on ``PATH``.
    """
    from .doctor import PASS, check_julia

    result = check_julia()
    if result.status == PASS:
        click.secho(f'[+] Julia: {result.message}', fg='green')
    else:
        click.secho(f'[!] Julia: {result.message}; fix: {result.fix_cmd}', fg='yellow')


def _env_with_proteus_julia(root: Path) -> dict[str, str]:
    """Return a copy of the environment in which a bare ``julia`` is the Julia PROTEUS uses.

    When ``PYTHON_JULIAPKG_EXE`` names an executable file, the folder of
    ``link_proteus_julia`` goes first on ``PATH``, so that the AGNI setup
    script runs the Julia that ``install.sh`` selected and not the default
    Julia of the user, and finds no other program of that binary's folder.
    A link that cannot be written (a read-only root) is a warning that the
    setup script stops on the same link (``use_proteus_julia`` in
    ``tools/_get_common.sh``), and ``PATH`` stays as it is.

    Parameters
    ----------
    root : Path
        PROTEUS source tree that holds the ``.proteus-julia`` folder.

    Returns
    -------
    dict[str, str]
        Environment for the setup-script subprocess.
    """
    env = os.environ.copy()
    try:
        folder = link_proteus_julia(root)
    except OSError as exc:
        click.secho(
            f'[!] Could not link {root / ".proteus-julia" / "julia"} to the Julia PROTEUS '
            f'uses ({exc}); the setup script will stop, because it makes the same link',
            fg='yellow',
        )
        folder = None
    if folder is not None:
        env['PATH'] = os.pathsep.join(p for p in (str(folder), env.get('PATH', '')) if p)
    return env


def _update_input_data(config_path: Path):
    if config_path.exists():
        # Only try data download if a config file is present.
        configuration = read_config_object(config_path)
        download_sufficient_data(configuration, clean=True)
        click.secho('[+] Additional data has been downloaded.', fg='green')
        return True

    else:
        click.echo(f'[!] No config file found at {config_path}, skipping data download.')
        return False


def _is_proteus_root(path: Path) -> bool:
    """Return True when ``path`` holds the PROTEUS source tree.

    Identity is checked by the project name in ``pyproject.toml``, so an
    arbitrary Python project, a stale editable-install pointer, or a
    directory that merely contains some ``pyproject.toml`` is never
    accepted as the install target. Any filesystem error while probing
    counts as "not a PROTEUS root".

    Parameters
    ----------
    path : Path
        Candidate directory.

    Returns
    -------
    bool
        Whether the directory holds the ``fwl-proteus`` project.
    """
    try:
        pyproject = path / 'pyproject.toml'
        if not pyproject.is_file():
            return False
        with open(pyproject, 'rb') as fh:
            name = tomllib.load(fh).get('project', {}).get('name', '')
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return name == 'fwl-proteus'


def _resolve_proteus_root() -> Path:
    """Return the PROTEUS source tree root, independent of the working directory.

    ``install-all`` and ``update-all`` clone the sibling submodules (SOCRATES,
    AGNI) next to the package source and run an editable ``pip install`` against
    the source tree, so they must locate the repository from the installed
    package itself rather than from the directory the command was invoked in.
    Deriving the root from the caller's current directory breaks the moment the
    command is run from anywhere other than the checkout.

    Both candidates are verified to actually hold the ``fwl-proteus`` project
    before being accepted, and the chosen root is announced. For a non-editable
    install (plain wheel) the package location does not sit in a source tree;
    in that case the current directory is used when it is itself a PROTEUS
    checkout, which keeps the command usable for wheel users standing inside a
    clone.

    Returns
    -------
    Path
        Absolute path to the PROTEUS source tree.

    Raises
    ------
    SystemExit
        If no PROTEUS source tree is reachable from either the installed
        package or the current directory. There is then no checkout to set up
        or update, so the command exits with a clear message instead of a
        confusing downstream error.
    """
    root = None
    try:
        candidate = Path(get_proteus_dir())
    except EnvironmentError:
        candidate = None
    if candidate is not None and _is_proteus_root(candidate):
        root = candidate
    else:
        try:
            cwd = Path.cwd()
        except OSError:
            cwd = None
        if cwd is not None and _is_proteus_root(cwd):
            root = cwd
    if root is None:
        click.secho(
            '[x] Cannot locate the PROTEUS source tree. '
            "'install-all' and 'update-all' need a PROTEUS git checkout: "
            'install PROTEUS in editable mode (pip install -e) or run the '
            'command from inside a clone of the repository.',
            fg='red',
        )
        raise SystemExit(1)
    click.secho(f'[i] PROTEUS root: {root}', fg='cyan')
    return root


_CVODE_IMPORT = 'from scikits_odes_sundials.cvode import CVODE, CV_RootFunction, StatusEnum'


def _install_cvode(root: Path) -> None:
    """Install the SUNDIALS CVODE solver, or warn and let the command go on.

    CVODE lives outside the pip dependency tree because it needs the SUNDIALS C
    library. Only a run with Aragog on ``solver_method = "cvode"`` needs it, and
    that run stops at setup without it, so a failed install is a warning here:
    SPIDER, ``radau`` and ``bdf`` users can finish the installation. The helper
    script is idempotent and returns early when CVODE already imports. Its own
    Python may not be the one running PROTEUS, so the import is checked again here.

    Parameters
    ----------
    root : Path
        PROTEUS source tree that holds ``tools/get_cvode.sh``.
    """
    click.secho('[+] Installing the SUNDIALS CVODE solver...', fg='blue')
    try:
        subprocess.run(['bash', str(root / 'tools' / 'get_cvode.sh')], cwd=root, check=True)
        subprocess.run([sys.executable, '-c', _CVODE_IMPORT], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        click.secho(
            '[!] CVODE (scikits-odes-sundials) is not installed or does not import in '
            f'{sys.executable}. Aragog with solver_method = "cvode" stops at setup until it '
            'does: fix the error above and run "bash tools/get_cvode.sh". Other interior '
            'modules, and Aragog with solver_method = "radau" or "bdf", do not need it.',
            fg='yellow',
        )
        return
    click.secho('[+] CVODE available', fg='green')


def _download_default_phoenix() -> None:
    """Download the PHOENIX grid of the default star, or warn and let the command go on.

    The default star reads the grid at [Fe/H] = 0.0 and [alpha/M] = 0.0, which
    ``download_sufficient_data`` does not fetch; ``install.sh`` fetches it in its
    essential data set. Without it a config with the default star stops at the
    star set-up of an offline run, so a failed download is a warning here.
    """
    from .utils.data import download_phoenix
    from .utils.phoenix_helper import phoenix_to_grid

    click.secho('[+] Downloading the PHOENIX grid of the default star...', fg='blue')
    grid = phoenix_to_grid(FeH=0.0, alpha=0.0)
    label = f'[Fe/H]={grid["FeH"]:+.1f}, [alpha/M]={grid["alpha"]:+.1f}'
    try:
        ok = download_phoenix(alpha=grid['alpha'], FeH=grid['FeH'])
    except OSError as exc:
        click.echo(exc)
        ok = False
    if not ok:
        click.secho(
            f'[!] The PHOENIX grid for {label} did not download. A config with the default '
            'star stops at the star set-up of an offline run until it is there: run '
            '"proteus get phoenix --feh 0.0 --alpha 0.0".',
            fg='yellow',
        )
        return
    click.secho(f'[+] PHOENIX grid for {label} available', fg='green')


_GIT_READ_TIMEOUT_S = 30
_NOT_OWN_CLONE = 'not a git clone of its own'


def _git_read(directory: Path, *args: str) -> str | None:
    """Return the output of a read-only git command in ``directory``, without surrounding white space.

    ``--no-optional-locks`` keeps ``git status`` from rewriting the index.

    Returns
    -------
    str or None
        The output, or None when git is missing, fails or runs longer than
        ``_GIT_READ_TIMEOUT_S`` s (``doctor._git_stdout``).
    """
    from .doctor import _git_stdout

    out = _git_stdout(
        str(directory), *args, options=('--no-optional-locks',), timeout=_GIT_READ_TIMEOUT_S
    )
    return None if out is None else out.strip()


def _agni_update_blocker(agni_dir: Path) -> str | None:
    """Return what keeps ``tools/get_agni.sh`` from moving the AGNI clone, or None.

    The script checks out the pinned commit in place, which git refuses over
    a conflicting change but does without a word over commits of a detached
    HEAD. The checks run in this order, each with read-only git commands:

    1. ``git rev-parse --show-toplevel`` fails or names another folder: the
       folder is no clone of its own, and git would act on the repository
       around it (an empty ``.git`` folder inside the PROTEUS checkout).
    2. ``git status --porcelain --untracked-files=no`` lists modified
       tracked files.
    3. ``git rev-list --count HEAD --not --remotes --tags`` counts commits
       of HEAD on no remote-tracking branch and under no tag: local work on
       a detached HEAD or on a branch that was not pushed.

    A git command that fails or times out blocks too, because the state of
    the clone is then unknown. Untracked files and stash entries do not
    block: the checkout keeps them.

    Parameters
    ----------
    agni_dir : Path
        The AGNI folder under the PROTEUS root.

    Returns
    -------
    str or None
        What was found, for the stay line, or None when the clone may move.
    """
    top = _git_read(agni_dir, 'rev-parse', '--show-toplevel')
    if top is None:
        return f'{_NOT_OWN_CLONE} (git does not report one there)'
    if Path(top).resolve() != agni_dir.resolve():
        return f'{_NOT_OWN_CLONE} (git answers for {top})'
    status = _git_read(agni_dir, 'status', '--porcelain', '--untracked-files=no')
    if status is None:
        return 'git could not report its modified files'
    if status:
        count = len(status.splitlines())
        return f'{count} modified tracked file{"s" if count != 1 else ""}'
    local = _git_read(agni_dir, 'rev-list', '--count', 'HEAD', '--not', '--remotes', '--tags')
    if local is None or not local.isdigit():
        return 'git could not list its local commits'
    if int(local) > 0:
        count = int(local)
        return (
            f'{count} local commit{"s" if count != 1 else ""} '
            '(on no remote-tracking branch and under no tag)'
        )
    return None


def _update_agni(root: Path) -> None:
    """Bring the AGNI clone under ``root`` to its pinned commit, or say why it stays.

    The pinned commit is ``AGNI_GIT_REF`` when set, as ``tools/get_agni.sh``
    reads it, else the ``ref`` of ``[tool.proteus.modules.agni]`` in
    ``pyproject.toml``. A clone that ``_agni_update_blocker`` holds back is
    not touched; the commands that move it by hand are printed instead, led
    by a ``git branch`` that keeps local commits. A failure of
    ``tools/get_agni.sh`` is reported with its exit status, and the update
    goes on with its next step either way.

    Parameters
    ----------
    root : Path
        PROTEUS source tree that holds ``AGNI/`` and ``tools/get_agni.sh``.
    """
    from .doctor import _module_pins

    agni_dir = root / 'AGNI'
    ref = os.environ.get('AGNI_GIT_REF') or _module_pins().get('agni', {}).get('ref')
    found = _agni_update_blocker(agni_dir)
    if found is not None:
        if ref is None:
            expected = 'the commit PROTEUS expects could not be read from pyproject.toml'
        else:
            expected = f'PROTEUS expects the commit {ref[:7] if len(ref) == 40 else ref}'
        click.secho(f'[!] agni: {agni_dir} stays as it is: {found}; {expected}', fg='yellow')
        if found.startswith(_NOT_OWN_CLONE):
            click.echo(f'    Move it away; then, in {root}, run  bash tools/get_agni.sh 0')
            click.echo('    which clones AGNI there at the pinned commit.')
            return
        click.echo('    To move it to that commit by hand, run:')
        if 'local commit' in found:
            click.echo(
                f'      git -C {agni_dir} branch my-agni-work'
                '    (keeps the local commits on a branch; choose any name)'
            )
        click.echo(f'      git -C {agni_dir} fetch origin')
        click.echo(f'      git -C {agni_dir} checkout {ref or "<agni ref in pyproject.toml>"}')
        click.echo(f'    and then, in {root}:  bash tools/get_agni.sh 0')
        return
    click.secho('[+] Updating AGNI...', fg='blue')
    try:
        subprocess.run(
            ['bash', str(root / 'tools' / 'get_agni.sh'), '0'],
            cwd=root,
            env=_env_with_proteus_julia(root),
            stdin=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.secho(
            f'[x] Failed to update AGNI: bash tools/get_agni.sh 0 exited with status '
            f'{e.returncode}; its messages are above',
            fg='red',
        )
        return
    click.secho('[+] AGNI updated', fg='green')


@cli.command()
@click.option(
    '--export-env',
    is_flag=True,
    help='Record FWL_DATA, RAD_DIR for a built SOCRATES, PYTHON_JULIAPKG_EXE when it names an '
    'executable Julia, and FC_DIR for a built FastChem, in the active conda env (not base) '
    'and in .proteus-env.sh.',
)
@click.option(
    '--config-path',
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help='Path to the TOML config file (default: input/all_options.toml under the PROTEUS root)',
)
def install_all(export_env: bool, config_path: Path | None):
    """Install PROTEUS, required submodules, and get lookup data from online sources."""
    # --- Step 0: Locate the source tree and check available disk space ---
    # The submodule checkouts and the build land under the source tree, so
    # the disk gate measures that filesystem, not the caller's.
    root = _resolve_proteus_root()
    if config_path is None:
        config_path = root / 'input' / 'all_options.toml'
    available_disk_space_in_B = shutil.disk_usage(root).free
    G = 1e9
    available_disk_space_in_GB = available_disk_space_in_B / G
    required_disk_space_in_GB = 5
    if not available_disk_space_in_GB > required_disk_space_in_GB:
        click.secho(
            f'[!] You have {available_disk_space_in_GB:.3f} GB of disk space at your disposal.',
            fg='yellow',
        )
        click.secho(
            f'   To be safe, make sure you have at least {required_disk_space_in_GB:d} GB of free disk space.',
            fg='yellow',
        )
        click.secho(
            "[x] Aborting installation -- 'proteus install-all'.",
            fg='red',
        )
        raise SystemExit(1)

    # --- Step 1: FWL_DATA directory ---
    fwl_data = resolve_fwl_data_dir()
    fwl_data.mkdir(parents=True, exist_ok=True)
    click.secho(f'[+] FWL_DATA directory: {fwl_data}', fg='green')

    # --- Step 1b: Install CVODE (Aragog's default solver) ---
    _install_cvode(root)

    # --- Step 2: Install SOCRATES ---
    socrates_dir = root / 'socrates'
    if not socrates_dir.exists():
        click.secho('[+] Installing SOCRATES...', fg='blue')
        try:
            subprocess.run(['bash', str(root / 'tools' / 'get_socrates.sh')], check=True)
        except subprocess.CalledProcessError as e:
            click.secho('[x] Failed to install SOCRATES', fg='red')
            click.echo(e)
            raise SystemExit(1)
    else:
        click.secho('[+] SOCRATES already present', fg='green')

    rad_dir = _built_socrates(root)
    if rad_dir is not None:
        os.environ.setdefault('RAD_DIR', str(rad_dir))

    # --- Step 3: Julia check ---
    if not is_julia_installed():
        click.secho('[!] Julia not found in PATH.', fg='yellow')
        click.secho(
            '   Proteus requires Julia for AGNI.',
            fg='yellow',
        )
        click.secho(
            '   Please install Julia and ensure it is accessible via your shell PATH.',
            fg='yellow',
        )
        click.secho(f'   Current PATH: {os.environ["PATH"]}', fg='white')
        click.secho(
            "[x] Aborting installation -- 'proteus install-all' cannot proceed without Julia.",
            fg='red',
        )
        raise SystemExit(1)
    _report_julia()

    # --- Step 4: Install AGNI at its pinned commit, as install.sh does ---
    agni_dir = root / 'AGNI'
    if not agni_dir.exists():
        click.secho('[+] Installing AGNI...', fg='blue')
        try:
            subprocess.run(
                ['bash', str(root / 'tools' / 'get_agni.sh'), '0'],
                cwd=root,
                env=_env_with_proteus_julia(root),
                stdin=subprocess.DEVNULL,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            click.secho('[x] Failed to install AGNI', fg='red')
            click.echo(e)
            raise SystemExit(1)
    else:
        click.secho('[+] AGNI already present', fg='green')

    # --- Step 5: Export environment variables ---
    if export_env:
        _export_env(root, _install_vars(root, fwl_data, rad_dir))

    # --- Step 6: Update input data ---
    _download_default_phoenix()
    _update_input_data(config_path)
    (root / 'output').mkdir(exist_ok=True)

    click.secho('[+] PROTEUS installation completed!', fg='green')


@cli.command()
@click.option(
    '--export-env',
    is_flag=True,
    help='Record FWL_DATA, RAD_DIR for a built SOCRATES, PYTHON_JULIAPKG_EXE when it names an '
    'executable Julia, and FC_DIR for a built FastChem, again in the active conda env (not '
    'base) and in .proteus-env.sh.',
)
@click.option(
    '--config-path',
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help='Path to the TOML config file (default: input/all_options.toml under the PROTEUS root)',
)
def update_all(export_env: bool, config_path: Path | None):
    """Update PROTEUS, submodules, and lookup data from online sources."""
    # --- Step 0: Locate the source tree and check available disk space ---
    # The submodule refreshes and the build land under the source tree, so
    # the disk gate measures that filesystem, not the caller's.
    root = _resolve_proteus_root()
    if config_path is None:
        config_path = root / 'input' / 'all_options.toml'
    available_disk_space_in_B = shutil.disk_usage(root).free
    G = 1e9
    available_disk_space_in_GB = available_disk_space_in_B / G
    required_disk_space_in_GB = 5
    if not available_disk_space_in_GB > required_disk_space_in_GB:
        click.secho(
            f'[!] You have {available_disk_space_in_GB:.3f} GB of disk space at your disposal.',
            fg='yellow',
        )
        click.secho(
            f'   To be safe, make sure you have at least {required_disk_space_in_GB:d} GB of free disk space.',
            fg='yellow',
        )
        click.secho(
            "[x] Aborting installation -- 'proteus update-all'.",
            fg='red',
        )
        raise SystemExit(1)

    # --- Step 1: update all Python packages ---
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', '-e', str(root)], check=True)

    # --- Step 1b: ensure the SUNDIALS CVODE solver (Aragog's default solver) ---
    _install_cvode(root)

    # --- Step 2: FWL_DATA check ---
    # resolve_fwl_data_dir always returns a path; an update only makes sense
    # when the data directory from a previous installation actually exists.
    fwl_data = resolve_fwl_data_dir()
    if not fwl_data.is_dir():
        click.secho(
            f'[x] FWL_DATA directory not found at {fwl_data}. Run `proteus install-all` first.',
            fg='red',
        )
        raise SystemExit(1)
    click.secho(f'[+] Using FWL_DATA: {fwl_data}', fg='green')

    # --- Step 3: Update SOCRATES ---
    socrates_dir = root / 'socrates'
    if socrates_dir.exists():
        click.secho('[+] Updating SOCRATES...', fg='blue')
        try:
            subprocess.run(['bash', str(root / 'tools' / 'get_socrates.sh')], check=True)
            click.secho('[+] SOCRATES updated', fg='green')
        except subprocess.CalledProcessError as e:
            click.secho('[x] Failed to update SOCRATES', fg='red')
            click.echo(e)
    else:
        click.secho('[!] SOCRATES not found. Run `proteus install-all`.', fg='yellow')

    rad_dir = _built_socrates(root)
    if rad_dir is not None:
        os.environ.setdefault('RAD_DIR', str(rad_dir))

    # --- Step 4: Julia check ---
    if not is_julia_installed():
        click.secho('[!] Julia not found in PATH.', fg='yellow')
        click.secho('   Cannot update AGNI without Julia.', fg='yellow')
    else:
        _report_julia()
        # --- Step 5: Bring AGNI to its pinned commit ---
        if (root / 'AGNI').exists():
            _update_agni(root)
        else:
            click.secho('[!] AGNI not found. Run `proteus install-all`.', fg='yellow')

    # --- Step 6: Refresh environment exports ---
    if export_env:
        _export_env(root, _install_vars(root, fwl_data, rad_dir))

    # --- Step 7: Update input data ---
    _update_input_data(config_path)

    click.secho('[+] PROTEUS update completed!', fg='green')


if __name__ == '__main__':
    cli()
