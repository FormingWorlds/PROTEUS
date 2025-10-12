from __future__ import annotations

import os

# Prevent workers from using each other's CPUs to avoid
#     oversubscription and improve performance
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # noqa

import shutil
import subprocess
import sys
from pathlib import Path

import click

from proteus import Proteus
from proteus import __version__ as proteus_version
from proteus.config import read_config_object
from proteus.utils.data import download_sufficient_data
from proteus.utils.logs import setup_logger

config_option = click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(
        exists=True, dir_okay=False, path_type=Path, resolve_path=True
    ),
    help="Path to config file",
    required=True,
)


@click.group()
@click.version_option(version=proteus_version)
def cli():
    pass


# ----------------
# 'plot' command
# ----------------


def list_plots(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    from .plot import plot_dispatch

    click.echo(" ".join(plot_dispatch))
    sys.exit()


@click.command()
@click.argument("plots", nargs=-1)
@config_option
@click.option(
    "-l",
    "--list",
    is_flag=True,
    default=False,
    help="List available plots and exit",
    is_eager=True,
    expose_value=False,
    callback=list_plots,
)
def plot(plots, config_path: Path):
    """(Re-)generate plots from completed run"""
    from .plot import plot_dispatch

    click.echo(f"Config: {config_path}")

    handler = Proteus(config_path=config_path)

    if "all" in plots:
        plots = list(plot_dispatch.keys())

    for plot in plots:
        if plot not in plot_dispatch.keys():
            click.echo(f"Invalid plot: {plot}")
        else:
            click.echo(f"Plotting: {plot}")
            plot_func = plot_dispatch[plot]
            plot_func(handler=handler)


cli.add_command(plot)

# ----------------
# 'start' command
# ----------------


@click.command()
@config_option
@click.option(
    "-r",
    "--resume",
    is_flag=True,
    default=False,
    help="Resume simulation from disk",
)
@click.option(
    "-o",
    "--offline",
    is_flag=True,
    default=False,
    help="Run in offline mode; do not connect to the internet",
)
def start(config_path: Path, resume: bool, offline: bool):
    """Start proteus run"""
    runner = Proteus(config_path=config_path)
    runner.start(resume=resume, offline=offline)


cli.add_command(start)

# --------------
# 'get' command, with subcommands
# --------------


@click.group()
def get():
    """Get data and modules"""
    pass


@click.command()
@click.option(
    "-n", "--name", "name", type=str, help="Name of spectral file group"
)
@click.option("-b", "--bands", "bands", type=str, help="Number of bands")
def spectral(**kwargs):
    """Get spectral files

    By default, download all files.
    """
    from .utils.data import download_spectral_file

    download_spectral_file(kwargs["name"], kwargs["bands"])


@click.command()
def stellar():
    """Get stellar spectra"""
    from .utils.data import download_stellar_spectra, download_stellar_tracks

    for track in ["Spada", "Baraffe"]:
        download_stellar_tracks(track)
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

# ----------------
# doctor utility
# ----------------


@click.command()
def doctor():
    """Diagnose your PROTEUS installation"""
    from .doctor import doctor_entry

    doctor_entry()


cli.add_command(doctor)

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
        logpath=runner.directories["output"] + "offchem.log",
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
        logpath=runner.directories["output"] + "observe.log",
        logterm=True,
        level=runner.config.params.out.logging,
    )
    runner.observe()


cli.add_command(offchem)
cli.add_command(observe)

# ----------------
# GridPROTEUS and BO inference scheme
# ----------------


@click.command()
@config_option
def grid(config_path: Path):
    """Run GridPROTEUS to generate a grid of forward models"""
    from proteus.grid.manage import grid_from_config

    grid_from_config(config_path)


@click.command()
@config_option
def infer(config_path: Path):
    """Use Bayesian optimisation to infer parameters from observables"""
    from proteus.inference.inference import infer_from_config

    infer_from_config(config_path)


cli.add_command(grid)
cli.add_command(infer)

# ----------------
# installer
# ----------------


def resolve_fwl_data_dir() -> Path:
    """Return the FWL_DATA path (env or default)."""
    if "FWL_DATA" in os.environ:
        return Path(os.environ["FWL_DATA"])
    else:
        # Return a default path to install FWL data.
        return Path(__file__).resolve().parent.parent / "FWL_DATA"


def append_to_shell_rc(
    var: str, value: str, shell: str | None = None
) -> Path | None:
    """Append an export line to the appropriate shell rc file."""
    shell = shell or os.environ.get("SHELL", "")
    shell_rc_map = {
        "bash": ".bashrc",
        "zsh": ".zshrc",
        "fish": ".config/fish/config.fish",
    }

    for key, rc in shell_rc_map.items():
        if key in shell:
            rc_path = Path.home() / rc
            export_line = f'export {var}="{value}"\n'

            if rc_path.exists() and export_line in rc_path.read_text():
                return None  # Already present

            rc_path.parent.mkdir(parents=True, exist_ok=True)
            with rc_path.open("a") as f:
                f.write(f"\n# Set by PROTEUS installer\n{export_line}")
            return rc_path

    return None


def is_julia_installed() -> bool:
    return shutil.which("julia") is not None


@cli.command()
@click.option(
    "--export-env", is_flag=True, help="Add FWL_DATA and RAD_DIR to shell rc."
)
def install_all(export_env: bool):
    # --- Step 0: Check available disk space---
    available_disk_space_in_B = shutil.disk_usage(".").free
    G = 1e9
    available_disk_space_in_GB = available_disk_space_in_B / G
    required_disk_space_in_GB = 5
    if not available_disk_space_in_GB > required_disk_space_in_GB:
        click.secho(
            f"⚠️ You have {available_disk_space_in_GB:.3f} GB of disk space at your disposal.",
            fg="yellow",
        )
        click.secho(
            f"   To be safe, make sure you have at least {required_disk_space_in_GB:d} GB of free disk space.",
            fg="yellow",
        )
        click.secho(
            "❌ Aborting installation — 'proteus install-all'.",
            fg="red",
        )
        raise SystemExit(1)

    """Install SOCRATES, AGNI, and configure PROTEUS environment."""

    # --- Step 1: FWL_DATA directory ---
    fwl_data = resolve_fwl_data_dir()
    fwl_data.mkdir(parents=True, exist_ok=True)
    click.secho(f"✅ FWL_DATA directory: {fwl_data}", fg="green")

    # --- Step 2: Install SOCRATES ---
    root = Path.cwd()
    socrates_dir = root / "socrates"
    if not socrates_dir.exists():
        click.secho("🌤️ Installing SOCRATES...", fg="blue")
        try:
            subprocess.run(["bash", "tools/get_socrates.sh"], check=True)
        except subprocess.CalledProcessError as e:
            click.secho("❌ Failed to install SOCRATES", fg="red")
            click.echo(e)
            raise SystemExit(1)
    else:
        click.secho("✅ SOCRATES already present", fg="green")

    rad_dir = socrates_dir.resolve()
    os.environ.setdefault("RAD_DIR", str(rad_dir))

    env = os.environ.copy()

    # --- Step 3: Julia check ---
    if not is_julia_installed():
        click.secho("⚠️ Julia not found in PATH.", fg="yellow")
        click.secho(
            "   Proteus requires Julia for AGNI.",
            fg="yellow",
        )
        click.secho(
            "   Please install Julia and ensure it is accessible via your shell PATH.",
            fg="yellow",
        )
        click.secho(f'   Current PATH: {os.environ["PATH"]}', fg="white")
        click.secho(
            "❌ Aborting installation — 'proteus install-all' cannot proceed without Julia.",
            fg="red",
        )
        raise SystemExit(1)

    # --- Step 4: Install AGNI ---
    agni_dir = root / "AGNI"
    if not agni_dir.exists():
        click.secho("🧪 Installing AGNI...", fg="blue")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/nichollsh/AGNI.git"],
                check=True,
            )
            subprocess.run(
                ["bash", "src/get_agni.sh"], cwd=agni_dir, env=env, check=True
            )
        except subprocess.CalledProcessError as e:
            click.secho("❌ Failed to install AGNI", fg="red")
            click.echo(e)
            raise SystemExit(1)
    else:
        click.secho("✅ AGNI already present", fg="green")

    # --- Step 5: Export environment variables ---
    if export_env:
        for var, value in {"FWL_DATA": fwl_data, "RAD_DIR": rad_dir}.items():
            rc_file = append_to_shell_rc(var, str(value))
            if rc_file:
                click.secho(f"✅ Exported {var} to {rc_file}", fg="green")
            else:
                click.secho(
                    f"ℹ️ {var} already exported or shell not recognized",
                    fg="cyan",
                )
        click.secho(
            "🔁 Please run: source ~/.bashrc (or your shell rc)", fg="yellow"
        )

    click.secho("🎉 PROTEUS installation completed!", fg="green")


@cli.command()
@click.option(
    "--export-env",
    is_flag=True,
    help="Re-add FWL_DATA and RAD_DIR to shell rc.",
)
@click.option(
    "--config-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("input/all_options.toml"),
    help="Path to the TOML config file",
)
def update_all(export_env: bool, config_path: Path):
    # --- Step 0: Check available disk space---
    available_disk_space_in_B = shutil.disk_usage(".").free
    G = 1e9
    available_disk_space_in_GB = available_disk_space_in_B / G
    required_disk_space_in_GB = 5
    if not available_disk_space_in_GB > required_disk_space_in_GB:
        click.secho(
            f"⚠️ You have {available_disk_space_in_GB:.3f} GB of disk space at your disposal.",
            fg="yellow",
        )
        click.secho(
            f"   To be safe, make sure you have at least {required_disk_space_in_GB:d} GB of free disk space.",
            fg="yellow",
        )
        click.secho(
            "❌ Aborting installation — 'proteus update-all'.",
            fg="red",
        )
        raise SystemExit(1)
    """Update SOCRATES, AGNI, and refresh PROTEUS environment."""

    root = Path.cwd()
    # --- Step 1: update all Python packages ---
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "-e", "."], check=True
    )

    # --- Step 2: FWL_DATA check ---
    try:
        fwl_data = resolve_fwl_data_dir()
    except EnvironmentError:
        click.secho(
            "❌ FWL_DATA not set. Run `proteus install-all` first.", fg="red"
        )
        raise SystemExit(1)
    click.secho(f"📂 Using FWL_DATA: {fwl_data}", fg="green")

    # --- Step 3: Update SOCRATES ---
    socrates_dir = root / "socrates"
    if socrates_dir.exists():
        click.secho("🌤️ Updating SOCRATES...", fg="blue")
        try:
            subprocess.run(["bash", "tools/get_socrates.sh"], check=True)
            click.secho("✅ SOCRATES updated", fg="green")
        except subprocess.CalledProcessError as e:
            click.secho("❌ Failed to update SOCRATES", fg="red")
            click.echo(e)
    else:
        click.secho(
            "⚠️ SOCRATES not found. Run `proteus install-all`.", fg="yellow"
        )

    rad_dir = socrates_dir.resolve()
    os.environ.setdefault("RAD_DIR", str(rad_dir))

    # --- Step 4: Julia check ---
    if not is_julia_installed():
        click.secho("⚠️ Julia not found in PATH.", fg="yellow")
        click.secho("   Cannot update AGNI without Julia.", fg="yellow")
    else:
        # --- Step 5: Update AGNI ---
        agni_dir = root / "AGNI"
        if agni_dir.exists():
            click.secho("🧪 Updating AGNI...", fg="blue")
            try:
                subprocess.run(["git", "pull"], cwd=agni_dir, check=True)
                subprocess.run(
                    ["bash", "src/get_agni.sh"],
                    cwd=agni_dir,
                    env=os.environ,
                    check=True,
                )
                click.secho("✅ AGNI updated", fg="green")
            except subprocess.CalledProcessError as e:
                click.secho("❌ Failed to update AGNI", fg="red")
                click.echo(e)
        else:
            click.secho(
                "⚠️ AGNI not found. Run `proteus install-all`.", fg="yellow"
            )

    # --- Step 6: Refresh environment exports ---
    if export_env:
        for var, value in {"FWL_DATA": fwl_data, "RAD_DIR": rad_dir}.items():
            rc_file = append_to_shell_rc(var, str(value))
            if rc_file:
                click.secho(f"✅ Exported {var} to {rc_file}", fg="green")
            else:
                click.secho(
                    f"ℹ️ {var} already exported or shell not recognized",
                    fg="cyan",
                )
        click.secho(
            "🔁 Please run: source ~/.bashrc (or your shell rc)", fg="yellow"
        )

    # --- Step 7: Update input data.
    if config_path.exists():
        # Only try data download if a config file is present.
        configuration = read_config_object(config_path)
        download_sufficient_data(configuration)
        click.secho("✅ Additional data has been downloaded.", fg="green")
    else:
        click.echo(
            f"⚠️ No config file found at {config_path}, skipping data download."
        )

    click.secho("🎉 PROTEUS update completed!", fg="green")


if __name__ == "__main__":
    cli()
