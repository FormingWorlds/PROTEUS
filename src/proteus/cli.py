from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import click

from proteus import Proteus
from proteus import __version__ as proteus_version
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
    from .utils.data import download_evolution_tracks, download_stellar_spectra

    for track in ["Spada", "Baraffe"]:
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

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "FWL_DATA"
print(f"[DEBUG] DEFAULT_DATA_DIR resolves to: {DEFAULT_DATA_DIR}")


def resolve_fwl_data_dir() -> Path:
    """Return the FWL_DATA path (env or default)."""
    return Path(os.environ.get("FWL_DATA", DEFAULT_DATA_DIR))


def append_to_shell_rc(var: str, value: str, shell: str = None) -> Path | None:
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


def install_julia_and_get_bin_path() -> Path | None:
    click.secho("🐹 Julia is not installed. Installing Julia...", fg="blue")
    if platform.system() == "Windows":
        click.secho(
            "❌ Auto-installing Julia on Windows is not supported.", fg="red"
        )
        click.secho(
            "👉 Please install Julia manually: https://julialang.org/downloads/",
            fg="yellow",
        )
        raise SystemExit(1)

    try:
        proc = subprocess.run(
            ["bash", "-c", "curl -fsSL https://install.julialang.org | sh"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Search for the line that shows the install path
        match = re.search(r"installed at: (.+/bin)", proc.stdout)
        click.secho(
            "✅ Julia installed. You may need to restart your shell.",
            fg="green",
        )
        if match:
            julia_bin = Path(match.group(1))
            return julia_bin
        else:
            click.secho(
                "⚠️  Could not determine Julia install path from output."
            )
            click.secho(proc.stdout)
            return None
    except subprocess.CalledProcessError as e:
        click.secho(
            "❌ Failed to install Julia. Please install manually.", fg="red"
        )
        if e.stderr:
            click.secho(e.stderr.strip(), fg="yellow")
        raise SystemExit(1)


@cli.command()
@click.option(
    "--export-env", is_flag=True, help="Add FWL_DATA and RAD_DIR to shell rc."
)
def install_all(export_env: bool):
    """Install Julia (if needed), SOCRATES, AGNI, and configure PROTEUS environment."""

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
    os.environ["RAD_DIR"] = str(rad_dir)

    env = os.environ.copy()

    # --- Step 3: Julia check ---
    if not is_julia_installed():
        julia_bin_path = install_julia_and_get_bin_path()
        if julia_bin_path is not None:
            # Pick up the current shell PATH (which includes the Julia path)
            env["PATH"] = f'{julia_bin_path}:{env["PATH"]}'
        else:
            click.secho(
                "⚠️ Julia path not found — subsequent steps may fail",
                fg="yellow",
            )

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
                ["bash", "-c", 'echo "PATH seen by get_agni.sh: $PATH"']
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


if __name__ == "__main__":
    cli()
