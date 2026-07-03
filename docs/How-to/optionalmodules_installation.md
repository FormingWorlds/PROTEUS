# Installation: optional modules

!!! info "Install PROTEUS first"
    Please follow the [quick installation guide](installation.md) or [manual installation guide](manual_installation.md) before installing optional modules.

## SPIDER and PETSc {#optional-setup-petsc}

SPIDER is a C-based interior thermal evolution solver. It requires PETSc
(a numerical computing library) and a C compiler. Most configurations use
Aragog instead, which is written in Python/JAX and needs no additional
compiled dependencies.

!!! warning
    PETSc requires Python <= 3.12. Make sure your conda environment uses
    Python 3.12.

```console
bash tools/get_petsc.sh
bash tools/get_spider.sh
```

=== "Linux"

    The PETSc script downloads a pre-compiled version from OSF and configures
    `PETSC_DIR` and `PETSC_ARCH` automatically.

    !!! note "Fedora / RHEL"
        If you encounter errors moving libraries, see
        [Troubleshooting: PETSc on Fedora/RHEL](troubleshooting.md#cannot-compile-petsc-error-moving-libraries-fedorarhel).

=== "macOS"

    The script detects Apple Silicon vs Intel and uses Homebrew's MPI. If you
    encounter issues, see
    [Troubleshooting: PETSc on Apple Silicon](troubleshooting.md#petsc-compilation-fails-on-apple-silicon).

## Multi-phase tidal heating (LovePy)

LovePy computes tidal dissipation for multi-phase planetary interiors. It is
written in Julia.

```console
bash tools/get_lovepy.sh
```

## Synthetic observations (PetitRADTRANS)

[PetitRADTRANS](https://petitradtrans.readthedocs.io/en/latest/) generates
synthetic transmission and secondary eclipse spectra.

```console
bash tools/get_prt.sh
```

!!! note
    PetitRADTRANS downloads around 2 GB of opacity data on first use.

## Alternative outgassing (atmodeller)

[atmodeller](https://github.com/djbower/atmodeller) is an optional
alternative to CALLIOPE, selected with `outgas.module = "atmodeller"`. It is
not installed with PROTEUS by default; a standard run uses CALLIOPE.

```console
pip install "fwl-proteus[atmodeller]"
```

!!! warning "License"
    atmodeller is distributed under the GPL-3.0 license; review its terms
    before installing.

## Rock vapours (LavAtmos + ThermoEngineLite)

[LavAtmos](https://github.com/FormingWorlds/LavAtmos) computes the
thermodynamic vapourisation equilibrium of silicate melt, adding rock-vapour
species to the outgassed atmosphere. It is optional, selected with
`outgas.vapourise = true`, and is not installed with PROTEUS by default.
LavAtmos needs [ThermoEngineLite](https://github.com/FormingWorlds/ThermoEngineLite)
for melt thermodynamics and the FastChem chemistry solver already installed
alongside AGNI (`FC_DIR`; see
[Install AGNI and FastChem](manual_installation.md#6-install-agni-and-fastchem)
if that variable is not set).

```console
bash tools/get_thermoenginelite.sh
bash tools/get_lavatmos.sh
```

!!! warning "ThermoEngineLite build requirements"
    ThermoEngineLite compiles a C++/Python extension and requires `clang` and
    `pip` on `PATH`. Compilation takes around 20 minutes.

`get_lavatmos.sh` clones the checkout but does not export an environment
variable for you. Export the printed path and add it to your shell rc file:

```console
export LAVA_DIR='<path-to-LavAtmos>/'
```

Enable rock vapour outgassing in your config:

```toml
[outgas]
vapourise = true
```

See [`input/vapourise.toml`](https://github.com/FormingWorlds/PROTEUS/blob/main/input/vapourise.toml)
for a full example configuration.

## Atmospheric chemistry (VULCAN)

VULCAN is an optional atmospheric-photochemistry backend, selected with
`atmos_chem.module = "vulcan"`. It is not required for a standard PROTEUS
run. Install it from PyPI:

```console
pip install "fwl-proteus[vulcan]"
```

For local development, install as an editable checkout instead:

```console
bash tools/get_vulcan.sh
```

!!! warning "License"
    VULCAN is distributed under the GPL-3.0 license; review its terms
    before installing.
