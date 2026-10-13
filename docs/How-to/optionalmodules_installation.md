# Installation: optional modules

!!! info "Install PROTEUS first"
    Please follow the [quick installation guide](installation.md) or [manual installation guide](manual_installation.md) before installing optional modules.

!!! note "Set up by the installer"
    `bash install.sh` sets up Obliqua, atmodeller, VULCAN and Morrigan. Their sections on this page repeat that step by hand.

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

    The PETSc script downloads the PETSc 3.19.0 source archive from Zenodo (or its
    DataverseNL mirror), checks its SHA-256, builds it and configures
    `PETSC_DIR` and `PETSC_ARCH` automatically.

    !!! note "Fedora / RHEL"
        If you encounter errors moving libraries, see
        [Troubleshooting: PETSc on Fedora/RHEL](troubleshooting.md#cannot-compile-petsc-error-moving-libraries-fedorarhel).

=== "macOS"

    The script uses the MPI on your PATH, for example Homebrew's Open MPI. If you
    encounter issues, see
    [Troubleshooting: PETSc on Apple Silicon](troubleshooting.md#petsc-compilation-fails-on-apple-silicon).

## Solid-phase tidal heating (LovePy)

LovePy computes tidal dissipation for solid-phase planetary interiors. It is
written in Julia.

```console
bash tools/get_lovepy.sh
```

## Multi-phase tidal response (Obliqua)

The installer sets Obliqua up automatically. Obliqua computes the multi-phase
tidal response of the planet from its interior structure and rheology. It is
written in Julia. To repeat the setup step by hand:

```console
bash tools/get_obliqua.sh
```

The options `--register` (adds Obliqua to the default Julia environment) and
`--test` (runs its own tests) exist. `--force` replaces an Obliqua clone with a
new one at the pin. The script accepts an empty folder and a path that does
not exist, and a link to a folder stands for that folder. With or without
`--force` it refuses the PROTEUS folder, your home folder and a folder that
holds one of them, a file or a broken link, a path that goes through a folder
that does not exist and then through `.` or `..`, and a folder with files that is
not an Obliqua project; with `--force` it also refuses a folder with files
that is not a git clone of its own with a commit, and a folder whose `.git` is a
file (a git worktree or a submodule).

## Synthetic observations (PetitRADTRANS)

[PetitRADTRANS](https://petitradtrans.readthedocs.io/en/latest/) generates
synthetic transmission and secondary eclipse spectra.

```console
bash tools/get_prt.sh
```

!!! note
    PetitRADTRANS downloads around 2 GB of opacity data on first use.

## Alternative outgassing (atmodeller)

The installer sets atmodeller up automatically as an editable clone in `Atmodeller/`.
[atmodeller](https://github.com/djbower/atmodeller) is an alternative to
CALLIOPE, selected with `outgas.module = "atmodeller"`. The clone comes from
[FormingWorlds/atmodeller](https://github.com/FormingWorlds/atmodeller) at the
commit that `pyproject.toml` pins under `[tool.proteus.modules.atmodeller]`.
`bash tools/get_atmodeller.sh --force` replaces whatever is in `Atmodeller/`
with a new clone, also a folder with files that is not a git clone.
To repeat the setup step by hand:

```console
bash tools/get_atmodeller.sh
```

For an installation without the installer, install the package via its extra:

```console
pip install "fwl-proteus[atmodeller]"
```

!!! warning "License"
    atmodeller is distributed under the GPL-3.0 license; review its terms
    before installing.

## Rock vapours (LavAtmos + ThermoEngineLite)

[LavAtmos](https://github.com/FormingWorlds/LavAtmos) computes the
thermodynamic vapourisation equilibrium of silicate melt, adding rock-vapour
species to the atmosphere. It is optional, selected with
`outgas.vapourise = true`, and is not installed with PROTEUS by default.

LavAtmos needs [ThermoEngineLite](https://github.com/FormingWorlds/ThermoEngineLite)
for melt thermodynamics and the FastChem chemistry solver already installed
alongside AGNI (`FC_DIR`; see [Install AGNI and FastChem](manual_installation.md#6-install-agni-and-fastchem)
if that variable is not set).

```console
bash tools/get_thermoenginelite.sh
bash tools/get_lavatmos.sh
```

!!! warning "ThermoEngineLite build requirements"
    ThermoEngineLite compiles a C++/Python extension and requires `clang` and
    `pip` on `PATH`. Compilation takes around 20 minutes.

`get_lavatmos.sh` clones the checkout but does not set an environment
variable for you. Set the printed path in your conda environment:

```console
conda env config vars set -n proteus LAVA_DIR='<path-to-LavAtmos>/'
conda deactivate && conda activate proteus
```

See [`input/tutorials/tutorial_vapourise.toml`](https://github.com/FormingWorlds/PROTEUS/blob/main/input/tutorials/tutorial_vapourise.toml) for an example configuration.

## Parameter inference (Bayesian optimisation)

`proteus infer` runs the Bayesian-optimisation scheme, which is built on
PyTorch, BoTorch and GPyTorch. Those three are a several-hundred-megabyte
install that nothing else in PROTEUS uses, so a forward model or a
`proteus grid` sweep does not need them:

```console
pip install "fwl-proteus[inference]"
```

On Linux the default PyTorch wheel also pulls in the CUDA runtime packages.
The scheme runs on CPU, so install PyTorch from its CPU index first if you do
not want them:

```console
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "fwl-proteus[inference]"
```

Without the extra, `proteus infer` stops with a message naming the missing
package; every other command is unaffected. See the
[inference guide](inference.md) for how to configure a run.

## Atmospheric chemistry (VULCAN)

The installer sets VULCAN up automatically as an editable clone in `VULCAN/`.
VULCAN is an atmospheric photochemistry module, selected with
`atmos_chem.module = "vulcan"`. To repeat the setup step by hand:

```console
bash tools/get_vulcan.sh
```

For an installation without the installer, install the package via its extra:

```console
pip install "fwl-proteus[vulcan]"
```

!!! warning "License"
    VULCAN is distributed under the GPL-3.0 license; review its terms
    before installing.

## Protoplanet accretion (Morrigan)

The installer sets Morrigan up automatically as an editable clone in `Morrigan/`.
Morrigan is a giant-impact accretion module, selected with
`accretion.module = "morrigan"`. It evolves a system of planetary embryos
through orbital crossings and collisions and returns the impact history
that the coupled run replays. To repeat the setup step by hand:

```console
bash tools/get_morrigan.sh
```

For an installation without the installer, install the package via its extra:

```console
pip install "fwl-proteus[morrigan]"
```

An accretion run needs an interior module that can re-melt the mantle
after an impact. Aragog is the production choice; SPIDER is refused at
configuration load because it has no re-melt path.

Two accretion modules need no installation at all. `accretion.module =
"dummy"` builds an impact history from scaling laws, and `accretion.module
= "timeline"` replays one from a file, which is how a published or
externally computed impact history drives a run. Both apply the same
impact physics as the dynamical model.
