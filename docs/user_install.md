# User Install (Deprecated)

!!! warning "Deprecated"
    This installation method is deprecated. We recommend following the main [Installation](installation.md) guide instead, which provides a full developer setup with editable submodule installations.

---

## Requirements

1. conda (miniconda or miniforge)
2. git (install via conda if needed: `conda install git`)
3. Julia installation (see [Installation guide](installation.md#3-install-julia))
4. ~20 GB of disk space
5. ~30 minutes of your time (mostly waiting)

!!! note "miniforge and HDF5/NetCDF conflicts"
    There is currently a known conflict between Julia and Conda versions of HDF5 and/or NetCDF libraries when using miniforge. If you encounter issues, try using miniconda instead.

## Steps

1. `git clone https://github.com/FormingWorlds/PROTEUS.git`
2. `cd PROTEUS`
3. `conda env create -f environment.yml`
4. `conda activate proteus`
5. `pip install -e .`
6. `proteus install-all --export-env`

!!! note "macOS: PETSc compilation on Apple Silicon"
    If `proteus install-all` fails during the PETSc step on Apple Silicon (M1/M2/M3/Ultra), run the automated fix script:

    ```console
    ./tools/fix_petsc_compile.sh
    ```

    See the [Troubleshooting](troubleshooting.md#petsc-compilation-fails-on-apple-silicon) page for details.

## Before running PROTEUS

If you want to start running PROTEUS right away (from the same shell that you used to install PROTEUS), reload your environment variables:

=== "bash"

    ```console
    source ~/.bashrc
    conda activate proteus
    ```

=== "zsh"

    ```console
    source ~/.zshrc
    conda activate proteus
    ```

If you chose `proteus install-all` without `--export-env`, you will need to set your `PATH`, `FWL_DATA`, Julia, and `RAD_DIR` environment variables manually.

When you log into the machine where you installed PROTEUS through `proteus install-all --export-env`, the environment variables `FWL_DATA` and `RAD_DIR` will already be set (because your shell config file was updated during installation). You still need to run `conda activate proteus`.

## Updating PROTEUS

1. `conda activate proteus` (if not already activated)
2. `proteus update-all`
