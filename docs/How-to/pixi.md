# Install with pixi

!!! warning "Experimental"
    Installing with [pixi](https://pixi.sh) is an experimental alternative to
    conda. It may change or be removed.

pixi builds the Python environment from the `pixi.toml` in the PROTEUS
repository, so it replaces only the conda steps of the
[installation guide](installation.md). Everything else is unchanged.

1. Install the system packages as in step 1 of the
   [installation guide](installation.md), and install pixi.
2. Clone PROTEUS as in step 2, but skip `conda create` and `conda activate`.
3. Run the installer through pixi instead of `bash install.sh`:

    ```console
    pixi run bash install.sh
    ```

    `pixi run` creates the environment on first use, so there is no separate
    setup step.

Afterwards, prefix commands with `pixi run`, or use `pixi shell` for an
activated shell:

```console
pixi run proteus start --offline -c input/dummy.toml
```

## The lock file

`pixi.lock` is not committed, so a fresh clone resolves against current
packages. pixi writes one on first use and reuses it afterwards, so a later
install in the same directory reproduces exactly what it records. Run
`pixi update` to re-resolve to newer versions, or delete `pixi.lock` and
`.pixi/` to start from nothing.

## Kapteyn

Kapteyn provides no netcdf-fortran. As in the
[conda instructions](kapteyn_cluster_guide.md), it has to come from the
environment, and SOCRATES needs its library directory on `LD_LIBRARY_PATH` at
run time:

```console
pixi add netcdf-fortran
```

and add to `pixi.toml`:

```toml
[activation.env]
LD_LIBRARY_PATH = "$CONDA_PREFIX/lib"
```

Both are local edits to a tracked file.
[#886](https://github.com/FormingWorlds/PROTEUS/issues/886) looks into removing
the need for the run-time path.
