# Local Machine Guide

These steps should be performed before installing PROTEUS on your computer.
They do not apply when running PROTEUS on a server or HPC cluster. For instructions on configuring PROTEUS on a remote machine, see the cluster guide pages.

Once you have followed these steps, go back to the main [installation](installation.md) guide page.

## macOS

1. Open the terminal and install the Xcode developer tools:

    ```console
    xcode-select --install
    ```

2. Install the required libraries via the most appropriate method for you.

    **[Homebrew](https://brew.sh/)** (recommended)

    ```console
    brew install netcdf netcdf-fortran wget gcc open-mpi
    ```

    **[MacPorts](https://www.macports.org/)**

    ```console
    sudo port install netcdf-fortran +gcc8 wget
    ```

    !!! note "Apple Silicon (M1/M2/M3/Ultra)"
        The `gcc` and `open-mpi` packages are required for compiling PETSc on Apple Silicon Macs. If you skip these, you may encounter compilation errors during the PETSc installation step. See the [Troubleshooting](troubleshooting.md#petsc-compilation-fails-on-apple-silicon) page for details.

3. macOS Catalina (10.15) and later uses `zsh` as the default shell. Replace `.bashrc` with `.zshrc` throughout the installation instructions if you are using the default shell.

## Debian / Ubuntu Linux

Install `gfortran` and the NetCDF libraries via your package manager:

```console
sudo apt install libnetcdff-dev gfortran
```

## Fedora / RedHat Linux

Install the required libraries via your package manager:

```console
sudo dnf install gcc gcc-gfortran gcc-c++ netcdf netcdf-fortran netcdf-fortran-devel \
    lapack lapack-devel lapack-static sundials-mpich openmpi openmpi-devel f2c f2c-libs
```

## Microsoft Windows

It is not recommended to install and use PROTEUS on Windows machines. The installation instructions are written with Linux and macOS in mind. If attempting on Windows, check out the section on Windows instructions in the [VS Code Instructions for Kapteyn Cluster](https://docs.google.com/document/d/1Hm1J8x9CQ10dnyDJo1iohZHU6go_hxiUR7gTD2csv-M/edit?usp=sharing).
