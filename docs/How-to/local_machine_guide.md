# Local Machine Guide

These steps should be performed before installing PROTEUS on your computer.
They do not apply when running PROTEUS on a server or HPC cluster. For
instructions on configuring PROTEUS on a remote machine, see the cluster
guide pages.

!!! info "What this guide covers"
    System packages, a Python package manager (conda), and Homebrew (macOS).
    Julia and PROTEUS itself are covered in the main
    [installation guide](installation.md), which you should follow after
    completing these steps.

## macOS

1. Open the terminal and install the Xcode developer tools:

    ```console
    xcode-select --install
    ```

2. Install [Homebrew](https://brew.sh/) if you do not have it:

    ```console
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

3. Install the required system libraries:

    ```console
    brew install netcdf netcdf-fortran wget gcc open-mpi cmake
    ```

    !!! note "Apple Silicon (M1/M2/M3/Ultra)"
        The `gcc` and `open-mpi` packages are required for compiling PETSc
        on Apple Silicon Macs. If you skip these, you may encounter
        compilation errors during the PETSc installation step. See the
        [Troubleshooting](troubleshooting.md#petsc-compilation-fails-on-apple-silicon)
        page for details.

4. Install [miniforge](https://github.com/conda-forge/miniforge) (recommended for Apple Silicon):

    ```console
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

    Follow the prompts, then restart your terminal or run `source ~/.zshrc`.

5. macOS Catalina (10.15) and later uses `zsh` as the default shell.
   Replace `.bashrc` with `.zshrc` throughout the installation instructions
   if you are using the default shell.

## Debian / Ubuntu Linux

1. Install the required system libraries:

    ```console
    sudo apt install libnetcdff-dev gfortran build-essential cmake unzip curl wget git
    ```

2. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux):

    ```console
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh
    rm ~/miniconda3/miniconda.sh
    ```

    Choose an install folder where you have plenty of disk space. Follow
    the prompts, then restart your terminal or run `source ~/.bashrc`.

## Fedora / RedHat Linux

1. Install the required system libraries:

    ```console
    sudo dnf install gcc gcc-gfortran gcc-c++ netcdf netcdf-fortran netcdf-fortran-devel \
        lapack lapack-devel lapack-static cmake unzip curl wget git
    ```

2. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux):

    ```console
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh
    rm ~/miniconda3/miniconda.sh
    ```

## Microsoft Windows

PROTEUS is not natively supported on Windows. To run PROTEUS on a Windows
machine, use [WSL2 (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install)
and follow the **Debian / Ubuntu Linux** instructions above.

```console
wsl --install -d Ubuntu
```

After installing WSL2 and launching an Ubuntu terminal, proceed with the
main [installation](installation.md) guide as on Linux.

For remote development on a cluster instead, see the
[VS Code Instructions for Kapteyn Cluster](https://docs.google.com/document/d/1Hm1J8x9CQ10dnyDJo1iohZHU6go_hxiUR7gTD2csv-M/edit?usp=sharing).
