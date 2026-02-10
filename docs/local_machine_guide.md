# Local Machine Guide

These steps should be performed before installing PROTEUS on your computer.
They do not apply when running PROTEUS on a server or HPC cluster. For instructions on configuring PROTEUS on a remote machine, see the cluster guide pages.

Once you have followed these steps, go back to the main [installation](installation.html) guide page.

## Apple MacOS

1.  Open the terminal to install the developer tools

    ```console
    xcode-select --install
    ```

2.  Install the FORTRAN NetCDF and wget libraries via the most appropriate method for you.

    **[Homebrew](https://brew.sh/)** (recommended)
    ```console
    brew install netcdf netcdf-fortran wget
    ```

    **[MacPorts](https://www.macports.org/)**
    ```console
    sudo port install netcdf-fortran +gcc8 wget
    ```

3. Pay attention to replace `.bashrc` throughout the instructions below with `.zshrc` if you are on Mac OS >10.15 (Catalina) and using the default shell.

## Debian / Ubuntu Linux

Install gfortran and the NetCDF libraries via your package manager
```console
sudo apt install libnetcdff-dev gfortran
```

## Fedora / RedHat Linux

Install similiar libraries via your package manager
```
sudo dnf install gcc gcc-gfortran gcc-c++ netcdf netcdf-fortran netcdf-fortran-devel lapack lapack-devel lapack-static sundials-mpich openmpi openmpi-devel f2c f2c-libs
```

## Microsoft Windows

It is not recommended to install and use PROTEUS on Windows machines. The installation instructions are written with Linux and MacOS in mind. If attempting on Windows, check out the section on Windows instructions in [VS Code Instructions Kapteyn Cluster](https://docs.google.com/document/d/1Hm1J8x9CQ10dnyDJo1iohZHU6go_hxiUR7gTD2csv-M/edit?usp=sharing).
