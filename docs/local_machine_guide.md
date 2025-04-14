# Local Machine Guide

These steps should be performed before installing PROTEUS on your computer.
They do not apply when running PROTEUS on a server or HPC cluster. For instructions on configuring PROTEUS on a remote machine, see the cluster guide pages.

Once you have followed these steps, go back to the main [installation](./installation.md) guide page.

## Apple MacOS

1.  Open the terminal to install the developer tools

    ```console
    xcode-select --install
    ```

2.  Install `FORTRAN NetCDF` library via the most appropriate method for you.

    **[Homebrew](https://brew.sh/)** (recommended)
    ```console
    brew install netcdf
    brew install netcdf-fortran
    ```

    **[MacPorts](https://www.macports.org/)**
    ```console
    sudo port install netcdf-fortran +gcc8
    ```

3. Pay attention to replace `.bashrc` throughout the instructions below with `.zshrc` if you are on Mac OS >10.15 (Catalina) and using the default shell.

## Linux (Debian/Ubuntu)

Install gfortran and the NetCDF libraries via your package manager
```console
sudo apt install libnetcdff-dev gfortran
```

## Microsoft Windows

Generally it is not recommended to install and use PROTEUS on Windows machines. The remainder of the installation instructions are written with Linux and Mac OS in mind. However, for attempting that, check out the section on Windows instructions in [VS Code Instructions Kapteyn Cluster](https://docs.google.com/document/d/1Hm1J8x9CQ10dnyDJo1iohZHU6go_hxiUR7gTD2csv-M/edit?usp=sharing).

