# Troubleshooting

This section includes troubleshooting advice for common errors, organized by
platform. If you encounter errors that you cannot solve via the standard
step-by-step guide or the advice below,
[contact the developers](contact.html).

---

## General (all platforms)

### Cannot clone module, or Permission denied (publickey)

Have you added your SSH key to GitHub? See these pages for guidance:

* [Checking for existing SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys)
* [Generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* [Adding a new SSH key to your Github account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
* [Testing your SSH connection](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)

### Out-of-date modules detected

You will see an error that looks like this:

```console
[ INFO  ] Validating module versions
[ ERROR ] (MODULENAME) module is out of date: (INSTALLED VER) < (REQUIRED VER)
Uncaught exception
    ...
OSError: Out-of-date modules detected. Refer to the Troubleshooting guide on the wiki
```

This means one of the required PROTEUS modules is outdated on your machine. You need to update the named module before running PROTEUS again.

You can check which modules are out of date by running:

```console
proteus doctor
```

If you have **not** manually installed this module, go to the **PROTEUS** folder and run:

```console
python -m pip install -e .
```

If you have a **developer installation** of the module, go to the **module's own folder** and run:

```console
git checkout main
git pull
python -m pip install -U -e .
```

### Data download errors or slow Zenodo downloads

PROTEUS automatically downloads input data files from Zenodo. The code works without an API token using public access, but you can optionally configure a token for enhanced rate limits.

??? tip "Optional: Configure a Zenodo API token"
    If you encounter rate limiting errors or slow downloads, you can set up a Zenodo API token:

    1. Get a personal access token from [Zenodo](https://zenodo.org/account/settings/applications/tokens/new/)
    2. Configure it:
        ```console
        pystow set zenodo api_token <your-token>
        ```
    3. Verify it's configured:
        ```console
        pystow get zenodo api_token
        ```

    The token is optional. PROTEUS uses public access by default and falls back gracefully if the token is unavailable.

### PETSc complains about being in the wrong directory

First, check that you are in the correct directory when running `make` or `./configure`. If you are, this could be caused by the environment variable `PETSC_DIR` remaining set after a previous PETSc installation. Run `unset PETSC_DIR` and try again.

### `libudev.so.1` not found

This happens when compiling SPIDER within a Python environment that is incompatible with PETSc. Try compiling SPIDER (and/or PETSc) in your system's base Python environment, and ensure that you are not using Python >= 3.13, as SPIDER is only supported for Python versions <= 3.12 currently.

### Julia compatibility error / OpenSSL_jll

There are incompatibilities between Python and some versions of Julia. Julia version 1.12+ is not yet supported because it requires a version of the OpenSSL library that is incompatible with Python.

You must use **Python 3.12** and **Julia 1.11** to avoid these problems:

```console
juliaup add 1.11
juliaup default 1.11
```

---

## macOS

### PETSc compilation fails on Apple Silicon

!!! warning "Applies to: macOS on Apple Silicon (M1, M2, M3, Ultra)"

If `get_petsc.sh` fails during configuration or compilation on Apple Silicon Macs, this is because the default PETSc configuration can conflict with the Apple Silicon toolchain.

**Prerequisites**

Make sure you have installed GCC and OpenMPI via Homebrew:

```console
brew install gcc open-mpi
```

**Automated fix**

Run the provided fix script from the PROTEUS root directory:

```console
./tools/fix_petsc_compile.sh
```

This script sets the correct environment variables and reconfigures PETSc with Apple Silicon-compatible flags.

??? info "Manual fix (if the script does not work)"
    1. Set environment variables in your shell config (`~/.zshrc` or `~/.bash_profile`):

        ```console
        export SDKROOT=$(xcrun --show-sdk-path)
        export CC=mpicc
        export CXX=mpicxx
        export FC=mpifort
        export F77=mpifort
        ```

    2. Navigate to `petsc/` and run the configure command manually:

        ```console
        cd petsc
        ./configure \
           PETSC_ARCH=arch-darwin-c-opt \
           CC=mpicc \
           CXX=mpicxx \
           FC=mpifort \
           F77=mpifort \
           LDFLAGS="-L/opt/homebrew/lib -Wl,-w" \
           --with-debugging=0 \
           --with-cxx-dialect=14 \
           --download-sundials2=1
        ```

    3. Build and test:

        ```console
        make PETSC_DIR=$(pwd) PETSC_ARCH=arch-darwin-c-opt all
        make PETSC_DIR=$(pwd) PETSC_ARCH=arch-darwin-c-opt check
        ```

    This has been tested on M1 Ultra (macOS Sequoia 15.6.1) and M2 (macOS 26.2).

### PETSc tests error (network configuration)

Error when running the PETSc tests:

```console
Fatal error in PMPI_Init_thread: Other MPI error, error stack:
MPIR_Init_thread(467)...............:
MPID_Init(177).....................: channel initialization failed
MPIDI_CH3_Init(70).................:
MPID_nem_init(319).................:
MPID_nem_tcp_init(171).............:
MPID_nem_tcp_get_business_card(418):
MPID_nem_tcp_init(377).............: gethostbyname failed, localhost (errno 3)
```

This is a network configuration issue. To fix it, add the following to `/etc/hosts`, where `computername` is your hostname:

```console
127.0.0.1   computername.local
127.0.0.1   computername
```

Then enable Remote Login in your Sharing settings and add your user to the *allowed access* list.

### Errors during PETSc configuration or compilation (x86 header)

One of the following errors during PETSc configuration or compilation:

```
This header is only meant to be used on x86 and x64 architecture
#error "This header is only meant to be used on x86 and x64 architecture
```

Install Python via Homebrew and use pip to manage packages, rather than Conda.

### Errors during the SOCRATES compilation

One of the following errors happen during the SOCRATES compilation:

1. `clang (LLVM option parsing): Unknown command line argument '-x86-pad-for-align=false'.  Try: 'clang (LLVM option parsing) --help'`
2. `clang (LLVM option parsing): Did you mean '--x86-slh-loads=false'?`

There is an issue with your compiler, either the standard Apple `clang` or `gcc` installed by `brew`. Follow the steps provided at the [StackOverflow thread](https://stackoverflow.com/questions/72428802/c-lang-llvm-option-parsing-unknown-command-line-argument-when-running-gfort):

```console
sudo rm -rf /Library/Developer/CommandLineTools
sudo xcode-select --install
```

### FastChem compilation with Apple Silicon

With Apple Silicon hardware (M1/M2/M3), the option `-march=native` sometimes causes issues.

Make sure to use the GNU version of `g++`, not the Apple one. The Apple version located at `/usr/bin/gcc` is actually a wrapper around `clang`.

The Homebrew version located at `/opt/homebrew/bin/` works well. Find out which `gcc` version Homebrew installed (`ls /opt/homebrew/bin/gcc-*`), and edit the file `make.globaloptions` in the FastChem directory to use, e.g. `g++-12` or `g++-13` instead of `g++`.

### Python / netCDF error

Error: `Library not loaded: '@rpath/libcrypto.3.dylib'`

Create a symlink in the local Python installation. See [this guide](https://pavcreations.com/dyld-library-not-loaded-libssl-1-1-dylib-fix-on-macos/) for details.

```console
brew install openssl
```

Follow the instructions at the end of the `openssl` installation (replace `USERNAME` with your own system username):

```console
echo 'export PATH="/usr/local/opt/openssl@3/bin:$PATH"' >> /Users/USERNAME/.bash_profile
echo 'export LDFLAGS="-L/usr/local/opt/openssl@3/lib"' >>/Users/USERNAME/.bash_profile
echo 'export CPPFLAGS="-I/usr/local/opt/openssl@3/include"' >>/Users/USERNAME/.bash_profile
ln -s /usr/local/opt/openssl/lib/libcrypto.3.dylib /Users/USERNAME/opt/anaconda3/envs/proteus/lib/python3.10/site-packages/netCDF4/../../../
ln -s /usr/local/opt/openssl/lib/libssl.3.dylib /Users/USERNAME/opt/anaconda3/envs/proteus/lib/python3.10/site-packages/netCDF4/../../../
```

### `ModuleNotFoundError: No module named 'yaml'`

Install `yaml` via pip rather than via conda:

```console
python -m pip install pyyaml
```

### Missing `ifort` compilers

If the SOCRATES installation steps complain about missing `ifort` compilers:

Install Intel compilers from the [Intel oneAPI Toolkit page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html). First install Intel oneAPI Base Toolkit, then Intel oneAPI HPC Toolkit. Follow the instructions provided after the installation to set the locations of `ifort` in your environment.

### `ModuleNotFoundError: No module named '_tkinter'`

Install the `tkinter` package using `brew`:

```console
brew install python-tk
```

### TkAgg backend / X-forwarding error

Error: `ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running`

If you are connecting to a computer over SSH, make sure to enable X-forwarding. This can be done in your SSH configuration file or as a command line parameter. Install [XQuartz](https://www.xquartz.org/) on your Mac.

### YAML library error

Error: `ld: unsupported tapi file type '!tapi-tbd' in YAML file '/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk/usr/lib/libSystem.tbd' for architecture arm64`

There is an issue with your `ld`, potentially caused by an existing installation of `anaconda`. Delete all traces of `anaconda` by following the steps at the [Anaconda uninstall guide](https://docs.anaconda.com/free/anaconda/install/uninstall/), then install Python via `brew`.

### gfortran cannot find system libraries

Error: `ld library not found for -lsystem`

This usually occurs when your operating system has been updated, but Xcode is not on the latest version. Try updating or reinstalling Xcode.

Alternatively, try adding the following flag to the gfortran compile and link steps:

```console
-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/
```

---

## Linux

### Cannot compile PETSc on compute cluster

There are various reasons that PETSc might have problems compiling. On a cluster, this could be caused by a mismatch between your MPICH version and SLURM. This is a known problem on Snellius.

Try loading the OpenMPI module:

```console
module load mpi/openmpi-x86_64
```

### Cannot compile PETSc, error moving libraries (Fedora/RHEL)

If you are using Fedora or RedHat Linux distributions, you might encounter an error:

```
Error moving
  /home/$USER/PROTEUS/petsc/arch-linux-c-opt/externalpackages/f2cblaslapack-3.8.0.q2
  libraries
```

To resolve, follow these steps:

1. Install libraries:
    ```console
    sudo dnf install lapack lapack-devel lapack-static sundials-mpich \
        openmpi openmpi-devel f2c f2c-libs
    ```
2. Enable module utility: `source /etc/profile.d/modules.sh`
3. Load OpenMPI module: `module load mpi/openmpi-x86_64`
4. Add the commands from steps 2 and 3 to your `~/.bashrc` file.
5. Run the `get_petsc.sh` script again.

### `netcdff.so` not found

This happens when SOCRATES cannot find the NetCDF-Fortran library. Make sure that it's installed. You can also try adding the library location to your `LD_LIBRARY_PATH` variable. Find the location of the NetCDF-Fortran library using:

```console
nf-config --flibs
```
