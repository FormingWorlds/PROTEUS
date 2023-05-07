## Troubleshooting advice

This document includes troubleshooting advice for common errors. Each entry is labelled with the platform(s) typically affected.

### MacOS: PETSc tests error
On I have experienced an error when running the PETSc tests.
It looked like something along the lines of:
```
Fatal error in PMPI_Init_thread: Other MPI error, error stack:
MPIR_Init_thread(467)..............:
MPID_Init(177).....................: channel initialization failed
MPIDI_CH3_Init(70).................:
MPID_nem_init(319).................:
MPID_nem_tcp_init(171).............:`
MPID_nem_tcp_get_business_card(418):
MPID_nem_tcp_init(377).............: gethostbyname failed, localhost (errno 3)
```
This is actually a network configuration issue. To fix it, you need to add the following to `/etc/hosts`:    
`127.0.0.1   computername.local`    
`127.0.0.1   computername`    
And then also enable Remote Login in your Sharing settings and add your user to the 'allowed access' list.

### All: PETSc complains about being in the wrong directory
Firstly, check that you are in the correct directory when running `make` or `./configure`. If you are, then this could be caused by the environment variable `PETSC_DIR` remaining set after a previous PETSc installation. Run `unset PETSC_DIR` and try again.

### All: SPIDER can't find PETSc
Have you sourced `PROTEUS.env`? If yes, check that the variables `PETSC_ARCH` and `PETSC_DIR` in that file are correct for your system.

### MacOS: The FastChem code distributed with VULCAN won't compile 
With the new Apple Silicon hardware, the option `-march=native` sometimes causes issues. In order to avoid this, you need to make sure to use the GNU version of `g++`, not the Apple one. The Apple one located at `/usr/bin/gcc` is actually a wrapped around `clang`. I have found that using the Homebrew version located at `/opt/homebrew/bin/` works well. To fix this error, find out which gcc version homebrew installed (`ls /opt/homebrew/bin/gcc-*`), and edit the file `make.globaloptions` in the FastChem directory to use, e.g., `g++-12` or `g++-13` instead of `g++`.

### Linux: ksh not found when running SOCRATES
Most Linux distributions do not come with `ksh` installed, while MacOS seems to. If you get an error relating to `ksh` not being found, check that you did all of the installation steps. One step under 'Setup SOCRATES' involves replacing `ksh` with `bash` in all of the SOCRATES executables.

### MacOS: Python / netCDF error `Library not loaded: '@rpath/libcrypto.3.dylib'`
Create a symlink in the local Python installation (here shown for `bash` terminal). See https://pavcreations.com/dyld-library-not-loaded-libssl-1-1-dylib-fix-on-macos/

`brew install openssl`.

Follow the instructions at the end of the `openssl` installation:

`echo 'export PATH="/usr/local/opt/openssl@3/bin:$PATH"' >> /Users/USERNAME/.bash_profile`

`echo 'export LDFLAGS="-L/usr/local/opt/openssl@3/lib"' >>/Users/USERNAME/.bash_profile`  

`echo 'export CPPFLAGS="-I/usr/local/opt/openssl@3/include"' >>/Users/USERNAME/.bash_profile` 

`ln -s /usr/local/opt/openssl/lib/libcrypto.3.dylib /Users/USERNAME/opt/anaconda3/envs/proteus/lib/python3.10/site-packages/netCDF4/../../../`  

`ln -s /usr/local/opt/openssl/lib/libssl.3.dylib /Users/USERNAME/opt/anaconda3/envs/proteus/lib/python3.10/site-packages/netCDF4/../../../` 

### MacOS: Python error `ModuleNotFoundError: No module named 'yaml'` despite `yaml` being installed via `conda`
`python -m pip install pyyaml`

### MacOS: If SOCRATES make routine complains about absent `ifort` compilers
Install Intel compilers from https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html.
* First Intel® oneAPI Base Toolkit
* Then Intel® oneAPI HPC Toolkit
* Follow the instructions that are provided after the installation to set the locations of `ifort` in your environment

### MacOS: One of the following errors during PETSC configuration or compilation steps
`"This header is only meant to be used on x86 and x64 architecture"`
`#error "This header is only meant to be used on x86 and x64 architecture"`
* Delete all traces of Anaconda package manager from your system and switch to a different Python environment, for example brew/pip
* Follow the steps at https://docs.anaconda.com/free/anaconda/install/uninstall/
* Delete all Anaconda-related entries from your .bash_profile (Intel) or .zshrc (ARM)
* Install Python via `brew`: 
  * `brew install python`
  * Update to the latest stable version: `brew upgrade python`
  * Refresh your shell / `source ~/.zsrhrc` (ARM) / `source ~/.bash_profile` (Intel)
  * Install all necessary packages: `pip install matplotlib pandas netcdf4 matplotlib numpy pandas scipy sympy natsort`
  * Make the new Python version the system default:
    * Intel: `export PATH="/opt/homebrew/opt/python/libexec/bin:$PATH"`
    * ARM: `export PATH="/usr/local/opt/python/libexec/bin:$PATH"`

### MacOS: `ModuleNotFoundError: No module named '_tkinter'`
* Install `tkinter` package.
* Using `brew`: `brew install python-tk`
