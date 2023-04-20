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
With the new Apple Silicon hardware, the option `-march=native` sometimes causes issues. In order to avoid this, you need to make sure to use the GNU version of `g++`, not the Apple one. The Apple one located at `/usr/bin/gcc` is actually a wrapped around `clang`. I have found that using the Homebrew version located at `/opt/homebrew/bin/` works well. To fix this error, edit the file `make.globaloptions` in the FastChem directory to use `g++-12` instead of `g++`.

### Linux: ksh not found when running SOCRATES
Most Linux distributions do not come with `ksh` installed, while MacOS seems to. If you get an error relating to `ksh` not being found, check that you did all of the installation steps. One step under 'Setup SOCRATES' involves replacing `ksh` with `bash` in all of the SOCRATES executables.

