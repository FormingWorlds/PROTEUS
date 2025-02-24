# Install PROTEUS on the Snellius Cluster

1. Connect via SSH. See instructions from SURF [here](https://servicedesk.surf.nl/wiki/display/WIKI/SSH+public-key+authentication).

2. Set up your working environment. You have to load some modules and set up the environment variables prior to the installation. To facilitate this, we suggest that you copy the following lines into your `~/.bashrc` file.

    ```console
    module load 2024
    module load MPICH/4.2.2-GCC-13.3.0
    export LD_LIBRARY_PATH=""
    module load netCDF-Fortran/4.6.1-gompi-2024a
    export RAD_DIR="$HOME/SOCRATES/"
    export FWL_DATA="$HOME/FWL_DATA/"
    ```
    You must then logout.

3. When you login again, run `module list` and confirm that you have loaded the following modules.

    ```
    Currently Loaded Modules:
    1) 2024                           11) libpciaccess/0.18.1-GCCcore-13.3.0  21) Szip/2.1.1-GCCcore-13.3.0
    2) GCCcore/13.3.0                 12) hwloc/2.10.0-GCCcore-13.3.0         22) HDF5/1.14.5-gompi-2024a
    3) zlib/1.3.1-GCCcore-13.3.0      13) OpenSSL/3                           23) cURL/8.7.1-GCCcore-13.3.0
    4) binutils/2.42-GCCcore-13.3.0   14) libevent/2.1.12-GCCcore-13.3.0      24) gzip/1.13-GCCcore-13.3.0
    5) GCC/13.3.0                     15) libfabric/1.21.0-GCCcore-13.3.0     25) lz4/1.9.4-GCCcore-13.3.0
    6) numactl/2.0.18-GCCcore-13.3.0  16) PMIx/5.0.2-GCCcore-13.3.0           26) zstd/1.5.6-GCCcore-13.3.0
    7) UCX/1.16.0-GCCcore-13.3.0      17) PRRTE/3.0.5-GCCcore-13.3.0          27) bzip2/1.0.8-GCCcore-13.3.0
    8) MPICH/4.2.2-GCC-13.3.0         18) UCC/1.3.0-GCCcore-13.3.0            28) netCDF/4.9.2-gompi-2024a
    9) XZ/5.4.5-GCCcore-13.3.0        19) OpenMPI/5.0.3-GCC-13.3.0            29) netCDF-Fortran/4.6.1-gompi-2024a
    10) libxml2/2.12.7-GCCcore-13.3.0  20) gompi/2024a
    ```
