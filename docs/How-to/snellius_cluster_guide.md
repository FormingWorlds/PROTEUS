# Snellius Cluster Guide

## Installation

1. Connect via SSH. See instructions from SURF [here](https://servicedesk.surf.nl/wiki/display/WIKI/SSH+public-key+authentication).

2. Set up your working environment. You have to load some modules prior to the installation. To facilitate this, we suggest that you copy the following lines into your `~/.bashrc` file. The installer sets `FWL_DATA` and `RAD_DIR` in the conda environment, so they do not go there.

    ```console
    module load 2024
    module load MPICH/4.2.2-GCC-13.3.0
    export LD_LIBRARY_PATH=""
    module load netCDF-Fortran/4.6.1-gompi-2024a
    ```
    **You must then logout**.

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

4. You can now follow the usual installation steps [here](installation.md).

## Running grids

The smallest job on the `rome` partition is 1/8 of a node (16 cores and 28 GiB), and a job is charged for that share whether it uses it or not. Pack 16 cases into one job with `cores_per_job = 16` in the grid file, or 128 for a whole node, and keep `max_mem` at or below 1.75 GiB per case to stay within the core share; see [Dispatching the grid with Slurm](usage_grids.md#with-slurm).

### How Snellius charges a grid

Snellius charges a job that fits on one node in eighths of a node: the share is the larger of the cores and the memory that the job gets, as a fraction of the node, rounded up to the next eighth ([Shared usage accounting](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/300974114/Shared+usage+accounting)). One charged core costs the SBU weight of the partition per hour, for the time that the job runs, not its walltime. The cores, memory and weights come from the [Snellius partitions](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions) page (read on 2026-11-21). The examples on the accounting page give other memory per eighth (for example 224 GiB on a fat node); this guide and the grid layout explorer follow the partition table.

| Partition | Cores per node | Memory per node (GiB) | Memory per core (GiB) | Smallest share | SBU per core-hour |
|---|---|---|---|---|---|
| `rome` | 128 | 224 | 1.75 | 16 cores and 28 GiB | 1.0 |
| `genoa` | 192 | 336 | 1.75 | 24 cores and 42 GiB | 1.0 |
| `fat_rome` | 128 | 960 | 7.5 | 16 cores and 120 GiB | 1.5 |
| `fat_genoa` | 192 | 1440 | 7.5 | 24 cores and 180 GiB | 1.5 |

The grid manager asks for `max_mem` per core (`--mem-per-cpu`). On `rome`, `sbatch --test-only` gave a job with `cores_per_job = 8` and `max_mem = 6.9` the whole eighth, 16 cores, each with `max_mem`, so 110 GiB and a charge of 64 cores, 8 SBU per case-hour. The explorer assumes that every partition rounds the cores of a job up to whole eighths in this way; that is checked only on `rome`. Above 1.75 GiB per core on `rome` or `genoa`, memory sets the charge, so the cases also pay for cores that they don't use. A whole-eighth `cores_per_job` (16 on `rome`, 24 on `genoa`) avoids the unasked cores; for cases that need more memory, a fat partition can cost less: on `fat_rome`, a job of 16 cases at 7.5 GiB is charged 16 cores, 1.5 SBU per case-hour. The [grid layout explorer](usage_grids.md#grid-layout-explorer) computes these charges for a grid.

Note that when logging into `snellius.surf.nl`, you will not always connect to the same computer. For example,
sometimes your hostname might be `int5` while other times it could be `int6`. This means that `tmux` sessions
might suddenly appear to vanish, but in reality they are running on a different node. You can access a particular
session (e.g. `int5`) by running `ssh int5` once logged into Snellius.
