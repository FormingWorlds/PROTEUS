# Running Grids of Simulations

## Overview

Running grids of simulations (ensembles) allows you to explore parameter space with multiple forward models. Each point in a grid represents a different set of parameters. PROTEUS provides command-line tools to manage and monitor these runs.

## Creating a Grid Configuration

Create a TOML file specifying your grid's axes and execution parameters. An example is available at:

```console
input/ensembles/example.grid.toml
```

This file typically references a "base" configuration file and specifies which parameters should be varied and over what ranges.

## Running a Grid

Execute a grid of simulations using:

```console
proteus grid -c input/ensembles/example.grid.toml
```

Grids can be run with or without a workload manager (Slurm), depending on your computational environment.

## Without Slurm (Local/Multi-core)

For running grids on a personal computer or server without Slurm:

1. Set `use_slurm = false` in your grid configuration file
2. Set `max_jobs` to the maximum number of CPU cores to use simultaneously (limited by your machine)
3. Run: `proteus grid -c [grid_config]`
4. Keep the PROTEUS process open—it manages the subprocesses for you

This method works without any job scheduler and is suitable for multicore personal computers.

## With Slurm (HPC Clusters)

For distributed execution on clusters with Slurm (e.g., Habrok, Snellius):

1. Set `use_slurm = true` in your grid configuration file
2. Set `max_mem` (typically 3 GB) and `max_days` (typically 2) for resource allocation per job
3. Run: `proteus grid -c [grid_config]`
4. PROTEUS will generate a dispatch script: `slurm_dispatch.sh` and a `cfgs/` subfolder with individual configs
5. **Then run**: `sbatch [path_to_slurm_dispatch.sh]`
6. You will be prompted to run this command in the terminal

When using Slurm, the original PROTEUS process does not need to stay open—Slurm manages all subprocesses.

## Monitoring Grid Status

View the status of a running or completed grid:

```console
proteus grid-summarise -o output/grid_demo/
```

To list cases with a specific status (e.g., completed):

```console
proteus grid-summarise -o output/grid_demo/ -s completed
```

Available status filters include: `completed`, `running`, `failed`, etc.

## Managing Grid Jobs (Slurm)

Monitor running jobs:

```console
squeue -u $USER
```

Cancel all your running jobs:

```console
scancel -u $USER
```

Cancel a specific job:

```console
scancel [job_id]
```

## Packaging Grid Results

Package grid results into a single zip file for sharing or backup:

```console
proteus grid-pack -o output/grid_demo/
```

This creates `pack.zip` in the grid folder containing the most important files (not all raw data).
