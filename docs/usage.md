# Usage

## Running PROTEUS from the terminal

Proteus has a command-line interface that can be accessed by running `proteus` on the command line.
Try `proteus --help` to see the available commands!

You can directly run PROTEUS using the command:

```console
proteus start -c [cfgfile]
```

Where `[cfgfile]` is the path to the required configuration file.
Pass the flag `--resume` in to resume the simulation from the disk.

A good first test is to run the `minimal.toml` config, which is located in the `input` folder:

```console
proteus start -c <PROTEUS path>/input/minimal.toml
```
This will run a simulation and output the results to the `<PROTEUS path>/output/` folder.

See the [config guide](https://fwl-proteus.readthedocs.io/en/latest/config/) for information
on how to edit the configurations files, and an explanation of their structure.


## Running PROTEUS on remote machines / servers

Using PROTEUS on a remote machine (e.g. Habrok, the Kapteyn cluster, etc.) is best done through tmux.
Tmux allows you to leave programs running in the 'background' for long periods of time.
You can find detailed documentation [here](https://tmuxcheatsheet.com/).

- For example, you can start a new tmux session with the command:
    ```console
    tmux new -s <session_name>
    ```
- Inside the tmux session, start your simulation:
    ```console
    proteus start -c input/all_options.toml
    ```
- To detach from the session, press `Ctrl + b`, then `d`. You can reattach to the session later with:
    ```console
    tmux attach -t <session_name>
    ```
- To list all tmux sessions, use:
    ```console
    tmux ls
    ```
- To kill a tmux session, use:
    ```console
    tmux kill-session -t <session_name>
    ```
- The above started simulation will store the output data in the PROTEUS `output/` folder. You can check the progress of the simulation by looking at the log files in this folder. The log files are named according to the simulation name and contain information about the simulation's progress and any errors that may have occurred.
- If you want to check if you are using CPUs on the cluster, use the command:
    ```console
    htop
    ```
- Press `Ctrl + c` to exit the `htop` command.

## Running grids of simulations

It is often useful to run grids of models, where each point in a grid represents a different
set of parameters. This can be done using the script `tools/grid_proteus.py`.

You can configure a grid of your choosing by editing the variables at the end of this file.
With the grid configured to your liking, you can then dispatch the grid in two ways.

### Without Slurm

Firstly, you can set `use_slurm=False`. In this case, `grid_proteus.py` will manage the
individual subprocesses which compose the grid. The variable `max_jobs` specifies the maximum number of CPU cores
which should be utilised by the grid at any one time. This is limited by the number of CPU
cores available on your machine. This method works without SLURM, and can be applied on servers or on multicore personal computers.

You will need to make sure that the `grid_proteus.py` process stays open in order to mange the subprocesses.

### With Slurm

Alternatively, you can access high performance compute nodes through the SLURM workload
manager (e.g. on Habrok and Snellius). This is a two-step process. To do this, set `use_slurm=True` in `grid_proteus.py`,
and set `max_mem` and `max_days` to specify how much memory should be allocated to each job (each simulation).
These are nominally 3 GB and 2 days respectively. Ensure that these values are within the limits of the server you are working on.

With these options enabled, running `grid_proteus.py` will produce a script called `slurm_dispatch.sh` in the
specified output folder, as well as write the required configuration files to a subfolder called `cfgs/`.

To dispatch your grid via Slurm, run `sbatch <path>` where `<path>` is the path to the dispatch script created
by `grid_proteus.py`. You will be prompted to do this in the terminal.

Monitor your running jobs with `squeue -u $USER`. To cancel **all** of your running jobs, use `scancel -u $USER`.


## Version checking

The `proteus doctor` commnd helps you to diagnose issues with your proteus installation.
It tells you about outdated or missing packages, and whether all environment variables have been set.

```console
$ proteus doctor
Dependencies
fwl-proteus: ok
fwl-mors: Update available 24.10.27 -> 24.11.18
fwl-calliope: ok
fwl-zephyrus: ok
aragog: Update available 0.1.0a0 -> 0.1.5a0
AGNI: No package metadata was found for AGNI is not installed.

Environment variables
FWL_DATA: Variable not set.
RAD_DIR: Variable not set.
```

## Postprocessing of results with 'offline' chemistry

PROTEUS includes an "offline" chemistry functionality, which uses results of a simulation
as an input to the VULCAN chemical kinetics model, capturing the additional physics.

You can access the offline chemistry via the command line interface:

```console
proteus offchem -c [cfgfile]
```
This will run VULCAN as a subprocess. This command should not be used in batch processing.

PROTEUS will perform this step automatically when the configuration variable
`atmos_chem.when` is set to `"offline"`.


## Postprocessing of results with synthetic observations

Similarly to the offline chemistry, PROTEUS results can be postprocessed to generate
synthetic observations. Transmission and emission spectra are generated based on the
modelled temperature-pressure profile, as well as atmospheric composition. The composition
can be set by the output of the offline chemistry calculation (see config file).

You can access the synthetic observation functionality via the command line interface:

```console
proteus observe -c [cfgfile]
```

PROTEUS will perform this step automatically if enabled in the configuration file.

## Archiving output files

Running PROTEUS can generate a large number of files, which is problematic when also running
large grids of simulations. To counter this, the `params.out.archive_mod` configuration
option can be used to tell PROTEUS when to archive its output files. This will gather the
output files of each run into `.tar` files.

Archiving the output files makes them inaccessible for analysis or plotting. To extract the
archives from a run, use the proteus command line interface:
```console
proteus extract-archives -c [cfgfile]
```

This is reversible. To pack the data files into `.tar` archives again:
```console
proteus create-archives -c [cfgfile]
```
