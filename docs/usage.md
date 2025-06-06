# Usage

## Running PROTEUS from the terminal

PROTEUS has a command-line interface that can be accessed by running `proteus` on the command line.
Try `proteus --help` to see the available commands!

You can directly run PROTEUS using the command:

```console
proteus start -c [cfgfile]
```

Where `[cfgfile]` is the path to the required configuration file.
Pass the flag `--resume` in to resume the simulation from the disk.

A good first test is to run the `all_options.toml` config, which is located in the `input` folder:

```console
proteus start -c input/all_options.toml
```
This will run a simulation and write the results to the `output/` folder inside your PROTEUS
directory.

See the [config guide](https://fwl-proteus.readthedocs.io/en/latest/config/) for information
on how to edit the configurations files, and an explanation of their structure.

## Output and results

Simulations with PROTEUS create several types of output files. For the `all_options` example,
the results are located at `output/all_options/`. The tree below outlines the purposes of
some files and subfolders contained within the simulation's output folder.

```
all_options/
 ├─runtime_helpfile.csv         <---- table containing the main simulation results
 ├─proteus_00.log               <---- the log file from the simulation
 ├─init_coupler.toml            <---- a completed copy of the configuration file
 ├─status                       <---- status of the simulation
 ├─data/
 │ ├─files ending in _atm.nc    <---- atmosphere data
 │ ├─files ending in .json      <---- interior data
 │ └─data.tar                   <---- atmosphere & interior data archive
 ├─observe/
 │ └─files ending in .csv       <---- synthetic/simulated observations of the planet
 ├─offchem/
 │ └─vulcan.csv                 <---- atmospheric mixing ratios calculated with VULCAN
 ├─plots
 │ ├─plot_chem_atmosphere.png   <---- plot of atmospheric mixing ratios
 │ ├─plot_escape.png            <---- plot of volatile inventories over time
 │ ├─plot_global_log.png        <---- plot containing an overview of the simulation
 │ └─other files                <---- any other plots
```

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

## Postprocessing of PROTEUS simulation grids 

Results from a PROTEUS grid can be post-processed using the `proteus grid_analyze` command. 

This will generate a CSV file with extracted data (`your_grid_name_extracted_data.csv`) from the grid results and ECDF plots 
(see [seaborn.ecdfplot doc](https://seaborn.pydata.org/generated/seaborn.ecdfplot.html)). 
Here is the structure of the generated `post_processing_grid` folder inside the grid directory :

```console
your_grid_name/
 ├─case_00000                               <---- case of your grid (for the structure refer to the tree from the [## Output and results] section)
 ├─case_00001         
 ├─...                  
 ├─cfgs                                     <---- folder with all the `input.toml` files for all cases
 ├─logs                                     <---- folder with all the `proteus_case_number.log` files for all cases
 ├─manager.log                              <---- the log file of the grid
 ├─slurm_dispatch.sh                        <---- if use_slurm=True in `grid_proteus.py`, this is the slurm file to submit with `sbatch` command
 ├─post_processing_grid                     <---- this folder contains all the output from this script
 │ └─extracted_data                         <---- folder with the generated CSV file 
 │   └─your_grid_name_extracted_data.csv    <---- CSV file containing the tested input parameters and extracted output from the grid
 │ └─plots_grid                             <---- folder with the generated plots
 │  ├─ecdf_grid_plot.png                    <---- Grid plot to visualize all tested input parameters vs extracted outputs using ECDF distribution
 │  ├─grid_statuses_summary.png             <---- Summary plot of statuses for all cases of the grid
 │  └─single_plots_ecdf                     <---- folder with all the single ECDF plots corresponding to all the panels from the grid plot
 │     ├─ecdf_[extracted_output]_per_[input_param].png     <---- Single plot using ECDF distribution to visualize one tested input parameter vs one extracted output for all cases 
 │     └─...
```

 
 To post-processed the grid and generate ECDF plots for further analysis, use the proteus command line interface:

```console
proteus grid-analyze /path/to/grid/ [grid_name] 
```

The user can also specify to update the CSV file with new output to extract for instance by adding the `--update-csv` flag, using :

```console
proteus grid-analyze /path/to/grid/ [grid_name] --update-csv
```

To get more information about this command, run :

```console
proteus grid-analyze --help
```

*Note to the user : update `output_to_extract` for your grid*

1. The user can choose the output to extract for each simulations at the last time-step (from the `runtime_helpfile.csv` file of each cases) like 'esc_rate_total','Phi_global','P_surf','T_surf','M_planet'... 
To do so, the user should go to `PROTEUS/src/proteus/grid/post_processing_grid.py` and modify the variable `output_to_extract` within the `run_grid_analyze` function. 

2. In the Step 2 of the same function, the user should also modify accordingly the `param_settings_single` and `output_settings_single` object for generating single plots (same for the grid plot). For this, the user should add the input parameters and output extracted from your grid if this is not already present in the scripe and comment the one useless for your grid. 

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

## Version checking

The `proteus doctor` command helps you to diagnose issues with your proteus installation.
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
