# Running a Single Simulation

## Basic Simulation Execution

Start a PROTEUS simulation using the command-line interface:

```console
proteus start -c [cfgfile]
```

Where `[cfgfile]` is the path to your configuration file. For example:

```console
proteus start -c input/all_options.toml
```

This will run a simulation and write results to the `output/` folder.

See the configuration guide for information on how to edit configuration files.

## Resuming Simulations

To resume an interrupted simulation from disk:

```console
proteus start -c [cfgfile] --resume
```

## Offline Mode

By default, PROTEUS will automatically check if any lookup tables or data need to be downloaded. To disable this functionality and run in offline mode:

```console
proteus start -c [cfgfile] --offline
```

## Understanding Output and Results

Simulations create multiple output files in `output/[simulation_name]/`. The structure is:

```text
output/
 ├─runtime_helpfile.csv         <---- main simulation results table
 ├─proteus_00.log               <---- simulation log file
 ├─init_coupler.toml            <---- completed copy of configuration file
 ├─status                       <---- simulation status
 ├─data/
 │ ├─files ending in _atm.nc    <---- atmosphere data (NetCDF format)
 │ ├─files ending in .json      <---- interior data
 │ └─data.tar                   <---- archived atmosphere & interior data
 ├─observe/
 │ └─files ending in .csv       <---- synthetic observations of the planet
 ├─offchem/
 │ └─vulcan.csv                 <---- atmospheric mixing ratios from VULCAN
 └─plots
   ├─plot_chem_atmosphere.png   <---- atmospheric mixing ratio plot
   ├─plot_escape.png            <---- volatile inventory evolution plot
   ├─plot_global_log.png        <---- simulation overview plot
   └─other files                <---- additional analysis plots
```

## Checking Installation Status

Use the `proteus doctor` command to diagnose potential issues:

```console
proteus doctor
```

This will report:

- Status of all required packages
- Availability of updates
- Environment variable configuration (FWL_DATA, RAD_DIR)

Example output:

```console
$ proteus doctor
Packages
aragog: ok
fwl-calliope: ok
fwl-janus: ok
fwl-proteus: ok
fwl-mors: ok
fwl-zephyrus: ok
AGNI: Update available 1.7.1 -> Ledoux, oceans, water, clouds, and blackbody stars

Environment variables
FWL_DATA: ok
RAD_DIR: ok
```
