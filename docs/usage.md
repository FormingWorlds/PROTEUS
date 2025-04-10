# Usage

## Running PROTEUS from the terminal

Proteus has a command-line interface that can be accessed by running `proteus` on the command line.
Try `proteus --help` to see the available commands!

You can directly run PROTEUS using the command:

```console
proteus start --config [cfgfile]
```

Where `[cfgfile]` is the path to the required configuration file.
Pass the flag `--resume` in to resume the simulation from the disk.

A good first test is to run the `minimal.toml` config, which is located in the `input` folder:

```console
proteus start --config <PROTEUS path>/input/minimal.toml
```
This will run a simulation and output the results to the `<PROTEUS path>/output/` folder.

See the [config guide](https://fwl-proteus.readthedocs.io/en/latest/config/) for information
on how to edit the configurations files, and an explanation of their structure.

## Running grids of simulations

It is often useful to run grids of models, where each point in a grid represents a different
set of parameters. This can be done using the script `tools/grid_proteus.py`.

You can configure a grid of your choosing by editing the variables in this file.

## Usage on clusters

If you are running PROTEUS on a cluster, you may need to use a job scheduler or `tmux` to run the simulations. Check out the [Kapteyn cluster guide](./kapteyn_cluster_guide.md) and [Snellius cluster guide](./snellius_cluster_guide.md) for more information.


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
proteus offchem --config [cfgfile]
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
proteus observe --config [cfgfile]
```

PROTEUS will perform this step automatically if enabled in the configuration file.
