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

See the [config guide](https://fwl-proteus.readthedocs.io/en/latest/config/) for information
on how to edit the configurations files, and an explanation of their structure.

## Running grids of simulations

It is often useful to run grids of models, where each point in a grid represents a different
set of parameters. This can be done using the script `tools/GridPROTEUS.py`.

You can configure a grid of your choosing by editing the variables in this file.


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
