# Usage

## Running PROTEUS

You can directly run PROTEUS using the Python command:

```console
proteus start --config [cfgfile]
```

Where `[cfgfile]` is the path to the required configuration file.
Pass the flag `--resume` in to resume the simulation from the disk.

You can also run PROTEUS inside a Screen session using:

```console
tools/RunPROTEUS.sh [cfgfile] [alias] [resume] [detach]
```

Which runs PROTEUS using the config file `[cfgfile]` inside a Screen
session with the name `[alias]`. The `[resume]` parameter (y/n) tells
the model whether to resume from a previous state. The `[detach]`
parameter (y/n) tells the session whether to immediately detach or not.
This allows multiple instances of the model to be dispatched easily and
safely.

## CLI

Proteus has a command-line interface that can be accessed by running `proteus` on the command line.
Try `proteus --help` to see the available commands!

### `proteus doctor`

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
