# Running and output

This page covers how to launch a single PROTEUS simulation from the command line, how to run it on a remote machine, where the results are written, and how to archive them. For running many simulations at once, see [Parameter grids](usage_grids.md).

## Running PROTEUS from the terminal

PROTEUS has a command-line interface (CLI) accessed by running `proteus` on the command line. Run `proteus --help` to list the available commands. To run a simulation:

```console
proteus start -c [cfgfile]
```

where `[cfgfile]` is the path to the configuration file. A good first test is the `all_options.toml` config in the `input` folder:

```console
proteus start -c input/all_options.toml
```

This writes its results to `output/all_options/` inside your PROTEUS directory. See the [configuration guide](config.md) for how to edit configuration files and an explanation of their structure.

The most useful flags for `proteus start` are:

| Flag | Effect |
|---|---|
| `-c, --config [cfgfile]` | Path to the configuration file (required). |
| `--offline` | Skip the check for lookup tables and reference data to download. Use when the data is already present. |
| `--resume` | Resume the simulation from the state saved on disk rather than starting from `t = 0`. |
| `--deterministic` | Pin the JAX/XLA reduction order on top of the always-on BLAS thread pins. Use this when a coupled run fails on noise-floor floating-point divergence between launches (see [troubleshooting](troubleshooting.md#numerically-fragile-coupled-runs)). |

By default PROTEUS checks whether any lookup tables or data need to be downloaded before it runs; pass `--offline` to disable that check.

!!! tip "Long runs"
    A coupled simulation can run for hours. Detach it from the terminal so it survives disconnects, for example with `nohup`:

    ```console
    nohup proteus start --offline -c input/all_options.toml > output/all_options/launch.log 2>&1 &
    ```

    On a remote machine, `tmux` is usually more convenient (see below).

## Running PROTEUS on remote machines and servers

Using PROTEUS on a remote machine (for example Habrok or the Kapteyn cluster) is best done through `tmux`, which keeps programs running in the background for long periods. Full documentation is [here](https://tmuxcheatsheet.com/).

- Start a new tmux session:
    ```console
    tmux new -s <session_name>
    ```
- Inside the session, start your simulation:
    ```console
    proteus start -c input/all_options.toml
    ```
- Detach from the session with `Ctrl + b`, then `d`. Reattach later with:
    ```console
    tmux attach -t <session_name>
    ```
- List all sessions with `tmux ls`, and kill one with `tmux kill-session -t <session_name>`.
- The simulation stores its output in the PROTEUS `output/` folder. Check progress by reading the log files there.
- To see whether you are using CPUs on the cluster, run `htop -u $USER`; press `Ctrl + c` to exit.

For site-specific instructions, see the cluster guides: [Kapteyn](kapteyn_cluster_guide.md), [Snellius](snellius_cluster_guide.md), and [Habrok](habrok_cluster_guide.md).

## Output and results

A PROTEUS simulation creates several types of output file. For the `all_options` example the results are at `output/all_options/`. The tree below outlines the purpose of the main files and subfolders:

```text
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
 │ ├─vulcan.csv                 <---- atmospheric mixing ratios (offline mode)
 │ └─vulcan_{year}.csv          <---- per-snapshot mixing ratios (online mode)
 ├─plots/
 │ ├─plot_chem_atmosphere.png   <---- plot of atmospheric mixing ratios
 │ ├─plot_escape.png            <---- plot of volatile inventories over time
 │ ├─plot_global_log.png        <---- plot containing an overview of the simulation
 │ └─other files                <---- any other plots
```

The full column layout of `runtime_helpfile.csv` and the data-file formats are documented in the [output format reference](../Reference/output.md).

To make plots manually, use `proteus plot`. For example, to plot the atmosphere temperature profiles:

```console
proteus plot -c input/all_options.toml atmosphere
```

To make every available plot:

```console
proteus plot -c input/all_options.toml all
```

## Archiving output files

A simulation can generate a large number of files, which becomes a problem when running large [parameter grids](usage_grids.md). The `params.out.archive_mod` configuration option tells PROTEUS when to gather a run's output files into `.tar` archives.

Archiving makes the files inaccessible for analysis or plotting. Extract the archives from a run using:

```console
proteus extract-archives -c [cfgfile]
```

This is reversible. To pack the data files back into `.tar` archives:

```console
proteus create-archives -c [cfgfile]
```

---

**See also:** [Usage overview](usage.md) | [Configuration file](config.md) | [Output format reference](../Reference/output.md) | [Parameter grids](usage_grids.md) | [Postprocessing and chemistry](usage_postprocessing.md) | [Troubleshooting](troubleshooting.md)
