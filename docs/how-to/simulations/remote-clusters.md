# Running Simulations on Remote Clusters

## Using tmux for Background Execution

Running PROTEUS on remote machines (e.g., Habrok, Kapteyn cluster) is best done through `tmux`, which allows you to leave programs running in the background for extended periods.

### Starting a Simulation in tmux

Create a new tmux session:

```console
tmux new -s <session_name>
```

Inside the session, start your simulation:

```console
proteus start -c input/all_options.toml
```

### Managing tmux Sessions

Detach from a session (keep it running):

- Press `Ctrl + b`, then `d`

Reattach to a session later:

```console
tmux attach -t <session_name>
```

List all active tmux sessions:

```console
tmux ls
```

Kill a tmux session:

```console
tmux kill-session -t <session_name>
```

For comprehensive tmux documentation, see [tmuxcheatsheet.com](https://tmuxcheatsheet.com/).

## Monitoring Progress

The simulation stores output data in the PROTEUS `output/` folder. Check progress by viewing log files:

```console
cat output/[simulation_name]/proteus_00.log
```

This displays information about the simulation's progress and any errors.

## Checking CPU Usage

To see if you are using CPUs on the cluster:

```console
htop -u $USER
```

Press `Ctrl + c` to exit `htop`.

## Running Grids with Slurm

For large grids on HPC clusters, use Slurm instead of tmux. See the [Running Grids of Simulations](grid-simulations.md) guide for Slurm-specific instructions, including:

- Setting `use_slurm = true` in your grid configuration
- Running `sbatch` to dispatch jobs
- Monitoring with `squeue`
