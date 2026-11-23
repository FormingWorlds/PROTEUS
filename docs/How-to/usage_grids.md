# Parameter grids (ensembles)

!!! tip "Worked example"
    For a runnable, end-to-end walkthrough (a fast all-dummy grid over planet mass, orbital distance, and hydrogen budget, with plots), see the [parameter grid sweep tutorial](../Tutorials/parameter_grid.md).

It is often useful to run grids of forward models, where each point in a grid represents a different set of parameters. This is done with the `proteus grid` command:

```console
proteus grid -c input/example.grid.toml
```

Configure a grid by creating a TOML file that specifies the grid's axes and how it should be run. The example at `input/example.grid.toml` uses the dummy configuration file as a reference and modifies it for every combination of the parameters declared in the `.grid.toml` file. See the [configuration guide](config.md) for how to edit configuration files, and the [parameter grid tutorial](../Tutorials/parameter_grid.md) for a worked end-to-end example.

## Terms used on this page

PROTEUS and Slurm use some of the same words for different things. This page uses each word with the meaning in this table.

| Term | Meaning |
|---|---|
| Case | One PROTEUS run with one set of parameter values. Each case has its own output folder `case_NNNNNN/` and uses one CPU core. |
| Grid | All cases that one grid file (`*.grid.toml`) defines. |
| Job | A Slurm job that runs several cases on `cores_per_job` cores. In Slurm terms it's one array task of the job array, with its own ID `<array ID>_<task index>`. Without Slurm, there are no jobs: PROTEUS starts one subprocess per case. The grid setting `max_jobs` counts cases, not jobs. |
| Job array | All jobs that one `sbatch slurm_dispatch.sh` submits. Slurm also calls the whole array a job, with one array ID. |
| Array task | One job of the job array. This page calls it a job. |
| Job step | One `srun` call in a job. Each case runs as one job step, so `sacct` shows each case as a step. A step runs one Slurm task (`--ntasks=1`): in this sense, a Slurm task is a process, not an array task. |
| Core | One CPU core. A case uses one core, and a job uses `cores_per_job` cores. |
| Wave | One round of cases on the cores of a job. Most grids need 1 wave. If a grid has more cases than MaxArraySize times `cores_per_job`, each core runs several cases one after the other, and the grid needs several waves. |
| Walltime | The most time that Slurm lets a job run (`--time`). Slurm stops the job at its walltime. |
| Throttle | The most jobs of the job array that run at the same time (`%N` at the end of `--array`). |
| MaxArraySize | The cluster's limit on the number of jobs in one job array. |

??? info "Intermezzo: why these words are confusing"
    No single standard names these things. Each field uses the same few words in its own way:

    - Slurm has a job (one `sbatch` submission), a job step (one `srun` call in a job) and a task (one process of a step). A job array is one submission of many array tasks, and each array task is a job of its own.
    - "Batch" doesn't help. A batch system runs batch jobs that you submit with `sbatch`, but in other fields a batch is a group of inputs that a program processes together.
    - `make -j 8` and GNU `parallel -j 8` run 8 jobs at the same time, and a job there is one command. `max_jobs` follows this meaning, which clashes with the Slurm meaning.
    - Tools for parameter studies name one parameter point differently: an evaluation in Dakota, a trial in Optuna, and an ensemble member or a realization in climate models. PROTEUS calls it a case.

    These docs use the Slurm words only for Slurm things, and "case" and "grid" for PROTEUS things.

## Defining grid axes

A grid configuration file (`*.grid.toml`) contains two kinds of entries: top-level settings that control the whole ensemble, and one TOML table per parameter axis.

The top-level settings are:

| Setting | Meaning |
|---|---|
| `ref_config` | Base ("reference") config file, relative to the PROTEUS root. Every case starts from this file and overrides only the swept parameters. |
| `output` | Output folder name created inside `output/`. Use `"auto"` for a timestamped `grid_YYYYMMDD_HHMMSS_xxxx` name, or any fixed string. |
| `symlink` | Absolute path used to redirect the output to alternative storage (for example a scratch disk); set to `""` to disable. The grid empties this folder, so it may not lie inside the PROTEUS folder or inside `output/`, also where `output/` links to (for example `/scratch/$USER/proteus_output`); it may not lie inside the run output root that `PROTEUS_OUTPUT_PATH` names; it may not be your home folder, the PROTEUS folder, `output/` (or its link target) or the run output root, or a folder above one of them; it may not hold the grid file or the base config, also through a link inside it, and it may not itself be a link or a file. A sibling such as `/scratch/$USER/mygrid` is fine. On a file system that ignores letter case (macOS), a path in another case is the same folder. |
| `use_slurm` | Whether to dispatch through Slurm (see below). |
| `max_jobs` | Maximum number of cases that run at the same time (not Slurm jobs; see the warning after this table). |
| `max_days`, `max_mem` | Walltime (days) and memory (GB) per case. Only the Slurm dispatch uses them. |
| `cores_per_job` | CPU cores per Slurm job; each core runs one case at a time. Default 1. Only the Slurm dispatch uses it. |
| `jax_cache` | Share a JAX compilation cache across Slurm array tasks (see below). Default `false`; only affects Slurm dispatch. |
| `config_version` | Config format version, for example `"3.0"`. Accepted so that a grid file can state it; the grid manager does not read it. |

`use_slurm` and `jax_cache` take only a TOML `true` or `false`; a quoted `"false"`, a `0` or a `1` stops the grid at load with an error that names the setting and the grid file.

!!! warning "Confusing names"
    Some setting names don't say what the setting counts. Read them as follows:

    - `max_jobs` counts cases, not jobs. Without Slurm, it's the number of cases that run at the same time; PROTEUS uses at most the number of CPU cores of the machine and at most the number of cases. With Slurm, it sets the throttle: `max_jobs // cores_per_job` jobs run at the same time, at least 1. See [How the grid settings become Slurm settings](#how-the-grid-settings-become-slurm-settings).
    - `max_days` is the walltime per case, not per job. A job gets `max_days` times the number of waves. Without Slurm, PROTEUS reads `max_days` but doesn't use it.
    - `max_mem` is the memory per case, so per core. A job gets `cores_per_job * max_mem` GB. Without Slurm, PROTEUS reads `max_mem` but doesn't use it.
    - `cores_per_job` is the number of cores of one Slurm job, and so the number of cases that one job runs at the same time. Without Slurm, it has no effect.
    - `symlink` is the folder that holds the output, not the link. PROTEUS makes the output folder in `output/` a link to it.

Each parameter axis is a TOML table whose **name is the dotted path of the config field to vary**. For example, `["planet.mass_tot"]` sweeps `config.planet.mass_tot`, and `["outgas.fO2_shift_IW"]` sweeps the mantle redox offset. Any field documented in `input/all_options.toml` (or the [configuration reference](config.md)) can serve as an axis. Every top-level key must be accounted for: it is either one of the settings above or the path of one configuration field, given as a table with one of the methods below and the keys that method requires. Any other key, a path that names a whole section, an unquoted dotted header such as `[planet.mass_tot]` (TOML reads it as the section `planet`), an axis table with an unknown method or a missing required key, or a `direct` axis whose `values` is not a non-empty list of numbers and booleans or of strings (an empty list, a NaN, a single value without brackets, a nested array or table) stops the grid at load, before an earlier output folder is removed, with an error that names the key and the grid file. The grid manager then builds every case config and reads it back with the config loader before it removes an earlier output folder of the same name; a value the configuration refuses (an unknown module, a value out of its range, a quoted boolean) stops the grid there with the case index, the key and the value. This check runs in a hidden folder `output/.grid-check-*` that the grid removes when the check ends; a check that is killed (for example at a time limit) can leave that folder behind, and it is safe to remove.

Each axis declares a `method` that controls how its values are generated:

| `method` | Required keys | Values produced |
|---|---|---|
| `direct` | `values = [...]` | Exactly the listed values, all of one type (the TOML parser refuses an array that mixes types, so write `[1.0, 2.5]`, not `[1, 2.5]`). Duplicates are removed at their first occurrence. Lists of numbers or booleans are then sorted; string lists keep their written order. |
| `arange` | `start`, `stop`, `step` | Evenly stepped values from `start` to `stop`, **including** the `stop` endpoint. |
| `linspace` | `start`, `stop`, `count` | `count` values evenly spaced between `start` and `stop` (both endpoints included). |
| `logspace` | `start`, `stop`, `count` | `count` values logarithmically spaced between `start` and `stop`. Here `start` and `stop` are the actual endpoint values, not their base-10 exponents. |

The following example sweeps planet mass and hydrogen inventory:

```toml
ref_config = "input/dummy.toml"
output     = "auto"
symlink    = ""
use_slurm  = false
max_jobs   = 10
max_days   = 1
max_mem    = 12
jax_cache  = false

# Planet mass [M_earth]: four explicit values
["planet.mass_tot"]
    method = "direct"
    values = [0.7, 1.0, 2.0, 3.0]

# Hydrogen inventory [ppmw]: 5000, 10000, 15000, 20000
["planet.elements.H_budget"]
    method = "arange"
    start  = 5000
    stop   = 20000
    step   = 5000
```

PROTEUS runs the **Cartesian product** of all axes, so this example produces 16 cases (the four masses combined with the four hydrogen budgets). Each additional axis multiplies the case count by its own length, so adding a third axis with three values would yield 48 cases. The total number of cases must stay below 1,000,000, which is the limit imposed by the `case_NNNNNN` folder-naming scheme.

Some parameters are only meaningful when a matching *mode* is set in the reference config, because the mode selects how the value is interpreted. For example, sweeping `planet.elements.H_budget` requires an `H_mode` such as `"ppmw"` in the base config, and `planet.elements.S_budget` requires the corresponding `S_mode` (for example `"S/H"`). Set the mode once in `ref_config`, then vary the budget along a grid axis. The [planet and volatiles reference](../Reference/config/planet.md) lists which fields depend on which modes.

## Dispatching the grid

Grids can be dispatched with or without using a workload manager. In PROTEUS, we use the [Slurm](https://slurm.schedmd.com/overview.html) workload manager, which can allow running large ensembles of models on high-performance compute clusters. The subsections below detail cases with/without Slurm.

Before committing compute to a large grid, validate it with a dry run, which generates the grid and writes every per-case config file without launching any simulations:

```console
proteus grid -c input/example.grid.toml --dry-run
```

| | Without Slurm | With Slurm |
|---|---|---|
| Config setting | `use_slurm = false` | `use_slurm = true` |
| Process management | PROTEUS manages subprocesses | Slurm manages jobs |
| PROTEUS must stay open? | Yes | No |
| Where to run | Servers, multicore desktops | HPC clusters (Habrok, Snellius) |

### Without Slurm

Firstly, set `use_slurm = false`. In this case, the GridPROTEUS routine will manage the
individual subprocesses which compose the grid. `max_jobs` sets the number of cases that run at the same time, each on one CPU core. PROTEUS uses at most the number of CPU cores of your machine. `max_days`, `max_mem`, `cores_per_job` and `jax_cache` have no effect without Slurm. This method works without Slurm, and can be applied on servers or
on multicore personal computers.

In this case, you will need to make sure that PROTEUS stays open in order to manage its subprocesses.

### With Slurm

Alternatively, you can access high performance compute nodes through the Slurm workload manager (e.g. on Habrok and Snellius). This is a two-step process. To do this, set `use_slurm = true` in your grid's configuration file. Then set `max_mem` and `max_days` to the memory and the time that each case needs. The shipped example uses 12 GB and 1 day. Ensure that these values are within the limits of the server you are working on.

With these options enabled, running PROTEUS will produce a script called `slurm_dispatch.sh` in the specified output folder, as well as write the required configuration files to a subfolder called `cfgs/`.

To dispatch your grid via Slurm, you **must then run** the command `sbatch <path>` where `<path>` is the path to the dispatch script created by the `proteus grid` command. You will be prompted to do this in the terminal.

Each Slurm job gets `cores_per_job` cores and `cores_per_job * max_mem` GB of memory, and runs that many cases at a time; when a case ends, its core takes the next case of the job. Each case runs as its own job step of one core and `max_mem` GB, named after the case, so `sacct -j <jobid> --format=JobName,Elapsed,MaxRSS,State` shows the time, peak memory and state of each case. Its output goes to `logs/case_NNNNNN.log`, and its temporary files go to its own folder `$TMPDIR/<job ID>_case_NNNNNN`, which is removed when the case ends. A case fails when it exits with an error code or ends with an error status (20 to 28). A case that fails does not stop the other cases of its job; the job then ends as failed, and the status file of each case tells which one. A case that was killed before PROTEUS could write a final status (for example by the kernel when out of memory) gets status 25 and also counts as failed. Slurm sends the job SIGTERM 15 minutes before the walltime (a tenth of it for jobs shorter than 150 minutes). The job script then logs the time of the signal in UTC, by the clock of the compute node, and the seconds since the job started, in `logs/proteus-<jobid>_<task>.log`, starts no further case, and sends SIGTERM on to its running cases, which write status 25. A `scancel` of the job logs its time the same way; Slurm then sends SIGTERM to the cases itself, so the log also shows the error `Invalid job id specified` from the forward. Slurm may send the early signal up to a minute early; without a signal, the job still starts no further case from a minute before the signal is due (for jobs shorter than 10 minutes, from one signal lead before it). Cases that the job didn't start stay not started. A job is not requeued: a requeued job would start its cases again from the beginning.

Clusters that charge whole cores or whole node fractions waste their allocation on one-core jobs. On Snellius, for example, the smallest job on the `rome` partition is 16 cores and 28 GiB, charged whether used or not, so set `cores_per_job = 16` (or 128 for a whole node) and choose the partition with `sbatch -p rome`. Memory counts too: a job is charged for the larger of its core share and its memory share, so a `max_mem` above 1.75 GB per case on `rome` raises the charge. Set `max_mem` close to the measured peak memory of a case. A case still running at the early SIGTERM is stopped and gets status 25, so set `max_days` to cover the longest case.

Monitor your running jobs with `squeue -u $USER`. To cancel **all** of your running jobs, use `scancel -u $USER`.
The original PROTEUS process does not need to stay open when using Slurm to manage the subprocesses.

Set `jax_cache = true` to add a shared JAX compilation cache to the dispatch script. Each job then exports `JAX_COMPILATION_CACHE_DIR` pointing at a `jax_cache/` subdirectory of the grid output, so array tasks reuse each other's compiled kernels instead of recompiling the same interior solver per task. The cache is bounded at 80 GiB (`JAX_COMPILATION_CACHE_MAX_SIZE`), with a 1 s minimum compile time and a 4 KiB minimum entry size so only worthwhile kernels are stored. These exports are written into the Slurm dispatch script only; a local grid run (`use_slurm = false`) spawns subprocesses without them.

The cluster guides give site-specific Slurm settings: [Habrok](habrok_cluster_guide.md), [Snellius](snellius_cluster_guide.md), and [Kapteyn](kapteyn_cluster_guide.md).

#### How the grid settings become Slurm settings

The grid manager turns the grid settings into these Slurm settings in `slurm_dispatch.sh`:

| Setting | What it sets | What it becomes in Slurm |
|---|---|---|
| `cores_per_job` | Cores per Slurm job. Each core runs 1 case at a time. | Cores per job (`--cpus-per-task`), so cases at a time within one job. |
| `max_jobs` | Cases that run at the same time, over the whole grid. | Jobs at a time (the throttle, `%` in `--array`) = `max_jobs // cores_per_job`, at least 1 and at most the number of jobs. |
| `max_mem` | Memory per case, in GB. | Memory per job (`--mem`) = `cores_per_job * max_mem`, as one request for the job. The job step of each case gets `max_mem` (`--mem-per-cpu` of its one core). |
| `max_days` | Walltime per case, in days. | Walltime per job (`--time`) = `max_days * waves`, in whole minutes and at least 1 minute, because Slurm reads a walltime of 0 as no time limit. |
| MaxArraySize (from Slurm) | Most jobs in one job array. The grid manager reads it with `scontrol show config`. Without `scontrol`, it uses the Slurm default of 1001. | Jobs (`--array`) = `min(ceil(cases / cores_per_job), MaxArraySize)`. |

For example, 1000 cases with `cores_per_job = 16`, `max_jobs = 64`, `max_mem = 1.75` and `max_days = 1` give 63 jobs of 16 cores, 28 GB and 1 day each, in 1 wave. 4 jobs run at a time, so 64 cases run at a time. The script header then has `--cpus-per-task=16`, `--mem=28672M`, `--time=1440` and `--array=0-62%4`.

If `max_jobs` is smaller than `cores_per_job`, one job still runs, so more cases run at the same time than `max_jobs` says. If `max_jobs` isn't a multiple of `cores_per_job`, the grid manager rounds the number of jobs at a time down.

#### Grid layout explorer

Move the sliders to see how the grid settings lay out the cases over Slurm jobs. Each box is one job with one column per core and one row per wave; each cell shows the case index. Jobs outside the throttle wait in the queue (dashed), and an empty cell with a dashed outline is an idle core. The MaxArraySize slider stops at 40 to keep the drawing readable; on Snellius, the real value is about 1000.

<div class="grid-explorer"><p>The grid layout explorer needs JavaScript.</p></div>

Settings to try:

- Waves: set MaxArraySize to 5, with 50 cases on 4 cores. Each of the 5 jobs gets 3 rows (waves), and the walltime per job becomes 3 times `max_days`.
- Rounding: set `max_jobs` below `cores_per_job`, for example 2 with 4 cores, or to a value that isn't a multiple of it, for example 10. The warning line shows that 4 cases run at once instead of 2, or that 10 is rounded down to 2 jobs, so 8 cases at once.
- Idle cores: set 20 cases with 16 cores per job. Each of the 2 jobs gets 10 cases, so 6 of its 16 cores stay idle. That's 37.5 % of the allocated cores, which the idle-core warning catches.

The explorer assumes that every case runs equally long, so each row of a job is one wave. In a real run, `xargs` gives a free core the next case of its job, so when run times differ, the cases don't stay in these rows.

The explorer uses the formulas of the grid manager. To check it, compare these layouts with `array_layout` and `idle_share` in `proteus.grid.slurm`:

- 50 cases with `cores_per_job = 4` give 13 jobs in 1 wave, with 3.8 % of the cores idle.
- 20 cases with `cores_per_job = 16` give 2 jobs in 1 wave, with 37.5 % idle.
- 50 cases with `cores_per_job = 4` and a MaxArraySize of 5 give 5 jobs in 3 waves, with 16.7 % idle.

#### How `max_jobs` is enforced

No lock or counter counts the running cases over the whole cluster. Two different mechanisms enforce `max_jobs`, one with Slurm and one without.

With Slurm, Slurm enforces the limit through the throttle of the job array: `--array=0-62%4` means that at most 4 jobs of this array run at the same time. The grid manager sets the throttle to `max_jobs // cores_per_job` jobs, and each job runs `cores_per_job` cases at the same time with `xargs -P`. So the limit on cases is the product of these two numbers, and Slurm never sees it as one number.

The throttle counts only the jobs of this one array. It doesn't count other grids, other job arrays that you submit, or the jobs of other users.

`max_jobs` is an upper limit, not a target. Slurm schedules each job of the array on its own, so fewer cases can run than `max_jobs` allows:

- Some jobs can still wait in the queue.
- A job keeps all of its cores until its last case ends, and it still counts for the throttle, also while some of its cores have no case (see [Idle cores](#idle-cores)).

Without Slurm, the `proteus grid` process counts its own running cases in a loop. When fewer than `max_jobs` cases run, it starts the next case. It starts at most one case per check, and it checks every second while it starts the first cases and every 15 s after that. It runs at most as many cases as the machine has CPU cores (`os.cpu_count()`). It counts only its own cases on that machine, not other grids or other programs.

#### How cases are spread over the jobs

Job `j` runs cases `j`, `j + jobs`, `j + 2 * jobs` and so on. For example, with 10 cases and `cores_per_job = 4`, the grid manager makes 3 jobs:

- Job 0 runs cases 0, 3, 6 and 9.
- Job 1 runs cases 1, 4 and 7.
- Job 2 runs cases 2, 5 and 8.

This way each job gets cases from across the parameter space of the grid, instead of a block of neighbouring cases.

#### Grids with more cases than MaxArraySize allows

If a grid has more cases than MaxArraySize times `cores_per_job`, each core runs several cases one after the other. The number of cases per core is `waves`, and the walltime of each job is `max_days * waves`. The grid manager logs a warning when `waves` is more than 1.

For example, 20000 cases with `cores_per_job = 16` and a MaxArraySize of 1000 give 1000 jobs of 20 cases. Each core runs at most 2 of these cases, so the walltime is 2 times `max_days`. In the second wave, only 4 of the 16 cores of each job have a case, so this layout leaves 37.5 % of the allocated cores idle, and the grid manager warns about it. With `cores_per_job = 20`, each job runs its 20 cases in 1 wave, without idle cores.

#### Idle cores

A job keeps all of its cores until its last case ends. If the cases of a job don't fill all of its cores in the last wave, some cores have no case. A cluster that charges for allocated cores, such as Snellius, charges for these cores too. The job asks for `max_mem` for each of its cores, so the same share of its memory is idle too.

The grid manager computes the share of allocated core time that the layout leaves idle, if every case runs equally long. If that share is more than 10 %, it logs a warning with the share and with the nearest `cores_per_job` values that leave at most 10 % idle. It doesn't change the layout. Real cases don't run equally long, so the real idle time is different.

For example:

- 1000 cases with `cores_per_job = 16` give 63 jobs with 1008 cores. 8 of these cores are idle, so less than 1 %: no warning.
- 20 cases with `cores_per_job = 16` give 2 jobs of 10 cases with 32 cores. 12 cores are idle, so 37.5 %: the warning suggests `cores_per_job` = 20, 11 or 21.

The warning starts at 10 % because a layout with 1 wave and at least 10 jobs has fewer idle cores than one job has, so less than 10 % idle. Only small grids and grids with more than 1 wave can get the warning.

## Grid output layout

Running a grid creates one folder per case inside the output directory, alongside copies of the inputs needed to reproduce the ensemble:

```text
output/<grid name>/
├── ref_config.toml      # copy of the reference config
├── copy.grid.toml       # copy of the grid definition
├── manager.log          # grid manager log
├── cfgs/                # generated per-case config files (case_000000.toml, ...)
├── logs/                # per-job logs (Slurm dispatch)
├── case_000000/         # full PROTEUS run directory for the first case
├── case_000001/
└── ...
```

Each `case_NNNNNN/` folder is a complete PROTEUS run directory, numbered in the same order as the grid points listed in `manager.log`. Because the reference config and grid definition are copied into the output folder, an ensemble can be regenerated or extended from its output directory alone. To run a grid again from these copies, copy them out of the output folder first: a grid whose output folder or symlink folder holds its grid file or base config, also through a link, stops before anything is removed. The per-case output files are described on the [Running and output](usage_running.md#output-and-results) page.

## Viewing grid status

Use the CLI to view the status of a grid, such as to check cases which are finished. For example, the command below will summarise the top-level statuses of the demo grid.

```console
proteus grid-summarise -o output/grid_demo/
```

Add `-s` to find out which cases have a particular status. For example, the command below will list all completed cases.

```console
proteus grid-summarise -o output/grid_demo/ -s completed
```

A case without a readable status file, because it stopped before PROTEUS wrote one or has not started yet (a config in `cfgs/` but no case folder: still queued, or its job ended before it started), counts as unknown with code -1, and the summary logs how many cases have not started; `-s unknown` lists those cases together with any unrecognised codes, and `-s code=-1` lists them alone.

## Packaging grid results

Use the CLI to package the results of a grid into a zip file; for example to share or back it up. The command below creates `pack.zip` in the `grid_demo/` folder. This does not store all the data for each case, only the most important files: the 1D resolved interior and atmospheric data are dropped, while files containing global parameters and auto-generated plots are preserved.

```console
proteus grid-pack -o output/grid_demo/
```

## Post-processing a grid

Once a grid has run, `proteus grid-analyse` extracts each case's final-row results into summary CSV files and makes overview plots. It is configured by an optional `[post_processing]` table in the same `.grid.toml` used to run the grid (so a grid needs only one config file). Point it at the grid's output folder:

```console
proteus grid-analyse -o output/<grid folder>
```

With `-o` alone, the options come from `copy.grid.toml` in that folder (the copy of the grid file saved when the grid was launched). Add `-c <grid file>` to analyse the folder with the `[post_processing]` options of that file instead, for example after changing the plotted outputs. `-c` alone also works when the grid file's `output` names a fixed folder; with `output = "auto"` the folder name is generated at launch, so pass it with `-o`.

The swept axes are always read from `copy.grid.toml` when the folder has one, so editing the grid file afterwards does not change them. If that copy no longer passes the current grid-file checks (for example a grid launched with an older PROTEUS), the analysis stops with an error naming the rejected keys; update them in the copy, or remove it to use the axes of the file passed with `-c`.

The `[post_processing]` table is checked when the grid is launched with `proteus grid` and again by `proteus grid-analyse`: an unknown key, a value of the wrong type, an unknown colour map, style or plot format stops the command with an error before any case runs.

The `[post_processing]` table accepts:

| Key | Meaning |
|---|---|
| `update_csv` | Rebuild the summary CSVs from the case folders before plotting (default `true`). Set `false` to plot from existing CSVs. |
| `plot_status` | Make the status-summary bar plot (default `true`). |
| `plot_ecdf` | Make the ECDF grid plot (default `true`). |
| `colormap` | Colour map for a parameter with more values than the colour cycle of the plot style: a Crameri scientific colour map from `cmcrameri` (for example `"batlowK_r"`, `"batlow"`, `"vik"`), a `proteus-mpl` colour map (`"proteus"`, `"proteus_div"`, `"proteus_phase"` and their `_r` forms, whatever the `style`) when `proteus-mpl` is installed, or any matplotlib colour map (`colourmap` is also accepted). Defaults to `batlowK_r`. |
| `output_variables` | List of `runtime_helpfile.csv` columns to plot, plus the derived `solidification_time`. Defaults to `solidification_time`, `Phi_global`, `T_surf`, `P_surf` and `atm_kg_per_mol`. |
| `plot_format` | Image format (`"png"` or `"pdf"`); defaults to the base config's `params.out.plot_fmt`, then `"png"`. |
| `style` | `"light"` or `"dark"` draws the plots in that variant of the [`proteus-mpl`](https://pypi.org/project/proteus-mpl/) style of the documentation figures (needs `pip install proteus-mpl`, included in the `develop` and `docs` extras); `"matplotlib"` keeps the active matplotlib style. Unset, the plots use `"light"` when `proteus-mpl` is installed and the matplotlib style otherwise. |

`solidification_time` is computed by the analysis as the first time a case's `Phi_global` is at or below its `params.stop.solid.phi_crit` (the solidification stop criterion). It is computed the same way when that stop criterion (`params.stop.solid.enabled`) is disabled, for example in a grid that runs every case to a fixed end time, and is empty (NaN) only for a case that never reached `phi_crit`.

The outputs are written under the grid's output folder:

```text
output/<grid name>/post_processing/
├── extracted_data/
│   ├── final_extracted_data_all.csv            # every case
│   ├── final_extracted_data_completed.csv      # completed cases
│   └── final_extracted_data_running_error.csv  # running, errored or unknown-status cases
└── grid_plots/
    ├── summary_grid_statuses.<fmt>             # case count per final status
    └── ecdf_grid_plot.<fmt>                    # ECDF per (parameter, output) panel
```

The CSVs are tab-separated (like other PROTEUS output), one row per started case, holding the case number, status, status category (`Completed`, `Running`, `Error`, or `Unknown` when the status file is missing, unreadable or holds an unrecognised code), `config_source`, the swept parameter values, every helpfile column at the final row, and `solidification_time`. The swept values come from the case's `init_coupler.toml`; a case that stopped before writing it (for example on a configuration error) takes them from `cfgs/case_<N>.toml`, the config the grid wrote for it, and `config_source` names the file used.

The plots take their colours from the plot style: the series colours from its colour cycle, the greys from its text and face colours, and the ticks, spines and grid from the style itself.

The status-summary plot has one horizontal bar per final status, plus a `Not started` bar for grid cases that have no case folder yet (for example still queued on SLURM). Completed, running and errored statuses take the colour of the style's cycle nearest to green, blue and red; unknown and not-started cases are grey.

The ECDF grid plot uses the completed cases. It has one row per swept parameter and one column per requested output; curves are coloured by the parameter value. A parameter with no more values than the colour cycle of the style takes the cycle colours and a legend; one with more takes the colour map, trimmed at each end until its colours stand out against the face colour of the style, and a colourbar with a tick at each swept value (string values get a legend). A numeric parameter is coloured on a log scale when its default plot setting is logarithmic and its values span at least a factor of 10. The layout follows Figure 5 of [Nicholls et al. (2024)](https://arxiv.org/abs/2411.19137). Each curve is a percentage of all completed cases with that parameter value, so it levels off below 100 % when some cases have no value for the output (for example cases that never solidified). Outputs whose default plot setting is logarithmic (for example `P_surf`) are drawn on a log axis when their values are all positive and span at least a factor of 10. When some cases have a zero value (for example a surface pressure of 0 bar after the atmosphere escaped), the axis is a 0 + log axis: a short linear segment from 0 to the decade below the smallest positive value, then logarithmic, so those cases appear at 0. Over a narrower range the axis is linear.

To run atmospheric chemistry on every case of a grid, use `proteus offchem -g output/<grid folder>`; see [Postprocessing and chemistry](usage_postprocessing.md#grids).

---

**See also:** [Running and output](usage_running.md) | [Configuration file](config.md) | [Execution and output reference](../Reference/config/params.md) | [Parameter grid tutorial](../Tutorials/parameter_grid.md) | [Bayesian inference](inference.md)
