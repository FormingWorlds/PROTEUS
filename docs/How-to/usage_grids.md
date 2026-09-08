# Parameter grids (ensembles)

!!! tip "Worked example"
    For a runnable, end-to-end walkthrough (a fast all-dummy grid over planet mass, orbital distance, and hydrogen budget, with plots), see the [parameter grid sweep tutorial](../Tutorials/parameter_grid.md).

It is often useful to run grids of forward models, where each point in a grid represents a different set of parameters. This is done with the `proteus grid` command:

```console
proteus grid -c input/example.grid.toml
```

Configure a grid by creating a TOML file that specifies the grid's axes and how it should be run. The example at `input/example.grid.toml` uses the dummy configuration file as a reference and modifies it for every combination of the parameters declared in the `.grid.toml` file. See the [configuration guide](config.md) for how to edit configuration files, and the [parameter grid tutorial](../Tutorials/parameter_grid.md) for a worked end-to-end example.

## Defining grid axes

A grid configuration file (`*.grid.toml`) contains two kinds of entries: top-level settings that control the whole ensemble, and one TOML table per parameter axis.

The top-level settings are:

| Setting | Meaning |
|---|---|
| `ref_config` | Base ("reference") config file, relative to the PROTEUS root. Every case starts from this file and overrides only the swept parameters. |
| `output` | Output folder name created inside `output/`. Use `"auto"` for a timestamped `grid_YYYYMMDD_HHMMSS_xxxx` name, or any fixed string. |
| `symlink` | Absolute path used to redirect the output to alternative storage (for example a scratch disk); set to `""` to disable. |
| `use_slurm` | Whether to dispatch through Slurm (see below). |
| `max_jobs` | Maximum number of cases running concurrently. |
| `max_days`, `max_mem` | Per-job walltime (days) and memory (GB) limits, used when dispatching through Slurm. |
| `jax_cache` | Share a JAX compilation cache across Slurm array tasks (see below). Default `false`; only affects Slurm dispatch. |

Each parameter axis is a TOML table whose **name is the dotted path of the config field to vary**. For example, `["planet.mass_tot"]` sweeps `config.planet.mass_tot`, and `["outgas.fO2_shift_IW"]` sweeps the mantle redox offset. Any field documented in `input/all_options.toml` (or the [configuration reference](config.md)) can serve as an axis. The grid manager treats every top-level key containing a dot as an axis, and every key without one as a setting, so axis names must always be given as the full dotted path.

Each axis declares a `method` that controls how its values are generated:

| `method` | Required keys | Values produced |
|---|---|---|
| `direct` | `values = [...]` | Exactly the listed values. Numeric lists are sorted and de-duplicated; string lists are kept as written. |
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
individual subprocesses which compose the grid. The variable `max_jobs` specifies the maximum number of CPU cores
which should be utilised by the grid at any one time. This is limited by the number of CPU
cores available on your machine. This method works without Slurm, and can be applied on servers or
on multicore personal computers.

In this case, you will need to make sure that PROTEUS stays open in order to manage its subprocesses.

### With Slurm

Alternatively, you can access high performance compute nodes through the Slurm workload manager (e.g. on Habrok and Snellius). This is a two-step process. To do this, set `use_slurm = true` in your grid's configuration file. Then set `max_mem` and `max_days` to specify how much memory should be allocated to each job (each simulation). The shipped example uses 12 GB and 1 day. Ensure that these values are within the limits of the server you are working on.

With these options enabled, running PROTEUS will produce a script called `slurm_dispatch.sh` in the specified output folder, as well as write the required configuration files to a subfolder called `cfgs/`.

To dispatch your grid via Slurm, you **must then run** the command `sbatch <path>` where `<path>` is the path to the dispatch script created by the `proteus grid` command. You will be prompted to do this in the terminal.

Monitor your running jobs with `squeue -u $USER`. To cancel **all** of your running jobs, use `scancel -u $USER`.
The original PROTEUS process does not need to stay open when using Slurm to manage the subprocesses.

Set `jax_cache = true` to add a shared JAX compilation cache to the dispatch script. Each job then exports `JAX_COMPILATION_CACHE_DIR` pointing at a `jax_cache/` subdirectory of the grid output, so array tasks reuse each other's compiled kernels instead of recompiling the same interior solver per task. The cache is bounded at 80 GiB (`JAX_COMPILATION_CACHE_MAX_SIZE`), with a 1 s minimum compile time and a 4 KiB minimum entry size so only worthwhile kernels are stored. These exports are written into the Slurm dispatch script only; a local grid run (`use_slurm = false`) spawns subprocesses without them.

The cluster guides give site-specific Slurm settings: [Habrok](habrok_cluster_guide.md), [Snellius](snellius_cluster_guide.md), and [Kapteyn](kapteyn_cluster_guide.md).

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

Each `case_NNNNNN/` folder is a complete PROTEUS run directory, numbered in the same order as the grid points listed in `manager.log`. Because the reference config and grid definition are copied into the output folder, an ensemble can be regenerated or extended from its output directory alone. The per-case output files are described on the [Running and output](usage_running.md#output-and-results) page.

## Viewing grid status

Use the CLI to view the status of a grid, such as to check cases which are finished. For example, the command below will summarise the top-level statuses of the demo grid.

```console
proteus grid-summarise -o output/grid_demo/
```

Add `-s` to find out which cases have a particular status. For example, the command below will list all completed cases.

```console
proteus grid-summarise -o output/grid_demo/ -s completed
```

## Packaging grid results

Use the CLI to package the results of a grid into a zip file; for example to share or back it up. The command below creates `pack.zip` in the `grid_demo/` folder. This does not store all the data for each case, only the most important files: the 1D resolved interior and atmospheric data are dropped, while files containing global parameters and auto-generated plots are preserved.

```console
proteus grid-pack -o output/grid_demo/
```

---

**See also:** [Running and output](usage_running.md) | [Configuration file](config.md) | [Execution and output reference](../Reference/config/params.md) | [Parameter grid tutorial](../Tutorials/parameter_grid.md) | [Bayesian inference](inference.md)
