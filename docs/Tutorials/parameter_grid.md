# Parameter grid sweep

This tutorial demonstrates how to run an ensemble of PROTEUS simulations
across a grid of parameter values using the `proteus grid` command. Grid
runs are the standard approach for parameter studies, sensitivity analyses,
and population synthesis.

## How grids work

PROTEUS generates the Cartesian product of all specified parameter axes.
Each combination becomes an independent simulation case with its own
configuration file, output directory, and status tracking.

For example, 4 planet masses $\times$ 4 hydrogen budgets = 16 cases, each
running the full coupled evolution.

## Configuration

Grid runs use a separate TOML file that specifies the base configuration,
output settings, and parameter axes. Create `input/tutorial_grid.toml`:

```toml
config_version = "3.0"

# Base config: each grid point starts from this file
ref_config = "input/dummy.toml"

# Output folder
output = "tutorial_grid"

# Local execution settings
use_slurm = false
max_jobs  = 4

# Parameter axes (Cartesian product)

["planet.mass_tot"]
    method = "direct"
    values = [0.5, 1.0, 2.0, 3.0]

["planet.elements.H_budget"]
    method = "arange"
    start  = 1000
    stop   = 5000
    step   = 1000
```

### Sweep methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `direct` | Explicit list of values | `values = [...]` |
| `arange` | Evenly spaced (inclusive endpoint) | `start`, `stop`, `step` |
| `linspace` | Evenly spaced, fixed count | `start`, `stop`, `count` |
| `logspace` | Log-spaced, fixed count | `start`, `stop`, `count` |

### Parameter paths

The parameter name in the TOML table header (e.g., `"planet.mass_tot"`)
is a dot-separated path into the PROTEUS configuration. Any parameter
from the [config reference](../Reference/config/params.md) can be swept.

## Running

```bash
conda activate proteus
proteus grid -c input/tutorial_grid.toml
```

The grid manager:

1. Generates all parameter combinations (4 masses $\times$ 5 H budgets = 20 cases)
2. Writes a configuration file for each case
3. Launches cases in parallel (up to `max_jobs` at a time)
4. Monitors completion and reports progress

## Monitoring progress

While the grid is running, check the status:

```bash
proteus grid-summarise -o output/tutorial_grid
```

This shows how many cases are running, completed, or failed.

To filter by status:

```bash
proteus grid-summarise -o output/tutorial_grid -s Completed
proteus grid-summarise -o output/tutorial_grid -s Error
```

## Results

Each case writes output to `output/tutorial_grid/case_NNNNNN/`. The grid
directory also contains:

- `manager.log`: the grid manager log
- `ref_config.toml`: copy of the base configuration
- `cfgs/`: per-case configuration files

### Packing results

To bundle the results for sharing or analysis:

```bash
proteus grid-pack -o output/tutorial_grid
```

This creates `output/tutorial_grid/pack.zip` containing the helpfile,
status, and log for each case (optionally including plots).

### Analysis

Load all helpfiles for cross-case comparison:

```python
import pandas as pd
from pathlib import Path

grid_dir = Path('output/tutorial_grid')
results = []
for case_dir in sorted(grid_dir.glob('case_*')):
    hf = case_dir / 'runtime_helpfile.csv'
    if hf.exists():
        df = pd.read_csv(hf, sep='\t')
        # Read the case config to get parameter values
        results.append(df.iloc[-1])  # final state

final = pd.DataFrame(results)
print(final[['T_magma', 'P_surf', 'Phi_global', 'M_atm']].describe())
```

## SLURM cluster execution

For large grids on HPC clusters, set `use_slurm = true` and configure:

```toml
use_slurm = true
max_jobs  = 500      # max concurrent SLURM tasks
max_days  = 2        # walltime limit per case [days]
max_mem   = 12       # memory per CPU [GB]
```

The grid manager generates a SLURM job array script and submits it.
Each case runs as an independent array task.

## Exercises

1. Add an fO$_2$ axis: sweep `outgas.fO2_shift_IW` from $-2$ to $+4$
2. Use `logspace` to sweep `escape.zephyrus.efficiency` from 0.01 to 0.3
3. Run with real physics modules instead of dummy (change `ref_config`
   to your Earth analogue config)
