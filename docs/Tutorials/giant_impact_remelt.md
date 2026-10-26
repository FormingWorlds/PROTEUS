# Giant impacts with a partial re-melt

This tutorial shows how a giant impact re-melts part of the mantle in a PROTEUS run.
You will write an impact timeline with a melt fraction and run the same planet with a full, a partial and no re-melt.
Then you read what the impact does to the mantle temperature, the liquid fraction of the mantle and the energy budget.

A giant impact melts the mass fraction $f$ of the mantle, the **melt fraction** of the impact.
PROTEUS raises the outermost mantle that holds this mass to the **initial condition** of the run, the thermal state that `planet.temperature_mode` sets at the start.
The deeper mantle keeps its state ([How much of the mantle an impact melts](../Explanations/model.md#how-much-of-the-mantle-an-impact-melts) explains the rule).
The setting `accretion.remelt` chooses $f$:

- `"full"`: $f = 1$, the whole mantle rises to the initial condition.
- `"computed"`: $f$ comes from the impact, as the Morrigan accretion module reports it or as the `f_melt` column of a timeline file gives it.
- Without a setting, a run with the Morrigan module uses `"computed"` and every other run `"full"`.

Four names on this page belong to the same quantity:

| Name | Where | What |
|---|---|---|
| $f$ | text | the melt fraction of an impact |
| `f_melt` | column of the timeline file (Step 1) | the $f$ of each impact, read by `"computed"` |
| `accretion.remelt` | configuration (Step 2) | chooses where $f$ comes from |
| `f_remelt` | output column (Step 4) | the $f$ that the impact applied on that row |

## Prerequisites

- PROTEUS installed with `bash install.sh` (see the [installation guide](../How-to/installation.md)), with the Aragog interior, and its environment active (`conda activate proteus`).
- `FWL_DATA` set (Step 3 downloads the interior data).
- `tomlkit` for Step 2; it comes with `pip install -e ".[develop]"`.

Run the Python blocks from the PROTEUS folder in one Python session, in the order of the steps.
Steps 4 and 6 use `out`, `runs` and `write_config` from Steps 1 and 2, and Steps 5 and 6 use `tables` from Step 4, so keep the session open during Step 3.
Run the commands of Step 3 in a second terminal, with the same environment and from the PROTEUS folder.
The 4 runs took 23 to 25 minutes in total on a workstation.

## Step 1: write an impact timeline with a melt fraction

The `timeline` accretion module reads its impacts from a file with one row per impact.
The row in the code block below is one impact at 1000 yr: an impactor of 0.1 Earth masses hits a planet of 1 Earth mass.
The columns are, in order:

| Column | Meaning | Unit |
|---|---|---|
| `time` | time of the impact | yr |
| `M_target_before`, `M_impactor`, `M_merged_after` | masses of the target before, the impactor, and the merged body | kg |
| `v_impact`, `v_esc` | collision velocity and mutual escape velocity | m/s |
| `impact_parameter` | sine of the impact angle (0 is head-on) | 1 |
| `R_target_before`, `R_impactor` | radii of the target and the impactor | m |
| `rho_target`, `rho_impactor` | bulk densities | kg/m$^3$ |
| `a_before`, `a_after` | semi-major axis of the target before and of the merged body after | m |
| `e_before`, `e_after` | eccentricity before and after | 1 |
| `id_target`, `id_impactor` | body ids | |
| `f_melt` | melt fraction of the impact, from 0 to 1 | 1 |

You can leave the whole `f_melt` column out of the file.
A run with `accretion.remelt = "full"` then still works, but a run with `"computed"` stops at the start.
A column that is present needs a number in every row.
The block writes 3 files: one with $f = 0.3$, one with $f = 0$ (the impact adds mass but re-melts nothing), and one with a blank `f_melt` cell.

```python
import os

out = os.path.abspath(os.path.join('output', 'tutorial_remelt'))
os.makedirs(out, exist_ok=True)

header = (
    'time,M_target_before,M_impactor,M_merged_after,v_impact,v_esc,impact_parameter,'
    'R_target_before,R_impactor,rho_target,rho_impactor,a_before,a_after,e_before,'
    'e_after,id_target,id_impactor,f_melt'
)
impact = (
    '1000.0,5.972e+24,5.972e+23,6.569200000000001e+24,11493.277490259983,'
    '9599.456365907872,0.5,6284776.941068927,3231205.3330786866,5743.301092030068,'
    '4226.078100956827,74798935350.0,74660017423.92729,0.1,0.08646396912736636,0,1,'
)
for name, f_melt in (('f03', '0.3'), ('f00', '0.0'), ('blank', '')):
    with open(os.path.join(out, f'timeline_{name}.csv'), 'w') as file:
        file.write(header + '\n' + impact + f_melt + '\n')
print(sorted(os.listdir(out)))
```

On our run this printed:

```text
['timeline_blank.csv', 'timeline_f00.csv', 'timeline_f03.csv']
```

A blank `f_melt` cell is not a valid melt fraction. PROTEUS refuses the file when it reads it, also with `accretion.remelt = "full"`, which ignores the values:

```python
from proteus.accretion.common import read_timeline

try:
    read_timeline(os.path.join(out, 'timeline_blank.csv'))
except ValueError as err:
    print(err)
```

On our run this printed:

```text
impact 0 at t = 1.0000e+03 yr: f_melt must be finite and in [0, 1], got nan
```

## Step 2: write the configurations

The configurations start from `input/dummy.toml` and change the interior to Aragog ([model description](../Explanations/model.md)).
Aragog divides the mantle into cells, thin spherical shells, each with its own entropy.
The `dummy` interior, used in the fourth run, is a fast stand-in with one mantle temperature and no cells.
On Aragog, `planet.temperature_mode = "isothermal"` with `tsurf_init = 4000` sets a nearly uniform entropy, not one temperature.
With one entropy, the temperature rises with depth by compression.
Each run writes its state at every time step as one row of the output table `runtime_helpfile.csv`.
Its first row in our run has `T_magma`, the temperature at the top of the mantle, at 4000 K, and `T_cmb`, the temperature at the core-mantle boundary, at 6197 K.
Its column `Phi_global`, the liquid fraction of the mantle (Step 4), is 1: the mantle starts fully molten.
The run stops at 1500 yr, 500 yr after the impact.
The stop at solidification is off, so each run ends at 1500 yr; `params.dt.minimum` and `params.dt.maximum` (10 and 100 yr here) bound the time step.
The structure module stays `dummy` (`interior_struct.module`, set in `input/dummy.toml`); it gives the planet radius and the core size from scaling laws.
PALEOS is the set of equation-of-state (EOS) tables of the mantle material.
The dummy structure still reads the table `interior_struct.zalmoxis`: it names the mantle EOS, and its `lookup_nP` and `lookup_nS` set the pressure and entropy points of the PALEOS EOS table that Aragog reads (150 by 60, to keep the runs short).
The configuration check requires `interior_struct.melting_dir` for Aragog unless the structure module generates a PALEOS table set, and the dummy structure generates none.
Aragog then reads the melting curves from the PALEOS EOS table and not from this folder, and the run prints a warning that you can ignore.
The block writes 4 configuration files: `full`, `f03` (`computed` with $f = 0.3$), `f00` (`computed` with $f = 0$) and `dummy_f03` (the dummy interior with $f = 0.3$).
The `full` run reads the $f = 0.3$ timeline but ignores its `f_melt` value.

```python
import copy
import tomllib

import tomlkit

with open(os.path.join('input', 'dummy.toml'), 'rb') as file:
    base = tomllib.load(file)


def write_config(name, remelt, timeline, interior='aragog'):
    """Write <out>/<name>.toml: input/dummy.toml with one impact and the given re-melt."""
    cfg = copy.deepcopy(base)
    cfg['params']['out'].update(path=os.path.join(out, name), write_mod=1, plot_mod=0)
    cfg['params']['out']['archive_mod'] = 'none'
    cfg['params']['stop']['time']['maximum'] = 1500.0
    cfg['params']['stop']['solid']['enabled'] = False
    cfg['params']['dt'].update(minimum=10.0, maximum=100.0)
    cfg['accretion'] = {
        'module': 'timeline',
        'remelt': remelt,
        'timeline': {'timeline_path': os.path.join(out, f'timeline_{timeline}.csv')},
    }
    cfg['interior_energetics']['module'] = interior
    if interior == 'aragog':
        cfg['planet'].update(temperature_mode='isothermal', tsurf_init=4000.0)
        cfg['interior_struct']['melting_dir'] = 'Monteux-600'
        cfg['interior_struct'].setdefault('zalmoxis', {}).update(lookup_nP=150, lookup_nS=60)
    path = os.path.join(out, f'{name}.toml')
    with open(path, 'w') as file:
        file.write(tomlkit.dumps(cfg))
    return path


runs = {
    'full': write_config('full', 'full', 'f03'),
    'f03': write_config('f03', 'computed', 'f03'),
    'f00': write_config('f00', 'computed', 'f00'),
    'dummy_f03': write_config('dummy_f03', 'computed', 'f03', interior='dummy'),
}
print(runs)
```

## Step 3: run the planet

Download the interior data once; the command reads `f03.toml` from Step 2:

```text
proteus get interiordata --config-path output/tutorial_remelt/f03.toml
```

Then run the 4 configurations one after the other in the second terminal; together they take 23 to 25 minutes.
Each run writes its output table `runtime_helpfile.csv` (the helpfile) into its output folder, and the last command shows how each run ended.

```console
proteus start --offline -c output/tutorial_remelt/full.toml
proteus start --offline -c output/tutorial_remelt/f03.toml
proteus start --offline -c output/tutorial_remelt/f00.toml
proteus start --offline -c output/tutorial_remelt/dummy_f03.toml
cat output/tutorial_remelt/*/status
```

In our 2 runs of this page on a workstation, the full re-melt took about half a minute, the run with $f = 0.3$ about 9 minutes, the run with $f = 0$ 14 to 15 minutes, and the dummy run a few seconds.
Each Aragog run warns that an isothermal initial condition is not guaranteed to be fully molten.
Only `planet.temperature_mode = "liquidus_super"` starts above the liquidus, the temperature above which the mantle is fully molten, for any mantle.
Here the mantle starts fully molten (liquid fraction `Phi_global` 1, defined in Step 4) and crystallises a little before the impact (0.980 at 925 yr).
The runs with $f = 0.3$ and $f = 0$ also log a few `EntropySolver: integration failed` errors; Aragog then retries the step with a shorter time step and a looser absolute tolerance of the solver (`atol_sf` 3 and then 5 times).
On our run all 4 ended with the status `Completed (target time)`, as the last command shows.

## Step 4: read the impact row

The impact row is the row at `Time` = 1000 yr.
On that row:

- `f_remelt` is the melt fraction that the impact applied, 0 on a row without an impact;
- `step_dE_impact_J` is the heat that the re-melt added to the mantle \[J\];
- `Phi_global` is the liquid fraction of the mantle: the mean liquid fraction $\phi$ of the cells, each cell counted by its mass;
- a cell has $\phi = 0$ when solid and $\phi = 1$ when fully molten;
- `T_magma` is the temperature at the top of the mantle \[K\], and `T_cmb` the temperature at the core-mantle boundary \[K\].

Other pages call `Phi_global` the global melt fraction or the mantle melt mass fraction; neither is the melt fraction $f$ of an impact.
Each table shows the first row, the row before the impact, the impact row, the row after it and the last row.

```python
import pandas as pd

cols = ['Time', 'f_remelt', 'Phi_global', 'T_magma', 'T_cmb', 'step_dE_impact_J']
tables = {}
for name in runs:
    tables[name] = pd.read_csv(os.path.join(out, name, 'runtime_helpfile.csv'), sep=r'\s+')
for name in ('full', 'f03', 'f00'):
    t = tables[name]
    i = int(t.index[t['Time'] == 1000.0][0])
    print(f'--- {name}')
    print(t.loc[[0, i - 1, i, i + 1, len(t) - 1], cols].to_string(float_format=lambda v: f'{v:.6g}'))
```

On our run this printed:

```text
--- full
      Time  f_remelt  Phi_global  T_magma   T_cmb  step_dE_impact_J
0        0         0           1     4000 6197.16                 0
17 924.905         0    0.980475  3540.43 6002.16                 0
18    1000         1           1  4000.07 6197.22       1.09936e+31
19 1060.08         0           1  3972.66 6217.12                 0
27    1500         0     0.99705  3757.62 6142.52                 0
--- f03
      Time  f_remelt  Phi_global  T_magma   T_cmb  step_dE_impact_J
0        0         0           1     4000 6197.16                 0
17 924.905         0    0.980475  3540.43 6002.16                 0
18    1000       0.3    0.976851  4000.07 6023.73       2.88137e+30
19 1060.08         0    0.975142   3856.2 6087.49                 0
33    1500         0    0.964204  3423.11 6064.77                 0
--- f00
      Time  f_remelt  Phi_global  T_magma   T_cmb  step_dE_impact_J
0        0         0           1     4000 6197.16                 0
17 924.905         0    0.980475  3540.43 6002.16                 0
18    1000         0     0.97718  3508.09 6023.73                 0
19 1120.15         0    0.969534  3463.06 6075.44                 0
25    1500         0    0.951639  3331.11 6041.32                 0
```

The full re-melt raises the whole mantle: `Phi_global` is 1, `T_cmb` rises to 6197 K, and the re-melt adds the most heat ($1.1 \times 10^{31}$ J).
The partial re-melt with $f = 0.3$ raises `T_magma` to 4000 K and adds $2.9 \times 10^{30}$ J; the run with $f = 0$ adds none.
`T_cmb` is the same in the $f = 0.3$ and $f = 0$ runs (6023.73 K), because the deeper mantle keeps its entropy.
In both, `T_cmb` rises across the impact also without a re-melt: the log of the $f = 0$ run shows 5994 K at the end of the step before the impact and 6024 K after it. This tutorial does not establish the cause.
The partial re-melt adds 0.26 of the heat of the full re-melt, not 0.30.
A separate check of this case, outside the tutorial, recomputed the heat cell by cell and found why.
The 16 fully re-melted cells take the same heat as in the full re-melt, and the partly melted cell at the melt front takes 6 % of it.
In the full re-melt the re-melted top cells take 28 % of the heat: per kilogram they take 3.71 MJ against 3.79 MJ for the deeper mantle, because their entropy rises more (907 against 711 J/kg/K) but at a lower temperature (about 4090 K against 5230 K along the path).

```python
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for name, label in (('full', 'full'), ('f03', 'computed, f = 0.3'), ('f00', 'computed, f = 0')):
    t = tables[name]
    axes[0].plot(t['Time'], t['T_magma'], label=label)
    axes[1].plot(t['Time'], t['Phi_global'], label=label)
for ax, ylabel in zip(axes, ('T_magma [K]', 'Phi_global')):
    ax.axvline(1000.0, ls=':', color='grey')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel(ylabel)
axes[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(out, 'remelt_runs.png'))
```

The block draws the plot into `output/tutorial_remelt/remelt_runs.png`; the page does not show it.
In the plot, the left panel shows `T_magma` and the right panel `Phi_global` against time, one line per run, and the dotted line marks the impact.
The 3 lines agree before the impact.
At the impact the full and the $f = 0.3$ runs jump to 4000 K, the full run also to `Phi_global` = 1, and the $f = 0$ run does not jump.

## Step 5: why the liquid fraction can fall at a partial re-melt

Compare `Phi_global` on the impact row of the runs with $f = 0.3$ and $f = 0$.

```python
phi = {name: float(tables[name].loc[tables[name]['Time'] == 1000.0, 'Phi_global'].iloc[0]) for name in ('f03', 'f00')}
print(f"Phi_global at the impact: f = 0.3 {phi['f03']:.6f}, f = 0 {phi['f00']:.6f}")
```

On our run this printed:

```text
Phi_global at the impact: f = 0.3 0.976851, f = 0 0.977180
```

At the impact, the run with $f = 0.3$ has the lower `Phi_global`, by 0.0003.
The difference is small, but its sign is the opposite of what a hotter mantle suggests.
The top is 492 K hotter (`T_magma` 4000 K against 3508 K), and the re-melt added $2.9 \times 10^{30}$ J.
The reason is how `Phi_global` weights the cells.
Each cell counts with its mass $\rho V$, and $\rho$ is the density at the current entropy.
The re-melt heats a cell, so its density falls and it counts for less.
In this planet the re-melted outer 30 % of the mantle mass is already fully molten before the impact, so the re-melt adds heat there but no melt.
These hotter molten cells count for less in the average.
Some deeper cells are not fully molten ($\phi < 1$); they now count for more, and the mean drops.
No melt is lost: the separate cell-by-cell check of this case, outside the tutorial, found the 17 re-melted cells fully molten ($\phi = 1$) in both runs, with a mass $\rho V$ 0.3 to 5.5 % lower at $f = 0.3$.
We expect a rise of `Phi_global` when the outer mantle is partly solid at the impact, because the re-melt then adds melt.
This tutorial does not run that case.
This mantle is 98 % molten before the impact (`Phi_global` 0.980 at 925 yr), so the effect of a partial re-melt shows in the heat and in `T_magma`, not in `Phi_global`.
After the impact the re-melted planet stays hotter and more molten: at 1500 yr `Phi_global` is 0.964 against 0.952 (Step 4).

## Step 6: other interiors

The dummy interior has one mantle temperature and no cells.
At an impact it moves that temperature the share $f$ of the way from its value just before the impact to the initial temperature `planet.tsurf_init`, so two impacts compound.
The dummy run keeps the default `tsurf_init` of 4000 K, because Step 2 changes the temperature settings only for Aragog.
The rule never cools the mantle: a mantle hotter than `tsurf_init` keeps its temperature.
The scalar interiors book no re-melt heat, so `step_dE_impact_J` is 0 on the impact row of the dummy run.
The row before the impact in the table is at 924.9 yr.
The mantle evolves further until the impact at 1000 yr, so the table does not show the value that the rule starts from.

```python
t = tables['dummy_f03']
i = int(t.index[t['Time'] == 1000.0][0])
print(t.loc[[i - 1, i, i + 1], ['Time', 'f_remelt', 'T_magma', 'Phi_global']].to_string(float_format=lambda v: f'{v:.6g}'))
```

On our run this printed:

```text
      Time  f_remelt  T_magma  Phi_global
17 924.905         0  3495.51           1
18    1000       0.3  3625.31           1
19 1120.15         0  3570.58           1
```

PROTEUS refuses an accretion run on the boundary interior.
That interior cannot shorten a time step to land on the impact time.
PROTEUS also refuses SPIDER, which has no re-melt path.
It also refuses `interior_struct.module = "spider"`.
The refusal comes when PROTEUS loads the configuration:

```python
from proteus.config import read_config_object

try:
    read_config_object(write_config('boundary', 'computed', 'f03', interior='boundary'))
except ValueError as err:
    print([line.strip() for line in str(err).splitlines() if 'cannot run with' in line][0])
```

On our run this printed:

```text
accretion.module = 'timeline' cannot run with interior_energetics.module = 'boundary': a giant impact re-melts the mantle and has to be applied where the timeline places it, but the boundary interior does not forward its interior state object to the time-stepper, so the step cannot be shortened to land on a scheduled impact. Use interior_energetics.module = 'aragog' (or 'dummy' for a test).
```

## Summary

- Set `accretion.remelt = "computed"` and give each impact an `f_melt` between 0 and 1, or use `"full"` for a whole-mantle re-melt.
- On the impact row, `f_remelt` gives the melt fraction applied and, on Aragog, `step_dE_impact_J` the heat the re-melt added (0 on the dummy interior).
- A partial re-melt can lower `Phi_global` when the re-melted top was already molten; read the temperature and the heat as well.
- Accretion runs need the Aragog or the dummy interior, and a structure module other than SPIDER.

The validation cases of the rule are on the [re-melt validation page](../Validation/interior_energetics/remelt.md).

## Next steps

- Start from a mantle that is partly solid at the impact, for example a lower `tsurf_init`, and compare `Phi_global` at the impact row with and without the re-melt.
- Run an accretion case with the Morrigan module, which reports $f$ for each merger, instead of a timeline file.
- Use `planet.temperature_mode = "liquidus_super"`, so that the re-melted cells are fully molten for any mantle.
