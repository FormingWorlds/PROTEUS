# Giant impacts with a partial re-melt

This tutorial shows how a giant impact re-melts part of the mantle in a PROTEUS run.
You will write an impact timeline with a melt fraction, run the same planet with a full and with a partial re-melt, and read what the impact does to the mantle temperature, the melt fraction and the energy budget.

A giant impact melts the mass fraction $f$ of the mantle, the **melt fraction** of the impact.
PROTEUS raises the outermost mantle that holds this mass to the initial condition of the run, and the deeper mantle keeps its state ([How much of the mantle an impact melts](../Explanations/model.md#how-much-of-the-mantle-an-impact-melts) explains the rule).
The setting `accretion.remelt` chooses $f$:

- `"full"`: $f = 1$, the whole mantle rises to the initial condition.
- `"computed"`: $f$ comes from the impact, as the Morrigan module reports it or as the `f_melt` column of a timeline file gives it.
- Without a setting, a run with the Morrigan module uses `"computed"` and every other run `"full"`.

## Prerequisites

- PROTEUS installed with `bash install.sh` (see the [installation guide](../How-to/installation.md)), with the Aragog interior.
- `FWL_DATA` set and the interior data downloaded; [Quick start (all-dummy)](quick_start_dummy.md) runs on your machine.

Run every block from the PROTEUS folder, in one Python session, in the order of the steps.
The 4 runs took about 23 minutes in total on a workstation (Step 3).

## Step 1: write an impact timeline with a melt fraction

The `timeline` accretion module reads its impacts from a file with one row per impact.
The row below is one impact at 1000 yr: an impactor of 0.1 Earth masses hits a planet of 1 Earth mass.
The last column, `f_melt`, is the melt fraction of the impact.
The block writes 2 files, one with $f = 0.3$ and one with $f = 0$ (the impact adds mass but re-melts nothing).

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

A blank `f_melt` cell is not a valid melt fraction; PROTEUS refuses the file when it reads it:

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

The configurations start from `input/dummy.toml` and change the interior to Aragog, with an isothermal initial condition at 4000 K.
The run stops at 1500 yr, 500 yr after the impact.
The block writes 3 Aragog runs (`full`, `computed` with $f = 0.3$, `computed` with $f = 0$) and 1 run on the dummy interior with $f = 0.3$.

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

Each run writes its rows to `runtime_helpfile.csv` in its output folder.

```console
proteus start --offline -c output/tutorial_remelt/full.toml
proteus start --offline -c output/tutorial_remelt/f03.toml
proteus start --offline -c output/tutorial_remelt/f00.toml
proteus start --offline -c output/tutorial_remelt/dummy_f03.toml
```

On a workstation the full re-melt took about 1 minute, the runs with $f = 0.3$ and $f = 0$ about 9 and 14 minutes, and the dummy run less than 1 minute.
The runs with $f = 0.3$ and $f = 0$ log a few `EntropySolver: integration failed` errors; on our run all 4 still ended with the status `Completed (target time)`.

## Step 4: read the impact row

The impact row is the row at 1000 yr.
`f_remelt` is the melt fraction that the impact applied on that row, 0 without an impact, and `step_dE_impact_J` the heat that the re-melt added to the mantle [J].
`Phi_global` is the mass fraction of the mantle that is molten, `T_magma` the temperature at the top of the mantle [K] and `T_cmb` the temperature at the core-mantle boundary [K].

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
    print(t.loc[[i - 1, i, i + 1, len(t) - 1], cols].to_string(float_format=lambda v: f'{v:.6g}'))
```

On our run this printed:

```text

```

The full re-melt raises the whole mantle: `Phi_global` is 1 and the heat booked at the impact is the largest.
The partial re-melt with $f = 0.3$ raises `T_magma` to the initial condition but books less heat, and the run with $f = 0$ books none.

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

Open `output/tutorial_remelt/remelt_runs.png`: the dotted line marks the impact.

## Step 5: why Phi_global can fall at a partial re-melt

Compare `Phi_global` on the impact row of the runs with $f = 0.3$ and $f = 0$.

```python
phi = {name: float(tables[name].loc[tables[name]['Time'] == 1000.0, 'Phi_global'].iloc[0]) for name in ('f03', 'f00')}
print(f"Phi_global at the impact: f = 0.3 {phi['f03']:.6f}, f = 0 {phi['f00']:.6f}")
```

On our run this printed:

```text
Phi_global at the impact: f = 0.3 0.976851, f = 0 0.977180
```

At the impact, the run with $f = 0.3$ has the lower `Phi_global`, although its top is 490 K hotter (`T_magma` 4000 K against 3508 K) and it booked $2.9 \times 10^{30}$ J of heat.
The reason is how `Phi_global` is computed: it is the melt fraction of each cell weighted by the cell mass, and Aragog takes that mass from the density at the current entropy.
In this case the outer 30 % of the mantle is already fully molten before the impact, so the re-melt adds heat there but no melt.
The hotter melt is less dense, so these molten cells weigh less in the average, and `Phi_global` falls.
A separate check of this case found that the 17 re-melted cells have a melt fraction of 1 in both runs and that their mass from the density falls by 5 %; with the densities of the $f = 0$ run, the melt fractions of the $f = 0.3$ run give exactly the `Phi_global` of the $f = 0$ run.
A planet whose outer mantle is partly solid at the impact shows a rise of `Phi_global` instead.
After the impact the re-melted planet stays hotter and more molten: at 1500 yr `Phi_global` is 0.964 against 0.952 in Step 4.

## Step 6: other interiors

The dummy interior has one mantle temperature and no cells.
At an impact it moves that temperature the share $f$ of the way from its value before the impact to the initial temperature, so two impacts compound.

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

The boundary interior and SPIDER cannot apply an impact at the time the timeline gives, so PROTEUS refuses an accretion run with them when it loads the configuration:

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
- On the impact row, `f_remelt` gives the melt fraction applied and `step_dE_impact_J` the heat the re-melt added.
- A partial re-melt can lower `Phi_global` when the re-melted top was already molten; read the temperature and the heat as well.
- Accretion runs need the Aragog or the dummy interior.

The validation cases of the rule are on the [re-melt validation page](../Validation/interior_energetics/remelt.md).
