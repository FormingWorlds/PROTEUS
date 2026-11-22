# The core temperature of the structure solve

This tutorial shows how the core temperature sets the core of the Zalmoxis structure.
You run the same planet twice: once with the core as part of the mantle energy balance, and once with its own energy budget.
Then you read which temperature each structure update gave the core, and how the core radius and the central pressure follow it.

Each Zalmoxis solve with a mantle temperature profile takes the temperature of its core nodes from a core temperature table: pressure and temperature pairs on the core adiabat. The adiabat goes through `T_cmb`, the core-mantle boundary (CMB) temperature of the run, or through the temperature of the live core with its own energy budget.
A hotter core is less dense, so it is larger and the central pressure is lower.
[The core temperature in the structure solve](../Explanations/model.md#the-core-temperature-in-the-structure-solve) explains the rule.

## Prerequisites

- PROTEUS installed with `bash install.sh` (see the [installation guide](../How-to/installation.md)), with the Aragog interior and the Zalmoxis structure, and its environment active (`conda activate proteus`).
- `FWL_DATA` set; the first run downloads the equation of state (EOS) tables.
- `tomlkit` and `pandas`; both come with `pip install -e ".[develop]"`.

Run the commands from the PROTEUS folder. Run the Python blocks in one Python session, in the order of the steps.
Each run took 6 to 8 minutes on a workstation.

## Step 1: the configuration

The configuration is `input/tutorials/tutorial_core_table.toml`.
An Earth-mass planet starts fully molten, on the coolest adiabat above the liquidus (`temperature_mode = "liquidus_super"`).
Zalmoxis solves the structure with a PALEOS mantle and an iron core, Aragog evolves the interior, and CALLIOPE the outgassing; the star, orbit, atmosphere, escape and chemistry are dummy modules.
The structure is updated every 100 yr, and the run stops at 450 yr:

```toml
[interior_struct.zalmoxis]
update_interval = 100.0

[interior_energetics.aragog]
core_bc = "energy_balance"
```

With `core_bc = "energy_balance"` the core is part of the energy balance at the base of the mantle, and the table goes through the `T_cmb` of the run.

## Step 2: run the planet

```console
proteus start --offline -c input/tutorials/tutorial_core_table.toml
```

The output goes to `output/tutorial_core_table/`.

## Step 3: run the same planet with a core energy budget

With `core_bc = "core_module"` the core has its own energy budget, and the table goes through the temperature of that core.
The block writes a copy of the configuration with this setting and its own output folder:

```python
import tomlkit

cfg = tomlkit.parse(open('input/tutorials/tutorial_core_table.toml').read())
cfg['params']['out']['path'] = 'tutorial_core_table_cm'
cfg['interior_energetics']['aragog']['core_bc'] = 'core_module'
open('input/tutorials/tutorial_core_table_cm.toml', 'w').write(tomlkit.dumps(cfg))
```

```console
proteus start --offline -c input/tutorials/tutorial_core_table_cm.toml
```

## Step 4: read the structure updates

The block prints the start, each structure update and the last row of both runs.
`struct_core_T_used` is 1 when the solve took its core temperature from the table, and `struct_core_T_excess` is how far the core nodes lie past the pressures of the table, as a part of its pressure range.

```python
import pandas as pd

for name in ('tutorial_core_table', 'tutorial_core_table_cm'):
    h = pd.read_csv(f'output/{name}/runtime_helpfile.csv', sep=r'\s+')
    updates = h[h['R_core'].diff() != 0]
    print(name)
    print(f'{"t [yr]":>8} {"T_cmb [K]":>10} {"R_core [km]":>12} {"P_center [GPa]":>15} {"used":>5} {"excess":>7}')
    for _, row in pd.concat([updates, h.tail(1)]).iterrows():
        print(
            f'{row["Time"]:8.0f} {row["T_cmb"]:10.1f} {row["R_core"] / 1e3:12.1f} '
            f'{row["P_center"] / 1e9:15.2f} {row["struct_core_T_used"]:5.0f} {row["struct_core_T_excess"]:7.4f}'
        )
```

```text
tutorial_core_table
  t [yr]  T_cmb [K]  R_core [km]  P_center [GPa]  used  excess
       0     5726.1       4076.8          434.40     1  0.0000
     102     5619.2       4074.1          435.69     1  0.0036
     246     5492.6       4069.2          438.04     1  0.0066
     371     5402.1       4065.4          439.86     1  0.0051
     450     5351.1       4065.4          439.86     1  0.0051
tutorial_core_table_cm
  t [yr]  T_cmb [K]  R_core [km]  P_center [GPa]  used  excess
       0     5727.1       4076.8          434.40     1  0.0000
     102     5725.4       4077.0          434.64     1  0.0007
     246     5718.2       4075.4          435.76     1  0.0032
     371     5709.1       4074.0          436.72     1  0.0027
     450     5701.2       4074.0          436.72     1  0.0027
```

## Step 5: compare the 2 cores

Every solve used the table, and the core nodes lie at most 0.0066 of its pressure range past it; PROTEUS stops a solve above 0.1.

The core with its own energy budget cools slowly: after 450 yr it is 350 K hotter than the core in the mantle energy balance (5701.2 against 5351.1 K).
The hotter core is less dense, so its radius is 8.6 km larger and its central pressure 3.1 GPa lower.
With `"energy_balance"` the core cools with the base of the mantle, and the core shrinks by 11.4 km over the 3 updates.

The rows between updates keep the structure of the last update, so `R_core` and `P_center` change only at the update rows.

## Where to go next

- [The core temperature in the structure solve](../Explanations/model.md#the-core-temperature-in-the-structure-solve): the rule and its limits.
- [Core temperature in the structure solve](../Validation/interior_struct/core_temperature.md): the check of the structure against 2 independent solves.
- [Core evolution](../How-to/usage_core_evolution.md): the settings of `core_bc = "core_module"`.
