# Check the energy closure of a run

This page shows how to sum the energy that a run books, step by step, and read how well the planet keeps its energy. For what the terms mean and why some stay open, see [the whole-run energy ledger](../Explanations/coupling_loop.md#the-whole-run-energy-ledger).

## Run the command on an output folder

```console
python tools/whole_run_ledger.py output/my_run
```

The command reads `runtime_helpfile.csv` and the core settings (`core_bc`, `core_tfac_avg`) from `init_coupler.toml` of the folder. `--core-bc` replaces the setting of the config. With an older run that has no config file, give the core with `--core-bc energy_balance`; the core temperature factor is then 1.147. Only `energy_balance` and `core_module` have a core ledger, and only runs with the Aragog interior and `surface_bc_mode = 'flux'`. `--start ROW` and `--end ROW` limit the window; the default starts at row 2, the last row at t = 0.

For a 450 yr run with `energy_balance` and 3 structure updates the output is:

```text
output/my_run (core_bc energy_balance, core_tfac 1.147)
steps 11, t 1 to 450 yr
S_state    -6.822665e+30 J
L_int      +7.499964e+30 J
C_cmb      +6.773096e+29 J
Q          +1.156540e+24 J
I_mantle   +0.000000e+00 J
D_core     -6.328395e+29 J
B_core     +0.000000e+00 J
E_atm      +7.519450e+30 J
R_m        -1.229789e+25 J  (-1.640e-06 of L_int)
R_c        +4.447006e+28 J  (+5.929e-03 of L_int)
G          -1.948594e+28 J  (-2.598e-03 of L_int)
R_p        +6.394370e+28 J  (+8.526e-03 of L_int)
R_c by step class: update -8.6788e+25 J, remap +4.4612e+28 J, other -5.5027e+25 J
steps by class: lid 0, update 3, remap 3, zero 1, impact 0, negative_F_atm 0
closure R_p/T +4.263e-03 (T 1.5000e+31 J); compression work (not summed) +5.8641e+29 J
mantle mass change +4.8347e+20 kg in 3 steps (energy not booked: it depends on the zero of the EOS energy)
```

## Read the output

- The first line names the run folder and the core settings that the command used. The second line gives the number of steps and the time span.
- The next block lists the terms summed over the window, in J. Every run books 5 of them: the change of the mantle heat (`S_state`), the heat that leaves the interior (`L_int`), the heat across the core-mantle boundary (`C_cmb`), radiogenic and tidal heat (`Q`) and impact heat (`I_mantle`). The booked core heat (`B_core`) comes from columns of `core_module` and is 0 with `energy_balance`. The change of the core heat (`D_core`) is a column of `core_module`; with `energy_balance` the ledger computes it. The ledger also computes the loss to space (`E_atm`).
- The residuals follow, each also as a fraction of `L_int`: `R_m` for the mantle, `R_c` for the core, `G` for the coupling between the interior and the atmosphere, `R_p = R_m + R_c - G` for the planet.
- `R_c by step class` splits the core residual into the steps that end at a structure update (`update`), the steps after one (`remap`) and the other steps. In this run nearly all of it is in `remap`: `energy_balance` books no core heat at an update.
- `steps by class` counts the steps with a live sub-grid lid at either end (`lid`), the `update` and `remap` steps, the steps where time passes but the interior books no heat (`zero`), the steps with impact heat (`impact`) and the steps that start with a negative `F_atm` (`negative_F_atm`). `zero 1` is the first step of the run: it carries a loss to space but no interior heat, and it makes all of `G`.
- `closure R_p/T` is the planet residual over the throughput `T`: the sum of the absolute values of `S_state`, `L_int`, `C_cmb`, `Q`, `I_mantle` and `B_core`, in J. `D_core` is not in it. Here `T` is an energy, not a temperature.
- The compression work of the structure updates is printed in J and is not added to any residual.
- The mantle mass change is reported in kg only. Its energy depends on the zero of the energy of the equation of state (EOS), so the ledger does not book it.

The command stops with exit code 2 and prints the reason when it cannot build the ledger. The reasons are:

- a folder without `runtime_helpfile.csv`;
- no `init_coupler.toml`, or one that names no Aragog `core_bc`, and no `--core-bc`;
- an interior other than Aragog, or a core setting without a ledger;
- `surface_bc_mode = 'grey_body'`: the interior then loses heat by its own grey-body law, not by `F_atm`, so the coupling identity does not hold by design;
- a core temperature factor that is not finite and positive;
- a missing or non-numeric column of an older helpfile;
- a value that is not finite;
- a window outside the table, or one in which no step books interior heat;
- a step that carries energy while the time does not increase;
- a failed check of the ledger: the coupling identity on a step without a lid, or, when the helpfile holds `E_residual_cons_J`, the mantle residual against it, each to the precision with which the helpfile stores its numbers.

## Use the function from Python

The same numbers come from `proteus.utils.energy_ledger.whole_run_ledger`. The rows of the run above are stored with the tests, so this runs from the root folder of a PROTEUS checkout:

```python
import json

import pandas as pd

from proteus.utils.energy_ledger import whole_run_ledger

d = json.load(open('tests/data/energy_ledger/eb_updates.json'))
led = whole_run_ledger(pd.DataFrame(d['columns']), 'energy_balance')
L = led.terms['L_int'].sum()
print(f"{led.R_m.sum() / L:.4e} {led.R_c.sum() / L:.4e} {led.split('R_c')['remap'] / L:.4e}")
```

It prints `-1.6397e-06 5.9294e-03 5.9483e-03`: the mantle residual, the core residual and its part in the steps after an update, each over `L_int`. The returned `Ledger` holds every term and residual per step, the step classes and the mantle mass change; see [the API reference](../Reference/api/utils_energy_ledger.md).
