# Volatiles held under a closed crust

This tutorial shows how a crust that closes over the magma ocean holds the volatiles of the melt, and how they then leak out.
You run the same planet twice, once with the lid seal and once without it.
Then you read when the seal closed, check the leak against its exact solution, and compare the 2 atmospheres.

When the seal closes, the volatiles dissolved in the melt move to the `{x}_kg_sealed` columns, and the chemistry stops.
While the seal holds, the held volatiles reach the atmosphere only by a leak with the relaxation time `tau_leak`, and by volcanism if `crust_rate` is above 0.
[Volatiles held under a closed crust](../Explanations/outgas.md#volatiles-held-under-a-closed-crust) explains the rule.

## Prerequisites

- PROTEUS installed with `bash install.sh` (see the [installation guide](../How-to/installation.md)), with the Aragog interior, and its environment active (`conda activate proteus`).
- `FWL_DATA` set; the first run downloads the interior data.
- `tomlkit` and `matplotlib`; both come with `pip install -e ".[develop]"`.

Run the commands from the PROTEUS folder. Run the Python blocks in one Python session, in the order of the steps.
Each run took about 20 minutes on a workstation.

## Step 1: the configuration

The configuration is `input/tutorials/tutorial_lid_seal.toml`.
An Earth-mass planet at 1 AU cools from a magma ocean under a grey (dummy) atmosphere, with the Aragog interior and the CALLIOPE chemistry.
Its seal section is:

```toml
[outgas.lid_seal]
    enabled = true
    tau_leak = 1.0e5             # [yr]; delta_close stays at its default, 10 km
```

The seal closes when the crust is at least `delta_close` thick (default 10 km) and the top mesh cell holds less than `phi_close` of melt (default 0.3).
Here no sub-grid lid runs, so the crust is the top run of solid mesh cells.
The run stops at 140 kyr.

## Step 2: run the planet with the seal

```console
proteus start --offline -c input/tutorials/tutorial_lid_seal.toml
```

The output goes to `output/tutorial_lid_seal/`.

## Step 3: run the same planet without the seal

The block writes a copy of the configuration with the seal off and its own output folder:

```python
import tomlkit

cfg = tomlkit.parse(open('input/tutorials/tutorial_lid_seal.toml').read())
cfg['params']['out']['path'] = 'tutorial_lid_seal_off'
cfg['outgas']['lid_seal']['enabled'] = False
open('input/tutorials/tutorial_lid_seal_off.toml', 'w').write(tomlkit.dumps(cfg))
```

```console
proteus start --offline -c input/tutorials/tutorial_lid_seal_off.toml
```

## Step 4: when did the seal close?

```python
import numpy as np
import pandas as pd

on = pd.read_csv('output/tutorial_lid_seal/runtime_helpfile.csv', sep=r'\s+')
off = pd.read_csv('output/tutorial_lid_seal_off/runtime_helpfile.csv', sep=r'\s+')
k = int(np.flatnonzero(on['seal_active'].values > 0.5)[0])
close = on.iloc[k]
print(f"closed at {close['Time']:.0f} yr, crust {close['seal_delta'] / 1e3:.1f} km, "
      f"top-cell melt fraction {close['seal_phi_top']:.1e}, mantle melt fraction {close['Phi_global']:.3f}")
print(f"held: H2O {close['H2O_kg_sealed']:.3e} kg, CO2 {close['CO2_kg_sealed']:.3e} kg")
```

The run printed:

```text
closed at 96938 yr, crust 65.8 km, top-cell melt fraction 5.0e-05, mantle melt fraction 0.192
held: H2O 6.924e+18 kg, CO2 2.572e+17 kg
```

The seal closes when the top mesh cell freezes.
The top cell of this mesh is 65.8 km thick, so the crust jumps from 0 to 65.8 km in one step, and any `delta_close` up to 65.8 km closes the seal at the same step.
About a fifth of the mantle is still molten, and its dissolved water and CO$_2$ are now held under the crust.

## Step 5: check the leak against its exact solution

With `crust_rate = 0`, the held mass decays as $M(t) = M_c\, e^{-(t - t_c)/\tau}$, with $M_c$ the held mass at the closing time $t_c$ and $\tau$ = `tau_leak`:

```python
tau = 1.0e5
held = on.iloc[k:]
exact = close['H2O_kg_sealed'] * np.exp(-(held['Time'] - close['Time']) / tau)
print(f"largest relative error {np.max(np.abs(held['H2O_kg_sealed'] / exact - 1)):.1e}")
print(f"held at {on['Time'].iloc[-1]:.0f} yr: {on['H2O_kg_sealed'].iloc[-1] / close['H2O_kg_sealed']:.3f} of the closing value")
```

The run printed:

```text
largest relative error 1.6e-15
held at 140000 yr: 0.650 of the closing value
```

The code uses the exact solution over each step, so only round-off remains, whatever the step length.

## Step 6: compare the 2 atmospheres

```python
end_on, end_off = on.iloc[-1], off.iloc[-1]
print(f"H2O in the atmosphere at 140 kyr: seal {end_on['H2O_kg_atm']:.2e} kg, no seal {end_off['H2O_kg_atm']:.2e} kg")
print(f"surface pressure: seal {end_on['P_surf']:.3f} bar, no seal {end_off['P_surf']:.3f} bar")
print(f"H2O in the melt without the seal: {end_off['H2O_kg_liquid']:.2e} kg")
print(f"mantle melt fraction without the seal: {end_off['Phi_global']:.3f}")
```

The run printed:

```text
H2O in the atmosphere at 140 kyr: seal 2.41e+18 kg, no seal 2.23e+15 kg
surface pressure: seal 1.316 bar, no seal 1.035 bar
H2O in the melt without the seal: 6.92e+18 kg
mantle melt fraction without the seal: 0.135
```

Before the closing step the 2 runs agree to the last digit.
After it they part.
Without the seal, the chemistry keeps the water dissolved in the melt, because the melt is still 13.5 % of the mantle at 140 kyr and water is very soluble.
With the seal, the leak takes a fixed share of the held water per year, and it does not depend on the solubility, so 35 % of the held water reaches the atmosphere in 43 kyr.
`tau_leak` therefore sets how fast a sealed planet degasses; with the defaults, `tau_leak` infinite and `crust_rate = 0`, the held water stays under the crust.

The block plots the water in the atmosphere and under the crust:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(on['Time'] / 1e3, on['H2O_kg_atm'], label='atmosphere, seal')
ax.plot(off['Time'] / 1e3, off['H2O_kg_atm'], '--', label='atmosphere, no seal')
ax.plot(held['Time'] / 1e3, held['H2O_kg_sealed'], ':', label='held under the crust')
ax.set_yscale('log')
ax.set_xlabel('Time [kyr]')
ax.set_ylabel('H$_2$O mass [kg]')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()
fig.savefig('output/tutorial_lid_seal/water.png', dpi=150)
```

## Step 7: try it yourself

- Set `tau_leak` to `1e4` or `1e6` and run again: the share of the held water that reaches the atmosphere by 140 kyr changes with $1 - e^{-(t - t_c)/\tau}$.
- Set `crust_rate` above 0 to add the volcanic leak; the columns `{e}_kg_leak_volc` and `{e}_kg_leak_tau` show the 2 parts in each step, and `seal_volc_capped` shows the steps where the crust production was cut to the melt mass.
- Set `delta_close` to 0.3 m: on this mesh the seal closes at the same step, because a crust on the mesh is 0 or a whole cell. A thin crust closes the seal earlier only through the Aragog sub-grid lid.
