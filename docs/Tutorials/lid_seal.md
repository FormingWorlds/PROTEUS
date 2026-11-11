# Volatiles held under a closed crust

This tutorial shows how a crust that closes over the magma ocean holds the volatiles of the melt, and how they then leak out down to a floor.
You run the same planet twice, once with the lid seal and once without it.
Then you read when the seal closed, check when each volatile starts to leak, and compare the 2 atmospheres.

When the seal closes, the volatiles dissolved in the melt move to the `{x}_kg_sealed` columns, and the chemistry stops.
While the seal holds, the held volatiles reach the atmosphere only by a leak with the relaxation time `tau_leak`, and by volcanism if `crust_rate` is above 0.
The leak stops at the floor `{x}_kg_eq`: the mass that melt and crystals dissolve at the pressure of the atmosphere.
[Volatiles held under a closed crust](../Explanations/lid_seal.md) explains the rule.

## Prerequisites

- PROTEUS installed with `bash install.sh` (see the [installation guide](../How-to/installation.md)), with the Aragog interior, and its environment active (`conda activate proteus`).
- `FWL_DATA` set; the first run downloads the interior data.
- `tomlkit` and `matplotlib`; both come with `pip install -e ".[develop]"`.

Run the commands from the PROTEUS folder. Run the Python blocks in one Python session, in the order of the steps.
Each run took about 20 minutes on a workstation.
The printed values come from runs with CALLIOPE 26.11.8.

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
closed at 97808 yr, crust 65.8 km, top-cell melt fraction 3.1e-04, mantle melt fraction 0.193
held: H2O 6.924e+18 kg, CO2 2.858e+17 kg
```

The seal closes when the top mesh cell freezes.
The top cell of this mesh is 65.8 km thick, so the crust jumps from 0 to 65.8 km in one step, and any `delta_close` up to 65.8 km closes the seal at the same step.
About a fifth of the mantle is still molten, and its dissolved water and CO$_2$ are now held under the crust.

## Step 5: when does each volatile start to leak?

At closure the held water is the water dissolved in the melt.
Its floor also counts the crystals, with the partition coefficient $D$ = `outgas.D_const_H2O` = 0.0017, so the held water starts below its floor.
It leaks only once the floor falls under it.
With the pressures fixed, that happens when the global melt fraction $\Phi$ falls below $(\Phi_c - D)/(1 - D)$, with $\Phi_c$ the value at closure:

```python
D = 0.0017
held = on.iloc[k:]
print(f"floor / held water on the first held step: {held['H2O_kg_eq'].iloc[1] / held['H2O_kg_sealed'].iloc[1]:.4f}")
for s in ('CO2', 'H2O'):
    leaks = held[f'{s}_kg_sealed'].diff() < 0
    first = held[leaks].iloc[0]
    print(f"{s} first leaks at {first['Time']:.0f} yr, Phi = {first['Phi_global']:.5f}")
print(f"(Phi_c - D) / (1 - D) = {(close['Phi_global'] - D) / (1 - D):.5f}")
leaks = held['H2O_kg_sealed'].diff() < 0
print(f"held / floor of water on the rows where it leaks: at least {(held['H2O_kg_sealed'] / held['H2O_kg_eq'])[leaks].min():.6f}")
```

The run printed:

```text
floor / held water on the first held step: 1.0061
CO2 first leaks at 97890 yr, Phi = 0.19231
H2O first leaks at 98470 yr, Phi = 0.19105
(Phi_c - D) / (1 - D) = 0.19113
held / floor of water on the rows where it leaks: at least 1.000127
```

At closure the floor of the water is $1 + D(1 - \Phi_c)/\Phi_c$ = 1.0071 times the held water.
On the first held step the ratio is already 1.0061, because $\Phi$ has fallen from 0.19250 to 0.19231, which shrinks the melt in the reservoir of the floor.
CO$_2$ has $D$ = 0, so its floor starts at the held mass and it leaks from the first held step.
Water does not leak on the first 7 held steps; it starts once $\Phi$ falls below 0.19113, about 660 yr after closure.
From then on the held water follows its floor down and stays above it.

## Step 6: compare the 2 atmospheres

```python
end_on, end_off = on.iloc[-1], off.iloc[-1]
print(f"H2O in the atmosphere at 140 kyr: seal {end_on['H2O_kg_atm']:.3e} kg, no seal {end_off['H2O_kg_atm']:.3e} kg")
print(f"surface pressure: seal {end_on['P_surf']:.3f} bar, no seal {end_off['P_surf']:.3f} bar")
print(f"held H2O: {end_on['H2O_kg_sealed']:.3e} kg, floor {end_on['H2O_kg_eq']:.3e} kg")
print(f"H2O in the melt without the seal: {end_off['H2O_kg_liquid']:.3e} kg")
print(f"mantle melt fraction without the seal: {end_off['Phi_global']:.3f}")
```

The run printed:

```text
H2O in the atmosphere at 140 kyr: seal 2.113e+15 kg, no seal 2.122e+15 kg
surface pressure: seal 1.037 bar, no seal 1.064 bar
held H2O: 6.923e+18 kg, floor 6.920e+18 kg
H2O in the melt without the seal: 6.923e+18 kg
mantle melt fraction without the seal: 0.136
```

Before the closing step the 2 runs agree to the last digit.
After it they part, but not by much.
The melt is still 13.6 % of the mantle at 140 kyr, and in both runs almost all the water stays in the mantle.
With the seal, the atmosphere holds 0.44 % less water than without the seal.
This small difference is the sum of 2 larger model terms of opposite sign, not a delay of the leak.
Replays of the held span of this run, with one term switched off at a time, give the terms in percentage points of the water in the atmosphere of the open run:

- the floor counts the water of the crystals formed after closure, with $D$ = 0.0017, which keeps 2.15 points more water in the mantle;
- the held gas keeps the species it had at closure, while the open run turns CO into CO$_2$ as the magma cools; the sealed atmosphere then holds less oxygen, the kept species lower its surface pressure by 1.7 %, and the water needs 1.86 points more mass in the atmosphere to reach the same partial pressure;
- the finite leak time leaves 0.12 points more water held, and the rest is 0.02 points;
- the closing temperature that the seal keeps for the CO$_2$ law changes nothing here: near 1 bar that law does not depend on temperature.

The path before closure is sensitive to round-off, so your run can close at another time and end with a different amount of water.
In 5 pairs of runs whose initial superheat differs by up to $2.5 \times 10^{-9}$ K, the water in the atmosphere at 140 kyr spreads by 0.64 % between the starts, and the run printed here ends 2.0 % below their mean.
Over these 5 pairs and the run printed here, the seal run holds between 0.428 % and 0.438 % less water than the open run of the same start; the printed run is at 0.438 %.
`tests/data/lid_seal/tutorial_ensemble.csv` holds these end values.

The block plots the water in the atmosphere, and the held water against its floor:

```python
import matplotlib.pyplot as plt

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
ax.plot(on['Time'] / 1e3, on['H2O_kg_atm'], label='seal')
ax.plot(off['Time'] / 1e3, off['H2O_kg_atm'], '--', label='no seal')
ax.set_ylabel('H$_2$O in the atmosphere [kg]')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
on_floor = held[held['H2O_kg_eq'] > 0]
ax2.plot(on_floor['Time'] / 1e3, on_floor['H2O_kg_sealed'] / on_floor['H2O_kg_eq'] - 1)
ax2.axhline(0.0, ls=':')
ax2.set_ylabel('held / floor - 1')
ax2.set_xlabel('Time [kyr]')
fig.tight_layout()
fig.savefig('output/tutorial_lid_seal/water.png', dpi=150)
```

## Step 7: try it yourself

- Set `tau_leak` to `1e2` and run again: the leak follows the floor more closely, and the water in the atmosphere at 140 kyr changes by less than 0.1 %, less than the spread of 0.64 % between starts that differ only by round-off.
- Set `crust_rate` above 0 to add the volcanic leak; the columns `{e}_kg_leak_volc` and `{e}_kg_leak_tau` show the 2 parts in each step, and `seal_volc_capped` shows the steps where the crust production was cut to the melt mass. Both parts stop at the same floor.
- Set `delta_close` to 0.3 m: on this mesh the seal closes at the same step, because a crust on the mesh is 0 or a whole cell. A thin crust closes the seal earlier only through the Aragog sub-grid lid.
