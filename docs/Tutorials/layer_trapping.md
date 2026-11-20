# Volatiles trapped in the freezing mantle, node by node

This tutorial shows how the layer trapping scheme keeps part of the water, CO$_2$ and N$_2$ of the magma ocean in the mantle as it freezes.
You run the same planet 3 times: with the layer scheme, with the front scheme and without trapping.
Then you read how much of each element the mantle keeps, follow the melt the chemistry still sees, and compare the 3 atmospheres.

In the layer scheme every cell of the interior mesh, a node, keeps its own record.
A node that leaves the magma ocean enters the mush, a mix of crystals and melt, and its melt between the crystals, the pore melt, buries its volatiles.
That melt drains upward and returns its volatiles to the ocean, and what a node still holds when it freezes is locked in the solid.
The chemistry dissolves volatiles into the melt of the ocean alone, because the pore melt no longer mixes with it.
[Layer drainage node by node](../Explanations/outgas.md#layer-drainage-node-by-node) explains the rules.

## Prerequisites

- PROTEUS installed with `bash install.sh` (see the [installation guide](../How-to/installation.md)), with the Aragog interior, and its environment active (`conda activate proteus`).
- `FWL_DATA` set; the first run downloads the interior data.
- `tomlkit`, `pandas` and `matplotlib`; all come with `pip install -e ".[develop]"`.

Run the commands from the PROTEUS folder. Run the Python blocks in one Python session, in the order of the steps.

## Step 1: the configuration

The configuration is `input/tutorials/tutorial_layer_trapping.toml`.
It is the Earth analogue of [the Earth tutorial](earth_analogue.md) at 1.5 AU, with the Aragog interior, the CALLIOPE chemistry and energy-limited escape.
It has 2 changes that keep a run short and the result easy to read:

- a grey (dummy) atmosphere in place of AGNI, so a run takes minutes;
- 4e18 kg of nitrogen, a volatile that dissolves poorly in the melt.

Its trapping setting is:

```toml
[outgas]
    trap_mode = "layers"
```

The run stops when the global melt fraction, `Phi_global`, falls to 0.05.

With the grey atmosphere the planet loses heat at a rate set by its surface temperature alone, not by the mass or the make-up of its atmosphere.
So the 3 runs cool along the same path, and any difference between them comes from where the volatiles go.
With AGNI a thicker steam atmosphere slows the cooling, and the 3 runs then freeze at different times.

## Step 2: run the planet with the layer scheme

```console
proteus start --offline -c input/tutorials/tutorial_layer_trapping.toml
```

The output goes to `output/tutorial_layer_trapping/`.

## Step 3: run the same planet with the front scheme and without trapping

The block writes 2 copies of the configuration, each with its own output folder:

```python
import tomlkit

base = open('input/tutorials/tutorial_layer_trapping.toml').read()
for mode in ('front', 'none'):
    cfg = tomlkit.parse(base)
    cfg['params']['out']['path'] = f'tutorial_layer_trapping_{mode}'
    cfg['outgas']['trap_mode'] = mode
    open(f'input/tutorials/tutorial_layer_trapping_{mode}.toml', 'w').write(tomlkit.dumps(cfg))
```

```console
proteus start --offline -c input/tutorials/tutorial_layer_trapping_front.toml
proteus start --offline -c input/tutorials/tutorial_layer_trapping_none.toml
```

The run without trapping is the control: it shows what the atmosphere gets when the mantle keeps nothing.

## Step 4: how much does the mantle keep?

The columns `{e}_kg_trapped` hold the mass of element `{e}` kept in the mantle, and `{e}_kg_total` the mass of the whole planet:

```python
import pandas as pd

runs = {
    mode: pd.read_csv(f'output/{path}/runtime_helpfile.csv', sep=r'\s+')
    for mode, path in (
        ('layers', 'tutorial_layer_trapping'),
        ('front', 'tutorial_layer_trapping_front'),
        ('none', 'tutorial_layer_trapping_none'),
    )
}
for mode, df in runs.items():
    end = df.iloc[-1]
    shares = ', '.join(
        f"{e} {100 * end[f'{e}_kg_trapped'] / end[f'{e}_kg_total']:.1f} %" for e in ('H', 'C', 'N')
    )
    print(f"{mode:6s} t = {end['Time'] / 1e6:.2f} Myr, P_surf = {end['P_surf']:.0f} bar, kept: {shares}")
```

The run without trapping keeps nothing.
Hydrogen, carried mostly as water, dissolves well in the melt, so the mush buries much of it.
Carbon and nitrogen dissolve poorly: most of them are in the atmosphere before the mush forms, and the mantle keeps only a small share.

## Step 5: the melt the chemistry still sees

`trap_Phi_ocean` is the melt of the nodes still in the magma ocean, as a fraction of the mantle mass, and `trap_n_mush` counts the nodes in the mush:

```python
layers = runs['layers']
mush = layers[layers['trap_n_mush'] > 0]
for frac in (0.25, 0.5, 0.75):
    row = mush.iloc[int(frac * (len(mush) - 1))]
    print(f"t = {row['Time'] / 1e6:.2f} Myr: Phi_global {row['Phi_global']:.3f}, "
          f"ocean melt {row['trap_Phi_ocean']:.3f}, mush nodes {row['trap_n_mush']:.0f}")
```

Once the mush forms, the ocean melt falls faster than the global melt fraction, because more and more of the melt sits between the crystals of the mush.
The chemistry dissolves the volatiles into the ocean melt alone, so the water the shrinking ocean cannot hold goes to the atmosphere, or is buried by the nodes that leave the ocean.

## Step 6: compare the 3 atmospheres

The block plots the trapped share of hydrogen and the water in the atmosphere of each run:

```python
import matplotlib.pyplot as plt

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
for mode, df in runs.items():
    ax.plot(df['Time'] / 1e6, 100 * df['H_kg_trapped'] / df['H_kg_total'], label=mode)
    ax2.plot(df['Time'] / 1e6, df['H2O_bar'], label=mode)
ax.set_ylabel('H kept in the mantle [%]')
ax.legend()
ax2.set_ylabel('H$_2$O in the atmosphere [bar]')
ax2.set_xlabel('Time [Myr]')
fig.tight_layout()
fig.savefig('output/tutorial_layer_trapping/trapping.png', dpi=150)
```

The trapped share of the layer run can step down as well as up: melt that drains out of the mush returns its volatiles to the ocean, and `trap_kg_returned` holds that mass in each step.
The CO$_2$ and N$_2$ pressures of a run can fall while the mass of carbon and nitrogen in its atmosphere stays the same: the water that the ocean releases lowers the mean molar mass of the atmosphere, so the same mass of CO$_2$ is a smaller share of the surface pressure.

## Step 7: try it yourself

- Set `interior_energetics.grain_size` to `1e-2`: grains 10 times larger make a mush 100 times more permeable, so more melt drains and the mantle keeps less.
- Set `outgas.trap_phi_min` to `0.03`: nodes count as solid, and lock their pore melt, at a higher porosity.
- Set `outgas.trap_mush_log10visc` to `18`: a softer crystal framework compacts faster and pushes out more melt.
- Switch the atmosphere to AGNI, as in `input/tutorials/tutorial_earth.toml`: the run takes hours, and the 3 runs then freeze at different times, because the atmosphere they keep sets how fast they cool.
