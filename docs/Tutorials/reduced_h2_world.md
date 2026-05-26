# Reduced H2-rich world

This tutorial simulates a planet with a highly reduced mantle (fO$_2$ = IW$-$2),
producing a hydrogen-dominated atmosphere rather than the steam/CO$_2$
atmospheres typical of oxidised planets. Reduced conditions fundamentally
change the outgassing chemistry: H$_2$ and CO dominate over H$_2$O and
CO$_2$, the greenhouse effect is weaker, and the atmospheric escape
behaviour differs.

## Configuration

The config file is at `input/tutorials/tutorial_reduced.toml`:

```toml
config_version = "3.0"

[params.out]
    path = "tutorial_reduced"

[params.dt]
    initial = 1e2
    minimum = 1e3
    maximum = 1e7

    [params.stop.time]
        maximum = 5e9

[star]
    module  = "mors"
    mass    = 1.0
    age_ini = 0.1

[orbit]
    semimajoraxis = 1.0

[planet]
    mass_tot         = 1.0
    temperature_mode = "liquidus_super"
    delta_T_super    = 500.0
    volatile_mode    = "elements"
    [planet.elements]
        H_mode   = "oceans"
        H_budget = 2.0
        C_mode   = "C/H"
        C_budget = 1.0
        N_mode   = "N/H"
        N_budget = 0.5
        S_mode   = "S/H"
        S_budget = 1.0

[interior_struct]
    module    = "zalmoxis"
    core_frac = 0.325

[interior_energetics]
    module = "aragog"

[outgas]
    module       = "calliope"
    fO2_shift_IW = -2

[atmos_clim]
    module = "agni"

[escape]
    module = "zephyrus"
```

### Key differences from the Earth analogue

The only change is `fO2_shift_IW = -2` (reduced) instead of `+4` (oxidised).
This single parameter transforms the entire atmospheric evolution:

- **Oxidised (IW+4)**: outgassing produces H$_2$O, CO$_2$, N$_2$, SO$_2$
- **Reduced (IW$-$2)**: outgassing produces H$_2$, CO, CH$_4$, H$_2$S, NH$_3$

## What to expect

1. **H$_2$-dominated atmosphere**: At IW$-$2, most hydrogen outgasses as
   H$_2$ rather than H$_2$O. The atmosphere is lighter (lower mean
   molecular weight) and more extended.

2. **Weaker greenhouse**: H$_2$ is a poor greenhouse gas compared to
   H$_2$O and CO$_2$. The planet cools faster through the magma ocean
   phase. Compare `T_surf` evolution against the Earth analogue tutorial.

3. **Faster escape**: The lighter, more extended H$_2$ atmosphere has a
   larger XUV absorption cross-section and lower escape velocity at the
   exobase. The escape rate is typically higher than for an oxidised
   atmosphere with the same total hydrogen budget.

4. **Different carbon chemistry**: CO dominates over CO$_2$. Check
   `CO_vmr` and `CO2_vmr` in the helpfile to see the partitioning.

5. **Sulfur species**: H$_2$S instead of SO$_2$. The sulfur cycle is
   fundamentally different under reducing conditions.

## Comparison exercise

Run both this tutorial and the [Earth analogue](earth_analogue.md) tutorial,
then compare:

```python
import pandas as pd
import matplotlib.pyplot as plt

ox = pd.read_csv('output/tutorial_earth/runtime_helpfile.csv', sep='\t')
red = pd.read_csv('output/tutorial_reduced/runtime_helpfile.csv', sep='\t')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Temperature
axes[0,0].plot(ox['Time'], ox['T_magma'], label='IW+4')
axes[0,0].plot(red['Time'], red['T_magma'], label='IW-2')
axes[0,0].set(xlabel='Time [yr]', ylabel='T_magma [K]', xscale='log')
axes[0,0].legend()

# Atmospheric composition
axes[0,1].plot(ox['Time'], ox['H2O_bar'], label='H2O (IW+4)')
axes[0,1].plot(red['Time'], red['H2_bar'], label='H2 (IW-2)')
axes[0,1].set(xlabel='Time [yr]', ylabel='Partial pressure [bar]', xscale='log')
axes[0,1].legend()

# Surface pressure
axes[1,0].plot(ox['Time'], ox['P_surf'], label='IW+4')
axes[1,0].plot(red['Time'], red['P_surf'], label='IW-2')
axes[1,0].set(xlabel='Time [yr]', ylabel='P_surf [bar]', xscale='log')
axes[1,0].legend()

# Escape rate
axes[1,1].plot(ox['Time'], ox['esc_rate_total'], label='IW+4')
axes[1,1].plot(red['Time'], red['esc_rate_total'], label='IW-2')
axes[1,1].set(xlabel='Time [yr]', ylabel='Escape rate [kg/s]', xscale='log', yscale='log')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('redox_comparison.pdf')
```

## Next tutorials

- [Sub-Neptune](sub_neptune.md): boundary interior module with thick H/He envelope
- [Parameter grid sweep](parameter_grid.md): systematically explore fO$_2$ and H budget

---

**See also:** [Model description](../Explanations/model.md) | [Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)
