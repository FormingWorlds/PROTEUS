# Lid seal

A crust that closes over the magma ocean can hold the volatiles dissolved in the melt; [Volatiles held under a closed crust](../Explanations/lid_seal.md) explains the model and the [tutorial](../Tutorials/lid_seal.md) runs it. This page explains how to switch the seal on, what it needs, and which output columns to read. The settings are listed with their units and defaults in the [config reference](../Reference/config/escape_outgas.md#lid-seal-outgaslid_seal).

## Switch it on

The seal needs the Aragog interior solver, because it reads the melt fraction and the radius of each mesh cell:

```toml
[interior_energetics]
module = "aragog"

[outgas.lid_seal]
enabled = true
delta_close = 1e4     # crust thickness at which the seal closes [m]
tau_leak = 1e5        # relaxation time of the held volatiles [yr]
```

The seal needs `outgas.module = "calliope"`, an escape reservoir of `"outgas"` or `"bulk"`, and, with the Zalmoxis structure, `interior_struct.zalmoxis.dry_mantle = true`. It cannot be combined with `params.stop.solid.freeze_volatiles = true`, which also freezes the melt inventory, or with `outgas.vapourise = true`.

## Read the output

These output columns show what the seal does ([full list](../Reference/output.md)):

- `seal_active` \[1\]: 1 while the seal holds, else 0.
- `seal_delta` \[m\] and `seal_phi_top` \[1\]: the crust thickness and the melt fraction of the top mesh cell that the seal reads in the step. On the mesh, `seal_delta` is the depth of the top run of cells below `phi_solid` while the seal is open, and below `phi_open` while it holds and on the step where it opens, so it can jump on the first held step.
- `seal_path` \[1\]: the crust that a closed seal reads in the step: 1 for the crust on the mesh, 2 for the sub-grid lid; 0 while the seal is open, also when `seal_delta` comes from the sub-grid lid.
- `{sp}_kg_sealed`, `{sp}_mol_sealed` and `{e}_kg_sealed`: the held mass of each species and element.
- `{sp}_kg_eq` \[kg\]: the floor of each soluble species at the end of the step; 0 while the seal is open, on the step where it closes, with `outgas.calliope.solubility = false`, and on held steps with no leak rate (`tau_leak` infinite and no crust production) or nothing held.
- `seal_t_close` \[yr\]: the time the seal last closed.
- `seal_Tmagma_close` \[K\]: the magma temperature of the chemistry solve that closed the seal, at least `outgas.T_floor`.
- `seal_phi_close` \[1\]: the global melt fraction of the step that closed the seal, clipped to \[0, 1\].
- `seal_phi_max` \[1\]: the highest global melt fraction since the seal closed, clipped to \[0, 1\]; the reservoir of the floor reads it under front trapping.
- `seal_crust_rate` \[kg yr$^{-1}$\] and `seal_volc_capped` \[1\]: the rate at which melt reaches the crust in the step, and whether `crust_rate dt` was cut to the melt mass.
- `{e}_kg_leak_tau` and `{e}_kg_leak_volc` \[kg\]: the mass of each element that leaked in the step by relaxation and by volcanism.
- `seal_leak_kg_cum` \[kg\]: the leaked mass since the start of the run.
- `seal_open_kg` \[kg\]: the held mass that returned to the melt in a step where the seal opened.
