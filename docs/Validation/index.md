# Scientific validation

These pages track the tests that pin PROTEUS physics against published
benchmarks, analytical limits, and cross-implementation checks. Each
page covers one source file and records the reference cited, the test
ids that carry the `reference_pinned` marker, and the scope of the
comparison. A physics module's behavior is considered validated when
every source file with a public physical API has at least one pinned
test inventoried here.

A case with a plot shows the tested values against their reference in a light and a dark version. The test writes the plot with the same numbers it asserts, through the `validation_plot` fixture of proteus-mpl. This command writes all plots to `docs/assets/validation/`:

```console
pytest -m reference_pinned tests --validation-plots
```

Without `--validation-plots` the tests assert and write nothing. The command makes every plot again; commit only the pair of the case you changed, because the bytes match only for the same matplotlib and Pillow versions. The physics inside a coupled module, such as Zalmoxis, is validated in the documentation of that module.

## Pages by process

| Process | Source file | Page | Plot |
|---|---|---|---|
| Accretion | `accretion/dummy.py` | [Analytical accretion](accretion/dummy.md) | none yet |
| Accretion | `accretion/wrapper.py` | [Impact atmosphere-loss dispatch](accretion/wrapper.md) | none yet |
| Atmosphere climate | `atmos_clim/agni.py` | [Transparent black-body limit](atmos_clim/agni.md) | none yet |
| Interior structure | `interior_struct/zalmoxis.py` | [Liquidus-super IC anchor](interior_struct/zalmoxis.md) | none yet |
| Orbit | `orbit/orbit.py` | [Orbital evolution](orbit/orbit.md) | none yet |
| Orbit | `orbit/parameterized.py` | [Prescribed migration laws](orbit/parameterized.md) | [eccentricity decay](orbit/parameterized.md#plot-eccentricity-decay) |
| Orbit | `orbit/satellite.py` | [Satellite angular momentum](orbit/satellite.md) | none yet |
| Orbit | `orbit/wrapper.py` | [Orbit module wrapper](orbit/wrapper.md) | none yet |
| Orbit | `orbit/obliqua.py` | [Tidal Love-number lookup](orbit/obliqua.md) | none yet |
| Outgassing | `outgas/atmodeller.py` | [Species-to-element mass split](outgas/atmodeller.md) | none yet |
| Outgassing | `outgas/binodal.py` | [H2-MgSiO3 binodal](outgas/binodal.md) | none yet |
| Outgassing | `outgas/burial_ledger.py` | [Burial ledger for remelting](outgas/burial_ledger.md) | none yet |
| Outgassing | `outgas/compaction.py` | [Melt drainage across the freezing front](outgas/compaction.md) | [drainage velocity](outgas/compaction.md#plot-drainage-velocity) |
| Outgassing | `outgas/lavatmos.py` | [Rock-vapour molar masses](outgas/lavatmos.md) | none yet |
| Outgassing | `outgas/layer_drainage.py` | [Pore-melt drainage of the layer scheme](outgas/layer_drainage.md) | [single node](outgas/layer_drainage.md#plot-single-node), [kinematic wave](outgas/layer_drainage.md#plot-kinematic-wave) |
| Outgassing | `outgas/layer_state.py` | [Per-node state of the layer scheme](outgas/layer_state.md) | [matrix-limited node](outgas/layer_state.md#plot-matrix-limited-node) |
| Outgassing | `outgas/layer_trapping.py` | [Layer trapping step](outgas/layer_trapping.md) | [no-drainage bound](outgas/layer_trapping.md#plot-no-drainage-bound) |
| Outgassing | `outgas/trapping.py` | [Solid-phase volatile trapping](outgas/trapping.md) | none yet |
| Star | `star/dummy.py` | [Stellar luminosity and instellation](star/dummy.md) | none yet |
| Star | `star/wrapper.py` | [Stellar wrapper dispatch](star/wrapper.md) | none yet |

## Adding a page

Create `docs/Validation/<module>/<file>.md` when the first
`reference_pinned` test for that source file lands, add it to the
table above and to the `Validation` tab in the `nav` of `mkdocs.yml`, and cite
the benchmark (paper, figure, table), the analytical limit, or the
cross-implementation setup the test compares against.

## Adding a plot

Give the test the `validation_plot` fixture and call it after the asserts with a function that draws a new figure from the tested values, and a case name: `validation_plot(figure, '<case>')`. The fixture writes `<case>.svg` and `<case>_dark.svg` (`kind='map'` writes AVIF for maps), at most 60 kB per file. Commit both files and show them on the page with the `#only-light` and `#only-dark` suffixes, followed by the reference, the tolerance, the test id and the command.
