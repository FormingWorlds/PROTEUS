# Scientific validation

These pages track the tests that pin PROTEUS physics against published
benchmarks, analytical limits, and cross-implementation checks. Each
page covers one source file and records the reference cited, the test
ids that carry the `reference_pinned` marker, and the scope of the
comparison. `reference_pinned` is a pytest marker for a test that compares
the code with a published value, an analytical limit or a second
implementation. A physics module's behavior is considered validated when
every source file with a public physical API has at least one pinned
test inventoried here.

A case with a plot shows the tested values against their reference in a light and a dark version. The test writes the plot with the same numbers it asserts, through `validation_plot`, a pytest fixture of the proteus-mpl package that draws and checks the plot of a test. This command writes all plots to `docs/assets/validation/`:

```console
pytest -m reference_pinned tests --validation-plots
```

Without `--validation-plots` the tests write nothing, but each plot test compares the data of its plot with the committed pair and fails when that pair is missing or stale. The command makes every plot again; commit only the pair of the case you changed, because the bytes match only for the same matplotlib and Pillow versions. The physics inside a coupled module, such as Zalmoxis, is validated in the documentation of that module.

## Pages by process

| Process | Source file | Page | Plot |
|---|---|---|---|
| Accretion | `accretion/dummy.py` | [Analytical accretion](accretion/dummy.md) | [collision speed](accretion/dummy.md#collision-speed-and-the-mutual-escape-speed) |
| Accretion | `accretion/wrapper.py` | [Impact atmosphere-loss dispatch](accretion/wrapper.md) | [Kegerreis loss law](accretion/wrapper.md#kegerreis-et-al-2020-loss-law) |
| Atmosphere climate | `atmos_clim/agni.py` | [Transparent black-body limit](atmos_clim/agni.md) | [grey-gas column](atmos_clim/agni.md#black-body-emission-of-a-grey-gas-transparent-column) |
| Interior energetics | `interior_energetics/remelt.py` | [Partial giant-impact re-melt](interior_energetics/remelt.md) | [melt front](interior_energetics/remelt.md#melt-front-against-the-closed-form) |
| Interior structure | `interior_struct/zalmoxis.py` | [Liquidus-super IC anchor](interior_struct/zalmoxis.md) | [MgSiO3 liquidus](interior_struct/zalmoxis.md#mgsio3-liquidus-at-135-gpa) |
| Orbit | `orbit/orbit.py` | [Orbital evolution](orbit/orbit.md) | [eccentricity rate](orbit/orbit.md#eccentricity-rate-of-sp0d) |
| Orbit | `orbit/parameterized.py` | [Prescribed migration laws](orbit/parameterized.md) | [eccentricity decay](orbit/parameterized.md#eccentricity-decay-of-high-eccentricity-migration) |
| Orbit | `orbit/satellite.py` | [Satellite angular momentum](orbit/satellite.md) | [Earth-Moon angular momentum](orbit/satellite.md#angular-momentum-of-the-earth-moon-system) |
| Orbit | `orbit/wrapper.py` | [Orbit module wrapper](orbit/wrapper.md) | [orbital period](orbit/wrapper.md#orbital-period-of-the-earth) |
| Orbit | `orbit/obliqua.py` | [Tidal Love-number lookup](orbit/obliqua.md) | [Love number](orbit/obliqua.md#love-number-at-a-negative-forcing-frequency) |
| Outgassing | `outgas/atmodeller.py` | [Species-to-element mass split](outgas/atmodeller.md) | [element mass fractions](outgas/atmodeller.md#element-mass-fractions-of-water) |
| Outgassing | `outgas/binodal.py` | [H2-MgSiO3 binodal](outgas/binodal.md) | [moles of H2](outgas/binodal.md#moles-of-h2-from-the-molecular-mass) |
| Outgassing | `outgas/burial_ledger.py` | [Burial ledger for remelting](outgas/burial_ledger.md) | [ledger against a fine grid](outgas/burial_ledger.md#the-ledger-against-a-fine-grid) |
| Outgassing | `outgas/compaction.py` | [Melt drainage across the freezing front](outgas/compaction.md) | [drainage velocity](outgas/compaction.md#drainage-velocity-in-each-regime) |
| Outgassing | `outgas/lavatmos.py` | [Rock-vapour molar masses](outgas/lavatmos.md) | [molar masses](outgas/lavatmos.md#molar-masses-from-fastchem-formulas) |
| Outgassing | `outgas/trapping.py` | [Solid-phase volatile trapping](outgas/trapping.md) | [interstitial-melt trapping](outgas/trapping.md#trapping-through-the-interstitial-melt) |
| Star | `star/dummy.py` | [Stellar luminosity and instellation](star/dummy.md) | [solar luminosity](star/dummy.md#luminosity-of-the-sun) |
| Star | `star/wrapper.py` | [Stellar wrapper dispatch](star/wrapper.md) | [equilibrium temperature](star/wrapper.md#equilibrium-temperature-of-an-earth-like-planet) |

## Adding a page

Create `docs/Validation/<module>/<file>.md` when the first
`reference_pinned` test for that source file lands, add it to the
table above and to the `Validation` tab in the `nav` of `mkdocs.yml`, and cite
the benchmark (paper, figure, table), the analytical limit, or the
cross-implementation setup the test compares against.

## Adding a plot

Give the test the `validation_plot` fixture and call it after the asserts with a function that draws a new figure from the tested values, and a case name: `validation_plot(figure, '<case>')`. The fixture writes `<case>.svg` and `<case>_dark.svg` (`kind='map'` writes AVIF for maps), at most 60 kB per file. Commit both files and show them on the page with the `#only-light` and `#only-dark` suffixes, followed by the reference, the tolerance, the test id and the command.
