# Scientific validation

These pages track the tests that pin PROTEUS physics against published
benchmarks, analytical limits, and cross-implementation checks. Each
page covers one source file and records the reference cited, the test
ids that carry the `reference_pinned` marker, and the scope of the
comparison. A physics module's behavior is considered validated when
every source file with a public physical API has at least one pinned
test inventoried here.

## Pages by module

| Module | Source file | Page |
|---|---|---|
| Interior structure | `interior_struct/zalmoxis.py` | [Liquidus-super IC anchor](interior_struct/zalmoxis.md) |
| Orbit | `orbit/orbit.py` | [Orbital evolution](orbit/orbit.md) |
| Orbit | `orbit/satellite.py` | [Satellite angular momentum](orbit/satellite.md) |
| Orbit | `orbit/wrapper.py` | [Orbit module wrapper](orbit/wrapper.md) |
| Outgassing | `outgas/binodal.py` | [H2-MgSiO3 binodal](outgas/binodal.md) |
| Star | `star/star.py` | [Stellar luminosity and instellation](star/star.md) |
| Star | `star/wrapper.py` | [Stellar wrapper dispatch](star/wrapper.md) |

## Adding a page

Create `docs/Validation/<module>/<file>.md` when the first
`reference_pinned` test for that source file lands, add it to the
table above and to the `Validation` section of `mkdocs.yml`, and cite
the benchmark (paper, figure, table), the analytical limit, or the
cross-implementation setup the test compares against.
