# Unified redox module (#57)

The `proteus.redox` subpackage is a coupled, self-consistent redox
accounting layer that tracks the **Redox Budget** `R` across three
reservoirs (atmosphere, mantle, core), debits escape, and solves a
transactional oxygen fugacity (`ΔIW`) each step so that Evans-style
electron conservation holds end-to-end.

It was introduced on the `tl/interior-refactor` branch (Commits A-E,
2026-04-22) and is exposed as three user modes. The default `static`
mode reproduces pre-#57 PROTEUS behaviour exactly.


## The three modes

### `static` (default)

```toml
[redox]
mode = "static"
```

`fO2_shift_IW` is pinned to `config.outgas.fO2_shift_IW` every step,
as before. The redox module still runs but only writes passive
diagnostics (`R_budget_atm`, `R_budget_mantle`, `R_budget_core`,
`R_budget_total`, `Fe3_frac`) into the helpfile. No solver runs, no
escape debit, no per-cell Fe persistence.

Use this when:

- Reproducing CHILI or earlier benchmark results.
- Running a comparison against a reference trajectory where `fO2`
  must stay constant.
- Debugging a regression: `static` is the null mode and should be
  bit-identical to the pre-#57 baseline modulo the new columns.

### `fO2_init`

```toml
[redox]
mode = "fO2_init"
```

At `t = 0`, `fO2_shift_IW` is seeded from `config.outgas.fO2_shift_IW`.
After the first few init loops, the transactional Brent solver runs
each step to close the budget:

```text
R_atm(t) + R_mantle(t) + R_core(t) = R_total(t-dt) + ΔR_escape - ΔR_dispro
```

Under the solver, the atmosphere's `fO2_shift_IW` is the free variable.
The solver probes via deep-copied `hf_row` calls into `run_outgassing`
(CALLIOPE or atmodeller) and converges within `redox.rtol` / `redox.atol`.

Use this when:

- You want the atmosphere's oxidation state to evolve self-consistently
  as escape removes H (oxidising) or C (reducing) from the inventory.
- The planet's initial composition is already specified as an `fO2`,
  not as a mantle composition.

### `composition`

```toml
[redox]
mode = "composition"
```

At `t = 0`, `fO2_shift_IW` is **derived** from
`config.interior_struct.mantle_comp` via the chosen oxybarometer
(`schaefer2024` by default). Subsequent steps run the same solver as
`fO2_init`.

The one-shot seed calls `redox.coupling.seed_composition_fO2`, which:

1. Reads `MantleComp` (9 BSE oxides + `Fe3_frac`).
2. Evaluates the oxybarometer at the Aragog mesh surface
   `(P_stag[-1], T_stag[-1], phi_max)`.
3. Converts `log10 fO2` to `ΔIW` via
   `log10_fO2_IW(T_surf, P_surf)`.
4. Writes `hf_row['fO2_shift_IW']`.

On NaN (missing `MantleComp`, oxybarometer failure), falls back to
`config.outgas.fO2_shift_IW` and logs a warning.

Use this when:

- You know the mantle composition (pyrolite, CI chondritic, etc.) but
  not the initial `fO2`.
- You want to test how mantle-redox-state choice propagates into
  atmospheric chemistry.


## Redox budget mechanics

Under the global reference states (Evans mantle-like, extended):

| Element | Reference oxidation state |
|---------|---------------------------|
| Fe | +2 |
| O | -2 |
| C | 0 |
| S | -2 |
| H | +1 |
| N | 0 |
| Si | +4 |
| Mg | +2 |
| Cr | +3 |

The signed RB per atom is `v_i = z_sample - z_ref`; per molecule it
sums over atoms. Key examples:

| Species | RB |
|---------|----|
| H2O | 0 |
| H2 | -2 |
| O2 | +4 |
| CO2 | +4 |
| CO | +2 |
| CH4 | -4 |
| N2 | 0 |
| NH3 | -3 |
| S2 | +4 |
| SO2 | +6 |
| H2S | 0 |
| SiO | +2 |
| SiO2 | 0 |
| MgO | 0 |
| FeO | 0 |
| Fe2O3 | +2 |
| Fe0 (metal) | -2 |
| FeS2 | +2 |

The full machinery is in `proteus.redox.budget`. Unknown species are
warn-and-skipped; species in `DEFERRED_SPECIES` (SO3, H2SO4, OCS, SO,
HCN, N2O, NO, NO2) are excluded because neither CALLIOPE nor
atmodeller emit them today.


## Buffers and oxybarometers

`proteus.redox.buffers` provides:

- **Buffers** (pressure in bar, Frost 1991 convention): `IW`, `QFM`,
  `NNO`.
- **Oxybarometers**:
  - `schaefer2024` (MO-active, `phi_max > phi_crit`): silicate-melt
    oxybarometer from Schaefer+24 Eq 13, implemented via Mariana's
    PDF Eq 25 (5-oxide coupling; natural log form; P in bar).
  - `stagno2013_peridotite` (MO-inactive fallback): garnet peridotite
    oxybarometer anchored at P=3 GPa, T=1573 K, Fe2O3=0.05 wt%.
  - `hirschmann2022`, `sossi2020`: registered placeholders for
    literature-trace coefficients; not wired to full calibrations.

`log10_fO2_mantle` is the dispatcher: on `phi_max > phi_crit` it
routes to the MO-active oxybarometer; otherwise to the peridotite
fallback.


## Stub physics (issues #653, #526, #432)

Three live physics hooks are stubbed as documented no-ops so the
redox module can ship without blocking downstream work:

- **#653 Mariana Fe3+/Fe2+/Fe0 evolution**
  (`partitioning.advance_fe_reservoirs`) — stub-bulk; treats the
  whole mantle as one cell, does not freeze or thaw Fe between melt
  and solid, returns NaN warm-start for fully-reduced mantles.
  Mariana's handover checklist is plan v6 §6.1.
- **#526 metal-silicate partitioning**
  (`partitioning.apply_metal_silicate_partitioning`) — stub zeros;
  no H/O/Si/C/S/N leak into the core.
- **#432 EOS density corrections**
  (`partitioning.eos_density_correction`) — identity factors; no
  consumer in PROTEUS today, API locked for future use.

`IMPLEMENTATION_STATUS` in `redox/__init__.py` surfaces these as a
registry. `proteus doctor` prints the current status line for
visibility.


## Regression tier

The redox module is backed by 103 unit tests and a live A/B
regression script against the CHILI Earth baseline.

### Unit tests

```bash
conda activate proteus-redox
cd /path/to/PROTEUS-redox
pytest tests/redox tests/test_redox_scaffolding.py -q
```

Covers: budget arithmetic (14 tests), buffer and oxybarometer values
(13 tests), stub contracts (6 tests), conservation invariants
(8 tests), solver transactional isolation (6 tests), NetCDF
persistence (5 tests), static-mode contract (5 tests), O-first-class
mass conservation (5 tests), modern-Earth anchors (6 tests),
plus scaffolding and config-parity tests.

### Live A/B static-mode regression

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Baseline: tl/interior-refactor @ 2401e6d2
conda activate proteus
cd /path/to/PROTEUS
proteus start --offline -c input/chili/chili_aragog_redox_ab_200iter.toml

# Branch: tl/redox-scaffolding with redox.mode='static'
conda activate proteus-redox
cd /path/to/PROTEUS-redox
proteus start --offline -c input/chili/chili_aragog_redox_ab_200iter.toml

# Diff
python scripts/regression_static_mode.py \
    /path/to/PROTEUS/output/chili_ab_200iter \
    /path/to/PROTEUS-redox/output/chili_ab_200iter_branch
```

`OPENBLAS_NUM_THREADS=1` is non-negotiable; without it the 1e-8
relative tolerance is not achievable due to BLAS reduction-order
non-determinism.

The script asserts:

- Row counts match.
- `fO2_shift_IW` is bit-identical.
- Non-O shared columns agree within 1e-8 relative.
- Branch-only columns are in the allow-list.


## Main-loop integration

Plan v6 §3.1 order:

```text
1. Interior solve (Aragog or SPIDER)         # writes NetCDF snapshot
2. Escape                                    # hf_row['esc_kg_*']
3. REDOX section (#57, when mode != static):
     a. populate_core_composition            # Fe_kg_core from CoreComp × M_core
     b. _write_passive_diagnostics           # R_budget_* bootstrap
     c. debit_escape                         # R_atm -= Σ RB * n_escaped
     d. advance_fe_reservoirs                # Mariana stub populates
                                             #   interior_o.redox_per_cell
                                             #   for next iter's NetCDF write
     e. solve_fO2                            # transactional Brent on ΔIW
     f. R_budget_total recomputed inline     # R_total = R_atm+R_mantle+R_core
                                             #   (NOT via _write_passive_diagnostics
                                             #    to avoid overwriting the
                                             #    debited R_atm — round-6 fix)
     g. assert_redox_conserved               # warn on soft_tol violation
4. Outgassing                                # picks up the pinned fO2
5. Atmosphere                                # AGNI / JANUS
```

Redox runs AFTER the interior solve, so the NetCDF snapshot at iter N
carries the per-cell Fe / fO2 state computed at iter N-1. This is the
same "snapshot is IC of the next step" pattern as entropy_s / T_stag.
