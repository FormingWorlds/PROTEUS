# Validation: the core temperature of the structure solve

## What this page checks

This page checks the core temperature that PROTEUS gives the Zalmoxis structure solve. PROTEUS builds a table of pressure and temperature pairs on the core adiabat, and Zalmoxis reads it at each core node. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. The page has 2 cases:

- The first case takes the right answer from 2 other ways to solve the same structure. It is a cross-check, not a published value.
- The second case checks that a resumed run builds the same table as the run before the stop.

If the table or its use were wrong, the core radius, the central pressure and the pressure at the core-mantle boundary (CMB) of every Zalmoxis run would be wrong. Each case says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case.

## Source under test

`src/proteus/interior_struct/core_temperature.py` (the table and its checks) and the call in `src/proteus/interior_struct/zalmoxis.py` that passes the table to Zalmoxis as `core_temperature_arrays`.

## The core table gives the structure of 2 independent paths

**Why this check exists.** The table changes the core density, so a fault in the table, in the call or in how Zalmoxis reads it moves the whole structure. Such a fault gives no error message: the solve still converges, at a wrong radius and pressure.

**What it tests.** The test starts from a stored state: the first structure update, at 102 yr, of a 1 Earth-mass planet with `core_bc = "energy_balance"`. It solves the structure 4 times with `zalmoxis_solver`:

1. with the core table on the JAX path, the default path;
2. with the same table on the numpy path, which uses another integrator and other temperature code;
3. without the table, with the core temperature given by radius and iterated 4 times on the core node pressures (a fixed point);
4. without the table, with the whole core at the temperature of the deepest mantle node.

Solves 2 and 3 must give the core radius $R_\mathrm{core}$, the planet radius $R_\mathrm{int}$, the central pressure $P_\mathrm{center}$ and the CMB pressure $P_\mathrm{cmb}$ of solve 1 within a relative 1e-5. The largest of the 4 differences of solve 4 must be more than 1e-2 (here $P_\mathrm{center}$, 4.6e-2), so the test can see a table that has no effect.

**How we know the right answer.** Solves 2 and 3 reach the same structure in 2 different ways. A fault in the JAX code of the table, or in the table read, moves solve 1 away from both.

**Why it matters.** With the table, the core radius is 0.74 % larger and the central pressure 4.4 % lower than with the core at the deepest mantle temperature (measured on this state). PROTEUS gives the table to every Zalmoxis solve that has a mantle temperature profile, which includes every structure update of a run.

![Temperature against radius in the core and the lower mantle for the 4 solves](../../assets/validation/structure_core_table_temperature.svg#only-light){ width="80%" }
![Temperature against radius in the core and the lower mantle for the 4 solves](../../assets/validation/structure_core_table_temperature_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the radius from the centre in km and the y-axis the temperature in K. The dotted line is solve 4: the whole core is at one temperature. The other 3 lines are solves 1, 2 and 3: the core temperature rises to the centre along the adiabat. A pass means that these 3 lines lie on each other.

![Upper limit 1e-5 for solves 2 and 3 and the measured difference of solve 4, for 4 structure values, on a log axis](../../assets/validation/structure_core_table_paths.svg#only-light){ width="80%" }
![Upper limit 1e-5 for solves 2 and 3 and the measured difference of solve 4, for 4 structure values, on a log axis](../../assets/validation/structure_core_table_paths_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis names 4 structure values. The y-axis is the relative difference to solve 1, on a log axis. Circles are solve 4, the core at the temperature of the deepest mantle node, rounded to 2 digits. The test asserts that the highest circle lies above 1e-2. Triangles stand for solves 2 and 3 and are drawn at the pass mark of 1e-5. They mark an upper limit, not a measured value: a pass means that the measured difference is below the triangle.

The measured differences of solves 2 and 3 come from 2 different numerical methods and change between Linux and macOS, so the stored plot shows the pass mark. The table gives the measured values.

| Value | Solve 2, numpy path | Solve 3, fixed point | Solve 4, deepest mantle temperature |
|---|---|---|---|
| $R_\mathrm{core}$ | 2.0e-8 | 3.0e-8 | 7.3e-3 |
| $R_\mathrm{int}$ | 6.8e-9 | 1.6e-7 | 2.2e-3 |
| $P_\mathrm{center}$ | 3.3e-8 | 5.5e-7 | 4.6e-2 |
| $P_\mathrm{cmb}$ | 4.8e-7 | 4.5e-8 | 1.8e-2 |

The values are the relative differences to solve 1, measured on macOS arm64. The percentages under **Why it matters** (0.74 % and 4.4 %) are relative to solve 4, so they differ a little from the solve 4 column, which is relative to solve 1. With `--junitxml=<file>`, as in the nightly run, the test also writes the 12 values to the JUnit XML report.

**Known limit.** The fixed point uses the same table values at the core node pressures, so a wrong table (a wrong adiabat or anchor) moves all 3 solves the same way. This test does not check the table values themselves. The checks in the next section do: one compares the Gaussian table with a separate integration of the adiabat, and the others check which core and which temperature each solve takes.

??? info "Test details"
    - Reference: cross-check against the numpy path and a fixed point without the table, on the stored state `tests/interior_struct/fixtures/structure_core_table/state_eb_102yr.npz`.
    - Tolerance: relative 1e-5 on each of $R_\mathrm{core}$, $R_\mathrm{int}$, $P_\mathrm{center}$ and $P_\mathrm{cmb}$ for solves 2 and 3; more than 1e-2 for the largest of the 4 differences of solve 4.
    - Test id: `tests/interior_struct/test_slow_structure_core_table.py::test_the_core_table_gives_the_structure_of_two_independent_paths`
    - Command: `pytest tests/interior_struct/test_slow_structure_core_table.py --validation-plots`; add `--junitxml=<file>` for the 12 values

## The core table is the same after a resume

**Why this check exists.** A resumed run must continue the physics of the stopped run. If the core table depended on a state that a resume loses or rounds, the first structure update after a resume would build another core than the stopped run.

**What it tests.** A run with `core_bc = "core_module"` stops at 250 yr and resumes. At the first solve of the core after the resume, before it advances, the test builds the 2 tables PROTEUS uses: the adiabat of the live core through its restored temperature, and the Gaussian fit to the restored row through the restored `T_cmb`. Both must equal, bit for bit, the tables of the stopped run at its last row. A change of the restored temperature by 1 unit in the last place must change each table, so the comparison can see a restore that is off by round-off.

**How we know the right answer.** PROTEUS keeps no state of the core table: each solve builds it from the row and the live core. With an exact restore, the 2 builds see the same inputs and give the same numbers.

**Why it matters.** A long run that the queue stops and resumes must give the same structure updates as one run.

**Known limit.** The resumed run is not the same trajectory as one run without a stop: the stop adds a row at 250 yr, and the time step starts small again after the resume. For information only, from a single sample at 450 yr (`core_module`, resumed against not stopped): $P_\mathrm{cmb}$ differs by 5.3e-4 with the table and by 6.5e-4 without it, $R_\mathrm{core}$ by 3.0e-5 and 8.1e-5. In this sample the difference is not larger with the table.

This case has no plot: it checks an exact identity of 2 tables.

??? info "Test details"
    - Reference: the tables of the stopped run at its last row (250 yr).
    - Tolerance: bit for bit; a 1-ulp change of the restored temperature must change the table.
    - Test id: `tests/integration/test_slow_core_table_resume.py::test_the_resumed_run_builds_the_core_table_of_the_stopped_run`
    - Command: `pytest tests/integration/test_slow_core_table_resume.py`

## What the table does not include

The table is the adiabat of a core with a constant expansion coefficient and heat capacity. No case on this page compares it with the isentrope of the PALEOS liquid iron table, which the structure solve and the core energy budget do not use. A single measurement gives its size. The run is `tests/integration/zalmoxis_aragog_calliope.toml` with `planet.temperature_mode = "liquidus_super"`, `core_bc = "energy_balance"`, `update_interval = 100.0` yr, `update_min_interval = 0.0` and a time step of at most 50 yr, at 1, 3 and 4 Earth masses. Its first structure update (102 yr) is solved once more with the isentrope through the same CMB temperature in place of the table.

| Planet mass | Centre temperature, table against isentrope | $R_\mathrm{int}$, isentrope against table | $R_\mathrm{core}$ | $P_\mathrm{center}$ |
|---|---|---|---|---|
| 1 Earth mass | -20 % | +0.18 % | +0.60 % | -2.8 % |
| 3 Earth masses | +30 % | -0.01 % | -0.03 % | +1.0 % |
| 4 Earth masses | +73 % | -0.10 % | -0.34 % | +2.8 % |

The same run with the table and a relative change of 1e-12 in the super-liquidus temperature offset of the start (`planet.delta_T_super`) gives differences of 1.3e-6 or less, so each value in the table lies far above the round-off of the run. The limits of the table, also after a giant impact, are in [Limits of the core temperature table](../../Explanations/model.md#limits-of-the-core-temperature-table).

## Other checks of the core table

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_struct/test_core_temperature.py::test_gaussian_pairs_follow_an_independent_integration_of_the_adiabat` | The adiabat equation $dT/dP = \alpha T / (\rho c_p)$, integrated numerically in pressure from the CMB with `solve_ivp` on the density of the Gaussian profile | 5 interior pairs of an Earth-like core at 5600 K, `rtol=1e-6` (measured 6.1e-12) | 2026-11-22 |
| `tests/interior_energetics/test_structure_core_table.py::test_energy_balance_fits_a_gaussian_core_to_the_row_through_t_cmb` | The table of `core_bc = "energy_balance"` is the Gaussian fit to the row through the row `T_cmb` | An Earth-like row at 4300 K, bit for bit | 2026-11-22 |
| `tests/interior_energetics/test_structure_core_table.py::test_a_live_aragog_core_sets_the_table_through_its_own_temperature` | With `core_bc = "core_module"` the table is the adiabat of the live core through its temperature | A live core at 5100 K, bit for bit | 2026-11-22 |
| `tests/interior_energetics/test_structure_core_table.py::test_a_re_solve_anchors_the_table_at_the_row_t_cmb_or_the_deepest_mantle_t` | A structure update takes the row `T_cmb`, else the deepest mantle temperature | Rows with `T_cmb` 4300 K, 0 and missing, `rel=1e-12` | 2026-11-22 |
| `tests/interior_energetics/test_wrapper_adiabat_ic.py::test_an_impact_solve_anchors_the_core_table_at_the_row_t_cmb` | A solve after a giant impact takes the row `T_cmb`, else the deepest mantle temperature; a start-up solve the deepest mantle temperature | Row `T_cmb` 5700 K, 0, NaN, negative; start-up | 2026-11-22 |

## Last verified
2026-11-22
