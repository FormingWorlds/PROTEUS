# core_temperature.py Validation

## What this page checks

This page checks the core temperature that PROTEUS gives the Zalmoxis structure solve. PROTEUS builds a table of pressure and temperature pairs on the core adiabat, and Zalmoxis reads it at each core node. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. Here the right answer comes from 2 other ways to solve the same structure, so the case is a cross-check, not a published value. If the table or its use were wrong, the core radius, the central pressure and the pressure at the core-mantle boundary (CMB) of every Zalmoxis run would be wrong. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

## Source under test
`src/proteus/interior_struct/core_temperature.py` (the table and its checks) and the call in `src/proteus/interior_struct/zalmoxis.py` that passes the table to Zalmoxis as `core_temperature_arrays`.

## The core table gives the structure of 2 independent paths

**Why this check exists.** The table changes the core density, so a fault in the table, in the call or in how Zalmoxis reads it moves the whole structure. Such a fault gives no error message: the solve still converges, at a wrong radius and pressure.

**What it tests.** The test starts from a stored state: the first structure update, at 102 yr, of a 1 Earth-mass planet with `core_bc = "energy_balance"`. It solves the structure 4 times with `zalmoxis_solver`:

1. with the core table on the JAX path, the default path;
2. with the same table on the numpy path, which uses another integrator and other temperature code;
3. without the table, with the core temperature given by radius and iterated 4 times on the core node pressures (a fixed point);
4. without the table, with the whole core at the temperature of the deepest mantle node.

Solves 2 and 3 must give the core radius $R_\mathrm{core}$, the planet radius $R_\mathrm{int}$, the central pressure $P_\mathrm{center}$ and the CMB pressure $P_\mathrm{cmb}$ of solve 1 within a relative 1e-4. Solve 4 must differ from solve 1 by more than 1e-2, so the test can see a table that has no effect.

**How we know the right answer.** Solves 2 and 3 reach the same structure in 2 different ways. A fault in the JAX code of the table, or in the table read, moves solve 1 away from both.

**Why it matters.** With the table, the core radius is 0.74 % larger and the central pressure 4.4 % lower than with the core at the deepest mantle temperature (measured on this state). PROTEUS gives the table to every Zalmoxis solve that has a mantle temperature profile, which includes every structure update of a run.

**Known limit.** The fixed point uses the same table values at the core node pressures, so a wrong table (a wrong adiabat or anchor) moves all 3 solves the same way. The test does not check the table values themselves; `tests/interior_struct/test_core_temperature.py` checks them against the adiabat slope at the centre.

![Temperature against radius in the core and the lower mantle for the 4 solves](../../assets/validation/structure_core_table_temperature.svg#only-light){ width="80%" }
![Temperature against radius in the core and the lower mantle for the 4 solves](../../assets/validation/structure_core_table_temperature_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the radius from the centre and the y-axis the temperature. The dotted line is solve 4: the core is at one temperature. The other 3 lines are solves 1, 2 and 3: the core temperature rises to the centre along the adiabat. A pass means that these 3 lines lie on each other.

![Relative difference of 4 structure values between the solves, on a log axis](../../assets/validation/structure_core_table_paths.svg#only-light){ width="80%" }
![Relative difference of 4 structure values between the solves, on a log axis](../../assets/validation/structure_core_table_paths_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis names 4 structure values and the y-axis is the relative difference to solve 1, on a log axis. Circles are the numpy path (solve 2) and squares the fixed point (solve 3); both are asserted. The dashed line is the tolerance of 1e-4. A pass means that every mark lies below the line. Measured: at most 4.8e-7 for the numpy path and 5.5e-7 for the fixed point.

??? info "Test details"
    - Reference: cross-check against the numpy path and a fixed point without the table, on the stored state `tests/interior_struct/fixtures/structure_core_table/state_eb_102yr.npz`.
    - Tolerance: relative 1e-4 on $R_\mathrm{core}$, $R_\mathrm{int}$, $P_\mathrm{center}$ and $P_\mathrm{cmb}$; more than 1e-2 against the core at the deepest mantle temperature.
    - Test id: `tests/interior_struct/test_slow_structure_core_table.py::test_the_core_table_gives_the_structure_of_two_independent_paths`
    - Command: `pytest tests/interior_struct/test_slow_structure_core_table.py --validation-plots`

## Last verified
2026-11-22
