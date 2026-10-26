# Validation: `src/proteus/outgas/atmodeller.py`

## What this page checks

This page checks how the atmodeller wrapper, `proteus.outgas.atmodeller`, splits the mass of each gas species into the masses of its elements. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this split is wrong, the element budgets of H, C, N, S and O in each reservoir are wrong. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case. The `reference_pinned` marker is registered in `pyproject.toml`.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/outgas/test_atmodeller.py::test_water_split_matches_closed_form_mass_fractions` | IUPAC atomic masses (https://iupac.qmul.ac.uk/AtWt/); the per-element mass fraction of a molecule is an analytic identity | n/a (closed form) | Pins the species-to-element mass split for water: the oxygen mass fraction of H2O is `m_O / (2 m_H + m_O) = 0.88809` and the hydrogen fraction is the complement `0.11191`. A discrimination guard rules out a naive equal (0.5/0.5) split. Also pins mass conservation (the split sums back to the species mass) and that the helper writes the reservoir split (`{e}_kg_atm`) but not the escape-owned `{e}_kg_total`. |

## Element mass fractions of water

**Why this check exists.** Each species mass must go to its elements by mass fraction. An equal split between the elements, or a split that loses mass, gives wrong element budgets without an error message.

**What it tests.** For a molecule, the mass fraction of element $e$ is $f_e = n_e m_e / \sum_i n_i m_i$, with $n_i$ the number of atoms and $m_i$ the atomic masses. For water the oxygen fraction is 0.88809 and the hydrogen fraction 0.11191. The test splits a mass of water with `_populate_volatile_element_reservoirs` and checks:

- the oxygen and hydrogen masses against these fractions;
- that the hydrogen mass is more than 0.3 of the water mass away from an equal 0.5 split;
- that the 2 element masses sum back to the water mass;
- that the helper writes `{e}_kg_atm` but not `{e}_kg_total`, which the escape step owns.

**How we know the right answer.** The mass fraction of an element in a molecule is a closed-form identity; with the IUPAC atomic masses it gives 0.88809 for oxygen in water.

**Why it matters.** The atmodeller wrapper calls `_populate_volatile_element_reservoirs` after each solve to fill the element reservoirs.

![Mass fractions of O and H after the split of a water inventory](../../assets/validation/outgas_atmodeller_water_split.svg#only-light){ width="80%" }
![Mass fractions of O and H after the split of a water inventory](../../assets/validation/outgas_atmodeller_water_split_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The bars are the asserted O and H masses that `_populate_volatile_element_reservoirs` gives, divided by the 4e21 kg of water. The hollow squares are the closed-form mass fractions 0.88809 and 0.11191, asserted to relative 1e-4. The dashed line is the equal split, which the test rejects. A pass means that each square sits on the top of its bar.

??? info "Test details"
    - Reference: IUPAC atomic masses (https://iupac.qmul.ac.uk/AtWt/); the per-element mass fraction of a molecule (closed form).
    - Tolerance: relative 1e-4 on the fractions 0.88809 and 0.11191; relative 1e-12 on the sum.
    - Test id: `tests/outgas/test_atmodeller.py::test_water_split_matches_closed_form_mass_fractions`
    - Command: `pytest -m reference_pinned tests/outgas/test_atmodeller.py --validation-plots`

## Re-derivation note

For a molecule with `n_e` atoms of element `e` and constituent atomic masses
`m_i`, the element's mass fraction is

```
f_e = n_e * m_e / sum_i (n_i * m_i)
```

For H2O (`m_H = 1.008e-3`, `m_O = 15.999e-3` kg/mol):

```
f_O = 15.999 / (2 * 1.008 + 15.999) = 15.999 / 18.015 = 0.88809
f_H = 1 - f_O = 0.11191
```

`_populate_volatile_element_reservoirs` applies this per species and per
reservoir, so the per-element reservoir masses conserve the total volatile mass
exactly (independent of the absolute accuracy of the atomic-mass table, because
the same table sets both numerator and denominator).

## Element-total ownership note

The helper writes only the per-element reservoir split
(`{e}_kg_atm` / `_kg_liquid` / `_kg_solid`). It does not write `{e}_kg_total`,
which is owned by the escape step (the running-budget debit) for H/C/N/S and by
the authoritative `O_kg_total` write for oxygen, matching CALLIOPE and
`outgas.common.expected_keys`. This keeps escape's debited total from being
overwritten by a species reconstruction each iteration.

## Atmospheric mass note

For each gas species, atmodeller's output gives the conserved constraint
`total_mass`, the melt share `dissolved_mass` and the solver's own gas-phase
reservoir `gas_mass`. The wrapper writes `gas_mass` to `{sp}_kg_atm`, as the
authoritative atmospheric mass, and `dissolved_mass`, limited to
`[0, total_mass]`, to `{sp}_kg_liquid`.

`total_mass - dissolved_mass` equals `gas_mass` only while no gas species holds
a solid-phase reservoir. That holds today: condensates are separate condensed
species and never enter a gas species' `total_mass`. The difference therefore
serves as a closure check, logged as a warning when it differs from `gas_mass`
by more than 1 kg and a relative 1e-6, and as the atmospheric mass when the
output omits `gas_mass`.

Either write replaces the mass first derived from the partial pressure, which
distributes the column by mole fraction and so misweights a species whose molar
mass differs from the mean, badly for a light noble gas.
