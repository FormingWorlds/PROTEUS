# fractionated_debit.py Validation

## What this page checks

This page checks `src/proteus/outgas/fractionated_debit.py`, which removes escaping elements from the gases of an atmosphere over a solidified mantle. Once the mantle has solidified, no outgassing step re-forms the gases, so escape has to take each element out of the gases that hold it. Escape that separates the elements (hydrogen leaves faster than oxygen) cannot shrink all gases by one factor. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the composition of the atmosphere drifts after solidification without an error message.

## Source under test
`src/proteus/outgas/fractionated_debit.py` (the per-element debit of the atmospheric gases, the gases formed by the atoms left over, and the graphite that unpaired carbon forms).

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_fractionated_debit::test_water_losing_hydrogen_leaves_its_oxygen_as_o2` | Analytical stoichiometry of water that loses hydrogen and no oxygen | The water and O2 masses after the loss of 5, 25, 50, and 90 % of the hydrogen, to `rel=1e-12`, with a check that the oxygen stays in the atmosphere. |

## Water that loses hydrogen leaves its oxygen as O2

**Why this check exists.** Under the escape dispatcher, hydrogen escapes faster than oxygen. A debit that shrinks all gases by one factor would take oxygen with the hydrogen and form no O2, and the atmosphere would keep the composition it had at solidification.

**What it tests.** An atmosphere of 10²⁰ kg of water loses a share $f$ of its hydrogen and no oxygen. The test checks the water and O2 masses after the loss, and that the oxygen mass of the atmosphere does not change.

**How we know the right answer.** Each water molecule that loses its 2 hydrogen atoms leaves 1 oxygen atom, and 2 oxygen atoms form 1 O2 molecule. With $n$ the moles of water, the water keeps $(1 - f)\,n$ moles and the O2 is $f n / 2$ moles.

**Why it matters.** Oxygen that builds up as hydrogen escapes is the main way escape changes the composition of a secondary atmosphere after the magma ocean has frozen.

**Known limit.** The debit applies stoichiometry, without chemistry. Leftover atoms form O2, H2, N2, and S2, and leftover carbon pairs with oxygen into CO and with hydrogen into CH4, or condenses as graphite. Equilibrium chemistry could prefer other gases, such as CO2 or H2O from the O2 and H2 that the debit forms. Rock vapours keep their masses.

![Water and O2 masses after the loss of a share of the hydrogen](../../assets/validation/outgas_fractionated_debit_water.svg#only-light){ width="80%" }
![Water and O2 masses after the loss of a share of the hydrogen](../../assets/validation/outgas_fractionated_debit_water_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the share of the hydrogen removed. The y-axis is a gas mass divided by the initial water mass. The solid line is the analytical water mass, the dashed line the analytical O2 mass. The circles and the hollow squares are the water and O2 masses of the debit. The dotted line is the atmosphere mass after a uniform debit of the same hydrogen mass, which keeps no O2. A pass means that the circles lie on the solid line and the squares on the dashed line.

??? info "Test details"
    - Reference: analytical stoichiometry of the hydrogen loss from water.
    - Tolerance: relative 1e-12 on the water and O2 masses.
    - Test id: `tests/outgas/test_fractionated_debit.py::test_water_losing_hydrogen_leaves_its_oxygen_as_o2`
    - Command: `pytest -m reference_pinned tests/outgas/test_fractionated_debit.py --validation-plots`
    - Last compared: 13 November 2026.
