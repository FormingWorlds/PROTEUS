# Validation: `src/proteus/outgas/atmodeller.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.outgas.atmodeller` against an analytical limit. The marker
is registered in `pyproject.toml`.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/outgas/test_atmodeller.py::test_water_split_matches_closed_form_mass_fractions` | IUPAC atomic masses (https://iupac.qmul.ac.uk/AtWt/); the per-element mass fraction of a molecule is an analytic identity | n/a (closed form) | Pins the species-to-element mass split for water: the oxygen mass fraction of H2O is `m_O / (2 m_H + m_O) = 0.88809` and the hydrogen fraction is the complement `0.11191`. A discrimination guard rules out a naive equal (0.5/0.5) split. Also pins mass conservation (the split sums back to the species mass) and that the helper writes the reservoir split (`{e}_kg_atm`) but not the escape-owned `{e}_kg_total`. |

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
