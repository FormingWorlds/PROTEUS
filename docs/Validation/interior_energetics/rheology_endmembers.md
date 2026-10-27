# Validation: `src/proteus/interior_energetics/rheology_endmembers.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the end members
of the crystal-melt mush (solid shear modulus and its Debye thermal term) against a
published formula and analytical limits. Last comparison: 2026-10-20.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/interior_energetics/test_rheology_endmembers.py::test_shear_law_returns_g0_at_reference_state` | Stixrude and Lithgow-Bertelloni (2005, Geophys. J. Int. 162, 610), eq. 33 | eq. 33 | At zero Eulerian strain and 300 K the shear modulus equals G0 for en, lp-cen, hp-cen, brg and ppv; at a strain of 0.02 the cold expression with its (1+2f)^(5/2) prefactor is reproduced and the (1+2f)^(3/2) variant is excluded. |
| `tests/interior_energetics/test_rheology_endmembers.py::test_debye_energy_high_and_low_temperature_limits` | Debye model: high-temperature expansion 3nRT(1 - 3x/8 + x^2/20) and low-temperature limit 3 pi^4 n R T^4 / (5 theta^3) | n/a (analytical limit) | The quadrature reproduces both limits to 1e-6; the classical 3nRT is excluded at T = theta/100. |

## Notes

The shear constants G0, G0', eta_S0, theta0, gamma0 and q0 are those of Stixrude and
Lithgow-Bertelloni (2011, Geophys. J. Int. 184, 1180, Table A1) for en, cen, hpcen, mgpv
and mppv (cen for the PALEOS lp-cen phase). The reference volumes and K0, K0' are those of
the PALEOS phases; at 1 bar and 300 K the PALEOS lp-cen density (3198.5 kg/m3) is within
0.3 % of M / V0. The adiabatic bulk modulus is checked against a synthetic table with an
analytic answer (`test_adiabatic_bulk_is_phase_k_and_not_isothermal`), which excludes both
the isothermal modulus and a difference taken across a polymorph boundary.
