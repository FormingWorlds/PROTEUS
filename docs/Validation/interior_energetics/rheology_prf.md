# Validation: `src/proteus/interior_energetics/rheology_prf.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the elastic-moduli
profiles: the self-consistent-scheme mush moduli against analytical limits of the
Hashin-Shtrikman bounds, and the Kervazo et al. (2021) fits against their Table 3. Last
comparison: 2026-10-27.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/interior_energetics/test_rheology_prf.py::test_hs_shear_bounds_analytical_limits` | Hashin and Shtrikman (1963, J. Mech. Phys. Solids 11, 127); Voigt and Reuss averages (Hill 1952, Proc. Phys. Soc. A 65, 349) | n/a (analytical limit) | Both HS shear bounds equal the common modulus when the phases share it, and lie inside the Voigt and Reuss averages otherwise; the Reuss average is excluded as a stand-in for the lower bound. |
| `tests/interior_energetics/test_rheology_prf.py::test_bulk_viscosity_prudhomme_bird_form` | Prud'homme and Bird (1978, J. Non-Newtonian Fluid Mech. 3, 261), as quoted by Simpson, Spiegelman and Weinstein (2010, eq. 11b; arXiv:0903.0437) | eq. 11b | zeta = (4/3) eta (1 - v)/v: 4 eta at v = 0.25, against 5.33 eta for Taylor's (1954) dilute form; 0 for pure melt, finite in the solid, and below the empty-pore HS bound. |
| `tests/interior_energetics/test_rheology_prf.py::test_nur_modified_voigt_matches_paper_formula` | Nur, Mavko, Dvorkin and Galmudi (1998, The Leading Edge 17(3), 357-362; doi:10.1190/1.1437977), modified Voigt bound | p. 361 | K_eff = (1 - v/v_c) K_s + (v/v_c) K_c with K_c the Reuss modulus at critical porosity, Reuss beyond v_c, and (1 - v/v_c) G_s for a melt of vanishing rigidity; continuous at v_c; the plain Voigt average, which ignores the critical porosity, is 3 % higher at v = 0.15. |
| `tests/interior_energetics/test_rheology_prf.py::test_gassmann_of_nur_frame_reproduces_saturated_modified_voigt` | Nur et al. (1998), p. 361; Gassmann (1951) | p. 361 | Biot-Gassmann saturation of the dry frame K_d = (1 - v/v_c) K_s (alpha = v/v_c, volume porosity) equals the saturated modified Voigt curve to 1e-10, and alpha - v stays non-negative, so the tidal solver recovers the model from the drained, grain and liquid moduli. |
| `tests/interior_energetics/test_rheology_prf.py::test_undrained_bulk_modulus_limits_and_volume_porosity` | Gassmann (1951) | n/a (analytical limits) | The undrained modulus K_d + alpha^2/S is the grain modulus without melt, the Reuss average for an empty frame (the melt modulus at v = 1), lies between K_d and K_s, and stays K_s for a frame as stiff as the grain; the melt mass fraction in place of the volume fraction gives a modulus more than 1 % stiffer at v = 0.6. |
| `tests/interior_energetics/test_rheology_prf.py::test_kervazo_table3_critical_crystal_fraction` | Kervazo et al. (2021, A&A 650, A72), with the law of Costa, Caricchi and Bagdassarov (2009, G-cubed 10, Q03010) | Table 3; eqs. 1-3 | The fits equal the Costa form in the crystal fraction 1 - phi with the tabulated critical crystal fraction phi_* (0.569, 0.597, 0.712 for eta, mu, K), so the melt-fraction law uses phist = 1 - phi_*; pure melt returns the melt reference value. Reading phi_* as a melt fraction is excluded by at least a factor 2.6 at phi = 0.5. |

## Notes

Against the prototype construction at 25 GPa (MgSiO3, G_inf = 1e10 Pa, framework collapse
at crystals = 0.8 phi_m), the unrelaxed shear modulus agrees to within 3 % across the melt
fraction. The drained bulk modulus differs by up to 8 % at low melt fraction because the
solid end member is now taken at the layer's (P, T) from the PALEOS unified table instead
of the solid side of the Aragog P-S tables.

The Kervazo fits previously used phist = 0.4 for all three channels. With the Table 3
values the log-midpoint of the transition, in melt fraction, moves from 0.303 to 0.336 for
the viscosity, from 0.353 to 0.356 for the shear modulus and from 0.534 to 0.468 for the
bulk modulus; the solid values at phi = 0 become 1.0e19 Pa s (was 1.3e19), 59.7 GPa (65.9)
and 200 GPa (148).

## Notes on the Nur profile

Nur et al. (1998) scale the Voigt and Reuss bounds with an empirical critical porosity per
rock class (0.05 to 0.8, their Table 1) and relate it to the porosity of the random close
packing (0.36 for identical spheres). The scaled interpolation is not a rigorous bound: the
dry frame (1 - v/v_c) K_s exceeds the Hashin-Shtrikman upper bound for empty pores for v_c
above about 0.4 (by 3 % at 0.45 and 15 % at 0.6), so the frame is capped there. For a melt
with a finite shear modulus the suspension modulus is the Hashin-Shtrikman lower bound rather
than the Reuss average, which lies 22 to 32 % below it for the MgSiO3 mush at 25 GPa; for a
melt of vanishing rigidity both vanish and the paper's formula is recovered.
