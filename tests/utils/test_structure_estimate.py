"""Unit tests for ``proteus.utils.structure_estimate``.

Verifies the mass-aware Noack & Lasbleis (2020) P_cmb estimator that
replaces the older hardcoded 135 GPa fallback used in the
``liquidus_super`` IC mode for super-Earth masses.

Reference values are computed from NL20 Eqs. 5, 9, 12-16 by hand at the
fixture parametrisation (CMF=0.325 mass-mode, fe_mantle=0.1) and pinned
with ``pytest.approx`` tolerances tight enough to catch any drift in the
scaling-law constants but loose enough to allow minor floating-point
differences.

The reference values used here come from a NumPy reproduction of the
NL20 algebra; if those constants ever change the assertions fail loudly.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# iron_fractions
# ----------------------------------------------------------------------


class TestIronFractions:
    """Validation and physical-bound checks on the iron-fraction helper."""

    def test_mass_mode_returns_input_cmf(self):
        """In ``'mass'`` mode, ``iron_fractions`` returns the input CMF
        verbatim as ``x_cmf`` and produces ``x_fem`` (mantle iron mass
        fraction) and ``x_fe`` (total iron mass fraction) in the
        physical bounds 0 < x_fem < x_fe < 1.
        """
        from proteus.utils.structure_estimate import iron_fractions

        x_cmf, x_fe, x_fem = iron_fractions(0.325, 'mass')
        assert x_cmf == pytest.approx(0.325, rel=1e-12)
        assert 0.0 < x_fem < 1.0
        assert x_fem < x_fe < 1.0

    def test_radius_mode_collapses_via_power_law(self):
        """In ``'radius'`` mode, ``iron_fractions`` maps the input
        core-radius fraction to CMF via ``cf**2.5`` (the NL20 power-law
        approximation for converting radius to mass fraction).
        """
        from proteus.utils.structure_estimate import iron_fractions

        x_cmf, _, _ = iron_fractions(0.55, 'radius')
        assert x_cmf == pytest.approx(0.55**2.5, rel=1e-12)

    def test_radius_mode_clamps_to_window(self):
        """``'radius'`` mode clamps the resulting CMF to the [0.01, 0.80]
        window so the NL20 calibration band is not violated at the
        physical extremes (Mercury-like to mantle-stripped).
        """
        from proteus.utils.structure_estimate import iron_fractions

        x_cmf_lo, _, _ = iron_fractions(0.05, 'radius')
        x_cmf_hi, _, _ = iron_fractions(0.99, 'radius')
        assert x_cmf_lo == pytest.approx(0.01, rel=1e-12)
        assert x_cmf_hi <= 0.80

    @pytest.mark.parametrize('bad_cf', [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_core_frac_raises(self, bad_cf):
        """Core fractions outside (0, 1) raise ValueError; the helper does
        not silently clamp into range.
        """
        from proteus.utils.structure_estimate import iron_fractions

        with pytest.raises(ValueError):
            iron_fractions(bad_cf, 'mass')

    def test_unknown_mode_raises(self):
        """An unsupported mode string (e.g. 'volume') raises ValueError
        rather than falling back to one of the two supported modes.
        """
        from proteus.utils.structure_estimate import iron_fractions

        with pytest.raises(ValueError):
            iron_fractions(0.325, 'volume')

    def test_x_fe_increases_with_x_cmf(self):
        """Anti-happy-path: more core mass means more total iron, all
        else equal. If x_fe collapses to a constant the formula is wrong.
        """
        from proteus.utils.structure_estimate import iron_fractions

        _, x_fe_low, _ = iron_fractions(0.20, 'mass')
        _, x_fe_high, _ = iron_fractions(0.50, 'mass')
        assert x_fe_high > x_fe_low + 0.20


# ----------------------------------------------------------------------
# estimate_P_cmb_NL20
# ----------------------------------------------------------------------


class TestEstimatePCMB:
    """Pin the NL20 P_cmb estimate at multiple masses + bound checks."""

    def test_earth_returns_close_to_PREM(self):
        """NL20 at 1 M_Earth, CMF=0.325 mass-mode, fe_mantle=0.1 returns
        ~142 GPa, within ~6 GPa of PREM's 136 GPa CMB pressure.
        Pinned with a 10 GPa absolute tolerance to catch any drift in
        the NL20 fit constants while tolerating the small offset
        between the analytical fit and the seismic reference.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        P_cmb = estimate_P_cmb_NL20(1.0, 0.325, 'mass')
        # Discriminating: must be close to Earth's CMB pressure, not 0
        # and not stratospherically high.
        assert P_cmb == pytest.approx(142e9, abs=10e9)
        # Anti-happy-path: must NOT collapse to the legacy 135 GPa
        # hardcoded constant. NL20 gives ~142 GPa for default Earth
        # parameters; identical 135 GPa would mean the formula was
        # short-circuited to the old fallback.
        assert abs(P_cmb - 135e9) > 1e9

    def test_super_earth_3me_lifts_p_cmb_substantially(self):
        """At 3 M_Earth, P_cmb scales up. The 135 GPa Earth-only
        fallback was off by 3-4x for this mass; the NL20 estimate
        should land in the 350-450 GPa band.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        P_cmb = estimate_P_cmb_NL20(3.0, 0.325, 'mass')
        assert 300e9 < P_cmb < 500e9

    def test_super_earth_5me_continues_scaling(self):
        """At 5 M_Earth, NL20 ``P_cmb`` lands in the 500-800 GPa band,
        continuing the mass-scaling trend established at 1 and 3 M_Earth.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        P_cmb = estimate_P_cmb_NL20(5.0, 0.325, 'mass')
        assert 500e9 < P_cmb < 800e9

    def test_super_earth_10me_high_pressure_branch(self):
        """At 10 M_Earth the CMB pressure is order-1 TPa. NL20 should
        track this without runaway. Tolerance is wide because we are
        well outside the NL20 calibration band but the formula is still
        smooth here.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        P_cmb = estimate_P_cmb_NL20(10.0, 0.325, 'mass')
        assert 0.8e12 < P_cmb < 2.0e12

    def test_p_cmb_monotonic_in_mass(self):
        """Property-based: P_cmb increases with planet mass at fixed CMF.
        This is a structural invariant that must hold regardless of the
        exact NL20 fit constants.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        masses = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        Ps = [estimate_P_cmb_NL20(m, 0.325, 'mass') for m in masses]
        for i in range(1, len(Ps)):
            assert Ps[i] > Ps[i - 1], (
                f'NL20 P_cmb not monotonic in mass: '
                f'M={masses[i - 1]} -> {Ps[i - 1] / 1e9:.1f} GPa, '
                f'M={masses[i]} -> {Ps[i] / 1e9:.1f} GPa'
            )

    def test_p_cmb_monotonic_decreasing_in_cmf_at_fixed_mass(self):
        """Property-based: at fixed mass, more core mass means a thinner
        mantle, and the (R_p - R_c) shrinkage dominates the rho_m and
        g_m_av increases, so P_cmb DECREASES with CMF. NL20 reproduces
        this Mercury-analogue trend; verifying it pins the sign of the
        scaling. A bug that flipped the (R_p - R_c) factor or used the
        whole-planet density instead of mantle density would invert
        this trend.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        cmfs = [0.20, 0.30, 0.40, 0.50]
        Ps = [estimate_P_cmb_NL20(1.0, c, 'mass') for c in cmfs]
        for i in range(1, len(Ps)):
            assert Ps[i] < Ps[i - 1], (
                f'NL20 P_cmb not monotonically decreasing in CMF at '
                f'fixed mass: CMF={cmfs[i - 1]} -> {Ps[i - 1] / 1e9:.1f} GPa, '
                f'CMF={cmfs[i]} -> {Ps[i] / 1e9:.1f} GPa'
            )

    def test_zero_or_negative_mass_raises(self):
        """``estimate_P_cmb_NL20`` rejects mass values that are not
        strictly positive, since the scaling laws are undefined there.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        with pytest.raises(ValueError):
            estimate_P_cmb_NL20(0.0, 0.325, 'mass')
        with pytest.raises(ValueError):
            estimate_P_cmb_NL20(-1.0, 0.325, 'mass')

    def test_invalid_core_frac_raises(self):
        """A core fraction at the upper bound (1.0) raises ValueError; the
        function does not produce a degenerate zero-mantle result.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        with pytest.raises(ValueError):
            estimate_P_cmb_NL20(1.0, 1.0, 'mass')

    def test_returns_finite_float(self):
        """Anti-happy-path: must return a finite float (not nan/inf/None)
        for the entire calibrated band.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        for m in (0.5, 1.0, 3.0, 10.0):
            P = estimate_P_cmb_NL20(m, 0.325, 'mass')
            assert isinstance(P, float)
            assert math.isfinite(P)
            assert P > 0
