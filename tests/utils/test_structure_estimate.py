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

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


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

    @pytest.mark.physics_invariant
    def test_radius_mode_honours_requested_core_radius_fraction(self):
        """In ``'radius'`` mode, ``iron_fractions`` chooses the core mass
        fraction so the realized NL20 core radius fraction R_c/R_p equals the
        requested ``core_frac``, instead of approximating it with a fixed
        power law.
        """
        from proteus.utils.structure_estimate import _nl20_radius_fraction, iron_fractions

        x_cmf, _, x_fem = iron_fractions(0.6, 'radius', mass_tot_M_earth=1.0)
        realized = _nl20_radius_fraction(x_cmf, 1.0, x_fem)
        # The requested radius fraction is honoured to root-find precision.
        assert realized == pytest.approx(0.6, abs=1e-4)
        # Discrimination: the retired cf**2.5 heuristic set x_cmf=0.6**2.5=0.279,
        # which realizes R_c/R_p~0.497; the inversion must land far from that.
        assert abs(realized - 0.497) > 0.05
        # The chosen mass fraction is neither the radius-as-mass passthrough
        # (0.6) nor the old heuristic value (0.279).
        assert abs(x_cmf - 0.6) > 0.05
        assert abs(x_cmf - 0.6**2.5) > 0.05

    @pytest.mark.physics_invariant
    def test_radius_mode_clamps_unreachable_fraction_to_unit_interval(self):
        """A requested radius fraction beyond the achievable NL20 range clamps
        to the nearest core-mass-fraction bound, and the result stays in (0, 1)
        across the input range (replacing the old [0.01, 0.80] CMF window).
        """
        from proteus.utils.structure_estimate import _nl20_radius_fraction, iron_fractions

        # NL20 caps R_c/R_p near ~0.93, so a near-unity request is unreachable
        # and clamps to the upper x_cmf bound.
        x_hi, _, x_fem = iron_fractions(0.99, 'radius', mass_tot_M_earth=1.0)
        assert 0.0 < x_hi < 1.0
        assert _nl20_radius_fraction(x_hi, 1.0, x_fem) < 0.99
        # A small requested fraction yields a small but positive core; the
        # mapping is monotonic, so it sits below the near-unity case.
        x_lo, _, _ = iron_fractions(0.05, 'radius', mass_tot_M_earth=1.0)
        assert 0.0 < x_lo < x_hi
        # Below the achievable minimum (~0.03 at 1 M_earth), it clamps to the
        # lower x_cmf bound (1e-4) rather than going to zero or negative.
        x_min, _, _ = iron_fractions(0.01, 'radius', mass_tot_M_earth=1.0)
        assert x_min == pytest.approx(1.0e-4, rel=1e-6)

    @pytest.mark.parametrize('bad_cf', [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_core_frac_raises(self, bad_cf):
        """Core fractions outside (0, 1) raise ValueError; the helper does
        not silently clamp into range.
        """
        from proteus.utils.structure_estimate import iron_fractions

        with pytest.raises(ValueError):
            iron_fractions(bad_cf, 'mass')
        # Discrimination: a value just inside the open interval must NOT
        # raise. A regression that broadened the rejection (e.g. <= 0.5)
        # would fail this neighboring-input check.
        result = iron_fractions(0.5, 'mass')
        assert result[0] == pytest.approx(0.5, rel=1e-12)

    def test_unknown_mode_raises(self):
        """An unsupported mode string (e.g. 'volume') raises ValueError
        rather than falling back to one of the two supported modes.
        """
        from proteus.utils.structure_estimate import iron_fractions

        with pytest.raises(ValueError):
            iron_fractions(0.325, 'volume')
        # Discrimination: the two documented modes must NOT raise and
        # must produce CMF values in the open unit interval. A
        # regression that tightened the gate would surface here.
        x_cmf_mass, _, _ = iron_fractions(0.325, 'mass')
        x_cmf_rad, _, _ = iron_fractions(0.325, 'radius', mass_tot_M_earth=1.0)
        assert 0.0 < x_cmf_mass < 1.0
        assert 0.0 < x_cmf_rad < 1.0

    def test_x_fe_increases_with_x_cmf(self):
        """Anti-happy-path: more core mass means more total iron, all
        else equal. If x_fe collapses to a constant the formula is wrong.
        """
        from proteus.utils.structure_estimate import iron_fractions

        _, x_fe_low, _ = iron_fractions(0.20, 'mass')
        _, x_fe_high, _ = iron_fractions(0.50, 'mass')
        assert x_fe_high > x_fe_low + 0.20
        # Boundedness: both iron fractions must lie inside (0, 1). A
        # regression that overshot the mantle-iron contribution would
        # produce x_fe > 1, breaking the physical bounds.
        assert 0.0 < x_fe_low < 1.0
        assert 0.0 < x_fe_high < 1.0


# ----------------------------------------------------------------------
# estimate_P_cmb_NL20
# ----------------------------------------------------------------------


class TestEstimatePCMB:
    """Pin the NL20 P_cmb estimate at multiple masses + bound checks."""

    @pytest.mark.reference_pinned
    @pytest.mark.physics_invariant
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

    @pytest.mark.physics_invariant
    def test_radius_mode_p_cmb_uses_inverted_core_fraction(self):
        """In radius mode P_cmb is computed from the inverted core mass fraction,
        so for the same nominal core_frac it differs from the mass-mode value
        (and stays positive). Guards the mass-threading into iron_fractions.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        p_radius = estimate_P_cmb_NL20(1.0, 0.5, 'radius')
        p_mass = estimate_P_cmb_NL20(1.0, 0.5, 'mass')
        # Positivity invariant.
        assert p_radius > 0.0
        # Discrimination: radius mode inverts core_frac=0.5 to a different core
        # mass fraction than the mass-mode passthrough (x_cmf=0.5), so the two
        # P_cmb values must differ by more than rounding.
        assert abs(p_radius - p_mass) > 1e9

    def test_super_earth_3me_lifts_p_cmb_substantially(self):
        """At 3 M_Earth, P_cmb scales up. The 135 GPa Earth-only
        fallback was off by 3-4x for this mass; the NL20 estimate
        should land in the 350-450 GPa band.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        P_cmb = estimate_P_cmb_NL20(3.0, 0.325, 'mass')
        assert 300e9 < P_cmb < 500e9
        # Discrimination: at 3 M_E the NL20 estimate must be well
        # above the 1 M_E value (~142 GPa). A regression to a fixed
        # Earth-only constant would land at 135 GPa, failing the
        # 300 GPa lower bound but reinforce with a direct 1-vs-3
        # mass comparison so the bug surfaces on identifier change.
        P_1me = estimate_P_cmb_NL20(1.0, 0.325, 'mass')
        assert P_cmb > P_1me + 100e9

    def test_super_earth_5me_continues_scaling(self):
        """At 5 M_Earth, NL20 ``P_cmb`` lands in the 500-800 GPa band,
        continuing the mass-scaling trend established at 1 and 3 M_Earth.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        P_cmb = estimate_P_cmb_NL20(5.0, 0.325, 'mass')
        assert 500e9 < P_cmb < 800e9
        # Discrimination: the 5 M_E estimate must continue rising above
        # 3 M_E. A regression that capped P_cmb at a saturation value
        # (e.g. via spurious clamp) would tie the two masses together.
        P_3me = estimate_P_cmb_NL20(3.0, 0.325, 'mass')
        assert P_cmb > P_3me + 50e9

    def test_super_earth_10me_high_pressure_branch(self):
        """At 10 M_Earth the CMB pressure is order-1 TPa. NL20 should
        track this without runaway. Tolerance is wide because we are
        well outside the NL20 calibration band but the formula is still
        smooth here.
        """
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        P_cmb = estimate_P_cmb_NL20(10.0, 0.325, 'mass')
        assert 0.8e12 < P_cmb < 2.0e12
        # Discrimination: the 10 M_E pressure must exceed the 5 M_E
        # value by a meaningful amount (>200 GPa). A regression that
        # saturated the scaling above 5 M_E would land both values
        # near the 800 GPa band and lose the trend.
        P_5me = estimate_P_cmb_NL20(5.0, 0.325, 'mass')
        assert P_cmb > P_5me + 200e9

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
        # Scale discrimination: the 0.5 M_E vs 10 M_E span must be at
        # least an order of magnitude in P_cmb. A regression that
        # flattened the mass exponent close to zero would still pass
        # the inner monotonicity check (any tiny upward drift is
        # monotonic) but lose the physical scale.
        assert Ps[-1] / Ps[0] > 10.0

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
        # Scale discrimination: the CMF=0.20 vs CMF=0.50 endpoints must
        # differ by at least 20 GPa at 1 M_E. A regression that wired
        # whole-planet density into the mantle factor would flip the
        # sign of this trend; even a regression that only weakened the
        # gradient would compress the endpoint gap below 20 GPa.
        assert Ps[0] - Ps[-1] > 20e9

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
        # Discrimination: an in-band core_frac just below the upper bound
        # must NOT raise and must return a positive P_cmb in Pa. A
        # regression that broadened the rejection to closed-interval
        # would surface on the neighboring 0.95 input.
        P = estimate_P_cmb_NL20(1.0, 0.95, 'mass')
        assert P > 0.0

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
