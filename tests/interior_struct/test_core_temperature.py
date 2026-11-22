"""Tests of the core temperature table that PROTEUS gives the Zalmoxis structure solve."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from proteus.interior_struct.core_temperature import CORE_T_POINTS, core_temperature_table

pytestmark = pytest.mark.unit

TAB = pathlib.Path(__file__).parents[1] / 'interior_energetics' / 'fixtures' / 'core_tabulated'


def _gaussian():
    """Gaussian core fitted to an Earth-like structure, with the config core alpha and c_p."""
    from aragog.core.profiles import GaussianCoreProfiles

    return GaussianCoreProfiles.from_structure(
        m_core=1.94e24, p_cen=3.6e11, r_cmb=3.48e6, p_cmb=1.36e11, alpha=1.35e-5, c_p=840.0
    )


@pytest.mark.physics_invariant
def test_gaussian_pairs_run_from_the_cmb_to_the_centre_on_the_adiabat():
    """The pairs start at (P_cmb, anchor) and end at the centre pressure and temperature of the
    profile; the slope of the last interval is the adiabat slope alpha T / (rho c_p) at the
    centre within 1e-3, so a linear extension past the centre follows the adiabat."""
    prof = _gaussian()
    p, t = core_temperature_table(prof, 5600.0)
    assert len(p) == len(t) == CORE_T_POINTS
    assert p[0] == pytest.approx(1.36e11, rel=1e-12)
    assert t[0] == pytest.approx(5600.0, rel=1e-12)
    assert p[-1] == pytest.approx(float(prof.pressure(0.0)), rel=1e-12)
    assert t[-1] == pytest.approx(float(prof.t_cen(5600.0)), rel=1e-12)
    slope = (t[-1] - t[-2]) / (p[-1] - p[-2])
    adiabat = 1.35e-5 * t[-1] / (float(prof.density(0.0)) * 840.0)
    assert slope == pytest.approx(adiabat, rel=1e-3)
    assert np.all(np.diff(t) > 0)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_gaussian_pairs_follow_an_independent_integration_of_the_adiabat():
    """Inside the core the pairs lie on the adiabat dT/dP = alpha T / (rho c_p), integrated in P
    from the CMB with solve_ivp on the density of the profile (rel 1e-6 at 5 interior pairs). The
    integration shares no code with the profile's own adiabat, so a pair of P and T from 2
    different radii, or a T with another alpha or c_p, falls outside."""
    from scipy.integrate import solve_ivp

    prof = _gaussian()
    p, t = core_temperature_table(prof, 5600.0)
    r = np.linspace(0.0, 3.48e6, 20001)
    p_r, rho_r = np.asarray(prof.pressure(r))[::-1], np.asarray(prof.density(r))[::-1]

    def rhs(pres, temp):
        return 1.35e-5 * temp / (np.interp(pres, p_r, rho_r) * 840.0)

    idx = [20, 60, 80, 120, 150]
    sol = solve_ivp(rhs, (p[0], p[-1]), [5600.0], t_eval=p[idx], rtol=1e-11, atol=1e-9)
    assert sol.success
    np.testing.assert_allclose(t[idx], sol.y[0], rtol=1e-6, atol=0.0)
    assert t[idx[-1]] / t[0] > 1.2  # the adiabat rises well above the anchor


@pytest.mark.physics_invariant
def test_tabulated_pairs_start_at_the_cmb_of_the_stored_core():
    """A PALEOS tabulated core from stored Zalmoxis core arrays (1 M_E, 2510 K): the pairs start at
    the CMB pressure of the arrays with the anchor and rise in P and T to the centre."""
    from aragog.core.module import build_core_module_budget, load_core_arrays

    a = load_core_arrays(TAB / 'zalmoxis_core_arrays_1me_2510K.npz')
    params = {
        'profile_source': 'paleos',
        'core_liquid_table': str(TAB / 'paleos_iron_liquid_cut_3tpa.npz'),
        'core_profile_file': str(TAB / 'zalmoxis_core_arrays_1me_2510K.npz'),
    }
    budget = build_core_module_budget(
        params,
        r_cmb=float(a['r'][-1]),
        p_cmb_fallback=float(a['P'][-1]),
        m_core=float(a['m'][-1]),
    )
    p, t = core_temperature_table(budget.profiles, 4200.0)
    assert p[0] == pytest.approx(float(a['P'][-1]), rel=1e-9)
    assert t[0] == pytest.approx(4200.0, rel=1e-9)
    assert p[-1] == pytest.approx(float(a['P'][0]), rel=1e-6)
    assert np.all(np.diff(t) > 0)


def test_anchors_outside_the_positive_reals_are_refused():
    """Hypothesis over anchors: bool, None, a string, NaN, inf, zero and negative numbers raise;
    a finite positive anchor scales the Gaussian pairs in T and leaves P unchanged."""
    hypothesis = pytest.importorskip('hypothesis')
    st = pytest.importorskip('hypothesis.strategies')
    prof = _gaussian()
    p_ref, t_ref = core_temperature_table(prof, 1000.0)
    bad = st.one_of(
        st.booleans(),
        st.none(),
        st.just('3000'),
        st.sampled_from([np.nan, np.inf, -np.inf, 0.0, -0.0]),
        st.floats(max_value=0.0, allow_nan=False),
    )

    @hypothesis.settings(max_examples=60, deadline=None)
    @hypothesis.given(bad)
    def _refused(x):
        with pytest.raises(ValueError, match='finite positive'):
            core_temperature_table(prof, x)

    @hypothesis.settings(max_examples=30, deadline=None)
    @hypothesis.given(st.floats(10.0, 1.0e5))
    def _scaled(x):
        p, t = core_temperature_table(prof, x)
        assert np.array_equal(p, p_ref)
        assert t == pytest.approx(t_ref * x / 1000.0, rel=1e-12)

    _refused()
    _scaled()


@pytest.mark.parametrize(
    ('pressure', 'adiabat'),
    [
        (lambda r: 1.0e11 + 0.0 * r, lambda r, t: t + 0.0 * r),  # flat P
        (lambda r: 1.0e11 + r, lambda r, t: t + 0.0 * r),  # P rising outward
        (lambda r: 1.0e11 - r, lambda r, t: np.where(r > 1.0e6, np.nan, t)),  # NaN T
        (lambda r: 1.0e11 - r, lambda r, t: t - 1.0e-3 * r),  # T <= 0 near the CMB
    ],
    ids=['flat P', 'P rising outward', 'NaN T', 'T not positive'],
)
def test_pairs_that_are_not_ascending_finite_and_positive_are_refused(pressure, adiabat):
    """A profile whose pairs are not strictly ascending in P, not finite or not positive in T
    raises instead of reaching the structure solve."""
    prof = SimpleNamespace(r_cmb=3.0e6, pressure=pressure, adiabat=adiabat)
    with pytest.raises(ValueError, match='ascending in P and positive in T'):
        core_temperature_table(prof, 3000.0)


def test_pairs_whose_temperature_falls_with_pressure_are_refused():
    """Hypothesis over one dip of size d at pair k of a table rising in T with P: the table is
    refused exactly when T falls with P somewhere (the Zalmoxis rule); a flat step passes."""
    hypothesis = pytest.importorskip('hypothesis')
    st = pytest.importorskip('hypothesis.strategies')
    r = np.linspace(0.0, 3.0e6, CORE_T_POINTS)
    rise = np.linspace(2000.0, 1000.0, CORE_T_POINTS)  # T falls outward: rises with P

    @hypothesis.settings(max_examples=80, deadline=None)
    @hypothesis.given(
        st.integers(0, CORE_T_POINTS - 1),
        st.one_of(st.just(0.0), st.floats(0.0, 50.0), st.just(1000.0 / (CORE_T_POINTS - 1))),
    )
    def _dip(k, d):
        shape = rise.copy()
        shape[k] -= d
        prof = SimpleNamespace(
            r_cmb=3.0e6,
            pressure=lambda x: 1.0e11 + 2.0e11 * (1.0 - x / 3.0e6),
            adiabat=lambda x, t: t / 1000.0 * np.interp(x, r, shape),
        )
        falls = np.any(np.diff(shape[::-1]) < 0)
        if falls:
            with pytest.raises(ValueError, match='falls with P'):
                core_temperature_table(prof, 1000.0)
        else:
            assert np.all(np.diff(core_temperature_table(prof, 1000.0)[1]) >= 0)

    _dip()
