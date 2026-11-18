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
