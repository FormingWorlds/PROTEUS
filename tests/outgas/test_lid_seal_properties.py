"""Property tests of ``proteus.outgas.lid_seal`` over the input domain of the leak step.

Invariants checked: at a constant floor the step kernel gives the closed form
for any held mass, floor and step fraction; a sealed row resumes only with a
finite closing temperature above 0 K; under front trapping the floor reservoir
counts the crystals formed after the highest melt fraction since closure, for any
path of the melt fraction, and without trapping it is the melt for any partition
coefficient; tau_leak is accepted only with a finite rate 1/tau_leak.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('hypothesis')

from hypothesis import given, settings
from hypothesis import strategies as st

from proteus.config._outgas import LidSeal
from proteus.outgas.lid_seal import floor_leak, reservoirs, restore_seal_state

from .test_lid_seal import D_H2O, M_MANTLE, _config, _interior, _row, _sealed_row, _step

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
@given(
    mass=st.floats(1.0, 1.0e24),
    share=st.floats(0.0, 2.0),
    frac=st.floats(1.0e-12, 1.0),
)
def test_floor_leak_at_a_constant_floor_is_the_closed_form(mass, share, frac):
    """At a constant floor F >= 0 the leak is frac max(M - F, 0), and in [0, frac M]."""
    floor = share * mass
    leak = floor_leak(mass, frac, lambda _: floor)
    assert 0.0 <= leak <= frac * mass
    assert leak == pytest.approx(frac * max(mass - floor, 0.0), rel=1e-12, abs=1e-12 * mass)


@given(temp=st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.just(0.0)))
def test_restore_refuses_a_sealed_row_without_a_closing_temperature(temp):
    """A sealed last row resumes only with a finite seal_Tmagma_close above 0 K;
    an open row resumes whatever the column holds."""
    row = {**_sealed_row(), 'seal_active': 1.0, 'seal_Tmagma_close': temp}
    if np.isfinite(temp) and temp > 0.0:
        assert restore_seal_state(_config(), row) is True
    else:
        with pytest.raises(ValueError, match='seal_Tmagma_close'):
            restore_seal_state(_config(), row)
    assert restore_seal_state(_config(), {**_row(), 'seal_Tmagma_close': temp}) is False
    del row['seal_Tmagma_close']
    with pytest.raises(ValueError, match='seal_Tmagma_close'):
        restore_seal_state(_config(), row)


@given(
    phi_max=st.one_of(
        st.floats(allow_nan=True, allow_infinity=True), st.sampled_from([0.4, 1.0])
    )
)
def test_restore_refuses_a_sealed_row_without_its_highest_melt_fraction(phi_max):
    """Under front trapping a sealed last row resumes only with the highest melt
    fraction since closure between the closing melt fraction (0.4) and 1, so a
    zero filled for a missing column is refused; without trapping it is not read."""
    row = {**_sealed_row(), 'seal_active': 1.0, 'seal_phi_max': phi_max}
    front = _config()
    front.outgas.trap_mode = 'front'
    if np.isfinite(phi_max) and 0.4 <= phi_max <= 1.0:
        assert restore_seal_state(front, dict(row)) is True
    else:
        with pytest.raises(ValueError, match='seal_phi_max'):
            restore_seal_state(front, dict(row))
    assert restore_seal_state(_config(), dict(row)) is True
    assert restore_seal_state(front, {**_row(), 'seal_phi_max': phi_max}) is False


@pytest.mark.physics_invariant
@settings(max_examples=60, deadline=None)
@given(phi_c=st.floats(-1.0e-9, 1.0 + 1.0e-9), path=st.lists(st.floats(0.0, 1.0), max_size=5))
def test_the_front_reservoir_follows_the_highest_melt_fraction_since_closure(phi_c, path):
    """A seal closing at any melt fraction, also past either end by round-off, and
    held over any path of the melt fraction: seal_phi_max is the highest clipped
    melt fraction since closure and the front reservoir of H2O is M_mantle (Phi +
    D max(Phi_max - Phi, 0)), between the melt and the melt plus D times the solid."""
    row = {**_row(phi=phi_c), 'lid_weight': 1.0, 'lid_delta': 0.5, 'esc_rate_total': 0.0}
    kw = {'delta_close': 0.3, 'trap_mode': 'front', 'tau_leak': 1.0e3}
    assert _step(row, _interior([0.6, 0.2]), False, **kw)[0]
    row['fO2_shift_IW_derived'] = 2.0
    for phi in path:
        row['Phi_global'] = phi
        assert _step(row, _interior([0.6, 0.2]), True, **kw)[0]
    top = max([min(max(phi_c, 0.0), 1.0), *path])
    assert row['seal_phi_max'] == top
    phi = min(max(row['Phi_global'], 0.0), 1.0)
    front = _config()
    front.outgas.trap_mode = 'front'
    r_h2o = reservoirs(front, row)[0]['H2O']
    assert r_h2o == pytest.approx(M_MANTLE * (phi + D_H2O * (top - phi)), rel=1e-14, abs=1.0)
    assert phi * M_MANTLE <= r_h2o <= M_MANTLE * (phi + D_H2O * (1.0 - phi)) * (1 + 1e-15)


@pytest.mark.physics_invariant
@given(
    phi=st.floats(-1.0e-9, 1.0 + 1.0e-9),
    d=st.floats(0.0, 1.0e3),
    phi_max=st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True)),
)
def test_without_trapping_the_floor_reservoir_is_the_melt(phi, d, phi_max):
    """trap_mode = 'none': for any melt fraction, also past either end by
    round-off, any partition coefficient and any stored seal_phi_max (or none),
    every sealed species has the reservoir M_mantle Phi of the clipped Phi."""
    cfg = _config()
    for s in ('H2O', 'CO2', 'N2'):
        setattr(cfg.outgas, f'D_const_{s}', d)
    row = {'M_mantle': M_MANTLE, 'Phi_global': phi}
    if phi_max is not None:
        row['seal_phi_max'] = phi_max
    res, clipped, m_melt = reservoirs(cfg, row)
    assert clipped == min(max(phi, 0.0), 1.0) and m_melt == clipped * M_MANTLE
    assert set(res.values()) == {m_melt}


@given(tau=st.floats(min_value=0.0, exclude_min=True, allow_infinity=True))
def test_tau_leak_takes_a_positive_value_with_a_finite_rate(tau):
    """tau_leak > 0 is accepted when 1/tau_leak is finite (inf gives rate 0); a
    subnormal tau_leak, whose rate overflows, is refused."""
    if np.isfinite(1.0 / tau):
        assert LidSeal(tau_leak=tau).tau_leak == tau
    else:
        with pytest.raises(ValueError, match='tau_leak'):
            LidSeal(tau_leak=tau)
