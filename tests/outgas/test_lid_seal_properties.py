"""Property tests of ``proteus.outgas.lid_seal`` over the input domain of the leak step.

Invariants checked: at a constant floor the step kernel gives the closed form
for any held mass, floor and step fraction; a sealed row resumes only with a
finite closing temperature above 0 K and, under front trapping, finite crystal
masses >= 0; under front trapping the floor reservoir counts the solid formed
since closure as trapping buries it, youngest solid remelting first, for any path
of the melt fraction and the trapped melt fraction, and without trapping it is
the melt for any partition coefficient; tau_leak is accepted only with a finite
rate 1/tau_leak.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('hypothesis')

from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from proteus.config._outgas import LidSeal
from proteus.outgas.lid_seal import (
    CRYSTAL_COLUMNS,
    add_crystals,
    floor_leak,
    reservoirs,
    restore_seal_state,
)

from .test_lid_seal import (
    D_H2O,
    M_MANTLE,
    _config,
    _front_held_steps,
    _row,
    _sealed_row,
)

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


MASS = st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.sampled_from([0.0, 1.0e22]))


@given(key=st.sampled_from(CRYSTAL_COLUMNS), mass=MASS)
def test_restore_refuses_a_sealed_row_without_its_crystal_masses(key, mass):
    """Under front trapping a sealed last row resumes only with finite crystal and
    trapped melt masses >= 0 since closure; without trapping they are not read."""
    row = {**_sealed_row(), 'seal_active': 1.0, key: mass}
    front = _config()
    front.outgas.trap_mode = 'front'
    if np.isfinite(mass) and mass >= 0.0:
        assert restore_seal_state(front, dict(row)) is True
    else:
        with pytest.raises(ValueError, match=key):
            restore_seal_state(front, dict(row))
    assert restore_seal_state(_config(), dict(row)) is True
    assert restore_seal_state(front, {**_row(), key: mass}) is False


@given(
    dm=st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.sampled_from([0.0, 1e22])),
    f_tl=st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.sampled_from([0.0, 1.0])),
)
def test_a_held_step_takes_its_crystals_only_from_a_readable_trapping_step(dm, f_tl):
    """A held freezing step adds (1 - F_tl) dM_RM crystals and F_tl dM_RM trapped
    melt for a finite dM_RM >= 0 and F_tl in [0, 1]; a value that is not finite or
    an F_tl outside [0, 1] stops the step."""
    readable = np.isfinite(dm) and 0.0 <= f_tl <= 1.0
    assume(not (readable and dm < 0.0))  # a remelt reads the completed rows instead
    row = {**_sealed_row(), 'trap_dM_RM': dm, 'trap_F_tl': f_tl}
    if not readable:
        with pytest.raises(ValueError, match='trap_dM_RM'):
            add_crystals(row, None)
        return
    add_crystals(row, None)
    assert (row['seal_xtal_kg'], row['seal_tl_kg']) == ((1.0 - f_tl) * dm, f_tl * dm)


def _stack(phi_c, path):
    """Crystals and trapped melt left since closure [M_mantle], youngest solid melting first."""
    layers, prev = [], phi_c
    for phi, f_tl in path:
        if phi < prev:
            layers.append([phi, prev, f_tl])
        while phi > prev and layers and layers[-1][0] < phi:
            if layers[-1][1] <= phi:
                layers.pop()
            else:
                layers[-1][0] = phi
        prev = phi
    return sum((h - lo) * (1 - f) for lo, h, f in layers), sum(
        (h - lo) * f for lo, h, f in layers
    )


@pytest.mark.physics_invariant
@settings(max_examples=60, deadline=None)
@example(phi_c=1.0 + 1.0e-9, path=[(0.0, 1.0)])
@given(
    phi_c=st.floats(-1.0e-9, 1.0 + 1.0e-9),
    path=st.lists(st.tuples(st.floats(0.0, 1.0), st.floats(0.0, 1.0)), max_size=5),
)
def test_the_front_reservoir_counts_the_solid_since_closure_as_trapping_buries_it(phi_c, path):
    """A seal closing at any melt fraction, also past either end by round-off, and
    held over any path of (Phi, F_tl): the crystals and trapped melt left are those
    of a layer stack in which the youngest solid melts first, and the front
    reservoir of H2O is M_melt + D M_xtal + M_tl, between the melt and the mantle
    (past it by the round-off of a closing Phi above 1, which trapping differences)."""
    row, res = _front_held_steps(path, phi0=phi_c)
    xtal, tl = _stack(phi_c, path)
    assert row['seal_xtal_kg'] == pytest.approx(xtal * M_MANTLE, rel=1e-12, abs=1.0e12)
    assert row['seal_tl_kg'] == pytest.approx(tl * M_MANTLE, rel=1e-12, abs=1.0e12)
    m_melt = min(max(row['Phi_global'], 0.0), 1.0) * M_MANTLE
    assert res['H2O'] == m_melt + D_H2O * row['seal_xtal_kg'] + row['seal_tl_kg']
    assert m_melt <= res['H2O'] <= M_MANTLE * (max(phi_c, 1.0) + 1.0e-12)  # dM_RM is unclipped


@pytest.mark.physics_invariant
@given(
    phi=st.floats(-1.0e-9, 1.0 + 1.0e-9),
    d=st.floats(0.0, 1.0e3),
    mass=st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True)),
)
def test_without_trapping_the_floor_reservoir_is_the_melt(phi, d, mass):
    """trap_mode = 'none': for any melt fraction, also past either end by
    round-off, any partition coefficient and any stored crystal masses (or none),
    every sealed species has the reservoir M_mantle Phi of the clipped Phi."""
    cfg = _config()
    for s in ('H2O', 'CO2', 'N2'):
        setattr(cfg.outgas, f'D_const_{s}', d)
    row = {'M_mantle': M_MANTLE, 'Phi_global': phi}
    if mass is not None:
        row['seal_xtal_kg'] = row['seal_tl_kg'] = mass
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
