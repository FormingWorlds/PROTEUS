"""Open part of the Aragog lid surface: solidus floor, start value and in-step response.

The open part of the surface (weight ``1 - w``) sits on melt held at least at the lid base,
``T_m = max(T_node, T_b)``. Its start value ``(F_atm - w L) / (1 - w)``, with the signed lid
flux ``L`` of the call, closes the weighted sum to ``F_atm``; inside the step it follows ``T_m``
with ``G_eff = G F' / (F' + w G_l' + (1 - w) G)`` per open area.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip('hypothesis')
from aragog.surface_lid import open_conductance  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proteus.atmos_clim.common import (  # noqa: E402
    LID_OPEN_AREA_FLOOR,
    LID_RESID_WARN,
    LID_SLOPE_GUARD_SIGN,
    LID_SLOPE_GUARD_SIZE,
    lid_acts,
    lid_boundary,
    lid_open_fields,
    lid_open_flux,
    lid_open_residual,
)
from proteus.utils.constants import const_sigma as SIGMA  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _row(w, G=4.0, T_b=1400.0, T_node=1500.0, **kw):
    return dict(
        lid_weight=w,
        lid_conductance=G,
        lid_T_base=T_b,
        lid_T_node=T_node,
        T_magma=1450.0,
        boundary_layer_thickness=kw.pop('d', 0.01),
        boundary_layer_conductivity=kw.pop('k', 2.0),
        **kw,
    )


def _g_eff(row, slope, T_other=None):
    """Response ``G_eff`` per open area and the guard code along the run path: the fields of
    ``lid_open_fields`` passed to Aragog's ``open_conductance``; ``(None, 0)`` without a live
    lid."""
    if not lid_acts(row):
        return None, 0
    f, code = lid_open_fields(row, slope, T_other)
    G_eff = open_conductance(
        float(row['lid_weight']),
        f['open_surface_slope'],
        f['open_skin_conductance'],
        f['open_lid_conductance'],
    )
    return G_eff, code


def _balanced(row, gap=1.0):
    d, k, T_melt = lid_boundary(row)
    row['F_atm'] = gap * k / d * (T_melt - row['T_surf'])
    return row


@pytest.mark.physics_invariant
def test_the_call_holds_the_open_node_at_the_lid_base():
    """With the node below the lid base the call sees the open part at ``T_b``."""
    row = _row(0.3, T_node=1350.0, T_surf=900.0)
    d, k, T_melt = lid_boundary(row)
    assert T_melt == pytest.approx(1400.0, rel=1e-14)
    above = _row(0.3, T_node=1500.0, T_surf=900.0)
    G = 0.3 * 4.0 + 0.7 * 200.0
    assert lid_boundary(above)[2] == pytest.approx(
        (1.2 * 1400.0 + 140.0 * 1500.0) / G, rel=1e-14
    )


@pytest.mark.physics_invariant
def test_the_start_value_takes_the_floored_skin():
    """At the balance of the call the start value is the skin flux ``G (T_m - T_s)`` of the
    floored node."""
    row = _balanced(_row(0.3, T_node=1350.0, T_surf=900.0))
    assert lid_open_flux(row) == pytest.approx(200.0 * (1400.0 - 900.0), rel=1e-14)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('T_node', [1350.0, 1500.0])
@pytest.mark.parametrize('w', [1e-12, 1e-11])
def test_the_floored_start_value_goes_to_f_atm_at_the_onset(w, T_node):
    """The start value tends to ``F_atm`` as the weight goes to 0, floored or not."""
    row = _balanced(_row(w, T_node=T_node, T_surf=1300.0), gap=0.97)
    assert lid_open_flux(row) == pytest.approx(row['F_atm'], rel=1e-10)


def _lid_flux(row):
    """Signed flux of the lid part in the call, ``min(G_lid, G_skin) (T_b - T_surf)``."""
    G_skin = row['boundary_layer_conductivity'] / row['boundary_layer_thickness']
    return min(row['lid_conductance'], G_skin) * (row['lid_T_base'] - row['T_surf'])


@pytest.mark.physics_invariant
@given(
    w=st.floats(1e-6, 1.0 - 4.0 * LID_OPEN_AREA_FLOOR),
    G_lid=st.floats(1e-3, 1e5),
    d=st.floats(1e-4, 1.0),
    k=st.floats(0.1, 10.0),
    T_node=st.floats(300.0, 4000.0),
    T_b=st.floats(300.0, 4000.0),
    T_surf=st.floats(200.0, 4500.0),
    F_atm=st.floats(-1e5, 1e5),
)
@settings(max_examples=400, deadline=None)
def test_the_weighted_parts_sum_to_f_atm_on_both_sides_of_the_lid_base(
    w, G_lid, d, k, T_node, T_b, T_surf, F_atm
):
    """The lid part at its signed call flux ``L`` and the open part sum to ``F_atm`` for any sign
    of ``F_atm`` and of ``T_b - T_surf``; relative 1e-12 of the larger term."""
    row = _row(w, G=G_lid, T_b=T_b, T_node=T_node, d=d, k=k, T_surf=T_surf, F_atm=F_atm)
    L = _lid_flux(row)
    F_open = lid_open_flux(row)
    assert math.isfinite(F_open)
    assert w * L + (1.0 - w) * F_open == pytest.approx(
        F_atm, rel=0.0, abs=1e-12 * (abs(F_atm) + abs(w * L)) + 1e-300
    )


@pytest.mark.physics_invariant
@given(
    w=st.floats(1e-6, 1.0 - 4.0 * LID_OPEN_AREA_FLOOR),
    T_node=st.floats(300.0, 4000.0),
    T_b=st.floats(300.0, 4000.0),
    T_surf=st.floats(200.0, 4500.0),
    gap=st.floats(-50.0, 50.0),
)
@settings(max_examples=200, deadline=None)
def test_the_residual_is_the_start_value_less_the_skin_flux(w, T_node, T_b, T_surf, gap):
    """The residual ``F_atm - B`` per total area, divided by ``1 - w``, is the start value less
    the skin flux ``G_skin (T_m - T_surf)`` of the floored node; relative 1e-9 of the larger
    term."""
    row = _row(w, T_b=T_b, T_node=T_node, T_surf=T_surf)
    d, k, T_melt = lid_boundary(row)
    B = k / d * (T_melt - T_surf)
    row['F_atm'] = B + gap
    S = 200.0 * (max(T_node, T_b) - T_surf)
    resid = lid_open_residual(row)
    # B is a difference of terms of size G T: its round-off sets the absolute tolerance
    tol = 1e-12 * 200.0 * max(T_node, T_b, T_surf)
    assert resid == pytest.approx(gap, rel=1e-9, abs=tol)
    per_open = resid / (1.0 - w)
    assert lid_open_flux(row) - S == pytest.approx(
        per_open, rel=1e-9, abs=tol / (1.0 - w) + 1e-12 * abs(S)
    )


def test_the_residual_is_0_without_a_lid():
    """Without a lid the call has no skin balance of its own: the residual is 0."""
    assert lid_open_residual(_row(0.0, T_surf=900.0, F_atm=-3.0)) == 0.0
    assert lid_open_residual({'F_atm': 5.0}) == 0.0


@pytest.mark.parametrize('w', [1.0 - 1e-10, 1.0 - 0.5 * LID_OPEN_AREA_FLOOR, 0.5])
def test_the_residual_per_total_area_does_not_grow_near_a_full_lid(w):
    """The residual is ``F_atm - B`` per total area: a call 0.01 W m-2 off its skin balance
    records 0.01 at any lid weight, also next to a full lid, where a share per open area would
    reach 0.01 / (1 - w)."""
    row = _row(w, T_surf=900.0)
    d, k, T_melt = lid_boundary(row)
    row['F_atm'] = k / d * (T_melt - 900.0) + 0.01
    assert lid_open_residual(row) == pytest.approx(0.01, abs=1e-8)
    assert abs(lid_open_residual(row)) < LID_RESID_WARN


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('w', 'T_surf', 'F_atm'), [(0.99, 1450.0, -0.91), (0.5, 1500.0, 7.0), (0.97, 1400.01, -1.0)]
)
def test_a_surface_above_the_lid_base_takes_the_sum_with_a_signed_lid_flux(
    w, T_surf, F_atm, caplog
):
    """With the surface above ``T_b`` the lid flux of the call is inward, and the open part
    carries ``F_atm`` less that flux, with no warning."""
    row = _row(w, T_surf=T_surf, F_atm=F_atm)
    L = 4.0 * (1400.0 - T_surf)
    assert L < 0.0
    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.common'):
        F_open = lid_open_flux(row)
    assert F_open == (F_atm - w * L) / (1.0 - w)
    assert abs(F_open - F_atm) > 1.0
    assert caplog.records == []


@pytest.mark.physics_invariant
@pytest.mark.parametrize('F_atm', [0.0, -5.0, -1200.0])
def test_an_atmosphere_flux_at_or_below_zero_gives_the_open_part_the_rest(F_atm):
    """A positive skin balance with ``F_atm`` at or below 0 gives the open part
    ``(F_atm - w L) / (1 - w)``; the run goes on."""
    row = _row(0.3, T_surf=900.0, F_atm=F_atm)
    L = 4.0 * 500.0
    assert lid_open_flux(row) == (F_atm - 0.3 * L) / 0.7


@pytest.mark.physics_invariant
@pytest.mark.parametrize('w', [1.0, 1.0 - 0.5 * LID_OPEN_AREA_FLOOR, 1.0 - 1e-10])
def test_an_open_area_at_the_floor_takes_f_atm(w):
    """Where ``1 - w`` is at or below ``LID_OPEN_AREA_FLOOR`` the open part takes ``F_atm``."""
    assert 1.0 - w <= LID_OPEN_AREA_FLOOR
    row = _row(w, T_surf=900.0, F_atm=-3.0)
    assert lid_open_flux(row) == row['F_atm']


@pytest.mark.physics_invariant
def test_an_open_area_just_above_the_floor_closes_the_sum():
    """Just above the floor the open part closes the sum, far from ``F_atm`` per open area."""
    w = 1.0 - 4.0 * LID_OPEN_AREA_FLOOR
    row = _row(w, T_surf=900.0, F_atm=-3.0)
    L = 4.0 * 500.0
    F_open = lid_open_flux(row)
    assert F_open == (-3.0 - w * L) / (1.0 - w)
    assert F_open < -1e11
    assert w * L + (1.0 - w) * F_open == pytest.approx(-3.0, rel=1e-6)


def _shared_surface(row, F_prime, T_m, F0=10.0, T_s0=900.0):
    """Surface of the call where a linear atmosphere ``F0 + F' (T_s - T_s0)`` meets the parallel
    balance of :func:`lid_boundary` with the open node at ``T_m``."""
    r = dict(row, lid_T_node=T_m)
    d, k, T_melt = lid_boundary(r)
    G = k / d
    return (G * T_melt - F0 + F_prime * T_s0) / (G + F_prime)


@pytest.mark.physics_invariant
def test_the_response_keeps_the_step_change_at_the_change_of_the_atmosphere():
    """On a stored lid-onset row: the open part's own response plus the lid part's
    change per total area is ``F' dT_s/dT_m``, which equals ``(1 - w) G_eff``."""
    w, G_l, F_p = 0.703, 7.27, 0.306
    row = _row(w, G=G_l, T_node=1600.0, T_surf=900.0, d=0.01, k=2.0, F_atm=1.0)
    G_eff, code = _g_eff(row, F_p)
    assert code == 0
    h = 1e-3
    dT_s = (_shared_surface(row, F_p, 1600.0 + h) - _shared_surface(row, F_p, 1600.0 - h)) / (
        2.0 * h
    )
    A_total = (1.0 - w) * 200.0 * (1.0 - dT_s)
    lid_part = -w * G_l * dT_s
    assert A_total + lid_part == pytest.approx(F_p * dT_s, rel=1e-9)
    assert (1.0 - w) * G_eff == pytest.approx(F_p * dT_s, rel=1e-9)
    assert (1.0 - w) * G_eff == pytest.approx(0.28043, rel=1e-4)
    assert G_eff == pytest.approx(0.94420, rel=1e-4)


@pytest.mark.physics_invariant
def test_the_response_limits():
    """At w = 0 the skin and the atmosphere in series; as w -> 1 ``G F' / (F' + G_l')`` per open
    area and 0 per total area."""
    F_p = 0.4
    G0, _ = _g_eff(_row(0.0 + 1e-300, G=7.0, T_surf=900.0, F_atm=1.0), F_p)
    assert G0 == pytest.approx(200.0 * F_p / (200.0 + F_p), rel=1e-12)
    w = 1.0 - 1e-12
    G1, _ = _g_eff(_row(w, G=7.0, T_surf=900.0, F_atm=1.0), F_p)
    assert G1 == pytest.approx(200.0 * F_p / (F_p + 7.0), rel=1e-9)
    assert (1.0 - w) * G1 < 1e-10


@pytest.mark.physics_invariant
@settings(max_examples=300, deadline=None)
@given(
    w=st.floats(1e-9, 1.0),
    G_lid=st.floats(1e-3, 1e5),
    k_over_d=st.floats(1.0, 1e4),
    F_p=st.floats(1e-6, 1e3),
    T_s=st.floats(200.0, 3000.0),
)
def test_the_response_is_positive_and_at_most_the_skin(w, G_lid, k_over_d, F_p, T_s):
    """``0 < G_eff <= G``, and the size guard never raises the response."""
    row = _row(w, G=G_lid, d=1.0 / k_over_d, k=1.0, T_surf=T_s, F_atm=1.0)
    G_eff, code = _g_eff(row, F_p)
    assert 0.0 < G_eff <= k_over_d * (1.0 + 1e-12)
    cap = 4.0 * SIGMA * T_s**3
    assert code == (LID_SLOPE_GUARD_SIZE if F_p > cap else 0)
    G_cap, _ = _g_eff(row, min(F_p, cap))
    assert G_eff == pytest.approx(G_cap, rel=1e-12)


@pytest.mark.physics_invariant
def test_a_secant_is_cut_at_the_grey_secant_of_its_interval():
    """The grey secant over [T_surf, T_end] passes uncut; above it, the cut is that secant."""
    T_s, T_end = 1500.0, 1520.0
    row = _row(0.4, T_surf=T_s, F_atm=1.0)
    grey = SIGMA * (T_end**4 - T_s**4) / (T_end - T_s)
    assert grey > 4.0 * SIGMA * T_s**3 * 1.01
    G_in, code_in = _g_eff(row, grey, T_end)
    assert code_in == 0
    G_above, code_above = _g_eff(row, 1.5 * grey, T_end)
    assert code_above == LID_SLOPE_GUARD_SIZE
    assert G_above == pytest.approx(G_in, rel=1e-12)
    _, code_tangent = _g_eff(row, grey)
    assert code_tangent == LID_SLOPE_GUARD_SIZE


@pytest.mark.physics_invariant
@pytest.mark.parametrize('T_end', [1480.0, 1520.0])
def test_the_grey_secant_cuts_a_cooling_and_a_warming_secant(T_end):
    """Below or above T_surf, a slope between the grey tangent at T_surf and the grey secant
    over [T_surf, T_end] is cut only when it lies above that secant."""
    T_s = 1500.0
    row = _row(0.4, T_surf=T_s, F_atm=1.0)
    tangent = 4.0 * SIGMA * T_s**3
    grey = SIGMA * (T_end**4 - T_s**4) / (T_end - T_s)
    between = 0.5 * (tangent + grey)
    _, code = _g_eff(row, between, T_end)
    assert code == (LID_SLOPE_GUARD_SIZE if T_end < T_s else 0)
    G_cut, code_cut = _g_eff(row, 2.0 * tangent, T_end)
    assert code_cut == LID_SLOPE_GUARD_SIZE
    at_cut = SIGMA * (T_end**2 + T_s**2) * (T_end + T_s)  # the cut itself is not cut
    assert _g_eff(row, at_cut, T_end)[1] == 0
    assert G_cut == pytest.approx(_g_eff(row, grey * (1 - 1e-12), T_end)[0], rel=1e-9)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('T_end', [None, math.nan, math.inf, 0.0, -1500.0, 1500.0])
def test_without_a_finite_other_end_the_cut_is_the_grey_tangent(T_end):
    """With no finite other end above 0, or the same surface, the cut is the tangent
    4 sigma T_surf^3."""
    T_s = 1500.0
    row = _row(0.4, T_surf=T_s, F_atm=1.0)
    tangent = 4.0 * SIGMA * T_s**3
    assert _g_eff(row, tangent * (1 - 1e-9), T_end)[1] == 0
    G_cut, code = _g_eff(row, 2.0 * tangent, T_end)
    assert code == LID_SLOPE_GUARD_SIZE
    assert G_cut == pytest.approx(_g_eff(row, tangent)[0], rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('F_p', [0.0, -1.0, math.nan, math.inf])
def test_a_slope_of_zero_or_a_bad_slope_holds_the_start_value(F_p):
    """A slope of 0 (no usable slope, or a sign fault) or a non-finite one gives ``G_eff = 0``."""
    G_eff, code = _g_eff(_row(0.4, T_surf=900.0, F_atm=1.0), F_p)
    assert G_eff == 0.0
    assert code == LID_SLOPE_GUARD_SIGN


@pytest.mark.physics_invariant
def test_no_response_without_a_live_lid():
    """Without a live lid there is no open response and no guard code."""
    for row in (_row(0.0, T_surf=900.0, F_atm=1.0), {'F_atm': 1.0, 'T_surf': 900.0}):
        assert _g_eff(row, 0.3) == (None, 0)


@pytest.mark.physics_invariant
def test_the_response_caps_the_lid_conductance_at_the_skin():
    """A lid conducting more than the skin enters ``G_eff`` at the skin conductance, as in the
    parallel sum of the call."""
    w, F_p, G = 0.5, 3.0, 200.0
    G_eff, code = _g_eff(_row(w, G=1000.0, T_surf=900.0, F_atm=1.0), F_p)
    assert code == 0
    assert G_eff == pytest.approx(G * F_p / (F_p + w * G + (1.0 - w) * G), rel=1e-14)
    uncapped = G * F_p / (F_p + w * 1000.0 + (1.0 - w) * G)
    assert abs(G_eff / uncapped - 1.0) > 0.5


@pytest.mark.unit
def test_the_open_flux_stop_writes_the_atmosphere_status():
    """The stop for an ``F_atm`` that is not finite records status 22 before it ends the run;
    a usable row, ``F_atm`` below 0 included, passes its start value through."""
    from unittest.mock import patch

    from proteus.atmos_clim.common import LidOpenFluxError
    from proteus.atmos_clim.wrapper import lid_or_stop

    with patch('proteus.atmos_clim.wrapper.UpdateStatusfile') as status:
        with pytest.raises(LidOpenFluxError, match='not finite'):
            lid_or_stop({}, lid_open_flux, _row(0.3, T_surf=900.0, F_atm=math.nan))
        assert [c.args[1] for c in status.call_args_list] == [22]
        for row in (_balanced(_row(0.3, T_surf=900.0)), _row(0.3, T_surf=900.0, F_atm=-5.0)):
            assert lid_or_stop({}, lid_open_flux, row) == lid_open_flux(row)
        assert status.call_count == 1


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_response_is_the_change_of_the_shared_surface(validation_plot):
    """Validation: at 3 slopes F' of the atmosphere and 19 lid weights, ``(1 - w) G_eff``
    equals ``F' dT_s/dT_m``, the change of the atmosphere flux at the shared surface of the call
    when the open node warms (central difference of the surface balance); relative 1e-7."""
    G_l = 7.27
    weights = np.linspace(0.05, 0.95, 19)
    slopes = (0.05, 0.306, 3.0)
    rows = {}
    for F_p in slopes:
        got, want = [], []
        for w in weights:
            row = _row(w, G=G_l, T_node=1600.0, T_surf=900.0, d=0.01, k=2.0, F_atm=1.0)
            G_eff, code = _g_eff(row, F_p)
            assert code == 0
            h = 1e-3
            dT_s = (
                _shared_surface(row, F_p, 1600.0 + h) - _shared_surface(row, F_p, 1600.0 - h)
            ) / (2.0 * h)
            got.append((1.0 - w) * G_eff)
            want.append(F_p * dT_s)
        got, want = np.array(got), np.array(want)
        np.testing.assert_allclose(got, want, rtol=1e-7)
        rows[F_p] = (got, want)
    # Discrimination: the open part alone (form A, G F'/(G + F')) misses at w = 0.5, F' = 3.
    w, F_p = 0.5, 3.0
    form_a = (1.0 - w) * 200.0 * F_p / (200.0 + F_p)
    assert abs(form_a / rows[F_p][1][9] - 1.0) > 0.1

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for F_p, mk in zip(slopes, ('o', 's', '^'), strict=True):
            got, want = rows[F_p]
            line = ax.plot(weights, want / F_p, label=f"F' = {F_p:g}")[0]
            ax.plot(weights, got / F_p, mk, ms=4, mfc='none', color=line.get_color())
        ax.set(xlabel='Lid weight w', ylabel="(1 - w) G_eff / F'")
        ax.legend(fontsize=8)
        # the difference is round-off: text, its value depends on the platform
        err = max(np.max(np.abs(got / want - 1.0)) for got, want in rows.values()) / 1e-7
        ax.text(
            0.03,
            0.08,
            f'max difference / tolerance\n= {err:.1e}',
            transform=ax.transAxes,
            fontsize=8,
        )
        fig.tight_layout()
        return fig

    validation_plot(figure, 'atmos_clim_lid_open_response')


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_a_lid_base_that_is_not_finite_stops_before_the_atmosphere_call(value):
    """A lid base that is not finite under a live lid stops the check that runs before the
    atmosphere call, with status 22 and the column named; without a live lid it is not read."""
    from unittest.mock import patch

    from proteus.atmos_clim.common import LidOpenFluxError, check_lid_inputs
    from proteus.atmos_clim.wrapper import lid_or_stop

    row = _row(0.3, T_surf=900.0, F_atm=5.0)
    row['lid_T_base'] = value
    with patch('proteus.atmos_clim.wrapper.UpdateStatusfile') as status:
        with pytest.raises(LidOpenFluxError, match='not finite') as err:
            lid_or_stop({'output': 'x'}, check_lid_inputs, row)
    assert f'lid_T_base {value}' in str(err.value)
    status.assert_called_once_with({'output': 'x'}, 22)
    off = _row(0.0, T_surf=900.0, F_atm=5.0)
    off['lid_T_base'] = value
    assert check_lid_inputs(off) is None


def test_only_the_open_flux_stop_writes_the_atmosphere_status():
    """Another error from the open-part flux passes up without status 22."""
    from unittest.mock import patch

    from proteus.atmos_clim import wrapper
    from proteus.atmos_clim.wrapper import lid_or_stop

    with (
        patch(
            'proteus.atmos_clim.wrapper.lid_open_flux', side_effect=RuntimeError('zero skin')
        ),
        patch('proteus.atmos_clim.wrapper.UpdateStatusfile') as status,
    ):
        with pytest.raises(RuntimeError, match='zero skin'):
            lid_or_stop({}, wrapper.lid_open_flux, _row(0.3, T_surf=900.0, F_atm=1.0))
        status.assert_not_called()


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('T_surf', math.nan),
        ('T_surf', math.inf),
        ('lid_T_base', math.nan),
        ('lid_T_base', -math.inf),
    ],
)
@pytest.mark.parametrize('w', [0.3, 1.0])
def test_a_surface_or_lid_base_that_is_not_finite_stops_under_a_live_lid(key, value, w):
    """A surface or lid-base temperature that is not finite stops under a live lid, also where
    the open area is at the floor, with a finite ``F_atm``."""
    from proteus.atmos_clim.common import LidOpenFluxError

    row = _row(w, T_surf=900.0, F_atm=5.0)
    row[key] = value
    with pytest.raises(LidOpenFluxError, match='not finite'):
        lid_open_flux(row)


@pytest.mark.parametrize(
    ('key', 'value', 'w'),
    [
        ('lid_weight', math.nan, 0.3),
        ('lid_weight', math.inf, 0.3),
        ('lid_conductance', math.nan, 0.3),
        ('lid_conductance', math.inf, 0.3),
        ('lid_T_node', math.nan, 0.3),
        ('lid_T_node', -math.inf, 1.0),
    ],
)
def test_a_lid_column_that_is_not_finite_stops_where_it_is_read(key, value, w):
    """A lid weight that is not finite, a lid conductance that is not finite under a weight
    above 0, and a node temperature that is not finite under a live lid stop the call, with a
    finite ``F_atm``, surface and lid base."""
    from proteus.atmos_clim.common import LidOpenFluxError

    row = _row(w, T_surf=900.0, F_atm=5.0)
    row[key] = value
    with pytest.raises(LidOpenFluxError, match='not finite') as err:
        lid_open_flux(row)
    assert f'{key} {value}' in str(err.value)


def test_a_lid_column_that_is_not_read_passes_when_not_finite():
    """Without a lid weight the conductance and the node are not read, and with a conductance
    of 0 the node is not read: a NaN there passes ``F_atm`` through."""
    for key in ('lid_conductance', 'lid_T_node'):
        row = _row(0.0, T_surf=900.0, F_atm=5.0)
        row[key] = math.nan
        assert lid_open_flux(row) == row['F_atm']
    row = _row(0.3, G=0.0, T_surf=900.0, F_atm=5.0)
    row['lid_T_node'] = math.nan
    assert lid_open_flux(row) == row['F_atm']


@pytest.mark.parametrize('F_atm', [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize('T_surf', [900.0, 1490.0])
def test_an_atmosphere_flux_that_is_not_finite_stops_under_a_live_lid(F_atm, T_surf):
    """A flux that is not finite stops under a positive skin balance and under one at or below 0."""
    from proteus.atmos_clim.common import LidOpenFluxError

    with pytest.raises(LidOpenFluxError, match='not finite'):
        lid_open_flux(_row(0.3, T_surf=T_surf, F_atm=F_atm))
