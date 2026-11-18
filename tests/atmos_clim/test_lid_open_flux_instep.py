"""Open part of the Aragog lid surface: solidus floor, start value and in-step response.

The open part of the surface (weight ``1 - w``) sits on melt held at least at the lid base,
``T_m = max(T_node, T_b)``. Its start value is the scaled skin flux where the skin balance of
the call is positive and ``F_atm`` elsewhere; inside the step it follows ``T_m`` with
``G_eff = G F' / (F' + w G_l' + (1 - w) G)`` per open area.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip('hypothesis')
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proteus.atmos_clim.common import (  # noqa: E402
    LID_SLOPE_GUARD_SIGN,
    LID_SLOPE_GUARD_SIZE,
    lid_boundary,
    lid_open_conductance,
    lid_open_flux,
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
    """The scaled branch reads ``G (T_m - T_s)`` with the floored node."""
    row = _balanced(_row(0.3, T_node=1350.0, T_surf=900.0))
    assert lid_open_flux(row) == pytest.approx(200.0 * (1400.0 - 900.0), rel=1e-14)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('T_node', [1350.0, 1500.0])
@pytest.mark.parametrize('w', [1e-12, 1e-11])
def test_the_floored_start_value_goes_to_f_atm_at_the_onset(w, T_node):
    """The start value tends to ``F_atm`` as the weight goes to 0, floored or not."""
    row = _balanced(_row(w, T_node=T_node, T_surf=1300.0), gap=0.97)
    assert lid_open_flux(row) == pytest.approx(row['F_atm'], rel=1e-10)


@pytest.mark.physics_invariant
def test_the_unscaled_branch_takes_the_flux_of_the_atmosphere(caplog, monkeypatch):
    """Where the skin balance of the call is not positive the open part starts at ``F_atm``, with
    one warning per process."""
    import proteus.atmos_clim.common as common

    monkeypatch.setattr(common, '_OPEN_FLUX_WARNED', [False])
    row = _row(0.99, T_surf=1450.0, F_atm=-0.91)
    d, k, T_melt = lid_boundary(row)
    assert k / d * (T_melt - 1450.0) < 0.0
    with caplog.at_level('WARNING', logger='fwl.proteus.atmos_clim.common'):
        assert lid_open_flux(row) == -0.91
        assert lid_open_flux(_row(0.5, T_surf=1500.0, F_atm=7.0)) == 7.0
    warned = [r for r in caplog.records if 'F_atm' in r.getMessage()]
    assert len(warned) == 1


@pytest.mark.physics_invariant
def test_a_scale_of_zero_or_below_stops_the_run():
    """A positive skin balance with ``F_atm`` at or below 0 gives ``c <= 0``: the run stops."""
    for F_atm in (0.0, -5.0):
        row = _row(0.3, T_surf=900.0, F_atm=F_atm)
        with pytest.raises(RuntimeError, match='scale'):
            lid_open_flux(row)


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
    G_eff, code = lid_open_conductance(row, F_p)
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
    G0, _ = lid_open_conductance(_row(0.0 + 1e-300, G=7.0, T_surf=900.0, F_atm=1.0), F_p)
    assert G0 == pytest.approx(200.0 * F_p / (200.0 + F_p), rel=1e-12)
    w = 1.0 - 1e-12
    G1, _ = lid_open_conductance(_row(w, G=7.0, T_surf=900.0, F_atm=1.0), F_p)
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
    G_eff, code = lid_open_conductance(row, F_p)
    assert 0.0 < G_eff <= k_over_d * (1.0 + 1e-12)
    cap = 4.0 * SIGMA * T_s**3
    assert code == (LID_SLOPE_GUARD_SIZE if F_p > cap else 0)
    G_cap, _ = lid_open_conductance(row, min(F_p, cap))
    assert G_eff == pytest.approx(G_cap, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('F_p', [0.0, -1.0, math.nan, math.inf])
def test_a_slope_of_zero_or_a_bad_slope_holds_the_start_value(F_p):
    """A slope of 0 (no usable slope, or a sign fault) or a non-finite one gives ``G_eff = 0``."""
    G_eff, code = lid_open_conductance(_row(0.4, T_surf=900.0, F_atm=1.0), F_p)
    assert G_eff == 0.0
    assert code == LID_SLOPE_GUARD_SIGN


@pytest.mark.physics_invariant
def test_no_response_without_a_live_lid():
    """Without a live lid there is no open response and no guard code."""
    for row in (_row(0.0, T_surf=900.0, F_atm=1.0), {'F_atm': 1.0, 'T_surf': 900.0}):
        assert lid_open_conductance(row, 0.3) == (None, 0)


@pytest.mark.physics_invariant
def test_the_response_caps_the_lid_conductance_at_the_skin():
    """A lid conducting more than the skin enters ``G_eff`` at the skin conductance, as in the
    parallel sum of the call."""
    w, F_p, G = 0.5, 3.0, 200.0
    G_eff, code = lid_open_conductance(_row(w, G=1000.0, T_surf=900.0, F_atm=1.0), F_p)
    assert code == 0
    assert G_eff == pytest.approx(G * F_p / (F_p + w * G + (1.0 - w) * G), rel=1e-14)
    uncapped = G * F_p / (F_p + w * 1000.0 + (1.0 - w) * G)
    assert abs(G_eff / uncapped - 1.0) > 0.5


@pytest.mark.unit
def test_the_scale_stop_writes_the_atmosphere_status():
    """The stop for a scale at or below 0 records status 22 before it ends the run; a usable row
    passes its start value through."""
    from unittest.mock import patch

    from proteus.atmos_clim.wrapper import lid_open_flux_or_stop

    with patch('proteus.atmos_clim.wrapper.UpdateStatusfile') as status:
        with pytest.raises(RuntimeError, match='scale'):
            lid_open_flux_or_stop({}, _row(0.3, T_surf=900.0, F_atm=-5.0))
        assert [c.args[1] for c in status.call_args_list] == [22]
        row = _balanced(_row(0.3, T_surf=900.0))
        assert lid_open_flux_or_stop({}, row) == lid_open_flux(row)
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
            G_eff, code = lid_open_conductance(row, F_p)
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

        fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 5.6))
        for F_p, mk in zip(slopes, ('o', 's', '^'), strict=True):
            got, want = rows[F_p]
            line = top.plot(weights, want / F_p, label=f"F' = {F_p:g}")[0]
            top.plot(weights, got / F_p, mk, ms=4, mfc='none', color=line.get_color())
            bottom.plot(weights, np.abs(got / want - 1.0) / 1e-7, mk, ms=4, mfc='none')
        top.set(ylabel="(1 - w) G_eff / F'")
        top.legend(fontsize=8)
        bottom.set(xlabel='Lid weight w', ylabel='Diff. / tol.', ylim=(0.0, 1.0))
        fig.tight_layout()
        return fig

    validation_plot(figure, 'atmos_clim_lid_open_response')


def test_only_the_scale_stop_writes_the_atmosphere_status():
    """Another error from the open-part flux passes up without status 22."""
    from unittest.mock import patch

    from proteus.atmos_clim.wrapper import lid_open_flux_or_stop

    with (
        patch(
            'proteus.atmos_clim.wrapper.lid_open_flux', side_effect=RuntimeError('zero skin')
        ),
        patch('proteus.atmos_clim.wrapper.UpdateStatusfile') as status,
    ):
        with pytest.raises(RuntimeError, match='zero skin'):
            lid_open_flux_or_stop({}, _row(0.3, T_surf=900.0, F_atm=1.0))
        status.assert_not_called()


def test_a_scale_that_is_not_a_number_stops_with_a_message_that_says_so():
    """A NaN atmosphere flux gives a scale that is not a number; the message says so."""
    from proteus.atmos_clim.common import LidOpenFluxError

    with pytest.raises(LidOpenFluxError, match='not positive'):
        lid_open_flux(_row(0.3, T_surf=900.0, F_atm=float('nan')))
