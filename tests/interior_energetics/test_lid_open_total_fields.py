"""Boundary fields of the open part of a live lid, passed as a total-area term.

Under a live lid in flux mode with the mush base, the interior takes ``F_atm`` as the outer
boundary value and 4 fields of the call balance: the lid flux ``L = G_l (T_b - T_surf)``, the
slope ``F'`` (cut at the grey secant), the skin conductance ``G`` and ``G_l = min(G_lid, G)``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from proteus.interior_energetics.aragog import AragogRunner
from proteus.utils.constants import const_sigma
from tests.atmos_clim.test_common import _row

pytestmark = pytest.mark.unit

FIELDS = (
    'open_lid_flux',
    'open_surface_slope',
    'open_skin_conductance',
    'open_lid_conductance',
)
UNSET = object()


def _update(w, lid=True, T=(1150.0, 1160.0, 1170.0), F=(14000.0, 14400.0, 14900.0)):
    bc = SimpleNamespace(
        outer_boundary_condition=4,
        subgrid_lid=lid,
        lid_mush_base=lid,
        outer_boundary_value=None,
        lid_surface_flux_cap=None,
        equilibrium_temperature=None,
    )
    params = SimpleNamespace(
        solver=SimpleNamespace(start_time=None, end_time=None),
        boundary_conditions=bc,
        energy=SimpleNamespace(tidal_array=None),
    )
    interior_o = SimpleNamespace(
        aragog_solver=SimpleNamespace(parameters=params, solution=None), tides=np.zeros(3)
    )
    hf_all = pd.DataFrame({'Time': [10.0, 20.0, 30.0], 'T_surf': T, 'F_atm': F})
    row = _row(w, G=21.9, T_b=1440.0, T_surf=T[-1], F_atm=F[-1], lid_F_open=165000.0)
    row.update(Time=30.0, T_eqm=255.0)
    AragogRunner.update_solver(25.0, row, interior_o, hf_all=hf_all)
    return bc, row


@pytest.mark.parametrize('w', [0.4, 1.0])
def test_a_live_lid_passes_f_atm_and_the_four_fields_of_the_call(w):
    """The outer value is F_atm (total area), and the 4 fields are those of the call balance:
    L = G_l (T_b - T_surf), F' the secant of the last 2 rows (50 W m-2 K-1, below the grey
    secant), G = k/d of the skin (200), G_l = min(21.9, 200)."""
    bc, row = _update(w)
    assert bc.outer_boundary_value == row['F_atm']
    got = {f: getattr(bc, f, UNSET) for f in FIELDS}
    assert got['open_lid_flux'] == pytest.approx(21.9 * (1440.0 - 1170.0), rel=1e-14)
    assert got['open_surface_slope'] == pytest.approx(50.0, rel=1e-12)
    assert got['open_skin_conductance'] == pytest.approx(2.0 / 0.01, rel=1e-14)
    assert got['open_lid_conductance'] == pytest.approx(21.9, rel=1e-14)


def test_the_slope_field_is_cut_at_the_grey_secant():
    """A secant above the grey secant of the same 2 rows is cut to it, as for the response."""
    bc, row = _update(0.4, F=(14000.0, 14400.0, 24400.0))
    T0, T1 = 1160.0, 1170.0
    grey = const_sigma * (T0**2 + T1**2) * (T0 + T1)
    assert getattr(bc, 'open_surface_slope', UNSET) == pytest.approx(grey, rel=1e-12)
    assert row['lid_slope_guard'] == 2


@pytest.mark.parametrize(('w', 'lid'), [(0.0, True), (0.4, False)])
def test_without_a_live_lid_the_fields_are_none(w, lid):
    """Without a live lid (no weight, or a solver without the lid) the 4 fields are None and the
    outer value is F_atm."""
    bc, row = _update(w, lid=lid)
    assert bc.outer_boundary_value == row['F_atm']
    assert all(getattr(bc, f, UNSET) is None for f in FIELDS)
