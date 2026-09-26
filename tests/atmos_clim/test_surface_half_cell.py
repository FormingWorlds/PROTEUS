"""
Unit tests for the Aragog top half cell in series with the atmosphere's conductive skin.

Module under test: ``proteus.atmos_clim.common.surface_skin_inputs`` and its three
callers (dummy, JANUS, AGNI wrappers), the ``surface_half_cell`` config check, and the
interior surface flux ``F_int`` of the Aragog wrapper. Invariants: the default path is
unchanged; the series conductance ``1 / G_eff = surface_d / surface_k + w / G_top_half``;
the conduction-limited flux of a solid lid stays below ``G_eff (T_base - T_eq)``.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import attrs
import pytest

from proteus.atmos_clim.common import surface_skin_inputs

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

K_SKIN = 2.0  # W/m/K, schema default
D_SKIN = 0.01  # m, schema default
G_HALF = 4.0 / 15.0e3  # k 4 W/m/K over a 15 km half cell [W/m^2/K]


def _cfg(module='aragog', half_cell=True):
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(surface_d=D_SKIN, surface_k=K_SKIN),
        interior_energetics=SimpleNamespace(
            module=module, aragog=SimpleNamespace(surface_half_cell=half_cell)
        ),
    )


def _row(w=1.0, G=G_HALF, T_top=400.0, T_magma=320.0):
    return {'T_magma': T_magma, 'T_top_cell': T_top, 'G_top_half': G, 'w_solid_top': w}


@pytest.mark.physics_invariant
def test_skin_inputs_put_the_solid_half_cell_in_series_with_the_skin():
    """A solid top cell (w = 1): the skin base is the top-cell temperature and the
    equivalent thickness gives the series conductance of skin and half cell."""
    T_base, d = surface_skin_inputs(_cfg(), _row())
    assert T_base == pytest.approx(400.0, rel=1e-15)
    G_eff = K_SKIN / d
    assert 1.0 / G_eff == pytest.approx(D_SKIN / K_SKIN + 1.0 / G_HALF, rel=1e-12)
    # Scale guard: the half cell dominates, 3750 m^2 K/W against 0.005 for the skin.
    assert 7000.0 < d < 8000.0
    # A dropped weight or a parallel sum would leave d near the 1 cm skin.
    assert d > 1e5 * D_SKIN


@pytest.mark.physics_invariant
def test_skin_inputs_blend_with_the_solid_weight_and_default_without_it():
    """Half-solid top cell blends base temperature and series resistance linearly in w;
    w = 0, G = 0, a non-Aragog interior, the option off, or missing columns all give
    the skin unchanged."""
    T_base, d = surface_skin_inputs(_cfg(), _row(w=0.5))
    assert T_base == pytest.approx(0.5 * 320.0 + 0.5 * 400.0, rel=1e-15)
    assert d == pytest.approx(D_SKIN + 0.5 * K_SKIN / G_HALF, rel=1e-12)
    for cfg, row in (
        (_cfg(), _row(w=0.0)),
        (_cfg(), _row(G=0.0)),
        (_cfg(module='spider'), _row()),
        (_cfg(half_cell=False), _row()),
        (_cfg(), {'T_magma': 320.0}),
    ):
        assert surface_skin_inputs(cfg, row) == pytest.approx((320.0, D_SKIN), rel=1e-15)


@pytest.mark.physics_invariant
def test_dummy_atmosphere_flux_of_a_solid_lid_is_conduction_limited():
    """Cold transparent case (T_eq 273 K) with the top state at the 320 K table edge:
    the 1 cm skin drains 269.7 W/m^2; the series half cell (T_top 400 K) gives
    0.0339 W/m^2, equal to G_eff (T_base - T_surf) and below G_eff (T_top - 273 K)."""
    from proteus.atmos_clim.dummy import RunDummyAtm
    from proteus.config import read_config_object

    cfg = read_config_object('input/dummy.toml')
    ac = attrs.evolve(
        cfg.atmos_clim,
        surf_state='skin',
        dummy=attrs.evolve(cfg.atmos_clim.dummy, gamma=0.0),
    )
    cfg = attrs.evolve(cfg, atmos_clim=ac)
    row = dict(
        F_ins=1749.9,
        albedo_pl=0.1,
        atm_kg_per_mol=0.029,
        gravity=9.81,
        R_int=6.371e6,
        P_surf=1.0e5,
        T_surf=300.0,
        T_magma=320.0,
        T_top_cell=400.0,
        G_top_half=G_HALF,
        w_solid_top=1.0,
    )
    off = RunDummyAtm({'output': '/tmp'}, cfg, dict(row))
    assert off['F_atm'] == pytest.approx(269.67, rel=1e-3)

    on_cfg = SimpleNamespace(**{f.name: getattr(cfg, f.name) for f in attrs.fields(type(cfg))})
    on_cfg.interior_energetics = SimpleNamespace(
        module='aragog', aragog=SimpleNamespace(surface_half_cell=True)
    )
    on = RunDummyAtm({'output': '/tmp'}, on_cfg, dict(row))
    T_base, d = surface_skin_inputs(on_cfg, row)
    G_eff = K_SKIN / d
    assert on['F_atm'] == pytest.approx(G_eff * (T_base - on['T_surf']), rel=1e-6)
    assert 0.0 < on['F_atm'] < G_eff * (400.0 - 273.0)
    # The surface sits a few mK above the zero-flux temperature of this instellation.
    assert 273.0 < on['T_surf'] < 273.02
    assert off['F_atm'] > 1e3 * on['F_atm']


@patch('janus.utils.atmos')
def test_janus_update_receives_the_series_skin(_atmos):
    """JANUS ``UpdateStateAtm`` sets ``tmp_magma`` and ``skin_d`` from the half cell
    when it is on, and to ``T_magma`` and ``surface_d`` when it is off."""
    from proteus.atmos_clim.janus import UpdateStateAtm
    from tests.atmos_clim.test_janus import _build_update_hf_row

    for half_cell, expected in (
        (True, surface_skin_inputs(_cfg(), _row(T_magma=2000.0))),
        (False, (2000.0, D_SKIN)),
    ):
        cfg = _cfg(half_cell=half_cell)
        cfg.atmos_clim.p_top = 1e-5
        hf_row = _build_update_hf_row()
        hf_row.update(_row(T_magma=2000.0))
        atm = MagicMock()
        UpdateStateAtm(atm, cfg, hf_row, tropopause=None)
        assert atm.tmp_magma == pytest.approx(expected[0], rel=1e-15)
        assert atm.skin_d == pytest.approx(expected[1], rel=1e-15)
    assert expected[1] == pytest.approx(D_SKIN, rel=1e-15)


def test_agni_update_receives_the_series_skin(monkeypatch):
    """AGNI ``update_agni_atmos`` writes the half-cell base temperature and skin
    thickness onto the struct; with the option off they stay ``T_magma`` and
    ``surface_d``."""
    import proteus.atmos_clim.agni as agni_mod
    from tests.atmos_clim.test_agni import (
        _install_profile_fakes,
        _ProfileAGNI,
        _ProfileAtmosphere,
    )

    for half_cell in (True, False):
        _install_profile_fakes(monkeypatch, _ProfileAGNI())
        cfg = _cfg(half_cell=half_cell)
        cfg.atmos_clim.agni = SimpleNamespace(psurf_thresh=1.0e-3, ini_profile='isothermal')
        atmos = _ProfileAtmosphere([1.0e1, 1.0e3, 1.0e5, 1.0e7], [200.0, 500.0, 1100.0, 1900.0])
        hf_row = dict(
            F_ins=1361.0,
            albedo_pl=0.1,
            T_surf=1900.0,
            P_surf=200.0,
            gravity=9.8,
            R_int=6.4e6,
            M_int=6.0e24,
            axial_period=86400.0,
            longitude=0.0,
            latitude=0.0,
            **_row(w=0.5, T_top=1500.0, T_magma=2000.0),
        )
        agni_mod.update_agni_atmos(atmos, hf_row, {'output': '/tmp/run'}, cfg)
        T_base, d = surface_skin_inputs(cfg, hf_row)
        assert atmos.tmp_magma == pytest.approx(T_base, rel=1e-15)
        assert atmos.skin_d == pytest.approx(d, rel=1e-15)
        if half_cell:
            assert atmos.tmp_magma == pytest.approx(1750.0, rel=1e-15)
            assert atmos.skin_d > 1e5 * D_SKIN
        else:
            assert (atmos.tmp_magma, atmos.skin_d) == pytest.approx((2000.0, D_SKIN), rel=1e-15)


def test_config_rejects_the_half_cell_without_a_skin_in_flux_mode():
    """``surface_half_cell`` acts through the atmosphere skin, so flux mode with
    ``surf_state = 'fixed'`` is rejected; grey-body mode needs no skin."""
    from proteus.config._config import surface_half_cell_needs_skin

    def inst(surf_state, bc_mode, half_cell=True):
        return SimpleNamespace(
            atmos_clim=SimpleNamespace(surf_state=surf_state),
            interior_energetics=SimpleNamespace(
                module='aragog',
                surface_bc_mode=bc_mode,
                aragog=SimpleNamespace(surface_half_cell=half_cell),
            ),
        )

    with pytest.raises(ValueError, match="surf_state='skin'"):
        surface_half_cell_needs_skin(inst('fixed', 'flux'), None, None)
    for ok in (inst('skin', 'flux'), inst('fixed', 'grey_body'), inst('fixed', 'flux', False)):
        assert surface_half_cell_needs_skin(ok, None, None) is None


@pytest.mark.physics_invariant
def test_interior_flux_is_the_applied_call_mean_with_the_half_cell():
    """With the half cell, ``F_int`` is the call-mean applied flux from the surface
    energy integral (below ``F_atm`` when the table-edge cutoff acted); without it,
    flux mode reports ``F_atm`` and grey-body mode the surface node flux."""
    from proteus.interior_energetics.aragog import _surface_flux_int
    from proteus.utils.constants import secs_per_year

    area, dt_yr = 5.1e14, 1.0e3
    out = SimpleNamespace(
        dt_actual=dt_yr,
        step_dE_F_int_J=-0.02 * area * dt_yr * secs_per_year,
        heat_flux=[0.0, 7.0],
    )
    hf_row = {'F_atm': 0.03}
    F = _surface_flux_int(out, hf_row, area, 'flux', True)
    assert F == pytest.approx(0.02, rel=1e-12)
    assert 0.0 < F < hf_row['F_atm']
    assert _surface_flux_int(out, hf_row, area, 'flux', False) == pytest.approx(0.03, rel=1e-15)
    assert _surface_flux_int(out, hf_row, area, 'grey_body', False) == pytest.approx(
        7.0, rel=1e-15
    )
    # Zero-length call: no integral to divide, fall back to the mode's flux.
    out.dt_actual = 0.0
    assert _surface_flux_int(out, hf_row, area, 'flux', True) == pytest.approx(0.03, rel=1e-15)
