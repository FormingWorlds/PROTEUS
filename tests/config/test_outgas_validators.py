"""Tests for outgas config validators.

This file targets _outgas.py (Outgas, Calliope, Atmodeller parameters).
See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_calliope_defaults():
    """Test Calliope module initializes with valid defaults."""
    from proteus.config._outgas import Calliope

    c = Calliope()
    assert c.include_H2O is True
    assert c.include_CO2 is True
    assert c.include_N2 is True
    assert c.solubility is True
    # The cold-start pressure ceiling defaults to CALLIOPE's own 1e5 bar, so
    # the default PROTEUS config leaves the solver guess range unchanged.
    assert c.p_guess_max == pytest.approx(1.0e5)


@pytest.mark.unit
def test_calliope_p_guess_max_validator_and_override():
    """p_guess_max accepts a raised ceiling within CALLIOPE's solver box and
    rejects values outside (0, 1e7]: a non-positive ceiling would invert the
    log-uniform draw range, and a ceiling above the 1e7 bar box is unreachable
    (the solver rejects roots beyond the box), so it must fail loudly."""
    from proteus.config._outgas import Calliope

    # A raised ceiling within the box is accepted and stored verbatim.
    c = Calliope(p_guess_max=5.0e6)
    assert c.p_guess_max == pytest.approx(5.0e6)
    # Discrimination: the override is honoured, not silently reset to the default.
    assert c.p_guess_max != pytest.approx(1.0e5)

    # Out-of-range ceilings are rejected: non-positive, above the 1e7 box, and
    # the non-finite inf that a bare gt(0) would have let through.
    for bad in (0.0, -1.0e5, 1.0e8, float('inf')):
        with pytest.raises(ValueError):
            Calliope(p_guess_max=bad)


@pytest.mark.unit
def test_calliope_is_included_h2o():
    """Test Calliope is_included method returns volatile inclusion status."""
    from proteus.config._outgas import Calliope

    c = Calliope()
    assert c.is_included('H2O') is True
    assert c.is_included('CO2') is True
    assert c.is_included('N2') is True


@pytest.mark.unit
def test_outgas_shared_solver_defaults():
    """Test shared solver parameters on Outgas (T_floor, tolerances)."""
    from proteus.config._outgas import Outgas

    o = Outgas(module='calliope', fO2_shift_IW=4.0)
    assert o.T_floor == pytest.approx(700.0, rel=1e-12)
    assert o.solver_rtol == pytest.approx(1e-4, rel=1e-12)
    assert o.solver_atol == pytest.approx(1e-6, rel=1e-12)


@pytest.mark.unit
def test_outgas_custom_solver_params():
    """Test Outgas accepts custom shared solver parameters."""
    from proteus.config._outgas import Outgas

    o = Outgas(
        module='calliope', fO2_shift_IW=4.0, T_floor=500.0, solver_rtol=1e-5, solver_atol=1e-7
    )
    assert o.T_floor == pytest.approx(500.0, rel=1e-12)
    assert o.solver_rtol == pytest.approx(1e-5, rel=1e-12)
    assert o.solver_atol == pytest.approx(1e-7, rel=1e-12)


@pytest.mark.unit
def test_outgas_prevent_desaturation_defaults_off_and_toggles():
    """Condensates may re-vapourise by default; the flag turns that off."""
    from proteus.config._outgas import Outgas

    assert Outgas(module='atmodeller', fO2_shift_IW=-4.0).prevent_desaturation is False
    on = Outgas(module='atmodeller', fO2_shift_IW=-4.0, prevent_desaturation=True)
    assert on.prevent_desaturation is True


@pytest.mark.unit
def test_lavatmos_defaults():
    """Lavatmos initializes with the documented rock-vapour defaults."""
    from proteus.config._outgas import Lavatmos

    lav = Lavatmos()
    assert lav.T_min == pytest.approx(1500.0, rel=1e-12)
    assert lav.melt_comp_name == 'BSE_palm'
    assert lav.P_melt == pytest.approx(0.01, rel=1e-12)
    assert lav.xatol == pytest.approx(1e-5, rel=1e-12)


@pytest.mark.unit
def test_lavatmos_custom_melt_pressure_and_tolerance():
    """
    P_melt and xatol are honoured verbatim and must stay strictly positive.
    """
    from proteus.config._outgas import Lavatmos

    # Custom melt pressure and solver tol.
    lav = Lavatmos(P_melt=0.05, xatol=2e-6)
    assert lav.P_melt == pytest.approx(0.05, rel=1e-12)
    assert lav.xatol == pytest.approx(2e-6, rel=1e-12)

    # Discrimination: overrides are stored, not reset to the defaults.
    assert lav.P_melt != pytest.approx(0.01, rel=1e-12)
    assert lav.xatol != pytest.approx(1e-5, rel=1e-12)

    # Check melt pressure and tol. are positive.
    for bad in (0.0, -0.01):
        with pytest.raises(ValueError):
            Lavatmos(P_melt=bad)
        with pytest.raises(ValueError):
            Lavatmos(xatol=bad)


def test_h2_binodal_enabled_is_rejected():
    """`h2_binodal = true` is rejected at config load.

    The H2-silicate binodal partitioning is not production ready, so
    enabling it must fail loudly instead of running an unsettled
    parameterisation. The default (off) constructs, pinning that the
    rejection is the flag value and not the field itself.
    """
    from proteus.config._outgas import Outgas

    with pytest.raises(ValueError, match='h2_binodal'):
        Outgas(h2_binodal=True)
    o = Outgas()
    assert o.h2_binodal is False


# Lid seal: each numeric field refuses a TOML bool, a string, NaN and the values
# outside its range, and keeps the values inside it.
_SEAL_BAD = [True, False, '1.0', float('nan')]


def _seal(**kw):
    from proteus.config._outgas import LidSeal

    return LidSeal(**kw)


def test_lid_seal_defaults_hold_and_derive_the_open_band():
    """Defaults: off, 10 km to close, top cell below 0.3 melt, tau inf, no crust
    production; the open band is 0.5 delta_close and phi_close + 0.05."""
    s = _seal()
    assert (s.enabled, s.delta_close, s.phi_close, s.tau_leak) == (
        False,
        1e4,
        0.3,
        float('inf'),
    )
    assert (s.crust_rate, s.chi_d, s.phi_solid) == (0.0, 0.02, 1e-3)
    assert s.delta_open_value == pytest.approx(5.0e3, rel=1e-15)
    assert s.phi_open_value == pytest.approx(0.35, rel=1e-15)
    s = _seal(delta_close=0.3, delta_open=0.2, phi_open=0.31)
    assert (s.delta_open_value, s.phi_open_value) == (0.2, 0.31)


@pytest.mark.parametrize(
    ('field', 'bad'),
    [
        *(('delta_close', b) for b in (*_SEAL_BAD, 0.0, -1.0, float('inf'))),
        *(('phi_close', b) for b in (*_SEAL_BAD, 0.0, 1.0, -0.1)),
        *(('tau_leak', b) for b in (*_SEAL_BAD, 0.0, -1.0)),
        *(('chi_d', b) for b in (*_SEAL_BAD, -1e-9, 1.0 + 1e-9)),
        *(('crust_rate', b) for b in (*_SEAL_BAD, -1.0, float('inf'))),
        *(('phi_solid', b) for b in (*_SEAL_BAD, 0.0, 0.3, 0.5)),
        *(('delta_open', b) for b in (*_SEAL_BAD, 0.0, 1e4 + 1.0, float('inf'))),
        *(('phi_open', b) for b in (*_SEAL_BAD, 0.29, 1.0)),
    ],
)
def test_lid_seal_refuses_values_outside_each_domain(field, bad):
    """Each field refuses a bool, a string, NaN and its out-of-range values; the
    bounds phi_solid < phi_close, delta_open <= delta_close and phi_open >=
    phi_close read the other fields (defaults 1e4 m and 0.3)."""
    with pytest.raises(ValueError, match=field):
        _seal(**{field: bad})
    assert isinstance(bad, (bool, str, float))


def test_lid_seal_refuses_a_derived_phi_open_at_or_above_one():
    """phi_open = none with phi_close >= 0.95 would derive a value >= 1."""
    with pytest.raises(ValueError, match='phi_open = None'):
        _seal(phi_close=0.95)
    assert _seal(phi_close=0.949).phi_open_value == pytest.approx(0.999, rel=1e-12)
    assert _seal(phi_close=0.95, phi_open=0.99).phi_open_value == 0.99


def test_lid_seal_accepts_the_limits_and_both_regimes():
    """Range limits that are inside: tau inf, chi_d 0 and 1, an int thickness,
    delta_open == delta_close, phi_open == phi_close; both the 0.3 m and the
    10 km closure regimes load."""
    assert _seal(tau_leak=float('inf'), chi_d=0.0).chi_d == 0.0
    assert _seal(chi_d=1.0, crust_rate=0).crust_rate == 0
    assert _seal(delta_close=2, delta_open=2).delta_open_value == 2.0
    assert _seal(phi_open=0.3).phi_open_value == 0.3
    assert _seal(delta_close=0.3).delta_open_value == pytest.approx(0.15, rel=1e-15)


def test_lid_seal_hypothesis_over_the_thickness_and_fraction_domains():
    """Property: any finite thickness > 0 and fraction in (0, 1) loads, and the
    derived band stays inside (0, delta_close] and [phi_close, 1)."""
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=200, deadline=None)
    @given(
        st.floats(min_value=1e-9, max_value=1e9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=2e-3, max_value=0.9499, allow_nan=False),
    )
    def check(delta, phi):
        s = _seal(delta_close=delta, phi_close=phi)
        assert 0.0 < s.delta_open_value <= s.delta_close
        assert s.phi_close <= s.phi_open_value < 1.0

    check()
    with pytest.raises(ValueError):
        _seal(delta_close=-1e-9)


def test_lid_seal_cross_field_rules(tmp_path):
    """With the seal on, a non-Aragog interior, freeze_volatiles and vapourise
    are refused at load; front trapping is allowed."""
    import tomllib

    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    with open(PROTEUS_ROOT / 'input' / 'all_options.toml', 'rb') as f:
        base = tomllib.load(f)
    assert base['interior_energetics']['module'] == 'aragog'

    def load(edit):
        import copy

        import tomlkit

        cfg = copy.deepcopy(base)
        cfg['outgas']['lid_seal']['enabled'] = True
        edit(cfg)
        path = tmp_path / 'c.toml'
        path.write_text(tomlkit.dumps(cfg))
        return read_config_object(path)

    assert load(lambda c: c['outgas'].update(trap_mode='front')).outgas.lid_seal.enabled
    with pytest.raises(ValueError, match='freeze_volatiles'):
        load(lambda c: c['params']['stop']['solid'].update(freeze_volatiles=True))
    with pytest.raises(ValueError, match='vapourise'):
        load(lambda c: c['outgas'].update(vapourise=True))
    with pytest.raises(ValueError, match='lid_seal.enabled = true requires'):
        load(lambda c: c['interior_energetics'].update(module='spider'))
    assert not load(lambda c: None).params.stop.solid.freeze_volatiles
