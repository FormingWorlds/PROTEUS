"""Unit tests for ``proteus.outgas.calliope``.

Exercises the CALLIOPE wrapper helper functions with mocked CALLIOPE
imports: ``_resolve_element`` (element mode-to-mass conversion),
``construct_guess`` (initial-guess construction for the solver),
and ``flag_included_volatiles`` (volatile inclusion logic).

Invariants tested:
  - _resolve_element: correct unit conversion for 'X/H', 'ppmw', 'kg' modes
  - _resolve_element: unknown mode raises ValueError
  - construct_guess: returns None at Time < 1 (IC phase)
  - construct_guess: zeros guess for depleted elements
  - flag_included_volatiles: O2 is always included

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# calliope is an optional dependency; skip if not installed
pytest.importorskip('calliope')

from proteus.outgas.calliope import (
    _resolve_element,
    _resolve_noble,
    construct_guess,
    construct_options,
    flag_included_volatiles,
)
from proteus.utils.constants import (
    element_list,
    noble_gases,
    noble_solar_mass_ratio,
    vol_list,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# -----------------------------------------------------------------------
# _resolve_element
# -----------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_resolve_element_ratio_mode():
    """In 'C/H' mode, element mass = budget * H_kg.

    For C with budget=0.5 and H_kg=1e20, result is 5e19 kg.
    Discrimination: 'ppmw' mode with the same budget would give
    0.5e-6 * M_reservoir, orders of magnitude different.
    """
    result = _resolve_element(mode='C/H', budget=0.5, H_kg=1e20, M_reservoir=4e24, name='C')
    assert result == pytest.approx(5e19, rel=1e-12)
    # Scale guard: ratio mode gives ~5e19, ppmw would give ~2e18
    assert result > 1e19
    assert result < 1e20


@pytest.mark.physics_invariant
def test_resolve_element_ppmw_mode():
    """In 'ppmw' mode, element mass = budget * 1e-6 * M_reservoir.

    For budget=100 ppmw and M_reservoir=4e24 kg, result is 4e20 kg.
    Discrimination: 'kg' mode with budget=100 would give 100 kg,
    20 orders of magnitude off.
    """
    result = _resolve_element(mode='ppmw', budget=100.0, H_kg=1e20, M_reservoir=4e24, name='N')
    expected = 100.0 * 1e-6 * 4e24  # = 4e20
    assert result == pytest.approx(expected, rel=1e-12)
    # Scale guard: 4e20, not 100 (kg mode) or 5e-5 (ratio mode)
    assert 1e20 < result < 1e21


@pytest.mark.physics_invariant
def test_resolve_element_kg_mode():
    """In 'kg' mode, element mass = budget directly.

    The budget value is returned without any conversion.
    """
    result = _resolve_element(mode='kg', budget=1.5e18, H_kg=1e20, M_reservoir=4e24, name='S')
    assert result == pytest.approx(1.5e18, rel=1e-12)
    # Pass-through invariant: changing H_kg or M_reservoir must NOT
    # affect the result in 'kg' mode
    result2 = _resolve_element(mode='kg', budget=1.5e18, H_kg=9e99, M_reservoir=9e99, name='S')
    assert result2 == pytest.approx(1.5e18, rel=1e-12)


def test_resolve_element_unknown_mode_raises():
    """An unrecognised mode raises ValueError with a message naming
    the element and listing valid modes.

    Edge case: typo in config ('ppb' instead of 'ppmw').
    """
    with pytest.raises(ValueError, match='Unknown C_mode'):
        _resolve_element(mode='ppb', budget=100.0, H_kg=1e20, M_reservoir=4e24, name='C')

    # Adjacent-valid: all three valid modes must NOT raise
    for mode, budget in [('C/H', 0.5), ('ppmw', 100.0), ('kg', 1e18)]:
        result = _resolve_element(
            mode=mode, budget=budget, H_kg=1e20, M_reservoir=4e24, name='C'
        )
        assert result > 0


# -----------------------------------------------------------------------
# construct_guess
# -----------------------------------------------------------------------


def test_construct_guess_returns_none_at_time_zero():
    """At Time < 1 (initial phase), construct_guess returns None to
    let CALLIOPE construct its own initial guess.

    Edge case: the very first outgassing call.
    """
    hf_row = {'Time': 0.0}
    target = {e: (1e20 if e not in noble_gases else 0.0) for e in element_list}
    result = construct_guess(hf_row, target, mass_thresh=1.0)
    assert result is None
    assert hf_row['Time'] == pytest.approx(0.0, abs=1e-12)  # hf_row untouched


def test_construct_guess_uses_previous_pressures():
    """After the IC phase (Time >= 1), construct_guess reads the
    previous partial pressures from hf_row and returns them as the
    guess dictionary.

    Discrimination: at Time < 1 the function returns None (tested above).
    """
    hf_row = {'Time': 100.0}
    for s in vol_list:
        hf_row[f'{s}_bar'] = 10.0
    target = {e: (1e20 if e not in noble_gases else 0.0) for e in element_list}

    result = construct_guess(hf_row, target, mass_thresh=1.0)
    assert result is not None
    assert isinstance(result, dict)
    # All volatiles should have the previous pressure
    for s in vol_list:
        assert result[s] == pytest.approx(10.0, rel=1e-12)


@pytest.mark.physics_invariant
def test_construct_guess_zeros_depleted_element():
    """When an element's target mass is below mass_thresh, the guess
    pressure for all volatiles containing that element is set to zero.

    Edge case: hydrogen is fully escaped, so H2O and H2 and H2S and
    NH3 and CH4 guess pressures should be zero.
    """
    hf_row = {'Time': 100.0}
    for s in vol_list:
        hf_row[f'{s}_bar'] = 10.0

    # H is depleted below threshold
    target = {e: (1e20 if e not in noble_gases else 0.0) for e in element_list}
    target['H'] = 0.1  # below mass_thresh=1.0

    result = construct_guess(hf_row, target, mass_thresh=1.0)
    assert result is not None
    # H-bearing volatiles must have zero guess
    for s in ('H2O', 'H2', 'H2S', 'NH3', 'CH4'):
        assert result[s] == pytest.approx(0.0, abs=1e-30), (
            f'{s} guess should be zero when H is depleted'
        )
    # Non-H volatiles with non-depleted elements keep their guess
    assert result['CO2'] == pytest.approx(10.0, rel=1e-12)
    assert result['N2'] == pytest.approx(10.0, rel=1e-12)


# -----------------------------------------------------------------------
# flag_included_volatiles
# -----------------------------------------------------------------------


def test_flag_included_volatiles_o2_always_included():
    """O2 is always included regardless of config settings.

    This is a hard-coded invariant in the CALLIOPE wrapper.
    """
    config = MagicMock()
    # Set all include_X to False
    for s in vol_list:
        if s != 'O2':
            setattr(config.outgas.calliope, f'include_{s}', False)

    result = flag_included_volatiles(guess=None, config=config)
    assert result['O2'] is True
    # Other volatiles should be excluded
    assert result['H2O'] is False


def test_flag_included_volatiles_zero_pressure_excludes():
    """A volatile with zero guess pressure is excluded even if the
    config says to include it.

    Edge case: the element backing a volatile has been fully escaped,
    so the guess is zero.
    """
    config = MagicMock()
    for s in vol_list:
        if s != 'O2':
            setattr(config.outgas.calliope, f'include_{s}', True)

    guess = {s: 10.0 for s in vol_list}
    guess['H2O'] = 0.0  # depleted

    result = flag_included_volatiles(guess=guess, config=config)
    assert result['H2O'] is False
    # CO2 still has positive pressure, should remain included
    assert result['CO2'] is True
    # O2 always included
    assert result['O2'] is True


# -----------------------------------------------------------------------
# _resolve_noble
# -----------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_resolve_noble_kg_mode_is_pass_through():
    """In 'kg' mode the noble gas mass is the budget itself, independent of
    the hydrogen inventory or the reservoir mass.
    """
    result = _resolve_noble(mode='kg', budget=3.0e16, H_kg=1.5e20, M_reservoir=4e24, name='He')
    assert result == pytest.approx(3.0e16, rel=1e-12)
    # Pass-through: changing H_kg and M_reservoir must not change the result.
    result2 = _resolve_noble(mode='kg', budget=3.0e16, H_kg=9e99, M_reservoir=9e99, name='He')
    assert result2 == pytest.approx(3.0e16, rel=1e-12)


@pytest.mark.physics_invariant
def test_resolve_noble_ppmw_mode_scales_with_reservoir():
    """In 'ppmw' mode the mass is budget * 1e-6 * M_reservoir.

    For 5 ppmw and a 4e24 kg reservoir the mass is 2e19 kg. Discrimination:
    the 'kg' mode with the same budget would give 5 kg, and the 'solar' mode
    would scale by the protosolar Ar/H ratio instead, both far from 2e19.
    """
    result = _resolve_noble(mode='ppmw', budget=5.0, H_kg=1.5e20, M_reservoir=4e24, name='Ar')
    assert result == pytest.approx(5.0 * 1e-6 * 4e24, rel=1e-12)
    assert 1e19 < result < 1e20
    # Wrong-mode guard: kg mode would return 5.0, off by ~19 orders.
    assert abs(result - 5.0) > 1e18


@pytest.mark.physics_invariant
def test_resolve_noble_solar_mode_uses_protosolar_ratio():
    """In 'solar' mode the mass is budget * (X/H)_solar * H_kg, so a budget
    of 1.0 gives exactly the protosolar noble gas complement for the given
    hydrogen inventory.
    """
    H_kg = 1.5e20
    result = _resolve_noble(mode='solar', budget=1.0, H_kg=H_kg, M_reservoir=4e24, name='He')
    expected = noble_solar_mass_ratio['He'] * H_kg
    assert result == pytest.approx(expected, rel=1e-12)
    # A budget of 0.5 is half solar, and the mode is linear in the budget.
    half = _resolve_noble(mode='solar', budget=0.5, H_kg=H_kg, M_reservoir=4e24, name='He')
    assert half == pytest.approx(0.5 * expected, rel=1e-12)
    # Discrimination: the solar mass ratios are distinct per gas, so Xe must
    # give a far smaller inventory than He at the same budget and H.
    xe = _resolve_noble(mode='solar', budget=1.0, H_kg=H_kg, M_reservoir=4e24, name='Xe')
    assert xe < 1e-4 * result


def test_resolve_noble_unknown_mode_raises():
    """An unrecognised noble gas mode raises ValueError naming the gas and
    the valid modes, so a config typo fails loudly rather than silently.
    """
    with pytest.raises(ValueError, match='Unknown He_mode'):
        _resolve_noble(mode='oceans', budget=1.0, H_kg=1e20, M_reservoir=4e24, name='He')
    # Adjacent-valid: all three real modes return a non-negative mass.
    for mode, budget in [('kg', 1e16), ('ppmw', 5.0), ('solar', 1.0)]:
        assert (
            _resolve_noble(mode=mode, budget=budget, H_kg=1e20, M_reservoir=4e24, name='He')
            >= 0
        )


# -----------------------------------------------------------------------
# construct_options :: noble gas wiring
# -----------------------------------------------------------------------


def _element_mode_config(noble_included, He_mode='kg', He_budget=0.0, reservoir='mantle'):
    """Minimal element-mode config for construct_options with noble control.

    `noble_included` maps a noble gas symbol to its include flag. Only the
    attributes construct_options reads in element mode are populated.
    """
    config = MagicMock()
    config.outgas.fO2_shift_IW = 4.0
    config.outgas.calliope.solubility = True
    config.planet.volatile_mode = 'elements'
    config.planet.volatile_reservoir = reservoir
    config.planet.gas_prs.get_pressure = lambda s: 0.0

    def _is_included(s):
        if s in noble_gases:
            return noble_included.get(s, False)
        return True

    config.outgas.calliope.is_included = _is_included

    elem = config.planet.elements
    elem.use_metallicity = False
    elem.H_mode, elem.H_budget = 'kg', 1.5e20
    elem.C_mode, elem.C_budget = 'kg', 1.0e20
    elem.N_mode, elem.N_budget = 'kg', 2.0e18
    elem.S_mode, elem.S_budget = 'kg', 5.0e19
    for gas in noble_gases:
        setattr(elem, f'{gas}_mode', 'kg')
        setattr(elem, f'{gas}_budget', 0.0)
    elem.He_mode, elem.He_budget = He_mode, He_budget
    return config


_HF_ROW = {
    'M_mantle': 4.0e24,
    'M_int': 4.5e24,
    'gravity': 9.81,
    'R_int': 6.37e6,
    'Phi_global': 1.0,
    'T_magma': 1800.0,
}


def test_construct_options_includes_noble_gas_with_budget():
    """A noble gas that is switched on and carries a positive budget reaches
    the CALLIOPE options as an inclusion flag and a ppmw budget relative to
    the mantle mass. This is the element-mode path a PROTEUS config uses.
    """
    config = _element_mode_config({'He': True}, He_mode='kg', He_budget=3.0e16)
    opts = construct_options({}, config, dict(_HF_ROW))

    assert opts['He_included'] == 1
    # He_ppmw is the mantle-relative ppmw of the 3e16 kg budget.
    expected_ppmw = 1e6 * 3.0e16 / _HF_ROW['M_mantle']
    assert opts['He_ppmw'] == pytest.approx(expected_ppmw, rel=1e-9)
    # The other noble gases stay off with a zero budget.
    for gas in ('Ne', 'Ar', 'Kr', 'Xe'):
        assert opts[f'{gas}_included'] == 0
        assert opts[f'{gas}_ppmw'] == 0.0
    # Discrimination guard: forgetting the 1e6 ppmw factor would make He_ppmw
    # a mass fraction (~7.5e-9) rather than a ppmw (~7.5e-3).
    assert opts['He_ppmw'] > 1e-4


def test_construct_options_excludes_noble_gas_off_or_zero_budget():
    """A noble gas that is switched off, or on but with a zero budget, does
    not enter the solve: its inclusion flag stays zero and its ppmw is zero.
    """
    # He switched off entirely.
    off = construct_options(
        {}, _element_mode_config({'He': False}, He_budget=3.0e16), dict(_HF_ROW)
    )
    assert off['He_included'] == 0
    assert off['He_ppmw'] == 0.0

    # He switched on but with no budget: still excluded from the solve.
    zero = construct_options(
        {}, _element_mode_config({'He': True}, He_budget=0.0), dict(_HF_ROW)
    )
    assert zero['He_included'] == 0
    assert zero['He_ppmw'] == 0.0


def test_construct_options_solar_mode_scales_with_hydrogen():
    """The 'solar' noble gas mode reaches CALLIOPE as a ppmw budget derived
    from the protosolar He/H ratio and the hydrogen inventory, so it tracks
    the hydrogen budget rather than being an absolute mass.
    """
    config = _element_mode_config({'He': True}, He_mode='solar', He_budget=1.0)
    opts = construct_options({}, config, dict(_HF_ROW))

    H_kg = 1.5e20  # matches the config H_budget in kg mode
    He_kg = noble_solar_mass_ratio['He'] * H_kg
    expected_ppmw = 1e6 * He_kg / _HF_ROW['M_mantle']
    assert opts['He_included'] == 1
    assert opts['He_ppmw'] == pytest.approx(expected_ppmw, rel=1e-9)


def test_construct_options_ppmw_mode_uses_selected_reservoir():
    """In 'ppmw' mode with volatile_reservoir='mantle+core', the He budget is
    ppmw relative to the mantle-plus-core mass, then re-expressed as a
    mantle-relative ppmw for CALLIOPE. This exercises the reservoir-selection
    branch that the kg and solar tests bypass.
    """
    config = _element_mode_config(
        {'He': True}, He_mode='ppmw', He_budget=10.0, reservoir='mantle+core'
    )
    opts = construct_options({}, config, dict(_HF_ROW))

    # He_kg = 10 ppmw of M_int; the mantle-relative ppmw CALLIOPE receives is
    # 1e6 * He_kg / M_mantle, which equals the input ppmw scaled by M_int/M_mantle.
    he_kg = 10.0 * 1e-6 * _HF_ROW['M_int']
    expected = 1e6 * he_kg / _HF_ROW['M_mantle']
    assert opts['He_included'] == 1
    assert opts['He_ppmw'] == pytest.approx(expected, rel=1e-9)
    assert opts['He_ppmw'] == pytest.approx(
        10.0 * _HF_ROW['M_int'] / _HF_ROW['M_mantle'], rel=1e-9
    )
    # Discrimination: a reservoir swap that used M_mantle instead of M_int
    # would give exactly the input 10.0 ppmw (the mantle factors cancel), so
    # the mantle-plus-core result must differ from 10.0 by the M_int/M_mantle
    # ratio (about 12.5 percent here).
    assert opts['He_ppmw'] != pytest.approx(10.0, rel=1e-3)


def test_construct_guess_defers_to_cold_start_when_noble_active():
    """When a noble gas has a positive target, construct_guess returns None
    so CALLIOPE runs its own cold start, because the helpfile does not carry
    a noble partial pressure to warm-start from.
    """
    hf_row = {'Time': 100.0}
    for s in vol_list:
        hf_row[f'{s}_bar'] = 10.0
    target = {e: (1e20 if e not in noble_gases else 0.0) for e in element_list}
    target['He'] = 3.0e16  # He is active

    assert construct_guess(hf_row, target, mass_thresh=1.0) is None

    # A trace noble inventory, far below the major-volatile mass threshold,
    # still defers to the cold start. This is the realistic regime (Earth-like
    # helium is orders of magnitude below the 1e16 kg threshold); a condition
    # that keyed on mass_thresh would miss it and hand CALLIOPE a warm guess
    # with no helium pressure.
    target['He'] = 3.0e12
    assert construct_guess(hf_row, target, mass_thresh=1.0e16) is None

    # Discrimination: with no active noble gas the same call returns a dict.
    target['He'] = 0.0
    result = construct_guess(hf_row, target, mass_thresh=1.0)
    assert isinstance(result, dict)
    assert result['CO2'] == pytest.approx(10.0, rel=1e-12)
