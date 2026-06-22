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

from unittest.mock import MagicMock, patch

import pytest

# calliope is an optional dependency; skip if not installed
pytest.importorskip('calliope')

from proteus.outgas.calliope import (
    _resolve_element,
    calc_surface_pressures,
    construct_guess,
    flag_included_volatiles,
)
from proteus.utils.constants import element_list, vol_list

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
    target = {e: 1e20 for e in element_list}
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
    target = {e: 1e20 for e in element_list}

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
    target = {e: 1e20 for e in element_list}
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
# calc_surface_pressures: graceful, labelled handling of a CALLIOPE
# volatile-equilibrium non-convergence.
# -----------------------------------------------------------------------


@pytest.mark.unit
def test_calc_surface_pressures_labels_calliope_nonconvergence(tmp_path):
    """A CALLIOPE non-convergence is relabelled and given outgassing status 27.

    When CALLIOPE cannot find a volatile-equilibrium solution, the wrapper
    must catch the bare 'max attempts' RuntimeError, write outgassing-error
    status 27, and re-raise a RuntimeError whose message names the
    unphysical-regime cause, so a degenerate cell is identifiable for
    exclusion instead of producing an opaque crash.

    Discriminating: the underlying CALLIOPE error carries only the generic
    'max attempts' wording; the wrapper's message must instead match the
    labelled 'Unphysical planetary regime' phrasing AND the status code must
    be 27 (outgassing), not propagated unlabelled. The status-write happens
    before the re-raise so an unattended grid cell leaves a parseable status.
    """
    from collections import defaultdict

    config = MagicMock()
    config.planet.fO2_source = 'calc'  # not 'from_O_budget' -> equilibrium_atmosphere path
    config.outgas.T_floor = 1000.0
    config.outgas.mass_thresh = 1e10
    hf_row = {e + '_kg_total': 1e20 for e in element_list}
    dirs = {'output': str(tmp_path)}

    with (
        patch('proteus.outgas.calliope.construct_options', return_value={'T_magma': 4200.0}),
        patch('proteus.outgas.calliope.construct_guess', return_value=None),
        patch(
            'proteus.outgas.calliope.flag_included_volatiles',
            return_value=defaultdict(int),
        ),
        patch(
            'proteus.outgas.calliope.equilibrium_atmosphere',
            side_effect=RuntimeError(
                'Could not find solution for volatile abundances (max attempts, 1000)'
            ),
        ),
        patch('proteus.outgas.calliope.UpdateStatusfile') as mock_update,
    ):
        with pytest.raises(RuntimeError, match='Unphysical planetary regime'):
            calc_surface_pressures(dirs, config, hf_row)
    mock_update.assert_called_once()
    args, _ = mock_update.call_args
    assert args[1] == 27  # outgassing-model error status code
