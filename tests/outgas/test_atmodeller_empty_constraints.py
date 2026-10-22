"""Tests for the atmodeller wrapper when no volatile element budget is above the threshold.

Module under test: ``proteus.outgas.atmodeller``.

Under ``user_constant`` the wrapper solves the O2-only network against the IW
buffer, as CALLIOPE does, so an atmosphere left from an earlier call is replaced
by the buffered O2 column. Under ``from_O_budget`` it returns without a solve.
A non-finite H, C, N or S total raises with the element named.

See ``docs/How-to/testing.md``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip('atmodeller')
pytest.importorskip('hypothesis')

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proteus.outgas.atmodeller import (  # noqa: E402
    _MODEL_CACHE,
    calc_surface_pressures_atmodeller,
)
from proteus.utils.constants import noble_gases, vol_list  # noqa: E402
from tests.outgas.test_atmodeller import (  # noqa: E402
    _CALLIOPE_SPECIES_FIELDS,
    _hf_row_with_HS_budget,
    _make_user_constant_config,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_THRESH = 1e8  # config.outgas.mass_thresh of _make_user_constant_config


def _empty_row() -> dict:
    """Helpfile row with every H, C, N, S and noble total at 0."""
    hf_row = _hf_row_with_HS_budget()
    hf_row['H_kg_total'] = 0.0
    hf_row['S_kg_total'] = 0.0
    return hf_row


def _o2_output(p_o2: float = 1.0e-6):
    """Fake atmodeller output holding only O2."""
    out = MagicMock()
    out.quick_look.return_value = {'O2_g': np.array(p_o2)}
    out.total_pressure.return_value = np.array(p_o2)
    out.asdict.return_value = {'O2_g': {'gas_mass': np.array(1.0e10)}}
    model = MagicMock()
    model.output = out
    return model


def _solve(config, hf_row, model=None):
    """Run the wrapper once; return the solved species names, or None without a model."""
    _MODEL_CACHE.clear()
    try:
        if model is None:
            calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)
        else:
            with patch('atmodeller.EquilibriumModel', return_value=model):
                calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)
        return [key[0] for key in _MODEL_CACHE][0] if _MODEL_CACHE else None
    finally:
        _MODEL_CACHE.clear()


@pytest.mark.physics_invariant
def test_empty_constraints_user_constant_solves_o2_only():
    """A real O2-only solve replaces a stale atmosphere: O2 at the IW+2 buffer, every
    other species 0 in all 10 fields, C_kg_atm 0, and O_kg_total = O_kg_atm."""
    from atmodeller.thermodata import IronWustiteBuffer

    config = _make_user_constant_config(2.0)
    hf_row = _empty_row()
    hf_row.update(
        {f'CO2_{f}': 1.0e10 * (i + 1) for i, f in enumerate(_CALLIOPE_SPECIES_FIELDS)}
    )
    hf_row.update(O2_bar=2.227, C_kg_atm=8.76e15, P_surf=2.232)

    solved = _solve(config, hf_row)

    p_o2 = 10.0 ** (float(np.squeeze(IronWustiteBuffer(0.0).log10_fugacity(2200.0, 1.0))) + 2.0)
    assert solved == ('O2_g',)
    assert hf_row['O2_bar'] == pytest.approx(p_o2, rel=1e-6)
    assert hf_row['P_surf'] == hf_row['O2_bar']
    assert hf_row['O2_vmr'] == 1.0
    others = {f'{s}_{f}': hf_row[f'{s}_{f}'] for s in vol_list + noble_gases if s != 'O2'
              for f in _CALLIOPE_SPECIES_FIELDS}  # fmt: skip
    assert others == dict.fromkeys(others, 0.0)
    assert hf_row['C_kg_atm'] == 0.0
    assert hf_row['O_kg_total'] == pytest.approx(hf_row['O_kg_atm'], rel=1e-12)
    assert hf_row['O_kg_atm'] > 0.0


def test_empty_constraints_from_o_budget_returns_without_a_solve():
    """Under from_O_budget an empty constraint set builds no model and marks the derived
    IW offset and the O residual as undefined."""
    config = _make_user_constant_config(2.0)
    config.planet.fO2_source = 'from_O_budget'
    hf_row = _empty_row()
    hf_row.update(fO2_shift_IW_derived=2.0, O_res=0.0)
    with patch('atmodeller.EquilibriumModel') as em:
        try:
            calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)
        except ValueError as e:
            pytest.fail(f'from_O_budget with no budget above the threshold must return: {e}')
    em.assert_not_called()
    assert np.isnan(hf_row['fO2_shift_IW_derived'])
    assert np.isnan(hf_row['O_res'])


@pytest.mark.parametrize('element', ['H', 'C', 'N', 'S'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_element_total_raises_with_the_element_named(element, value):
    """A NaN or infinite H, C, N or S total raises before any model is built."""
    config = _make_user_constant_config(2.0)
    hf_row = _hf_row_with_HS_budget()
    hf_row[f'{element}_kg_total'] = value
    with patch('atmodeller.EquilibriumModel') as em:
        with pytest.raises(ValueError, match=f'{element}_kg_total is not finite'):
            calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)
    em.assert_not_called()


@pytest.mark.physics_invariant
def test_atmosphere_mass_of_an_element_outside_the_solve_raises():
    """A solver output that names CO2 while C is outside the solve would give C_kg_atm > 0;
    the wrapper raises with the element named instead of writing it."""
    config = _make_user_constant_config(2.0)
    model = _o2_output()
    model.output.quick_look.return_value = {'O2_g': np.array(1.0e-6), 'CO2_g': np.array(1.0)}
    model.output.asdict.return_value['CO2_g'] = {'gas_mass': np.array(1.0e15)}
    with pytest.raises(RuntimeError, match='C_kg_atm'):
        _solve(config, _empty_row(), model)


_TOTALS = st.sampled_from(
    [0.0, -1.0, _THRESH * (1 - 1e-12), _THRESH, _THRESH * (1 + 1e-12), 1e30]
)


@settings(max_examples=60, deadline=None)
@given(
    totals=st.fixed_dictionaries({e: _TOTALS for e in 'HCNS'}),
    he=st.sampled_from([0.0, 1.0]),
    t_magma=st.sampled_from([500.0, 1200.0, 5000.0]),
    shift=st.floats(-10.0, 10.0),
)
def test_o2_only_network_exactly_when_no_budget_is_above_the_threshold(
    totals, he, t_magma, shift
):
    """The solved network is O2 alone exactly when no H, C, N or S total exceeds
    mass_thresh and no noble gas total is positive; the IW constraint is always set."""
    config = _make_user_constant_config(shift)
    hf_row = _empty_row()
    hf_row.update({f'{e}_kg_total': v for e, v in totals.items()})
    hf_row.update(He_kg_total=he, T_magma=t_magma)
    model = _o2_output()

    solved = _solve(config, hf_row, model)

    empty = max(totals.values()) <= _THRESH and he == 0.0
    assert (solved == ('O2_g',)) == empty
    assert 'O2_g' in model.solve.call_args.kwargs['fugacity_constraints']
    assert float(np.squeeze(model.solve.call_args.kwargs['state'].temperature)) >= 1200.0


@pytest.mark.physics_invariant
def test_o2_only_solve_with_a_second_pressure_raises():
    """On the O2-only path P_surf must equal O2_bar; an output that also gives He a
    partial pressure while He has no budget raises."""
    config = _make_user_constant_config(2.0)
    model = _o2_output()
    model.output.quick_look.return_value['He_g'] = np.array(1.0)
    with pytest.raises(RuntimeError, match='O2-only solve: P_surf'):
        _solve(config, _empty_row(), model)
