"""Unit and Hypothesis property tests for the ``[invariants]`` config schema.

Verifies default values, type enforcement, bound validation (rejecting zero
or negative tolerances where disallowed, NaNs, infinities, booleans, and
invalid modes), and cross-field ordering rules.
"""

from __future__ import annotations

import math

import pytest

from proteus.config import Config, read_config_object, structure_config
from proteus.config._invariants import Invariants

hypothesis = pytest.importorskip('hypothesis')
given = hypothesis.given
settings = hypothesis.settings
st = hypothesis.strategies

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_STRICT_POSITIVE_FIELDS = (
    'energy_residual_rtol',
    'step_energy_rtol',
    'step_energy_smin_rtol',
    'solver_residual_rtol',
    'core_residual_rtol',
    'struct_mass_desync_warn_rtol',
    'struct_mass_desync_rtol',
    'interior_mass_target_rtol',
    'core_mass_target_rtol',
    'element_closure_rtol',
    'element_closure_atol_kg',
    'volatile_gas_sum_rtol',
    't_magma_max',
)

_NON_NEGATIVE_FIELDS = (
    'planet_mass_cap_rtol',
    'escape_bound_atol_kg',
    'phi_global_atol',
)


def test_invariants_config_default_values():
    """Default ``Invariants`` values match configured tolerances."""
    cfg = Config()
    inv = cfg.invariants
    assert isinstance(inv, Invariants)
    assert inv.mode == 'report'
    assert inv.energy_residual_rtol == pytest.approx(2.6e-4, rel=1e-12)
    assert inv.step_energy_rtol == pytest.approx(1.0e-2, rel=1e-12)
    assert inv.step_energy_smin_rtol == pytest.approx(5.0e-2, rel=1e-12)
    assert inv.solver_residual_rtol == pytest.approx(1.0e-12, rel=1e-12)
    assert inv.core_residual_rtol == pytest.approx(1.0e-4, rel=1e-12)
    assert inv.struct_mass_desync_warn_rtol == pytest.approx(5.0e-3, rel=1e-12)
    assert inv.struct_mass_desync_rtol == pytest.approx(1.0e-2, rel=1e-12)
    assert inv.interior_mass_target_rtol == pytest.approx(3.0e-3, rel=1e-12)
    assert inv.core_mass_target_rtol == pytest.approx(2.5e-2, rel=1e-12)
    assert inv.element_closure_rtol == pytest.approx(2.5e-4, rel=1e-12)
    assert inv.element_closure_atol_kg == pytest.approx(1.0, rel=1e-12)
    assert inv.volatile_gas_sum_rtol == pytest.approx(1.0e-6, rel=1e-12)
    assert inv.planet_mass_cap_rtol == pytest.approx(1.0e-6, rel=1e-12)
    assert inv.escape_bound_factor == pytest.approx(1.5, rel=1e-12)
    assert inv.escape_bound_atol_kg == pytest.approx(1.0e3, rel=1e-12)
    assert inv.t_magma_max == pytest.approx(1.0e6, rel=1e-12)
    assert inv.phi_global_atol == pytest.approx(1.0e-6, rel=1e-12)


@pytest.mark.parametrize('valid_mode', ['off', 'report', 'stop'])
def test_invariants_mode_accepts_valid_choices(valid_mode):
    """``Invariants.mode`` accepts only ``'off'``, ``'report'``, and ``'stop'``."""
    inv = Invariants(mode=valid_mode)
    assert inv.mode == valid_mode
    assert inv.struct_mass_desync_rtol > inv.struct_mass_desync_warn_rtol


@settings(max_examples=40, deadline=None)
@given(
    bad_mode=st.text(min_size=0, max_size=16).filter(
        lambda s: s not in {'off', 'report', 'stop'}
    )
)
def test_hypothesis_invariants_mode_rejects_arbitrary_strings(bad_mode):
    """Any string outside ``{'off', 'report', 'stop'}`` is rejected for ``mode``."""
    with pytest.raises((ValueError, TypeError)):
        Invariants(mode=bad_mode)


@settings(max_examples=40, deadline=None)
@given(
    field_name=st.sampled_from(_STRICT_POSITIVE_FIELDS),
    bad_val=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
        st.just(float('nan')),
        st.just(float('inf')),
        st.just(float('-inf')),
    ),
)
def test_hypothesis_strict_positive_tolerances_reject_non_positive_and_non_finite(
    field_name, bad_val
):
    """Strictly positive tolerance fields reject ``<= 0``, ``NaN``, and ``+/-inf``."""
    with pytest.raises((ValueError, TypeError)):
        Invariants(**{field_name: bad_val})


@settings(max_examples=40, deadline=None)
@given(
    field_name=st.sampled_from(_STRICT_POSITIVE_FIELDS),
    pos_val=st.floats(
        min_value=1.0e-15, max_value=1.0e9, allow_nan=False, allow_infinity=False
    ),
)
def test_hypothesis_strict_positive_tolerances_accept_valid_positive_floats(
    field_name, pos_val
):
    """Strictly positive tolerance fields accept finite positive floats respecting ordering."""
    kwargs = {field_name: pos_val}
    if field_name == 'struct_mass_desync_warn_rtol':
        kwargs['struct_mass_desync_rtol'] = max(pos_val * 2.0, 1.0e-2)
    elif field_name == 'struct_mass_desync_rtol':
        kwargs['struct_mass_desync_warn_rtol'] = min(pos_val * 0.5, 5.0e-3)
    elif field_name == 'step_energy_rtol':
        kwargs['step_energy_smin_rtol'] = max(pos_val * 2.0, 5.0e-2)
    elif field_name == 'step_energy_smin_rtol':
        kwargs['step_energy_rtol'] = min(pos_val * 0.5, 1.0e-2)
    inv = Invariants(**kwargs)
    assert math.isfinite(getattr(inv, field_name))
    assert getattr(inv, field_name) == pytest.approx(pos_val, rel=1e-12)


@settings(max_examples=40, deadline=None)
@given(
    field_name=st.sampled_from(_NON_NEGATIVE_FIELDS),
    bad_val=st.one_of(
        st.floats(max_value=-1.0e-15, allow_nan=False, allow_infinity=True),
        st.just(float('nan')),
        st.just(float('inf')),
        st.just(float('-inf')),
    ),
)
def test_hypothesis_non_negative_tolerances_reject_negative_and_non_finite(field_name, bad_val):
    """Non-negative tolerance fields reject ``< 0``, ``NaN``, and ``+/-inf``."""
    with pytest.raises((ValueError, TypeError)):
        Invariants(**{field_name: bad_val})


@settings(max_examples=30, deadline=None)
@given(
    bad_factor=st.one_of(
        st.floats(max_value=0.999999, allow_nan=False, allow_infinity=True),
        st.just(float('nan')),
        st.just(float('inf')),
        st.just(float('-inf')),
    )
)
def test_hypothesis_escape_bound_factor_rejects_below_one_and_non_finite(bad_factor):
    """``escape_bound_factor`` must be finite and ``>= 1.0``."""
    with pytest.raises((ValueError, TypeError)):
        Invariants(escape_bound_factor=bad_factor)


@pytest.mark.parametrize(
    'field_name',
    _STRICT_POSITIVE_FIELDS + _NON_NEGATIVE_FIELDS + ('escape_bound_factor',),
)
@pytest.mark.parametrize('bad_type_val', [True, False, '1e-4', None, [1e-4]])
def test_invariants_numeric_fields_reject_booleans_and_non_numeric_types(
    field_name, bad_type_val
):
    """Numeric tolerance fields reject booleans, strings, ``None``, and containers."""
    with pytest.raises((ValueError, TypeError)):
        Invariants(**{field_name: bad_type_val})


def test_invariants_cross_field_ordering_validators():
    """Stop thresholds cannot be smaller than their lower/warn counterparts."""
    with pytest.raises(ValueError, match='struct_mass_desync_rtol'):
        Invariants(struct_mass_desync_warn_rtol=1.0e-2, struct_mass_desync_rtol=5.0e-3)
    with pytest.raises(ValueError, match='step_energy_smin_rtol'):
        Invariants(step_energy_rtol=5.0e-2, step_energy_smin_rtol=1.0e-2)


def test_invariants_section_loads_from_toml_and_rejects_orphan_keys(tmp_path):
    """``read_config_object`` structures ``[invariants]`` and rejects unknown keys."""
    from helpers import PROTEUS_ROOT

    dummy_text = (PROTEUS_ROOT / 'input' / 'dummy.toml').read_text()
    valid_toml = tmp_path / 'valid_inv.toml'
    valid_toml.write_text(
        dummy_text + '\n[invariants]\nmode = "stop"\nstruct_mass_desync_rtol = 0.015\n'
    )
    cfg = read_config_object(valid_toml)
    assert cfg.invariants.mode == 'stop'
    assert cfg.invariants.struct_mass_desync_rtol == pytest.approx(0.015)

    orphan_toml = tmp_path / 'orphan_inv.toml'
    orphan_toml.write_text(dummy_text + '\n[invariants]\nnonexistent_tol = 1e-3\n')
    with pytest.raises(ValueError, match='invariants.nonexistent_tol'):
        read_config_object(orphan_toml)
    assert isinstance(structure_config({'config_version': '3.0'}, valid_toml), Config)


def test_energy_residual_coarse_dt_rtol_key_is_rejected(tmp_path):
    """An energy_residual_coarse_dt_rtol key in [invariants] is rejected as an unknown key."""
    from helpers import PROTEUS_ROOT

    dummy_text = (PROTEUS_ROOT / 'input' / 'dummy.toml').read_text()
    coarse_toml = tmp_path / 'coarse_inv.toml'
    coarse_toml.write_text(
        dummy_text + '\n[invariants]\nenergy_residual_coarse_dt_rtol = 5.0e-4\n'
    )
    with pytest.raises(ValueError, match='invariants.energy_residual_coarse_dt_rtol'):
        read_config_object(coarse_toml)

    with pytest.raises(TypeError, match='energy_residual_coarse_dt_rtol'):
        Invariants(energy_residual_coarse_dt_rtol=5.0e-4)  # type: ignore[call-arg]


@settings(max_examples=30, deadline=None)
@given(
    base_rtol=st.floats(
        min_value=1.0e-8, max_value=1.0e-2, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_step_energy_rtol_ordering_and_non_kink_dispatch(base_rtol, scale):
    """``step_energy_rtol`` governs mushy rows above 301 K while ``step_energy_smin_rtol`` governs T <= 301 K."""
    from types import SimpleNamespace

    import pandas as pd

    from proteus.utils.invariants import check_energy_invariants

    smin_rtol = base_rtol * scale
    inv = Invariants(step_energy_rtol=base_rtol, step_energy_smin_rtol=smin_rtol)
    cfg = SimpleNamespace(invariants=inv)
    hf_all = pd.DataFrame(
        [
            {
                'Time': 100.0,
                'E_state_cons_J': 1.0e31,
                'E_state_heat_cons_J': -1.0e30,
                'step_dE_F_int_J': -1.0e30,
                'step_dE_F_cmb_J': 0.0,
                'step_dE_Q_radio_J': 0.0,
                'step_dE_Q_tidal_J': 0.0,
            }
        ]
    )
    mid_frac = base_rtol * 1.5
    step_gross = 1.0e28
    hf_row = {
        'Time': 200.0,
        'T_surf': 1500.0,
        'T_magma': 2200.0,
        'Phi_global': 0.5,
        'RF_depth': 0.3,
        'E_state_cons_J': 1.0e31,
        'E_state_heat_cons_J': -1.01e30,
        'E_residual_cons_J': 0.0,
        'step_dE_state_heat_J': -step_gross * (1.0 - mid_frac),
        'step_dE_F_int_J': -step_gross,
        'step_dE_F_cmb_J': 0.0,
        'step_dE_Q_radio_J': 0.0,
        'step_dE_Q_tidal_J': 0.0,
        'solver_residual_J': 0.0,
        'core_C_eff': 0.0,
    }
    v_non_kink = check_energy_invariants(hf_row, cfg, hf_all)
    assert [v.name for v in v_non_kink] == ['step_dE_inc_frac']
    v_tsurf_cold = check_energy_invariants(dict(hf_row, T_surf=300.5), cfg, hf_all)
    assert [v.name for v in v_tsurf_cold] == ['step_dE_inc_frac']
    assert v_tsurf_cold[0].tolerance == pytest.approx(base_rtol)

    hf_row_kink = dict(hf_row, T_magma=300.5)
    assert check_energy_invariants(hf_row_kink, cfg, hf_all) == []
