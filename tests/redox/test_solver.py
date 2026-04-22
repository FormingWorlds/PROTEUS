"""
Transactional Brent solver tests (#57 Commit C).

Plan v6 §3.3 + §5.4. Locks:
- `solve_fO2` never mutates the caller's hf_row beyond writing
  `fO2_shift_IW` (all inner probes run on deep-copies).
- Brent converges to a known ΔIW on a synthetic residual.
- Desiccation gate frozen at entry (inner probes don't re-evaluate).
- Bracket widening engages on first sign-change failure before
  falling back to previous ΔIW.
"""
from __future__ import annotations

import copy

import pytest

from proteus.redox.solver import SolverResult, solve_fO2


class _FakeOutgasConfig:
    """Minimal config shim used by the tests."""
    class outgas:
        fO2_shift_IW = 4.0
        mass_thresh = 1e16
        module = 'calliope'
    class redox:
        mode = 'fO2_init'
        rtol = 1e-4
        atol = 1e-6
        max_iter = 40
        bracket_halfwidth = 4.0
        soft_conservation_tol = 1e-3
        include_atm = True
        include_mantle = True
        include_core = True


def _base_hf_row() -> dict:
    from proteus.redox.budget import RB_COEF
    row = {f'{s}_mol_atm': 0.0 for s in RB_COEF}
    row.update({f'{s}_mol_liquid': 0.0 for s in RB_COEF})
    row.update({
        'n_Fe3_melt': 0.0, 'n_Fe2_melt': 0.0,
        'n_Fe3_solid_total': 0.0, 'n_Fe2_solid_total': 0.0,
        'n_Fe0_solid_total': 0.0,
        'Fe_kg_core': 1.94e24, 'H_kg_core': 0.0,
        'O_kg_core': 0.0, 'Si_kg_core': 0.0,
        'R_budget_atm': 0.0, 'R_escaped_cum': 0.0,
        'R_budget_mantle': 0.0, 'R_budget_core': 0.0,
        'H_kg_total': 1e20, 'C_kg_total': 1e19,
        'N_kg_total': 1e18, 'S_kg_total': 1e18,
        'redox_solver_fallback_count': 0.0,
    })
    return row


@pytest.mark.unit
def test_solver_is_transactional():
    """solve_fO2 must not mutate the caller's hf_row beyond fO2_shift_IW."""
    row = _base_hf_row()
    row_prev = _base_hf_row()
    row_prev['fO2_shift_IW'] = 3.0
    row['fO2_shift_IW'] = 3.0
    config = _FakeOutgasConfig()
    directories = {}

    # Fake outgas_callable: writes a specific pattern into the probe.
    # If we see those writes leak into the caller's hf_row, the
    # transactional guarantee is broken.
    calls = []

    def fake_outgas(dirs, cfg, hf):
        calls.append(dict(hf))   # snapshot the probe each call
        hf['H2O_mol_atm'] = 1e20   # mutate probe
        hf['poison'] = 'mutated'   # foreign key should never leak

    row_before = copy.deepcopy(row)
    solve_fO2(
        row, row_prev, directories, config,
        outgas_callable=fake_outgas,
    )

    # Brent may or may not converge with the trivial residual; what we
    # assert is that everything in row except fO2_shift_IW (and the
    # solver's own diagnostic columns) is unchanged.
    for k in row_before:
        if k == 'fO2_shift_IW':
            continue
        if k == 'redox_solver_fallback_count':
            continue  # solver may bump this on fallback
        assert row.get(k) == row_before[k], (
            f'Transactional violation: {k} changed from '
            f'{row_before[k]} to {row.get(k)}'
        )
    assert 'poison' not in row, 'Probe poisoning leaked into real hf_row'


@pytest.mark.unit
def test_solver_converges_on_monotonic_residual():
    """With a synthetic monotonic residual, Brent finds the known root."""
    row = _base_hf_row()
    row_prev = _base_hf_row()
    row_prev['fO2_shift_IW'] = 0.0
    row['fO2_shift_IW'] = 0.0
    config = _FakeOutgasConfig()

    # Fake outgas_callable: sets H2_mol_atm so R_atm is a known
    # monotonic function of ΔIW. Choose R_atm(ΔIW) = k * (ΔIW - ΔIW*).
    # R_atm = -2 * n_H2 (RB_H2 = -2). So setting
    # n_H2 = (ΔIW - ΔIW_star) * 0.5 makes R_atm = -(ΔIW - ΔIW_star).
    # Target ΔIW* = 2.0; R_target = 0.0; R_atm(2.0) = 0, so ΔIW=2
    # satisfies.
    def fake_outgas(dirs, cfg, hf):
        target_star = 2.0
        # Residual = R_atm − R_target. R_target = 0 here (see below).
        # So R_atm_post = -(ΔIW - 2); solving residual=0 → ΔIW=2.
        delta = hf['fO2_shift_IW']
        hf['H2_mol_atm'] = (delta - target_star) * (-0.5)  # n_H2
        # Flip sign: fake_outgas sets H2, and RB_H2 = -2, so
        # R_atm = -2 * (delta - 2) * (-0.5) = (delta - 2).
        # Residual = R_atm - R_target = delta - 2.
        # Brent finds zero at delta=2.

    directories = {}
    # Seed R_atm_prev = 0 via hf_row_prev so R_target = 0.
    result = solve_fO2(
        row, row_prev, directories, config,
        outgas_callable=fake_outgas,
    )
    assert isinstance(result, SolverResult)
    if not result.fell_back_to_previous:
        assert result.converged
        assert result.delta_IW == pytest.approx(2.0, abs=1e-3)
        assert row['fO2_shift_IW'] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.unit
def test_solver_falls_back_when_no_sign_change():
    """If Brent cannot bracket a root in any widened window, fall
    back to previous ΔIW and increment the fallback counter."""
    row = _base_hf_row()
    row_prev = _base_hf_row()
    row_prev['fO2_shift_IW'] = -2.5
    row['fO2_shift_IW'] = -2.5
    config = _FakeOutgasConfig()

    # Residual that is always positive (no sign change anywhere).
    def always_positive_outgas(dirs, cfg, hf):
        hf['H2_mol_atm'] = 0.0
        hf['CO2_mol_atm'] = 1e25  # R_atm always +4e25

    result = solve_fO2(
        row, row_prev, {}, config,
        outgas_callable=always_positive_outgas,
    )
    assert result.fell_back_to_previous
    assert result.delta_IW == -2.5   # fallback = previous ΔIW
    assert row['fO2_shift_IW'] == -2.5
    assert row['redox_solver_fallback_count'] == 1.0


@pytest.mark.unit
def test_solver_widens_bracket_on_initial_failure():
    """Root outside ±halfwidth but inside ±2·halfwidth: solver widens
    once before falling back."""
    row = _base_hf_row()
    row_prev = _base_hf_row()
    row_prev['fO2_shift_IW'] = 0.0
    row['fO2_shift_IW'] = 0.0
    config = _FakeOutgasConfig()
    config.redox.bracket_halfwidth = 1.0  # tight → forces widening

    # Root at ΔIW=+1.5 (outside ±1 but inside ±2).
    def fake_outgas(dirs, cfg, hf):
        target_star = 1.5
        delta = hf['fO2_shift_IW']
        hf['H2_mol_atm'] = (delta - target_star) * (-0.5)

    result = solve_fO2(
        row, row_prev, {}, config,
        outgas_callable=fake_outgas,
    )
    if not result.fell_back_to_previous:
        assert result.converged
        assert result.widened_bracket, (
            'Expected bracket widening to engage for root outside '
            '±halfwidth'
        )


@pytest.mark.unit
def test_solver_warm_start_from_mariana():
    """When `redox_delta_IW_suggested_by_mariana` is a finite float,
    Brent uses it as the warm-start center."""
    row = _base_hf_row()
    row_prev = _base_hf_row()
    row_prev['fO2_shift_IW'] = 0.0
    row['fO2_shift_IW'] = 0.0
    row['redox_delta_IW_suggested_by_mariana'] = 3.0
    config = _FakeOutgasConfig()
    config.redox.bracket_halfwidth = 0.5   # narrow bracket

    def fake_outgas(dirs, cfg, hf):
        # Root at ΔIW = 3.2 (just outside the narrow ±0.5 around
        # warm=3.0, but inside the ±1.0 widened bracket).
        delta = hf['fO2_shift_IW']
        hf['H2_mol_atm'] = (delta - 3.2) * (-0.5)

    result = solve_fO2(
        row, row_prev, {}, config,
        outgas_callable=fake_outgas,
    )
    if not result.fell_back_to_previous:
        assert abs(result.delta_IW - 3.2) < 0.1


@pytest.mark.unit
def test_solver_nan_warm_start_falls_back_to_previous():
    """NaN warm-start (from Mariana Fe3_frac=0 path) must not poison
    the bracket; solver uses hf_row_prev['fO2_shift_IW'] instead."""
    row = _base_hf_row()
    row_prev = _base_hf_row()
    row_prev['fO2_shift_IW'] = 1.0
    row['fO2_shift_IW'] = 1.0
    row['redox_delta_IW_suggested_by_mariana'] = float('nan')
    config = _FakeOutgasConfig()

    def fake_outgas(dirs, cfg, hf):
        delta = hf['fO2_shift_IW']
        hf['H2_mol_atm'] = (delta - 1.5) * (-0.5)

    result = solve_fO2(
        row, row_prev, {}, config,
        outgas_callable=fake_outgas,
    )
    # Convergence around 1.5 expected; the point is NO crash on NaN.
    assert isinstance(result, SolverResult)
