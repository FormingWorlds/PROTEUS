"""
Unit tests for E.2 static-mode contract.

These are fast pure-Python tests that lock the invariant: in
``redox.mode = 'static'``, ``run_redox_step`` MUST NOT modify
``hf_row['fO2_shift_IW']`` beyond pinning it to ``config.outgas.fO2_shift_IW``,
and MUST NOT call the Brent solver. Full A/B regression against the
baseline branch lives in ``scripts/regression_static_mode.py`` and is
documented in ``docs/redox.md``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_static_config(fO2_shift_IW: float = 4.0):
    """Config mock with redox.mode='static' and CHILI-canonical fO2."""
    config = MagicMock()
    config.redox.mode = 'static'
    config.redox.include_atm = True
    config.redox.include_mantle = True
    config.redox.include_core = True
    config.outgas.fO2_shift_IW = fO2_shift_IW
    # CHILI Earth pyrolite (plan v6 §3.5).
    config.interior_struct.mantle_comp.FeO_total_wt = 7.82
    config.interior_struct.mantle_comp.Fe3_frac = 0.10
    config.interior_struct.mantle_comp.SiO2_wt = 45.5
    config.interior_struct.mantle_comp.Al2O3_wt = 4.49
    config.interior_struct.mantle_comp.MgO_wt = 38.3
    config.interior_struct.mantle_comp.CaO_wt = 3.58
    config.interior_struct.mantle_comp.Na2O_wt = 0.36
    config.interior_struct.mantle_comp.K2O_wt = 0.029
    config.interior_struct.mantle_comp.TiO2_wt = 0.21
    config.interior_struct.mantle_comp.P2O5_wt = 0.021
    return config


def _seed_hf_row(fO2_shift_IW: float = 4.0) -> dict:
    """Minimum hf_row a static-mode pass writes back into."""
    return {
        'Time': 0.0,
        'fO2_shift_IW': fO2_shift_IW,
        'R_budget_atm': 0.0,
        'R_budget_mantle': 0.0,
        'R_budget_core': 0.0,
        'R_budget_total': 0.0,
        'R_escaped_cum': 0.0,
        'Fe3_frac': 0.10,
        'n_Fe3_melt': 0.0,
        'n_Fe2_melt': 0.0,
        'dm_Fe0_to_core_cum': 0.0,
        'redox_solver_fallback_count': 0.0,
        'redox_conservation_residual': 0.0,
        'M_int': 4.0e24,
        'Fe_kg_core': 1.87e24,
        'M_mantle': 4.0e24,
    }


@pytest.mark.unit
def test_static_mode_pins_fO2_from_config() -> None:
    """Static mode overwrites fO2_shift_IW with config.outgas.fO2_shift_IW."""
    from proteus.redox.coupling import run_redox_step

    config = _make_static_config(fO2_shift_IW=4.0)
    hf_row = _seed_hf_row(fO2_shift_IW=0.0)    # mismatched on purpose
    hf_row_prev = dict(hf_row)

    run_redox_step(
        hf_row=hf_row,
        hf_row_prev=hf_row_prev,
        mesh_state=None,
        config=config,
        dirs={'output': '/tmp'},
        escape_fluxes_species_kg=None,
    )

    assert hf_row['fO2_shift_IW'] == 4.0, (
        'static mode must pin fO2_shift_IW to config.outgas.fO2_shift_IW'
    )


@pytest.mark.unit
def test_static_mode_does_not_call_solver() -> None:
    """Static path must not enter the Brent solver, period."""
    from proteus.redox import coupling

    config = _make_static_config()
    hf_row = _seed_hf_row()
    hf_row_prev = dict(hf_row)

    with patch('proteus.redox.solver.solve_fO2') as mock_solver:
        coupling.run_redox_step(
            hf_row=hf_row,
            hf_row_prev=hf_row_prev,
            mesh_state=None,
            config=config,
            dirs={'output': '/tmp'},
            escape_fluxes_species_kg=None,
        )

    mock_solver.assert_not_called()


@pytest.mark.unit
def test_static_mode_does_not_debit_escape() -> None:
    """Static path must not call debit_escape even if fluxes are present."""
    from proteus.redox import coupling

    config = _make_static_config()
    hf_row = _seed_hf_row()
    hf_row_prev = dict(hf_row)

    escape_fluxes = {'H2O': 1.0e15, 'H2': 5.0e14}    # non-empty on purpose

    with patch('proteus.redox.conservation.debit_escape') as mock_debit:
        coupling.run_redox_step(
            hf_row=hf_row,
            hf_row_prev=hf_row_prev,
            mesh_state=None,
            config=config,
            dirs={'output': '/tmp'},
            escape_fluxes_species_kg=escape_fluxes,
        )

    mock_debit.assert_not_called()


@pytest.mark.unit
def test_static_mode_does_not_populate_redox_per_cell() -> None:
    """
    `_redox_per_cell` is only written on the evolving path (`fO2_init` /
    `composition`). In static mode it must stay absent so the NetCDF
    writer falls through cleanly.
    """
    import numpy as np

    from proteus.interior_energetics.aragog import AragogMeshState
    from proteus.redox.coupling import run_redox_step

    config = _make_static_config()
    hf_row = _seed_hf_row()
    hf_row_prev = dict(hf_row)

    n_stag = 10
    mesh_state = AragogMeshState(
        pressure_profile=np.linspace(135e9, 1e5, n_stag),
        temperature_profile=np.linspace(4000.0, 1800.0, n_stag),
        melt_fraction_profile=np.linspace(0.0, 1.0, n_stag),
        melt_fraction_profile_prev=np.linspace(0.0, 1.0, n_stag),
        cell_mass_profile=np.full(n_stag, 1e23),
        n_Fe3_solid_cell_prev=np.zeros(n_stag),
        n_Fe2_solid_cell_prev=np.zeros(n_stag),
    )

    run_redox_step(
        hf_row=hf_row,
        hf_row_prev=hf_row_prev,
        mesh_state=mesh_state,
        config=config,
        dirs={'output': '/tmp'},
        escape_fluxes_species_kg=None,
    )

    assert '_redox_per_cell' not in hf_row, (
        'static mode must not populate _redox_per_cell'
    )


@pytest.mark.unit
def test_static_mode_writes_passive_diagnostics() -> None:
    """
    Plan v6 §3.6: static mode must still populate the R_budget_* and
    Fe3_frac diagnostic columns so the helpfile schema is consistent
    across modes. Only the solver path is skipped.
    """
    from proteus.redox.coupling import run_redox_step

    config = _make_static_config()
    hf_row = _seed_hf_row()
    # Add enough atm inventory that passive R_budget_atm is non-zero.
    hf_row['H2O_mol_atm'] = 1.0e20
    hf_row['CO2_mol_atm'] = 1.0e19
    hf_row_prev = dict(hf_row)

    run_redox_step(
        hf_row=hf_row,
        hf_row_prev=hf_row_prev,
        mesh_state=None,
        config=config,
        dirs={'output': '/tmp'},
        escape_fluxes_species_kg=None,
    )

    # Passive R_budget_atm from CO2 (+4 coef) and H2O (0 coef).
    # Expect R_atm ≈ 4 × 1e19 = 4e19.
    assert 'R_budget_atm' in hf_row
    assert abs(hf_row['R_budget_atm'] - 4.0e19) < 1.0e17, (
        f"Passive R_budget_atm = {hf_row['R_budget_atm']:.3e}, "
        'expected ~4e19 from CO2 inventory'
    )
