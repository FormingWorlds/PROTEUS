"""
Unit tests for E.1: NetCDF per-cell Fe / fO2 write path.

Covers `aragog.AragogRunner._write_output_ncdf` with the new
`redox_per_cell` kwarg, plus the `coupling.run_redox_step` hook that
populates `hf_row['_redox_per_cell']`.

See plan v6 §3.7 for the write-path contract and §3.1 for the
interior → redox ordering (iter N's snapshot carries iter N-1's
reservoir state).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import netCDF4 as nc
import numpy as np
import pytest


def _fake_solver_output(n_stag: int = 10, n_basic: int = 11):
    """Minimal SolverOutput stub with the fields _write_output_ncdf reads."""
    out = MagicMock()
    out.S_final = np.linspace(3900.0, 2000.0, n_stag)
    out.T_stag = np.linspace(4000.0, 2000.0, n_stag)
    out.phi_stag = np.linspace(1.0, 0.0, n_stag)
    out.r_stag = np.linspace(3.5e6, 6.37e6, n_stag)
    out.P_stag = np.linspace(135e9, 1e5, n_stag)
    out.r_basic = np.linspace(3.48e6, 6.37e6, n_basic)
    out.visc_stag = np.full(n_stag, 1e21)
    out.rho_stag = np.full(n_stag, 4500.0)
    out.heat_flux = np.full(n_basic, 100.0)
    out.heating = np.full(n_stag, 1e-12)
    out.mass_stag = np.full(n_stag, 1e22)
    out.Phi_global = 0.5
    return out


@pytest.mark.unit
def test_write_ncdf_without_redox(tmp_path: Path) -> None:
    """Baseline: no redox_per_cell → NetCDF has core vars only."""
    from proteus.interior_energetics.aragog import AragogRunner

    (tmp_path / 'data').mkdir()
    out = _fake_solver_output()

    AragogRunner._write_output_ncdf(
        output_dir=str(tmp_path),
        time=1000.0,
        out=out,
        redox_per_cell=None,
    )

    ds = nc.Dataset(str(tmp_path / 'data' / '1000_int.nc'))
    try:
        assert 'entropy_s' in ds.variables
        assert 'temp_s' in ds.variables
        assert 'phi_s' in ds.variables
        # New redox vars must be absent when redox_per_cell is None.
        assert 'n_Fe3_solid_cell' not in ds.variables
        assert 'n_Fe2_solid_cell' not in ds.variables
        assert 'log10_fO2_profile' not in ds.variables
    finally:
        ds.close()


@pytest.mark.unit
def test_write_ncdf_with_redox(tmp_path: Path) -> None:
    """Redox_per_cell populates three new staggered-dim vars with right units."""
    from proteus.interior_energetics.aragog import AragogRunner

    (tmp_path / 'data').mkdir()
    out = _fake_solver_output(n_stag=10)

    redox_per_cell = {
        'n_Fe3_solid_cell': np.full(10, 1.23e20),
        'n_Fe2_solid_cell': np.full(10, 4.56e20),
        'log10_fO2_profile': np.linspace(-12.0, -8.0, 10),
    }

    AragogRunner._write_output_ncdf(
        output_dir=str(tmp_path),
        time=2000.0,
        out=out,
        redox_per_cell=redox_per_cell,
    )

    ds = nc.Dataset(str(tmp_path / 'data' / '2000_int.nc'))
    try:
        assert 'n_Fe3_solid_cell' in ds.variables
        assert 'n_Fe2_solid_cell' in ds.variables
        assert 'log10_fO2_profile' in ds.variables

        assert ds['n_Fe3_solid_cell'][:].shape == (10,)
        assert ds['n_Fe2_solid_cell'][:].shape == (10,)
        assert ds['log10_fO2_profile'][:].shape == (10,)

        assert ds['n_Fe3_solid_cell'].units == 'mol'
        assert ds['n_Fe2_solid_cell'].units == 'mol'
        assert ds['log10_fO2_profile'].units == 'log10 bar'

        np.testing.assert_allclose(
            ds['n_Fe3_solid_cell'][:], np.full(10, 1.23e20))
        np.testing.assert_allclose(
            ds['n_Fe2_solid_cell'][:], np.full(10, 4.56e20))
        np.testing.assert_allclose(
            ds['log10_fO2_profile'][:], np.linspace(-12.0, -8.0, 10))
    finally:
        ds.close()


@pytest.mark.unit
def test_write_ncdf_shape_mismatch_raises(tmp_path: Path) -> None:
    """Wrong array length is a hard error, not silent truncation."""
    from proteus.interior_energetics.aragog import AragogRunner

    (tmp_path / 'data').mkdir()
    out = _fake_solver_output(n_stag=10)

    bad = {
        'n_Fe3_solid_cell': np.zeros(7),
        'n_Fe2_solid_cell': np.zeros(10),
        'log10_fO2_profile': np.zeros(10),
    }

    with pytest.raises(ValueError, match='expected.*10'):
        AragogRunner._write_output_ncdf(
            output_dir=str(tmp_path),
            time=3000.0,
            out=out,
            redox_per_cell=bad,
        )


@pytest.mark.unit
def test_write_ncdf_partial_redox_dict(tmp_path: Path) -> None:
    """Missing keys in redox_per_cell skip those vars without erroring."""
    from proteus.interior_energetics.aragog import AragogRunner

    (tmp_path / 'data').mkdir()
    out = _fake_solver_output(n_stag=10)

    partial = {'log10_fO2_profile': np.linspace(-10.0, -8.0, 10)}

    AragogRunner._write_output_ncdf(
        output_dir=str(tmp_path),
        time=4000.0,
        out=out,
        redox_per_cell=partial,
    )

    ds = nc.Dataset(str(tmp_path / 'data' / '4000_int.nc'))
    try:
        assert 'log10_fO2_profile' in ds.variables
        assert 'n_Fe3_solid_cell' not in ds.variables
        assert 'n_Fe2_solid_cell' not in ds.variables
    finally:
        ds.close()


@pytest.mark.unit
def test_coupling_populates_redox_per_cell_in_fO2_init_mode(tmp_path: Path) -> None:
    """
    run_redox_step must stash `_redox_per_cell` on hf_row when the
    evolving path runs, so the next Aragog write picks it up.
    """
    from proteus.interior_energetics.aragog import AragogMeshState
    from proteus.redox.coupling import run_redox_step

    # Minimal config mock with redox.mode='fO2_init' path active.
    config = MagicMock()
    config.redox.mode = 'fO2_init'
    config.redox.oxybarometer = 'schaefer2024'
    config.redox.phi_crit = 0.4
    config.redox.rtol = 1e-3
    config.redox.atol = 1e-4
    config.redox.bracket_halfwidth = 4.0
    config.redox.max_iter = 20
    config.redox.soft_conservation_tol = 1e-3
    config.redox.include_atm = True
    config.redox.include_mantle = True
    config.redox.include_core = True
    config.redox.buffer = 'IW'
    config.outgas.fO2_shift_IW = 4.0
    config.interior_struct.mantle_comp.FeO_total_wt = 7.82
    config.interior_struct.mantle_comp.Fe3_frac = 0.10
    config.interior_struct.mantle_comp.SiO2_wt = 45.5
    config.interior_struct.mantle_comp.Al2O3_wt = 4.49
    config.interior_struct.mantle_comp.CaO_wt = 3.58
    config.interior_struct.mantle_comp.Na2O_wt = 0.36
    config.interior_struct.mantle_comp.K2O_wt = 0.029
    config.interior_struct.mantle_comp.MgO_wt = 38.3
    config.interior_struct.mantle_comp.TiO2_wt = 0.21
    config.interior_struct.mantle_comp.P2O5_wt = 0.021

    n_stag = 12
    mesh_state = AragogMeshState(
        pressure_profile=np.linspace(135e9, 1e5, n_stag),
        temperature_profile=np.linspace(4000.0, 1800.0, n_stag),
        melt_fraction_profile=np.linspace(0.0, 1.0, n_stag),
        melt_fraction_profile_prev=np.linspace(0.0, 1.0, n_stag),
        cell_mass_profile=np.full(n_stag, 1e23),
        n_Fe3_solid_cell_prev=np.zeros(n_stag),
        n_Fe2_solid_cell_prev=np.zeros(n_stag),
    )

    hf_row = {
        'Time': 0.0,
        'fO2_shift_IW': 4.0,
        'R_budget_atm': 0.0,
        'R_budget_mantle': 0.0,
        'R_budget_core': 0.0,
        'R_budget_total': 0.0,
        'R_escaped_cum': 0.0,
        'Fe3_frac': 0.10,
        'n_Fe3_melt': 1.0e20,
        'n_Fe2_melt': 9.0e20,
        'dm_Fe0_to_core_cum': 0.0,
        'redox_solver_fallback_count': 0.0,
        'redox_conservation_residual': 0.0,
        'M_int': 4.0e24,
        'Fe_kg_core': 1.87e24,
        'M_mantle': 4.0e24,
    }
    hf_row_prev = dict(hf_row)

    # Best-effort call; solver may raise but we want to see whether the
    # _redox_per_cell dict is populated before the solver runs.
    try:
        run_redox_step(
            hf_row=hf_row,
            hf_row_prev=hf_row_prev,
            mesh_state=mesh_state,
            config=config,
            dirs={'output': str(tmp_path)},
            escape_fluxes_species_kg=None,
        )
    except Exception:
        # Solver path may fail in this isolated unit-test fixture.
        # The contract we test: _redox_per_cell is set regardless.
        pass

    assert '_redox_per_cell' in hf_row, (
        '_redox_per_cell must be stashed by run_redox_step so the next '
        'Aragog NetCDF write can pick it up'
    )
    per_cell = hf_row['_redox_per_cell']
    assert 'n_Fe3_solid_cell' in per_cell
    assert 'n_Fe2_solid_cell' in per_cell
    assert 'log10_fO2_profile' in per_cell
    assert per_cell['n_Fe3_solid_cell'].shape == (n_stag,)
    assert per_cell['n_Fe2_solid_cell'].shape == (n_stag,)
    assert per_cell['log10_fO2_profile'].shape == (n_stag,)
