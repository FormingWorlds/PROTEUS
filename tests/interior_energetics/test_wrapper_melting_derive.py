"""Unit tests for v2.1 P-S melting-curve derivation in
proteus.interior_energetics.wrapper.

Tests the ``_load_spider_ps_phase_table`` reader,
``_derive_ps_melting_curve`` inverter, and ``_override_melting_curves_from_pt``
glue function added 2026-04-29 to close the v2-paper bookkeeping leak
where WB17 runs configured with ``melting_dir = "PALEOS-Fei2021"`` were
silently using WB+2018 P-S curves at runtime.

See finding_2026_04_29_v2_melting_curve_mismatch.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from proteus.interior_energetics.wrapper import (
    _derive_ps_melting_curve,
    _load_spider_ps_phase_table,
    _override_melting_curves_from_pt,
)


def _write_synthetic_ps_phase_table(
    path: Path,
    P_grid: np.ndarray,
    S_grid: np.ndarray,
    T_func,
    P_scale: float = 1.0e9,
    S_scale: float = 4824266.84604467,
    val_scale: float = 1.0,
) -> None:
    """Write a synthetic SPIDER P-S phase table for testing.

    Layout follows SPIDER convention: P inner, S outer; 5 header lines.
    """
    nP = len(P_grid)
    nS = len(S_grid)
    with open(path, 'w') as f:
        f.write(f'# 5 {nP} {nS}\n')
        f.write('# Pressure, Entropy, Quantity\n')
        f.write('# column * scaling factor should be SI units\n')
        f.write('# scaling factors (constant) for each column given on line below\n')
        f.write(f'# {P_scale} {S_scale} {val_scale}\n')
        for s in S_grid:
            for p in P_grid:
                T_val = T_func(p, s)
                f.write(f'{p / P_scale:.18e}\t{s / S_scale:.18e}\t{T_val / val_scale:.18e}\n')


def _write_pt_curve(path: Path, P: np.ndarray, T: np.ndarray) -> None:
    arr = np.column_stack([P, T])
    np.savetxt(path, arr, fmt='%.18e', header='Pressure_Pa Temperature_K')


@pytest.mark.unit
def test_load_spider_ps_phase_table_roundtrip(tmp_path):
    """Synthesised P-S phase table loads with correct shape and SI scaling."""
    P_grid = np.array([0.0, 1e10, 5e10, 1e11, 5e11, 1e12])
    S_grid = np.array([100.0, 500.0, 1000.0, 2000.0, 3000.0])

    def T_func(p, s):
        return 300.0 + 0.5 * s + 1e-9 * p

    table_path = tmp_path / 'temperature_solid.dat'
    _write_synthetic_ps_phase_table(table_path, P_grid, S_grid, T_func)

    P_loaded, S_loaded, T_loaded = _load_spider_ps_phase_table(str(table_path))

    np.testing.assert_allclose(P_loaded, P_grid)
    np.testing.assert_allclose(S_loaded, S_grid)
    assert T_loaded.shape == (len(P_grid), len(S_grid))

    for i, p in enumerate(P_grid):
        for j, s in enumerate(S_grid):
            np.testing.assert_allclose(T_loaded[i, j], T_func(p, s), rtol=1e-12)


@pytest.mark.unit
def test_derive_ps_melting_curve_recovers_target_temperature(tmp_path):
    """Round-trip: derive S(P) from P-T, then check T_grid(P, S_derived) == T_target."""
    P_grid = np.linspace(0, 1e12, 200)
    S_grid = np.linspace(100.0, 3000.0, 100)

    def T_func(p, s):
        # Linear T(P, S) for analytic invertibility
        return 1500.0 + 1.5 * s + 2e-9 * p

    phase_path = tmp_path / 'temperature_solid.dat'
    _write_synthetic_ps_phase_table(phase_path, P_grid, S_grid, T_func)

    P_target = np.linspace(1e9, 9e11, 50)
    # T_func produces T(P, S_min=100) = 1650 + 2e-9*P, T(P, S_max=3000) = 6000 + 2e-9*P.
    # Pick T_target = 4000 + 1e-9*P; well inside the grid for all P.
    T_target = 4000.0 + 1e-9 * P_target
    pt_path = tmp_path / 'solidus_P-T.dat'
    _write_pt_curve(pt_path, P_target, T_target)

    out_path = tmp_path / 'solidus_P-S.dat'
    summary = _derive_ps_melting_curve(
        str(pt_path),
        str(phase_path),
        str(out_path),
        label='test_solidus',
    )

    assert summary['n_points'] == len(P_target)
    assert summary['n_clipped_below'] == 0
    assert summary['n_clipped_above'] == 0
    assert summary['max_inversion_residual_K'] < 1e-6

    # Now verify by re-reading the output and walking through the grid
    arr = np.loadtxt(out_path, comments='#')
    P_scale, S_scale = 1.0e9, 4824266.84604467
    P_out = arr[:, 0] * P_scale
    S_out = arr[:, 1] * S_scale

    # Each (P_out[i], S_out[i]) should map back to T_target[i] under T_func
    for i, P in enumerate(P_target):
        np.testing.assert_allclose(P_out[i], P, rtol=1e-9)
        T_back = T_func(P_out[i], S_out[i])
        np.testing.assert_allclose(T_back, T_target[i], atol=1.0)  # < 1 K target


@pytest.mark.unit
def test_derive_ps_melting_curve_clips_above_grid(tmp_path, caplog):
    """T_target above EoS T ceiling clips to S_max with warning."""
    P_grid = np.linspace(0, 1e12, 200)
    S_grid = np.linspace(100.0, 3000.0, 100)

    def T_func(p, s):
        return 1500.0 + 1.5 * s + 2e-9 * p

    # T_max(P) = 1500 + 1.5*3000 + 2e-9*P = 6000 + small. Pick T_target > 6500 K.

    phase_path = tmp_path / 'temperature_melt.dat'
    _write_synthetic_ps_phase_table(phase_path, P_grid, S_grid, T_func)

    # T_max(P) = 6000 + 2e-9*P; at P=9e11, T_max=7800. Pick T_target=8500
    # (above grid ceiling for all P up to 1 TPa).
    P_target = np.linspace(1e9, 9e11, 10)
    T_target = np.full_like(P_target, 8500.0)
    pt_path = tmp_path / 'liquidus_P-T.dat'
    _write_pt_curve(pt_path, P_target, T_target)

    out_path = tmp_path / 'liquidus_P-S.dat'
    import logging

    with caplog.at_level(logging.WARNING):
        summary = _derive_ps_melting_curve(
            str(pt_path),
            str(phase_path),
            str(out_path),
            label='test_clipped',
        )
    assert summary['n_clipped_above'] == len(P_target)
    assert summary['n_clipped_below'] == 0
    assert any('clipped' in rec.message.lower() for rec in caplog.records)

    # All derived S should be at S_max
    arr = np.loadtxt(out_path, comments='#')
    S_scale = 4824266.84604467
    S_out = arr[:, 1] * S_scale
    np.testing.assert_allclose(S_out, S_grid[-1])


@pytest.mark.unit
def test_override_melting_curves_writes_files(tmp_path):
    """override_melting_curves_from_pt writes both solidus_P-S and liquidus_P-S."""
    P_grid = np.linspace(0, 1e12, 100)
    S_grid = np.linspace(100.0, 3000.0, 50)

    def T_sol_func(p, s):
        return 1000.0 + 1.5 * s + 2e-9 * p

    def T_liq_func(p, s):
        return 2000.0 + 1.5 * s + 2e-9 * p

    eos_dir = tmp_path / 'spider_eos'
    eos_dir.mkdir()
    _write_synthetic_ps_phase_table(
        eos_dir / 'temperature_solid.dat', P_grid, S_grid, T_sol_func
    )
    _write_synthetic_ps_phase_table(
        eos_dir / 'temperature_melt.dat', P_grid, S_grid, T_liq_func
    )

    P_target = np.linspace(1e9, 9e11, 20)
    sol_pt = tmp_path / 'solidus_PT.dat'
    liq_pt = tmp_path / 'liquidus_PT.dat'
    _write_pt_curve(sol_pt, P_target, np.full_like(P_target, 3000.0))
    _write_pt_curve(liq_pt, P_target, np.full_like(P_target, 4000.0))

    _override_melting_curves_from_pt(
        str(eos_dir),
        str(sol_pt),
        str(liq_pt),
        label_prefix='unittest',
    )

    assert (eos_dir / 'solidus_P-S.dat').is_file()
    assert (eos_dir / 'liquidus_P-S.dat').is_file()

    # Derived T_back at the recovered S should equal target
    sol_arr = np.loadtxt(eos_dir / 'solidus_P-S.dat', comments='#')
    P_scale, S_scale = 1.0e9, 4824266.84604467
    P_sol = sol_arr[:, 0] * P_scale
    S_sol = sol_arr[:, 1] * S_scale
    for P_i, S_i in zip(P_sol, S_sol):
        np.testing.assert_allclose(T_sol_func(P_i, S_i), 3000.0, atol=1.0)


@pytest.mark.unit
def test_override_melting_curves_skips_when_phase_tables_missing(tmp_path, caplog):
    """If temperature_{solid,melt}.dat are absent, override is a no-op + warns."""
    eos_dir = tmp_path / 'spider_eos_empty'
    eos_dir.mkdir()
    sol_pt = tmp_path / 'solidus_PT.dat'
    liq_pt = tmp_path / 'liquidus_PT.dat'
    _write_pt_curve(sol_pt, np.array([1e10, 1e11]), np.array([3000.0, 4000.0]))
    _write_pt_curve(liq_pt, np.array([1e10, 1e11]), np.array([4000.0, 5000.0]))

    import logging

    with caplog.at_level(logging.WARNING):
        _override_melting_curves_from_pt(
            str(eos_dir),
            str(sol_pt),
            str(liq_pt),
            label_prefix='nofiles',
        )

    assert not (eos_dir / 'solidus_P-S.dat').is_file()
    assert any('missing temperature' in rec.message.lower() for rec in caplog.records)
