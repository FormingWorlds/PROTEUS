# Regression test: AW mesh vs Zalmoxis mesh for Earth-mass planet
#
# Purpose: Verify that for an Earth-mass planet, SPIDER with a
# Zalmoxis-derived mesh produces results consistent with SPIDER
# using its built-in Adams-Williamson mesh. The two pathways use
# different density profiles (AW parameterized vs. hydrostatic with
# tabulated EOS) but should yield broadly consistent interior evolution.
#
# Tolerances are generous because the meshes are genuinely different:
# - T_magma: within 10% (same initial entropy, similar adiabats)
# - Phi_global: both should be ~1.0 (fully molten at early times)
# - F_int: within 30% (sensitive to mesh geometry)
#
# Runtime: ~60-90s (two PROTEUS runs of ~30s each)
#
# Documentation:
# - docs/test_infrastructure.md
# - docs/test_categorization.md
#
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus


def _run_proteus(config_path, tmpdir, label, struct_module):
    """Run a short PROTEUS simulation and return the final helpfile row.

    Parameters
    ----------
    config_path : Path
        Path to the base TOML config file.
    tmpdir : str
        Temporary directory for output.
    label : str
        Subdirectory label for this run.
    struct_module : str
        Structure module to use ("self" or "zalmoxis").

    Returns
    -------
    pandas.Series
        Final row of the helpfile.
    """
    runner = Proteus(config_path=config_path)
    runner.config.params.out.path = str(Path(tmpdir) / label)
    runner.config.interior_struct.module = struct_module

    if struct_module == 'self':
        runner.config.interior_struct.zalmoxis = None

    runner.init_directories()
    runner.start(resume=False, offline=True)

    assert runner.hf_all is not None, f'{label}: helpfile should be created'
    assert len(runner.hf_all) > 1, f'{label}: should have > 1 row'
    return runner.hf_all.iloc[-1]


@pytest.mark.integration
def test_aw_vs_zalmoxis_earth_mass():
    """Compare AW and Zalmoxis mesh pathways for 1 M_earth SPIDER run.

    Both runs use the same initial entropy, atmosphere, and volatile
    inventory. The only difference is the static mesh source:
    - AW: SPIDER's internal Adams-Williamson parameterization
    - Zalmoxis: hydrostatic solver with Seager2007/WolfBower2018 EOS

    For Earth-mass planets, the AW and Zalmoxis density profiles are
    similar enough that interior evolution should be broadly consistent.

    Runtime: ~60-90s
    """
    config_path = PROTEUS_ROOT / 'input' / 'tests' / 'zalmoxis_spider.toml'

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run both pathways
        zalmoxis_row = _run_proteus(config_path, tmpdir, 'zalmoxis', 'zalmoxis')
        aw_row = _run_proteus(config_path, tmpdir, 'aw_self', 'self')

        # --- Both should produce physical results ---
        for label, row in [('AW', aw_row), ('Zalmoxis', zalmoxis_row)]:
            assert not np.isnan(row['T_magma']), f'{label}: T_magma is NaN'
            assert not np.isnan(row['Phi_global']), f'{label}: Phi_global is NaN'
            assert not np.isnan(row['F_int']), f'{label}: F_int is NaN'
            assert row['T_magma'] > 1000, f'{label}: T_magma too low'
            assert row['F_int'] > 0, f'{label}: F_int should be positive'

        # --- T_magma within 10% ---
        T_rel = abs(aw_row['T_magma'] - zalmoxis_row['T_magma']) / aw_row['T_magma']
        assert T_rel < 0.10, (
            f'T_magma differs by {T_rel:.1%}: '
            f'AW={aw_row["T_magma"]:.1f} K, Zalmoxis={zalmoxis_row["T_magma"]:.1f} K'
        )

        # --- Phi_global: both near 1.0 at early times ---
        assert aw_row['Phi_global'] == pytest.approx(1.0, abs=0.05), (
            f'AW Phi_global={aw_row["Phi_global"]:.4f}, expected ~1.0'
        )
        assert zalmoxis_row['Phi_global'] == pytest.approx(1.0, abs=0.05), (
            f'Zalmoxis Phi_global={zalmoxis_row["Phi_global"]:.4f}, expected ~1.0'
        )

        # --- F_int within 30% (sensitive to mesh details) ---
        F_rel = abs(aw_row['F_int'] - zalmoxis_row['F_int']) / aw_row['F_int']
        assert F_rel < 0.30, (
            f'F_int differs by {F_rel:.1%}: '
            f'AW={aw_row["F_int"]:.3e} W/m^2, Zalmoxis={zalmoxis_row["F_int"]:.3e} W/m^2'
        )

        # --- R_int should be Earth-like for both ---
        for label, row in [('AW', aw_row), ('Zalmoxis', zalmoxis_row)]:
            assert 4e6 <= row['R_int'] <= 10e6, (
                f'{label}: R_int={row["R_int"]:.3e} m, expected Earth-like'
            )
