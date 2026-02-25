# Smoke test: Zalmoxis structure + SPIDER interior coupling
#
# Purpose: Validate that Zalmoxis computes the planetary structure,
# writes a SPIDER-format mesh file, and SPIDER evolves the interior
# on the external mesh without errors. Uses dummy modules for
# atmosphere, star, and escape to keep runtime short.
#
# Tests validate:
# - Zalmoxis solver completes and sets R_int, M_int, gravity
# - SPIDER mesh file is written and used (dirs['spider_mesh'])
# - Interior evolution produces physical T_magma, Phi_global, F_int
# - No NaN or Inf values in outputs
#
# Runtime: ~30-60s (Zalmoxis structure solve + SPIDER init + 2-3 timesteps)
#
# Documentation:
# - docs/test_infrastructure.md
# - docs/test_categorization.md
#
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus


@pytest.mark.smoke
def test_smoke_zalmoxis_spider_coupling():
    """Test Zalmoxis structure + SPIDER interior end-to-end (few timesteps).

    Physical scenario: 1 M_earth planet with Zalmoxis computing the static
    structure (P, rho, g vs r) using the temperature-dependent silicate EOS,
    then SPIDER evolving the entropy field on the Zalmoxis-derived mesh.
    Dummy atmosphere provides a simple boundary condition.

    Validates:
    - Zalmoxis sets R_int, M_int, gravity in hf_row
    - SPIDER mesh file exists at dirs['spider_mesh']
    - T_magma is physical (1000-5000 K for magma ocean)
    - Phi_global is in [0, 1]
    - F_int is positive and finite
    - Time progresses beyond init stage

    Runtime: ~30-60s
    """
    config_path = PROTEUS_ROOT / 'input' / 'tests' / 'zalmoxis_spider.toml'

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Proteus(config_path=config_path)

        runner.config.params.out.path = str(Path(tmpdir) / 'output')
        runner.init_directories()

        runner.start(resume=False, offline=True)

        # --- Helpfile was created ---
        assert runner.hf_all is not None, 'Helpfile should be created'
        assert len(runner.hf_all) > 1, 'Should have more than 1 row (init + evolution)'

        final_row = runner.hf_all.iloc[-1]

        # --- Structure was solved (Zalmoxis sets these) ---
        assert 'R_int' in final_row
        R_int = final_row['R_int']
        assert not np.isnan(R_int), 'R_int should not be NaN'
        assert 4e6 <= R_int <= 10e6, f'R_int should be Earth-like (4-10 Mm), got {R_int:.3e}'

        assert 'M_int' in final_row
        M_int = final_row['M_int']
        assert not np.isnan(M_int), 'M_int should not be NaN'
        assert 1e24 <= M_int <= 1e26, (
            f'M_int should be Earth-like (1e24-1e26 kg), got {M_int:.3e}'
        )

        assert 'gravity' in final_row
        gravity = final_row['gravity']
        assert not np.isnan(gravity), 'gravity should not be NaN'
        assert 5 <= gravity <= 20, (
            f'Surface gravity should be Earth-like (5-20 m/s^2), got {gravity:.3f}'
        )

        # --- SPIDER mesh file was written ---
        spider_mesh = runner.directories.get('spider_mesh')
        assert spider_mesh is not None, 'dirs["spider_mesh"] should be set by Zalmoxis'
        assert os.path.isfile(spider_mesh), f'SPIDER mesh file should exist at {spider_mesh}'

        # --- Interior evolution produced physical results ---
        assert 'T_magma' in final_row
        T_magma = final_row['T_magma']
        assert not np.isnan(T_magma), 'T_magma should not be NaN'
        assert not np.isinf(T_magma), 'T_magma should not be Inf'
        assert 1000 <= T_magma <= 5000, (
            f'T_magma should be physical for magma ocean (1000-5000 K), got {T_magma:.1f}'
        )

        assert 'Phi_global' in final_row
        Phi_global = final_row['Phi_global']
        assert not np.isnan(Phi_global), 'Phi_global should not be NaN'
        assert 0 <= Phi_global <= 1, f'Phi_global should be in [0, 1], got {Phi_global:.4f}'

        assert 'F_int' in final_row
        F_int = final_row['F_int']
        assert not np.isnan(F_int), 'F_int should not be NaN'
        assert not np.isinf(F_int), 'F_int should not be Inf'
        assert F_int >= 0, f'F_int should be non-negative, got {F_int:.3e}'

        # --- Time progressed ---
        assert final_row['Time'] > 0, 'Simulation time should have progressed'
