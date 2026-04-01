# Regression test: periodic Zalmoxis structure re-computation
#
# Purpose: Verify that the Phase 2 feedback loop (SPIDER T(r) ->
# Zalmoxis prescribed mode -> updated mesh) produces stable, continuous
# evolution without crashes or wild oscillations.
#
# Validates:
# - R_int stays in a physically reasonable range throughout
# - M_int stays Earth-like (within 10% of target) despite T-dependent
#   EOS changing density with each structure update
# - T_magma is continuous (no jumps > 200 K between steps)
# - At least one structure update actually occurred
# - Time progresses normally
#
# Note: M_int is NOT expected to be exactly conserved because the
# WolfBower2018 T-dependent EOS produces different densities for
# different temperature profiles. The initial linear T(r) and SPIDER's
# evolved T(r) give ~5% different total masses for the same target.
#
# Runtime: ~5 min (structure updates add Zalmoxis re-solves per step)
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
from proteus.utils.constants import M_earth


@pytest.mark.integration
def test_structure_update_consistency():
    """Verify periodic Zalmoxis re-computation is stable and continuous.

    Runs a Zalmoxis+SPIDER simulation with update_interval=100 yr so the
    structure is re-computed multiple times during the ~4000 yr evolution.
    Checks that the simulation completes without crashes and that R_int,
    M_int, and T_magma stay in physically reasonable ranges.

    Runtime: ~5 min
    """
    config_path = PROTEUS_ROOT / 'input' / 'tests' / 'zalmoxis_spider.toml'

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Proteus(config_path=config_path)
        runner.config.params.out.path = str(Path(tmpdir) / 'output')

        # Enable structure updates every 100 yr
        runner.config.interior_struct.update_interval = 100.0

        runner.init_directories()
        runner.start(resume=False, offline=True)

        # --- Helpfile was created with multiple rows ---
        assert runner.hf_all is not None, 'Helpfile should be created'
        n_rows = len(runner.hf_all)
        assert n_rows > 3, f'Need > 3 rows for meaningful checks, got {n_rows}'

        hf = runner.hf_all

        # --- R_int stays Earth-like throughout ---
        R_int = hf['R_int'].values
        for i in range(len(R_int)):
            assert 4e6 <= R_int[i] <= 10e6, (
                f'R_int at step {i} = {R_int[i]:.3e} m, outside Earth-like range'
            )

        # --- M_int stays within 10% of target mass ---
        # The T-dependent EOS causes M_int to shift when the temperature
        # profile changes (linear init -> SPIDER evolved), so we check
        # against the target mass, not against the initial M_int.
        M_target = runner.config.planet.planet_mass_tot * M_earth
        M_int = hf['M_int'].values
        for i in range(len(M_int)):
            rel_dev = abs(M_int[i] - M_target) / M_target
            assert rel_dev < 0.10, (
                f'M_int at step {i} = {M_int[i]:.6e} kg, '
                f'{rel_dev:.1%} from target {M_target:.6e} kg'
            )

        # --- M_int is stable after structure updates settle ---
        # The first structure update (init T -> SPIDER T) causes a ~5%
        # M_int shift because the T-dependent EOS gives different densities.
        # After that transition, M_int should stabilize. Skip init + 2
        # steps to let the first update settle, then check for stability.
        init_iters = runner.config.params.stop.iters.minimum
        settle_offset = init_iters + 2
        settled = M_int[settle_offset:]
        if len(settled) > 1:
            for i in range(1, len(settled)):
                rel_jump = abs(settled[i] - settled[i - 1]) / settled[i - 1]
                assert rel_jump < 0.10, (
                    f'M_int jump at step {settle_offset + i}: '
                    f'{settled[i - 1]:.6e} -> {settled[i]:.6e} ({rel_jump:.1%})'
                )

        # --- T_magma is continuous (no jumps > 200 K between steps) ---
        T_magma = hf['T_magma'].values
        for i in range(1, len(T_magma)):
            dT = abs(T_magma[i] - T_magma[i - 1])
            assert dT < 200, (
                f'T_magma jump at step {i}: '
                f'{T_magma[i - 1]:.1f} -> {T_magma[i]:.1f} K (delta={dT:.1f} K)'
            )

        # --- No NaN or Inf in key columns ---
        for col in ['R_int', 'M_int', 'T_magma', 'F_int', 'Phi_global']:
            vals = hf[col].values
            assert not np.any(np.isnan(vals)), f'{col} contains NaN'
            assert not np.any(np.isinf(vals)), f'{col} contains Inf'

        # --- Time progressed ---
        assert hf['Time'].iloc[-1] > 0, 'Simulation time should have progressed'

        # --- At least one structure update occurred ---
        # R_int should not be constant if updates happened.
        R_range = R_int.max() - R_int.min()
        assert R_range > 100, (
            f'R_int range = {R_range:.1f} m, too small to confirm '
            'structure updates occurred (threshold: 100 m)'
        )

        # --- Final values are physical ---
        final = hf.iloc[-1]
        assert 1000 <= final['T_magma'] <= 5000, f'T_magma={final["T_magma"]:.1f} K'
        assert 0 <= final['Phi_global'] <= 1, f'Phi_global={final["Phi_global"]:.4f}'
        assert final['F_int'] > 0, f'F_int={final["F_int"]:.3e} should be positive'
