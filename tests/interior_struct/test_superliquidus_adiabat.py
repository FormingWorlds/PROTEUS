"""Real-EOS validation of the super-liquidus initial-condition solver.

``solve_superliquidus_adiabat`` is physics code: it fixes the magma-ocean
initial condition for ``temperature_mode = 'liquidus_super'``. These tests run
the REAL PALEOS entropy adiabat (no mocks) and pin its output, the
mass-independence of the solved entropy, and the unreachable-superheat error
path. Each solve runs a surface-temperature scan plus bisection over the PALEOS
adiabat, so they live in the slow nightly tier; the unit suite exercises the
solver control flow against a synthetic adiabat in ``test_liquidus_super_ic``.

See also: ``docs/How-to/testing.md``,
``docs/Explanations/test_framework.md``.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip('zalmoxis')

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

# Self-contained fixture configs for the pinned super-liquidus solves.
_GRID = os.path.join(os.path.dirname(__file__), 'data', 'superliquidus')


def _cfg(name: str):
    from proteus.config import read_config_object

    return read_config_object(os.path.join(_GRID, name))


@pytest.fixture(autouse=True)
def _clear_solver_cache():
    from proteus.interior_struct.zalmoxis import _clear_superliquidus_cache

    _clear_superliquidus_cache()
    yield
    _clear_superliquidus_cache()


class TestSolveSuperliquidusReal:
    """Pinned real-PALEOS behaviour of the super-liquidus solve."""

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_m1_solved_adiabat_pinned(self):
        """Pin the 1 M_Earth PALEOS solve (delta_T_super = 500 K).

        The surface temperature, the uniform initial entropy and the achieved
        superheat are pinned; the discrimination guards distinguish the solve
        from the removed CMB-liquidus anchor, which gave S ~ 10400 J/kg/K and a
        cold ~2922 K surface at high mass.
        """
        from proteus.interior_struct.zalmoxis import solve_superliquidus_adiabat

        r = solve_superliquidus_adiabat(_cfg('S1_m1_dyn_IW4.toml'), {'P_cmb': 1.42e11})
        # Pinned values (PALEOS MgSiO3 unified, delta_T_super = 500 K). The
        # tolerances absorb the surface-T bisection resolution.
        assert r['achieved_superheat'] == pytest.approx(500.0, abs=20.0)
        assert r['surface_T'] == pytest.approx(4241.0, abs=40.0)
        assert r['S_target'] == pytest.approx(10588.0, rel=3e-3)
        # Physics invariants: positive temperatures, fully molten (the adiabat
        # is hotter at depth), and the binding depth is below the Fei+2021
        # liquidus calibration (~500 GPa), so the superheat is not set against
        # an extrapolated liquidus.
        assert r['cmb_T'] > r['surface_T'] > 0.0
        assert r['binding_P'] < 500e9
        # Discrimination: a wrong (cold-surface CMB) anchor would put the
        # surface near ~2922 K; the solved surface must be far warmer.
        assert r['surface_T'] > 3500.0

    @pytest.mark.physics_invariant
    def test_solved_entropy_is_mass_independent(self):
        """The shallow PALEOS binding depth makes the solved entropy
        essentially mass-independent, keeping the mass grid on a common initial
        adiabat. Pin that 1, 5 and 10 M_Earth agree to better than 1 %.
        """
        from proteus.interior_struct.zalmoxis import (
            _clear_superliquidus_cache,
            solve_superliquidus_adiabat,
        )

        entropies = []
        for name, p_cmb in (
            ('S1_m1_dyn_IW4.toml', 1.42e11),
            ('S1_m5_dyn_IW4.toml', 6.73e11),
            ('S1_m10_dyn_IW4.toml', 1.474e12),
        ):
            _clear_superliquidus_cache()
            entropies.append(
                solve_superliquidus_adiabat(_cfg(name), {'P_cmb': p_cmb})['S_target']
            )
        spread = max(entropies) - min(entropies)
        assert spread < 0.01 * min(entropies), (
            f'solved entropy not mass-independent: {entropies}'
        )
        # Discrimination: the removed CMB anchor gave S falling with mass
        # (~10400, ~9650, ~7766), a ~25 % spread; mass-independence must hold to
        # well within that.
        assert spread < 0.05 * min(entropies)

    def test_unreachable_superheat_raises_real(self):
        """A superheat too large for the EOS table to support raises and names
        the largest achievable value, rather than returning a partial-melt IC.
        """
        from proteus.interior_struct.zalmoxis import solve_superliquidus_adiabat

        cfg = _cfg('S1_m10_dyn_IW4.toml')
        object.__setattr__(cfg.planet, 'delta_T_super', 5000.0)
        with pytest.raises(RuntimeError, match='cannot initialise a fully molten') as exc:
            solve_superliquidus_adiabat(cfg, {'P_cmb': 1.474e12})
        assert 'largest achievable superheat' in str(exc.value)
