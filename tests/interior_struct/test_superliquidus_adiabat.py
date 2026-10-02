"""Real-EOS validation of the super-liquidus initial-condition solver.

``solve_superliquidus_adiabat`` is physics code: it fixes the magma-ocean
initial condition for ``temperature_mode = 'liquidus_super'``. These tests run
the REAL PALEOS entropy adiabat and pin its output, the mass-independence of the
solved entropy, the clamp for an unreachable superheat, and the raise for a
liquidus the table cannot clear (the one test that shifts the liquidus). Each solve runs a surface-temperature scan plus bisection over the PALEOS
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


@pytest.fixture(scope='module', autouse=True)
def _two_phase_tables():
    """Fetch the tables the pinned solves read, and require the 2-phase pair on disk.

    The configs name the unified MgSiO3 mantle, whose fetch also gets the pair the
    adiabat is built from; the zero-superheat pin is for the pair.
    """
    from proteus.interior_struct.zalmoxis import (
        load_zalmoxis_material_dictionaries,
        resolve_2phase_mgsio3_paths,
    )
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS:MgSiO3', core_eos='PALEOS:iron')
    tables = resolve_2phase_mgsio3_paths('PALEOS:MgSiO3', load_zalmoxis_material_dictionaries())
    solid, liquid = tables
    assert solid and liquid, f'2-phase MgSiO3 tables missing from FWL_DATA: {solid=}, {liquid=}'


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
    @pytest.mark.timeout(5400)  # 90 min ceiling; a single solve measures ~47 min on the runner
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
    # Three real solves in series (1, 5, 10 M_Earth); this test measures ~141 min
    # on the runner, so it needs well above the file-level 3600 s net.
    @pytest.mark.timeout(14400)
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

    @pytest.mark.reference_pinned
    @pytest.mark.timeout(5400)  # 90 min ceiling; one solve, like test_m1_solved_adiabat_pinned
    def test_m1_zero_superheat_pinned(self):
        """Pin the 1 M_Earth PALEOS solve at delta_T_super = 0.

        The coarse scan here starts in a ~100 K band just above the surface
        liquidus where the adiabat is not monotonic (a near-liquidus kink of
        a few K), so the first scan point is invalid; the solve tolerates that
        one leading band and lands on the coolest fully molten adiabat,
        surface T = 3860.57 K.
        """
        from proteus.interior_struct.zalmoxis import solve_superliquidus_adiabat

        cfg = _cfg('S1_m1_dyn_IW4.toml')
        object.__setattr__(cfg.planet, 'delta_T_super', 0.0)
        r = solve_superliquidus_adiabat(cfg, {'P_cmb': 1.42e11})
        # The tolerance absorbs the surface-T bisection resolution (~0.1 K).
        assert r['surface_T'] == pytest.approx(3860.57, abs=1.0)
        assert 0.0 <= r['achieved_superheat'] < 1.0
        assert r['clamped'] is False

    @pytest.mark.timeout(5400)  # 90 min ceiling; about 3 min locally
    def test_unreachable_superheat_raises_real(self, monkeypatch):
        """A liquidus the EOS table cannot clear raises instead of returning a
        partially molten initial condition. The real PALEOS adiabat clears the
        PALEOS liquidus by at most ~1180 K, so the liquidus is raised by 1500 K;
        the hottest valid adiabat is then the table-ceiling one (surface T
        ~4775 K, as in the clamp test below), still below the liquidus.
        """
        import re

        import numpy as np

        from proteus.interior_struct.zalmoxis import (
            load_zalmoxis_solidus_liquidus_functions,
            solve_superliquidus_adiabat,
        )

        real = load_zalmoxis_solidus_liquidus_functions

        def _hot_liquidus(mantle_eos, config):
            sol, liq = real(mantle_eos, config)
            return sol, (lambda P: np.asarray(liq(P), dtype=float) + 1500.0)

        monkeypatch.setattr(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            _hot_liquidus,
        )
        cfg = _cfg('S1_m1_dyn_IW4.toml')
        object.__setattr__(cfg.planet, 'delta_T_super', 0.0)
        with pytest.raises(
            RuntimeError, match='no fully-molten initial condition is reachable'
        ) as exc:
            solve_superliquidus_adiabat(cfg, {'P_cmb': 1.42e11})
        msg = str(exc.value)
        assert 'below the liquidus' in msg
        assert re.search(r'surface T=47[67]\d K', msg), msg

    @pytest.mark.timeout(5400)  # 90 min ceiling; about 3 min locally
    def test_unreachable_superheat_clamps_real(self, caplog):
        """A superheat larger than the EOS table supports, on a molten
        adiabat, clamps to the largest achievable value (~1180 K, set by the
        table ceiling at the 1 bar surface), flags the result and warns that
        the table, not the search window, is the limit.
        """
        import logging

        from proteus.interior_struct.zalmoxis import solve_superliquidus_adiabat

        cfg = _cfg('S1_m10_dyn_IW4.toml')
        object.__setattr__(cfg.planet, 'delta_T_super', 5000.0)
        with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_struct.zalmoxis'):
            res = solve_superliquidus_adiabat(cfg, {'P_cmb': 1.474e12})
        assert res['clamped'] is True
        assert res['achieved_superheat'] == pytest.approx(1181.0, abs=20.0)
        assert res['surface_T'] == pytest.approx(4777.0, abs=20.0)
        msgs = [r.getMessage() for r in caplog.records]
        assert any('not reachable' in m and 'within the EOS table' in m for m in msgs), msgs

    @pytest.mark.physics_invariant
    @pytest.mark.timeout(9000)  # 150 min; ~82 min on the runner, 8-9 min locally
    def test_zalmoxis_structure_ic_on_own_p_s_tables(self, tmp_path):
        """With the Zalmoxis structure, the initial entropy is solved on the
        run's own P-S tables (generated from PALEOS for S1_m1_dyn_IW4): at
        delta_T_super = 0 and 500 K the adiabat is at least delta - 0.5 K
        above the tables' liquidus at every pressure from 1 bar to P_cmb, and
        Aragog's and SPIDER's IC entry points pass that entropy through
        unchanged.
        """
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import numpy as np
        from aragog.eos.entropy import EntropyEOS

        from proteus.interior_energetics.aragog import AragogRunner
        from proteus.interior_energetics.spider import _compute_spider_initial_entropy
        from proteus.interior_struct.zalmoxis import generate_spider_tables

        cfg = _cfg('S1_m1_dyn_IW4.toml')
        assert cfg.interior_struct.module == 'zalmoxis'
        tables = generate_spider_tables(cfg, str(tmp_path))
        eos = EntropyEOS(tables['eos_dir'])
        hf_row = {'P_cmb': 1.42e11, 'R_int': 6.4e6, 'R_core': 3.5e6}
        P = np.geomspace(1e5, hf_row['P_cmb'], 20000)
        T_liq = eos.temperature(P, eos.liquidus_entropy(P))

        for delta in (0.0, 500.0):
            object.__setattr__(cfg.planet, 'delta_T_super', delta)
            object.__setattr__(cfg.planet, 'ini_dsdr', 0.0)

            solver = MagicMock()
            solver._P_stag_flat = np.array([1e10, 5e10])
            solver._r_basic_flat = np.array([3.5e6, 5.0e6, 6.4e6])
            interior_o = SimpleNamespace(
                aragog_solver=solver, _spider_eos_dir=tables['eos_dir']
            )
            AragogRunner._set_entropy_ic(cfg, interior_o, str(tmp_path), hf_row)
            S_aragog = float(solver.set_initial_entropy.call_args[0][0][0])
            S_spider = _compute_spider_initial_entropy(cfg, hf_row, tables['eos_dir'])

            assert S_aragog == pytest.approx(S_spider, abs=1e-9)
            margin = eos.temperature(P, np.full_like(P, S_spider)) - T_liq
            assert np.all(np.isfinite(margin))
            assert margin.min() >= delta - 0.5, (delta, float(margin.min()))
            # Coolest such adiabat: the binding margin sits within 1 K of delta.
            assert margin.min() == pytest.approx(delta, abs=1.0)
