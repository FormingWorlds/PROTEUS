"""Unit tests for the liquidus_super initial-condition mode.

Covers:

(1) ``_resolve_zalmoxis_temperature_mode``: PROTEUS -> Zalmoxis mode
    mapping (liquidus_super -> adiabatic_from_cmb, accretion/isentropic
    -> adiabatic, others pass through).

(2) ``_resolve_zalmoxis_cmb_temperature``: liquidus_super delegates to the
    super-liquidus solve and returns the solved adiabat's CMB temperature;
    every other mode echoes ``tcmb_init`` verbatim.

(2b) ``solve_superliquidus_adiabat``: the solver finds the surface
    temperature (hence entropy) of the coolest adiabat that is fully molten
    everywhere with at least ``delta_T_super`` of superheat above the
    configured liquidus, and raises when that superheat is unreachable. The
    expensive PALEOS adiabat is mocked so the bracketing and error logic are
    unit-tested deterministically.

(3) ``load_zalmoxis_configuration`` end-to-end: liquidus_super propagates the
    solved cmb_temperature into the ``config_params`` dict consumed by
    ``zalmoxis.solver.main``.

The Fei+2021 / Belonoshko+2005 piecewise liquidus, still used to evaluate the
superheat, is:
    T = 1831 * (1 + P/4.6)**0.33      for P < 2.55 GPa  (Belonoshko+2005)
    T = 6000 * (P/140)**0.26          for P >= 2.55 GPa (Fei+2021)
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.fixture(autouse=True)
def _clear_superliquidus_solver_cache():
    """Drop the per-process super-liquidus solve cache around every test, so a
    cached result from one case (real or mocked) never leaks into another."""
    from proteus.interior_struct.zalmoxis import _clear_superliquidus_cache

    _clear_superliquidus_cache()
    yield
    _clear_superliquidus_cache()


# ----------------------------------------------------------------------
# (1) PROTEUS -> Zalmoxis temperature_mode mapping
# ----------------------------------------------------------------------


class TestResolveZalmoxisTemperatureMode:
    """``_resolve_zalmoxis_temperature_mode`` mapping rules."""

    def test_liquidus_super_maps_to_adiabatic_from_cmb(self):
        """PROTEUS ``liquidus_super`` mode maps to Zalmoxis
        ``adiabatic_from_cmb``. The PROTEUS side solves the fully-molten
        anchor temperature before handing the mode to Zalmoxis as a standard
        CMB adiabat.
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_temperature_mode,
        )

        assert _resolve_zalmoxis_temperature_mode('liquidus_super') == 'adiabatic_from_cmb'
        # Discrimination: the output must NOT pass the input through
        # verbatim. A regression that returned 'liquidus_super' to Zalmoxis
        # would fail in the structure solver downstream (Zalmoxis does not
        # recognise that token); pin the rename explicitly.
        assert _resolve_zalmoxis_temperature_mode('liquidus_super') != 'liquidus_super'

    @pytest.mark.parametrize('mode', ['accretion', 'isentropic'])
    def test_accretion_isentropic_collapse_to_adiabatic(self, mode):
        """PROTEUS ``accretion`` and ``isentropic`` both collapse to the
        Zalmoxis ``adiabatic`` mode (no separate CMB anchor).
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_temperature_mode,
        )

        out = _resolve_zalmoxis_temperature_mode(mode)
        assert out == 'adiabatic'
        # Discrimination: must NOT collapse to ``adiabatic_from_cmb``
        # (which would silently anchor at config.planet.tcmb_init even
        # for non-CMB modes) or pass the input string through verbatim.
        assert out != 'adiabatic_from_cmb'
        assert out != mode

    @pytest.mark.parametrize(
        'mode',
        ['isothermal', 'linear', 'adiabatic', 'adiabatic_from_cmb'],
    )
    def test_pass_through_unchanged(self, mode):
        """Modes that exist verbatim on the Zalmoxis side
        (``isothermal``, ``linear``, ``adiabatic``, ``adiabatic_from_cmb``)
        are passed through unchanged by the resolver.
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_temperature_mode,
        )

        out = _resolve_zalmoxis_temperature_mode(mode)
        assert out == mode
        # Discrimination: must not silently remap to the special
        # ``adiabatic_from_cmb`` token used by ``liquidus_super``. A
        # regression that defaulted everything to the CMB-anchored branch
        # would still pass an equality-only check for ``adiabatic_from_cmb``
        # but fail for the other three modes here.
        if mode != 'adiabatic_from_cmb':
            assert out != 'adiabatic_from_cmb'

    def test_unknown_mode_passes_through_unchanged(self):
        """Discriminator: validator (in _planet.py) is the source of
        truth for accept/reject; this helper must NOT add a second
        rejection point. Otherwise, adding a new mode would require
        two updates and silently mismap on miss.
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_temperature_mode,
        )

        out = _resolve_zalmoxis_temperature_mode('hypothetical_future_mode')
        assert out == 'hypothetical_future_mode'
        # Discrimination: the resolver must not silently coerce an unknown
        # mode to one of the two special remapped tokens. A regression that
        # added a fallthrough branch would land on one of these.
        assert out not in ('adiabatic', 'adiabatic_from_cmb')


# ----------------------------------------------------------------------
# (2) liquidus_super CMB temperature: gate and delegation to the solve
# ----------------------------------------------------------------------


def _make_minimal_config(
    delta_T_super=500.0,
    tcmb_init=6000.0,
    temperature_mode='liquidus_super',
):
    """Build a stub config exposing only fields the resolver reads."""
    cfg = MagicMock()
    cfg.planet.delta_T_super = delta_T_super
    cfg.planet.tcmb_init = tcmb_init
    cfg.planet.temperature_mode = temperature_mode
    return cfg


class TestResolveZalmoxisCMBTemperature:
    """Mode gate and delegation to ``solve_superliquidus_adiabat``."""

    def test_non_liquidus_super_returns_tcmb_init_verbatim(self):
        """For any non-liquidus_super mode the resolver echoes
        ``config.planet.tcmb_init`` verbatim and does not run the
        super-liquidus solve or consult ``hf_row['P_cmb']``.
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_cmb_temperature,
        )

        cfg = _make_minimal_config(tcmb_init=7199.0, temperature_mode='adiabatic_from_cmb')
        # hf_row carries a P_cmb value but it must NOT be consulted here.
        T = _resolve_zalmoxis_cmb_temperature(cfg, {'P_cmb': 135e9}, 'adiabatic_from_cmb')
        assert T == pytest.approx(7199.0)
        # Discrimination: must be exactly tcmb_init, not diverted into the
        # liquidus_super solve (which ignores tcmb_init and returns a much
        # hotter, fully-molten CMB temperature). Even a nearby-but-different
        # value would signal the gate failing.
        assert abs(T - 7199.0) < 1e-6

    def test_liquidus_super_returns_solved_cmb_temperature(self, monkeypatch):
        """liquidus_super delegates to ``solve_superliquidus_adiabat`` and
        returns the solved adiabat's CMB temperature, NOT ``tcmb_init`` and
        NOT the legacy liquidus-plus-offset anchor.
        """
        import proteus.interior_struct.zalmoxis as zmod

        sentinel = {
            'surface_T': 4241.0,
            'S_target': 10591.0,
            'cmb_T': 8765.0,
            'achieved_superheat': 500.0,
            'binding_P': 1.2e11,
            'P_cmb': 6.7e11,
        }
        seen = {}

        def fake_solve(config, hf_row):
            seen['hf_row'] = hf_row
            return sentinel

        monkeypatch.setattr(zmod, 'solve_superliquidus_adiabat', fake_solve)
        cfg = _make_minimal_config(delta_T_super=500.0, tcmb_init=6000.0)
        T = zmod._resolve_zalmoxis_cmb_temperature(cfg, {'P_cmb': 6.7e11}, 'liquidus_super')
        assert T == pytest.approx(8765.0)
        # The resolver must use the solve result, not tcmb_init.
        assert T != pytest.approx(6000.0)
        # Discrimination against the removed behaviour: the old anchor was
        # liquidus(670 GPa) + 500 ~ 9580 K, distinct from the solved 8765 K.
        legacy_anchor = 6000.0 * (670.0 / 140.0) ** 0.26 + 500.0
        assert abs(T - legacy_anchor) > 100.0
        # The helpfile row is forwarded to the solve unchanged.
        assert seen['hf_row'] == {'P_cmb': 6.7e11}


# ----------------------------------------------------------------------
# (2b) The super-liquidus solve itself (logic, with a mocked adiabat)
# ----------------------------------------------------------------------


_FAKE_BINDING_OFFSET = 3700.0  # synthetic: min superheat = T_surface - this


def _install_fake_solver_deps(monkeypatch, ceiling_T=4800.0):
    """Patch the solver's EOS dependencies with deterministic stubs.

    The synthetic adiabat is ``T(P) = (T_surface - 3700) + T_liq(P)``: monotone
    in pressure, with a uniform superheat above the liquidus of
    ``T_surface - 3700`` K (independent of the pressure range, so the same
    requested margin is reachable for any P_cmb). The offset places the
    delta_T_super = 500 K solution at T_surface ~ 4200 K, inside the solver's
    surface-temperature scan window. Above ``ceiling_T`` the deep half is set to
    NaN so the profile reads as EOS-table-exhausted, which is how the solver
    detects its upper limit. The real ``paleos_liquidus`` is used for the
    liquidus, so the superheat arithmetic is exercised without loading any
    PALEOS table.
    """
    import numpy as np
    import zalmoxis.eos_export as eos_export
    from zalmoxis.melting_curves import paleos_liquidus

    import proteus.interior_struct.zalmoxis as zmod

    monkeypatch.setattr(
        zmod,
        'load_zalmoxis_material_dictionaries',
        lambda: {'PALEOS:MgSiO3': {'eos_file': '/stub/eos.dat'}},
    )
    monkeypatch.setattr(zmod, 'resolve_2phase_mgsio3_paths', lambda *a, **k: (None, None))
    monkeypatch.setattr(zmod, 'load_zalmoxis_solidus_liquidus_functions', lambda *a, **k: None)

    def fake_adiabat(
        eos_file,
        T_surface,
        P_surface,
        P_cmb,
        n_points,
        solidus_func,
        liquidus_func,
        solid_eos_file,
        liquid_eos_file,
    ):
        P = np.linspace(P_surface, P_cmb, n_points)
        liq = np.asarray(liquidus_func(P), dtype=float)
        T = (T_surface - _FAKE_BINDING_OFFSET) + liq
        if T_surface > ceiling_T:
            T[n_points // 2 :] = np.nan  # EOS table exhausted at depth
        return {
            'P': P,
            'T': T,
            'S_target': float(T_surface),  # monotone proxy for entropy
            'S_profile': np.full(n_points, float(T_surface)),
        }

    monkeypatch.setattr(eos_export, 'compute_entropy_adiabat', fake_adiabat)
    return paleos_liquidus


class TestSolveSuperliquidusAdiabat:
    """Solver logic: it finds the adiabat with the requested superheat."""

    def _cfg(self, delta_T_super=500.0, mass_tot=1.0):
        cfg = MagicMock()
        cfg.planet.delta_T_super = delta_T_super
        cfg.planet.mass_tot = mass_tot
        cfg.interior_struct.core_frac = 0.325
        cfg.interior_struct.core_frac_mode = 'mass'
        cfg.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
        return cfg

    def test_solves_for_requested_superheat(self, monkeypatch):
        """The bisection lands the solve on the requested superheat margin.

        This exercises the scan/bracket/bisection CONTROL FLOW against a
        synthetic adiabat (min superheat = T_surf - 3700 by construction); it
        is not a physics invariant. The real-EOS numerical output is pinned in
        the slow-tier test_superliquidus_adiabat tests.
        """
        _install_fake_solver_deps(monkeypatch)
        from proteus.interior_struct.zalmoxis import solve_superliquidus_adiabat

        cfg = self._cfg(delta_T_super=500.0)
        res = solve_superliquidus_adiabat(cfg, {'P_cmb': 1.5e11})
        # The solver never under-melts (achieved superheat is at least the
        # requested margin) and lands close to it.
        assert res['achieved_superheat'] >= 500.0 - 1.0
        assert res['achieved_superheat'] == pytest.approx(500.0, abs=30.0)
        # In the synthetic model min superheat = T_surf - 3700, so delta pins
        # the solved surface temperature to ~4200 K.
        assert res['surface_T'] == pytest.approx(4200.0, abs=30.0)
        # Discrimination: a solver that ignored delta and returned the
        # marginally-molten adiabat would give surface_T ~3700 K (superheat ~0);
        # the solved adiabat must also be hotter at depth than at the surface.
        assert res['surface_T'] > 3900.0
        assert res['cmb_T'] > res['surface_T']

    def test_unachievable_superheat_raises(self, monkeypatch):
        """A superheat the synthetic table cannot support raises RuntimeError
        rather than silently returning a partially-solid initial condition, and
        the error reports the largest achievable superheat.
        """
        _install_fake_solver_deps(monkeypatch, ceiling_T=4800.0)
        from proteus.interior_struct.zalmoxis import solve_superliquidus_adiabat

        cfg = self._cfg(delta_T_super=6000.0)  # beyond the synthetic ceiling
        with pytest.raises(RuntimeError, match='cannot initialise a fully molten') as exc:
            solve_superliquidus_adiabat(cfg, {'P_cmb': 1.5e11})
        msg = str(exc.value)
        # The message must quote a concrete, achievable ceiling below the
        # unreachable 6000 K request (so the user knows what to lower to).
        assert 'largest achievable superheat' in msg
        ceiling = re.search(r'largest achievable superheat is (\d+) K', msg)
        assert ceiling is not None and 0.0 < float(ceiling.group(1)) < 6000.0

    def test_missing_p_cmb_uses_nl20_estimate(self, monkeypatch):
        """When hf_row lacks P_cmb the solve falls back to the
        Noack & Lasbleis (2020) mass-aware estimate, and carries that
        pressure into the result, instead of crashing.
        """
        _install_fake_solver_deps(monkeypatch)
        import proteus.utils.structure_estimate as se
        from proteus.interior_struct.zalmoxis import solve_superliquidus_adiabat

        seen = {}
        real_nl20 = se.estimate_P_cmb_NL20

        def spy(mass, core_frac, core_frac_mode):
            seen['p'] = real_nl20(mass, core_frac, core_frac_mode)
            return seen['p']

        monkeypatch.setattr(se, 'estimate_P_cmb_NL20', spy)
        cfg = self._cfg(delta_T_super=300.0, mass_tot=5.0)
        res = solve_superliquidus_adiabat(cfg, None)
        assert 'p' in seen  # NL20 was consulted
        assert res['P_cmb'] == pytest.approx(seen['p'])
        # A 5 M_Earth core-mantle pressure is far above the Earth-like 135 GPa.
        assert res['P_cmb'] > 4e11


# ----------------------------------------------------------------------
# (3) End-to-end plumbing through load_zalmoxis_configuration
# ----------------------------------------------------------------------


def _make_full_mock_config(
    temperature_mode='liquidus_super',
    delta_T_super=500.0,
    tsurf_init=4000.0,
    tcmb_init=6000.0,
):
    """Build a complete mock Config exercising the Zalmoxis-config builder."""
    config = MagicMock()
    config.planet.mass_tot = 1.0
    config.planet.tsurf_init = tsurf_init
    config.planet.tcmb_init = tcmb_init
    config.planet.tcenter_init = 5000.0
    config.planet.temperature_mode = temperature_mode
    config.planet.delta_T_super = delta_T_super
    config.interior_struct.core_frac = 0.325
    config.interior_struct.core_frac_mode = 'mass'
    config.interior_struct.zalmoxis.core_eos = 'PALEOS:iron'
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS-2phase:MgSiO3'
    config.interior_struct.zalmoxis.ice_layer_eos = None
    config.interior_struct.zalmoxis.mushy_zone_factor = 0.8
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.interior_struct.zalmoxis.num_levels = 100
    config.interior_struct.zalmoxis.solver_tol_outer = 3.0e-3
    config.interior_struct.zalmoxis.solver_tol_inner = 1.0e-4
    config.interior_struct.zalmoxis.solver_max_iter_outer = 100
    config.interior_struct.zalmoxis.solver_max_iter_inner = 100
    config.interior_struct.zalmoxis.use_jax = False
    config.interior_struct.zalmoxis.use_anderson = False
    config.interior_struct.zalmoxis.dry_mantle = True
    config.interior_struct.zalmoxis.outer_solver = 'newton'
    config.interior_struct.zalmoxis.newton_max_iter = 30
    config.interior_struct.zalmoxis.newton_tol = 1.0e-4
    config.interior_struct.zalmoxis.newton_relative_tolerance = 1.0e-9
    config.interior_struct.zalmoxis.newton_absolute_tolerance = 1.0e-10
    config.interior_struct.zalmoxis.global_miscibility = False
    return config


def _make_hf_row(P_cmb=None):
    row = {
        'M_volatiles': 0.0,
        'H_kg_total': 0.0,
        'C_kg_total': 0.0,
        'N_kg_total': 0.0,
        'S_kg_total': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_total': 0.0,
    }
    if P_cmb is not None:
        row['P_cmb'] = P_cmb
    return row


def _stub_target_surface_pressure(monkeypatch):
    import proteus.interior_struct.zalmoxis as _mod

    monkeypatch.setattr(_mod, '_get_target_surface_pressure', lambda *a, **kw: 1.0e5)


def _stub_solver(monkeypatch, cmb_T=8765.0):
    """Replace the expensive super-liquidus solve with a fixed result so the
    plumbing tests stay fast unit tests."""
    import proteus.interior_struct.zalmoxis as _mod

    monkeypatch.setattr(
        _mod,
        'solve_superliquidus_adiabat',
        lambda config, hf_row: {
            'surface_T': 4241.0,
            'S_target': 10591.0,
            'cmb_T': cmb_T,
            'achieved_superheat': 500.0,
            'binding_P': 1.2e11,
            'P_cmb': 6.7e11,
        },
    )


class TestLoadZalmoxisConfigurationLiquidusSuper:
    """Integration: liquidus_super flows through load_zalmoxis_configuration."""

    def test_temperature_mode_remapped_for_zalmoxis(self, monkeypatch):
        """PROTEUS liquidus_super -> Zalmoxis adiabatic_from_cmb, with the
        solved CMB temperature plumbed alongside the remap."""
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_full_mock_config(temperature_mode='liquidus_super')
        _stub_target_surface_pressure(monkeypatch)
        _stub_solver(monkeypatch, cmb_T=8765.0)
        cp = load_zalmoxis_configuration(config, _make_hf_row(P_cmb=135e9))
        assert cp['temperature_mode'] == 'adiabatic_from_cmb', (
            'liquidus_super must collapse to adiabatic_from_cmb on the '
            f'Zalmoxis structure side; got {cp["temperature_mode"]!r}'
        )
        # The cmb_temperature must be the solved anchor, not the raw token or
        # an unset placeholder. A regression that remapped the mode but left
        # cmb_temperature stale would fail this.
        assert cp['cmb_temperature'] == pytest.approx(8765.0)

    def test_cmb_temperature_overrides_tcmb_init(self, monkeypatch):
        """When mode=liquidus_super, cmb_temperature is the solved anchor,
        NOT config.planet.tcmb_init.
        """
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_full_mock_config(
            temperature_mode='liquidus_super',
            tcmb_init=6000.0,  # deliberately != the solved anchor
        )
        _stub_target_surface_pressure(monkeypatch)
        _stub_solver(monkeypatch, cmb_T=8765.0)
        cp = load_zalmoxis_configuration(config, _make_hf_row(P_cmb=135e9))
        assert cp['cmb_temperature'] == pytest.approx(8765.0)
        # Discrimination: the default tcmb_init must not leak through.
        assert cp['cmb_temperature'] != pytest.approx(6000.0)

    def test_cmb_temperature_unchanged_for_adiabatic_from_cmb(self, monkeypatch):
        """Backward compatibility: adiabatic_from_cmb must still echo
        config.planet.tcmb_init unchanged (the solve is not invoked).
        """
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_full_mock_config(
            temperature_mode='adiabatic_from_cmb',
            tcmb_init=7199.0,
        )
        _stub_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row(P_cmb=135e9))
        assert cp['cmb_temperature'] == pytest.approx(7199.0)
        assert cp['temperature_mode'] == 'adiabatic_from_cmb'

    def test_isentropic_unaffected_by_delta_T_super(self, monkeypatch):
        """Setting delta_T_super on a non-liquidus_super run must not change
        anything on the Zalmoxis side. Anti-happy-path: ensures the field
        does not leak into other modes.
        """
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_full_mock_config(
            temperature_mode='isentropic',
            delta_T_super=12345.0,  # absurd value to highlight any leak
            tcmb_init=4321.0,
        )
        _stub_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row(P_cmb=135e9))
        assert cp['temperature_mode'] == 'adiabatic'  # isentropic -> adiabatic
        assert cp['cmb_temperature'] == pytest.approx(4321.0)


# ----------------------------------------------------------------------
# Sanity check: paleos_liquidus is the source-of-truth function
# ----------------------------------------------------------------------


class TestPaleosLiquidusSourceOfTruth:
    """Pin the Zalmoxis paleos_liquidus values the IC solve evaluates.

    If Zalmoxis ever changes the Fei+2021 fit, these tests fail loudly and
    the IC mode must be re-validated.
    """

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_paleos_liquidus_135gpa(self):
        """Pin the Fei+2021 value at 135 GPa: T_liq ~ 5944 K.

        Hand-derivation: 6000 * (135/140)**0.26 = 6000 * 0.99059 ~ 5943.5.
        Tolerance is intentionally tight (1 K) so any drift in the fit
        constants (6000, 0.26, 140 GPa reference) is caught immediately.
        """
        from zalmoxis.melting_curves import paleos_liquidus

        T = float(paleos_liquidus(135e9))
        assert T == pytest.approx(5943.53, abs=1.0)
        # Sign guard (Section 3 positivity): liquidus temperature must
        # be positive Kelvin everywhere.
        assert T > 0.0
        # Exponent-error guard: a regression that swapped the 0.26
        # exponent for 0.5 would give 6000 * (135/140)**0.5 ~ 5892 K
        # (50 K below the correct 5944 K); the abs=1.0 tolerance
        # discriminates that even before this scale check fires, but
        # the explicit guard keeps the failure message readable.
        wrong_exp = 6000.0 * (135.0 / 140.0) ** 0.5
        assert abs(T - wrong_exp) > 30.0

    @pytest.mark.physics_invariant
    def test_paleos_liquidus_continuous_at_crossover(self):
        """Crossover at 2.5517 GPa: Belonoshko and Fei branches must meet."""
        from zalmoxis.melting_curves import paleos_liquidus

        T_below = float(paleos_liquidus(2.55e9))
        T_above = float(paleos_liquidus(2.56e9))
        # Both branches at the crossover should give ~1972 K (within 5 K).
        assert abs(T_above - T_below) < 5.0
        # Positivity and scale guard: both branches must produce
        # positive Kelvin temperatures in the documented ~1970 K
        # neighbourhood at the crossover. A regression that emitted
        # zero or negative on one side (e.g. branch dispatch error)
        # would fail this even when the difference happens to be small.
        assert T_below > 0.0
        assert T_above > 0.0
        assert 1500.0 < T_below < 2500.0

    @pytest.mark.physics_invariant
    def test_paleos_liquidus_monotonic(self):
        """Liquidus must increase with pressure across the magma-ocean range."""
        from zalmoxis.melting_curves import paleos_liquidus

        Ps = [1e9, 10e9, 50e9, 100e9, 135e9, 200e9, 400e9, 600e9]
        Ts = [float(paleos_liquidus(P)) for P in Ps]
        for i in range(1, len(Ts)):
            assert Ts[i] > Ts[i - 1], (
                f'Liquidus not monotonic at P={Ps[i]:.1e}: '
                f'T(P[i-1])={Ts[i - 1]:.0f} >= T(P[i])={Ts[i]:.0f}'
            )
        # Positivity guard across the full sampled range and minimum
        # scale separation between endpoints. The 1 GPa Belonoshko
        # value is ~1942 K and the 600 GPa Fei value is ~8417 K, so
        # the span must be > 5000 K. A regression that flattened the
        # fit (e.g. clipped exponent to zero) would still pass the
        # monotonicity check above as long as each step is tiny.
        for T in Ts:
            assert T > 0.0
        assert Ts[-1] - Ts[0] > 5000.0
