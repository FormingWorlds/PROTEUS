"""Unit tests for the liquidus_super initial-condition mode.

Covers:

(1) ``_resolve_zalmoxis_temperature_mode``: PROTEUS -> Zalmoxis mode
    mapping (liquidus_super -> adiabatic_from_cmb, accretion/isentropic
    -> adiabatic, others pass through).

(2) ``_resolve_zalmoxis_cmb_temperature``: Fei+2021 liquidus + delta_T_super
    arithmetic at the converged P_cmb (or 135 GPa fallback when hf_row
    has not yet been populated).

(3) ``load_zalmoxis_configuration`` end-to-end: liquidus_super propagates
    the right cmb_temperature into the ``config_params`` dict consumed by
    ``zalmoxis.solver.main``.

The Fei+2021 liquidus formula is:
    T = 1831 * (1 + P/4.6)**0.33      for P < 2.55 GPa  (Belonoshko+2005)
    T = 6000 * (P/140)**0.26          for P >= 2.55 GPa (Fei+2021)

Anchor values used in the tests below (from Zalmoxis melting_curves.py):

    P =   135 GPa -> T_liq ~ 5935 K   (1 M_E reference)
    P =   400 GPa -> T_liq ~ 7716 K   (3 M_E reference)
    P =   600 GPa -> T_liq ~ 8417 K   (super-Earth)
    P =     1 GPa -> T_liq ~ 1942 K   (low-pressure Belonoshko branch)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ----------------------------------------------------------------------
# (1) PROTEUS -> Zalmoxis temperature_mode mapping
# ----------------------------------------------------------------------


class TestResolveZalmoxisTemperatureMode:
    """``_resolve_zalmoxis_temperature_mode`` mapping rules."""

    def test_liquidus_super_maps_to_adiabatic_from_cmb(self):
        """PROTEUS ``liquidus_super`` mode maps to Zalmoxis
        ``adiabatic_from_cmb``. PROTEUS-side handles the liquidus anchor
        before handing the mode to Zalmoxis as a standard CMB adiabat.
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
# (2) liquidus_super CMB temperature anchor
# ----------------------------------------------------------------------


def _make_minimal_config(
    delta_T_super=500.0,
    tcmb_init=6000.0,
    temperature_mode='liquidus_super',
):
    """Build a stub config exposing only fields _resolve_zalmoxis_cmb_temperature reads."""
    cfg = MagicMock()
    cfg.planet.delta_T_super = delta_T_super
    cfg.planet.tcmb_init = tcmb_init
    cfg.planet.temperature_mode = temperature_mode
    return cfg


class TestResolveZalmoxisCMBTemperature:
    """Fei+2021 + delta_T_super arithmetic and fallback discipline."""

    def test_non_liquidus_super_returns_tcmb_init_verbatim(self):
        """For any non-liquidus_super mode, ``_resolve_zalmoxis_cmb_temperature``
        echoes ``config.planet.tcmb_init`` verbatim and does not consult
        ``hf_row['P_cmb']``. This pins the contract that the Fei+2021
        anchor logic is gated on the mode flag.
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_cmb_temperature,
        )

        cfg = _make_minimal_config(tcmb_init=7199.0, temperature_mode='adiabatic_from_cmb')
        # hf_row carries a P_cmb value but it must NOT be consulted in
        # non-liquidus_super modes.
        T = _resolve_zalmoxis_cmb_temperature(
            cfg,
            {'P_cmb': 135e9},
            'adiabatic_from_cmb',
        )
        assert T == pytest.approx(7199.0)
        # Discrimination: T must NOT be the Fei+2021 anchor at 135 GPa
        # (~5944 K). If the gate on the mode flag failed, the anchor
        # would override tcmb_init and the test would still see a
        # plausible Kelvin value but the wrong one.
        T_liq_135 = 6000.0 * (135.0 / 140.0) ** 0.26
        assert abs(T - T_liq_135) > 1000.0

    @pytest.mark.physics_invariant
    def test_liquidus_super_uses_hf_row_p_cmb_at_135_gpa(self):
        """At P=135 GPa, Fei+2021 gives T_liq ~ 5935 K. With
        delta_T_super=500 K, the anchor should be ~6435 K.

        Discriminating: tests at a real super-Earth pressure (135 GPa,
        not 1 Pa where all the constants disappear).
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_cmb_temperature,
        )

        cfg = _make_minimal_config(delta_T_super=500.0)
        T = _resolve_zalmoxis_cmb_temperature(
            cfg,
            {'P_cmb': 135e9},
            'liquidus_super',
        )
        # Fei+2021: 6000 * (135/140)**0.26 ~ 5935 K
        T_liq_expected = 6000.0 * (135.0 / 140.0) ** 0.26
        assert T == pytest.approx(T_liq_expected + 500.0, rel=1e-9)
        # Sign + scale guard: T must be positive Kelvin (Section 3 positivity)
        # and the offset must be additive, not multiplicative. A regression
        # that multiplied by (1 + delta_T_super) instead of adding 500 K
        # would put T at ~3e6 K, far outside the magma-ocean band.
        assert T > 0.0
        assert 5000.0 < T < 10000.0

    @pytest.mark.physics_invariant
    def test_liquidus_super_zero_offset_lands_on_liquidus(self):
        """delta_T_super = 0 K -> anchor exactly on the Fei liquidus.
        This is the boundary physical case (validator allows ge(0)).
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_cmb_temperature,
        )

        cfg = _make_minimal_config(delta_T_super=0.0)
        T = _resolve_zalmoxis_cmb_temperature(
            cfg,
            {'P_cmb': 135e9},
            'liquidus_super',
        )
        T_liq_expected = 6000.0 * (135.0 / 140.0) ** 0.26
        assert T == pytest.approx(T_liq_expected, rel=1e-9)
        # Discrimination: changing delta_T_super from 0 to 500 K must
        # shift T by exactly 500 K. A regression that ignored delta_T_super
        # (or applied it multiplicatively) would not produce that exact
        # additive shift.
        cfg_offset = _make_minimal_config(delta_T_super=500.0)
        T_offset = _resolve_zalmoxis_cmb_temperature(
            cfg_offset,
            {'P_cmb': 135e9},
            'liquidus_super',
        )
        assert T_offset - T == pytest.approx(500.0, rel=1e-9)

    @pytest.mark.physics_invariant
    def test_liquidus_super_super_earth_pressure(self):
        """At P=400 GPa (3 M_E typical), T_liq ~ 7716 K, anchor with
        delta=500 K ~ 8216 K. Verifies the high-pressure Fei+2021 branch
        gives a meaningfully different answer from the 135 GPa case.
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_cmb_temperature,
        )

        cfg = _make_minimal_config(delta_T_super=500.0)
        T = _resolve_zalmoxis_cmb_temperature(
            cfg,
            {'P_cmb': 400e9},
            'liquidus_super',
        )
        T_liq_expected = 6000.0 * (400.0 / 140.0) ** 0.26
        assert T == pytest.approx(T_liq_expected + 500.0, rel=1e-9)
        # Discriminator: 400 GPa branch must be strictly hotter than 135 GPa.
        T_135 = _resolve_zalmoxis_cmb_temperature(
            cfg,
            {'P_cmb': 135e9},
            'liquidus_super',
        )
        assert T > T_135

    @pytest.mark.physics_invariant
    def test_liquidus_super_low_pressure_belonoshko_branch(self):
        """At P=1 GPa, the piecewise fit drops to the Belonoshko+2005
        branch: T = 1831 * (1 + 1/4.6)**0.33 ~ 1942 K. With delta=500 K
        the anchor is ~2442 K.

        Discriminating: 1 GPa is below the 2.55 GPa crossover, so this
        test fails if we accidentally use the Fei branch on the whole
        domain. Fei at 1 GPa would give ~3148 K, which is very different.
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_cmb_temperature,
        )

        cfg = _make_minimal_config(delta_T_super=500.0)
        T = _resolve_zalmoxis_cmb_temperature(
            cfg,
            {'P_cmb': 1e9},
            'liquidus_super',
        )
        T_liq_expected = 1831.0 * (1.0 + 1.0 / 4.6) ** 0.33
        assert T == pytest.approx(T_liq_expected + 500.0, rel=1e-9)
        # Negative discrimination: must NOT match the Fei branch.
        T_fei_branch = 6000.0 * (1.0 / 140.0) ** 0.26 + 500.0
        assert abs(T - T_fei_branch) > 100.0

    @pytest.mark.parametrize(
        'hf_row', [None, {}, {'P_cmb': None}, {'P_cmb': 0.0}, {'P_cmb': -1e9}]
    )
    def test_liquidus_super_fallback_to_NL20_at_1me(self, hf_row):
        """When hf_row lacks a usable P_cmb at 1 M_Earth, fall back to
        the Noack & Lasbleis (2020) mass-aware estimate (~142 GPa for
        CMF=0.325 mass-mode), NOT the legacy hardcoded 135 GPa.

        Anti-happy-path: tests four ways P_cmb can be missing/unusable
        (None, empty dict, explicit None, 0.0, negative). All must
        collapse to the same NL20 fallback; otherwise an unset hf_row
        would silently feed Zalmoxis a meaningless tcmb.
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_cmb_temperature,
        )
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        cfg = _make_minimal_config(delta_T_super=500.0)
        # Make config look like a 1 M_Earth Earth-like planet so
        # the NL20 fallback matches the resolver path.
        cfg.planet.mass_tot = 1.0
        cfg.interior_struct.core_frac = 0.325
        cfg.interior_struct.core_frac_mode = 'mass'
        T = _resolve_zalmoxis_cmb_temperature(cfg, hf_row, 'liquidus_super')
        P_expected = estimate_P_cmb_NL20(1.0, 0.325, 'mass')
        T_liq_expected = 6000.0 * (P_expected / 1e9 / 140.0) ** 0.26
        assert T == pytest.approx(T_liq_expected + 500.0, rel=1e-9)
        # Discriminating: NL20 at 1 M_E is ~142 GPa, distinguishable
        # from the legacy 135 GPa fallback.
        T_legacy = 6000.0 * (135.0 / 140.0) ** 0.26 + 500.0
        assert abs(T - T_legacy) > 5.0, (
            f'Resolver still uses the legacy 135 GPa fallback (T={T:.1f}); '
            f'NL20 mass-aware T={T_liq_expected + 500.0:.1f}, legacy T={T_legacy:.1f}'
        )

    @pytest.mark.parametrize(
        'mass_tot,low_GPa,high_GPa',
        [
            (1.0, 130, 160),
            (3.0, 350, 480),
            (5.0, 550, 800),
            (10.0, 900, 2000),
        ],
    )
    def test_liquidus_super_fallback_scales_with_mass(
        self,
        mass_tot,
        low_GPa,
        high_GPa,
    ):
        """Super-Earth check: when hf_row['P_cmb'] is missing, the
        fallback P_cmb must scale with planet mass (3 M_E -> ~400 GPa,
        5 M_E -> ~600 GPa, 10 M_E -> ~1000+ GPa) and the resulting T_cmb
        anchor must reflect the higher pressure. The legacy hardcoded
        135 GPa fallback failed this check by misplacing the 5 M_E
        anchor by ~2700 K. Discriminating: a regression to the legacy
        constant would put T_cmb under 6500 K for every mass.
        """
        from proteus.interior_struct.zalmoxis import (
            _resolve_zalmoxis_cmb_temperature,
        )
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        cfg = _make_minimal_config(delta_T_super=500.0)
        cfg.planet.mass_tot = mass_tot
        cfg.interior_struct.core_frac = 0.325
        cfg.interior_struct.core_frac_mode = 'mass'
        T = _resolve_zalmoxis_cmb_temperature(cfg, None, 'liquidus_super')
        P_expected = estimate_P_cmb_NL20(mass_tot, 0.325, 'mass')
        # The fallback P_cmb must land in the documented physical band.
        assert low_GPa * 1e9 < P_expected < high_GPa * 1e9, (
            f'NL20 P_cmb at {mass_tot} M_E is {P_expected / 1e9:.1f} GPa, '
            f'outside the expected [{low_GPa}, {high_GPa}] GPa band'
        )
        T_liq_expected = 6000.0 * (P_expected / 1e9 / 140.0) ** 0.26
        assert T == pytest.approx(T_liq_expected + 500.0, rel=1e-9)
        # Regression guard: must NOT be the legacy 135 GPa anchor for
        # any super-Earth mass. The 1 M_E case is allowed to be close
        # but for mass > 2 M_E the anchor must differ by > 1000 K.
        if mass_tot > 2.0:
            T_legacy = 6000.0 * (135.0 / 140.0) ** 0.26 + 500.0
            assert T - T_legacy > 1000.0, (
                f'Resolver regressed to the legacy 135 GPa fallback at '
                f'{mass_tot} M_E (T={T:.0f}, legacy T={T_legacy:.0f}); '
                f'expected NL20 anchor near {T_liq_expected + 500.0:.0f}'
            )


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


class TestLoadZalmoxisConfigurationLiquidusSuper:
    """Integration: liquidus_super flows through load_zalmoxis_configuration."""

    def test_temperature_mode_remapped_for_zalmoxis(self, monkeypatch):
        """PROTEUS liquidus_super -> Zalmoxis adiabatic_from_cmb."""
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_full_mock_config(temperature_mode='liquidus_super')
        _stub_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row(P_cmb=135e9))
        assert cp['temperature_mode'] == 'adiabatic_from_cmb', (
            'liquidus_super must collapse to adiabatic_from_cmb on the '
            f'Zalmoxis structure side; got {cp["temperature_mode"]!r}'
        )
        # Discrimination: the cmb_temperature plumbed alongside the remap
        # must be the Fei-derived anchor (positive Kelvin in the magma-ocean
        # band), not the raw token or an unset placeholder. A regression
        # that remapped the mode but left cmb_temperature stale would fail
        # this positivity + scale guard.
        T_anchor = cp['cmb_temperature']
        assert T_anchor > 0.0
        assert 5000.0 < T_anchor < 10000.0

    def test_cmb_temperature_overrides_tcmb_init(self, monkeypatch):
        """When mode=liquidus_super, cmb_temperature is the Fei-derived
        anchor, NOT config.planet.tcmb_init.

        Discriminating: the default config tcmb_init=6000 K is intentionally
        100 K different from the expected liquidus_super anchor at
        P=135 GPa (~6435 K), so a buggy implementation that returns
        tcmb_init verbatim would fail this test.
        """
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_full_mock_config(
            temperature_mode='liquidus_super',
            delta_T_super=500.0,
            tcmb_init=6000.0,  # deliberately != Fei + 500 at 135 GPa
        )
        _stub_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row(P_cmb=135e9))
        T_liq_expected = 6000.0 * (135.0 / 140.0) ** 0.26
        assert cp['cmb_temperature'] == pytest.approx(
            T_liq_expected + 500.0,
            rel=1e-9,
        )
        assert cp['cmb_temperature'] != pytest.approx(6000.0)

    def test_cmb_temperature_unchanged_for_adiabatic_from_cmb(self, monkeypatch):
        """Backward compatibility: adiabatic_from_cmb must still echo
        config.planet.tcmb_init unchanged.
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

    def test_first_call_uses_NL20_fallback_when_p_cmb_missing(self, monkeypatch):
        """First call (P_cmb not yet populated) at 1 M_Earth: use the
        NL20 mass-aware fallback (~142 GPa) instead of the legacy
        hardcoded 135 GPa. The energetics IC step then re-derives the
        anchor against the converged Zalmoxis P_cmb, so this only
        matters for the very first structure call.
        """
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration
        from proteus.utils.structure_estimate import estimate_P_cmb_NL20

        config = _make_full_mock_config(
            temperature_mode='liquidus_super',
            delta_T_super=500.0,
        )
        # _make_full_mock_config sets mass_tot=1.0; explicitly stub
        # the structure-side fields the NL20 fallback now reads.
        config.interior_struct.core_frac = 0.325
        config.interior_struct.core_frac_mode = 'mass'
        _stub_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row(P_cmb=None))
        P_expected = estimate_P_cmb_NL20(1.0, 0.325, 'mass')
        T_liq_expected = 6000.0 * (P_expected / 1e9 / 140.0) ** 0.26
        assert cp['cmb_temperature'] == pytest.approx(
            T_liq_expected + 500.0,
            rel=1e-9,
        )
        # Discrimination: the fallback must differ from the legacy
        # hardcoded 135 GPa anchor by > 5 K at 1 M_E (NL20 lands at
        # ~142 GPa). A regression that re-introduced the legacy
        # constant would collapse the two anchors.
        T_legacy = 6000.0 * (135.0 / 140.0) ** 0.26 + 500.0
        assert abs(cp['cmb_temperature'] - T_legacy) > 5.0

    def test_isentropic_unaffected_by_delta_T_super(self, monkeypatch):
        """Setting delta_T_super on a non-liquidus_super run must not
        change anything on the Zalmoxis side. Anti-happy-path: ensures
        the new field doesn't leak into other modes.
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
    """Pin the Zalmoxis paleos_liquidus values used as the IC anchor.

    If Zalmoxis ever changes the Fei+2021 fit, these tests fail loudly
    and the IC mode must be re-validated.
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
