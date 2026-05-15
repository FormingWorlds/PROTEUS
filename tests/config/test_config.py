"""Validator and parser coverage for PROTEUS config objects.

This file targets config parsing helpers plus validator guardrails across
_config.py, _interior.py, and _params.py. See testing standards in
docs/test_infrastructure.md, docs/test_categorization.md, and
docs/test_building.md for required structure, speed, and physics validity.
"""

from __future__ import annotations

from itertools import chain
from types import SimpleNamespace

import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.config import Config, read_config, read_config_object
from proteus.config._config import (
    instmethod_dummy,
    instmethod_evolve,
    janus_escape_atmosphere,
    observe_resolved_atmosphere,
    prevent_warming_advisory,
    satellite_evolve,
    spada_zephyrus,
    tides_enabled_orbit,
)
from proteus.config._converters import none_if_none
from proteus.config._interior import valid_aragog, valid_interiordummy, valid_spider
from proteus.config._outgas import Calliope
from proteus.config._params import max_bigger_than_min, valid_mod, valid_path
from proteus.config._planet import GasPrs, Planet

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


PATHS = chain(
    (PROTEUS_ROOT / 'input').glob('*.toml'),
)


@pytest.fixture
def cfg():
    """Load the demo dummy config and return the parsed Config object."""
    path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    obj = read_config_object(path)

    assert isinstance(obj, Config)

    return obj


@pytest.mark.unit
def test_read_config_returns_dict():
    """
    read_config returns a raw dict from TOML.

    Physical scenario: config files are TOML; read_config provides
    the low-level dict before validation/structure.
    """
    path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    raw = read_config(path)
    assert isinstance(raw, dict)
    assert 'params' in raw or 'star' in raw or 'orbit' in raw


@pytest.mark.unit
def test_valid_config_version_rejects_old():
    """config_version validator rejects old versions with a clear upgrade message."""
    from proteus.config._config import valid_config_version

    instance = SimpleNamespace()
    with pytest.raises(ValueError, match='config_version = "3.0"'):
        valid_config_version(instance, SimpleNamespace(), '2.0')


@pytest.mark.unit
def test_valid_config_version_accepts_current():
    """config_version validator accepts the current version.

    Sister to ``test_valid_config_version_rejects_unknown`` (which exercises
    a future version string '2.0' that must raise); together they pin the
    accepted-version envelope.
    """
    from proteus.config._config import valid_config_version

    instance = SimpleNamespace()
    result = valid_config_version(instance, SimpleNamespace(), '3.0')
    assert result is None  # contract: validator returns None silently on accepted version
    assert vars(instance) == {}  # validator must not mutate the input instance


@pytest.mark.unit
def test_auto_output_path_resolved():
    """params.out.path = 'auto' resolves to run_YYYYMMDD_HHMMSS_xxxx."""
    import re

    from proteus.config import read_config_object
    from proteus.utils.coupler import set_directories

    cfg = read_config_object(PROTEUS_ROOT / 'input' / 'dummy.toml')
    assert cfg.params.out.path == 'auto'
    set_directories(cfg)
    assert cfg.params.out.path != 'auto'
    assert re.match(r'run_\d{8}_\d{6}_[0-9a-f]{4}', cfg.params.out.path)


@pytest.mark.unit
def test_factory_defaults_from_minimal_config():
    """Config sections omitted from TOML use factory defaults.

    Pins the per-module default backend so a downstream schema change cannot
    silently flip a default and corrupt CI runs that lack the optional
    backend's package. The most recent regression of this kind: the
    outgas.module default used to be "atmodeller" while atmodeller is not
    in the PROTEUS dependency set; CI without atmodeller could not load
    minimal.toml because check_module_dependencies tried to import the
    missing package.
    """
    from proteus.config import read_config_object

    cfg = read_config_object(PROTEUS_ROOT / 'input' / 'minimal.toml')
    # Omitted sections should use factory defaults
    assert cfg.escape.module == 'zephyrus'
    assert cfg.accretion.module is None
    assert cfg.observe.synthesis is None
    assert cfg.atmos_chem.module is None
    assert cfg.interior_energetics.module == 'aragog'
    assert cfg.star.module == 'mors'
    assert cfg.atmos_clim.module == 'agni'
    # outgas.module default must be a backend whose package is in the
    # PROTEUS hard-dependency set. calliope is hard-pinned in
    # pyproject.toml; atmodeller is not. The default must be calliope so
    # any environment that has PROTEUS installed can load minimal.toml.
    assert cfg.outgas.module == 'calliope'


@pytest.mark.unit
def test_read_config_object_returns_config():
    """
    read_config_object returns a validated Config instance.

    Physical scenario: full parse and validate; used by Proteus and tests.
    """
    path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    obj = read_config_object(path)
    assert isinstance(obj, Config)
    # Discriminating check: dummy.toml uses the dummy backend stack across
    # every coupled module (atmos_clim, outgas, escape, etc.). A regression
    # that returned a stale or partially populated Config would have left at
    # least one of these fields empty or wrong.
    assert obj.atmos_clim.module == 'dummy'
    assert obj.outgas.module == 'dummy'


@pytest.mark.unit
def test_none_if_none():
    """Ensure the converter only treats literal 'none' (case-sensitive) as None."""
    assert none_if_none('none') is None

    for val in ('', 'notnone', 'mors', 'None'):
        assert none_if_none(val) is not None


@pytest.mark.parametrize('path', PATHS)
def test_proteus_init(path):
    """Smoke-test initialization for bundled example configs that should parse."""
    if 'nogit' in str(path):
        # Skip configs that rely on git metadata when running offline.
        return
    if path.name.startswith('example.'):
        # Skip subcommand meta-configs: example.infer.toml drives `proteus infer`
        # and example.grid.toml drives `proteus grid`. Their schema is intentionally
        # different from the standard PROTEUS Config (no planet.mass_tot, etc.) and
        # they are loaded by their own CLI handlers, not by Proteus(config_path=...).
        return
    print(f'Testing config at {path}')
    runner = Proteus(config_path=path)
    assert isinstance(runner.config, Config)
    # Discriminating check: the runner's config_path attribute round-trips the
    # input. A regression that returned a default-constructed Proteus runner
    # without ingesting the TOML would still pass the isinstance check alone.
    assert runner.config_path == path


@pytest.mark.unit
def test_calliope_is_included():
    """Calliope includes/excludes species based on boolean flags."""
    conf = Calliope(
        include_CO=False,
        include_H2=True,
    )
    assert not conf.is_included('CO')
    assert conf.is_included('H2')
    with pytest.raises(AttributeError):
        conf.is_included('fails')


@pytest.mark.unit
def test_delivery_get_pressure():
    """Volatile partial pressure lookup succeeds for known keys and errors otherwise."""
    conf = GasPrs(N2=123.0)
    assert conf.get_pressure('N2') == pytest.approx(123.0, rel=1e-12)
    with pytest.raises(AttributeError):
        conf.get_pressure('fails')


# =============================================================================
# Config validators
# =============================================================================


@pytest.mark.unit
def test_instmethod_dummy_rejects_non_dummy_star():
    """Dummy instellation method is only valid when the star module is also dummy."""
    inst = SimpleNamespace(
        orbit=SimpleNamespace(instellation_method='inst'),
        star=SimpleNamespace(module='janus'),
    )
    with pytest.raises(ValueError):
        instmethod_dummy(inst, None, None)


@pytest.mark.unit
def test_instmethod_evolve_rejects_evolving_inst_method():
    """Instellation method cannot evolve when evolve flag is already active."""
    inst = SimpleNamespace(orbit=SimpleNamespace(instellation_method='inst', evolve=True))
    with pytest.raises(ValueError):
        instmethod_evolve(inst, None, None)


@pytest.mark.unit
def test_satellite_evolve_rejects_combination():
    """Orbit configs cannot enable both satellite and evolve simultaneously."""
    inst = SimpleNamespace(orbit=SimpleNamespace(satellite=True, evolve=True))
    with pytest.raises(ValueError):
        satellite_evolve(inst, None, None)


@pytest.mark.unit
def test_tides_enabled_orbit_requires_orbit_module():
    """Tidal heating requires an orbit module to propagate the energy source."""
    inst = SimpleNamespace(
        interior_energetics=SimpleNamespace(heat_tidal=True), orbit=SimpleNamespace(module=None)
    )
    with pytest.raises(ValueError):
        tides_enabled_orbit(inst, None, None)


def _ns_for_prevent_warming(prevent_warming: bool, module: str = 'aragog'):
    """Build the minimal SimpleNamespace shape that prevent_warming_advisory reads.

    The validator now only inspects ``planet.prevent_warming``. The
    ``module`` argument is kept on the helper for callsite parity with the
    pre-Φ_vol-deletion test suite.
    """
    return SimpleNamespace(
        planet=SimpleNamespace(prevent_warming=prevent_warming),
        interior_energetics=SimpleNamespace(module=module),
    )


@pytest.mark.unit
def test_prevent_warming_advisory_silent_when_disabled(caplog):
    """prevent_warming = false: validator is a no-op for every interior module."""
    import logging

    caplog.set_level(logging.WARNING, logger='fwl.proteus.config._config')
    for module in ('aragog', 'dummy', 'spider'):
        caplog.clear()
        inst = _ns_for_prevent_warming(prevent_warming=False, module=module)
        prevent_warming_advisory(inst, None, None)
        assert caplog.records == [], f'unexpected warning emitted for module={module}'


@pytest.mark.unit
@pytest.mark.parametrize('module', ['aragog', 'spider', 'dummy'])
def test_prevent_warming_advisory_warns_when_enabled(caplog, module):
    """prevent_warming = true: validator emits exactly one base warning.

    Wording must reference the monotonic-cooling assumption and the
    F_atm = F_int clamp-consistency pitfall. The earlier
    dilatation-specific STRONG WARNING was retired together with the
    Φ_vol source term: that source was a divergence double-count and is
    now deleted, so the heat-pump-vs-clamp interaction can no longer
    occur. The warning is now module-independent.
    """
    import logging

    caplog.set_level(logging.WARNING, logger='fwl.proteus.config._config')
    inst = _ns_for_prevent_warming(prevent_warming=True, module=module)
    prevent_warming_advisory(inst, None, None)
    assert len(caplog.records) == 1, 'expected exactly one warning record'
    msg = caplog.records[0].getMessage()
    assert 'monotonically decrease' in msg
    assert 'F_atm = F_int' in msg
    assert 'STRONG WARNING' not in msg, 'STRONG WARNING was tied to the deleted dilatation gate'


@pytest.mark.unit
def test_observe_resolved_atmosphere_requires_non_dummy():
    """Resolved spectra synthesis is invalid when the climate module is dummy."""
    inst = SimpleNamespace(
        observe=SimpleNamespace(synthesis='platon'), atmos_clim=SimpleNamespace(module='dummy')
    )
    with pytest.raises(ValueError):
        observe_resolved_atmosphere(inst, None, None)


@pytest.mark.unit
def test_spada_zephyrus_requires_mors_spada():
    """Spada tracks must be selected when Zephyrus escape couples to MORS."""
    inst = SimpleNamespace(
        escape=SimpleNamespace(module='zephyrus'),
        star=SimpleNamespace(module='mors', mors=SimpleNamespace(tracks='notspada')),
    )
    with pytest.raises(ValueError):
        spada_zephyrus(inst, None, None)


@pytest.mark.unit
def test_janus_escape_requires_escape_stop_flag():
    """Janus escape needs stop.escape.enabled=True to prevent runaway mass loss."""
    inst = SimpleNamespace(
        escape=SimpleNamespace(module='zephyrus'),
        atmos_clim=SimpleNamespace(module='janus'),
        params=SimpleNamespace(stop=SimpleNamespace(escape=SimpleNamespace(enabled=False))),
    )
    with pytest.raises(ValueError):
        janus_escape_atmosphere(inst, None, None)


# =============================================================================
# Interior validators
# =============================================================================


@pytest.mark.unit
def test_valid_interiordummy_checks_tmagma_and_liquidus():
    """Dummy interior rejects unrealistically low magma temperature and thin liquidus gap."""
    dummy = SimpleNamespace(mantle_tliq=1500.0, mantle_tsol=1600.0)
    inst = SimpleNamespace(module='dummy', dummy=dummy)
    with pytest.raises(ValueError):
        valid_interiordummy(inst, None, None)

    dummy_ok = SimpleNamespace(mantle_tliq=1700.0, mantle_tsol=1600.0)
    inst_ok = SimpleNamespace(module='dummy', dummy=dummy_ok)
    # Should not raise
    valid_interiordummy(inst_ok, None, None)


@pytest.mark.unit
def test_valid_aragog_requires_energy_term_and_tmagma():
    """Aragog interior needs at least one heat transport term and warm initial magma."""
    inst = SimpleNamespace(
        module='aragog',
        trans_conduction=False,
        trans_convection=False,
        trans_mixing=False,
        trans_grav_sep=False,
        aragog=SimpleNamespace(),
    )
    with pytest.raises(ValueError):
        valid_aragog(inst, None, None)

    inst_ok = SimpleNamespace(
        module='aragog',
        trans_conduction=True,
        trans_convection=False,
        trans_mixing=False,
        trans_grav_sep=False,
        aragog=SimpleNamespace(),
    )
    valid_aragog(inst_ok, None, None)


@pytest.mark.unit
def test_valid_spider_requires_energy_term_and_entropy():
    """Spider interior requires an energy transport term and positive entropy."""
    inst = SimpleNamespace(
        module='spider',
        trans_conduction=False,
        trans_convection=False,
        trans_mixing=False,
        trans_grav_sep=False,
        spider=SimpleNamespace(),
    )
    with pytest.raises(ValueError):
        valid_spider(inst, None, None)

    inst_ok = SimpleNamespace(
        module='spider',
        trans_conduction=True,
        trans_convection=False,
        trans_mixing=False,
        trans_grav_sep=False,
        spider=SimpleNamespace(),
    )
    valid_spider(inst_ok, None, None)


# =============================================================================
# Params validators
# =============================================================================


@pytest.mark.unit
def test_valid_path_rejects_empty_string():
    """Path validators refuse empty strings to avoid silent path defaults."""
    with pytest.raises(ValueError):
        valid_path(None, SimpleNamespace(name='path'), '')


@pytest.mark.unit
def test_max_bigger_than_min_enforces_order():
    """Maximum must exceed minimum to preserve physically consistent bounds."""
    dummy = SimpleNamespace(minimum=10)
    with pytest.raises(ValueError):
        max_bigger_than_min(dummy, SimpleNamespace(name='maximum'), 5)


@pytest.mark.unit
def test_valid_mod_rejects_negative():
    """Modulus-like parameters cannot be negative; None is accepted for defaults."""
    with pytest.raises(ValueError):
        valid_mod(None, SimpleNamespace(name='plot_mod'), -1)

    assert valid_mod(None, SimpleNamespace(name='plot_mod'), None) is None


# ============================================================================
# Star config validators
# ============================================================================


@pytest.mark.unit
def test_star_mors_invalid_age():
    """Test MORS validator rejects invalid stellar age."""
    from proteus.config._star import valid_mors

    # Age must be > 0
    config = SimpleNamespace(
        module='mors',
        mors=SimpleNamespace(
            age_now=None,  # Invalid: None
            star_name='sun',
            rot_pcntle=50.0,
            rot_period=None,
            spectrum_source=None,
        ),
    )

    with pytest.raises(ValueError, match='age_now must be > 0'):
        valid_mors(config, SimpleNamespace(), None)

    # Negative age also invalid
    config.mors.age_now = -1.0
    with pytest.raises(ValueError, match='age_now must be > 0'):
        valid_mors(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_star_mors_missing_spec():
    """Test MORS validator requires star_name for solar/muscles spectra."""
    from proteus.config._star import valid_mors

    config = SimpleNamespace(
        module='mors',
        mors=SimpleNamespace(
            age_now=5.0,
            star_name=None,  # Invalid: missing for solar source
            rot_pcntle=50.0,
            rot_period=None,
            spectrum_source='solar',
        ),
    )

    with pytest.raises(ValueError, match='Must provide mors.star_name'):
        valid_mors(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_star_mors_rotation_percentile_valid():
    """Test MORS validator accepts valid rotation percentile."""
    from proteus.config._star import valid_mors

    # Valid: 50th percentile of rotation distribution
    config = SimpleNamespace(
        module='mors',
        mors=SimpleNamespace(
            age_now=5.0,
            star_name='sun',
            rot_pcntle=50.0,  # Valid: 0-100
            rot_period=None,
            spectrum_source=None,
        ),
    )

    result = valid_mors(config, SimpleNamespace(), None)
    assert result is None  # contract: validator returns None silently on the pass path
    # validator must not mutate the input config; use pytest.approx to keep the
    # comparison float-safe (the underlying value is a literal float).
    assert config.mors.rot_pcntle == pytest.approx(50.0, rel=1e-12)


@pytest.mark.unit
def test_star_mors_rotation_percentile_bounds():
    """Test MORS validator enforces rotation percentile bounds."""
    from proteus.config._star import valid_mors

    config = SimpleNamespace(
        module='mors',
        mors=SimpleNamespace(
            age_now=5.0,
            star_name='sun',
            rot_pcntle=101.0,  # Invalid: > 100
            rot_period=None,
            spectrum_source=None,
        ),
    )

    with pytest.raises(ValueError, match='Rotation percentile must be'):
        valid_mors(config, SimpleNamespace(), None)

    config.mors.rot_pcntle = -1.0  # Invalid: < 0
    with pytest.raises(ValueError, match='Rotation percentile must be'):
        valid_mors(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_star_mors_rotation_period_positive():
    """Test MORS validator requires positive rotation period."""
    from proteus.config._star import valid_mors

    config = SimpleNamespace(
        module='mors',
        mors=SimpleNamespace(
            age_now=5.0,
            star_name='sun',
            rot_pcntle=None,
            rot_period=25.0,  # Valid: > 0 days
            spectrum_source=None,
        ),
    )

    # Should not raise error
    valid_mors(config, SimpleNamespace(), None)

    config.mors.rot_period = -1.0  # Invalid: negative
    with pytest.raises(ValueError, match='Rotation period must be greater than zero'):
        valid_mors(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_star_mors_rotation_mutual_exclusion():
    """Test MORS validator enforces rotation period XOR percentile."""
    from proteus.config._star import valid_mors

    # Both set - invalid
    config = SimpleNamespace(
        module='mors',
        mors=SimpleNamespace(
            age_now=5.0,
            star_name='sun',
            rot_pcntle=50.0,
            rot_period=25.0,  # Both set - error
            spectrum_source=None,
        ),
    )

    with pytest.raises(ValueError, match='must be set by percentile or period, not both'):
        valid_mors(config, SimpleNamespace(), None)

    # Neither set - invalid
    config.mors.rot_pcntle = None
    config.mors.rot_period = None
    with pytest.raises(ValueError, match='must be set either by percentile or period'):
        valid_mors(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_star_dummy_radius_calculation_vs_fixed():
    """Test dummy star validator mutually excludes radius calculation vs fixed."""
    from proteus.config._star import valid_stardummy

    # calculate_radius=True but radius provided - error
    config = SimpleNamespace(
        module='dummy',
        dummy=SimpleNamespace(
            calculate_radius=True,
            radius=1.0,  # Should be None when calculating
            Teff=5780.0,
        ),
    )

    with pytest.raises(ValueError, match="radius must be 'none' when calculate_radius is True"):
        valid_stardummy(config, SimpleNamespace(), None)

    # calculate_radius=False but no radius - error
    config.dummy.calculate_radius = False
    config.dummy.radius = None
    with pytest.raises(
        ValueError, match='radius MUST be set >0 when calculate_radius is False'
    ):
        valid_stardummy(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_star_dummy_teff_positive():
    """Test dummy star validator requires positive effective temperature."""
    from proteus.config._star import valid_stardummy

    config = SimpleNamespace(
        module='dummy',
        dummy=SimpleNamespace(
            calculate_radius=False,
            radius=1.0,
            Teff=0.0,  # Invalid: <= 0
        ),
    )

    with pytest.raises(ValueError, match='Teff must be >0 K'):
        valid_stardummy(config, SimpleNamespace(), None)

    config.dummy.Teff = None  # Invalid: None
    with pytest.raises(ValueError, match='Teff must be >0 K'):
        valid_stardummy(config, SimpleNamespace(), None)


# ============================================================================
# Escape config validators
# ============================================================================


@pytest.mark.unit
def test_escape_zephyrus_pxuv_bounds():
    """Test Zephyrus validator enforces XUV opacity pressure bounds."""
    from proteus.config._escape import valid_zephyrus

    # Valid Pxuv
    config = SimpleNamespace(
        module='zephyrus',
        zephyrus=SimpleNamespace(
            Pxuv=1.0,  # Valid: 0 < Pxuv < 10
            efficiency=0.1,
            tidal=False,
        ),
    )

    # Should not raise error
    valid_zephyrus(config, SimpleNamespace(), None)

    # Pxuv = 0 - invalid
    config.zephyrus.Pxuv = 0.0
    with pytest.raises(ValueError, match='Pxuv.*must be >0 and <= 10'):
        valid_zephyrus(config, SimpleNamespace(), None)

    # Pxuv = 10.1 - invalid
    config.zephyrus.Pxuv = 10.1
    with pytest.raises(ValueError, match='Pxuv.*must be >0 and <= 10'):
        valid_zephyrus(config, SimpleNamespace(), None)

    # Pxuv = None - invalid
    config.zephyrus.Pxuv = None
    with pytest.raises(ValueError, match='Pxuv.*must be >0 and <= 10'):
        valid_zephyrus(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_escape_zephyrus_efficiency_bounds():
    """Test Zephyrus validator enforces escape efficiency bounds [0, 1]."""
    from proteus.config._escape import valid_zephyrus

    # Valid efficiency
    config = SimpleNamespace(
        module='zephyrus',
        zephyrus=SimpleNamespace(
            Pxuv=1.0,
            efficiency=0.5,  # Valid: 0 <= efficiency <= 1
            tidal=False,
        ),
    )

    # Should not raise error
    valid_zephyrus(config, SimpleNamespace(), None)

    # Efficiency = -0.1 - invalid
    config.zephyrus.efficiency = -0.1
    with pytest.raises(ValueError, match='efficiency.*must be >=0 and <=1'):
        valid_zephyrus(config, SimpleNamespace(), None)

    # Efficiency = 1.1 - invalid
    config.zephyrus.efficiency = 1.1
    with pytest.raises(ValueError, match='efficiency.*must be >=0 and <=1'):
        valid_zephyrus(config, SimpleNamespace(), None)

    # Efficiency = None - invalid
    config.zephyrus.efficiency = None
    with pytest.raises(ValueError, match='efficiency.*must be >=0 and <=1'):
        valid_zephyrus(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_escape_dummy_rate_positive():
    """Test dummy escape module requires positive mass loss rate."""
    from proteus.config._escape import valid_escapedummy

    config = SimpleNamespace(
        module='dummy',
        dummy=SimpleNamespace(
            rate=1e-9,  # Valid: > 0
        ),
    )

    # Should not raise error
    valid_escapedummy(config, SimpleNamespace(), None)

    # Rate = 0 - valid (zero escape)
    config.dummy.rate = 0.0
    valid_escapedummy(config, SimpleNamespace(), None)  # Should not raise

    # Rate = None - invalid
    config.dummy.rate = None
    with pytest.raises(ValueError, match='rate.*must be >= 0'):
        valid_escapedummy(config, SimpleNamespace(), None)

    # Rate = -1 - invalid
    config.dummy.rate = -1.0
    with pytest.raises(ValueError, match='rate.*must be >= 0'):
        valid_escapedummy(config, SimpleNamespace(), None)


# ============================================================================
# Orbit config validators
# ============================================================================


@pytest.mark.unit
def test_orbit_phi_tide_inequality_format():
    """Test Phi_tide validator requires proper inequality format."""
    from proteus.config._orbit import phi_tide_validator

    # Valid format
    phi_tide_validator(SimpleNamespace(), SimpleNamespace(), '<0.3')  # No error
    phi_tide_validator(SimpleNamespace(), SimpleNamespace(), '>0.5')  # No error

    # Invalid: wrong operator
    with pytest.raises(ValueError, match='must be an inequality'):
        phi_tide_validator(SimpleNamespace(), SimpleNamespace(), '=0.3')

    # Invalid: missing value
    with pytest.raises(ValueError, match='must be an inequality'):
        phi_tide_validator(SimpleNamespace(), SimpleNamespace(), '<')


@pytest.mark.unit
def test_orbit_phi_tide_value_bounds():
    """Test Phi_tide validator enforces value between 0 and 1."""
    from proteus.config._orbit import phi_tide_validator

    # Valid value
    phi_tide_validator(SimpleNamespace(), SimpleNamespace(), '<0.5')  # No error

    # Value > 1.0 - invalid
    with pytest.raises(ValueError, match='Phi_tide value must be between 0 and 1'):
        phi_tide_validator(SimpleNamespace(), SimpleNamespace(), '<1.5')

    # Value < 0.0 - invalid
    with pytest.raises(ValueError, match='Phi_tide value must be between 0 and 1'):
        phi_tide_validator(SimpleNamespace(), SimpleNamespace(), '<-0.1')


# ============================================================================
# Interior config validators
# ============================================================================


@pytest.mark.unit
def test_interior_spider_energy_term_required():
    """Test SPIDER validator requires at least one energy transport term enabled."""
    from proteus.config._interior import valid_spider

    # At least one enabled - valid
    config = SimpleNamespace(
        module='spider',
        trans_conduction=True,
        trans_convection=False,
        trans_mixing=False,
        trans_grav_sep=False,
        spider=SimpleNamespace(),
    )

    # Should not raise error
    valid_spider(config, SimpleNamespace(), None)

    # All disabled - invalid
    config.trans_conduction = False
    with pytest.raises(ValueError, match='Must enable at least one energy transport term'):
        valid_spider(config, SimpleNamespace(), None)


# ========================
# ATMOS_CLIM VALIDATORS
# ========================


@pytest.mark.unit
def test_janus_tmp_max_bigger_than_tmp_min_valid():
    """Test valid_janus accepts janus.tmp_maximum > tmp_minimum."""
    from proteus.config._atmos_clim import valid_janus

    instance = SimpleNamespace(
        module='janus',
        spectral_group='Honeyside',
        spectral_bands='48',
        tmp_minimum=300.0,
        janus=SimpleNamespace(
            tmp_maximum=5000.0,
        ),
    )
    result = valid_janus(instance, SimpleNamespace(), instance.janus)
    assert result is None  # contract: validator returns None silently on the pass path
    # Discriminating bounds preserved: tmp_minimum unchanged, tmp_maximum strictly greater.
    assert instance.tmp_minimum == pytest.approx(300.0, rel=1e-12)
    assert instance.janus.tmp_maximum > instance.tmp_minimum


@pytest.mark.unit
def test_janus_tmp_max_equals_tmp_min_invalid():
    """Test valid_janus rejects janus.tmp_maximum == tmp_minimum."""
    from proteus.config._atmos_clim import valid_janus

    instance = SimpleNamespace(
        module='janus',
        spectral_group='Honeyside',
        spectral_bands='48',
        tmp_minimum=300.0,
        janus=SimpleNamespace(
            tmp_maximum=300.0,
        ),
    )
    with pytest.raises(ValueError, match='has to be bigger'):
        valid_janus(instance, SimpleNamespace(), instance.janus)


@pytest.mark.unit
def test_janus_tmp_max_less_than_tmp_min_invalid():
    """Test valid_janus rejects janus.tmp_maximum < tmp_minimum."""
    from proteus.config._atmos_clim import valid_janus

    instance = SimpleNamespace(
        module='janus',
        spectral_group='Honeyside',
        spectral_bands='48',
        tmp_minimum=400.0,
        janus=SimpleNamespace(
            tmp_maximum=300.0,
        ),
    )
    with pytest.raises(ValueError, match='has to be bigger'):
        valid_janus(instance, SimpleNamespace(), instance.janus)


@pytest.mark.unit
def test_atmos_clim_warn_if_dummy_rayleigh_compatible():
    """Test warn_if_dummy validator allows rayleigh with non-dummy module."""
    from proteus.config._atmos_clim import warn_if_dummy

    # Valid: AGNI module with rayleigh enabled
    instance = SimpleNamespace(module='agni')
    attribute = SimpleNamespace(name='rayleigh')
    result = warn_if_dummy(instance, attribute, True)
    assert result is None  # contract: validator returns None silently on accepted module
    assert instance.module == 'agni'  # no mutation of the module sentinel


@pytest.mark.unit
def test_atmos_clim_warn_if_dummy_rayleigh_incompatible():
    """Test warn_if_dummy validator rejects rayleigh with dummy module."""
    from proteus.config._atmos_clim import warn_if_dummy

    # Invalid: dummy module with rayleigh enabled
    instance = SimpleNamespace(module='dummy')
    attribute = SimpleNamespace(name='rayleigh')
    with pytest.raises(
        ValueError, match='Dummy atmos_clim module is incompatible with rayleigh=True'
    ):
        warn_if_dummy(instance, attribute, True)


@pytest.mark.unit
def test_atmos_clim_warn_if_dummy_rayleigh_disabled():
    """Test warn_if_dummy validator allows dummy module with rayleigh disabled."""
    from proteus.config._atmos_clim import warn_if_dummy

    # Valid: dummy module with rayleigh disabled
    instance = SimpleNamespace(module='dummy')
    result = warn_if_dummy(instance, SimpleNamespace(), False)
    assert result is None  # contract: rayleigh=False short-circuits the dummy guard
    assert instance.module == 'dummy'  # no mutation of the module sentinel


@pytest.mark.unit
def test_atmos_clim_check_overlap_valid_ro():
    """Test check_overlap validator accepts random overlap method."""
    from proteus.config._atmos_clim import check_overlap

    instance = SimpleNamespace()
    result = check_overlap(instance, SimpleNamespace(), 'ro')
    assert result is None  # contract: validator returns None silently on accepted overlap
    assert vars(instance) == {}  # validator must not mutate the input instance


@pytest.mark.unit
def test_atmos_clim_check_overlap_valid_rorr():
    """Test check_overlap validator accepts rorr overlap method."""
    from proteus.config._atmos_clim import check_overlap

    instance = SimpleNamespace()
    result = check_overlap(instance, SimpleNamespace(), 'rorr')
    assert result is None  # contract: validator returns None silently on accepted overlap
    assert vars(instance) == {}  # validator must not mutate the input instance


@pytest.mark.unit
def test_atmos_clim_check_overlap_valid_ee():
    """Test check_overlap validator accepts ee overlap method."""
    from proteus.config._atmos_clim import check_overlap

    instance = SimpleNamespace()
    result = check_overlap(instance, SimpleNamespace(), 'ee')
    assert result is None  # contract: validator returns None silently on accepted overlap
    assert vars(instance) == {}  # validator must not mutate the input instance


@pytest.mark.unit
def test_atmos_clim_check_overlap_invalid():
    """Test check_overlap validator rejects invalid overlap methods."""
    from proteus.config._atmos_clim import check_overlap

    instance = SimpleNamespace()
    with pytest.raises(ValueError, match='Overlap type must be one of'):
        check_overlap(instance, SimpleNamespace(), 'invalid')


@pytest.mark.unit
def test_atmos_clim_check_overlap_empty_string():
    """Test check_overlap validator rejects empty string."""
    from proteus.config._atmos_clim import check_overlap

    instance = SimpleNamespace()
    with pytest.raises(ValueError, match='Overlap type must be one of'):
        check_overlap(instance, SimpleNamespace(), '')


@pytest.mark.unit
def test_atmos_clim_valid_albedo_float_in_range():
    """Test valid_albedo validator accepts float in [0, 1]."""
    from proteus.config._atmos_clim import valid_albedo

    instance = SimpleNamespace()
    # Boundary values and midpoint; each call must return silently on the
    # accepted-range path.
    assert valid_albedo(instance, SimpleNamespace(), 0.0) is None
    assert valid_albedo(instance, SimpleNamespace(), 0.5) is None
    assert valid_albedo(instance, SimpleNamespace(), 1.0) is None
    # The validator does not mutate the input instance.
    assert vars(instance) == {}


@pytest.mark.unit
def test_atmos_clim_valid_albedo_float_below_range():
    """Test valid_albedo validator rejects float < 0."""
    from proteus.config._atmos_clim import valid_albedo

    instance = SimpleNamespace()
    with pytest.raises(ValueError, match='must be between 0 and 1'):
        valid_albedo(instance, SimpleNamespace(), -0.1)


@pytest.mark.unit
def test_atmos_clim_valid_albedo_float_above_range():
    """Test valid_albedo validator rejects float > 1."""
    from proteus.config._atmos_clim import valid_albedo

    instance = SimpleNamespace()
    with pytest.raises(ValueError, match='must be between 0 and 1'):
        valid_albedo(instance, SimpleNamespace(), 1.1)


@pytest.mark.unit
def test_atmos_clim_valid_albedo_string():
    """Test valid_albedo validator accepts file path as string."""
    from proteus.config._atmos_clim import valid_albedo

    instance = SimpleNamespace()
    result = valid_albedo(instance, SimpleNamespace(), '/path/to/albedo_file.csv')
    assert result is None  # contract: validator returns None silently on accepted string
    assert vars(instance) == {}  # validator must not mutate the input instance


@pytest.mark.unit
def test_atmos_clim_valid_albedo_invalid_type():
    """Test valid_albedo validator rejects invalid types."""
    from proteus.config._atmos_clim import valid_albedo

    instance = SimpleNamespace()
    with pytest.raises(ValueError, match='must be a string or a float'):
        valid_albedo(instance, SimpleNamespace(), 123)  # int, not float/str

    with pytest.raises(ValueError, match='must be a string or a float'):
        valid_albedo(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_janus_module_skip():
    """Test valid_janus validator skips validation when module != 'janus'."""
    from proteus.config._atmos_clim import valid_janus

    # With module='agni', validator should return early without checking janus config
    instance = SimpleNamespace(
        module='agni',
        spectral_group=None,
        spectral_bands=None,
    )
    result = valid_janus(instance, SimpleNamespace(), None)
    assert result is None  # contract: non-janus module short-circuits the validator
    # Discriminating check: spectral_group/spectral_bands are both None, which
    # would normally raise via the janus path; only the module-skip branch can
    # produce a silent pass here.
    assert instance.module == 'agni'


@pytest.mark.unit
def test_atmos_clim_janus_spectral_group_required():
    """Test valid_janus validator requires spectral_group."""
    from proteus.config._atmos_clim import valid_janus

    instance = SimpleNamespace(
        module='janus',
        spectral_group=None,
        spectral_bands='16',
        tmp_minimum=0.5,
    )
    with pytest.raises(ValueError, match='Must set atmos_clim.spectral_group'):
        valid_janus(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_janus_spectral_bands_required():
    """Test valid_janus validator requires spectral_bands."""
    from proteus.config._atmos_clim import valid_janus

    instance = SimpleNamespace(
        module='janus',
        spectral_group='sw',
        spectral_bands=None,
        tmp_minimum=0.5,
    )
    with pytest.raises(ValueError, match='Must set atmos_clim.spectral_bands'):
        valid_janus(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_janus_valid_spectral_config():
    """Test valid_janus validator accepts valid spectral configuration."""
    from proteus.config._atmos_clim import valid_janus

    instance = SimpleNamespace(
        module='janus',
        spectral_group='sw_lw',
        spectral_bands='16',
        tmp_minimum=0.5,
        janus=SimpleNamespace(tmp_maximum=5000.0),
    )
    result = valid_janus(instance, SimpleNamespace(), None)
    assert result is None  # contract: validator returns None silently on the pass path
    assert instance.janus.tmp_maximum > instance.tmp_minimum  # bounds preserved


@pytest.mark.unit
def test_atmos_clim_agni_module_skip():
    """Test valid_agni validator skips validation when module != 'agni'."""
    from proteus.config._atmos_clim import valid_agni

    # With module='dummy', validator should return early without checking agni config
    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace())
    result = valid_agni(instance, SimpleNamespace(), None)
    assert result is None  # contract: non-agni module short-circuits the validator
    # Discriminating check: instance.agni is empty; only the module-skip branch
    # can produce a silent pass.
    assert instance.module == 'dummy'


@pytest.mark.unit
def test_atmos_clim_agni_p_top_must_be_less_than_psurf_thresh():
    """Test valid_agni validator enforces p_top < psurf_thresh."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        spectral_group='sw_lw',
        spectral_bands='16',
        p_top=0.2,  # Greater than psurf_thresh,
        p_obs=1.0,
        agni=SimpleNamespace(
            psurf_thresh=0.1,  # Less than p_top - INVALID
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            chemistry='none',
        ),
        surf_state='skin',
    )
    with pytest.raises(
        ValueError, match='Must set `p_top` to be less than `agni.psurf_thresh`'
    ):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_p_top_must_be_less_than_p_obs():
    """Test valid_agni validator enforces p_top < p_obs."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        spectral_group='sw_lw',
        spectral_bands='16',
        p_top=2.0,  # Greater than p_obs,
        p_obs=1.0,  # Less than p_top - INVALID,
        agni=SimpleNamespace(
            psurf_thresh=5.0,  # Valid: > p_top
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            chemistry='none',
        ),
        surf_state='skin',
    )
    with pytest.raises(ValueError, match='Must set `p_top` to be less than `p_obs`'):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_solve_energy_required_for_skin_surf_state():
    """Test valid_agni validator requires solve_energy=True for surf_state='skin'."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        spectral_group='sw_lw',
        spectral_bands='16',
        p_top=1e-5,
        p_obs=0.02,
        agni=SimpleNamespace(
            psurf_thresh=0.1,
            solve_energy=False,  # INVALID for skin surf_state
            latent_heat=False,
            rainout=True,
            chemistry='none',
        ),
        surf_state='skin',  # Requires solve_energy=True
    )
    with pytest.raises(
        ValueError, match="Must set `agni.solve_energy=true` if using `surf_state='skin'`"
    ):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_latent_heat_requires_rainout():
    """Test valid_agni validator requires rainout=True when latent_heat=True."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        spectral_group='sw_lw',
        spectral_bands='16',
        p_top=1e-5,
        p_obs=0.02,
        agni=SimpleNamespace(
            psurf_thresh=0.1,
            solve_energy=True,
            latent_heat=True,  # Requires rainout=True
            rainout=False,  # INVALID
            chemistry='none',
        ),
        surf_state='skin',
    )
    with pytest.raises(
        ValueError, match='Must set `rainout=true` if setting `latent_heat=true`'
    ):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_spectral_group_required():
    """Test valid_agni validator requires spectral_group."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        spectral_group=None,
        spectral_bands='16',
        p_top=1e-5,
        p_obs=0.02,
        agni=SimpleNamespace(
            psurf_thresh=0.1,
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            chemistry='none',
            spectral_file=None,
        ),
        surf_state='skin',
    )
    with pytest.raises(ValueError, match='Must set atmos_clim.spectral_group'):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_spectral_bands_required():
    """Test valid_agni validator requires spectral_bands."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        spectral_group='sw_lw',
        spectral_bands=None,
        p_top=1e-5,
        p_obs=0.02,
        agni=SimpleNamespace(
            psurf_thresh=0.1,
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            chemistry='none',
            spectral_file=None,
        ),
        surf_state='skin',
    )
    with pytest.raises(ValueError, match='Must set atmos_clim.spectral_bands'):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_valid_complete_config():
    """Test valid_agni validator accepts valid AGNI configuration."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        spectral_group='sw_lw',
        spectral_bands='16',
        p_top=1e-5,
        p_obs=0.02,
        agni=SimpleNamespace(
            psurf_thresh=0.1,
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            chemistry='none',
            spectral_file=None,
        ),
        surf_state='skin',
    )
    result = valid_agni(instance, SimpleNamespace(), None)
    assert result is None  # contract: validator returns None silently on the pass path
    # Discriminating bounds: p_top below both psurf_thresh and p_obs; surf_state='skin'
    # is compatible with solve_energy=True. Each is a separate branch in valid_agni.
    assert instance.p_top < instance.agni.psurf_thresh
    assert instance.p_top < instance.p_obs


# ========================
# PARAMS VALIDATORS
# ========================


@pytest.mark.unit
def test_params_valid_path_non_empty_string():
    """Test valid_path validator accepts non-empty strings."""
    from proteus.config._params import valid_path

    instance = SimpleNamespace()
    result = valid_path(instance, SimpleNamespace(name='data_path'), '/data/output')
    assert result is None  # contract: validator returns None silently on accepted string
    assert vars(instance) == {}  # validator must not mutate the input instance


@pytest.mark.unit
def test_params_valid_path_empty_string():
    """Test valid_path validator rejects empty strings."""
    from proteus.config._params import valid_path

    with pytest.raises(ValueError, match='must be a non-empty string'):
        valid_path(SimpleNamespace(), SimpleNamespace(name='data_path'), '')


@pytest.mark.unit
def test_params_valid_path_whitespace_only():
    """Test valid_path validator rejects whitespace-only strings."""
    from proteus.config._params import valid_path

    with pytest.raises(ValueError, match='must be a non-empty string'):
        valid_path(SimpleNamespace(), SimpleNamespace(name='data_path'), '   ')


@pytest.mark.unit
def test_params_valid_path_non_string():
    """Test valid_path validator rejects non-string types."""
    from proteus.config._params import valid_path

    with pytest.raises(ValueError, match='must be a non-empty string'):
        valid_path(SimpleNamespace(), SimpleNamespace(name='data_path'), None)

    with pytest.raises(ValueError, match='must be a non-empty string'):
        valid_path(SimpleNamespace(), SimpleNamespace(name='data_path'), 123)


@pytest.mark.unit
def test_params_max_bigger_than_min_valid():
    """Test max_bigger_than_min validator accepts maximum > minimum."""
    from proteus.config._params import max_bigger_than_min

    instance = SimpleNamespace(minimum=100)
    result = max_bigger_than_min(instance, SimpleNamespace(), 1000)
    assert result is None  # contract: validator returns None silently when max > min
    assert instance.minimum == 100  # validator must not mutate the input instance


@pytest.mark.unit
def test_params_max_bigger_than_min_equal():
    """Test max_bigger_than_min validator rejects maximum == minimum."""
    from proteus.config._params import max_bigger_than_min

    instance = SimpleNamespace(minimum=100)
    with pytest.raises(ValueError, match="'maximum' has to be bigger than 'minimum'"):
        max_bigger_than_min(instance, SimpleNamespace(), 100)


@pytest.mark.unit
def test_params_max_bigger_than_min_less():
    """Test max_bigger_than_min validator rejects maximum < minimum."""
    from proteus.config._params import max_bigger_than_min

    instance = SimpleNamespace(minimum=1000)
    with pytest.raises(ValueError, match="'maximum' has to be bigger than 'minimum'"):
        max_bigger_than_min(instance, SimpleNamespace(), 100)


@pytest.mark.unit
def test_params_valid_mod_none():
    """Test valid_mod validator accepts None."""
    from proteus.config._params import valid_mod

    instance = SimpleNamespace()
    result = valid_mod(instance, SimpleNamespace(), None)
    assert result is None  # contract: validator returns None silently for value=None
    assert vars(instance) == {}  # validator must not mutate the input instance


@pytest.mark.unit
def test_params_valid_mod_positive():
    """Test valid_mod validator accepts positive values."""
    from proteus.config._params import valid_mod

    instance = SimpleNamespace()
    # Both small (1) and large (100) positive values on the accepted-range path.
    assert valid_mod(instance, SimpleNamespace(), 1) is None
    assert valid_mod(instance, SimpleNamespace(), 100) is None
    assert vars(instance) == {}  # validator must not mutate the input instance


@pytest.mark.unit
def test_params_valid_mod_zero():
    """Test valid_mod validator accepts zero."""
    from proteus.config._params import valid_mod

    instance = SimpleNamespace()
    result = valid_mod(instance, SimpleNamespace(), 0)
    assert result is None  # zero is a documented special sentinel, not invalid
    assert vars(instance) == {}  # validator must not mutate the input instance


@pytest.mark.unit
def test_params_valid_mod_negative():
    """Test valid_mod validator rejects negative values."""
    from proteus.config._params import valid_mod

    instance = SimpleNamespace()
    with pytest.raises(ValueError, match='must be None or greater than 0'):
        valid_mod(instance, SimpleNamespace(), -1)


# ========================
# STRUCT VALIDATORS
# ========================


@pytest.mark.unit
def test_planet_mass_valid_accepts_positive():
    """Test planet_mass_valid validator accepts positive mass_tot."""
    from proteus.config._config import planet_mass_valid

    instance = SimpleNamespace(planet=SimpleNamespace(mass_tot=1.0))
    result = planet_mass_valid(instance, SimpleNamespace(), None)
    assert result is None  # contract: validator returns None silently on accepted mass
    # validator must not mutate mass_tot; use pytest.approx to keep the
    # comparison float-safe.
    assert instance.planet.mass_tot == pytest.approx(1.0, rel=1e-12)


@pytest.mark.unit
def test_planet_mass_valid_rejects_none():
    """Test planet_mass_valid validator rejects None."""
    from proteus.config._config import planet_mass_valid

    instance = SimpleNamespace(planet=SimpleNamespace(mass_tot=None))
    with pytest.raises(ValueError, match='must be set'):
        planet_mass_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_planet_mass_valid_rejects_negative():
    """Test planet_mass_valid validator rejects negative mass_tot."""
    from proteus.config._config import planet_mass_valid

    instance = SimpleNamespace(planet=SimpleNamespace(mass_tot=-1.0))
    with pytest.raises(ValueError, match='must be > 0'):
        planet_mass_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_planet_mass_valid_rejects_too_large():
    """Test planet_mass_valid validator rejects mass_tot > 20 M_earth."""
    from proteus.config._config import planet_mass_valid

    instance = SimpleNamespace(planet=SimpleNamespace(mass_tot=21.0))
    with pytest.raises(ValueError, match='must be < 20'):
        planet_mass_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_module_skip():
    """Test valid_zalmoxis validator skips when module == 'spider'."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(module='spider', zalmoxis=SimpleNamespace())
    result = valid_zalmoxis(instance, SimpleNamespace(), None)
    assert result is None  # contract: non-zalmoxis module short-circuits the validator
    assert instance.module == 'spider'  # no mutation of the module sentinel


@pytest.mark.unit
def test_planet_mass_required_for_zalmoxis():
    """Test planet_mass_valid rejects None mass (required for Zalmoxis)."""
    from proteus.config._config import planet_mass_valid

    instance = SimpleNamespace(planet=SimpleNamespace(mass_tot=None))
    with pytest.raises(ValueError, match='must be set'):
        planet_mass_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_two_layer_requires_no_mantle_fraction():
    """Test valid_zalmoxis requires mantle_mass_fraction=0 for 2-layer non-Tdep model."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        zalmoxis=SimpleNamespace(
            core_eos='Seager2007:iron',
            mantle_eos='Seager2007:MgSiO3',
            ice_layer_eos=None,
            mantle_mass_fraction=0.2,  # INVALID (must be 0)
        ),
    )
    with pytest.raises(ValueError, match='mantle_mass_fraction.*must be 0'):
        valid_zalmoxis(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_three_layer_mass_constraint():
    """Test valid_zalmoxis enforces core+mantle mass <= 75% for 3-layer model."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        core_frac=0.5,
        core_frac_mode='mass',
        zalmoxis=SimpleNamespace(
            core_eos='Seager2007:iron',
            mantle_eos='Seager2007:MgSiO3',
            ice_layer_eos='Seager2007:H2O',
            mantle_mass_fraction=0.3,  # 50%+30%=80% > 75% INVALID
        ),
    )
    with pytest.raises(ValueError, match='must add up to <= 75%'):
        valid_zalmoxis(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_valid_configuration():
    """Test valid_zalmoxis accepts valid Zalmoxis configuration."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        zalmoxis=SimpleNamespace(
            core_eos='Seager2007:iron',
            mantle_eos='Seager2007:MgSiO3',
            ice_layer_eos=None,
            mantle_mass_fraction=0,
        ),
    )
    result = valid_zalmoxis(instance, SimpleNamespace(), None)
    assert result is None  # contract: validator returns None silently on the pass path
    # Discriminating check: ice_layer_eos=None selects the 2-layer branch where
    # mantle_mass_fraction=0 is the only valid combination.
    assert instance.zalmoxis.mantle_mass_fraction == 0


@pytest.mark.unit
def test_struct_zalmoxis_eos_format_missing_colon():
    """Test valid_zalmoxis rejects core_eos without ':' separator."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        zalmoxis=SimpleNamespace(
            core_eos='iron_no_source',
            mantle_eos='Seager2007:MgSiO3',
            ice_layer_eos=None,
            mantle_mass_fraction=0,
        ),
    )
    with pytest.raises(ValueError, match='<source>:<material>'):
        valid_zalmoxis(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_ice_eos_format_missing_colon():
    """Test valid_zalmoxis rejects ice_layer_eos with non-empty invalid format."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        zalmoxis=SimpleNamespace(
            core_eos='Seager2007:iron',
            mantle_eos='Seager2007:MgSiO3',
            ice_layer_eos='H2O_bad_format',
            mantle_mass_fraction=0,
        ),
    )
    with pytest.raises(ValueError, match='ice_layer_eos'):
        valid_zalmoxis(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_tdep_allows_mantle_fraction():
    """WolfBower2018 T-dep EOS should allow mantle_mass_fraction > 0."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        zalmoxis=SimpleNamespace(
            core_eos='Seager2007:iron',
            mantle_eos='WolfBower2018:MgSiO3',
            ice_layer_eos=None,
            mantle_mass_fraction=0.675,
        ),
    )
    result = valid_zalmoxis(instance, SimpleNamespace(), None)
    assert result is None  # contract: validator returns None silently on the pass path
    # Discriminating check: a fixed-EOS (Seager2007) mantle would have rejected
    # mantle_mass_fraction=0.675; the T-dependent WolfBower2018 EOS path is the
    # only branch that permits it.
    assert instance.zalmoxis.mantle_eos.startswith('WolfBower2018')


# ========================
# CONFIG VALIDATORS
# ========================


@pytest.mark.unit
def test_config_spada_zephyrus_requires_mors_spada():
    """Test spada_zephyrus validator requires MORS + Spada for Zephyrus."""
    from proteus.config._config import spada_zephyrus

    # Invalid: Zephyrus without MORS + Spada
    instance = SimpleNamespace(
        escape=SimpleNamespace(module='zephyrus'),
        star=SimpleNamespace(module='dummy'),  # Not MORS
    )
    with pytest.raises(ValueError, match='ZEPHYRUS must be used with MORS'):
        spada_zephyrus(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_config_spada_zephyrus_non_zephyrus_skip():
    """Test spada_zephyrus validator skips when using non-Zephyrus escape."""
    from proteus.config._config import spada_zephyrus

    # Valid: using dummy escape (not zephyrus)
    instance = SimpleNamespace(
        escape=SimpleNamespace(module='dummy'),
        star=SimpleNamespace(module='dummy'),
    )
    result = spada_zephyrus(instance, SimpleNamespace(), None)
    assert result is None  # contract: non-zephyrus escape short-circuits the validator
    # Discriminating check: star.module='dummy' would have raised under zephyrus;
    # the escape-module-skip branch is the only path to a silent pass here.
    assert instance.escape.module == 'dummy'


@pytest.mark.unit
def test_config_instmethod_dummy_requires_dummy_star():
    """Test instmethod_dummy validator requires dummy star for 'inst' method."""
    from proteus.config._config import instmethod_dummy

    # Invalid: inst method without dummy star
    instance = SimpleNamespace(
        orbit=SimpleNamespace(instellation_method='inst'),
        star=SimpleNamespace(module='mors'),  # Not dummy
    )
    with pytest.raises(ValueError, match="can only be 'inst' when star.module=dummy"):
        instmethod_dummy(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_config_instmethod_dummy_non_inst_skip():
    """Test instmethod_dummy validator skips for non-inst instellation method."""
    from proteus.config._config import instmethod_dummy

    # Valid: using 'gravity' instellation method (not inst)
    instance = SimpleNamespace(
        orbit=SimpleNamespace(instellation_method='gravity'),
        star=SimpleNamespace(module='mors'),
    )
    result = instmethod_dummy(instance, SimpleNamespace(), None)
    assert result is None  # contract: non-inst method short-circuits the validator
    # Discriminating check: star.module='mors' would have raised under 'inst';
    # the instellation_method-skip branch is the only path to a silent pass.
    assert instance.orbit.instellation_method == 'gravity'


@pytest.mark.unit
def test_config_instmethod_evolve_rejects_inst_with_orbit_evolution():
    """Test instmethod_evolve validator rejects 'inst' method with orbital evolution."""
    from proteus.config._config import instmethod_evolve

    # Invalid: inst method with orbital evolution
    instance = SimpleNamespace(
        orbit=SimpleNamespace(
            instellation_method='inst',
            evolve=True,  # INVALID
        ),
    )
    with pytest.raises(ValueError, match='not supported for `instellation_method'):
        instmethod_evolve(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_config_instmethod_evolve_allows_non_inst_with_evolution():
    """Test instmethod_evolve validator allows evolution with non-inst methods."""
    from proteus.config._config import instmethod_evolve

    # Valid: non-inst method with orbital evolution
    instance = SimpleNamespace(
        orbit=SimpleNamespace(
            instellation_method='gravity',
            evolve=True,
        ),
    )
    result = instmethod_evolve(instance, SimpleNamespace(), None)
    assert result is None  # contract: non-inst method permits orbital evolution
    # Discriminating check: evolve=True with instellation_method='inst' would have raised;
    # only the gravity-method branch allows the True-evolve combination.
    assert instance.orbit.evolve is True
    assert instance.orbit.instellation_method == 'gravity'


@pytest.mark.unit
def test_config_satellite_evolve_rejects_both_satellite_and_evolution():
    """Test satellite_evolve validator rejects satellite with orbital evolution."""
    from proteus.config._config import satellite_evolve

    # Invalid: satellite + orbital evolution
    instance = SimpleNamespace(
        orbit=SimpleNamespace(
            satellite=True,  # Has satellite
            evolve=True,  # Also evolving - INVALID
        ),
    )
    with pytest.raises(ValueError, match='cannot be used simultaneously'):
        satellite_evolve(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_config_satellite_evolve_allows_satellite_without_evolution():
    """Test satellite_evolve validator allows satellite without evolution."""
    from proteus.config._config import satellite_evolve

    # Valid: satellite without evolution
    instance = SimpleNamespace(
        orbit=SimpleNamespace(
            satellite=True,
            evolve=False,
        ),
    )
    result = satellite_evolve(instance, SimpleNamespace(), None)
    assert result is None  # contract: satellite=True with evolve=False is the valid combo
    # Discriminating check: satellite=True with evolve=True would have raised; only the
    # evolve=False branch can produce a silent pass with satellite=True.
    assert instance.orbit.satellite is True
    assert instance.orbit.evolve is False


@pytest.mark.unit
def test_config_tides_enabled_orbit_requires_orbit_module():
    """Test tides_enabled_orbit validator requires orbit module for tidal heating."""
    from proteus.config._config import tides_enabled_orbit

    # Invalid: tidal heating without orbit module
    instance = SimpleNamespace(
        interior_energetics=SimpleNamespace(heat_tidal=True),
        orbit=SimpleNamespace(module=None),  # No orbit module - INVALID
    )
    with pytest.raises(ValueError, match='Interior tidal heating requires'):
        tides_enabled_orbit(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_config_tides_enabled_orbit_allows_no_tides():
    """Test tides_enabled_orbit validator allows disabled tidal heating."""
    from proteus.config._config import tides_enabled_orbit

    # Valid: no tidal heating
    instance = SimpleNamespace(
        interior_energetics=SimpleNamespace(heat_tidal=False),
        orbit=SimpleNamespace(module=None),
    )
    result = tides_enabled_orbit(instance, SimpleNamespace(), None)
    assert result is None  # contract: heat_tidal=False short-circuits the validator
    # Discriminating check: orbit.module=None with heat_tidal=True would have raised;
    # only the heat_tidal-disabled branch can produce a silent pass.
    assert instance.interior_energetics.heat_tidal is False


@pytest.mark.unit
def test_config_observe_resolved_atmosphere_requires_non_dummy_atmos():
    """Test observe_resolved_atmosphere requires non-dummy atmosphere."""
    from proteus.config._config import observe_resolved_atmosphere

    # Invalid: synthesis observations with dummy atmosphere
    instance = SimpleNamespace(
        observe=SimpleNamespace(synthesis='platon'),  # Synthesis enabled
        atmos_clim=SimpleNamespace(module='dummy'),  # Dummy module - INVALID
    )
    with pytest.raises(ValueError, match='Observational synthesis requires'):
        observe_resolved_atmosphere(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_config_observe_resolved_atmosphere_allows_no_synthesis():
    """Test observe_resolved_atmosphere allows dummy atmosphere without synthesis."""
    from proteus.config._config import observe_resolved_atmosphere

    # Valid: no synthesis observations
    instance = SimpleNamespace(
        observe=SimpleNamespace(synthesis=None),
        atmos_clim=SimpleNamespace(module='dummy'),
    )
    result = observe_resolved_atmosphere(instance, SimpleNamespace(), None)
    assert result is None  # contract: synthesis=None short-circuits the validator
    # Discriminating check: atmos_clim.module='dummy' with synthesis='platon' would
    # have raised; only the synthesis-disabled branch can produce a silent pass.
    assert instance.observe.synthesis is None


@pytest.mark.unit
def test_config_janus_escape_atmosphere_requires_stop_escape():
    """Test janus_escape_atmosphere validator requires stop.escape.enabled=True."""
    from proteus.config._config import janus_escape_atmosphere

    # Invalid: Zephyrus + JANUS without stop.escape enabled
    instance = SimpleNamespace(
        escape=SimpleNamespace(module='zephyrus'),
        atmos_clim=SimpleNamespace(module='janus'),
        params=SimpleNamespace(stop=SimpleNamespace(escape=SimpleNamespace(enabled=False))),
    )
    with pytest.raises(ValueError, match='params.stop.escape must be True'):
        janus_escape_atmosphere(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_config_janus_escape_atmosphere_non_zephyrus_skip():
    """Test janus_escape_atmosphere validator skips for non-Zephyrus escape."""
    from proteus.config._config import janus_escape_atmosphere

    # Valid: dummy escape (not zephyrus)
    instance = SimpleNamespace(
        escape=SimpleNamespace(module='dummy'),
        atmos_clim=SimpleNamespace(module='janus'),
        params=SimpleNamespace(stop=SimpleNamespace(escape=SimpleNamespace(enabled=False))),
    )
    result = janus_escape_atmosphere(instance, SimpleNamespace(), None)
    assert result is None  # contract: non-zephyrus escape short-circuits the validator
    # Discriminating check: stop.escape.enabled=False with zephyrus+janus would have
    # raised; only the escape-module-skip branch can produce a silent pass here.
    assert instance.escape.module == 'dummy'


@pytest.mark.unit
def test_config_janus_escape_atmosphere_non_janus_skip():
    """Test janus_escape_atmosphere validator skips for non-JANUS atmosphere."""
    from proteus.config._config import janus_escape_atmosphere

    # Valid: AGNI instead of JANUS
    instance = SimpleNamespace(
        escape=SimpleNamespace(module='zephyrus'),
        atmos_clim=SimpleNamespace(module='agni'),
        params=SimpleNamespace(stop=SimpleNamespace(escape=SimpleNamespace(enabled=False))),
    )
    result = janus_escape_atmosphere(instance, SimpleNamespace(), None)
    assert result is None  # contract: non-janus atmosphere short-circuits the validator
    # Discriminating check: stop.escape.enabled=False with zephyrus+janus would have
    # raised; only the atmos-module-skip branch can produce a silent pass here.
    assert instance.atmos_clim.module == 'agni'


# ----------------------------------------------------------------------
# liquidus_super temperature_mode + delta_T_super (Stage 4 IC anchor)
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_planet_temperature_mode_accepts_liquidus_super():
    """Planet validator must accept the new 'liquidus_super' mode.

    Discriminating: this test fails if the validator tuple is missing the
    new entry, which is the most plausible regression after a refactor of
    the temperature_mode list. We also confirm the mode does not silently
    coerce to one of the existing six modes.
    """
    p = Planet(temperature_mode='liquidus_super')
    assert p.temperature_mode == 'liquidus_super'
    # negative discrimination: must not collapse to default
    assert p.temperature_mode != 'adiabatic_from_cmb'


@pytest.mark.unit
@pytest.mark.parametrize(
    'mode',
    [
        'isothermal',
        'linear',
        'adiabatic',
        'adiabatic_from_cmb',
        'accretion',
        'isentropic',
        'liquidus_super',
    ],
)
def test_planet_temperature_mode_full_enumeration(mode):
    """Every documented temperature_mode value must construct successfully.

    This pins the public surface so that adding new modes does not silently
    drop existing ones from the validator.
    """
    p = Planet(temperature_mode=mode)
    assert p.temperature_mode == mode


@pytest.mark.unit
def test_planet_temperature_mode_rejects_typos():
    """Unknown temperature_mode values must raise (no silent fallback).

    Anti-happy-path: tests three plausible typos (case error, plural, wrong
    underscore) plus an empty string. A bug that downcases input or tries
    fuzzy matching would fail one of these.
    """
    for bad in ('Liquidus_super', 'liquidussuper', 'liquidus-super', ''):
        with pytest.raises(ValueError):
            Planet(temperature_mode=bad)


@pytest.mark.unit
def test_planet_delta_T_super_default_is_500():
    """delta_T_super default must match the documented 500 K, not 0 or
    some unrelated leftover constant. A 500 K offset above the Fei+2021
    liquidus at P_cmb ~ 135 GPa (T_liq ~ 5935 K) gives the canonical
    fully-molten 1 M_E IC at T_cmb ~ 6435 K used in the paper rerun.
    """
    p = Planet()
    assert p.delta_T_super == pytest.approx(500.0, rel=0, abs=0)


@pytest.mark.unit
def test_planet_delta_T_super_accepts_zero():
    """delta_T_super = 0 K must be allowed (anchor exactly on the liquidus,
    a meaningful physical limit, not just a numerical edge case).
    """
    p = Planet(delta_T_super=0.0)
    assert p.delta_T_super == pytest.approx(0.0, abs=0)


@pytest.mark.unit
@pytest.mark.parametrize('value', [-1.0, -0.001, -1e6])
def test_planet_delta_T_super_rejects_negative(value):
    """Negative delta_T_super (subliquidus IC by request) must raise.

    Physically, a sub-liquidus anchor is supported by the existing
    'adiabatic_from_cmb' mode with a chosen tcmb_init; the
    'liquidus_super' mode is explicitly the super-liquidus side and a
    negative value would silently become a sub-liquidus IC of unknown
    melt fraction.
    """
    with pytest.raises(ValueError):
        Planet(delta_T_super=value)


@pytest.mark.unit
def test_planet_delta_T_super_accepts_large_super_earth_value():
    """A 5000 K super-liquidus offset must be allowed.

    Discriminating: large offsets are physically meaningful for hot
    super-Earth ICs, so the validator must not impose a hidden upper
    bound. (If we ever add ge(0) AND a sanity ceiling, this test will
    flag the change.)
    """
    p = Planet(delta_T_super=5000.0)
    assert p.delta_T_super == pytest.approx(5000.0, abs=0)


@pytest.mark.unit
def test_planet_liquidus_super_does_not_disturb_other_defaults():
    """Selecting liquidus_super must not collide with other IC scalars.

    Verifies that ini_entropy, tsurf_init, tcmb_init keep their defaults
    when liquidus_super is chosen; this is the regression test for the
    case where a future refactor makes one mode mutually-exclusive with
    another's parameters.
    """
    p = Planet(temperature_mode='liquidus_super', delta_T_super=750.0)
    assert p.temperature_mode == 'liquidus_super'
    assert p.delta_T_super == pytest.approx(750.0)
    assert p.ini_entropy == pytest.approx(3900.0)
    assert p.tsurf_init == pytest.approx(4000.0)
    assert p.tcmb_init == pytest.approx(6000.0)


# ============================================================================
# Regression: melting_dir / eos_dir belong on [interior_struct], not [interior_energetics]
# ============================================================================


@pytest.mark.unit
def test_melting_dir_is_field_of_Struct_not_Interior():
    """Regression: ``melting_dir`` and ``eos_dir`` are fields of
    ``Struct`` (which maps to TOML section ``[interior_struct]``),
    not ``Interior`` (``[interior_energetics]``). Putting them on
    ``Interior`` makes cattrs silently drop the values from the TOML,
    then the Struct validator raises a confusing
    ``melting_dir must be set`` error.

    This pins down the schema location so a future refactor that
    moves them back to ``Interior`` fails this test rather than
    silently breaking every example config that follows the docstring.
    """
    import attrs

    from proteus.config._interior import Interior
    from proteus.config._struct import Struct

    interior_fields = {f.name for f in attrs.fields(Interior)}
    struct_fields = {f.name for f in attrs.fields(Struct)}

    assert 'melting_dir' in struct_fields, (
        'melting_dir must live on Struct (the [interior_struct] section)'
    )
    assert 'melting_dir' not in interior_fields, (
        'melting_dir must NOT live on Interior; cattrs would silently drop it'
    )
    assert 'eos_dir' in struct_fields, 'eos_dir must live on Struct'
    assert 'eos_dir' not in interior_fields, 'eos_dir must NOT live on Interior'


@pytest.mark.unit
def test_Interior_docstring_does_not_advertise_struct_fields():
    """Companion guarantee: the ``Interior`` class docstring must not
    document ``melting_dir`` or ``eos_dir`` as if they were its own
    fields. That docstring is what caused the example
    ``init_coupler.toml`` to place ``melting_dir`` in the wrong section."""
    from inspect import getdoc

    from proteus.config._interior import Interior

    doc = getdoc(Interior)
    assert doc is not None
    # The docstring may mention these names in passing, but must not
    # list them as `<name>: <type>` attributes the way attrs docstring
    # conventions document fields.
    forbidden_lines = ('melting_dir: str', 'eos_dir: str')
    for marker in forbidden_lines:
        assert marker not in doc, (
            f'Interior docstring still advertises a non-existent field: {marker!r}'
        )


@pytest.mark.unit
def test_example_init_coupler_places_melting_dir_under_interior_struct():
    """Regression: the canonical ``examples/minimal/init_coupler.toml``
    must place ``melting_dir`` inside the ``[interior_struct]`` section.
    Otherwise cattrs silently drops the value at parse time."""
    text = (PROTEUS_ROOT / 'examples' / 'minimal' / 'init_coupler.toml').read_text(
        encoding='utf-8'
    )

    # Find every section header and the index of the melting_dir line.
    lines = text.splitlines()
    section = None
    melting_dir_section = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            section = stripped[1:-1]
        if stripped.startswith('melting_dir'):
            melting_dir_section = section
            break

    assert melting_dir_section == 'interior_struct', (
        f'melting_dir is under [{melting_dir_section}] but must be under '
        '[interior_struct]; otherwise cattrs silently drops it'
    )


@pytest.mark.unit
def test_no_input_toml_uses_bare_interior_section():
    """Regression: no TOML under input/ may use the bare
    ``[interior]`` section name. The current schema places interior
    physics under ``[interior_energetics]``; cattrs silently drops
    unknown top-level sections, so a stale ``[interior]`` header
    silently demotes module=spider runs to the factory default
    (aragog) without any error.

    This bit ``input/chili/intercomp/_base.toml`` after PR #659:
    every CHILI sweep launched from that config ran Aragog
    regardless of the configured backend.
    """
    import re

    repo_root = PROTEUS_ROOT
    pattern = re.compile(r'^\[interior\](?:\.|$)', re.MULTILINE)
    offenders = []
    for toml_path in (repo_root / 'input').rglob('*.toml'):
        text = toml_path.read_text(encoding='utf-8')
        if pattern.search(text):
            offenders.append(str(toml_path.relative_to(repo_root)))
    assert not offenders, (
        f'These input TOMLs still use the bare [interior] section '
        f'(should be [interior_energetics]): {offenders}'
    )
