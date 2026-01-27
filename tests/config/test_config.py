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
    satellite_evolve,
    spada_zephyrus,
    tides_enabled_orbit,
)
from proteus.config._converters import none_if_none
from proteus.config._delivery import Volatiles
from proteus.config._interior import valid_aragog, valid_interiordummy, valid_spider
from proteus.config._outgas import Calliope
from proteus.config._params import max_bigger_than_min, valid_mod, valid_path

PATHS = chain(
    (PROTEUS_ROOT / 'input').glob('*.toml'),
)


@pytest.fixture
def cfg():
    """Load the demo dummy config and return the parsed Config object."""
    path = PROTEUS_ROOT / 'input' / 'demos' / 'dummy.toml'
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
    path = PROTEUS_ROOT / 'input' / 'demos' / 'dummy.toml'
    raw = read_config(path)
    assert isinstance(raw, dict)
    assert 'params' in raw or 'star' in raw or 'orbit' in raw


@pytest.mark.unit
def test_read_config_object_returns_config():
    """
    read_config_object returns a validated Config instance.

    Physical scenario: full parse and validate; used by Proteus and tests.
    """
    path = PROTEUS_ROOT / 'input' / 'demos' / 'dummy.toml'
    obj = read_config_object(path)
    assert isinstance(obj, Config)


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
    print(f'Testing config at {path}')
    runner = Proteus(config_path=path)
    assert isinstance(runner.config, Config)


@pytest.mark.unit
def test_calliope_is_included():
    """Calliope includes/excludes species based on boolean flags."""
    conf = Calliope(
        T_floor=0.1,
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
    conf = Volatiles(N2=123.0)
    assert conf.get_pressure('N2') == 123.0
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
        interior=SimpleNamespace(tidal_heat=True), orbit=SimpleNamespace(module=None)
    )
    with pytest.raises(ValueError):
        tides_enabled_orbit(inst, None, None)


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
    """Janus escape needs stop.escape flag to prevent runaway mass loss."""
    inst = SimpleNamespace(
        escape=SimpleNamespace(module='zephyrus'),
        atmos_clim=SimpleNamespace(module='janus'),
        params=SimpleNamespace(stop=SimpleNamespace(escape=False)),
    )
    with pytest.raises(ValueError):
        janus_escape_atmosphere(inst, None, None)


# =============================================================================
# Interior validators
# =============================================================================


@pytest.mark.unit
def test_valid_interiordummy_checks_tmagma_and_liquidus():
    """Dummy interior rejects unrealistically low magma temperature and thin liquidus gap."""
    dummy = SimpleNamespace(ini_tmagma=150.0, mantle_tliq=1500.0, mantle_tsol=1600.0)
    inst = SimpleNamespace(module='dummy', dummy=dummy)
    with pytest.raises(ValueError):
        valid_interiordummy(inst, None, None)

    dummy_ok = SimpleNamespace(ini_tmagma=500.0, mantle_tliq=1700.0, mantle_tsol=1600.0)
    inst_ok = SimpleNamespace(module='dummy', dummy=dummy_ok)
    # Should not raise
    valid_interiordummy(inst_ok, None, None)


@pytest.mark.unit
def test_valid_aragog_requires_energy_term_and_tmagma():
    """Aragog interior needs at least one heat transport term and warm initial magma."""
    aragog = SimpleNamespace(
        ini_tmagma=150.0,
        conduction=False,
        convection=False,
        mixing=False,
        gravitational_separation=False,
    )
    inst = SimpleNamespace(module='aragog', aragog=aragog)
    with pytest.raises(ValueError):
        valid_aragog(inst, None, None)

    aragog_ok = SimpleNamespace(
        ini_tmagma=800.0,
        conduction=True,
        convection=False,
        mixing=False,
        gravitational_separation=False,
    )
    inst_ok = SimpleNamespace(module='aragog', aragog=aragog_ok)
    valid_aragog(inst_ok, None, None)


@pytest.mark.unit
def test_valid_spider_requires_energy_term_and_entropy():
    """Spider interior requires an energy transport term and positive entropy."""
    spider = SimpleNamespace(
        ini_entropy=150.0,
        conduction=False,
        convection=False,
        mixing=False,
        gravitational_separation=False,
    )
    inst = SimpleNamespace(module='spider', spider=spider)
    with pytest.raises(ValueError):
        valid_spider(inst, None, None)

    spider_ok = SimpleNamespace(
        ini_entropy=400.0,
        conduction=True,
        convection=False,
        mixing=False,
        gravitational_separation=False,
    )
    inst_ok = SimpleNamespace(module='spider', spider=spider_ok)
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
    """Test MORS validator requires star_name."""
    from proteus.config._star import valid_mors

    config = SimpleNamespace(
        module='mors',
        mors=SimpleNamespace(
            age_now=5.0,  # Valid age
            star_name=None,  # Invalid: missing star name
            rot_pcntle=50.0,
            rot_period=None,
            spectrum_source=None,
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

    # Should not raise error
    valid_mors(config, SimpleNamespace(), None)


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
    with pytest.raises(ValueError, match='Pxuv.*must be >0 and < 10'):
        valid_zephyrus(config, SimpleNamespace(), None)

    # Pxuv = 10.1 - invalid
    config.zephyrus.Pxuv = 10.1
    with pytest.raises(ValueError, match='Pxuv.*must be >0 and < 10'):
        valid_zephyrus(config, SimpleNamespace(), None)

    # Pxuv = None - invalid
    config.zephyrus.Pxuv = None
    with pytest.raises(ValueError, match='Pxuv.*must be >0 and < 10'):
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

    # Rate = 0 - invalid
    config.dummy.rate = 0.0
    with pytest.raises(ValueError, match='rate.*must be >0'):
        valid_escapedummy(config, SimpleNamespace(), None)

    # Rate = None - invalid
    config.dummy.rate = None
    with pytest.raises(ValueError, match='rate.*must be >0'):
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
def test_interior_spider_entropy_minimum():
    """Test SPIDER validator requires minimum initial entropy."""
    from proteus.config._interior import valid_spider

    # Valid entropy
    config = SimpleNamespace(
        module='spider',
        spider=SimpleNamespace(
            ini_entropy=250.0,  # Valid: > 200
            conduction=True,
            convection=True,
            mixing=False,
            gravitational_separation=False,
        ),
    )

    # Should not raise error
    valid_spider(config, SimpleNamespace(), None)

    # Entropy = 200.0 - invalid (must be > 200)
    config.spider.ini_entropy = 200.0
    with pytest.raises(ValueError, match='ini_entropy.*must be >200'):
        valid_spider(config, SimpleNamespace(), None)

    # Entropy = None - invalid
    config.spider.ini_entropy = None
    with pytest.raises(ValueError, match='ini_entropy.*must be >200'):
        valid_spider(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_interior_spider_energy_term_required():
    """Test SPIDER validator requires at least one energy transport term enabled."""
    from proteus.config._interior import valid_spider

    # At least one enabled - valid
    config = SimpleNamespace(
        module='spider',
        spider=SimpleNamespace(
            ini_entropy=250.0,
            conduction=True,  # At least this one
            convection=False,
            mixing=False,
            gravitational_separation=False,
        ),
    )

    # Should not raise error
    valid_spider(config, SimpleNamespace(), None)

    # All disabled - invalid
    config.spider.conduction = False
    with pytest.raises(ValueError, match='Must enable at least one energy transport term'):
        valid_spider(config, SimpleNamespace(), None)


@pytest.mark.unit
def test_interior_valid_path_string_required():
    """Test valid_path validator requires non-empty string."""
    from proteus.config._interior import valid_path

    # Valid path
    valid_path(SimpleNamespace(), SimpleNamespace(name='path_field'), '/some/path')  # OK

    # Empty string - invalid
    with pytest.raises(ValueError, match='must be a non-empty string'):
        valid_path(SimpleNamespace(), SimpleNamespace(name='path_field'), '')

    # Non-string - invalid
    with pytest.raises(ValueError, match='must be a non-empty string'):
        valid_path(SimpleNamespace(), SimpleNamespace(name='path_field'), None)

    # Whitespace only - invalid
    with pytest.raises(ValueError, match='must be a non-empty string'):
        valid_path(SimpleNamespace(), SimpleNamespace(name='path_field'), '   ')


# ========================
# ATMOS_CLIM VALIDATORS
# ========================


@pytest.mark.unit
def test_atmos_clim_tmp_max_bigger_than_tmp_min_valid():
    """Test tmp_max_bigger_than_tmp_min validator with valid inputs."""
    from proteus.config._atmos_clim import tmp_max_bigger_than_tmp_min

    # Valid case: tmp_maximum > tmp_minimum
    instance = SimpleNamespace(tmp_minimum=300.0)
    tmp_max_bigger_than_tmp_min(instance, SimpleNamespace(), 5000.0)  # Should not raise


@pytest.mark.unit
def test_atmos_clim_tmp_max_equals_tmp_min_invalid():
    """Test tmp_max_bigger_than_tmp_min validator rejects equal values."""
    from proteus.config._atmos_clim import tmp_max_bigger_than_tmp_min

    # Invalid: tmp_maximum == tmp_minimum
    instance = SimpleNamespace(tmp_minimum=300.0)
    with pytest.raises(ValueError, match="'tmp_maximum' has to be bigger than 'tmp_minimum'"):
        tmp_max_bigger_than_tmp_min(instance, SimpleNamespace(), 300.0)


@pytest.mark.unit
def test_atmos_clim_tmp_max_less_than_tmp_min_invalid():
    """Test tmp_max_bigger_than_tmp_min validator rejects tmp_max < tmp_min."""
    from proteus.config._atmos_clim import tmp_max_bigger_than_tmp_min

    # Invalid: tmp_maximum < tmp_minimum
    instance = SimpleNamespace(tmp_minimum=400.0)
    with pytest.raises(ValueError, match="'tmp_maximum' has to be bigger than 'tmp_minimum'"):
        tmp_max_bigger_than_tmp_min(instance, SimpleNamespace(), 300.0)


@pytest.mark.unit
def test_atmos_clim_warn_if_dummy_rayleigh_compatible():
    """Test warn_if_dummy validator allows rayleigh with non-dummy module."""
    from proteus.config._atmos_clim import warn_if_dummy

    # Valid: AGNI module with rayleigh enabled
    instance = SimpleNamespace(module='agni')
    warn_if_dummy(instance, SimpleNamespace(), True)  # Should not raise


@pytest.mark.unit
def test_atmos_clim_warn_if_dummy_rayleigh_incompatible():
    """Test warn_if_dummy validator rejects rayleigh with dummy module."""
    from proteus.config._atmos_clim import warn_if_dummy

    # Invalid: dummy module with rayleigh enabled
    instance = SimpleNamespace(module='dummy')
    with pytest.raises(
        ValueError, match='Dummy atmos_clim module is incompatible with Rayleigh scattering'
    ):
        warn_if_dummy(instance, SimpleNamespace(), True)


@pytest.mark.unit
def test_atmos_clim_warn_if_dummy_rayleigh_disabled():
    """Test warn_if_dummy validator allows dummy module with rayleigh disabled."""
    from proteus.config._atmos_clim import warn_if_dummy

    # Valid: dummy module with rayleigh disabled
    instance = SimpleNamespace(module='dummy')
    warn_if_dummy(instance, SimpleNamespace(), False)  # Should not raise


@pytest.mark.unit
def test_atmos_clim_check_overlap_valid_ro():
    """Test check_overlap validator accepts random overlap method."""
    from proteus.config._atmos_clim import check_overlap

    instance = SimpleNamespace()
    check_overlap(instance, SimpleNamespace(), 'ro')  # Should not raise


@pytest.mark.unit
def test_atmos_clim_check_overlap_valid_rorr():
    """Test check_overlap validator accepts rorr overlap method."""
    from proteus.config._atmos_clim import check_overlap

    instance = SimpleNamespace()
    check_overlap(instance, SimpleNamespace(), 'rorr')  # Should not raise


@pytest.mark.unit
def test_atmos_clim_check_overlap_valid_ee():
    """Test check_overlap validator accepts ee overlap method."""
    from proteus.config._atmos_clim import check_overlap

    instance = SimpleNamespace()
    check_overlap(instance, SimpleNamespace(), 'ee')  # Should not raise


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
    # Boundary values and midpoint
    valid_albedo(instance, SimpleNamespace(), 0.0)  # Lower bound
    valid_albedo(instance, SimpleNamespace(), 0.5)  # Midpoint
    valid_albedo(instance, SimpleNamespace(), 1.0)  # Upper bound


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
    valid_albedo(instance, SimpleNamespace(), '/path/to/albedo_file.csv')  # Should not raise


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
        janus=SimpleNamespace(spectral_group=None, spectral_bands=None),
    )
    valid_janus(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_atmos_clim_janus_spectral_group_required():
    """Test valid_janus validator requires spectral_group."""
    from proteus.config._atmos_clim import valid_janus

    instance = SimpleNamespace(
        module='janus',
        janus=SimpleNamespace(spectral_group=None, spectral_bands='16'),
    )
    with pytest.raises(ValueError, match='Must set atmos_clim.janus.spectral_group'):
        valid_janus(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_janus_spectral_bands_required():
    """Test valid_janus validator requires spectral_bands."""
    from proteus.config._atmos_clim import valid_janus

    instance = SimpleNamespace(
        module='janus',
        janus=SimpleNamespace(spectral_group='sw', spectral_bands=None),
    )
    with pytest.raises(ValueError, match='Must set atmos_clim.janus.spectral_bands'):
        valid_janus(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_janus_valid_spectral_config():
    """Test valid_janus validator accepts valid spectral configuration."""
    from proteus.config._atmos_clim import valid_janus

    instance = SimpleNamespace(
        module='janus',
        janus=SimpleNamespace(spectral_group='sw_lw', spectral_bands='16'),
    )
    valid_janus(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_atmos_clim_agni_module_skip():
    """Test valid_agni validator skips validation when module != 'agni'."""
    from proteus.config._atmos_clim import valid_agni

    # With module='dummy', validator should return early without checking agni config
    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace())
    valid_agni(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_atmos_clim_agni_p_top_must_be_less_than_psurf_thresh():
    """Test valid_agni validator enforces p_top < psurf_thresh."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        agni=SimpleNamespace(
            p_top=0.2,  # Greater than psurf_thresh
            psurf_thresh=0.1,  # Less than p_top - INVALID
            p_obs=1.0,
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            spectral_group='sw_lw',
            spectral_bands='16',
            chemistry='none',
        ),
        surf_state='skin',
    )
    with pytest.raises(
        ValueError, match='Must set `agni.p_top` to be less than `agni.psurf_thresh`'
    ):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_p_top_must_be_less_than_p_obs():
    """Test valid_agni validator enforces p_top < p_obs."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        agni=SimpleNamespace(
            p_top=2.0,  # Greater than p_obs
            psurf_thresh=5.0,  # Valid: > p_top
            p_obs=1.0,  # Less than p_top - INVALID
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            spectral_group='sw_lw',
            spectral_bands='16',
            chemistry='none',
        ),
        surf_state='skin',
    )
    with pytest.raises(ValueError, match='Must set `agni.p_top` to be less than `agni.p_obs`'):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_solve_energy_required_for_skin_surf_state():
    """Test valid_agni validator requires solve_energy=True for surf_state='skin'."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        agni=SimpleNamespace(
            p_top=1e-5,
            psurf_thresh=0.1,
            p_obs=0.02,
            solve_energy=False,  # INVALID for skin surf_state
            latent_heat=False,
            rainout=True,
            spectral_group='sw_lw',
            spectral_bands='16',
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
        agni=SimpleNamespace(
            p_top=1e-5,
            psurf_thresh=0.1,
            p_obs=0.02,
            solve_energy=True,
            latent_heat=True,  # Requires rainout=True
            rainout=False,  # INVALID
            spectral_group='sw_lw',
            spectral_bands='16',
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
        agni=SimpleNamespace(
            p_top=1e-5,
            psurf_thresh=0.1,
            p_obs=0.02,
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            spectral_group=None,  # INVALID
            spectral_bands='16',
            chemistry='none',
        ),
        surf_state='skin',
    )
    with pytest.raises(ValueError, match='Must set atmos_clim.agni.spectral_group'):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_spectral_bands_required():
    """Test valid_agni validator requires spectral_bands."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        agni=SimpleNamespace(
            p_top=1e-5,
            psurf_thresh=0.1,
            p_obs=0.02,
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            spectral_group='sw_lw',
            spectral_bands=None,  # INVALID
            chemistry='none',
        ),
        surf_state='skin',
    )
    with pytest.raises(ValueError, match='Must set atmos_clim.agni.spectral_bands'):
        valid_agni(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_atmos_clim_agni_valid_complete_config():
    """Test valid_agni validator accepts valid AGNI configuration."""
    from proteus.config._atmos_clim import valid_agni

    instance = SimpleNamespace(
        module='agni',
        agni=SimpleNamespace(
            p_top=1e-5,
            psurf_thresh=0.1,
            p_obs=0.02,
            solve_energy=True,
            latent_heat=False,
            rainout=True,
            spectral_group='sw_lw',
            spectral_bands='16',
            chemistry='none',
        ),
        surf_state='skin',
    )
    valid_agni(instance, SimpleNamespace(), None)  # Should not raise


# ========================
# PARAMS VALIDATORS
# ========================


@pytest.mark.unit
def test_params_valid_path_non_empty_string():
    """Test valid_path validator accepts non-empty strings."""
    from proteus.config._params import valid_path

    valid_path(SimpleNamespace(), SimpleNamespace(name='data_path'), '/data/output')  # OK


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
    max_bigger_than_min(instance, SimpleNamespace(), 1000)  # Should not raise


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
    valid_mod(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_params_valid_mod_positive():
    """Test valid_mod validator accepts positive values."""
    from proteus.config._params import valid_mod

    instance = SimpleNamespace()
    valid_mod(instance, SimpleNamespace(), 1)  # OK
    valid_mod(instance, SimpleNamespace(), 100)  # OK


@pytest.mark.unit
def test_params_valid_mod_zero():
    """Test valid_mod validator accepts zero."""
    from proteus.config._params import valid_mod

    instance = SimpleNamespace()
    valid_mod(instance, SimpleNamespace(), 0)  # OK (0 means special behavior, not invalid)


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
def test_struct_mass_radius_valid_mass_tot_set():
    """Test mass_radius_valid validator accepts mass_tot being set."""
    from proteus.config._struct import mass_radius_valid

    instance = SimpleNamespace(mass_tot=1.0, radius_int=None)  # Only mass_tot set
    mass_radius_valid(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_struct_mass_radius_valid_radius_int_set():
    """Test mass_radius_valid validator accepts radius_int being set."""
    from proteus.config._struct import mass_radius_valid

    instance = SimpleNamespace(mass_tot=None, radius_int=1.0)  # Only radius_int set
    mass_radius_valid(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_struct_mass_radius_neither_set():
    """Test mass_radius_valid validator rejects when neither mass_tot nor radius_int set."""
    from proteus.config._struct import mass_radius_valid

    instance = SimpleNamespace(mass_tot=None, radius_int=None)
    with pytest.raises(ValueError, match='Must set one of'):
        mass_radius_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_mass_radius_both_set():
    """Test mass_radius_valid validator rejects when both mass_tot and radius_int set."""
    from proteus.config._struct import mass_radius_valid

    instance = SimpleNamespace(mass_tot=1.0, radius_int=1.0)
    with pytest.raises(ValueError, match='Must set either'):
        mass_radius_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_mass_radius_mass_tot_negative():
    """Test mass_radius_valid validator rejects negative mass_tot."""
    from proteus.config._struct import mass_radius_valid

    instance = SimpleNamespace(mass_tot=-1.0, radius_int=None)
    with pytest.raises(ValueError, match='must be > 0'):
        mass_radius_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_mass_radius_mass_tot_too_large():
    """Test mass_radius_valid validator rejects mass_tot > 20 M_earth."""
    from proteus.config._struct import mass_radius_valid

    instance = SimpleNamespace(mass_tot=21.0, radius_int=None)
    with pytest.raises(ValueError, match='must be < 20'):
        mass_radius_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_mass_radius_radius_int_negative():
    """Test mass_radius_valid validator rejects negative radius_int."""
    from proteus.config._struct import mass_radius_valid

    instance = SimpleNamespace(mass_tot=None, radius_int=-1.0)
    with pytest.raises(ValueError, match='must be > 0'):
        mass_radius_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_mass_radius_radius_int_too_large():
    """Test mass_radius_valid validator rejects radius_int > 10 R_earth."""
    from proteus.config._struct import mass_radius_valid

    instance = SimpleNamespace(mass_tot=None, radius_int=11.0)
    with pytest.raises(ValueError, match='must be < 10'):
        mass_radius_valid(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_module_skip():
    """Test valid_zalmoxis validator skips when module != 'zalmoxis'."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(module='self', zalmoxis=SimpleNamespace())
    valid_zalmoxis(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_struct_zalmoxis_mass_tot_required():
    """Test valid_zalmoxis validator requires mass_tot."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        mass_tot=None,  # INVALID
        zalmoxis=SimpleNamespace(
            max_iterations_outer=100,
            max_iterations_inner=100,
            max_iterations_pressure=200,
            EOSchoice='Tabulated:iron/silicate',
            coremassfrac=0.325,
            mantle_mass_fraction=0,
        ),
    )
    with pytest.raises(ValueError, match='`mass_tot` must be set'):
        valid_zalmoxis(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_max_iterations_outer_minimum():
    """Test valid_zalmoxis validator enforces max_iterations_outer > 2."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        mass_tot=1.0,
        zalmoxis=SimpleNamespace(
            max_iterations_outer=2,  # INVALID (must be > 2)
            max_iterations_inner=100,
            max_iterations_pressure=200,
            EOSchoice='Tabulated:iron/silicate',
            coremassfrac=0.325,
            mantle_mass_fraction=0,
        ),
    )
    with pytest.raises(ValueError, match='max_iterations_outer.*must be > 2'):
        valid_zalmoxis(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_max_iterations_inner_minimum():
    """Test valid_zalmoxis validator enforces max_iterations_inner > 12."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        mass_tot=1.0,
        zalmoxis=SimpleNamespace(
            max_iterations_outer=100,
            max_iterations_inner=12,  # INVALID (must be > 12)
            max_iterations_pressure=200,
            EOSchoice='Tabulated:iron/silicate',
            coremassfrac=0.325,
            mantle_mass_fraction=0,
        ),
    )
    with pytest.raises(ValueError, match='max_iterations_inner.*must be > 12'):
        valid_zalmoxis(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_max_iterations_pressure_minimum():
    """Test valid_zalmoxis validator enforces max_iterations_pressure > 12."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        mass_tot=1.0,
        zalmoxis=SimpleNamespace(
            max_iterations_outer=100,
            max_iterations_inner=100,
            max_iterations_pressure=12,  # INVALID (must be > 12)
            EOSchoice='Tabulated:iron/silicate',
            coremassfrac=0.325,
            mantle_mass_fraction=0,
        ),
    )
    with pytest.raises(ValueError, match='max_iterations_pressure.*must be > 12'):
        valid_zalmoxis(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_iron_silicate_eos_requires_no_mantle():
    """Test valid_zalmoxis requires mantle_mass_fraction=0 for Tabulated:iron/silicate."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        mass_tot=1.0,
        zalmoxis=SimpleNamespace(
            max_iterations_outer=100,
            max_iterations_inner=100,
            max_iterations_pressure=200,
            EOSchoice='Tabulated:iron/silicate',
            coremassfrac=0.325,
            mantle_mass_fraction=0.2,  # INVALID (must be 0)
        ),
    )
    with pytest.raises(ValueError, match='mantle_mass_fraction.*must be 0'):
        valid_zalmoxis(instance, SimpleNamespace(), None)


@pytest.mark.unit
def test_struct_zalmoxis_water_eos_mass_constraint():
    """Test valid_zalmoxis enforces core+mantle mass < 75% for Tabulated:water EOS."""
    from proteus.config._struct import valid_zalmoxis

    instance = SimpleNamespace(
        module='zalmoxis',
        mass_tot=1.0,
        zalmoxis=SimpleNamespace(
            max_iterations_outer=100,
            max_iterations_inner=100,
            max_iterations_pressure=200,
            EOSchoice='Tabulated:water',
            coremassfrac=0.5,  # 50%
            mantle_mass_fraction=0.3,  # 30%, sum=80% > 75% INVALID
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
        mass_tot=1.0,
        zalmoxis=SimpleNamespace(
            max_iterations_outer=100,
            max_iterations_inner=100,
            max_iterations_pressure=200,
            EOSchoice='Tabulated:iron/silicate',
            coremassfrac=0.325,
            mantle_mass_fraction=0,
        ),
    )
    valid_zalmoxis(instance, SimpleNamespace(), None)  # Should not raise


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
    spada_zephyrus(instance, SimpleNamespace(), None)  # Should not raise


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
    instmethod_dummy(instance, SimpleNamespace(), None)  # Should not raise


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
    instmethod_evolve(instance, SimpleNamespace(), None)  # Should not raise


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
    satellite_evolve(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_config_tides_enabled_orbit_requires_orbit_module():
    """Test tides_enabled_orbit validator requires orbit module for tidal heating."""
    from proteus.config._config import tides_enabled_orbit

    # Invalid: tidal heating without orbit module
    instance = SimpleNamespace(
        interior=SimpleNamespace(tidal_heat=True),
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
        interior=SimpleNamespace(tidal_heat=False),
        orbit=SimpleNamespace(module=None),
    )
    tides_enabled_orbit(instance, SimpleNamespace(), None)  # Should not raise


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
    observe_resolved_atmosphere(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_config_janus_escape_atmosphere_requires_stop_escape():
    """Test janus_escape_atmosphere validator requires stop.escape=True."""
    from proteus.config._config import janus_escape_atmosphere

    # Invalid: Zephyrus + JANUS without stop.escape
    instance = SimpleNamespace(
        escape=SimpleNamespace(module='zephyrus'),
        atmos_clim=SimpleNamespace(module='janus'),
        params=SimpleNamespace(stop=SimpleNamespace(escape=False)),  # INVALID
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
        params=SimpleNamespace(stop=SimpleNamespace(escape=False)),
    )
    janus_escape_atmosphere(instance, SimpleNamespace(), None)  # Should not raise


@pytest.mark.unit
def test_config_janus_escape_atmosphere_non_janus_skip():
    """Test janus_escape_atmosphere validator skips for non-JANUS atmosphere."""
    from proteus.config._config import janus_escape_atmosphere

    # Valid: AGNI instead of JANUS
    instance = SimpleNamespace(
        escape=SimpleNamespace(module='zephyrus'),
        atmos_clim=SimpleNamespace(module='agni'),
        params=SimpleNamespace(stop=SimpleNamespace(escape=False)),
    )
    janus_escape_atmosphere(instance, SimpleNamespace(), None)  # Should not raise
