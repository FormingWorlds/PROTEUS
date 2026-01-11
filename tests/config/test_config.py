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
from proteus.config import Config, read_config_object
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
