"""Tests for atmos_clim config validators."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from proteus.config._atmos_clim import (
    surf_state_to_int,
    valid_agni,
    valid_p_obs,
    valid_rayleigh,
    valid_surface_skin,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_agni_instance(**kwargs):
    """Build a stand-in for the AtmosClim attrs instance used by valid_agni.

    On this branch, `spectral_group`, `spectral_bands`, `p_top`, and `p_obs`
    live on the parent AtmosClim class (shared with JANUS), and AGNI-specific
    fields live under `agni`.
    """
    parent_defaults = {
        'module': 'agni',
        'surf_state': 'fixed',
        'spectral_group': 'Honeyside',
        'spectral_bands': '64',
        'p_top': 1e-5,
        'p_obs': 1e-3,
    }
    agni_defaults = {
        'psurf_thresh': 1.0,
        'solve_energy': True,
        'latent_heat': False,
        'rainout': True,
        'spectral_file': None,
        'chemistry': 'none',
        'load_all_opacities': True,
    }
    parent_overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in parent_defaults}
    parent_defaults.update(parent_overrides)
    agni_defaults.update(kwargs)
    return SimpleNamespace(agni=SimpleNamespace(**agni_defaults), **parent_defaults)


@pytest.mark.unit
def test_valid_rayleigh_rejects_dummy_module():
    """Rayleigh scattering cannot be enabled with the dummy atmos_clim
    module; the validator raises with an 'incompatible with Rayleigh
    scattering' message.
    """
    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace(spectral_file=None))
    with pytest.raises(ValueError, match='incompatible with Rayleigh scattering'):
        valid_rayleigh(instance, attribute=None, value=True)
    # Discrimination: the same instance with value=False must NOT raise,
    # so the rejection is driven by the rayleigh flag, not the dummy
    # module alone. A regression that hard-raised on every call would
    # also raise here.
    assert valid_rayleigh(instance, attribute=None, value=False) is None


@pytest.mark.unit
def test_valid_rayleigh_rejects_agni_greygas():
    """Rayleigh scattering cannot be enabled with AGNI's grey-gas RT
    (no band structure to scatter in); the validator raises with a
    'grey gas is incompatible' message.
    """
    instance = SimpleNamespace(module='agni', agni=SimpleNamespace(spectral_file='greygas'))
    with pytest.raises(ValueError, match='grey gas is incompatible with Rayleigh scattering'):
        valid_rayleigh(instance, attribute=None, value=True)
    # Discrimination: swapping the spectral_file from greygas to a real
    # band-resolved path on the same AGNI instance must make the
    # validator pass. A regression that rejected every AGNI value
    # regardless of the spectral file would still raise here.
    instance.agni.spectral_file = '/path/to/sw_lw.spc'
    assert valid_rayleigh(instance, attribute=None, value=True) is None


@pytest.mark.unit
def test_valid_rayleigh_passes_when_disabled():
    """valid_rayleigh is a no-op when rayleigh is False.

    Both module='dummy' and the AGNI-greygas combination would normally
    raise; passing value=False must short-circuit before either guard.
    """
    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace(spectral_file='greygas'))
    result = valid_rayleigh(instance, attribute=None, value=False)
    assert result is None  # contract: validator returns None silently on the pass path
    assert instance.module == 'dummy'  # validator must not mutate the input instance


@pytest.mark.unit
def test_valid_agni_allows_greygas_spectral_file():
    """AGNI with ``spectral_file='greygas'`` is the analytic grey-gas
    configuration; the validator accepts it silently.

    The other AGNI checks (psurf_thresh > p_top, p_obs > p_top, surf_state,
    solve_energy) all pass with the defaults; the spectral_file branch is
    what this test pins.
    """
    instance = _make_agni_instance(spectral_file='greygas')
    result = valid_agni(instance, attribute=None, value=None)
    assert result is None  # contract: validator returns None silently on the pass path
    assert instance.agni.spectral_file == 'greygas'  # no mutation of the greygas marker


@pytest.mark.unit
def test_valid_agni_rejects_missing_spectral_file_path():
    """A non-existent spectral file path raises FileNotFoundError with the
    'AGNI spectral file not found' message, so a typo in config does not
    silently fall back to a different file.
    """
    instance = _make_agni_instance(spectral_file='/this/path/does/not/exist.spc')
    with pytest.raises(FileNotFoundError, match='AGNI spectral file not found'):
        valid_agni(instance, attribute=None, value=None)
    # Discrimination: swapping to the analytic 'greygas' sentinel must
    # pass on the same instance, so the rejection is driven by the
    # missing path, not by all string spectral_files. A regression
    # that always rejected non-None spectral_file would still raise
    # here.
    instance.agni.spectral_file = 'greygas'
    assert valid_agni(instance, attribute=None, value=None) is None


@pytest.mark.unit
def test_valid_agni_requires_spectral_group_when_no_file():
    """With no explicit spectral file, ``spectral_group`` must be set so
    the runtime can construct the file path; an empty value raises.
    """
    instance = _make_agni_instance(spectral_file=None, spectral_group='')
    with pytest.raises(ValueError, match='Must set atmos_clim.spectral_group'):
        valid_agni(instance, attribute=None, value=None)
    # Discrimination: restoring a non-empty spectral_group on the same
    # instance must make the validator pass. A regression that
    # rejected every spectral_file=None instance regardless of the
    # group would still raise here.
    instance.spectral_group = 'Honeyside'
    assert valid_agni(instance, attribute=None, value=None) is None


@pytest.mark.unit
def test_valid_agni_requires_spectral_bands_when_no_file():
    """With no explicit spectral file, ``spectral_bands`` must be set so
    the runtime can construct the file path; an empty value raises.
    """
    instance = _make_agni_instance(spectral_file=None, spectral_bands='')
    with pytest.raises(ValueError, match='Must set atmos_clim.spectral_bands'):
        valid_agni(instance, attribute=None, value=None)
    # Discrimination: restoring a non-empty spectral_bands on the same
    # instance must make the validator pass. A regression that
    # rejected every spectral_file=None instance regardless of the
    # band setting would still raise here.
    instance.spectral_bands = '64'
    assert valid_agni(instance, attribute=None, value=None) is None


# ============================================================================
# Regression: aerosols + grey gas / dummy must raise
# ============================================================================


@pytest.mark.unit
def test_valid_aerosols_enabled_rejects_agni_greygas():
    """Regression: AGNI grey-gas RT has no spectral bands, so aerosol
    Mie data is either silently ignored or crashes Julia-side. The
    validator must catch this at config-load time."""
    from proteus.config._atmos_clim import valid_aerosols_enabled

    instance = SimpleNamespace(module='agni', agni=SimpleNamespace(spectral_file='greygas'))
    with pytest.raises(ValueError, match='aerosols'):
        valid_aerosols_enabled(instance, SimpleNamespace(name='aerosols_enabled'), True)
    # Discrimination: passing value=False on the same greygas instance
    # must short-circuit and return None. A regression that hard-raised
    # on greygas regardless of the aerosols flag would still raise here.
    assert (
        valid_aerosols_enabled(instance, SimpleNamespace(name='aerosols_enabled'), False)
        is None
    )


@pytest.mark.unit
def test_valid_aerosols_enabled_rejects_dummy_module():
    """Aerosols also incompatible with the dummy atmos_clim module
    (the dummy uses analytic grey-body opacity; no aerosols loop)."""
    from proteus.config._atmos_clim import valid_aerosols_enabled

    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace(spectral_file=None))
    with pytest.raises(ValueError, match='Dummy atmos_clim'):
        valid_aerosols_enabled(instance, SimpleNamespace(name='aerosols_enabled'), True)
    # Discrimination: swapping the module to AGNI with a real spectral
    # file on the same instance must make the validator pass. A
    # regression that rejected every value=True call regardless of the
    # module would still raise here.
    instance.module = 'agni'
    instance.agni.spectral_file = '/path/to/sw_lw.spc'
    assert (
        valid_aerosols_enabled(instance, SimpleNamespace(name='aerosols_enabled'), True) is None
    )


@pytest.mark.unit
def test_valid_aerosols_enabled_passes_for_agni_with_path():
    """AGNI with a real band-resolved spectral file is the canonical
    aerosols-enabled configuration. Must NOT raise."""
    from proteus.config._atmos_clim import valid_aerosols_enabled

    instance = SimpleNamespace(
        module='agni', agni=SimpleNamespace(spectral_file='/path/to/sw_lw.spc')
    )
    result = valid_aerosols_enabled(instance, SimpleNamespace(name='aerosols_enabled'), True)
    assert result is None  # contract: validator returns None silently on the pass path
    assert instance.agni.spectral_file == '/path/to/sw_lw.spc'  # spectral_file unchanged


@pytest.mark.unit
def test_valid_aerosols_enabled_no_op_when_disabled():
    """A False value bypasses every guard, regardless of module.

    Both module='dummy' and the AGNI-greygas combination would normally
    raise; passing value=False must short-circuit before either guard.
    """
    from proteus.config._atmos_clim import valid_aerosols_enabled

    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace(spectral_file='greygas'))
    result = valid_aerosols_enabled(instance, SimpleNamespace(name='aerosols_enabled'), False)
    assert result is None  # contract: validator returns None silently on the pass path
    assert instance.module == 'dummy'  # validator must not mutate the input instance


# ============================================================================
# valid_agni: pressure-ordering and surface-state guards
# ============================================================================


@pytest.mark.unit
def test_valid_agni_rejects_p_top_above_psurf_thresh():
    """`p_top` must sit below AGNI's `psurf_thresh`, otherwise the top-of-
    atmosphere pressure is above the surface threshold and the column is
    ill-posed; the validator raises."""
    instance = _make_agni_instance(p_top=2.0)  # psurf_thresh defaults to 1.0
    with pytest.raises(ValueError, match='p_top'):
        valid_agni(instance, attribute=None, value=None)
    # Discrimination: a p_top safely below psurf_thresh passes, so the guard
    # is the ordering, not any non-default p_top.
    instance.p_top = 1e-4
    assert valid_agni(instance, attribute=None, value=None) is None


@pytest.mark.unit
def test_valid_agni_rejects_p_top_above_p_obs():
    """`p_top` must also sit below the observation pressure `p_obs`; a p_top
    below psurf_thresh but above p_obs is still rejected."""
    # p_top=1e-2 is below psurf_thresh (1.0) but above p_obs (1e-3).
    instance = _make_agni_instance(p_top=1e-2)
    with pytest.raises(ValueError, match='p_obs'):
        valid_agni(instance, attribute=None, value=None)
    # Discrimination: lowering p_top below p_obs on the same instance passes,
    # so the rejection is the p_top/p_obs ordering specifically.
    instance.p_top = 1e-5
    assert valid_agni(instance, attribute=None, value=None) is None


@pytest.mark.unit
def test_valid_agni_skips_p_obs_ordering_check_when_p_obs_is_none():
    """p_obs=None (AGNI determines the photosphere from optical depth,
    issue #694) has no fixed pressure to compare against p_top, so the
    ordering check must not raise regardless of p_top.

    Discrimination: the same p_top=1e-2 rejected in
    test_valid_agni_rejects_p_top_above_p_obs (against a numeric p_obs of
    1e-3) must pass here once p_obs is None, so the guard is specifically
    the numeric-comparison path, not a blanket bypass of the whole function.
    """
    instance = _make_agni_instance(p_top=1e-2, p_obs=None)
    assert valid_agni(instance, attribute=None, value=None) is None
    # Discrimination: setting p_obs back to a numeric value below p_top on
    # this same instance raises, so the bypass above is specifically about
    # p_obs=None, not a coincidental pass from some other field.
    instance.p_obs = 1e-3
    with pytest.raises(ValueError, match='p_obs'):
        valid_agni(instance, attribute=None, value=None)


@pytest.mark.unit
def test_valid_p_obs_accepts_none_only_for_agni():
    """p_obs=None delegates the photosphere to AGNI's optical-depth
    determination (issue #694), which only AGNI implements; JANUS and the
    dummy module have no equivalent, so None is rejected for them.
    """
    instance = SimpleNamespace(module='agni')
    assert valid_p_obs(instance, attribute=None, value=None) is None

    for other_module in ('janus', 'dummy'):
        instance = SimpleNamespace(module=other_module)
        with pytest.raises(ValueError, match='agni'):
            valid_p_obs(instance, attribute=None, value=None)


@pytest.mark.unit
def test_valid_p_obs_rejects_non_positive_numeric_value():
    """A numeric p_obs must be a positive pressure; zero or negative values
    are not physical pressure levels.

    Edge case: exactly zero is rejected, not just negative values.
    """
    instance = SimpleNamespace(module='agni')
    with pytest.raises(ValueError, match='p_obs'):
        valid_p_obs(instance, attribute=None, value=0.0)
    with pytest.raises(ValueError, match='p_obs'):
        valid_p_obs(instance, attribute=None, value=-1e-3)
    # Discrimination: a small but positive value passes.
    assert valid_p_obs(instance, attribute=None, value=1e-10) is None


@pytest.mark.unit
def test_valid_agni_rejects_mixed_layer_surf_state():
    """AGNI has no mixed-layer surface boundary condition; selecting it raises
    rather than silently falling back to a different state."""
    instance = _make_agni_instance(surf_state='mixed_layer')
    with pytest.raises(ValueError, match='mixed_layer'):
        valid_agni(instance, attribute=None, value=None)
    # Discrimination: a supported surf_state on the same instance passes, so
    # the rejection is mixed_layer specifically, not any non-default state.
    instance.surf_state = 'skin'
    assert valid_agni(instance, attribute=None, value=None) is None


# ============================================================================
# AtmosClim properties: surf_state_int
# ============================================================================


@pytest.mark.unit
def test_surf_state_int_maps_skin_and_rejects_unknown():
    """`surf_state_int` maps the AGNI surface state to the integer code the
    Julia side expects: 'skin' -> 2, with an unknown state raising."""
    from proteus.config._atmos_clim import AtmosClim

    assert AtmosClim.surf_state_int.fget(SimpleNamespace(surf_state='skin')) == 2
    # Discrimination: 'fixed' maps to a different code (1), so the property is
    # a real mapping, not a constant.
    assert AtmosClim.surf_state_int.fget(SimpleNamespace(surf_state='fixed')) == 1
    # Error contract: an unsupported state raises rather than returning a
    # silent default integer.
    with pytest.raises(ValueError, match='surf_state'):
        AtmosClim.surf_state_int.fget(SimpleNamespace(surf_state='mixed_layer'))


# ============================================================================
# Agni: linesearch and Broyden options
# ============================================================================


@pytest.mark.unit
def test_agni_ls_default_takes_only_agni_linesearch_methods():
    """`ls_default` takes each of AGNI's linesearch methods, including 3,
    the non-monotone Armijo linesearch added in AGNI 1.14.0, and refuses a
    code AGNI does not have. AGNI would otherwise stop the solve with a
    configuration error on its first linesearch."""
    from proteus.config._atmos_clim import Agni

    for method in (0, 1, 2, 3):
        assert Agni(ls_default=method).ls_default == method
    # Edges: the codes either side of the valid range.
    for method in (-1, 4):
        with pytest.raises(ValueError, match='ls_default'):
            Agni(ls_default=method)


@pytest.mark.unit
def test_agni_defaults_to_broyden_with_the_armijo_linesearch():
    """PROTEUS defaults to Broyden jacobian updates with the non-monotone
    Armijo linesearch, the pairing AGNI intends Broyden for, although AGNI
    itself defaults to neither. Each can still be turned back."""
    from proteus.config._atmos_clim import Agni

    defaults = Agni()
    assert defaults.broyden is True
    assert defaults.ls_default == 3
    assert Agni(broyden=False).broyden is False


@pytest.mark.unit
def test_agni_loads_every_opacity_unless_asked_not_to():
    """`load_all_opacities` defaults to true, AGNI's own default, so a config
    written before the option existed loads every gas of the spectral file as
    before. A configured false is kept, and `input/all_options.toml` states the
    default."""
    from proteus.config import read_config_object
    from proteus.config._atmos_clim import Agni

    assert Agni().load_all_opacities is True
    assert Agni(load_all_opacities=False).load_all_opacities is False
    root = Path(__file__).parents[2]
    cfg = read_config_object(root / 'input' / 'all_options.toml')
    assert cfg.atmos_clim.agni.load_all_opacities is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ('chemistry', 'load_all', 'warned'),
    [
        (None, False, False),
        ('eq', False, True),
        ('eq', True, False),
    ],
    ids=[
        'modelled_gases_without_chemistry',
        'eq_chemistry_with_modelled_gases_only',
        'eq_chemistry_with_all_gases',
    ],
)
def test_valid_agni_warns_when_eq_chemistry_makes_load_all_opacities_false_moot(
    monkeypatch, tmp_path, caplog, chemistry, load_all, warned
):
    """With `chemistry = "eq"` FastChem can add any gas of the spectral file, so
    AGNI loads every opacity. A configured `load_all_opacities = false` is then
    reported once, at config load, and the config value is left unchanged."""
    monkeypatch.setenv('FC_DIR', str(tmp_path))
    instance = _make_agni_instance(chemistry=chemistry, load_all_opacities=load_all)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.config._atmos_clim'):
        assert valid_agni(instance, attribute=None, value=None) is None
    assert instance.agni.load_all_opacities is load_all
    assert ('load_all_opacities = false has no effect' in caplog.text) is warned


# ============================================================================
# valid_surface_skin: a positive number, or 'saturation' for both with a skin
# ============================================================================


def _skin_instance(surface_d=0.01, surface_k=2.0, surf_state='skin'):
    """Stand-in for the AtmosClim fields that valid_surface_skin reads."""
    return SimpleNamespace(surface_d=surface_d, surface_k=surface_k, surf_state=surf_state)


@pytest.mark.unit
def test_valid_surface_skin_accepts_numbers_and_a_saturation_pair():
    """A positive float or int is accepted for each field, and 'saturation' when both
    fields hold it with surf_state='skin'."""
    for name in ('surface_d', 'surface_k'):
        attribute = SimpleNamespace(name=name)
        assert valid_surface_skin(_skin_instance(), attribute, 0.01) is None
        assert valid_surface_skin(_skin_instance(), attribute, 3) is None
        pair = _skin_instance('saturation', 'saturation')
        assert valid_surface_skin(pair, attribute, 'saturation') is None
    # A numeric skin is allowed with any surface state
    fixed = _skin_instance(surf_state='fixed')
    assert valid_surface_skin(fixed, SimpleNamespace(name='surface_d'), 1.0e-4) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ('value', 'match'),
    [(0.0, 'positive'), (-0.01, 'positive'), (True, 'positive'), ('bogus', 'positive')],
    ids=['zero-thickness', 'negative-thickness', 'boolean', 'unknown-string'],
)
def test_valid_surface_skin_rejects_a_non_positive_or_non_numeric_value(value, match):
    """Zero, a negative value, a boolean (which would pass `> 0`) and any other string
    are refused for the skin thickness."""
    attribute = SimpleNamespace(name='surface_d')
    with pytest.raises(ValueError, match=match):
        valid_surface_skin(_skin_instance(), attribute, value)
    # Discrimination: the smallest positive value on the same instance passes
    assert valid_surface_skin(_skin_instance(), attribute, 1.0e-12) is None


@pytest.mark.unit
def test_valid_surface_skin_requires_both_fields_and_the_skin_state():
    """'saturation' on one field alone, or with surf_state other than 'skin', raises;
    the error names the field that is missing."""
    attribute_d = SimpleNamespace(name='surface_d')
    attribute_k = SimpleNamespace(name='surface_k')
    with pytest.raises(ValueError, match='surface_k'):
        valid_surface_skin(_skin_instance('saturation', 2.0), attribute_d, 'saturation')
    with pytest.raises(ValueError, match='surface_d'):
        valid_surface_skin(_skin_instance(0.01, 'saturation'), attribute_k, 'saturation')
    fixed_pair = _skin_instance('saturation', 'saturation', surf_state='fixed')
    with pytest.raises(ValueError, match="surf_state='skin'"):
        valid_surface_skin(fixed_pair, attribute_d, 'saturation')


@pytest.mark.unit
def test_saturation_skin_loads_from_toml_and_a_half_pair_is_refused(tmp_path):
    """The TOML string 'saturation' reaches both fields through the config loader, and
    a file that sets it for surface_d alone fails to load."""
    import re

    from proteus.config import read_config_object

    root = Path(__file__).resolve().parents[2]
    text = (root / 'input' / 'all_options.toml').read_text()
    text = re.sub(r'surface_d\s*=\s*0\.01', "surface_d = 'saturation'", text)
    both = re.sub(r'surface_k\s*=\s*2\.0', "surface_k = 'saturation'", text)
    (tmp_path / 'both.toml').write_text(both)
    cfg = read_config_object(tmp_path / 'both.toml')
    assert (cfg.atmos_clim.surface_d, cfg.atmos_clim.surface_k) == ('saturation', 'saturation')
    assert cfg.atmos_clim.surf_state == 'skin'

    (tmp_path / 'half.toml').write_text(text)
    with pytest.raises(ValueError, match="surface_k='saturation'"):
        read_config_object(tmp_path / 'half.toml')


@pytest.mark.unit
def test_surf_state_to_int_maps_the_effective_state():
    """The module-level mapping gives AGNI 1 for 'fixed' and 2 for 'skin', which the
    coupler uses for the state of each iteration, and raises for 'mixed_layer'."""
    assert surf_state_to_int('fixed') == 1
    assert surf_state_to_int('skin') == 2
    with pytest.raises(ValueError, match='mixed_layer'):
        surf_state_to_int('mixed_layer')
