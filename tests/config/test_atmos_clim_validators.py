"""Tests for atmos_clim config validators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from proteus.config._atmos_clim import valid_agni, valid_rayleigh

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
