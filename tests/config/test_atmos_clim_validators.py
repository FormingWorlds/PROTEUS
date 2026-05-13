"""Tests for atmos_clim config validators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from proteus.config._atmos_clim import valid_agni, valid_rayleigh


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
    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace(spectral_file=None))
    with pytest.raises(ValueError, match='incompatible with Rayleigh scattering'):
        valid_rayleigh(instance, attribute=None, value=True)


@pytest.mark.unit
def test_valid_rayleigh_rejects_agni_greygas():
    instance = SimpleNamespace(module='agni', agni=SimpleNamespace(spectral_file='greygas'))
    with pytest.raises(ValueError, match='grey gas is incompatible with Rayleigh scattering'):
        valid_rayleigh(instance, attribute=None, value=True)


@pytest.mark.unit
def test_valid_rayleigh_passes_when_disabled():
    """valid_rayleigh is a no-op when rayleigh is False."""
    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace(spectral_file='greygas'))
    valid_rayleigh(instance, attribute=None, value=False)


@pytest.mark.unit
def test_valid_agni_allows_greygas_spectral_file():
    instance = _make_agni_instance(spectral_file='greygas')
    valid_agni(instance, attribute=None, value=None)


@pytest.mark.unit
def test_valid_agni_rejects_missing_spectral_file_path():
    instance = _make_agni_instance(spectral_file='/this/path/does/not/exist.spc')
    with pytest.raises(FileNotFoundError, match='AGNI spectral file not found'):
        valid_agni(instance, attribute=None, value=None)


@pytest.mark.unit
def test_valid_agni_requires_spectral_group_when_no_file():
    instance = _make_agni_instance(spectral_file=None, spectral_group='')
    with pytest.raises(ValueError, match='Must set atmos_clim.spectral_group'):
        valid_agni(instance, attribute=None, value=None)


@pytest.mark.unit
def test_valid_agni_requires_spectral_bands_when_no_file():
    instance = _make_agni_instance(spectral_file=None, spectral_bands='')
    with pytest.raises(ValueError, match='Must set atmos_clim.spectral_bands'):
        valid_agni(instance, attribute=None, value=None)


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


@pytest.mark.unit
def test_valid_aerosols_enabled_rejects_dummy_module():
    """Aerosols also incompatible with the dummy atmos_clim module
    (the dummy uses analytic grey-body opacity; no aerosols loop)."""
    from proteus.config._atmos_clim import valid_aerosols_enabled

    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace(spectral_file=None))
    with pytest.raises(ValueError, match='Dummy atmos_clim'):
        valid_aerosols_enabled(instance, SimpleNamespace(name='aerosols_enabled'), True)


@pytest.mark.unit
def test_valid_aerosols_enabled_passes_for_agni_with_path():
    """AGNI with a real band-resolved spectral file is the canonical
    aerosols-enabled configuration. Must NOT raise."""
    from proteus.config._atmos_clim import valid_aerosols_enabled

    instance = SimpleNamespace(
        module='agni', agni=SimpleNamespace(spectral_file='/path/to/sw_lw.spc')
    )
    # No raise.
    valid_aerosols_enabled(instance, SimpleNamespace(name='aerosols_enabled'), True)


@pytest.mark.unit
def test_valid_aerosols_enabled_no_op_when_disabled():
    """A False value bypasses every guard, regardless of module."""
    from proteus.config._atmos_clim import valid_aerosols_enabled

    instance = SimpleNamespace(module='dummy', agni=SimpleNamespace(spectral_file='greygas'))
    valid_aerosols_enabled(instance, SimpleNamespace(name='aerosols_enabled'), False)
