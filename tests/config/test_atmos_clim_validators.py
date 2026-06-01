"""Tests for atmos_clim config validators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from proteus.config._atmos_clim import valid_agni, valid_rayleigh


def _make_agni_instance(**agni_kwargs):
    agni_defaults = {
        'p_top': 1e-5,
        'psurf_thresh': 1.0,
        'p_obs': 1e-3,
        'solve_energy': True,
        'latent_heat': False,
        'rainout': True,
        'spectral_file': None,
        'spectral_group': 'Honeyside',
        'spectral_bands': '64',
        'chemistry': 'none',
    }
    agni_defaults.update(agni_kwargs)
    return SimpleNamespace(
        module='agni',
        surf_state='mixed_layer',
        agni=SimpleNamespace(**agni_defaults),
    )


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
    with pytest.raises(ValueError, match='Must set atmos_clim.agni.spectral_group'):
        valid_agni(instance, attribute=None, value=None)


@pytest.mark.unit
def test_valid_agni_requires_spectral_bands_when_no_file():
    instance = _make_agni_instance(spectral_file=None, spectral_bands='')
    with pytest.raises(ValueError, match='Must set atmos_clim.agni.spectral_bands'):
        valid_agni(instance, attribute=None, value=None)
