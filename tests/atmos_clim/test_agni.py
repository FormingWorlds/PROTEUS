"""
Unit tests for proteus.atmos_clim.agni module.

This module tests the AGNI atmosphere interface including:
- Aerosol discovery (_determine_aerosols)
- Condensate species determination (_determine_condensates)
- AGNI atmosphere initialization (init_agni_atmos)
- Temperature-profile carry-over between iterations (_validate_stored_profile,
  update_agni_atmos), covering pressure and temperature positivity, profile
  monotonicity under interpolation, and the atmosphere-failure contract when
  the surface boundary condition cannot seed a new profile

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import proteus.atmos_clim.agni as agni_mod
from proteus.atmos_clim.agni import (
    _determine_aerosols,
    _determine_condensates,
    _validate_stored_profile,
    init_agni_atmos,
    write_atmos_ncdf,
)
from proteus.utils.constants import noble_gases

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.listdir')
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_success(mock_isdir, mock_listdir):
    """
    Test aerosol discovery when scattering data directory exists.

    Physical scenario: Scattering data for aerosols (e.g., sulfate, silicate)
    is available in FWL_DATA/scattering/scattering/*.mon files.
    """
    mock_isdir.return_value = True
    mock_listdir.return_value = [
        'Sulfate.mon',
        'Silicate.mon',
        'Haze.mon',
        'other_file.txt',  # Should be ignored
        'readme.md',  # Should be ignored
    ]

    dirs = {'fwl': '/fake/fwl/path'}
    aerosols = _determine_aerosols(dirs)

    # Verify correct aerosols found and sorted
    assert len(aerosols) == 3
    assert aerosols == ['Haze', 'Silicate', 'Sulfate']  # alphabetically sorted

    # Verify correct directory was checked
    mock_isdir.assert_called_once_with('/fake/fwl/path/scattering/scattering')


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_missing_directory(mock_isdir):
    """
    Test aerosol discovery when scattering directory doesn't exist.

    Physical scenario: FWL_DATA not properly downloaded or scattering
    data not installed. Should return empty list and warn.
    """
    mock_isdir.return_value = False

    dirs = {'fwl': '/nonexistent/path'}
    aerosols = _determine_aerosols(dirs)

    # Should return empty list without crashing
    assert aerosols == []
    mock_isdir.assert_called_once()


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.listdir')
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_empty_directory(mock_isdir, mock_listdir):
    """
    Test aerosol discovery when directory exists but has no .mon files.

    Physical scenario: Scattering directory present but empty or only
    contains non-aerosol files.
    """
    mock_isdir.return_value = True
    mock_listdir.return_value = ['readme.txt', 'config.yaml']

    dirs = {'fwl': '/path/to/fwl'}
    aerosols = _determine_aerosols(dirs)

    # Should return empty list
    assert aerosols == []
    # Discrimination guard: the directory existed, so isdir must have
    # been queried AND listdir must have been called to inspect the
    # contents. A regression that returned [] without inspecting (e.g.
    # always short-circuited) would still pass the assertion above.
    mock_isdir.assert_called_once_with('/path/to/fwl/scattering/scattering')
    mock_listdir.assert_called_once()
    # Type guard: returning None or a non-list would also satisfy
    # `== []` against another empty container, so pin the type.
    assert isinstance(aerosols, list)


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.listdir')
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_single_species(mock_isdir, mock_listdir):
    """
    Test aerosol discovery with only one aerosol type.

    Physical scenario: Limited scattering data with only one aerosol species
    available (e.g., only sulfate aerosols).
    """
    mock_isdir.return_value = True
    mock_listdir.return_value = ['Sulfate.mon']

    dirs = {'fwl': '/path/to/fwl'}
    aerosols = _determine_aerosols(dirs)

    assert len(aerosols) == 1
    assert aerosols == ['Sulfate']


@pytest.mark.unit
def test_determine_condensates():
    """
    Test condensate species determination from volatile list.

    Physical scenario: given a list of volatile species, filter out the
    always-dry gases (H2, N2, CO, and the inert noble gases) to leave the
    condensable species like H2O, CO2, CH4.
    """
    # Test with mixed list of dry, condensable, and noble species
    vol_list = ['H2O', 'CO2', 'N2', 'CH4', 'He', 'Ar', 'H2', 'CO']
    condensates = _determine_condensates(vol_list)

    # H2O, CO2, CH4 remain condensable
    assert 'H2O' in condensates
    assert 'CO2' in condensates
    assert 'CH4' in condensates
    # N2, H2, CO are always dry
    assert 'N2' not in condensates
    assert 'H2' not in condensates
    assert 'CO' not in condensates
    # Noble gases are chemically inert and never condense
    assert 'He' not in condensates
    assert 'Ar' not in condensates


@pytest.mark.unit
def test_determine_condensates_all_dry():
    """
    Test condensate determination with only dry species.

    Physical scenario: Hydrogen-nitrogen-CO dominated atmosphere with no
    condensable species.
    """
    vol_list = ['H2', 'N2', 'CO']
    condensates = _determine_condensates(vol_list)

    # Should return empty list
    assert condensates == []
    # Discrimination guard: the input list is non-empty (len 3), so a
    # regression that returned the input unchanged would land at
    # ['H2', 'N2', 'CO'] not []. Pin the explicit filter behaviour.
    assert len(condensates) < len(vol_list)
    # The input is not mutated: filtering must not change the caller's
    # list (single-gas warning path notwithstanding).
    assert vol_list == ['H2', 'N2', 'CO']


@pytest.mark.unit
def test_determine_condensates_all_condensable():
    """
    Test condensate determination with all condensable species.

    Physical scenario: Rocky planet atmosphere with water, CO2, and other
    condensable volatiles but no hydrogen/helium.
    """
    vol_list = ['H2O', 'CO2', 'NH3', 'CH4', 'SO2']
    condensates = _determine_condensates(vol_list)

    # All should remain
    assert len(condensates) == len(vol_list)
    assert set(condensates) == set(vol_list)


@pytest.mark.unit
def test_determine_condensates_empty_list():
    """
    Test condensate determination with empty volatile list.

    Physical scenario: Edge case where no volatiles are specified.
    Empty input must traverse the list-comprehension path (not the
    single-gas warning short-circuit), so the return must be a fresh
    empty list, not the caller's list aliased back.
    """
    input_list = []
    condensates = _determine_condensates(input_list)
    assert condensates == []
    # Type guard: a regression returning None on empty input would
    # also satisfy `== []` against the empty container of the wrong
    # type? `None == []` is False, so `==` alone catches that. Pin
    # the list type explicitly for clarity.
    assert isinstance(condensates, list)
    # Idempotency: a second call with the same empty input must return
    # the same empty result. A regression that introduced caller-state
    # would diverge.
    second = _determine_condensates([])
    assert second == condensates


class _FakeAtmosphere:
    """Stand-in struct used by the `init_agni_atmos` dispatch tests.

    Carries every field listed in `_REQUIRED_ATMOS_FIELDS` so the
    schema check at allocate succeeds. Tests that need to vary a
    specific field assign it on the instance after construction.
    """

    def __init__(self):
        self.transparent = False
        # Pressure-temperature state
        self.tmp = [300.0]
        self.tmpl = [295.0, 305.0]
        self.pl = [1e4, 1e5]
        self.p_boa = 1e5
        self.p_oboa = 1e5
        self.tmp_surf = 1500.0
        self.tmp_magma = 1500.0
        # Cell-centre gravity, read at the XUV level
        self.g = [9.8]
        # Solver flags
        self.is_converged = True
        # Allocation flag, gating write_atmos_ncdf
        self.is_alloc = True
        # Radiative fluxes
        self.flux_d_sw = [100.0]
        self.flux_u_lw = [300.0]
        self.flux_u_sw = [10.0]
        self.flux_tot = [100.0, 200.0]
        # AGNI 1.10.2 additive
        self.tau_band = [[0.0], [0.5]]
        self.diagnostic_Ra = [1.0]
        self.timescale_conv = [1e3]
        self.timescale_rad = [1e6]
        # Gas composition
        self.gas_names = ['H2O']
        self.gas_vmr = {'H2O': [1.0]}
        self.gas_ovmr = {'H2O': [1.0]}
        # Ocean diagnostics
        self.ocean_areacov = 0.0
        self.ocean_maxdepth = 0.0
        self.ocean_tot = 0.0
        # Stellar / transit
        self.instellation = 1361.0
        self.transspec_p = 1e2
        self.transspec_r = 6.371e6
        self.transspec_tmp = 295.0
        self.transspec_grav = 9.8
        # Chemistry workspace
        self.fastchem_work = ''


class _FakeAGNI:
    def __init__(self):
        self.last_setup_args = None
        self.last_setup_kwargs = None
        self.last_allocate_input_star = None
        self.atmosphere = SimpleNamespace(
            Atmos_t=lambda: _FakeAtmosphere(),
            setup_b=self._setup_b,
            allocate_b=self._allocate_b,
        )
        # setpt routines: record-only stubs
        self.setpt = SimpleNamespace(
            fromncdf_b=lambda *_a, **_k: None,
            loglinear_b=lambda *_a, **_k: None,
            isothermal_b=lambda *_a, **_k: None,
            dry_adiabat_b=lambda *_a, **_k: None,
            analytic_b=lambda *_a, **_k: None,
            stratosphere_b=lambda *_a, **_k: None,
        )

    def _setup_b(self, atmos, *args, **kwargs):
        self.last_setup_args = args
        self.last_setup_kwargs = kwargs
        return True

    def _allocate_b(self, atmos, input_star, **kwargs):
        self.last_allocate_input_star = input_star
        return True


def _build_greygas_config():
    """Build a config object that triggers the grey-gas dispatch."""
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(
            aerosols_enabled=False,
            cloud_enabled=False,
            rayleigh=False,
            surf_greyalbedo=0.3,
            surface_d=0.0,
            surface_k=0.0,
            tmp_minimum=50.0,
            num_levels=40,
            p_top=1e-5,
            overlap_method='ro',
            agni=SimpleNamespace(
                spectral_file='greygas',
                verbosity=2,
                oceans=False,
                rainout=False,
                chemistry='none',
                surf_material='greybody',
                surf_roughness=0.0,
                surf_windspeed=0.0,
                phs_timescale=1.0,
                evap_efficiency=1.0,
                thermo_functions=True,
                fastchem_floor=1e-30,
                fastchem_maxiter_chem=1,
                fastchem_maxiter_solv=1,
                fastchem_xtol_chem=1e-6,
                fastchem_xtol_elem=1e-6,
                real_gas=False,
                mlt_criterion='a',
                ini_profile='isothermal',
                grey_opacity_lw=0.1,
                grey_opacity_sw=0.2,
                check_safe_gas=False,
            ),
        ),
        orbit=SimpleNamespace(s0_factor=1.0, zenith_angle=48.0),
        params=SimpleNamespace(out=SimpleNamespace(logging='INFO')),
    )


@pytest.mark.unit
def test_init_agni_atmos_greygas_bypasses_spectral_copy(monkeypatch, tmp_path):
    """Greygas path should not call get_spfile_path or pass a stellar spectrum.

    When spectral_file='greygas' is set, AGNI uses the grey-gas RT scheme and
    does not need a SOCRATES spectral file or stellar flux to be copied into
    the runtime directory.
    """
    fake_agni = _FakeAGNI()
    fake_jl = SimpleNamespace(AGNI=fake_agni, Dict=dict, Char=str)

    output_dir = tmp_path / 'out'
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / '100.sflux').write_text('sflux', encoding='utf-8')

    dirs = {'output': str(output_dir), 'agni': '/fake/agni', 'fwl': '/fake/fwl'}
    config = _build_greygas_config()
    hf_row = {
        'F_ins': 1000.0,
        'albedo_pl': 0.2,
        'T_surf': 900.0,
        'gravity': 9.8,
        'R_int': 6.4e6,
        'P_surf': 1.0,
    }

    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'convert', lambda _typ, value: value)
    monkeypatch.setattr(agni_mod, '_construct_voldict', lambda *_a, **_k: {'H2O': 1.0})
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *_a, **_k: None)
    monkeypatch.setattr(
        agni_mod,
        'get_spfile_path',
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError('get_spfile_path should not be called for greygas')
        ),
    )

    atmos = init_agni_atmos(dirs, config, hf_row)

    assert atmos is not None

    # setup_b positional args: [dirs['agni'], dirs['output'], input_sf, ...]
    assert fake_agni.last_setup_args[2] == 'greygas'

    # Empty stellar path prevents AGNI from modifying/copying runtime spectral assets.
    assert fake_agni.last_allocate_input_star == ''

    # grey_opacity_lw/sw should be forwarded as the Greek-named AGNI kwargs.
    assert fake_agni.last_setup_kwargs['κ_grey_lw'] == pytest.approx(0.1)
    assert fake_agni.last_setup_kwargs['κ_grey_sw'] == pytest.approx(0.2)


@pytest.mark.unit
def test_init_agni_atmos_passes_unscaled_surface_pressure(monkeypatch, tmp_path):
    """init_agni_atmos hands AGNI the true surface pressure from hf_row, with
    no rescaling by the composition's mixing-ratio sum.

    Every modelled gas, including the noble gases, is in the composition, so
    AGNI's unit-sum renormalization already preserves each partial pressure;
    the total column pressure passed to setup_b must equal hf_row['P_surf'].

    Discrimination: _construct_voldict here returns a set whose VMR sum is 0.9.
    A regression reintroducing ``p_surf *= sum(vol_dict.values())`` would pass
    0.9 * P_surf, differing from the correct value by 10% of the column.
    """
    fake_agni = _FakeAGNI()
    fake_jl = SimpleNamespace(AGNI=fake_agni, Dict=dict, Char=str)

    output_dir = tmp_path / 'out'
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / '100.sflux').write_text('sflux', encoding='utf-8')

    dirs = {'output': str(output_dir), 'agni': '/fake/agni', 'fwl': '/fake/fwl'}
    config = _build_greygas_config()
    p_surf_true = 200.0  # bar, far above the p_top floor so no clamping occurs
    hf_row = {
        'F_ins': 1000.0,
        'albedo_pl': 0.2,
        'T_surf': 900.0,
        'gravity': 9.8,
        'R_int': 6.4e6,
        'P_surf': p_surf_true,
    }

    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'convert', lambda _typ, value: value)
    # VMR sum deliberately below 1 so a rescaling regression is detectable.
    monkeypatch.setattr(
        agni_mod, '_construct_voldict', lambda *_a, **_k: {'H2O': 0.6, 'CO2': 0.3}
    )
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *_a, **_k: None)

    atmos = init_agni_atmos(dirs, config, hf_row)
    assert atmos is not None

    # setup_b positional arg index 11 is p_surf (see the setup_b call in agni.py).
    passed_p_surf = fake_agni.last_setup_args[11]
    assert passed_p_surf == pytest.approx(p_surf_true)
    # The removed hack would have scaled by the 0.9 VMR sum.
    assert passed_p_surf != pytest.approx(0.9 * p_surf_true)


@pytest.mark.unit
def test_init_agni_atmos_greygas_does_not_glob_sflux(monkeypatch, tmp_path):
    """Regression: in grey-gas mode, init_agni_atmos must not require any
    *.sflux file to exist. Before this fix, an unconditional
    `glob.glob('*.sflux'); sorted(...)[-1]` at the top of the function
    would crash with IndexError on a fresh output dir before reaching the
    grey-gas dispatch.
    """
    fake_agni = _FakeAGNI()
    fake_jl = SimpleNamespace(AGNI=fake_agni, Dict=dict, Char=str)

    output_dir = tmp_path / 'out'
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True)
    # NOTE: no *.sflux file written. Pre-fix this would have crashed.

    dirs = {'output': str(output_dir), 'agni': '/fake/agni', 'fwl': '/fake/fwl'}
    config = _build_greygas_config()
    hf_row = {
        'F_ins': 1000.0,
        'albedo_pl': 0.2,
        'T_surf': 900.0,
        'gravity': 9.8,
        'R_int': 6.4e6,
        'P_surf': 1.0,
    }

    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'convert', lambda _typ, value: value)
    monkeypatch.setattr(agni_mod, '_construct_voldict', lambda *_a, **_k: {'H2O': 1.0})
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *_a, **_k: None)
    monkeypatch.setattr(
        agni_mod,
        'get_spfile_path',
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError('get_spfile_path should not be called for greygas')
        ),
    )

    # Must not raise.
    atmos = init_agni_atmos(dirs, config, hf_row)

    assert atmos is not None
    assert fake_agni.last_setup_args[2] == 'greygas'
    assert fake_agni.last_allocate_input_star == ''


@pytest.mark.unit
def test_init_agni_atmos_non_greygas_no_sflux_raises_filenotfound(monkeypatch, tmp_path):
    """When AGNI needs a fresh spectral file (no runtime.sf, no
    user-provided path), it must have a stellar spectrum to seed from.
    A missing *.sflux in that branch should raise FileNotFoundError
    instead of IndexError, so the caller sees a clear diagnostic."""
    fake_agni = _FakeAGNI()
    fake_jl = SimpleNamespace(AGNI=fake_agni, Dict=dict, Char=str)

    output_dir = tmp_path / 'out'
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True)
    # No *.sflux; no runtime.sf either.

    dirs = {'output': str(output_dir), 'agni': '/fake/agni', 'fwl': '/fake/fwl'}
    # Use the same scaffold as greygas test but flip spectral_file to None
    # so the function takes the "AGNI copy from FWL_DATA" branch.
    config = _build_greygas_config()
    config.atmos_clim.agni.spectral_file = None
    hf_row = {
        'F_ins': 1000.0,
        'albedo_pl': 0.2,
        'T_surf': 900.0,
        'gravity': 9.8,
        'R_int': 6.4e6,
        'P_surf': 1.0,
    }

    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'convert', lambda _typ, value: value)
    monkeypatch.setattr(agni_mod, '_construct_voldict', lambda *_a, **_k: {'H2O': 1.0})
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *_a, **_k: None)
    monkeypatch.setattr(agni_mod, 'UpdateStatusfile', lambda *_a, **_k: None)
    monkeypatch.setattr(agni_mod, 'get_spfile_path', lambda *_a, **_k: '/fake/spfile')

    with pytest.raises(FileNotFoundError, match='No stellar spectrum'):
        init_agni_atmos(dirs, config, hf_row)
    # Discrimination guard: a regression that hard-raised FileNotFoundError
    # on every path (regardless of spectral_file or *.sflux presence) would
    # also pass the pytest.raises block. Verify that flipping back to the
    # grey-gas branch (which does not need a stellar spectrum) does NOT
    # raise on the same hf_row and same empty data dir.
    config.atmos_clim.agni.spectral_file = 'greygas'
    atmos = init_agni_atmos(dirs, config, hf_row)
    assert atmos is not None
    # The grey-gas branch must have allocated with an empty stellar path
    # (no spectral file copied) to confirm the dispatch took the correct
    # branch rather than a no-op fallback.
    assert fake_agni.last_allocate_input_star == ''


# ---------------------------------------------------------------------------
# AgniSchemaMismatch: lightweight Atmos_t field-list check at allocate
# ---------------------------------------------------------------------------


def _build_complete_atmos_stub() -> SimpleNamespace:
    """Stand-in struct that carries every field PROTEUS expects from Atmos_t.

    Mirrors `_REQUIRED_ATMOS_FIELDS` in `proteus.atmos_clim.agni`. The
    helper exists so each schema test can compose its own missing-field
    subset without rewriting the full list.
    """
    return SimpleNamespace(
        tmp=[300.0],
        tmpl=[295.0, 305.0],
        pl=[1e4, 1e5],
        p_boa=1e5,
        p_oboa=1e5,
        tmp_surf=1500.0,
        tmp_magma=1500.0,
        g=[9.8],
        is_converged=True,
        transparent=False,
        flux_d_sw=[100.0],
        flux_u_lw=[300.0],
        flux_u_sw=[10.0],
        flux_tot=[100.0, 200.0],
        tau_band=[[0.0], [0.5]],
        diagnostic_Ra=[1.0, 2.0],
        timescale_conv=[1e3, 2e3],
        timescale_rad=[1e6, 1e6],
        gas_names=['H2O'],
        gas_vmr={'H2O': [1.0]},
        gas_ovmr={'H2O': [1.0]},
        ocean_areacov=0.0,
        ocean_maxdepth=0.0,
        ocean_tot=0.0,
        instellation=1361.0,
        transspec_p=1e2,
        transspec_r=6.371e6,
        transspec_tmp=295.0,
        fastchem_work='',
    )


def test_check_agni_schema_accepts_complete_struct():
    """A struct carrying every required field must pass without raising.

    Edge: this is the positive baseline. Any change to
    `_REQUIRED_ATMOS_FIELDS` that drops a name silently would still pass
    this test, but a *new* required field would trip it because the stub
    builder above lists fields explicitly.
    """
    atmos = _build_complete_atmos_stub()
    # Returns None (implicit) on success; raising would fail the test.
    assert agni_mod._check_agni_schema(atmos) is None
    # Discrimination guard: confirm the stub actually had every required
    # field, so a passing test cannot mean the checker is a no-op against
    # an under-populated input.
    for name in agni_mod._REQUIRED_ATMOS_FIELDS:
        assert hasattr(atmos, name), f'stub is missing {name}'


def test_check_agni_schema_rejects_missing_tau_band():
    """A struct missing `tau_band` (the AGNI 1.10.2 additive field) must raise.

    Discriminating: `tau_band` is the field a roll-back to AGNI 1.10.1
    would silently drop. The check must catch this at IC rather than
    surface as a `KeyError('tau_band')` mid-coupling-loop.
    """
    atmos = _build_complete_atmos_stub()
    del atmos.tau_band
    with pytest.raises(agni_mod.AgniSchemaMismatch) as excinfo:
        agni_mod._check_agni_schema(atmos)
    # The error must name the missing field (not a generic message), so
    # the next maintainer can fix the pin or the field list directly.
    assert 'tau_band' in str(excinfo.value)
    # Side-effect guard: the failing path must NOT silently mutate the
    # input struct. The other fields remain intact.
    assert hasattr(atmos, 'flux_tot')


def test_check_agni_schema_rejects_multiple_missing_fields():
    """When several fields are absent the error names every one of them.

    Edge: the next AGNI bump could rename a cluster of related fields
    in one move. The schema check must surface the entire missing set
    so the maintainer is not forced through one-at-a-time discovery.
    """
    atmos = _build_complete_atmos_stub()
    del atmos.tau_band
    del atmos.flux_tot
    del atmos.gas_vmr
    with pytest.raises(agni_mod.AgniSchemaMismatch) as excinfo:
        agni_mod._check_agni_schema(atmos)
    message = str(excinfo.value)
    for name in ('tau_band', 'flux_tot', 'gas_vmr'):
        assert name in message, f'{name} must appear in the schema-mismatch message'
    # Discriminating: a regression that reported only the first missing
    # field would fail this test because the other two would be absent
    # from the message.


# ---------------------------------------------------------------------------
# tau_band and diagnostic summarisers (AGNI 1.10.2 outputs into hf_row)
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_summarise_tau_band_returns_monotonic_TOA_below_surface():
    """At a realistic IR optical-depth profile, tau at TOA < tau at surface.

    Discriminating: a regression that swapped the TOA and surface
    indices, or read the array in the wrong axis order, would land on
    the inverted ordering. The TOA value of 0.05 vs the surface value
    of 4.5 is well outside any plausible aggregation noise.
    """
    # Layout matches AGNI's Julia storage: (nlev_l, nbands), with the
    # level axis on cell edges (nlev_l = nlev_c + 1).
    atmos = SimpleNamespace(
        nlev_l=3,
        nbands=2,
        tau_band=[[0.0, 0.1], [1.0, 1.2], [4.0, 5.0]],
    )
    toa, surf = agni_mod._summarise_tau_band(atmos)
    assert toa == pytest.approx(0.1, rel=1e-6)
    assert surf == pytest.approx(5.0, rel=1e-6)

    # A cell-centre-sized level axis (one row fewer than nlev_l) is
    # accepted too: TOA and surface sit at indices 0 and -1 on either
    # grid, so the reduction must not depend on the convention.
    atmos_c = SimpleNamespace(
        nlev_l=4,
        nbands=2,
        tau_band=[[0.0, 0.1], [1.0, 1.2], [4.0, 5.0]],
    )
    toa_c, surf_c = agni_mod._summarise_tau_band(atmos_c)
    assert toa_c == pytest.approx(0.1, rel=1e-6)
    assert surf_c == pytest.approx(5.0, rel=1e-6)

    # Monotonicity: optical depth integrated from TOA downwards must
    # grow with depth. Discrimination guard against wrong-direction
    # integration: a flipped sum would invert this inequality.
    assert toa < surf

    # Scale guard: the inversion-resistant form. Rejects 0.5 *
    # surface = 2.25 > TOA, which still passes the strict inequality
    # but would mean the gap is shrinking.
    assert toa < 0.5 * surf


def test_summarise_tau_band_returns_nan_on_unreadable_array():
    """If atmos.tau_band cannot be coerced into a numpy array, both
    aggregates are zero so the helpfile column remains well-formed.

    Edge: a transparent-mode solve never populates tau_band, leaving
    the field unset on the struct. The summariser must not raise.
    """
    atmos = SimpleNamespace()  # no tau_band attribute at all

    class _Raises:
        def __array__(self, *_a, **_k):
            raise RuntimeError('not readable from this Julia context')

    atmos.tau_band = _Raises()
    atmos.nlev_c = 3
    atmos.nbands = 2
    toa, surf = agni_mod._summarise_tau_band(atmos)

    assert toa == pytest.approx(0.0, abs=1e-10)
    assert surf == pytest.approx(0.0, abs=1e-10)


@pytest.mark.physics_invariant
def test_summarise_diagnostics_picks_extrema():
    """Get diagnostics at extrema of the profile

    We extract the profile-maximum values of both:
    - Ra, which identifies the most convective level and its Ra number.
    - The timescale ratio of t_conv/t_rad
    """
    atmos = SimpleNamespace(
        diagnostic_Ra=[0.0, 5.0, 4.0, 3.0],
        timescale_conv=[0.0, 1.0e3, 2.0e3, 1.0e2],
        timescale_rad=[1.0e6, 1.0e6, 1.0e5, 1.0e4],
    )
    ra_max, ratio = agni_mod._summarise_diagnostics(atmos)
    assert ra_max == pytest.approx(5.0, rel=1e-12)
    assert ratio == pytest.approx(0.02, rel=1e-6)

    # Sign + scale guards. A negative or zero ratio would mean the
    # convective timescale was read as zero, which is unphysical in this case.
    assert ratio > 0
    assert ratio < 1.0


def test_summarise_diagnostics_emits_zero_when_no_level_is_convective():
    """A purely radiative profile has timescale ratio set to zero."""
    atmos = SimpleNamespace(
        diagnostic_Ra=[0.0, 0.0, 0.0],
        timescale_conv=[0.0, 0.0, 0.0],
        timescale_rad=[1e6, 1e6, 1e6],
    )
    ra_max, ratio = agni_mod._summarise_diagnostics(atmos)
    assert ra_max == pytest.approx(0.0, abs=1e-10)
    assert ratio == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# init_agni_atmos: spectral file + surface material validation branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_agni_spectral_file_path_not_found_raises(monkeypatch, tmp_path):
    """A user-provided spectral_file path that does not exist on disk
    must raise FileNotFoundError with a diagnostic message, not
    silently fall through to AGNI's internal copy path.

    The adjacent-valid case (spectral_file='greygas') is covered by
    the greygas tests above; this pins the error-contract branch for
    a non-greygas path that points to a missing file.
    """
    fake_agni = _FakeAGNI()
    fake_jl = SimpleNamespace(AGNI=fake_agni, Dict=dict, Char=str)
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)

    config = _build_greygas_config()
    config.atmos_clim.agni.spectral_file = '/nonexistent/path/to/spec.sf'

    dirs = {'output': str(tmp_path), 'fwl': '/fake/fwl', 'agni': '/fake/agni'}
    hf_row = {'T_surf': 1500.0, 'P_surf': 100.0}

    with pytest.raises(FileNotFoundError, match='AGNI spectral file not found') as excinfo:
        init_agni_atmos(dirs, config, hf_row)

    # The error message must name the missing file path so the operator
    # can locate the gap. A regression that emitted a bare error without
    # interpolation would match the regex but lose the diagnostic.
    assert '/nonexistent/path/to/spec.sf' in str(excinfo.value)

    # NOTE: tests for the surf_material FileNotFoundError (L536-540) and
    # the ini_profile ValueError (L649-651) are deferred because they
    # require mocking deep into the Julia init path of init_agni_atmos
    # (past Atmos_t construction, spectral file dispatch, gas dict
    # population, and schema check). The existing greygas tests cover
    # the Julia mocking surface; extending them for these two branches
    # is a follow-up when the _FakeAGNI scaffold gains the missing
    # attributes (vol_mixing, gas_names, aerosol setup).


# ---------------------------------------------------------------------------
# _validate_agni_state: post-solve struct validation
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_validate_agni_state_accepts_valid_struct():
    """A physically valid struct passes validation.

    Discrimination: pin that (ok=True, reason='') and confirm the check
    actually inspected the fields (not a no-op returning True).
    """
    atmos = _build_complete_atmos_stub()
    atmos.tmp_surf = 1500.0
    atmos.flux_tot = [100.0, 200.0, 150.0]
    atmos.is_converged = True
    ok, reason = agni_mod._validate_agni_state(atmos)
    assert ok is True
    assert reason == ''
    # Scale guard: positive T_surf is accepted
    assert atmos.tmp_surf > 0


@pytest.mark.physics_invariant
def test_validate_agni_state_rejects_non_converged():
    """When is_converged is False despite solver returning True, the
    struct is contradictory and must be rejected.

    Discrimination: the specific reason string identifies the field,
    not a generic 'validation failed'.
    """
    atmos = _build_complete_atmos_stub()
    atmos.is_converged = False
    atmos.tmp_surf = 1500.0
    atmos.flux_tot = [100.0, 200.0]
    ok, reason = agni_mod._validate_agni_state(atmos)
    assert ok is False
    assert 'is_converged' in reason
    # Adjacent-valid: same struct with is_converged=True passes
    atmos.is_converged = True
    ok2, reason2 = agni_mod._validate_agni_state(atmos)
    assert ok2 is True
    assert reason2 == ''


def test_validate_agni_state_rejects_nan_tmp_surf():
    """A NaN tmp_surf must fail validation (non-finite or <= 0 guard).

    Edge: observed in CHILI sweep R12/R17 where line-search collapse
    left NaN on the struct after is_converged=True.
    """
    atmos = _build_complete_atmos_stub()
    atmos.is_converged = True
    atmos.tmp_surf = float('nan')
    atmos.flux_tot = [100.0]
    ok, reason = agni_mod._validate_agni_state(atmos)
    assert ok is False
    assert 'tmp_surf' in reason


def test_validate_agni_state_rejects_zero_tmp_surf():
    """T_surf = 0 K is unphysical and must be rejected.

    Discrimination: the guard is '<= 0', not '< 0', so 0.0 must fail.
    A regression that used '< 0' would pass 0.0 through.
    """
    atmos = _build_complete_atmos_stub()
    atmos.is_converged = True
    atmos.tmp_surf = 0.0
    atmos.flux_tot = [100.0]
    ok, reason = agni_mod._validate_agni_state(atmos)
    assert ok is False
    assert 'tmp_surf' in reason
    # Adjacent-valid: T = 1 K is positive and passes
    atmos.tmp_surf = 1.0
    ok2, _ = agni_mod._validate_agni_state(atmos)
    assert ok2 is True


def test_validate_agni_state_rejects_nonfinite_flux_tot():
    """Non-finite elements in flux_tot must fail validation.

    Edge: a single NaN in the flux array poisons the downstream
    F_atm = tot_flux[0] assignment in run_agni.
    """
    atmos = _build_complete_atmos_stub()
    atmos.is_converged = True
    atmos.tmp_surf = 1500.0
    atmos.flux_tot = [100.0, float('inf'), 200.0]
    ok, reason = agni_mod._validate_agni_state(atmos)
    assert ok is False
    assert 'non-finite' in reason
    # Pin the count: exactly 1 bad element out of 3
    assert '1' in reason


def test_validate_agni_state_rejects_empty_flux_tot():
    """Empty flux_tot must be rejected (size == 0 guard).

    Discrimination: a regression that skipped the size check and went
    straight to np.all(np.isfinite([])) would get True (vacuous truth).
    """
    atmos = _build_complete_atmos_stub()
    atmos.is_converged = True
    atmos.tmp_surf = 1500.0
    atmos.flux_tot = []
    ok, reason = agni_mod._validate_agni_state(atmos)
    assert ok is False
    assert 'empty' in reason


def test_validate_agni_state_handles_missing_tmp_surf_attr():
    """When tmp_surf attribute is absent, the except clause returns
    False with a 'could not be read' reason.
    """
    from types import SimpleNamespace

    atmos = SimpleNamespace(is_converged=True, flux_tot=[100.0])
    ok, reason = agni_mod._validate_agni_state(atmos)
    assert ok is False
    assert 'could not be read' in reason


# ---------------------------------------------------------------------------
# _summarise_tau_band: transposed (nbands, nlev_c) layout
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_summarise_tau_band_unexpected_shape_returns_zero():
    """A shape that matches neither (nlev_c, nbands) nor (nbands, nlev_c)
    must return zero for both values and log a warning.
    """
    atmos = SimpleNamespace(
        nlev_c=3,
        nbands=2,
        tau_band=[[[0.0]], [[0.1]]],
    )
    toa, surf = agni_mod._summarise_tau_band(atmos)
    assert toa == pytest.approx(0.0, abs=1e-10)
    assert surf == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# run_agni: transparent + opaque + prevent_warming + ocean output
# ---------------------------------------------------------------------------


def _make_run_agni_atmos(*, transparent=False):
    """Build an atmos stub for run_agni exercising the parse-results block."""
    atmos = _FakeAtmosphere()
    atmos.transparent = transparent
    atmos.tmp_surf = 1500.0
    atmos.tmp_magma = 1500.0
    atmos.p_boa = 1.0e5
    atmos.transspec_p = 1.0e4
    atmos.transspec_r = 6.4e6
    atmos.transspec_tmp = 280.0
    atmos.flux_tot = [150.0, 200.0, 100.0]
    atmos.flux_u_lw = [120.0]
    atmos.flux_u_sw = [20.0]
    atmos.flux_d_sw = [100.0]
    atmos.ocean_areacov = 0.5
    atmos.ocean_maxdepth = 3000.0
    atmos.ocean_tot = {g: (1e20 if g == 'H2O' else 0.0) for g in agni_mod.gas_list}
    atmos.gas_names = ['H2O', 'CO2']
    atmos.gas_vmr = {'H2O': [0.9, 0.8], 'CO2': [0.1, 0.2]}
    atmos.gas_ovmr = {'H2O': [0.9], 'CO2': [0.1]}
    atmos.p = [1e3, 1e5]
    atmos.r = [6.5e6, 6.4e6]
    # Cell-centre profiles read at the XUV level; two entries to match p
    atmos.tmp = [280.0, 900.0]
    atmos.g = [9.5, 9.8]
    atmos.is_converged = True
    atmos.tau_band = [[0.01, 0.02], [0.5, 0.6]]
    atmos.diagnostic_Ra = [0.1, 5.0]
    atmos.timescale_conv = [1e3, 2e3]
    atmos.timescale_rad = [1e6, 1e5]
    atmos.nlev_c = 2
    atmos.nbands = 2
    return atmos


def _make_run_agni_config(
    *,
    solve_energy=True,
    prevent_warming=False,
    oceans=False,
    xuv_defined_by_radius=False,
    hill_clamp=False,
):
    """Build the config namespace run_agni reads."""
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(
            p_obs=1e-3,
            p_top=1e-5,
            agni=SimpleNamespace(
                solve_energy=solve_energy,
                oceans=oceans,
                rainout=False,
                chemistry='none',
                psurf_thresh=1e-4,
                solution_atol=1e-3,
                solution_rtol=1e-3,
                ls_default=1,
                dx_max=100.0,
                perturb_all=False,
                verbosity=0,
            ),
            surf_state_int=1,
        ),
        planet=SimpleNamespace(prevent_warming=prevent_warming),
        escape=SimpleNamespace(
            xuv_defined_by_radius=xuv_defined_by_radius,
            hill_clamp=hill_clamp,
            hill_clamp_frac=1.0,
        ),
        params=SimpleNamespace(
            out=SimpleNamespace(
                logging='WARNING',
                plot_mod=None,
                plot_fmt='png',
            )
        ),
    )


@pytest.mark.physics_invariant
def test_run_agni_transparent_returns_R_obs_equal_R_int(monkeypatch):
    """In transparent mode, R_obs = R_int (no atmosphere adds radius).

    Discrimination: in the opaque branch, R_obs comes from
    atmos.transspec_r, which is typically larger than R_int.
    """
    atmos = _make_run_agni_atmos(transparent=True)
    config = _make_run_agni_config(solve_energy=True)
    hf_row = {
        'P_surf': 1e-5,
        'R_int': 6.371e6,
        'R_xuv': 6.5e6,
        'p_xuv': 1e-3,
        'gravity': 9.8,
        'Time': 0.0,
    }
    for g in ['H2O', 'CO2']:
        hf_row[g + '_vmr'] = 0.5

    dirs = {'output': '/tmp/fake_output', 'output/plots': '/tmp/fake_plots'}

    fake_jl = SimpleNamespace(
        AGNI=SimpleNamespace(
            atmosphere=SimpleNamespace(estimate_photosphere_b=lambda *a, **kw: None),
            save=SimpleNamespace(write_ncdf=lambda a, p: None),
            plotting=SimpleNamespace(plot_contfunc1=lambda a, p: None),
            solver=SimpleNamespace(solve_transparent_b=lambda *a, **kw: None),
        ),
    )
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *a: [])
    monkeypatch.setattr(agni_mod, 'get_oarr_from_parr', lambda p_arr, r_arr, val: (0, val))

    _, output = agni_mod.run_agni(atmos, 1, dirs, config, hf_row)

    assert output['R_obs'] == pytest.approx(6.371e6, rel=1e-12)
    assert output['T_obs'] == pytest.approx(atmos.tmp_surf, rel=1e-12)
    # Opaque branch would give transspec_r = 6.4e6; pin the difference
    assert abs(output['R_obs'] - 6.4e6) > 1e3
    assert output['agni_converged'] is True


@pytest.mark.physics_invariant
def test_run_agni_prevent_warming_clamps_negative_flux(monkeypatch):
    """When prevent_warming is True and F_atm < 0, the output F_atm
    is clamped to 1e-8 W/m^2 (no planet warming from negative net flux).

    Discrimination: without the clamp, F_atm would be -50.0. The
    difference (1e-8 vs -50) is unambiguous.
    """
    atmos = _make_run_agni_atmos(transparent=False)
    atmos.flux_tot = [-50.0, 200.0]
    config = _make_run_agni_config(solve_energy=False, prevent_warming=True)
    hf_row = {
        'P_surf': 100.0,
        'p_xuv': 1e-3,
        'R_xuv': 6.5e6,
        'gravity': 9.8,
        'Time': 100.0,
    }
    for g in ['H2O', 'CO2']:
        hf_row[g + '_vmr'] = 0.5

    dirs = {'output': '/tmp/fake', 'output/plots': '/tmp/fake_plots'}
    fake_jl = SimpleNamespace(
        AGNI=SimpleNamespace(
            atmosphere=SimpleNamespace(estimate_photosphere_b=lambda *a, **kw: None),
            save=SimpleNamespace(write_ncdf=lambda a, p: None),
            plotting=SimpleNamespace(plot_contfunc1=lambda a, p: None),
            chemistry=SimpleNamespace(calc_composition_b=lambda *a: False),
            setpt=SimpleNamespace(
                dry_adiabat_b=lambda a: None,
                saturation_b=lambda a, g: None,
                stratosphere_b=lambda a, v: None,
            ),
            energy=SimpleNamespace(
                calc_fluxes_b=lambda a, **kw: None,
                fill_Kzz_b=lambda a: None,
            ),
        ),
    )
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *a: [])
    monkeypatch.setattr(agni_mod, 'get_oarr_from_parr', lambda p_arr, r_arr, val: (0, val))

    _, output = agni_mod.run_agni(atmos, 1, dirs, config, hf_row)

    assert output['F_atm'] == pytest.approx(1e-8, rel=1e-6)
    # Without prevent_warming, F_atm would be -50.0
    assert output['F_atm'] > 0


def test_run_agni_ocean_output_keys_populated(monkeypatch):
    """When oceans=True in config, the output dict must contain
    ocean_areacov, ocean_maxdepth, and per-gas ocean totals.

    Edge: tests the gas-in-gas_names vs gas-not-in-gas_names branching.
    """
    atmos = _make_run_agni_atmos(transparent=False)
    config = _make_run_agni_config(solve_energy=False, oceans=True)
    hf_row = {
        'P_surf': 100.0,
        'p_xuv': 1e-3,
        'R_xuv': 6.5e6,
        'Time': 100.0,
    }
    for g in ['H2O', 'CO2']:
        hf_row[g + '_vmr'] = 0.5

    dirs = {'output': '/tmp/fake', 'output/plots': '/tmp/fake_plots'}
    fake_jl = SimpleNamespace(
        AGNI=SimpleNamespace(
            atmosphere=SimpleNamespace(estimate_photosphere_b=lambda *a, **kw: None),
            save=SimpleNamespace(write_ncdf=lambda a, p: None),
            plotting=SimpleNamespace(plot_contfunc1=lambda a, p: None),
            chemistry=SimpleNamespace(calc_composition_b=lambda *a: False),
            setpt=SimpleNamespace(
                dry_adiabat_b=lambda a: None,
                saturation_b=lambda a, g: None,
                stratosphere_b=lambda a, v: None,
            ),
            energy=SimpleNamespace(
                calc_fluxes_b=lambda a, **kw: None,
                fill_Kzz_b=lambda a: None,
            ),
        ),
    )
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *a: [])
    monkeypatch.setattr(agni_mod, 'get_oarr_from_parr', lambda p_arr, r_arr, val: (0, val))

    _, output = agni_mod.run_agni(atmos, 1, dirs, config, hf_row)

    assert output['ocean_areacov'] == pytest.approx(0.5, rel=1e-12)
    assert output['ocean_maxdepth'] == pytest.approx(3000.0, rel=1e-12)
    # H2O is in gas_names -> reads from atmos.ocean_tot dict
    assert output['H2O_ocean'] == pytest.approx(1e20, rel=1e-6)
    # Gases NOT in gas_names get 0.0
    for g in agni_mod.gas_list:
        if g not in ['H2O', 'CO2']:
            assert output[g + '_ocean'] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.physics_invariant
def test_run_agni_opaque_gobs_inverse_square_of_robs(monkeypatch):
    """Opaque AGNI reports g_obs and R_obs that obey the inverse-square law.

    AGNI computes gravity on its own height grid inside the Julia solver and
    surfaces it as transspec_grav at the photospheric radius transspec_r.
    However, it doesn't use a simple geometric scaling of the surface gravity.
    """
    g_surf = 9.8
    R_int = 6.371e6
    R_obs = 6.45e6  # photosphere above the surface
    g_obs_expected = g_surf * (R_int / R_obs) ** 2  # geometric scaling

    atmos = _make_run_agni_atmos(transparent=False)
    atmos.transspec_r = R_obs
    atmos.transspec_grav = g_obs_expected

    config = _make_run_agni_config(solve_energy=False)
    hf_row = {
        'P_surf': 100.0,
        'p_xuv': 1e-3,
        'R_xuv': 6.5e6,
        'R_int': R_int,
        'gravity': g_surf,
        'Time': 100.0,
    }
    for g in ['H2O', 'CO2']:
        hf_row[g + '_vmr'] = 0.5

    dirs = {'output': '/tmp/fake', 'output/plots': '/tmp/fake_plots'}
    fake_jl = SimpleNamespace(
        AGNI=SimpleNamespace(
            atmosphere=SimpleNamespace(estimate_photosphere_b=lambda *a, **kw: None),
            save=SimpleNamespace(write_ncdf=lambda a, p: None),
            plotting=SimpleNamespace(plot_contfunc1=lambda a, p: None),
            chemistry=SimpleNamespace(calc_composition_b=lambda *a: False),
            setpt=SimpleNamespace(
                dry_adiabat_b=lambda a: None,
                saturation_b=lambda a, g: None,
                stratosphere_b=lambda a, v: None,
            ),
            energy=SimpleNamespace(
                calc_fluxes_b=lambda a, **kw: None,
                fill_Kzz_b=lambda a: None,
            ),
        ),
    )
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *a: [])
    monkeypatch.setattr(agni_mod, 'get_oarr_from_parr', lambda p_arr, r_arr, val: (0, val))

    _, output = agni_mod.run_agni(atmos, 1, dirs, config, hf_row)

    # R_obs is the photosphere,
    assert output['R_obs'] == pytest.approx(R_obs, rel=1e-12)
    assert output['R_obs'] > R_int

    # Gravity is scaled by the inverse-square law from the surface value here
    assert output['g_obs'] == pytest.approx(g_surf * (R_int / output['R_obs']) ** 2, rel=1e-10)
    assert output['g_obs'] < g_surf

    # Discrimination guard: rules out a pass-through of the surface gravity
    assert output['g_obs'] != pytest.approx(g_surf, rel=1e-3)


@pytest.mark.physics_invariant
def test_run_agni_xuv_level_temperature_and_gravity_from_profiles(monkeypatch):
    """T_xuv and g_xuv are read off the cell-centre profiles at the XUV level.

    With p_xuv = 1e-3 bar = 100 Pa against a profile p = [1e3, 1e5] Pa, the
    nearest level is the upper cell, so T_xuv = 280 K and g_xuv = 9.5 m/s2.
    The lower cell holds 900 K and 9.8 m/s2, so reading the wrong end of the
    column misses by a factor of 3.2 in temperature, far outside tolerance.
    """
    atmos = _make_run_agni_atmos(transparent=False)
    config = _make_run_agni_config(solve_energy=False)
    hf_row = {
        'P_surf': 100.0,
        'p_xuv': 1e-3,
        'R_xuv': 6.5e6,
        'R_int': 6.371e6,
        'gravity': 9.8,
        'Time': 100.0,
    }
    for g in ['H2O', 'CO2']:
        hf_row[g + '_vmr'] = 0.5

    dirs = {'output': '/tmp/fake', 'output/plots': '/tmp/fake_plots'}
    fake_jl = SimpleNamespace(
        AGNI=SimpleNamespace(
            atmosphere=SimpleNamespace(estimate_photosphere_b=lambda *a, **kw: None),
            save=SimpleNamespace(write_ncdf=lambda a, p: None),
            plotting=SimpleNamespace(plot_contfunc1=lambda a, p: None),
            chemistry=SimpleNamespace(calc_composition_b=lambda *a: False),
            setpt=SimpleNamespace(
                dry_adiabat_b=lambda a: None,
                saturation_b=lambda a, g: None,
                stratosphere_b=lambda a, v: None,
            ),
            energy=SimpleNamespace(
                calc_fluxes_b=lambda a, **kw: None,
                fill_Kzz_b=lambda a: None,
            ),
        ),
    )
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *a: [])
    # The real interpolator, so the pin discriminates which cell was read

    _, output = agni_mod.run_agni(atmos, 1, dirs, config, hf_row)

    assert output['T_xuv'] == pytest.approx(280.0, rel=1e-12)
    assert output['g_xuv'] == pytest.approx(9.5, rel=1e-12)
    assert output['R_xuv'] == pytest.approx(6.5e6, rel=1e-12)
    # Discrimination: the other end of the column is far outside tolerance.
    assert output['T_xuv'] != pytest.approx(900.0, rel=1e-2)
    assert output['g_xuv'] != pytest.approx(9.8, rel=1e-3)


def _make_run_agni_jl():
    """Julia stand-in for the run_agni post-solve path."""
    return SimpleNamespace(
        AGNI=SimpleNamespace(
            atmosphere=SimpleNamespace(estimate_photosphere_b=lambda *a, **kw: None),
            save=SimpleNamespace(write_ncdf=lambda a, p: None),
            plotting=SimpleNamespace(plot_contfunc1=lambda a, p: None),
            chemistry=SimpleNamespace(calc_composition_b=lambda *a: False),
            setpt=SimpleNamespace(
                dry_adiabat_b=lambda a: None,
                saturation_b=lambda a, g: None,
                stratosphere_b=lambda a, v: None,
            ),
            energy=SimpleNamespace(
                calc_fluxes_b=lambda a, **kw: None,
                fill_Kzz_b=lambda a: None,
            ),
        ),
    )


def _clip_test_hf_row():
    """Row for the clip tests: Hill radius between the two profile cells."""
    hf_row = {
        'P_surf': 100.0,
        'p_xuv': 1e-3,
        'R_xuv': 6.5e6,
        'R_int': 6.371e6,
        'gravity': 9.8,
        'Time': 100.0,
        'hill_radius': 6.42e6,
    }
    for g in ['H2O', 'CO2']:
        hf_row[g + '_vmr'] = 0.5
    return hf_row


@pytest.mark.physics_invariant
def test_run_agni_clips_the_xuv_level_to_the_hill_radius(monkeypatch):
    """With the clip enabled and a Hill radius inside the unclipped XUV level,
    the level moves to the Hill radius and p, T, g are re-read there, so the
    whole group describes the clipped level rather than mixing two levels.

    Unclipped, p_xuv = 1e-3 bar lands on the upper cell (r = 6.5e6 m, 280 K,
    9.5 m/s2). The Hill radius at 6.42e6 m forces the level down; the nearest
    profile point is then the lower cell, so every quantity flips to its other
    end: p 1e5 Pa, T 900 K, g 9.8 m/s2. A clip that moved only the radius
    would keep 280 K and fail by a factor of 3.2.
    """
    atmos = _make_run_agni_atmos(transparent=False)
    config = _make_run_agni_config(solve_energy=False, hill_clamp=True)
    hf_row = _clip_test_hf_row()

    dirs = {'output': '/tmp/fake', 'output/plots': '/tmp/fake_plots'}
    monkeypatch.setattr(agni_mod, 'jl', _make_run_agni_jl())
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *a: [])

    _, output = agni_mod.run_agni(atmos, 1, dirs, config, hf_row)

    # The level sits at the Hill radius, not at the unclipped 6.5e6 m.
    assert output['R_xuv'] == pytest.approx(6.42e6, rel=1e-12)
    # Pressure, temperature and gravity are re-read at the clipped level.
    assert output['p_xuv'] == pytest.approx(1.0, rel=1e-12)  # 1e5 Pa in bar
    assert output['T_xuv'] == pytest.approx(900.0, rel=1e-12)
    assert output['g_xuv'] == pytest.approx(9.8, rel=1e-12)
    # Discrimination: the unclipped level's values are far outside tolerance.
    assert output['T_xuv'] != pytest.approx(280.0, rel=1e-2)
    assert output['p_xuv'] != pytest.approx(1e-3, rel=1e-2)


def test_run_agni_clip_disabled_leaves_the_level_alone(monkeypatch):
    """With the clip disabled the same setup keeps the unclipped level, so the
    clip cannot silently engage for configs that turned it off."""
    atmos = _make_run_agni_atmos(transparent=False)
    config = _make_run_agni_config(solve_energy=False, hill_clamp=False)
    hf_row = _clip_test_hf_row()

    dirs = {'output': '/tmp/fake', 'output/plots': '/tmp/fake_plots'}
    monkeypatch.setattr(agni_mod, 'jl', _make_run_agni_jl())
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *a: [])

    _, output = agni_mod.run_agni(atmos, 1, dirs, config, hf_row)

    assert output['R_xuv'] == pytest.approx(6.5e6, rel=1e-12)
    assert output['T_xuv'] == pytest.approx(280.0, rel=1e-12)
    assert output['g_xuv'] == pytest.approx(9.5, rel=1e-12)


# ---------------------------------------------------------------------------
# _solve_transparent: transparent solver dispatch
# ---------------------------------------------------------------------------


def test_solve_transparent_passes_config_tolerances(monkeypatch):
    """_solve_transparent must pass solution_atol and solution_rtol
    from config to AGNI's solve_transparent_b.

    Discrimination: pin the kwargs of the Julia call. A regression
    that hardcoded atol/rtol would not match the config values.
    """
    captured = {}

    def fake_solve_transparent_b(atmos, **kwargs):
        captured.update(kwargs)

    fake_jl = SimpleNamespace(
        AGNI=SimpleNamespace(
            solver=SimpleNamespace(solve_transparent_b=fake_solve_transparent_b),
        ),
    )
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)

    config = _make_run_agni_config()
    config.atmos_clim.agni.solution_atol = 1e-4
    config.atmos_clim.agni.solution_rtol = 1e-5
    config.atmos_clim.surf_state_int = 2

    atmos = _make_run_agni_atmos(transparent=True)
    result = agni_mod._solve_transparent(atmos, config)

    assert captured['conv_atol'] == pytest.approx(1e-4, rel=1e-12)
    assert captured['conv_rtol'] == pytest.approx(1e-5, rel=1e-12)
    assert captured['sol_type'] == 2
    assert captured['max_steps'] == 120
    assert result is atmos


# ---------------------------------------------------------------------------
# _solve_once: prescribed T(p) solver
# ---------------------------------------------------------------------------


def test_solve_once_calls_composition_and_fluxes(monkeypatch):
    """_solve_once must call calc_composition_b, dry_adiabat_b,
    calc_fluxes_b, and fill_Kzz_b in sequence.

    Discrimination: track call order. A regression that skipped
    the flux calculation would show as a missing call.
    """
    call_order = []

    def _track(name):
        def _fn(*args, **kwargs):
            call_order.append(name)
            return False

        return _fn

    fake_jl = SimpleNamespace(
        AGNI=SimpleNamespace(
            chemistry=SimpleNamespace(calc_composition_b=_track('composition')),
            setpt=SimpleNamespace(
                dry_adiabat_b=_track('dry_adiabat'),
                saturation_b=_track('saturation'),
                stratosphere_b=_track('stratosphere'),
            ),
            energy=SimpleNamespace(
                calc_fluxes_b=_track('calc_fluxes'),
                fill_Kzz_b=_track('fill_Kzz'),
            ),
        ),
    )
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)

    config = _make_run_agni_config(solve_energy=False)
    config.atmos_clim.agni.oceans = False
    config.atmos_clim.agni.rainout = False

    atmos = _make_run_agni_atmos()
    result = agni_mod._solve_once(atmos, config)

    assert result is atmos
    assert 'composition' in call_order
    assert 'dry_adiabat' in call_order
    assert 'calc_fluxes' in call_order
    assert 'fill_Kzz' in call_order
    # Rainout disabled => no saturation call
    assert 'saturation' not in call_order


def test_solve_once_rainout_skips_noble_gases(monkeypatch):
    """With rainout enabled, _solve_once calls AGNI's saturation routine for
    the reactive gases but never for a noble gas.

    Noble gases are chemically inert and tracked as conserved reservoirs, so
    condensing them would break mass closure. A regression that dropped the
    noble skip in the rainout loop would pass He/Ne/Ar/Kr/Xe to saturation_b.
    """
    saturated = []

    fake_jl = SimpleNamespace(
        AGNI=SimpleNamespace(
            chemistry=SimpleNamespace(calc_composition_b=lambda *_a, **_k: False),
            setpt=SimpleNamespace(
                dry_adiabat_b=lambda *_a, **_k: None,
                saturation_b=lambda _atmos, gas: saturated.append(gas),
                stratosphere_b=lambda *_a, **_k: None,
            ),
            energy=SimpleNamespace(
                calc_fluxes_b=lambda *_a, **_k: None,
                fill_Kzz_b=lambda *_a, **_k: None,
            ),
        ),
    )
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)

    config = _make_run_agni_config(solve_energy=False)
    config.atmos_clim.agni.oceans = False
    config.atmos_clim.agni.rainout = True

    atmos = _make_run_agni_atmos()
    agni_mod._solve_once(atmos, config)

    # No noble gas reaches the saturation routine.
    for noble in noble_gases:
        assert noble not in saturated, f'{noble} must not be rained out'
    # Edge guard: the loop still ran and the skip is selective, not a blanket
    # skip of every gas. At least one reactive gas was saturated.
    assert len(saturated) > 0
    assert all(g not in noble_gases for g in saturated)


# ---------------------------------------------------------------------------
# _construct_voldict: VMR assembly from hf_row
# ---------------------------------------------------------------------------


def test_construct_voldict_raises_on_zero_vmr(monkeypatch):
    """When all volatile VMRs sum to < 1e-4, _construct_voldict must raise
    ValueError and call UpdateStatusfile with code 20.

    Discrimination: the error string must mention 'zero'. A regression
    that raised on threshold=1e-2 instead of 1e-4 would pass a
    sum=1e-3 input; test with sum=0 for the clear-cut case.
    """
    hf_row = {}
    for g in agni_mod.gas_list:
        hf_row[g + '_vmr'] = 0.0

    update_calls = []
    monkeypatch.setattr(
        agni_mod, 'UpdateStatusfile', lambda dirs, code: update_calls.append(code)
    )

    cfg = SimpleNamespace(atmos_clim=SimpleNamespace(agni=SimpleNamespace(chemistry='none')))
    with pytest.raises(ValueError, match='zero'):
        agni_mod._construct_voldict(cfg, hf_row, {'output': '/tmp'})

    assert update_calls == [20]


def test_construct_voldict_includes_noble_gases():
    """_construct_voldict returns {gas: vmr} for every modelled gas, including
    the noble gases, which enter the composition handed to the AGNI solve
    alongside the reactive gases.

    Discrimination: the returned dict has exactly the full gas_list key set and
    each noble gas keeps its input VMR. The noble-excluding formula would drop
    the sum by 0.05, far above the tolerance.
    """
    hf_row = {}
    for g in agni_mod.gas_list:
        hf_row[g + '_vmr'] = 0.01

    cfg = SimpleNamespace(atmos_clim=SimpleNamespace(agni=SimpleNamespace(chemistry='none')))
    vol_dict = agni_mod._construct_voldict(cfg, hf_row, {'output': '/tmp'})
    assert set(vol_dict.keys()) == set(agni_mod.gas_list)
    for gas in noble_gases:
        assert vol_dict[gas] == pytest.approx(0.01, rel=1e-12)
    assert sum(vol_dict.values()) == pytest.approx(0.01 * len(agni_mod.gas_list), rel=1e-12)
    # A noble-excluding regression would land at 0.01 * (len - 5); the 0.05
    # gap discriminates it from the correct full-set sum.
    excl_sum = 0.01 * (len(agni_mod.gas_list) - len(noble_gases))
    assert abs(sum(vol_dict.values()) - excl_sum) > 0.04


def test_construct_voldict_handles_noble_only_atmosphere():
    """When every reactive gas has zero VMR but the noble gases are present,
    the total mixing-ratio sum is non-zero, so _construct_voldict returns the
    noble composition and the empty-atmosphere guard does not trip.

    Edge case: every reactive gas is at zero VMR. With the five noble gases at
    0.2 the mixing-ratio sum is 1.0, well above the 1e-4 floor, so the function
    returns the noble gases. The noble-excluding formula would drop the sum to
    zero and raise ValueError.
    """
    hf_row = {g + '_vmr': 0.0 for g in agni_mod.gas_list}
    for gas in noble_gases:
        hf_row[gas + '_vmr'] = 0.2  # noble-only atmosphere

    cfg = SimpleNamespace(atmos_clim=SimpleNamespace(agni=SimpleNamespace(chemistry='none')))
    vol_dict = agni_mod._construct_voldict(cfg, hf_row, {'output': '/tmp'})
    for gas in noble_gases:
        assert vol_dict[gas] == pytest.approx(0.2, rel=1e-12)
    assert sum(vol_dict.values()) == pytest.approx(0.2 * len(noble_gases), rel=1e-12)


# ---------------------------------------------------------------------------
# _determine_Hfraction: hydrogen floor for equilibrium chemistry
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_determine_Hfraction_leaves_hydrogen_rich_atmosphere_unchanged():
    """When the atmosphere already contains hydrogen, _determine_Hfraction does
    not inject the H2 floor and leaves the mixing ratios untouched.

    Physical scenario: an H2O/CO2 atmosphere. H2O carries 2 H atoms per molecule,
    so the H abundance (1.0) is far above the 1e-10 floor and no H2 is added.
    """
    vol_dict = {'H2O': 0.5, 'CO2': 0.5}
    result = agni_mod._determine_Hfraction({}, vol_dict)

    # No hydrogen floor injected; the reactive VMRs are preserved.
    assert 'H2' not in result
    assert result['H2O'] == pytest.approx(0.5, rel=1e-12)
    # All mixing ratios remain valid.
    for vmr in result.values():
        assert 0.0 <= vmr <= 1.0


@pytest.mark.physics_invariant
def test_determine_Hfraction_injects_floor_when_hydrogen_absent():
    """When no molecule contributes hydrogen atoms, _determine_Hfraction injects
    a small H2 floor so FastChem can define metallicity ratios.

    Edge case: a CO2/N2 atmosphere with a noble gas (He) whose name contains 'H'
    but contributes zero H atoms. The substring 'H' in 'He' must not be counted
    as hydrogen, so the floor still fires. Pins the injected VMR to 1e-10 (not
    the earlier 1e-10/2), and confirms it is positive and bounded.
    """
    vol_dict = {'CO2': 0.6, 'N2': 0.1, 'He': 0.3}
    result = agni_mod._determine_Hfraction({}, vol_dict)

    assert result['H2'] == pytest.approx(1e-10, rel=1e-9)
    # Discrimination: the old formula injected half this (5e-11); the values
    # differ by 5e-11, well outside the tolerance.
    assert abs(result['H2'] - 5e-11) > 1e-11
    assert result['H2'] > 0.0
    assert 0.0 <= result['H2'] <= 1.0


# ---------------------------------------------------------------------------
# sync_log_files: logfile content migration
# ---------------------------------------------------------------------------


def test_sync_log_files_returns_empty_on_missing_logfile(tmp_path):
    """When the AGNI logfile does not exist, sync_log_files returns []
    without crashing.

    Edge: first iteration before AGNI writes anything.
    """
    result = agni_mod.sync_log_files(str(tmp_path))
    assert result == []
    # Type guard: a regression returning None would also satisfy
    # `== []` as False, but not isinstance.
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _determine_condensates: single-gas warning path (line 397-399)
# ---------------------------------------------------------------------------


def test_determine_condensates_single_gas_returns_empty():
    """With only one gas, condensation is impossible (no dry gas backup).

    Discrimination: the single-gas guard returns [] directly, not the
    filtered list. A regression that removed the guard would return
    ['H2O'] for a condensable single gas.
    """
    result = agni_mod._determine_condensates(['H2O'])
    assert result == []
    # Adjacent-valid: two gases (one dry, one condensable) works
    result2 = agni_mod._determine_condensates(['H2O', 'N2'])
    assert result2 == ['H2O']


# ---------------------------------------------------------------------------
# write_atmos_ncdf: AGNI NetCDF snapshot writer
# ---------------------------------------------------------------------------


def test_write_atmos_ncdf_uses_rounded_time_and_data_dir(monkeypatch):
    """The AGNI writer serialises the struct to ``<output>/data/<%.0f>_atm.nc``
    via ``jl.AGNI.save.write_ncdf``.

    Discrimination: time=1000.6 rounds to 1001 under AGNI's ``%.0f``
    convention (not 1000 as an ``int()`` truncation would give), so the pinned
    path distinguishes the rounding convention. The struct is passed through
    unchanged as the first argument.
    """
    fake_write = MagicMock()
    fake_jl = SimpleNamespace(AGNI=SimpleNamespace(save=SimpleNamespace(write_ncdf=fake_write)))
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)

    # Write atmosphere to appropriately named file
    atmos_sentinel = SimpleNamespace(is_alloc=True)
    write_atmos_ncdf(atmos_sentinel, {'output': '/tmp/run'}, 1000.6)

    # Check function was called and file path is correct.
    # This test doesn't actually write the file, since we mock the function.
    fake_write.assert_called_once()
    called_atmos, called_path = fake_write.call_args.args
    assert called_atmos is atmos_sentinel
    assert called_path == '/tmp/run/data/1001_atm.nc'

    # A regression to int() truncation would have produced 1000_atm.nc.
    assert '1000_atm.nc' not in called_path


def test_write_atmos_ncdf_skips_unallocated_struct(monkeypatch, caplog):
    """An unallocated AGNI struct contains no profile, so the writer
    must warn rather than attempt to write it (and likely crash).
    """
    fake_write = MagicMock()
    fake_jl = SimpleNamespace(AGNI=SimpleNamespace(save=SimpleNamespace(write_ncdf=fake_write)))
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)

    # Suppress the warning log to avoid polluting the tests, but capture.
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.atmos_clim.agni'):
        write_atmos_ncdf(SimpleNamespace(is_alloc=False), {'output': '/tmp/run'}, 1000.6)

    # Check that file is not written, and the refusal is surfaced to the log.
    fake_write.assert_not_called()
    assert any('unallocated' in rec.message for rec in caplog.records)

    # Adjacent-valid: the identical call with is_alloc True does write, so the
    # skip above is attributable to the flag and not to the fake or the path.
    write_atmos_ncdf(SimpleNamespace(is_alloc=True), {'output': '/tmp/run'}, 1000.6)
    fake_write.assert_called_once()


class _ProfileAtmosphere:
    """Struct stand-in carrying a temperature profile on a pressure grid.

    Holds the fields `update_agni_atmos` reads and writes. The grid routine
    and the profile-guess routines of `_ProfileAGNI` operate on it, so a test
    can read the profile the function leaves behind rather than only the calls
    it made.
    """

    def __init__(self, p_centres, t_centres):
        self.p = list(p_centres)
        self.tmp = list(t_centres)
        # Level edges bracket the cell centres, so there is one more of them
        self.pl = (
            [self.p[0] / 2.0]
            + [0.5 * (a + b) for a, b in zip(self.p[:-1], self.p[1:])]
            + [self.p[-1] * 2.0]
        )
        self.tmpl = [self.tmp[0]] * (len(self.tmp) + 1)
        self.p_boa = self.pl[-1]
        self.p_oboa = self.pl[-1]
        self.tmp_surf = 1500.0
        self.tmp_magma = 1500.0
        self.instellation = 1361.0
        self.albedo_b = 0.0
        self.transparent = False
        # Broadcast assignment mirrors the Julia arrays the real struct holds
        self.gas_vmr = {'H2O': np.ones(len(self.p))}
        self.gas_ovmr = {'H2O': np.ones(len(self.p))}


class _ProfileAGNI:
    """AGNI stand-in whose grid and profile routines write real numbers.

    `generate_pgrid_b` lays a log-spaced grid between a fixed top pressure and
    the surface pressure held on the struct; `isothermal_b` fills the profile;
    `stratosphere_b` applies its argument as a floor. The guess routines each
    record their call so a test can tell which branch of the config ran.
    """

    P_TOP = 1.0  # [Pa]

    def __init__(self):
        self.calls = []
        self.guess_arg = None
        self.atmosphere = SimpleNamespace(
            generate_pgrid_b=self._generate_pgrid_b,
            make_transparent_b=self._record('make_transparent'),
        )
        self.setpt = SimpleNamespace(
            loglinear_b=self._record('loglinear'),
            isothermal_b=self._isothermal_b,
            dry_adiabat_b=self._record('dry_adiabat'),
            analytic_b=self._record('analytic'),
            stratosphere_b=self._stratosphere_b,
        )

    def _record(self, name):
        """Stub that notes the guess that ran and the value it was given."""

        def _stub(atmos, *args):
            self.calls.append(name)
            self.guess_arg = args[0] if args else None
            # A guess routine always leaves a profile behind, whatever its shape
            atmos.tmp = [1.0] * len(atmos.p)
            atmos.tmpl = [1.0] * len(atmos.pl)

        return _stub

    def _generate_pgrid_b(self, atmos):
        self.calls.append('generate_pgrid')
        nlev_c = len(atmos.p)
        edges = np.logspace(np.log10(self.P_TOP), np.log10(atmos.p_boa), nlev_c + 1)
        atmos.pl = [float(x) for x in edges]
        atmos.p = [float(np.sqrt(a * b)) for a, b in zip(edges[:-1], edges[1:])]

    def _isothermal_b(self, atmos, value):
        self.calls.append('isothermal')
        atmos.tmp = [float(value)] * len(atmos.p)
        atmos.tmpl = [float(value)] * len(atmos.pl)

    def _stratosphere_b(self, atmos, value):
        self.calls.append('stratosphere')
        atmos.tmp = [max(float(value), t) for t in atmos.tmp]
        atmos.tmpl = [max(float(value), t) for t in atmos.tmpl]


def _build_profile_config(ini_profile='isothermal'):
    """Config carrying the fields read while the profile is updated."""
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(
            agni=SimpleNamespace(psurf_thresh=1.0e-3, ini_profile=ini_profile),
        ),
    )


def _install_profile_fakes(monkeypatch, fake_agni):
    """Point the module at the AGNI stand-in and a single-gas composition."""
    fake_jl = SimpleNamespace(
        AGNI=fake_agni,
        Symbol=lambda name: name,
        **{'setfield!': lambda obj, name, value: setattr(obj, name, value)},
    )
    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, '_construct_voldict', lambda *_a, **_k: {'H2O': 1.0})


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_validate_stored_profile_accepts_a_physical_profile():
    """A profile of positive pressures and temperatures is interpolable.

    Physical scenario: the profile left by a converged solve, spanning seven
    decades of pressure from the top of the atmosphere down to a hot surface.
    """
    p_centres = [1.0e-1, 1.0e2, 1.0e5, 1.0e7]
    t_centres = [180.0, 400.0, 1200.0, 2400.0]

    usable, reason = _validate_stored_profile(p_centres, t_centres)
    assert usable
    assert reason == ''

    # The pins above are only meaningful if this profile really does survive
    # the interpolator the guard protects; a guard that passes something
    # scipy rejects would be worthless.
    PchipInterpolator(np.log10(p_centres), t_centres)

    # Edge case: a single-cell profile carries no gradient but is still
    # a valid structure to hold.
    usable_one, reason_one = _validate_stored_profile([5.0e6], [1900.0])
    assert usable_one
    assert reason_one == ''


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_validate_stored_profile_rejects_non_finite_pressure():
    """A NaN or infinite pressure cannot be turned into a log-pressure.

    Physical scenario: the solver diverged and wrote NaN into the pressure
    grid, which is what reaches the interpolator on the next iteration.
    """
    p_nan = [1.0e2, float('nan'), 1.0e6]
    t_ok = [400.0, 900.0, 1800.0]

    usable, reason = _validate_stored_profile(p_nan, t_ok)
    assert not usable
    assert 'atmos.p' in reason
    assert '1 non-finite' in reason

    # Both non-finite kinds are caught, and the count reports every one.
    usable_inf, reason_inf = _validate_stored_profile([float('inf'), float('nan'), 1.0e6], t_ok)
    assert not usable_inf
    assert '2 non-finite' in reason_inf

    # Discrimination: this is exactly the input scipy refuses, so a guard
    # that let it through would return the run to the crash it replaces.
    with pytest.raises(ValueError, match='finite'):
        PchipInterpolator(np.log10(p_nan), t_ok)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_validate_stored_profile_rejects_non_positive_pressure():
    """Pressure must be positive for the profile to be interpolable.

    Physical scenario: a collapsed grid holding zero or negative pressure.
    Finiteness alone does not catch it, since the value only goes non-finite
    once the logarithm is taken.
    """
    t_ok = [400.0, 900.0, 1800.0]

    usable_zero, reason_zero = _validate_stored_profile([1.0e2, 0.0, 1.0e6], t_ok)
    assert not usable_zero
    assert 'atmos.p' in reason_zero
    assert '<= 0 Pa' in reason_zero

    usable_neg, reason_neg = _validate_stored_profile([1.0e2, -3.0e4, 1.0e6], t_ok)
    assert not usable_neg
    assert '<= 0 Pa' in reason_neg

    # Discrimination against a finiteness-only guard: the offending values
    # are finite, and only the logarithm of them is not.
    assert np.all(np.isfinite([1.0e2, 0.0, 1.0e6]))
    with np.errstate(divide='ignore'):
        log_p = np.log10([1.0e2, 0.0, 1.0e6])
    with pytest.raises(ValueError, match='finite'):
        PchipInterpolator(log_p, t_ok)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_validate_stored_profile_rejects_a_grid_that_stops_rising():
    """Pressure must rise with depth for the profile to be interpolable.

    Physical scenario: a grid that repeats a level or folds back on itself. It
    is finite and positive everywhere, so only the ordering distinguishes it
    from a usable column.
    """
    t_ok = [400.0, 900.0, 1800.0, 2200.0]

    p_flat = [1.0e2, 1.0e5, 1.0e5, 1.0e7]
    usable_flat, reason_flat = _validate_stored_profile(p_flat, t_ok)
    assert not usable_flat
    assert 'atmos.p' in reason_flat
    assert '1 level(s)' in reason_flat

    # A grid that folds back on itself is caught at the step that descends.
    usable_back, reason_back = _validate_stored_profile([1.0e2, 1.0e6, 1.0e4, 1.0e7], t_ok)
    assert not usable_back
    assert '1 level(s)' in reason_back

    # Discrimination against a finiteness-and-positivity guard: every value
    # passes both of those, and only the ordering is wrong.
    assert np.all(np.isfinite(p_flat))
    assert min(p_flat) > 0.0
    with pytest.raises(ValueError, match='strictly increasing'):
        PchipInterpolator(np.log10(p_flat), t_ok)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_validate_stored_profile_rejects_unphysical_temperature():
    """Temperature must be finite and above absolute zero.

    Physical scenario: the pressure grid survived the rejected solve but the
    temperatures did not. Carrying them forward would seed the next solve
    with a profile that is not a temperature.
    """
    p_ok = [1.0e2, 1.0e4, 1.0e6]

    usable_nan, reason_nan = _validate_stored_profile(p_ok, [400.0, float('nan'), 1800.0])
    assert not usable_nan
    assert 'atmos.tmp' in reason_nan
    assert 'non-finite' in reason_nan

    usable_zero, reason_zero = _validate_stored_profile(p_ok, [400.0, 0.0, 1800.0])
    assert not usable_zero
    assert '<= 0 K' in reason_zero

    usable_neg, _ = _validate_stored_profile(p_ok, [400.0, -12.0, 1800.0])
    assert not usable_neg

    # The pressures here are the ones accepted above, so the rejection is
    # attributable to the temperatures alone.
    assert _validate_stored_profile(p_ok, [400.0, 900.0, 1800.0])[0]


@pytest.mark.unit
def test_validate_stored_profile_rejects_empty_and_ragged_input():
    """A profile with no cells, or with mismatched arrays, is not a profile.

    Contract clause: the guard runs before any array is indexed, so it has to
    survive the degenerate shapes rather than raise on them.
    """
    usable_empty, reason_empty = _validate_stored_profile([], [])
    assert not usable_empty
    assert 'empty' in reason_empty

    usable_half, reason_half = _validate_stored_profile([1.0e5], [])
    assert not usable_half
    assert 'empty' in reason_half

    usable_ragged, reason_ragged = _validate_stored_profile([1.0e2, 1.0e5], [300.0])
    assert not usable_ragged
    assert '2 pressures' in reason_ragged
    assert '1 temperatures' in reason_ragged


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_update_agni_atmos_interpolates_a_usable_profile(monkeypatch):
    """A usable profile is carried onto the new grid, not thrown away.

    Physical scenario: an ordinary iteration after a converged solve, where
    the surface pressure has grown by outgassing and the stored profile is
    stretched onto the wider grid.
    """
    fake_agni = _ProfileAGNI()
    _install_profile_fakes(monkeypatch, fake_agni)

    atmos = _ProfileAtmosphere([1.0e1, 1.0e3, 1.0e5, 1.0e7], [200.0, 500.0, 1100.0, 1900.0])
    hf_row = {
        'F_ins': 1361.0,
        'albedo_pl': 0.1,
        'T_surf': 1900.0,
        'T_magma': 2000.0,
        'P_surf': 200.0,  # bar
    }

    out = agni_mod.update_agni_atmos(
        atmos, hf_row, {'output': '/tmp/run'}, _build_profile_config()
    )

    assert out is atmos
    # No guess routine ran: the stored profile was used.
    assert 'isothermal' not in fake_agni.calls
    assert fake_agni.calls.count('generate_pgrid') == 1

    # The new grid reaches the new surface pressure [bar -> Pa].
    assert atmos.p_boa == pytest.approx(2.0e7, rel=1e-12)

    # Boundedness: every temperature stays inside the range spanned by the
    # stored profile and the magma clamp above it.
    assert np.all(np.isfinite(atmos.tmp))
    assert min(atmos.tmp) >= 200.0
    assert max(atmos.tmp) <= max(1000.0, hf_row['T_magma'])

    # Monotonicity: the stored profile grows with pressure, and a shape
    # preserving interpolation of it must too.
    assert np.all(np.diff(atmos.tmp) >= 0.0)

    # Discrimination against a guess-profile regression: an isothermal
    # rebuild would have left a single repeated value.
    assert max(atmos.tmp) - min(atmos.tmp) > 100.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_update_agni_atmos_rebuilds_profile_left_non_finite(monkeypatch, caplog):
    """A profile poisoned by a rejected solve is replaced, not interpolated.

    Physical scenario: AGNI exhausted its attempts with NaN in the solution,
    and the next iteration would otherwise take the logarithm of that grid.
    """
    fake_agni = _ProfileAGNI()
    _install_profile_fakes(monkeypatch, fake_agni)
    no_interp = MagicMock(side_effect=AssertionError('interpolator must not be built'))
    monkeypatch.setattr(agni_mod, 'PchipInterpolator', no_interp)
    status = MagicMock()
    monkeypatch.setattr(agni_mod, 'UpdateStatusfile', status)

    atmos = _ProfileAtmosphere(
        [1.0e1, float('nan'), 1.0e5, float('nan')],
        [200.0, 500.0, 1100.0, 1900.0],
    )
    hf_row = {
        'F_ins': 1361.0,
        'albedo_pl': 0.1,
        'T_surf': 2400.0,
        'T_magma': 2400.0,
        'P_surf': 260.0,  # bar
    }

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.atmos_clim.agni'):
        out = agni_mod.update_agni_atmos(
            atmos, hf_row, {'output': '/tmp/run'}, _build_profile_config()
        )

    assert out is atmos
    no_interp.assert_not_called()

    # The run continues, so no failure status is written.
    status.assert_not_called()

    # The struct comes back on a finite, positive grid reaching the surface.
    assert np.all(np.isfinite(atmos.p))
    assert min(atmos.p) > 0.0
    assert atmos.p_boa == pytest.approx(2.6e7, rel=1e-12)

    # Positivity: the profile is the surface boundary condition, and the
    # stratospheric floor sits below it so it is left alone.
    assert np.all(np.isfinite(atmos.tmp))
    assert atmos.tmp == pytest.approx([2400.0] * len(atmos.p))
    assert np.all(np.isfinite(atmos.tmpl))
    assert min(atmos.tmpl) > 0.0

    # The log says which quantity went bad, so the cause is recoverable
    # from the run's own output. Only the pressures were poisoned here, so
    # naming the temperatures would be wrong.
    messages = ' '.join(rec.getMessage() for rec in caplog.records)
    assert 'atmos.p holds 2 non-finite value(s)' in messages
    assert 'atmos.tmp' not in messages


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_update_agni_atmos_rebuilds_when_only_temperatures_are_poisoned(monkeypatch, caplog):
    """A NaN temperature on a clean grid is caught in its own right.

    Physical scenario: the solver iterates on temperature, so a rejected solve
    can leave the pressure grid intact and the profile unusable. The
    interpolator refuses that too, and for a different reason.
    """
    fake_agni = _ProfileAGNI()
    _install_profile_fakes(monkeypatch, fake_agni)
    no_interp = MagicMock(side_effect=AssertionError('interpolator must not be built'))
    monkeypatch.setattr(agni_mod, 'PchipInterpolator', no_interp)

    p_clean = [1.0e1, 1.0e3, 1.0e5, 1.0e7]
    atmos = _ProfileAtmosphere(p_clean, [200.0, float('nan'), 1100.0, 1900.0])
    hf_row = {
        'F_ins': 1361.0,
        'albedo_pl': 0.1,
        'T_surf': 2000.0,
        'T_magma': 2000.0,
        'P_surf': 50.0,
    }

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.atmos_clim.agni'):
        agni_mod.update_agni_atmos(
            atmos, hf_row, {'output': '/tmp/run'}, _build_profile_config()
        )

    no_interp.assert_not_called()
    assert 'isothermal' in fake_agni.calls

    # Positivity: the profile that comes back is the surface condition.
    assert np.all(np.isfinite(atmos.tmp))
    assert atmos.tmp == pytest.approx([2000.0] * len(atmos.p))

    # The offending quantity is named, and the clean pressures are not.
    messages = ' '.join(rec.getMessage() for rec in caplog.records)
    assert 'atmos.tmp holds 1 non-finite value(s)' in messages
    assert 'atmos.p ' not in messages

    # Discrimination: the pressures used here are the ones the interpolated
    # path accepts, so the rebuild is attributable to the temperatures.
    assert _validate_stored_profile(p_clean, [200.0, 600.0, 1100.0, 1900.0])[0]


@pytest.mark.unit
def test_update_agni_atmos_honours_the_configured_guess_on_rebuild(monkeypatch):
    """The rebuild uses the initial guess the config asks for.

    Contract clause: recovery reuses the same guess routine as the first
    iteration, so a run does not silently switch profile shape mid-flight.
    """
    fake_agni = _ProfileAGNI()
    _install_profile_fakes(monkeypatch, fake_agni)

    atmos = _ProfileAtmosphere([1.0e2, 1.0e5], [float('nan'), 1200.0])
    hf_row = {
        'F_ins': 1361.0,
        'albedo_pl': 0.0,
        'T_surf': 1500.0,
        'T_magma': 1500.0,
        'P_surf': 10.0,
    }

    agni_mod.update_agni_atmos(
        atmos, hf_row, {'output': '/tmp/run'}, _build_profile_config('dry_adiabat')
    )
    assert 'dry_adiabat' in fake_agni.calls
    assert 'isothermal' not in fake_agni.calls

    # The floor is applied after the guess, in that order.
    assert fake_agni.calls.index('stratosphere') > fake_agni.calls.index('dry_adiabat')

    # Every guess the config offers is reachable from the rebuild, and the
    # log-linear one is anchored on the surface temperature it is given.
    for name in ('loglinear', 'analytic', 'isothermal'):
        fresh = _ProfileAtmosphere([1.0e2, 1.0e5], [float('nan'), 1200.0])
        agni_mod.update_agni_atmos(
            fresh, hf_row, {'output': '/tmp/run'}, _build_profile_config(name)
        )
        assert fake_agni.calls[-2] == name
        assert fake_agni.calls[-1] == 'stratosphere'
        if name == 'loglinear':
            assert fake_agni.guess_arg == pytest.approx(-750.0)

    # An unknown guess is a configuration error, not an atmosphere one.
    status = MagicMock()
    monkeypatch.setattr(agni_mod, 'UpdateStatusfile', status)
    bad = _ProfileAtmosphere([1.0e2, 1.0e5], [float('nan'), 1200.0])
    with pytest.raises(ValueError, match='Invalid initial'):
        agni_mod.update_agni_atmos(
            bad, hf_row, {'output': '/tmp/run'}, _build_profile_config('parabolic')
        )
    status.assert_called_once_with({'output': '/tmp/run'}, 20)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_update_agni_atmos_fails_as_atmosphere_error_without_a_usable_bc(monkeypatch):
    """With no usable surface condition the run ends as an atmosphere failure.

    Contract clause: there is nothing left to rebuild the profile from, so the
    run stops with status 22 rather than the generic status a bare scipy
    exception would leave behind.
    """
    fake_agni = _ProfileAGNI()
    _install_profile_fakes(monkeypatch, fake_agni)
    status = MagicMock()
    monkeypatch.setattr(agni_mod, 'UpdateStatusfile', status)
    dirs = {'output': '/tmp/run'}
    config = _build_profile_config()

    hf_row = {
        'F_ins': 1361.0,
        'albedo_pl': 0.0,
        'T_surf': float('nan'),
        'T_magma': 2000.0,
        'P_surf': 100.0,
    }
    atmos = _ProfileAtmosphere([1.0e2, 1.0e5], [float('nan'), 1200.0])
    with pytest.raises(RuntimeError, match='T_surf'):
        agni_mod.update_agni_atmos(atmos, hf_row, dirs, config)
    status.assert_called_once_with(dirs, 22)

    # Edge case: a surface pressure that is not a number sits above the
    # transparent-mode threshold on every comparison, so it arrives here.
    status.reset_mock()
    hf_row_p = dict(hf_row, T_surf=1800.0, P_surf=float('nan'))
    atmos_p = _ProfileAtmosphere([1.0e2, 1.0e5], [float('nan'), 1200.0])
    with pytest.raises(RuntimeError, match='P_surf'):
        agni_mod.update_agni_atmos(atmos_p, hf_row_p, dirs, config)
    status.assert_called_once_with(dirs, 22)

    # The same surface condition with a usable profile is not an error, so
    # the failures above are attributable to the profile and the boundary
    # condition together.
    status.reset_mock()
    atmos_ok = _ProfileAtmosphere([1.0e2, 1.0e5], [600.0, 1200.0])
    agni_mod.update_agni_atmos(atmos_ok, dict(hf_row, T_surf=1800.0), dirs, config)
    status.assert_not_called()

    # A surface pressure of zero leaves no column to lay a grid over. In a run
    # it is diverted to transparent mode upstream, so the rebuild is called
    # directly to hold it to its own contract.
    status.reset_mock()
    atmos_z = _ProfileAtmosphere([1.0e2, 1.0e5], [float('nan'), 1200.0])
    calls_before = len(fake_agni.calls)
    with pytest.raises(RuntimeError, match='P_surf'):
        agni_mod._rebuild_agni_profile(
            atmos_z, dict(hf_row, T_surf=1800.0, P_surf=0.0), dirs, config, 'test reason'
        )
    status.assert_called_once_with(dirs, 22)

    # It fails before touching the struct, so the grid is never laid down.
    assert len(fake_agni.calls) == calls_before
    assert atmos_z.p_boa == pytest.approx(2.0e5)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_update_agni_atmos_rejects_a_bad_surface_state_behind_a_good_profile(monkeypatch):
    """A surface state that is not finite and positive ends the run even when
    the stored profile is perfectly usable.

    Contract clause: temperature and pressure are positive quantities, and the
    check on the stored profile cannot see the boundary condition carried
    beside it. Edge case: a surface pressure of infinity, which sets the top of
    the interpolation grid and raises inside scipy, and one of NaN, which is
    invisible because every comparison against it is false.
    """
    fake_agni = _ProfileAGNI()
    _install_profile_fakes(monkeypatch, fake_agni)
    status = MagicMock()
    monkeypatch.setattr(agni_mod, 'UpdateStatusfile', status)
    dirs = {'output': '/tmp/run'}
    config = _build_profile_config()

    healthy = {
        'F_ins': 1361.0,
        'albedo_pl': 0.0,
        'T_surf': 1800.0,
        'T_magma': 3000.0,
        'P_surf': 260.0,
    }

    # The same profile with a usable surface state goes through, so each
    # failure below is attributable to the boundary condition alone.
    agni_mod.update_agni_atmos(
        _ProfileAtmosphere([1.0e2, 1.0e5], [600.0, 1200.0]), dict(healthy), dirs, config
    )
    status.assert_not_called()

    for field, value in (
        ('P_surf', float('inf')),
        ('P_surf', float('nan')),
        ('P_surf', -1.0),
        ('T_surf', float('nan')),
        ('T_surf', 0.0),
        ('T_magma', float('nan')),
        ('T_magma', -1.0),
    ):
        status.reset_mock()
        atmos = _ProfileAtmosphere([1.0e2, 1.0e5], [600.0, 1200.0])
        with pytest.raises(RuntimeError, match=field):
            agni_mod.update_agni_atmos(atmos, dict(healthy, **{field: value}), dirs, config)
        status.assert_called_once_with(dirs, 22)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_surface_state_accepts_a_planet_that_has_lost_its_atmosphere():
    """A surface pressure of zero is a planet with no atmosphere left, not a
    spoiled boundary condition, so it must pass the surface-state check.

    Contract clause: zero means different things either side of the boundary
    condition. Desiccation sets ``P_surf`` to exactly zero deliberately and an
    escape step that outpaces the column clamps it there, and transparent mode
    handles that case without ever reading the value. A temperature of zero is
    not a temperature and stays refused. Edge case: both boundaries at once.
    """
    from proteus.atmos_clim.agni import _validate_surface_state

    healthy = {'T_surf': 1800.0, 'T_magma': 3000.0, 'P_surf': 260.0}

    with patch('proteus.atmos_clim.agni.UpdateStatusfile') as status:
        # The terminal state, and a pressure below the transparent threshold.
        _validate_surface_state(dict(healthy, P_surf=0.0), {})
        _validate_surface_state(dict(healthy, P_surf=1.0e-5), {})
        status.assert_not_called()

    # Discrimination: the neighbouring values on either side of that zero are
    # still refused, so accepting it is not the check going slack.
    for field, value in (('P_surf', -1.0e-30), ('T_surf', 0.0), ('T_magma', 0.0)):
        with patch('proteus.atmos_clim.agni.UpdateStatusfile') as status:
            with pytest.raises(RuntimeError, match=field):
                _validate_surface_state(dict(healthy, **{field: value}), {})
            status.assert_called_once_with({}, 22)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_lost_atmosphere_goes_to_transparent_mode_not_the_pressure_grid(monkeypatch):
    """A planet with no atmosphere left must reach transparent mode, which is
    what makes accepting a surface pressure of zero safe.

    Contract clause: the surface-state check lets zero through because the
    branch below it handles that case without reading the value. Laying a
    log-spaced grid on a surface pressure of zero is what the routing avoids,
    so the test pins the routing rather than the check. Edge case: exactly
    zero, which desiccation produces, against a pressure that takes the grid.
    """
    fake_agni = _ProfileAGNI()
    _install_profile_fakes(monkeypatch, fake_agni)
    status = MagicMock()
    monkeypatch.setattr(agni_mod, 'UpdateStatusfile', status)
    dirs = {'output': '/tmp/run'}
    config = _build_profile_config()
    healthy = {
        'F_ins': 1361.0,
        'albedo_pl': 0.0,
        'T_surf': 1800.0,
        'T_magma': 3000.0,
        'P_surf': 260.0,
    }

    agni_mod.update_agni_atmos(
        _ProfileAtmosphere([1.0e2, 1.0e5], [600.0, 1200.0]),
        dict(healthy, P_surf=0.0),
        dirs,
        config,
    )
    status.assert_not_called()
    assert 'make_transparent' in fake_agni.calls
    assert 'isothermal' in fake_agni.calls
    # The grid is what a surface pressure of zero must never reach.
    assert 'generate_pgrid' not in fake_agni.calls

    # Discrimination: a pressure that clears the threshold does take the grid,
    # so the routing above is the branch and not simply what the fake does.
    fake_agni.calls.clear()
    agni_mod.update_agni_atmos(
        _ProfileAtmosphere([1.0e2, 1.0e5], [600.0, 1200.0]), dict(healthy), dirs, config
    )
    assert 'generate_pgrid' in fake_agni.calls
    assert 'make_transparent' not in fake_agni.calls
    status.assert_not_called()
