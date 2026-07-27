"""Unit tests for proteus.atmos_clim.janus module.

Covers the JANUS-side atmosphere-wrapper functions without booting
the real JANUS / SOCRATES stack. The actual JANUS imports happen
inside each function body and are mocked via unittest.mock.patch so
the wrapper logic (overlap-method dispatch, unit conversions, branch
selection, error contracts) is exercised in isolation.

Module under test ships these public callables:

- ``InitStellarSpectrum``: write the merged stellar spectrum into
  the SOCRATES spectral file. The wrapper delegates to
  ``janus.utils.PrepareStellarSpectrum`` /
  ``janus.utils.InsertStellarSpectrum`` and removes the staged file.
- ``InitAtm``: build the ``janus.utils.atmos`` object and dispatch
  ``atm.overlap_type`` from ``config.atmos_clim.overlap_method``
  (``'ro'`` -> 2, ``'ee'`` -> 4, ``'rorr'`` -> 8, otherwise raise
  ``ValueError``).
- ``UpdateStateAtm``: push the current iteration state into the
  ``atmos`` mutable. Tests cover the bar -> Pa pressure conversion,
  the rock-vapour warning branch keyed off ``vap_list[gas]_vmr >
  1e-5``, and the ``tropopause='skin'`` vs default branch.

Invariants asserted:

- Overlap-method dispatch is a 3-of-4 enumerated map (positivity +
  pinned distinct values 2 / 4 / 8 act as the discrimination guard
  against a regression that homogenises the codes).
- Surface-pressure conversion uses the 1e5 bar -> Pa factor; pinned
  with a discriminating value (10 bar -> 1e6 Pa, not 1e2 Pa or 1e8
  Pa).
- Rock-vapour warning fires only when at least one ``vap_list``
  species sits above the ``1e-5`` mixing-ratio threshold.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.atmos_clim import janus as janus_mod
from proteus.atmos_clim.janus import (
    InitAtm,
    InitStellarSpectrum,
    RunJANUS,
    UpdateStateAtm,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _build_overlap_config(overlap_method: str) -> SimpleNamespace:
    """Build a minimal Config object that drives ``InitAtm``.

    Only the fields read inside ``InitAtm`` need to exist; everything
    else stays unset so a regression that started reading new fields
    would surface as an ``AttributeError`` rather than silently
    picking up a default.
    """
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(
            p_top=1e-5,
            num_levels=40,
            cloud_enabled=False,
            surf_greyalbedo=0.3,
            tmp_minimum=50.0,
            overlap_method=overlap_method,
            surface_d=0.0,
            surface_k=0.0,
            janus=SimpleNamespace(
                cloud_alpha=1.0,
                tmp_maximum=3500.0,
            ),
        ),
        orbit=SimpleNamespace(zenith_angle=48.0, s0_factor=1.0),
    )


def _build_update_config() -> SimpleNamespace:
    """Build a Config object for ``UpdateStateAtm``."""
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(p_top=1e-5),
    )


def _build_update_hf_row(*, T_surf=1500.0, P_surf=10.0, vap_vmr=0.0):
    """Construct an hf_row dict with all keys ``UpdateStateAtm``
    consumes. ``vap_vmr`` controls the rock-vapour threshold path.
    """
    from proteus.utils.constants import vap_list, vol_list

    hf_row = {
        'T_surf': T_surf,
        'P_surf': P_surf,
        'R_int': 6.371e6,
        'M_int': 5.972e24,
        'F_ins': 1361.0,
        'albedo_pl': 0.3,
        'T_magma': 2000.0,
        'T_skin': 250.0,
    }
    for gas in vol_list:
        hf_row[gas + '_vmr'] = 1.0 / len(vol_list)
    for vap in vap_list:
        hf_row[vap + '_vmr'] = vap_vmr
    return hf_row


@pytest.mark.unit
@pytest.mark.physics_invariant
@patch('janus.utils.atmos')
@patch('janus.utils.ReadBandEdges')
def test_init_atm_overlap_method_ro_maps_to_2(mock_read, mock_atmos):
    """Overlap method ``'ro'`` must drive ``atm.overlap_type = 2``.

    JANUS / SOCRATES expects the integer overlap code, not the string;
    the wrapper does the translation. ``'ro'`` is the random-overlap
    method, pinned to overlap_type 2 in the SOCRATES spectral handler.
    """
    fake_atm = MagicMock()
    mock_atmos.return_value = fake_atm

    cfg = _build_overlap_config('ro')
    result = InitAtm({'output': '/tmp/run/'}, cfg)

    assert result is fake_atm
    assert result.overlap_type == 2
    # Discrimination guard: a regression to a homogenised dispatch
    # (e.g. always 8, or always 0) would not satisfy the pinned value.
    assert result.overlap_type != 4
    assert result.overlap_type != 8


@pytest.mark.unit
@pytest.mark.physics_invariant
@patch('janus.utils.atmos')
@patch('janus.utils.ReadBandEdges')
def test_init_atm_overlap_method_ee_maps_to_4(mock_read, mock_atmos):
    """Overlap method ``'ee'`` (equivalent-extinction) maps to 4."""
    fake_atm = MagicMock()
    mock_atmos.return_value = fake_atm

    cfg = _build_overlap_config('ee')
    result = InitAtm({'output': '/tmp/run/'}, cfg)

    assert result.overlap_type == 4
    # Discrimination guard.
    assert result.overlap_type != 2
    assert result.overlap_type != 8


@pytest.mark.unit
@pytest.mark.physics_invariant
@patch('janus.utils.atmos')
@patch('janus.utils.ReadBandEdges')
def test_init_atm_overlap_method_rorr_maps_to_8(mock_read, mock_atmos):
    """Overlap method ``'rorr'`` maps to 8 (random-overlap with
    resorting-and-rebinning)."""
    fake_atm = MagicMock()
    mock_atmos.return_value = fake_atm

    cfg = _build_overlap_config('rorr')
    result = InitAtm({'output': '/tmp/run/'}, cfg)

    assert result.overlap_type == 8
    # Discrimination guard.
    assert result.overlap_type != 2
    assert result.overlap_type != 4


@pytest.mark.unit
@patch('janus.utils.atmos')
@patch('janus.utils.ReadBandEdges')
def test_init_atm_unknown_overlap_method_raises(mock_read, mock_atmos):
    """Unknown overlap methods must hit the ``case _`` raise branch.

    The wrapper accepts only the documented enum; any other string is
    a config-side mistake and the ``match/case`` default arm raises
    ``ValueError``. Picking ``'rrr'`` (a plausible typo for ``'rorr'``)
    confirms the error contract fires for near-miss inputs rather
    than silently falling through to a default code.
    """
    mock_atmos.return_value = MagicMock()

    cfg = _build_overlap_config('rrr')
    with pytest.raises(ValueError, match='Invalid overlap method selected for SOCRATES/JANUS!'):
        InitAtm({'output': '/tmp/run/'}, cfg)

    # Discrimination: switching to a valid overlap method on the same
    # config scaffold must succeed. Rules out a regression that
    # hard-raises regardless of the method string.
    cfg_valid = _build_overlap_config('ro')
    result = InitAtm({'output': '/tmp/run/'}, cfg_valid)
    assert result is not None


@pytest.mark.unit
@pytest.mark.physics_invariant
@patch('janus.utils.atmos')
@patch('janus.utils.ReadBandEdges')
def test_init_atm_passes_top_pressure_in_pa(mock_read, mock_atmos):
    """The wrapper feeds ``atmos`` the top pressure in Pa, not bar.

    The Config holds ``p_top`` in bar. JANUS' ``atmos`` constructor
    expects SI Pa. The conversion is the third positional argument:
    ``p_top * 1e5``. Pinning the value 1e-3 bar -> 1e2 Pa
    discriminates from a regression that forgot the conversion
    (would land at 1e-3) or applied it twice (would land at 1e7).
    """
    mock_atmos.return_value = MagicMock()

    cfg = _build_overlap_config('ro')
    cfg.atmos_clim.p_top = 1e-3
    InitAtm({'output': '/tmp/run/'}, cfg)

    args, _ = mock_atmos.call_args
    p_top_arg = args[2]
    assert p_top_arg == pytest.approx(1e2, rel=1e-12)
    # Discrimination guard: a bar-to-Pa regression would land 5 orders
    # of magnitude lower; a double-conversion would land 5 orders higher.
    assert p_top_arg > 1.0
    assert p_top_arg < 1e6


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_update_state_atm_converts_surface_pressure_to_pa():
    """``setSurfacePressure`` receives bar -> Pa conversion x 1e5.

    Pin P_surf = 10 bar (= 1e6 Pa). A regression that dropped the
    1e5 factor would land at 10; a regression that used 1e3 instead
    would land at 1e4. The cross-magnitude gap is large enough that
    pytest.approx with default tolerance discriminates trivially.
    """
    fake_atm = MagicMock()
    hf_row = _build_update_hf_row(P_surf=10.0)
    cfg = _build_update_config()

    UpdateStateAtm(fake_atm, cfg, hf_row, tropopause=None)

    # max(P_surf, p_top * 1.1) = max(10, 1.1e-5) = 10 bar -> 1e6 Pa
    assert fake_atm.setSurfacePressure.call_count == 1
    pa_arg = fake_atm.setSurfacePressure.call_args.args[0]
    assert pa_arg == pytest.approx(1e6, rel=1e-12)
    # Discrimination guard: rule out the missing-conversion regression
    # (would be 10) and the double-conversion regression (would be 1e11).
    assert pa_arg > 1e3
    assert pa_arg < 1e9
    # Also confirm the surface temperature was set, with the pinned value.
    fake_atm.setSurfaceTemperature.assert_called_once_with(1500.0)


@pytest.mark.unit
def test_update_state_atm_warns_on_rock_vapours(caplog):
    """Rock-vapour warning fires only when ``vap_list`` species exceed
    1e-5 mixing ratio.

    The threshold is the limit-input behaviour. The test pairs two
    cases inside one function as a threshold-discrimination guard:
    first the above-threshold value 2e-5 must trigger exactly one
    warning record; then 0.0 must not. Splitting the pair into two
    tests would let a regression that fires the warning
    unconditionally pass the above-threshold half silently.
    """
    import logging

    fake_atm = MagicMock()
    hf_row = _build_update_hf_row(vap_vmr=2e-5)
    cfg = _build_update_config()

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.atmos_clim.janus'):
        UpdateStateAtm(fake_atm, cfg, hf_row, tropopause=None)

    # At least one warning record was emitted, and its text mentions
    # the wrapper's documented intent of neglecting rock vapours.
    rock_records = [rec for rec in caplog.records if 'rock vapour' in rec.getMessage().lower()]
    assert len(rock_records) == 1
    # Discrimination guard against a regression that fires the warning
    # unconditionally: the call below uses vap_vmr = 0 and must NOT
    # warn. Running in the same caplog context to keep one fixture.
    caplog.clear()
    hf_row_clean = _build_update_hf_row(vap_vmr=0.0)
    UpdateStateAtm(fake_atm, cfg, hf_row_clean, tropopause=None)
    rock_records_clean = [
        rec for rec in caplog.records if 'rock vapour' in rec.getMessage().lower()
    ]
    assert len(rock_records_clean) == 0


@pytest.mark.unit
def test_update_state_atm_skin_tropopause_uses_t_skin():
    """``tropopause='skin'`` plumbs ``hf_row['T_skin']`` into ``atm.trppT``.

    The non-skin branch falls back to the literal 0.5 (an arbitrary
    floor used when no skin model is active). Pinning T_skin = 250 K
    discriminates from the floor value clearly.
    """
    fake_atm = MagicMock()
    hf_row = _build_update_hf_row()
    cfg = _build_update_config()

    UpdateStateAtm(fake_atm, cfg, hf_row, tropopause='skin')

    assert fake_atm.trppT == pytest.approx(250.0, rel=1e-12)
    # Discrimination guard: the fallback floor is 0.5; the skin path
    # must not pick that up.
    assert fake_atm.trppT > 1.0


@pytest.mark.unit
def test_update_state_atm_default_tropopause_uses_floor():
    """Anything other than ``'skin'`` (including ``None``) yields the
    0.5 K floor on ``atm.trppT``.

    The behaviour is intentional: JANUS still needs a finite floor
    value when no skin model is active. A regression that propagated
    ``T_skin`` regardless of the tropopause flag would lift the value
    well above 0.5.
    """
    fake_atm = MagicMock()
    hf_row = _build_update_hf_row()
    cfg = _build_update_config()

    UpdateStateAtm(fake_atm, cfg, hf_row, tropopause=None)

    assert fake_atm.trppT == pytest.approx(0.5, rel=1e-12)
    # Discrimination guard: a regression that picked up T_skin here
    # would land at 250.0 K, three orders of magnitude above the floor.
    assert fake_atm.trppT < 10.0


def _build_run_config():
    """Config namespace covering the fields ``RunJANUS`` reads."""
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(
            p_obs=1e-3,  # bar
            rayleigh=False,
            surf_state='fixed',
            janus=SimpleNamespace(tropopause=None, F_atm_bc=0),
        ),
        planet=SimpleNamespace(prevent_warming=False),
        escape=SimpleNamespace(xuv_defined_by_radius=False),
    )


def _build_run_atm(*, g_surf, z_obs_height, stale_grav):
    """Fake ``atmos`` returned by the (mocked) JANUS solver.

    ``grav_z`` is deliberately populated with a stale/wrong profile so the
    test fails if ``RunJANUS`` ever reads gravity off that array again instead
    of deriving it from the returned radius.
    """
    atm = SimpleNamespace()
    # Height array (m above surface); the observed level is the last entry via
    # the patched interpolation below.
    atm.z = np.array([0.0, z_obs_height * 0.5, z_obs_height], dtype=float)
    atm.grav_z = np.array([stale_grav, stale_grav, stale_grav], dtype=float)
    atm.tmp = np.array([1400.0, 900.0, 500.0], dtype=float)
    atm.p = np.array([1e6, 1e4, 1e2], dtype=float)
    atm.net_flux = np.array([120.0, 80.0, 40.0], dtype=float)
    atm.LW_flux_up = np.array([110.0, 70.0, 30.0], dtype=float)
    atm.SW_flux_up = np.array([5.0, 4.0, 3.0], dtype=float)
    atm.SW_flux_down = np.array([200.0, 150.0, 100.0], dtype=float)
    atm.clfr = np.array([0.0, 0.0, 0.0], dtype=float)
    atm.ps = 1.0e6  # Pa
    atm.ts = 1400.0  # K
    atm.height_error = False
    atm.x_gas = {}
    atm.write_ncdf = lambda path: None
    return atm


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_janus_gobs_inverse_square_of_robs(monkeypatch, tmp_path):
    """RunJANUS returns g_obs consistent with R_obs via the inverse-square law."""
    g_surf = 9.81
    R_int = 6.371e6
    z_obs_height = 8.0e4  # m; observed level sits this far above the surface

    atm = _build_run_atm(g_surf=g_surf, z_obs_height=z_obs_height, stale_grav=g_surf)

    # Solver is mocked to return the fake atmosphere unchanged.
    monkeypatch.setattr('janus.modules.MCPA', lambda *a, **kw: atm)
    # State push and hydrostatic solve are covered by their own unit tests.
    monkeypatch.setattr(janus_mod, 'UpdateStateAtm', lambda *a, **kw: None)
    # Deterministic "observed level" = last array entry, independent of p_obs.
    monkeypatch.setattr(
        janus_mod, 'get_oarr_from_parr', lambda p_arr, o_arr, val: (0, o_arr[-1])
    )

    config = _build_run_config()
    hf_row = {
        'Time': 100.0,
        'R_int': R_int,
        'gravity': g_surf,
        'p_xuv': 1e-3,  # bar
        'T_surf': 1400.0,
    }
    dirs = {'output': str(tmp_path)}

    # Run JANUS
    output = RunJANUS(atm, dirs, config, hf_row, None, write_in_tmp_dir=False)

    # Check additive radii
    R_obs_expected = R_int + z_obs_height
    assert output['R_obs'] == pytest.approx(R_obs_expected, rel=1e-12)

    # Check gravity scales as inverse square.
    g_obs_expected = g_surf * (R_int / R_obs_expected) ** 2
    assert output['g_obs'] == pytest.approx(g_obs_expected, rel=1e-10)

    # The observed level is above the surface, so gravity must be weaker there.
    assert output['R_obs'] > R_int
    assert output['g_obs'] < g_surf

    # Discrimination guard: the stale grav_z value is the surface gravity
    assert output['g_obs'] != pytest.approx(g_surf, rel=1e-3)


@pytest.mark.unit
@patch('proteus.atmos_clim.janus.os.remove')
@patch('janus.utils.InsertStellarSpectrum')
@patch('janus.utils.PrepareStellarSpectrum')
def test_init_stellar_spectrum_calls_janus_utilities_in_order(
    mock_prepare, mock_insert, mock_remove
):
    """The wrapper stages a SOCRATES spectrum file then inserts it.

    Both janus.utils helpers must run, in order, with the stage path
    that the wrapper composes from ``dirs['output']``. The staged
    file is removed afterwards to keep the run directory clean.
    """
    dirs = {'output': '/tmp/runXYZ/'}
    wl = [100.0, 200.0, 300.0]
    fl = [1e-3, 2e-3, 3e-3]

    InitStellarSpectrum(dirs, wl, fl, '/tmp/spectral.nostar')

    mock_prepare.assert_called_once()
    mock_insert.assert_called_once()
    # PrepareStellarSpectrum stages to dirs['output'] + 'socrates_star.txt'
    prep_args = mock_prepare.call_args.args
    assert prep_args[0] is wl
    assert prep_args[1] is fl
    assert prep_args[2] == '/tmp/runXYZ/socrates_star.txt'
    # InsertStellarSpectrum reads from the staged file and writes the
    # merged spectrum back into the output dir.
    ins_args = mock_insert.call_args.args
    assert ins_args[0] == '/tmp/spectral.nostar'
    assert ins_args[1] == '/tmp/runXYZ/socrates_star.txt'
    assert ins_args[2] == '/tmp/runXYZ/'
    # Discrimination guard: the staged file must be removed exactly
    # once; a regression that forgot the cleanup would leave the
    # staged file behind.
    mock_remove.assert_called_once_with('/tmp/runXYZ/socrates_star.txt')
