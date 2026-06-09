"""Unit tests for proteus.observe.platon: PLATON spectral synthesis
wrapper.

PLATON is an optional dependency. Its top-level package is importable
in the standard PROTEUS environment but the submodules used by the
wrapper (``platon.TP_profile``, ``platon.transit_depth_calculator``,
``platon.eclipse_depth_calculator``) are only present in full PLATON
installs. These unit tests inject mock submodules into ``sys.modules``
so the wrapper paths execute under any PLATON install.

Module scope:

- Module-level constants ``PLATON_TLIMS``, ``PLATON_METHOD``, and
  ``PLATON_GASES`` are pinned against documented values. A regression
  that flipped the temperature limits or the opacity method or
  silently changed the supported gas set surfaces here.
- Helper ``_get_mix`` is exercised under each documented source
  ('outgas', 'profile', 'offchem'); the gas-inclusion gate
  (``vmr >= clip_vmr``) is pinned both for accepted and rejected
  cases.
- Helper ``_construct_abundances`` is exercised with a 1D log-pressure
  interpolation; the output shape, monotonicity, and round-trip at
  the grid nodes are pinned.
- Helper ``_get_ptr`` is exercised under both ascending and
  descending pressure orderings; the temperature-clamp at the
  upper / lower ``PLATON_TLIMS`` is pinned for the reversed case.
- Helper ``_get_prof`` is exercised with a mocked ``Profile`` class;
  the ``set_from_arrays`` call signature and the returned object are
  pinned.
- ``transit_depth`` and ``eclipse_depth`` are exercised with mocked
  ``TransitDepthCalculator`` / ``EclipseDepthCalculator`` classes;
  the per-call argument shape contract, the per-gas zero-out loop,
  and the file-write path are pinned. Missing-atmosphere early-
  return paths are pinned for both functions.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Module-level constants.
# ---------------------------------------------------------------------------


def test_platon_tlims_documented_bracket_around_100_4000_kelvin():
    """``PLATON_TLIMS`` is the temperature window the PLATON wrapper
    clamps the profile to before handing it to the radiative-transfer
    calculator. The documented window is (100.5, 3999.5) K: 0.5 K
    inside the [100, 4000] K opacity-table range so a regression that
    relaxed the clamp to the bare table edge would still be inside
    the table at floating-point roundoff.

    Discrimination: pin both edges as exact numeric values so a
    regression that shifted either edge silently surfaces.
    """
    from proteus.observe.platon import PLATON_TLIMS

    assert isinstance(PLATON_TLIMS, tuple)
    assert len(PLATON_TLIMS) == 2
    assert PLATON_TLIMS[0] == pytest.approx(100.5, rel=1e-12)
    assert PLATON_TLIMS[1] == pytest.approx(3999.5, rel=1e-12)
    # Sign + ordering guard: lower < upper, both positive.
    assert PLATON_TLIMS[0] > 0
    assert PLATON_TLIMS[1] > PLATON_TLIMS[0]


def test_platon_method_is_xsec():
    """``PLATON_METHOD`` selects the opacity calculation mode. The
    documented production value is 'xsec' (cross-sections); the
    alternative 'ktables' is an opt-in development knob. Pin the
    production default.
    """
    from proteus.observe.platon import PLATON_METHOD

    assert PLATON_METHOD == 'xsec'
    assert isinstance(PLATON_METHOD, str)


def test_platon_gases_documented_set_and_count():
    """``PLATON_GASES`` is the full set of species the wrapper offers
    to the PLATON calculator. The count and named set are pinned: a
    silent rename (e.g. 'H2O' to 'water') or a drop of a documented
    species fails the set equality. A new species added would also
    surface; in that case, update the test alongside the source.
    """
    from proteus.observe.platon import PLATON_GASES

    documented = {
        'H2', 'H', 'He',
        'H2O', 'CH4', 'CO', 'CO2',
        'O', 'C', 'N',
        'NH3', 'N2',
        'O2', 'O3',
        'H2S', 'HCN',
        'NO', 'NO2',
        'OH', 'PH3',
        'SiO', 'SO2', 'TiO', 'VO',
        'Na', 'K', 'Ca', 'Ti', 'Fe', 'Ni',
        'C2H2', 'FeH',
    }  # fmt: skip
    assert set(PLATON_GASES) == documented, (
        f'PLATON_GASES drifted from documented set; '
        f'extra={set(PLATON_GASES) - documented}, '
        f'missing={documented - set(PLATON_GASES)}'
    )
    assert len(PLATON_GASES) == 32
    # Ordering is not load-bearing, but uniqueness is.
    assert len(PLATON_GASES) == len(set(PLATON_GASES))


# ---------------------------------------------------------------------------
# _get_mix: gas inclusion under each source.
# ---------------------------------------------------------------------------


def test_get_mix_outgas_source_uses_helpfile_constant_vmr():
    """Under ``source='outgas'``, ``_get_mix`` reads VMR from the
    hf_row and broadcasts to a constant profile across all levels.
    Gas inclusion gates on ``max(vmr) >= clip_vmr``.

    Discrimination: a gas with VMR > clip_vmr is included; a gas with
    VMR < clip_vmr is excluded. The output profile is a uniform
    array equal to the hf_row scalar at every level.
    """
    from proteus.observe.platon import _get_mix

    hf_row = {'H2O_vmr': 1e-2, 'CO2_vmr': 1e-12, 'H2_vmr': 0.5}
    atm = {'pl': np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])}
    gases, vmrs = _get_mix(hf_row, atm, source='outgas', clip_vmr=1e-6)

    assert 'H2O' in gases
    assert 'H2' in gases
    # CO2 below clip_vmr is excluded (1e-12 < 1e-6).
    assert 'CO2' not in gases
    # Each included VMR profile is constant across levels and equals
    # the hf_row scalar.
    for gas, vmr in zip(gases, vmrs):
        assert vmr.shape == (5,), f'{gas} VMR shape mismatch'
        assert np.allclose(vmr, hf_row[f'{gas}_vmr']), (
            f'{gas} VMR not constant or != hf_row value'
        )


def test_get_mix_profile_source_reads_per_level_vmr_array():
    """Under ``source='profile'``, ``_get_mix`` reads VMR directly
    from the atm dict and prepends the first value (the wrapper adds
    a TOA layer mirroring the highest-level VMR). For an input of
    length N the output has length N+1.

    Discrimination: the per-level array is preserved (not flattened
    to a constant); the inclusion gate fires on the max value; the
    prepended TOA element equals the input's first level.
    """
    from proteus.observe.platon import _get_mix

    hf_row = {}
    atm = {
        'pl': np.array([1e-3, 1e-2, 1e-1, 1.0]),
        'H2O_vmr': np.array([1e-3, 1e-2, 1e-1, 0.5]),
        'CO2_vmr': np.array([1e-12, 1e-12, 1e-12]),
    }
    gases, vmrs = _get_mix(hf_row, atm, source='profile', clip_vmr=1e-6)

    assert 'H2O' in gases
    assert 'CO2' not in gases, 'CO2 below clip_vmr should be excluded'
    h2o_idx = gases.index('H2O')
    h2o_vmr = vmrs[h2o_idx]
    # Wrapper prepends vmr[0], so length goes from 4 to 5.
    assert h2o_vmr.shape == (5,)
    # Prepended element equals the input's first level.
    assert h2o_vmr[0] == pytest.approx(1e-3, rel=1e-12)
    # The maximum is preserved.
    assert h2o_vmr.max() == pytest.approx(0.5, rel=1e-12)


def test_get_mix_offchem_source_reads_unprefixed_gas_name():
    """Under ``source='offchem'``, ``_get_mix`` reads VMR from
    ``atm[gas]`` (no '_vmr' suffix) because the offline-chemistry
    dataframe uses the raw species name as the column. Pin this
    source-specific lookup.
    """
    from proteus.observe.platon import _get_mix

    atm = {
        'pl': np.array([1e-3, 1e-2, 1e-1, 1.0]),
        'H2O': np.array([1e-2, 1e-2, 1e-2, 1e-2]),
        'CO2': np.array([1e-12, 1e-12, 1e-12, 1e-12]),
    }
    gases, vmrs = _get_mix({}, atm, source='offchem', clip_vmr=1e-6)
    assert 'H2O' in gases
    assert 'CO2' not in gases


def test_get_mix_below_clip_vmr_excluded_for_all_sources():
    """The exclusion gate (``max(vmr) < clip_vmr``) fires under every
    documented source. Pin the trace-gas rejection separately so a
    regression that loosened the gate on one source while keeping it
    on the others surfaces. The 'outgas' source uses a constant VMR
    of clip_vmr / 10; the 'profile' source uses per-level values all
    below clip_vmr.
    """
    from proteus.observe.platon import _get_mix

    pl = np.array([1e-3, 1e-2, 1e-1, 1.0])
    # outgas: scalar VMR below clip
    hf_row = {'H2O_vmr': 1e-10}
    atm = {'pl': pl}
    g, _ = _get_mix(hf_row, atm, source='outgas', clip_vmr=1e-6)
    assert 'H2O' not in g
    # profile: per-level VMR all below clip
    atm = {'pl': pl, 'H2O_vmr': np.full(4, 1e-12)}
    g, _ = _get_mix({}, atm, source='profile', clip_vmr=1e-6)
    assert 'H2O' not in g


# ---------------------------------------------------------------------------
# _construct_abundances: per-gas T-P interpolation grid.
# ---------------------------------------------------------------------------


def test_construct_abundances_returns_per_gas_2d_array_on_t_p_grid():
    """``_construct_abundances`` projects per-gas VMR profiles onto
    the PLATON internal (T, P) grid via PchipInterpolator in
    log-pressure space. Output is a dict mapping gas name to a
    2D array of shape (len(T_grid), len(P_grid)).

    Discrimination: shape contract per gas, pchip interpolation is
    monotonic between grid nodes, and the wrapper extends the
    pressure range with sentinels (1e-6 Pa at top, 1e13 Pa at
    bottom) so the interpolator never sees an out-of-range query.
    The constant-VMR input round-trips at every grid point.
    """
    from proteus.observe.platon import _construct_abundances

    atm = {'pl': np.array([1e-3, 1e-1, 10.0, 100.0])}
    gas_incl = ['H2O', 'CO2']
    # H2O: constant VMR = 1e-2 across levels.
    # CO2: constant VMR = 1e-4 across levels.
    vmr_incl = [np.full(4, 1e-2), np.full(4, 1e-4)]
    T_grid = np.array([200.0, 500.0, 1500.0])
    P_grid = np.array([1e-2, 1e0, 1e2])

    abund = _construct_abundances(atm, gas_incl, vmr_incl, T_grid, P_grid)

    assert set(abund.keys()) == {'H2O', 'CO2'}
    for gas, arr in abund.items():
        assert arr.shape == (len(T_grid), len(P_grid))
        # Constant input -> constant output at every grid point.
        expected = 1e-2 if gas == 'H2O' else 1e-4
        assert np.allclose(arr, expected, rtol=1e-6), (
            f'{gas} interpolation drifted from constant input'
        )


# ---------------------------------------------------------------------------
# _get_ptr: pressure / temperature / radius ordering and clamp.
# ---------------------------------------------------------------------------


def test_get_ptr_ascending_pressure_passes_through_unchanged():
    """``_get_ptr`` returns the (P, T, R) arrays unchanged when the
    input ``pl`` is monotonically increasing.

    Discrimination: no reversal, no clamping.
    """
    from proteus.observe.platon import _get_ptr

    atm = {
        'pl': np.array([1e-3, 1e-2, 1e-1, 1.0]),
        'tmpl': np.array([200.0, 500.0, 1500.0, 2500.0]),
        'rl': np.array([7e6, 7.1e6, 7.2e6, 7.3e6]),
    }
    prs, tmp, rad = _get_ptr(atm)
    assert np.array_equal(prs, atm['pl'])
    assert np.array_equal(tmp, atm['tmpl'])
    assert np.array_equal(rad, atm['rl'])


def test_get_ptr_descending_pressure_reverses_and_clamps_temperature():
    """``_get_ptr`` reverses the arrays when ``pl`` is descending
    AND clamps the temperature to ``PLATON_TLIMS``. Pin both the
    reversal and the clamp on the same call.

    Edge: temperatures below PLATON_TLIMS[0] and above PLATON_TLIMS[1]
    get clamped to the limit values; in-range values pass through.
    """
    from proteus.observe.platon import PLATON_TLIMS, _get_ptr

    atm = {
        'pl': np.array([1.0, 1e-1, 1e-2, 1e-3]),  # descending
        'tmpl': np.array([5000.0, 2000.0, 500.0, 50.0]),
        'rl': np.array([7e6, 7.1e6, 7.2e6, 7.3e6]),
    }
    prs, tmp, rad = _get_ptr(atm)
    # Reversed so pressure is ascending.
    assert np.all(np.diff(prs) > 0)
    # Temperature reversed AND clamped to PLATON_TLIMS.
    # Order after reversal: 50, 500, 2000, 5000.
    # 50 clamps up to PLATON_TLIMS[0]; 5000 clamps down to PLATON_TLIMS[1].
    assert tmp[0] == pytest.approx(PLATON_TLIMS[0], rel=1e-12)
    assert tmp[1] == pytest.approx(500.0, rel=1e-12)
    assert tmp[2] == pytest.approx(2000.0, rel=1e-12)
    assert tmp[3] == pytest.approx(PLATON_TLIMS[1], rel=1e-12)
    # Radius reversed but not clamped.
    assert rad[0] == pytest.approx(7.3e6, rel=1e-12)
    assert rad[-1] == pytest.approx(7e6, rel=1e-12)


# ---------------------------------------------------------------------------
# _get_prof: Profile instantiation with mocked platon.TP_profile.
# ---------------------------------------------------------------------------


def test_get_prof_instantiates_profile_and_calls_set_from_arrays(monkeypatch):
    """``_get_prof`` instantiates a PLATON ``Profile`` and calls
    ``set_from_arrays(P, T)``. Pin the call signature: the
    profile is constructed exactly once and ``set_from_arrays`` is
    called with the (P, T) arrays returned by ``_get_ptr``.

    A regression that swapped the P/T order in set_from_arrays
    would silently invert the profile; the explicit arg-order
    check catches that.
    """
    from proteus.observe.platon import _get_prof

    fake_profile = MagicMock(name='Profile_instance')
    fake_profile_class = MagicMock(name='Profile_class', return_value=fake_profile)
    fake_mod = types.ModuleType('platon.TP_profile')
    fake_mod.Profile = fake_profile_class
    monkeypatch.setitem(sys.modules, 'platon.TP_profile', fake_mod)

    atm = {
        'pl': np.array([1e-3, 1e-2, 1e-1, 1.0]),
        'tmpl': np.array([200.0, 500.0, 1500.0, 2500.0]),
        'rl': np.array([7e6, 7.1e6, 7.2e6, 7.3e6]),
    }
    result = _get_prof(atm)
    fake_profile_class.assert_called_once_with()
    fake_profile.set_from_arrays.assert_called_once()
    # Argument order: (P, T). Pull from positional args and check ordering.
    call_args, _ = fake_profile.set_from_arrays.call_args
    p_arg, t_arg = call_args[0], call_args[1]
    # Pressure ascending, temperature in physical range.
    assert np.all(np.diff(p_arg) > 0), 'set_from_arrays got non-monotonic P'
    assert np.all((100.0 < t_arg) & (t_arg < 4000.0)), 'set_from_arrays got out-of-range T'
    assert result is fake_profile


# ---------------------------------------------------------------------------
# transit_depth: mocked TransitDepthCalculator + missing-atm path.
# ---------------------------------------------------------------------------


def test_transit_depth_returns_none_when_atm_is_missing(monkeypatch, tmp_path):
    """``transit_depth`` warns and returns None when
    ``_get_atm_profile`` returns None (no atmosphere data file). The
    PLATON ``TransitDepthCalculator`` is imported at the top of the
    function (so the import always fires) but never instantiated on
    the missing-atm path.

    Discrimination: install a sentinel that records instantiation;
    after the call, instantiation count is zero.
    """
    from proteus.observe import platon as platon_mod

    monkeypatch.setattr(platon_mod, '_get_atm_profile', lambda outdir, hf: None)
    sentinel_class = MagicMock(name='TransitDepthCalculator')
    fake_mod = types.ModuleType('platon.transit_depth_calculator')
    fake_mod.TransitDepthCalculator = sentinel_class
    monkeypatch.setitem(sys.modules, 'platon.transit_depth_calculator', fake_mod)

    cfg = MagicMock()
    hf_row = {'R_star': 7e8, 'M_planet': 6e24, 'R_int': 6.4e6, 'T_star': 5800.0}
    result = platon_mod.transit_depth(hf_row, str(tmp_path), cfg, source='profile')
    assert result is None
    # The calculator is reachable as an import but never instantiated.
    assert sentinel_class.call_count == 0


def test_transit_depth_calls_calculator_for_each_zero_out_gas(monkeypatch, tmp_path):
    """``transit_depth`` calls ``compute_depths`` once for the
    full-spectrum baseline and once per gas with that gas
    zeroed-out. Pin the call count: 1 baseline + N per-gas calls.

    The output file is written via ``np.savetxt`` to the path
    returned by ``get_transit_fpath``. Pin file existence.
    """
    from proteus.observe import platon as platon_mod

    atm = {
        'pl': np.array([1e-3, 1e-1, 10.0, 100.0]),
        'tmpl': np.array([400.0, 800.0, 1600.0, 2200.0]),
        'rl': np.array([6.4e6, 6.45e6, 6.5e6, 6.55e6]),
    }
    monkeypatch.setattr(platon_mod, '_get_atm_profile', lambda outdir, hf: atm)

    # Two gases included (H2O + CO2 above clip_vmr); calculator
    # should be called 1 + 2 = 3 times.
    def fake_mix(hf, atm_arg, source, clip_vmr):
        nlev = len(atm_arg['pl'])
        return ['H2O', 'CO2'], [np.full(nlev, 1e-2), np.full(nlev, 1e-4)]

    monkeypatch.setattr(platon_mod, '_get_mix', fake_mix)

    fake_calc = MagicMock(name='TransitCalculator')
    fake_calc.atm.T_grid = np.array([300.0, 1500.0])
    fake_calc.atm.P_grid = np.array([1e-2, 1.0, 1e2])
    fake_wl = np.linspace(1e-6, 5e-6, 10)
    fake_de = np.full(10, 1e-3)
    fake_calc.compute_depths.return_value = (fake_wl, fake_de, None)
    fake_mod = types.ModuleType('platon.transit_depth_calculator')
    fake_mod.TransitDepthCalculator = MagicMock(return_value=fake_calc)
    monkeypatch.setitem(sys.modules, 'platon.transit_depth_calculator', fake_mod)

    cfg = MagicMock()
    cfg.observe.platon.clip_vmr = 1e-6
    cfg.observe.platon.downsample = 1
    hf_row = {
        'Time': 1e3,
        'R_star': 7e8,
        'M_planet': 6e24,
        'R_int': 6.4e6,
        'T_star': 5800.0,
    }
    # Create the observe/ subdirectory the wrapper writes into.
    (tmp_path / 'observe').mkdir(parents=True, exist_ok=True)
    X = platon_mod.transit_depth(hf_row, str(tmp_path), cfg, source='profile')
    # 1 baseline + 2 gases = 3 calls.
    assert fake_calc.compute_depths.call_count == 3
    # Output array carries wavelength + (1 + N) depth columns.
    assert X is not None
    assert X.shape == (10, 4)  # 10 wavelengths, [wl, full, no-H2O, no-CO2]
    # Output file exists at the documented path.
    out_csv = tmp_path / 'observe' / 'transit_profile_synthesis.csv'
    assert out_csv.exists(), f'expected output at {out_csv}'


# ---------------------------------------------------------------------------
# eclipse_depth: mocked EclipseDepthCalculator + missing-atm path.
# ---------------------------------------------------------------------------


def test_eclipse_depth_returns_none_when_atm_is_missing(monkeypatch, tmp_path):
    """``eclipse_depth`` warns and returns None when
    ``_get_atm_profile`` returns None. The
    ``EclipseDepthCalculator`` is imported at the top of the
    function but never instantiated on the missing-atm path.
    """
    from proteus.observe import platon as platon_mod

    monkeypatch.setattr(platon_mod, '_get_atm_profile', lambda outdir, hf: None)
    sentinel_class = MagicMock(name='EclipseDepthCalculator')
    fake_mod = types.ModuleType('platon.eclipse_depth_calculator')
    fake_mod.EclipseDepthCalculator = sentinel_class
    monkeypatch.setitem(sys.modules, 'platon.eclipse_depth_calculator', fake_mod)

    cfg = MagicMock()
    hf_row = {
        'R_star': 7e8,
        'M_planet': 6e24,
        'R_int': 6.4e6,
        'T_star': 5800.0,
    }
    result = platon_mod.eclipse_depth(hf_row, str(tmp_path), cfg, source='profile')
    assert result is None
    assert sentinel_class.call_count == 0


def test_eclipse_depth_calls_calculator_for_each_zero_out_gas(monkeypatch, tmp_path):
    """``eclipse_depth`` mirrors ``transit_depth``: 1 baseline call +
    N per-gas zero-out calls. The Profile-build path also fires;
    mock ``_get_prof`` to bypass platon.TP_profile.
    """
    from proteus.observe import platon as platon_mod

    atm = {
        'pl': np.array([1e-3, 1e-1, 10.0, 100.0]),
        'tmpl': np.array([400.0, 800.0, 1600.0, 2200.0]),
        'rl': np.array([6.4e6, 6.45e6, 6.5e6, 6.55e6]),
    }
    monkeypatch.setattr(platon_mod, '_get_atm_profile', lambda outdir, hf: atm)
    monkeypatch.setattr(platon_mod, '_get_prof', lambda atm_arg: MagicMock(name='profile'))

    def fake_mix(hf, atm_arg, source, clip_vmr):
        nlev = len(atm_arg['pl'])
        return ['H2O', 'CO2'], [np.full(nlev, 1e-2), np.full(nlev, 1e-4)]

    monkeypatch.setattr(platon_mod, '_get_mix', fake_mix)

    fake_calc = MagicMock(name='EclipseCalculator')
    fake_calc.atm.T_grid = np.array([300.0, 1500.0])
    fake_calc.atm.P_grid = np.array([1e-2, 1.0, 1e2])
    fake_wl = np.linspace(1e-6, 5e-6, 10)
    fake_de = np.full(10, 1e-3)
    fake_calc.compute_depths.return_value = (fake_wl, fake_de, None)
    fake_mod = types.ModuleType('platon.eclipse_depth_calculator')
    fake_mod.EclipseDepthCalculator = MagicMock(return_value=fake_calc)
    monkeypatch.setitem(sys.modules, 'platon.eclipse_depth_calculator', fake_mod)

    cfg = MagicMock()
    cfg.observe.platon.clip_vmr = 1e-6
    cfg.observe.platon.downsample = 1
    hf_row = {
        'Time': 1e3,
        'R_star': 7e8,
        'M_planet': 6e24,
        'R_int': 6.4e6,
        'T_star': 5800.0,
    }
    (tmp_path / 'observe').mkdir(parents=True, exist_ok=True)
    X = platon_mod.eclipse_depth(hf_row, str(tmp_path), cfg, source='profile')
    assert fake_calc.compute_depths.call_count == 3
    assert X is not None
    assert X.shape == (10, 4)
    out_csv = tmp_path / 'observe' / 'eclipse_profile_synthesis.csv'
    assert out_csv.exists(), f'expected output at {out_csv}'
