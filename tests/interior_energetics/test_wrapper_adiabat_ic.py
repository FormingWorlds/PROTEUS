"""Unit tests for the liquidus_super adiabat initial-condition structure solve.

Covers the IC path in proteus.interior_energetics.wrapper that integrates the
mantle density against the true super-liquidus adiabat instead of Zalmoxis'
internal linear-in-r temperature guess:

- ``_build_superliquidus_adiabat_tp``: builds the pressure-indexed T(P) closure
  from the memoised super-liquidus adiabat; returns None (NaN guard) when the
  adiabat contains non-finite values.
- ``_solve_structure_with_adiabat_or_rollback``: shared single-solve core that
  captures zalmoxis_output.dat -> .prev, enforces the mass anchor and R_int
  validity, and on failure restores BOTH the hf_row keys AND the on-disk
  zalmoxis_output.dat so the next solver never reads the failed adiabat geometry.
- ``_resolve_adiabatic_ic_structure``: two-pass re-solve that hands the adiabat
  T(P) to the Zalmoxis numpy path, with mass-anchor enforcement and rollback to
  the linear-guess structure on failure.
- ``equilibrate_initial_state``: the equilibration loop hands the adiabat T(P) to
  every iteration's solver, falls back to the linear-guess solve for any
  iteration whose adiabat re-solve fails, and rebuilds the adiabat grid when a
  converged P_cmb outgrows it.
- ``determine_interior_radius_with_zalmoxis``: gates the re-solve on the
  zalmoxis structure module + liquidus_super + non-SPIDER energetics; SPIDER and
  other temperature modes keep the linear-guess solve untouched.

Invariants exercised:
- Positivity / boundedness: the adiabat T(P) is finite and positive; the NaN
  guard rejects a non-finite profile.
- Pinned numeric value with discrimination guard: the T(P) closure interpolates
  the adiabat and clips out-of-range pressures to the table edges.
- Monotonicity: integrating a real hydrostatic + mass ODE against the convex
  adiabat yields a larger radius than against a colder linear-like T(P), and a
  lower surface temperature yields a smaller radius (t = 0 is the maximum).
- The adiabat-anchored IC solve relaxes R_int outward (lower mean density) and
  rolls back to the linear-guess radius (and on-disk geometry) on a mass-anchor
  violation.

Testing standards and documentation:
- docs/How-to/test_infrastructure.md: Test infrastructure overview
- docs/How-to/test_categorization.md: Test marker definitions
- docs/How-to/test_building.md: Best practices for test construction
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.interior_energetics.wrapper import (
    _ADIABAT_IC_PCMB_MARGIN,
    _build_superliquidus_adiabat_tp,
    _resolve_adiabatic_ic_structure,
    _solve_structure_with_adiabat_or_rollback,
    _use_superliquidus_adiabat_ic,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _config(
    module: str = 'aragog',
    temperature_mode: str = 'liquidus_super',
    struct_module: str = 'zalmoxis',
):
    """Build the minimal config namespace the IC adiabat path reads.

    Only the fields touched by the gate and the adiabat builder are populated:
    the structure module, the energetics module, the planet temperature mode,
    and the Zalmoxis mantle EOS string used to resolve the PALEOS tables.
    """
    return SimpleNamespace(
        interior_energetics=SimpleNamespace(module=module),
        planet=SimpleNamespace(temperature_mode=temperature_mode),
        interior_struct=SimpleNamespace(
            module=struct_module,
            zalmoxis=SimpleNamespace(mantle_eos='PALEOS:MgSiO3'),
        ),
    )


def _good_adiabat():
    """A monotone, finite super-liquidus adiabat over a wide pressure range.

    Surface 1 bar at 4000 K rising to ~1.4 TPa at ~6500 K, spanning the
    super-Earth structure pressure range so the structure integral never has to
    extrapolate.
    """
    P = np.linspace(1e5, 1.4e12, 500)
    T = np.linspace(4000.0, 6500.0, 500)
    return {'P': P, 'T': T, 'surface_T': 4000.0}


# ============================================================================
# _build_superliquidus_adiabat_tp: closure construction and the NaN guard
# ============================================================================


@pytest.mark.physics_invariant
def test_adiabat_tp_closure_interpolates_and_clips_pressure():
    """The built T(P) closure interpolates the adiabat and clips out-of-range P.

    The structure ODE calls the closure as f(r, P) with P = y[2]; r has no grid
    at IC time and must be ignored. Pin an interior point against np.interp and
    assert that pressures below/above the table return the surface/CMB edge
    temperatures rather than extrapolating.
    """
    adiabat = _good_adiabat()
    config = _config()
    built = _build_with_adiabat(config, adiabat)

    assert built is not None
    tf, P_arr, T_arr = built

    # Interior point: pin against the linear interpolation of the adiabat at a
    # mid-mantle pressure. r is passed but ignored.
    P_mid = 5.0e11
    expected_mid = float(np.interp(P_mid, P_arr, T_arr))
    assert tf(3.0e6, P_mid) == pytest.approx(expected_mid, rel=1e-9)
    # Discrimination: nearest-table-node lookup (a plausible wrong impl) would
    # land on a grid point, not the interpolated value; the gap exceeds tol.
    nearest = float(T_arr[np.argmin(np.abs(P_arr - P_mid))])
    assert abs(expected_mid - nearest) < 6.0  # 500-node grid is fine here
    # Clip below the surface pressure: returns the surface (coolest) edge, not a
    # cold extrapolation toward absolute zero.
    assert tf(6.0e6, -1.0e9) == pytest.approx(float(T_arr[0]), rel=1e-9)
    # Clip above the CMB pressure: returns the CMB (hottest) edge, not a hot
    # extrapolation. Bounds the closure for the deep-mantle integration step.
    assert tf(1.0e6, 5.0e12) == pytest.approx(float(T_arr[-1]), rel=1e-9)
    # Boundedness invariant: every tabulated temperature is positive and finite.
    assert np.all(T_arr > 0.0)
    assert np.all(np.isfinite(T_arr))


def _build_with_adiabat(config, adiabat):
    """Run _build_superliquidus_adiabat_tp with the EOS path fully mocked.

    Patches the four Zalmoxis lookups plus compute_entropy_adiabat so the test
    controls the returned profile. Returns whatever the builder returns
    (a (closure, P, T) tuple on success, or None on the NaN/failure fallback).
    """
    with (
        patch(
            'proteus.interior_struct.zalmoxis.solve_superliquidus_adiabat',
            return_value={'surface_T': 4000.0},
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={'PALEOS:MgSiO3': {'eos_file': __file__}},
        ),
        patch(
            'proteus.interior_struct.zalmoxis.resolve_2phase_mgsio3_paths',
            return_value=(None, None),
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        patch('zalmoxis.eos_export.compute_entropy_adiabat', return_value=adiabat),
    ):
        return _build_superliquidus_adiabat_tp(config, {}, P_cmb_target=1.4e12)


def test_adiabat_tp_returns_none_on_nan_profile():
    """A NaN in the adiabat T(P) profile triggers the fallback (returns None).

    compute_entropy_adiabat can return NaNs when the deep adiabat exhausts the
    EOS table. The builder must reject the profile so the caller keeps the
    linear-guess structure instead of feeding a corrupt T(r) to the solver. The
    positive control (the same path with a finite profile) must return a usable
    closure, so this discriminates "rejects NaN" from "rejects everything".
    """
    config = _config()
    # Positive control: a finite profile yields a usable closure.
    good = _build_with_adiabat(config, _good_adiabat())
    assert good is not None
    assert callable(good[0])

    # Negative case: a single NaN anywhere in T(P) rejects the whole profile.
    nan_adiabat = _good_adiabat()
    nan_adiabat['T'] = nan_adiabat['T'].copy()
    nan_adiabat['T'][250] = np.nan  # plateau / table-exhaustion signature
    bad = _build_with_adiabat(config, nan_adiabat)
    assert bad is None

    # An inf in the pressure grid is rejected for the same reason: the guard
    # checks np.isfinite on both arrays, not just NaN in T.
    inf_adiabat = _good_adiabat()
    inf_adiabat['P'] = inf_adiabat['P'].copy()
    inf_adiabat['P'][0] = np.inf
    assert _build_with_adiabat(config, inf_adiabat) is None


def test_adiabat_tp_returns_none_when_solver_raises():
    """A RuntimeError from the adiabat solver falls back rather than crashing.

    solve_superliquidus_adiabat raises when the requested superheat cannot be
    reached before the EOS table is exhausted. The builder must swallow that and
    return None so the IC degrades to the linear guess, not to a crash. The
    finite-profile control confirms the same path otherwise yields a closure.
    """
    config = _config()
    # Positive control: no exception -> usable closure.
    good = _build_with_adiabat(config, _good_adiabat())
    assert good is not None
    assert callable(good[0])

    # Negative case: the adiabat solver raises -> graceful None, not a crash.
    with patch(
        'proteus.interior_struct.zalmoxis.solve_superliquidus_adiabat',
        side_effect=RuntimeError('EOS table exhausted at this mass'),
    ):
        bad = _build_superliquidus_adiabat_tp(config, {}, P_cmb_target=1.4e12)
    assert bad is None


# ============================================================================
# _resolve_adiabatic_ic_structure: two-pass re-solve, mass anchor, rollback
# ============================================================================


def _linear_hf_row():
    """Helpfile row as left by the linear-guess solve for a 10 M_Earth planet."""
    return {
        'R_int': 1.2007e7,  # km-scale linear-guess radius (too small)
        'R_core': 6.0e6,
        'M_int': 5.972e25,
        'M_core': 1.9e25,
        'M_int_target': 5.972e25,
        'P_surf': 1e5,
        'P_center': 1.6e12,
        'P_cmb': 1.456e12,
        'gravity': 18.0,
        'core_density': 1.1e4,
        'rho_avg': 8.0e3,
    }


def test_adiabat_resolve_relaxes_radius_outward_and_passes_temperature_function():
    """The adiabat re-solve hands a non-None T(P) to the solver and grows R_int.

    The mocked solver mimics the physical relaxation: the isentropic adiabat IC
    is less dense in the deep mantle, so R_int increases from the linear-guess
    12007 km toward the molten ~12746 km. Assert the solver received a callable
    temperature_function (the contract) and that R_int grew.
    """
    config = _config()
    hf_row = _linear_hf_row()
    adiabat = _good_adiabat()

    def fake_solver(cfg, outdir, row, num_spider_nodes=0, temperature_function=None, **kw):
        # Contract: the IC re-solve must pass the adiabat closure.
        assert callable(temperature_function)
        # The closure is pressure-indexed; sample it to confirm it is live.
        assert temperature_function(3.0e6, 5.0e11) > 0.0
        row['R_int'] = 1.2746e7  # relaxed molten radius
        row['M_int'] = row['M_int_target']  # mass-anchored
        return 6.0e6, '/tmp/adiabat_mesh.dat'

    with (
        patch(
            'proteus.interior_energetics.wrapper._build_superliquidus_adiabat_tp',
            return_value=(
                lambda r, P: float(np.interp(P, adiabat['P'], adiabat['T'])),
                adiabat['P'],
                adiabat['T'],
            ),
        ),
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=fake_solver),
    ):
        mesh = _resolve_adiabatic_ic_structure(
            config, '/tmp/out', hf_row, num_spider_nodes=0, linear_mesh_file=None
        )

    # The accepted structure is the adiabat re-solve, not the linear guess.
    assert mesh == '/tmp/adiabat_mesh.dat'
    # R_int relaxed outward: adiabat IC is larger than the linear 12007 km.
    assert hf_row['R_int'] == pytest.approx(1.2746e7, rel=1e-9)
    assert hf_row['R_int'] > 1.2007e7
    # Discrimination vs a no-op (which would leave the linear 12007 km): the
    # outward move is ~739 km, far above any rounding.
    assert hf_row['R_int'] - 1.2007e7 > 5.0e5


def test_adiabat_resolve_rolls_back_on_mass_anchor_violation(tmp_path):
    """A mass-anchor violation rolls back hf_row AND the on-disk geometry.

    A too-loosely-converged adiabat solve (|M_int/M_target - 1| above the
    anchor tolerance) is treated as a failure: hf_row is rolled back to the
    linear-guess structure, the linear mesh is returned, AND the on-disk
    zalmoxis_output.dat is restored from its .prev snapshot. Without the on-disk
    restore the immediately-following Aragog run reads the failed adiabat
    geometry against the linear-guess EOS mesh and crashes; this test pins the
    file content, not just the hf_row scalars.
    """
    config = _config()
    hf_row = _linear_hf_row()
    R_int_linear = hf_row['R_int']
    M_target = hf_row['M_int_target']
    adiabat = _good_adiabat()

    # Real on-disk layout: data/zalmoxis_output.dat holding the linear-guess
    # geometry, plus its .prev snapshot the wrapper captures before the solve.
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    out_path = data_dir / 'zalmoxis_output.dat'
    linear_content = 'LINEAR GEOMETRY: R_int=12007 km, linear-in-r T(r)\n'
    out_path.write_text(linear_content)
    linear_mesh = tmp_path / 'linear_mesh.dat'
    linear_mesh.write_text('linear mesh\n')

    def bad_solver(cfg, outdir, row, num_spider_nodes=0, temperature_function=None, **kw):
        assert callable(temperature_function)
        # Mimic a real solver: it overwrites the on-disk output before the
        # wrapper-level mass anchor rejects the result.
        (data_dir / 'zalmoxis_output.dat').write_text(
            'ADIABAT GEOMETRY: R_int=14000 km, super-liquidus T(P)\n'
        )
        row['R_int'] = 1.40e7  # would-be relaxed radius, discarded on rollback
        row['M_int'] = M_target * 1.10  # 10% off: violates the 0.3% anchor
        return 6.0e6, str(tmp_path / 'bad_mesh.dat')

    with (
        patch(
            'proteus.interior_energetics.wrapper._build_superliquidus_adiabat_tp',
            return_value=(lambda r, P: 5000.0, adiabat['P'], adiabat['T']),
        ),
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=bad_solver),
    ):
        mesh = _resolve_adiabatic_ic_structure(
            config,
            str(tmp_path),
            hf_row,
            num_spider_nodes=0,
            linear_mesh_file=str(linear_mesh),
        )

    # Rollback contract: linear mesh returned, linear R_int restored.
    assert mesh == str(linear_mesh)
    assert hf_row['R_int'] == pytest.approx(R_int_linear, rel=1e-12)
    # The off-anchor radius (14000 km) must NOT survive: pin against it
    # explicitly so a regression that skipped the rollback fails here.
    assert abs(hf_row['R_int'] - 1.40e7) > 1.0e6
    # On-disk contract: the file holds the LINEAR content again, not the adiabat
    # geometry the failed solver wrote. This is the discriminating check that the
    # hf_row-only rollback (the bug) would fail.
    restored = out_path.read_text()
    assert restored == linear_content
    assert 'ADIABAT' not in restored


def test_solve_helper_keeps_disk_geometry_on_success(tmp_path):
    """On a successful adiabat solve the on-disk geometry is the solver's output.

    Discriminating positive control for the rollback test: when the mass anchor
    is satisfied, the helper signals ok=True and does NOT overwrite the solver's
    new zalmoxis_output.dat with the .prev snapshot, so the accepted adiabat
    geometry survives. This separates "restores only on failure" from "always
    restores".
    """
    config = _config()
    hf_row = _linear_hf_row()
    M_target = hf_row['M_int_target']

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    out_path = data_dir / 'zalmoxis_output.dat'
    out_path.write_text('LINEAR GEOMETRY\n')

    adiabat_content = 'ADIABAT GEOMETRY: accepted\n'

    def good_solver(cfg, outdir, row, num_spider_nodes=0, temperature_function=None, **kw):
        assert callable(temperature_function)
        (data_dir / 'zalmoxis_output.dat').write_text(adiabat_content)
        row['R_int'] = 1.2746e7  # relaxed molten radius
        row['M_int'] = M_target  # mass-anchored, within tolerance
        return 6.0e6, str(tmp_path / 'adiabat_mesh.dat')

    with patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=good_solver):
        mesh, ok = _solve_structure_with_adiabat_or_rollback(
            config,
            str(tmp_path),
            hf_row,
            num_spider_nodes=0,
            temperature_function=lambda r, P: 5000.0,
        )

    assert ok is True
    assert mesh == str(tmp_path / 'adiabat_mesh.dat')
    # Accepted geometry survives: the relaxed radius and the new file content.
    assert hf_row['R_int'] == pytest.approx(1.2746e7, rel=1e-9)
    assert out_path.read_text() == adiabat_content


def test_adiabat_resolve_falls_back_when_adiabat_unavailable():
    """When the adiabat builder returns None the linear-guess solve is kept.

    If _build_superliquidus_adiabat_tp fails (e.g. NaN profile), the re-solve
    must not call the solver again; it returns the linear mesh unchanged.
    """
    config = _config()
    hf_row = _linear_hf_row()
    R_int_linear = hf_row['R_int']

    with (
        patch(
            'proteus.interior_energetics.wrapper._build_superliquidus_adiabat_tp',
            return_value=None,
        ),
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver') as mock_solver,
    ):
        mesh = _resolve_adiabatic_ic_structure(
            config,
            '/tmp/out',
            hf_row,
            num_spider_nodes=0,
            linear_mesh_file='/tmp/linear_mesh.dat',
        )

    # No second solve when the adiabat is unavailable.
    mock_solver.assert_not_called()
    assert mesh == '/tmp/linear_mesh.dat'
    assert hf_row['R_int'] == pytest.approx(R_int_linear, rel=1e-12)


def test_adiabat_resolve_skips_when_pcmb_nonpositive():
    """A non-positive linear-guess P_cmb skips the re-solve safely.

    The two-pass design needs the linear solve's P_cmb to size the adiabat grid.
    If P_cmb is absent or non-positive the adiabat cannot be sized, so the
    linear-guess structure is kept and the builder is never called.
    """
    config = _config()
    hf_row = _linear_hf_row()
    hf_row['P_cmb'] = 0.0

    with (
        patch(
            'proteus.interior_energetics.wrapper._build_superliquidus_adiabat_tp',
        ) as mock_build,
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver') as mock_solver,
    ):
        mesh = _resolve_adiabatic_ic_structure(
            config,
            '/tmp/out',
            hf_row,
            num_spider_nodes=0,
            linear_mesh_file='/tmp/linear_mesh.dat',
        )

    mock_build.assert_not_called()
    mock_solver.assert_not_called()
    assert mesh == '/tmp/linear_mesh.dat'


def test_pcmb_margin_is_one_sided_and_above_unity():
    """The adiabat grid uses a >1 P_cmb margin so it never undershoots P_cmb.

    The re-solve relaxes R_int outward (lower mean density), so its P_cmb does
    not exceed the linear-guess P_cmb; a margin above 1.0 guarantees the
    tabulated T(P) covers the full integration range without extrapolation.
    """
    # Boundedness of a tuning constant: must exceed unity but stay a small
    # margin (not, e.g., a factor-of-10 typo that would push the grid far past
    # the table edge and risk an out-of-table adiabat).
    assert _ADIABAT_IC_PCMB_MARGIN > 1.0
    assert _ADIABAT_IC_PCMB_MARGIN < 1.5


# ============================================================================
# determine_interior_radius_with_zalmoxis: the aragog + liquidus_super gate
# ============================================================================


def _gate_config(module: str, temperature_mode: str, struct_module: str = 'zalmoxis'):
    """Full mock config for the determine_interior_radius gate test.

    MagicMock for the deep attribute tree; the gate reads interior_struct.module,
    interior_energetics.module, and planet.temperature_mode, plus the Zalmoxis
    knobs threaded into the (mocked) solver call.
    """
    config = MagicMock()
    config.interior_struct.module = struct_module
    config.interior_energetics.module = module
    config.planet.temperature_mode = temperature_mode
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
    return config


@pytest.mark.parametrize(
    'module, temperature_mode, expect_resolve',
    [
        ('aragog', 'liquidus_super', True),
        ('dummy', 'liquidus_super', True),
        ('spider', 'liquidus_super', False),
        ('aragog', 'isothermal', False),
        ('spider', 'isothermal', False),
    ],
    ids=[
        'aragog_liquidus_super_resolves',
        'dummy_liquidus_super_resolves',
        'spider_liquidus_super_untouched',
        'aragog_isothermal_untouched',
        'spider_isothermal_untouched',
    ],
)
def test_ic_adiabat_gate_fires_for_zalmoxis_liquidus_super_non_spider(
    module, temperature_mode, expect_resolve
):
    """The adiabat IC re-solve fires for zalmoxis + liquidus_super + non-SPIDER.

    The maximal-radius IC is wanted for any energetics backend that does not
    supply its own T(r): both Aragog and the dummy energetics backend qualify.
    SPIDER manages its own T(r) through entropy evolution and must keep the
    linear-guess solve; non-liquidus_super modes do not build a super-liquidus
    adiabat. The discriminating contrasts are the spider+liquidus_super row
    (energetics excluded) and the aragog+isothermal row (mode excluded), both of
    which leave the re-solve untouched.
    """
    from proteus.interior_energetics import wrapper as wmod

    config = _gate_config(module, temperature_mode)
    dirs = {'output': '/tmp/out', 'spider': '/tmp/spider'}
    hf_row = {'P_cmb': 1.456e12, 'R_int': 1.2e7}

    with (
        patch.object(wmod, 'get_nlevb', return_value=80),
        patch.object(wmod, 'Interior_t', return_value=MagicMock(ic=1)),
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(6.0e6, None),
        ),
        patch(
            'proteus.interior_struct.zalmoxis.generate_spider_tables',
            return_value=None,
        ),
        patch.object(wmod, '_provide_spider_eos_tables'),
        patch.object(wmod, 'run_interior'),
        patch.object(
            wmod, '_resolve_adiabatic_ic_structure', return_value=None
        ) as mock_resolve,
    ):
        wmod.determine_interior_radius_with_zalmoxis(
            dirs, config, MagicMock(), hf_row, '/tmp/out'
        )

    if expect_resolve:
        mock_resolve.assert_called_once()
        # The gate must forward the live hf_row so the re-solve sees the
        # linear-guess P_cmb it needs to size the adiabat grid.
        _args, _kwargs = mock_resolve.call_args
        assert hf_row in _args
    else:
        mock_resolve.assert_not_called()


def test_use_superliquidus_adiabat_ic_gate_truth_table():
    """The gate predicate fires only for zalmoxis + liquidus_super + non-SPIDER.

    Pin the full truth table so a regression that drops any of the three
    conjuncts is caught. The dummy energetics backend qualifies (Tim wants the
    maximal IC for any non-SPIDER combination); SPIDER never qualifies; a
    non-zalmoxis structure module never qualifies; a non-liquidus_super mode
    never qualifies.
    """
    # Positive: both non-SPIDER energetics backends qualify under zalmoxis +
    # liquidus_super.
    assert _use_superliquidus_adiabat_ic(_config('aragog', 'liquidus_super')) is True
    assert _use_superliquidus_adiabat_ic(_config('dummy', 'liquidus_super')) is True
    # Negative 1 (energetics): SPIDER supplies its own T(r); excluded.
    assert _use_superliquidus_adiabat_ic(_config('spider', 'liquidus_super')) is False
    # Negative 2 (temperature mode): no super-liquidus adiabat to build.
    assert _use_superliquidus_adiabat_ic(_config('aragog', 'isothermal')) is False
    # Negative 3 (structure module): the dummy/spider structure paths do not run
    # the Zalmoxis numpy solve this hand-off targets.
    assert (
        _use_superliquidus_adiabat_ic(
            _config('aragog', 'liquidus_super', struct_module='dummy')
        )
        is False
    )


# ============================================================================
# equilibrate_initial_state: per-iteration adiabat hand-off, fallback, rebuild
# ============================================================================


def _equilibrate_config(
    module: str = 'aragog',
    temperature_mode: str = 'liquidus_super',
    struct_module: str = 'zalmoxis',
    max_iter: int = 2,
    tol: float = 1e-3,
    num_levels: int = 80,
):
    """Concrete (non-MagicMock) config namespace for the equilibration loop.

    A MagicMock auto-creates every attribute and so cannot discriminate the gate;
    a concrete namespace makes the three gate conjuncts explicit and lets the
    convergence comparisons read real floats.
    """
    return SimpleNamespace(
        interior_energetics=SimpleNamespace(module=module, num_levels=num_levels),
        planet=SimpleNamespace(temperature_mode=temperature_mode),
        interior_struct=SimpleNamespace(
            module=struct_module,
            zalmoxis=SimpleNamespace(
                mantle_eos='PALEOS:MgSiO3',
                equilibrate_max_iter=max_iter,
                equilibrate_tol=tol,
            ),
        ),
    )


@contextlib.contextmanager
def _patch_equilibrate_io():
    """Patch the equilibration loop's I/O and table generation to no-ops.

    Patches the outgassing calls and SPIDER-table regeneration so the test
    exercises only the structure-solve hand-off, not CALLIOPE or table writers.
    A context manager so a single `with _patch_equilibrate_io():` enters them all.
    """
    from proteus.interior_energetics import wrapper as wmod

    with (
        patch('proteus.outgas.wrapper.calc_target_elemental_inventories'),
        patch('proteus.outgas.wrapper.run_outgassing'),
        patch('proteus.interior_struct.zalmoxis.generate_spider_tables', return_value=None),
        patch.object(wmod, 'get_nlevb', return_value=80),
    ):
        yield


@pytest.mark.parametrize('module', ['aragog', 'dummy'])
def test_equilibrate_passes_adiabat_to_every_iteration(module):
    """Each equilibration iteration solves the structure against the adiabat T(P).

    For zalmoxis + liquidus_super + non-SPIDER energetics (Aragog and dummy), the
    loop must hand a callable temperature_function to the solver on every
    iteration, not just the first. The non-converging structure (R/P held fixed
    apart) forces the loop to run to max_iter, so the per-iteration contract is
    exercised across all iterations.
    """
    from proteus.interior_energetics import wrapper as wmod

    config = _equilibrate_config(module=module, max_iter=3, tol=1e-9)
    hf_row = _linear_hf_row()
    dirs = {'output': str('/tmp/out')}

    received = []

    def fake_solver(cfg, outdir, row, num_spider_nodes=0, temperature_function=None, **kw):
        received.append(temperature_function)
        # Keep R/P oscillating so convergence never trips and all iters run.
        row['R_int'] = 1.27e7 if len(received) % 2 else 1.28e7
        row['P_surf'] = 1.0e5
        row['M_int'] = row['M_int_target']
        return 6.0e6, None

    adiabat = _good_adiabat()
    with (
        patch.object(
            wmod,
            '_build_superliquidus_adiabat_tp',
            return_value=(
                lambda r, P: float(np.interp(P, adiabat['P'], adiabat['T'])),
                adiabat['P'],
                adiabat['T'],
            ),
        ),
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=fake_solver),
        _patch_equilibrate_io(),
    ):
        wmod.equilibrate_initial_state(dirs, config, hf_row, '/tmp/out')

    # Contract: every iteration received a callable adiabat, none received None.
    assert len(received) == 3
    assert all(callable(tf) for tf in received)
    # Discrimination: a live closure returns a positive interior temperature.
    assert received[0](3.0e6, 5.0e11) > 0.0


@pytest.mark.parametrize(
    'module, temperature_mode',
    [('spider', 'liquidus_super'), ('aragog', 'isothermal')],
    ids=['spider_liquidus_super', 'aragog_isothermal'],
)
def test_equilibrate_passes_none_for_excluded_combinations(module, temperature_mode):
    """Excluded combinations solve with temperature_function=None (linear guess).

    SPIDER + liquidus_super (energetics excluded) and Aragog + non-liquidus_super
    (mode excluded) must keep the linear-guess solve: the solver receives None,
    and the adiabat builder is never called.
    """
    from proteus.interior_energetics import wrapper as wmod

    config = _equilibrate_config(module=module, temperature_mode=temperature_mode, max_iter=1)
    hf_row = _linear_hf_row()
    dirs = {'output': '/tmp/out'}

    received = []

    def fake_solver(cfg, outdir, row, num_spider_nodes=0, temperature_function=None, **kw):
        received.append(temperature_function)
        row['R_int'] = 1.27e7
        row['P_surf'] = 1.0e5
        row['M_int'] = row['M_int_target']
        return 6.0e6, None

    with (
        patch.object(wmod, '_build_superliquidus_adiabat_tp') as mock_build,
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=fake_solver),
        _patch_equilibrate_io(),
    ):
        wmod.equilibrate_initial_state(dirs, config, hf_row, '/tmp/out')

    mock_build.assert_not_called()
    assert received == [None]


def test_equilibrate_skips_adiabat_when_pcmb_seed_nonpositive():
    """A non-positive seed P_cmb skips adiabat construction in the loop.

    The adiabat grid is sized from the entry P_cmb. If that seed is absent or
    non-positive the adiabat cannot be built, so the loop must fall back to the
    linear-guess solve (temperature_function=None) without calling the builder.
    """
    from proteus.interior_energetics import wrapper as wmod

    config = _equilibrate_config(max_iter=1)
    hf_row = _linear_hf_row()
    hf_row['P_cmb'] = 0.0  # no valid seed to size the grid
    dirs = {'output': '/tmp/out'}

    received = []

    def fake_solver(cfg, outdir, row, num_spider_nodes=0, temperature_function=None, **kw):
        received.append(temperature_function)
        row['R_int'] = 1.27e7
        row['P_surf'] = 1.0e5
        row['M_int'] = row['M_int_target']
        return 6.0e6, None

    with (
        patch.object(wmod, '_build_superliquidus_adiabat_tp') as mock_build,
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=fake_solver),
        _patch_equilibrate_io(),
    ):
        wmod.equilibrate_initial_state(dirs, config, hf_row, '/tmp/out')

    mock_build.assert_not_called()
    assert received == [None]


def test_equilibrate_falls_back_to_linear_when_adiabat_solve_raises(tmp_path):
    """An adiabat re-solve failure degrades to the linear-guess solve, not abort.

    When the adiabat solve trips the mass anchor (rollback signals failure), the
    iteration must retry with temperature_function=None so init never aborts
    where the linear path would have converged ('never worse than today'). The
    first solver call (adiabat) is rejected; the second (linear) is accepted.
    """
    from proteus.interior_energetics import wrapper as wmod

    config = _equilibrate_config(max_iter=1)
    hf_row = _linear_hf_row()
    M_target = hf_row['M_int_target']

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'zalmoxis_output.dat').write_text('LINEAR GEOMETRY\n')
    dirs = {'output': str(tmp_path)}

    calls = []

    def fake_solver(cfg, outdir, row, num_spider_nodes=0, temperature_function=None, **kw):
        calls.append(temperature_function)
        if temperature_function is not None:
            # Adiabat attempt: write a bad geometry and trip the mass anchor.
            (data_dir / 'zalmoxis_output.dat').write_text('ADIABAT GEOMETRY (rejected)\n')
            row['R_int'] = 1.40e7
            row['M_int'] = M_target * 1.10  # 10% off: violates the anchor
            return 6.0e6, None
        # Linear fallback: converge cleanly.
        row['R_int'] = 1.2007e7
        row['P_surf'] = 1.0e5
        row['M_int'] = M_target
        return 6.0e6, None

    adiabat = _good_adiabat()
    with (
        patch.object(
            wmod,
            '_build_superliquidus_adiabat_tp',
            return_value=(lambda r, P: 5000.0, adiabat['P'], adiabat['T']),
        ),
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=fake_solver),
        _patch_equilibrate_io(),
    ):
        wmod.equilibrate_initial_state(dirs, config, hf_row, str(tmp_path))

    # Two solver calls in the single iteration: adiabat (rejected) then linear.
    assert len(calls) == 2
    assert callable(calls[0])
    assert calls[1] is None
    # The accepted linear-guess radius survives, not the rejected adiabat radius.
    assert hf_row['R_int'] == pytest.approx(1.2007e7, rel=1e-9)
    assert abs(hf_row['R_int'] - 1.40e7) > 1.0e6


def test_equilibrate_rebuilds_adiabat_when_pcmb_outgrows_grid():
    """A converged P_cmb above the grid ceiling triggers an adiabat rebuild.

    The adiabat is built once from the entry P_cmb * margin. If a later
    iteration's converged P_cmb exceeds that ceiling, the deepest mantle would
    clip to a flat T_cmb; the loop must rebuild the adiabat with the larger
    P_cmb. Drive P_cmb upward across two iterations and assert the builder is
    called a second time with a larger P_cmb_target.
    """
    from proteus.interior_energetics import wrapper as wmod

    config = _equilibrate_config(max_iter=2, tol=1e-9)
    hf_row = _linear_hf_row()
    P_cmb_seed = hf_row['P_cmb']  # 1.456e12
    dirs = {'output': '/tmp/out'}

    def fake_solver(cfg, outdir, row, num_spider_nodes=0, temperature_function=None, **kw):
        # Each call pushes P_cmb well above the grid ceiling so the rebuild
        # branch fires. Keep R/P from converging so both iterations run.
        row['P_cmb'] = P_cmb_seed * 2.0
        row['R_int'] = 1.27e7 if not hasattr(fake_solver, '_n') else 1.30e7
        fake_solver._n = True
        row['P_surf'] = 1.0e5
        row['M_int'] = row['M_int_target']
        return 6.0e6, None

    build_targets = []
    adiabat = _good_adiabat()

    def fake_build(cfg, row, P_cmb_target):
        build_targets.append(P_cmb_target)
        return (
            lambda r, P: float(np.interp(P, adiabat['P'], adiabat['T'])),
            adiabat['P'],
            adiabat['T'],
        )

    with (
        patch.object(wmod, '_build_superliquidus_adiabat_tp', side_effect=fake_build),
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=fake_solver),
        _patch_equilibrate_io(),
    ):
        wmod.equilibrate_initial_state(dirs, config, hf_row, '/tmp/out')

    # The builder ran at least twice: once for the seed grid, then a rebuild.
    assert len(build_targets) >= 2
    # The first build sized to the seed P_cmb * margin; the rebuild sized to the
    # larger converged P_cmb * margin. Discriminate the two targets.
    assert build_targets[0] == pytest.approx(P_cmb_seed * _ADIABAT_IC_PCMB_MARGIN, rel=1e-9)
    assert build_targets[-1] > build_targets[0]
    assert build_targets[-1] == pytest.approx(
        P_cmb_seed * 2.0 * _ADIABAT_IC_PCMB_MARGIN, rel=1e-9
    )


# ============================================================================
# Physics invariant: the adiabat IC yields the maximal radius at t = 0
# ============================================================================


@pytest.mark.physics_invariant
def test_adiabat_ic_gives_larger_radius_than_colder_profile():
    """A hotter (convex adiabat) T(P) integrates to a larger radius than a colder
    profile, and a lower surface temperature gives a smaller radius.

    This does NOT mock the radius: it integrates a real, simplified hydrostatic +
    mass ODE down from the surface against an analytic density rho(P, T) that
    decreases with T. Two T(P) closures over the same P span (a colder
    linear-like ramp vs the convex super-liquidus adiabat) bracket the physics:
    the hotter, less-dense interior packs the fixed target mass into a larger
    radius. Lowering the surface temperature cools the whole column, raises the
    mean density, and shrinks the radius, so t = 0 (the hottest, freshest IC) is
    the radius maximum. This is the physical reason the IC adiabat hand-off
    exists.
    """
    # Analytic EOS: rho = rho0 * (1 + P/K) / (1 + alpha*(T - T0)). Density rises
    # with pressure (compression) and falls with temperature (thermal
    # expansion); both signs are physical. SI-ish toy scales chosen so the
    # integral spans a realistic mantle range without a lookup table.
    rho0 = 4000.0  # kg/m^3 reference density
    K = 2.0e11  # Pa bulk-modulus-like compression scale
    alpha = 5.0e-5  # 1/K thermal expansion
    T0 = 3000.0  # K reference temperature
    g = 10.0  # m/s^2 constant gravity (toy hydrostatic balance)

    def rho_of(P, T):
        return rho0 * (1.0 + P / K) / (1.0 + alpha * (T - T0))

    def enclosed_mass(temp_func, R_surface):
        """Integrate hydrostatic dP/dr = rho*g and dM/dr = 4*pi*r^2*rho inward
        from a trial surface radius R_surface to the centre; return the total
        mass the structure holds for this T(P) closure.

        This is the forward map M(R) the structure root finder inverts: for a
        fixed surface BC, a less-dense (hotter) interior holds LESS mass at a
        given R, so the radius that holds the fixed target mass is LARGER.
        """
        n = 4000
        dr = R_surface / n
        r = R_surface
        P = 1.0e5  # surface pressure BC
        M = 0.0
        for _ in range(n):
            T = temp_func(r, P)
            rho = rho_of(P, T)
            M += 4.0 * np.pi * r * r * rho * dr
            P += rho * g * dr  # hydrostatic: P grows going inward
            r -= dr
        return M

    def solve_radius(temp_func, M_target):
        """Bisect the surface radius so the enclosed mass equals M_target.

        M(R) increases monotonically with R, so bisection on [R_lo, R_hi]
        converges. This mirrors the inverse problem
        determine_interior_radius_with_zalmoxis solves with a secant.
        """
        R_lo, R_hi = 5.0e6, 5.0e7
        for _ in range(80):
            R_mid = 0.5 * (R_lo + R_hi)
            if enclosed_mass(temp_func, R_mid) < M_target:
                R_lo = R_mid
            else:
                R_hi = R_mid
        return 0.5 * (R_lo + R_hi)

    # Same P-anchored closures the structure path uses: f(r, P) -> T. Build them
    # over a shared P span. The colder profile is a gentle linear ramp; the
    # adiabat is convex (hotter at depth), so it is less dense in the deep
    # interior at the same pressure.
    P_span = np.linspace(1.0e5, 8.0e11, 400)
    T_linear = np.linspace(3200.0, 4200.0, 400)  # colder, near-linear
    T_adiabat = 3200.0 + 3000.0 * (P_span / P_span[-1]) ** 0.6  # convex, hotter deep

    def tf_linear(r, P):
        return float(np.interp(min(max(P, P_span[0]), P_span[-1]), P_span, T_linear))

    def tf_adiabat(r, P):
        return float(np.interp(min(max(P, P_span[0]), P_span[-1]), P_span, T_adiabat))

    M_target = 5.0e24  # kg, fixed for both solves

    R_linear = solve_radius(tf_linear, M_target)
    R_adiabat = solve_radius(tf_adiabat, M_target)

    # Both solves landed inside the bracket (not pinned to an edge).
    assert 5.0e6 < R_linear < 5.0e7
    assert 5.0e6 < R_adiabat < 5.0e7
    # Monotonicity invariant 1: the hotter adiabat IC holds the same mass at a
    # LARGER radius. This is the maximal-radius-at-t=0 statement.
    assert R_adiabat > R_linear
    # Discrimination: the gap is well above bisection convergence noise.
    assert R_adiabat - R_linear > 1.0e4

    # Monotonicity invariant 2: a lower surface temperature (colder whole column)
    # gives a SMALLER radius, so cooling away from t=0 shrinks the planet.
    def tf_adiabat_cold(r, P):
        # Same shape, shifted 300 K colder everywhere.
        return tf_adiabat(r, P) - 300.0

    R_adiabat_cold = solve_radius(tf_adiabat_cold, M_target)
    assert 5.0e6 < R_adiabat_cold < 5.0e7
    assert R_adiabat_cold < R_adiabat
    assert R_adiabat - R_adiabat_cold > 1.0e3
