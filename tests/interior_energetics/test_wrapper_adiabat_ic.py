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
- ``_sample_adiabat_temperature_arrays``: samples the P-indexed adiabat closure
  onto the previous structure's mantle (r, P) grid so the solve carries
  r-indexed ``temperature_arrays`` alongside the callable and the Zalmoxis JAX
  inner path can consume the adiabat on JAX-viable EOS layouts (issue #719);
  returns None (callable-only degradation) on any missing or malformed
  previous structure.
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
- ``update_structure_from_interior``: the dynamic-evolution structure re-solve
  enforces a monotonic-radius guard under the super-liquidus adiabat IC. A
  re-solve that would raise R_int above the running minimum is rejected as a
  cross-table representation artifact (the IC adiabat is a P-T table; the evolved
  interior temperature is a P-S representation), the previous structure is
  retained, and the consecutive-failure counter is left untouched; a genuine
  contraction is accepted, and the guard is a no-op for non-liquidus_super runs.

Invariants exercised:
- Positivity / boundedness: the adiabat T(P) is finite and positive; the NaN
  guard rejects a non-finite profile.
- Pinned numeric value with discrimination guard: the T(P) closure interpolates
  the adiabat and clips out-of-range pressures to the table edges.
- Monotonicity: integrating a real hydrostatic + mass ODE against the convex
  adiabat yields a larger radius than against a colder linear-like T(P), and a
  lower surface temperature yields a smaller radius (t = 0 is the maximum). The
  recorded R_int never rises above its running minimum once the super-liquidus
  IC is active, because a fully molten start can only cool and crystallise.
- The adiabat-anchored IC solve relaxes R_int outward (lower mean density) and
  rolls back to the linear-guess radius (and on-disk geometry) on a mass-anchor
  violation.

Testing standards and documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t
from proteus.interior_energetics.wrapper import (
    _ADIABAT_IC_PCMB_MARGIN,
    _MONOTONIC_RINT_REL_TOL,
    _build_superliquidus_adiabat_tp,
    _resolve_adiabatic_ic_structure,
    _sample_adiabat_temperature_arrays,
    _solve_structure_with_adiabat_or_rollback,
    _use_superliquidus_adiabat_ic,
    update_structure_from_interior,
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


# ============================================================================
# _sample_adiabat_temperature_arrays: adiabat sampling for the JAX arrays path
# ============================================================================


def _write_structure_file(tmp_path, n=40, r0=6.0e6, r1=1.2007e7, p0=1.4e12, p1=1.0e5):
    """Write a numeric zalmoxis_output.dat (mantle rows; columns r, P, rho, g, T).

    Radii ascend from the CMB to the surface and pressure descends, matching
    the file the previous structure solve writes. Returns the (r, P) columns.
    """
    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    r = np.linspace(r0, r1, n)
    P = np.linspace(p0, p1, n)
    rho = np.linspace(5500.0, 3300.0, n)
    g = np.full(n, 10.0)
    T = np.linspace(6200.0, 4000.0, n)
    np.savetxt(
        data_dir / 'zalmoxis_output.dat',
        np.column_stack([r, P, rho, g, T]),
    )
    return r, P


def _adiabat_closure():
    """P-indexed adiabat closure over the _good_adiabat table (r is ignored)."""
    adiabat = _good_adiabat()
    P_arr, T_arr = adiabat['P'], adiabat['T']

    def tf(r, P):
        P_clipped = min(max(float(P), float(P_arr[0])), float(P_arr[-1]))
        return float(np.interp(P_clipped, P_arr, T_arr))

    return tf


def test_sample_adiabat_arrays_matches_closure_on_previous_grid(tmp_path):
    """The sampled arrays are the closure evaluated on the previous (r, P) grid.

    The r axis is the previous structure's mantle radii (strictly increasing,
    spanning CMB to surface, the domain the outer radius search perturbs
    around; radii beyond it are endpoint-clamped by the JAX RHS), and every
    T value equals the closure at that row's (r, P). Because the closure is
    pressure-indexed and the file's P decreases outward, T decreases
    monotonically toward the surface.
    """
    r_file, p_file = _write_structure_file(tmp_path)
    tf = _adiabat_closure()

    sampled = _sample_adiabat_temperature_arrays(str(tmp_path), tf)

    assert sampled is not None
    r_arr, t_arr = sampled
    np.testing.assert_allclose(r_arr, r_file, rtol=0, atol=0)
    assert np.all(np.diff(r_arr) > 0.0)
    # Domain coverage: the sampling spans the previous mantle exactly.
    assert r_arr[0] == pytest.approx(6.0e6, rel=1e-12)
    assert r_arr[-1] == pytest.approx(1.2007e7, rel=1e-12)
    # T pins to the closure row by row, and stays positive and finite.
    expected = np.array([tf(r, P) for r, P in zip(r_file, p_file)])
    np.testing.assert_allclose(t_arr, expected, rtol=1e-12)
    assert np.all(np.isfinite(t_arr)) and np.all(t_arr > 0.0)
    assert np.all(np.diff(t_arr) <= 0.0)
    # Discrimination: sampling T at a fixed pressure (a plausible wrong
    # implementation) would be constant; the true profile spans > 1000 K.
    assert t_arr[0] - t_arr[-1] > 1000.0


def test_sample_adiabat_arrays_returns_none_on_bad_input(tmp_path):
    """Any missing or malformed previous structure degrades to None.

    None routes the caller to the callable-only solve, the pre-#719 behavior,
    so every guard here is a safety valve rather than an error path: missing
    file, non-numeric content, non-monotone radii, non-finite pressure, and a
    closure returning non-finite temperatures.
    """
    tf = _adiabat_closure()

    # Missing file.
    assert _sample_adiabat_temperature_arrays(str(tmp_path), tf) is None

    # Non-numeric content (the rollback tests write such placeholders).
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    out_path = data_dir / 'zalmoxis_output.dat'
    out_path.write_text('LINEAR GEOMETRY placeholder\n')
    assert _sample_adiabat_temperature_arrays(str(tmp_path), tf) is None

    # Single row: no interpolable grid.
    np.savetxt(out_path, np.array([[6.0e6, 1.4e12, 5500.0, 10.0, 6200.0]]))
    assert _sample_adiabat_temperature_arrays(str(tmp_path), tf) is None

    # Non-monotone radii.
    _write_structure_file(tmp_path)
    data = np.loadtxt(out_path)
    data[5, 0] = data[3, 0]  # duplicate radius breaks strict monotonicity
    np.savetxt(out_path, data)
    assert _sample_adiabat_temperature_arrays(str(tmp_path), tf) is None

    # Non-finite pressure.
    _write_structure_file(tmp_path)
    data = np.loadtxt(out_path)
    data[5, 1] = np.nan
    np.savetxt(out_path, data)
    assert _sample_adiabat_temperature_arrays(str(tmp_path), tf) is None

    # Closure returning non-finite temperatures.
    _write_structure_file(tmp_path)
    assert _sample_adiabat_temperature_arrays(str(tmp_path), lambda r, P: float('nan')) is None

    # Closure raising outright (e.g. a P outside its tabulated span with a
    # strict interpolator): degrade to the callable-only call, never raise.
    def _raising_closure(r, P):
        raise ValueError('P outside adiabat table span')

    _write_structure_file(tmp_path)
    assert _sample_adiabat_temperature_arrays(str(tmp_path), _raising_closure) is None


def test_adiabat_solve_hands_sampled_arrays_alongside_callable(tmp_path):
    """The adiabat solve passes both the closure and its r-indexed sampling.

    With a numeric previous structure on disk, the solver receives the same
    callable object plus temperature_arrays sampled from it on the previous
    (r, P) grid, so zalmoxis_solver's temperature-source dispatch decides the
    JAX-vs-numpy question exactly as it does for evolved re-solves (#719).
    """
    r_file, p_file = _write_structure_file(tmp_path)
    tf = _adiabat_closure()
    config = _config()
    hf_row = _linear_hf_row()
    M_target = hf_row['M_int_target']

    received = {}

    def fake_solver(
        cfg,
        outdir,
        row,
        num_spider_nodes=0,
        temperature_function=None,
        temperature_arrays=None,
        **kw,
    ):
        received['tf'] = temperature_function
        received['arrays'] = temperature_arrays
        row['R_int'] = 1.2746e7
        row['M_int'] = M_target
        return 6.0e6, str(tmp_path / 'adiabat_mesh.dat')

    with patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=fake_solver):
        _mesh, ok = _solve_structure_with_adiabat_or_rollback(
            config,
            str(tmp_path),
            hf_row,
            num_spider_nodes=0,
            temperature_function=tf,
        )

    assert ok is True
    # The callable still reaches the solver unmodified (non-JAX-viable
    # configurations keep consuming it on the numpy path).
    assert received['tf'] is tf
    # The arrays ride alongside: previous r grid, closure-sampled T.
    r_arr, t_arr = received['arrays']
    np.testing.assert_allclose(r_arr, r_file, rtol=0, atol=0)
    expected = np.array([tf(r, P) for r, P in zip(r_file, p_file)])
    np.testing.assert_allclose(t_arr, expected, rtol=1e-12)


def test_adiabat_solve_degrades_to_callable_only_without_previous_structure(tmp_path):
    """Without a usable previous structure the solve carries the callable alone.

    temperature_arrays arrives as None, which reproduces the pre-#719 call
    exactly: the solver-side dispatch sees no arrays and keeps the callable on
    the numpy path.
    """
    tf = _adiabat_closure()
    config = _config()
    hf_row = _linear_hf_row()
    M_target = hf_row['M_int_target']

    received = {}

    def fake_solver(
        cfg,
        outdir,
        row,
        num_spider_nodes=0,
        temperature_function=None,
        temperature_arrays=None,
        **kw,
    ):
        received['tf'] = temperature_function
        received['arrays'] = temperature_arrays
        row['R_int'] = 1.2746e7
        row['M_int'] = M_target
        return 6.0e6, str(tmp_path / 'adiabat_mesh.dat')

    with patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=fake_solver):
        _mesh, ok = _solve_structure_with_adiabat_or_rollback(
            config,
            str(tmp_path),
            hf_row,
            num_spider_nodes=0,
            temperature_function=tf,
        )

    assert ok is True
    assert received['tf'] is tf
    assert received['arrays'] is None


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


# ============================================================================
# update_structure_from_interior: monotonic-radius guard (super-liquidus IC)
# ============================================================================


def _monotonic_config(
    module: str = 'aragog',
    temperature_mode: str = 'liquidus_super',
    struct_module: str = 'zalmoxis',
):
    """Concrete config namespace for the dynamic structure-update guard tests.

    A concrete SimpleNamespace (not a MagicMock) is used so the gate's three
    conjuncts and the trigger thresholds read real values; a MagicMock would
    auto-create attributes and silently satisfy the gate. The trigger knobs are
    set so a single ceiling crossing fires the re-solve: update_interval is short
    and every change-based threshold is left high so the ceiling is the trigger.
    """
    return SimpleNamespace(
        interior_energetics=SimpleNamespace(module=module, num_levels=50),
        planet=SimpleNamespace(temperature_mode=temperature_mode),
        interior_struct=SimpleNamespace(
            module=struct_module,
            zalmoxis=SimpleNamespace(
                mantle_eos='PALEOS:MgSiO3',
                update_interval=1000.0,
                update_min_interval=100.0,
                mesh_convergence_interval=10.0,
                update_stale_ceiling=0.0,
                update_dphi_abs=10.0,
                update_dtmagma_frac=10.0,
                update_dw_comp_abs=10.0,
                mesh_max_shift=0.05,
                global_miscibility=False,
            ),
        ),
    )


def _monotonic_interior_o(n_stag=49):
    """Real Interior_t with a mantle T(r) profile for the re-solve hand-off.

    A real Interior_t (not a MagicMock) so zalmoxis_fail_count is a true integer
    the guard must leave untouched and last_successful_struct_time is settable.
    The radius/temp arrays feed the temperature_function the wrapper builds.
    """
    interior_o = Interior_t(n_stag + 1)
    interior_o.radius = np.linspace(6.371e6, 3.504e6, n_stag + 1)
    interior_o.temp = np.full(n_stag, 3000.0)
    interior_o.zalmoxis_fail_count = 0
    interior_o.last_successful_struct_time = float('-inf')
    return interior_o


def _monotonic_hf_row(R_int_prev, M_target=5.972e24):
    """Helpfile row entering a ceiling-triggered re-solve at R_int_prev.

    Time is past update_interval so the ceiling fires. M_int is mass-anchored to
    M_target so the mass-anchor check passes and control reaches the radius guard.
    """
    return {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': R_int_prev,
        'R_core': 3.504e6,
        'M_int': M_target,
        'M_int_target': M_target,
        'M_core': 1.9e24,
        'M_mantle': M_target - 1.9e24,
        'P_surf': 1.0e5,
        'P_center': 1.6e12,
        'P_cmb': 1.4e12,
        'rho_avg': 8.0e3,
        # Surface gravity consistent with R_int_prev, so a restore reverting it
        # in lockstep with R_int is checkable against G*M/R^2.
        'gravity': 6.674e-11 * M_target / R_int_prev**2,
    }


def _run_monotonic_resolve(config, dirs, hf_row, interior_o, R_int_returned, M_target):
    """Drive one ceiling-triggered update_structure_from_interior re-solve.

    The mocked zalmoxis_solver writes the on-disk geometry and sets the returned
    R_int / mass anchor, mimicking the real solver's writeback. blend_mesh_files
    is mocked to a no-op shift so the success path's mesh handling does not touch
    the toy mesh files. Returns the (last_struct_time, last_Tmagma, last_Phi)
    sentinel tuple the function returns.
    """

    def _solver(cfg, outdir, row, num_spider_nodes=0, temperature_function=None, **kw):
        # The real solver overwrites zalmoxis_output.dat before returning, and
        # writes gravity / P_cmb derived from the new (here: up-step) radius.
        out_path = os.path.join(outdir, 'data', 'zalmoxis_output.dat')
        with open(out_path, 'w') as f:
            f.write('RESOLVE GEOMETRY: R_int=%.6e m\n' % R_int_returned)
        row['R_int'] = R_int_returned
        row['M_int'] = M_target  # mass-anchored so the anchor check passes
        row['M_int_target'] = M_target
        row['gravity'] = 6.674e-11 * M_target / R_int_returned**2
        row['P_cmb'] = 1.5e12
        return 3.504e6, dirs['spider_mesh']

    with (
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=_solver),
        patch('proteus.interior_energetics.spider.blend_mesh_files', return_value=0.0),
    ):
        return update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )


def _monotonic_dirs(tmp_path, prev_content):
    """Lay out data/zalmoxis_output.dat + .prev and a mesh + .prev on disk.

    The .prev snapshots hold the previous (running-minimum) geometry the guard
    restores when it rejects an up-step. Returns the dirs dict the wrapper reads.
    """
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    out_path = data_dir / 'zalmoxis_output.dat'
    out_path.write_text(prev_content)
    # Pre-seed the .prev snapshot the wrapper will also (re)write before the
    # solve; the guard restores from it on a rejected up-step.
    (data_dir / 'zalmoxis_output.dat.prev').write_text(prev_content)
    mesh = tmp_path / 'mesh.dat'
    mesh.write_text('prev mesh\n')
    mesh_prev = tmp_path / 'mesh.dat.prev'
    mesh_prev.write_text('prev mesh\n')
    return {
        'output': str(tmp_path),
        'spider': str(tmp_path / 'spider'),
        'spider_mesh': str(mesh),
        'spider_mesh_prev': str(mesh_prev),
        'mesh_shift_active': False,
        'mesh_convergence_steps': 0,
    }, out_path


@pytest.mark.physics_invariant
def test_monotonic_radius_guard_rejects_representation_up_step(tmp_path):
    """A re-solve that raises R_int above the running minimum is a clean no-op.

    Physical invariant: a fully molten super-liquidus start can only cool and
    crystallise, so the solid-body R_int is non-increasing. A dynamic re-solve
    against the evolved P-S interior temperature can land slightly ABOVE the IC
    P-T-adiabat radius purely as a cross-table representation artifact. The guard
    must reject it: the recorded R_int stays at the previous (running-minimum)
    value, the function returns the unchanged sentinel tuple, and the
    consecutive-failure counter does NOT increment (a rejected up-step is not a
    solver failure and must not consume the abort budget). The on-disk geometry
    is restored from the .prev snapshot, not left at the rejected up-step file.
    """
    R_int_prev = 1.2007e7
    R_int_up = R_int_prev * (1.0 + 1.0e-4)  # well above the 1e-9 guard tolerance
    M_target = 5.972e24
    config = _monotonic_config(module='aragog', temperature_mode='liquidus_super')
    interior_o = _monotonic_interior_o()
    prev_content = 'PREV GEOMETRY: R_int=%.6e m\n' % R_int_prev
    dirs, out_path = _monotonic_dirs(tmp_path, prev_content)
    hf_row = _monotonic_hf_row(R_int_prev, M_target=M_target)

    sentinel = _run_monotonic_resolve(
        config, dirs, hf_row, interior_o, R_int_returned=R_int_up, M_target=M_target
    )

    # No-op contract 1: the recorded radius is the previous running minimum,
    # NOT the rejected up-step. Pin against R_int_prev and discriminate against
    # the up-step value, which is ~1.2 km larger (far above any rounding).
    assert hf_row['R_int'] == pytest.approx(R_int_prev, rel=1e-12)
    assert abs(hf_row['R_int'] - R_int_up) > 1.0e3
    # No-op contract 2: the trigger clock and sentinels ADVANCE to the current
    # time/state (the retained structure is the current best, "checked at
    # current_time"), so a persistent artifact re-fires only at the normal
    # cadence, not an expensive re-solve every timestep. T_magma/Phi are
    # unchanged by the rejected solve, so they equal their entry values.
    assert sentinel[0] == pytest.approx(1100.0, rel=1e-12)
    assert sentinel[1] == pytest.approx(3000.0, rel=1e-12)
    assert sentinel[2] == pytest.approx(0.8, rel=1e-12)
    # The stale-aware ceiling anchor advances too, so it does not immediately
    # re-arm and force a wasteful re-solve next step.
    assert interior_o.last_successful_struct_time == pytest.approx(1100.0, rel=1e-12)
    # No-op contract 3 (the load-bearing one): a rejected up-step is NOT a solver
    # failure, so the consecutive-failure counter stays at zero and the run's
    # abort budget is untouched.
    assert interior_o.zalmoxis_fail_count == 0
    # The retained structure is consistent, so the stale flag is not raised
    # (which would otherwise force Aragog onto a frozen-mesh recovery path).
    assert interior_o.structure_stale is False
    assert '_structure_stale' not in hf_row, 'flag must not live on hf_row'
    # Consistency: gravity is reverted in lockstep with R_int, not left at the
    # rejected up-step value. After the no-op gravity must match G*M/R_int**2 for
    # the restored R_int (the bug: gravity left ~2e-4 high from the up-step).
    g_expected = 6.674e-11 * M_target / hf_row['R_int'] ** 2
    assert hf_row['gravity'] == pytest.approx(g_expected, rel=1e-9)
    g_upstep = 6.674e-11 * M_target / R_int_up**2
    assert abs(hf_row['gravity'] - g_upstep) > 1e-5 * g_expected
    # On-disk contract: zalmoxis_output.dat holds the PREV geometry again, not
    # the rejected up-step file the solver wrote. This is the discriminating
    # check that an hf_row-only no-op (the bug) would fail.
    restored = out_path.read_text()
    assert restored == prev_content
    assert 'RESOLVE GEOMETRY' not in restored


@pytest.mark.physics_invariant
def test_monotonic_radius_guard_accepts_genuine_contraction(tmp_path):
    """A re-solve that LOWERS R_int (real cooling) is accepted, sentinels advance.

    Discriminating negative for the guard: genuine crystallisation contracts the
    planet, so a re-solve below the running minimum is the physically correct
    direction and must be accepted. The recorded R_int updates downward, the
    returned sentinel advances to the current time/state, and the on-disk
    geometry is the new (contracted) file, not the .prev snapshot. This separates
    'rejects up-steps' from 'rejects every re-solve'.
    """
    R_int_prev = 1.2007e7
    R_int_down = R_int_prev * (1.0 - 5.0e-4)  # genuine contraction
    M_target = 5.972e24
    config = _monotonic_config(module='aragog', temperature_mode='liquidus_super')
    interior_o = _monotonic_interior_o()
    dirs, out_path = _monotonic_dirs(tmp_path, 'PREV GEOMETRY\n')
    hf_row = _monotonic_hf_row(R_int_prev, M_target=M_target)

    sentinel = _run_monotonic_resolve(
        config, dirs, hf_row, interior_o, R_int_returned=R_int_down, M_target=M_target
    )

    # Acceptance contract: R_int moved DOWN to the contracted value, not held at
    # the previous minimum. Pin the new value and discriminate against a no-op.
    assert hf_row['R_int'] == pytest.approx(R_int_down, rel=1e-9)
    assert hf_row['R_int'] < R_int_prev
    assert R_int_prev - hf_row['R_int'] > 1.0e3
    # The sentinel advanced to the current time (1100.0) and state: a contraction
    # is a successful re-solve, so the trigger clock and anchors move forward.
    assert sentinel[0] == pytest.approx(1100.0, rel=1e-12)
    # On-disk contract: the contracted geometry survives (not restored to .prev).
    assert 'RESOLVE GEOMETRY' in out_path.read_text()
    # A success also resets the failure streak (it was already zero here).
    assert interior_o.zalmoxis_fail_count == 0


@pytest.mark.physics_invariant
def test_monotonic_radius_guard_inactive_for_non_liquidus_super(tmp_path):
    """The same up-step is accepted for a non-liquidus_super config (gate off).

    Discriminating negative for the gating: regimes that are not the molten
    super-liquidus IC can re-inflate for real (tidal reheating, volatile
    re-dissolution), so the monotonic clamp must NOT fire there. With an
    isothermal temperature mode the gate is off, so an identical R_int up-step is
    accepted: R_int rises, the sentinel advances, and the failure counter stays
    at zero (no regression for other regimes).
    """
    R_int_prev = 1.2007e7
    R_int_up = R_int_prev * (1.0 + 1.0e-4)  # same up-step as the rejected case
    M_target = 5.972e24
    # Gate off via the temperature mode (isothermal), everything else identical.
    config = _monotonic_config(module='aragog', temperature_mode='isothermal')
    assert _use_superliquidus_adiabat_ic(config) is False
    interior_o = _monotonic_interior_o()
    dirs, out_path = _monotonic_dirs(tmp_path, 'PREV GEOMETRY\n')
    hf_row = _monotonic_hf_row(R_int_prev, M_target=M_target)

    sentinel = _run_monotonic_resolve(
        config, dirs, hf_row, interior_o, R_int_returned=R_int_up, M_target=M_target
    )

    # No clamp: the up-step is recorded because re-inflation is physical here.
    assert hf_row['R_int'] == pytest.approx(R_int_up, rel=1e-9)
    assert hf_row['R_int'] > R_int_prev
    # The sentinel advanced (accepted re-solve), so this is not the no-op path.
    assert sentinel[0] == pytest.approx(1100.0, rel=1e-12)
    # The new geometry survives on disk; the guard did not restore the snapshot.
    assert 'RESOLVE GEOMETRY' in out_path.read_text()
    assert interior_o.zalmoxis_fail_count == 0


def test_monotonic_radius_tolerance_is_tiny_and_positive():
    """The guard tolerance is a tiny positive relative band (floating-point noise).

    Boundedness of the tuning constant: it must be strictly positive so an exact
    re-solve that reproduces the previous radius (rel change 0) is accepted, and
    far below any physical contraction step so a real representation up-step
    (~1e-4 in the production trace) is rejected rather than absorbed.
    """
    assert _MONOTONIC_RINT_REL_TOL > 0.0
    # Discrimination: the band is well below the ~1e-4 representation up-step the
    # guard must catch, and not, e.g., a 1e-2 typo that would absorb real rises.
    assert _MONOTONIC_RINT_REL_TOL < 1.0e-6
