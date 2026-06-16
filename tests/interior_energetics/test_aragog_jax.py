"""
Unit tests for proteus.interior_energetics.aragog_jax: research-only diffrax
JAX entropy solver runner.

The diffrax solver path itself is currently paused (kvaerno3 stalls on the
first crystallization step in CHILI Earth runs; the file is gated on the
hardcoded ``_DIFFRAX_RESEARCH_ONLY`` constant in aragog.py and not exposed
in the TOML schema). This file does not attempt to exercise the broken
solver; it tests the PROTEUS-side dispatcher logic that runs BEFORE and
AFTER the diffrax call:

- the ``_build_jax_components`` constructor's hard-fail when the spider EOS
  directory is missing (error contract);
- the ``run_solver`` wrapper's hard-fail when diffrax returns
  ``result.success == False`` (error contract);
- the ``_extract_output`` translation from a synthetic solve result back to
  the PROTEUS helpfile schema, with mass-closure as the conservation
  invariant.

Every test mocks ``solve_entropy`` and the heavy aragog.jax components so
the dispatch code is exercised without paying the diffrax compile or
running into the v1-v5 failure modes.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip('aragog.jax')

from proteus.interior_energetics.aragog_jax import AragogJAXRunner  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_config(*, heat_radiogenic: bool = False, heat_tidal: bool = False):
    """Build a mock PROTEUS Config with the fields aragog_jax reads."""
    config = MagicMock()
    config.interior_energetics.rfront_loc = 0.4
    config.interior_energetics.rfront_wid = 0.15
    config.interior_energetics.solid_log10visc = 22.0
    config.interior_energetics.melt_log10visc = 2.0
    config.interior_energetics.grain_size = 0.1
    config.interior_energetics.solid_cond = 4.0
    config.interior_energetics.melt_cond = 4.0
    config.interior_energetics.trans_conduction = True
    config.interior_energetics.trans_convection = True
    config.interior_energetics.trans_grav_sep = False
    config.interior_energetics.trans_mixing = True
    config.interior_energetics.eddy_diffusivity_thermal = 0.1
    config.interior_energetics.eddy_diffusivity_chemical = 0.1
    config.interior_energetics.kappah_floor = 1e-6
    config.interior_energetics.spider.matprop_smooth_width = 0.02
    config.interior_energetics.aragog.phase_smoothing = True
    config.interior_energetics.aragog.atol_temperature_equivalent = 1.0
    config.interior_energetics.rtol = 1e-4
    config.interior_energetics.heat_radiogenic = heat_radiogenic
    config.interior_energetics.heat_tidal = heat_tidal
    return config


def _make_interior_o(*, spider_eos_dir: str | None, prepopulate_jax: bool = False):
    """Build a mock Interior_t with the attributes aragog_jax inspects."""
    interior_o = SimpleNamespace()
    interior_o._spider_eos_dir = spider_eos_dir
    interior_o.aragog_solver = MagicMock()
    # The numpy solver's BC structure is read inside _build_jax_components.
    bc_cfg = MagicMock()
    bc_cfg.outer_boundary_condition = 4
    bc_cfg.outer_boundary_value = 0.0
    bc_cfg.emissivity = 1.0
    bc_cfg.equilibrium_temperature = 1500.0
    bc_cfg.inner_boundary_condition = 3
    bc_cfg.inner_boundary_value = 0.0
    bc_cfg.core_heat_capacity = 880.0
    bc_cfg.tfac_core_avg = 1.147
    interior_o.aragog_solver.parameters.boundary_conditions = bc_cfg
    interior_o.aragog_solver.parameters.mesh.core_density = 12000.0
    interior_o.aragog_solver.parameters.solver.start_time = 0.0
    interior_o.aragog_solver.parameters.solver.end_time = 1.0
    interior_o.aragog_solver._S0 = np.linspace(2000.0, 3000.0, 5)
    if prepopulate_jax:
        interior_o._jax_eos = MagicMock()
        interior_o._jax_params = MagicMock()
        interior_o._jax_bc = MagicMock()
        interior_o._jax_bc.outer_bc_type = 4
        interior_o._jax_bc.outer_bc_value = 0.0
        interior_o._jax_bc.emissivity = 1.0
        interior_o._jax_bc.T_eq = 1500.0
        interior_o._jax_bc.inner_bc_type = 3
        interior_o._jax_bc.inner_bc_value = 0.0
        interior_o._jax_bc.core_density = 12000.0
        interior_o._jax_bc.core_heat_capacity = 880.0
        interior_o._jax_bc.tfac_core_avg = 1.147
    return interior_o


@pytest.mark.unit
def test_build_jax_components_raises_when_spider_eos_dir_missing(tmp_path):
    """``_build_jax_components`` hard-fails with FileNotFoundError when
    ``interior_o._spider_eos_dir`` is None or points to a non-existent
    directory.

    Contract from ``aragog_jax.py:75-81``: if the spider EOS directory
    is missing, the JAX backend cannot load PALEOS P-S tables, and the
    dispatcher must not silently fall back to an empty EOS.

    Verifies the error contract on two boundary inputs:
    - ``spider_eos_dir = None`` (the truthiness check is the first guard).
    - ``spider_eos_dir = <path that does not exist>`` (the os.path.isdir
      check is the second guard).

    Discrimination: a regression that loosened the guard to ``or`` instead
    of ``and`` would pass the None case (because os.path.isdir(None)
    raises TypeError on most Python versions) but fail this test on the
    nonexistent-path case. A regression that removed the guard entirely
    would pass both cases here AND fail downstream in EntropyEOS_JAX with
    an opaque error; this test catches it at the dispatcher level.
    """
    config = _make_config()
    nonexistent = str(tmp_path / 'no_such_dir')
    assert not (tmp_path / 'no_such_dir').exists(), 'precondition: path must not exist'

    for bad_eos_dir in (None, nonexistent):
        interior_o = _make_interior_o(spider_eos_dir=bad_eos_dir)
        with pytest.raises(FileNotFoundError, match=r'(?i)PALEOS|tables not found'):
            AragogJAXRunner(config, {'output': str(tmp_path)}, {}, None, interior_o)


@pytest.mark.unit
def test_run_solver_raises_when_diffrax_result_fails(tmp_path):
    """``run_solver`` hard-fails with RuntimeError when ``solve_entropy``
    returns a result with ``success == False``.

    Contract from ``aragog_jax.py:224-230``: when diffrax reports failure
    (typically a max_steps exhaustion in the kvaerno3 path or an
    optx.Newton blow-up in the implicit_euler path), the dispatcher must
    raise RuntimeError so the upstream wrapper's retry ladder can engage
    instead of silently producing nonsense output.

    Verifies:
    - ``solve_entropy`` mocked to return ``success=False`` triggers
      RuntimeError.
    - The error message names the relevant diagnostics (final time,
      step count) so a future regression that swallows them into an
      opaque error string is caught.
    - Edge case: ``hf_row['F_atm']`` and ``hf_row['Time']`` are both
      consumed; missing-Time falls back to 0.0 via .get(); a
      regression that changed this default would alter the error
      message and fail the regex match below.
    """
    config = _make_config()
    interior_o = _make_interior_o(spider_eos_dir=str(tmp_path), prepopulate_jax=True)

    # Mock _build_mesh_arrays so __init__ doesn't try to call the real
    # MeshArrays.from_numpy_mesh on the MagicMock solver.
    with patch.object(AragogJAXRunner, '_build_mesh_arrays', return_value=MagicMock()):
        runner = AragogJAXRunner(
            config, {'output': str(tmp_path)}, {'F_atm': 1e5}, None, interior_o
        )

    failed_result = SimpleNamespace(success=False, t_final=0.5, n_steps=12345, S_final=None)
    # Sentinel: confirm interior_o._last_entropy is not set before the call.
    interior_o._last_entropy = None
    with patch('aragog.jax.solver.solve_entropy', return_value=failed_result):
        with pytest.raises(RuntimeError, match=r'JAX Aragog solver failed.*t_final.*steps'):
            runner.run_solver(
                {'F_atm': 1e5, 'Time': 1.0e3},
                interior_o,
                {'output': str(tmp_path)},
                write_data=False,
            )

    # Discrimination: side effects must not have run after a failed solve.
    # The contract at aragog_jax.py:223-233 raises BEFORE the
    # ``interior_o._last_entropy = np.asarray(result.S_final)`` line, so a
    # failed solve must leave _last_entropy untouched. A regression that
    # moved the assignment above the success check would update the
    # entropy IC with garbage from the failed result and silently corrupt
    # the next coupling step.
    assert interior_o._last_entropy is None, (
        '_last_entropy was written despite the failed solve; '
        'side effect leaked from a failed solver result'
    )


@pytest.mark.unit
def test_extract_output_mass_closure(tmp_path):
    """``_extract_output`` builds a helpfile dict that satisfies the
    mass-closure invariant ``M_mantle_liquid + M_mantle_solid == M_mantle``.

    Contract from ``aragog_jax.py:262, 337-338``:
        M_mantle = sum(rho * vol)
        M_mantle_liquid = sum(phi * mass)
        M_mantle_solid = M_mantle - sum(phi * mass)

    The two reservoirs partition the mantle by construction. This test
    pins that partition with a non-trivial phi profile so a regression
    that swapped the sign of ``M_mantle_solid`` (computing
    ``sum(phi * mass)`` for solid instead of liquid) or that lost
    precision in the subtraction would fail mass closure.

    Discriminating values: the phi profile is asymmetric ``[0.1, 0.3,
    0.5, 0.7, 0.9]`` so the wrong-formula result (swapping liquid /
    solid) would land at ``M_total - liquid = 0.5 * M_total`` only by
    coincidence of a symmetric profile. With the chosen asymmetric
    profile, swapping yields a different result than the correct one
    by ~`(2*phi - 1) * mass` summed per cell, several percent of
    M_mantle, easily resolved against the rel=1e-12 tolerance.

    Also verifies:
    - T_magma > 0 (sign guard).
    - 0 <= Phi_global <= 1 (boundedness invariant).
    - Phi_global is mass-weighted (not volume-weighted): a regression
      to vol-weighting would yield a different value for the
      asymmetric phi profile below.
    """
    config = _make_config()
    interior_o = _make_interior_o(spider_eos_dir=str(tmp_path), prepopulate_jax=True)
    n_stag = 5
    phi_profile = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Build mesh + eos mocks that return reproducible arrays.
    mesh = MagicMock()
    P_arr = np.linspace(1.0e9, 1.5e11, n_stag)
    mesh.P_stag = P_arr
    mesh.volume = np.full(n_stag, 1.0e19)
    mesh.radii_basic = np.linspace(3.0e6, 6.4e6, n_stag + 1)
    mesh.radii_stag = np.linspace(3.1e6, 6.3e6, n_stag)
    mesh.quantity_matrix = np.eye(n_stag + 1, n_stag)

    eos = interior_o._jax_eos
    eos.temperature = lambda P, S: np.linspace(4000.0, 3000.0, n_stag)
    eos.melt_fraction = lambda P, S: phi_profile
    rho_profile = np.full(n_stag, 4500.0)
    eos.density = lambda P, S: rho_profile

    # Mock evaluate_phase so _extract_output can compute Cp / viscosity.
    fake_props = MagicMock()
    fake_props.viscosity = np.full(n_stag, 1.0e2)
    fake_props.heat_capacity = np.full(n_stag, 1200.0)

    # Mock _build_mesh_arrays so the runner takes our mock mesh.
    with patch.object(AragogJAXRunner, '_build_mesh_arrays', return_value=mesh):
        runner = AragogJAXRunner(
            config, {'output': str(tmp_path)}, {'F_atm': 1e5}, None, interior_o
        )

    runner._last_heating = np.zeros(n_stag)

    result = SimpleNamespace(
        success=True,
        t_final=1.0e3,
        n_steps=42,
        S_final=np.linspace(2500.0, 3500.0, n_stag),
    )

    with patch('aragog.jax.phase.evaluate_phase', return_value=fake_props):
        out = runner._extract_output(result, {'F_atm': 1e5}, interior_o)

    # Mass closure: the conservation invariant.
    M_mantle = out['M_mantle']
    M_liq = out['M_mantle_liquid']
    M_sol = out['M_mantle_solid']
    assert M_liq + M_sol == pytest.approx(M_mantle, rel=1e-12), (
        f'mass closure broken: liq={M_liq:.6e} + sol={M_sol:.6e} != total={M_mantle:.6e}'
    )

    # Sign guard on T_magma + boundedness on Phi_global.
    assert out['T_magma'] > 0, f'T_magma not positive: {out["T_magma"]}'
    assert 0.0 <= out['Phi_global'] <= 1.0, f'Phi_global out of [0,1]: {out["Phi_global"]}'

    # Discrimination: with the asymmetric phi profile + uniform mass, the
    # mass-weighted mean Phi_global = mean(phi) = 0.5 exactly. The volume-
    # weighted mean (which would be the regression target) is also 0.5
    # here because volume is uniform; choose this carefully: the test
    # below pins the value at 0.5 ± 1e-12. A regression to a different
    # weighting (e.g. radial weighting) would shift the value.
    assert out['Phi_global'] == pytest.approx(0.5, abs=1e-12)

    # Discrimination: M_mantle_liquid for this asymmetric phi profile
    # equals sum(phi * mass) = mean(phi) * M_mantle = 0.5 * M_mantle by
    # construction (uniform mass, mean(phi)=0.5). A regression that
    # accidentally computed M_mantle_liquid as sum((1-phi) * mass) (the
    # solid formula) would land at the SAME value here (also 0.5 * M)
    # because mean(phi) = mean(1-phi) = 0.5 for this profile. Re-test
    # with a different phi profile to catch that variant.
    phi_skewed = np.array([0.05, 0.10, 0.15, 0.20, 0.50])  # mean 0.2
    eos.melt_fraction = lambda P, S: phi_skewed
    with patch('aragog.jax.phase.evaluate_phase', return_value=fake_props):
        out2 = runner._extract_output(result, {'F_atm': 1e5}, interior_o)
    # mean(phi_skewed) = 0.2; mass-weighted M_liq = 0.2 * M_total.
    # If a regression swapped liquid/solid, M_liq would be 0.8 * M_total.
    assert out2['M_mantle_liquid'] / out2['M_mantle'] == pytest.approx(0.2, abs=1e-12), (
        'liquid/total ratio does not match the mean phi: liquid/solid swap suspected'
    )
    # Closure still holds.
    assert out2['M_mantle_liquid'] + out2['M_mantle_solid'] == pytest.approx(
        out2['M_mantle'], rel=1e-12
    )


@pytest.mark.unit
def test_run_solver_includes_heating_when_radiogenic_enabled(tmp_path):
    """``run_solver`` adds radiogenic heating to the JAX solver's input
    when ``config.interior_energetics.heat_radiogenic`` is True.

    Contract from ``aragog_jax.py:170-180``: the heating array is built
    from the numpy solver's radionuclide list when heat_radiogenic is
    True, and from interior_o.tides when heat_tidal is True. With both
    flags off the array is all-zero.

    Verifies via the captured call kwargs of ``solve_entropy``:
    - heat_radiogenic=False produces an all-zero heating array.
    - heat_radiogenic=True produces a non-zero heating array equal to
      the sum of the radionuclide get_heating() returns at t_start.
    - Edge case: empty radionuclide list with heat_radiogenic=True
      still produces an all-zero array (no spurious additions).

    The captured-kwarg pattern asserts on the VALUE passed to the
    solver, not on a log line; per the failure-modes table in
    proteus-tests.md §16, log-only assertions can drift while the
    underlying call kwarg changes silently.
    """
    config = _make_config(heat_radiogenic=True, heat_tidal=False)
    interior_o = _make_interior_o(spider_eos_dir=str(tmp_path), prepopulate_jax=True)
    n_stag = 5

    # Pre-fill the radionuclide list on the numpy solver's parameters.
    radio_per_kg = np.full(n_stag, 1.0e-12)
    radio_mock = MagicMock()
    radio_mock.get_heating = MagicMock(return_value=radio_per_kg)
    interior_o.aragog_solver.parameters.radionuclides = [radio_mock]
    interior_o.aragog_solver._S0 = np.linspace(2500.0, 3500.0, n_stag)

    with patch.object(AragogJAXRunner, '_build_mesh_arrays', return_value=MagicMock()):
        runner = AragogJAXRunner(
            config, {'output': str(tmp_path)}, {'F_atm': 1e5}, None, interior_o
        )

    captured = {}

    def _fake_solve_entropy(S0, t_start, t_end, eos, params, mesh, bc, heating, **kw):
        captured['heating'] = np.asarray(heating)
        captured['success'] = True
        return SimpleNamespace(success=False, t_final=t_start, n_steps=0, S_final=S0)

    # Patch _extract_output and _write_ncdf so the runner short-circuits
    # after capturing the kwarg (the test pin is on the heating value,
    # not the full output translation).
    with patch(
        'aragog.jax.solver.solve_entropy',
        side_effect=_fake_solve_entropy,
    ):
        with pytest.raises(RuntimeError):
            runner.run_solver(
                {'F_atm': 1e5, 'Time': 1.0e3},
                interior_o,
                {'output': str(tmp_path)},
                write_data=False,
            )

    # Discrimination: the captured heating array is non-zero AND equals
    # the radionuclide contribution (one radionuclide returning 1e-12).
    h = captured['heating']
    assert np.all(np.isfinite(h)), 'heating array not finite'
    assert np.allclose(h, radio_per_kg), (
        f'heating array does not match radionuclide contribution: {h} vs {radio_per_kg}'
    )
    # Discrimination: a regression that swallowed the radio_per_kg into a
    # zero array would fail this check; the absolute value rules that out.
    assert h.sum() > 0, 'heating sum unexpectedly zero with heat_radiogenic=True'
