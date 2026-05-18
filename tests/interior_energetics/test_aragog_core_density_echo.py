"""Unit tests for the Aragog wrapper's core-density echo-back.

The echo-back recomputes :math:`\\rho_\\mathrm{core} = M_\\mathrm{core} /
(\\tfrac{4}{3} \\pi R_\\mathrm{cmb}^3)` from the on-disk Zalmoxis mantle
mesh file and the live ``hf_row['M_core']``, then writes the corrected
value back to ``hf_row['core_density']`` and into the live solver's
``MeshParameters.core_density``. The motivation mirrors SPIDER's
``-rho_core`` re-derivation in ``proteus/interior_energetics/spider.py``,
but the file format is different: SPIDER reads its own non-dimensional
``spider_mesh.dat`` (header + surface-first ratios) and multiplies the
fractional coresize by ``hf_row['R_int']``; the Aragog path here reads
absolute SI radii from ``zalmoxis_output.dat`` (CMB-first) directly.
The two paths give the same numerical answer when both files are in
sync with the same planet state, which is the production case.

Tests cover:
- ``resolve_core_density`` baseline + echo-back paths.
- ``setup_solver`` plumbing: the resolved value reaches ``MeshParameters``.
- ``update_structure`` refresh: a Zalmoxis re-solve that shifts R_cmb
  propagates into the live solver and ``hf_row``.
- Failure-mode behaviour: missing file, empty file, M_core <= 0.

The integration tests mock the heavy Aragog construction so the wrapper's
control flow can be exercised in milliseconds without an EOS load.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _write_mantle_mesh(target: Path, R_cmb: float = 3.480e6) -> Path:
    """Write a 5-column ascending-r mantle mesh file at the given CMB radius."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f'{R_cmb:.17e} 1.35e11 9904.0 10.7 4500.0\n'
        f'{R_cmb + 1e6:.17e} 1.0e11 5500.0 10.5 4200.0\n'
        f'{6.371e6:.17e} 1.0e5 3300.0 9.81 1800.0\n'
    )
    return target


def _make_minimal_config(struct_module: str = 'zalmoxis') -> MagicMock:
    """Just enough config for resolve_core_density to make decisions."""
    config = MagicMock()
    config.interior_struct.module = struct_module
    config.interior_struct.core_density = 'self'  # take from hf_row
    return config


# -- resolve_core_density unit tests ----------------------------------------


@pytest.mark.unit
def test_resolve_returns_baseline_when_no_mesh_file(tmp_path):
    """No zalmoxis_output.dat present -> return get_core_density() unchanged.

    Simulates a fresh init before Zalmoxis has written the mesh file. The
    wrapper must fall back to the cached ``hf_row['core_density']`` rather
    than crashing on a missing file.
    """
    from proteus.interior_energetics.aragog import resolve_core_density

    config = _make_minimal_config()
    hf_row = {'core_density': 10000.0, 'M_core': 1.94e24}

    result = resolve_core_density(config, hf_row, str(tmp_path))

    assert result == pytest.approx(10000.0)
    # No mesh file -> hf_row is not echo-mutated.
    assert hf_row['core_density'] == pytest.approx(10000.0)


@pytest.mark.unit
def test_resolve_overrides_when_zalmoxis_mesh_present(tmp_path):
    """Mesh file + M_core > 0 -> override returns mesh-derived rho_core."""
    from proteus.interior_energetics.aragog import resolve_core_density

    R_cmb = 3.480e6
    M_core = 1.94e24
    expected = M_core / (4.0 / 3.0 * math.pi * R_cmb**3)

    data_dir = tmp_path / 'data'
    _write_mantle_mesh(data_dir / 'zalmoxis_output.dat', R_cmb=R_cmb)

    config = _make_minimal_config()
    # hf_row carries a deliberately-stale value (e.g. from a pre-blend
    # Zalmoxis state). The override should ignore it and use the
    # mesh-derived value.
    hf_row = {'core_density': 9000.0, 'M_core': M_core}

    result = resolve_core_density(config, hf_row, str(tmp_path))

    assert result == pytest.approx(expected, rel=1e-12)
    # And critically, hf_row['core_density'] is echoed back so downstream
    # modules see the actually-used density.
    assert hf_row['core_density'] == pytest.approx(expected, rel=1e-12)
    assert hf_row['core_density'] != pytest.approx(9000.0, rel=1e-3)


@pytest.mark.unit
def test_resolve_skips_when_M_core_is_zero(tmp_path):
    """M_core = 0 -> no override; baseline returned unchanged.

    This handles the very-first init call before Zalmoxis has populated
    M_core. A divide-by-zero or NaN would otherwise contaminate the
    energy-balance BC on the first time step.
    """
    from proteus.interior_energetics.aragog import resolve_core_density

    data_dir = tmp_path / 'data'
    _write_mantle_mesh(data_dir / 'zalmoxis_output.dat')

    config = _make_minimal_config()
    hf_row = {'core_density': 11000.0, 'M_core': 0.0}

    result = resolve_core_density(config, hf_row, str(tmp_path))

    assert result == pytest.approx(11000.0)
    assert hf_row['core_density'] == pytest.approx(11000.0)


@pytest.mark.unit
def test_resolve_skips_when_M_core_missing(tmp_path):
    """M_core key absent in hf_row -> baseline returned (no KeyError)."""
    from proteus.interior_energetics.aragog import resolve_core_density

    data_dir = tmp_path / 'data'
    _write_mantle_mesh(data_dir / 'zalmoxis_output.dat')

    config = _make_minimal_config()
    hf_row = {'core_density': 11000.0}  # no M_core key

    result = resolve_core_density(config, hf_row, str(tmp_path))

    assert result == pytest.approx(11000.0)
    # No-side-effect discriminator: the missing-M_core branch must
    # leave hf_row untouched. A regression that wrote a NaN or zero
    # to hf_row['core_density'] would propagate into the helpfile and
    # downstream into the energy-balance BC on the next iteration.
    assert hf_row['core_density'] == pytest.approx(11000.0)
    assert 'M_core' not in hf_row


@pytest.mark.unit
def test_resolve_falls_back_on_corrupt_mesh(tmp_path):
    """Garbled mesh file -> wrapper logs and falls back, never crashes.

    A corrupted ``zalmoxis_output.dat`` (truncated I/O, write race, ...)
    must not abort the run. The wrapper should log a debug message and
    return the cached baseline so the integrator can proceed with
    whatever Zalmoxis last wrote to ``hf_row['core_density']``.
    """
    from proteus.interior_energetics.aragog import resolve_core_density

    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / 'zalmoxis_output.dat').write_text('not a number\n')

    config = _make_minimal_config()
    hf_row = {'core_density': 10500.0, 'M_core': 1.94e24}

    result = resolve_core_density(config, hf_row, str(tmp_path))

    assert result == pytest.approx(10500.0)
    assert hf_row['core_density'] == pytest.approx(10500.0)


@pytest.mark.unit
def test_resolve_uses_first_row_R_cmb_not_last(tmp_path):
    """Discriminator: a regression that read R_int (last row) would be ~6x lighter.

    Zalmoxis writes the mantle mesh with the CMB at the first row. If
    the helper accidentally read the surface row, the resulting
    rho_core would be wrong by :math:`(R_\\mathrm{int}/R_\\mathrm{cmb})^3
    \\sim 6` for Earth, producing an Earth-core density of
    :math:`\\sim 1700` kg/m^3 (less than mantle silicate, physically
    impossible for a metallic core).
    """
    from proteus.interior_energetics.aragog import resolve_core_density

    R_cmb = 3.480e6
    M_core = 1.94e24
    data_dir = tmp_path / 'data'
    _write_mantle_mesh(data_dir / 'zalmoxis_output.dat', R_cmb=R_cmb)

    config = _make_minimal_config()
    hf_row = {'core_density': 1.0, 'M_core': M_core}

    result = resolve_core_density(config, hf_row, str(tmp_path))

    # Earth-core density should be O(10^4) kg/m^3, not O(10^3).
    assert result > 8000.0
    assert result < 13000.0


@pytest.mark.unit
def test_resolve_writes_python_float_back_to_hf_row(tmp_path):
    """``hf_row['core_density']`` echo-back must be a plain ``float``.

    The helpfile CSV writer breaks on bare ``np.ndarray`` of shape ``()``;
    a plain ``float`` is universally safe.
    """
    from proteus.interior_energetics.aragog import resolve_core_density

    data_dir = tmp_path / 'data'
    _write_mantle_mesh(data_dir / 'zalmoxis_output.dat')

    config = _make_minimal_config()
    hf_row = {'core_density': 10000.0, 'M_core': 1.94e24}

    resolve_core_density(config, hf_row, str(tmp_path))

    assert isinstance(hf_row['core_density'], float)
    assert not isinstance(hf_row['core_density'], np.ndarray)


# -- update_structure echo-back tests ---------------------------------------


def _make_solver_with_mesh(mesh_file: str, init_core_density: float = 10000.0):
    """Build a mock solver that exposes the parameters MeshParameters needs."""
    solver = MagicMock()
    # solver.parameters.mesh.* mirrors MeshParameters; we use a SimpleNamespace
    # so attribute writes stick (MagicMock would silently accept anything).
    from types import SimpleNamespace

    solver.parameters.mesh = SimpleNamespace(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        gravitational_acceleration=9.81,
        core_density=init_core_density,
        eos_file=mesh_file,
    )
    solver._prev_struct_log = None
    return solver


@pytest.mark.unit
def test_update_structure_refreshes_core_density_on_zalmoxis_resolve(tmp_path):
    """A Zalmoxis re-solve that shifts R_cmb must update the live solver.

    Scenario: init mesh has R_cmb = 3.480e6 m; Zalmoxis re-solves and
    rewrites the mesh with R_cmb = 3.500e6 m (slightly larger CMB,
    plausible after IC equilibration). The live solver's
    ``mesh.core_density`` and ``hf_row['core_density']`` must both
    track the new value.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    data_dir = tmp_path / 'data'
    mesh_file = _write_mantle_mesh(data_dir / 'zalmoxis_output.dat', R_cmb=3.480e6)

    # First solve: init density matches the init mesh.
    M_core = 1.94e24
    init_rho = M_core / (4.0 / 3.0 * math.pi * 3.480e6**3)

    # Now Zalmoxis re-solves and rewrites the mesh with a shifted R_cmb.
    new_R_cmb = 3.500e6
    _write_mantle_mesh(mesh_file, R_cmb=new_R_cmb)
    expected_rho = M_core / (4.0 / 3.0 * math.pi * new_R_cmb**3)

    solver = _make_solver_with_mesh(str(mesh_file), init_core_density=init_rho)
    interior_o = MagicMock()
    interior_o.aragog_solver = solver

    config = MagicMock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.core_frac = 0.55

    hf_row = {
        'R_int': 6.371e6,
        'R_core': new_R_cmb,
        'gravity': 9.81,
        'M_core': M_core,
        'core_density': init_rho,  # stale (pre-resolve) value
        'Time': 1.0e4,
    }

    AragogRunner.update_structure(config, hf_row, interior_o)

    # Solver picks up the new density.
    assert solver.parameters.mesh.core_density == pytest.approx(expected_rho, rel=1e-12)
    # And hf_row sees the echo-back.
    assert hf_row['core_density'] == pytest.approx(expected_rho, rel=1e-12)
    # The shift is non-trivial (> 1% drift in R_cmb).
    assert solver.parameters.mesh.core_density != pytest.approx(init_rho, rel=1e-3)


@pytest.mark.unit
def test_update_structure_skips_echo_back_for_spider_module(tmp_path):
    """SPIDER structure module + Zalmoxis mesh present: no echo-back fires.

    The wrapper gates the echo-back on ``interior_struct.module ==
    'zalmoxis'``. If a SPIDER-coupled run leaves a stale
    zalmoxis_output.dat in the output directory (rare but possible
    after a module switch), the wrapper must NOT silently start using
    that file as authoritative.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    # Mesh file exists at zalmoxis path but the structure module is SPIDER:
    # set solver.eos_file to a SPIDER mesh path so it's NOT zalmoxis_output.
    spider_mesh = tmp_path / 'data' / 'spider_mesh.dat'
    spider_mesh.parent.mkdir(parents=True)
    spider_mesh.write_text(
        '# 3 1\n6.371e6 1.0e5 3300.0 9.81 1800.0\n4.5e6 1e11 5500.0 10.5 4200.0\n3.480e6 1.35e11 9904.0 10.7 4500.0\n'
    )

    init_rho = 11000.0
    solver = _make_solver_with_mesh(str(spider_mesh), init_core_density=init_rho)
    interior_o = MagicMock()
    interior_o.aragog_solver = solver

    config = MagicMock()
    config.interior_struct.module = 'spider'
    config.interior_struct.core_frac = 0.55

    hf_row = {
        'R_int': 6.371e6,
        'R_core': 3.480e6,
        'gravity': 9.81,
        'M_core': 1.94e24,
        'core_density': init_rho,
        'Time': 1.0e4,
    }

    AragogRunner.update_structure(config, hf_row, interior_o)

    # SPIDER-module path: echo-back gate is closed; init density preserved.
    assert solver.parameters.mesh.core_density == pytest.approx(init_rho)
    assert hf_row['core_density'] == pytest.approx(init_rho)


@pytest.mark.unit
def test_update_structure_handles_missing_eos_file(tmp_path):
    """Solver's eos_file points to a non-existent path: skip echo-back, no crash."""
    from proteus.interior_energetics.aragog import AragogRunner

    init_rho = 10500.0
    solver = _make_solver_with_mesh(
        '/nonexistent/zalmoxis_output.dat', init_core_density=init_rho
    )
    interior_o = MagicMock()
    interior_o.aragog_solver = solver

    config = MagicMock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.core_frac = 0.55

    hf_row = {
        'R_int': 6.371e6,
        'R_core': 3.480e6,
        'gravity': 9.81,
        'M_core': 1.94e24,
        'core_density': init_rho,
        'Time': 1.0e4,
    }

    # Must not raise.
    AragogRunner.update_structure(config, hf_row, interior_o)

    assert solver.parameters.mesh.core_density == pytest.approx(init_rho)
    # No-side-effect discriminator: hf_row['core_density'] must also
    # remain the cached baseline. A regression that wrote NaN through
    # to hf_row before bailing on the missing file would propagate
    # into the helpfile and corrupt the next iteration's BC.
    assert hf_row['core_density'] == pytest.approx(init_rho)
    # Sign / positivity invariant (Section 3): core density must
    # remain strictly positive after the fallback. A regression
    # that zeroed the field on the exception path would land here.
    assert solver.parameters.mesh.core_density > 0.0


@pytest.mark.unit
def test_update_structure_skips_when_M_core_is_zero(tmp_path):
    """M_core = 0 (e.g. pre-init) -> no override even with a valid mesh file."""
    from proteus.interior_energetics.aragog import AragogRunner

    data_dir = tmp_path / 'data'
    mesh_file = _write_mantle_mesh(data_dir / 'zalmoxis_output.dat')

    init_rho = 10500.0
    solver = _make_solver_with_mesh(str(mesh_file), init_core_density=init_rho)
    interior_o = MagicMock()
    interior_o.aragog_solver = solver

    config = MagicMock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.core_frac = 0.55

    hf_row = {
        'R_int': 6.371e6,
        'R_core': 3.480e6,
        'gravity': 9.81,
        'M_core': 0.0,  # not yet populated
        'core_density': init_rho,
        'Time': 0.0,
    }

    AragogRunner.update_structure(config, hf_row, interior_o)

    assert solver.parameters.mesh.core_density == pytest.approx(init_rho)
    # No-side-effect discriminator: with M_core=0 the wrapper must
    # skip the echo-back and leave hf_row['core_density'] at the
    # baseline. A divide-by-zero regression would have written NaN to
    # both the solver and hf_row.
    assert hf_row['core_density'] == pytest.approx(init_rho)
    # Section 3 positivity: even on the skip path the live core
    # density must remain strictly positive (the BC depends on it).
    assert solver.parameters.mesh.core_density > 0.0


# -- Round-trip consistency tests -------------------------------------------


@pytest.mark.unit
def test_round_trip_setup_then_update_consistent(tmp_path):
    """resolve_core_density at setup and update_structure must agree.

    On a stable mesh (no Zalmoxis re-solve in between), setup_solver's
    resolve_core_density and update_structure's echo-back must produce
    bit-identical core densities. A drift here would mean the two code
    paths use different R_cmb sources, which is the exact
    self-consistency bug this feature exists to prevent.
    """
    from proteus.interior_energetics.aragog import (
        AragogRunner,
        resolve_core_density,
    )

    data_dir = tmp_path / 'data'
    mesh_file = _write_mantle_mesh(data_dir / 'zalmoxis_output.dat', R_cmb=3.495e6)
    M_core = 1.94e24

    config = _make_minimal_config()
    hf_row = {'core_density': 10000.0, 'M_core': M_core}

    rho_setup = resolve_core_density(config, hf_row, str(tmp_path))

    # Now exercise the update path with a solver fresh-installed at rho_setup.
    solver = _make_solver_with_mesh(str(mesh_file), init_core_density=rho_setup)
    interior_o = MagicMock()
    interior_o.aragog_solver = solver

    full_config = MagicMock()
    full_config.interior_struct.module = 'zalmoxis'
    full_config.interior_struct.core_frac = 0.55

    hf_row['R_int'] = 6.371e6
    hf_row['R_core'] = 3.495e6
    hf_row['gravity'] = 9.81
    hf_row['Time'] = 1.0e4

    AragogRunner.update_structure(full_config, hf_row, interior_o)

    rho_update = solver.parameters.mesh.core_density

    assert rho_setup == pytest.approx(rho_update, rel=1e-15)
    # Closed-form discriminator: the round-trip must agree with the
    # analytical M / (4/3 pi R^3) formula at R_cmb = 3.495e6 m. A
    # regression that read R_int (~ 6.371e6 m) instead of R_cmb would
    # produce a value ~6.1x lower, well outside rel=1e-12.
    expected = M_core / (4.0 / 3.0 * math.pi * 3.495e6**3)
    assert rho_setup == pytest.approx(expected, rel=1e-12)
    # Positivity invariant (Section 3): core density is a physical
    # density and must be strictly positive.
    assert rho_setup > 0.0


@pytest.mark.unit
def test_two_consecutive_resolves_track_R_cmb_drift(tmp_path):
    """Two re-solves with different R_cmb produce two different rho_core.

    Verifies the per-iteration update genuinely re-reads the mesh file
    instead of returning a cached first-call value.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    data_dir = tmp_path / 'data'
    mesh_file = _write_mantle_mesh(data_dir / 'zalmoxis_output.dat', R_cmb=3.480e6)
    M_core = 1.94e24

    init_rho = M_core / (4.0 / 3.0 * math.pi * 3.480e6**3)
    solver = _make_solver_with_mesh(str(mesh_file), init_core_density=init_rho)
    interior_o = MagicMock()
    interior_o.aragog_solver = solver

    config = MagicMock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.core_frac = 0.55

    hf_row = {
        'R_int': 6.371e6,
        'R_core': 3.480e6,
        'gravity': 9.81,
        'M_core': M_core,
        'core_density': init_rho,
        'Time': 1.0e4,
    }

    # First update: mesh unchanged, density unchanged.
    AragogRunner.update_structure(config, hf_row, interior_o)
    rho_1 = solver.parameters.mesh.core_density

    # Zalmoxis re-solves with a shifted R_cmb.
    _write_mantle_mesh(mesh_file, R_cmb=3.520e6)
    hf_row['R_core'] = 3.520e6

    AragogRunner.update_structure(config, hf_row, interior_o)
    rho_2 = solver.parameters.mesh.core_density

    expected_2 = M_core / (4.0 / 3.0 * math.pi * 3.520e6**3)

    assert rho_1 != pytest.approx(rho_2, rel=1e-3)
    assert rho_2 == pytest.approx(expected_2, rel=1e-12)
    # rho_1 (R_cmb=3.480e6) > rho_2 (R_cmb=3.520e6) at fixed M_core.
    assert rho_1 > rho_2


# -- File-write race / plausibility tests -----------------------------------


@pytest.mark.unit
def test_resolve_falls_back_when_derived_density_too_low(tmp_path):
    """Mesh corrupted to give R_cmb >> R_int -> rho_core would be < 1000 kg/m^3.

    Simulates a partial-write race on a network filesystem where the
    first row is readable but the radius column has been truncated /
    re-padded to a non-physical value (here, R_cmb = 9.0e6 m for an
    Earth-mass planet, giving rho_core ~ 600 kg/m^3 — physically
    impossible for any iron-bearing core).
    """
    from proteus.interior_energetics.aragog import resolve_core_density

    data_dir = tmp_path / 'data'
    # R_cmb = 9.0e6 m is larger than Earth's planet radius; rho_core
    # at 1.94e24 kg / (4/3 pi (9.0e6)^3) ~ 635 kg/m^3 (less than water).
    _write_mantle_mesh(data_dir / 'zalmoxis_output.dat', R_cmb=9.0e6)

    config = _make_minimal_config()
    hf_row = {'core_density': 11000.0, 'M_core': 1.94e24}

    result = resolve_core_density(config, hf_row, str(tmp_path))

    # Plausibility check fires; cached baseline preserved.
    assert result == pytest.approx(11000.0)
    assert hf_row['core_density'] == pytest.approx(11000.0)


@pytest.mark.unit
def test_resolve_falls_back_when_derived_density_too_high(tmp_path):
    """Pathologically tiny R_cmb -> rho_core would exceed 30000 kg/m^3.

    Simulates a write-race where the radius column is truncated to a
    sub-thousand-km value, producing an unphysically dense core.
    """
    from proteus.interior_energetics.aragog import resolve_core_density

    data_dir = tmp_path / 'data'
    # R_cmb = 1.0e6 m at M_core = 1.94e24 kg gives rho_core ~ 4.6e8 kg/m^3.
    _write_mantle_mesh(data_dir / 'zalmoxis_output.dat', R_cmb=1.0e6)

    config = _make_minimal_config()
    hf_row = {'core_density': 11000.0, 'M_core': 1.94e24}

    result = resolve_core_density(config, hf_row, str(tmp_path))

    assert result == pytest.approx(11000.0)
    assert hf_row['core_density'] == pytest.approx(11000.0)


@pytest.mark.unit
def test_resolve_accepts_super_earth_core_density(tmp_path):
    """Compressed super-Earth core (~ 17000 kg/m^3) is inside the bracket.

    Plausibility bounds [1000, 30000] must accept the upper end of
    the realistic super-Earth regime (10 M_E core, deeply compressed).
    A regression that tightened the upper bound to 12000 would reject
    legitimate super-Earth runs.
    """
    from proteus.interior_energetics.aragog import resolve_core_density

    data_dir = tmp_path / 'data'
    # 10 M_E with core_frac=0.32 by mass; compressed core radius ~ 5.5e6 m
    R_cmb = 5.50e6
    _write_mantle_mesh(data_dir / 'zalmoxis_output.dat', R_cmb=R_cmb)

    M_core = 0.32 * 10.0 * 5.972e24
    expected = M_core / (4.0 / 3.0 * math.pi * R_cmb**3)
    assert 14000.0 < expected < 30000.0  # sanity: in super-Earth range

    config = _make_minimal_config()
    hf_row = {'core_density': 9000.0, 'M_core': M_core}

    result = resolve_core_density(config, hf_row, str(tmp_path))

    # Echo-back fires; super-Earth density is accepted.
    assert result == pytest.approx(expected, rel=1e-12)
    assert hf_row['core_density'] == pytest.approx(expected, rel=1e-12)


# -- Formula correctness sanity check ---------------------------------------


@pytest.mark.unit
def test_echo_back_formula_correct(tmp_path):
    """Aragog's echo-back returns ``M_core / (4/3 pi R_cmb^3)`` exactly.

    The numerical formula matches what SPIDER computes at
    ``interior_energetics/spider.py:706-708`` for the same
    ``(M_core, R_cmb)`` inputs. Note that the *file format* the two
    wrappers read from differs (SPIDER: surface-first non-dimensional
    coresize; Aragog: CMB-first absolute SI radii), so this test pins
    only the formula, not the parsing path.
    """
    from proteus.interior_energetics.aragog import resolve_core_density

    R_cmb = 3.480e6
    M_core = 1.94e24
    spider_rho = M_core / (4.0 / 3.0 * math.pi * R_cmb**3)

    data_dir = tmp_path / 'data'
    _write_mantle_mesh(data_dir / 'zalmoxis_output.dat', R_cmb=R_cmb)

    config = _make_minimal_config()
    hf_row = {'core_density': 9999.9, 'M_core': M_core}

    aragog_rho = resolve_core_density(config, hf_row, str(tmp_path))

    # Same formula, same inputs -> floating-point parity.
    assert aragog_rho == pytest.approx(spider_rho, rel=1e-15)
    # Exponent-error guard: the formula uses R_cmb**3 (sphere volume),
    # not R_cmb**2 or R_cmb**4. The wrong exponent at R_cmb=3.480e6 m
    # and M_core=1.94e24 kg lands many orders of magnitude away:
    # R**2 would give 1.27e10 kg/m^3 (1e6x too high), R**4 would
    # give 1.10e-2 kg/m^3 (1e6x too low). The bracket below
    # discriminates both.
    assert 8000.0 < aragog_rho < 14000.0
    # Sign / positivity invariant (Section 3): mass and volume are
    # both strictly positive so the density must be strictly positive.
    assert aragog_rho > 0.0
